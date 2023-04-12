"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import argparse
import time
import logging
import math
import pickle
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.model.gpt_model import GPTConfig, GPT
from src.data_io.data_loaders import DataLoader, DataConfig
from config.configurator import override_config


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GPTTrainer:
    def __init__(self, job_config, model_config, model, checkpoint=None):
        self.job_config = job_config
        self.model_config = model_config
        self.model = model
        self.checkpoint = checkpoint
        self.scaler = None
        self.optimizer = None
        self.unoptimized_model = None
    
    def init_trainer(self):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.job_config.dtype == 'float16'))

        # optimizer
        self.optimizer = self.model.configure_optimizers(
            self.job_config.weight_decay,
            self.job_config.learning_rate,
            (self.job_config.beta1, self.job_config.beta2),
            self.job_config.device_type
        )
        if self.job_config.init_from == 'resume':
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # compile the model
        if self.job_config.compile_model:
            logger.info("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0

        # wrap model into DDP container
        if self.job_config.ddp:
            self.model = DDP(self.model, device_ids=[self.job_config.ddp_local_rank])

        # logging
        if self.job_config.wandb_log and self.job_config.master_process:
            import wandb
            wandb.init(project=self.job_config.wandb_project, name=self.job_config.wandb_run_name, config=asdict(self.job_config))
        
    def training_loop(self, data_loader, context, iter_num, best_val_loss):
        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(self.job_config.eval_iters)
                for k in range(self.job_config.eval_iters):
                    X, Y = data_loader.get_batch(split)
                    with context:
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train()
            return out

        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < self.job_config.warmup_iters:
                return self.job_config.learning_rate * it / self.job_config.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > self.job_config.lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - self.job_config.warmup_iters) / (self.job_config.lr_decay_iters - self.job_config.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return self.job_config.min_lr + coeff * (self.job_config.learning_rate - self.job_config.min_lr)

        # training loop
        X, Y = data_loader.get_batch('train') # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.job_config.ddp else self.model # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if self.job_config.decay_lr else self.job_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.job_config.eval_interval == 0 and self.job_config.master_process:
                losses = estimate_loss()
                logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    self.job_config.wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or self.job_config.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': asdict(self.model_config),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': asdict(self.job_config),
                        }
                        logger.info(f"saving checkpoint to {self.job_config.out_dir}")
                        torch.save(checkpoint, os.path.join(self.job_config.out_dir, 'ckpt.pt'))
            if iter_num == 0 and self.job_config.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.job_config.gradient_accumulation_steps):
                if self.job_config.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == self.job_config.gradient_accumulation_steps - 1)
                with context:
                    logits, loss = self.model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = data_loader.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.job_config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.job_config.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.job_config.log_interval == 0 and self.job_config.master_process:
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.job_config.batch_size * self.job_config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.job_config.max_iters:
                break

def init_job():
    Configs = namedtuple('Configs', 'job_config data_config model_config context')
    job_args = parse_args()
    job_config = override_config(job_args.config_file)
    
    data_config = DataConfig(
        dataset=job_config.dataset,
        block_size=job_config.block_size,
        batch_size=job_config.batch_size,
        device=job_config.device,
        device_type=job_config.device_type,)
    
    model_config = GPTConfig(
        n_layer=job_config.n_layer,
        n_head=job_config.n_head,
        n_embd=job_config.n_embd,
        block_size=job_config.block_size,
        bias=job_config.bias,
        vocab_size=None,
        dropout=job_config.dropout,
    )
    
    # various inits, derived attributes, I/O setup
    job_config.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if job_config.ddp:
        init_process_group(backend=job_config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        job_config.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(job_config.device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        job_config.gradient_accumulation_steps *= 8 # simulate 8 gpus

    if master_process:
        os.makedirs(job_config.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[job_config.dtype]
    ctx = nullcontext() if job_config.device_type == 'cpu' else torch.amp.autocast(device_type=job_config.device_type, dtype=ptdtype)
    
    return Configs(job_config, data_config, model_config, ctx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="", default=None)
    parser.add_argument("--resume", type=int, default=0, help="")
    return parser.parse_args()


def main():
    configs = init_job()
    config = asdict(configs.job_config)
    logging.debug(configs.job_config)
    
    data_loader = DataLoader(configs.data_config)
    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None
    if configs.job_config.init_from == 'scratch':
        # init a new model from scratch
        logger.info("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if data_loader.meta_vocab_size is None:
            logger.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        gptconf = configs.model_config
        gptconf.vocab_size = data_loader.meta_vocab_size if data_loader.meta_vocab_size is not None else 50304
        model = GPT(gptconf)
    elif configs.job_config.init_from == 'resume':
        logger.info(f"Resuming training from {configs.job_config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(configs.job_config.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=configs.job_config.device)
        checkpoint_model_args = checkpoint['model_args']
        gptconf = configs.model_config
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            setattr(gptconf, k, checkpoint_model_args[k])
        # create the model
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif configs.job_config.init_from.startswith('gpt2'):
        logger.info(f"Initializing from OpenAI GPT-2 weights: {configs.job_config.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=configs.model_config.dropout)
        model = GPT.from_pretrained(configs.job_config.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            setattr(configs.model_config, k, getattr(model.config, k))
    
    # crop down the model block size if desired, using model surgery
    if configs.job_config.block_size < model.config.block_size:
        model.crop_block_size(configs.job_config.block_size)
        configs.model_config.block_size = configs.job_config.block_size # so that the checkpoint will have the right value
    model.to(configs.job_config.device)

    trainer = GPTTrainer(configs.job_config, configs.model_config, model, checkpoint)
    trainer.init_trainer()
    trainer.training_loop(
        data_loader=data_loader,
        context=configs.context,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
    )

    if configs.job_config.ddp:
        destroy_process_group()

    
if __name__ == "__main__":
    main()
