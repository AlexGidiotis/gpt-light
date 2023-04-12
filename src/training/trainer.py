import time
import logging
import math
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


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