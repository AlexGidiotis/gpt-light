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
from src.training.trainer import GPTTrainer
from src.data_io.data_loaders import DataLoader, DataConfig
from config.configurator import override_config


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
