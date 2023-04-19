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
import sys
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
from src.training.trainer import GPTTrainer, TrainModelInitialiser
from src.data_io.data_loaders import DataLoader, DataConfig
from src.config.configurator import override_config


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init_job(args):
    Configs = namedtuple("Configs", "job_config data_config model_config context")
    job_args = parse_args(args)
    job_config = override_config(job_args.config_file)

    data_config = DataConfig(
        dataset=job_config.dataset,
        block_size=job_config.block_size,
        batch_size=job_config.batch_size,
        device=job_config.device,
        device_type=job_config.device_type,
    )

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
    job_config.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if job_config.ddp:
        init_process_group(backend=job_config.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        job_config.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(job_config.device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        job_config.gradient_accumulation_steps *= 8  # simulate 8 gpus

    if master_process:
        os.makedirs(job_config.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[job_config.dtype]
    ctx = (
        nullcontext()
        if job_config.device_type == "cpu"
        else torch.amp.autocast(device_type=job_config.device_type, dtype=ptdtype)
    )

    return Configs(job_config, data_config, model_config, ctx)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="", default=None)
    return parser.parse_args(args)


def main():
    configs = init_job(sys.argv[1:])
    config = asdict(configs.job_config)
    logging.debug(configs.job_config)

    data_loader = DataLoader(configs.data_config)
    model_init = TrainModelInitialiser(configs)
    initialised = model_init.init_model(data_loader)
    configs = (
        model_init.configs
    )  # some configs might have been changed when loading a model

    trainer = GPTTrainer(
        configs.job_config, configs.model_config, initialised.checkpoint
    )
    trainer.init_trainer(initialised.model)
    trainer.training_loop(
        data_loader=data_loader,
        model=initialised.model,
        context=configs.context,
        iter_num=initialised.iter_num,
        best_val_loss=initialised.best_val_loss,
    )

    if configs.job_config.ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
