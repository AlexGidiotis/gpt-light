import os

import pytest
from collections import namedtuple
from contextlib import nullcontext
import torch
from tempfile import TemporaryDirectory

from src.training.trainer import JobConfig, GPTTrainer, TrainModelInitialiser
from src.model.gpt_model import GPTConfig, GPT
from src.data_io.data_loaders import DataLoader, DataConfig
from src.features.gpt_encoding import DataEncoder
from test.shared.shared_testing import init_data


def init_training_model(dataset, tmpdirname, num_iters):
    InitModel = namedtuple("InitModel", "initialised data_loader configs")
    Configs = namedtuple("Configs", "job_config data_config model_config context")
    job_config = JobConfig(
        init_from="scratch",
        eval_interval=100,  # we set those high to avoid checkpointing
        eval_iters=100,  # we set those high to avoid checkpointing
        log_interval=1,
        always_save_checkpoint=False,
        dataset=dataset,
        batch_size=2,
        block_size=8,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        learning_rate=1.0e-3,
        max_iters=num_iters,
        lr_decay_iters=num_iters,
        min_lr=1.0e-4,
        beta2=0.99,
        device="cpu",
        device_type="cpu",
        dtype="float32",
        compile_model=False,
    )

    data_config = DataConfig(
        dataset=dataset,
        block_size=job_config.block_size,
        batch_size=job_config.batch_size,
        device=job_config.device,
        device_type=job_config.device_type,
    )
    data_loader = DataLoader(data_config, path=tmpdirname)

    model_config = GPTConfig(
        n_layer=job_config.n_layer,
        n_head=job_config.n_head,
        n_embd=job_config.n_embd,
        block_size=job_config.block_size,
        bias=job_config.bias,
        vocab_size=None,
        dropout=job_config.dropout,
    )

    master_process = True
    seed_offset = 0
    job_config.gradient_accumulation_steps *= 8  # simulate 8 gpus

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

    configs = Configs(job_config, data_config, model_config, ctx)

    model_init = TrainModelInitialiser(configs)
    initialised = model_init.init_model(data_loader)
    configs = (
        model_init.configs
    )  # some configs might have been changed when loading a model

    return InitModel(initialised, data_loader, configs)


def test_training_loop():
    with TemporaryDirectory() as tmpdirname:
        dataset = "test_dataset"
        num_iters = 1
        init_data(dataset, tmpdirname)
        init_model = init_training_model(dataset, tmpdirname, num_iters)

        trainer = GPTTrainer(
            init_model.configs.job_config,
            init_model.configs.model_config,
            init_model.initialised.checkpoint,
        )
        trainer.init_trainer(init_model.initialised.model)
        intial_loses = trainer.estimate_loss(
            init_model.data_loader, init_model.configs.context
        )
        train_logs = trainer.training_loop(
            data_loader=init_model.data_loader,
            context=init_model.configs.context,
            iter_num=init_model.initialised.iter_num,
            best_val_loss=init_model.initialised.best_val_loss,
        )
        final_losses = train_logs.losses
        assert train_logs.iter_num == num_iters + 1
        assert isinstance(final_losses, dict)
