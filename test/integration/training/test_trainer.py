import sys
import os
import logging
from tempfile import TemporaryDirectory

from src.training.trainer import GPTTrainer, TrainModelInitialiser
from src.data_io.data_loaders import DataLoader
from src.training.train_main import init_job


def test_scratch_training():
    configs = init_job(
        [
            "--config_file",
            os.path.join(os.path.dirname(__file__), "test_scratch_training.yml"),
        ]
    )
    os.rmdir(
        configs.job_config.out_dir
    )  # this was created during init but we don't need it here
    with TemporaryDirectory() as tmpdirname:
        configs.job_config.out_dir = os.path.join(
            tmpdirname, configs.job_config.out_dir
        )
        os.makedirs(
            configs.job_config.out_dir, exist_ok=True
        )  # we need to create this manually since it's a tempdir

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
        out_dir = os.listdir(configs.job_config.out_dir)
        assert out_dir
        assert out_dir[0] == "ckpt.pt"


def test_finetune_training():
    configs = init_job(
        [
            "--config_file",
            os.path.join(os.path.dirname(__file__), "test_finetune_training.yml"),
        ]
    )
    os.rmdir(
        configs.job_config.out_dir
    )  # this was created during init but we don't need it here
    logging.debug(configs.job_config)

    with TemporaryDirectory() as tmpdirname:
        configs.job_config.out_dir = os.path.join(
            tmpdirname, configs.job_config.out_dir
        )
        os.makedirs(
            configs.job_config.out_dir, exist_ok=True
        )  # we need to create this manually since it's a tempdir

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
        out_dir = os.listdir(configs.job_config.out_dir)
        assert out_dir
        assert out_dir[0] == "ckpt.pt"
