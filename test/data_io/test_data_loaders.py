import os
from tempfile import TemporaryDirectory

import pytest

from src.data_io.data_loaders import DataLoader, DataConfig
from src.features.gpt_encoding import DataEncoder


def init_data(dataset, tmpdirname):
    data_dir = os.path.join(tmpdirname, "data")
    dataset_dir = os.path.join(data_dir, dataset)
    os.mkdir(data_dir)
    os.mkdir(dataset_dir)
    
    train_data = "This is a dataset created for training loaders"
    val_data = "This is a dataset created for validation loaders"
    data_encoder = DataEncoder()
    train_ids = data_encoder.encode(train_data)
    data_encoder.save_data(train_ids, dir_path=dataset_dir, fname="train")
    val_ids = data_encoder.encode(val_data)
    data_encoder.save_data(val_ids, dir_path=dataset_dir, fname="val")
    data_encoder.save_metadata(dir_path=dataset_dir)
        

def test_load_metadata():
    with TemporaryDirectory() as tmpdirname:
        dataset = "test_dataset"
        batch_size = 2
        block_size = 8
        init_data(dataset, tmpdirname)
        data_config = DataConfig(
            dataset=dataset,
            block_size=block_size,
            batch_size=batch_size,
            device="cpu",
            device_type="cpu",
        )
        data_loader = DataLoader(data_config, path=tmpdirname)
        assert data_loader.meta_vocab_size == 50257


def test_get_batch():
    with TemporaryDirectory() as tmpdirname:
        dataset = "test_dataset"
        batch_size = 2
        block_size = 8
        init_data(dataset, tmpdirname)
        data_config = DataConfig(
            dataset=dataset,
            block_size=block_size,
            batch_size=batch_size,
            device="cpu",
            device_type="cpu",
        )
        data_loader = DataLoader(data_config, path=tmpdirname)
        X, Y = data_loader.get_batch('train')
        assert X.shape == (batch_size, block_size)
        assert Y.shape == (batch_size, block_size)
