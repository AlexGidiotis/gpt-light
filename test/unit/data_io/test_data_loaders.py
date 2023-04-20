import os
from tempfile import TemporaryDirectory

import pytest

from src.data_io.data_loaders import DataLoader, DataConfig
from src.features.gpt_encoding import DataEncoder
from test.shared.shared_testing import init_data


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
        X, Y = data_loader.get_batch("train")
        assert X.shape == (batch_size, block_size)
        assert Y.shape == (batch_size, block_size)
