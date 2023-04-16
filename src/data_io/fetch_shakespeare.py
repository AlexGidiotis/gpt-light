"""
Prepare the Shakespeare dataset for language modeling.
"""
import os
import logging

import numpy as np

from src.features.gpt_encoding import DataEncoder
from src.data_io.data_fetchers import fetch_txt_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    length of dataset in characters:  1115394
    all the unique characters:
        #!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    vocab size: 65
    train has 1003854 tokens
    val has 111540 tokens
    """
    data = fetch_txt_data(
        data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        output_dir_path="data/tinyshakespeare",
    )
    data_builder = DataEncoder()
    train_data, val_data = data_builder.create_splits(data)
    
    # encode both to integers
    train_ids = data_builder.encode(train_data)
    val_ids = data_builder.encode(val_data)
    
    logger.info(f"vocabulary has {data_builder.enc.n_vocab} tokens")
    num_train_ids = train_ids["len"]
    num_val_ids = val_ids["len"]
    logger.info(f"train has {num_train_ids} tokens")
    logger.info(f"val has {num_val_ids} tokens")
    
    data_builder.save_data(train_ids, dir_path="data/tinyshakespeare", fname="train")
    data_builder.save_data(val_ids, dir_path="data/tinyshakespeare", fname="val")
    data_builder.save_metadata(dir_path="data/tinyshakespeare")


if __name__ == "__main__":
    main()
