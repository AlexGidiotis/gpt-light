import os

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
