import os
import pickle
import requests
import logging

import numpy as np
import tiktoken


logger = logging.getLogger(__name__)


class DataEncoder:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, s, train=True):
        if train:  # training preprocessing
            ids = self.enc.encode_ordinary(s) # encode_ordinary ignores any special tokens
            ids.append(self.enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
        else:  # inference
            out = self.enc.encode(s, allowed_special={"<|endoftext|>"})
        return out
    
    def decode(self, s):
        return self.enc.decode(s)

    @staticmethod
    def create_splits(data):
        """create the train and test splits"""
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]
        return train_data, val_data
    
    @staticmethod
    def save_data(data, dir_path, fname):
        """export to bin files"""
        data_ids = np.array(data["ids"], dtype=np.uint16)
        data_ids.tofile(os.path.join(dir_path, f"{fname}.bin"))
    
    def save_metadata(self, dir_path):
        """save the meta information as well, to help us encode/decode later"""
        meta = {
            'vocab_size': self.enc.n_vocab,
        }
        with open(os.path.join(dir_path, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
