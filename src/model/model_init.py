import os
import time
import logging
import math
from dataclasses import asdict
from collections import namedtuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.gpt_model import GPTConfig, GPT


logger = logging.getLogger(__name__)


class ModelInitialiser:
    def __init__(self, configs):
        self.configs = configs
        
    def init_resume(self, training_args=True):
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(self.configs.job_config.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.configs.job_config.device)
        if training_args:
            # resume training from a checkpoint.
            gptconf = self.configs.model_config
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                setattr(gptconf, k, checkpoint['model_args'][k])
        else:
            gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        return model, checkpoint
