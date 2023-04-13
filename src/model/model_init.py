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
        

class InferenceModelInitialiser(ModelInitialiser):
    def __init__(self, configs, **kwargs):
        super(InferenceModelInitialiser, self).__init__(configs, **kwargs)

    def init_model(self):
        InitialisedModel = namedtuple('InitialisedModel', 'model checkpoint')
        # model
        if self.configs.job_config.init_from == 'resume':
            # init from a model saved in a specific directory
            model, checkpoint = self.init_resume(training_args=False)
        elif self.configs.job_config.init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            model = GPT.from_pretrained(self.configs.job_config.init_from, dict(dropout=0.0))
            
        return InitialisedModel(model, checkpoint)
            

class TrainModelInitialiser(ModelInitialiser):
    def __init__(self, configs, **kwargs):
        super(TrainModelInitialiser, self).__init__(configs, **kwargs)
        
    def init_model(self, data_loader):
        InitialisedModel = namedtuple('InitialisedModel', 'model best_val_loss iter_num checkpoint')
        iter_num = 0
        best_val_loss = 1e9
        checkpoint = None
        if self.configs.job_config.init_from == 'scratch':
            # init a new model from scratch
            logger.info("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if data_loader.meta_vocab_size is None:
                logger.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            gptconf = self.configs.model_config
            gptconf.vocab_size = data_loader.meta_vocab_size if data_loader.meta_vocab_size is not None else 50304
            model = GPT(gptconf)
        elif self.configs.job_config.init_from == 'resume':
            logger.info(f"Resuming training from {self.configs.job_config.out_dir}")
            # resume training from a checkpoint.
            model, checkpoint = self.init_resume(training_args=True)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
        elif self.configs.job_config.init_from.startswith('gpt2'):
            logger.info(f"Initializing from OpenAI GPT-2 weights: {self.configs.job_config.init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=self.configs.model_config.dropout)
            model = GPT.from_pretrained(self.configs.job_config.init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                setattr(self.configs.model_config, k, getattr(model.config, k))

        # crop down the model block size if desired, using model surgery
        if self.configs.job_config.block_size < model.config.block_size:
            model.crop_block_size(self.configs.job_config.block_size)
            self.configs.model_config.block_size = self.configs.job_config.block_size # so that the checkpoint will have the right value
        model.to(self.configs.job_config.device)
        
        return InitialisedModel(model, best_val_loss, iter_num, checkpoint)
