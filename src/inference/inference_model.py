import os
import argparse
import logging
import pickle
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import tiktoken

from src.model.model_init import ModelInitialiser
from src.model.gpt_model import GPTConfig, GPT
from src.features.gpt_encoding import DataEncoder


logger = logging.getLogger(__name__)


@dataclass
class InferenceJobConfig:
    """default config values designed to sample from a gpt2"""
    out_dir: str = 'out'
    start: str = "\n"
    init_from: str = 'resume' # 'resume' or 'gpt2*'
    num_samples: int = 10
    max_new_tokens: int = 500 # number of tokens generated in each sample
    temperature: float = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device_type: str = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    dtype: str = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile_model: bool = True # use PyTorch 2.0 to compile the model to be faster


class InferenceModel:
    """This class can be used to call a GPT model for inference"""
    def __init__(self, configs):
        self.job_config = configs.job_config
        self.context = configs.context
        self.encode_fn = None
        self.decode_fn = None
        self.model = None
    
    def init_inference(self, model, checkpoint=None):        
        # ok let's assume gpt-2 encodings by default
        logger.info("No meta.pkl found, assuming GPT-2 encodings...")
        data_encoder = DataEncoder()
        self.encode_fn = lambda s: data_encoder.encode(s, train=False)
        self.decode_fn = lambda l: data_encoder.decode(l)
        
        self.model = model
        self.model.eval()
        self.model.to(self.job_config.device)
        if self.job_config.compile_model:
            self.model = torch.compile(self.model) # requires PyTorch 2.0 (optional)
        
    def generate_sample(self, s):
        s_ids = self.encode_fn(s)
        x = (torch.tensor(s_ids, dtype=torch.long, device=self.job_config.device)[None, ...])
        with torch.no_grad():
            with self.context:
                y = self.model.generate(
                    x,
                    self.job_config.max_new_tokens,
                    temperature=self.job_config.temperature,
                    top_k=self.job_config.top_k
                )
        
        return self.decode_fn(y[0].tolist())
            
    def generate_samples(self, num_samples):
        start = self.job_config.start
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        # run generation
        out = []
        for k in range(num_samples):
            y = self.generate_sample(start)
            out.append(y)
            
        return out
            

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
