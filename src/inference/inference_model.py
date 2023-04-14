import os
import argparse
import logging
import pickle
from collections import namedtuple
from contextlib import nullcontext
import torch
import tiktoken

from config.configurator import override_config
from src.model.model_init import InferenceModelInitialiser
from src.model.gpt_model import GPTConfig, GPT
from src.features.gpt_encoding import DataEncoder


logger = logging.getLogger(__name__)


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
        for k in range(num_samples):
            y = self.generate_sample(start)
            logger.info(f"{y}\n---------------")
