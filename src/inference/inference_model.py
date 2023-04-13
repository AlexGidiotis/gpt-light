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
    def __init__(self, configs):
        self.job_config = configs.job_config
        self.context = configs.context
        self.encode_fn = None
        self.decode_fn = None
    
    def init_inference(self, checkpoint=None):        
        # ok let's assume gpt-2 encodings by default
        logger.info("No meta.pkl found, assuming GPT-2 encodings...")
        data_encoder = DataEncoder()
        self.encode_fn = lambda s: data_encoder.encode(s, train=False)
        self.decode_fn = lambda l: data_encoder.decode(l)
            
    def generate_samples(self, model, num_samples):
        model.eval()
        model.to(self.job_config.device)
        if self.job_config.compile_model:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)
        
        start = self.job_config.start
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = self.encode_fn(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.job_config.device)[None, ...])

        # run generation
        with torch.no_grad():
            with self.context:
                for k in range(num_samples):
                    y = model.generate(
                        x,
                        self.job_config.max_new_tokens,
                        temperature=self.job_config.temperature,
                        top_k=self.job_config.top_k)
                    logger.info(f"{self.decode_fn(y[0].tolist())}\n---------------")
