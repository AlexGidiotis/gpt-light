"""
Sample from a trained model
"""
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
from src.inference.inference_model import InferenceModel


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="", default=None)
    return parser.parse_args()


def init_job():
    Configs = namedtuple('Configs', 'job_config context')
    job_args = parse_args()
    job_config = override_config(job_args.config_file, inference=True)
    
    torch.manual_seed(job_config.seed)
    torch.cuda.manual_seed(job_config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[job_config.dtype]
    ctx = nullcontext() if job_config.device_type == 'cpu' else torch.amp.autocast(device_type=job_config.device_type, dtype=ptdtype)
    
    return Configs(job_config, ctx)

    
def main():
    configs = init_job()
    logging.debug(configs.job_config)
    
    model_init = InferenceModelInitialiser(configs)
    initialised = model_init.init_model()
    configs = model_init.configs  # some configs might have been changed when loading a model
    
    inference_model = InferenceModel(configs)
    inference_model.init_inference(initialised.model, initialised.checkpoint)
    inference_model.generate_samples(configs.job_config.num_samples)


if __name__ == "__main__":
    main()
