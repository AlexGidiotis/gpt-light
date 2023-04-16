"""
Poor Man's Configurator
"""

import logging
import yaml
from dataclasses import dataclass

from src.training.trainer import JobConfig
from src.inference.inference_model import InferenceJobConfig


logger = logging.getLogger(__name__)



def override_config(config_file=None, inference=False):
    loaded_configs = {}
    if config_file:
        logger.info(f"Overriding config with {config_file}:")
        with open(config_file, "r") as stream:
            loaded_configs = yaml.safe_load(stream)
    
    if inference:
        job_config = InferenceJobConfig(**loaded_configs)
    else:
        job_config = JobConfig(**loaded_configs)
    
    return job_config
