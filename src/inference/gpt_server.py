import os
import logging
from collections import namedtuple

from contextlib import nullcontext
import torch

from config.configurator import override_config
from src.model.model_init import InferenceModelInitialiser
from src.inference.inference_model import InferenceModel


logger = logging.getLogger(__name__)


class GPTServer:
    """This class is used to serve an InferenceModel"""
    def __init__(self, config_file):
        configs = self.init_server(config_file)
        logger.debug(configs.job_config)

        model_init = InferenceModelInitialiser(configs)
        initialised = model_init.init_model()
        logger.info("GPT-2 model loaded")
        configs = model_init.configs  # some configs might have been changed when loading a model

        self.inference_model = InferenceModel(configs)
        self.inference_model.init_inference(initialised.model, initialised.checkpoint)
        logger.info("GPT-2 inference ready")
        
    def init_server(self, config_file):
        Configs = namedtuple('Configs', 'job_config context')
        job_config = override_config(config_file, inference=True)

        torch.manual_seed(job_config.seed)
        torch.cuda.manual_seed(job_config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[job_config.dtype]
        ctx = nullcontext() if job_config.device_type == 'cpu' else torch.amp.autocast(device_type=job_config.device_type, dtype=ptdtype)

        return Configs(job_config, ctx)

    def generate_sample(self, prompt_txt):
        out = self.inference_model.generate_sample(prompt_txt)
        return out
