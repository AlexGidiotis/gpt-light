"""
Poor Man's Configurator
"""

import logging
import yaml
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False # if True, script exits right after the first eval
    always_save_checkpoint: bool = True # if True, always save a checkpoint after each eval
    init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log: bool = False # disabled by default
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2' # 'run' + str(time.time())
    # data
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5 # used to simulate larger batch sizes
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024
    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    ddp: bool = False
    ddp_local_rank: int = -1
    # system
    master_process: bool = True
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device_type: str = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    dtype: str = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile_model: bool = True # use PyTorch 2.0 to compile the model to be faster
    

def override_config(config_file=None):
    loaded_configs = {}
    if config_file:
        logger.info(f"Overriding config with {config_file}:")
        with open(config_file, "r") as stream:
            loaded_configs = yaml.safe_load(stream)

    job_config = JobConfig(**loaded_configs)
    
    return job_config
