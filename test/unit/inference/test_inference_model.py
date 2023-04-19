import os

import pytest
from collections import namedtuple
from contextlib import nullcontext
import torch

from src.inference.inference_model import InferenceJobConfig, InferenceModel
from src.model.gpt_model import GPTConfig, GPT
from src.features.gpt_encoding import DataEncoder


@pytest.fixture(scope="module")
def init_inference_model():
    InitModel = namedtuple("InitModel", "inference_model data_encoder inference_config")
    Configs = namedtuple("Configs", "job_config context")
    job_config = InferenceJobConfig(
        start="\n",
        max_new_tokens=12,
        temperature=0.8,
        top_k=5,
        seed=1337,
        device="cpu",
        device_type="cpu",
        compile_model=False,
    )
    torch.manual_seed(job_config.seed)
    torch.cuda.manual_seed(job_config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[job_config.dtype]
    ctx = (
        nullcontext()
        if job_config.device_type == "cpu"
        else torch.amp.autocast(device_type=job_config.device_type, dtype=ptdtype)
    )
    inference_config = Configs(job_config, ctx)
    data_encoder = DataEncoder()
    model = GPT(GPTConfig())
    inference_model = InferenceModel(configs=inference_config)
    inference_model.init_inference(
        model=model,
    )

    return InitModel(inference_model, data_encoder, inference_config)


def test_generate_sample(init_inference_model):
    prompt = "\n"
    out = init_inference_model.inference_model.generate_sample(prompt)
    assert isinstance(out, str)


def test_generate_samples(init_inference_model):
    num_samples = 3
    out = init_inference_model.inference_model.generate_samples(num_samples)
    assert len(out) == 3
    for y in out:
        assert isinstance(y, str)
