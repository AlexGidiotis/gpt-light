import sys
import os
from src.inference.inference_model import InferenceModel, InferenceModelInitialiser
from src.inference.sample_main import init_job


def test_gpt2():
    configs = init_job(
        ["--config_file", os.path.join(os.path.dirname(__file__), "test_sample.yml")]
    )

    model_init = InferenceModelInitialiser(configs)
    initialised = model_init.init_model()
    configs = model_init.configs

    inference_model = InferenceModel(configs)
    inference_model.init_inference(initialised.model, initialised.checkpoint)
    out = inference_model.generate_samples(configs.job_config.num_samples)
    assert len(out) == 1
    for y in out:
        assert isinstance(out, list)
        assert isinstance(out[0], str)
