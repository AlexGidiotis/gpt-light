"""Creates a FastAPI service that serves GPT"""
import logging

from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI

from src.inference.gpt_server import GPTServer


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = FastAPI()
models = {}


class Prompt(BaseModel):
    prompt_txt: Union[str, None] = None


@app.on_event("startup")
async def startup_event():
    models["gpt2_model"] = GPTServer(config_file="config/sample_shakespeare.yml")

    
@app.get("/")
def read_root():
    return {"Hello": "From Light ChatGPT"}


@app.post("/lightchatgpt/")
async def read_item(prompt: Prompt):
    out = models["gpt2_model"].generate_sample(prompt.prompt_txt)
    return {"prompt": prompt.prompt_txt, "out": out}
