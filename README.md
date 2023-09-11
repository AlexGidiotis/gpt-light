# GPT-light

## TLDR
The goal of this repo is to provide a simple implementation of the GPT models that can be used for production applications. The code currently supports the following use cases:

1) Training a GPT architecture from scratch.
2) Fine-tuning an existing GPT model (e.g. the open source GPT2 models) on new data.
3) Using a trained GPT model to make batch predictions.
4) Deploying a trained GPT model as a REST API for serving.

Feel free to fork this repo and use it as a template for developing your own GPT applications.

### Future Work

We want to eventualy include the RL-based fine-tuning of ChatGPT. In general this does not change the underlying knowledge model but makes the interaction with the model more human-like.

## Setup


### Conda
Setting up the code is fairly minimal and can be easily reproduced in local or cloud VM environments.\*

1. Install Pytorch (with optional GPU support)
2. Create Conda environment:
```
conda create -n python39 python=3.9
conda activate python39
```
3. Install dependencies:
```
conda install --file requirements.txt
```

\* In the future we will try to provide a docker version of this setup.
### Docker
You could create your docker through the dockerfile provided under the docker directory and build your docker using the following command if you wish to use gpu:

**Make sure you have the nvidia-container-toolkit**

**GPU**
`docker build -t <name:tag> -f ./docker/Dockerfile-gpu .`

and run your docker using the following command if you use the gpu dockerfile:
`docker run --gpus all -it --name "container_name" -v /project/path/:/home/project/path <image_id> bash`

**CPU**

`docker build -t <name:tag> -f ./docker/Dockerfile-cpu .`

The conda environment is still created in the docker under the path `/opt/gpt-light`. Feel free to change your env name. You can run your scripts by invoking `/opt/gpt-light/bin/python script.py`

Remember to export the project's dir in PYTHONPATH
`export PYTHONPATH=/gpt-light/:$PYTHONPATH`

## Running things
### Preparing tiny-shakespeare data

We provide sample code that downloads and preprocesses the tiny-shakespeare dataset for training and fine-tuning. Use this code as a template to create data preprocessing for your own data.
```
python src/data_io/fetch_shakespeare.py
```

### Training tiny-shakespeare

Training a very small GPT model from scratch on the tiny-shakespeare dataset created in the previous step. The resulting model won't be very generalizable as it is rather small and trained on a very small dataset but it can generate Shakespeare-like quotes.

Change the configuration file and in order to train a model from scratch on your own data. At this point you can technically scale to very large model sizes according to your data size and resources.\*

```bash
python src/training/train_main.py --config_file config/train_shakespeare_small.yml
```

\* For this step you will probably require one or multiple GPUs.

### Fine-tuning GPT2 on shakespeare

Fine-tuning the open source GPT2 model on your own data. It is possible to use any open source GPT-like model (gpt2-medium, gpt2-xl etc). This is the more common option as it requires fewer data and resources (it is possible to run this even on a CPU) and is way faster than training from scratch.
```
python src/training/train_main.py --config_file config/finetune_shakespeare.yml
```

### Sampling from the shakespeare model

Sample a number of outputs from a trained model given an input prompt. This can be used for batch inference.
```
python src/inference/sample_main.py --config_file config/sample_shakespeare.yml
```

### Start server

A lightweigth server that serves a trained model in a REST API.
```
uvicorn src.inference.service_main:app --reload --reload-include config/sample_gpt2.yml
```

Once the server is running you can query the endpoint with a prompt (as a POST request). For more details on the query specifics have a look at the following test script.

#### Test endpoint
Once your server is running, you can test the endpoint using the following script.
```
python src/inference/test_query.py
```

## Testing
This repo has been deployed having a production application in mind and includes unit and integration testing.

\* Tesing is not extensive atm but we will try to increase coverage in the future.

### Run unit tests
Unit tests have been included to cover the main functionality. You can run the full unit testing suite with the following command:

```
pytest test/unit/
```

### Run integration tests
In addition to unit tests we have included some integration tests for the training, fine-tuning and sampling workflows.

```
pytest test/integration/
```

\* Integration testing currently requires data to work (see above).

### Pre-commit setup
It is generally advisable to enable pre-commit hooks when working with the repo. Currently runs some basic formatting checks and runs the unit testing suite (but not the integration tests).

```
conda install -c conda-forge pre-commit
pre-commit install
```

## acknowledgements
The GPT implementation in this repo is inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repo by Andrej Karpathy. Our goal was to re-implement, re-structure and extend it with modules that make it easy to build different types of production applications on top of it.
