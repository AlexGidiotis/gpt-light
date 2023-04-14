# chatgpt-light

## Setup
1. Install Pytorch
2. Create Conda environment:
```
conda create -n python38 python=3.8
conda activate python38
```
3. Install dependencies:
``` conda install --file requirements.txt ```

## Running things
### Preparing tiny-shakespeare data
``` python data/tinyshakespeare/fetch_shakespeare.py ```

### Training tiny-shakespeare
``` python src/training/train_main.py --config_file config/train_shakespeare_small.yml ```

### Fine-tuning GPT2 on shakespeare
``` python src/training/train_main.py --config_file config/finetune_shakespeare.yml ```

### Sampling from the shakespeare model
``` python src/inference/sample_main.py --config_file config/sample_shakespeare.yml ```
