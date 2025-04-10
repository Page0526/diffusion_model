
<div align="center">

# Diffusion
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

At first, the repo is mainly about my process of implementing basic diffusion models from scratch when I'm studying diffusion models. Then, I started researching about image restoration problem and want to apply diffusion model, more specificially, on medical image, so this repo is more like a place where I record my researching journey. I may have made some (more like a lot :>) mistakes, I will certainly happy if you let me know about it. 

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd diffusion_model

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements using pip 
pip install -r requirements.txt
# or you can install requirements using conda
conda 
```
## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
You can test [MoDL](https://arxiv.org/pdf/1712.02862) by running this experiment & modify any params in it
```bash
python src/train.py experiment=modl.yaml
```
Or [DDIM](https://arxiv.org/pdf/2010.02502) by running this experiment & modify net params in it
```bash
python src/train.py experiment=diffusion.yaml
```
Or [DDPM](https://arxiv.org/abs/2006.11239) by running this experiment & modify net params in it
```bash
python src/train.py experiment=diffusion.yaml
```
You can see the result on wandb by run this script
```bash
export WANDB_API_KEY=
python src/train.py experiment=diffusion.yaml
```
## My result
I will update sooner or later :> (this is my happy face incase you don't know)
<br>
<sub>A little confession as usual. Recently, I have applied to lots of AI jobs, but I got rejected from all of them, which is sad. However, I believe that when one door shuts, another one opens. I want to try my best and learn a lot, so I won’t have any regrets one day. </sub>
