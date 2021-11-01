# Explainable Knowledge Graph Embedding (XKGE)
Supporting code for [XKGE]() paper.

## Pre-requisites
1. This repo has been tested for a system running Ubuntu 18.04 LTS, PyTorch (1.2.0), and 
hardware CPU or Nvidia GPU (GeForce GTX 1060 6GB or better).
2. For GPU functionality Nvidia drivers, CUDA, and cuDNN are required.

## Installation
1. To install using conda run: `conda env create -f xkge_env.yml` in the repo root and activate
environment via `conda activate xkge_env`

## Check install
After activating the environment, run `python`. Python version 3.6 should run. Next, check if `import torch` works.
Next, for GPU usage check if `torch.cuda.is_available()` is `True`. If all these checks passed, the installation should
be working.

## Repo Conents

    
## Run the Paper Experiments


## ToDos
I. Try different versions of logit
    e. Plot all results
    d. Run c. for the FB dataset
II. Add toggle for using decision tree
    a. How to generate dataset
III. Add toggle for enrich the knowledge graph
    a. add reverse and similarity relations to specific entities
IV. Add methods to corrupt the dataset
V. Check explanations coming out of corrupted dataset