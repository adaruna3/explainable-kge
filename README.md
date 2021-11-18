# Explainable Knowledge Graph Embedding (XKGE)
Supporting code for [XKGE]() paper.

## Pre-requisites
1. This repo has been tested for a system running Ubuntu 18.04 LTS, PyTorch (1.2.0), and 
hardware CPU or Nvidia GPU (GeForce GTX 1060 6GB or better).
2. For GPU functionality Nvidia drivers, CUDA, and cuDNN are required.

## Installation
1. To install using conda run: `conda env create -f xkge_env.yml` in the repo root and activate
environment via `conda activate xkge_env`
2. Create a file `vim ~/[PATH_TO_ANACONDA]/envs/xkge_env2/lib/python3.6/site-packages/xkge.pth`
containing a single line with the absolute path to this repo. This file lets conda find the 
`explainable_kge` module when doing imports, see [here](https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath).
3. Get submodules: `git submodule update --init`
4. Modify the `cd` path in `explainable_kge/run_pra.sh` to the absolute path on your PC for `pra` submodule.

## Check install
1. After activating the environment, run `python`. Python version 3.6 should run. Next, check if `import torch` works.
Next, for GPU usage check if `torch.cuda.is_available()` is `True`. 
2. The `pra` submodule requires (sbt)[https://www.scala-sbt.org/download.html]. Make sure to install 
it and run `sbt test` inside the `pra` submodule.
3. If all these checks passed, the installation should be working.

## Repo Conents

    
## Run the Paper Experiments


## ToDos
I. Post-process explanation outputs
    c. Final grammar fixes
II. User-study Design
    b. Design the survey UI with data output
    c. Test with Devleena and others
    d. Put live
III. Improve methods to corrupt the dataset
IV. Try to get feedback and retrain KGE
