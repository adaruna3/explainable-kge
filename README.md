# Explainable Knowledge Graph Embedding (XKGE)
Supporting code for [Explainable Knowledge Graph Embedding: Inference Reconciliation for Knowledge Inferences Supporting Robot Actions]().

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
5. The `pra` submodule requires [sbt](https://www.scala-sbt.org/download.html). Make sure to install 
it and run `sbt test` inside the `pra` submodule.

## Check install
1. After activating the conda environment, run `python`. Python version 3.6 should run. Next, check if `import torch` works.
Next, for GPU usage check if `torch.cuda.is_available()` is `True`.
2. If all these checks passed, the installation should be working.

## Repo Conents

* Knowledge Graph Embedding models: [TuckER](https://arxiv.org/pdf/1901.09590.pdf) implemented [here](https://github.com/adaruna3/explainable-kge/blob/dev/explainable_kge/models/standard_models.py)
* Interpretable Graph Feature models: [XKE](https://github.com/arthurcgusmao/XKE) implemented [here](https://github.com/adaruna3/explainable-kge/blob/74e2f968dff7c17a230ad6f75bdbcdbdab938b4a/explainable_kge/models/explain_utils.py#L1791), and our approach [XKGE](https://github.com/adaruna3/explainable-kge/blob/74e2f968dff7c17a230ad6f75bdbcdbdab938b4a/explainable_kge/models/explain_utils.py#L1809)
* Sub-graph Feature Extraction (SFE): [SFE](https://aclanthology.org/D15-1173.pdf) implemented [here](https://github.com/adaruna3/pra/tree/786f93213b054b1c3ba33a82283b4ccaca5f34b7)
* Datasets: [VH+_CLEAN_RAN](https://github.com/adaruna3/explainable-kge/tree/dev/explainable_kge/datasets/VH+_CLEAN_RAN/), which is <img src="https://render.githubusercontent.com/render/math?math=D}"> in the paper, and [VH+_CORR_RAN](https://github.com/adaruna3/explainable-kge/tree/dev/explainable_kge/datasets/VH+_CORR_RAN/), which is <img src="https://render.githubusercontent.com/render/math?math=\hat{D}}"> in the paper.

    
## Run the Paper Experiments
1. Evaluation of Interpretable Graph Feature Model
    - Run [alg_eval.sh](https://github.com/adaruna3/explainable-kge/blob/dev/explainable_kge/experiments/scripts/alg_eval.sh) using: `./explainable_kge/experiments/scripts/alg_eval.sh`
    - If this script runs correctly, it will produce 5 output folders in `./explainable_kge/logger/logs`, one for each fold of cross-validation: `VH+_CLEAN_RAN_tucker_[X]` where [X] is the fold number 0 to 4. Additionally, inside `./explainable_kge/logger/logs/VH+_CLEAN_RAN_tucker_0/` there should be 3 PDF files, each containing a set of results corresponding to the last 4 rows of the table (the last row and 4th from last row are repeats since our model without locality or decision trees is XKE)
    - There is a possibility that some of the folds error out when running due to internal configuations of SFE code, resulting in missing PDF result files. This will be seen from the Python Stack Trace as one of the runs involving `explainable_setting.py` in `alg_eval.sh` failing. That can be fixed by removing the entire `VH+_CLEAN_RAN_tucker_[X]` directory and re-running the commands in `./explainable_kge/experiments/scripts/alg_eval.sh` to generate the directory followed by the corresponding plotting comamnd to generate the PDF. Please see [alg_eval.sh](https://github.com/adaruna3/explainable-kge/blob/dev/explainable_kge/experiments/scripts/alg_eval.sh).
2. 

## ToDos
