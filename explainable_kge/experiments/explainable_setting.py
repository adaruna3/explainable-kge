import os
from copy import copy, deepcopy
import pickle
import numpy as np
import pandas as pd
import torch
import tqdm

import __main__  # used to get the original execute module

from explainable_kge.models import model_utils
from explainable_kge.logger import viz_utils
from explainable_kge.models import explain_utils as x_utils
from explainable_kge.logger.terminal_utils import logout, load_config

import pdb


def setup_experiment(args):
    # loads the training and valid dataset
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    tr_dataset = model_utils.load_dataset(train_args)
    dev_args = copy(args)
    dev_args["dataset"]["set_name"] = "0_valid2id"
    dev_args["continual"]["session"] = 0
    de_dataset = model_utils.load_dataset(dev_args)
    # combines datasets
    tr_de_dataset = deepcopy(tr_dataset)
    tr_de_dataset.triples = np.unique(np.concatenate((tr_dataset.triples,
                                                      de_dataset.triples), axis=0), axis=0)
    tr_de_dataset.load_bernouli_sampling_stats()
    if args["dataset"]["reverse"]:
        tr_de_dataset.reload_er_vocab()
    tr_de_dataset.load_current_ents_rels()
    tr_de_dataset.load_current_ents_rels()
    tr_de_dataset.reverse = False

    # loads trained embedding model
    model_optim_args = copy(args)
    model_optim_args["model"]["num_ents"] = len(tr_dataset.e2i)
    model_optim_args["model"]["num_rels"] = len(tr_dataset.r2i)
    model = model_utils.init_model(model_optim_args)
    if model_optim_args["cuda"]:
        model.to(torch.device("cuda"), non_blocking=True)
    else:
        model.to(torch.device("cpu"), non_blocking=True)
    model_optim_args["tag"] = "standard_setting"
    model_optim_args["sess"] = str(0)
    model = model_utils.load_model(model_optim_args, model)

    # makes experiment path
    fp = os.path.join("explainable_kge/logger/logs", args["dataset"]["name"] + "_" + args["model"]["name"])
    if not os.path.exists(fp):
        os.makedirs(fp)
    fp = os.path.abspath(fp)

    return model, tr_de_dataset, fp


if __name__ == "__main__":
    # parse arguments
    exp_config = load_config("Standard setting experiment")
    # select hardware to use
    if exp_config["cuda"] and torch.cuda.is_available():
        logout("Running with CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        exp_config["cuda"] = False
        logout("Running with CPU, experiments will be slow", "w")
    # prepare experiment objects
    exp_model, tr_de_d, exp_fp = setup_experiment(exp_config)
    # 1. Generate G^
    #     a. For each training triple, perturb with NN to be all possible triples in G^
    ent_embeddings = exp_model.E.weight.cpu().detach().numpy()
    ent_embeddings = ent_embeddings / np.linalg.norm(ent_embeddings, ord=2, axis=1, keepdims=True)
    knn = x_utils.get_ent_embedding_knn(ent_embeddings, tr_de_d.e2i, k=10)
    #     b. Prepare to classify triples as T/F with embedding
    rel_thresholds = model_utils.get_rel_thresholds(exp_config, exp_model)
    #     c. Form G^ from all T triples classified by embedding
    ghat_fp = os.path.join(exp_fp, "ghat.tsv")
    if not os.path.exists(ghat_fp):
        x_utils.generate_ghat(exp_config, knn, tr_de_d, exp_model, rel_thresholds, device, ghat_fp)
    # 2. Run SFE on G^
    sfe_fp = os.path.join(exp_fp, "results", exp_config["model"]["name"])
    if not os.path.exists(sfe_fp):
        split_fp = os.path.join(exp_fp, "splits")
        split_name = exp_config["model"]["name"]
        x_utils.run_sfe(exp_config, exp_model, device, 
                        rel_thresholds, tr_de_d.i2e, tr_de_d.i2r, 
                        split_fp, split_name, ghat_fp, exp_fp)
    # 3. Train local explainable model for each test triple
    results_fp = os.path.join(sfe_fp, "{}.pkl".format(exp_config["explain"]["xmodel"] + "_" + exp_config["explain"]["locality"]))
    if not os.path.exists(results_fp):
        results = x_utils.get_explainable_results(exp_config, knn, 
                                                  tr_de_d.r2i, tr_de_d.e2i, tr_de_d.i2e,
                                                  sfe_fp, results_fp)
    else:
        with open(results_fp, "rb") as f:
            results = pickle.load(f)
    viz_utils.plot_ex_results(results, tr_de_d.r2i)
    pdb.set_trace()
