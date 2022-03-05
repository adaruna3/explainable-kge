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
    train_args["continual"]["session"] = str(args["logging"]["log_num"])
    tr_dataset = model_utils.load_dataset(train_args)
    dev_args = copy(args)
    dev_args["dataset"]["set_name"] = "0_valid2id"
    dev_args["continual"]["session"] = str(args["logging"]["log_num"])
    de_dataset = model_utils.load_dataset(dev_args)
    # combines datasets
    tr_de_dataset = deepcopy(tr_dataset)
    tr_de_dataset.triples = np.unique(np.concatenate((tr_dataset.triples,
                                                      de_dataset.triples), axis=0), axis=0)
    tr_de_dataset.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        tr_de_dataset.reload_er_vocab()
    tr_de_dataset.load_current_ents_rels()
    tr_de_dataset.load_current_ents_rels()
    tr_de_dataset.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs

    # load the gt clean triples
    dirty_ds_name = copy(args["dataset"]["name"])
    clean_ds_name = dirty_ds_name.split("_")[0] + "_CLEAN_" + dirty_ds_name.split("_")[-1]
    clean_args = copy(args)
    clean_args["dataset"]["name"] = clean_ds_name
    clean_args["dataset"]["set_name"] = "0_train2id"
    clean_args["continual"]["session"] = str(args["logging"]["log_num"])
    clean_dataset = model_utils.load_dataset(clean_args)
    clean_triples = clean_dataset.load_triples(["0_gt2id.txt"], num_skip=0)

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
    model_optim_args["sess"] = str(args["logging"]["log_num"])
    model = model_utils.load_model(model_optim_args, model)

    # makes experiment path
    fp = os.path.join("explainable_kge/logger/logs", args["dataset"]["name"] + "_" + args["model"]["name"] + "_" + str(args["logging"]["log_num"]))
    if not os.path.exists(fp):
        os.makedirs(fp)
    fp = os.path.abspath(fp)

    return model, tr_de_dataset, clean_triples, fp
    # return model, tr_de_dataset, fp


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
    exp_model, tr_de_d, clean_gt_triples, exp_fp = setup_experiment(exp_config)
    # exp_model, tr_de_d, exp_fp = setup_experiment(exp_config)
    # 1. Generate G^
    #     a. For each training triple, perturb with NN to be all possible triples in G^
    ent_embeddings = exp_model.E.weight.cpu().detach().numpy()
    ent_embeddings = ent_embeddings / np.linalg.norm(ent_embeddings, ord=2, axis=1, keepdims=True)
    knn = x_utils.get_typed_knn(ent_embeddings, tr_de_d.e2i, k=exp_config["explain"]["knn_k"])
    #     b. Prepare to classify triples as T/F with embedding
    rel_thresholds = model_utils.get_rel_thresholds(exp_config, exp_model)
    #     c. Form G^ from all T triples classified by embedding
    ghat_fp = os.path.join(exp_fp, "ghat.tsv")
    if not os.path.exists(ghat_fp):
        max_nbrs = (exp_config["explain"]["ghat_head_k"], exp_config["explain"]["ghat_tail_k"])
        g_hat = x_utils.generate_ghat(exp_config, knn, tr_de_d, exp_model, rel_thresholds, device, max_nbrs, ghat_fp)
    else:
        g_hat = pd.read_csv(ghat_fp, sep="\t", header=None).to_numpy(dtype=str)
    g_hat_dicts = x_utils.process_ghat(g_hat, tr_de_d.e2i, tr_de_d.i2e, tr_de_d.r2i, tr_de_d.i2r, clean_gt_triples)
    # 2. Run SFE on G^
    sfe_fp = os.path.join(exp_fp, "results")
    if not os.path.exists(os.path.join(sfe_fp,exp_config["model"]["name"])):
        split_fp = os.path.join(exp_fp, "splits")
        split_name = exp_config["model"]["name"]
        x_utils.run_sfe(exp_config, exp_model, device, 
                        rel_thresholds, tr_de_d.i2e, tr_de_d.i2r, 
                        split_fp, split_name, ghat_fp, exp_fp)
    # 3. Train explainable model to predict each test triple
    locality_str = str(exp_config["explain"]["locality_k"]) if type(exp_config["explain"]["locality_k"]) == int else "best"
    results_fp = os.path.join(sfe_fp, "{}.pkl".format(exp_config["explain"]["xmodel"] + "_" + exp_config["explain"]["locality"] + "_" + locality_str))
    if not os.path.exists(results_fp):
        results = x_utils.get_explainable_results(exp_config, knn, exp_config["explain"]["locality_k"],
                                                  tr_de_d.r2i, tr_de_d.e2i, tr_de_d.i2e,
                                                  sfe_fp, results_fp, ent_embeddings,
                                                  exp_model, g_hat_dicts, device)
    else:
        with open(results_fp, "rb") as f:
            results = pickle.load(f)
    viz_utils.get_summary(results)
