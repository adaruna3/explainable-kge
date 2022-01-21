import os
from copy import copy, deepcopy
import pickle
import numpy as np
import pandas as pd
import torch
import tqdm
import json
import shutil

import __main__  # used to get the original execute module

from explainable_kge.models import model_utils
from explainable_kge.logger import viz_utils
from explainable_kge.models import explain_utils as x_utils
from explainable_kge.logger.terminal_utils import logout, load_config

import pdb


"""
load exp json
filter exp json for train facts the were corrupted
select X% fraction of corrupted train facts
make new dataset with X% fraction of corrupted train facts subsituted for train facts
make new dataset with 100% fraction of corrupted train facts subsituted for train facts
"""

def setup_experiment(args):
    # loads the training and valid dataset
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    tr_dataset = model_utils.load_dataset(train_args)
    tr_dataset.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        tr_dataset.reload_er_vocab()
    tr_dataset.load_current_ents_rels()
    tr_dataset.load_current_ents_rels()
    tr_dataset.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    dev_args = copy(args)
    dev_args["dataset"]["set_name"] = "0_valid2id"
    dev_args["continual"]["session"] = 0
    de_dataset = model_utils.load_dataset(dev_args)
    de_dataset.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        de_dataset.reload_er_vocab()
    de_dataset.load_current_ents_rels()
    de_dataset.load_current_ents_rels()
    de_dataset.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    test_args = copy(args)
    test_args["dataset"]["set_name"] = "0_test2id"
    test_args["continual"]["session"] = 0
    te_dataset = model_utils.load_dataset(test_args)
    te_dataset.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        te_dataset.reload_er_vocab()
    te_dataset.load_current_ents_rels()
    te_dataset.load_current_ents_rels()
    te_dataset.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs

    # makes experiment path
    fp = os.path.join("explainable_kge/logger/logs", args["dataset"]["name"] + "_" + args["model"]["name"] + "_" + str(args["logging"]["log_num"]))
    if not os.path.exists(fp):
        os.makedirs(fp)
    fp = os.path.abspath(fp)

    return tr_dataset, de_dataset, te_dataset, fp


def load_corrupt_json(args, main_fp):
    sfe_fp = os.path.join(main_fp, "results")
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    json_name = "explanations_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str
    json_fp = os.path.join(sfe_fp, json_name + '.json')
    with open(json_fp, "r") as f:
        corrupt_json = json.load(f)
    return corrupt_json


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
    tr_d, de_d, te_d, exp_fp = setup_experiment(exp_config)
    # load exp json
    corr_json = load_corrupt_json(exp_config, exp_fp)
    # get unique json corrupted facts
    updates = []
    for example in corr_json:
        split_fact = example["fact"].split(",")
        r = split_fact[1]
        if r[0] == "_":
            h = split_fact[-1]
            t = split_fact[0]
        else:
            h = split_fact[0]
            t = split_fact[-1]
        hi = tr_d.e2i[h]
        ri = tr_d.r2i[r.replace("_","")]
        ti = tr_d.e2i[t]
        split_fact_ = example["fact_corrupted"].split(",")
        r_ = split_fact_[1]
        if r_[0] == "_":
            h_ = split_fact_[-1]
            t_ = split_fact_[0]
        else:
            h_ = split_fact_[0]
            t_ = split_fact_[-1]
        hi_ = tr_d.e2i[h_]
        ri_ = tr_d.r2i[r_.replace("_","")]
        ti_ = tr_d.e2i[t_]
        updates.append([hi,ri,ti,hi_,ri_,ti_])
        for part in example["parts"]:
            if part["correct_id"] != -1:
                split_fact = part["fact"].split(",")
                r = split_fact[1]
                if r[0] == "_":
                    h = split_fact[-1]
                    t = split_fact[0]
                else:
                    h = split_fact[0]
                    t = split_fact[-1]
                hi = tr_d.e2i[h]
                ri = tr_d.r2i[r.replace("_","")]
                ti = tr_d.e2i[t]
                split_fact_ = part["fact_corrupted"].split(",")
                r_ = split_fact_[1]
                if r_[0] == "_":
                    h_ = split_fact_[-1]
                    t_ = split_fact_[0]
                else:
                    h_ = split_fact_[0]
                    t_ = split_fact_[-1]
                hi_ = tr_d.e2i[h_]
                ri_ = tr_d.r2i[r_.replace("_","")]
                ti_ = tr_d.e2i[t_]
                updates.append([hi,ri,ti,hi_,ri_,ti_])
    updates = np.unique(updates, axis=0)
    # filter out test set updates
    te_triples = te_d.triples
    nontest_updates = []
    for update in updates:
        if not any(np.equal(te_triples,update[0:3]).all(1)) and not any(np.equal(te_triples,update[3:]).all(1)):
            nontest_updates.append(update.tolist())
    # filter out valid set updates
    nontest_updates = np.asarray(nontest_updates)
    de_triples = de_d.triples
    nontede_updates = []
    for update in nontest_updates:
        if not any(np.equal(de_triples,update[0:3]).all(1)) and not any(np.equal(de_triples,update[3:]).all(1)):
            nontede_updates.append(update.tolist())
    nontede_updates = np.asarray(nontede_updates)
    # report corruptions
    rel_counts = [0 for _ in tr_d.r2i.keys()]
    for update in nontede_updates:
        rel_counts[update[1]] += 1
    print("Number of corruptions per relation type:")
    for r, r_id in tr_d.r2i.items():
        print(r + ": " + str(rel_counts[r_id]))
    tr_triples = tr_d.triples
    gt_triples = tr_d.load_triples(["gt2id.txt"])
    # generate dataset with 100% corrupions
    # first delete sampled triples
    mask = np.zeros(shape=(tr_triples.shape[0]), dtype=bool)
    for update in nontede_updates:
        mask = mask | np.equal(tr_triples, update[0:3]).all(1)
    tr_triples_updated = np.delete(tr_triples, mask, axis=0)
    # then add corrupted triples
    new_triples = np.unique(nontede_updates[:,3:], axis=0)
    tr_triples_updated_ = np.append(tr_triples_updated, new_triples, axis=0)
    # first delete sampled triples
    mask = np.zeros(shape=(gt_triples.shape[0]), dtype=bool)
    for update in nontede_updates:
        mask = mask | np.equal(gt_triples, update[0:3]).all(1)
    gt_triples_updated = np.delete(gt_triples, mask, axis=0)
    # then add corrupted triples
    gt_triples_updated_ = np.append(gt_triples_updated, new_triples, axis=0)
    # output the fully corrupted dataset
    tr_triples_updated_[:, [1, 2]] = tr_triples_updated_[:, [2, 1]]
    gt_triples_updated_[:, [1, 2]] = gt_triples_updated_[:, [2, 1]]
    original_dataset_fp = tr_d.fp[:-1]
    fully_corrupted_dataset_fp = original_dataset_fp + "_CORR_100"
    if not os.path.exists(fully_corrupted_dataset_fp):
        shutil.copytree(original_dataset_fp, fully_corrupted_dataset_fp)
        with open(fully_corrupted_dataset_fp + "/0_train.txt", "w") as f:
            for triple in tr_triples_updated_:
                str_triple = [tr_d.i2e[triple[0]],tr_d.i2e[triple[1]],tr_d.i2r[triple[2]]]
                f.write("\t".join(str_triple) + "\n")
        with open(fully_corrupted_dataset_fp + "/0_train2id.txt", "w") as f:
            f.write(str(len(tr_triples_updated_)) + "\n")
            for triple in tr_triples_updated_:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
        with open(fully_corrupted_dataset_fp + "/gt2id.txt", "w") as f:
            f.write(str(len(gt_triples_updated_)) + "\n")
            for triple in gt_triples_updated_:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
    # generate dataset with X% corrupions
    partial = int(exp_config["feedback"]["corrupt_rate"] * nontede_updates.shape[0])
    nontede_updates_partial = nontede_updates[:partial,:]
    # first delete sampled triples
    mask = np.zeros(shape=(tr_triples.shape[0]), dtype=bool)
    for update in nontede_updates_partial:
        mask = mask | np.equal(tr_triples, update[0:3]).all(1)
    tr_triples_updated = np.delete(tr_triples, mask, axis=0)
    # then add corrupted triples
    new_triples = np.unique(nontede_updates_partial[:,3:], axis=0)
    tr_triples_updated_ = np.append(tr_triples_updated, new_triples, axis=0)
    # first delete sampled triples
    mask = np.zeros(shape=(gt_triples.shape[0]), dtype=bool)
    for update in nontede_updates_partial:
        mask = mask | np.equal(gt_triples, update[0:3]).all(1)
    gt_triples_updated = np.delete(gt_triples, mask, axis=0)
    # then add corrupted triples
    gt_triples_updated_ = np.append(gt_triples_updated, new_triples, axis=0)
    # output the partially corrupted dataset
    tr_triples_updated_[:, [1, 2]] = tr_triples_updated_[:, [2, 1]]
    gt_triples_updated_[:, [1, 2]] = gt_triples_updated_[:, [2, 1]]
    partial_str = str(int(exp_config["feedback"]["corrupt_rate"] * 100))
    partially_corrupted_dataset_fp = original_dataset_fp + "_CORR_" + partial_str
    if not os.path.exists(partially_corrupted_dataset_fp):
        shutil.copytree(original_dataset_fp, partially_corrupted_dataset_fp)
        with open(partially_corrupted_dataset_fp + "/0_train.txt", "w") as f:
            for triple in tr_triples_updated_:
                str_triple = [tr_d.i2e[triple[0]],tr_d.i2e[triple[1]],tr_d.i2r[triple[2]]]
                f.write("\t".join(str_triple) + "\n")
        with open(partially_corrupted_dataset_fp + "/0_train2id.txt", "w") as f:
            f.write(str(len(tr_triples_updated_)) + "\n")
            for triple in tr_triples_updated_:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
        with open(fully_corrupted_dataset_fp + "/gt2id.txt", "w") as f:
            f.write(str(len(gt_triples_updated_)) + "\n")
            for triple in gt_triples_updated_:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
