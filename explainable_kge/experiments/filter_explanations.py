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

def load_dataset(args):
    ds = model_utils.load_dataset(args)
    ds.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        ds.reload_er_vocab()
    ds.load_current_ents_rels()
    ds.load_current_ents_rels()
    ds.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    return ds


def not_in_filter_triples(triples, filter_triples):
    filtered_triples = []
    for triple in triples:
        if not any(np.equal(triple, filter_triples).all(1)):
            filtered_triples.append(triple.tolist())
    return np.asarray(filtered_triples)


def in_filter_triples(triples, filter_triples):
    filtered_triples = []
    for triple in triples:
        if any(np.equal(triple, filter_triples).all(1)):
            filtered_triples.append(triple.tolist())
    return np.asarray(filtered_triples)


def report_counts(triples, r2i, detail_str):
    rel_counts = [0 for _ in r2i.keys()]
    for triple in triples:
        rel_counts[triple[1]] += 1
    print("Number of " + detail_str + ":")
    for r, r_id in r2i.items():
        print(r + ": " + str(rel_counts[r_id]))


def remove_triples(triples, remove_triples):
    mask = np.zeros(shape=(triples.shape[0]), dtype=bool)
    for remove_triple in remove_triples:
        mask = mask | np.equal(triples, remove_triple).all(1)
    masked_triples = np.delete(triples, mask, axis=0)
    return masked_triples


def extract_triples(triples, valid_triples):
    mask = np.zeros(shape=(triples.shape[0]), dtype=bool)
    for valid_triple in valid_triples:
        mask = mask | np.equal(triples, valid_triple).all(1)
    masked_triples = triples[mask,:]
    return masked_triples


def setup_experiment(args):
    # loads current (corrutped) train, valid, test sets
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    tr_dataset = load_dataset(train_args)
    dev_args = copy(args)
    dev_args["dataset"]["set_name"] = "0_valid2id"
    dev_args["continual"]["session"] = 0
    de_dataset = load_dataset(dev_args)
    test_args = copy(args)
    test_args["dataset"]["set_name"] = "0_test2id"
    test_args["continual"]["session"] = 0
    te_dataset = load_dataset(test_args)
    # loads clean train, valid, +/- test sets
    dirty_ds_name = copy(args["dataset"]["name"])
    clean_ds_name = dirty_ds_name.split("_")[0] + "_CLEAN_" + dirty_ds_name.split("_")[-1]
    clean_train_args = copy(args)
    clean_train_args["dataset"]["name"] = clean_ds_name
    clean_train_args["dataset"]["set_name"] = "0_train2id"
    clean_train_args["continual"]["session"] = 0
    clean_tr_dataset = load_dataset(clean_train_args)
    clean_dev_args = copy(args)
    clean_dev_args["dataset"]["name"] = clean_ds_name
    clean_dev_args["dataset"]["set_name"] = "0_valid2id"
    clean_dev_args["continual"]["session"] = 0
    clean_de_dataset = load_dataset(clean_dev_args)
    clean_test_args = copy(args)
    clean_test_args["dataset"]["name"] = clean_ds_name
    clean_test_args["dataset"]["set_name"] = "0_test2id"
    clean_test_args["continual"]["session"] = 0
    clean_te_dataset = load_dataset(clean_test_args)
    clean_test_neg_args = copy(args)
    clean_test_neg_args["dataset"]["name"] = clean_ds_name
    clean_test_neg_args["dataset"]["set_name"] = "0_test2id_neg"
    clean_test_neg_args["continual"]["session"] = 0
    clean_te_neg_dataset = load_dataset(clean_test_neg_args)
    # makes experiment path
    fp = os.path.join("explainable_kge/logger/logs", args["dataset"]["name"] + "_" + args["model"]["name"] + "_" + str(args["logging"]["log_num"]))
    if not os.path.exists(fp):
        os.makedirs(fp)
    fp = os.path.abspath(fp)

    return tr_dataset, de_dataset, te_dataset, clean_tr_dataset, clean_de_dataset, clean_te_dataset, clean_te_neg_dataset, fp


def load_corrupt_json(args, main_fp):
    sfe_fp = os.path.join(main_fp, "results")
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    json_name = "explanations_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str + "_clean"
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
    tr_d, de_d, te_d, clean_tr_d, clean_de_d, clean_te_d, clean_te_neg_d, exp_fp = setup_experiment(exp_config)
    
    # LOAD EXPLANATIONS
    corr_json = load_corrupt_json(exp_config, exp_fp)

    # GET REMOVES AND ADDS in explanations
    removes = []
    adds = []
    for example in corr_json:
        if example["y_bb"] != example["y_xm"] or (example["y_bb"] == example["y_xm"] and not example["y_xm"]):
            continue
        for part in example["parts"]:
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
            removes.append([hi,ri,ti])
            for triple_str in part["correction_triples"]:
                split_fact = triple_str.split(",")
                r = split_fact[1]
                if r[0] == "_":
                    h = split_fact[-1]
                    t = split_fact[0]
                else:
                    h = split_fact[0]
                    t = split_fact[-1]
                adds.append([tr_d.e2i[h],tr_d.r2i[r.replace("_","")],tr_d.e2i[t]])
    # DEBUG: SANITY CHECK
    # clean_gt_triples = clean_tr_d.load_triples(["0_gt2id.txt"])
    # trde_triples = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    # removes = not_in_filter_triples(trde_triples, clean_gt_triples)
    # adds = copy(clean_gt_triples)
    # DEBUG: SANITY CHECK

    # FILTER REMOVES
    removes = np.asarray(removes)
    # include removes done by classification
    ghat_fp = os.path.join(exp_fp, "ghat.tsv")
    g_hat = pd.read_csv(ghat_fp, sep="\t", header=None).to_numpy(dtype=str)
    g_hat = np.asarray([[tr_d.e2i[triple[0]],tr_d.r2i[triple[1]],tr_d.e2i[triple[-1]]] for triple in g_hat])
    trde_triples = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    removes = np.append(removes, not_in_filter_triples(trde_triples, g_hat), axis=0)
    print("Number of total removes: " + str(removes.shape[0]))
    removes = np.unique(removes, axis=0)
    print("Number of unique removes: " + str(removes.shape[0]))
    # filter out clean gt set removes
    clean_gt_triples = clean_tr_d.load_triples(["0_gt2id.txt"])
    non_clean_triples = not_in_filter_triples(removes, clean_gt_triples)
    print("Number of removes after removing gt pos set: " + str(non_clean_triples.shape[0]))
    # report removes in explanations
    report_counts(non_clean_triples, tr_d.r2i, "removes per relation type in explanations")
    # report corruptions in dataset
    trde_triples = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    nongt_triples = not_in_filter_triples(trde_triples, clean_gt_triples)
    report_counts(nongt_triples, tr_d.r2i, "removes per relation type in train/valid triple set")

    # FILTER ADDS
    # include clean gt set removes to adds
    adds = np.asarray(adds)
    adds = np.append(adds, in_filter_triples(removes, clean_gt_triples), axis=0)
    print("Number of total adds: " + str(adds.shape[0]))
    adds = np.unique(adds, axis=0)
    print("Number of unique adds: " + str(adds.shape[0]))
    # filter out gt test triples (FP in test)
    nontegt_adds = not_in_filter_triples(adds, clean_te_d.triples)
    print("Number of adds after removing gt pos test set: " + str(nontegt_adds.shape[0]))
    # filter out non gt clean train/valid (other FP)
    gttrde_only_adds = in_filter_triples(nontegt_adds, clean_gt_triples)
    print("Number of adds after removing non gt clean train/valid set: " + str(gttrde_only_adds.shape[0]))
    report_counts(gttrde_only_adds, tr_d.r2i, "adds per relation type in explanations")

    # generate dataset with X% corrupions
    tr_triples = tr_d.triples
    de_triples = de_d.triples
    gt_triples = tr_d.load_triples(["0_gt2id.txt"])

    # set X% for removes and adds
    partial = int(exp_config["feedback"]["noise_reduction_rate"] * non_clean_triples.shape[0])
    partial_idxs = np.random.choice(np.arange(non_clean_triples.shape[0]), partial, False)
    non_clean_cleanteneg_triples_partial = non_clean_triples[partial_idxs,:]
    partial = int(exp_config["feedback"]["correction_rate"] * gttrde_only_adds.shape[0])
    partial_idxs = np.random.choice(np.arange(gttrde_only_adds.shape[0]), partial, False)
    gttrde_only_adds_partial = gttrde_only_adds[partial_idxs,:]

    # generate tr & de dataset with X% corrupions
    # first delete X% corrupt triples
    tr_triples_updated = remove_triples(tr_triples, non_clean_cleanteneg_triples_partial)
    print("Number of train set triples after remove corrupt: " + str(tr_triples_updated.shape[0]))
    de_triples_updated = remove_triples(de_triples, non_clean_cleanteneg_triples_partial)
    print("Number of dev set triples after remove corrupt: " + str(de_triples_updated.shape[0]))
    gt_triples_updated = remove_triples(gt_triples, non_clean_cleanteneg_triples_partial)
    print("Number of gt set triples after remove corrupt: " + str(gt_triples_updated.shape[0]))
    # then add X% correct triples
    tr_triples_add = extract_triples(gttrde_only_adds_partial, clean_tr_d.triples)
    print("Number of triples to add to train set: " + str(tr_triples_add.shape[0]))
    de_triples_add = extract_triples(gttrde_only_adds_partial, clean_de_d.triples)
    print("Number of triples to add to valid set: " + str(de_triples_add.shape[0]))
    tr_triples_updated = np.unique(np.append(tr_triples_updated, tr_triples_add, axis=0), axis=0)
    print("New num train triples: " + str(tr_triples_updated.shape[0]))
    de_triples_updated = np.unique(np.append(de_triples_updated, de_triples_add, axis=0), axis=0)
    print("New num valid triples (+): " + str(de_triples_updated.shape[0]))
    gt_triples_updated = np.unique(np.append(gt_triples_updated, gttrde_only_adds_partial, axis=0),axis=0)
    print("New num gt triples: " + str(gt_triples_updated.shape[0]))

    # output the partially corrupted dataset
    tr_triples_updated[:, [1, 2]] = tr_triples_updated[:, [2, 1]]
    de_triples_updated[:, [1, 2]] = de_triples_updated[:, [2, 1]]
    gt_triples_updated[:, [1, 2]] = gt_triples_updated[:, [2, 1]]
    partial_str1 = str(int(exp_config["feedback"]["noise_reduction_rate"] * 100))
    partial_str2 = str(int(exp_config["feedback"]["correction_rate"] * 100))
    original_dataset_fp = tr_d.fp[:-1]
    # partially_corrupted_dataset_fp = original_dataset_fp + "_GT_DENOISED_" + partial_str1 + "_CORRECTED_" + partial_str2
    partially_corrupted_dataset_fp = original_dataset_fp + "_DENOISED_" + partial_str1 + "_CORRECTED_" + partial_str2
    if not os.path.exists(partially_corrupted_dataset_fp):
        shutil.copytree(original_dataset_fp, partially_corrupted_dataset_fp)
        with open(partially_corrupted_dataset_fp + "/0_train.txt", "w") as f:
            for triple in tr_triples_updated:
                str_triple = [tr_d.i2e[triple[0]],tr_d.i2e[triple[1]],tr_d.i2r[triple[2]]]
                f.write("\t".join(str_triple) + "\n")
        with open(partially_corrupted_dataset_fp + "/0_train2id.txt", "w") as f:
            f.write(str(len(tr_triples_updated)) + "\n")
            for triple in tr_triples_updated:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
        with open(partially_corrupted_dataset_fp + "/0_valid2id.txt", "w") as f:
            f.write(str(len(de_triples_updated)) + "\n")
            for triple in de_triples_updated:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
        with open(partially_corrupted_dataset_fp + "/0_gt2id.txt", "w") as f:
            for triple in gt_triples_updated:
                str_triple = [str(item) for item in triple]
                f.write("\t".join(str_triple) + "\n")
