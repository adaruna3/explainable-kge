import os
from copy import copy, deepcopy
import pickle
import numpy as np
import pandas as pd
import torch
import tqdm
import json
import random
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


def get_updates(ex_json, e2i, r2i):
    ex_adds = []
    ex_removes = []
    t2e = {}
    for i, example in enumerate(ex_json):
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
            hi = e2i[h]
            ri = r2i[r.replace("_","")]
            ti = e2i[t]
            ex_removes.append([hi,ri,ti])
            if (hi,ri,ti) in t2e:
                t2e[(hi,ri,ti)].append(i)
            else:
                t2e[(hi,ri,ti)] = [i]
            for triple_str in part["correction_triples"]:
                split_fact = triple_str.split(",")
                r = split_fact[1]
                if r[0] == "_":
                    h = split_fact[-1]
                    t = split_fact[0]
                else:
                    h = split_fact[0]
                    t = split_fact[-1]
                ex_adds.append([e2i[h],r2i[r.replace("_","")],e2i[t]])
                if (e2i[h],r2i[r.replace("_","")],e2i[t]) in t2e:
                    t2e[(e2i[h],r2i[r.replace("_","")],e2i[t])].append(i)
                else:
                    t2e[(e2i[h],r2i[r.replace("_","")],e2i[t])] = [i]
    return np.asarray(ex_adds), np.asarray(ex_removes), t2e


def filter_removes(ex_removes, main_fp, tr_d, de_d, clean_tr_d):
    # include removes done by classification
    ghat_fp = os.path.join(main_fp, "ghat.tsv")
    g_hat = pd.read_csv(ghat_fp, sep="\t", header=None).to_numpy(dtype=str)
    g_hat = np.asarray([[tr_d.e2i[triple[0]],tr_d.r2i[triple[1]],tr_d.e2i[triple[-1]]] for triple in g_hat])
    trde = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    ghat_removes = not_in_filter_triples(trde, g_hat)
    ex_removes = np.append(ex_removes, ghat_removes, axis=0)
    print("Number of total removes: " + str(ex_removes.shape[0]))
    ex_removes = np.unique(ex_removes, axis=0)
    print("Number of unique removes: " + str(ex_removes.shape[0]))
    # filter out clean gt set removes
    clean_gt = clean_tr_d.load_triples(["0_gt2id.txt"])
    ex_removes = not_in_filter_triples(ex_removes, clean_gt)
    print("Number of removes after removing gt pos set: " + str(ex_removes.shape[0]))
    # report removes in explanations
    report_counts(ex_removes, tr_d.r2i, "removes per relation type in explanations")
    return ex_removes, ghat_removes


def filter_adds(ex_adds, ex_removes, clean_tr_d, clean_te_d):
    # include clean gt set removes to adds
    clean_gt = clean_tr_d.load_triples(["0_gt2id.txt"])
    ex_adds = np.append(ex_adds, in_filter_triples(ex_removes, clean_gt), axis=0)
    print("Number of total adds: " + str(ex_adds.shape[0]))
    ex_adds = np.unique(ex_adds, axis=0)
    print("Number of unique adds: " + str(ex_adds.shape[0]))
    # filter out gt test triples (FP in test)
    ex_adds = not_in_filter_triples(ex_adds, clean_te_d.triples)
    print("Number of adds after removing gt pos test set: " + str(ex_adds.shape[0]))
    # filter out non gt clean train/valid (other FP)
    ex_adds = in_filter_triples(ex_adds, clean_gt)
    print("Number of adds after removing non gt clean train/valid set: " + str(ex_adds.shape[0]))
    report_counts(ex_adds, tr_d.r2i, "adds per relation type in explanations")
    return ex_adds


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


def store_corrupt_json(args, main_fp, exp_json, num):
    sfe_fp = os.path.join(main_fp, "results")
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    json_name = "explanations_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str + "_clean_filtered"
    json_fp = os.path.join(sfe_fp, json_name + '.json')
    with open(json_fp, "w") as f:
        random.shuffle(exp_json)
        json.dump(exp_json[:num], f)


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
    adds, removes, triple2example = get_updates(corr_json, tr_d.e2i, tr_d.r2i)
    # DEBUG: SANITY CHECK
    # clean_gt_triples = clean_tr_d.load_triples(["0_gt2id.txt"])
    # trde_triples = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    # removes = not_in_filter_triples(trde_triples, clean_gt_triples)
    # adds = copy(clean_gt_triples)
    # DEBUG: SANITY CHECK

    # FILTER REMOVES
    all_removes, classify_removes = filter_removes(removes, exp_fp, tr_d, de_d, clean_tr_d)
    # report corruptions in dataset as check
    trde_triples = np.unique(np.append(tr_d.triples, de_d.triples, axis=0), axis=0)
    clean_gt_triples = clean_tr_d.load_triples(["0_gt2id.txt"])
    nongt_triples = not_in_filter_triples(trde_triples, clean_gt_triples)
    report_counts(nongt_triples, tr_d.r2i, "removes per relation type in train/valid triple set")

    # FILTER ADDS
    all_adds = filter_adds(adds, np.unique(removes,axis=0), clean_tr_d, clean_te_d)
    
    # select explanations for AMT
    add_examples = []
    for add in all_adds:
        add_examples.append(triple2example[tuple(add)][0])
    remove_examples = []
    for remove in all_removes:
        try:
            remove_examples.append(triple2example[tuple(remove)][0])
        except:
            if in_filter_triples([remove],classify_removes).shape[0]:
                continue
            else:
                pdb.set_trace()
    amt_ex_ids = list(set(add_examples).union(set(remove_examples)))
    amt_exs = [corr_json[amt_ex_id] for amt_ex_id in amt_ex_ids]
    store_corrupt_json(exp_config, exp_fp, amt_exs, 330)
    exit()

    # generate dataset with X% corrupions
    tr_triples = tr_d.triples
    de_triples = de_d.triples
    gt_triples = tr_d.load_triples(["0_gt2id.txt"])

    # set X% for removes and adds
    partial = int(exp_config["feedback"]["noise_reduction_rate"] * all_removes.shape[0])
    partial_idxs = np.random.choice(np.arange(all_removes.shape[0]), partial, False)
    non_clean_cleanteneg_triples_partial = all_removes[partial_idxs,:]
    partial = int(exp_config["feedback"]["correction_rate"] * all_adds.shape[0])
    partial_idxs = np.random.choice(np.arange(all_adds.shape[0]), partial, False)
    gttrde_only_adds_partial = all_adds[partial_idxs,:]

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
