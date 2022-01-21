from enum import unique
import os
import multiprocessing
import subprocess
from copy import copy, deepcopy
import itertools
import pickle
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import tqdm

from gingerit.gingerit import GingerIt
import torch

from explainable_kge.models import model_utils
from explainable_kge.logger.terminal_utils import logout
from explainable_kge.logger.viz_utils import figs2pdf

import pdb


def get_typed_knn(ent_embeddings, e2i=None, k=10):
    """
    Provides k nearest neighbors of entities in embedding space
    :param ent_embeddings:
    """
    knn = {}
    num_cpu = multiprocessing.cpu_count()
    if e2i is not None:
        ent_types = list(set([e[-1] for e in e2i.keys()]))
    else:
        ent_types = [None]
    for ent_type in ent_types:
        if ent_type is not None:
            same_type_ents = [i for e, i in e2i.items() if e[-1] == ent_type]
        else:
            same_type_ents = list(e2i.values())
        same_type_embeddings = ent_embeddings[same_type_ents]
        type_k = min(k, len(same_type_ents))
        nbrs = NearestNeighbors(n_neighbors=type_k, n_jobs=num_cpu, metric="euclidean").fit(same_type_embeddings)
        _, knn_indices = nbrs.kneighbors(same_type_embeddings)
        for row_id in range(knn_indices.shape[0]):
            row = knn_indices[row_id]
            knn[same_type_ents[row[0]]] = [same_type_ents[row[i]] for i in range(row.shape[0])]
    logout("Typed KNN learning finished.","s")
    return knn


def get_knn(embeddings):
    num_cpu = multiprocessing.cpu_count()
    nbrs = NearestNeighbors(n_neighbors=embeddings.shape[0], n_jobs=num_cpu, metric="euclidean").fit(embeddings)
    return nbrs.kneighbors(embeddings)
    

def get_batch_from_generator(triples_iter, batch_size):
    batch_heads = []
    batch_rels = []
    batch_tails = []

    for i in range(batch_size):
        try:
            triple = next(triples_iter)
        except StopIteration:
            break
        batch_heads.append(triple[0])
        batch_rels.append(triple[1])
        batch_tails.append(triple[2])
    
    batch_heads = torch.tensor(batch_heads, dtype=torch.long)
    batch_rels = torch.tensor(batch_rels, dtype=torch.long)
    batch_tails = torch.tensor(batch_tails, dtype=torch.long)

    return (batch_heads, batch_rels, batch_tails), len(batch_heads)


def get_p_triples(batch_head, batch_rel, batch_tail, i2e, i2r, scores, thresholds):
    batch_head = batch_head.cpu().detach().numpy()
    batch_rel = batch_rel.cpu().detach().numpy()
    batch_tail = batch_tail.cpu().detach().numpy()
    positives = []
    for i in range(len(scores)):
        if scores[i] > thresholds[batch_rel[i]]:
            h = i2e[batch_head[i]]
            r = i2r[batch_rel[i]]
            t = i2e[batch_tail[i]]
            positives.append([h,r,t])
    return positives


def ghat_triple_generator(nbrs, dataset, max_nbrs):
    num_head_nbrs = list(max_nbrs[0])
    num_tail_nbrs = list(max_nbrs[1])
    for batch in dataset:
        h, r, t, _ = batch[0,:]
        for ghat_triple in itertools.product(nbrs[h][:num_head_nbrs[r]], [r], nbrs[t][:num_tail_nbrs[r]]):
            yield [ghat_triple[0], ghat_triple[1], ghat_triple[2]]


def generate_ghat(args, knn, dataset, model, thresholds, device, max_neighbors, ghat_path=None):    
    """
    Generates graph of positive triples for SFE input
    :param args: experiment config
    :param knn: nearest neighbors dict
    :param dataset: dataset of triples to classify
    :param model: pytorch nn.Module embedding object
    :param thresholds: dict of relation thresholds {"rel": threshold}
    :param device: cuda device
    :param ghat_path: optional output path
    :param max_neighbors: optional max number of neighbors to use for g_hat
    :return: g_hat []
    """
    g_hat = []
    gen = ghat_triple_generator(knn, dataset, max_neighbors)
    model.eval()
    with torch.no_grad():
        while True:
            (bh, br, bt), bs = get_batch_from_generator(gen, 1000)
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device))
            g_hat += get_p_triples(bh, br, bt, dataset.i2e, dataset.i2r, scores, thresholds)
            if bs < 1000:
                break
    if ghat_path is not None:
        pd.DataFrame(g_hat).to_csv(ghat_path, sep="\t", index=False, header=False)
    logout("Generated G_hat. Size is " + str(len(g_hat)), "s")
    return np.asarray(g_hat, dtype=str)


def process_ghat(ghat, e2i, i2e, r2i, i2r, num_gt_triples):
    # combine all triples from ghat and gt
    str_gt_triples = []
    for row_id in range(num_gt_triples.shape[0]):
        hi, ri, ti = num_gt_triples[row_id,:]
        str_gt_triples.append([i2e[hi],i2r[ri],i2e[ti]])
    str_gt_triples = np.asarray(str_gt_triples)
    all_triples = np.unique(np.append(ghat, str_gt_triples, axis=0), axis=0)
    # all_triples = ghat
    
    # convert ghat into usable lookup table
    valid_heads_rt = {}
    valid_heads_r = {}
    valid_tails_rh = {}
    valid_tails_r = {}
    for i in range(all_triples.shape[0]):
        h, r, t = all_triples[i,:]
        h_i = e2i[h]
        r_i = r2i[r]
        t_i = e2i[t]

        if (r_i,t_i) not in valid_heads_rt:
            valid_heads_rt[(r_i,t_i)] = [h_i]
        elif h_i not in valid_heads_rt[(r_i,t_i)]:
            valid_heads_rt[(r_i,t_i)].append(h_i)

        if r_i not in valid_heads_r:
            valid_heads_r[r_i] = [h_i]
        elif h_i not in valid_heads_r[r_i]:
            valid_heads_r[r_i].append(h_i)

        if (r_i,h_i) not in valid_tails_rh:
            valid_tails_rh[(r_i,h_i)] = [t_i]
        elif t_i not in valid_tails_rh[(r_i,h_i)]:
            valid_tails_rh[(r_i,h_i)].append(t_i)

        if r_i not in valid_tails_r:
            valid_tails_r[r_i] = [t_i]
        elif t_i not in valid_tails_r[r_i]:
            valid_tails_r[r_i].append(t_i)

    # also load corrupt domains
    rel_heads = {}
    rel_tails = {}
    for row_id in range(all_triples.shape[0]):
        h, r, t = all_triples[row_id, :]
        if r not in rel_heads:
            rel_heads[r] = []
        if r not in rel_tails:
            rel_tails[r] = []
        if h not in rel_heads[r]:
            rel_heads[r].append(h)
        if t not in rel_tails[r]:
            rel_tails[r].append(t)

    # assign the full domains
    h_dom = {}
    t_dom = {}
    for r in rel_heads.keys():
        for t in rel_tails[r]:
            h_dom[(r,t)] = deepcopy(rel_heads[r])
        for h in rel_heads[r]:
            t_dom[(r,h)] = deepcopy(rel_tails[r])
    
    # remove all head/tails from relation domain in triples
    for row_id in range(all_triples.shape[0]):
        h, r, t = all_triples[row_id,:]
        if t in t_dom[(r,h)]:
            del t_dom[(r,h)][t_dom[(r,h)].index(t)]
        if h in h_dom[(r,t)]:
            del h_dom[(r,t)][h_dom[(r,t)].index(h)]

    return valid_heads_rt, valid_heads_r, valid_tails_rh, valid_tails_r, h_dom, t_dom
    

def load_datasets_to_dataframes(args):
    """
    Loads all splits for a dataset into pandas DataFrames given the experiment config
    :param args: experiment config
    :return: tuple of pandas DataFrames for each split
    """
    # dev set
    dev_args = copy(args)
    dev_args["continual"]["session"] = 0
    dev_args["dataset"]["set_name"] = "0_valid2id"
    de_p_d = model_utils.load_dataset(dev_args)
    p_triples = copy(de_p_d.triples)
    p_triples = np.append(p_triples, np.ones(shape=(p_triples.shape[0],1), dtype=np.int), axis=1)
    dev_args["dataset"]["set_name"] = "0_valid2id_neg"
    de_n_d = model_utils.load_dataset(dev_args)
    n_triples = copy(de_n_d.triples)
    n_triples = np.append(n_triples, -np.ones(shape=(n_triples.shape[0],1), dtype=np.int), axis=1)
    de_df = pd.DataFrame(np.concatenate((p_triples, n_triples), axis=0))
    # test set
    test_args = copy(args)
    test_args["continual"]["session"] = 0
    test_args["dataset"]["set_name"] = "0_test2id"
    te_p_d = model_utils.load_dataset(test_args)
    p_triples = copy(te_p_d.triples)
    p_triples = np.append(p_triples, np.ones(shape=(p_triples.shape[0],1), dtype=np.int), axis=1)
    test_args["dataset"]["set_name"] = "0_test2id_neg"
    te_n_d = model_utils.load_dataset(test_args)
    n_triples = copy(te_n_d.triples)
    n_triples = np.append(n_triples, -np.ones(shape=(n_triples.shape[0],1), dtype=np.int), axis=1)
    te_df = pd.DataFrame(np.concatenate((p_triples, n_triples), axis=0))
    # train set
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    train_args["dataset"]["neg_ratio"] = 1
    tr_dataset = model_utils.load_dataset(train_args)
    tr_dataset.load_bernouli_sampling_stats()
    tr_dataset.load_corrupt_domains()
    tr_dataset.load_current_ents_rels()
    tr_dataset.load_known_ent_set()
    tr_dataset.load_known_rel_set()
    tr_dataset.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    tr_df = pd.DataFrame(np.concatenate([tr_dataset[i] for i in range(len(tr_dataset))], axis=0))
    return tr_df, de_df, te_df


def create_split(dfs, splits_dirpath, split_name):
    """Creates a split directory that PRA algorithm can use for the respective dataset.

    Arguments:
    - dfs: a dict whose keys are fold names (e.g. "train", "test") and values are DataFrames with
    head, tail, relation, and label columns.
    - split_dirpath: path where the split should be created.
    """
    if not os.path.exists(splits_dirpath):
        os.makedirs(splits_dirpath)
    this_split_path = splits_dirpath + '/' + split_name
    if not os.path.exists(this_split_path):
        os.makedirs(this_split_path)

    # get relations
    rels = set()
    for _, df in dfs.items():
        rels.update(df[1].unique())

    # create relations_to_run.tsv file
    with open(this_split_path + '/relations_to_run.tsv', 'w') as f:
        for rel in rels:
            f.write('{}\n'.format(rel))

    # create each relation dir and its files
    for rel in rels:
        for fold_name, df in dfs.items():
            relpath = '{}/{}/'.format(this_split_path, rel)
            if not os.path.exists(relpath):
                os.makedirs(relpath)
            filtered = df.loc[df[1] == rel]
            filtered.to_csv('{}/{}.tsv'.format(relpath, fold_name),
                            columns=[0, 2, 3], index=False, header=False, sep='\t')


def run_sfe(args, model, device, rel_thresholds, i2e, i2r,
            split_fp, split_name, ghat_path, main_fp):
    """

    """
    train_df, dev_df, test_df = load_datasets_to_dataframes(args)
    with torch.no_grad():
        for df in [train_df, dev_df, test_df]:
            bh = torch.tensor(df[0], dtype=torch.long)
            br = torch.tensor(df[1], dtype=torch.long)
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), df[2]].cpu().detach().numpy()
            else:
                bt = torch.tensor(df[2], dtype=torch.long)
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device))
            for i in range(len(scores)):
                if scores[i] > rel_thresholds[df.loc[i,1]]:
                    df.loc[i,3] = 1
                else:
                    df.loc[i,3] = -1
                df.loc[i,0] = i2e[df.loc[i,0]]
                df.loc[i,1] = i2r[df.loc[i,1]]
                df.loc[i,2] = i2e[df.loc[i,2]]
    create_split({"train":train_df, "valid": dev_df, "test": test_df}, split_fp, split_name)

    rel_meta_str = ""
    if args["dataset"]["reverse"]:
        assert len(i2r) % 2 == 0
        rev_offset = int(len(i2r) / 2)
        rev_rel_map = pd.DataFrame()
        for rel_idx in range(rev_offset):
            rel = i2r[rel_idx]
            rev_rel = i2r[rel_idx + rev_offset]
            rev_rel_map = rev_rel_map.append({0: rel, 1: rev_rel}, ignore_index=True)
        rel_meta_fp = '{}/{}.tsv'.format(main_fp, "inverses")
        rev_rel_map.to_csv(rel_meta_fp,
                            columns=[0, 1], index=False, header=False, sep='\t')
        rel_meta_str = '"relation metadata": "{}",'.format(rel_meta_fp)
    
    spec = """
        {{
            "graph": {{
                "name": "{}",
                "relation sets": [
                    {{
                        "is kb": false,
                        "relation file": "{}"
                    }}
                ]
            }},
            {}
            "split": "{}",
            "operation": {{
                "type": "create matrices",
                "features": {{
                    "type": "subgraphs",
                    "path finder": {{
                        "type": "BfsPathFinder",
                        "number of steps": 2
                    }},
                    "feature extractors": [
                        {}
                    ],
                    "feature size": -1
                }},
                "data": "{}"
            }},
            "output": {{ "output matrices": true }}
        }}
    """.format("ghat", ghat_path, rel_meta_str, split_name, '"PraFeatureExtractor"', "onefold")
    if not os.path.exists('{}/experiment_specs'.format(main_fp)):
        os.makedirs('{}/experiment_specs'.format(main_fp))
    spec_fpath = '{}/experiment_specs/{}.json'.format(main_fp, split_name)
    with open(spec_fpath, 'w') as f:
        f.write(spec)
    
    bash_command = '/media/adaruna3/melodic/explainable-kge/explainable_kge/run_pra.sh {} {}'.format(main_fp, split_name)
    n_runs = len(i2r) * 3
    for r in tqdm.tqdm(range(n_runs)):
        print("Running #{}: {}".format(r, bash_command))
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if str(error) != "None":
            logout(error, "f")
            logout(output, "d")
            exit()
    logout("SFE finished", "s")


def parse_feature_matrix(filepath):
    """Returns four objects: three lists (of heads, tails and labels) and a sparse matrix (of
    features) for the input (a path to a feature matrix file).
    """
    heads = []
    tails = []
    labels = []
    feat_dicts = []
    with open(filepath, 'r') as f:
        for line in f:
            ent_pair, label, features = line.replace('\n', '').split('\t')
            head, tail = ent_pair.split(',')
            d = {}
            if features:
                for feat in features.split(' -#- '):
                    feat_name, value = feat.split(',')
                    d[feat_name] = float(value)

            heads.append(head)
            tails.append(tail)
            labels.append(int(label))
            feat_dicts.append(d)

    return np.array(heads), np.array(tails), np.array(labels), feat_dicts


def get_reasons(row, n=10):
    # Remove zero elements
    reasons = row[row != 0]
    # Select the top n_examples elements
    top_reasons_abs = reasons.abs().nlargest(n=n, keep='first')
    # Create a pandas series with these
    output = pd.Series()
    counter = 1
    for reason, _ in top_reasons_abs.iteritems():
        output['reason' + str(counter)] = reason
        output['relevance' + str(counter)] = reasons[reason]
        counter = counter + 1
        if counter == n:
            break
    for i in range(counter, n):
        output['reason' + str(i)] = "n/a"
        output['relevance' + str(i)] = "n/a"
    return output, top_reasons_abs.index.to_numpy(dtype=str)


def get_logit_explain_paths(ex_fp, rel, example_num, feats, feat_names, coeff, head, tail, pred, label):
    if not os.path.exists(ex_fp):
        os.makedirs(ex_fp)
    feats = feats.todense()
    explanations = np.multiply(feats, coeff).reshape(1, -1)
    example_df = pd.DataFrame(explanations, columns=feat_names)
    final_reasons, paths = example_df.apply(get_reasons, axis=1)[0]
    final_reasons['head'] = head
    final_reasons['tail'] = tail
    final_reasons['y_logit'] = pred
    final_reasons['y_hat'] = label
    final_reasons.to_csv(os.path.join(ex_fp, rel + "_ex" + str(example_num) + "_" + str(head) + '_' + str(tail) + '.tsv'), sep='\t')
    return paths


def get_dt_explain_paths(ex_fp, rel, example_num, example, model, feat_names, head, tail, pred, label, plot_tree):
    if not os.path.exists(ex_fp):
        os.makedirs(ex_fp)
    file_name = rel + "_ex" + str(example_num) + "_" + str(head) + '_' + str(tail)
    if plot_tree:
        # save whole tree
        plt.axis("tight")
        plot_tree(model, class_names=["False", "True"], feature_names=feat_names, node_ids=True, filled=True)
        plt.savefig(os.path.join(ex_fp, file_name + ".pdf"), bbox_inches='tight', dpi=100)
    # save decision path
    explanation_df = pd.DataFrame(columns=["rule","head","tail","y_logit","y_hat"])
    explanation_df = explanation_df.append({"head": head, "tail": tail, "y_logit": pred, "y_hat": label}, ignore_index=True)
    decision_path_nodes = model.decision_path(example).indices
    leaf = model.apply(example)
    paths = []
    for node in decision_path_nodes:
        if node == leaf: continue
        feat_idx = model.tree_.feature[node]
        path = feat_names[feat_idx].replace("-",",").replace("_","Reverse ")[1:-1]
        if example[0,feat_idx]:
            explanation_df = explanation_df.append({"rule": "Path {} exists".format(path)}, ignore_index=True)
            paths.append(feat_names[feat_idx])
        else:
            explanation_df = explanation_df.append({"rule": "Path {} missing".format(path)}, ignore_index=True)
    explanation_df.to_csv(os.path.join(ex_fp, file_name + '.tsv'), sep='\t')
    return np.asarray(paths, dtype=str)


def fr_hop(args, r_i, h_i, ghat, next_r, r2i, dflag, model, device):
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tails_r, _, _ = ghat
    # get tails that are compatible with next relation
    t_i = valid_tails_rt[(r_i,h_i)]
    if dflag == "fr":
        if next_r[0] == "_":
            t_i_valid = valid_tails_r[r2i[next_r[1:]]]
        else:
            t_i_valid = valid_heads_r[r2i[next_r]]
    else:
        if next_r[0] == "_":
            t_i_valid = valid_heads_r[r2i[next_r[1:]]]
        else:
            t_i_valid = valid_tails_r[r2i[next_r]]
    t_i = list(set(t_i).intersection(set(t_i_valid)))
    assert len(t_i)
    # given head and relation, return most likely tail
    bh = torch.tensor(h_i, dtype=torch.long).repeat(len(t_i))
    br = torch.tensor(r_i, dtype=torch.long).repeat(len(t_i))
    bt = torch.tensor(t_i, dtype=torch.long)
    if args["model"]["name"] == "tucker":
        scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))[0,bt].cpu().detach().numpy()
    else:
        scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device))
    return t_i, scores


def bk_hop(args, r_i, t_i, ghat, next_r, r2i, dflag, model, device):
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tails_r, _, _ = ghat
    h_i = valid_heads_rh[(r_i,t_i)]
    if dflag == "fr":
        if next_r[0] == "_":
            h_i_valid = valid_tails_r[r2i[next_r[1:]]]
        else:
            h_i_valid = valid_heads_r[r2i[next_r]]
    else:
        if next_r[0] == "_":
            h_i_valid = valid_heads_r[r2i[next_r[1:]]]
        else:
            h_i_valid = valid_tails_r[r2i[next_r]]
    h_i = list(set(h_i).intersection(set(h_i_valid)))
    assert len(h_i)
    # given tail and relation, return most likely head
    bh = torch.tensor(h_i, dtype=torch.long)
    br = torch.tensor(r_i, dtype=torch.long).repeat(len(bh))
    bt = torch.tensor(t_i, dtype=torch.long).repeat(len(bh))
    if args["model"]["name"] == "tucker":
        scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
        scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
    else:
        scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device))
    return h_i, scores


def forward_path_search(args, path, head, ghat, e2i, i2e, r2i, model, device):
    # get tails for current path step
    hop_str = path[0]
    if hop_str[0] == "_":
        # fr 'reverse' hop
        r = hop_str[1:]
        t_i_s, scores = bk_hop(args, r2i[r], e2i[head], ghat, path[1], r2i, "fr", model, device)
    else:
        # fr hop
        r = hop_str
        t_i_s, scores = fr_hop(args, r2i[r], e2i[head], ghat, path[1], r2i, "fr", model, device)
    # if path size is two, return the tail paths for head with scores
    if len(path) == 2:
        return [[[head,hop_str,i2e[t_i]]] for t_i in t_i_s], scores
    # if path greater than two, recursively get rest of path and merge with current
    gnd_paths_fr = []
    gnd_scores = []
    for t_i_idx in range(len(t_i_s)):
        t_i = t_i_s[t_i_idx]
        gnd_paths, scores2 = forward_path_search(args, path[1:], i2e[t_i], ghat, e2i, i2e, r2i, model, device)
        for gnd_idx in range(len(gnd_paths)):
            gnd_paths_fr.append(gnd_paths[gnd_idx].insert(0, [head,hop_str,i2e[t_i]]))
            gnd_scores.append(scores[t_i_idx] + scores2[gnd_idx])
    return gnd_paths_fr, gnd_scores


def backward_path_search(args, path, tail, ghat, e2i, i2e, r2i, model, device):
    hop_str = path[-1]
    if hop_str[0] == "_":
        # bk 'reverse' hop
        r = hop_str[1:]
        h_i_s, scores = fr_hop(args, r2i[r], e2i[tail], ghat, path[-2], r2i, "bk", model, device)
    else:
        # bk hop
        r = hop_str
        h_i_s, scores = bk_hop(args, r2i[r], e2i[tail], ghat, path[-2], r2i, "bk", model, device)
    # if path size is two, return the tail paths for head with scores
    if len(path) == 2:
        return [[[i2e[h_i],hop_str,tail]] for h_i in h_i_s], scores
    # if path greater than two, recursively get rest of path and merge with current
    gnd_paths_bk = []
    gnd_scores = []
    for h_i_idx in range(len(h_i_s)):
        h_i = h_i_s[h_i_idx]
        gnd_paths, scores2 = backward_path_search(args, path[:-1], i2e[h_i], ghat, e2i, i2e, r2i, model, device)
        for gnd_idx in range(len(gnd_paths)):
            gnd_paths_bk.append(gnd_paths[gnd_idx].append([i2e[h_i],hop_str,tail]))
            gnd_scores.append(scores[h_i_idx] + scores2[gnd_idx])
    return gnd_paths_bk, gnd_scores


def order_possible_paths(args, scores1, scores2):
    score_tbl = {}
    scores = []
    if scores1.shape[0] and scores2.shape[0]:
        for i in range(len(scores1)):
            for j in range(len(scores2)):
                score_tbl[len(scores)] = (i, j)
                scores.append(scores1[i] + scores2[j])
        if args["model"]["name"] == "tucker":
            _, sort_idxs = torch.sort(torch.tensor(scores), descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
        else:
            sort_idxs = np.argsort(scores)
        ids1 = []
        ids2 = []
        for idx in range(len(sort_idxs)):
            id1, id2 = score_tbl[sort_idxs[idx]]
            ids1.append(id1)
            ids2.append(id2)
    elif scores1.shape[0] and not scores2.shape[0]:
        for i in range(len(scores1)):
            score_tbl[len(scores)] = (i, None)
            scores.append(scores1[i])
        if args["model"]["name"] == "tucker":
            _, sort_idxs = torch.sort(torch.tensor(scores), descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
        else:
            sort_idxs = np.argsort(scores)
        ids1 = []
        ids2 = []
        for idx in range(len(sort_idxs)):
            id1, id2 = score_tbl[sort_idxs[idx]]
            ids1.append(id1)
            ids2.append(id2)
    elif not scores1.shape[0] and not scores2.shape[0]:
        ids1 = [None]
        ids2 = [None]
    else:
        pdb.set_trace()
    return ids1, ids2


def get_connection_ends(h, fr_path, fr_idx, t, bk_path, bk_idx):
    if fr_idx is None:
        fr_h = h
    else:
        fr_h = fr_path[fr_idx][-1][-1]
    if bk_idx is None:
        bk_t = t
    else:
        bk_t = bk_path[bk_idx][0][0]
    return fr_h, bk_t


def path_connection(args, fr_h_i, fr_r_str, bk_t_i, bk_r_str, ghat, r2i, i2e, model, device):
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tail_r, _, _ = ghat
    if fr_r_str[0] != "_" and bk_r_str[0] != "_":
        # neither hop is reversed
        fr_r_i = r2i[fr_r_str]
        fr_t_i = valid_tails_rt[(fr_r_i,fr_h_i)]
        bk_r_i = r2i[bk_r_str]
        bk_h_i = valid_heads_rh[(bk_r_i,bk_t_i)]
        connections = list(set(fr_t_i).intersection(set(bk_h_i)))
        if not len(connections): return False
        if len(connections) == 1: return connections
        bc = torch.tensor(connections, dtype=torch.long)
        fr_bh = torch.tensor(fr_h_i, dtype=torch.long).repeat(len(connections))
        fr_br = torch.tensor(fr_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            fr_scores = model.predict(fr_bh.contiguous().to(device),
                                      fr_br.contiguous().to(device))[0,bc]
        else:
            fr_scores = model.predict(fr_bh.contiguous().to(device),
                                      fr_br.contiguous().to(device),
                                      bc.contiguous().to(device))
        bk_bt = torch.tensor(bk_t_i, dtype=torch.long).repeat(len(connections))
        bk_br = torch.tensor(bk_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            bk_scores = model.predict(bc.contiguous().to(device), 
                                      bk_br.contiguous().to(device))
            bk_scores = bk_scores[torch.arange(0, len(bc), device=device, dtype=torch.long), bk_bt]
        else:
            bk_scores = model.predict(bc.contiguous().to(device), 
                                      bk_br.contiguous().to(device), 
                                      bk_bt.contiguous().to(device))
    elif fr_r_str[0] == "_" and bk_r_str[0] != "_":
        # only forward hop reversed
        fr_r_i = r2i[fr_r_str[1:]]
        fr_t_i = valid_heads_rh[(fr_r_i,fr_h_i)]
        bk_r_i = r2i[bk_r_str]
        bk_h_i = valid_heads_rh[(bk_r_i,bk_t_i)]
        connections = list(set(fr_t_i).intersection(set(bk_h_i)))
        if not len(connections): return False
        if len(connections) == 1: return connections
        bc = torch.tensor(connections, dtype=torch.long)
        fr_bh = torch.tensor(fr_h_i, dtype=torch.long).repeat(len(connections))
        fr_br = torch.tensor(fr_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            fr_scores = model.predict(bc.contiguous().to(device), 
                                      fr_br.contiguous().to(device))
            fr_scores = fr_scores[torch.arange(0, len(bc), device=device, dtype=torch.long), fr_bh]
        else:
            fr_scores = model.predict(bc.contiguous().to(device), 
                                      fr_br.contiguous().to(device), 
                                      fr_bh.contiguous().to(device))
        bk_bt = torch.tensor(bk_t_i, dtype=torch.long).repeat(len(connections))
        bk_br = torch.tensor(bk_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            bk_scores = model.predict(bc.contiguous().to(device), 
                                      bk_br.contiguous().to(device))
            bk_scores = bk_scores[torch.arange(0, len(bc), device=device, dtype=torch.long), bk_bt]
        else:
            bk_scores = model.predict(bc.contiguous().to(device), 
                                      bk_br.contiguous().to(device), 
                                      bk_bt.contiguous().to(device))
    elif fr_r_str[0] != "_" and bk_r_str[0] == "_":
        # only backward hop reversed
        fr_r_i = r2i[fr_r_str]
        fr_t_i = valid_tails_rt[(fr_r_i,fr_h_i)]
        bk_r_i = r2i[bk_r_str[1:]]
        bk_h_i = valid_tails_rt[(bk_r_i,bk_t_i)]
        connections = list(set(fr_t_i).intersection(set(bk_h_i)))
        if not len(connections): return False
        if len(connections) == 1: return connections
        bc = torch.tensor(connections, dtype=torch.long)
        fr_bh = torch.tensor(fr_h_i, dtype=torch.long).repeat(len(connections))
        fr_br = torch.tensor(fr_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            fr_scores = model.predict(fr_bh.contiguous().to(device),
                                      fr_br.contiguous().to(device))[0,bc]
        else:
            fr_scores = model.predict(fr_bh.contiguous().to(device),
                                      fr_br.contiguous().to(device),
                                      bc.contiguous().to(device))
        bk_bt = torch.tensor(bk_t_i, dtype=torch.long).repeat(len(connections))
        bk_br = torch.tensor(bk_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            bk_scores = model.predict(bk_bt.contiguous().to(device),
                                      bk_br.contiguous().to(device))[0,bc]
        else:
            bk_scores = model.predict(bk_bt.contiguous().to(device),
                                      bk_br.contiguous().to(device),
                                      bc.contiguous().to(device))
    else:
        # both hops reversed
        fr_r_i = r2i[fr_r_str[1:]]
        fr_t_i = valid_heads_rh[(fr_r_i,fr_h_i)]
        bk_r_i = r2i[bk_r_str[1:]]
        bk_h_i = valid_tails_rt[(bk_r_i,bk_t_i)]
        connections = list(set(fr_t_i).intersection(set(bk_h_i)))
        if not len(connections): return False
        if len(connections) == 1: return connections
        bc = torch.tensor(connections, dtype=torch.long)
        fr_bh = torch.tensor(fr_h_i, dtype=torch.long).repeat(len(connections))
        fr_br = torch.tensor(fr_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            fr_scores = model.predict(bc.contiguous().to(device), 
                                      fr_br.contiguous().to(device))
            fr_scores = fr_scores[torch.arange(0, len(bc), device=device, dtype=torch.long), fr_bh]
        else:
            fr_scores = model.predict(bc.contiguous().to(device), 
                                      fr_br.contiguous().to(device), 
                                      fr_bh.contiguous().to(device))
        bk_bt = torch.tensor(bk_t_i, dtype=torch.long).repeat(len(connections))
        bk_br = torch.tensor(bk_r_i, dtype=torch.long).repeat(len(connections))
        if args["model"]["name"] == "tucker":
            bk_scores = model.predict(bk_bt.contiguous().to(device),
                                      bk_br.contiguous().to(device))[0,bc]
        else:
            bk_scores = model.predict(bk_bt.contiguous().to(device),
                                      bk_br.contiguous().to(device),
                                      bc.contiguous().to(device))
    scores = fr_scores + bk_scores
    if args["model"]["name"] == "tucker":
        _, sort_idxs = torch.sort(scores, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
    else:
        sort_idxs = np.argsort(scores)
    return np.asarray(connections)[sort_idxs].tolist()


def get_path_corrupt_parts(short_path):
    if len(short_path) > 2:
        num_corrupt_parts = np.random.randint(1,3)
    else:
        num_corrupt_parts = 1
    possible_corrupt_parts = [i for i in range(len(short_path) + 1)]
    corrupt_parts = []
    while len(corrupt_parts) < num_corrupt_parts:
        if len(corrupt_parts) == 0: # ensure head or tail of predicted triple corrupted
            corrupt_parts.append(np.random.choice([0,len(short_path)]))
            if corrupt_parts[-1] == 0:
                possible_corrupt_parts.pop(len(short_path))
            else:
                possible_corrupt_parts.pop(0)
        else:
            corrupt_parts.append(np.random.choice(possible_corrupt_parts))
        possible_corrupt_parts.pop(possible_corrupt_parts.index(corrupt_parts[-1]))
        if corrupt_parts[-1]+1 in possible_corrupt_parts:
            possible_corrupt_parts.pop(possible_corrupt_parts.index(corrupt_parts[-1]+1))
        if corrupt_parts[-1]-1 in possible_corrupt_parts:
            possible_corrupt_parts.pop(possible_corrupt_parts.index(corrupt_parts[-1]-1))
    return corrupt_parts


def get_bad_ents(args, correct_part, ghat, e2i, i2e, r2i, head_flag, corrupting_ends, end_head, end_rel, end_tail, model, device, num_corrections=3):
    valid_heads_rt, _, valid_tails_rh, _, h_dom, t_dom = ghat
    h, r, t = correct_part
    if head_flag:
        # corrupt head with least likely, invalid, same type heads
        if r[0] == "_":
            incorrect_ents = t_dom[(r[1:],t)]
        else:
            incorrect_ents = h_dom[(r,t)]
        if corrupting_ends:
            # account for corrupting ends
            if end_rel[0] == "_":
                if (end_rel[1:],end_tail) in t_dom:
                    incorrect_ents = list(set(incorrect_ents).union(set(t_dom[(end_rel[1:],end_tail)])))
            else:
                if (end_rel,end_tail) in h_dom:
                    incorrect_ents = list(set(incorrect_ents).union(set(h_dom[(end_rel,end_tail)])))
        if len(incorrect_ents):
            # rank according to likelihood of satisfying relationship
            bh = torch.tensor([e2i[ent] for ent in incorrect_ents], dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bh))
            bt = torch.tensor(e2i[t], dtype=torch.long).repeat(len(bh))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_heads = np.asarray(incorrect_ents)[sort_idxs].tolist()
            bad_heads = get_non_repeats_end(all_bad_heads, num_corrections)
        else:
            bad_heads = []
        if len(bad_heads) < 3:
            # add least likely, invalid corrupt heads if needed
            possible_bad_heads = np.arange(len(e2i))
            if r[0] == "_":
                if (r2i[r[1:]],e2i[t]) in valid_tails_rh:
                    valid_heads = valid_tails_rh[(r2i[r[1:]],e2i[t])]
                    possible_bad_heads[np.isin(possible_bad_heads, valid_heads, invert=True)]
            else:
                if (r2i[r],e2i[t]) in valid_heads_rt:
                    valid_heads = valid_heads_rt[(r2i[r],e2i[t])]
                    possible_bad_heads[np.isin(possible_bad_heads, valid_heads, invert=True)]
            if corrupting_ends:
                if end_rel[0] == "_":
                    if (r2i[end_rel[1:]],e2i[end_tail]) in valid_tails_rh:
                        valid_heads = valid_tails_rh[(r2i[end_rel[1:]],e2i[end_tail])]
                        possible_bad_heads[np.isin(possible_bad_heads, valid_heads, invert=True)]
                else:
                    if (r2i[end_rel],e2i[end_tail]) in valid_heads_rt:
                        valid_heads = valid_heads_rt[(r2i[end_rel],e2i[end_tail])]
                        possible_bad_heads[np.isin(possible_bad_heads, valid_heads, invert=True)]
            # rank according to likelihood of satisfying relationship
            bh = torch.tensor(possible_bad_heads, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bh))
            bt = torch.tensor(e2i[t], dtype=torch.long).repeat(len(bh))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_heads = [i2e[e_id] for e_id in np.asarray(possible_bad_heads)[sort_idxs].tolist()] + bad_heads
            bad_heads = get_non_repeats_end(all_bad_heads, num_corrections)
        return bad_heads
    else:
        # corrupt tail with least likely, invalid, same type tails
        if r[0] == "_":
            incorrect_ents = h_dom[(r[1:],h)]
        else:
            incorrect_ents = t_dom[(r,h)]
        if corrupting_ends:
            # account for corrupting ends
            if end_rel[0] == "_":
                if (end_rel[1:],end_head) in h_dom:
                    incorrect_ents = list(set(incorrect_ents).union(set(h_dom[(end_rel[1:],end_head)])))
            else:
                if (end_rel,end_head) in t_dom:
                    incorrect_ents = list(set(incorrect_ents).union(set(t_dom[(end_rel,end_head)])))
        # rank according to likelihood of satisfying relationship
        if len(incorrect_ents):
            bt = torch.tensor([e2i[ent] for ent in incorrect_ents], dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bt))
            bh = torch.tensor(e2i[h], dtype=torch.long).repeat(len(bt))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_tails = np.asarray(incorrect_ents)[sort_idxs].tolist()
            bad_tails = get_non_repeats_end(all_bad_tails, num_corrections)
        else:
            bad_tails = []
        if len(bad_tails) < 3:
            # add least likely, invalid corrupt tails if needed
            possible_bad_tails = np.arange(len(e2i))
            if r[0] == "_":
                if (r2i[r[1:]],e2i[h]) in valid_heads_rt:
                    valid_tails = valid_heads_rt[(r2i[r[1:]],e2i[h])]
                    possible_bad_tails[np.isin(possible_bad_tails, valid_tails, invert=True)]
            else:
                if (r2i[r],e2i[h]) in valid_tails_rh:
                    valid_tails = valid_tails_rh[(r2i[r],e2i[h])]
                    possible_bad_tails[np.isin(possible_bad_tails, valid_tails, invert=True)]
            if corrupting_ends:
                if end_rel[0] == "_":
                    if (r2i[end_rel[1:]],e2i[end_head]) in valid_heads_rt:
                        valid_tails = valid_heads_rt[(r2i[end_rel[1:]],e2i[end_head])]
                        possible_bad_tails[np.isin(possible_bad_tails, valid_tails, invert=True)]
                else:
                    if (r2i[end_rel],e2i[end_head]) in valid_tails_rh:
                        valid_tails = valid_tails_rh[(r2i[end_rel],e2i[end_head])]
                        possible_bad_tails[np.isin(possible_bad_tails, valid_tails, invert=True)]
            # rank according to likelihood of satisfying relationship
            bt = torch.tensor(possible_bad_tails, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bt))
            bh = torch.tensor(e2i[h], dtype=torch.long).repeat(len(bt))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_tails = [i2e[e_id] for e_id in np.asarray(possible_bad_tails)[sort_idxs].tolist()] + bad_tails
            bad_tails = get_non_repeats_end(all_bad_tails, num_corrections)
        return bad_tails


def get_other_ents(args, correct_part, ghat, e2i, i2e, r2i, head_flag, corrupting_ends, end_head, end_rel, end_tail, model, device, num_corrections=3):
    valid_heads_rt, valid_heads_r, valid_tails_rh, valid_tails_r, _, _ = ghat
    h, r, t = correct_part
    if head_flag:
        filter_heads = [h, end_head] if corrupting_ends else [h]
        # corrupt head with most likely, valid, same type heads
        if r[0] == "_":
            incorrect_ents = valid_tails_rh[(r2i[r[1:]],e2i[t])]
        else:
            incorrect_ents = valid_heads_rt[(r2i[r],e2i[t])]
        if corrupting_ends:
            # account for corrupting ends
            if end_rel[0] == "_":
                if (r2i[end_rel[1:]],e2i[end_tail]) in valid_tails_rh:
                    incorrect_ents = list(set(incorrect_ents).union(set(valid_tails_rh[(r2i[end_rel[1:]],e2i[end_tail])])))
            else:
                if (r2i[end_rel],e2i[end_tail]) in valid_heads_rt:
                    incorrect_ents = list(set(incorrect_ents).union(set(valid_heads_rt[(r2i[end_rel],e2i[end_tail])])))
        if len(incorrect_ents):
            # rank according to likelihood of satisfying relationship
            bh = torch.tensor(incorrect_ents, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bh))
            bt = torch.tensor(e2i[t], dtype=torch.long).repeat(len(bh))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_heads = [i2e[e_id] for e_id in np.asarray(incorrect_ents)[sort_idxs].tolist()]
            bad_heads = get_non_repeats_front(all_bad_heads, num_corrections, filter_heads)
        else:
            bad_heads = []
        if len(bad_heads) < 3:
            # add most likely, valid corrupt heads if needed
            if r[0] == "_":
                possible_bad_heads = valid_tails_r[r2i[r[1:]]]
            else:
                possible_bad_heads = valid_heads_r[r2i[r]]
            if corrupting_ends:
                if end_rel[0] == "_":
                    possible_bad_heads = list(set(possible_bad_heads).union(set(valid_tails_r[r2i[end_rel[1:]]])))
                else:
                    possible_bad_heads = list(set(possible_bad_heads).union(set(valid_heads_r[r2i[end_rel]])))
            # rank according to likelihood of satisfying relationship
            bh = torch.tensor(possible_bad_heads, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bh))
            bt = torch.tensor(e2i[t], dtype=torch.long).repeat(len(bh))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends:
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bh))
                    bt = torch.tensor(e2i[end_tail], dtype=torch.long).repeat(len(bh))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_heads = bad_heads + [i2e[e_id] for e_id in np.asarray(possible_bad_heads)[sort_idxs].tolist()]
            bad_heads = get_non_repeats_front(all_bad_heads, num_corrections, filter_heads)
        return bad_heads
    else:
        filter_tails = [t, end_tail] if corrupting_ends else [t]
        # corrupt tail with most likely, valid, same type tails
        if r[0] == "_":
            incorrect_ents = valid_heads_rt[(r2i[r[1:]],e2i[h])]
        else:
            incorrect_ents = valid_tails_rh[(r2i[r],e2i[h])]
        if corrupting_ends:
            # account for corrupting ends
            if end_rel[0] == "_":
                if (r2i[end_rel[1:]],e2i[end_head]) in valid_heads_rt:
                    incorrect_ents = list(set(incorrect_ents).union(set(valid_heads_rt[(r2i[end_rel[1:]],e2i[end_head])])))
            else:
                if (r2i[end_rel],e2i[end_head]) in valid_tails_rh:
                    incorrect_ents = list(set(incorrect_ents).union(set(valid_tails_rh[(r2i[end_rel],e2i[end_head])])))
        # rank according to likelihood of satisfying relationship
        if len(incorrect_ents):
            bt = torch.tensor(incorrect_ents, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bt))
            bh = torch.tensor(e2i[h], dtype=torch.long).repeat(len(bt))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_tails = [i2e[e_id] for e_id in np.asarray(incorrect_ents)[sort_idxs].tolist()]
            bad_tails = get_non_repeats_front(all_bad_tails, num_corrections, filter_tails)
        else:
            bad_tails = []
        if len(bad_tails) < 3:
            # add least likely, valid corrupt tails if needed
            if r[0] == "_":
                possible_bad_tails = valid_heads_r[r2i[r[1:]]]
            else:
                possible_bad_tails = valid_tails_r[r2i[r]]
            if corrupting_ends:
                if end_rel[0] == "_":
                    possible_bad_tails = list(set(possible_bad_tails).union(set(valid_heads_r[r2i[end_rel[1:]]])))
                else:
                    possible_bad_tails = list(set(possible_bad_tails).union(set(valid_tails_r[r2i[end_rel]])))
            # rank according to likelihood of satisfying relationship
            bt = torch.tensor(possible_bad_tails, dtype=torch.long)
            br = torch.tensor(r2i[r.replace("_","")], dtype=torch.long).repeat(len(bt))
            bh = torch.tensor(e2i[h], dtype=torch.long).repeat(len(bt))
            if args["model"]["name"] == "tucker":
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores_ = scores_[torch.arange(0, len(bh), device=device, dtype=torch.long), bt]
                    scores += scores_
                _, sort_idxs = torch.sort(scores, descending=True)
                sort_idxs = sort_idxs.cpu().detach().numpy()
            else:
                scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                if corrupting_ends: # corrupting last tail, account for that
                    br = torch.tensor(r2i[end_rel.replace("_","")], dtype=torch.long).repeat(len(bt))
                    bh = torch.tensor(e2i[end_head], dtype=torch.long).repeat(len(bt))
                    scores_ = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device)).cpu().detach().numpy()
                    scores += scores_
                sort_idxs = np.argsort(scores)
            all_bad_tails = bad_tails + [i2e[e_id] for e_id in np.asarray(possible_bad_tails)[sort_idxs].tolist()]
            bad_tails = get_non_repeats_front(all_bad_tails, num_corrections, filter_tails)
        return bad_tails


def get_non_repeats_end(ent_list, num_corrupt=3):
    if len(ent_list) <= num_corrupt:
        # remove repeats from small list (inefficiently)
        i = len(ent_list) - 1
        while i > 0:
            j = i - 1
            while j >= 0:
                if ent_list[i][:-2] == ent_list[j][:-2]:
                    del ent_list[j]
                    i -= 1
                    j -= 1
                else:
                    j -= 1
            i -= 1
        return ent_list
    else:
        # find 3 non-repeating ents (inefficiently)
        ent_list_norepeat = [ent_list[-1]]
        i = len(ent_list) - 2
        while len(ent_list_norepeat) < 3 and i >= 0:
            j = 0
            skip = False
            while j < len(ent_list_norepeat):
                if ent_list_norepeat[j][:-2] == ent_list[i][:-2]:
                    i -= 1
                    skip = True
                    break
                else:
                    j += 1
            if not skip:
                ent_list_norepeat.insert(0, ent_list[i])
                i -= 1
        return ent_list_norepeat


def get_non_repeats_front(ent_list, num_corrupt=3, filter_ents=[]):
    filter_names = [ent[:-2] for ent in filter_ents]
    ent_list_filtered = [ent for ent in ent_list if ent[:-2] not in filter_names]
    if len(ent_list_filtered) <= num_corrupt:
        # remove repeats from small list (inefficiently)
        i = 0
        while i < len(ent_list_filtered):
            j = i + 1
            while j < len(ent_list_filtered):
                if ent_list_filtered[i][:-2] == ent_list_filtered[j][:-2]:
                    del ent_list_filtered[j]
                else:
                    j += 1
            i += 1
        return ent_list_filtered
    else:
        # find 3 non-repeating ents (inefficiently)
        ent_list_norepeat = [ent_list_filtered[0]]
        i = 1
        while len(ent_list_norepeat) < 3 and i < len(ent_list_filtered):
            j = 0
            skip = False
            while j < len(ent_list_norepeat):
                if ent_list_norepeat[j][:-2] == ent_list_filtered[i][:-2]:
                    i += 1
                    skip = True
                    break
                else:
                    j += 1
            if not skip:
                ent_list_norepeat.append(ent_list_filtered[i])
                i += 1
        return ent_list_norepeat


def fmt_corrupt_part(args, head, rel, tail, correct_path, corrupt_ids, ghat, e2i, i2e, r2i, model, device):
    corrupt_path = []
    prev_tail = None
    num_corrupt = 3
    for part_id in range(len(correct_path)):
        correct_part = correct_path[part_id]
        h, r, t = correct_part
        # double check not corrupting head & tail of same triple
        assert not (part_id in corrupt_ids and part_id+1 in corrupt_ids)
        assert not (part_id in corrupt_ids and part_id-1 in corrupt_ids)
        # form corrupt_part
        corrupt_part = {}
        corrupt_part["idx"] = str(part_id)
        if args["explain"]["corrupt_json"] and (part_id in corrupt_ids): # corrupting head of current part
            bad_heads = get_bad_ents(args, correct_part, ghat, e2i, i2e, r2i, True, part_id == 0, head, rel, tail, model, device, num_corrupt)
            swap_idx = np.random.randint(0, len(bad_heads))
            if prev_tail is None:
                bad_h = bad_heads.pop(swap_idx)
            else:
                bad_heads.pop(swap_idx)
                bad_h = copy(prev_tail)
                prev_tail = None
            bad_heads.insert(swap_idx, h)
            corrupt_part["fact"] = ",".join([h,r,t])
            corrupt_part["fact_corrupted"] = ",".join([bad_h,r,t])
            corrupt_part["str"] = format_triple(bad_h,r,t)
            corrupt_part["str_list"] = format_triple_list(bad_h,r,t)
            if r in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]:
                corrupt_part["colors"] = ["b","b","r"]
            else:
                corrupt_part["colors"] = ["r","b","b"]
            corrupt_part["corrections"] = [format_triple_list(bad_h,r,t) for bad_h in bad_heads]
            corrupt_part["correct_id"] = swap_idx
        elif args["explain"]["corrupt_json"] and (part_id+1 in corrupt_ids): # corrupting tail of current part
            bad_tails = get_bad_ents(args, correct_part, ghat, e2i, i2e, r2i, False, part_id+1 == len(correct_path), head, rel, tail, model, device, num_corrupt)
            swap_idx = np.random.randint(0, len(bad_tails))
            bad_t = bad_tails.pop(swap_idx)
            prev_tail = bad_t
            bad_tails.insert(swap_idx, t)
            corrupt_part["fact"] = ",".join([h,r,t])
            corrupt_part["fact_corrupted"] = ",".join([h,r,bad_t])
            corrupt_part["str"] = format_triple(h,r,bad_t)
            corrupt_part["str_list"] = format_triple_list(h,r,bad_t)
            if r in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]:
                corrupt_part["colors"] = ["r","b","b"]
            else:
                corrupt_part["colors"] = ["b","b","r"]
            corrupt_part["corrections"] = [format_triple_list(h,r,bad_t) for bad_t in bad_tails]
            corrupt_part["correct_id"] = swap_idx
        else: # not corrupting either
            corrupt_part["fact"] = ",".join([h,r,t])
            corrupt_part["fact_corrupted"] = ",".join([h,r,t])
            corrupt_part["str"] = format_triple(h,r,t)
            corrupt_part["str_list"] = format_triple_list(h,r,t)
            flag = np.random.randint(0,2)
            corrupt_part["colors"] = ["r","b","b"] if (flag == 1 and r not in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]) or (flag == 0 and r in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]) else ["b","b","r"]
            if args["explain"]["corrupt_json"]:
                options = get_bad_ents(args, correct_part, ghat, e2i, i2e, r2i, flag, part_id == 0 or part_id+1 == len(correct_path), head, rel, tail, model, device, num_corrupt)
            else:
                options = get_other_ents(args, correct_part, ghat, e2i, i2e, r2i, flag, part_id == 0 or part_id+1 == len(correct_path), head, rel, tail, model, device, num_corrupt)
            if flag:
                corrections = [format_triple_list(option, r, t) for option in options]
                correction_triples = [",".join([option,r,t]) for option in options]
            else:
                corrections = [format_triple_list(h, r, option) for option in options]
                correction_triples = [",".join([h,r,option]) for option in options]
            corrupt_part["corrections"] = corrections
            corrupt_part["correct_id"] = -1
            if not args["explain"]["corrupt_json"]:
                corrupt_part["correction_triples"] = correction_triples
            prev_tail = None
        corrupt_path.append(corrupt_part)
    return corrupt_path


def get_grounded_explanation(ex_fp, args, paths, example_num, rel, head, tail, pred, label, true_label, model, ghat, e2i, i2e, r2i, device):
    if not os.path.exists(ex_fp):
        os.makedirs(ex_fp)
    file_name = rel + "_ex" + str(example_num) + "_" + str(head) + '_' + str(tail) + "_gnd"
    gnd_paths = []
    for path_str in paths:
        gnd_paths_fr = []
        gnd_paths_fr_scores = np.asarray([])
        gnd_paths_bk = []
        gnd_paths_bk_scores = np.asarray([])
        path = path_str[1:-1].split("-")
        if len(path) == 1:
            gnd_paths.append([[head, path[0], tail]])
            continue
        fr_hops = len(path)//2 if len(path) % 2 != 0 else (len(path)//2)-1
        bk_hops = len(path)//2 if len(path) % 2 == 0 else (len(path)//2)+1
        if len(path) > 2:
            # perform forward hops
            gnd_paths_fr, gnd_paths_fr_scores = forward_path_search(args,
                                                                    path[:fr_hops+1],
                                                                    head, ghat,
                                                                    e2i, i2e, r2i,
                                                                    model, device)
            # perform backward hops
            if len(path) > 3:
                gnd_paths_bk, gnd_paths_bk_scores = backward_path_search(args,
                                                                         path[bk_hops:],
                                                                         tail, ghat,
                                                                         e2i, i2e, r2i,
                                                                         model, device)
        fr_id, bk_id = order_possible_paths(args, gnd_paths_fr_scores, gnd_paths_bk_scores)
        pair_idx = -1
        # attempts to form all complete grounded explanation paths in order of likelihood
        while pair_idx < len(fr_id)-1:
            pair_idx += 1
            lh, rt = get_connection_ends(head, gnd_paths_fr, fr_id[pair_idx], tail, gnd_paths_bk, bk_id[pair_idx])
            joint_ents = path_connection(args, e2i[lh], path[fr_hops], e2i[rt], path[bk_hops], ghat, r2i, i2e, model, device)
            if joint_ents:
                for joint_ent in joint_ents:
                    connection = [[lh,path[fr_hops],i2e[joint_ent]],[i2e[joint_ent],path[bk_hops],rt]]
                    if fr_id[pair_idx] is None:
                        gnd_path_fr = []
                    else:
                        gnd_path_fr = gnd_paths_fr[fr_id[pair_idx]]
                    if bk_id[pair_idx] is None:
                        gnd_path_bk = []
                    else:
                        gnd_path_bk = gnd_paths_bk[bk_id[pair_idx]]
                    gnd_paths.append(gnd_path_fr + connection + gnd_path_bk)
    # stores the grounded explanation for debug
    explanation_df = pd.DataFrame(columns=["explanation","head","tail","y_logit","y_hat"])
    explanation_df = explanation_df.append({"head": head, "tail": tail, "y_logit": pred, "y_hat": label}, ignore_index=True)
    # stores the grounded, possibly corrupted, explanations for AMT in JSON
    exp_json = []
    for path in gnd_paths:
        if type(path) == str:
            path_str = path
        else:
            short_path = remove_redundancies(path)
            path_str, fmt_path = format_path(short_path)
            # only test path where BB and XM agree, and path is not a repeat of predicted triple
            # if label == pred and pred == 1 and (len(short_path) > 1 or short_path[0][1].replace("_","") != rel.replace("_","")):
            # prepare for path corruption
            if args["explain"]["corrupt_json"]:
                corrupt_parts = get_path_corrupt_parts(short_path)
            else:
                corrupt_parts = []
            exp = {"y_bb": str(label), "y_xm": str(pred), "parts": []}
            # corrupt corrupt_parts of path
            exp["parts"] = fmt_corrupt_part(args, head, rel, tail, short_path, corrupt_parts, ghat, e2i, i2e, r2i, model, device)
            # set predicted triple, accounting for new corruptions
            if 0 in corrupt_parts and len(short_path) in corrupt_parts:
                fmt_bad_head = exp["parts"][0]["fact_corrupted"].split(",")[0]
                fmt_bad_tail = exp["parts"][-1]["fact_corrupted"].split(",")[-1]
                exp["str"] = format_triple(fmt_bad_head, rel, fmt_bad_tail)
                exp["str_list"] = format_triple_list(fmt_bad_head, rel, fmt_bad_tail)
                exp["fact_corrupted"] = ",".join([fmt_bad_head, rel, fmt_bad_tail])
                exp["fact"] = ",".join([head, rel, tail])
            elif 0 in corrupt_parts: # head needs to be first path head, which was corrupted
                fmt_bad_head = exp["parts"][0]["fact_corrupted"].split(",")[0]
                exp["str"] = format_triple(fmt_bad_head, rel, tail)
                exp["str_list"] = format_triple_list(fmt_bad_head, rel, tail)
                exp["fact_corrupted"] = ",".join([fmt_bad_head, rel, tail])
                exp["fact"] = ",".join([head, rel, tail])
            elif len(short_path) in corrupt_parts: # tail needs to be last path tail, which was corrupted
                fmt_bad_tail = exp["parts"][-1]["fact_corrupted"].split(",")[-1]
                exp["str"] = format_triple(head, rel, fmt_bad_tail)
                exp["str_list"] = format_triple_list(head, rel, fmt_bad_tail)
                exp["fact_corrupted"] = ",".join([head, rel, fmt_bad_tail])
                exp["fact"] = ",".join([head, rel, tail])
            else: # neither first head, nor last tail of path were corrupted
                exp["str"] = format_triple(head, rel, tail)
                exp["str_list"] = format_triple_list(head, rel, tail)
                exp["fact_corrupted"] = ",".join([head, rel, tail])
                exp["fact"] = ",".join([head, rel, tail])
            # store the example predict triple/path combo
            exp_json.append(exp)
        explanation_df = explanation_df.append({"explanation": path_str}, ignore_index=True)
    explanation_df.to_csv(os.path.join(ex_fp, file_name + '.tsv'), sep='\t')
    return exp_json


def remove_redundancies(path):
    # remove redundant parts of paths
    i = 0
    while i < len(path):
        h_i, r_i, t_i = path[i]
        j = i
        redundancy = False
        while j < len(path):
            h_j, r_j, t_j = path[j]
            if h_i == t_j:
                redundancy = True
                break
            else:
                j += 1
        if redundancy:
            if i == j:
                del path[i]
            elif i == 0 and j == len(path)-1:
                i += 1
            else:
                del path[i:j+1]
        else:
            i += 1
    return path


def format_path(path):
    # format the triple to a template
    fmt_path = []
    path_str = ""
    for triple_idx in range(len(path)):
        triple = path[triple_idx]
        h, r, t = triple
        h_str = format_ent(h)
        r_str = format_relation(r)
        t_str = format_ent(t)

        fmt_path.append([h_str, r_str[1:-1], t_str])

        path_str += format_triple(h, r, t)
    return path_str, fmt_path


def add_ing(word):
    if word[-1] == "e":
        word = word[:-1]
        return word + "ing"
    elif word[-1] in ["g","b"]:
        return word + word[-1] + "ing"
    elif word in ["put"]:
        return word + "ting"
    elif word in ["mop","drop"]:
        return word + "ping"
    else:
        return word + "ing"


def add_ed(word):
    if word[-1] == "e":
        return word + "d"
    elif word[-1] in ["g","b"]:
        return word + word[-1] + "ed"
    elif word[-1] == "y":
        return word[:-1] + "ied"
    elif word in ["put"]:
        return word
    elif word in ["mop","drop"]:
        return word + "ped"
    elif word in ["sweep"]:
        return word[:-2] + "pt"
    elif word in ["break"]:
        return "broke"
    elif word in ["throw"]:
        return "thrown"
    else:
        return word + "ed"


def format_triple(head, rel, tail):
    format_rel = {
        "HasEffect": "The act of {} an object will make it {}.",
        "_HasEffect": "An object is {} after {} it.",
        "InverseActionOf": "{} an object is the opposite of {} an object.",
        "_InverseActionOf": "{} an object is the opposite of {} an object.",
        "InverseStateOf": "An object being {} is the opposite of being {}.",
        "_InverseStateOf": "An object being {} is the opposite of being {}.",
        "LocInRoom": "{} can often be found in {}.",
        "_LocInRoom": "{} often can contain {}.",
        "ObjCanBe": "{} can be {}.",
        "_ObjCanBe": "{} can be {}.",
        "ObjInLoc": "{} is often in {}.",
        "_ObjInLoc": "{} often can contain {}.",
        "ObjInRoom": "{} can often be found in {}.",
        "_ObjInRoom": "{} often can contain {}.",
        "ObjOnLoc": "{} is often in {}.",
        "_ObjOnLoc": "{} often can contain {}.",
        "ObjUsedTo": "{} is used to {}.",
        "_ObjUsedTo": "{} can be done using {}.",
        "ObjhasState": "{} can be {}.",
        "_ObjhasState": "{} can be {}.",
        "OperatesOn": "{} is usually used on {}.",
        "_OperatesOn": "{} is usually used on {}.",
    }
    if rel in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]:
        fmt_head = tail[:-2]
        head_type = tail[-1]
        fmt_tail = head[:-2]
        tail_type = head[-1]
    else:
        fmt_head = head[:-2]
        head_type = head[-1]
        fmt_tail = tail[:-2]
        tail_type = tail[-1]
    # article of head
    if rel in ["LocInRoom", "_LocInRoom", "ObjCanBe", "_ObjCanBe", "ObjInLoc", \
               "_ObjInLoc", "ObjInRoom", "_ObjInRoom", "ObjOnLoc", "_ObjOnLoc", \
               "ObjUsedTo", "ObjhasState", "_ObjhasState", "OperatesOn", "_OperatesOn"]:
        if fmt_head in ["alchohol", "bananas", "cereal", "chicken", "chinesefood", \
                        "chocolate_syrup", "pants", "cutlets", "dishwashingliquid", \
                        "facecream", "glasses", "hairproduct", "milk", "mincedmeat", \
                        "multicleaner", "pudding", "salmon", "toilet_paper", "tooth_paste"]:
            pass
        elif fmt_head in ["floor", "ceiling"]:
            fmt_head = "the " + fmt_head
        elif fmt_head[0] in ["a","e","i","o","u"]:
            fmt_head = "an " + fmt_head
        else:
            fmt_head = "a " + fmt_head
    # article of tail
    if rel in ["LocInRoom", "_LocInRoom", "ObjInLoc", "_ObjInLoc", "ObjInRoom", \
               "_ObjInRoom", "ObjOnLoc", "_ObjOnLoc", "_ObjUsedTo", "OperatesOn", \
               "_OperatesOn"]:
        if fmt_tail in ["alchohol", "bananas", "cereal", "chicken", "chinesefood", \
                        "chocolate_syrup", "pants", "cutlets", "dishwashingliquid", \
                        "facecream", "glasses", "hairproduct", "milk", "mincedmeat", \
                        "multicleaner", "pudding", "salmon", "toilet_paper", "tooth_paste"]:
            pass
        elif fmt_tail in ["floor", "ceiling"]:
            fmt_tail = "the " + fmt_tail
        elif fmt_tail[0] in ["a","e","i","o","u"]:
            fmt_tail = "an " + fmt_tail
        else:
            fmt_tail = "a " + fmt_tail
    # tense of head
    if rel in ["HasEffect", "InverseActionOf", "_InverseActionOf", "_ObjUsedTo"] and head_type == "a":
        fmt_head_list = fmt_head.split("_")
        fmt_head_list[0] = add_ing(fmt_head_list[0])
        fmt_head = " ".join(fmt_head_list)
    else:
        fmt_head = fmt_head.replace("_"," ")
    # caps head
    # if rel in ["InverseActionOf","_InverseActionOf","LocInRoom","_LocInRoom", \
    #            "ObjCanBe","_ObjCanBe","ObjInLoc","_ObjInLoc","ObjInRoom", \
    #            "_ObjInRoom","ObjOnLoc","_ObjOnLoc","ObjUsedTo","_ObjUsedTo", \
    #            "ObjhasState","_ObjhasState","OperatesOn","_OperatesOn"]:
    #     fmt_head = fmt_head.capitalize()
    # tense of tail
    if rel in ["InverseActionOf", "_HasEffect", "_InverseActionOf"] and tail_type == "a":
        fmt_tail_list = fmt_tail.split("_")
        fmt_tail_list[0] = add_ing(fmt_tail_list[0])
        fmt_tail = " ".join(fmt_tail_list)
    elif rel in ["ObjCanBe", "_ObjCanBe"] and tail_type == "a":
        fmt_tail_list = fmt_tail.split("_")
        fmt_tail_list[0] = add_ed(fmt_tail_list[0])
        fmt_tail = " ".join(fmt_tail_list)
    else:
        fmt_tail = fmt_tail.replace("_"," ")
    # final formatting
    fmt_triple = format_rel[rel].format(fmt_head, fmt_tail)
    # parser = GingerIt()
    # return parser.parse(fmt_triple)['result']
    return fmt_triple


def format_triple_list(head, rel, tail):
    if rel in ["_ObjCanBe", "_ObjhasState", "_OperatesOn"]:
        fmt_head = tail[:-2]
        head_type = tail[-1]
        fmt_tail = head[:-2]
        tail_type = head[-1]
    else:
        fmt_head = head[:-2]
        head_type = head[-1]
        fmt_tail = tail[:-2]
        tail_type = tail[-1]
    # article of head
    if rel in ["LocInRoom", "_LocInRoom", "ObjCanBe", "_ObjCanBe", "ObjInLoc", \
               "_ObjInLoc", "ObjInRoom", "_ObjInRoom", "ObjOnLoc", "_ObjOnLoc", \
               "ObjUsedTo", "ObjhasState", "_ObjhasState", "OperatesOn", "_OperatesOn"]:
        if fmt_head in ["alchohol", "bananas", "cereal", "chicken", "chinesefood", \
                        "chocolate_syrup", "pants", "cutlets", "dishwashingliquid", \
                        "facecream", "glasses", "hairproduct", "milk", "mincedmeat", \
                        "multicleaner", "pudding", "salmon", "toilet_paper", "tooth_paste"]:
            pass
        elif fmt_head in ["floor", "ceiling"]:
            fmt_head = "the " + fmt_head
        elif fmt_head[0] in ["a","e","i","o","u"]:
            fmt_head = "an " + fmt_head
        else:
            fmt_head = "a " + fmt_head
    # article of tail
    if rel in ["LocInRoom", "_LocInRoom", "ObjInLoc", "_ObjInLoc", "ObjInRoom", \
               "_ObjInRoom", "ObjOnLoc", "_ObjOnLoc", "_ObjUsedTo", "OperatesOn", \
               "_OperatesOn"]:
        if fmt_tail in ["alchohol", "bananas", "cereal", "chicken", "chinesefood", \
                        "chocolate_syrup", "pants", "cutlets", "dishwashingliquid", \
                        "facecream", "glasses", "hairproduct", "milk", "mincedmeat", \
                        "multicleaner", "pudding", "salmon", "toilet_paper", "tooth_paste"]:
            pass
        elif fmt_tail in ["floor", "ceiling"]:
            fmt_tail = "the " + fmt_tail
        elif fmt_tail[0] in ["a","e","i","o","u"]:
            fmt_tail = "an " + fmt_tail
        else:
            fmt_tail = "a " + fmt_tail
    # tense of head
    if rel in ["HasEffect", "InverseActionOf", "_InverseActionOf", "_ObjUsedTo"] and head_type == "a":
        fmt_head_list = fmt_head.split("_")
        fmt_head_list[0] = add_ing(fmt_head_list[0])
        fmt_head = " ".join(fmt_head_list)
    else:
        fmt_head = fmt_head.replace("_"," ")
    # caps head
    # if rel in ["InverseActionOf","_InverseActionOf","LocInRoom","_LocInRoom", \
    #            "ObjCanBe","_ObjCanBe","ObjInLoc","_ObjInLoc","ObjInRoom", \
    #            "_ObjInRoom","ObjOnLoc","_ObjOnLoc","ObjUsedTo","_ObjUsedTo", \
    #            "ObjhasState","_ObjhasState","OperatesOn","_OperatesOn"]:
    #     fmt_head = fmt_head.capitalize()
    # tense of tail
    if rel in ["InverseActionOf", "_HasEffect", "_InverseActionOf"] and tail_type == "a":
        fmt_tail_list = fmt_tail.split("_")
        fmt_tail_list[0] = add_ing(fmt_tail_list[0])
        fmt_tail = " ".join(fmt_tail_list)
    elif rel in ["ObjCanBe", "_ObjCanBe"] and tail_type == "a":
        fmt_tail_list = fmt_tail.split("_")
        fmt_tail_list[0] = add_ed(fmt_tail_list[0])
        fmt_tail = " ".join(fmt_tail_list)
    else:
        fmt_tail = fmt_tail.replace("_"," ")
    return fmt_head, rel, fmt_tail


def format_relation(rel):
    format_rel = {
        "HasEffect": " results in ", # "The act of {}ing an object will make it {}."
        "InverseActionOf": " is the opposite action of ", # "An object being {} is the oppossite of the object being {}."
        "InverseStateOf": " is the opposite state of ", # "{}ing an object is the oppossite of {}ing an object."
        "LocInRoom": " can often be found in ",
        "ObjCanBe": " can have performed on it ", # "A {} can be {}ed."
        "ObjInLoc": " is often in ",
        "ObjInRoom": " can often be found in ",
        "ObjOnLoc": " is often on ",
        "ObjUsedTo": " can be used to perform ", # "A {} is used to {}."
        "ObjhasState": " has a possible state of ", # "A {} can be {}."
        "OperatesOn": " can change the state of ", # "A {} can be used on {}."
        "_HasEffect": " is caused by ", # "An object is {} after {}ing it."
        "_InverseActionOf": " is the opposite action of ", # "An object being {} is the oppossite of the object being {}."
        "_InverseStateOf": " is the opposite state of ", # "{}ing an object is the oppossite of {}ing an object."
        "_LocInRoom": " often can contain ",
        "_ObjCanBe": " can be performed on ", # "{}ing is an action that can be done to a {}."
        "_ObjInLoc": " often can contain ",
        "_ObjInRoom": " often can contain ",
        "_ObjOnLoc": " often can contain ",
        "_ObjUsedTo": " can be done using ", # "{}ing can be done using a {}."
        "_ObjhasState": " is a possible state of ", # "{} is a condition a {} can be in."
        "_OperatesOn": " can have its state changed by ", # "{} can be acted on by a {}."
    }
    return format_rel[rel]
    

def format_ent(ent):
    if ent[-1] == "a":
        ent_str = "the {} action".format(ent.replace('_',' ')[:-2])
    elif ent[-1] == "s":
        ent_str = "the {} state".format(ent.replace('_',' ')[:-2])
    else:
        ent_str = "a {}".format(ent.replace('_',' ')[:-2])
    return ent_str


def get_local_data1(head, tail, tr_heads, tr_tails, toggle='tail'):
    """ select train instances with same tail or same head as test """
    indices = []
    if toggle == "tail":
        indices.extend(np.where(tr_tails == tail)[0])
    else:
        indices.extend(np.where(tr_heads == head)[0])
    return indices


def get_local_data2(knn, k, te_head, te_tail, tr_heads, tr_tails, e2i, i2e):
    """ get # nearest neighbor HEADS/TAILS in training set weighted by embedding """
    nbr_heads = knn[e2i[te_head]][1:k]
    nbr_tails = knn[e2i[te_tail]][1:k]
    examples_indices = []
    for head_id in nbr_heads:
        examples_indices.extend(np.where(tr_heads == i2e[head_id])[0])
    for tail_id in nbr_tails:
        examples_indices.extend(np.where(tr_tails == i2e[tail_id])[0])
    return examples_indices


def get_local_data3(head, tail, tr_heads, tr_tails, embeddings, e2i, locality, mask):
    """ get # nearest neighbor TRIPLES in training set weighted by the embedding """
    # get weights for train heads/tails w.r.t. test head/tail
    heads = np.insert(tr_heads.astype("<U35"), 0 , head)
    head_embeddings = np.asarray([embeddings[e2i[h],:] for h in heads])
    dist, idxs = get_knn(head_embeddings)
    head_dists = [None] * len(heads)
    for i, h in enumerate(heads): head_dists[idxs[0,i]] = dist[0,i]
    tails = np.insert(tr_tails.astype("<U35"), 0 , tail)
    tail_embeddings = np.asarray([embeddings[e2i[t],:] for t in tails])
    dist, idxs = get_knn(tail_embeddings)
    tail_dists = [None] * len(tails)
    for i, t in enumerate(tails): tail_dists[idxs[0,i]] = dist[0,i]
    # select k most relevant examples based on combined head/tail weights
    assert len(heads) == len(tails)
    example_dists = [head_dists[i] + tail_dists[i] for i in range(len(heads))]
    example_idxs = np.argsort(example_dists)
    masked_example_idxs = example_idxs[np.insert(mask, 0, False)[example_idxs]]
    return masked_example_idxs[:locality]-1


def get_explainable_results(args, knn, k, r2i, e2i, i2e, sfe_fp, results_fp, ent_embeddings, kg_embedding, ghat, device):
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    exp_name = "explanations" + "_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str
    results = pd.DataFrame(columns=["rel", "sample", "label", "predict", "train_size", "params"])
    if args["explain"]["ground_explanations"]:
        exp_json = []
    train_df, dev_df, test_df = load_datasets_to_dataframes(args)
    # DEBUG
    for rel, rel_id in r2i.items():
    #     a. Load the extracted SFE features/labels
        print("Working on " + str(rel))
        train_fp = os.path.join(sfe_fp, args["model"]["name"], rel, "train.tsv")
        valid_fp = os.path.join(sfe_fp, args["model"]["name"], rel, "valid.tsv")
        test_fp = os.path.join(sfe_fp, args["model"]["name"], rel, "test.tsv")
        tr_heads, tr_tails, tr_y, tr_feat_dicts = parse_feature_matrix(train_fp)
        de_heads, de_tails, de_y, de_feat_dicts = parse_feature_matrix(valid_fp)
        v = DictVectorizer(sparse=True)
        v.fit(tr_feat_dicts + de_feat_dicts)
        train_x = v.transform(tr_feat_dicts)
        valid_x = v.transform(de_feat_dicts)
        tr_heads = np.concatenate((tr_heads, de_heads))
        tr_tails = np.concatenate((tr_tails, de_tails))
        tr_y = np.concatenate((tr_y, de_y))
        train_x = vstack((train_x, valid_x))
        feature_names = v.get_feature_names()
        te_heads, te_tails, te_y, te_feat_dicts = parse_feature_matrix(test_fp)
        # DEBUG
        # v = DictVectorizer(sparse=True)
        # v.fit(tr_feat_dicts)
        # train_x = v.transform(tr_feat_dicts)
        # feature_names = v.get_feature_names()
        # te_heads, te_tails, te_y, te_feat_dicts = parse_feature_matrix(valid_fp)
        # DEBUG
        test_x = v.transform(te_feat_dicts)
        fit_model = True
    #     b. Sample local training set for each test triple
        for test_idx, test_pair in tqdm.tqdm(enumerate(zip(te_heads, te_tails))):
            te_head, te_tail = test_pair
            # prepares xmodel train data
            if args["explain"]["locality"] == "global":
                train_x_local = train_x
                train_y_local = tr_y
            else:
                # various methods for selecting subgraph g to sample
                locality_k = k if type(k) == int else list(k)[rel_id]
                if args["explain"]["locality"] == "local1":
                    examples_indices = get_local_data1(te_head, te_tail, tr_heads, tr_tails)
                elif args["explain"]["locality"] == "local2":
                    examples_indices = get_local_data2(knn, locality_k, te_head, te_tail, tr_heads, tr_tails, e2i, i2e)
                elif args["explain"]["locality"] == "local3":
                    pos_mask = tr_y == 1
                    neg_mask = tr_y == -1
                    min_examples = min(np.count_nonzero(pos_mask), np.count_nonzero(neg_mask))
                    locality = min(locality_k, min_examples)
                    pos = get_local_data3(te_head, te_tail, tr_heads, tr_tails, ent_embeddings, 
                                          e2i, locality, pos_mask)
                    neg = get_local_data3(te_head, te_tail, tr_heads, tr_tails, ent_embeddings, 
                                          e2i, locality, neg_mask)
                    examples_indices = np.append(pos, neg)
                examples_indices = np.unique(examples_indices)
                # get features and labels
                try:
                    train_x_local = train_x[examples_indices, :]
                    train_y_local = tr_y[examples_indices]
                except IndexError:
                    results = results.append({"rel": rel, "sample": test_pair, "label": te_y[test_idx], 
                                              "predict": 0, "train_size": None, "params": None}, ignore_index=True)
                    continue
            # checks if training feasible
            classes, counts = np.unique(train_y_local, return_counts=True)
            if len(classes) <= 1:
                logout("Cannot train for `{}` because singular class.".format(str(test_pair)), "w")
                results = results.append({"rel": rel, "sample": test_pair, "label": te_y[test_idx], 
                                         "predict": 0, "train_size": None, "params": None}, ignore_index=True)
                continue
            else:
                if min(counts) < 2:
                    logout("Cannot train for `{}` because less then 3 examples in a class.".format(str(test_pair)), "w")
                    results = results.append({"rel": rel, "sample": test_pair, "label": te_y[test_idx], 
                                             "predict": 0, "train_size": None, "params": None}, ignore_index=True)
                    continue
    #     c. Train logit model using scikit-learn
            if fit_model:
                if args["explain"]["xmodel"] == "logit":
                    n_jobs = multiprocessing.cpu_count()
                    param_grid_logit = [{
                        'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
                        'alpha': [0.01, 0.001, 0.0001],
                        'loss': ["log"],
                        'penalty': ["elasticnet"],
                        'max_iter': [100000],
                        'tol': [1e-3],
                        'class_weight': ["balanced"],
                        'n_jobs': [n_jobs],
                        'random_state': [1],
                    }]
                    num_folds = min(5, min(counts))
                    gs = GridSearchCV(SGDClassifier(), param_grid_logit, n_jobs=n_jobs, refit=True, cv=num_folds)
                    gs.fit(train_x_local, train_y_local)
                    xmodel = gs.best_estimator_
                    best_params = str(gs.best_params_)
                elif args["explain"]["xmodel"] == "decision_tree":
                    xmodel = DecisionTreeClassifier(random_state=1)
                    xmodel.fit(train_x_local, train_y_local)
                    best_params = str(xmodel.get_params())
            prediction = xmodel.predict(test_x[test_idx]).item()
            if args["explain"]["xmodel"] == "logit":
                paths = get_logit_explain_paths(os.path.join(sfe_fp, exp_name), rel, test_idx, test_x[test_idx], feature_names, xmodel.coef_, te_head, te_tail, prediction, te_y[test_idx])
            else:
                paths = get_dt_explain_paths(os.path.join(sfe_fp, exp_name), rel, test_idx, test_x[test_idx].toarray(), xmodel, feature_names, te_head, te_tail, prediction, te_y[test_idx], args["explain"]["save_tree"])
            if args["explain"]["ground_explanations"]:
                true_label = test_df.loc[(test_df[1] == rel_id) & (test_df[0] == e2i[te_head]) & (test_df[2] == e2i[te_tail])][3].values[0]
                exp = get_grounded_explanation(os.path.join(sfe_fp, exp_name), args, paths, test_idx, rel, te_head, te_tail, prediction, te_y[test_idx], true_label, kg_embedding, ghat, e2i, i2e, r2i, device)
                exp_json += exp
            results = results.append({"rel": rel,
                                      "sample": test_pair,
                                      "label": te_y[test_idx],
                                      "predict": prediction,
                                      "train_size": train_x_local.shape,
                                      "params": best_params}, ignore_index=True)
            if args["explain"]["locality"] == "global":
                fit_model = False
    if args["explain"]["ground_explanations"]:
        facts = []
        for example in exp_json:
            facts.append(tuple(example["fact"].split(",")))
            for part in example["parts"]:
                if part["correct_id"] != -1:
                    facts.append(tuple(part["fact"].split(",")))
        facts = np.unique(facts, axis=0)
        rel_counts = [0 for _ in r2i.keys()]
        for fact in facts:
            rel_counts[r2i[fact[1].replace("_","")]] += 1
        print("Number of corruptions per relation type:")
        for r, r_id in r2i.items():
            print(r + ": " + str(rel_counts[r_id]))
        locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
        corrupt_str = "corrupted" if args["explain"]["corrupt_json"] else "clean"
        json_name = "explanations_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str + "_" + corrupt_str
        json_fp = os.path.join(sfe_fp, json_name + '.json')
        with open(json_fp, "w") as f:
            random.shuffle(exp_json)
            json.dump(exp_json, f)
    with open(results_fp, "wb") as f:
        pickle.dump(results, f)
    logout("Finished getting xmodel results", "s")
    return results