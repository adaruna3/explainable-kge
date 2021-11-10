import os
import multiprocessing
import subprocess
from copy import copy
import itertools
import pickle
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


def ghat_triple_generator(nbrs, triples, max_nbrs):
    for triple in triples:
        for ghat_triple in itertools.product(nbrs[triple[0,0]][:max_nbrs], [triple[0,1]], nbrs[triple[0,2]][:max_nbrs]):
            yield [ghat_triple[0], ghat_triple[1], ghat_triple[2]]


def generate_ghat(args, knn, dataset, model, thresholds, device, ghat_path=None, max_neighbors=3):    
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


def process_ghat(ghat, e2i, r2i):
    # convert ghat into usable lookup table
    valid_heads_rt = {}
    valid_heads_r = {}
    valid_tails_rh = {}
    valid_tails_r = {}
    for i in range(ghat.shape[0]):
        h, r, t = ghat[i,:]
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

    return valid_heads_rt, valid_heads_r, valid_tails_rh, valid_tails_r
    

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
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tails_r = ghat
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
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tails_r = ghat
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


def order_possible_paths(scores1, scores2):
    score_tbl = {}
    scores = []
    if scores1.shape[0] and scores2.shape[0]:
        for i in range(len(scores1)):
            for j in range(len(scores2)):
                score_tbl[len(scores)] = (i, j)
                scores.append(scores1[i] + scores2[j])
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
    valid_heads_rh, valid_heads_r, valid_tails_rt, valid_tail_r = ghat
    if fr_r_str[0] != "_" and bk_r_str[0] != "_":
        # neither hop is reversed
        fr_r_i = r2i[fr_r_str]
        fr_t_i = valid_tails_rt[(fr_r_i,fr_h_i)]
        bk_r_i = r2i[bk_r_str]
        bk_h_i = valid_heads_rh[(bk_r_i,bk_t_i)]
        connections = list(set(fr_t_i).intersection(set(bk_h_i)))
        if not len(connections): return False
        if len(connections) == 1: return connections[0]
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
        if len(connections) == 1: return connections[0]
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
        if len(connections) == 1: return connections[0]
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
        if len(connections) == 1: return connections[0]
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
    return connections[sort_idxs[0]]


def get_grounded_explanation(ex_fp, args, paths, example_num, rel, head, tail, pred, label, model, ghat, e2i, i2e, r2i, device):
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
        fr_id, bk_id = order_possible_paths(gnd_paths_fr_scores, gnd_paths_bk_scores)
        pair_idx = -1
        # attempts to form complete grounded explanation path
        joint_ent = False
        while not joint_ent and pair_idx < len(fr_id)-1:
            pair_idx += 1
            lh, rt = get_connection_ends(head, gnd_paths_fr, fr_id[pair_idx], tail, gnd_paths_bk, bk_id[pair_idx])
            joint_ent = path_connection(args, e2i[lh], path[fr_hops], e2i[rt], path[bk_hops], ghat, r2i, i2e, model, device)
        if not joint_ent:
            logout("Failed to find grounded explanation for {} and {} following: {}".format(head, tail, str(path)),"w")
            gnd_paths.append("Failed to find grounded explanation for {} and {} following: {}".format(head, tail, str(path)))
        else:
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
    # store the grounded explanation
    explanation_df = pd.DataFrame(columns=["explanation","head","tail","y_logit","y_hat"])
    explanation_df = explanation_df.append({"head": head, "tail": tail, "y_logit": pred, "y_hat": label}, ignore_index=True)
    for path in gnd_paths:
        if type(path) == str:
            path_str = path
        else:
            path_str = []
            for triple in path:
                path_str.append(",".join(triple))
            path_str = ";".join(path_str)
        explanation_df = explanation_df.append({"explanation": path_str}, ignore_index=True)
    explanation_df.to_csv(os.path.join(ex_fp, file_name + '.tsv'), sep='\t')


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
    heads = np.insert(tr_heads, 0 , head)
    head_embeddings = np.asarray([embeddings[e2i[h],:] for h in heads])
    dist, idxs = get_knn(head_embeddings)
    head_dists = [None] * len(heads)
    for i, h in enumerate(heads): head_dists[idxs[0,i]] = dist[0,i]
    tails = np.insert(tr_tails, 0 , tail)
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
    exp_name = "explanations" + "_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + str(args["explain"]["locality_k"])
    results = pd.DataFrame(columns=["rel", "sample", "label", "predict", "train_size", "params"])
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
                if args["explain"]["locality"] == "local1":
                    examples_indices = get_local_data1(te_head, te_tail, tr_heads, tr_tails)
                elif args["explain"]["locality"] == "local2":
                    examples_indices = get_local_data2(knn, k, te_head, te_tail, tr_heads, tr_tails, e2i, i2e)
                elif args["explain"]["locality"] == "local3":
                    pos_mask = tr_y == 1
                    neg_mask = tr_y == -1
                    min_examples = min(np.count_nonzero(pos_mask), np.count_nonzero(neg_mask))
                    locality = min(args["explain"]["locality_k"], min_examples)
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
                paths = get_logit_explain_paths(os.path.join(sfe_fp, exp_name), rel, test_idx, test_x[test_idx], feature_names, xmodel.coef_, te_heads[test_idx], te_tails[test_idx], prediction, te_y[test_idx])
            else:
                paths = get_dt_explain_paths(os.path.join(sfe_fp, exp_name), rel, test_idx, test_x[test_idx].toarray(), xmodel, feature_names, te_heads[test_idx], te_tails[test_idx], prediction, te_y[test_idx], args["explain"]["save_tree"])
            if args["explain"]["ground_explanations"]:
                get_grounded_explanation(os.path.join(sfe_fp, exp_name), args, paths, test_idx, rel, te_heads[test_idx], te_tails[test_idx], prediction, te_y[test_idx], kg_embedding, ghat, e2i, i2e, r2i, device)
            results = results.append({"rel": rel,
                                      "sample": test_pair,
                                      "label": te_y[test_idx],
                                      "predict": prediction,
                                      "train_size": train_x_local.shape,
                                      "params": best_params}, ignore_index=True)
            if args["explain"]["locality"] == "global":
                fit_model = False
    with open(results_fp, "wb") as f:
        pickle.dump(results, f)
    logout("Finished getting xmodel results", "s")
    return results