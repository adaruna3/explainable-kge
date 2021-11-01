import os
import multiprocessing
import subprocess
from copy import copy
import itertools
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import tqdm

import torch

from explainable_kge.models import model_utils
from explainable_kge.logger.terminal_utils import logout

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
    return g_hat
    

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


def get_reasons(row, n=3):
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
    return output


def explain_example(ex_fp, rel, example_num, feats, feat_names, coeff, head, tail, pred, label):
    if not os.path.exists(ex_fp):
        os.makedirs(ex_fp)
    feats = feats.todense()
    explanations = np.multiply(feats, coeff).reshape(1, -1)
    example_df = pd.DataFrame(explanations, columns=feat_names)
    final_reasons = example_df.apply(get_reasons, axis=1)
    final_reasons['head'] = head
    final_reasons['tail'] = tail
    final_reasons['y_logit'] = pred
    final_reasons['y_hat'] = label
    final_reasons.to_csv(os.path.join(ex_fp, rel + "_ex" + str(example_num) + "_" + str(head) + '_' + str(tail) + '.tsv'), sep='\t')


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


def get_explainable_results(args, knn, k, r2i, e2i, i2e, sfe_fp, results_fp, embeddings):
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
                    pos = get_local_data3(te_head, te_tail, tr_heads, tr_tails, embeddings, 
                                          e2i, locality, pos_mask)
                    neg = get_local_data3(te_head, te_tail, tr_heads, tr_tails, embeddings, 
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
    #     c. Train logit model using scikit-learn
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
            if fit_model:
                num_folds = min(5, min(counts))
                gs = GridSearchCV(SGDClassifier(), param_grid_logit, n_jobs=n_jobs, refit=True, cv=num_folds)
                gs.fit(train_x_local, train_y_local)
                xmodel = gs.best_estimator_
            prediction = xmodel.predict(test_x[test_idx]).item()
            explain_example(os.path.join(sfe_fp, exp_name), rel, test_idx, test_x[test_idx], feature_names, xmodel.coef_, te_heads[test_idx], te_tails[test_idx], prediction, te_y[test_idx])
            results = results.append({"rel": rel,
                                      "sample": test_pair,
                                      "label": te_y[test_idx],
                                      "predict": prediction,
                                      "train_size": train_x_local.shape,
                                      "params": str(gs.best_params_)}, ignore_index=True)
            if args["explain"]["locality"] == "global":
                fit_model = False
    with open(results_fp, "wb") as f:
        pickle.dump(results, f)
    logout("Finished getting xmodel results", "s")
    return results