from posixpath import join
import numpy as np
import time
import itertools
import multiprocessing
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier, LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from copy import copy, deepcopy
import torch
from torch.utils.data import DataLoader
import os
import subprocess
import __main__  # used to get the original execute module
import pandas as pd
import tqdm
import pickle

from explainable_kge.models import model_utils
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

    return model, tr_de_dataset

def get_rel_thresholds(args, model):
    # collect set of positive/negative triples from training and validation sets
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    train_args["dataset"]["neg_ratio"] = 1
    tr_dataset = model_utils.load_dataset(train_args)
    tr_dataset.load_corrupt_domains()
    tr_dataset.load_known_ent_set()
    tr_dataset.load_known_rel_set()
    tr_dataset.reverse = False
    labeled_triples = np.concatenate([tr_dataset[i] for i in range(len(tr_dataset))], axis=0)
    p_triples = labeled_triples[labeled_triples[:,-1]==1,:3]
    n_triples = labeled_triples[labeled_triples[:,-1]==-1,:3]
    dev_args = copy(args)
    dev_args["continual"]["session"] = 0
    dev_args["dataset"]["set_name"] = "0_valid2id"
    de_p_d = model_utils.load_dataset(dev_args)
    de_p_d.triples = np.unique(np.concatenate((de_p_d.triples, p_triples), axis=0), axis=0)
    de_p_d.reverse = False
    dev_args["dataset"]["set_name"] = "0_valid2id_neg"
    de_n_d = model_utils.load_dataset(dev_args)
    de_n_d.triples = np.unique(np.concatenate((de_n_d.triples, n_triples), axis=0), axis=0)
    de_n_d.reverse = False
    # get scores for each positive/negative triples from embedding
    if args["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    collate_fn = model_utils.collate_tucker_batch if exp_config["dataset"]["reverse"] else model_utils.collate_batch
    model.eval()
    data_loader = DataLoader(de_p_d, shuffle=False, pin_memory=True,
                             batch_size=exp_config["train"]["batch_size"],
                             num_workers=exp_config["train"]["num_workers"],
                             collate_fn=collate_fn)
    p_scores = np.zeros(shape=(0))
    with torch.no_grad():
        for idx_b, batch in enumerate(data_loader):
            bh, br, bt, by = batch
            scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
            scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
            p_scores = np.append(p_scores, scores)

    data_loader = DataLoader(de_n_d, shuffle=False, pin_memory=True,
                             batch_size=exp_config["train"]["batch_size"],
                             num_workers=exp_config["train"]["num_workers"],
                             collate_fn=collate_fn)
    n_scores = np.zeros(shape=(0))
    with torch.no_grad():
        for idx_b, batch in enumerate(data_loader):
            bh, br, bt, by = batch
            scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
            scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
            n_scores = np.append(n_scores, scores)
    # find best thresholds for each relation type
    rel_thresholds = {}
    rel_accs = {}
    for rel, rel_id in de_p_d.r2i.items():
        # count total +/- triples in set
        total = np.count_nonzero(de_p_d.triples[:,1]==rel_id)
        total += np.count_nonzero(de_n_d.triples[:,1]==rel_id)
        # get range of +/- scores
        min_score = float("inf")
        max_score = -float("inf")
        max_p_score = np.max(p_scores[de_p_d.triples[:,1]==rel_id])
        if max_p_score > max_score: max_score = max_p_score
        min_p_score = np.min(p_scores[de_p_d.triples[:,1]==rel_id])
        if min_p_score < min_score: min_score = min_p_score
        max_n_score = np.max(n_scores[de_n_d.triples[:,1]==rel_id])
        if max_n_score > max_score: max_score = max_n_score
        min_n_score = np.min(n_scores[de_n_d.triples[:,1]==rel_id])
        if min_n_score < min_score: min_score = min_n_score
        # approximate best threshold
        n_interval = 1000
        interval = (max_score - min_score) / n_interval
        best_threshold = min_score + 0 * interval
        best_acc = 0.0
        for i in range(0, n_interval+1):
            temp_threshold = min_score + i * interval
            correct = np.count_nonzero(p_scores[de_p_d.triples[:,1]==rel_id] > temp_threshold)
            correct += np.count_nonzero(n_scores[de_n_d.triples[:,1]==rel_id] < temp_threshold)
            temp_acc = 1.0 * correct / (total * 2.0)
            if temp_acc > best_acc:
                best_acc = copy(temp_acc)
                best_threshold = copy(temp_threshold)
        rel_thresholds[rel_id] = best_threshold
        rel_accs[rel] = best_acc
    logout("Classification mean acc: " + str(np.mean([acc for acc in rel_accs.values()])), "s")
    logout("Classification acc std: " + str(np.std([acc for acc in rel_accs.values()])), "s")
    return rel_thresholds


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


def load_datasets_to_dataframes(args):
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
    tr_dataset.reverse = False
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

    exp_model, tr_de_d = setup_experiment(exp_config)

    # 1. Generate G^
    #     a. For each training triple, perturb with NN to be all possible triples in G^
    ent_embeddings = exp_model.E.weight.cpu().detach().numpy()
    ent_embeddings = ent_embeddings / np.linalg.norm(ent_embeddings, ord=2, axis=1, keepdims=True)
    ent_types = ['s','a','l','r','o']
    knn = {}
    start_time = time.time()
    num_cpu = multiprocessing.cpu_count()
    k = 10
    for ent_type in ent_types:
        same_type_ents = [i for i, e in tr_de_d.i2e.items() if e[-1] == ent_type]
        same_type_embeddings = ent_embeddings[same_type_ents]
        type_k = min(k,len(same_type_ents))
        nbrs = NearestNeighbors(n_neighbors=type_k, n_jobs=num_cpu, metric="euclidean").fit(same_type_embeddings)
        knn_distance, knn_indices = nbrs.kneighbors(same_type_embeddings)
        for row_id in range(knn_indices.shape[0]):
            row = knn_indices[row_id]
            knn[same_type_ents[row[0]]] = [same_type_ents[row[i]] for i in range(row.shape[0])]
    knn_learning_time = time.time() - start_time
    logout("KNN finished, learning time: {}".format(knn_learning_time), "s")

    #     b. Prepare to classify triples as T/F with embedding
    rel_thresholds = get_rel_thresholds(exp_config, exp_model)

    #     c. Form G^ from all T triples classified by embedding
    if not os.path.exists("explainable_kge/logger/logs/g_hat.tsv"):
        def ghat_triple_generator(nbrs, dataset):
            max_nbrs = 3
            for triple in dataset:
                for ghat_triple in itertools.product(nbrs[triple[0,0]][:max_nbrs], [triple[0,1]], nbrs[triple[0,2]][:max_nbrs]):
                    yield [ghat_triple[0], ghat_triple[1], ghat_triple[2]]
        gen = ghat_triple_generator(knn, tr_de_d)
        g_hat = []
        exp_model.eval()
        with torch.no_grad():
            while True:
                (bh, br, bt), bs = get_batch_from_generator(gen, 1000)
                scores = exp_model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
                g_hat += get_p_triples(bh, br, bt, tr_de_d.i2e, tr_de_d.i2r, scores, rel_thresholds)
                if bs < 1000:
                    break
        pd.DataFrame(g_hat).to_csv("explainable_kge/logger/logs/g_hat.tsv", sep="\t", index=False, header=False)
        logout("Generated G_hat. Size is " + str(len(g_hat)), "s")

    #     d. Run SFE on G^
    if not os.path.exists("explainable_kge/logger/logs/results/test"):
        train_df, dev_df, test_df = load_datasets_to_dataframes(exp_config)
        with torch.no_grad():
            for df in [train_df, dev_df, test_df]:
                bh = torch.tensor(df[0], dtype=torch.long).contiguous().to(device)
                br = torch.tensor(df[1], dtype=torch.long).contiguous().to(device)
                scores = exp_model.predict(bh, br)
                scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), df[2]].cpu().detach().numpy()
                for i in range(len(scores)):
                    if scores[i] > rel_thresholds[df.loc[i,1]]:
                        df.loc[i,3] = 1
                    else:
                        df.loc[i,3] = -1
                    df.loc[i,0] = tr_de_d.i2e[df.loc[i,0]]
                    df.loc[i,1] = tr_de_d.i2r[df.loc[i,1]]
                    df.loc[i,2] = tr_de_d.i2e[df.loc[i,2]]
        create_split({"train":train_df, "valid": dev_df, "test": test_df}, "explainable_kge/logger/logs/splits", "test")
        
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
        """.format("ghat2id", "/media/adaruna3/melodic/explainable-kge/explainable_kge/logger/logs/g_hat.tsv", "test", '"PraFeatureExtractor"', "onefold")
        if not os.path.exists('{}/experiment_specs'.format("/media/adaruna3/melodic/explainable-kge/explainable_kge/logger/logs/")):
            os.makedirs('{}/experiment_specs'.format("/media/adaruna3/melodic/explainable-kge/explainable_kge/logger/logs/"))
        spec_fpath = '{}/experiment_specs/{}.json'.format("/media/adaruna3/melodic/explainable-kge/explainable_kge/logger/logs/", "test")
        with open(spec_fpath, 'w') as f:
            f.write(spec)
        
        bash_command = '/media/adaruna3/melodic/explainable-kge/explainable_kge/run_pra.sh {} {}'.format("/media/adaruna3/melodic/explainable-kge/explainable_kge/logger/logs/", "test")
        n_runs = len(tr_de_d.r2i) * 3
        for r in tqdm.tqdm(range(n_runs)):
            print("Running #{}: {}".format(r, bash_command))
            process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            if str(error) != "None":
                logout(error, "f")
                logout(output, "d")
                exit()
        logout("SFE finished", "s")
    
    # 3. Train local explainable model for each test triple
    results_fp = "explainable_kge/logger/logs/results/test.pkl"
    if not os.path.exists(results_fp):
        results = pd.DataFrame(columns=["rel", "sample", "label", "predict"])
        for rel, rel_id in tr_de_d.r2i.items():
        #     a. Load the extracted SFE features/labels
            if "reverse" in rel:
                continue
            print("Working on " + str(rel))
            train_fp = "explainable_kge/logger/logs/results/test/{}/train.tsv".format(rel)
            valid_fp = "explainable_kge/logger/logs/results/test/{}/valid.tsv".format(rel)
            test_fp = "explainable_kge/logger/logs/results/test/{}/test.tsv".format(rel)
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
            skip = False
            
        #     b. Sample local training set for each test triple
            for test_idx, test_pair in tqdm.tqdm(enumerate(zip(te_heads, te_tails))):
                te_head, te_tail = test_pair
                nbr_heads = knn[tr_de_d.e2i[te_head]]
                nbr_tails = knn[tr_de_d.e2i[te_tail]]

                # get the corresponding training examples
                examples_indices = []
                for head_id in nbr_heads[1:]:
                    examples_indices.extend(np.where(tr_heads == tr_de_d.i2e[head_id])[0])
                for tail_id in nbr_tails[1:]:
                    examples_indices.extend(np.where(tr_tails == tr_de_d.i2e[tail_id])[0])
                n_nearby_examples = len(examples_indices)

                # get features
                train_x_local = train_x[examples_indices, :]
                # get labels or scores
                train_y_local = tr_y[examples_indices]
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
                }]
                if len(np.unique(train_y_local)) <= 1:
                    print("Not possible to train explainable model in triple `{}` because training set contains a single class.".format(str(test_pair)))
                    skip = True
                else:
                    for class_ in np.unique(train_y_local):
                        if len(train_y_local[train_y_local==class_]) < 3:
                            print("Not possible to train explainable model in relation `{}` because training set contains too few examples for one of the classes.".format(str(test_pair)))
                            print("Class: " + str(class_) + " has only " + str(len(train_y_local[train_y_local==class_])) + " examples")
                            skip = True
                if skip:
                    results = results.append({"rel": rel, "sample": test_pair, "label": te_y[test_idx], "predict": 0}, ignore_index=True)
                    skip = False
                    continue
                gs = GridSearchCV(SGDClassifier(), param_grid_logit, n_jobs=n_jobs, refit=True, cv=5)
                gs.fit(train_x_local, train_y_local)
                xmodel = gs.best_estimator_
                prediction = xmodel.predict(test_x[test_idx]).item()
                explain_example("explainable_kge/logger/logs/results/explanations", rel, test_idx, test_x[test_idx], feature_names, xmodel.coef_, te_heads[test_idx], te_tails[test_idx], prediction, te_y[test_idx])
                results = results.append({"rel": rel, "sample": test_pair, "label": te_y[test_idx], "predict": prediction}, ignore_index=True)
        with open(results_fp, "wb") as f:
            pickle.dump(results, f)
        logout("Finished getting xmodel results", "s")
    else:
        with open(results_fp, "rb") as f:
            results = pickle.load(f)
    f1s = []
    fids = []
    covs = []
    for rel in tr_de_d.r2i.keys():
        if "reverse" in rel:
            continue
        rel_results = results.loc[results["rel"].isin([rel]),:]
        coverage = float(rel_results.shape[0] - rel_results["predict"].isin([0]).sum()) / float(rel_results.shape[0])
        rel_pred_results = rel_results.loc[rel_results["predict"].isin([1,-1]),:]
        fidelity = accuracy_score(rel_pred_results["label"].to_numpy(np.int), rel_pred_results["predict"].to_numpy(np.int))
        f1_fidelity = f1_score(rel_pred_results["label"].to_numpy(np.int), rel_pred_results["predict"].to_numpy(np.int))
        logout("For relation " + rel + " coverage is " + str(coverage), "s")
        logout("For relation " + rel + " fidelity is " + str(fidelity), "s")
        logout("For relation " + rel + " f1-fidelity is " + str(f1_fidelity), "s")
        f1s.append(f1_fidelity)
        fids.append(fidelity)
        covs.append(coverage)
    logout("Coverage Mean: " + str(np.mean(covs)), "s")
    logout("Coverage Std: " + str(np.std(covs)), "s")
    logout("Fidelity Mean: " + str(np.mean(fids)), "s")
    logout("Fidelity Std: " + str(np.std(fids)), "s")
    logout("F1-Fidelity Mean: " + str(np.mean(f1s)), "s")
    logout("F1-Fidelity Std: " + str(np.std(f1s)), "s")
    pdb.set_trace()
