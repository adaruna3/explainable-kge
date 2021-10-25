from collections import defaultdict
from os.path import abspath, dirname, exists

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from copy import deepcopy

from explainable_kge.logger.terminal_utils import logout

import pdb


class TripleDataset(Dataset):
    def __init__(self, dataset_name, neg_ratio=0, neg_type="random", reverse=False, session=0):
        """
        Represents a triples dataset
        :param dataset_name: dataset folder name
        """
        super(TripleDataset, self).__init__()
        datasets_fp = abspath(dirname(__file__)) + "/"
        self.fp = datasets_fp + dataset_name + "/"
        self.neg_ratio = neg_ratio
        self.reverse = reverse
        self.session = session
        self.e2i, self.i2e = self.load_id_map(str(session) + "_entity2id.txt")
        self.r2i, self.i2r = self.load_id_map(str(session) + "_relation2id.txt")
        if self.reverse:
            self.r2i, self.i2r = self.add_reverse_rels(self.r2i, self.i2r)
        self.triple_ents = []
        self.triple_rels = []
        self.known_ents = []
        self.known_rels = []
        self.triples = None
        self.berns = None
        self.h_mask = defaultdict(list)
        self.t_mask = defaultdict(list)
        self.counts = None
        self.er_vocab = defaultdict(list)
        self.er_vocab_pairs = []
        self.h_dom = {}
        self.t_dom = {}
        self.neg_type = neg_type

    def load_id_map(self, label_file):
        """
        loads a mapping between triples/strings and IDs
        :param label_file: filename of labels
        :return: ID mapping(s) for the set of labels in a file
        """
        try:
            labels = pd.read_csv(self.fp + label_file, sep="\t", skiprows=1, header=None,
                                 dtype={0: np.str, 1: np.int32})
        except IOError as e:
            logout("Could not load " + str(label_file), "f")
            raise IOError

        label2index = {labels.iloc[idx, 0]: labels.iloc[idx, 1] for idx in range(len(labels))}
        index2label = {labels.iloc[idx, 1]: labels.iloc[idx, 0] for idx in range(len(labels))}
        return label2index, index2label

    def add_reverse_rels(self, r2i, i2r):
        num_rels = len(r2i.keys())
        rev_r2i = deepcopy(r2i)
        rev_i2r = deepcopy(i2r)
        for rel, idx in r2i.items():
            rev_rel = rel + "_reverse"
            rev_idx = int(idx + num_rels)
            rev_r2i[rev_rel] = rev_idx
            rev_i2r[rev_idx] = rev_rel
        return rev_r2i, rev_i2r

    def load_triple_set(self, names):
        """
        Loads the dataset object with triples in set `name` of the dataset
        :param name: `name` of the set to load (i.e. train2id, test2id, valid2id)
        :return: None
        """
        if type(names) == str:
            names = [names]
        self.triples = self.load_triples([name + ".txt" for name in names])
        self.load_bernouli_sampling_stats()
        if self.reverse:
            self.reload_er_vocab()

    def load_triples(self, triples_files):
        """
        loads all triples in the triples file
        :param triples_file: contains triples for train, valid, or test
        :return:
        """
        triples = np.ndarray(shape=(0, 3), dtype=int)
        for triples_file in triples_files:
            try:
                file_triples = pd.read_csv(self.fp + triples_file, sep=" |,|\t", skiprows=1, header=None,
                                     dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
                file_triples[:, [1, 2]] = file_triples[:, [2, 1]]
                if self.reverse:
                    reverse_triples = []
                    for idx in range(file_triples.shape[0]):
                        h_i, r_i, t_i = file_triples[idx, :]
                        rev_r = self.i2r[r_i]+ "_reverse"
                        reverse_triples.append([t_i, self.r2i[rev_r], h_i])
                    file_triples = np.append(file_triples, reverse_triples, axis=0)
                triples = np.append(triples, file_triples, axis=0)
            except IOError as e:
                logout('Could not load ' + str(triples_file), "f")
                raise IOError
        return triples

    def load_corrupt_domains(self):
        all_triples = self.load_triples(["gt2id.txt"])
        # get full head/tail domain for relations
        rel_heads = {r: [] for r in self.i2r.keys()}
        rel_tails = {r: [] for r in self.i2r.keys()}
        for row_id in range(all_triples.shape[0]):
            h, r, t = all_triples[row_id]
            if h not in rel_heads[r]:
                rel_heads[r].append(h)
            if t not in rel_tails[r]:
                rel_tails[r].append(t)
        # assign the full domains
        h_dom = {}
        t_dom = {}
        for r in self.i2r.keys():
            for t in rel_tails[r]:
                h_dom[(r,t)] = deepcopy(rel_heads[r])
            for h in rel_heads[r]:
                t_dom[(r,h)] = deepcopy(rel_tails[r])
        # remove all head/tails from relation domain in triples
        for row_id in range(all_triples.shape[0]):
            h, r, t = all_triples[row_id]
            if t in t_dom[(r,h)]:
                del t_dom[(r,h)][t_dom[(r,h)].index(t)]
            if h in h_dom[(r,t)]:
                del h_dom[(r,t)][h_dom[(r,t)].index(h)]
        self.h_dom = h_dom
        self.t_dom = t_dom

    def reload_er_vocab(self):
        # generate the er_vocab used to train tucker
        self.er_vocab = defaultdict(list)
        self.er_vocab_pairs = []
        for idx in range(self.triples.shape[0]):
            h_i, r_i, t_i = self.triples[idx, :]
            self.er_vocab[(h_i, r_i)].append(t_i)
        self.er_vocab_pairs = list(self.er_vocab.keys())

    def load_known_ent_set(self):
        """
        loads the known ents array used during negative sampling and regularization
        :return:
        """
        known_ents_file = self.fp + str(self.session) + "_known_ents.txt"
        if exists(known_ents_file):
            with open(known_ents_file, "r") as f:
                for line in f:
                    ent = line.strip()
                    self.known_ents.append(self.e2i[ent])
        else:
            self.known_ents = list(self.e2i.values())
        self.known_ents.sort()

    def load_known_rel_set(self):
        """
        loads the known rels array used for regularization
        unknown entities
        :return:
        """
        known_rels_file = self.fp + str(self.session) + "_known_rels.txt"
        if exists(known_rels_file):
            with open(known_rels_file, "r") as f:
                for line in f:
                    rel = line.strip()
                    self.known_rels.append(self.r2i[rel])
        else:
            self.known_rels = list(self.r2i.values())
        self.known_rels.sort()

    def load_current_ents_rels(self):
        for triple in self.triples:
            h, r, t = triple.tolist()
            if h not in self.triple_ents:
                self.triple_ents.append(int(h))
            if t not in self.triple_ents:
                self.triple_ents.append(int(t))
            if r not in self.triple_rels:
                self.triple_rels.append(int(r))
        self.triple_ents.sort()
        self.triple_rels.sort()

    def load_bernouli_sampling_stats(self):
        """
        calculates probabilities needed to do negative sampling based on Bernoulli method
        :return:
        """
        probs = {}
        for rel in self.r2i.values():
            hpt = {}
            tph = {}
            for idx in range(len(self.triples)):
                h, r, t = self.triples[idx, :].tolist()
                if r == rel:
                    if h not in tph:
                        tph[h] = {t}
                    else:
                        tph[h].add(t)
                    if t not in hpt:
                        hpt[t] = {h}
                    else:
                        hpt[t].add(h)
            if len(tph) > 0 and len(hpt) > 0:
                avg_tph = np.average([float(len(tph[h])) for h in tph])
                avg_hpt = np.average([float(len(hpt[t])) for t in hpt])
                probs[rel] = avg_tph / (avg_tph + avg_hpt)
            else:
                probs[rel] = 0.0
        self.berns = probs

    def __len__(self):
        """
        Used by dataloader, returns set size
        :return: triples set size
        """
        if self.reverse:
            return len(self.er_vocab_pairs)
        else:
            return self.triples.shape[0]

    def __getitem__(self, idx):
        """
        :param idx: index of triple to return
        :return: training triples sample
        """
        # modify label behavior for when using tucker
        if self.reverse:
            # positives and negatives together
            h_i, r_i = self.er_vocab_pairs[idx]
            targets = np.zeros(dtype=np.int32, shape=(1, len(self.e2i)))
            targets[0, self.er_vocab[(h_i, r_i)]] = 1
            samples = np.concatenate([[[h_i, r_i, -1]], targets], axis=1)
        else:
            # positives
            samples = np.asarray([self.triples[idx, :].tolist()+[1]])
            # negatives
            if self.neg_type == "random":
                samples = np.concatenate([samples, self.corrupt_random_known(self.triples[idx, :], self.neg_ratio)])
            elif self.neg_type == "type_constrained":
                samples = np.concatenate([samples, self.corrupt_type_constrained(self.triples[idx, :], self.neg_ratio)])
            else:
                logout("Negative sampling method " + str(self.neg_type) + " not supported.","f")
                exit()
        return samples

    def corrupt_random_known(self, triple, num):
        """
        uses Bernoulli method to make corrupted triples
        assumes berns are populated
        :param triple: triple used for generating negative samples
        :param num: number of negative samples
        :return: np.ndarray of negative samples
        """
        h, r, t = triple.tolist()
        corrupted_triples = np.ndarray(shape=(0, 4), dtype=np.int32)
        try:
            prob = self.berns[r]
        except KeyError as e: # for dealing with UNK relations...
            prob = 0.5
        for i in range(num):
            if np.random.uniform() < prob:
                hh = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[hh, r, t, -1]], axis=0)
            else:
                tt = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[h, r, tt, -1]], axis=0)
        return corrupted_triples

    def corrupt_type_constrained(self, triple, num):
        """
        uses relation domains and Bernoulli method to make corrupted triples
        assumes berns, h_dom, and t_dom are populated
        :param triple: triple used for generating negative samples
        :param num: number of negative samples
        :return: np.ndarray of negative samples
        """
        h, r, t = triple.tolist()
        corrupted_triples = np.ndarray(shape=(0, 4), dtype=np.int32)
        try:
            prob = self.berns[r]
        except KeyError as e: # for dealing with UNK relations...
            prob = 0.5
        for i in range(num):
            if np.random.uniform() < prob:
                if len(self.h_dom[(r,t)]):
                    hh = self.h_dom[(r,t)][np.random.randint(len(self.h_dom[(r,t)]), dtype=np.int32)]
                else:
                    hh = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[hh, r, t, -1]], axis=0)
            else:
                if len(self.t_dom[(r,h)]):
                    tt = self.t_dom[(r,h)][np.random.randint(len(self.t_dom[(r,h)]), dtype=np.int32)]
                else:
                    tt = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[h, r, tt, -1]], axis=0)
        return corrupted_triples

    def load_mask(self, dataset_fps=None):
        """
        loads the hr -> o & rt -> h vocab used for "filtering" during evaluation
        """
        t_mask = defaultdict(list)
        h_mask = defaultdict(list)
        all_triples = np.ndarray(shape=(0, 3), dtype=np.int32)

        if dataset_fps is None:
            dataset_fps = [self.fp]
        else:
            dataset_fps += [self.fp]
        dataset_fps = list(set(dataset_fps))

        # loads all train, valid, and test triples
        triple_file_names = ["_train2id", "_valid2id", "_test2id"]
        for dataset_fp in dataset_fps:
            for filename in triple_file_names:
                triples_file = dataset_fp + str(self.session) + filename + ".txt"
                try:
                    new_triples = pd.read_csv(triples_file, sep=" |,|\t", skiprows=1, header=None,
                                         dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
                    new_triples[:, [1, 2]] = new_triples[:, [2, 1]]
                    if self.reverse:
                        reverse_triples = []
                        for idx in range(new_triples.shape[0]):
                            h_i, r_i, t_i = new_triples[idx, :]
                            rev_r = self.i2r[r_i]+ "_reverse"
                            reverse_triples.append([t_i, self.r2i[rev_r], h_i])
                        new_triples = np.append(new_triples, reverse_triples, axis=0)
                    all_triples = np.append(all_triples, new_triples, axis=0)
                except IOError as e:
                    logout('Could not load ' + str(triples_file), "f")
                    exit()
        all_triples = np.unique(all_triples, axis=0)

        # sets the hr -> t & rt -> h vocabs
        for triple in all_triples:
            h, r, t = triple
            h_mask[(r, t)].append(h)
            t_mask[(h, r)].append(t)

        self.h_mask = h_mask
        self.t_mask = t_mask

    def load_counts(self, ground_truth_file, filtering_file=None):
        # loads the ground truth triples from the full dataset
        gt_triples = pd.read_csv(self.fp + ground_truth_file, sep=" |,|\t", skiprows=1, header=None,
                                 dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
        gt_triples[:, [1, 2]] = gt_triples[:, [2, 1]]

        # populates the counts matrix
        self.counts = np.zeros(shape=(len(self.r2i), len(self.e2i), len(self.e2i)), dtype=np.int64)
        for idx in range(gt_triples.shape[0]):
            h, r, t = gt_triples[idx, :]
            self.counts[r, h, t] += 1.0

        if filtering_file is not None:  # TODO consider further what SHOULD be filtered...
            # loads the train triples from the full dataset
            train_triples = pd.read_csv(self.fp + filtering_file, sep=" |,|\t", skiprows=1, header=None,
                                        dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
            train_triples[:, [1, 2]] = train_triples[:, [2, 1]]

            # removes training triples from counts matrix
            for idx in range(train_triples.shape[0]):
                h, r, t = train_triples[idx, :]
                self.counts[r, h, t] = 0.0

    def predict(self, h, r, t):
        return -self.counts[r.cpu().data.numpy(), h.cpu().data.numpy(), t.cpu().data.numpy()]

    def substitute_unks(self):
        # sets the UNK id
        ent_unk_id = len(self.e2i)
        rel_unk_id = len(self.r2i)

        # loads the UNK entities and relations
        unk_ents = set()
        unk_rels = set()
        with open(self.fp+"UNK_ents.txt","r") as f:
            next(f)
            for line in f:
                ent, id = line.strip().split("\t")
                unk_ents.add(int(id))
        with open(self.fp+"UNK_rels.txt","r") as f:
            next(f)
            for line in f:
                rel, id = line.strip().split("\t")
                unk_rels.add(int(id))

        # substitutes UNKs into triples
        for idx in range(self.triples.shape[0]):
            h,r,t = self.triples[idx,:]
            self.triples[idx, 0] = ent_unk_id if h in unk_ents else h
            self.triples[idx, 2] = ent_unk_id if t in unk_ents else t
            self.triples[idx, 1] = rel_unk_id if r in unk_rels else r


class TripleSequenceDataset(TripleDataset):
    def __init__(self, dataset_name, reverse, session):
        super(TripleSequenceDataset, self).__init__(dataset_name, reverse=reverse, session=session)
        self.vocab, self.w2i, self.i2w, self.sot, self.eot = self.load_vocab_map()

    def load_vocab_map(self):
        vocab = list(self.e2i.keys()) + list(self.r2i.keys()) + ["<sot>", "<eot>"]
        w2i = {vocab[idx]: idx for idx in range(len(vocab))}
        i2w = {idx: vocab[idx] for idx in range(len(vocab))}
        sot = w2i["<sot>"]
        eot = w2i["<eot>"]
        return vocab, w2i, i2w, sot, eot

    def load_triple_set(self, names):
        if type(names) == str:
            names = [names]
        self.load_triples([name + ".txt" for name in names])
        self.load_bernouli_sampling_stats()

    def load_triples(self, triples_file):
        self.triples = super().load_triples(triples_file)
        triples = np.zeros_like(self.triples, dtype=int)
        for row in range(self.triples.shape[0]):
            triples[row, 0] = self.w2i[self.i2r[self.triples[row, 1]]]
            triples[row, 1] = self.w2i[self.i2e[self.triples[row, 0]]]
            triples[row, 2] = self.w2i[self.i2e[self.triples[row, 2]]]
        self.triples = triples

    def load_bernouli_sampling_stats(self):
        probs = {}
        for rel in [self.w2i[rel] for rel in self.r2i.keys()]:
            hpt = {}
            tph = {}
            for idx in range(len(self.triples)):
                r, h, t = self.triples[idx, :].tolist()
                if r == rel:
                    if h not in tph:
                        tph[h] = {t}
                    else:
                        tph[h].add(t)
                    if t not in hpt:
                        hpt[t] = {h}
                    else:
                        hpt[t].add(h)
            if len(tph) > 0 and len(hpt) > 0:
                avg_tph = np.average([float(len(tph[h])) for h in tph])
                avg_hpt = np.average([float(len(hpt[t])) for t in hpt])
                probs[rel] = avg_tph / (avg_tph + avg_hpt)
            else:
                probs[rel] = 0.0
        self.berns = probs

    def __getitem__(self, idx):
        return {"input": np.concatenate(([self.w2i["<sot>"]], self.triples[idx, :])),
                "target": np.concatenate((self.triples[idx, :], [self.w2i["<eot>"]]))}

    def __len__(self):
        """
        Used by dataloader, returns set size
        :return: triples set size
        """
        return self.triples.shape[0]

    def corrupt(self, triple, num):
        raise NotImplementedError

    def load_mask(self, dataset_fps=None):
        raise NotImplementedError


if __name__ == "__main__":
    # TODO add unit tests
    pass
