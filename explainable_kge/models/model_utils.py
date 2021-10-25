import enum
from os.path import abspath, dirname
import numpy as np
from copy import copy, deepcopy
from enum import Enum

# torch imports
import torch
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import rankdata

# user module imports
from explainable_kge.logger.terminal_utils import logout, log_train
import explainable_kge.datasets.data_utils as data_utils
from explainable_kge.models.pytorch_modelsize import SizeEstimator
import explainable_kge.models.standard_models as std_models

import pdb
import time


def load_dataset(cmd_args):
    dataset = data_utils.TripleDataset(cmd_args["dataset"]["name"], 
                                       cmd_args["dataset"]["neg_ratio"],
                                       cmd_args["dataset"]["neg_type"],
                                       cmd_args["dataset"]["reverse"],
                                       cmd_args["continual"]["session"])
    dataset.load_triple_set(cmd_args["dataset"]["set_name"])
    dataset.load_current_ents_rels()
    return dataset

#######################################################
#  Standard Processors (finetune/offline)
#######################################################
class TrainBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args["dataset"]["name"], 
                                                self.args["dataset"]["neg_ratio"],
                                                self.args["dataset"]["reverse"],
                                                self.args["continual"]["session"])
        self.dataset.load_triple_set(self.args["dataset"]["set_name"])
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.dataset.load_current_ents_rels()
        collate_fn = collate_tucker_batch if self.args["dataset"]["reverse"] else collate_batch
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args["train"]["batch_size"],
                                      num_workers=self.args["train"]["num_workers"],
                                      collate_fn=collate_fn,
                                      pin_memory=True)
        if self.args["cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def reset_data_loader(self):
        collate_fn = collate_tucker_batch if self.args["dataset"]["reverse"] else collate_batch
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args["train"]["batch_size"],
                                      num_workers=self.args["train"]["num_workers"],
                                      collate_fn=collate_fn,
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model_was_training = model.training
        if not model_was_training:
            model.train()

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            try:
                batch_loss = model.forward(bh.contiguous().to(self.device),
                                        br.contiguous().to(self.device),
                                        bt.contiguous().to(self.device),
                                        by.contiguous().to(self.device))
            except:
                pdb.set_trace()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class DevBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args["dataset"]["name"], 
                                                self.args["dataset"]["neg_ratio"],
                                                self.args["dataset"]["reverse"],
                                                self.args["continual"]["session"])
        self.dataset.load_triple_set(self.args["dataset"]["set_name"])
        self.dataset.load_mask(self.args["dataset"]["dataset_fps"])
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.dataset.reverse = False  # makes __getitem__ only retrieve triples instead of triple pairs
        collate_fn = collate_batch
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.args["train"]["batch_size"],
                                      num_workers=self.args["train"]["num_workers"],
                                      collate_fn=collate_fn,
                                      pin_memory=True)
        self.cutoff = int(self.args["train"]["valid_cutoff"] / self.args["train"]["batch_size"]) if self.args["train"]["valid_cutoff"] != str(None) else None
        if self.args["cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def process_epoch(self, model):
        model_was_training = model.training
        if model_was_training:
            model.eval()

        ranks = np.ndarray(shape=0, dtype=np.float64)
        with no_grad():
            for idx_b, batch in enumerate(self.data_loader):
                if self.cutoff is not None:  # validate on less triples for large datasets
                    if idx_b > self.cutoff:
                        break

                if self.args.cuda and torch.cuda.is_available() and self.args["continual"]["cl_method"] == "DGR":
                    torch.cuda.empty_cache()

                # get ranks for each triple in the batch
                bh, br, bt, by = batch
                if self.args["dataset"]["reverse"]:
                    # get tucker ranks
                    ranks = np.append(ranks, self._rank_tucker(model, bh, br, bt), axis=0)
                else:
                    # get transe/analogy ranks
                    ranks = np.append(ranks, self._rank_head(model, bh, br, bt), axis=0)
                    ranks = np.append(ranks, self._rank_tail(model, bh, br, bt), axis=0)

        # calculate hits10 & mrr
        hits10 = np.count_nonzero(ranks <= 10) / len(ranks)
        mrr = np.mean(1.0 / ranks)

        return hits10, mrr

    def _rank_tucker(self, model, h, r, t):
        scores = model.predict(h.contiguous().to(self.device),
                               r.contiguous().to(self.device))
        for idx in range(len(h)):
            filt = self.dataset.t_mask[(h[idx].item(), r[idx].item())]
            target_value = scores[idx, t[idx].item()].item()
            scores[idx, filt] = 0.0
            scores[idx, t[idx]] = target_value
        
        sort_vals, sort_idxs = torch.sort(scores, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
        ranks = []
        for idx in range(len(h)):
            ranks.append(np.where(sort_idxs[idx]==t[idx].item())[0][0] + 1)
        return ranks

    def _rank_head(self, model, h, r, t):
        rank_heads = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(h.shape[0], 1)
        scores = model.predict(rank_heads.contiguous().to(self.device),
                               r.unsqueeze(-1).contiguous().to(self.device),
                               t.unsqueeze(-1).contiguous().to(self.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(r[i].numpy()), int(t[i].numpy())) in self.dataset.h_mask:
                h_mask = copy(self.dataset.h_mask[(int(r[i].numpy()), int(t[i].numpy()))])
                h_mask.remove(int(h[i].numpy()))
                ents = known_ents[np.isin(known_ents, h_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(h[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks

    def _rank_tail(self, model, h, r, t):
        rank_tails = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(t.shape[0], 1)
        scores = model.predict(h.unsqueeze(-1).contiguous().to(self.device),
                               r.unsqueeze(-1).contiguous().to(self.device),
                               rank_tails.contiguous().to(self.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(h[i].numpy()), int(r[i].numpy())) in self.dataset.t_mask:
                t_mask = copy(self.dataset.t_mask[(int(h[i].numpy()), int(r[i].numpy()))])
                t_mask.remove(int(t[i].numpy()))
                ents = known_ents[np.isin(known_ents, t_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(t[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks


def collate_batch(batch):
    batch = tensor(batch)
    batch_h = batch[:, :, 0].flatten()
    batch_r = batch[:, :, 1].flatten()
    batch_t = batch[:, :, 2].flatten()
    batch_y = batch[:, :, 3].flatten()
    return batch_h, batch_r, batch_t, batch_y


def collate_tucker_batch(batch):
    batch = tensor(batch)
    batch_h = batch[:, :, 0].flatten()
    batch_r = batch[:, :, 1].flatten()
    batch_t = batch[:, :, 2].flatten()
    batch_y = batch[:, :, 3:].squeeze(1).float()
    return batch_h, batch_r, batch_t, batch_y


class TransEModels(Enum):
    finetune = std_models.TransE
    offline = std_models.TransE


class AnalogyModels(Enum):
    finetune = std_models.Analogy
    offline = std_models.Analogy


class TuckERModels(Enum):
    finetune = std_models.TuckER
    offline = std_models.TuckER


def init_model(args):
    # sets the cuda device to use for model
    if args["cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # initializes the model
    model = None
    if args["model"]["name"] == "transe":
        model_class = TransEModels[args["continual"]["cl_method"]].value
        if args["continual"]["cl_method"] == "CWR":
            cw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"], 
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   args["dataset"]["neg_ratio"], 
                                   args["train"]["batch_size"],
                                   device,
                                   **args["model"]["arch"])
            tw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"], 
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   args["dataset"]["neg_ratio"], 
                                   args["train"]["batch_size"], 
                                   device,
                                   **args["model"]["arch"])
            model = [tw_model, cw_model]
        else:
            model = model_class(args["model"]["num_ents"], 
                                args["model"]["num_rels"], 
                                args["model"]["ent_dim"],
                                args["model"]["rel_dim"],
                                args["dataset"]["neg_ratio"], 
                                args["train"]["batch_size"],
                                device,
                                **args["model"]["arch"])
    elif args["model"]["name"] == "analogy":
        model_class = AnalogyModels[args["continual"]["cl_method"]].value
        if args["continual"]["cl_method"] == "CWR":
            cw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"], 
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   device)
            tw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"], 
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   device)
            model = [tw_model, cw_model]
        else:
            model = model_class(args["model"]["num_ents"], 
                                args["model"]["num_rels"], 
                                args["model"]["ent_dim"],
                                args["model"]["rel_dim"],
                                device)
    elif args["model"]["name"] == "tucker":
        model_class = TuckERModels[args["continual"]["cl_method"]].value
        if args["continual"]["cl_method"] == "CWR":
            cw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"],
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   args["train"]["label_smooth_rate"],
                                   device,
                                   **args["model"]["arch"])
            tw_model = model_class(args["model"]["num_ents"], 
                                   args["model"]["num_rels"],
                                   args["model"]["ent_dim"],
                                   args["model"]["rel_dim"],
                                   args["train"]["label_smooth_rate"],
                                   device,
                                   **args["model"]["arch"])
            model = [tw_model, cw_model]
        else:
            model = model_class(args["model"]["num_ents"], 
                                args["model"]["num_rels"],
                                args["model"]["ent_dim"],
                                args["model"]["rel_dim"],
                                args["train"]["label_smooth_rate"],
                                device,
                                **args["model"]["arch"])
    else:
        logout("The model '" + str(args.model) + "' to be used is not implemented.", "f")
        exit()
    return model


class Optims(Enum):
    adagrad = optim.Adagrad
    adadelta = optim.Adadelta
    adam = optim.Adam
    sgd = optim.SGD


def init_optimizer(args, model):
    # gets model params to be optimized
    if args["continual"]["cl_method"] == "CWR":
        tw_model, cw_model = model
        optim_model = tw_model
    else:
        optim_model = model
    # gets optimizer params and type, then initialize
    opt_class = Optims[args["train"]["opt_method"]].value
    optimizer = opt_class(optim_model.parameters(), **args["train"]["opt_params"])
    # creates scheduler if needed
    return optimizer


def save_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args["tag"]) + "__"
    checkpoint_name += "sess" + str(args["sess"]) + "_"
    checkpoint_name += str(args["dataset"]["name"]) + "_"
    checkpoint_name += "mt" + str(args["model"]["name"]) + "_"
    checkpoint_name += "clm" + str(args["continual"]["cl_method"]) + "_"
    checkpoint_name += "ln" + str(args["logging"]["log_num"])

    if args["continual"]["cl_method"] == "CWR":
        tw_model, cw_model = model
        save_checkpoint(tw_model.state_dict(), checkpoints_fp + checkpoint_name + "_tw")
        save_checkpoint(cw_model.state_dict(), checkpoints_fp + checkpoint_name + "_cw")
    else:
        save_checkpoint(model.state_dict(), checkpoints_fp + checkpoint_name)


def save_checkpoint(params, filename):
    try:
        torch.save(params, filename)
        # logout('Written to: ' + filename)
    except Exception as e:
        logout("Could not save: " + filename, "w")
        raise e


def load_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args["tag"]) + "__"
    checkpoint_name += "sess" + str(args["sess"]) + "_"
    checkpoint_name += str(args["dataset"]["name"]) + "_"
    checkpoint_name += "mt" + str(args["model"]["name"]) + "_"
    checkpoint_name += "clm" + str(args["continual"]["cl_method"]) + "_"
    checkpoint_name += "ln" + str(args["logging"]["log_num"])

    if args["continual"]["cl_method"] == "CWR":
        tw_model, cw_model = model
        tw_model = load_checkpoint(tw_model, checkpoints_fp + checkpoint_name + "_tw")
        cw_model = load_checkpoint(cw_model, checkpoints_fp + checkpoint_name + "_cw")
        model = tw_model, cw_model
    else:
        model = load_checkpoint(model, checkpoints_fp + checkpoint_name)
    return model


def load_checkpoint(model, filename):
    try:
        model.load_state_dict(load(filename), strict=False)
    except Exception as e:
        logout("Could not load: " + filename, "w")
        raise e
    return model


def evaluate_model(args, sess, batch_processors, model):
    performances = np.ndarray(shape=(0, 2))
    for valid_sess in range(args["continual"]["num_sess"]):
        eval_bp = batch_processors[valid_sess]
        if args["continual"]["cl_method"] == "CWR":
            tw_model, cw_model = model
            if valid_sess == sess:
                performance = eval_bp.process_epoch(tw_model)
            else:
                performance = eval_bp.process_epoch(cw_model)
        else:
            performance = eval_bp.process_epoch(model)
        performances = np.append(performances, [performance], axis=0)
    return performances


class EarlyStopTracker:
    def __init__(self, args):
        self.args = args
        if int(self.args["sess"]) == 0:
            self.num_epoch = 1000.0
        else:
            self.num_epoch = args["train"]["num_epochs"]
        self.epoch = 0
        self.valid_freq = args["train"]["valid_freq"]
        self.patience = args["train"]["patience"]
        self.early_stop_trigger = -int(self.patience / self.valid_freq)
        self.last_early_stop_value = 0.0
        self.best_performances = None
        self.best_measure = 0.0
        self.best_epoch = None

    def continue_training(self):
        return not bool(self.epoch > self.num_epoch or self.early_stop_trigger > 0)

    def get_epoch(self):
        return self.epoch

    def validate(self):
        return bool(self.epoch % self.valid_freq == 0)

    def update_best(self, sess, performances, model):
        measure = performances[sess, 1]
        # checks for new best model and saves if so
        # if measure > self.best_measure:
        if True:
            self.best_measure = copy(measure)
            self.best_epoch = copy(self.epoch)
            self.best_performances = np.copy(performances)
            save_model(self.args, model)
        # checks for reset of early stop trigger
        if measure - 0.01 > self.last_early_stop_value:
            self.last_early_stop_value = copy(measure)
            self.early_stop_trigger = -int(self.patience / self.valid_freq)
        else:
            self.early_stop_trigger += 1
        # adjusts valid frequency throughout training
        if self.epoch >= 400:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 50.0
            self.valid_freq = 50
        elif self.epoch >= 200:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 25.0
            self.valid_freq = 25
        elif self.epoch >= 50:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 10.0
            self.valid_freq = 10

    def step_epoch(self):
        self.epoch += 1

    def get_best(self):
        return self.best_performances, self.best_epoch


def get_rel_thresholds(args, model):
    """
    Searches for best classification threshold to use for each relation type
    :param args: experiment config args
    :param model: pytorch nn.Module KGE object
    :return: classification thresholds {"relation_name": threshold}
    """
    # collect set of positive/negative triples from training and validation sets
    train_args = copy(args)
    train_args["dataset"]["set_name"] = "0_train2id"
    train_args["continual"]["session"] = 0
    train_args["dataset"]["neg_ratio"] = 1
    tr_dataset = load_dataset(train_args)
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
    de_p_d = load_dataset(dev_args)
    de_p_d.triples = np.unique(np.concatenate((de_p_d.triples, p_triples), axis=0), axis=0)
    de_p_d.reverse = False
    dev_args["dataset"]["set_name"] = "0_valid2id_neg"
    de_n_d = load_dataset(dev_args)
    de_n_d.triples = np.unique(np.concatenate((de_n_d.triples, n_triples), axis=0), axis=0)
    de_n_d.reverse = False
    # get scores for each positive/negative triples from embedding
    if args["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    collate_fn = collate_tucker_batch if args["model"]["name"] == "tucker" else collate_batch
    model.eval()
    p_scores = np.zeros(shape=(0))
    n_scores = np.zeros(shape=(0))
    for i, de_d in enumerate([de_p_d, de_n_d]):
        data_loader = DataLoader(de_d, shuffle=False, pin_memory=True, collate_fn=collate_fn,
                                 batch_size=args["train"]["batch_size"],
                                 num_workers=args["train"]["num_workers"])
        with torch.no_grad():
            for idx_b, batch in enumerate(data_loader):
                bh, br, bt, _ = batch
                if args["model"]["name"] == "tucker":
                    scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device))
                    scores = scores[torch.arange(0, len(bh), device=device, dtype=torch.long), bt].cpu().detach().numpy()
                else:
                    scores = model.predict(bh.contiguous().to(device), br.contiguous().to(device), bt.contiguous().to(device))
                if i:
                    n_scores = np.append(n_scores, scores)
                else:
                    p_scores = np.append(p_scores, scores)
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


if __name__ == "__main__":
    # TODO add unit tests below
    pass
