from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import pdb


class TuckER(nn.Module):
    def __init__(self, num_ents, num_rels, ent_dim, rel_dim, label_smooth_rate, device, **kwargs):
        super(TuckER, self).__init__()
        self.E = nn.Embedding(num_ents, ent_dim).to(device)
        self.R = nn.Embedding(num_rels, rel_dim).to(device)
        self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rel_dim, ent_dim, ent_dim)), 
                              dtype=torch.float, device=device, requires_grad=True)).to(device)

        self.input_dropout = nn.Dropout(kwargs["input_dropout"]).to(device)
        self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"]).to(device)
        self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"]).to(device)
        self.loss = nn.BCELoss().to(device)

        self.bn0 = nn.BatchNorm1d(ent_dim).to(device)
        self.bn1 = nn.BatchNorm1d(ent_dim).to(device)
        self.device = device
        self.ent_dim = ent_dim

        self.smooth_rate = label_smooth_rate
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        nn.init.uniform_(self.W.data, -1, 1)
        self.bn0 = None
        self.bn1 = None
        self.bn0 = nn.BatchNorm1d(self.ent_dim).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.ent_dim).to(self.device)

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        if len(batch_h) == 1:  # small batch fix for batchnorm torch bug
            batch_h = batch_h.repeat(2)
            batch_r = batch_r.repeat(2)
            batch_y = batch_y.repeat(2,1)

        e1 = self.E(batch_h)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(batch_r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        score = torch.sigmoid(x)
        if self.smooth_rate:
            batch_y = ((1.0 - self.smooth_rate) * batch_y) + (1.0 / batch_y.size(1))
        return self.loss(score, batch_y)

    def predict(self, batch_h, batch_r):
        if len(batch_h) == 1:  # small batch fix for batchnorm torch bug
            batch_h = batch_h.repeat(2)
            batch_r = batch_r.repeat(2)

        e1 = self.E(batch_h)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(batch_r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        return torch.sigmoid(x)


class Analogy(nn.Module):
    def __init__(self, num_ents, num_rels, ent_dim, rel_dim, device):
        super(Analogy, self).__init__()
        self.ent_re_embeddings = nn.Embedding(num_ents, int(ent_dim / 2.0)).to(device)
        self.ent_im_embeddings = nn.Embedding(num_ents, int(ent_dim / 2.0)).to(device)
        self.rel_re_embeddings = nn.Embedding(num_rels, int(rel_dim / 2.0)).to(device)
        self.rel_im_embeddings = nn.Embedding(num_rels, int(rel_dim / 2.0)).to(device)
        self.ent_embeddings = nn.Embedding(num_ents, int(ent_dim / 2.0)).to(device)
        self.rel_embeddings = nn.Embedding(num_rels, int(rel_dim / 2.0)).to(device)
        self.criterion = nn.Sigmoid().to(device)
        self.device = device
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return torch.sum(r_re * h_re * t_re + r_re * h_im * t_im + r_im * h_re * t_im - r_im * h_im * t_re, -1) + \
               torch.sum(h * t * r, -1)

    def loss(self, score, batch_y):
        return torch.sum(-torch.log(self.criterion(score * batch_y.float())))

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return self.loss(score, batch_y)

    def predict(self, batch_h, batch_r, batch_t):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return -score.cpu().data.numpy()


class TransE(nn.Module):
    def __init__(self, num_ents, num_rels, ent_dim, rel_dim, neg_ratio, batch_size, device, **kwargs):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(num_ents, ent_dim).to(device)
        self.rel_embeddings = nn.Embedding(num_rels, rel_dim).to(device)
        self.criterion = nn.MarginRankingLoss(kwargs["margin"], reduction="sum").to(device)
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.device = device
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, r, t):
        h = nn.functional.normalize(h, 2, -1)
        r = nn.functional.normalize(r, 2, -1)
        t = nn.functional.normalize(t, 2, -1)
        return torch.norm(h + r - t, 1, -1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1])).to(self.device)
        return self.criterion(p_score, n_score, y)

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self, batch_h, batch_r, batch_t):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)
        return score.cpu().data.numpy()

    def get_positive_score(self, score):
        return score[0:len(score):self.neg_ratio+1]

    def get_negative_score(self, score):
        negs = torch.tensor([], dtype=torch.float32).to(self.device)
        for idx in range(0, len(score), self.neg_ratio + 1):
            batch_negs = score[idx + 1:idx + self.neg_ratio + 1]
            negs = torch.cat((negs, torch.mean(batch_negs,0,keepdim=True)))
        return negs