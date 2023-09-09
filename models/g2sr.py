#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Jan, 2020

@author: Zhi-Hong Deng
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import *

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class G2SR(Module):
    def __init__(self, opt, n_node):
        super(G2SR, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.tau = opt.tau
        self.g_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        print("Initialize parameters done.")

    def set_optimizer(self):     
        embedding_params = list(map(id, self.g_embedding.parameters()))
        base_params = filter(lambda p: id(p) not in embedding_params, self.parameters())
        self.optimizer_ft = torch.optim.SGD(self.g_embedding.parameters(), lr=self.opt.lr/100, weight_decay=self.opt.l2)
        self.optimizer = torch.optim.Adam(base_params, lr=self.opt.lr, weight_decay=self.opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lr_dc_step, gamma=self.opt.lr_dc)

    def compute_scores(self, hidden, mask, position_weights):
        batch_size = hidden.shape[0]
        hidden = hidden*mask.view(batch_size, -1, 1).float()
        hidden = hidden*position_weights[torch.sum(mask, 1)-1].view(batch_size, -1, 1)
        ht = torch.sum(hidden, 1)
        a = ht
        b = self.g_embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.g_embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden



