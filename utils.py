#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Jan, 2020

@author: Zhi-Hong Deng
"""

import networkx as nx
import numpy as np
import torch

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois] # 得到每一个session的长度
    len_max = max(us_lens)                           # 最大session长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] # 二级列表，把每个session补0到等长，也即最大session长度
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]                              # 二级列表，得到对应的mask，补0的地方mask为0，否则为1
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0] # 二级列表，包含所有的session，[session1, session2, ...]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs) # 二维numpy数组，session数*最大session长度，保存补0后等长的所有session
        self.mask = np.asarray(mask)     # 二维numpy数组，session数*最大session长度，对应inputs，mask为0表示该位置是补0的
        self.len_max = len_max           # 最大session长度
        self.targets = np.asarray(data[1]) # 二维numpy数组，session数*1
        self.length = len(inputs)          # session数
        self.shuffle = shuffle             # 每个epoch是否需要打乱顺序
        self.graph = graph                 # 用于训练的全局graph，默认None（SR-GNN没用上）

    def generate_batch(self, batch_size): # 每个epoch会重新生成batch
        if self.shuffle: # 打乱数据
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        # 设置索引
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)          # n_batch个batch的索引
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))] # 最后一个batch可能不满batch_size，要特殊处理一下
        return slices # 二维numpy数组，n_batch*batch_size（最后一个batch除外），表示每个batch应取打乱后数据集的哪些session

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        # inputs：二维numpy数组，batch_size*最大session长度，当前batch的所有session（补0至等长）
        # mask：二维numpy数组，batch_size*最大session长度，对应inputs，mask为0表示该位置是补0的
        # targets：维numpy数组，batch_size*1，当前batch每个session的下一件物品
        items, n_node, A, alias_inputs = [], [], [], []
        # 定义batch i中session最多包含的物品数为max_n_node
        # items：二级列表，batch_size*max_n_node。保存了batch i中每个session所包含的不同物品（数量不足max_n_node则用0补足）
        # n_node：列表，batch i中每一个session graph的结点数（session内不同物品的数量）
        # A：列表，max_n_node*(2*max_n_node)，batch i中每一个session的邻接矩阵
        # alias_inputs：二级列表，batch_size*最大session长度，对应inputs，记录了物品在session graph中被映射到什么结点ID（0~max_n_node）
        for u_input in inputs: # 对当前batch的每一个session
            n_node.append(len(np.unique(u_input))) # n_node列表保存每一个session的session graph的结点数（session内不同物品的数量）
        max_n_node = np.max(n_node)                # 当前batch中session graph结点数的最大值
        for u_input in inputs: # 对当前batch的每一个session
            node = np.unique(u_input)                                       # 获得该session内不同的物品
            items.append(node.tolist() + (max_n_node - len(node)) * [0])    # 物品数不足max_n_node则补0，补0后存进items列表
            u_A = np.zeros((max_n_node, max_n_node))                        # 出边&入边邻接矩阵，同一个batch的session不管长度如何，邻接矩阵的size都一样
            # 构建当前session的邻接矩阵u_A
            for i in np.arange(len(u_input) - 1): 
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            # 对当前session的邻接矩阵进行归一化
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
