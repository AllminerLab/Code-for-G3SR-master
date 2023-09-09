#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Jan, 2020

@author: Zhi-Hong Deng
This script is used to evaluate the performance for a trained model only. It will not change its parameters.
"""

import argparse
import pickle
import time
import gensim
import torch
import models
from run import *
from utils import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='G3SR', help='model name: G3SR/SRGNN')
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--test_model', default=None, help='use which model, defualt is None')
parser.add_argument('--tau', type=float, default=1.0, help='temparature parameter')

opt = parser.parse_args()
print(opt)


def main():
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310 # 'sample'

    model = getattr(models, opt.model)(opt, n_node)

    # load pre-trained item embeddings
    model.load_state_dict(torch.load('save/'+opt.test_model), strict=False)
    print("Load pretrained model done.")

    model = trans_to_cuda(model)
    if torch.cuda.is_available():
        print("Using GPU.")
    else:
        print("Using CPU.")

    start = time.time()
    hit, mrr = test(model, test_data, 0, None)
    print('Recall@20:\t%.4f\tMRR@20:\t%.4f'% (hit, mrr))
    end = time.time()
    print("Total run time: %f s" % (end - start))


if __name__ == '__main__':
    main()