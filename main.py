#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Jan, 2020

@author: Zhi-Hong Deng
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
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--pretrained', action='store_false', help='use pretrained item embeddings')
parser.add_argument('--pretrain_dataset', default=None, help='use which pretrained embeddings')
parser.add_argument('--tau', type=float, default=1.0, help='temparature parameter')
parser.add_argument('--p', default='0.25', help='the return parameter p')

opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    n_batch = int(train_data.length/opt.batchSize) if train_data.length%opt.batchSize==0 else int(train_data.length/opt.batchSize)+1 
    opt.n_batch = n_batch
    print("number of batches: ", n_batch)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310 # 'sample'

    model = getattr(models, opt.model)(opt, n_node)

    # load pre-trained item embeddings
    dataset, out_of_dict = 'None', []
    if opt.pretrained:
        if opt.pretrain_dataset == None:
            dataset = opt.dataset
        else:
            dataset = opt.pretrain_dataset
        wv = gensim.models.KeyedVectors.load_word2vec_format('datasets/' + dataset + '/p' + opt.p + 'embeddings')
        weights = torch.zeros([n_node, opt.hiddenSize])
        for i in range(n_node):
            try:
                weights[i] = torch.FloatTensor(wv.get_vector(str(i)))
            except:
                out_of_dict.append(i)
                pass # id 'i' doesn't exist
        model.g_embedding = torch.nn.Embedding.from_pretrained(weights, freeze=True)
        print("Load pretrained embeddings done.")
        # print("These items are out of dict: ", out_of_dict)

    model.set_optimizer()
    model = trans_to_cuda(model)
    if torch.cuda.is_available():
        print("Using GPU.")
    else:
        print("Using CPU.")
    print(model)
    print(model.optimizer)

    start = round(time.time(), 3)
    writer = SummaryWriter('logs/'+opt.model+'+'+opt.dataset+'+'+dataset+'+'+str(start)+'/')
    best_result, best_epoch = [0, 0], [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = run(model, train_data, test_data, epoch, writer)
        flag = 0
        if hit >= best_result[0]:
            best_result[0], best_epoch[0] = hit, epoch
            torch.save(model.state_dict(), 'save/'+opt.model+'+'+opt.dataset+'+'+dataset+'+'+str(start)+'.pt')
            flag, bad_counter = 1, 0
        if mrr >= best_result[1]:
            best_result[1], best_epoch[1] = mrr, epoch
            flag, bad_counter = 1, 0
        print('Result:      Recall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'% (hit, mrr, epoch, epoch))
        print('Best Result: Recall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience: # early stopping
            break
    print('-------------------------------------------------------')
    end = round(time.time(), 3)
    print("Total run time: %f s" % (end - start))
    print("Save path: ", opt.model+'+'+opt.dataset+'+'+dataset+'+'+str(start))


if __name__ == '__main__':
    main()
