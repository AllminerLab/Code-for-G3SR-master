#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Jan, 2020
@author: Zhi-Hong Deng
"""

import torch
import datetime
from utils import *

def compute_weights(max_len, tau): # yoochoose1.64的max_len是145，diginetica的是69
    W = np.zeros([max_len, max_len])
    for i in range(max_len):
        for j in range(i+1):
            W[i,j] = np.exp(-(i-j)/tau)
    W /= np.sum(W, axis=1)[:, np.newaxis]
    return trans_to_cuda(torch.Tensor(W).float())

def forward(model, i, data, position_weights):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, position_weights)

def train(model, train_data, epoch, writer):
    position_weights = compute_weights(train_data.len_max, model.tau)
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    count = 0
    for i, j in zip(slices, np.arange(len(slices))): # 每一个batch
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data, position_weights)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if writer:
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * len(slices) + count)
        count += 1
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    if writer:
        writer.add_scalar('loss/train_loss', total_loss.mean().item(), epoch)

def test(model, test_data, epoch, writer):
    position_weights = compute_weights(test_data.len_max, model.tau)
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data, position_weights)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    if writer:
        print('epoch: ', epoch)
        writer.add_scalar('eval/hit', hit, epoch)
        writer.add_scalar('eval/mrr', mrr, epoch)
    return hit, mrr

def run(model, train_data, test_data, epoch, writer): # 一个epoch
    train(model, train_data, epoch, writer)
    hit, mrr = test(model, test_data, epoch, writer)
    return hit, mrr