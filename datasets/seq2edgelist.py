#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on Aug, 2019

@author: Zhi-Hong Deng
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose_64/yoochoose1_4/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample/'
if opt.dataset == 'yoochoose1_64':
    dataset = 'yoochoose1_64/'
elif opt.dataset =='yoochoose1_4':
    dataset = 'yoochoose1_4/'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose/'
elif opt.dataset =='yoochoose1_2':
    dataset = 'yoochoose1_2/'
elif opt.dataset =='yoochoose1_8':
    dataset = 'yoochoose1_8/'
elif opt.dataset =='yoochoose1_16':
    dataset = 'yoochoose1_16/'
elif opt.dataset =='yoochoose1_32':
    dataset = 'yoochoose1_32/'
else:
    dataset = 'diginetica/'
seq = pickle.load(open(dataset+'all_train_seq.txt', 'rb'))

def zero():
    return 0
seqDict = defaultdict(zero)
itemDict = defaultdict(zero)

print('Counting edges.')
for session in seq:
    itemDict[session[0]] += 1
    for p, q in zip(session[:-1], session[1:]):
        seqDict[(p, q)] += 1
        itemDict[q] += 1

print('Constructing EdgeList.')
with open(dataset+'edgelist.txt','w') as f:
    for k, v in seqDict.items():
        a, b = k
        f.write(str(a)+' '+str(b)+' '+str(v))
        f.write('\n')

seqDict = sorted(seqDict.items(), key=lambda x: x[1], reverse=True) # 高到低排序
print(seqDict[:50])
print('Done.')
