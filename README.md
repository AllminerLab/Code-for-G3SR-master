# G3SR-private
Code for the paper "G3SR: Global Graph Guided Session-based Recommendation"

## Pre-processing

First, please download the `diginetica` dataset and `yoochoose` dataset, and put `train-item-views.csv` and `yoochoose-clicks.dat` into the `datasets` directory，

1. Extract sessions from the raw dataset and split them into a training set and a test set. Take yoochoose as an example:

```
python preprocess.py --dataset yoochoose
```

2. Construct the edge list for unsupervised pre-training (G3SR uses node2vec, a simple yet effective method). Take yoochoose 1/64 as an example

```
python seq2edgelist.py --dataset yoochoose1_64
```

## Phrase 1: Pre-training

Take yoochoose 1/64 as an example:

```
python node2vec/src/main.py --input yoochoose1_64/edgelist.txt --output yoochoose1_64/embeddings --dimensions 100 --walk-length 80 --num-walks 10 --window-size 10 --p 0.25 --q 4 --iter 5 --weighted --directed
```

The code takes the edge list as input and outputs the pre-trained embeddings. Please refer to the code of node2vec for more details.

## Phrase 2: Session-based Recommendation

The code fixes the embeddings obtained in phase 1, and learns task-specific bias and GNN parameters. Take yoochoose 1/64 as an example:

```
python main.py --dataset yoochoose1_64 --pretrain_dataset yoochoose1_64 --model G3SR
```

The above code uses the embeddings pre-trained on yoochoose 1/64 in the session recommendation task of yoochoose 1/64. One may also use a larger dataset to obtain pre-trained embeddings.

Reference:

Zhi-Hong Deng, Chang-Dong Wang, Ling Huang, Jian-Huang Lai and Philip S. Yu. "G3SR: Global Graph Guided Session-Based Recommendation", TNNLS2022.
