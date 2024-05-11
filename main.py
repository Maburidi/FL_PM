import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


import zipfile
import json
import platform
from sklearn.model_selection import train_test_split

import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import os.path as osp
from torch.nn.modules.module import Module

import torch.nn as nn
import math
from copy import deepcopy
from sklearn.metrics import f1_score
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import scipy.io
import argparse



from Datasets import Dataset, PrePtbDataset  
from defense import * 
from defense import GCN 
from utils import * 
from models import ProjMetattack 

def Parser():
    parser = argparse.ArgumentParser(description="RobustFM")
    parser.add_argument('--ptb_rate', type=float, default=0.1, help="Rate of perturbation for the PTB attack")
    parser.add_argument('--seed', type=int, default=42, help="Seed for random number generation")
    parser.add_argument('--dataset', type=str, default='citeseer', help="Name of the dataset to use, 'cora', 'citeseer'")
    parser.add_argument('--hidden', type=int, default=16, help="Number of hidden units in the model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for regularization")
    parser.add_argument('--epochs', type=int, default=250, help="Number of training epochs")


    return parser

def main(args):
    #no_cuda = args.no_cuda
    ptb_rate = args.ptb_rate
    seed = args.seed
    dataset = args.dataset
    #attack = args.attack   #random, meta, nettack
    hidden = args.hidden
    dropout = args.dropout
    epochs =  args.epochs
    
    #only_gcn= only_gcn.args

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ############### DATA - import data ##############
    data = Dataset(root='/content', name='cora', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    ###### set the perturbation rate #####
    perturbations = int(ptb_rate * (adj.sum()//2))                # number of edges to be perturbated
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    ###### Set the targeted model ##########
    surrogate = GCN(nfeat=features.shape[1],nhid=16, nclass=labels.max().item() + 1, dropout=dropout, device=device,
                 with_relu=False, with_bias=True, weight_decay=5e-4, lr=0.01)
    # train it on clean data
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)


    ####### Perform the attack on the graph data
    lambda_ = 1
    model = ProjMetattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
    model = model.to(device)
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)

    ## get the attached graoh data 
    pertrubed_matrix = model.modified_adj
    ## predict - use model trained on clean data to do predictions based on pertrubed data (pertrubations are generated using ProjMettaAttack) 
    output = model.Model_forward(features.to('cuda'), pertrubed_matrix.to('cuda'))

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))




 
 






   
if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    main(args)
