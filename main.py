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
from models import FL_PMttack  

def Parser():
    parser = argparse.ArgumentParser(description="RobustFM")
    parser.add_argument('--ptb_rate', type=float, default=0.1, help="Rate of perturbation for the PTB attack")
    parser.add_argument('--seed', type=int, default=42, help="Seed for random number generation")
    parser.add_argument('--dataset', type=str, default='citeseer', help="Name of the dataset to use, 'cora', 'citeseer'")
    parser.add_argument('--hidden', type=int, default=16, help="Number of hidden units in the model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for regularization")
    parser.add_argument('--epochs', type=int, default=250, help="Number of training epochs")


    return parser


def test(new_adj, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(features, new_adj, labels, idx_train, idx_val, patience=30) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
    else:
        gcn.eval()
        output = gcn.predict(features.to(device), new_adj.to(device)).cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item(), gcn


def main(args):
    ptb_rate = args.ptb_rate
    seed = args.seed
    dataset = args.dataset
    hidden = args.hidden
    dropout = args.dropout
    epochs =  args.epochs
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ############### DATA - import data ##############
 
    data = Dataset(root='/content', name='citeseer', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    features = normalize_feature(features)

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    #idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8)
    #idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.54126, test_size=0.40241)

    ##### Set the targeted model #####
    target_gcn = GCN(nfeat=features.shape[1],nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device, lr=0.01)
    target_gcn = target_gcn.to(device)

    #--- train the model on clean data - and test it, print accuracy
    target_gcn.fit(features, adj, labels, idx_train, idx_val,train_iters=200 , patience=300)
    print('=== testing GCN on clean graph ===')
    print(f" Accuracy = {test(adj, target_gcn)}")

    #---- get predictions of all nodes
    fake_labels = target_gcn.predict(features.to(device), adj.to(device))
    fake_labels = torch.argmax(fake_labels, 1).cpu()
    # Besides, we need to add the idx into the whole process
    idx_fake = np.concatenate([idx_train,idx_test])
    

    ###### set the perturbation rate #####
    perturbations = int(ptb_rate * (adj.sum()//2))                # number of edges to be perturbated

    ####### SET UP ATTACK MODEL #######
    print('=== setup attack model ===')
    model = FL_PMttack(model=target_gcn, nnodes=adj.shape[0], loss_type='FL', device=device)
    model = model.to(device)

    fake_labels = target_gcn.predict(features.to(device), adj.to(device))
    fake_labels = torch.argmax(fake_labels, 1).cpu()
    # Besides, we need to add the idx into the whole process
    idx_fake = np.concatenate([idx_train,idx_test])

    idx_others = list(set(np.arange(len(labels))) - set(idx_train))
    fake_labels = torch.cat([labels[idx_train], fake_labels[idx_others]])

    ##### ATTTACK 
    model.attack(features, adj, fake_labels, idx_fake, perturbations, epochs=200)
    
    ##################### Evaluate ######################
    print('=== testing GCN on Poisoning attack ===')
    modified_adj = model.modified_adj
    print(f" Accuracy_ Natural on clean data = {test(adj, target_gcn)}")

    acc_0,model =test(modified_adj, target_gcn)
    print(f" Misclassification Rate_posinoed data on natural = {1-acc_0}")
    print(f" Accuracy _ Natural on poisoned data on natural = {acc_0}")

    acc, gcn_robust = test(modified_adj)
    print(f" Accuracy_ posinoed data on robust = {acc}")
    print(f" Accuracy_ clean on robust = {test(adj, gcn_robust)}")

 
 
   
if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    main(args)
