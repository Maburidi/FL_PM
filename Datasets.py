import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
import urllib 
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


class Dataset():
    """Dataset class contains four citation network datasets "cora", "cora-ml", "citeseer" and "pubmed",
    and one blog dataset "Polblogs". Datasets "ACM", "BlogCatalog", "Flickr", "UAI",
    "Flickr" are also available. See more details in https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph#supported-datasets.
    The 'cora', 'cora-ml', 'polblogs' and 'citeseer' are downloaded from https://github.com/danielzuegner/gnn-meta-attack/tree/master/data, and 'pubmed' is from https://github.com/tkipf/gcn/tree/master/gcn/data.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be chosen from ['cora', 'citeseer', 'cora_ml', 'polblogs',
        'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr']
    setting : string
        there are two data splits settings. It can be chosen from ['nettack', 'gcn', 'prognn']
        The 'nettack' setting follows nettack paper where they select the largest connected
        components of the graph and use 10%/10%/80% nodes for training/validation/test .
        The 'gcn' setting follows gcn paper where they use the full graph and 20 samples
        in each class for traing, 500 nodes for validation, and 1000
        nodes for test. (Note here 'netack' and 'gcn' setting do not provide fixed split, i.e.,
        different random seed would return different data splits)
    seed : int
        random seed for splitting training/validation/test.
    require_mask : bool
        setting require_mask True to get training, validation and test mask
        (self.train_mask, self.val_mask, self.test_mask)

    Examples
    --------
	We can first create an instance of the Dataset class and then take out its attributes.

	>>> from deeprobust.graph.data import Dataset
	>>> data = Dataset(root='/tmp/', name='cora', seed=15)
	>>> adj, features, labels = data.adj, data.features, data.labels
	>>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    """

    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False):
        self.name = name.lower()
        self.setting = setting.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs',
                'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr'], \
                'Currently only support cora, citeseer, cora_ml, ' + \
                'polblogs, pubmed, acm, blogcatalog, flickr'
        assert self.setting in ['gcn', 'nettack', 'prognn'], "Settings should be" + \
                        " choosen from ['gcn', 'nettack', 'prognn']"

        self.seed = seed
        # self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name

        if platform.system() == 'Windows':
            self.root = root
        else:
        	self.root = osp.expanduser(osp.normpath(root))

        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        self.require_mask = require_mask

        self.require_lcc = False if setting == 'gcn' else True
        self.adj, self.features, self.labels = self.load_data()

        if setting == 'prognn':
            assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml', 'polblogs'], "ProGNN splits only " + \
                        "cora, citeseer, pubmed, cora_ml, polblogs"
            self.idx_train, self.idx_val, self.idx_test = self.get_prognn_splits()
        else:
            self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_train_val_test(self):
        """Get training, validation, test splits according to self.setting (either 'nettack' or 'gcn').
        """
        if self.setting == 'nettack':
            return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels, seed=self.seed)
        if self.setting == 'gcn':
            return get_train_val_test_gcn(self.labels, seed=self.seed)

    def get_prognn_splits(self):
        """Get target nodes incides, which is the nodes with degree > 10 in the test set."""
        url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/' + \
                     'master/splits/{}_prognn_splits.json'.format(self.name)
        json_file = osp.join(self.root,
                '{}_prognn_splits.json'.format(self.name))

        if not osp.exists(json_file):
            self.download_file(url, json_file)
        # with open(f'/mnt/home/jinwei2/Projects/nettack/{dataset}_nettacked_nodes.json', 'r') as f:
        with open(json_file, 'r') as f:
            idx = json.loads(f.read())
        return np.array(idx['idx_train']), \
               np.array(idx['idx_val']), np.array(idx['idx_test'])

    def load_data(self):
        print('Loading {} dataset...'.format(self.name))
        if self.name == 'pubmed':
            return self.load_pubmed()

        if self.name in ['acm', 'blogcatalog', 'uai', 'flickr']:
            return self.load_zip()

        if not osp.exists(self.data_filename):
            self.download_npz()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_file(self, url, file):
        print('Dowloading from {} to {}'.format(url, file))
        try:
            urllib.request.urlretrieve(url, file)
        except:
            raise Exception("Download failed! Make sure you have \
                    stable Internet connection and enter the right name")

    def download_npz(self):
        """Download adjacen matrix npz file from self.url.
        """
        print('Downloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        try:
            print('Downloading', url)
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_zip(self, name):
        url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/other_datasets/{}.zip'.\
                format(name)
        try:
            print('Downlading', url)
            urllib.request.urlretrieve(url, osp.join(self.root, name+'.zip'))
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def load_zip(self):
        data_filename = self.data_folder + '.zip'
        name = self.name
        if not osp.exists(data_filename):
            self.download_zip(name)
            with zipfile.ZipFile(data_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root)

        feature_path = osp.join(self.data_folder, '{0}.feature'.format(name))
        label_path = osp.join(self.data_folder, '{0}.label'.format(name))
        graph_path = osp.join(self.data_folder, '{0}.edge'.format(name))

        f = np.loadtxt(feature_path, dtype = float)
        l = np.loadtxt(label_path, dtype = int)
        features = sp.csr_matrix(f, dtype=np.float32)
        # features = torch.FloatTensor(np.array(features.todense()))
        struct_edges = np.genfromtxt(graph_path, dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        n = features.shape[0]
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n, n), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        label = np.array(l)

        return sadj, features, label

    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)


        test_idx_file = "ind.{}.test.index".format(dataset)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def get_mask(self):
        idx_train, idx_val, idx_test = self.idx_train, self.idx_val, self.idx_test
        labels = self.onehot(self.labels)

        def get_mask(idx):
            mask = np.zeros(labels.shape[0], dtype=np.bool)
            mask[idx] = 1
            return mask

        def get_y(idx):
            mx = np.zeros(labels.shape)
            mx[idx] = labels[idx]
            return mx

        self.train_mask = get_mask(self.idx_train)
        self.val_mask = get_mask(self.idx_val)
        self.test_mask = get_mask(self.idx_test)
        self.y_train, self.y_val, self.y_test = get_y(idx_train), get_y(idx_val), get_y(idx_test)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx




def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    #assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test





def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20: ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test

class PrePtbDataset:
    """Dataset class manages pre-attacked adjacency matrix on different datasets. Note metattack is generated by deeprobust/graph/global_attack/metattack.py. While PrePtbDataset provides pre-attacked graph generate by Zugner, https://github.com/danielzuegner/gnn-meta-attack. The attacked graphs are downloaded from https://github.com/ChandlerBang/Pro-GNN/tree/master/meta.

    Parameters
    ----------
    root :
        root directory where the dataset should be saved.
    name :
        dataset name. It can be choosen from ['cora', 'citeseer', 'polblogs', 'pubmed']
    attack_method :
        currently this class only support metattack and  nettack. Note 'meta', 'metattack' or 'mettack' will be interpreted as the same attack.
    seed :
        random seed for splitting training/validation/test.

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset, PrePtbDataset
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> # Load meta attacked data
    >>> perturbed_data = PrePtbDataset(root='/tmp/',
                            name='cora',
                            attack_method='meta',
                            ptb_rate=0.05)
    >>> perturbed_adj = perturbed_data.adj
    >>> # Load nettacked data
    >>> perturbed_data = PrePtbDataset(root='/tmp/',
                            name='cora',
                            attack_method='nettack',
                            ptb_rate=1.0)
    >>> perturbed_adj = perturbed_data.adj
    >>> target_nodes = perturbed_data.target_nodes
    """


    def __init__(self, root, name, attack_method='meta', ptb_rate=0.05):

        if attack_method == 'mettack' or attack_method == 'metattack':
            attack_method = 'meta'

        assert attack_method in ['meta', 'nettack'], \
            ' Currently the database only stores graphs perturbed by metattack, nettack'
        # assert attack_method in ['meta'], \
        #     ' Currently the database only stores graphs perturbed by metattack. Will update nettack soon.'

        self.name = name.lower()
        assert self.name in ['cora', 'citeseer', 'polblogs', 'pubmed', 'cora_ml'], \
            'Currently only support cora, citeseer, pubmed, polblogs, cora_ml'

        self.attack_method = attack_method
        self.ptb_rate = ptb_rate
        self.url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/{}/{}_{}_adj_{}.npz'.\
                format(self.attack_method, self.name, self.attack_method, self.ptb_rate)
        self.root = osp.expanduser(osp.normpath(root))
        self.data_filename = osp.join(root,
                '{}_{}_adj_{}.npz'.format(self.name, self.attack_method, self.ptb_rate))
        self.target_nodes = None
        self.adj = self.load_data()

    def load_data(self):
        if not osp.exists(self.data_filename):
            self.download_npz()
        print('Loading {} dataset perturbed by {} {}...'.format(self.name, self.ptb_rate, self.attack_method))

        if self.attack_method == 'meta':
            warnings.warn("The pre-attacked graph is perturbed under the data splits provided by ProGNN. So if you are going to verify the attacking performance, you should use the same data splits  (setting='prognn').")
            adj = sp.load_npz(self.data_filename)

        if self.attack_method == 'nettack':
            # assert True, "Will update pre-attacked data by nettack soon"
            warnings.warn("The pre-attacked graph is perturbed under the data splits provided by ProGNN. So if you are going to verify the attacking performance, you should use the same data splits  (setting='prognn').")
            adj = sp.load_npz(self.data_filename)
            self.target_nodes = self.get_target_nodes()
        return adj

    def get_target_nodes(self):
        """Get target nodes incides, which is the nodes with degree > 10 in the test set."""
        url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/nettack/{}_nettacked_nodes.json'.format(self.name)
        json_file = osp.join(self.root,
                '{}_nettacked_nodes.json'.format(self.name))

        if not osp.exists(json_file):
            self.download_file(url, json_file)
        # with open(f'/mnt/home/jinwei2/Projects/nettack/{dataset}_nettacked_nodes.json', 'r') as f:
        with open(json_file, 'r') as f:
            idx = json.loads(f.read())
        return idx["attacked_test_nodes"]

    def download_file(self, url, file):
        print('Dowloading from {} to {}'.format(url, file))
        try:
            urllib.request.urlretrieve(url, file)
        except:
            raise Exception("Download failed! Make sure you have \
                    stable Internet connection and enter the right name")

    def download_npz(self):
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception("Download failed! Make sure you have \
                    stable Internet connection and enter the right name")


class RandomAttack():

    def __init__(self):
        self.name = 'RandomAttack'

    def attack(self, adj, ratio=0.4):
        print('random attack: ratio=%s' % ratio)
        modified_adj = self._random_add_edges(adj, ratio)
        return modified_adj

    def _random_add_edges(self, adj, add_ratio):

        def sample_zero_forever(mat):
            nonzero_or_sampled = set(zip(*mat.nonzero()))
            while True:
                t = tuple(np.random.randint(0, mat.shape[0], 2))
                if t not in nonzero_or_sampled:
                    yield t
                    nonzero_or_sampled.add(t)
                    nonzero_or_sampled.add((t[1], t[0]))

        def sample_zero_n(mat, n=100):
            itr = sample_zero_forever(mat)
            return [next(itr) for _ in range(n)]

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        non_zeros = [(x, y) for x,y in np.argwhere(adj != 0) if x < y] # (x, y)

        added = sample_zero_n(adj, n=int(add_ratio * len(non_zeros)))
        for x, y in added:
            adj[x, y] = 1
            adj[y, x] = 1
        return adj

