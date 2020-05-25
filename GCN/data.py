import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

class MyData():
    def __init__(self,data_filename):
        self.data_filename= data_filename
        
        self.load_data()
    
    def parse_index_file(self,filename):
        index=[]
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(self,idx,l):
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def load_data(self):
        """
        ind.dataset_str.x -> 训练的特征向量
        ind.dataset_str.tx  -> 测试的特征向量
        ind.dataset_str.allx
        ind.dataset_str.y ->  训练集数据的标签  one-hot
        ind.dataset_str.ty->  测试集数据的标签  one-hot
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]}
        ind.dataset_str.test.index => the indices of test instances in graph

        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/cora/ind.{}.{}".format(self.data_filename, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file("data/cora/ind.{}.test.index".format(self.data_filename))
        test_idx_range = np.sort(test_idx_reorder)

        if self.data_filename == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        self.adj=adj
        self.features= features
        self.y_train=y_train
        self.y_val=y_val
        self.y_test=y_test
        self.train_mask= train_mask
        self.val_mask=val_mask
        self.test_mask = test_mask


    def _normalize_adj(self,adj):
        """
        公式中的  D^{-0.5} A D^{-0.5}
        """
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        adj_sqrt = np.power(rowsum,-0.5).flatten()
        #对很小的值进行处理
        adj_sqrt[np.isinf(adj_sqrt)] = 0
        adj_sqrt_matrix = sp.diags(adj_sqrt)
        return adj.dot(adj_sqrt_matrix).transpose().dot(adj_sqrt_matrix).tocoo()


    def _sparse_to_tuple(self,sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def normal_and_selfconnct(self,adj):
        adj_nomal = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        self.adj = self._sparse_to_tuple(adj_nomal)

    def normalize_features(self):
        """
        对行进行标准化
        """
        row = np.array(self.features.sum(1))
        r_inv = np.power(row,-1).flatten()
        #对很小的值进行处理
        r_inv[np.isinf(r_inv)] = 0
        r_inv_matrix = sp.diags(r_inv)
        features = r_inv_matrix.dot(self.features)
        self.features = self._sparse_to_tuple(features)