from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import _pickle as pkl
import os
import h5py
import pandas as pd
import random

def map_data(data):
    """
    将所有的节点创建字典索引
    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n

def load(dir,fname, seed=1234, verbose=True):
    """ 
    Loads dataset and creates adjacency matrix
    and feature matrix

    fname : str, dataset
    seed: int, dataset shuffling seed
    verbose: to print out statements or not

    Returns
    num_users : int
        Number of users and items respectively

    num_items : int

    u_nodes : np.int32 arrays
        User indices

    v_nodes : np.int32 array
        item (movie) indices

    ratings : np.float32 array
        User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
        item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
        not necessarily all u_nodes[k] or all v_nodes[k] separately.

    u_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    v_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    seed: int,
        For datashuffling seed with pythons own random.shuffle, as in CF-NADE.
    """

    u_features = None
    v_features = None

    print('Loading dataset', fname)

    data_dir = dir + fname

    files = ['/ratings.dat', '/movies.dat', '/users.dat']

    sep = r'\:\:'
    filename = data_dir + files[0]

    dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

    # [1000209 rows x 4 columns] 6040     1097      4.0  956715569.0
    data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

    data_array = data.as_matrix().tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    
    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)

    # Load movie features
    movies_file = data_dir + files[1]

    movies_headers = ['movie_id', 'title', 'genre']
    movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

    # Extracting all genres
    genres = []
    for s in movies_df['genre'].values:
        genres.extend(s.split('|'))

    genres = list(set(genres))
    num_genres = len(genres)

    genres_dict = {g: idx for idx, g in enumerate(genres)}

    # Creating 0 or 1 valued features for all genres
    v_features = np.zeros((num_items, num_genres), dtype=np.float32)
    for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
        # Check if movie_id was listed in ratings file and therefore in mapping dictionary
        if movie_id in v_dict.keys():
            gen = s.split('|')
            for g in gen:
                v_features[v_dict[movie_id], genres_dict[g]] = 1.

     # Load user features
    users_file = data_dir + files[2]
    users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
    users_df = pd.read_csv(users_file, sep=sep, header=None,
                        names=users_headers, engine='python')

    # Extracting all features
    cols = users_df.columns.values[1:]

    cntr = 0
    feat_dicts = []
    for header in cols:
        d = dict()
        feats = np.unique(users_df[header].values).tolist()
        d.update({f: i for i, f in enumerate(feats, start=cntr)})
        feat_dicts.append(d)
        cntr += len(d)

    num_feats = sum(len(d) for d in feat_dicts)

    u_features = np.zeros((num_users, num_feats), dtype=np.float32)
    for _, row in users_df.iterrows():
        u_id = row['user_id']
        if u_id in u_dict.keys():
            for k, header in enumerate(cols):
                u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    # 将user和item的信息转换成multil-hot向量
    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))
    # Number of users = 6040
    # Number of items = 3706
    # Number of links = 1000209
    # Fraction of positive links = 0.0447
    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features

def create_split_data(dir,dataset, seed=1234, testing=False, datasplit_path=None, 
                datasplit_from_file=False,verbose=True):
    """
    分割train/val/test集合  以及二部图的邻接矩阵
    """
    # print(datasplit_path)
    if datasplit_from_file and os.path.isfile(dataset):
        print('Reading dataset splits from file...')
        with open(datasplit_from_file) as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)
        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))
    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load(dir,dataset,seed=seed,verbose=verbose)

        with open(datasplit_path, 'wb+') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    neutral_rating = -1
    #打分种类的标签
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    #初始打分   为 -1
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    #存储打分值对应的索引
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    #分割数据  0.1用户test
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    #验证集数据
    num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    num_train = ratings.shape[0] - num_val - num_test
    #将u list和v list按照[u,v]格式存储
    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    # 将成对的索引改成行索引
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:num_train]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]
    #[u,v]这种存储方式的索引
    train_pairs_idx = pairs_nonzero[0:num_train]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]
    #将[u,v]方式的id索引转换成  [u],[v] index
    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    #对于test方式   将val和train数据集进行合并
    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values

def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)
    #按列拼接
    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features

def normalize_features(feat):
    """将每行进行标准化，即每个打分除以改用户打分总和"""
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm

if __name__ == "__main__":
    pass
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # create_split_data(current_dir,'/data/ml_1m',datasplit_path=current_dir+'/data/ml_1m/split1234.pickle',verbose=True)

    # num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features=load(current_dir,'/data/ml_1m')
