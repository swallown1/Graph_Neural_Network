

import argparse
import datetime
import time
import os

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from load_data import *

if __name__ == "__main__":
    # seed = 123 # use only for unit testing
    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', '/data/ml_1m', 'Dataset string.')   
    flags.DEFINE_string('model', 'gcn', 'Model string.')  
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden', [500, 75], 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('feat_hidden', 64, 'Number of units in feat_hidden size.')
    flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('feature', True, 'Dropout rate (1 - keep probability).')

    SELFCONNECTIONS = False
    SPLITFROMFILE = True
    VERBOSE = True

    NUMCLASSES = 5

    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasplit_path = current_dir+'/data/ml_1m/split'+str(seed)+'.pickle'

    print("Using random dataset split ...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = create_split_data(current_dir,
                                            FLAGS.dataset,seed=seed,testing=False,
                                            datasplit_path=datasplit_path,
                                            datasplit_from_file=SPLITFROMFILE,
                                            verbose=VERBOSE)
    
    num_users, num_items = adj_train.shape

    num_side_features = 0

    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')
     
    #是否考虑自身连接
    if not FLAGS.feature:
        # 对角线矩阵
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')
        #用户mutil-hot 特征向量 +  num_u * num_u的0矩阵
        u_features,v_features = preprocess_user_item_features(u_features,v_features)
    elif FLAGS.feature and u_features is not None and v_features is not None:
        # use features as side information and node_id's as node input features

        print("Normalizing feature vectors...")
        #将所有的打分，即节点的边进行Normalizing
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)
        
        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)
        # 用户的特征维度
        num_side_features = u_features_side.shape[1]
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)
    
    


