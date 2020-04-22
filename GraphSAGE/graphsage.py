"""
author: 酱油 
time:2020-04-22
paper:https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform,Zeros
from tensorflow.python.keras.layers import Input,Dense,Dropout,Layer,LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

class MeanAggregator(Layer):
    def __init__(self, units, input_dim, neigh_max, concat=True, 
                 dropout_rate=0.0, 
                 activation=tf.nn.relu, 
                 l2_reg=0,
                 use_bias=False,
                 seed=1024, **kwargs):
        super(MeanAggregator,self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.activation =activation
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.seed = seed
        self.input_dim = input_dim
    
    def build(self,input_shapes):
        self.neigh_weights = self.add_weight(
                                shape=(self.input_dim,self.units),
                                initializer=glorot_uniform(seed=self.seed),
                                regularizer = l2(self.l2_reg),
                                name="neigh_weights")
        
        if self.use_bias:
            self.bias=self.add_weight(
            shape=(self.units,),initializer = Zeros(),name="bias_weight")
            
        self.dropout = Dropout(self.dropout_rate)
        self.built = True
    
    def call(self,inputs,training=None):
        features,node,neighbours  = inputs
        
        node_feat = tf.nn.embedding_lookup(features,node)
        neigh_feat = tf.nn.embedding_lookup(features,neighbours)
        
        node_feat = self.dropout(node_feat,training = training)
        neigh_feat = self.dropout(neigh_feat,training = training)
        
        concat_feat = tf.concat([neigh_feat,node_feat],axis = 1)
        # 1 * input_dims
        concat_mean = tf.reduce_mean(concat_feat, axis=1)
        
        output = tf.matmul(concat_mean,self.neigh_weights) #1 * units
        if self.use_bias:
            output +=self.bias
        
        if self.activation:
            output = self.activation(output)
            
        output._uses_learning_phase = True
        return output
    
    def get_config(self):
        config = {'units':self.units,
                 'concat':self.concat,
                 'seed':self.seed}
        base_config = super(MeanAggregator,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True,
              activation=tf.nn.relu,
              aggregator_type='mean', 
              dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]

    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])  #

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model
    
    
def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):
   #self_loop : 是否自连接
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]
    if sample_num:
        if self_loop:
            sample_num -=1
        #采样规则
        sample_neighs = [
           list(_sample(neigh,sample_num,replace=False)) if len(neigh) >=sample_num else list(
           _sample(neigh,sample_num,replace=True)) for neigh in neighs]
    
        if self_loop:
            sample_neighs=[
                 samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        if shuffle:
            #多个节点的邻居节点
            sample_neighs = [list(np.random.permutation(x)) for x in sample_neighs]
    else:
        #不进行采样
        sample_neighs = neighs
    # 所有节点的邻居节点，所有节点的邻居节点的个数
    return np.asarray(sample_neighs),np.asarray(list(map(len,sample_neighs)))