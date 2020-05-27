import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    if layer_name in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
    else:
        _LAYER_UIDS[layer_name] = 1
        return 1

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x,m,sparse=False):
    """对于稀疏向量和稠密向量两种的向量相乘"""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x,m)
    else:
        res = tf.matmul(x,m)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '-' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

        logging = kwargs.get('logging',False)
        self.logging = logging

        self.spare_inputs = False

    def _call(self,inputs):
        return inputs
    
    def __call__(self,inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.spare_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            
            outputs = self._call(inputs)
            
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name+'/vars/'+var,self.vars[var])


class Dense(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.spare_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias


        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name+'_vars'):
            self.vars['weights'] = glorot([input_dim,output_dim],name='weights')
            
            if self.bias:
                self.vars['bias'] = zeros([output_dim],name='bias')
        
        if self.logging:
            self._log_vars()

    def _call(self,inputs):
        x = inputs

        #dropout
        if self.spare_inputs:
            x = sparse_dropout(x,1-self.dropout,self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,1-self.dropout)

        # transform
        output = dot(x,self.vars['weights'],sparse=self.spare_inputs)

        if self.bias:
            output +=self.vars['bias']
            
        return self.act(output)
        
class GraphConvolution(Layer):
    """单层图卷积"""
    def __init__(self,input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False,**kwargs):
        super(GraphConvolution,self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.support = placeholders['support']#单层卷积层中卷积的次数
        self.placeholders = placeholders
        self.spare_inputs = sparse_inputs
        self.act = act
        self.bias= bias
        self.featureless=featureless   #输入的数据带不带特征矩阵

        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope(self.name+'_vars'):
            for i in range(len(self.support)):
                self.vars['weights_'+str(i)] = glorot([input_dim,output_dim],name='weight_'+str(i))
            
            if self.bias:
                self.vars['bias'] = zeros([output_dim],name='bias')

        if self.logging:
            self._log_vars()

    def _call(self,inputs):
        x = inputs

        # dropout
        if self.spare_inputs:
            x = sparse_dropout(x,1-self.placeholders['dropout'],self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,1-self.placeholders['dropout'])

        
        # support是处理过的理解矩阵  具体的是 {D}^{-1/2}\tilde{A}^{-1/2}X
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_x = dot(x,self.vars['weights_'+str(i)],sparse=self.spare_inputs)
            else:
                pre_x = self.vars['weights_'+str(i)]
            support = dot(self.support[i],pre_x,sparse=True)# 一层卷积的结果
            supports.append(support) # 一层图卷积层的多次卷积
        output = tf.add_n(supports)
        
        if self.bias:
            output +=self.vars['bias']
        
        return self.act(output)
        




def glorot(shape,name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
    return tf.Variable(initial,name=name)

def zeros(shape,name=None):
    initial = tf.zeros(shape,dtype=tf.float32)
    return tf.Variable(initial,name=name)

def ones(shape,name=None):
    initial = tf.ones(shape,dtype=tf.float32)
    return tf.Variable(initial,name=name)

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

