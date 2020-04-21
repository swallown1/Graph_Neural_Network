from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  #经过一层注意力机制层后节点的维度
        self.attn_heads = attn_heads  #指的是K个注意力权值
        self.attn_heads_reduction = attn_heads_reduction 
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        
        #对参数的初始化方式
        #对于W的初始化 方式
        self.kernel_initializer = initializers.get(kernel_initializer)
        #对偏值的初始化方式
        self.bias_initializer = initializers.get(bias_initializer)
        #其实是公式中 a的初始化  a的目的是将向量转化成 1x1
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        
        #正则化部分
        # 对W的正则化方式
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 对偏值的正则化方式
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        #增加约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False
        
        
        self.kernels = []       # 公式中的W
        self.biases = []        # 偏值参数
        self.attn_kernels = []  # 公式中的  a
        
        #根据连接方式决定节点的维度
        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads
        else:
            self.output_dim = self.F_
        
        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]
        
        #初始化权值
        for head in range(self.attn_heads):
            w = self.add_weight(shape=(F,self.F_),
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint,
                               name='kernel_{}'.format(head))
            self.kernels.append(w)
            
            # 偏值
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_),
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               name='bias_{}'.format(head))
                self.biases.append(bias)
            
            
            #计算attend 部分
            #这里分成两部分算，一个是自身注意力，一个是邻居注意力
            atten_kernel_self = self.add_weight(shape=(self.F_,1),
                               initializer=self.attn_kernel_initializer,
                               regularizer=self.attn_kernel_regularizer,
                               constraint=self.attn_kernel_constraint,
                               name='attn_kernel_self_{}'.format(head))
            
            atten_kernel_neighs = self.add_weight(shape=(self.F_,1),
                               initializer=self.attn_kernel_initializer,
                               regularizer=self.attn_kernel_regularizer,
                               constraint=self.attn_kernel_constraint,
                               name='attn_kernel_neigh_{}'.format(head))
            
            self.attn_kernels.append([atten_kernel_self,atten_kernel_neighs])
        self.built = True
    
    def call(self,inputs):
        X = inputs[0]  #shape  N x F
        A = inputs[1]  #shape  N x N
        
        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  #论文中的W  F X F'
            # 公式中的  a   2F ’ x 1
            attention_kernel = self.attn_kernels[head]
            
            # 公式中的 W hi
            features = K.dot(X,kernel) #N x F'
            
            
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features,attention_kernel[0]) # N x 1 
            attn_for_neighs = K.dot(features,attention_kernel[1]) # N x 1
            
            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs) # N x N
            
            
            # 激活函数
            denses = LeakyReLU(alpha=0.2)(dense)
            
            
            # Mask values before activation
            #主要是为了解决结构性信息丢失，只对邻居节点进行softmax归一化
            # 主要的做法是将为0 部分 (即没有连接) 转化成负无穷   这样exp之后为0
            
            mask = -10e9 * (1.0-A)
            dense +=mask
            
            #进行softmax
            dense = K.softmax(dense)  # N x N
            
            dropout_attn = Dropout(self.dropout_rate)(dense) #(N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)# (N x F')
            
            # 对邻居节点信息进行线性结合
            node_features = K.dot(dropout_attn,dropout_feat) ## (N x F')
            
            
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])
            
            # 将多个attention后的结果放在output中
            outputs.append(node_features)

        # 对于K次结果的聚合
        if self.attn_heads_reduction == 'concat':
            output  = K.concatenate(outputs)# (N x KF')
        else:
            output  = K.mean(K.stack(outputs),axis = 0)# N x F')
        
        output  = self.activation(output)
        
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
        
        
        
        
        
                 
                 