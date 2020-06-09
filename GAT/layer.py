from keras import activations,constraints,initializers,regularizers
from keras.layers import Dropout,LeakyReLU,Layer
from keras import backend as K

class GraphAtten(Layer):
    def __init__(self,F_,attn_heads=1,
                attn_heads_reduction='average',dropout_rate=0.5,
                activation='relu',use_bias=True,
                kernel_initializer = 'glorot_uniform',
                bias_initializer='zeros',
                attn_kernel_initializer='glorot_uniform',
                kernel_regularizer=None,bias_regularizer=None,
                attn_kernel_regularizer=None,activity_regularizer=None,
                kernel_constraint=None,bias_constraint=None,
                attn_kernel_constraint=None,**kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        
        self.F_ = F_   #输出节点特征维度 F’
        self.attn_heads = attn_heads  #表示mutil-head 的个数k
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation=activations.get(activation) #激活函数
        self.use_bias = use_bias  #使用偏置
        
        #权重初始化方式
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_regularizer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        
        # 正则化方式
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        
        self.supports_masking = False
        
        # Populated by build()
        self.kernels = []       # 公式 中的W 
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads
        
        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads
        else:
            self.output_dim = self.F_
        super(GraphAtten,self).__init__(**kwargs)
        
    def build(self,input_shape):
        """搭建模型结构"""
        assert len(input_shape) >=2
        F = input_shape[0][-1]
        
        for k in range(self.attn_heads):
            # 使用layer 类中的add_weight方式初始化参数
            weight = self.add_weight(shape=(F,self.F_),
                                    initializer = self.kernel_initializer,
                                    regularizer = self.kernel_regularizer,
                                    constraint = self.kernel_constraint,
                                    name='kernel_{}'.format(k))
            self.kernels.append(weight)
            
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      name='kernel_{}'.format(k))
                self.biases.append(bias)
            
            # attention
            attn_kernel_self = self.add_weight(shape=(self.F_,1),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer = self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name='attn_kernel_self_{}'.format(k))
            attn_kernel_neighs=self.add_weight(shape=(self.F_,1),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name='attn_kernel_neighs_{}'.format(k))
            self.attn_kernels.append([attn_kernel_self,attn_kernel_neighs])
        
        self.built = True
    
    def call(self,inputs):
        X = inputs[0] #指的是所有的节点 的特征 (N x F)
        A = inputs[1] #指的是邻接矩阵
        
        outputs = []
        for head in range(self.attn_heads):
            weight = self.kernels[head]  # W in paper (F X F')
            attention_weight = self.attn_kernels[head] # 注意力权重 （2F' X 1）
            
            #将mutil-hot特征转换成 1*F' 的向量
            features = K.dot(X,weight) # N X F'
            
            attn_for_self = K.dot(features,attention_weight[0]) #(N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features,attention_weight[1]) # (N x 1), [a_2]^T [Wh_j]
            
            dense = attn_for_self+K.transpose(attn_for_neighs) # N X N
            
            dense = LeakyReLU(alpha=0.2)(dense)
            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense,axis=0)  # (N x N)
            
            #进行dropout
            dropout_attn = Dropout(self.dropout_rate)(dense) # N X N
            dropout_feat = Dropout(self.dropout_rate)(features) #NXF'
            
            #对所有的节点进行聚合更新
            node_features = K.dot(dropout_attn,dropout_feat) # N X F'
            
            if self.use_bias:
                node_features = K.bias_add(node_features,self.biases[head])
            
            outputs.append(node_features)
        
        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # N X (K X F')
        else:
            output = K.mean(K.stack(outputs),axis=0) # N X F'
        
        output = self.activation(output)
        return output
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape