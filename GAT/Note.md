# GRAPH ATTENTION NETWORKS

----
作者：Petar Velickovi ˇ c´、Guillem Cucurull∗
链接：https://arxiv.org/pdf/1710.10903.pdf
code：https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/
会议： ICLR 2018
---

###  GAT模型细节：
1.  GRAPH ATTENTIONAL LAYER
模型的输入： 节点特征集合，$h = \{ \vec h_1,\vec h_2...\vec h_N\},\vec h_i \in R^F$
其中N是节点的个数，F是节点的特征维度。
该层的输出:  节点维度的转换，$h^ {'} = \{ \vec h_1^ {'},\vec h_2^ {'}...\vec h_N^ {'}\},\vec h_i^ {'} \in R^{F'}$

该模型主要的创新点是对于节点i的不同邻居节点可能对节点i的影响程度不同，所以该模型主要将的就是如何学习这不同程度的权重以及如何通过聚合邻居节点来更新自身的节点表示。

学习注意力权重的方式如下：$$e_{ij} = a( W\vec h_i,W\vec h_j)$$
其中a表示单层的神经网络
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e9dfcc06f7f6d466.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

eij表示的是节点i和节点j的一个相关程度，可以理解为节点相似程度，为后面计算注意力权值做铺垫。$j \in N_i$表示节点i的所有一阶邻居节点，保留括节点i，也就是加上自循环。所以节点i的邻居节点的所有eij记性softmax得到注意力权重：
$$a_{ij} = softmax_j(e_{ij})=\frac{exp(e_{ij})}{\sum_{k \in N_i} exp(e_{ik})}$$
其中$\vec a \in R^{2F'}$是一个2F ' 的向量。对于第一个公式中a在试验中采用的LeakyReLU非线性激活函数，因此注意力层的总体公式为：
![注意力公式](https://upload-images.jianshu.io/upload_images/3426235-8057359e965ab0dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中||表示concatenation。

得到注意力权值之后，就是对节点i进行更新，具体的更新公式：
![image.png](https://upload-images.jianshu.io/upload_images/3426235-fa817634cf10446f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中aij是前面学习到注意力权重，W是将hj进行维度转换，在对注意力权值相乘，得到i的更新节点表示。其中Ni中包括了i节点本身，也就是加入了节点本身的特征。

GAT在此之后又提出了mutil-head，就是计算多了注意力权值，通过不同的权值更细腻的去聚合邻居节点的信息。具体的公式：
![image.png](https://upload-images.jianshu.io/upload_images/3426235-9909eb4e1bbce5b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中的K表示的就是多个注意力权重，那么对于节点i得到的就是1 * k * F ' .接下来在预测层为了让结果更稳定，这里不是将k个F ' 向量进行拼接，而是取平局值。得到i的更新向量表示：![image.png](https://upload-images.jianshu.io/upload_images/3426235-6db67593d0f5b679.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
也就是这幅图所表示的。
![image.png](https://upload-images.jianshu.io/upload_images/3426235-92c615bb7bb9f070.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
总结
优点：对于不同的节点具有不同的权重，这也是注意力机制本身的优点，而且多注意力机制感觉更加的细腻。

缺点：只有一阶邻居的信息，没有考虑更高阶的节点影响。