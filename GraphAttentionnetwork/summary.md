## Graph Attention Networks
---
###GAT网络的特点
GAT网络是处理有向图，通常定义入度的节点进行聚合，也是空域GNN，

### GAT的创新点
计算每个节点的邻居节点和自身的权重关系(重要性)，根据权值关系进行聚合，

### GAT的attention满足不对称性
不对称性指的是，节点i对节点j的重要性和节点j对节点i的重要性不同。

具体的是因为设计了两个权重，分别进行对节点i和节点j进行转化，
也就是说，设计了两个权重矩阵，w1，w2处向量 hi和hj。

### 公式中a向量的作用是什么

由w1hi，w2hj得到的两个1xF’ 向量，
在代码中：把一个 2F' x 1的attention kernel当作两个F' x 1
的小kernel，一个负责自注意力，一个负责邻节点注意力。通过用这两
个小kernel分别对 w1hi 和 w2hj 相乘，就能得到2个 N x 1 的
张量，即自注意力指标和邻注意力指标(本人定义的名字)。

### softmax部分的不同
为了保持结构信息，因此在softmax的时候只对节点的邻居节点进行
归一化，不对全局节点进行归一化。

### k个注意力值
GAT的特别之处，在没两个节点之间，计算的不是一个注意力值
而是计算k和注意力值，我的理解就是k和注意力值学习k个不同
方面的注意力权值。

对于k'个注意力值 文中有两种方式进行处理
- 将k个注意力值进行拼接
- 将k个注意力值进行均值

### GAT 的trick
1. 使用attention机制来描述邻居节点的重要性
2. 用邻接矩阵进行卷积操作，也就是空域GCN
3. 引入attention heads 即k  用来扩展attention 机制的channel

### 与GCN的联系与区别
本质上而言：GCN与GAT都是将邻居顶点的特征聚合到中心顶点上（一种aggregate运算）
，利用graph上的local stationary学习新的顶点特征表达。
不同的是GCN利用了拉普拉斯矩阵，GAT利用attention系数。一定程度上
而言，GAT会更强，因为 顶点特征之间的相关性被更好地融入到模型中。

### 为什么GAT适用于有向图？
我认为最根本的原因是GAT的运算方式是逐顶点的运算。就是说在遍历的时候
对所有节点进行计算，因此注意力系数有非对称性，所以适用于处理有向图。

### 为什么GAT适用于inductive任务？

为什么适合处理动态加入节点的。因为我们知道本文使用的是mask attention
也就是针对邻居节点计算权重，所以在加入新的节点的时候，只需要
只需要改变 Ni ，重新计算即可。相比GCN计算全图的方式，每次计算都是对
全图的节点进行更新，学习的参数很大程度与图的结构相关。


### 空域GNN 和 谱域GNN
谱域GNN指的是在图的原空间不容易做卷积，就在图的拉普拉斯
算子的傅里叶域做。

空域GNN指的是直接在图结构上进行卷积。关注于节点之间有无连接。
所以对于空域GNN，直观的理解就是直接用邻接矩阵进行卷积。

所以对于谱域来说可以省略很多的参数，但是缺点是很难处理动态图。
因为节点的变化对图的拉普拉斯会发生变化。

空域GCN:

《Learning Convolutional Neural Networks for Graphs》
[论文笔记：Learning Convolutional Neural Networks for Graphs](https://zhuanlan.zhihu.com/p/27587371)
[特别是论文对应的PPT讲解，非常有参考价值](https://link.zhihu.com/?target=http%3A//www.matlog.net/icml2016_slides.pdf)

谱域GCN：
《Semi-Supervised Classification with Graph Convolutional Networks》
[Semi-Supervised Classification with Graph Convolutional Networks阅读笔记](https://zhuanlan.zhihu.com/p/31067515)
