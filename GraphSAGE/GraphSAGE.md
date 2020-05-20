# GraphSAGE
----

论文：InductiveRepresentationLearningonLargeGraphs

作者：WilliamL.Hamilton, RexYing, JureLeskovec

来源：NIPS 2017

代码：https://github.com/williamleif/GraphSAGE

## GraphSAGE提出的背景
对于GCN中存在一些问题，第一就是在GCN训练的时候需要对所有节点的邻接矩阵和特征矩阵放在内存，对于大规模的图上，这存在着问题。其次一个问题是，无法处理动态度，也就是说对于新节点的加入的时候，需要重新训练，对新加入的节点无法处理。

## 概述
---
GraphSAGE是一个inductive模型[什么是inductive](https://mp.weixin.qq.com/s/1DHvLLysMU24dBeLzbSpUA)，也就是说将训练样本中的节点和测试样本的边进行保留，就是说不划分到训练集中。这一点也GraphSAGE可以通过训练已知的节点信息为未知节点生成Embedding。

GraphSAGE中包括两部分，1. sample，针对节点的邻居节点很多的情况，会进行采样，用部分节点的信息进行聚合。 2. aggreGatE，指的是如何用采样节点的Embedding来更新自己的Embedding信息。

来看一下采用和聚合的图：
![](https://upload-images.jianshu.io/upload_images/3426235-f93714acd9b8d8db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 模型
---
1.  节点的Embedding生成
先看一下论文中说的步骤
![](https://upload-images.jianshu.io/upload_images/3426235-a0c82a1c3be6e333.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 就这个伪代码看一下具体的过程。

- 初始化每个节点，通过节点本身的features进行初始化。

- K表达的是卷积的层数，也就是聚合几跳之后节点的信息。

- 对于每个节点v，得到其采样后的邻居节点Embedding  $h_u ,u \in N(v)$,然后进行聚合

-  根据聚合后的邻居Embedding $h_{N_v} $,再加上自身的嵌入向量，对自身的嵌入向量进行更新。然后重复K次这样的步骤。

在这个过程中K除了是卷积层数，同时代表的是聚合器的数量，也是权重矩阵的数量，因为每层网络的权重矩阵的共享的。

2. 采样过程
--- 
之前说由于节点的邻居节点数量可能很大，因此我们值进行采用，聚合部分邻居节点的信息。GraphSAGE采用的是定长的又放回采样。具体的是采样S个，如果邻居节点不够，那就进行又放回的重采用达到S个。如果数量够S个，就进行非重复采用。

3. 聚合器
---
本文中提出很多聚合器，效果最好的是平均聚合器。就是将所有邻居节点Embedding在每个维度进行求均值。
![](https://upload-images.jianshu.io/upload_images/3426235-4309394fa56fbc34.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3426235-026723e226feabf9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


论文还提出了另外两种Aggre:LSTM  aggregator和 Pooling aggregator。

Pooling aggregator：
![](https://upload-images.jianshu.io/upload_images/3426235-9d1bfb8c0121b684.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Pooling aggregator 先对目标顶点的邻接点表示向量进行一次非线性变换，之后进行一次pooling操作(maxpooling or meanpooling)，将得到结果与目标顶点的表示向量拼接，最后再经过一次非线性变换得到目标顶点的第k层表示向量。

LSTM aggregator：
LSTM相比简单的求平均操作具有更强的表达能力，然而由于LSTM函数不是关于输入对称的，所以在使用时需要对顶点的邻居进行一次乱序操作。

### 模型训练
----
论文提出两种学习方式，监督学习和无监督学习。

-  监督学习：无监督损失函数的设定来学习节点embedding 可以供下游多个任务使用，若仅使用在特定某个任务上就是根据任务的不同设置不中的目标函数，比如二分类时，可以采用交叉熵损失。

- 无监督学习：基于图的损失函数希望临近的定点具有相似的向量表示，同时对分类的定点尽量的区分。目标函数：
![](https://upload-images.jianshu.io/upload_images/3426235-1b59a9b0cc1c9ad8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
-  Zu 为节点u通过GraphSAGE生成的embedding
-  节点v是节点u随机游走访达“邻居”。
-   vn ~ Pn(u)表示负采样：节点vn 是从节点u的负采样分布 Pn 采样的，Q为采样样本数。
-  embedding之间相似度通过向量点积计算得到.

与DeepWalk不同的是，这里的顶点表示向量是通过聚合顶点的邻接点特征产生的，而不是简单的进行一个embedding lookup操作得到。


#### 参考
---
[GraphSAGE: 算法原理，实现和应用](https://zhuanlan.zhihu.com/p/79637787)




















































