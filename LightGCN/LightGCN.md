## LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
-----
作者：酱油
论文连接：https://arxiv.org/abs/2002.02126
实现：https://github.com/kuandeng/lightgcn
----
本片论文的观点是由于图神经网络中的特征转换以及非线性激活对协同过滤的性能几乎无影响，反而会加大性能使用，因此本文主要为了简化GCN，仅包含GCN中最重要的组成部分-邻域聚合-用于协作过滤，使得更适合推荐模型。

本文指出NGCF，从图卷积网络（GCN）中汲取灵感，遵循相同的传播规则来细化嵌入：特征变换，邻域聚合和非线性激活。虽然结果很好，但是其设计相当繁重且繁重-许多操作没有理由就直接从GCN继承，很多操作可能对协同过滤不一定存在作用。GCN原本的设计是对属性图上的节点进行分类，而每个节点都有丰富的属性作为输入特征，然而在CF中每个节点只利用ID属性来进行卷积，这可能并不会带来多大益处。通过论文的研究得出从GCN继承的两个操作-特征转换和非线性激活-对NGCF的有效性没有任何贡献，删除它们可以显着提高准确性。因此本文提出了一个名为LightGCN的新模型，其中包括GCN的最重要组成部分-邻域聚合-用于协作过滤。

本文的数据集有3个：Gowalla，Amazon-Book，yelp2018. 
对比模型：NGCF
评优方法：recall，ndcg

### NGCF简介
NGCF模型中，用户(项目)的初始嵌入是用户(项目)ID 嵌入 $e^{(0)}_u$ 
 ($e^{(0)}_i$)。然后，NGCF利用useritem交互图将嵌入传播为：
![公式1](https://upload-images.jianshu.io/upload_images/3426235-3b1da894d0bdb225.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
 其中$e^{(k)}_u$ 和 $e^{(k)}_i$ 表示第k层图卷积后用户和项的嵌入表示。
Ni表示用户的邻接项目，Nu表示项目的邻接用户。通过传播L层，NGCF获得L + 1个嵌入来表示所有的用户和项目。然后利用其得到最终的用户嵌入和项目嵌入。

NGCF很大程度上遵循标准GCN [21]，包括使用非线性激活函数σ（·）和特征转换矩阵W1和W2。 但是，我们认为这两个操作对于协同过滤没有用。 论文指出如果节点有丰富的语义特征非线性激活才能更起作用，如果只利用ID作为输入，那么非线性激活就没有必要。为什么非线性激活以及特征转换会对结果产生负面影响呢？是因为NGCF的恶化是由于训练困难而不是过度拟合，非线性激活的结合进一步加剧了表示能力和泛化性能之间的差异。

### LightGCN模型：
GCN的基本思想是迭代地执行图卷积，将邻居的特征聚合为目标节点的新表示。 这种邻域聚合可以抽象为：

![公式2](https://upload-images.jianshu.io/upload_images/3426235-25975632c377a3e2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

AGG是聚合函数，通过聚合目标节点的邻居节点来更新目标节点表示。现在有很多的聚合方式，GCN中的sum aggregator，GIN中的Means aggregator 以及GraphSage中的LSTM aggregator。虽然大多数任务中都使用的特征变化和非线性变换，使得结果很好，但是对于协同过滤来说是负担。

** Light Graph Convolution (LGC)**

LightGCN中使用sum aggregator并且放弃使用特征转化。在LGC中卷积操作定义为：
![公式3](https://upload-images.jianshu.io/upload_images/3426235-2d1e83747338b1ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中正则化部分来源于GCN的正则化标准，为了嵌入的规模随着图卷积运算的增加而增加；当然也可以使用L1正则化，但是实验证明此正则化方式表现更好。

在LGC中，我们仅聚合连接的邻居，而不集成目标节点本身。此文不进行自连接是因为下面介绍的层组合可以达到类似的自连接效果。

**  Layer Combination ** 
在LightGCN中，需要训练的参数只需要训练用户和项目的嵌入矩阵，在通过k层卷积操作，得到最终嵌入。 在K层LGC之后，考虑每层LGC后的嵌入表示以形成用户（项）的最终表示形式：
![公式4](https://upload-images.jianshu.io/upload_images/3426235-dcc8fc19ab3f9a58.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中，αk≥0表示第k层嵌入在构成最终嵌入中的重要性，也就是注意力权值。既可以作为超参数也可以学习生成。试验中，$\alpha_k$设置为$\frac{1}{(K+1)}$,也是为了使得LightGCN更加的简单。
考虑各层输出结果的原因有三：
1.  由于卷积层数增加，嵌入向量更加平滑，用最后一层卷积结果表示并不好。
2. 不同层的卷积效果有不同的语义，考虑更多的语义使得最后的嵌入表示包含更过信息，层数越高的嵌入向量捕获更高级别的邻近度。
3. 将不同层的嵌入与加权和相结合，可以捕获图卷积和自连接的效果。

通过层级连接也使得在卷积过程中没有考虑自连接，因为层级连接也考虑节点本身每层的原始嵌入表示。

**Model Prediction**
模型的预测部分被定义为：
![公式5](https://upload-images.jianshu.io/upload_images/3426235-1158d57c5028be8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将得到的结果按照分数进行排名。模型的整体结构图如下：
![LightGCN](https://upload-images.jianshu.io/upload_images/3426235-51447a8612e49821.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**矩阵形式**
下面给出模型的矩阵形式。 设用户-项目交互矩阵为R∈RM×N，其中M和N分别表示用户和项目数，如果u与项目i进行交互，则每个条目Rui为1，否则为0。 用户项目图的邻接矩阵为
![公式6](https://upload-images.jianshu.io/upload_images/3426235-8278bbd6f9fffd77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

设第0层嵌入矩阵为E（0）∈R（M + N）×T，其中T为嵌入大小。 然后我们可以得出LGC的矩阵等效形式为：
![公式7](https://upload-images.jianshu.io/upload_images/3426235-e89a73616bc4b5f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中D是(M +N)×(M +N)的对角矩阵，其中每个条目Di i表示邻接矩阵A的第i行向量中非零条目的数量（也称为度矩阵），最终的用于预测部分的所有嵌入矩阵E为：
![公式8](https://upload-images.jianshu.io/upload_images/3426235-f76acaa637ce56de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 模型训练
因为模型需要学习的参数只有初始的嵌入矩阵，其复杂度类似于矩阵分解，同时本文使用的是BPR损失，因此文中的损失函数为：
     
![loss损失   ](https://upload-images.jianshu.io/upload_images/3426235-6af74898d40fc81a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

本实验中没有进行负采样操作，同时也没有使用dropout来方式过拟合，因为LightGCN中没有特征变换权重矩阵，因此在嵌入层上强制执行L2正则化足以防止过度拟合。这展示了LightGCN简单的优势-比NGCF更容易训练和调整。对于注意力权值$\alpha_k$没有通过自学习，作者认为可能是因为训练数据没有足够的信号来学习可以推广到未知数据的良好α。 我们还尝试从验证数据中学习α   ，该方法学习了验证数据中的超参数。 性能略有提高（小于1％）。







