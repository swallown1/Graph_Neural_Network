# A Neural Influence Diffusion Model for Social Recommendation


-----

- 论文 ：  https://arxiv.org/pdf/1904.10322.pdf 
- 代码 ：  https://github.com/PeiJieSun/diffnet/ 
- 来源 ：  SIGIR2019



### 摘要
传统的CF 通过 user-item 二部图学习user和item的嵌入表示。
但是在社交推荐中数据的稀疏性，通过利用用户的邻居来弥补数据稀疏而更好的学习user的嵌入表示

本文的出发点是  由于当前的社交网络都是利用用过本地邻居来开发静态模型，没有模拟全球社交网络中的递归扩散，从而导致推荐效果不佳。

而本文是对于每个用户，扩散过程从融合相关特征的初始嵌入和捕获潜在行为
偏好的免费用户潜在向量开始。

提出的模型的关键思想是，我们设计一种分层影响传播结构，以模拟随着社会扩散过程的继续，
用户的潜在嵌入如何演变。

**论文主要贡献**：
本文提出了一个具有分层影响传播结构的扩散模型来模拟社会推荐中的递归动态社会扩散。此外，
DiffNet有一个融合层，这样每个用户和每个项目都可以表示为包含协作和特征内容信息的嵌入。

### 问题的定义
在社会推荐中有两个实体：
用户集合U (|U | = M) ， 项集合V (|V | = N) 。用户项的交互表示出用户的喜好。
由于隐式反馈(例如，看电影、购买物品、听歌曲)更为常见，我们还考虑了具有隐式反馈的推荐场景。

$R \in R^{(MxN)}$ 表示用户隐式反馈的打分矩阵，rai表示用户a对项目i有交互。

社交网络被视为 用户用户 图$G = [U,S \in R{(MxM)}]$,其中U表示用户集合，S表示
用户间的连接。如果用户a信任用户b则sba = 1 否则为0。如果是社交网络是无向图，那么
用户a信任用户b 和用户b信任用户a是一样的  即sba = 1 ^ sab=1。

那么对于a信任的所有用户b的集合表示为  Sa=[b|sba=1]

除此之外，用户a还和属性值相连表示为Xa，其中用户属性矩阵$X \in R^{(d1 x M)}$.
同样的项目i有属性向量yi 来自于项目属性矩阵 $Y \in R^(d2 X N)$.

于是，社会推荐问题可以定义为：

给定一个评分矩阵R，一个社交网络S，以及用户和项目的关联实值特征矩阵X和Y。
目标就是预测用户对项目的偏好：$\hat R = f(R,S,X,Y)$ 其中$\hat R \in R^{(M x N)}$
表示预测用户对项目的喜好。

###  之前的做法

** 经典的嵌入模型**：
1.  将用户和项目都嵌入到一个较小的潜在空间中，这样，每个用户对未知项目的
	预测偏好就变成了相应用户和项目嵌入之间的内部产品：
![image.png](https://upload-images.jianshu.io/upload_images/3426235-bd87e562ef6f2ebe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


2.  SVD++和 1 中的区别在于对于用户的嵌入不分加入了用户a对所有交互过的item的信息，更好的捕获用户a的兴趣
![image.png](https://upload-images.jianshu.io/upload_images/3426235-d0eb9c2d833bb704.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中e Ra = [j : ra j = 1] 是用户a的隐式反馈，yi是项目信息的低维向量。

由于项目和用户都有属性，将各自属性也编码进去可以更好建模嵌入模型，更好的预测用户对项目的兴趣。
![image.png](https://upload-images.jianshu.io/upload_images/3426235-7f90ed4ff103be05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中第一项使用特征工程捕获偏差项，第二项对用户和项目之间的第二级交互进行建模。 不同的基于嵌入的模型的嵌入矩阵公式和优化函数各不相同

例如BPR中对隐式反馈的成对优化函数 效果很不错。在BPR中，对于嵌入矩阵U和V初始化是满足高斯分布，然后再优化函数中加入L2正则化进行不断学习嵌入矩阵。
![image.png](https://upload-images.jianshu.io/upload_images/3426235-0e0cd601cae90d4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中i是用户a交互过的item，j是未交互item，也就是说j是负样本。

**社交推荐模型**:
由于用户的兴趣和其用户的连接会收到影响，由于嵌入模型在推荐中的广泛使用，大部分模型都是在理在嵌入模型上的。这些社会嵌入模型可以归纳为以下两类:基于社会正则化的方法[16,17,24]和基于用户行为增强的方法[7,8]。具体来说，基于社会规则化的方法假设连接用户会在社会影响扩散下表现出类似的嵌入。

除了上述的优化函数，在整体优化函数中加入社会调节项:
![](https://upload-images.jianshu.io/upload_images/3426235-949e425bf2988f76.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中D是对角矩阵，daa = ∑ sbb

一些研究人员认为，社交网络提供了有价值的信息，增强了每个用户的行为，缓解了数据稀疏问题，而不是社会规则化术语。
TrustSVD就是这样一个模型，它显示了最先进的性能[7,8]。由于研究人员已经很好地认识到，每个用户都表现出类似的偏好作为他们的社会联系，用户的社会邻居对物品的隐性反馈可以被视为该用户的辅助反馈，建模过程为:
![](https://upload-images.jianshu.io/upload_images/3426235-e6877aa6cfae61a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中Sa就是用户a信任的用户集合，其考虑的就是加入其信任用户的嵌入，嵌入到用户a自己的嵌入向量中。

问题： 尽管将社交网络用于社交推荐的性能有所提高，但我们注意到，几乎所有当前的社交推荐模型都利用观察到的社交联系(每个用户的社交邻居)进行推荐，同时使用一个静态流程。

然而，社会影响不是一个静态的过程，而是一个递归的过程，随着时间的推移，每个用户都会受到社会关系的影响。每一次，用户都需要在之前的偏好与社交邻居的影响之间进行平衡，形成更新的潜在兴趣。每一次，用户都需要在之前的偏好与社交邻居的影响之间进行平衡，形成更新的潜在兴趣。然后，随着当前用户兴趣的演变，社会邻居的影响也在变化。这个过程在社交网络中是递归扩散的。
因此，目前的解决方案忽略了社会推荐的迭代社会扩散过程。

更糟糕的是，当用户特性可用时，需要重新设计这些社会推荐模型，以利用特性数据在用户之间建立更好的相关性模型。

### 模型介绍
下图是DiffNet的总体结构
![](https://upload-images.jianshu.io/upload_images/3426235-0b46e94d8b4eacd7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

模型的输入是用户项目交互对 <a , i> ，输入是用户a对项目i的打分  rai。DiffNet包含4个部分：the embedding layer、the fusion layer、 the layer-wise influence diffusion layers、 the prediction layer

具体地，

通过获取相关输入，嵌入层输出用户和项目的免费嵌入。

 对于每个用户（项目），融合层通过融合用户（一项）的免费嵌入和相关功能来生成混合用户（项目）嵌入。

然后将融合的用户嵌入发送到影响扩散层。 影响扩散层采用分层结构构建，以建模社交网络中的递归社交扩散过程

这是DiffNet的关键思想。 在影响扩散过程达到稳定之后，输出层将生成用户-项目对的最终预测偏好。

####  Embedding Layer
$P \in R^{(D x M)}$和$Q \in R^{(D x M)}$是用户和项的嵌入矩阵，用来捕获用户和项的协同隐式表示。其中用户a和项目i是one-hot向量，通过嵌入矩阵得到低维的嵌入向量pa和qi

####  Fusion Layer
输入是用户a的初始嵌入向量和属性向量Xa，输出是$h_a^0$，从不同类型的输入数据捕获用户的初始兴趣。具体的说就是一个一层的全连接:
![](https://upload-images.jianshu.io/upload_images/3426235-db2d38eb7431d82e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中W0是转换矩阵  g(x)是非线性激活函数。为了方便标记，我们省略了全连通神经网络中的偏置项。其中通过设置W0来对不同维度的$h_a^0=[x_a,p_a]$转换成统一维度。

对item同理，其中qi是初始嵌入，yi是项目的属性。
![](https://upload-images.jianshu.io/upload_images/3426235-4984b8844d779f1f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###  Influence Diffusion Layers
通过将每个用户a的融合h 0 a的输出从融合层馈送到影响扩散部分，影响扩散层可以模拟社交网络S中用户潜在偏好扩散的动态。影响扩散部分类似于社交网络中信息的扩散，每一层k将用户从前一层的嵌入作为输入，输出用户在当前社会扩散过程完成后更新的嵌入 ，将更新后的用户嵌入发送到第k + 1层进行下一个扩散过程 。

具体的说 更新$h_a^(k+1)$包括两部分：从用户a信任的用户的第层嵌入表示，通过信息聚合得到。其中h^(k+1)_(Sa)是用户a信任的所有用户的影响。
![](https://upload-images.jianshu.io/upload_images/3426235-adb801a6c507ab0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中Pool函数为平均池化，它对所有可信用户的隐藏嵌入在第k层执行一个平均操作。Pool也可以是最大池化，目的是为了选出所有信任用户在第k层的隐藏嵌入中最大元素，形成$h_(Sa)^(k+1)$。

然后，第a个更新的嵌入$h_(a)^(k+1)$是她在第k层的潜在嵌入$h_(a)^(k)$和来自她信任的用户的影响扩散嵌入聚合$h_(Sa)^(k+1)$的组合。
由于不知道每个用户如何平衡这两部分，我们使用了一个非线性神经网络来处理两部分的连接
![](https://upload-images.jianshu.io/upload_images/3426235-26b609a8b10eb0dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中s(k+1)是非线性转换函数。















