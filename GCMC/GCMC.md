# GCMC：Graph Convolutional Matrix Completion
- 论文 ：  https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf
- 代码 ：  
- 来源 ： KDD2018

------
### Introduction
本文将矩阵补全看作是user-item 二分图的链接预测问题，每种链接边可以看做是一种label（例如多种交互行为：点击，收藏，喜欢，下载等；在评分中，1~5分可以分别看做一种label）。作者提出了一个差异化信息传递(differentiable message passing)的概念，通过在bipartite interaction graph上进行差异化信息传递来学习结点的嵌入，再通过一个bilinear decoder进行链接预测。

### Problem Definition
用户-物品二分图，  G = (W,E,R) W是节点，包括用户节点Wu和物品节点Wv。
E为边((ui,r,vj) ∈ E)表示ui和vj的交互r ，r指的是交互类型(也就是边的类型)。文中将打分1-5作为r，因此R=5

![GCMC](https://upload-images.jianshu.io/upload_images/3426235-0b65de0e8446e650.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

GCMC模型最主要的部分就是Graph Encoder Model：[Zu,Zv]=f(Xu,Xv,M1...Mr)，其中,Xu,Xv是用户和物品的feature 矩阵。$Mr ∈ {0,1}^(Nu x Nv)$是交互的邻接矩阵。Zu，Zv是用户和物品的编码后的嵌入向量。

### GCMC
作者的主要想法是将图按照边的不同类型分割成子图，然后将子图进行分割，进行信息聚合，再进行汇聚的encoder模型。这种局部卷积可以看作是信息在图中不同边上进行传递。具体的公式如下：
![](https://upload-images.jianshu.io/upload_images/3426235-dd3276150b4dec90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中cij是归一化因子，具体的为|N(ui)或 $\sqrt{|N(u_i)|N(v_j)}$。Wr是对于每个不同的边类型有一个不同的参数矩阵。$x^v_j$是item j的初始特征，然后对于用户节点i，将所有同类型的节点聚合过来，$\sum_{j \in N_r(u_i)} u_{j->i,r} $  $r \in R$

在对不同类型边进行汇聚，
![image.png](https://upload-images.jianshu.io/upload_images/3426235-f7b269d9f942a1a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
上式中accum是一个汇聚操作，例如可以采用连接操作或求和操作。上式操作可以看作是一个卷积层，可以对该层进行多层叠加。

对于卷积后的用户ui进行维度变化，加了一个全连接进行变换：
![](https://upload-images.jianshu.io/upload_images/3426235-3ec4d5ae79923de4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于item i的embedding和用户ui的过程是一致的，通过其所连接的用户进行聚合。
 
**Side Information**
为了更好的将节点的特征融入到模型中，作者在对$h^u_i$做全连接的时候，加入了节点自身的特征：
![](https://upload-images.jianshu.io/upload_images/3426235-03b41c0ac8fb1ee7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体的，先对特征$x_i^u$做一个线性变化得到$f_i^u$。在对$f_i^u$做一个维度变换和$h^u_i$变换后的维度相同，然后将两部分加起来进行激活，得到最终的节点嵌入表示。


**Bilinear decoder**：也就是模型的预测部分，对于得到的$Z^u_i$和$Z^v_j$进行链路预测，具体的就是把不同的rating level分别看做一个类别。bilinear operation后跟一个softmax输出每种类别的概率：
![](https://upload-images.jianshu.io/upload_images/3426235-5e761d2ad3eebb8f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$Q_r$是可训练的edge-type r特定的参数。最终预测的评分是关于上述概率分布的期望：
![image.png](https://upload-images.jianshu.io/upload_images/3426235-d8060bc5c4c3ce2d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### Model Training
本文使用多分类问题的交叉熵损失函数进行最小化损失。
![](https://upload-images.jianshu.io/upload_images/3426235-fb3399108c64cc16.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- Weight sharing
因为不是所有的用户或物品在不同rating level上都拥有相等的评分数量。因此Wr的某些列的参数得不到优化，因此需要对于不同r之间的参数进行共享，具体的是
![](https://upload-images.jianshu.io/upload_images/3426235-5924e142b22f05ed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中Ts是每种r的初始矩阵，所以出，评分越高，Wr中包含的Ts数量越多。这样在可以避免有些Wr中的参数得不到优化。

作者采用了一种基于基础参数矩阵的线性组合的参数共享方法：
![](https://upload-images.jianshu.io/upload_images/3426235-6d141c158f22d6e6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中，nb 是基础权重矩阵的个数， Ps是基础权重矩阵。ars是可学习的系数。


### conclusion

本文最主要的点就是对于边的不同类型，分组进行卷积来汇聚user邻域的item的信息，并将这些汇聚的信息进行一个非线性变换来得到user的表达，
这样的操作可以迭代若干次得到user的最终表示，item同理。最终使用一个双线性变换矩阵Qr来预测user和item之间的边属于类别r的概率。












