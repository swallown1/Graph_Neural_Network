# Graph Learning Approaches to Recommender Systems: A Review

##  摘要
近年来，基于图学习的推荐系统(GLRS)这一新兴话题得到了快速的发展。
GLRS主要采用先进的图形学习方法来建模用户的偏好和意图，以及项目的特征和受欢迎程度的推荐系统(RS)。与传统的RS(包括基于内容的过滤和协作过滤)不同，GLRS构建在简单或复杂的图形上，其中各种对象(例如用户、项和属性)显式或隐式地连接在一起。与图形学的快速发展,探索和利用同构或异构关系图是一个有前途的方向为构建先进的RS。本文我们提供GLRS的系统回顾,他们如何获得知识从图表提高对推荐的准确性、可靠性和可解释性。

首先，我们对GLRS进行了描述和形式化，然后对这一新研究领域的关键挑战进行了总结和分类。然后，我们调查了该地区最近和最重要的发展。最后，我们分享了这个充满活力的领域的一些新的研究方向。

##  1. 简介
推荐系统(RS)作为人工智能(AI)最受欢迎和最重要的应用之一，已经被广泛应用，帮助许多流行的web内容共享和电子商务用户更容易地找到相关的内容、产品或服务。
同时，图数据学习(Graph Learning, GL)，即:基于图结构数据的机器学习作为一种新兴的人工智能技术，近年来发展迅速，前景广阔。得益于GL对关系数据的学习能力，近年来出现了一种基于GL的新型RS模式，即基于图学习的推荐系统(GLRS)，并得到了广泛的研究。

**这促使我们系统地回顾动机:为什么RS要学习图形?**
> RS中的大多数数据本质上是一个图结构。在现实世界中，我们周围的大多数物体都或明或暗地彼此相连;换句话说，我们生活在一个图形的世界里。这种特征在RS中更为明显，**这里考虑的对象，包括用户、物品、属性等，彼此之间紧密相连，通过各种关系相互影响**，如图1所示。在实践中，RS所使用的数据中存在着各种各样的图形，并对推荐做出了显著的贡献。这种内在的数据特性使得在提出建议时必须考虑复杂的对象间关系。
![图1](https://upload-images.jianshu.io/upload_images/3426235-562bebcb9920994a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 图论学习具有学习复杂关系的能力。作为最有前途的机器学习技术之一，GL在获取嵌入在不同图形中的知识方面显示出了巨大的潜力。具体来说，许多GL技术，如随机漫步和图神经网络，已经被开发用于学习在图上建模的特殊类型的关系，并被证明是非常有效的。因此，利用GL对RS中的各种关系进行建模是一种自然而明智的选择。

> 图形学习有助于建立可解释的RS。如今，除了准确性之外，推荐信的可解释性也越来越受到学术界和工业界的关注。这一趋势在近年来更加明显，因为RS被深度学习技术所主导，而这些技术通常在黑箱工作机制下运行。GLRS得益于GL对关系的因果推理能力，可以很容易地根据RS中涉及的不同对象之间的推断关系来支持对推荐结果的解释，这实际上极大地促进了可解释RS的发展。

**形式化:图形学习如何帮助RS?**
> 到目前为止，还没有对所有的glr进行统一的形式化描述。事实上，对于具有特定特征的不同数据，有不同的、特定的实现应用不同的模型。因此，我们通常从高层次的角度来形式化glr。

>给定一个数据集,考虑图G = {V E}的对象,例如,用户,项目,视为节点设置V和关系,例如,购买历史,社会关系,它们之间的边缘设置E .那么,GLRS使用图G作为输入来生成相应的推荐结果R,通过造型G .正式的拓扑和内容信息,
![image.png](https://upload-images.jianshu.io/upload_images/3426235-8431ea6b3f4a0450.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 根据具体的数据和推荐场景，**图G可以是同构的，也可以是异构的，可以是静态的，也可以是动态的**，而R可以是各种形式的，例如预测评级或项目排名。**具体GLRS的优化目标也不同:根据图的拓扑结构，它们可以是最大的选择效用，也可以是节点间形成链路的最大概率**。

这项工作的主要贡献总结如下:
- 我们系统地分析了GLRS中各种图形上普遍存在的关键问题，并从数据驱动的角度对这些问题进行了分类，为深入理解GLRS的特性提供了一个新的视角。
- 我们通过从技术角度对艺术作品进行系统分类，总结了目前GLRS的研究进展。
- 我们分享并讨论了一些GLRS的开放研究方向，为社区提供参考 

## 2. 数据特征和挑战
一般来说，RS可以考虑很多不同的对象，例如用户、项目、属性、上下文等，而且几乎所有这些对象都是通过某种类型的链接相互联系的。例如用户之上的社会关系、用户与项目之间的交互。这些不同的关系本质上导致了RS中的自然图。它们之间产生了对应不同挑战的不同类型的图。在本节中，我们将从数据驱动的角度出发，系统地分析RS中的数据复杂性和特征，并相应地说明在具有不同特定特征的不同图上构建RS所面临的挑战。表1提供了一个简要的摘要。
![表1](https://upload-images.jianshu.io/upload_images/3426235-f069e7cfd71b0408.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###  2.1 树形图上的RS
通常，事务数据集中用于推荐的所有项都按照一个层次结构进行组织，这个层次结构也称为树图，它根据项的某个属性进行组织，例如类别[Huang et al.， 2019]。例如,在Amazon.com上出售的物品首先被分为不同的类别(例如,电子产品和体育产品),然后每个类别分为几个子类别(例如,无线产品在电子产品和电脑),而每个子范畴包括多个项目(如iPhone XR,华为在无线产品看GT)。这种层次图本质上揭示了物品背后的丰富关系[Gao et al.， 2019]，例如，来自不同但密切相关类别的两个物品(如iPhone XR和一个配件)很可能具有互补关系，即它们的功能。项目之间的这种层次关系可以极大地提高推荐的性能，例如避免向用户重复推荐同一子类别的类似项目，从而使推荐列表多样化。因此，如何有效地学习项目之间的这种层次关系，并将其融入到后续的推荐任务中是一个重要的挑战。

### 2.2关于单部图的RS
在RS中，至少可以定义两个同构的单部图，一个是用户图，另一个是项图。具体而言，一方面，用户之间的线上或线下社交关系构成了用户的同质社交图[Bagci and Karagoz, 2016];另一方面，同一购物篮或会话中物品的共同出现将事务数据中的所有物品连接在一起，从而形成物品的同构会话图[Xu et al.， 2019]。用户在社交图中通常会相互影响，包括他们对物品的偏好以及他们的购物行为。因此，在提出建议时，有必要考虑到社会影响。此外，这种社会影响通常会在社交图上传播，因此应该会对推荐结果产生连锁影响。

因此，如何了解用户间的社会影响及其在推荐用户图上的传播成为一个具体的挑战。

同样，物品之间的共现关系通常不仅反映了物品之间的某些潜在关系，如物品功能的互补或竞争关系，还揭示了用户的一些购物模式。
因此，在会话图上合并项之间的共现关系有助于生成更准确的建议。

这带来了另一个具体的挑战:如何在项目图中充分捕捉项目之间的关系，并适当地利用它们来提高建议的准确性。

###  2.3 二部图上的RS
连接用户和项目的交互(例如，单击、购买)是RS的核心信息，所有这些交互信息一起自然形成用户-项目二部图。
根据图上考虑的相互作用类型的数量，在二部图中建模的相互作用可以是齐次的(只有一种类型的相互作用)或异构的(多类型的相互作用)。
**一般来说，向某个用户推荐项的任务可以看作是对用户-项二部图的链接预测，**
即给定图中已知的边来预测可能的未知边[Li and Chen, 2013]。在这种情况下，一个典型的挑战是**如何学习具有同构交互的图上复杂的用户-项交互，以及这些交互之间的综合关系以获得建议**。
此外，**如何在具有异构交互的图上捕获不同类型交互之间的影响(例如，单击对购买的影响)**，以提供更丰富的信息来生成更准确的建议，这是另一个更大的挑战。

###  2.4属性图上的RS
除了上述同构用户/项图外，异构用户图、项图或用户项交互图在RS中也很常见。例如，在一个异构用户图中，至少有两种不同类型的边，分别表示不同的关系:一个表示用户之间的社会关系，另一个表示用户具有一定的属性值(如男性)，共享相同属性值的用户在图上间接连接[Yin et al.， 2010]。用户的属性值和它们建立在间接联系为朋友推荐(时et al ., 2019)和社会推荐(推荐项目,而将社会关系)具有重要意义(风扇et al ., 2019)通过提供额外的信息,以更好地捕捉个性化用户的偏好和inter-user影响力。
这实际上给如何在异构用户图上建模不同类型的关系以及它们之间的相互影响带来了挑战，然后如何将它们适当地集成到推荐任务中[Wang et al.， 2019a, Wang et al.， 2019b]。
> Sequential recommender systems: challenges, progress and prospects
> Modeling multi-purpose sessions for next-item recommendations via mixture-channel purpose routing networks.

同样，项与项属性值之间的共现关系形成异类项图。这两种关系对于理解项目的分布、出现模式和本质都很重要，从而有利于推荐。因此，**如何在异构项图上有效地对异构关系进行建模以提高推荐性能成为该分支面临的另一个挑战。**


### 2.5 RS on Complex Heterogeneous Graphs
为了解决用户-项目交互数据中的稀疏性问题，以便更好地理解用户偏好和项目特征，辅助信息(如社会关系或项目特征)常常与用户-项目交互信息相结合，以获得更好的推荐。

一方面，为了考虑项目的用户间影响偏好，用户之间的社会关系通常与用户-项目交互相结合，构建所谓的社交RS [Guy, 2015];另一方面，为了更深入地描述项目，项目特征通常与用户-项目交互结合在一起，提供关于冷启动项目的建议[Palumbo et al.， 2017;Han et al.， 2018]。
> [Palumbo et al., 2017] Entity2rec: Learning user-item relatedness from¨ knowledge graphs for top-n item recommendation. 
> [Hu et al., 2018]  Leveraging meta-path based context for top-n recommendation with a neural co-attention model.

这两类异质推荐信息的组合形成了两种异质图:**一种是基于用户-条目交互的二部图，另一种是用户之间的社交图或条目-特征图。**两个图中的共享用户或项充当连接它们的桥梁。社会关系或项目特性对于深入了解用户非常重要，因为它们考虑了用户的偏好传播，或者通过考虑项目的自然属性来更好地描述项目。然而，如何使来自两个图的异构信息能够适当地相互通信，并能够天生地组合在一起以使推荐任务受益，这是一个相当具有挑战性的问题。

### 2.6 RS对多源异构图的影响
有效解决无处不在的数据稀疏和冷起动问题，构建更健壮和可靠的RS,除了user-item交互,很多相关的信息有很大的显式或隐式的影响深入的推荐可以有效地利用和集成到RS [Cen et al., 2019]. 
>   Representation learning for  attributed multiplex heterogeneous network. 
例如,用户配置文件的用户信息表,在线或离线社交网络的社会关系,项目特点的项目信息表、项目同现关系事务表等.

同时可以利用,以帮助更好的理解用户的偏好和项目特点改善建议。因此，我们共同构建了多个异构的推荐图:基于用户-项目交互的二分图提供了用户选择建模的关键信息，**基于用户属性的图和社交图提供了用户的辅助信息**，**基于项目属性的图和基于项目共现的图提供了项目的辅助信息**。一般来说，我们对用户的喜好和购物行为了解得越多，从所有可用信息中了解商品的特点和受欢迎程度，我们就能做出更好的推荐。

然而，由于这种异构性，不同图上的信息是相对不相关的，不能立即使用，**因此如何利用不同的图来相互补充和推荐是第一个挑战**.
而且，**异构图越多，表示不同图之间可能存在噪声甚至矛盾的风险就越高。因此，如何从多源异构图中提取相关信息，减少噪声和不相关信息，提高下游推荐是另一大挑战**

###  3 Graph Learning Approaches to RS
在本节中，我们将首先从技术角度对构建GLRS的这些挑战的解决方案进行分类，然后讨论在每个类别中取得的进展。

GLRS方法的分类如图2所示。GLRS首先被分为四类，一些类别(如Graph neural network approach)被进一步分为多个子类别。一般来说，这些类别会从简单变为复杂，并依次进行报告。接下来，我们总结了这四类研究的进展。

![image.png](https://upload-images.jianshu.io/upload_images/3426235-d17b8fc837bd9e63.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###  3.1 Random Walk Approach
基于随机游走的RS在过去的15年中得到了广泛的研究，并被广泛应用于各种图(例如用户之间的社交图、项目之间的协作图)中，以捕获节点之间的复杂关系，从而提供建议。通常，基于随机游走的RS首先让随机游走者在以用户和物品为节点的构造图上行走，并具有针对每个步骤的预定义转移概率，以对用户和物品之间的隐式偏好或交互传播建模，然后采用 经过一定步骤对这些候选节点进行排名，随机沃克在节点上着陆。基于随机游走的RS得益于其特殊的工作机制，能够很好地捕捉图上各种节点(例如用户和项目)之间复杂的、高阶的和间接的关系，因此能够解决同构或异构图中生成建议的重要挑战。

基于随机游走的RS有不同的变体。
除了基本的基于随机游走的RS（如Baluja等人，2008）之外，基于随机游走并基于重启的RS（Bagci和Karagoz，2016； Jiang et al。，2018]是基于随机游走的RS的另一种代表性类型。

为了提供更多个性化的建议，一些基于随机游走的RS [Eksombatchai等人，2018]计算了每个步骤的用户特定过渡概率。
>  [Eksombatchai et al., 2018] Chantat Eksombatchai, Pranav Jindal,
Jerry Zitao Liu, and et al. Pixie: A system for recommending
3+ billion items to 200+ million users in real-time.
基于随机游走的RS的其他典型应用包括排名项目w.r.t. 它们在项目-项目共视图图上的重要性[Gori et al。，2007]，同时在用户项目二分图上对用户-项目交互进行建模，同时使用项目-项目接近关系来指导用户推荐前n个项目 过渡[Nikolakopoulos和Karypis，2019年]。
>  [Nikolakopoulos and Karypis, 2019] Athanasios N Nikolakopoulos and George Karypis. Recwalk: Nearly uncoupled random walks for top-n recommendation. In
WSDM

尽管基于随机游走的RS已得到广泛应用，但其缺点也很明显：（1）它们需要在每个步骤的每个用户的所有候选项目上生成排名得分，因此由于以下原因，它们难以应用于大型图形： 低效率（2）与大多数基于学习的范式不同，基于随机游走的RS是基于启发式的，缺乏用于优化推荐目标的模型参数，这大大降低了推荐性能。

###  3.2 Graph Representation Learning Approach
图表示学习是分析嵌入在图上的复杂关系的另一种有效方法，近年来发展迅速。它将每个节点映射到一个潜在的低维表示，以便将图形结构信息编码到其中。

研究人员将图表示学习引入到RS中，为后续的推荐建立各个节点(如用户、项目)之间的复杂关系模型，从而构建基于图表示学习的RS (GRLRS)。

根据表示学习的具体方法，GRLRS一般可分为三类:
(1)基于图分解机的RS (GFMRS)，
(2)基于图分布式表示的RS (GDRRS)，
(3)基于图神经嵌入的RS (GNERS)。 

###### 基于图分解机的RS（GFMRS）。
GFMRS首先使用分解机（例如矩阵分解），基于图上的元路径分解节点间通勤矩阵，以获得每个节点（例如，用户或物品）的潜在表示，将其用作 后续推荐任务的输入[Wang等，2019d]。 这样，将嵌入图形中的节点之间的复杂关系编码为潜在表示，以使建议受益。 由于具有处理节点异质性的能力，GFMRS已被广泛应用于捕获不同类型的节点之间的关系，例如，用户和物品。
>  Unified embedding model over heterogeneous information network for personalized recommendation. In IJCAI, 
这类模型虽然简单有效，但由于观测数据的稀疏性，很容易受到影响，难以实现理想的推荐。

##### 基于图形分布式表示的RS (GDRRS)。
不同于GFMRS, GDRRS通常遵循Skip-gram模型(Mikolov et al ., 2013)学习分布式表示图中的每个用户或项目编码的自我信息用户或项目及其相邻关系到一个低维向量(Shi et al ., 2018),在准备后续的推荐。
>  Heterogeneous information network embedding for  recommendation

具体来说，GDRRS通常首先使用随机游走来生成在一个元路径中同时出现的节点序列，然后使用skipgram或类似的模型来生成推荐的节点表示。GDRRS利用其强大的在图上编码节点间连接的能力，被广泛应用于同构或异构图上，以捕获RS中不同对象之间的关系[Cen et al.， 2019]。
>   Representation learning for  attributed multiplex heterogeneous network. In SIGKDD

由于GDRRS没有深度或复杂的网络结构，近年来由于其简单、高效和功效而显示出巨大的潜力[Wang et al.， 2020a, Wang et al.， 2020b]。
>   Intention Nets: Psychology-inspired User Choice Behavior Modeling for Next-basket Prediction In AAAI, 
>  Intention2Basket: A Neural Intention-driven Approach for Dynamic Next-basket Planning. In IJCAI

基于图神经嵌入的RS（GNERS）。 GNERS通常利用神经网络，例如多层感知器
（MLP），以了解用户的嵌入或图中的项目，然后将学习到的嵌入用于建议。 神经嵌入模型很容易与其他下游神经元整合在一起推荐模型（例如，基于RNN的模型）来构建可以共同训练两个模型的端到端RS一起进行更好的优化[Han et al。，2018]。 对此最后，GNERS已广泛应用于各种像属性图一样的图[Han et al。，2018]，异构图[Hu et al。，2018]，多源异构图[Cen et al。，2019]等来解析
提出建议的各种挑战。
>  [Han et al., 2018]  Aspect-level deep collaborative filtering via heterogeneous information networks. In IJCAI,
> [Hu et al。，2018] Leveraging meta-path based context for top-n recommendation with a neural co-attention model. In SIGKDD
>  Modeling influential contexts with heterogeneous relations for sparse and cold-start  recommendation. In AAAI,

###  3.3 Graph Neural Network Approach
近年来，将神经网络应用于图数据的图神经网络（GNN）迅速发展，并显示出解决该问题的巨大潜力。 各种图表上的各种挑战。 得益于GNN的优势，通过引入GNN提出了许多基于GNN的RS，以应对网络中的各种挑战。 

具体来说，基于GNN的RS可以根据具体使用的GNN模型主要分为三类:
(1)基于图注意网络的RS (GATRS)，
(2)基于门控图神经网络的RS (GGNNRS)，
(3)基于图卷积网络的RS (GCNRS)。

##### 基于图形注意网络的RS (GATRS)。
图注意力网络(GAT)将注意机制引入到GNN中，有区别地学习其他用户或项目的不同相关性和影响程度，即目标用户或项目对用户或项目图的影响程度。具体来说，学习注意力权重是为了专心地将来自邻居的信息集成到目标用户或项目的表示中。GATRS建立在GAT之上，以便更精确地了解后续建议的用户间或项目关系。
在这种情况下，来自那些更重要的用户或物品的影响 强调特定的用户或项目，这更符合实际情况，因此改进建议的好处。

由于GAT具有良好的识别能力，它被广泛应用于不同类型的图中，包括社交图[Fan et al.， 2019]、项目会话图[Xu et al.， 2019]、知识图[Wang et al.， 2019c]，以构建各种推荐性能良好的GATRS 
>   Graph neural networks for social recommendation. In WWW
>    Graph  contextualized self-attention network for session-based recommendation. In IJCAI
>   A  Survey on Session-based Recommender Systems.

#####  基于门控图神经网络的RS (GGNNRS)。
门控图神经网络(GGNN)将门控回归单元(GRU)引入到GNN中，通过迭代吸收图中其他节点的影响来学习优化的节点表示，从而全面捕捉节点间的关系。在GGNN上构建GGNNRS，综合考虑嵌入在相应用户或项图上的复杂用户间或项间关系，学习用户或项嵌入的推荐信息。

GGNNRS建立在GGNN之上，通过全面考虑嵌入在相应用户或项目图上的复杂的用户间或项目间关系来学习用户或项目嵌入以获取建议。 由于捕获图上复杂关系的力量，GGNN被广泛用于为基于会话的推荐建模会话图中项目之间的复杂转换[Wu等人，2019b]，或为不同模型之间的复杂交互建模 用于时尚推荐的时尚产品类别[Cui等，2019，Wang等，2019c]，并取得了出色的推荐性能。
>  Session-based recommendation with graph neural networks.  In AAAI,
>  Dressing as a whole: Outfit compatibility learning based on node-wise graph neural networks. In WWW

###### 基于RS (GCNRS)的图卷积网络。
图卷积网络(GCN)通常学习如何利用图结构和节点特征信息，利用神经网络从局部图邻域迭代地聚合特征信息。一般来说，通过使用卷积和池操作，GCN能够通过有效地将用户和项的邻域信息以图的形式聚合，从而学习用户和项的信息嵌入。

GCNRS是建立在GCN之上的，它可以学习用户或项目在图中的嵌入，同时全面利用用户或/和项目之间的复杂关系，以及他们自己的信息，以提供建议[Ying et al.， 2018]。
>  Graph convolutional neural networks for web-scale recommender
systems. In SIGKDD

得益于强大的特征提取和学习能力，特别是结合图结构和节点内容信息的力量，GCN被广泛应用于RS中的各种图以构建GCNRS，并被证明具有很大的发展前景。 例如，GCN用于在社交推荐中对社交图进行影响扩散[Wu等人，2019a]，
>  A neural  influence diffusion model for social recommendation. In SIGIR,

在用户-项目交互图上挖掘隐藏的用户-项目连接信息，以减轻协作过滤中的数据稀疏性问题[Wang等。 等人，2019a]，
>  Binarized collaborative filtering with distilling graph  convolutional networks. In IJCAI,

并通过在基于项目属性的知识图上挖掘其相关属性来捕获项目间的相关性[Wang等人，2019b]。
>  Knowledge-aware graph neural networks with label smoothness regularization for recommender systems. In SIGKDD

###  3.4 Knowledge-Graph Approach
基于知识图的RS (knowledge -graph based RS, KGRS)通常基于外部知识(如边信息)构建知识图KG (knowledge -graph, KG)，探索用户或项目之间的隐式或高阶连通关系，丰富用户或项目的表示，从而提高推荐性能。更重要的是，由于使用了额外的知识，KGRS能够更好地理解用户行为和项目特征，这导致了更可解释的建议[Wang et al.， 2018a]。

KGRS主要关注于RS早期阶段KG的构建，而现有的各种技术，包括因式分解机、图神经网络等，用于从构建的KG中提取信息，并将其集成到后续的建议中。
根据构建KG所使用的知识，一般可以将KGRS分为三个类，下面将依次介绍。

基于本体的KGRS（OKGRS）。 OKGRS基于用户或项的本体构建分层的KG，以在树状图中表示分层的所属关系。 分层KG的一个典型示例是Amazon.com中使用的树形图，其中产品类别用于组织平台上所有待售商品。 在此图中，根节点表示最粗糙的类别（例如食物），而叶节点表示特定的项（例如面包）。

最近几年，OKGRS已被广泛研究以增强建议的可解释性，例如，使用它来提取
项目本体图的多层次用户兴趣 [Gao等，2019，Wang等，2017]。
> Explainable recommendation through attentive multiview learning. In AAAI


