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

 ，


