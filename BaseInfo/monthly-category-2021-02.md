# 2021-02 月度论文分类汇总

共有1篇相关领域论文, 另有0篇其他

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 1 篇论文

### Self-Supervised Pretraining for RGB-D Salient Object Detection 
[[arxiv](https://arxiv.org/abs/2101.12482)] [[cool](https://papers.cool/arxiv/2101.12482)] [[pdf](https://arxiv.org/pdf/2101.12482)]
> **Authors**: Xiaoqi Zhao,Youwei Pang,Lihe Zhang,Huchuan Lu,Xiang Ruan
> **First submission**: 2021-01-29
> **First announcement**: 2021-02-01
> **comment**: This work was accepted by AAAI 2022
- **标题**: RGB-D显着对象检测预处理预处理
- **领域**: 计算机视觉和模式识别
- **摘要**: 需要在ImageNet上预定基于CNNS的RGB-D显着对象检测（SOD）网络，以了解有助于提供良好初始化的层次结构功能。但是，大规模数据集的收集和注释是耗时且昂贵的。在本文中，我们利用自我监督的表示学习（SSL）来设计两个借口任务：跨模式自动编码器和深度估计。我们的借口任务仅需要几个未标记的RGB-D数据集来执行训练，这使网络捕获了丰富的语义上下文并减少了两种模式之间的差距，从而为下游任务提供了有效的初始化。此外，对于RGB-D SOD中跨模式融合的固有问题，我们提出了一个一致性差异聚集（CDA）模块，该模块将单个特征融合分为多路径融合，以实现对一致和差异信息的充分感知。 CDA模块是通用的，适用于跨模式和跨级特征融合。在六个基准数据集上进行的广泛实验表明，我们的自我监管预处理的模型对大多数在ImageNet上预告片的最新方法都表现出色。源代码将在\ textColor {red} {\ url {https://github.com/xiaoqi-zhao-dlut/sslsod}}上公开获得。

