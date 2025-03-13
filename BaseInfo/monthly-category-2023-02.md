# 2023-02 月度论文分类汇总

共有2篇相关领域论文, 另有0篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 1 篇论文

### Grounding Language Models to Images for Multimodal Inputs and Outputs 
[[arxiv](https://arxiv.org/abs/2301.13823)] [[cool](https://papers.cool/arxiv/2301.13823)] [[pdf](https://arxiv.org/pdf/2301.13823)]
> **Authors**: Jing Yu Koh,Ruslan Salakhutdinov,Daniel Fried
> **First submission**: 2023-01-31
> **First announcement**: 2023-02-01
> **comment**: Published in ICML 2023. Project page: https://jykoh.com/fromage
- **标题**: 将语言模型接地到多模式输入和输出的图像
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 我们提出了一种有效的方法，将仅预读文本语言模型接地到视觉域，使它们能够处理任意交织的图像和文本数据，并生成与检索到的图像交织的文本。我们的方法利用了从大规模文本预处理中学到的语言模型的能力，例如内在的学习和自由形式的文本生成。我们将语言模型冻结，并进行芬太纳输入和输出线性层以实现交叉模式相互作用。这使我们的模型可以处理任意交织的图像和文本输入，并生成与检索的图像交织在一起的自由形式文本。我们在诸如上下文图像检索和多模式对话等诸如上下文的任务上实现了强劲的零射击性能，并展示了令人信服的互动能力。我们的方法与任何现成的语言模型一起使用，并为在视觉接地的设置中利用预审计的语言模型的有效，通用解决方案铺平了道路。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 1 篇论文

### UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers 
[[arxiv](https://arxiv.org/abs/2301.13741)] [[cool](https://papers.cool/arxiv/2301.13741)] [[pdf](https://arxiv.org/pdf/2301.13741)]
> **Authors**: Dachuan Shi,Chaofan Tao,Ying Jin,Zhendong Yang,Chun Yuan,Jiaqi Wang
> **First submission**: 2023-01-31
> **First announcement**: 2023-02-01
> **comment**: ICML 2023. Website: https://dachuanshi.com/UPop-Project
- **标题**: UPOP：统一和渐进的修剪，以压缩视力语言变压器
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 现实世界中的数据包含大量的多模式信息，其中愿景和语言是两个最具代表性的方式。此外，越来越重的模型，\ textit {e}。\ textit {g}。变形金刚吸引了研究人员对压缩建模的关注。但是，如何压缩多模型模型，尤其是Vison语言变压器，仍未探索。本文提出了\ textbf {u} nified和\ textbf {p} r \ textbf {o} gressive \ textbf {p} runing（\ \ emph {\ emph {upOp}}模型，可以在可压缩的方式和结构之间自动分配修剪比； 2）逐步搜索和重新培训子网，该子网保持搜索和重新培训之间的融合以达到更高的压缩比。对各种任务，数据集和模型体系结构进行的实验证明了拟议的UPOP框架的有效性和多功能性。该代码可在https://github.com/sdc17/upop上找到。

