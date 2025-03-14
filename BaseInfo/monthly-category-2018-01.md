# 2018-01 月度论文分类汇总

共有20篇相关领域论文, 另有11篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 3 篇论文

### The CAPIO 2017 Conversational Speech Recognition System 
[[arxiv](https://arxiv.org/abs/1801.00059)] [[cool](https://papers.cool/arxiv/1801.00059)] [[pdf](https://arxiv.org/pdf/1801.00059)]
> **Authors**: Kyu J. Han,Akshay Chandrashekaran,Jungsuk Kim,Ian Lane
> **First submission**: 2017-12-29
> **First announcement**: 2018-01-02
> **comment**: 8 page, 3 figures, 8 tables; extra experimental results added
- **标题**: Capio 2017会话识别系统
- **领域**: 计算语言学
- **摘要**: 在本文中，我们展示了如何在行业标准NIST 2000 HUB5英语评估集上实现最先进的表现。我们探索密集连接的LSTM，灵感来自最近引入的图像分类任务的密集连接的卷积网络。我们还提出了一种声学模型适应方案，该方案只需平均种子神经网络声学模型及其适应性版本的参数。该方法用于Callhome培训语料库，并将单个系统性能提高6.1％（相对），而评估集的Callhome部分则没有且无绩效损失。通过在三个不同的手机组中训练的5个系统上的RNN-LM逆转和晶格组合，我们的2017年语音识别系统分别获得了总机和Callhome的5.0％和9.1％，这两者都是迄今为止报告的最佳单词错误率。根据IBM在比较人类和机器转录的最新工作中，我们报告的总机单词错误率可以被认为超过了人类的均等（5.1％）转录对话性电话演讲的均等（5.1％）。

### Learning Multimodal Word Representation via Dynamic Fusion Methods 
[[arxiv](https://arxiv.org/abs/1801.00532)] [[cool](https://papers.cool/arxiv/1801.00532)] [[pdf](https://arxiv.org/pdf/1801.00532)]
> **Authors**: Shaonan Wang,Jiajun Zhang,Chengqing Zong
> **First submission**: 2018-01-01
> **First announcement**: 2018-01-02
> **comment**: To be appear in AAAI-18
- **标题**: 通过动态融合方法学习多模式单词表示
- **领域**: 计算语言学
- **摘要**: 事实证明，多模型模型在学习语义单词表示方面的表现要优于基于文本的模型。几乎所有以前的多模式模型通常都平等地对待不同方式的表示。但是，很明显，来自不同方式的信息对单词的含义有所不同。这促使我们构建一个多模型，该模型可以根据不同类型的单词动态地融合不同方式的语义表示。为此，我们提出了三种新型的动态融合方法，以将重要的权重分配给每种方式，在这种方式下，在单词关联对的弱监督下学习了权重。广泛的实验表明，所提出的方法的表现优于强大的单峰基线和最先进的多模型模型。

### Exploring Architectures, Data and Units For Streaming End-to-End Speech Recognition with RNN-Transducer 
[[arxiv](https://arxiv.org/abs/1801.00841)] [[cool](https://papers.cool/arxiv/1801.00841)] [[pdf](https://arxiv.org/pdf/1801.00841)]
> **Authors**: Kanishka Rao,Haşim Sak,Rohit Prabhavalkar
> **First submission**: 2018-01-02
> **First announcement**: 2018-01-03
> **comment**: In Proceedings of IEEE ASRU 2017
- **标题**: 通过RNN-TransDucer探索用于流端到端语音识别的架构，数据和单位
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 我们研究了使用复发性神经网络传感器（RNN-T）的端到端语音识别模型：一种流媒体，全中性，序列到序列体系结构，共同从转录的声学数据中学习了声学和语言模型组件。我们探索各种模型架构，并说明如果有其他文本或发音数据，如何进一步改进模型。该模型由“编码器”组成，该模型是从基于连接的时间分类（CTC）声学模型和“解码器”初始初始化的，该模型是从单独培训文本数据训练的复发性神经网络语言模型中部分初始初始初始初始初始初始初始初始初始初始初始初始初始初始的。整个神经网络都经过RNN-T损失的训练，并直接输出公认的成绩单作为一系列素描，从而执行端到端语音识别。我们发现，通过使用子字单元（``单词''）可以进一步提高性能，从而捕获更长的上下文并显着减少替代错误。最佳的RNN-T系统是一个十二层LSTM编码器，其两层LSTM解码器接受了30,000个文字作为输出目标训练的训练，可以在语音搜索上达到8.5％的单词错误率，而语音范围为5.2 \％，在语音任务上可与正面的基准计算，在8.3 \％的语音搜索中相当，并且是语音search和5.4 n.4 forseard和5.4 ossearch and 5.4。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 11 篇论文

### On-the-fly Augmented Reality for Orthopaedic Surgery Using a Multi-Modal Fiducial 
[[arxiv](https://arxiv.org/abs/1801.01560)] [[cool](https://papers.cool/arxiv/1801.01560)] [[pdf](https://arxiv.org/pdf/1801.01560)]
> **Authors**: Sebastian Andress,Alex Johnson,Mathias Unberath,Alexander Winkler,Kevin Yu,Javad Fotouhi,Simon Weidert,Greg Osgood,Nassir Navab
> **First submission**: 2018-01-04
> **First announcement**: 2018-01-05
> **comment**: S. Andress, A. Johnson, M. Unberath, and A. Winkler have contributed equally and are listed in alphabetical order
- **标题**: 使用多模式基金会的骨科手术的增强现实
- **领域**: 计算机视觉和模式识别
- **摘要**: 透视X射线指导是经皮骨科手术程序的基石。然而，三维解剖结构的二维观察遭受了投影简化的影响。因此，需要获得许多来自各种取向的X射线图像，以便外科医生准确评估患者的解剖结构与手术工具之间的空间关系。在本文中，我们提出了一种直立的外科手术支持系统，该系统使用增强现实提供指导，可用于准准备的手术室。所提出的系统建立在多模式标记和同时定位和映射技术的基础上，以将光学透明的头部安装显示到C型臂荧光镜检查系统。然后，在2D X射线图像上的注释可以作为3D提供手术指导的虚拟对象渲染。我们定量评估了所提出的系统的组件，最后设计了一项关于半伪像幻影的可行性研究。我们系统的准确性与传统的图像指导技术相媲美，同时大大减少了所获得的X射线图像的数量以及程序时间。我们有希望的结果鼓励对虚拟对象和真实对象之间的相互作用进行进一步研究，我们认为这将直接受益于所提出的方法。此外，我们希望在针对常见骨科干预措施的大型研究中探索我们即将增强现实支持系统的能力。

### Bridging the Gap: Simultaneous Fine Tuning for Data Re-Balancing 
[[arxiv](https://arxiv.org/abs/1801.02548)] [[cool](https://papers.cool/arxiv/1801.02548)] [[pdf](https://arxiv.org/pdf/1801.02548)]
> **Authors**: John McKay,Isaac Gerg,Vishal Monga
> **First submission**: 2018-01-08
> **First announcement**: 2018-01-09
> **comment**: Submitted to IGARSS 2018, 4 Pages, 8 Figures
- **标题**: 弥合差距：同时进行数据重新平衡的微调
- **领域**: 计算机视觉和模式识别
- **摘要**: 在许多现实世界中的分类问题中，数据不平衡问题（数据集包含一个/多个类别的样本比其他类别的样本大得多）是不可避免的。尽管有问题的类是一个常见的解决方案，但当大数据类本身是多种多样和/或有限的数据类特别小时，这并不是一个引人注目的选择。我们提出了一种基于有关有限数据问题的最新工作的策略，该策略利用了一组与有限数据类具有相似属性的图像集，以帮助培训神经网络。我们在现实世界合成孔径声纳数据集上针对其他典型方法显示了模型的结果。代码可以在github.com/johnmckay/dataimbalance上找到。

### DeepStyle: Multimodal Search Engine for Fashion and Interior Design 
[[arxiv](https://arxiv.org/abs/1801.03002)] [[cool](https://papers.cool/arxiv/1801.03002)] [[pdf](https://arxiv.org/pdf/1801.03002)]
> **Authors**: Ivona Tautkute,Tomasz Trzcinski,Aleksander Skorupa,Lukasz Brocki,Krzysztof Marasek
> **First submission**: 2018-01-08
> **First announcement**: 2018-01-09
> **comment**: Copyright held by IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
- **标题**: DeepStyle：时尚和室内设计的多模式搜索引擎
- **领域**: 计算机视觉和模式识别
- **摘要**: 在本文中，我们提出了一个多模式搜索引擎，该引擎结合了视觉和文本提示，以从与查询上相似的多媒体数据库中检索项目。我们的发动机的目的是实现时尚商品（例如衣服或家具）的直观检索。现有的搜索引擎仅将文本输入视为有关查询映像的附加信息来源，并且与用户寻找“同一衬衫但牛仔布”的现实情况不符。我们的新方法被称为“ Deepstyle”，通过使用关节神经网络结构来模拟不同模态特征之间的上下文依赖性，从而减轻了这些缺陷。我们证明了这种方法在两个不同挑战性的时尚物品和家具数据集上的鲁棒性，在该数据集中，我们的DeepStyle Engine在经过测试的数据集上优于基线方法18-21％。我们的搜索引擎是商业部署的，并通过基于Web的应用程序获得。

### Enhancing Underwater Imagery using Generative Adversarial Networks 
[[arxiv](https://arxiv.org/abs/1801.04011)] [[cool](https://papers.cool/arxiv/1801.04011)] [[pdf](https://arxiv.org/pdf/1801.04011)]
> **Authors**: Cameron Fabbri,Md Jahidul Islam,Junaed Sattar
> **First submission**: 2018-01-11
> **First announcement**: 2018-01-12
> **comment**: Submitted to ICRA 2018
- **标题**: 使用生成对抗网络增强水下图像
- **领域**: 计算机视觉和模式识别,机器人技术
- **摘要**: 自动的水下车辆（AUV）依赖于各种传感器 - 声学，惯性和视觉 - 用于智能决策。由于其非侵入性，被动的性质和高信息内容，视觉是一种有吸引力的感应方式，尤其是在较浅的深度下。但是，诸如光折射和吸收，水中悬浮颗粒以及颜色失真等因素会影响视觉数据的质量，从而导致嘈杂和扭曲的图像。因此，依靠视觉传感的AUV因此面临困难的挑战，因此在视力驱动的任务上表现出差的性能。本文提出了一种使用生成对抗网络（GAN）提高视觉水下场景质量的方法，目的是在自主管道下进一步改善视觉驱动行为的输入。此外，我们还展示了最近提出的方法能够生成数据集，以实现此类水下图像恢复的目的。对于任何视觉引导的水下机器人，这种改进可以通过强大的视觉感知提高安全性和可靠性。为此，我们提供了定量和定性数据，这些数据表明，通过所提出的方法纠正的图像会产生更具吸引力的图像，并且还为潜水员跟踪算法提供了提高的准确性。

### Deep Multi-Spectral Registration Using Invariant Descriptor Learning 
[[arxiv](https://arxiv.org/abs/1801.05171)] [[cool](https://papers.cool/arxiv/1801.05171)] [[pdf](https://arxiv.org/pdf/1801.05171)]
> **Authors**: Nati Ofir,Shai Silberstein,Hila Levi,Dani Rozenbaum,Yosi Keller,Sharon Duvdevani Bar
> **First submission**: 2018-01-16
> **First announcement**: 2018-01-17
> **comment**: No comments
- **标题**: 使用不变的描述符学习深度多光谱注册
- **领域**: 计算机视觉和模式识别
- **摘要**: 在本文中，我们引入了一种新颖的深度学习方法来对齐跨光谱图像。我们的方法依赖于学习的描述符，该描述符对不同的光谱不变。同一场景的多模式图像捕获了不同的信号，因此它们的注册具有挑战性，并且没有通过经典方法来解决。为此，我们开发了一种基于功能的方法，该方法解决了可见的（VIS）到近fra-Red（NIR）注册问题。我们的算法通过Harris检测到角落，并通过在CIFAR-10网络描述符顶部学习的贴片 - 图表进行匹配。当我们的实验证明时，我们以亚像素精度达到了跨光谱图像的高质量比对。与其他现有方法相比，我们的方法在VIS NIR注册的任务中更为准确。

### End-to-end Multi-Modal Multi-Task Vehicle Control for Self-Driving Cars with Visual Perception 
[[arxiv](https://arxiv.org/abs/1801.06734)] [[cool](https://papers.cool/arxiv/1801.06734)] [[pdf](https://arxiv.org/pdf/1801.06734)]
> **Authors**: Zhengyuan Yang,Yixuan Zhang,Jerry Yu,Junjie Cai,Jiebo Luo
> **First submission**: 2018-01-20
> **First announcement**: 2018-01-22
> **comment**: 6 pages, 5 figures
- **标题**: 具有视觉感知的自动驾驶汽车的端到端多模式多任务车辆控制
- **领域**: 计算机视觉和模式识别
- **摘要**: 卷积神经网络（CNN）已成功地应用于自动驾驶任务，许多端到端方式。以前的端到端转向控制方法将图像或图像序列作为输入，并直接用CNN预测转向角度。尽管在转向角上的单个任务学习报告了良好的表现，但仅转向角度不足以控制车辆。在这项工作中，我们提出了一个多任务学习框架，以端到端的方式同时预测转向角度和速度控制。由于仅使用视觉输入预测准确的速度值是不平凡的，因此我们首先提出一个网络来预测具有图像序列的离散速度命令和转向角。此外，我们提出了一个多模式的多任务网络，以通过将先前的反馈速度和视觉记录作为输入来预测速度值和转向角度。实验是对公共Udacity数据集和新收集的SAIC数据集进行的。结果表明，所提出的模型可以准确预测转向角度和速度值。此外，我们改善了在实际路测试中解决错误积累问题的故障数据合成方法。

### Food recognition and recipe analysis: integrating visual content, context and external knowledge 
[[arxiv](https://arxiv.org/abs/1801.07239)] [[cool](https://papers.cool/arxiv/1801.07239)] [[pdf](https://arxiv.org/pdf/1801.07239)]
> **Authors**: Luis Herranz,Weiqing Min,Shuqiang Jiang
> **First submission**: 2018-01-22
> **First announcement**: 2018-01-23
> **comment**: Survey about contextual food recognition andmultimodalrecipe analysis
- **标题**: 食品识别和食谱分析：整合视觉内容，上下文和外部知识
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 食物在我们的个人和社会生活中的核心作用，再加上最近的技术进步，激发了人们对应用的日益兴趣，这些应用有助于更好地监测饮食习惯，以及与食品相关信息的探索和检索。我们回顾了如何将视觉内容，上下文和外部知识有效整合到面向食品的应用中，特别关注食谱分析和检索，食品推荐以及餐厅环境作为新兴方向。

### Survey on Emotional Body Gesture Recognition 


[[arxiv](https://arxiv.org/abs/1801.07481)] [[cool](https://papers.cool/arxiv/1801.07481)] [[pdf](https://arxiv.org/pdf/1801.07481)]
> **Authors**: Fatemeh Noroozi,Ciprian Adrian Corneanu,Dorota Kamińska,Tomasz Sapiński,Sergio Escalera,Gholamreza Anbarjafari
> **First submission**: 2018-01-23
> **First announcement**: 2018-01-24
> **comment**: No comments
- **标题**: 情感身体手势识别的调查
- **领域**: 计算机视觉和模式识别
- **摘要**: 在过去的十年中，自动情绪识别已成为一个热门的研究主题。尽管基于面部表情或语音的作品比比皆是，但认识到身体手势的影响仍然是一个不太探索的话题。我们提出了一项新的综合调查，希望促进该领域的研究。我们首先将情感身体手势作为通常称为“肢体语言”的组成部分，并将一般方面评为性别差异和文化依赖。然后，我们为自动情感身体手势识别定义了一个完整的框架。我们介绍了RGB和3D中的人检测和评论静态和动态的身体姿势估计方法。然后，我们评论了与表示情感表达性手势图像的表示和情感识别有关的最新文献。我们还讨论了将语音或面对面与身体手势相结合的多模式方法，以改善情绪识别。如今，预处理方法学（例如人类的检测和姿势估计）是完全开发用于强大的大规模分析的成熟技术，但我们表明，对于情感识别，标记的数据的数量是稀缺的，但在明确定义的输出空间上尚无共识，并且代表性很小，并且基于天真的差异代表。

### Convolutional Invasion and Expansion Networks for Tumor Growth Prediction 
[[arxiv](https://arxiv.org/abs/1801.08468)] [[cool](https://papers.cool/arxiv/1801.08468)] [[pdf](https://arxiv.org/pdf/1801.08468)]
> **Authors**: Ling Zhang,Le Lu,Ronald M. Summers,Electron Kebebew,Jianhua Yao
> **First submission**: 2018-01-25
> **First announcement**: 2018-01-26
> **comment**: ef:IEEE Transactions on Medical Imaging, 15 November 2017, Volume:PP Issue: 99
- **标题**: 用于肿瘤生长预测的卷积入侵和扩张网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 肿瘤的生长与细胞侵袭和质量效应有关，这些细胞侵袭和质量效应是通过数学模型（即反应扩散方程式和生物力学）提出的。可以基于临床测量值来个性化此类模型，以构建肿瘤生长的预测模型。在本文中，我们研究了使用深卷积神经网络（Convnets）直接表示和学习细胞侵袭和质量效应，并预测肿瘤随后的参与区域的可能性。入侵网络从与来自多模式成像数据的代谢率，细胞密度和肿瘤边界有关的信息中学习细胞侵袭。扩展网络对肿瘤质量运动不断增长的质量效应建模。我们还研究了融合入侵和扩展网络的不同体系结构，以利用它们之间的固有相关性。与大多数未能合并人口数据的数学建模方法不同，我们的网络可以轻松地接受人群数据并为目标患者进行个性化培训。对胰腺肿瘤数据集的定量实验表明，所提出的方法在准确性和效率方面基本上优于基于数学模型的最新模型方法，并且两个子网络中的每一个捕获的信息都是互补的。

### A Multi-Biometrics for Twins Identification Based Speech and Ear 
[[arxiv](https://arxiv.org/abs/1801.09056)] [[cool](https://papers.cool/arxiv/1801.09056)] [[pdf](https://arxiv.org/pdf/1801.09056)]
> **Authors**: Cihan Akin,Umit Kacar,Murvet Kirci
> **First submission**: 2018-01-27
> **First announcement**: 2018-01-29
> **comment**: No comments
- **标题**: 基于双胞胎识别语音和耳朵的多生物测量学
- **领域**: 计算机视觉和模式识别
- **摘要**: 技术生物识别技术的开发变得更加重要。用于定义人类特征生物识别系统，但由于传统的生物识别系统无法识别双胞胎，因此开发了多模式生物识别系统。在这项研究中，提出了多模式的生物识别识别系统，以使用图像和语音数据来识别彼此和其他人的双胞胎。语音或图像数据足以互相认识到彼此，但是双胞胎不能用其中一个数据来区分。因此，需要使用语音和耳朵图像的结合来实现强大的识别系统。作为数据库，使用了39个双胞胎的照片和语音数据。为了进行语音识别，使用MFCC和DTW算法。另外，Gabor滤波器和DCVA算法用于耳朵识别。通过使匹配得分水平融合来提高多生物计量学的成功率。特别是，排名5的达到100％。我们认为语音和耳朵可以是互补的。因此，结果是基于多生物计量学的语音和耳朵对人类的识别有效。

### Object-based reasoning in VQA 
[[arxiv](https://arxiv.org/abs/1801.09718)] [[cool](https://papers.cool/arxiv/1801.09718)] [[pdf](https://arxiv.org/pdf/1801.09718)]
> **Authors**: Mikyas T. Desta,Larry Chen,Tomasz Kornuta
> **First submission**: 2018-01-29
> **First announcement**: 2018-01-30
> **comment**: 10 pages, 15 figures, published as a conference paper at 2018 IEEE Winter Conf. on Applications of Computer Vision (WACV'2018)
- **标题**: VQA中基于对象的推理
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉问题回答（VQA）是一个新的问题域，必须处理多模式输入，以解决以自然语言形式给出的任务。由于解决方案本质上需要将视觉和自然语言处理与抽象推理相结合，因此该问题被视为AI完整。最近的进步表明，使用从输入中提取的高级抽象事实可能有助于推理。遵循该方向，我们决定开发一种结合最新对象检测和推理模块的解决方案。在均衡的CLEVR数据集上实现的结果，确认了承诺，并且在复杂的“计数”任务上的准确性提高了很大的提高。

## 计算机与社会(cs.CY:Computers and Society)

该领域共有 2 篇论文

### How will the Internet of Things enable Augmented Personalized Health? 
[[arxiv](https://arxiv.org/abs/1801.00356)] [[cool](https://papers.cool/arxiv/1801.00356)] [[pdf](https://arxiv.org/pdf/1801.00356)]
> **Authors**: Amit Sheth,Utkarshani Jaimini,Hong Yung Yip
> **First submission**: 2017-12-31
> **First announcement**: 2018-01-02
> **comment**: No comments
- **标题**: 物联网将如何实现增强的个性化健康？
- **领域**: 计算机与社会,人工智能
- **摘要**: Things的Internet（IoT）正在深刻重新定义我们创建，消费和共享信息的方式。卫生爱好者和公民越来越多地使用物联网技术来跟踪其睡眠，食物摄入，活动，重要的身体信号和其他生理观察。 IoT系统补充了这一点，这些系统不断从环境和生活区内收集与健康相关的数据。这些共同为新一代的医疗保健解决方案创造了机会。但是，解释数据以了解个人的健康是具有挑战性的。通常有必要查看该人的临床记录和行为信息，以及影响该人的社会和环境信息。解释患者的状况还需要查看他对各自的健康目标的遵守，相关临床知识的应用和预期的结果。我们诉诸于增强个性化医疗保健（APH）的愿景，以使用人工智能（AI）技术来利用广泛的相关数据和医学知识，以扩展和增强人类健康，以提出增强健康管理策略的各个阶段：自我监控，自我监控，自我认识，自我管理，自我管理，自我管理，干预，干预，干预和预测和预测。 KHealth技术是APH的特定化身及其在哮喘和其他疾病中的应用来提供插图并讨论技术辅助健康管理的替代方案。还确定了涉及物联网和患者生成的健康数据（PGHD）的几项重要努力，尊重将多模式数据转换为可操作的信息（对智能数据的大数据）。讨论了三个组成部分在基于证据的语义感知方法中的角色 - 情境化，抽象和个性化。

### Deep Learning for Fatigue Estimation on the Basis of Multimodal Human-Machine Interactions 
[[arxiv](https://arxiv.org/abs/1801.06048)] [[cool](https://papers.cool/arxiv/1801.06048)] [[pdf](https://arxiv.org/pdf/1801.06048)]
> **Authors**: Yuri Gordienko,Sergii Stirenko,Yuriy Kochura,Oleg Alienin,Michail Novotarskiy,Nikita Gordienko
> **First submission**: 2017-12-30
> **First announcement**: 2018-01-18
> **comment**: 12 pages, 10 figures, 1 table; presented at XXIX IUPAP Conference in Computational Physics (CCP2017) July 9-13, 2017, Paris, University Pierre et Marie Curie - Sorbonne (https://ccp2017.sciencesconf.org/program)
- **标题**: 根据多模式人机相互作用进行深度学习以进行疲劳估计
- **领域**: 计算机与社会,人机交互,机器学习
- **摘要**: 提出了新方法来通过几种客观和主观特征来监测当前物理负荷的水平和累积的疲劳水平。它应用于针对的数据集，以通过几种统计和机器学习方法估算物理负载和疲劳。通过多种统计和机器学习方法（时刻分析，集群分析，主成分分析等）收集，整合和分析了外围传感器（加速度计，GPS，陀螺仪，磁力计）和脑部计算界面（脑电图）的数据（脑电图）的数据。提出了假设1，并证明体育活动不仅可以通过客观参数进行分类，还可以通过主观参数进行分类。提出了假设2（可以估计疲劳水平，可以识别出独特的模式）（可以识别出疲劳水平），并提出了一些证明其证明它的方法。提出了几种“物理负载”和“疲劳”指标。提出的结果允许扩展机器学习方法的应用，以表征复杂的人类活动模式（例如，估计其实际的身体负荷和疲劳，并给予注意力和建议）。

## 人机交互(cs.HC:Human-Computer Interaction)

该领域共有 1 篇论文

### Proceedings of eNTERFACE 2015 Workshop on Intelligent Interfaces 
[[arxiv](https://arxiv.org/abs/1801.06349)] [[cool](https://papers.cool/arxiv/1801.06349)] [[pdf](https://arxiv.org/pdf/1801.06349)]
> **Authors**: Matei Mancas,Christian Frisson,Joëlle Tilmanne,Nicolas d'Alessandro,Petr Barborka,Furkan Bayansar,Francisco Bernard,Rebecca Fiebrink,Alexis Heloir,Edgar Hemery,Sohaib Laraba,Alexis Moinet,Fabrizio Nunnari,Thierry Ravet,Loïc Reboursière,Alvaro Sarasua,Mickaël Tits,Noé Tits,François Zajéga,Paolo Alborno,Ksenia Kolykhalova,Emma Frid,Damiano Malafronte,Lisanne Huis in't Veld,Hüseyin Cakmak, et al. (49 additional authors not shown)
> **First submission**: 2018-01-19
> **First announcement**: 2018-01-22
> **comment**: 159 pages
- **标题**: Enterface 2015智能接口研讨会论文集
- **领域**: 人机交互,人工智能,计算机视觉和模式识别
- **摘要**: 2015年8月10日至2015年8月10日，Numediart Creative Technologies举办了关于多模式接口Enterface 2015的第11个夏季研讨会。在2015年8月10日至9月。在四个星期中，来自世界各地的学生和研究人员聚集在蒙斯大学的Numediart Institute of Mons大学的Numediart Institute，以围绕智能交织的智能项目进行八个选定的项目。选择了八个项目，并在此处显示其报告。

## 信息检索(cs.IR:Information Retrieval)

该领域共有 1 篇论文

### Cross-modal Embeddings for Video and Audio Retrieval 
[[arxiv](https://arxiv.org/abs/1801.02200)] [[cool](https://papers.cool/arxiv/1801.02200)] [[pdf](https://arxiv.org/pdf/1801.02200)]
> **Authors**: Didac Surís,Amanda Duarte,Amaia Salvador,Jordi Torres,Xavier Giró-i-Nieto
> **First submission**: 2018-01-07
> **First announcement**: 2018-01-08
> **comment**: 6 pages, 3 figures
- **标题**: 视频和音频检索的跨模式嵌入
- **领域**: 信息检索,计算机视觉和模式识别,声音,音频和语音处理
- **摘要**: 越来越多的在线视频为培训自我监督的神经网络带来了一些机会。创建大型视频（例如YouTube-8M）的大规模数据集使我们能够以可管理的方式处理大量数据。在这项工作中，我们通过利用其提供的多模式信息来找到利用此数据集的新方法。通过神经网络，我们能够通过将它们投影到特征空间的一个共同区域，从而获得联合视听嵌入方式，从而在音频和视觉文档之间创建链接。这些链接用于检索非常适合给定无声视频的音频样本，还可以检索与给定查询音频相匹配的图像。在YouTube-8M视频子集的子集中获得的Recomn@K的结果显示了这种无监督方法的跨模式特征学习的潜力。我们在检索问题中训练嵌入式的嵌入式，并在检索问题中评估其质量，以根据一种模式提取的功能根据其他模式中计算出的功能来检索最相似的视频。

## 多代理系统(cs.MA:Multiagent Systems)

该领域共有 1 篇论文

### Stable Marriage with Multi-Modal Preferences 
[[arxiv](https://arxiv.org/abs/1801.02693)] [[cool](https://papers.cool/arxiv/1801.02693)] [[pdf](https://arxiv.org/pdf/1801.02693)]
> **Authors**: Jiehua Chen,Rolf Niedermeier,Piotr Skowron
> **First submission**: 2018-01-08
> **First announcement**: 2018-01-09
> **comment**: No comments
- **标题**: 稳定的婚姻具有多模式的偏好
- **领域**: 多代理系统,数据结构和算法
- **摘要**: 我们介绍了著名的稳定婚姻问题的广义版，现在基于多模式的偏好列表。这里的中心扭曲是允许每个代理基于多种“评估模式”（例如，一个以上的标准）对其潜在匹配的对应物进行排名；因此，每个代理都配备了多个优先列表，每个座席都以可能不同的方式对对应物进行排名。我们介绍并研究了三种自然稳定性概念，研究它们的相互关系，并将重点放在计算复杂性方面，相对于这些新场景中的计算稳定匹配。大多数情况下，我们还可以发现很少的障碍岛，并与\ textsc {Graph isomorthismismist}问题建立令人惊讶的联系。

## 机器学习(stat.ML:Machine Learning)

该领域共有 1 篇论文

### Improving Bi-directional Generation between Different Modalities with Variational Autoencoders 
[[arxiv](https://arxiv.org/abs/1801.08702)] [[cool](https://papers.cool/arxiv/1801.08702)] [[pdf](https://arxiv.org/pdf/1801.08702)]
> **Authors**: Masahiro Suzuki,Kotaro Nakayama,Yutaka Matsuo
> **First submission**: 2018-01-26
> **First announcement**: 2018-01-29
> **comment**: Updated version of arXiv:1611.01891
- **标题**: 通过各种自动编码器，改善不同方式之间的双向生成
- **领域**: 机器学习,机器学习
- **摘要**: 我们研究了可以通过双向交换多种模式的深层生成模型，例如，从相应的文本中生成图像，反之亦然。实现这一目标的主要方法是训练一个模型，该模型将不同模式的所有信息整合到联合表示中，然后通过该联合表示从相应的其他方式产生一种模式。我们只是将这种方法应用于变分自动编码器（VAE），我们称之为联合多模式变异自动编码器（JMVAE）。但是，我们发现，当该模型试图在输入处产生较大的维数时，关节表示崩溃了，并且无法成功生成这种模态。此外，我们确认即使使用已知解决方案也无法解决这个困难。因此，在这项研究中，我们提出了两个模型以防止这种困难：JMVAE-KL和JMVAE-H。我们的实验结果表明，这些方法可以防止上述难度，并且它们比传统的VAE方法以相等或更高的可能性在双向方向上产生模态，而传统VAE方法仅在一个方向上产生。此外，我们确认这些方法可以适当地获得联合表示形式，以便它们可以通过移动关节表示或更改另一种方式的价值来产生各种方式。

## 其他论文

共有 11 篇其他论文

- [A Study on the Use of Eye Tracking to Adapt Gameplay and Procedural Content Generation in First-Person Shooter Games](https://arxiv.org/abs/1801.01565)
  - **标题**: 一项有关使用眼动追踪在第一人称射击游戏中适应游戏玩法和程序内容的研究的研究
  - **Filtered Reason**: none of cs.HC in whitelist
- [Narrating Networks](https://arxiv.org/abs/1801.01322)
  - **标题**: 叙述网络
  - **Filtered Reason**: none of physics.soc-ph,cs.SI in whitelist
- [Binning based algorithm for Pitch Detection in Hindustani Classical Music](https://arxiv.org/abs/1801.02155)
  - **标题**: 基于binning的算法，用于印度斯坦古典音乐中的音高检测
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [DCASE 2017 Task 1: Acoustic Scene Classification Using Shift-Invariant Kernels and Random Features](https://arxiv.org/abs/1801.02690)
  - **标题**: Dcase 2017任务1：使用Shift-Invariant内核和随机功能的声学场景分类
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [Speech Dereverberation Based on Integrated Deep and Ensemble Learning Algorithm](https://arxiv.org/abs/1801.04052)
  - **标题**: 基于集成深层和合奏学习算法的语音修复
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [To Relay or not to Relay: Open Distance and Optimal Deployment for Linear Underwater Acoustic Networks](https://arxiv.org/abs/1801.03641)
  - **标题**: 继电器或不中继：线性水下声学网络的开放距离和最佳部署
  - **Filtered Reason**: none of cs.IT in whitelist
- [Compact Real-time avoidance on a Humanoid Robot for Human-robot Interaction](https://arxiv.org/abs/1801.05671)
  - **标题**: 在人形机器人相互作用的人形机器人上的紧凑实时回避
  - **Filtered Reason**: none of cs.RO in whitelist
- [Coactivated Clique Based Multisource Overlapping Brain Subnetwork Extraction](https://arxiv.org/abs/1801.09589)
  - **标题**: 基于集团的多元化重叠大脑子网提取
  - **Filtered Reason**: none of cs.SI,q-bio.NC in whitelist
- [A Comparison of Visualisation Methods for Disambiguating Verbal Requests in Human-Robot Interaction](https://arxiv.org/abs/1801.08760)
  - **标题**: 比较人类机器人互动中歧义口头请求的可视化方法的比较
  - **Filtered Reason**: none of cs.HC,cs.RO in whitelist
- [A Single-Planner Approach to Multi-Modal Humanoid Mobility](https://arxiv.org/abs/1801.10225)
  - **标题**: 多模式类人动物迁移率的单拼手方法
  - **Filtered Reason**: none of cs.RO in whitelist
- [CREATE: Multimodal Dataset for Unsupervised Learning, Generative Modeling and Prediction of Sensory Data from a Mobile Robot in Indoor Environments](https://arxiv.org/abs/1801.10214)
  - **标题**: 创建：用于无监督学习，生成建模和从室内环境中移动机器人的感觉数据预测的多模式数据集
  - **Filtered Reason**: none of cs.RO in whitelist
