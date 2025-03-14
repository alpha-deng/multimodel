# 2017-01 月度论文分类汇总

共有17篇相关领域论文, 另有7篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 6 篇论文

### Unsupervised neural and Bayesian models for zero-resource speech processing 
[[arxiv](https://arxiv.org/abs/1701.00851)] [[cool](https://papers.cool/arxiv/1701.00851)] [[pdf](https://arxiv.org/pdf/1701.00851)]
> **Authors**: Herman Kamper
> **First submission**: 2017-01-03
> **First announcement**: 2017-01-04
> **comment**: PhD thesis, University of Edinburgh, 107 pages, submitted and accepted 2016
- **标题**: 无监督的神经和贝叶斯模型用于零资源语音处理
- **领域**: 计算语言学,机器学习
- **摘要**: 在只有未标记的语音数据的设置中，需要开发零资源的语音技术，而无需抄录，发音词典或语言建模文本。零资源的语音处理中有两个主要问题：（i）查找框架级特征表示形式，这些特征表示更容易区分语言单位（电话或单词），以及（ii）将未标记的语音分割为有意义的单位。在本论文中，我们认为自上而下的建模和自下而上的建模在解决这两个问题方面是有利的。为了解决框架级表示学习的问题，我们介绍了通信自动编码器（CAE），这是一种神经网络，该神经网络受到无监督术语发现系统的自上而下监督的训练。通过将这种自上而下的监督与无监督的自下而上初始化相结合，CAE比以前的方法产生的歧视性更大。然后，我们介绍了无监督的分段贝叶斯模型，该模型将无标记的语音分为假设的单词。通过施加一致的自上而下的细分，同时还使用了来自检测到的音节边界的自下而上的知识，我们的系统在多演讲者对话英语和Xitsonga语音数据上表现优于其他几个知识。最后，我们表明，通过使用CAE而不是传统的声学特征，可以使贝叶斯模型发现的群集变得较少。总而言之，本文中介绍的不同模型和系统表明，自上而下和自下而上的建模都可以改善未标记语音数据的表示，分割和聚类的表示。

### Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks 
[[arxiv](https://arxiv.org/abs/1701.02720)] [[cool](https://papers.cool/arxiv/1701.02720)] [[pdf](https://arxiv.org/pdf/1701.02720)]
> **Authors**: Ying Zhang,Mohammad Pezeshki,Philemon Brakel,Saizheng Zhang,Cesar Laurent Yoshua Bengio,Aaron Courville
> **First submission**: 2017-01-10
> **First announcement**: 2017-01-11
> **comment**: No comments
- **标题**: 通过深度卷积神经网络进行端到端的语音识别
- **领域**: 计算语言学,机器学习,机器学习
- **摘要**: 卷积神经网络（CNN）是减少声学特征中自动语音识别（ASR）中频谱变化和建模光谱相关的有效模型。混合语音识别系统将CNN与隐藏的Markov模型/高斯混合模型（HMMS/GMM）结合在一起，已在各种基准中实现了最新的。同时，提出的用于标记未分段序列的连接派时间分类（CTC），这使得训练端到端的语音识别系统而不是混合设置，这是可行的。但是，RNN在计算上很昂贵，有时很难训练。在本文中，受到CNN和CTC方法的优势的启发，我们通过将层次CNN与CTC直接直接无需复发连接结合，提出了一个序列标记的端到端语音框架。通过评估TIMIT音素识别任务上的方法，我们表明所提出的模型不仅在计算上是有效的，而且还与现有基线系统竞争。此外，我们认为CNN具有与适当的上下文信息建模时间相关性的能力。

### End-to-End ASR-free Keyword Search from Speech 
[[arxiv](https://arxiv.org/abs/1701.04313)] [[cool](https://papers.cool/arxiv/1701.04313)] [[pdf](https://arxiv.org/pdf/1701.04313)]
> **Authors**: Kartik Audhkhasi,Andrew Rosenberg,Abhinav Sethy,Bhuvana Ramabhadran,Brian Kingsbury
> **First submission**: 2017-01-13
> **First announcement**: 2017-01-16
> **comment**: Published in the IEEE 2017 International Conference onAcoustics, Speech, andSignalProcessing (ICASSP 2017), scheduled for 5-9 March 2017 in New Orleans, Louisiana, USA
- **标题**: 来自语音的端到端无ASR关键字搜索
- **领域**: 计算语言学,信息检索,机器学习,神经和进化计算
- **摘要**: 与常规的混合隐藏模型（HMM） - 基于基于基于的自动语音识别（ASR）系统相比，端到端（E2E）系统已经取得了竞争成果。由于缺乏对训练过程中输入声和输出素或HMM状态序列之间对齐的依赖，因此这种E2E系统具有吸引力。本文探讨了通过最少的监督训练的语音，用于基于文本查询的关键字搜索（KWS）的无ASR端到端系统的设计。我们的E2E KWS系统由三个子系统组成。第一个子系统是基于经常性的神经网络（RNN）的声学自动编码器，该系统通过有限的维度表示，可以通过有限的尺寸表示来重建音频。第二个子系统是使用从卷积神经网络中学到的嵌入的角色级RNN语言模型。由于声学和文本查询嵌入占据了不同的表示空间，因此它们输入了第三个馈送前向神经网络，该网络可以预测查询是否发生在声音中。尽管缺乏常规的ASR系统，并且训练速度要快得多，但该无ASR KWS系统的性能表现出色。

### Incorporating Global Visual Features into Attention-Based Neural Machine Translation 
[[arxiv](https://arxiv.org/abs/1701.06521)] [[cool](https://papers.cool/arxiv/1701.06521)] [[pdf](https://arxiv.org/pdf/1701.06521)]
> **Authors**: Iacer Calixto,Qun Liu,Nick Campbell
> **First submission**: 2017-01-23
> **First announcement**: 2017-01-24
> **comment**: 8 pages (11 including references), 5 figures
- **标题**: 将全局视觉特征纳入基于注意的神经机器翻译
- **领域**: 计算语言学
- **摘要**: 我们引入了多模式，基于注意力的神经机器翻译（NMT）模型，该模型将视觉特征纳入编码器和解码器的不同部分。我们利用使用预训练的卷积神经网络提取的全局图像特征，并将其（i）作为源句子中的单词（i）纳入（ii）来初始化编码器隐藏状态，而（iii）作为其他数据来初始化解码器隐藏状态。在我们的实验中，我们评估了这些不同的策略如何合并全球图像特征并表现最佳。我们还研究了添加合成多模式的多语言数据带来的影响，并发现其他数据对多模式模型具有积极影响。我们报告了新的最先进结果，我们的最佳模型也可以显着改善基于所有评估的指标的基于多330K数据集的基于可比的统计MT（PBSMT）模型。据我们所知，这是对此数据集评估的所有指标的PBSMT模型，这是第一次纯粹的神经模型。

### Learning Word-Like Units from Joint Audio-Visual Analysis 
[[arxiv](https://arxiv.org/abs/1701.07481)] [[cool](https://papers.cool/arxiv/1701.07481)] [[pdf](https://arxiv.org/pdf/1701.07481)]
> **Authors**: David Harwath,James R. Glass
> **First submission**: 2017-01-25
> **First announcement**: 2017-01-26
> **comment**: No comments
- **标题**: 从联合视听分析中学习类似单词的单元
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 给定图像和口语音频字幕的集合，我们提出了一种在连续语音信号中发现类似单词的声学单元并将其接地到语义相关的图像区域的方法。例如，我们的模型能够检测说话中“灯塔”一词的口语实例，并将它们与包含灯塔的图像区域相关联。我们不使用任何形式的常规自动语音识别，也不使用任何文本转录或常规语言注释。我们的模型有效地实现了一种口语获取的形式，其中计算机不仅可以通过声音识别单词类别，还可以通过将其通过图像将其接地来丰富它在语义上学习的单词。

### Image-Grounded Conversations: Multimodal Context for Natural Question and Response Generation 
[[arxiv](https://arxiv.org/abs/1701.08251)] [[cool](https://papers.cool/arxiv/1701.08251)] [[pdf](https://arxiv.org/pdf/1701.08251)]
> **Authors**: Nasrin Mostafazadeh,Chris Brockett,Bill Dolan,Michel Galley,Jianfeng Gao,Georgios P. Spithourakis,Lucy Vanderwende
> **First submission**: 2017-01-28
> **First announcement**: 2017-01-30
> **comment**: No comments
- **标题**: 图像接地的对话：自然问题和回答产生的多模式上下文
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 图像共享在社交媒体上的普及及其在用户之间创造的参与反映了视觉上下文在日常对话中所扮演的重要作用。我们提出了一项新颖的任务，图像接地的对话（IGC），其中生成了关于共享图像的自然对话。为了进行基准进步，我们引入了一个新的多次参考数据集，其中包括以事件为中心的图像对话。 IGC属于Chit-Chat和目标导向的对话模型之间的连续性，在这些对话模型中，视觉接地将对话主题限制为事件驱动的话语。在社交媒体数据上训练的模型的实验表明，视觉和文本上下文的组合增强了产生的对话转弯的质量。在人类评估中，人类绩效与神经和检索结构之间的差距表明，多模式IGC对对话研究提出了一个有趣的挑战。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 7 篇论文

### Vid2speech: Speech Reconstruction from Silent Video 
[[arxiv](https://arxiv.org/abs/1701.00495)] [[cool](https://papers.cool/arxiv/1701.00495)] [[pdf](https://arxiv.org/pdf/1701.00495)]
> **Authors**: Ariel Ephrat,Shmuel Peleg
> **First submission**: 2017-01-02
> **First announcement**: 2017-01-03
> **comment**: Accepted for publication at ICASSP 2017
- **标题**: vid2speech：无声视频中的语音重建
- **领域**: 计算机视觉和模式识别,声音
- **摘要**: 对于人类而言，语音阅读是一项艰巨的任务。在本文中，我们提出了一种基于卷积神经网络（CNN）的端到端模型，该模型用于从讲话人的无声视频帧中发出可理解的声学语音信号。提出的CNN根据其相邻帧为每个帧生成声音功能。然后从学习的语音特征中综合波形以产生可理解的语音。我们表明，通过利用CNN的自动功能学习能力，我们可以在网格数据集中获得最先进的单词清晰度，并显示出令人鼓舞的结果，用于学习播放唱片（OOV）单词。

### A Unified RGB-T Saliency Detection Benchmark: Dataset, Baselines, Analysis and A Novel Approach 
[[arxiv](https://arxiv.org/abs/1701.02829)] [[cool](https://papers.cool/arxiv/1701.02829)] [[pdf](https://arxiv.org/pdf/1701.02829)]
> **Authors**: Chenglong Li,Guizhao Wang,Yunpeng Ma,Aihua Zheng,Bin Luo,Jin Tang
> **First submission**: 2017-01-10
> **First announcement**: 2017-01-11
> **comment**: No comments
- **标题**: 统一的RGB-T显着性检测基准：数据集，基准，分析和一种新颖的方法
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管取得了重大进展，但在复杂的场景和环境中，图像显着性检测仍然仍然是一项具有挑战性的任务。整合多个不同但互补的提示，例如RGB和Thermal（RGB-T），可能是提高显着性检测性能的有效方法。但是，目前朝这个方向的研究受到缺乏全面基准的限制。这项工作贡献了这样的RGB-T图像数据集，其中包括821个空间对齐的RGB-T图像对及其以实现显着性检测目的的地面真相注释。图像对具有在不同的场景和环境条件下记录的高多样性，我们在这些图像对上注释了11个挑战，用于对不同显着性检测算法进行挑战敏感分析。我们还实施了三种具有不同模态输入的基线方法，以提供全面的比较平台。通过此基准，我们提出了一种新颖的方法，具有跨模式一致性的多任务歧管排名，用于RGB-T显着性检测。特别是，我们引入了每种模式的权重来描述可靠性，并将它们集成到基于图的歧管排名算法中，以实现不同源数据的自适应融合。此外，我们结合了跨模式一致的约束，以协作整合不同的模式。为了进行优化，我们针对迭代的几个子问题设计了有效的算法，该算法使用封闭式溶液求解。针对新创建的基准测试的其他基线方法进行了广泛的实验，证明了拟议方法的有效性，我们还为RGB-T显着性检测提供了基本的见解和潜在的未来研究方向。

### Attention-Based Multimodal Fusion for Video Description 
[[arxiv](https://arxiv.org/abs/1701.03126)] [[cool](https://papers.cool/arxiv/1701.03126)] [[pdf](https://arxiv.org/pdf/1701.03126)]
> **Authors**: Chiori Hori,Takaaki Hori,Teng-Yok Lee,Kazuhiro Sumi,John R. Hershey,Tim K. Marks
> **First submission**: 2017-01-11
> **First announcement**: 2017-01-12
> **comment**: Resubmitted to the rebuttal for CVPR 2017 for review, 8 pages, 4 figures
- **标题**: 基于注意的视频描述的多模式融合
- **领域**: 计算机视觉和模式识别,计算语言学,多媒体
- **摘要**: 目前，用于视频描述的成功方法基于使用经常租赁神经网络（RNN）的编码器句子生成。最近的工作表明，将时间和/或空间注意机制集成到这些模型中的优点，在这些模型中，解码器的网络工作可以通过选择性地对特定时间范围（时间关注）或特定空间区域（空间注意）的特定特定时间范围（时间范围）的特征进行选择更多地对描述中的每个单词进行预测。在本文中，我们建议将注意力模型扩展到有选择地参加特定时间或空间区域，而是针对输入的特定方式，例如图像功能，运动功能和音频功能。我们称之为多模式注意的新的新模式依赖性注意机制为视频描述融合多模式信息提供了一种自然的方式。我们在YouTube2Text数据集上评估了我们的方法，从而实现了与当前最新状态具有竞争力的结果。更重要的是，我们证明了我们的模型结合了多模式的关注以及时间的关注明显优于单独使用时间关注的模型。

### Auxiliary Multimodal LSTM for Audio-visual Speech Recognition and Lipreading 
[[arxiv](https://arxiv.org/abs/1701.04224)] [[cool](https://papers.cool/arxiv/1701.04224)] [[pdf](https://arxiv.org/pdf/1701.04224)]
> **Authors**: Chunlin Tian,Weijun Ji
> **First submission**: 2017-01-16
> **First announcement**: 2017-01-17
> **comment**: 8 pages, 4 figures
- **标题**: 辅助多模式LSTM用于视听语音识别和唇读
- **领域**: 计算机视觉和模式识别
- **摘要**: 使用视频和音频信息进行自动语音识别（ASR）的Aduio-Visual语音识别（AVSR）是多模式倾斜的应用之一，使ASR系统更加稳健和准确。传统模型通常将AVSR视为推理或投影，但严格的先验限制了其能力。随着深度学习的复兴，深度神经网络（DNN）成为许多传统分类任务（包括ASR，图像分类，自然语言处理）的重要工具包。一些DNN模型在AVSR中使用，例如多模式深度自动编码器（MDAE），多模式深信信仰网络（MDBN）和多模式的深玻尔兹曼（MDBM），实际上比传统方法更好。但是，这样的DNN模型有几个缺点：（1）它们不平衡模态融合和时间融合，甚至没有时间融合； （2）这些模型的体系结构不是端到端的，而是训练和测试变得繁琐。我们提出了DNN模型，即辅助多模式LSTM（AM-LSTM），以克服这种弱点。可以对AM-LSTM进行一次训练和测试，此外，易于训练并防止自动过度拟合。还考虑了可扩展性和灵活性。实验表明，AM-LSTM比三个数据集中的传统方法和其他DNN模型好得多。

### Multimodal Fusion via a Series of Transfers for Noise Removal 
[[arxiv](https://arxiv.org/abs/1701.06121)] [[cool](https://papers.cool/arxiv/1701.06121)] [[pdf](https://arxiv.org/pdf/1701.06121)]
> **Authors**: Chang-Hwan Son,Xiao-Ping Zhang
> **First submission**: 2017-01-21
> **First announcement**: 2017-01-23
> **comment**: No comments
- **标题**: 通过一系列传输进行噪声的多模式融合
- **领域**: 计算机视觉和模式识别
- **摘要**: 近红外成像被认为是在昏暗的照明条件下提供高质量照片的解决方案。该成像系统捕获了两种类型的多模式图像：一个是近红外灰色图像（NGI），另一个是可见的颜色图像（VCI）。 NGI是无噪音的，但它是灰度，而VCI的颜色，但包含噪音。此外，NGI和VCI之间存在严重的边缘和亮度差异。为了解决这个问题，提出了一种新的基于转移的融合方法以删除噪声。与常规融合方法不同，所提出的方法进行了一系列转移：对比度，细节和颜色转移。首先，提出的对比和细节转移旨在解决严重的差异问题，从而创造出新的免噪声和细节的NGI。其次，提出的色彩转移通过线性变换从DeNoed VCI中模拟未知的颜色，然后将自然颜色转移到新生成的NGI中。实验结果表明，提出的基于转移的融合方法在解决差异问题方面非常成功，从而清楚地描述了边缘和纹理，并完全消除了融合图像上的噪声。最重要的是，所提出的方法优于常规融合方法和指导过滤，甚至基于规模图和层分解的最新融合方法。

### Speech Map: A Statistical Multimodal Atlas of 4D Tongue Motion During Speech from Tagged and Cine MR Images 
[[arxiv](https://arxiv.org/abs/1701.06708)] [[cool](https://papers.cool/arxiv/1701.06708)] [[pdf](https://arxiv.org/pdf/1701.06708)]
> **Authors**: Jonghye Woo,Fangxu Xing,Maureen Stone,Jordan Green,Timothy G. Reese,Thomas J. Brady,Van J. Wedeen,Jerry L. Prince,Georges El Fakhri
> **First submission**: 2017-01-23
> **First announcement**: 2017-01-24
> **comment**: Accepted at Journal of Computer Methods in Biomechanics and Biomedical Engineering
- **标题**: 语音图：在标记和Cine MR图像的语音中，4D舌头运动的统计多模式图集
- **领域**: 计算机视觉和模式识别
- **摘要**: 在言语或其他舌外行为过程中，4D舌运动的功能和解剖特征的定量测量仍然是科学研究和临床应用中的主要挑战。在这里，我们使用健康受试者引入了4D舌头运动的统计多模式地图集，这可以在参考解剖构型中对舌头运动的合并定量表征。该地图集的框架称为语音图，结合了Cine和标记的MRI，以便在语音期间提供解剖参考和运动信息。我们的方法涉及一系列步骤，包括（1）从Cine-MRI中构建常见的参考解剖构型，（2）从标记的MRI中估算运动估计，（3）运动估计到参考解剖结构的转换，以及（4）运动量的计算，例如Lagrangian菌株。使用此框架，舌头的解剖结构似乎一动不动，而运动场和相关的应变测量值在语音的时间过程中变化。此外，为了形成高维和复杂运动场的简洁表示，进行主成分分析以表征我们语音任务的运动场的中心趋势和变化。我们提出的方法为定量和客观地解释舌头运动的差异和可变性提供了一个平台，通过阐明迄今为止棘手的内部运动和压力。这些发现用于了解舌头功能如何受到异常内运动和震荡切除术患者的应变的限制。

### MSCM-LiFe: Multi-scale cross modal linear feature for horizon detection in maritime images 
[[arxiv](https://arxiv.org/abs/1701.08378)] [[cool](https://papers.cool/arxiv/1701.08378)] [[pdf](https://arxiv.org/pdf/1701.08378)]
> **Authors**: D. K. Prasad,D. Rajan,C. K. Prasath,L. Rachmawati,E. Rajabaly,C. Quek
> **First submission**: 2017-01-29
> **First announcement**: 2017-01-30
> **comment**: 5 pages, 4 figures, IEEE TENCON 2016
- **标题**: MSCM生活：海上图像中的地平线检测的多尺度交叉模态线性特征
- **领域**: 计算机视觉和模式识别
- **摘要**: 本文提出了一种新方法，用于地平线检测，称为多尺度交叉模态线性特征。该方法整合了与海洋图像中地平线存在有关的三种不同概念，以提高地平线检测的准确性。具体而言，它使用了多尺度中值滤波中的地平线的持久性，并且它作为通常通过两种不同方法检测到的线性特征，即EDGEMAP和强度梯度的霍夫变换。我们证明了该方法超过13个视频的性能，其中包括3000多个帧，并表明所提出的方法在大多数情况下都以微小错误检测地平线，表现优于三种最先进的方法。

## 机器学习(cs.LG:Machine Learning)

该领域共有 1 篇论文

### Unsupervised Latent Behavior Manifold Learning from Acoustic Features: audio2behavior 
[[arxiv](https://arxiv.org/abs/1701.03198)] [[cool](https://papers.cool/arxiv/1701.03198)] [[pdf](https://arxiv.org/pdf/1701.03198)]
> **Authors**: Haoqi Li,Brian Baucom,Panayiotis Georgiou
> **First submission**: 2017-01-11
> **First announcement**: 2017-01-12
> **comment**: Accepted by ICASSP 2017
- **标题**: 无监督的潜在行为歧管从声学特征学习：Audio2Behavior
- **领域**: 机器学习,声音
- **摘要**: 使用信号处理和机器学习的行为注释高度取决于训练数据和行为标签的手动注释。先前的研究表明，语音信息编码重要的行为信息，并用于各种自动行为识别任务。但是，由于训练数据的稀少度以及语音的复杂，高度的语音以及其编码的复杂和多个信息流，因此从语音中提取行为信息仍然是一项艰巨的任务。在这项工作中，我们利用了人类行为的慢速变化特性。我们假设附近的语音部分具有相同的行为环境，因此在潜在空间中共享相似的基础代表。具体而言，我们提出了一个深神网络（DNN）模型，以连接行为上下文并以无监督的方式得出行为歧管。我们评估了夫妻治疗领域中提出的歧管，并提供了公开可用数据（例如站立喜剧）的示例。我们进一步研究了夫妻治疗领域和电影数据中的培训。结果极为令人鼓舞，并有望以无监督的方式改善行为量化，并需要在一系列应用中进一步研究。

## 多代理系统(cs.MA:Multiagent Systems)

该领域共有 1 篇论文

### LocDyn: Robust Distributed Localization for Mobile Underwater Networks 
[[arxiv](https://arxiv.org/abs/1701.08027)] [[cool](https://papers.cool/arxiv/1701.08027)] [[pdf](https://arxiv.org/pdf/1701.08027)]
> **Authors**: Cláudia Soares,João Gomes,Beatriz Ferreira,João Paulo Costeira
> **First submission**: 2017-01-27
> **First announcement**: 2017-01-30
> **comment**: No comments
- **标题**: LOCDYN：移动水下网络的强大分布本地化
- **领域**: 多代理系统,优化与控制,机器学习
- **摘要**: 如何仅使用嘈杂的范围测量值来自我定位的大型水下节点？如何以分布式的方式进行操作，并将动态纳入问题？如何拒绝异常值并产生可信赖的立场估计？严格的声学通信渠道和我们的地球物理调查应用程序的准确性需求要求更快，更准确的定位方法。我们将动态定位作为一个地图估计问题，其中先前的编码动力学，并设计了一种凸宽松方法，该方法在每个测量采集步骤中利用先前的估计值；对于一阶方法，该算法以最佳速率收敛。 LOCDYN是分布式的：没有负责处理获得的数据的融合中心，并且为每个节点执行相同的简单计算。 LOCDYN是准确的：实验证明了比可比的卡尔曼滤波器较小的定位误差。 LOCDYN很健壮：它拒绝异常噪声，而比较方法在定位误差方面屈服。

## 机器学习(stat.ML:Machine Learning)

该领域共有 2 篇论文

### Manifold Alignment Determination: finding correspondences across different data views 
[[arxiv](https://arxiv.org/abs/1701.03449)] [[cool](https://papers.cool/arxiv/1701.03449)] [[pdf](https://arxiv.org/pdf/1701.03449)]
> **Authors**: Andreas Damianou,Neil D. Lawrence,Carl Henrik Ek
> **First submission**: 2017-01-12
> **First announcement**: 2017-01-13
> **comment**: NIPS workshop onMulti-ModalMachine Learning, 2015
- **标题**: 歧管对准确定：从不同数据视图中查找对应关系
- **领域**: 机器学习,机器学习,可能性
- **摘要**: 我们提出了多种对准测定（MAD），这是一种从多种视图或模态之间学习数据点之间学习对准的算法。该方法能够学习视图之间的对应关系以及单个数据点之间的对应关系。所提出的方法仅需要几个对齐示例，从而可以通过概率模型从中恢复全局对齐。生成模型提供的强大而灵活的正则化足以使视图对齐。我们提供有关合成和真实数据的实验，以突出提出的方法的好处。

### Diffusion-based nonlinear filtering for multimodal data fusion with application to sleep stage assessment 
[[arxiv](https://arxiv.org/abs/1701.03619)] [[cool](https://papers.cool/arxiv/1701.03619)] [[pdf](https://arxiv.org/pdf/1701.03619)]
> **Authors**: Ori Katz,Ronen Talmon,Yu-Lun Lo,Hau-Tieng Wu
> **First submission**: 2017-01-13
> **First announcement**: 2017-01-16
> **comment**: :62-07
- **标题**: 基于扩散的非线性滤波，用于多模式数据融合，并应用于睡眠阶段评估
- **领域**: 机器学习,机器学习,数据分析、统计和概率
- **摘要**: 多年来，从多模式传感器获得的多个数据集中获得的信息融合问题吸引了大量的研究关注。在本文中，我们专注于一个特定的问题设置，该设置由物理现象或多个传感器观察到的感兴趣系统组成。我们假设所有传感器都通过其他传感器特异性和无关紧要的组件来测量关注系统的某些方面。我们的目标是恢复与观察到的系统相关的变量，并过滤传感器特异性变量的滋扰作用。我们提出了一种基于流形学习的方法，该方法特别适合于多种方式问题，因为它旨在捕获数据的内在结构并依赖于最少的先前模型知识。具体而言，我们提出了一种非线性过滤方案，该方案提取了由两个或多个传感器捕获的可变性源的隐藏源，它们与传感器特异性组件无关。除了提出理论分析外，我们还基于多个多模态传感器测量值的睡眠阶段评估来证明我们对实际测量数据的技术。我们表明，如果没有关于不同模式和测量系统的知识，我们的方法会引起数据驱动的表示，该表示与基础睡眠过程非常相关，并且与噪声和传感器特异性效应相关。

## 其他论文

共有 7 篇其他论文

- [Loss and Bandwidth Studies on Multimode Polymer Waveguide Components for On-Board High-Speed Optical Interconnects](https://arxiv.org/abs/1701.00846)
  - **标题**: 板上高速光学互连的多模聚合物波导组件的损失和带宽研究
  - **Filtered Reason**: none of cs.ET in whitelist
- [Compressive Sensing-Based Detection with Multimodal Dependent Data](https://arxiv.org/abs/1701.01352)
  - **标题**: 具有多模式依赖数据的基于压缩感应的检测
  - **Filtered Reason**: none of cs.IT,stat.AP in whitelist
- [Modeling Grasp Motor Imagery through Deep Conditional Generative Models](https://arxiv.org/abs/1701.03041)
  - **标题**: 通过深层有条件生成模型对抓住运动图像进行建模
  - **Filtered Reason**: none of stat.ML,cs.RO in whitelist
- [Optimized Spatial Partitioning via Minimal Swarm Intelligence](https://arxiv.org/abs/1701.05553)
  - **标题**: 通过最小群体智能进行优化的空间分区
  - **Filtered Reason**: none of stat.ME,cs.NE in whitelist
- [Learning Mid-Level Auditory Codes from Natural Sound Statistics](https://arxiv.org/abs/1701.07138)
  - **标题**: 从自然声音统计数据中学习中层听觉代码
  - **Filtered Reason**: none of cs.SD,q-bio.NC in whitelist
- [Design and Implementation of a Semantic Dialogue System for Radiologists](https://arxiv.org/abs/1701.07381)
  - **标题**: 放射科医生的语义对话系统的设计和实施
  - **Filtered Reason**: none of cs.HC in whitelist
- [Source localization in an ocean waveguide using supervised machine learning](https://arxiv.org/abs/1701.08431)
  - **标题**: 使用有监督的机器学习中的海洋波导中的源头定位
  - **Filtered Reason**: none of cs.NE,physics.geo-ph,physics.ao-ph in whitelist
