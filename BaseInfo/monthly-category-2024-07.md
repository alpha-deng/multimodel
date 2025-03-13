# 2024-07 月度论文分类汇总

共有27篇相关领域论文, 另有23篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 5 篇论文

### NAIST Simultaneous Speech Translation System for IWSLT 2024 
[[arxiv](https://arxiv.org/abs/2407.00826)] [[cool](https://papers.cool/arxiv/2407.00826)] [[pdf](https://arxiv.org/pdf/2407.00826)]
> **Authors**: Yuka Ko,Ryo Fukuda,Yuta Nishikawa,Yasumasa Kano,Tomoya Yanagita,Kosuke Doi,Mana Makinae,Haotian Tan,Makoto Sakai,Sakriani Sakti,Katsuhito Sudoh,Satoshi Nakamura
> **First submission**: 2024-06-30
> **First announcement**: 2024-07-01
> **comment**: IWSLT 2024 system paper
- **标题**: IWSLT 2024的Naist同时语音翻译系统
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 本文描述了NAIST对IWSLT 2024评估活动的同时曲目的提交：英语至 -  {德语，日语，中文}语音到文本翻译和英语到日本的语音到语音翻译。我们开发了一个多语言的端到端语音到文本翻译模型，结合了两个预训练的语言模型Hubert和Mbart。我们通过两个解码政策，本地协议（LA）和Alignatt培训了该模型。提交的模型采用了洛杉矶政策，因为它在以前的模型中优于Alignatt策略。我们的语音到语音翻译方法是上述语音到文本模型的级联和一个增量文本到语音（TTS）模块，该模块包含音素估计模型，平行声学模型和平行的Woveallan vocoder。我们通过使用Alignatt策略应用估算模型的Alignatt策略来提高增量TT。结果表明，我们升级的TTS模块有助于改善系统性能。

### Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0 
[[arxiv](https://arxiv.org/abs/2407.03005)] [[cool](https://papers.cool/arxiv/2407.03005)] [[pdf](https://arxiv.org/pdf/2407.03005)]
> **Authors**: Marianne de Heer Kloots,Willem Zuidema
> **First submission**: 2024-07-03
> **First announcement**: 2024-07-04
> **comment**: Accepted to Interspeech 2024. For code and materials, see https://github.com/mdhk/phonotactic-sensitivity
- **标题**: 神经语音模型中的人类语言偏见：wav2vec2.0中的语音分类和语音限制
- **领域**: 计算语言学,人工智能,声音,音频和语音处理
- **摘要**: 深度神经语音模型对语音学有什么了解？现有工作研究了这些模型中单个语言单元（例如音素）的编码。在这里，我们研究单元之间的互动。受到人类语音感知的经典实验的启发，我们研究了WAV2VEC2如何解决音量约束。我们在 /l /和 /r /之间的声学​​连续体上合成声音，并将它们嵌入仅 /l /，仅 /r /r /的受控上下文中，或者都不出现在英语中。像人类一样，WAV2VEC2模型在处理这种模棱两可的声音时对语音上可录的类别有偏见。使用简单的措施来分析单个刺激水平上的模型内部设备，我们发现这种偏差出现在模型变压器模块的早期层中。 ASR芬太尼会放大这种效果，但也存在于完全自我监管的模型中。我们的方法表明，受控刺激设计如何​​帮助将特定的语言知识定位在神经语音模型中。

### Textless Dependency Parsing by Labeled Sequence Prediction 
[[arxiv](https://arxiv.org/abs/2407.10118)] [[cool](https://papers.cool/arxiv/2407.10118)] [[pdf](https://arxiv.org/pdf/2407.10118)]
> **Authors**: Shunsuke Kando,Yusuke Miyao,Jason Naradowsky,Shinnosuke Takamichi
> **First submission**: 2024-07-14
> **First announcement**: 2024-07-15
> **comment**: Accepted to Interspeech 2024
- **标题**: 通过标记的序列预测来解析无文本依赖性
- **领域**: 计算语言学
- **摘要**: 传统的口头语言处理涉及将自动语音识别（ASR）系统级联到文本处理模型中。相比之下，“无文本”方法处理没有ASR系统的语音表示形式，从而可以直接使用声学语音特征。尽管它们的有效性在捕获声学特征时表现出来，但在捕获词汇知识时尚不清楚。本文提出了一种无文化的方法，用于依赖解析，检查其有效性和局限性。我们提出的方法可预测来自语音信号的依赖树，而无需转录，将树代表为标记的序列。在整体解析精度中，刺激方法的表现优于无文本方法，在具有重要的声学特征的情况下，后者出色。我们的发现突出了融合单词级表示和句子级韵律的重要性，以增强解析性能。代码和模型可公开可用：https：//github.com/mynlp/speechparser。

### dMel: Speech Tokenization made Simple 
[[arxiv](https://arxiv.org/abs/2407.15835)] [[cool](https://papers.cool/arxiv/2407.15835)] [[pdf](https://arxiv.org/pdf/2407.15835)]
> **Authors**: He Bai,Tatiana Likhomanenko,Ruixiang Zhang,Zijin Gu,Zakaria Aldeneh,Navdeep Jaitly
> **First submission**: 2024-07-22
> **First announcement**: 2024-07-23
> **comment**: under review
- **标题**: DMEL：语音令牌化变得简单
- **领域**: 计算语言学,人工智能,声音,音频和语音处理
- **摘要**: 大型语言模型通过利用自我监督的大量文本数据预处理来彻底改变了自然语言处理。受到这一成功的启发，研究人员研究了复杂的语音令牌化方法，以离散连续的语音信号，以便将语言建模技术应用于语音数据。但是，现有方法要么模型语义（内容）令牌，可能会丢失声学信息或模型声音令牌，从而冒着语义（内容）信息丢失的风险。具有多种令牌类型也使体系结构复杂化，需要额外的预处理。在这里，我们表明将MEL滤波器通道转化为离散强度箱会产生一个简单的表示（DMEL），该表示的性能比其他现有的语音令牌化方法更好。使用LM风格的变压器体系结构进行语音文本建模，我们全面评估了语音识别（ASR）和语音合成（TTS）的不同语音令牌化方法。我们的结果表明，DMEL在统一框架内在这两个任务上实现高性能的有效性，为语音和文本的有效和有效的联合建模铺平了道路。

### The Development of a Comprehensive Spanish Dictionary for Phonetic and Lexical Tagging in Socio-phonetic Research (ESPADA) 
[[arxiv](https://arxiv.org/abs/2407.15375)] [[cool](https://papers.cool/arxiv/2407.15375)] [[pdf](https://arxiv.org/pdf/2407.15375)]
> **Authors**: Simon Gonzalez
> **First submission**: 2024-07-22
> **First announcement**: 2024-07-23
> **comment**: Proceedings of the 16th Linguistic Annotation Workshop (LAW-XVI) within LREC2022
- **标题**: 在社会表达研究中的语音和词汇标记（ESPADA）中的全面的西班牙语词典开发（ESPADA）
- **领域**: 计算语言学
- **摘要**: 发音词典是语音强制对齐过程中的重要组成部分。这些词典的准确性对对齐的语音数据有很强的影响，因为它们有助于拼写和声学信号之间的映射。在本文中，我介绍了以西班牙语（ESPADA）为单词的综合发音词典的创建，该字典可用于西班牙数据的大多数方言变体。当前的词典集中在特定的区域变体上，但是凭借我们工具的灵活性，它可以很容易地用于捕获主要方言变体中最常见的语音差异。我们建议改进当前的发音词典，并绘制其他相关注释，例如形态和词汇信息。就规模而言，它是目前最完整的词典，其中有超过628,000个条目，代表来自16个国家 /地区的单词。所有条目均带有相应的发音，形态学和词汇标记以及其他相关信息，以进行语音分析：应力模式，音调，IPA转录等。这旨在为社会表达研究人员提供完整的开源工具，从而增强西班牙语中社会形式框架内的方言研究。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 4 篇论文

### SOAF: Scene Occlusion-aware Neural Acoustic Field 
[[arxiv](https://arxiv.org/abs/2407.02264)] [[cool](https://papers.cool/arxiv/2407.02264)] [[pdf](https://arxiv.org/pdf/2407.02264)]
> **Authors**: Huiyu Gao,Jiahao Ma,David Ahmedt-Aristizabal,Chuong Nguyen,Miaomiao Liu
> **First submission**: 2024-07-02
> **First announcement**: 2024-07-03
> **comment**: No comments
- **标题**: SOAF：场景闭塞感知神经声学领域
- **领域**: 计算机视觉和模式识别,声音,音频和语音处理
- **摘要**: 本文鉴于来自其他已知的现场轨迹的音频录音，沿室内场景中的任意轨迹沿任意轨迹沿着任意轨迹解决了新颖的视听构成问题。现有方法通常忽略了房间几何形状的效果，尤其是墙壁遮挡声音传播，从而使它们在多房间环境中的准确性降低。在这项工作中，我们提出了一种新的方法，称为场景闭塞声学场（SOAF），以进行准确的声音生成。我们的方法使用远距离感知的参数声音传播建模得出了声音能场的先验，然后根据从输入视频中学到的场景传播进行了转换。我们使用斐波那契球体从围绕接收器的局部声场中提取特征，以产生双耳音频，以使用方向感知的注意机制来产生新型视图。在真实数据集RWAV和合成数据集Soundspaces上进行的广泛实验表明，我们的方法在音频生成中的表现优于先前最先进的技术。项目页面：https：//github.com/huiyu-gao/soaf/。

### A Deep Learning Framework for Three Dimensional Shape Reconstruction from Phaseless Acoustic Scattering Far-field Data 
[[arxiv](https://arxiv.org/abs/2407.09525)] [[cool](https://papers.cool/arxiv/2407.09525)] [[pdf](https://arxiv.org/pdf/2407.09525)]
> **Authors**: Doga Dikbayir,Abdel Alsnayyan,Vishnu Naresh Boddeti,Balasubramaniam Shanker,Hasan Metin Aktulga
> **First submission**: 2024-06-24
> **First announcement**: 2024-07-12
> **comment**: 13 pages, 14 Figures
- **标题**: 从无相相散射远场数据的三维形状重建的深度学习框架
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 反向散射问题在许多领域中至关重要，包括医学成像，声纳，传感，无损评估等。感兴趣的问题可能从检测形状到障碍物的本构特性不等。两者的挑战在于，当信息有限时，这个问题是不适合的。也就是说，多年来，在为解决这个问题的解决方案开发解决方案方面已经花费了巨大的努力。在这里，我们使用了一种不同的方法，该方法建立在数据上。具体而言，我们使用有限的信息，单个事件波，单频率和无相位远场数据开发了一个深度学习框架，以用于形状重建。这是通过（a）使用由3D变异自动编码器学习的紧凑概率的潜在潜在空间和（b）训练有素的卷积神经网络，该卷积神经网络将声学散射信息映射到此形状表示。在合成的3D粒子数据集以及Shapenet（一种流行的3D形状识别数据集）上评估了所提出的框架。如通过多个结果所证明的那样，尽管数据中存在显着变化，但提出的方法能够为大批次的复杂散射器形状（例如飞机和汽车）产生准确的重建。

### Pose Estimation from Camera Images for Underwater Inspection 
[[arxiv](https://arxiv.org/abs/2407.16961)] [[cool](https://papers.cool/arxiv/2407.16961)] [[pdf](https://arxiv.org/pdf/2407.16961)]
> **Authors**: Luyuan Peng,Hari Vishnu,Mandar Chitre,Yuen Min Too,Bharath Kalyan,Rajat Mishra,Soo Pieng Tan
> **First submission**: 2024-07-23
> **First announcement**: 2024-07-24
> **comment**: Submitted to IEEE Journal of Oceanic Engineering
- **标题**: 从相机图像进行水下检查的姿势估算
- **领域**: 计算机视觉和模式识别,机器人技术,图像和视频处理
- **摘要**: 高精度定位在水下再调查任务中至关重要。传统的本地化方法，例如惯性导航系统，多普勒速度记录仪和声学定位，面临重大挑战，对于某些应用而言并不具有成本效益。在这种情况下，视觉定位是一种具有成本效益的替代方法，利用已经配备了检查车辆的相机从周围场景的图像中估算出姿势。在这些情况下，从图像中的基于机器学习的姿势估计显示了水下环境中的希望，使用基于先前映射的场景训练的模型进行有效的重新定位。我们探讨了基于学习的姿势估计器在清晰和浑浊的水检查任务中的功效，评估了图像格式，模型架构和培训数据多样性的影响。我们通过采用新型视图合成模型来生成增强的训练数据，从而显着增强未开发区域的姿势估计。此外，我们通过通过扩展的Kalman滤波器将姿势估计器输出与传感器数据集成在一起，从而提高了定位精度，从而证明了轨迹的平滑度和准确性提高。

### MicroEmo: Time-Sensitive Multimodal Emotion Recognition with Micro-Expression Dynamics in Video Dialogues 
[[arxiv](https://arxiv.org/abs/2407.16552)] [[cool](https://papers.cool/arxiv/2407.16552)] [[pdf](https://arxiv.org/pdf/2407.16552)]
> **Authors**: Liyun Zhang
> **First submission**: 2024-07-23
> **First announcement**: 2024-07-24
> **comment**: No comments
- **标题**: Microemo：在视频对话中，具有微表达动态的时间敏感的多模式情绪识别
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 多模式的大语言模型（MLLM）表现出了非凡的多模式识别能力，在视频中将视觉，声学和语言环境中的多模式线索整合在一起，以识别人类的情感状态。但是，现有的方法忽略了捕获微表达时间动态的局部面部特征，而不会利用视频中听众感知到的时间段的上下文依赖性，从而在一定程度上限制了他们的期望有效性。在这项工作中，我们提出了Microemo，这是一种时间敏感的MLLM，旨在将注意力集中在本地面部微表达动力学和语音感知视频片段的上下文依赖性。我们的模型结合了两个关键的架构贡献：（1）全局本地注意的视觉编码器，将全局框架级时板的图像特征与微表达时间动力学的局部面部特征集成在一起； （2）通过为每个话语段以及整个视频生成视觉令牌序列，然后将它们结合在一起，可以通过生成视觉令牌序列来捕获多尺度和上下文依赖性的视频Q形式。初步定性实验表明，在新的可解释的多模式情感识别（EMER）任务中，利用多模式和多方面的线索来预测以开放式摄氏（OV）方式预测情绪，Microemo与最新方法相比证明了其有效性。

## 机器学习(cs.LG:Machine Learning)

该领域共有 2 篇论文

### Modality-Order Matters! A Novel Hierarchical Feature Fusion Method for CoSAm: A Code-Switched Autism Corpus 
[[arxiv](https://arxiv.org/abs/2407.14328)] [[cool](https://papers.cool/arxiv/2407.14328)] [[pdf](https://arxiv.org/pdf/2407.14328)]
> **Authors**: Mohd Mujtaba Akhtar,Girish,Muskaan Singh,Orchid Chetia Phukan
> **First submission**: 2024-07-19
> **First announcement**: 2024-07-22
> **comment**: No comments
- **标题**: 模式 - 订购事项！ COSAM的一种新颖的分层功能融合方法：一种代码开关的自闭症语料库
- **领域**: 机器学习
- **摘要**: 自闭症谱系障碍（ASD）是一项复杂的神经发展挑战，在不同情况下，社会互动，交流和重复行为的表达中遇到了一系列困难。这种越来越多的患病率强调了ASD作为主要的公共卫生关注的重要性，并且需要对综合研究计划提高我们对疾病及其早期检测方法的理解。这项研究介绍了一种新型的分层特征融合方法，旨在通过分析代码切换的语音（英语和印地语）来增强儿童对ASD的早期检测。该研究采用先进的音频处理技术，使用变压器编码器整合了声学，副语言和语言信息。这种创新的融合策略旨在提高分类鲁棒性和准确性，对于早期和精确的ASD识别至关重要。该方法涉及从被诊断为ASD和匹配的对照组的儿童那里收集代码开关语料库COSAM。该数据集包含30名诊断为ASD的儿童的61次语音录音和3至13岁之间的神经型儿童的31个儿童，总共有159.75分钟的语音录音。该功能分析侧重于MFCC和广泛的统计属性，以捕获语音模式变异性和复杂性。最佳的模型性能是使用层次融合技术实现的，其精度为98.75％，首先是声学和语言特征的组合，然后以层次结构的方式进行副语言特征。

### Neural Networks for Generating Better Local Optima in Topology Optimization 
[[arxiv](https://arxiv.org/abs/2407.17957)] [[cool](https://papers.cool/arxiv/2407.17957)] [[pdf](https://arxiv.org/pdf/2407.17957)]
> **Authors**: Leon Herrmann,Ole Sigmund,Viola Muning Li,Christian Vogl,Stefan Kollmannsberger
> **First submission**: 2024-07-25
> **First announcement**: 2024-07-26
> **comment**: No comments
- **标题**: 神经网络，用于在拓扑优化方面产生更好的本地最优
- **领域**: 机器学习
- **摘要**: 最近，神经网络被用作伴随优化框架内的材料离散框架，用于反问题和拓扑优化。尽管在某些反问题上发现了有利的正则效果和更好的最佳选择，但拓扑优化的好处是有限的 - 调查的重点是合规性问题。我们证明了神经网络材料在某些条件下如何在更具挑战性的优化问题中找到更好的局部最佳选择，在这里我们特别考虑声学拓扑优化。通过运行具有不同神经网络初始化的多个部分优化，可以显着提高确定更好最佳的机会。此外，我们表明，神经网络材料离散化的优势来自与Adam Optimizer的相互作用，并在与受约束和高阶优化技术竞争时强调其当前局限性。目前，这种离散化仅被证明对不受限制的一阶优化有益。

## 机器人技术(cs.RO:Robotics)

该领域共有 1 篇论文

### Missile detection and destruction robot using detection algorithm 
[[arxiv](https://arxiv.org/abs/2407.07452)] [[cool](https://papers.cool/arxiv/2407.07452)] [[pdf](https://arxiv.org/pdf/2407.07452)]
> **Authors**: Md Kamrul Siam,Shafayet Ahmed,Md Habibur Rahman,Amir Hossain Mollah
> **First submission**: 2024-07-10
> **First announcement**: 2024-07-11
> **comment**: 67 pages
- **标题**: 使用检测算法的导弹检测和破坏机器人
- **领域**: 机器人技术,人工智能
- **摘要**: 这项研究基于世界上当前的导弹检测技术以及对这些技术的分析，以找到在孟加拉国实施该系统的经济有效解决方案。本文将使用电流传感器和脉冲多普勒雷达给出导弹检测技术的想法。制作系统以检测目标导弹。在超声波声纳，金属探测器传感器和烟雾探测器传感器的帮助下，自动检测和破坏。该系统主要基于超声波声纳传感器。它具有传感器，发射器和接收器。传感器与控制器连接的连接。当它通过遵循算法检测对象时，它会找到其距离和角度。它还可以通过使用其他算法的模拟来确保系统是否可以破坏对象。

## 声音(cs.SD:Sound)

该领域共有 11 篇论文

### Speaker- and Text-Independent Estimation of Articulatory Movements and Phoneme Alignments from Speech 
[[arxiv](https://arxiv.org/abs/2407.03132)] [[cool](https://papers.cool/arxiv/2407.03132)] [[pdf](https://arxiv.org/pdf/2407.03132)]
> **Authors**: Tobias Weise,Philipp Klumpp,Kubilay Can Demir,Paula Andrea Pérez-Toro,Maria Schuster,Elmar Noeth,Bjoern Heismann,Andreas Maier,Seung Hee Yang
> **First submission**: 2024-07-03
> **First announcement**: 2024-07-04
> **comment**: to be published in Interspeech 2024 proceedings
- **标题**: 言语运动和语音对齐的扬声器和文本无关的估计
- **领域**: 声音,人工智能,计算语言学,机器学习,音频和语音处理
- **摘要**: 本文介绍了两个任务的新组合，以前是单独处理的：声学到发音的语音反演（AAI）和音素到发音（PTA）运动估计。我们将这一联合任务称为声音到发音的语音反演（APTAI），并在推断期间探索了两种不同的方法，无论是在推理过程中与众不同的说话者和文本独立的。我们使用多任务学习设置，其端到端的目标是将原始语音作为输入并估算相应的关节运动，音素序列和音素对齐。尽管两种建议的方法都共享了这些相同的要求，但它们在实现音素相关预测的方式方面有所不同：一种是基于框架分类，另一个基于两期训练程序和强迫对齐。与最先进的文本依赖性音素力对准器相比，我们达到AAI任务的竞争性能为0.73平均相关性，并达到大约87％的帧重叠。

### A Toolchain for Comprehensive Audio/Video Analysis Using Deep Learning Based Multimodal Approach (A use case of riot or violent context detection) 
[[arxiv](https://arxiv.org/abs/2407.03110)] [[cool](https://papers.cool/arxiv/2407.03110)] [[pdf](https://arxiv.org/pdf/2407.03110)]
> **Authors**: Lam Pham,Phat Lam,Tin Nguyen,Hieu Tang,Alexander Schindler
> **First submission**: 2024-05-02
> **First announcement**: 2024-07-04
> **comment**: No comments
- **标题**: 使用基于深度学习的多模式方法（暴动或暴力环境检测的用例），用于全面音频/视频分析的工具链
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在本文中，我们通过利用基于深度学习的多模式方法提出了一个工具链，以进行全面的音频/视频分析。为此，对文本（S2T），声学场景分类（ASC），声学事件检测（AED），视觉对象检测（VOD），图像字幕（IC）和视频字幕（VC）进行了不同的特定语音任务。通过组合单个任务并分析从输入视频中提取的音频\和视觉数据，该工具链提供了各种基于音频/视频的应用程序：音频/视频聚类的两个一般应用，全面的音频/视频摘要以及RIOT或暴力上下文检测的特定应用。此外，该工具链提供了一种灵活且适应性的体系结构，该体系结构有效地集成了新的模型，以进行进一步的音频/视频应用程序。

### Real-time Timbre Remapping with Differentiable DSP 
[[arxiv](https://arxiv.org/abs/2407.04547)] [[cool](https://papers.cool/arxiv/2407.04547)] [[pdf](https://arxiv.org/pdf/2407.04547)]
> **Authors**: Jordie Shier,Charalampos Saitis,Andrew Robertson,Andrew McPherson
> **First submission**: 2024-07-05
> **First announcement**: 2024-07-08
> **comment**: Accepted for publication at the 24th International Conference on New Interfaces for Musical Expression in Utrecht, Netherlands
- **标题**: 实时音色与可区分的DSP重新映射
- **领域**: 声音,人工智能,机器学习,音频和语音处理,信号处理
- **摘要**: 音色是在各种音乐环境中表达的主要方式。然而，普遍的音频驱动的合成方法主要依赖于音高和响度信封，从而有效地从输入中表达了音色的表达。我们的方法借鉴了音色类比的概念，并研究了如何将来自输入信号的Timbral表达映射到控制器上以进行合成器。利用可不同的数字信号处理，我们的方法通过新的特征差损失来促进合成参数的直接优化。这种损失函数旨在学习音乐事件之间的相对时间差异，优先考虑短语中分级音色调制的微妙，从而可以在音色空间中进行有意义的翻译。使用SNARE鼓表演作为案例研究，其中Timbral表达是中心的，我们演示了实时的音色从声学圈圈鼓到以Roland TR-808为模型的可区分合成器。

### Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models 
[[arxiv](https://arxiv.org/abs/2407.04482)] [[cool](https://papers.cool/arxiv/2407.04482)] [[pdf](https://arxiv.org/pdf/2407.04482)]
> **Authors**: Vyas Raina,Mark Gales
> **First submission**: 2024-07-05
> **First announcement**: 2024-07-08
> **comment**: No comments
- **标题**: 控制耳语：通用声学对抗攻击以控制语音基础模型
- **领域**: 声音,计算语言学,音频和语音处理
- **摘要**: 具有灵活的基于语音识别系统的形式或音频宣传的大语言模型（LLMS）的形式，启用语音的基础模型正在变得越来越流行。这些模型的有趣方面之一是他们使用适当的提示可以执行自动语音识别（ASR）以外的任务的能力。例如，Openai Whisper模型可以同时执行语音转录和语音翻译。随着音频宣传的LLM的发展，有可能获得更大的控制选择。在这项工作中，我们证明，有了更大的灵活性，系统可能会受到模型控制对抗攻击的影响。无需任何访问模型提示即可通过适当更改音频输入来修改系统的行为。为了说明这种风险，我们证明可以将一个简短的通用对抗性声段预先到任何输入语音信号，以覆盖ASR基础模型的及时设置。具体而言，尽管设置进行语音转录，但我们成功地使用通用的对抗性声段来控制耳语，以始终执行语音翻译。总体而言，这项工作展示了对启用多任务语音的基础模型的一种新形式的对抗性攻击，在部署这种模型形式之前，需要考虑这些模型。

### Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion 
[[arxiv](https://arxiv.org/abs/2407.10373)] [[cool](https://papers.cool/arxiv/2407.10373)] [[pdf](https://arxiv.org/pdf/2407.10373)]
> **Authors**: Jian Ma,Wenguan Wang,Yi Yang,Feng Zheng
> **First submission**: 2024-07-14
> **First announcement**: 2024-07-15
> **comment**: ECCV 2024; Project page: https://hechang25.github.io/MVSD
- **标题**: 通过视觉场景驱动的扩散进行声学匹配和冲突的相互学习
- **领域**: 声音,人工智能,计算机视觉和模式识别,音频和语音处理
- **摘要**: 视力匹配（VAM）对于增强身临其境的体验至关重要，而缩放的任务有效地提高了音频的清晰度。现有方法独立处理每个任务，忽略了它们之间的固有互惠。此外，这些方法取决于配对的培训数据，这是一项挑战，可以阻碍广泛的未配对数据的利用。在本文中，我们介绍了MVSD，这是一个基于扩散模型的相互学习框架。 MVSD对称地考虑了这两个任务，从而利用了相互关系以促进从逆任务中学习并克服数据稀缺性。此外，我们采用扩散模型作为基础条件转换器来规避训练的不稳定性和传统gan体系结构的过度平滑缺陷。具体而言，MVSD使用两个转换器：一个用于VAM的Reverberator，一个用于Dereverberation，称为Dereverberator。遗迹判断混响者产生的混响音频是否听起来像处于有条件的视觉场景，反之亦然。通过形成封闭循环，这两个转换器可以生成信息丰富的反馈信号以优化逆任务，即使使用易于获得的单向不成对数据。在两个标准基准测试（即Soundspaces语音和声学上的Avspeech）上进行了广泛的实验，表明我们的框架可以改善混响器和替补架的性能，并更好地匹配指定的视觉场景。

### Pre-Trained Foundation Model representations to uncover Breathing patterns in Speech 
[[arxiv](https://arxiv.org/abs/2407.13035)] [[cool](https://papers.cool/arxiv/2407.13035)] [[pdf](https://arxiv.org/pdf/2407.13035)]
> **Authors**: Vikramjit Mitra,Anirban Chatterjee,Ke Zhai,Helen Weng,Ayuko Hill,Nicole Hay,Christopher Webb,Jamie Cheng,Erdrin Azemi
> **First submission**: 2024-07-17
> **First announcement**: 2024-07-18
> **comment**: 8 pages, 6 figures, BioKDD workshop paper
- **标题**: 预先训练的基础模型表示，以发现语音中的呼吸模式
- **领域**: 声音,计算语言学,机器学习,音频和语音处理
- **摘要**: 人类言语产生的过程涉及协调的呼吸作用，以引起声学信号。通常，当空气被强迫从肺部强迫时会产生语音，并由声带调节，在这种行动中，这种动作被空气中的呼吸时刻（吸入）散布在散布着肺部（吸入）再次重新填充肺部。呼吸率（RR）是一个重要的指标，用于评估个人的整体健康，适应性和一般福祉。现有的方法测量RR（一分钟一分钟的呼吸次数）是使用专用设备或培训进行的。研究表明，机器学习算法可用于使用生物传感器信号作为输入来估计RR。基于语音的RR估计可以提供一种有效的方法来测量重要指标，而无需任何专门的设备或传感器。这项工作调查了一种基于机器学习的方法，以从与近语麦克风设备的受试者获得的语音段估算RR。从n = 26个个体收集数据，在该个体中，通过商业级胸皮获得了地面RR，然后手动校正任何错误。提出了卷积长期术语记忆网络（CORV-LSTM），以估算语音信号的呼吸时间序列数据。我们证明，与基线相比，可以使用从wav2Vec2等基础模型（例如WAV2VEC2）获得的使用预训练的表示形式（例如WAV2VEC2）估计呼吸均值误差和高相关系数。模型驱动的时间序列可用于估计$ RR $，其平均绝对错误（MAE）〜1.6呼吸/分钟。

### Underwater Acoustic Signal Denoising Algorithms: A Survey of the State-of-the-art 
[[arxiv](https://arxiv.org/abs/2407.13264)] [[cool](https://papers.cool/arxiv/2407.13264)] [[pdf](https://arxiv.org/pdf/2407.13264)]
> **Authors**: Ruobin Gao,Maohan Liang,Heng Dong,Xuewen Luo,P. N. Suganthan
> **First submission**: 2024-07-18
> **First announcement**: 2024-07-19
> **comment**: No comments
- **标题**: 水下声学信号降级算法：最先进的调查
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 本文全面回顾了水下声学信号的最新进展，这对于提高水下通信和监测系统的可靠性和清晰度至关重要。尽管该领域取得了重大进展，但水下环境的复杂性质却带来了使转化过程复杂化的独特挑战。我们首先概述与水下声信号处理相关的基本挑战，包括信号衰减，噪声变异性以及环境因素的影响。然后，该综述系统地对并讨论了各种脱氧算法，例如常规，基于分解和基于学习的技术，突出了它们的应用，优势和局限性。还审查了评估指标和实验数据集。本文以未来研究方向的开头问题和建议列表结束，强调需要开发更强大的剥离技术，以适应动态的水下声学环境。

### On the Utility of Speech and Audio Foundation Models for Marmoset Call Analysis 
[[arxiv](https://arxiv.org/abs/2407.16417)] [[cool](https://papers.cool/arxiv/2407.16417)] [[pdf](https://arxiv.org/pdf/2407.16417)]
> **Authors**: Eklavya Sarkar,Mathew Magimai. -Doss
> **First submission**: 2024-07-23
> **First announcement**: 2024-07-24
> **comment**: Accepted at Interspeech 2024 satellite event (VIHAR 2024)
- **标题**: 关于语音和音频基础模型的效用，用于摩尔马塞特通话分析
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: Marmoset Monkeys在他们的呼叫中编码重要信息，并作为神经生物学家了解人声交流的进化起源的替代模型。传统上，通过基于信号处理的特征进行分析，最近的方法利用了在人类语音中预先培训的自我监督模型，以提取特征，从而利用了它们独立于其声学领域学习信号的内在结构的能力。但是，此类基础模型的实用性在多级分类，带宽和训练前域而言，对于Marmoset呼叫分析的效用仍不清楚。这项研究评估了来自语音和一般音频域的特征表示，跨越4、8和16 kHz的训练带宽，用于玛格丽特呼叫类型和呼叫者分类任务。结果表明，带宽较高的模型可改善性能，并在语音或一般音频上进行预训练可相当的结果，从而改善光谱基线。

### Model-driven Heart Rate Estimation and Heart Murmur Detection based on Phonocardiogram 
[[arxiv](https://arxiv.org/abs/2407.18424)] [[cool](https://papers.cool/arxiv/2407.18424)] [[pdf](https://arxiv.org/pdf/2407.18424)]
> **Authors**: Jingping Nie,Ran Liu,Behrooz Mahasseni,Erdrin Azemi,Vikramjit Mitra
> **First submission**: 2024-07-25
> **First announcement**: 2024-07-26
> **comment**: 6 pages, 10 figures
- **标题**: 模型驱动的心率估计和基于声音图的心脏杂音检测
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: 声学信号对于健康监测至关重要，尤其是心脏声音，它们提供了基本数据，例如心率和检测心脏异常（例如杂音）。这项研究利用公开可用的Phonocartiogram（PCG）数据集使用模型驱动的方法来估算心率，并将表现最佳的模型扩展到同时心率估计和杂音检测的多任务学习（MTL）框架（MTL）框架。心率估计是使用在心脏声音片段上的滑动窗口技术得出的，并结合了声学特征（MEL频谱图，sepstral系数，功率谱密度，均方根能量）的组合。我们的发现表明，2D卷积神经网络（\ textbf {\ texttt {2dcnn}}）对于心率估计最有效，达到1.312 bpm的平均绝对误差（MAE）。我们系统地研究了不同特征组合的影响，发现利用所有四个功能都会产生最佳结果。 MTL模型（\ textbf {\ texttt {2dcnn-mtl}}）在杂音检测中实现了超过95％的准确性，超过了现有模型，同时保持了1.636 bpm的心率估计，并满足了医疗仪器（AAMI AAMI）的协会所指出的要求。

### Innovative Speech-Based Deep Learning Approaches for Parkinson's Disease Classification: A Systematic Review 
[[arxiv](https://arxiv.org/abs/2407.17844)] [[cool](https://papers.cool/arxiv/2407.17844)] [[pdf](https://arxiv.org/pdf/2407.17844)]
> **Authors**: Lisanne van Gelderen,Cristian Tejedor-García
> **First submission**: 2024-07-25
> **First announcement**: 2024-07-26
> **comment**: van Gelderen, L., & Tejedor-García, C. (2024). Innovative Speech-Based Deep Learning Approaches for Parkinson's DiseaseClassification: A Systematic Review. Applied Sciences, 14(17). doi:10.3390/app14177873 This research was funded by the NWO research programme NGF AiNed Fellowship Grants under the project Responsible AI for Voice Diagnostics (RAIVD) - grant number NGF.1607.22.013
- **标题**: 帕金森氏病分类的创新性言语深度学习方法：系统评价
- **领域**: 声音,人工智能,计算语言学,机器学习,音频和语音处理
- **摘要**: 帕金森氏病（PD）是全世界第二大流行的神经退行性疾病，经常出现早期的言语障碍。通过分析语音数据，人工智能（AI）的最新进展（AI），尤其是深度学习（DL），已显着增强了PD诊断。然而，研究的进展受到公开访问的基于语音的PD数据集有限的限制，这主要是由于隐私问题所致。这项系统评价的目的是根据2020年1月至2024年3月之间发表的33项科学作品探索当前基于语音的DL方法的景观。我们讨论了他们的可用资源，能力，潜在的局限性以及与偏见，解释性和隐私相关的问题。此外，这篇综述提供了有关PD的公共访问数据集和开源材料的概述。确定的DL方法分为端到端（E2E）学习，转移学习（TL）和深度声学特征提取（DAFE）。在E2E方法中，卷积神经网络（CNN）很普遍，尽管变压器越来越受欢迎。 E2E方法面临诸如有限的数据和计算资源之类的挑战，尤其是在变压器的挑战中。 TL通过提供更强大的PD诊断和跨语言的更好的推广性来解决这些问题。 DAFE旨在通过检查深度特征对其他DL方法和更传统的机器学习（ML）方法的特定影响来提高结果的可解释性和解释性。但是，与E2E和TL方法相比，它通常表现不佳。

### SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection 
[[arxiv](https://arxiv.org/abs/2407.18517)] [[cool](https://papers.cool/arxiv/2407.18517)] [[pdf](https://arxiv.org/pdf/2407.18517)]
> **Authors**: Yi Zhu,Surya Koppisetti,Trang Tran,Gaurav Bharaj
> **First submission**: 2024-07-26
> **First announcement**: 2024-07-29
> **comment**: No comments
- **标题**: Slim：通用音频深盘检测的样式语言学不匹配模型
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 音频深击检测（ADD）对于打击生成AI模型合成的语音滥用至关重要。现有的添加模型遭受了概括性问题的困扰，并且在域内和室外数据之间存在巨大的性能差异。此外，现有模型的黑框性质限制了它们在现实情况下的使用，在现实世界中，模型决策需要解释。为了减轻这些问题，我们引入了一个新的添加模型，该模型在假语音中明确使用样式语言学不匹配（Slim），以将它们与真实的语音分开。 Slim首先仅对实际样本进行自我监督的预处理，以学习真实阶级的样式语言依赖性。然后，学习的功能与标准预处理的声学特征（例如WAV2VEC）一起使用，以学习真实和假班级的分类器。当功能编码器被冷冻时，Slim在室外数据集上的基准方法优于基准方法，同时在内域数据上获得竞争结果。 Slim学到的功能使我们能够量化样本中样式和语言内容之间的（MIS）匹配，从而促进了模型决策的解释。

## 音频和语音处理(eess.AS:Audio and Speech Processing)

该领域共有 3 篇论文

### Vibravox: A Dataset of French Speech Captured with Body-conduction Audio Sensors 
[[arxiv](https://arxiv.org/abs/2407.11828)] [[cool](https://papers.cool/arxiv/2407.11828)] [[pdf](https://arxiv.org/pdf/2407.11828)]
> **Authors**: Julien Hauret,Malo Olivier,Thomas Joubaud,Christophe Langrenne,Sarah Poirée,Véronique Zimpfer,Éric Bavu
> **First submission**: 2024-07-16
> **First announcement**: 2024-07-17
> **comment**: 23 pages, 42 figures
- **标题**: Vibravox：由人体传导音频传感器捕获的法国演讲数据集
- **领域**: 音频和语音处理,机器学习
- **摘要**: Vibravox是一个数据集，该数据集符合使用五个不同的身体传导音频传感器的通用数据保护调节（GDPR）（GDPR）：两个内耳麦克风，两个骨传导振动拾音器和一个喉头。该数据集还包括来自用作参考的机载麦克风的音频数据。 Vibravox语料库包含45小时的语音样本和生理声音，由188名参与者记录在高级Ambisonics 3D Spatializer施加的不同声学条件下。关于记录条件和语言转录的注释也包括在语料库中。我们对各种与语音有关的任务进行了一系列实验，包括语音识别，言语增强和扬声器验证。这些实验是使用最先进的模型进行的，以评估和比较颤音数据集提供的不同音频传感器捕获的信号的性能，以便更好地掌握其个体特征。

### Self-supervised ASR Models and Features For Dysarthric and Elderly Speech Recognition 
[[arxiv](https://arxiv.org/abs/2407.13782)] [[cool](https://papers.cool/arxiv/2407.13782)] [[pdf](https://arxiv.org/pdf/2407.13782)]
> **Authors**: Shujie Hu,Xurong Xie,Mengzhe Geng,Zengrui Jin,Jiajun Deng,Guinan Li,Yi Wang,Mingyu Cui,Tianzi Wang,Helen Meng,Xunying Liu
> **First submission**: 2024-07-03
> **First announcement**: 2024-07-19
> **comment**: IEEE/ACM Transactions on Audio, Speech, and Language Processing
- **标题**: 自我监督的ASR模型和违反障碍和老年语音识别的功能
- **领域**: 音频和语音处理,人工智能,声音
- **摘要**: 基于自我监督的学习（SSL）的语音基础模型已应用于广泛的ASR任务。但是，它们通过数据密集型参数微调应用于违反障碍和老年人语音的应用面临着域中数据稀缺性和不匹配。为此，本文探讨了一系列将域的精细SSL预先训练模型及其特征整合到TDNN和构象ASR系统中，以构造违规和老年人的语音识别。其中包括：a）标准声前端和域微调SSL语音表示之间的输入特征融合； b）单独使用标准声学特征和具有其他域微调SSL特征的TDNN系统之间的框架级关节解码； c）涉及将使用域微调预训练的ASR模型委员的TDNN/构象系统输出的多通解码。此外，微调的SSL语音特征用于构造多模式ASR系统的声学到发音（A2A）反转。实验是针对四个任务进行的：英语Uapseech和Torgo违反语音语音；以及英国痴呆症皮特（Pitt）和粤语JCCOCC MOCA老年语音数据集。通过集成了域适应的Hubert，WAV2VEC2-CONFORMER或多语言XLSR模型及其功能始终优于独立的微型SSL预训练模型的TDNN系统。这些系统分别在四个任务上产生了统计学上的显着性或CER降低，绝对量为6.53％，1.90％，2.04％和7.97％（24.10％，23.84％，10.14％和31.39％）。还使用Dementiabank Pitt老年语音识别输出获得了阿尔茨海默氏病检测准确性的一致提高。

### Explaining Spectrograms in Machine Learning: A Study on Neural Networks for Speech Classification 
[[arxiv](https://arxiv.org/abs/2407.17416)] [[cool](https://papers.cool/arxiv/2407.17416)] [[pdf](https://arxiv.org/pdf/2407.17416)]
> **Authors**: Jesin James,Balamurali B. T.,Binu Abeysinghe,Junchen Liu
> **First submission**: 2024-07-10
> **First announcement**: 2024-07-24
> **comment**: 5th International Conference on Artificial Intelligence and Speech Technology (AIST-2023), New Delhi, India
- **标题**: 解释机器学习中的频谱图：关于语音分类的神经网络的研究
- **领域**: 音频和语音处理,人工智能,计算语言学
- **摘要**: 这项研究调查了神经网络学到的歧视模式，以进行准确的语音分类，并特别关注元音分类任务。通过检查神经网络的激活和特征以进行元音分类，我们可以洞悉网络在频谱图中“看到”的内容。通过使用类激活映射，我们确定了有助于元音分类的频率，并将这些发现与语言知识进行比较。在美国英语元音数据集上进行的实验展示了神经网络的解释性，并在将它们与未发音的语音区分开时，对错误分类及其特征的原因提供了宝贵的见解。这项研究不仅增强了我们对元音分类中潜在的声学提示的理解

## 数值分析(math.NA:Numerical Analysis)

该领域共有 1 篇论文

### Early Recognition of Parkinson's Disease Through Acoustic Analysis and Machine Learning 
[[arxiv](https://arxiv.org/abs/2407.16091)] [[cool](https://papers.cool/arxiv/2407.16091)] [[pdf](https://arxiv.org/pdf/2407.16091)]
> **Authors**: Niloofar Fadavi,Nazanin Fadavi
> **First submission**: 2024-07-22
> **First announcement**: 2024-07-23
> **comment**: N/A
- **标题**: 通过声学分析和机器学习对帕金森氏病的早期认识
- **领域**: 数值分析,机器学习,神经元和认知
- **摘要**: 帕金森氏病（PD）是一种进行性神经退行性疾病，可显着影响运动和非运动功能，包括语音。通过语音分析对PD的早期识别可以通过及时干预来大大提高患者的预后。本文对使用语音数据进行了对PD识别方法的全面综述，突出了机器学习和数据驱动方法的进步。我们讨论了数据争吵的过程，包括数据收集，清洁，转换和探索性数据分析，以准备机器学习应用程序的数据集。探索了各种分类算法，包括有和没有特征选择的逻辑回归，SVM和神经网络。根据准确性，精度和训练时间对每种方法进行评估。我们的发现表明，特定的声学特征和先进的机器学习技术可以有效地区分PD和健康对照的个体。该研究以比较不同模型的比较结束，确定了PD识别的最有效方法，并提出了未来研究的潜在方向。

## 其他论文

共有 23 篇其他论文

- [RealMAN: A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization](https://arxiv.org/abs/2406.19959)
  - **标题**: Realman：用于动态语音增强和本地化的实录和注释的麦克风阵列数据集
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Constant Directivity Loudspeaker Beamforming](https://arxiv.org/abs/2407.01860)
  - **标题**: 恒定方向性扬声器横梁成形
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [VAE-based Phoneme Alignment Using Gradient Annealing and SSL Acoustic Features](https://arxiv.org/abs/2407.02749)
  - **标题**: 使用梯度退火和SSL声学特征的基于VAE的音素对齐
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [A New Classification of Clustering-based for Different Problems in Different Wireless Ad-hoc Networks](https://arxiv.org/abs/2407.02498)
  - **标题**: 针对不同无线临时网络中不同问题的基于聚类的新分类
  - **Filtered Reason**: none of cs.NI,cs.DC in whitelist
- [Research on the Acoustic Emission Source Localization Methodology in Composite Materials based on Artificial Intelligence](https://arxiv.org/abs/2407.05405)
  - **标题**: 基于人工智能的复合材料中的声发射源定位方法的研究
  - **Filtered Reason**: none of physics.data-an,eess.AS,cs.SD in whitelist
- [Wireless teleoperation of HSURF artificial fish in complex paths](https://arxiv.org/abs/2407.05120)
  - **标题**: HSURF人造鱼类在复杂路径中的无线近距离
  - **Filtered Reason**: none of cs.RO in whitelist
- [WildDESED: An LLM-Powered Dataset for Wild Domestic Environment Sound Event Detection System](https://arxiv.org/abs/2407.03656)
  - **标题**: 野性：野生家庭环境声音事件检测系统的LLM驱动数据集
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Cervical Auscultation Machine Learning for Dysphagia Assessment](https://arxiv.org/abs/2407.05870)
  - **标题**: 宫颈诊断机器的吞咽困难评估
  - **Filtered Reason**: none of eess.AS,cs.HC,cs.SD in whitelist
- [Dirichlet process mixture model based on topologically augmented signal representation for clustering infant vocalizations](https://arxiv.org/abs/2407.05760)
  - **标题**: Dirichlet工艺混合模型基于拓扑增强的信号表示，用于聚集婴儿发声
  - **Filtered Reason**: none of eess.AS,cs.SD,stat.AP,stat.ML in whitelist
- [AIRA: A Low-cost IR-based Approach Towards Autonomous Precision Drone Landing and NLOS Indoor Navigation](https://arxiv.org/abs/2407.05619)
  - **标题**: AIRA：一种基于低成本的IR基于自主精度无人机着陆和NLOS室内导航的方法
  - **Filtered Reason**: none of eess.SY,cs.RO in whitelist
- [TimeTravel: Real-time Timing Drift Attack on System Time Using Acoustic Waves](https://arxiv.org/abs/2407.06853)
  - **标题**: 时间旅行：使用声波对系统时间的实时定时漂移攻击
  - **Filtered Reason**: none of cs.CR in whitelist
- [Gaunt coefficients for complex and real spherical harmonics with applications to spherical array processing and Ambisonics](https://arxiv.org/abs/2407.06847)
  - **标题**: 与球形阵列处理和Ambisonics应用的复杂和真实球形谐波的GANT系数
  - **Filtered Reason**: none of eess.SP,cs.GR,eess.AS,cs.SD in whitelist
- [Speech dereverberation constrained on room impulse response characteristics](https://arxiv.org/abs/2407.08657)
  - **标题**: 语音消失限制在房间脉冲响应特征上
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Geometry-based Multi-beam Survey Line Layout Problem](https://arxiv.org/abs/2407.08184)
  - **标题**: 基于几何的多光束调查线布局问题
  - **Filtered Reason**: none of cs.CG,math.NA in whitelist
- [Few-Shot Bioacoustic Event Detection with Frame-Level Embedding Learning System](https://arxiv.org/abs/2407.10182)
  - **标题**: 使用框架级嵌入学习系统的几射击生物声学事件检测
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Disentangled Acoustic Fields For Multimodal Physical Scene Understanding](https://arxiv.org/abs/2407.11333)
  - **标题**: None
  - **Filtered Reason**: none of eess.AS,cs.RO,cs.SD in whitelist
- [SELM: Enhancing Speech Emotion Recognition for Out-of-Domain Scenarios](https://arxiv.org/abs/2407.15300)
  - **标题**: SELM：增强对跨域情景的语音情感识别
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Towards Realistic Emotional Voice Conversion using Controllable Emotional Intensity](https://arxiv.org/abs/2407.14800)
  - **标题**: 使用可控的情感强度来实现现实的情感语音转换
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Topology-Independent GEVD-Based Distributed Adaptive Node-Specific Signal Estimation in Ad-Hoc Wireless Acoustic Sensor Networks](https://arxiv.org/abs/2407.14172)
  - **标题**: 临时无线声学传感器网络中基于拓扑独立于GEVD的分布式自适应节点特异性信号估计
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [DSP-informed bandwidth extension using locally-conditioned excitation and linear time-varying filter subnetworks](https://arxiv.org/abs/2407.15624)
  - **标题**: 使用本地条件的激发和线性时变滤波器子网的DSP信息带宽扩展
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Secrecy Performance Analysis of Integrated RF-UWOC IoT Networks Enabled by UAV and Underwater-RIS](https://arxiv.org/abs/2407.18766)
  - **标题**: 无人机和水下RIS启用的集成RF-UWOC IOT网络的保密性能分析
  - **Filtered Reason**: none of eess.SP,cs.IT in whitelist
- [Abusive Speech Detection in Indic Languages Using Acoustic Features](https://arxiv.org/abs/2407.20808)
  - **标题**: 使用声学特征以指示语言进行滥用语音检测
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [DeepSpeech models show Human-like Performance and Processing of Cochlear Implant Inputs](https://arxiv.org/abs/2407.20535)
  - **标题**: 深脚的模型显示了类似人类的性能和人工耳蜗输入的处理
  - **Filtered Reason**: none of cs.NE,eess.AS,cs.SD in whitelist
