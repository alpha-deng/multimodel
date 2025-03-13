# 2024-06 月度论文分类汇总

共有32篇相关领域论文, 另有21篇其他

## 人工智能(cs.AI:Artificial Intelligence)

该领域共有 1 篇论文

### Reinforcement Learning Based Escape Route Generation in Low Visibility Environments 
[[arxiv](https://arxiv.org/abs/2406.07568)] [[cool](https://papers.cool/arxiv/2406.07568)] [[pdf](https://arxiv.org/pdf/2406.07568)]
> **Authors**: Hari Srikanth
> **First submission**: 2024-05-27
> **First announcement**: 2024-06-12
> **comment**: No comments
- **标题**: 在低知名度环境中，基于学习的基于学习的逃生路线生成
- **领域**: 人工智能,机器学习,机器人技术
- **摘要**: 结构大火是全国大多数与火灾有关的死亡的原因。为了协助被困人的快速疏散，本文提出了使用一个系统，该系统根据环境测量结果确定消防员的最佳搜索路径，并实时退出平民的路径。通过使用从声纳和烟雾浓度数据得出的信任范围评估和验证的LIDAR映射系统，测试了提出的低可见性映射解决方案。然后将这些独立的点云用于创建不同的地图，这些图是通过使用基于RANSAC的对齐方法合并的，并将其简化为可见性图。然后，使用温度和湿度数据将每个节点标记为危险评分，从而产生环境张量。在展示了基于线性函数近似的自然策略梯度RL方法如何优于稳健性和速度方面的复杂竞争者之后，本文概述了两个系统（救世主和难民）处理环境张量以创建安全的救援和逃生路线的系统。

## 计算语言学(cs.CL:Computation and Language)

该领域共有 6 篇论文

### Spontaneous Speech-Based Suicide Risk Detection Using Whisper and Large Language Models 
[[arxiv](https://arxiv.org/abs/2406.03882)] [[cool](https://papers.cool/arxiv/2406.03882)] [[pdf](https://arxiv.org/pdf/2406.03882)]
> **Authors**: Ziyun Cui,Chang Lei,Wen Wu,Yinan Duan,Diyang Qu,Ji Wu,Runsen Chen,Chao Zhang
> **First submission**: 2024-06-06
> **First announcement**: 2024-06-07
> **comment**: Accepted by Interspeech 2024
- **标题**: 使用耳语和大语言模型自发基于语音的自杀风险检测
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 早期发现自杀风险很重要，因为它可以进行干预以防止潜在的自杀企图。本文研究了基于青少年的自发性演讲的自杀风险的自动检测，并收集了一个普通话数据集，并从15个小时的自杀式言语中，来自一千多名青少年从10岁到十八岁的青少年进行实验。为了利用自发语音嵌入的各种声学和语言特征，低语语音模型和文本大语言模型（LLMS）都用于自杀风险检测。全参数登录和参数效率高的芬太尼方法均用于适应自杀风险检测的预训练模型，并评估了多种音频文本融合方法，以结合Whisper和LLM的表示形式。所提出的系统的检测准确性为0.807，在测试组上具有119名受试者的F1得分为0.846，这表明实际自杀风险检测应用的潜力有望。

### Sustained Vowels for Pre- vs Post-Treatment COPD Classification 
[[arxiv](https://arxiv.org/abs/2406.06355)] [[cool](https://papers.cool/arxiv/2406.06355)] [[pdf](https://arxiv.org/pdf/2406.06355)]
> **Authors**: Andreas Triantafyllopoulos,Anton Batliner,Wolfgang Mayr,Markus Fendler,Florian Pokorny,Maurice Gerczuk,Shahin Amiriparian,Thomas Berghaus,Björn Schuller
> **First submission**: 2024-06-10
> **First announcement**: 2024-06-11
> **comment**: Accepted to INTERSPEECH 2024
- **标题**: 持续的元音，用于治疗后COPD分类
- **领域**: 计算语言学
- **摘要**: 慢性阻塞性肺疾病（COPD）是一种严重的炎症性肺部疾病，影响了世界各地数百万。由于肺部的气流阻塞，它在患者的声音行为中也会显现出来。尤其重要的是检测加重发作，该发作标志着急性阶段，通常需要住院和治疗。先前的工作表明，可以使用自动分析读取语音来区分治疗前和治疗后状态。在这项贡献中，我们检查了持续的元音是否可以提供互补的镜头来分开这两个州。使用50名患者的队列，我们​​表明，使用读取语音从71 \％的基线中加入持续元音可以提高性能，最多可提高79 \％未加权的平均召回率。我们进一步识别和解释最重要的声学特征，这些特征表征了持续元音中COPD的表现。

### Multimodal Belief Prediction 
[[arxiv](https://arxiv.org/abs/2406.07466)] [[cool](https://papers.cool/arxiv/2406.07466)] [[pdf](https://arxiv.org/pdf/2406.07466)]
> **Authors**: John Murzaku,Adil Soubki,Owen Rambow
> **First submission**: 2024-06-11
> **First announcement**: 2024-06-12
> **comment**: John Murzaku and Adil Soubki contributed equally to this work
- **标题**: 多模式信念预测
- **领域**: 计算语言学,机器学习,声音,音频和语音处理
- **摘要**: 认识说话者对信仰的承诺水平是一项艰巨的任务。人类不仅可以解释上下文中单词的含义，还可以理解音调和音频信号其他方面的提示。 NLP社区中的许多论文和语料库都使用仅文本方法来实现信仰预测任务。我们是第一个对多模式信念预测任务进行框架并提出结果的人。我们使用CB-prosody语料库（CBP），其中包含与说话者信念注释的对齐文本和音频。我们首先使用声学特征和传统的机器学习方法报告基准和重要特征。然后，我们为CBP语料库分别在Bert和Whisper上介绍了文本和音频基线。最后，我们介绍了多模式体系结构，该体系结构在Bert和Whisper上进行微调，并使用多种融合方法，仅改善了这两种方式。

### Missingness-resilient Video-enhanced Multimodal Disfluency Detection 
[[arxiv](https://arxiv.org/abs/2406.06964)] [[cool](https://papers.cool/arxiv/2406.06964)] [[pdf](https://arxiv.org/pdf/2406.06964)]
> **Authors**: Payal Mohapatra,Shamika Likhite,Subrata Biswas,Bashima Islam,Qi Zhu
> **First submission**: 2024-06-11
> **First announcement**: 2024-06-12
> **comment**: Accepted to Interspeech 2024
- **标题**: 缺失 - 弹性视频增强多模式的差异检测
- **领域**: 计算语言学,多媒体,声音,音频和语音处理
- **摘要**: 大多数现有的语音不足检测技术仅依赖于声学数据。在这项工作中，我们提出了一种实用的多模式出现检测方法，该方法将可用的视频数据与音频一起利用。我们策划了一个视听数据集，并提出了一种新颖的融合技术，该技术具有统一的重量分担模态 - 不合命相编码器，以学习时间和语义上下文。我们的弹性设计可容纳现实的场景，在推理过程中有时可能会缺少视频方式。当确保两种方式完成时，我们还会提出替代的融合策略。在五项差异检测任务的实验中，我们的统一的多模式方法显着超过了仅听众的单峰方法，在视频和音频方式始终可用时，即使在样品中缺少一半的视频模式时，平均绝对的绝对改善（即10个百分点增加）也会产生7％。

### Joint vs Sequential Speaker-Role Detection and Automatic Speech Recognition for Air-traffic Control 
[[arxiv](https://arxiv.org/abs/2406.13842)] [[cool](https://papers.cool/arxiv/2406.13842)] [[pdf](https://arxiv.org/pdf/2406.13842)]
> **Authors**: Alexander Blatt,Aravind Krishnan,Dietrich Klakow
> **First submission**: 2024-06-19
> **First announcement**: 2024-06-21
> **comment**: Accepted at Interspeech 2024
- **标题**: 关节与顺序的扬声器角色检测和空中流量控制的自动语音识别
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 利用气交通控制（ATC）数据进行下游自然语言处理任务需要进行预处理步骤。关键步骤是通过自动语音识别（ASR）和说话者诊断的数据转录，分别是说话者角色检测（SRD），以将转录本分为飞行员和空中交通控制器（ATCO）转录本。尽管传统方法分别执行这些任务，但我们提出了一个基于变压器的联合ASR-SRD系统，该系统在依靠标准ASR体系结构的同时共同解决这两个任务。我们将该关节系统与多个ATC数据集的ASR和SRD的两种级联方法进行了比较。我们的研究表明，在哪些情况下，我们的联合系统可以胜过两种传统方法，而在哪些情况下，其他架构更可取。我们还评估声学和词汇差异如何影响所有架构，并展示如何克服我们的共同体系结构。

### Children's Speech Recognition through Discrete Token Enhancement 
[[arxiv](https://arxiv.org/abs/2406.13431)] [[cool](https://papers.cool/arxiv/2406.13431)] [[pdf](https://arxiv.org/pdf/2406.13431)]
> **Authors**: Vrunda N. Sukhadia,Shammur Absar Chowdhury
> **First submission**: 2024-06-19
> **First announcement**: 2024-06-21
> **comment**: Accepted at Interspeech 2024
- **标题**: 通过离散令牌增强儿童的语音识别
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 儿童的语音识别被认为是一项低资源任务，这主要是由于缺乏公开可用的数据。此类数据稀缺的原因有很多，包括昂贵的数据收集和注释过程以及数据隐私等。将语音信号转换为不带有敏感信息但捕获语言和声学信息的离散令牌可能是解决隐私问题的解决方案。在这项研究中，我们研究了离散语音令牌将其集成到儿童语音识别系统中，这是输入的，而不会显着降低ASR性能。此外，我们探索了创建这些离散标签的单视图和多视图策略。此外，我们还测试了具有看不见的域和耶稣降生数据集的概括功能的模型。结果表明，儿童的离散代币ASR实现了几乎同等的性能，参数降低了约83％。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 4 篇论文

### A Self-Supervised Denoising Strategy for Underwater Acoustic Camera Imageries 
[[arxiv](https://arxiv.org/abs/2406.02914)] [[cool](https://papers.cool/arxiv/2406.02914)] [[pdf](https://arxiv.org/pdf/2406.02914)]
> **Authors**: Xiaoteng Zhou,Katsunori Mizuno,Yilong Zhang
> **First submission**: 2024-06-05
> **First announcement**: 2024-06-06
> **comment**: 8 pages
- **标题**: 水下声学摄像头图像的自我监督的denoising策略
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 在以浊度和黑暗为特征的低可见性海洋环境中，声学摄像头是能够产生高分辨率2D声纳图像的视觉传感器。但是，声学相机图像会被复杂的噪声干扰，并且很难被下游视觉算法直接摄入。本文介绍了一种使用深度学习技术来降低声学相机图像的新型策略，该技术包括两个主要组成部分：一个自我监督的DeNoising框架和精美的特征指导块。此外，该研究还探讨了图像降级水平与特征匹配性能的改善之间的关系。实验结果表明，所提出的剥离策略可以有效地过滤声学摄像头图像，而无需事先了解噪声模型。不复杂的参数调整和后处理几乎是端到端的。它在保留精美的功能细节的同时成功地消除了噪音，从而增强了本地功能匹配的性能。

### VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling 
[[arxiv](https://arxiv.org/abs/2406.04321)] [[cool](https://papers.cool/arxiv/2406.04321)] [[pdf](https://arxiv.org/pdf/2406.04321)]
> **Authors**: Zeyue Tian,Zhaoyang Liu,Ruibin Yuan,Jiahao Pan,Qifeng Liu,Xu Tan,Qifeng Chen,Wei Xue,Yike Guo
> **First submission**: 2024-06-06
> **First announcement**: 2024-06-07
> **comment**: The code and datasets will be available at https://github.com/ZeyueT/VidMuse/
- **标题**: vidmuse：一个简单的视频到音乐生成框架，具有长期建模
- **领域**: 计算机视觉和模式识别,机器学习,多媒体,声音
- **摘要**: 在这项工作中，我们系统地研究音乐发电仅以视频为条件。首先，我们提出一个大型数据集，其中包括360k视频音乐对，包括各种流派，例如电影预告片，广告和纪录片。此外，我们提出了Vidmuse，这是一个简单的框架，用于生成与视频输入对齐的音乐。 Vidmuse通过制作具有声学和语义与视频一致的高保真音乐来脱颖而出。通过合并本地和全局的视觉提示，Vidmuse可以创建音乐连贯的音轨，这些音轨通过长期建模始终匹配视频内容。通过广泛的实验，Vidmuse在音频质量，多样性和视听一致性方面优于现有模型。代码和数据集将在https://github.com/zeyuet/vidmuse/上找到。

### RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker 
[[arxiv](https://arxiv.org/abs/2406.07189)] [[cool](https://papers.cool/arxiv/2406.07189)] [[pdf](https://arxiv.org/pdf/2406.07189)]
> **Authors**: Yunfeng Li,Bo Wang,Jiuran Sun,Xueyi Wu,Ye Li
> **First submission**: 2024-06-11
> **First announcement**: 2024-06-12
> **comment**: No comments
- **标题**: RGB-SONAR跟踪基准和空间跨注意变形金刚跟踪器
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉摄像头和声纳在水下环境中自然是互补的。结合两种方式的信息将促进更好地观察水下目标。但是，在先前的研究中，这个问题尚未得到足够的关注。因此，本文介绍了一项新的挑战性RGB-SONAR（RGB-S）跟踪任务，并研究了如何通过RGB和声纳方式的相互作用来实现水下目标的有效跟踪。具体而言，我们首先提出一个RGBS50基准数据集，其中包含50个序列和87000多个高质量注释的边界框。实验结果表明，RGBS50基准对当前流行的SOT跟踪器构成了挑战。其次，我们提出了一个称为SCANET的RGB-S跟踪器，其中包括一个空间跨意义模块（SCAM），该模块由新型的空间跨意义层和两个独立的全局整合模块组成。空间交叉注意力用于克服RGB和声纳图像之间空间未对准的问题。第三，我们提出了一种基于SOT数据的RGB-S仿真培训方法（SRST），以克服缺乏RGB-S培训数据集。它将RGB图像转换为类似声纳的显着图像以构建伪数据对，从而使模型能够学习RGB-S样数据的语义结构。综合实验表明，提出的空间跨注意力有效地实现了RGB与声纳方式和扫描仪之间的相互作用，在提议的基准上实现了最先进的性能。该代码可在https://github.com/liyunfenglyf/rgbs50上找到。

### IDA-UIE: An Iterative Framework for Deep Network-based Degradation Aware Underwater Image Enhancement 
[[arxiv](https://arxiv.org/abs/2406.18628)] [[cool](https://papers.cool/arxiv/2406.18628)] [[pdf](https://arxiv.org/pdf/2406.18628)]
> **Authors**: Pranjali Singh,Prithwijit Guha
> **First submission**: 2024-06-26
> **First announcement**: 2024-06-27
> **comment**: No comments
- **标题**: IDA-UIE：一个迭代框架，用于基于网络的深度退化意识到水下图像增强
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 水下图像质量受荧光，低照明，吸收和散射的影响。水下图像增强的最新研究提出了不同的深网架构来解决这些问题。这些作品中的大多数都提出了一个网络来应对所有挑战。我们认为，接受过特定条件的深层网络比从所有退化案例中学到的单个网络提供了更好的性能。因此，这项工作的第一个贡献在于迭代框架的提议，在该框架中确定并解决了单个主导降解条件。该提案考虑了以下八种降解条件 - 低照明，低对比度，朦胧，图像模糊，噪声和颜色失衡在三个不同的通道中的存在。深层网络旨在确定主要的降解条件。因此，选择了适当的深网络以进行降解条件特定的增强。这项工作的第二个贡献是从两个标准数据集（UIEB和EUVP）的高质量图像中构建降解条件的特定数据集。该数据集用于学习特定条件的增强网络。发现所提出的方法在UIEB和EUVP数据集上的表现优于9个基线方法。

## 声音(cs.SD:Sound)

该领域共有 12 篇论文

### MaskSR: Masked Language Model for Full-band Speech Restoration 
[[arxiv](https://arxiv.org/abs/2406.02092)] [[cool](https://papers.cool/arxiv/2406.02092)] [[pdf](https://arxiv.org/pdf/2406.02092)]
> **Authors**: Xu Li,Qirui Wang,Xiaoyu Liu
> **First submission**: 2024-06-04
> **First announcement**: 2024-06-05
> **comment**: Accepted by INTERSPEECH 2024. Demo page: https://masksr.github.io/MaskSR/
- **标题**: MASKSR：全面语音恢复的蒙版语言模型
- **领域**: 声音,人工智能,机器学习,音频和语音处理,信号处理
- **摘要**: 语音恢复旨在在存在各种扭曲的情况下恢复高质量的语音。尽管已经为这项任务研究了几个深度学习范式，但尚未充分探索最近新兴语言模型的力量。在本文中，我们提出了MaskSr，这是一种掩盖语言模型，该模型能够恢复全带44.1 kHz语音，共同考虑噪音，混响，剪裁和低带宽。 MASKSR与使用预训练的神经编解码器提取的离散声音令牌一起工作。在训练过程中，MaskSR进行了优化，以预测从高质量的目标语音中提取的随机掩盖的令牌，并以各种失真为条件。在推论过程中，MaskSR通过有效的迭代采样重建目标语音令牌。广泛的实验表明，与广泛的模型相比，MASKSR在全带语音恢复任务以及子任务上都获得了竞争结果。

### Speech-based Clinical Depression Screening: An Empirical Study 
[[arxiv](https://arxiv.org/abs/2406.03510)] [[cool](https://papers.cool/arxiv/2406.03510)] [[pdf](https://arxiv.org/pdf/2406.03510)]
> **Authors**: Yangbin Chen,Chenyang Xu,Chunfeng Liang,Yanbao Tao,Chuan Shi
> **First submission**: 2024-06-05
> **First announcement**: 2024-06-06
> **comment**: 5 pages, 3 figures
- **标题**: 基于语音的临床抑郁筛查：一项实证研究
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 这项研究调查了跨不同互动场景的基于AI的抑郁症筛查的语音信号的实用性，包括精神科访谈，聊天机器人对话和文本阅读。参与者包括从北京大学第六医院门诊诊所招募的沮丧患者和社区对照组成员，所有患者均由标准化诊断方案的精神病医生诊断。我们从每个参与者的分段录音中提取了声学和深厚的语音特征。使用神经网络或SVM进行分类，并具有确定最终评估的汇总结果。我们在交互情况，语音处理技术和特征类型的分析中证实了语音是抑郁症筛查的关键标记。具体而言，人类计算机的互动与临床访谈功效相匹配，超过阅读任务。段持续时间和数量显着影响模型性能，深层语音特征基本上优于传统声学特征。

### BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification 
[[arxiv](https://arxiv.org/abs/2406.06786)] [[cool](https://papers.cool/arxiv/2406.06786)] [[pdf](https://arxiv.org/pdf/2406.06786)]
> **Authors**: June-Woo Kim,Miika Toikkanen,Yera Choi,Seoung-Eun Moon,Ho-Young Jung
> **First submission**: 2024-06-10
> **First announcement**: 2024-06-11
> **comment**: Accepted INTERSPEECH 2024
- **标题**: BTS：元数据辅助呼吸道声音分类的桥接文本和声音方式
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 呼吸道声音分类（RSC）由于多样化的声学特征而具有挑战性，主要受患者人口统计和记录环境的影响。为了解决这个问题，我们介绍了一种使用呼吸道元数据的文本原模型模型，该模型为RSC提供了有用的互补信息。具体而言，我们使用来自声音样本的元数据得出的自由文本描述微调了预算的文本原模型模型，该描述包括患者的性别和年龄，记录设备的类型以及患者体内的记录位置。我们的方法在洲际论术语数据集上实现了最先进的性能，超过了先前的最佳结果1.17％。该结果验证了利用元数据和呼吸声样本在增强RSC性能方面的有效性。此外，我们研究了元数据部分不可用的情况，这可能发生在现实世界中的临床环境中。

### Predicting Heart Activity from Speech using Data-driven and Knowledge-based features 
[[arxiv](https://arxiv.org/abs/2406.06341)] [[cool](https://papers.cool/arxiv/2406.06341)] [[pdf](https://arxiv.org/pdf/2406.06341)]
> **Authors**: Gasser Elbanna,Zohreh Mostaani,Mathew Magimai. -Doss
> **First submission**: 2024-06-10
> **First announcement**: 2024-06-11
> **comment**: Accepted at Interspeech 2024
- **标题**: 使用数据驱动和基于知识的特征从语音中预测心脏活动
- **领域**: 声音,人工智能,音频和语音处理,信号处理
- **摘要**: 准确地预测心脏活动和其他生物学信号对于诊断和监测至关重要。鉴于语音是多个生理系统的结果，因此一大批作品研究了心脏活动的声学相关性。最近，与传统声学方法相比，自我监督的模型在与语音有关的任务方面表现出色。但是，数据驱动表示在预测心脏活动中的鲁棒性尚未探索。在这项研究中，我们证明了自我监督的语音模型在预测心脏活动参数方面优于声学特征。我们还强调了个体变异性对模型概括性的影响。这些发现强调了数据驱动表示在此类任务中的价值，以及需要更多基于语音的生理数据来减轻与说话者相关的挑战的价值。

### Bridging Language Gaps in Audio-Text Retrieval 
[[arxiv](https://arxiv.org/abs/2406.07012)] [[cool](https://papers.cool/arxiv/2406.07012)] [[pdf](https://arxiv.org/pdf/2406.07012)]
> **Authors**: Zhiyong Yan,Heinrich Dinkel,Yongqing Wang,Jizhong Liu,Junbo Zhang,Yujun Wang,Bin Wang
> **First submission**: 2024-06-11
> **First announcement**: 2024-06-12
> **comment**: interspeech2024
- **标题**: 音频文本检索中的桥接语言差距
- **领域**: 声音,计算语言学,音频和语音处理
- **摘要**: 音频文本检索是一项具有挑战性的任务，需要在数据库中搜索音频剪辑或文本字幕。鉴于实际数据中的大量非英语内容，现有研究的主要重点是对此类模型的适用性的限制。为了解决这些语言差异，我们建议使用多语言文本编码器（声纳）提出一种语言增强性（LE），并使用特定于语言的信息对文本数据进行编码。此外，我们通过应用一致的集合蒸馏（CED）来优化音频编码器，从而增强对可变长度音频文本检索的支持。我们的方法在英语音频文本检索中表现出色，在常用数据集（例如听力胶囊和衣服）上展示了最先进的（SOTA）性能。同时，该方法表现出熟练程度，可以在其他七种语言中检索内容，只有10％的其他语言增强培训数据，从而产生了令人鼓舞的结果。源代码可公开可用https://github.com/zyyan4/ml-clap。

### CTC-aligned Audio-Text Embedding for Streaming Open-vocabulary Keyword Spotting 
[[arxiv](https://arxiv.org/abs/2406.07923)] [[cool](https://papers.cool/arxiv/2406.07923)] [[pdf](https://arxiv.org/pdf/2406.07923)]
> **Authors**: Sichen Jin,Youngmoon Jung,Seungjin Lee,Jaeyoung Roh,Changwoo Han,Hoonyoung Cho
> **First submission**: 2024-06-12
> **First announcement**: 2024-06-13
> **comment**: ef:Proceedings of Interspeech 2024
- **标题**: CTC对准的音频文本嵌入用于流式开放式唱片集的关键字斑点
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 本文介绍了一种新颖的方法，用于通过基于文本的关键字注册流式开放vocabulary关键字斑点（KWS）。对于每个输入框架，提出的方法使用连接式时间分类（CTC）在框架结束的最佳对齐方式，并汇总框架级别的声学嵌入（AE）以获得与文本嵌入式（TE TE）的高级（即字符，单词或短语）获得更高级别的（即字符，单词或短语）AE。之后，我们计算了聚集的AE和TE的相似性。据我们所知，这是第一次尝试将音频和关键字文本对齐的尝试，以实现KWS的联合音频文本嵌入。尽管以流式传输方式运行，但与仅使用155k模型参数的非流式方法相比，我们的方法在Libriphrase数据集上实现了竞争性能，并且具有时间复杂性O（U）的解码算法，其中U是推理时目标关键字的目标长度。

### AV-GS: Learning Material and Geometry Aware Priors for Novel View Acoustic Synthesis 
[[arxiv](https://arxiv.org/abs/2406.08920)] [[cool](https://papers.cool/arxiv/2406.08920)] [[pdf](https://arxiv.org/pdf/2406.08920)]
> **Authors**: Swapnil Bhosale,Haosen Yang,Diptesh Kanojia,Jiankang Deng,Xiatian Zhu
> **First submission**: 2024-06-13
> **First announcement**: 2024-06-14
> **comment**: No comments
- **标题**: AV-GS：学习材料和几何学意识到新颖的观点声学综合
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 鉴于在3D场景中声源发出的单声道音频，新型视图声学（NVAS）旨在在任何目标角度呈现双耳音频。现有方法提出了基于NERF的隐式模型，以利用视觉提示作为合成双耳音频的条件。但是，除了源自重型NERF渲染的低效率外，这些方法还具有表征整个场景环境的能力有限的能力，例如房间几何形状，材料属性以及听众和声源之间的空间关系。为了解决这些问题，我们提出了一种新颖的视听高斯裂（AV-GS）模型。为了获得音频综合的材料感知和几何感知条件，我们在本地初始化的高斯点上学习了一个基于积分的场景表示，并考虑了来自听众和声音源的空间关系。为了使视觉场景模型音频自适应，我们提出了一个点致密化和修剪策略，以最佳分发高斯点，并在声音传播中的每点贡献（例如，在影响声音路径转移时，无纹理壁表面所需的更多点需要更多点）。广泛的实验验证了我们的AV-GS优于现实世界RWA和基于仿真的Soundspaces数据集的现有替代方案的优势。

### Large Language Models for Dysfluency Detection in Stuttered Speech 
[[arxiv](https://arxiv.org/abs/2406.11025)] [[cool](https://papers.cool/arxiv/2406.11025)] [[pdf](https://arxiv.org/pdf/2406.11025)]
> **Authors**: Dominik Wagner,Sebastian P. Bayerl,Ilja Baumann,Korbinian Riedhammer,Elmar Nöth,Tobias Bocklet
> **First submission**: 2024-06-16
> **First announcement**: 2024-06-17
> **comment**: Accepted at Interspeech 2024
- **标题**: 大型语言模型在口吃中检测失调障碍
- **领域**: 声音,计算语言学,音频和语音处理
- **摘要**: 准确地检测口语中的功能障碍可以帮助提高自动语音和语言处理组件的性能，并支持开发更具包容性的语音和语言技术。受到最新部署大语言模型（LLM）的趋势的启发，我们作为非时光输入的通用学习者和处理器（例如音频和视频），我们处理了多标签功能障碍检测的任务作为语言建模问题。我们提出了由自动语音识别系统和从音频编码器模型提取到LLM的声音表示产生的假设候选者，并在三个包含英语和德国口吃语音的数据集上预测系统的系统，以预测该系统。实验结果表明，我们的系统有效地结合了声学和词汇信息，并在多标签的口吃检测任务上取得了竞争成果。

### A Mel Spectrogram Enhancement Paradigm Based on CWT in Speech Synthesis 
[[arxiv](https://arxiv.org/abs/2406.12164)] [[cool](https://papers.cool/arxiv/2406.12164)] [[pdf](https://arxiv.org/pdf/2406.12164)]
> **Authors**: Guoqiang Hu,Huaning Tan,Ruilai Li
> **First submission**: 2024-06-17
> **First announcement**: 2024-06-18
> **comment**: Accepted by IALP 2024
- **标题**: None
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 声学特征在提高合成语音的质量方面起着重要作用。当前，MEL频谱图是大多数声学模型中广泛使用的声学特征。但是，由于其傅立叶变换过程造成的细粒损失，MEL频谱图合成的语音的清晰度在突变信号中受到损害。为了获得更详细的MEL频谱图，我们提出了基于连续小波变换（CWT）的MEL频谱增强范式。该范式引入了一个附加的任务：一个更详细的小波谱图，就像后处理网络一样，解码器将其作为输入MEL频谱图输出。我们选择tacotron2和fastspeech2进行实验验证，以分别测试自回归（AR）和非自动回应（NAR）语音系统。实验结果表明，使用模型与MEL频谱增强范式合成的语音表现出更高的MOS，与基线模型相比，分别提高了0.14和0.09。这些发现为增强范式的普遍性提供了一些验证，因为它们证明了不同体系结构中范式的成功。

### Automated Bioacoustic Monitoring for South African Bird Species on Unlabeled Data 
[[arxiv](https://arxiv.org/abs/2406.13579)] [[cool](https://papers.cool/arxiv/2406.13579)] [[pdf](https://arxiv.org/pdf/2406.13579)]
> **Authors**: Michael Doell,Dominik Kuehn,Vanessa Suessle,Matthew J. Burnett,Colleen T. Downs,Andreas Weinmann,Elke Hergenroether
> **First submission**: 2024-06-19
> **First announcement**: 2024-06-21
> **comment**: preprint
- **标题**: 在未标记的数据上对南非鸟类的自动化生物声监测
- **领域**: 声音,计算机视觉和模式识别,音频和语音处理
- **摘要**: 基于被动声学监测（PAM）记录的生物多样性监测的分析是耗时的，并且由于录音中存在背景噪声的存在而挑战。现有的声音事件检测模型（SED）仅适用于某些鸟类物种，而开发进一步的模型需要标记的数据。开发的框架自动从可用平台的选定禽类中提取了标记的数据。标记的数据嵌入了录音中，包括环境声音和噪声，并用于训练卷积复发性神经网络（CRNN）模型。对这些模型进行了评估，该模型是在城市夸祖鲁 - 纳塔尔栖息地中记录的未经处理的现实世界数据中的评估。改编的SED-CRNN模型达到了0.73的F1分数，证明了其在嘈杂的现实世界中的效率。所提出的方法自动提取所选鸟类物种标记的数据，使PAM轻松适应其他物种和栖息地，以供将来的保护项目。

### Predicting Individual Depression Symptoms from Acoustic Features During Speech 
[[arxiv](https://arxiv.org/abs/2406.16000)] [[cool](https://papers.cool/arxiv/2406.16000)] [[pdf](https://arxiv.org/pdf/2406.16000)]
> **Authors**: Sebastian Rodriguez,Sri Harsha Dumpala,Katerina Dikaios,Sheri Rempel,Rudolf Uher,Sageev Oore
> **First submission**: 2024-06-22
> **First announcement**: 2024-06-24
> **comment**: No comments
- **标题**: 在语音中预测声学特征的单个抑郁症状
- **领域**: 声音,人工智能,机器学习,音频和语音处理
- **摘要**: 当前的自动抑郁检测系统可直接提供预测，而无需依赖临床抑郁量表中表示的单个症状/抑郁症。相比之下，临床医生在临床环境中评估抑郁级评级量表的每个项目，因此隐含地为抑郁症诊断提供了更详细的理由。在这项工作中，我们迈出了使用语音的声学特征来预测抑郁量表的单个项目，然后再获得最终的抑郁预测。为此，我们使用卷积（CNN）和经常性（长期记忆（LSTM））神经网络。我们考虑了学习语音的时间背景的不同方法。此外，我们分析了两个投票方案的两种变体，以进行单个项目预测和抑郁检测。我们还包括一个动画可视化，该可视化显示了随着时间的流逝，随着时间的流逝，项目预测的示例。

### AI-based Drone Assisted Human Rescue in Disaster Environments: Challenges and Opportunities 
[[arxiv](https://arxiv.org/abs/2406.15875)] [[cool](https://papers.cool/arxiv/2406.15875)] [[pdf](https://arxiv.org/pdf/2406.15875)]
> **Authors**: Narek Papyan,Michel Kulhandjian,Hovannes Kulhandjian,Levon Hakob Aslanyan
> **First submission**: 2024-06-22
> **First announcement**: 2024-06-24
> **comment**: :68U10; 68T50(Primary) 68T45 (Secondary)ACM Class:I.2.7; I.2.10; I.4.0
- **标题**: 基于人工智能的无人机协助人类救援在灾难环境中：挑战和机遇
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在这项调查中，我们专注于利用基于无人机的系统来检测个体，尤其是通过识别人类的尖叫和其他遇险信号。这项研究在灾后场景中具有重要意义，包括地震，飓风，军事冲突，野火等事件。这些无人机能够悬停在灾难的地区上，这可能使救援队直接访问可能具有挑战性。在灾难情况下，通常将无人驾驶汽车（UAV）（无人机）经常用于搜索和救援任务。通常，无人机捕获空中图像以评估结构性损害并确定灾难的程度。他们还采用热成像技术来检测人体热特征，这可以帮助定位个人。在某些情况下，较大的无人机用于向陷入孤立灾难地区滞留的人们提供必要的供应。在我们的讨论中，我们深入研究了通过空中声学来定位人类的独特挑战。听觉系统必须区分人类的哭声和自然发生的声音，例如动物呼唤和风。此外，它应该能够识别与大喊，鼓掌或其他人试图向救援团队发出信号的信号相关的不同模式。为了应对这一挑战，一种解决方案涉及利用人工智能（AI）分析声音频率并确定常见的音频签名。可以使用这些签名来训练基于深度学习的网络，例如卷积神经网络（CNN），以滤除无人机电机和其他环境因素产生的噪声。此外，采用信号处理技术，例如基于麦克风阵列信号的到达方向（DOA）可以增强跟踪人类噪声源的精度。

## 音频和语音处理(eess.AS:Audio and Speech Processing)

该领域共有 5 篇论文

### RevRIR: Joint Reverberant Speech and Room Impulse Response Embedding using Contrastive Learning with Application to Room Shape Classification 
[[arxiv](https://arxiv.org/abs/2406.03120)] [[cool](https://papers.cool/arxiv/2406.03120)] [[pdf](https://arxiv.org/pdf/2406.03120)]
> **Authors**: Jacob Bitterman,Daniel Levi,Hilel Hagai Diamandi,Sharon Gannot,Tal Rosenwein
> **First submission**: 2024-06-05
> **First announcement**: 2024-06-06
> **comment**: Accepted to Interspeech 2024
- **标题**: Revrir：使用对比度学习嵌入与房间形状分类的对比度学习嵌入的联合混响语音和房间冲动响应
- **领域**: 音频和语音处理,机器学习,声音
- **摘要**: 本文着重于房间指纹，这项任务涉及对音频记录的分析，以确定捕获其捕获的房间的特定音量和形状。虽然从房间脉冲响应（RIR）确定基本房间参数相对简单，但通过语音信号进行操作是一项繁琐的任务。为了应对这一挑战，我们介绍了一种双重编码体架构，该体系结构直接从语音话语中促进了房间参数的估计。在预训练期间，一个编码器接收RIR，而另一个编码器会处理回响信号。使用对比损失函数将语音和声学响应嵌合在一起。在微调阶段，训练了特定的分类任务。在测试阶段，只有回响的话语，并且其嵌入用于房间形状分类的任务。使用模拟的声学环境对所提出的方案进行了广泛的评估。

### Relational Proxy Loss for Audio-Text based Keyword Spotting 
[[arxiv](https://arxiv.org/abs/2406.05314)] [[cool](https://papers.cool/arxiv/2406.05314)] [[pdf](https://arxiv.org/pdf/2406.05314)]
> **Authors**: Youngmoon Jung,Seungjin Lee,Joon-Young Yang,Jaeyoung Roh,Chang Woo Han,Hoon-Young Cho
> **First submission**: 2024-06-07
> **First announcement**: 2024-06-10
> **comment**: 5 pages, 2 figures, Accepted by Interspeech 2024
- **标题**: 基于音频文本的关键字发现的关系代理损失
- **领域**: 音频和语音处理,人工智能,信号处理
- **摘要**: 近年来，人们对用户便利的关注越来越多，从而增加了对基于文本的关键字注册系统（KWS）的兴趣。由于系统在注册阶段和实际使用过程中使用音频输入时使用文本输入，因此我们称此任务是基于音频文本的KWS。为了启用这项任务，声学和文本编码器通常都使用深度度量学习损失功能（例如基于三重态和代理损失）进行训练。这项研究旨在通过利用声学嵌入和文本嵌入中的结构关系来改善现有方法。与以前仅在点对点上比较声学和文本嵌入的研究不同，我们的方法通过引入关系代理损失的概念（RPL）来重点关注嵌入空间内的关系结构。通过合并RPL，我们在《华尔街日报》（WSJ）语料库上表现出了提高的性能。

### Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring 
[[arxiv](https://arxiv.org/abs/2406.07250)] [[cool](https://papers.cool/arxiv/2406.07250)] [[pdf](https://arxiv.org/pdf/2406.07250)]
> **Authors**: Tomoya Nishida,Noboru Harada,Daisuke Niizumi,Davide Albertini,Roberto Sannino,Simone Pradolini,Filippo Augusti,Keisuke Imoto,Kota Dohi,Harsh Purohit,Takashi Endo,Yohei Kawaguchi
> **First submission**: 2024-06-11
> **First announcement**: 2024-06-12
> **comment**: anomalydetection,acousticcondition monitoring, domain shift, first-shot problem, DCASE Challenge. arXiv admin note: text overlap with arXiv:2305.07828
- **标题**: DCASE 2024挑战任务2：机器状况监控的第一次射击异常检测
- **领域**: 音频和语音处理,机器学习,声音
- **摘要**: 我们介绍了声学场景和事件的检测和分类的任务描述（DCASE）2024挑战任务2：用于机器条件监视的第一张无监督异常检测（ASD）。从去年的DCASE 2023挑战任务2开始，我们将任务组织为首次弹出问题，在域泛化所需的设置下。首次发射问题的主要目的是在不需要机器特定的高参数调谐的情况下快速部署新型机器的ASD系统。通过（1）仅为每种机器类型提供一个部分，以及（2）具有完全不同的机器类型以进行开发和评估数据集来实现此问题设置。对于DCASE 2024挑战任务2，新收集了全新机器类型的数据，并作为评估数据集提供。此外，将几种机器类型的属性信息（例如机器操作条件）隐藏在模仿不可用的信息的情况下。我们将添加挑战结果和挑战提交截止日期后提交的分析。

### Tool Wear Prediction in CNC Turning Operations using Ultrasonic Microphone Arrays and CNNs 
[[arxiv](https://arxiv.org/abs/2406.08957)] [[cool](https://papers.cool/arxiv/2406.08957)] [[pdf](https://arxiv.org/pdf/2406.08957)]
> **Authors**: Jan Steckel,Arne Aerts,Erik Verreycken,Dennis Laurijssen,Walter Daems
> **First submission**: 2024-06-13
> **First announcement**: 2024-06-14
> **comment**: No comments
- **标题**: 使用超声波麦克风阵列和CNN的CNC转动操作中的工具磨损预测
- **领域**: 音频和语音处理,人工智能,声音,信号处理
- **摘要**: 本文介绍了一种用于预测CNC转动操作中工具磨损的新方法，结合了超声波麦克风阵列和卷积神经网络（CNN）。使用波束形成技术增强了0 kHz至60 kHz之间的高频声学发射，以提高信号效率比率。然后通过CNN分析处理后的声学数据，该数据预测切割工具的其余使用寿命（RUL）。该模型对350件工件的数据进行了培训，该模型可以准确预测碳化物插入物的Rul。我们的结果表明，通过将先进的超声传感器与深度学习的整合到CNC加工中的准确预测维护任务中获得的潜力。

### Evaluating Speaker Identity Coding in Self-supervised Models and Humans 
[[arxiv](https://arxiv.org/abs/2406.10401)] [[cool](https://papers.cool/arxiv/2406.10401)] [[pdf](https://arxiv.org/pdf/2406.10401)]
> **Authors**: Gasser Elbanna
> **First submission**: 2024-06-14
> **First announcement**: 2024-06-17
> **comment**: Masters Thesis
- **标题**: 评估说话者身份编码在自我监督模型和人类中
- **领域**: 音频和语音处理,人工智能,声音
- **摘要**: 说话者身份在人类交流中起着重要作用，并且越来越多地用于社会应用中，许多通过机器学习的进步。说话者身份感知是一种基本的认知现象，可以将可以大致降低到两个主要任务：识别声音或在声音之间进行区分。几项研究试图识别身份感知的声学相关性，以查明该任务的显着参数。与其他交流社会信号不同，大多数努力都得出了无效的结论。此外，语音身份处理的当前神经认知模型将感知基础视为声学维度，例如基本频率，谐波与噪声比率和共振体分散。但是，这些发现并不能说明自然主义的语音和言论中的变异性。当前的自我监督模型的代表空间在各种语音相关的任务中表现出显着的性能。在这项工作中，我们证明了来自不同家庭（例如生成，对比和预测模型）的自我监督的表示，对于声音代表而言，代言人的识别明显更好。我们还表明，这种说话者身份识别任务可用于更好地理解这些强大网络不同层中声学信息表示的性质。通过评估声音，音素，韵律和语言变体之间的说话者识别精度，我们报告了模型性能与人类认同感知之间的相似性。我们通过将模型和人类的编码空间并列，并挑战使用距离指标作为说话者接近的代理来进一步研究这些相似性。最后，我们表明某些模型可以预测自然主义刺激期间听觉和语言区域的大脑反应。

## 信号处理(eess.SP:Signal Processing)

该领域共有 3 篇论文

### Toward Fully-End-to-End Listened Speech Decoding from EEG Signals 
[[arxiv](https://arxiv.org/abs/2406.08644)] [[cool](https://papers.cool/arxiv/2406.08644)] [[pdf](https://arxiv.org/pdf/2406.08644)]
> **Authors**: Jihwan Lee,Aditya Kommineni,Tiantian Feng,Kleanthis Avramidis,Xuan Shi,Sudarsana Kadiri,Shrikanth Narayanan
> **First submission**: 2024-06-12
> **First announcement**: 2024-06-13
> **comment**: accepted to Interspeech2024
- **标题**: 倾向于全端到最终的聆听语音解码
- **领域**: 信号处理,人工智能,声音,音频和语音处理
- **摘要**: 从EEG信号解码的语音解码是一项艰巨的任务，在该任务中，大脑活动的建模以估计声刺激的显着特征。我们提出了FESDE，这是一个新颖的框架，用于从EEG信号中解码全端到最终的语音。我们的方法旨在直接在脑电图信号下直接重建听力的语音波形，而无需中间声学特征处理步骤。提出的方法由EEG模块和语音模块以及连接器组成。脑电图模块学会更好地表示脑电图，而语音模块从模型表示产生语音波形。连接器学会桥接脑电图和语音的潜在空间的分布。提出的框架既简单又有效，可以通过允许单步推理，并且在目标指标上的先前工作优于先前的工作。进行细粒的音素分析以揭示语音解码的模型特征。源代码可在此处找到：github.com/lee-jhwn/fesde。

### MEMS and ECM Sensor Technologies for Cardiorespiratory Sound Monitoring - A Comprehensive Review 
[[arxiv](https://arxiv.org/abs/2406.12432)] [[cool](https://papers.cool/arxiv/2406.12432)] [[pdf](https://arxiv.org/pdf/2406.12432)]
> **Authors**: Yasaman Torabi,Shahram Shirani,James P. Reilly,Gail M Gauvreau
> **First submission**: 2024-06-18
> **First announcement**: 2024-06-19
> **comment**: ef:Sensors, Vol. 24, Issue 21, Page 7036, 2024
- **标题**: 用于心肺声音监测的MEMS和ECM传感器技术 - 全面评论
- **领域**: 信号处理,人工智能,机器学习,音频和语音处理
- **摘要**: 本文介绍了心肺听觉感应设备（即听诊器）的全面综述，这对于理解理论方面和实用设计说明非常有用。在本文中，我们首先介绍心脏和肺的声学特性，以及听诊器演变的简短历史。然后，我们讨论了基于它们基于它们的eLLITRET冷凝器麦克风（ECM）的基本概念。然后，我们讨论微电学系统（MEMSS）技术，尤其是关注压电传感器传感器。本文全面回顾了心脏呼吸诱导的传感技术，并强调了过去十年中基于MEMS的可穿戴设计。据我们所知，这是第一篇概述ECM和MEMS应用程序的文章，以进行心脏和肺部声音分析。

### SGSM: A Foundation-model-like Semi-generalist Sensing Model 
[[arxiv](https://arxiv.org/abs/2406.16933)] [[cool](https://papers.cool/arxiv/2406.16933)] [[pdf](https://arxiv.org/pdf/2406.16933)]
> **Authors**: Tianjian Yang,Hao Zhou,Shuo Liu,Kaiwen Guo,Yiwen Hou,Haohua Du,Zhi Liu,Xiang-Yang Li
> **First submission**: 2024-06-15
> **First announcement**: 2024-06-24
> **comment**: No comments
- **标题**: SGSM：基础模型的半将军感应模型
- **领域**: 信号处理,人工智能
- **摘要**: 智能传感系统的重要性在智能服务领域增长。这些系统提取相关信号功能并为特定任务生成信息表示。但是，为此类系统构建功能提取组件需要广泛的特定领域专业知识或数据。基础模型的极快发展可能会在这种智能感知中引入新发现的能力。我们提出了一种新的传感模型方案，我们将其称为半将军感应模型（SGSM）。与传统系统相比，SGSM能够使用相对较少的特定于任务标记的数据来半息术解决各种任务。 SGSM通过对共同理论模型的分析构建，可以描述不同的方式，例如声学和Wi-Fi信号。这两个异质传感器的实验结果表明，SGSM在各种场景中的功能，从而建立了其广泛的适用性。在某些情况下，SGSM甚至比传感器特定的专业解决方案的性能更好。 Wi-Fi评估表明，将SGSM应用于现有传感模型时有20 \％的精度提高。

## 医学物理(physics.med-ph:Medical Physics)

该领域共有 1 篇论文

### Deep-Learning Approach for Tissue Classification using Acoustic Waves during Ablation with an Er:YAG Laser (Updated) 
[[arxiv](https://arxiv.org/abs/2406.14570)] [[cool](https://papers.cool/arxiv/2406.14570)] [[pdf](https://arxiv.org/pdf/2406.14570)]
> **Authors**: Carlo Seppi,Philippe C. Cattin
> **First submission**: 2024-06-06
> **First announcement**: 2024-06-21
> **comment**: This paper is an updated version of Deep-Learning Approach for TissueClassificationusingAcousticWaves during Ablation with an Er:YAG Laser originally published in DOI:10.1109/ACCESS.2021.3113055. This update addresses several issues and incorporates corrections as outlined in DOI:10.1109/ACCESS.2024.3395071. We provide here a detailed description of our experiments and the new models we used
- **标题**: 用ER：YAG激光消融过程中使用声波的深度学习方法（更新）
- **领域**: 医学物理,人工智能,图像和视频处理,组织和器官
- **摘要**: 当今的骨切割机械工具（截骨切开术）会导致机械创伤，从而延长了愈合过程。医疗设备制造商的目的是将这种创伤最小化，使用激光切割作为一种创新，最小的侵入性手术。这种方法使用激光光而不是机械工具消灭组织，从而减少了手术后的愈合时间。可靠的反馈系统在激光手术期间至关重要，以防止周围组织损害。我们提出了一种组织分类方法，分析激光消融过程中产生的声波，证明其在前体验实验中的适用性。用微秒脉冲ER：YAG激光器的消融过程产生的声波，并用空气耦合传感器获得。这些波被用来对五种猪组织类型进行分类：硬骨，软骨，肌肉，脂肪和皮肤。对于自动组织分类，我们将五种神经网络（NN）方法与时间相关输入进行了比较：一种一维卷积神经网络（CNN），一种完全连接的神经网络（FCNN）与频率频谱的频率频谱或频率频率组成的频率组件（作为输入和CNN的频率组合）和fcnn的频率组合和频率相结合。连续的声波用于提高分类精度。 Grad-CAM确定了频率的激活图，显示低频率是该任务最重要的。我们的结果表明，将时间依赖性数据与其频谱相结合达到了最高分类精度（65.5％-75.5％）。我们还发现，仅使用频率频谱就足够了，因此应用主组件分析（PCA）没有其他好处。

## 其他论文

共有 21 篇其他论文

- [MunchSonic: Tracking Fine-grained Dietary Actions through Active Acoustic Sensing on Eyeglasses](https://arxiv.org/abs/2405.21004)
  - **标题**: Munchsonic：通过眼镜上的主动声传感跟踪细粒度的饮食动作
  - **Filtered Reason**: none of cs.ET,cs.HC in whitelist
- [A Frame-based Attention Interpretation Method for Relevant Acoustic Feature Extraction in Long Speech Depression Detection](https://arxiv.org/abs/2406.03138)
  - **标题**: 长语音抑郁检测中相关声学特征提取的一种基于框架的注意力解释方法
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [InaGVAD : a Challenging French TV and Radio Corpus Annotated for Speech Activity Detection and Speaker Gender Segmentation](https://arxiv.org/abs/2406.04429)
  - **标题**: Inagvad：挑战性的法国电视和电台语料库注释，用于语音活动检测和演讲者性别细分
  - **Filtered Reason**: none of cs.DL,eess.AS,cs.SD,cs.MM in whitelist
- [Introducing the Brand New QiandaoEar22 Dataset for Specific Ship Identification Using Ship-Radiated Noise](https://arxiv.org/abs/2406.04353)
  - **标题**: 引入全新的Qiandaoear22数据集，用于使用船舶辐射噪声进行特定的船舶身份证明
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Soundscape Captioning using Sound Affective Quality Network and Large Language Model](https://arxiv.org/abs/2406.05914)
  - **标题**: 使用声音情感质量网络和大语言模型的音景字幕
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [XANE: eXplainable Acoustic Neural Embeddings](https://arxiv.org/abs/2406.05199)
  - **标题**: Xane：可解释的声学神经嵌入
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Stochastic Guidance of Buoyancy Controlled Vehicles under Ice Shelves using Ocean Currents](https://arxiv.org/abs/2406.06724)
  - **标题**: 使用洋流在冰架下浮力控制车辆的随机指导
  - **Filtered Reason**: none of cs.RO in whitelist
- [Stabilized Adaptive Steering for 3D Sonar Microphone Arrays with IMU Sensor Fusion](https://arxiv.org/abs/2406.06255)
  - **标题**: 3D声纳麦克风阵列的稳定自适应转向与IMU传感器融合
  - **Filtered Reason**: none of cs.RO in whitelist
- [Broadband MEMS Microphone Arrays with Reduced Aperture Through 3D-Printed Waveguides](https://arxiv.org/abs/2406.07663)
  - **标题**: 通过3D打印的波导，宽带MEMS麦克风阵列降低了光圈
  - **Filtered Reason**: none of eess.SY,eess.AS,cs.RO,cs.SD in whitelist
- [Noise-Robust Voice Conversion by Conditional Denoising Training Using Latent Variables of Recording Quality and Environment](https://arxiv.org/abs/2406.07280)
  - **标题**: 通过记录质量和环境的潜在变量，通过有条件的降级训练来噪音噪声转换
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Low-Complexity Acoustic Scene Classification Using Parallel Attention-Convolution Network](https://arxiv.org/abs/2406.08119)
  - **标题**: 使用平行注意卷积网络的低复杂声音场景分类
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [DCASE 2024 Task 4: Sound Event Detection with Heterogeneous Data and Missing Labels](https://arxiv.org/abs/2406.08056)
  - **标题**: Dcase 2024任务4：具有异质数据和缺少标签的声音事件检测
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [FlowAVSE: Efficient Audio-Visual Speech Enhancement with Conditional Flow Matching](https://arxiv.org/abs/2406.09286)
  - **标题**: Flowavse：有条件流动匹配的有效视听语音增强
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Cascaded noise reduction and acoustic echo cancellation based on an extended noise reduction](https://arxiv.org/abs/2406.08974)
  - **标题**: 基于扩展降噪的降低降噪和消除声音的消除
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [SMRU: Split-and-Merge Recurrent-based UNet for Acoustic Echo Cancellation and Noise Suppression](https://arxiv.org/abs/2406.11175)
  - **标题**: SMRU：基于反复的分裂和合并的UNET，用于消除声音的消除和抑制噪声
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [EchoGuide: Active Acoustic Guidance for LLM-Based Eating Event Analysis from Egocentric Videos](https://arxiv.org/abs/2406.10750)
  - **标题**: Echoguide：基于LLM的饮食活动分析的主动声学指南
  - **Filtered Reason**: none of cs.HC in whitelist
- [Underwater Human-Robot and Human-Swarm Interaction: A Review and Perspective](https://arxiv.org/abs/2406.12473)
  - **标题**: 水下人类机器人和人与人之间的互动：评论和观点
  - **Filtered Reason**: none of cs.RO in whitelist
- [Improved Remixing Process for Domain Adaptation-Based Speech Enhancement by Mitigating Data Imbalance in Signal-to-Noise Ratio](https://arxiv.org/abs/2406.13982)
  - **标题**: 通过减轻信噪比的数据不平衡，改进了基于域的适应语音增强的混合过程
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Online Domain-Incremental Learning Approach to Classify Acoustic Scenes in All Locations](https://arxiv.org/abs/2406.13386)
  - **标题**: 在线域内收入学习方法在所有位置对声学场景进行分类
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [SonicSense: Object Perception from In-Hand Acoustic Vibration](https://arxiv.org/abs/2406.17932)
  - **标题**: 超声义：来自声学振动的物体感知
  - **Filtered Reason**: none of eess.AS,cs.RO,cs.SD,cs.MM in whitelist
- [Speaker-Independent Acoustic-to-Articulatory Inversion through Multi-Channel Attention Discriminator](https://arxiv.org/abs/2406.17329)
  - **标题**: 通过多渠道注意歧视器，与说话者无关的声学反转
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD,physics.bio-ph in whitelist
