# 2024-03 月度论文分类汇总

共有12篇相关领域论文, 另有18篇其他

## 人工智能(cs.AI:Artificial Intelligence)

该领域共有 1 篇论文

### Medical Speech Symptoms Classification via Disentangled Representation 
[[arxiv](https://arxiv.org/abs/2403.05000)] [[cool](https://papers.cool/arxiv/2403.05000)] [[pdf](https://arxiv.org/pdf/2403.05000)]
> **Authors**: Jianzong Wang,Pengcheng Li,Xulong Zhang,Ning Cheng,Jing Xiao
> **First submission**: 2024-03-07
> **First announcement**: 2024-03-08
> **comment**: Accepted by the 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD 2024)
- **标题**: 通过分离表示的医学语音症状分类
- **领域**: 人工智能
- **摘要**: 目的是为了理解现有作品中的口语定义。医学语音涉及的文本特征和声学特征都包含意图，这对于症状诊断很重要。在本文中，我们提出了一个名为DRSC的医学语音分类模型，该模型会自动学习从文本 - 声学数据中解散意图和内容表示形式，以进行分类。文本域和MEL光谱域的意图表示是通过意图编码提取的，然后通过两个交换获得了重建的文本特征和MEL-Spectrogram特征。将两个域的意图结合到一个联合表示形式之后，综合意图表示被送入决策层进行分类。实验结果表明，我们的模型在检测25种不同的医疗症状时获得了95％的平均准确率。

## 计算语言学(cs.CL:Computation and Language)

该领域共有 3 篇论文

### What has LeBenchmark Learnt about French Syntax? 
[[arxiv](https://arxiv.org/abs/2403.02173)] [[cool](https://papers.cool/arxiv/2403.02173)] [[pdf](https://arxiv.org/pdf/2403.02173)]
> **Authors**: Zdravko Dugonjić,Adrien Pupier,Benjamin Lecouteux,Maximin Coavoux
> **First submission**: 2024-03-04
> **First announcement**: 2024-03-05
> **comment**: Accepted to LREC-COLING 2024
- **标题**: Lebenchmark了解法语语法的知识是什么？
- **领域**: 计算语言学
- **摘要**: 该论文报告了一系列旨在探测Lebenchmark的实验，该实验是一种经过验证的声学模型，该模型在7k小时的法语中训练，以获取句法信息。预处理的声学模型越来越多地用于下游语音任务，例如自动语音识别，语音翻译，口语理解或语音解析。他们接受了非常低级别的信息（原始语音信号）的培训，并且没有明确的词汇知识。尽管如此，他们还是在需要更高级别的语言知识的任务上获得了合理的结果。结果，新出现的问题是这些模型是否编码句法信息。我们使用OrféoTreebank探测了Lebenchmark的每个表示层，并观察到它已经学习了一些句法信息。我们的结果表明，句法信息更容易从网络的中层中提取，之后观察到非常急剧的降低。

### TopicDiff: A Topic-enriched Diffusion Approach for Multimodal Conversational Emotion Detection 
[[arxiv](https://arxiv.org/abs/2403.04789)] [[cool](https://papers.cool/arxiv/2403.04789)] [[pdf](https://arxiv.org/pdf/2403.04789)]
> **Authors**: Jiamin Luo,Jingjing Wang,Guodong Zhou
> **First submission**: 2024-03-04
> **First announcement**: 2024-03-06
> **comment**: No comments
- **标题**: 主题降：多模式对话情感检测的富含主题的扩散方法
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 多模式对话情绪（MCE）检测通常跨越声学，视觉和语言方式，引起了人们对多媒体社区的兴趣越来越多。先前的研究主要集中在对话中学习上下文信息，仅考虑单语言方式的主题信息，同时始终忽略声学和视觉主题信息。在此基础上，我们提出了一种模型不合时宜的主题扩散（topicDiff）方法，用于捕获MCE任务中的多模式主题信息。特别是，我们将扩散模型集成到神经主题模型中，以减轻捕获主题信息的神经主题模型的多样性缺陷问题。详细的评估表明，主题降价对最先进的MCE基线的显着改善，证明了多模式主题信息对MCE的重要性以及主题降价在捕获此类信息方面的有效性。此外，我们观察到一个有趣的发现，与语言相比，声学和视觉中的主题信息更具歧视性和健壮性。

### A Multimodal Approach to Device-Directed Speech Detection with Large Language Models 
[[arxiv](https://arxiv.org/abs/2403.14438)] [[cool](https://papers.cool/arxiv/2403.14438)] [[pdf](https://arxiv.org/pdf/2403.14438)]
> **Authors**: Dominik Wagner,Alexander Churchill,Siddharth Sigtia,Panayiotis Georgiou,Matt Mirsamadi,Aarshee Mishra,Erik Marchi
> **First submission**: 2024-03-21
> **First announcement**: 2024-03-22
> **comment**: arXiv admin note: text overlap with arXiv:2312.03632
- **标题**: 使用大语言模型的多模式导向语音检测
- **领域**: 计算语言学,机器学习,音频和语音处理
- **摘要**: 与虚拟助手的交互通常以预定义的触发短语开始，然后是用户命令。为了使与助手更直观的互动，我们探讨是否可以删除用户必须使用触发短语开始每个命令的要求。我们以三种方式探索此任务：首先，我们仅使用从音频波形获得的声学信息训练分类器。其次，我们将自动语音识别（ASR）系统的解码器输出（例如1好的假设）作为大语言模型（LLM）的输入功能。最后，我们探索了一个多模式系统，该系统结合了声学和词汇特征，以及LLM中的ASR解码器信号。使用多模式信息可产生相对相等的率改进，而相对于仅文本和仅音频的模型的相对率改进高达39％和61％。增加LLM的大小和低级适应性的训练会导致我们的数据集的相对相对降低高达18％。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 2 篇论文

### A Density-Guided Temporal Attention Transformer for Indiscernible Object Counting in Underwater Video 
[[arxiv](https://arxiv.org/abs/2403.03461)] [[cool](https://papers.cool/arxiv/2403.03461)] [[pdf](https://arxiv.org/pdf/2403.03461)]
> **Authors**: Cheng-Yen Yang,Hsiang-Wei Huang,Zhongyu Jiang,Hao Wang,Farron Wallace,Jenq-Neng Hwang
> **First submission**: 2024-03-05
> **First announcement**: 2024-03-06
> **comment**: Accepted by ICASSP 2024 (IEEE International Conference onAcoustics, Speech, andSignalProcessing)
- **标题**: 在水下视频中计数不可见的对象的密度引导的时间注意变压器
- **领域**: 计算机视觉和模式识别
- **摘要**: 由于视觉社区的最新发展，密集的对象计数或人群计数已经走了很长一段路。但是，旨在计算与周围环境混合的目标数量的不可分化的对象计数一直是一个挑战。基于图像的对象计数数据集已成为当前公开可用数据集的主流。因此，我们提出了一个称为YouTubeFish-35的大规模数据集，其中包含35个序列的高清视频序列，每秒较高的帧高和超过150,000个带注释的中心点在选定的各种场景中。为了进行基准测试，我们选择了三种主流方法来进行密集对象计数，并在新收集的数据集中仔细评估它们。我们提出了TransVidCount，这是一种新的强基线，在统一的框架中结合了沿时间域的密度和回归分支，可以有效地解决不可见的对象对YouTbefish-35数据集的最先进性能。

### Knowledge Distillation in YOLOX-ViT for Side-Scan Sonar Object Detection 
[[arxiv](https://arxiv.org/abs/2403.09313)] [[cool](https://papers.cool/arxiv/2403.09313)] [[pdf](https://arxiv.org/pdf/2403.09313)]
> **Authors**: Martin Aubard,László Antal,Ana Madureira,Erika Ábrahám
> **First submission**: 2024-03-14
> **First announcement**: 2024-03-15
> **comment**: No comments
- **标题**: Yolox-Vit中的知识蒸馏侧扫描声纳对象检测
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在本文中，我们提出了Yolox-Vit，这是一种新型的对象检测模型，并研究了知识蒸馏对模型尺寸降低的功效而无需牺牲性能。我们的研究专注于水下机器人技术，解决了有关较小模型的可行性以及视觉变压器层在Yolox中的影响的关键问题。此外，我们引入了一个新的侧扫描声纳图像数据集，并使用它来评估对象探测器的性能。结果表明，知识蒸馏有效地降低了墙检测中的假阳性。此外，引入的视觉变压器层显着提高了水下环境中的对象检测准确性。 Yolox-Vit中知识蒸馏的源代码位于https://github.com/remaro-network/kd-yolox-vit。

## 机器学习(cs.LG:Machine Learning)

该领域共有 2 篇论文

### MPIPN: A Multi Physics-Informed PointNet for solving parametric acoustic-structure systems 
[[arxiv](https://arxiv.org/abs/2403.01132)] [[cool](https://papers.cool/arxiv/2403.01132)] [[pdf](https://arxiv.org/pdf/2403.01132)]
> **Authors**: Chu Wang,Jinhong Wu,Yanzhi Wang,Zhijian Zha,Qi Zhou
> **First submission**: 2024-03-02
> **First announcement**: 2024-03-04
> **comment**: The number of figures is 16. The number of tables is 5. The number of words is 9717
- **标题**: MPIPN：用于求解参数声学系统的多物理信息点网
- **领域**: 机器学习,声音,音频和语音处理
- **摘要**: 机器学习用于解决由一般非线性偏微分方程（PDE）管辖的物理系统。但是，复杂的多物理系统（例如声学结构耦合）通常由一系列包含可变物理量的PDE描述，这些PDE被称为参数系统。缺乏解决由涉及明确和隐式数量的PDE管辖的参数系统的策略。在本文中，提出了一个基于深度学习的多物理学意识到的点网（MPIPN）来求解参数的声学结构系统。首先，MPIPN诱导了增强的点云架构，该体系结构涵盖了计算域的明确物理量和几何特征。然后，MPIPN分别将重建点云的本地和全局特征提取为解决参数系统标准的一部分。此外，隐式物理量是通过编码技术作为解决标准的另一部分嵌入的。最后，将表征参数系统的所有求解标准合并为形成独特的序列，作为MPIPN的输入，其输出是系统的解决方案。所提出的框架是通过适应物理学的损失函数来训练的，适用于相应的计算域。该框架被概括为处理系统的新参数条件。通过将其应用于求解由Helmholtz方程控制的稳定参数的原声结构耦合系统来验证MPIPN的有效性。已经实施了消融实验，以证明具有少数监督数据的物理信息影响的功效。所提出的方法在恒定的参数条件下以及声学系统的参数条件的可更改组合均可在所有计算域中获得合理的精确度。

### HeAR -- Health Acoustic Representations 
[[arxiv](https://arxiv.org/abs/2403.02522)] [[cool](https://papers.cool/arxiv/2403.02522)] [[pdf](https://arxiv.org/pdf/2403.02522)]
> **Authors**: Sebastien Baur,Zaid Nabulsi,Wei-Hung Weng,Jake Garrison,Louis Blankemeier,Sam Fishman,Christina Chen,Sujay Kakarmath,Minyoi Maimbolwa,Nsala Sanjase,Brian Shuma,Yossi Matias,Greg S. Corrado,Shwetak Patel,Shravya Shetty,Shruthi Prabhakara,Monde Muyoyeta,Diego Ardila
> **First submission**: 2024-03-04
> **First announcement**: 2024-03-05
> **comment**: 4 tables, 4 figures, 6 supplementary tables, 3 supplementary figures
- **标题**: 听到 - 健康声明
- **领域**: 机器学习,人工智能
- **摘要**: 众所周知，诸如咳嗽和呼吸之类的健康声音会包含有用的健康信号，具有监测健康和疾病的重要潜力，但在医疗机器学习社区中却没有忽视。现有的用于健康声学的深度学习系统通常受到狭义的培训和评估，并在一项任务上受到数据的限制，并可能阻碍对其他任务的概括。为了减轻这些差距，我们开发了一种可扩展的自我监督的基于学习的深度学习系统，该系统使用掩盖的自动编码器，该系统在大型数据集中训练有3.13亿个两秒长的音频剪辑。通过线性探针，我们在6个数据集的33个健康声学任务的基准上建立了一种最先进的健康音频嵌入模型。通过介绍这项工作，我们希望能够启用和加速进一步的健康声学研究。

## 声音(cs.SD:Sound)

该领域共有 3 篇论文

### Probing the Information Encoded in Neural-based Acoustic Models of Automatic Speech Recognition Systems 
[[arxiv](https://arxiv.org/abs/2402.19443)] [[cool](https://papers.cool/arxiv/2402.19443)] [[pdf](https://arxiv.org/pdf/2402.19443)]
> **Authors**: Quentin Raymondaud,Mickael Rouvier,Richard Dufour
> **First submission**: 2024-02-29
> **First announcement**: 2024-03-01
> **comment**: No comments
- **标题**: 探测自动语音识别系统基于神经的声学模型中编码的信息
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在许多研究领域，深度学习体系结构在绩效方面取得了重大进展。因此，自动语音识别（ASR）领域从这些科学和技术进步中受益，特别是对于声学建模，现在整合了深层神经网络体系结构。但是，这些性能收益已转化为通过这些黑盒架构所学的信息和传达的信息的增加。经过神经网络可解释性的许多研究，我们在本文中提出了一种旨在确定信息在ASR声学模型（AM）中的方案。为此，我们建议使用中间表示（在此，在不同的层级别上）评估确定的一组任务的AM性能。关于性能变化和目标任务，我们可以在不同的体系结构步骤中提高或扰动哪些信息进行假设。在说话者验证，声学环境分类，性别分类，速度降低检测系统和语音情感/情感识别上进行实验。分析表明，基于神经的AMS拥有异质信息，这些信息似乎与音素识别（例如情感，情感或说话者身份）无关。全球低级隐藏层似乎对信息的结构有用，而上层则倾向于删除无用的信息以获得音素识别。

### An AI-Driven Approach to Wind Turbine Bearing Fault Diagnosis from Acoustic Signals 
[[arxiv](https://arxiv.org/abs/2403.09030)] [[cool](https://papers.cool/arxiv/2403.09030)] [[pdf](https://arxiv.org/pdf/2403.09030)]
> **Authors**: Zhao Wang,Xiaomeng Li,Na Li,Longlong Shu
> **First submission**: 2024-03-13
> **First announcement**: 2024-03-14
> **comment**: No comments
- **标题**: 通过声信号诊断的风力涡轮机的AI驱动方法
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: 这项研究旨在开发一种深度学习模型，以通过声音信号中风力涡轮机发生轴承断层分类。通过使用五种预定义故障类型的音频数据进行训练和验证，成功构建和培训了卷积LSTM模型。为了创建数据集，收集了原始音频信号数据并在框架中处理以捕获时间和频域信息。该模型在训练样本上表现出杰出的精度，并在验证过程中表现出了出色的概括能力，表明其具有概括能力的能力。在测试样本上，该模型的分类性能出色，总体准确性超过99.5％，正常状态的假正率小于1％。这项研究的发现为风力涡轮机发电机中轴承断层的诊断和维持提供了必不可少的支持，有可能提高风能发电的可靠性和效率。

### Multitask frame-level learning for few-shot sound event detection 
[[arxiv](https://arxiv.org/abs/2403.11091)] [[cool](https://papers.cool/arxiv/2403.11091)] [[pdf](https://arxiv.org/pdf/2403.11091)]
> **Authors**: Liang Zou,Genwei Yan,Ruoyu Wang,Jun Du,Meng Lei,Tian Gao,Xin Fang
> **First submission**: 2024-03-17
> **First announcement**: 2024-03-18
> **comment**: 6 pages, 4 figures, conference
- **标题**: 多任务框架级学习，用于几次声音事件检测
- **领域**: 声音,计算机视觉和模式识别,音频和语音处理
- **摘要**: 本文重点介绍了几个声音事件检测（SED），该检测旨在自动识别和对样本有限的声音事件进行分类。但是，几乎没有弹出的主要方法主要依赖于细分级预测，这些预测通常提供详细的，细粒度的预测，尤其是对于短暂持续时间的事件。尽管已经提出了框架级别的预测策略来克服这些局限性，但这些策略通常面临着由背景噪声引起的预测截断的困难。为了减轻此问题，我们引入了创新的多任务框架级SED框架。此外，我们推出了TimeFilteraug，这是一种用于数据增强的线性正时面罩，以提高模型对不同声学环境的鲁棒性和适应性。所提出的方法达到了63.8％的F评分，在声学场景和事件的检测和分类挑战2023的检测和分类的几个射击生物声学事件检测类别中获得了第一名。

## 信号处理(eess.SP:Signal Processing)

该领域共有 1 篇论文

### Deep Learning based acoustic measurement approach for robotic applications on orthopedics 
[[arxiv](https://arxiv.org/abs/2403.05879)] [[cool](https://papers.cool/arxiv/2403.05879)] [[pdf](https://arxiv.org/pdf/2403.05879)]
> **Authors**: Bangyu Lan,Momen Abayazid,Nico Verdonschot,Stefano Stramigioli,Kenan Niu
> **First submission**: 2024-03-09
> **First announcement**: 2024-03-11
> **comment**: No comments
- **标题**: 针对骨科的机器人应用的基于深度学习的声学测量方法
- **领域**: 信号处理,机器学习,机器人技术
- **摘要**: 总共膝关节置换术（TKA），手术机器人技术可以提供图像引导的导航，以高精度地拟合植入物。它的跟踪方法高度依赖于将骨针插入光学跟踪系统跟踪的骨骼中。通常，这是通过侵入性的，辐射的方式（可植入的标记和CT扫描）来完成的，这会引入不必要的创伤并延长患者的准备时间。为了解决这个问题，基于超声的骨跟踪可以提供替代方案。在这项研究中，我们提出了一种新颖的深度学习结构，以通过A模式超声（US）提高骨跟踪的准确性。我们首先从尸体实验中获得了一组超声数据集，其中使用骨针计算了骨骼的地面真相位置。这些数据用于训练所提出的Casatt-Unet，以自动，稳健地预测骨骼位置。地面真相骨位置和我们的那些位置同时记录了。因此，我们可以在原始的美国信号中标记骨峰。结果，我们的方法在所有八个骨区域中都达到了亚毫米精度，脚踝中的一个通道唯一。该方法可以从1D RAW超声信号中对下肢骨位置进行稳健的测量。从安全，方便和有效的角度来看，它显示出在骨科手术中应用A模式超声波的巨大潜力。

## 其他论文

共有 18 篇其他论文

- [Robust Wake Word Spotting With Frame-Level Cross-Modal Attention Based Audio-Visual Conformer](https://arxiv.org/abs/2403.01700)
  - **标题**: 通过框架级别的跨模式注意的稳健的唤醒单词斑点基于视听构象异构体
  - **Filtered Reason**: none of eess.AS,cs.SD,cs.MM in whitelist
- [6DoF SELD: Sound Event Localization and Detection Using Microphones and Motion Tracking Sensors on self-motioning human](https://arxiv.org/abs/2403.01670)
  - **标题**: 6DOF SELD：使用麦克风和运动跟踪传感器在自动化人类上进行的声音事件定位和检测
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [PAVITS: Exploring Prosody-aware VITS for End-to-End Emotional Voice Conversion](https://arxiv.org/abs/2403.01494)
  - **标题**: 铺路：探索韵律意识的VIT，以进行端到端的情感语音转换
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Design and Performance Evaluation of SEANet, a Software-defined Networking Platform for the Internet of Underwater Things](https://arxiv.org/abs/2403.01009)
  - **标题**: Seanet的设计和绩效评估，Seanet是一个软件定义的网络平台，用于水下互联网
  - **Filtered Reason**: none of cs.NI,eess.SP in whitelist
- [Robust Online Epistemic Replanning of Multi-Robot Missions](https://arxiv.org/abs/2403.00641)
  - **标题**: 强大的在线认知重建多机器人任务
  - **Filtered Reason**: none of cs.RO in whitelist
- [The Impact of Frequency Bands on Acoustic Anomaly Detection of Machines using Deep Learning Based Model](https://arxiv.org/abs/2403.00379)
  - **标题**: 使用基于深度学习的模型，频带对机器的声学异常检测的影响
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [SonoTraceLab -- A Raytracing-Based Acoustic Modelling System for Simulating Echolocation Behavior of Bats](https://arxiv.org/abs/2403.06847)
  - **标题**: Sonotracelab-一种基于射线疗法的声学建模系统，用于模拟蝙蝠的回声定位行为
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [The Neural-SRP method for positional sound source localization](https://arxiv.org/abs/2403.09455)
  - **标题**: 位置声源定位的神经SRP方法
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [CoPlay: Audio-agnostic Cognitive Scaling for Acoustic Sensing](https://arxiv.org/abs/2403.10796)
  - **标题**: Coplay：声学感测的音频 - 反应性缩放
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Fully Distributed Cooperative Multi-agent Underwater Obstacle Avoidance](https://arxiv.org/abs/2403.10759)
  - **标题**: 完全分布的合作多代理水下障碍
  - **Filtered Reason**: none of cs.RO in whitelist
- [Virtual Elastic Tether: a New Approach for Multi-agent Navigation in Confined Aquatic Environments](https://arxiv.org/abs/2403.10629)
  - **标题**: 虚拟弹性束缚：一种在狭窄的水生环境中进行多代理导航的新方法
  - **Filtered Reason**: none of eess.SY,cs.RO in whitelist
- [Collaborative Aquatic Positioning System Utilising Multi-beam Sonar and Depth Sensors](https://arxiv.org/abs/2403.10397)
  - **标题**: 使用多光束声纳和深度传感器的协作水生定位系统
  - **Filtered Reason**: none of cs.RO in whitelist
- [Opti-Acoustic Semantic SLAM with Unknown Objects in Underwater Environments](https://arxiv.org/abs/2403.12837)
  - **标题**: 在水下环境中，具有未知物体的光声语义大满贯
  - **Filtered Reason**: none of cs.RO in whitelist
- [Reproducing the Acoustic Velocity Vectors in a Circular Listening Area](https://arxiv.org/abs/2403.12630)
  - **标题**: 在圆形聆听区域重现声速向量
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [KunquDB: An Attempt for Speaker Verification in the Chinese Opera Scenario](https://arxiv.org/abs/2403.13356)
  - **标题**: KunqudB：在中国歌剧场景中进行演讲者验证的尝试
  - **Filtered Reason**: none of eess.AS,cs.SD,eess.IV in whitelist
- [ACCESS: Assurance Case Centric Engineering of Safety-critical Systems](https://arxiv.org/abs/2403.15236)
  - **标题**: 访问：保证案件以安全至关重要系统的为中心工程
  - **Filtered Reason**: none of cs.SE in whitelist
- [Speaker Distance Estimation in Enclosures from Single-Channel Audio](https://arxiv.org/abs/2403.17514)
  - **标题**: 单渠道音频的外壳中的扬声器距离估算
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [A Robust UWOC-assisted Multi-hop Topology for Underwater Sensor Network Nodes](https://arxiv.org/abs/2403.19180)
  - **标题**: 水下传感器网络节点的强大UWOC辅助多跳拓扑
  - **Filtered Reason**: none of eess.SP,cs.ET in whitelist
