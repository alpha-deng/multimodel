# 2024-05 月度论文分类汇总

共有16篇相关领域论文, 另有19篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 2 篇论文

### Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models 
[[arxiv](https://arxiv.org/abs/2405.06134)] [[cool](https://papers.cool/arxiv/2405.06134)] [[pdf](https://arxiv.org/pdf/2405.06134)]
> **Authors**: Vyas Raina,Rao Ma,Charles McGhee,Kate Knill,Mark Gales
> **First submission**: 2024-05-09
> **First announcement**: 2024-05-10
> **comment**: No comments
- **标题**: 静音耳语：通用的声学对抗性攻击语音基础模型
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 大型语音基础模型（如Whisper）的最新发展导致它们在许多自动语音识别（ASR）应用中广泛使用。这些系统将“特殊令牌”包含在其词汇中，例如$ \ texttt {<| endoftext |>} $，以指导其语言生成过程。但是，我们证明可以通过对抗性攻击来操纵模型的行为来利用这些令牌。我们提出了一种简单而有效的方法，以了解Whisper的$ \ texttt {<| endoftext |>} $令牌的普遍声学实现，当该$上的任何语音信号预先使用时，它会鼓励该模型忽略演讲并仅转录特殊的标志，并有效地“有效地”模型。我们的实验表明，相同的，通用的0.64秒的对手音频段可以成功地将超过97 \％的语音样本静音目标耳语ASR模型。此外，我们发现这个通用的对手音频细分通常会将其转移到新的数据集和任务。总体而言，这项工作表明了耳语模型易受“变形”对抗性攻击的脆弱性，在这种情况下，这种攻击可以在现实世界中构成风险和潜在的好处：例如，该攻击可用于绕过语音调节系统，或者相反，也可以使用攻击来保护私人语音数据。

### A predictive learning model can simulate temporal dynamics and context effects found in neural representations of continuous speech 
[[arxiv](https://arxiv.org/abs/2405.08237)] [[cool](https://papers.cool/arxiv/2405.08237)] [[pdf](https://arxiv.org/pdf/2405.08237)]
> **Authors**: Oli Danyi Liu,Hao Tang,Naomi Feldman,Sharon Goldwater
> **First submission**: 2024-05-13
> **First announcement**: 2024-05-14
> **comment**: Accepted to CogSci 2024
- **标题**: 预测学习模型可以模拟连续语音的神经表示中发现的时间动态和上下文效应
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 语音感知涉及存储和集成顺序呈现的项目。认知神经科学方面的最新工作已经确定了人类对语音的神经编码的时间和上下文特征，这可能有助于这种时间处理。在这项研究中，我们模拟了类似的分析，该分析是从未经标记的语音训练的计算模型中提取的表示，并以预测即将到来的声学的学习目标进行了培训。我们的模拟揭示了与大脑信号中类似的时间动力学，这意味着这些特性可以在没有语言知识的情况下出现。大脑和模型之间共享的另一个属性是，音素的编码模式支持一定程度的跨文本概括。但是，我们发现证据表明这些概括的有效性取决于特定的上下文，这表明仅此分析不足以支持上下文不变编码的存在。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 1 篇论文

### A Calibration Tool for Refractive Underwater Vision 
[[arxiv](https://arxiv.org/abs/2405.18018)] [[cool](https://papers.cool/arxiv/2405.18018)] [[pdf](https://arxiv.org/pdf/2405.18018)]
> **Authors**: Felix Seegräber,Mengkun She,Felix Woelk,Kevin Köser
> **First submission**: 2024-05-28
> **First announcement**: 2024-05-29
> **comment**: 7 pages, 5 figures, the paper is submitted to the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)
- **标题**: 用于折射水下视觉的校准工具
- **领域**: 计算机视觉和模式识别
- **摘要**: 许多依赖视觉传感器的水下机器人应用都需要适当的相机校准，即知道图像中每个像素的传入光线。对于理想的针孔摄像机模型，所有观看射线都以一个3D点相交，而水下相机可能会在水，玻璃和空气的界面处遭受 - 可能多个光线的折射。方向的这些变化取决于相机在防水外壳内的位置和方向，以及光学窗口，端口本身的形状和特性。近年来，已经提出了在常见端口（例如平坦或圆顶端口）背后的水下视觉模型，但是水下社区仍缺乏校准工具，该工具可以通过折射校准来确定端口参数。在这项工作中，我们提供了第一个开源实现水下折射摄像头校准工具箱。它允许对水下视觉系统的端到端校准，包括具有圆顶或平坦端口的系统的相机，立体声和外壳校准。使用渲染的数据集和现实世界实验验证了实现。

## 机器学习(cs.LG:Machine Learning)

该领域共有 2 篇论文

### End-to-End Autoencoder for Drill String Acoustic Communications 
[[arxiv](https://arxiv.org/abs/2405.03840)] [[cool](https://papers.cool/arxiv/2405.03840)] [[pdf](https://arxiv.org/pdf/2405.03840)]
> **Authors**: Iurii Lezhenin,Aleksandr Sidnev,Vladimir Tsygan,Igor Malyshev
> **First submission**: 2024-05-06
> **First announcement**: 2024-05-07
> **comment**: No comments
- **标题**: 钻弦声通信的端到端自动编码器
- **领域**: 机器学习,信号处理
- **摘要**: 钻头通信对于钻孔效率和安全性很重要。具有高吞吐量和可靠性的低潜伏期钻头通信系统的设计仍然是一个开放的挑战。在本文中，基于深度学习自动编码器（AE）的端到端通信系统，在其中提出了用于Acousticdrill String Communications的发射器和接收器作为Feed向前神经网络实现的。仿真表明，AE系统能够在BER和PAPR方面胜过基线非连接的OFDM系统，并且延迟较低。

### ISR: Invertible Symbolic Regression 
[[arxiv](https://arxiv.org/abs/2405.06848)] [[cool](https://papers.cool/arxiv/2405.06848)] [[pdf](https://arxiv.org/pdf/2405.06848)]
> **Authors**: Tony Tohme,Mohammad Javad Khojasteh,Mohsen Sadr,Florian Meyer,Kamal Youcef-Toumi
> **First submission**: 2024-05-10
> **First announcement**: 2024-05-13
> **comment**: No comments
- **标题**: ISR：可逆符号回归
- **领域**: 机器学习,人工智能,信息论,机器学习
- **摘要**: 我们引入了可逆符号回归（ISR）方法。这是一种机器学习技术，可通过可逆地图（或架构）在给定数据集的输入和输出之间产生分析关系。提出的ISR方法自然结合了可逆神经网络（INNS）和方程学习者（EQL）的原理，这是一种基于神经网络的符号架构，用于功能学习。特别是，我们将旅馆的仿射耦合块转换为符号框架，从而导致端到端可区分的符号可逆体系结构，从而允许有效的基于梯度的学习。拟议的ISR框架还依赖于促进正则化的稀疏性，从而发现简洁而可解释的可逆表达式。我们表明，ISR可以作为密度估计任务的（符号）正常化流量。此外，我们强调了它在解决反问题方面的实际适用性，包括基准逆运动学问题，值得注意的是，海洋学中旨在推断来自声学信号的海底参数的后验分布中的地球倒数问题。

## 机器人技术(cs.RO:Robotics)

该领域共有 2 篇论文

### A Sonar-based AUV Positioning System for Underwater Environments with Low Infrastructure Density 
[[arxiv](https://arxiv.org/abs/2405.01971)] [[cool](https://papers.cool/arxiv/2405.01971)] [[pdf](https://arxiv.org/pdf/2405.01971)]
> **Authors**: Emilio Olivastri,Daniel Fusaro,Wanmeng Li,Simone Mosco,Alberto Pretto
> **First submission**: 2024-05-03
> **First announcement**: 2024-05-06
> **comment**: Accepted to the IEEE ICRA Workshop on Field Robotics 2024
- **标题**: 基于声纳的AUV定位系统，用于低基础设施密度的水下环境
- **领域**: 机器人技术,计算机视觉和模式识别
- **摘要**: 对水下车辆的需求不断增长，强调了在检查任务中进行强大定位解决方案的必要性。在这项工作中，我们提出了一种新型的基于Sonar的水下全球定位算法，用于AUVS（自动水下车辆），专为人工资产分布稀疏的环境而设计。我们的方法利用了将两个协同数据解释的前端应用于由多片前看起来的声纳（FSD）获取的同一声流数据流的前端。这些观察结果被融合在粒子滤波器（PF）中，以称量属于高象征区域或解决对称歧义的更多粒子。在类似于实际水下厂的模拟环境上进行的初步实验提供了令人鼓舞的结果。这项工作代表了该方法未来发展的起点，在现实世界中也进行了详尽的评估。

### EchoPT: A Pretrained Transformer Architecture that Predicts 2D In-Air Sonar Images for Mobile Robotics 
[[arxiv](https://arxiv.org/abs/2405.12573)] [[cool](https://papers.cool/arxiv/2405.12573)] [[pdf](https://arxiv.org/pdf/2405.12573)]
> **Authors**: Jan Steckel,Wouter Jansen,Nico Huebel
> **First submission**: 2024-05-21
> **First announcement**: 2024-05-22
> **comment**: No comments
- **标题**: ECHOPT：预处理的变压器体系结构，可预测移动机器人技术的2D空气声纳图像
- **领域**: 机器人技术,机器学习,信号处理,系统与控制
- **摘要**: 预测性大脑假设表明，感知可以解释为最大程度地减少内部世界模型产生的预测感知令牌和实际感觉输入令牌之间的误差的过程。在实施该假设的工作示例时，由于控制超声传感的反射模型的稀疏性质，出现了重大困难。尽管面临这些挑战，但使用声纳数据创建一致的世界模型对于在机器人技术中实施超声数据的预测处理至关重要。为了努力使用超声波作为唯一的外部感受传感器模态实现强大的机器人行为，本文介绍了Echopt，这是一种验证的变压器体系结构，旨在预测先前的感觉数据和机器人自我动感信息的2D声纳图像。我们详细介绍了驱动Echopt并将模型的性能与几种最新技术的性能进行比较的变压器体系结构。除了介绍和评估我们的ECHOPT模型外，我们还证明了在两个机器人任务中这种预测感知方法的有效性。

## 声音(cs.SD:Sound)

该领域共有 8 篇论文

### Who is Authentic Speaker 
[[arxiv](https://arxiv.org/abs/2405.00248)] [[cool](https://papers.cool/arxiv/2405.00248)] [[pdf](https://arxiv.org/pdf/2405.00248)]
> **Authors**: Qiang Huang
> **First submission**: 2024-04-30
> **First announcement**: 2024-05-01
> **comment**: No comments
- **标题**: 谁是真实的演讲者
- **领域**: 声音,人工智能,多媒体,音频和语音处理
- **摘要**: 使用深度学习技术的语音转换（VC）现在可以产生高质量的一对一声音，因此已在某些实用的应用程序领域（例如娱乐和医疗保健）中使用。但是，当用操纵声音用于欺骗性目的时，语音转换可能会构成潜在的社会问题。此外，由于源说话者的声学特征发生了很大的变化，因此找到来自转换后的声音的真实演讲者是一个巨大的挑战。在本文中，我们试图探索从转换的声音中识别真实的说话者的可行性。这项研究的假设是，即使来自源说话者的某些信息仍然存在，即使他们的声音转换为不同的目标声音。因此，我们的实验旨在鉴于转换后的声音识别源说话者，这些声音是通过在源和目标扬声器的随机配对的话语上使用fragmentVc生成的。为了提高针对转化的声音的鲁棒性，我们的识别模型是通过在深神经网络中使用局部汇总描述符（VLAD）的层次向量构建的。真实的说话者识别系统主要在两个方面进行测试，包括转换后的声音质量和VLAD的变化的影响。这项工作中使用的数据集是VCTK语料库，其中源和目标扬声器是随机配对的。在转换的话语中获得的结果表明，从转换的声音中识别出真实的扬声器方面表现出色。

### SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound 
[[arxiv](https://arxiv.org/abs/2405.00233)] [[cool](https://papers.cool/arxiv/2405.00233)] [[pdf](https://arxiv.org/pdf/2405.00233)]
> **Authors**: Haohe Liu,Xuenan Xu,Yi Yuan,Mengyue Wu,Wenwu Wang,Mark D. Plumbley
> **First submission**: 2024-04-30
> **First announcement**: 2024-05-01
> **comment**: Accepted by Journal of Selected Topics inSignalProcessing (JSTSP). Demo and code: https://haoheliu.github.io/SemantiCodec/
- **标题**: Semanticodec：一般声音的超低比特率语义音频编解码器
- **领域**: 声音,人工智能,多媒体,音频和语音处理,信号处理
- **摘要**: 大型语言模型（LLMS）通过音频编解码器具有显着高级的音频处理，这些音频编解码器将音频转换为离散令牌，从而使语言建模技术应用于音频数据。但是，传统的编解码器通常在高比特率或狭窄领域（例如语音）中运行，并且缺乏有效语言建模所需的语义线索。在解决这些挑战时，我们介绍了Semanticodec，这是一种新颖的编解码器，旨在将音频压缩为每秒不到一百个标记，包括语音，一般声音和音乐，而不会损害质量。 Semanticodec具有双编码器体系结构：使用自我监视的预训练的预训练的音频掩盖自动编码器（Audiomae）的语义编码器，使用K-Means聚类在广泛的音频数据上离散，并在广泛的音频数据上进行离散，并使用声学编码器来捕获剩余的细节。语义和声学编码器输出用于通过基于扩散模型的解码器重建音频。 Semanticodec以三种变体呈现，令牌速率为每秒25、50和100，支持在0.31 kbps和1.40 kbps之间的一系列超低比率。实验结果表明，Semanticodec在重建质量上的最先进描述编解码器明显胜过。我们的结果还表明，Semanticodec包含的语义信息明显比所有评估的最先进的音频编解码器，即使在比特率明显较低的情况下。我们的代码和演示可从https://haoheliu.github.io/semanticodec/获得。

### Deep Space Separable Distillation for Lightweight Acoustic Scene Classification 
[[arxiv](https://arxiv.org/abs/2405.03567)] [[cool](https://papers.cool/arxiv/2405.03567)] [[pdf](https://arxiv.org/pdf/2405.03567)]
> **Authors**: ShuQi Ye,Yuan Tian
> **First submission**: 2024-05-06
> **First announcement**: 2024-05-07
> **comment**: No comments
- **标题**: 深空可分离蒸馏，用于轻质声学场景分类
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在现实世界中，声学场景分类（ASC）非常重要。最近，基于深度学习的方法已被广泛用于声学场景分类。但是，这些方法目前不够轻巧，而且性能并不令人满意。为了解决这些问题，我们提出了一个可分开的蒸馏网络。首先，该网络在日志频谱图上执行高低频率分解，在维持模型性能的同时显着降低了计算复杂性。其次，我们专门为ASC设计了三个轻型操作员，包括可分离卷积（SC），正顺序分离卷积（OSC）和可分离的部分卷积（SPC）。这些操作员在声学场景分类任务中具有高效的功能提取功能。实验结果表明，与当前流行的深度学习方法相比，所提出的方法的性能增长为9.8％，同时也具有较小的参数计数和计算复杂性。

### FastSAG: Towards Fast Non-Autoregressive Singing Accompaniment Generation 
[[arxiv](https://arxiv.org/abs/2405.07682)] [[cool](https://papers.cool/arxiv/2405.07682)] [[pdf](https://arxiv.org/pdf/2405.07682)]
> **Authors**: Jianyi Chen,Wei Xue,Xu Tan,Zhen Ye,Qifeng Liu,Yike Guo
> **First submission**: 2024-05-13
> **First announcement**: 2024-05-14
> **comment**: IJCAI 2024
- **标题**: FastSag：迈向快速非自动回忆唱歌伴奏
- **领域**: 声音,人工智能,计算语言学,多媒体,音频和语音处理
- **摘要**: 伴奏伴奏（SAG）产生乐器音乐以伴随输入人声，这对于开发人类的共生艺术创造系统至关重要。 Singong的最先进方法利用了SAG的多阶段自回归（AR）模型，但是，由于它会递归生成语义和声学令牌，因此此方法非常慢，这使得它无法用于实时应用程序。在本文中，我们旨在开发一种可以创建高质量且连贯的伴奏的快速下垂方法。开发了基于非AR扩散的框架，该框架通过仔细设计从声带推论的条件直接生成目标伴奏的MEL频谱图。通过扩散和MEL频谱建模，所提出的方法显着简化了基于AR令牌的Singsong框架，并在很大程度上加速了生成。我们还设计语义投影，先前的投影块以及一组损失函数，以确保生成的伴奏具有语义和节奏与人声信号的连贯性。通过密集的实验研究，我们证明所提出的方法可以产生比Singsong更好的样品，并至少加速30次。音频示例和代码可在https://fastsag.github.io/上找到。

### A Novel Fusion Architecture for PD Detection Using Semi-Supervised Speech Embeddings 
[[arxiv](https://arxiv.org/abs/2405.17206)] [[cool](https://papers.cool/arxiv/2405.17206)] [[pdf](https://arxiv.org/pdf/2405.17206)]
> **Authors**: Tariq Adnan,Abdelrahman Abdelkader,Zipei Liu,Ekram Hossain,Sooyong Park,MD Saiful Islam,Ehsan Hoque
> **First submission**: 2024-05-21
> **First announcement**: 2024-05-24
> **comment**: 31 pages, 6 figures, and 8 tables
- **标题**: 使用半监督的语音嵌入用于PD检测的新型融合体系结构
- **领域**: 声音,机器学习
- **摘要**: 我们提出了一个框架，通过使用来自不同录音设置和环境（包括参与者家）的Web应用程序收集的英语pangram语音演讲来识别帕金森氏病（PD）。我们的数据集包括1306名参与者的全球队列，其中392名被诊断为PD。利用数据集的多样性，涵盖了各种人口统计学特性（例如年龄，性别和种族），我们使用了从半监督模型（例如WAV2VEC 2.0，wavlm和imageBind）等半监督模型中得出的深度学习嵌入，这些模型代表了与PD相关的语音动力学。我们用于PD分类的新型融合模型将不同的语音嵌入到一个凝聚力的特征空间中，表现出优于基于标准串联的融合模型和其他基线（包括建立在传统声学特征的模型）上。在随机数据拆分配置中，该模型在接收器操作特征曲线（AUROC）下达到了88.94％的区域，精度为85.65％。严格的统计分析证实，在性别，种族和年龄方面，我们的模型在各种人口统计亚组中的表现都公平，并且无论疾病持续时间如何，我们的模型仍然保持强劲。此外，我们的模型在两个从临床环境和PD护理中心收集的完全看不见的测试数据集进行测试时，AUROC得分分别为82.12％和78.44％。这肯定了该模型的鲁棒性，并有可能在现实世界应用中提高可及性和健康公平性。

### Luganda Speech Intent Recognition for IoT Applications 
[[arxiv](https://arxiv.org/abs/2405.19343)] [[cool](https://papers.cool/arxiv/2405.19343)] [[pdf](https://arxiv.org/pdf/2405.19343)]
> **Authors**: Andrew Katumba,Sudi Murindanyi,John Trevor Kasule,Elvis Mugume
> **First submission**: 2024-05-16
> **First announcement**: 2024-05-29
> **comment**: Presented as a conference paper at ICLR 2024/AfricaNLP
- **标题**: 卢冈达对物联网应用的演讲意图认可
- **领域**: 声音,人工智能,计算语言学,音频和语音处理
- **摘要**: 物联网（IoT）技术的出现引起了人们对语音控制的智能房屋的极大兴趣。尽管许多语音控制的智能家庭系统旨在理解和支持英语（例如英语）的语言，但诸如Luganda之类的低资源语言的说话者可能需要更多的支持。该研究项目旨在为物联网应用程序开发卢甘达语音意图分类系统，以将本地语言整合到智能家庭环境中。该项目将硬件组件（例如Raspberry Pi，Wio终端和ESP32节点）作为微控制器。 Raspberry Pi处理Luganda语音命令，WIO终端是显示设备，ESP32节点控制着IoT设备。这项工作的最终目标是使用卢甘达启用语音控制，这是通过部署在Raspberry Pi上的自然语言处理（NLP）模型来完成的。 NLP模型将MEL频率曲线系数（MFCC）用作声学特征和卷积神经网络（CORV2D）体系结构进行语音意图分类。为此目的策划了卢甘达语音命令的数据集，并将其进行了开源。这项工作通过合并Luganda语音命令来解决物联网应用程序中的本地化挑战和语言多样性，使用户能够与智能家庭设备互动而无需英语能力，尤其是在本地语言主要是主要的地区。

### NeRAF: 3D Scene Infused Neural Radiance and Acoustic Fields 
[[arxiv](https://arxiv.org/abs/2405.18213)] [[cool](https://papers.cool/arxiv/2405.18213)] [[pdf](https://arxiv.org/pdf/2405.18213)]
> **Authors**: Amandine Brunetto,Sascha Hornauer,Fabien Moutarde
> **First submission**: 2024-05-28
> **First announcement**: 2024-05-29
> **comment**: Project Page: https://amandinebtto.github.io/NeRAF
- **标题**: NERAF：3D场景注入了神经辐射和声场
- **领域**: 声音,计算机视觉和模式识别,音频和语音处理
- **摘要**: 声音在人类感知中起着重要作用。除愿景外，它还提供了理解我们周围环境的重要信息。尽管神经隐式表示的进步，但学习与视觉场景保持一致的声学仍然是一个挑战。我们提出了NERAF，这种方法可以共同学习声学和辐射场。 Neraf通过调节3D场景几何和Radiance场上的外观先验的声场来综合新位置的新型视图和空间化的房间冲动响应（RIR）。生成的RIR可以应用于任何音频信号。每种方式都可以独立且在空间不同的位置上渲染，从而提供更大的多功能性。我们证明NERAF在Soundspaces和RAF数据集上产生高质量的音频，从而在先前的方法上实现了显着的性能改进，同时提高了数据效率。此外，NERAF增强了通过跨模式学习培训稀疏数据的复杂场景的新型视图综合。 Neraf设计为Nerfstudio模块，可方便地访问现实的视听生成。

### On the Condition Monitoring of Bolted Joints through Acoustic Emission and Deep Transfer Learning: Generalization, Ordinal Loss and Super-Convergence 
[[arxiv](https://arxiv.org/abs/2405.20887)] [[cool](https://papers.cool/arxiv/2405.20887)] [[pdf](https://arxiv.org/pdf/2405.20887)]
> **Authors**: Emmanuel Ramasso,Rafael de O. Teloli,Romain Marcel
> **First submission**: 2024-05-29
> **First announcement**: 2024-05-31
> **comment**: No comments
- **标题**: 关于通过声发射和深度转移学习对螺栓关节监测的条件：概括，序数损失和超级连接
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: 本文研究了基于卷积神经网络（CNN）的深度转移学习，以使用声学排放来监测螺栓接头的状况。螺栓结构是许多机械系统中的关键组成部分，监测其状况状况的能力对于有效的结构健康监测至关重要。我们使用Orion-AE基准评估了方法的性能，Orion-AE基准是由三个螺栓连接的两个薄梁组成的结构，在该螺栓中进行了高度嘈杂的声发射测量，以检测螺栓施加的拧紧扭矩的变化。从这种结构中使用的数据是从声发射数据流到图像中使用连续小波变换的，并利用预审计的CNN进行特征提取和降解的。我们的实验比较了单传感器与多传感器融合，以估算螺栓的拧紧水平（松动），并评估了对性能的原始数据和预滤波数据的使用。我们特别关注在不同的测量活动中基于CNN的转移学习的概括能力，我们研究了序数损失功能，以至于在接近地面真理时严重地惩罚了错误的预测，从而鼓励在邻近类中进行错误分类错误。还研究了网络配置以及学习率调度程序，并获得了超级连接，即，在具有不同网络的一些迭代中，可以实现高分类精度。此外，结果证明了基于CNN的转移学习的概括能力，用于通过声发射来监测螺栓结构，并在训练过程中需要不同数量的先前信息。

## 音频和语音处理(eess.AS:Audio and Speech Processing)

该领域共有 1 篇论文

### Benchmarking Representations for Speech, Music, and Acoustic Events 
[[arxiv](https://arxiv.org/abs/2405.00934)] [[cool](https://papers.cool/arxiv/2405.00934)] [[pdf](https://arxiv.org/pdf/2405.00934)]
> **Authors**: Moreno La Quatra,Alkis Koudounas,Lorenzo Vaiani,Elena Baralis,Luca Cagliero,Paolo Garza,Sabato Marco Siniscalchi
> **First submission**: 2024-05-01
> **First announcement**: 2024-05-02
> **comment**: No comments
- **标题**: 语音，音乐和声学事件的基准测试表示
- **领域**: 音频和语音处理,机器学习,声音
- **摘要**: 评估音频表示学习（ARL）方法的标准化基准测试的多样性有限，可能会阻碍当前方法能力的系统比较。我们提出了Arch，这是一种评估各种音频分类域的ARL方法的综合基准，涵盖声学事件，音乐和语音。 ARCH包含12个数据集，使我们能够彻底评估不同大小的预训练的SSL模型。 Arch通过统一访问广泛的域以及容易合并新的数据集和模型的能力来简化ARL技术的基准测试。为了解决当前缺乏非语音音频的开源，预训练的模型，我们还发布了新的预训练模型，这些模型在非语音数据集上表现出强烈的性能。我们认为，提出的广泛评估为最先进的ARL方法提供了有价值的见解，并且对指出有希望的研究方向很有用。

## 其他论文

共有 19 篇其他论文

- [Third Medium Finite Element Contact Formulation for Pneumatically Actuated Systems](https://arxiv.org/abs/2405.01185)
  - **标题**: 为气动驱动系统的第三个中等有限元接触公式
  - **Filtered Reason**: none of cs.CE in whitelist
- [Performance Analysis of Underwater Acoustic Channel Amid Jamming by Random Jammers](https://arxiv.org/abs/2405.02885)
  - **标题**: 在随机干扰器上堵塞的水下声道的性能分析
  - **Filtered Reason**: none of eess.SP,cs.ET in whitelist
- [AFDM Channel Estimation in Multi-Scale Multi-Lag Channels](https://arxiv.org/abs/2405.02660)
  - **标题**: 多尺度多lag通道中的AFDM通道估计
  - **Filtered Reason**: none of eess.SP,cs.IT in whitelist
- [Real-time multichannel deep speech enhancement in hearing aids: Comparing monaural and binaural processing in complex acoustic scenarios](https://arxiv.org/abs/2405.01967)
  - **标题**: 实时多通道深度语音增强助听器：比较复杂的声学场景中的单声道和双耳处理
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [BERP: A Blind Estimator of Room Acoustic and Physical Parameters for Single-Channel Noisy Speech Signals](https://arxiv.org/abs/2405.04476)
  - **标题**: BERP：单渠道嘈杂的语音信号的房间声学和物理参数的盲目估计器
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [WixUp: A General Data Augmentation Framework for Wireless Perception in Tracking of Humans](https://arxiv.org/abs/2405.04804)
  - **标题**: WIXUP：无线知觉的一般数据增强框架跟踪人类
  - **Filtered Reason**: none of cs.NI in whitelist
- [ReefGlider: A highly maneuverable vectored buoyancy engine based underwater robot](https://arxiv.org/abs/2405.06033)
  - **标题**: Reefglider：高度可操纵的矢量浮力发动机的水下机器人
  - **Filtered Reason**: none of eess.SY,cs.RO in whitelist
- [NeuRSS: Enhancing AUV Localization and Bathymetric Mapping with Neural Rendering for Sidescan SLAM](https://arxiv.org/abs/2405.05807)
  - **标题**: 神经：增强AUV的定位和测深图映射，并通过侧扫的神经渲染
  - **Filtered Reason**: none of cs.RO in whitelist
- [IPDnet: A Universal Direct-Path IPD Estimation Network for Sound Source Localization](https://arxiv.org/abs/2405.07021)
  - **标题**: IPDNET：声音源本地化的通用直接路径IPD IPD估计网络
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Diff-ETS: Learning a Diffusion Probabilistic Model for Electromyography-to-Speech Conversion](https://arxiv.org/abs/2405.08021)
  - **标题**: DIFF-ET：学习肌电图转换的扩散概率模型
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [A tunable binaural audio telepresence system capable of balancing immersive and enhanced modes](https://arxiv.org/abs/2405.08742)
  - **标题**: 一种可调双耳音频电视系统，能够平衡沉浸式和增强模式
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [TunnelSense: Low-power, Non-Contact Sensing using Tunnel Diodes](https://arxiv.org/abs/2405.09155)
  - **标题**: Tunnelsense：使用隧道二极管的低功率，非接触感测
  - **Filtered Reason**: none of cs.ET in whitelist
- [Monaural speech enhancement on drone via Adapter based transfer learning](https://arxiv.org/abs/2405.10022)
  - **标题**: 通过基于适配器的转移学习对无人机上的单声道语音增强
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Data-Efficient Low-Complexity Acoustic Scene Classification in the DCASE 2024 Challenge](https://arxiv.org/abs/2405.10018)
  - **标题**: Dcase 2024挑战中的数据有效的低复杂性声学场景分类
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Towards Optimal Beacon Placement for Range-Aided Localization](https://arxiv.org/abs/2405.11550)
  - **标题**: 迈向范围辅助本地化的最佳信标
  - **Filtered Reason**: none of cs.RO in whitelist
- [Optimizing Underwater IoT Routing with Multi-Criteria Decision Making and Uncertainty Weights](https://arxiv.org/abs/2405.11513)
  - **标题**: 通过多标准决策和不确定性权重优化水下物联网路由
  - **Filtered Reason**: none of cs.NI in whitelist
- [Acoustical Features as Knee Health Biomarkers: A Critical Analysis](https://arxiv.org/abs/2405.15085)
  - **标题**: 声学特征作为膝盖健康生物标志物：关键分析
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [1st Place Solution to Odyssey Emotion Recognition Challenge Task1: Tackling Class Imbalance Problem](https://arxiv.org/abs/2405.20064)
  - **标题**: 奥德赛情绪识别挑战的第一名解决方案任务1：解决阶级不平衡问题
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Semantic Landmark Detection & Classification Using Neural Networks For 3D In-Air Sonar](https://arxiv.org/abs/2405.19869)
  - **标题**: 使用神经网络进行3D空气声纳的语义里程碑检测和分类
  - **Filtered Reason**: none of cs.RO in whitelist
