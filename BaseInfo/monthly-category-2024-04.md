# 2024-04 月度论文分类汇总

共有22篇相关领域论文, 另有12篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 2 篇论文

### Double Mixture: Towards Continual Event Detection from Speech 
[[arxiv](https://arxiv.org/abs/2404.13289)] [[cool](https://papers.cool/arxiv/2404.13289)] [[pdf](https://arxiv.org/pdf/2404.13289)]
> **Authors**: Jingqi Kang,Tongtong Wu,Jinming Zhao,Guitao Wang,Yinwei Wei,Hao Yang,Guilin Qi,Yuan-Fang Li,Gholamreza Haffari
> **First submission**: 2024-04-20
> **First announcement**: 2024-04-22
> **comment**: The first two authors contributed equally to this work
- **标题**: 双重混合物：从语音中进行持续的事件检测
- **领域**: 计算语言学,多媒体,声音,音频和语音处理
- **摘要**: 语音事件检测对于多媒体检索至关重要，涉及语义和声学事件的标记。传统的ASR系统通常会忽略这些事件之间的相互作用，仅关注内容，即使对话的解释可能随环境环境而变化。本文在语音事件检测中解决了两个主要的挑战：新事件的持续整合而不忘记了以前的事件，以及与声学事件的语义分开。我们从语音中介绍了一个新任务，不断的事件检测，我们还提供了两个基准数据集。为了应对灾难性遗忘和有效的解散的挑战，我们提出了一种新颖的方法“双重混合物”。该方法将语音专业知识与强大的记忆机制融合在一起，以增强适应性并防止忘记。我们的全面实验表明，这项任务提出了重大挑战，这些挑战在计算机视觉或自然语言处理中没有有效地解决。我们的方法达到了最低的遗忘速率和最高水平的概括，证明了各种持续学习序列的稳健性。我们的代码和数据可在https://anonymon.4open.science/status/continual-speched-6461上找到。

### Towards Dog Bark Decoding: Leveraging Human Speech Processing for Automated Bark Classification 
[[arxiv](https://arxiv.org/abs/2404.18739)] [[cool](https://papers.cool/arxiv/2404.18739)] [[pdf](https://arxiv.org/pdf/2404.18739)]
> **Authors**: Artem Abzaliev,Humberto Pérez Espinosa,Rada Mihalcea
> **First submission**: 2024-04-29
> **First announcement**: 2024-04-30
> **comment**: to be published in LREC-COLING 2024
- **标题**: 迈向狗皮解码：利用人类的语音处理自动树皮分类
- **领域**: 计算语言学
- **摘要**: 与人类类似，动物广泛使用言语和非语言形式的交流，包括各种音频信号。在本文中，我们解决了狗的发声，并探讨了在人类语音上预先训练的自我监督语音表示模型的使用，以解决狗皮分类任务，以在语音识别中找到以人为中心的任务中的相似之处。我们专门解决了四个任务：狗的识别，品种识别，性别分类和上下文基础。我们表明，使用语音嵌入表示形式可显着改善更简单的分类基线。此外，我们还发现，在大型人类语音声学上预先训练的模型可以为多个任务提供更多的绩效提高。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 4 篇论文

### Z-Splat: Z-Axis Gaussian Splatting for Camera-Sonar Fusion 
[[arxiv](https://arxiv.org/abs/2404.04687)] [[cool](https://papers.cool/arxiv/2404.04687)] [[pdf](https://arxiv.org/pdf/2404.04687)]
> **Authors**: Ziyuan Qu,Omkar Vengurlekar,Mohamad Qadri,Kevin Zhang,Michael Kaess,Christopher Metzler,Suren Jayasuriya,Adithya Pediredla
> **First submission**: 2024-04-06
> **First announcement**: 2024-04-08
> **comment**: No comments
- **标题**: Z-SPLAT：Z轴高斯为相机融合
- **领域**: 计算机视觉和模式识别,图形,机器学习
- **摘要**: 可区分的3D高斯分裂（GS）正在作为计算机视觉和图形中的突出技术，用于重建3D场景。 GS代表一个场景作为一组3D高斯人，具有不同的不相处，并采用了计算有效的脱落操作以及分析衍生物，以计算从各种观点捕获的场景图像给定的3D高斯参数。不幸的是，在许多现实世界成像场景中，捕获环境视图（$ 360^{\ circ} $ viewpoint）图像是不可能或不切实际的，包括水下成像，建筑物内的房间和自主导航。在这些受限制的基线成像方案中，GS算法遭受了众所周知的“缺失锥体”问题，从而导致沿深度轴的重建不良。在本手稿中，我们证明，使用瞬态数据（来自声纳）使我们能够通过沿深度轴采样高频数据来解决缺失的锥体问题。我们扩展了两个常用声纳的高斯脱算算法，并提出了同时利用RGB相机数据和声纳数据的融合算法。通过在各种成像场景中进行的模拟，仿真和硬件实验，我们表明拟议的融合算法可显着更好的新型视图合成（PSNR改善5 dB）和3D几何重建（较低的倒角距离）。

### StrideNET: Swin Transformer for Terrain Recognition with Dynamic Roughness Extraction 
[[arxiv](https://arxiv.org/abs/2404.13270)] [[cool](https://papers.cool/arxiv/2404.13270)] [[pdf](https://arxiv.org/pdf/2404.13270)]
> **Authors**: Maitreya Shelare,Neha Shigvan,Atharva Satam,Poonam Sonar
> **First submission**: 2024-04-20
> **First announcement**: 2024-04-22
> **comment**: 6 pages, 5 figures, 3rd IEEE International Conference on Computer Vision and Machine Intelligence (IEEE CVMI)
- **标题**: StrideNet：具有动态粗糙度提取的地形识别的Swin Transformer
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 遥感图像分类的领域随着卷积神经网络的兴起以及视觉变压器的兴起而取得了巨大进步。与传统的卷积模型相反，这些模型凭借其自我发挥机制可以有效地捕获图像贴片之间的全球关系和远程依赖性。本文介绍了StrideNet，这是一种基于双支化的变压器模型，用于地形识别和表面粗糙度提取。 Terrain识别分支采用Swin Transformer通过利用其捕获本地和全球特征的能力来对各种地形进行分类。补充这一点，粗糙度提取分支利用统计纹理功能分析技术来动态提取重要的土地表面特性，例如粗糙度和滑水。该模型在一个定制数据集上进行了培训，该数据集由四个地形类别组成 - 草，沼泽，桑迪和岩石，并且通过在所有类别中达到99％以上的平均测试准确性，优于基于基础的CNN和基于变压器的模型。这项工作的应用扩展到不同领域，例如环境监测，土地使用和涵盖分类，灾难响应和精确农业。

### FDCE-Net: Underwater Image Enhancement with Embedding Frequency and Dual Color Encoder 
[[arxiv](https://arxiv.org/abs/2404.17936)] [[cool](https://papers.cool/arxiv/2404.17936)] [[pdf](https://arxiv.org/pdf/2404.17936)]
> **Authors**: Zheng Cheng,Guodong Fan,Jingchun Zhou,Min Gan,C. L. Philip Chen
> **First submission**: 2024-04-27
> **First announcement**: 2024-04-29
> **comment**: 16pages,13 figures
- **标题**: FDCE-net：带有嵌入频率和双色编码器的水下图像增强
- **领域**: 计算机视觉和模式识别
- **摘要**: 水下图像通常会遭受各种问题，例如亮度低，颜色移位，模糊的细节以及由于水和悬浮颗粒引起的光吸收和散射引起的噪音。以前的水下图像增强（UIE）方法主要集中在空间域增强上，忽略了图像中固有的频域信息。但是，水下图像的降解因子在空间结构域紧密相互交织。尽管某些方法着重于增强频域中的图像，但它们忽略了图像降解因子与频域中存在的信息之间的固有关系。结果，这些方法经常增强改进图像的某些属性，同时不充分地解决甚至加剧其他属性。此外，许多现有方法在很大程度上依赖于先验知识来解决水下图像中的颜色转移问题，从而限制了它们的灵活性和鲁棒性。为了克服这些局限性，我们在论文中提出了嵌入频率和双色编码器网络（FDCE-net）。 FDCE-NET由两个主要结构组成：（1）频率空间网络（FS-NET）旨在通过利用我们设计的频率空间残留块（FSRB）来实现初始增强，以将频域中的图像降解因子解释并分别增强不同属性。 （2）为了解决彩色问题，我们介绍了双色编码器（DCE）。 DCE通过跨注意事项建立了颜色和语义表示之间的相关性，并利用多尺度图像特征来指导自适应颜色查询的优化。最终增强图像是通过通过融合网络组合FS-NET和DCE的输出来生成的。这些图像显示了丰富的细节，清晰的纹理，低噪声和自然色。

### Terrain characterisation for online adaptability of automated sonar processing: Lessons learnt from operationally applying ATR to sidescan sonar in MCM applications 
[[arxiv](https://arxiv.org/abs/2404.18663)] [[cool](https://papers.cool/arxiv/2404.18663)] [[pdf](https://arxiv.org/pdf/2404.18663)]
> **Authors**: Thomas Guerneve,Stephanos Loizou,Andrea Munafo,Pierre-Yves Mignotte
> **First submission**: 2024-04-29
> **First announcement**: 2024-04-30
> **comment**: Presented at UACE (UnderwaterAcousticsConference & Exhibition) 2023, Kalamata, Greece
- **标题**: 自动声纳处理的在线适应性的地形表征：从操作中汲取的经验教训是将ATR应用于MCM应用程序的侧can声纳
- **领域**: 计算机视觉和模式识别,机器学习,机器人技术,软件工程
- **摘要**: 自动识别（ATR）算法在侧扫声纳图像上的性能已显示在部署在非良性环境中时会迅速降解。复杂的海底和声伪影以强烈的纹理模式的形式构成干扰物，从而产生错误的检测或防止对真实物体的检测。本文介绍了两种在线海底表征技术，以提高自动水下车辆（AUV）任务中的解释性。重要的是，与以前在域中的工作相反，这些技术不是基于模型，并且需要人类操作员的投入有限，这使其适用于实时的船上处理。两种技术都依赖于无监督的机器学习方法来提取与人类对地形复杂性的理解有关的地形特征。第一种技术根据ATR算法的性能提供了定量的，应用驱动的地形表征。第二种方法提供了一种纳入主题专业知识并实现情境化和解释性的方法，以支持方案依赖的主观地形特征。与传统的无监督方法相比，地形复杂性与经验丰富的用户的期望与经验丰富的用户的期望相匹配。最终，我们详细介绍了这些技术以维修矿山对策（MCM）任务与Seebyte自主框架Neptune一起进行的应用。

## 机器学习(cs.LG:Machine Learning)

该领域共有 5 篇论文

### Audio Simulation for Sound Source Localization in Virtual Evironment 
[[arxiv](https://arxiv.org/abs/2404.01611)] [[cool](https://papers.cool/arxiv/2404.01611)] [[pdf](https://arxiv.org/pdf/2404.01611)]
> **Authors**: Yi Di Yuan,Swee Liang Wong,Jonathan Pan
> **First submission**: 2024-04-01
> **First announcement**: 2024-04-02
> **comment**: 2024 IEEE World Forum on Public Safety Technology
- **标题**: 在虚拟环境中的声源本地化的音频仿真
- **领域**: 机器学习,声音,音频和语音处理
- **摘要**: 信号剥夺环境中的非视线本地化是一个具有挑战性但相关的问题。由于性质的回响性质，在这种主要室内场景中的声学方法遇到了困难。在这项研究中，我们旨在通过利用物理接地的声音传播模拟和机器学习方法来定位虚拟环境中特定位置的声音源。该过程试图克服数据不足问题，以将声音来源定位到其发生的位置，尤其是在事后本地化中。我们使用音频变压器谱图方法实现了0.786 +/- 0.0136 F1得分。

### On the Efficiency and Robustness of Vibration-based Foundation Models for IoT Sensing: A Case Study 
[[arxiv](https://arxiv.org/abs/2404.02461)] [[cool](https://papers.cool/arxiv/2404.02461)] [[pdf](https://arxiv.org/pdf/2404.02461)]
> **Authors**: Tomoyoshi Kimura,Jinyang Li,Tianshi Wang,Denizhan Kara,Yizhuo Chen,Yigong Hu,Ruijie Wang,Maggie Wigness,Shengzhong Liu,Mani Srivastava,Suhas Diggavi,Tarek Abdelzaher
> **First submission**: 2024-04-03
> **First announcement**: 2024-04-04
> **comment**: No comments
- **标题**: 关于基于振动的基础模型的物联网传感的效率和鲁棒性：案例研究
- **领域**: 机器学习,信号处理
- **摘要**: 本文展示了基于振动的基础模型（FMS）的潜力，该模型（FMS）预先训练了未标记的感应数据，以提高（一类）IoT应用程序中运行时推断的鲁棒性。提出了一个案例研究，其中包含使用声学和地震传感的车辆分类应用。这项工作是由于基础模型在自然语言处理和计算机视觉领域的成功所激发的，从而导致FM概念对其他领域的概括也概括了，在这种情况下，存在大量的未标记数据，可用于自我监督的预训练。一个这样的领域就是IoT应用程序。可以使用可用的未标记的传感器数据以环境不知所措的方式预先训练IoT域中选定感应方式的基础模型，然后使用少量标记的数据对手头部署进行微调。该论文表明，预训练/微调方法改善了下游推断的鲁棒性，并促进了适应不同环境条件的。更具体地说，我们在现实世界中介绍了一个案例研究，以评估一个称为焦点的简单（基于振动）的FM样模型，与常规监督的深神经网络（DNN）相比，表明其出色的鲁棒性和适应性。我们还证明了它优于监督解决方案。我们的发现突出了基于推理的鲁棒性，运行时效率和模型适应（通过微调）在资源有限的IoT设置中的推理鲁棒性，运行时效率和模型适应（通过微调）的提示，其优势（通常是FM启发的自我监督模型）。

### Continual Learning of Range-Dependent Transmission Loss for Underwater Acoustic using Conditional Convolutional Neural Net 
[[arxiv](https://arxiv.org/abs/2404.08091)] [[cool](https://papers.cool/arxiv/2404.08091)] [[pdf](https://arxiv.org/pdf/2404.08091)]
> **Authors**: Indu Kant Deo,Akash Venkateshwaran,Rajeev K. Jaiman
> **First submission**: 2024-04-11
> **First announcement**: 2024-04-12
> **comment**: 14 pages, 18 figures
- **标题**: 使用条件卷积神经网持续学习依赖范围的传播损失
- **领域**: 机器学习,信号处理,流体动力学
- **摘要**: 非常需要精确，可靠的预测运输船上发出的远场噪声。基于Navier-Stokes方程的常规全阶模型不合适，而复杂的模型还原方法可能无效，无法准确预测具有海拔的环境和测深的显着变化的环境中的远场噪声。减少阶模型的最新进展，尤其是基于卷积和复发性神经网络的模型，提供了更快，更准确的替代方案。这些模型使用卷积神经网络有效地减少数据维度。但是，当前的深度学习模型在预测长时间和远程位置的波浪传播方面面临挑战，通常依靠自动回火预测和缺乏远场测深的信息。这项研究旨在提高深度学习模型的准确性，以预测远场场景中的水下辐射噪声。我们提出了一个新型的范围条件卷积神经网络，该网络将海洋测深数据纳入输入中。通过将这种体系结构整合到一个持续的学习框架中，我们旨在概括全球各种测深的模型。为了证明我们的方法的有效性，我们在几个测试案例和基准场景上分析了我们的模型，涉及东北太平洋迪金的海山的远场预测。我们提出的架构有效地捕获了依赖范围内的，不同的测深剖面的传输损失。该体系结构可以集成到用于水下辐射噪声的自适应管理系统中，从而提供近场船舶噪声源之间的实时端到端映射，并在海洋哺乳动物的位置接收到噪声。

### AudioProtoPNet: An interpretable deep learning model for bird sound classification 
[[arxiv](https://arxiv.org/abs/2404.10420)] [[cool](https://papers.cool/arxiv/2404.10420)] [[pdf](https://arxiv.org/pdf/2404.10420)]
> **Authors**: René Heinrich,Lukas Rauch,Bernhard Sick,Christoph Scholz
> **First submission**: 2024-04-16
> **First announcement**: 2024-04-17
> **comment**: Work in progress
- **标题**: AudiiOprotopnet：一种可解释的鸟类声音分类的深度学习模型
- **领域**: 机器学习
- **摘要**: 深度学习模型可以根据其发声来识别众多鸟类，从而具有显着高级的声学鸟类监测。但是，传统的深度学习模型是黑匣子，无法洞悉其基本计算，从而限制了它们对鸟类学家和机器学习工程师的用处。可解释的模型可以促进调试，知识发现，信任和跨学科合作。这项研究介绍了AudioProtopnet，这是用于多标签鸟类声音分类的原型零件网络（Protopnet）的适应。这是一个可解释的模型，它使用Convnext主链提取嵌入式，分类层被训练在这些嵌入的原型学习分类器所取代。分类器从训练实例的光谱图中学习了每种鸟类的发声的原型模式。在推断期间，通过将录音与嵌入空间中的学识渊博原型进行比较，从而对模型的决策进行了比较，从而对每种鸟类最有用的嵌入方式进行了解释。该模型在Birdset训练数据集上进行了培训，该数据集由9,734种鸟类和6,800小时的记录组成。它的性能在鸟类的七个测试数据集上进行了评估，涵盖了不同的地理区域。 AudioProtopnet的表现优于最先进的模型，平均AUROC为0.90，CMAP为0.42，相对提高分别为7.1％和16.7％。这些结果表明，即使对于多标签鸟类声音分类的具有挑战性的任务，也有可能开发强大但可解释的深度学习模型，为鸟类学家和机器学习工程师提供宝贵的见解。

### Dynamical Mode Recognition of Coupled Flame Oscillators by Supervised and Unsupervised Learning Approaches 
[[arxiv](https://arxiv.org/abs/2404.17801)] [[cool](https://papers.cool/arxiv/2404.17801)] [[pdf](https://arxiv.org/pdf/2404.17801)]
> **Authors**: Weiming Xu,Tao Yang,Peng Zhang
> **First submission**: 2024-04-27
> **First announcement**: 2024-04-29
> **comment**: research paper (29 pages, 20 figures)
- **标题**: 通过监督和无监督的学习方法对耦合火焰振荡器的动态模式识别
- **领域**: 机器学习
- **摘要**: 作为燃烧研究中最具挑战性的问题之一，燃气轮机和火箭发动机中的燃烧不稳定性源于火焰之间的复杂相互作用，这些相互作用也受到化学反应，热量和传播和声学的影响。识别和理解燃烧不稳定性对于确保许多燃烧系统的安全可靠操作至关重要，在探索和对复杂火焰系统的动态行为进行分类是核心。为了促进基本研究，目前的工作涉及对闪烁的浮力扩散火焰制成的耦合火焰振荡器的动态模式，这些振荡器近年来引起了人们的注意，但尚未充分理解。火焰振荡器的时间序列数据是通过完全验证的反应流仿真生成的。由于基于专业知识的模型的局限性，采用了数据驱动的方法。在这项研究中，使用变异自动编码器（VAE）的非线性尺寸还原模型将模拟数据投射到二维潜在空间上。基于潜在空间中的相轨迹，建议分别针对具有众所周知的标签和没有的数据集进行监督和无监督的分类器。对于标记的数据集，我们为模式识别建立了基于Wasserstein-Distance的分类器（WDC）；对于未标记的数据集，我们开发了一种新颖的无监督分类器（GMM-DTWC），结合了动态时间扭曲（DTW）和高斯混合模型（GMM）。通过与传统的降低和分类的方法进行比较，提出的监督和无监督的基于VAE的方法表现出了区分动态模式的突出性能，这意味着它们的潜在扩展到对复杂燃烧问题的动态模式识别。

## 多代理系统(cs.MA:Multiagent Systems)

该领域共有 1 篇论文

### Multi-AUV Cooperative Underwater Multi-Target Tracking Based on Dynamic-Switching-enabled Multi-Agent Reinforcement Learning 
[[arxiv](https://arxiv.org/abs/2404.13654)] [[cool](https://papers.cool/arxiv/2404.13654)] [[pdf](https://arxiv.org/pdf/2404.13654)]
> **Authors**: Shengbo Wang,Chuan Lin,Guangjie Han,Shengchao Zhu,Zhixian Li,Zhenyu Wang,Yunpeng Ma
> **First submission**: 2024-04-21
> **First announcement**: 2024-04-22
> **comment**: No comments
- **标题**: 基于动态转换的多代理增强学习的多AUV合作水下多目标跟踪
- **领域**: 多代理系统
- **摘要**: 近年来，自主水下汽车（AUV）群正在逐渐流行，并在海洋探索或水下跟踪中广泛促进。在本文中，我们提出了一个多AUV合作的水下多距离跟踪跟踪算法，尤其是当考虑到真正的水下因素时。我们首先为基于水下声纳的检测和海洋电流干扰目标跟踪过程提供了通常的建模方法。然后，基于软件定义的网络（SDN），我们将AUV群视为水下临时网络，并提出了层次软件定义的多AUV增强学习（HSARL）体系结构。根据提议的HSARL架构，我们提出了“动态转换”机制，其中包括“动态转换注意力”和“动态转换重新采样”机制，这些机制可以加速HSARL算法的收敛速度，并有效地阻止其固定在当地的最佳状态。此外，我们引入了奖励重塑机制，以进一步加速提出的HSARL算法在早期阶段的收敛速度。最后，基于提出的AUV分类方法，我们提出了一种合作跟踪算法，称为基于动态转换的MARL（DSBM）驱动的跟踪算法。评估结果表明，我们提出的DSBM跟踪算法可以执行精确的水下多目标跟踪，并与许多最近的研究产品相比，就各种重要的指标而言。

## 多媒体(cs.MM:Multimedia)

该领域共有 1 篇论文

### TCAN: Text-oriented Cross Attention Network for Multimodal Sentiment Analysis 
[[arxiv](https://arxiv.org/abs/2404.04545)] [[cool](https://papers.cool/arxiv/2404.04545)] [[pdf](https://arxiv.org/pdf/2404.04545)]
> **Authors**: Ming Zhou,Weize Quan,Ziqi Zhou,Kai Wang,Tong Wang,Dong-Ming Yan
> **First submission**: 2024-04-06
> **First announcement**: 2024-04-08
> **comment**: No comments
- **标题**: TCAN：面向文本的跨注意网络用于多模式情感分析
- **领域**: 多媒体,计算语言学
- **摘要**: 多模式情感分析（MSA）努力通过利用语言，视觉和声学方式来理解人类情感。尽管以前的MSA方法表现出了显着的性能，但固有的多模式异质性的存在却带来了挑战，不同方式的贡献很大。过去的研究主要集中于改善表示技术和特征融合策略。但是，其中许多努力忽略了不同方式之间语义丰富性的变化，从而统一地对待每种方式。这种方法可能导致低估了强烈方式的重要性，同时过分强调了弱者的重要性。在这些见解的推动下，我们引入了面向文本的跨注意网络（TCAN），强调了文本模式在MSA中的主要作用。具体而言，对于每个多模式样本，通过将三种模态的未对齐序列作为输入，我们最初将提取的单峰特征分配为视文本和声学文本对。随后，我们对文本方式实施自我注意事项，并将文本引人入胜的交叉注意力应用于视觉和声学方式。为了减轻噪声信号和冗余特征的影响，我们将封闭的控制机制纳入框架中。此外，我们引入了单峰联合学习，以通过反向传播更深入地了解各种方式的同质情感倾向。实验结果表明，TCAN在两个数据集（CMU-MOSI和CMU-MOSEI）上始终优于最先进的MSA方法。

## 机器人技术(cs.RO:Robotics)

该领域共有 1 篇论文

### Seamless Underwater Navigation with Limited Doppler Velocity Log Measurements 
[[arxiv](https://arxiv.org/abs/2404.13742)] [[cool](https://papers.cool/arxiv/2404.13742)] [[pdf](https://arxiv.org/pdf/2404.13742)]
> **Authors**: Nadav Cohen,Itzik Klein
> **First submission**: 2024-04-21
> **First announcement**: 2024-04-22
> **comment**: No comments
- **标题**: 多普勒速度日志测量有限的无缝水下导航
- **领域**: 机器人技术,人工智能,系统与控制
- **摘要**: 自动水下车辆（AUV）通常使用惯性导航系统（INS）和多普勒速度日志（DVL）进行水下导航。为此，它们的测量是通过非线性过滤器（例如扩展的卡尔曼滤波器（EKF））集成的。 DVL速度矢量估计取决于从海床中检索反射，从而确保其四个传输的声学束中至少有三个成功返回。当获得少于三个光束时，DVL无法提供速度更新来绑定导航解决方案漂移。为了应对这一挑战，在本文中，我们提出了一种混合神经耦合（HNC）方法，用于在有限的DVL测量值下进行无缝的AUV导航。首先，我们以回归两个或三个缺失的DVL光束的方法。然后，将这些梁与测得的梁一起掺入EKF中。我们在松散和紧密耦合的方法中检查了INS/DVL融合。我们的方法在两种不同的情况下对来自地中海的AUV实验的记录数据进行了培训和评估。结果表明，我们提出的方法的表现比基线松散，紧密耦合的基于模型的方法平均高96.15％。与基于模型的梁估计器相比，它的性能也表现出了较高的性能，而对于涉及两个或三个缺失光束的场景，就速度精度而言，平均速度为12.41％。因此，我们证明我们的方法在有限的光束测量值的情况下提供了无缝的AUV导航。

## 声音(cs.SD:Sound)

该领域共有 6 篇论文

### Sound event localization and classification using WASN in Outdoor Environment 
[[arxiv](https://arxiv.org/abs/2403.20130)] [[cool](https://papers.cool/arxiv/2403.20130)] [[pdf](https://arxiv.org/pdf/2403.20130)]
> **Authors**: Dongzhe Zhang,Jianfeng Chen,Jisheng Bai,Mou Wang
> **First submission**: 2024-03-29
> **First announcement**: 2024-04-01
> **comment**: No comments
- **标题**: 在室外环境中使用的声音事件本地化和分类
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: 基于深度学习的声音事件本地化和分类是无线声学传感器网络中的新兴研究领域。但是，当前的声音事件定位和分类方法通常依赖于单个麦克风阵列，使它们容易受到信号衰减和环境噪声的影响，从而限制了其监视范围。此外，使用多个麦克风阵列的方法通常仅着眼于源本地化，忽略了声音事件分类的方面。在本文中，我们提出了一种基于深度学习的方法，该方法采用多种功能和注意力机制来估计声源的位置和类别。我们引入了声音示意功能，以捕获多个频段的空间信息。我们还使用伽马酮过滤器来生成更适合室外环境的声学特征。此外，我们整合了注意机制，以在声学特征中学习渠道关系和时间依赖性。为了评估我们提出的方法，我们使用具有不同级别的噪声和监视区域大小的模拟数据集以及不同的阵列和源位置进行实验。实验结果表明，在声音事件分类和声音源本地化任务中，我们提出的方法比最先进的方法具有优越性。我们提供进一步的分析来解释观察到的错误的原因。

### Prior-agnostic Multi-scale Contrastive Text-Audio Pre-training for Parallelized TTS Frontend Modeling 
[[arxiv](https://arxiv.org/abs/2404.09192)] [[cool](https://papers.cool/arxiv/2404.09192)] [[pdf](https://arxiv.org/pdf/2404.09192)]
> **Authors**: Quanxiu Wang,Hui Huang,Mingjie Wang,Yong Dai,Jinzuomu Zhong,Benlai Tang
> **First submission**: 2024-04-14
> **First announcement**: 2024-04-15
> **comment**: No comments
- **标题**: 对并行的TTS前端建模的先前不稳定的多尺度对比文本预训练
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在过去的十年中，一系列坚定的努力致力于开发高度表现力和可控制的文本到语音（TTS）系统。通常，整体TTS包括两个互连组件：前端模块和后端模块。前端从原始文本输入中捕获语言表示形式方面表现出色，而后端模块将语言提示转换为语音。研究界表明，人们对前端组件的研究越来越兴趣，认识到其在文本到语音系统中的关键作用，包括文本归一化（TN），韵律边界预测（PBP）和多人歧义（PD）。尽管如此，带注释的文本数据不足所带来的局限性以及对均质文本信号的依赖极大地破坏了其监督学习的有效性。为了逃避这一障碍，本文提出了一个新颖的两阶段TTS前端预测管道，名为TAP-FM。具体而言，在第一个学习阶段，我们提出了一种多尺度的对比文本原告预训练方案（MC-TAP），该协议通过以不受监督的方式通过多范围的对比预训练来获取更丰富的见解。我们的框架没有在先前的培训方法中采矿均匀的特征，而是展示了深入研究全球和本地文本审计语义和声学表示的能力。此外，在第二阶段，精心设计了一个并行的TTS前端模型，分别设计为执行TN，PD和PBP预测任务。最后，广泛的实验说明了我们提出的方法的优越性，从而实现了最新的性能。

### Non-Invasive Suicide Risk Prediction Through Speech Analysis 
[[arxiv](https://arxiv.org/abs/2404.12132)] [[cool](https://papers.cool/arxiv/2404.12132)] [[pdf](https://arxiv.org/pdf/2404.12132)]
> **Authors**: Shahin Amiriparian,Maurice Gerczuk,Justina Lutz,Wolfgang Strube,Irina Papazova,Alkomiet Hasan,Alexander Kathan,Björn W. Schuller
> **First submission**: 2024-04-18
> **First announcement**: 2024-04-19
> **comment**: :I.2
- **标题**: 非侵入性自杀风险通过语音分析预测
- **领域**: 声音,计算语言学,音频和语音处理
- **摘要**: 延迟获得急诊科有自杀趋势的患者的专业精神病学评估和护理，在及时干预方面显着差距，阻碍了在关键情况下提供适当的心理健康支持。为了解决这个问题，我们提出了一种非侵入性的，基于语音的方法，用于自杀风险评估。在我们的研究中，我们从20美元的患者那里收集了一个新颖的语音录音数据集。我们提取三组功能，包括WAV2VEC，可解释的语音和声学特征以及基于深度学习的光谱表示。我们通过进行二进制分类来进行进行进行，以评估一项受试者的样式的自杀风险。我们最有效的语音模型达到了$ 66.2 \，\％$的平衡精度。此外，我们表明，将我们的语音模型与一系列患者的元数据相结合，例如自杀企图或使用枪支的历史，可以改善总体结果。元数据集成的准确性为$ 94.4 \，\％$，标志着$ 28.2 \，\％$的绝对提高，证明了我们提议的自动自杀风险评估急诊医学评估的疗效。

### Retrieval-Augmented Audio Deepfake Detection 
[[arxiv](https://arxiv.org/abs/2404.13892)] [[cool](https://papers.cool/arxiv/2404.13892)] [[pdf](https://arxiv.org/pdf/2404.13892)]
> **Authors**: Zuheng Kang,Yayun He,Botao Zhao,Xiaoyang Qu,Junqing Peng,Jing Xiao,Jianzong Wang
> **First submission**: 2024-04-22
> **First announcement**: 2024-04-23
> **comment**: Accepted by the 2024 International Conference on Multimedia Retrieval (ICMR 2024)
- **标题**: 检索声音的音频深击检测
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 随着语音综合的最新进展，包括文本到语音（TTS）和语音转换（VC）系统，可以产生超现实的音频深击，因此人们对它们的潜在滥用越来越关注。但是，大多数DeepFake（DF）检测方法仅依赖于单个模型所学的模糊知识，从而导致性能瓶颈和透明度问题。受检索型生成（RAG）的启发，我们提出了一个检索型检测（RAD）框架，增强了使用类似检索样品的样品进行测试以增强检测。我们还扩展了多融合细心的分类器，以将其与我们建议的RAD框架集成在一起。广泛的实验表明，所提出的RAD框架比基线方法的出色性能，在ASVSPOOF 2021 DF集合中实现了最先进的结果，并在2019年和2021 LA组中获得了竞争性结果。进一步的样本分析表明，猎犬始终从同一扬声器中检索样品，其声学特性与查询音频高度一致，从而提高了检测性能。

### StoryTTS: A Highly Expressive Text-to-Speech Dataset with Rich Textual Expressiveness Annotations 
[[arxiv](https://arxiv.org/abs/2404.14946)] [[cool](https://papers.cool/arxiv/2404.14946)] [[pdf](https://arxiv.org/pdf/2404.14946)]
> **Authors**: Sen Liu,Yiwei Guo,Xie Chen,Kai Yu
> **First submission**: 2024-04-23
> **First announcement**: 2024-04-24
> **comment**: Accepted by ICASSP 2024
- **标题**: Storytts：具有丰富文本表达注释的高度表现力的文本到语音数据集
- **领域**: 声音,计算语言学,音频和语音处理
- **摘要**: 尽管长期以来在表达文本到语音（ETT）中研究了声学表达性，但文本中的固有表现力缺乏足够的关注，尤其是对于艺术作品的ETT。在本文中，我们介绍了Storytts，这是一个高度的ETTS数据集，其中包含声学和文本视角上的丰富表现力，从录制了普通话的讲故事节目。提出了一个系统的全面标签框架，以提高文本表达。我们在讲故事中分析并定义与语音相关的文本表达性，包括通过语言学，言辞等五个不同的维度。然后，我们采用大型语言模型，并提示他们使用一些手动注释示例进行批处理注释。由此产生的语料库包含61个小时的连续和高度韵律的语音，配备了准确的文本抄录和丰富的文本表达注释。因此，讲故事可以帮助未来的ETT研究，以充分挖掘丰富的内在文本和声学特征。进行实验是为了验证TTS模型在与讲故事的带注释的文本标签集成时，可以提高表达能力。

### Leveraging tropical reef, bird and unrelated sounds for superior transfer learning in marine bioacoustics 
[[arxiv](https://arxiv.org/abs/2404.16436)] [[cool](https://papers.cool/arxiv/2404.16436)] [[pdf](https://arxiv.org/pdf/2404.16436)]
> **Authors**: Ben Williams,Bart van Merriënboer,Vincent Dumoulin,Jenny Hamer,Eleni Triantafillou,Abram B. Fleishman,Matthew McKown,Jill E. Munger,Aaron N. Rice,Ashlee Lillis,Clemency E. White,Catherine A. D. Hobbs,Tries B. Razak,Kate E. Jones,Tom Denton
> **First submission**: 2024-04-25
> **First announcement**: 2024-04-26
> **comment**: 18 pages, 5 figures
- **标题**: 利用热带礁石，鸟和无关的声音，以进行海洋生物源的出色转移学习
- **领域**: 声音,人工智能,机器学习,音频和语音处理
- **摘要**: 机器学习有可能彻底改变被动声监测（PAM）进行生态评估。但是，高注释和计算成本限制了该领域的功效。可概括的预处理网络可以克服这些成本，但是高质量的预处理需要大量注释的图书馆，这主要限制了其当前的适用性，主要是对鸟类分类单元。在这里，我们使用珊瑚礁生物声学确定了数据缺陷域的最佳预处理策略。我们组装了礁石，这是一个大型注释的珊瑚礁声音库，尽管与鸟类图书馆相比，以样本计数的2％相比，虽然谦虚。通过测试很少的转移学习表现，我们观察到，与单独的雷或不相关的音频进行预处理相比，在鸟音频上进行预处理可提供高出的概括性。但是，我们的主要发现表明，在预处理过程中利用鸟，珊瑚礁和无关的音频的跨域混合可以最大程度地推广礁。我们验证的网络Surfperch为对海洋PAM数据的自动分析奠定了坚实的基础，并具有最低的注释和计算成本。

## 数值分析(math.NA:Numerical Analysis)

该领域共有 1 篇论文

### Improved impedance inversion by deep learning and iterated graph Laplacian 
[[arxiv](https://arxiv.org/abs/2404.16324)] [[cool](https://papers.cool/arxiv/2404.16324)] [[pdf](https://arxiv.org/pdf/2404.16324)]
> **Authors**: Davide Bianchi,Florian Bossmann,Wenlong Wang,Mingming Liu
> **First submission**: 2024-04-25
> **First announcement**: 2024-04-26
> **comment**: mber:submitted to SEG Geophysics (June 2024)
- **标题**: 通过深度学习和迭代图拉普拉斯主义者改善阻抗反转
- **领域**: 数值分析,机器学习,信号处理
- **摘要**: 在近年来，深度学习技术在许多应用中都表现出了巨大的潜力。所达到的结果通常优于传统技术。但是，神经网络的质量在很大程度上取决于使用的训练数据。嘈杂，不足或有偏见的培训数据导致次优结果。我们提出了一种混合方法，该方法将深度学习与迭代图拉普拉斯式结合在一起，并显示其在声学阻抗反演中的应用，这是地震探索中的常规程序。神经网络用于获得潜在的声阻抗的第一个近似值，并通过此近似构建图形laplacian矩阵。之后，我们使用类似Tikhonov的变分方法来解决正规器基于构造的图形laplacian的阻抗反转问题。与神经网络获得的初始猜测相比，获得的解决方案可以证明相对于噪声更准确和稳定。每次从最新的重建中构造新的图拉普拉斯矩阵时，都可以迭代多次。该方法仅在几次迭代返回更准确的重建后收敛。我们在两个不同的数据集和不同级别的噪声下证明了我们方法的潜力。我们使用了以前作品中引入的两个不同的神经网络。实验表明，我们的方法在存在噪声的情况下提高了重建质量。

## 医学物理(physics.med-ph:Medical Physics)

该领域共有 1 篇论文

### Score-Based Diffusion Models for Photoacoustic Tomography Image Reconstruction 
[[arxiv](https://arxiv.org/abs/2404.00471)] [[cool](https://papers.cool/arxiv/2404.00471)] [[pdf](https://arxiv.org/pdf/2404.00471)]
> **Authors**: Sreemanti Dey,Snigdha Saha,Berthy T. Feng,Manxiu Cui,Laure Delisle,Oscar Leong,Lihong V. Wang,Katherine L. Bouman
> **First submission**: 2024-03-30
> **First announcement**: 2024-04-01
> **comment**: 5 pages
- **标题**: 基于分数的光声断层扫描图像重建的扩散模型
- **领域**: 医学物理,计算机视觉和模式识别,机器学习,图像和视频处理
- **摘要**: 光声断层扫描（PAT）是一种快速发展的医学成像方式，将光吸收对比度与超声成像深度的对比相结合。 PAT中的一个挑战是由于传感器覆盖率有限或由于传感器阵列的密度而导致声信号不足的图像重建。这些案例要求解决一个不适合的反重建问题。在这项工作中，我们使用基于得分的扩散模型来解决从有限的PAT测量中重建图像的反问题。所提出的方法使我们能够在模拟的血管结构上通过扩散模型纳入表达的先验，同时仍然坚固以改变换能器的稀疏条件。

## 其他论文

共有 12 篇其他论文

- [A Comparative Analysis of Poetry Reading Audio: Singing, Narrating, or Somewhere In Between?](https://arxiv.org/abs/2404.00789)
  - **标题**: 诗歌阅读音频的比较分析：唱歌，叙述或介于两者之间的某个地方？
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Robust STL Control Synthesis under Maximal Disturbance Sets](https://arxiv.org/abs/2404.05535)
  - **标题**: 在最大干扰集中的强大的STL控制合成
  - **Filtered Reason**: none of cs.RO in whitelist
- [Multiple Mobile Target Detection and Tracking in Active Sonar Array Using a Track-Before-Detect Approach](https://arxiv.org/abs/2404.10316)
  - **标题**: 使用轨道前定义方法，在活动声纳阵列中多个移动目标检测和跟踪
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [AquaSonic: Acoustic Manipulation of Underwater Data Center Operations and Resource Management](https://arxiv.org/abs/2404.11815)
  - **标题**: Aquasonic：水下数据中心操作和资源管理的声学操纵
  - **Filtered Reason**: none of cs.CR in whitelist
- [Sparse Direction of Arrival Estimation Method Based on Vector Signal Reconstruction with a Single Vector Sensor](https://arxiv.org/abs/2404.13568)
  - **标题**: 基于矢量信号重建的稀疏到达估计方法与单个矢量传感器
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Ring-a-Pose: A Ring for Continuous Hand Pose Tracking](https://arxiv.org/abs/2404.12980)
  - **标题**: 环姿势：连续手姿势跟踪的戒指
  - **Filtered Reason**: none of cs.HC in whitelist
- [ActSonic: Recognizing Everyday Activities from Inaudible Acoustic Wave Around the Body](https://arxiv.org/abs/2404.13924)
  - **标题**: Actsonic：认识到身体周围听不清的声波的日常活动
  - **Filtered Reason**: none of cs.ET,cs.HC in whitelist
- [Vector Signal Reconstruction Sparse and Parametric Approach of direction of arrival Using Single Vector Hydrophone](https://arxiv.org/abs/2404.15160)
  - **标题**: 使用单个矢量水文的矢量信号重建稀疏和参数到达方向的方法
  - **Filtered Reason**: none of cs.SD in whitelist
- [SCR-Auth: Secure Call Receiver Authentication on Smartphones Using Outer Ear Echoes](https://arxiv.org/abs/2404.15000)
  - **标题**: scr-auth：使用外耳回声上的智能手机上的安全通话接收器身份验证
  - **Filtered Reason**: none of cs.CR in whitelist
- [Bathymetric Surveying with Imaging Sonar Using Neural Volume Rendering](https://arxiv.org/abs/2404.14819)
  - **标题**: 使用神经音量渲染的成像声纳测量测量
  - **Filtered Reason**: none of cs.RO in whitelist
- [An Investigation of Time-Frequency Representation Discriminators for High-Fidelity Vocoder](https://arxiv.org/abs/2404.17161)
  - **标题**: 对高保真声码器的时频表示歧视者的调查
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Audio-Visual Target Speaker Extraction with Reverse Selective Auditory Attention](https://arxiv.org/abs/2404.18501)
  - **标题**: 视听目标扬声器提取有反向选择性听觉的注意
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
