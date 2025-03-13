# 2025-02 月度论文分类汇总

共有676篇相关领域论文, 另有71篇其他

## 星系天体物理学(astro-ph.GA:Astrophysics of Galaxies)

该领域共有 1 篇论文

### Shared Stochastic Gaussian Process Latent Variable Models: A Multi-modal Generative Model for Quasar Spectra 
[[arxiv](https://arxiv.org/abs/2502.19824)] [[cool](https://papers.cool/arxiv/2502.19824)] [[pdf](https://arxiv.org/pdf/2502.19824)]
> **Authors**: Vidhi Lalchand,Anna-Christina Eilers
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Published in TMLR, https://openreview.net/pdf?id=LzmsvRTqaJ. The code for this work is available at: https://github.com/vr308/Quasar-GPLVM
- **标题**: 共享随机高斯流程潜在变量模型：类星体光谱的多模式生成模型
- **领域**: 星系天体物理学,机器学习,应用领域,方法论
- **摘要**: 这项工作提出了一个基于高斯过程的可扩展概率潜在变量模型（Lawrence，2004），在多个观察空间的背景下。我们专注于天体物理学的应用，其中数据集通常包含观察到的光谱特征和天体物理物体（例如星系或系外行星）的科学特性。在我们的应用中，我们研究了称为类星体的非常发光星系的光谱，以及它们的性质，例如其中央超级黑洞的质量，积聚率和在多个观测空间中的光度差异。然后，一个数据点的特征是不同类别的观测值，每个观测值都有不同的可能性。我们提出的模型扩展了Lalchand等人引入的基线随机变量高斯过程潜伏变量模型（GPLVM）。 （2022）在此设置中，提出了一个无缝的生成模型，其中可以使用共享潜在空间同时生成类星体光谱和科学标签，作为对不同高斯工艺解码器集的输入，每个观测空间一个。此外，此框架可以在缺少的数据设置中进行培训，在缺少的数据设置中，每个数据点可能未知或未观察到大量维度。我们证明了在测试时间推理期间光谱和科学标签的高保真重建，并简要讨论了结果的科学解释，以及这种生成模型的重要性。

## 天体物理学仪器和方法(astro-ph.IM:Instrumentation and Methods for Astrophysics)

该领域共有 1 篇论文

### Evaluation of EAS directions based on TAIGA HiSCORE data using fully connected neural networks 
[[arxiv](https://arxiv.org/abs/2502.13851)] [[cool](https://papers.cool/arxiv/2502.13851)] [[pdf](https://arxiv.org/pdf/2502.13851)]
> **Authors**: A. P. Kryukov,S. P. Polyakov,Yu. Yu. Dubenskaya,E. O. Gres,E. B. Postnikov,P. A. Volchugov,D. P. Zhurov
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: The work was reported on the 8th International Conference on Deep Learning in Computational Physics (DLCP2025), June 19-21, 2024, Moscow, Russia (https://dlcp2024.sinp.msu.ru/). To bee published in Moscow University Physics Bulletin
- **标题**: 使用完全连接的神经网络根据TAIGA HISCORE数据评估EAS方向
- **领域**: 天体物理学仪器和方法,高能天体物理现象,机器学习
- **摘要**: 广泛的空气淋浴的方向可用于确定γ量子的来源，并在估计主要粒子能量中起重要作用。来自TAIGA实验的一系列非成像Cherenkov检测站HISCORE的数据可以使用高精度来估算淋浴方向。在这项工作中，我们使用对gamma Quanta进行蒙特卡洛模拟的TAIGA HISCORE数据训练的人工神经网络来获得淋浴方向估计。神经网络是具有跳过连接的多层感知器，使用来自几个HISCORE站的部分数据作为输入；复合估计值来自神经网络的多个单独估计。我们应用了两阶段算法，其中使用第一阶段获得的方向估计值用于转换输入数据并完善估计值。最终估计的平均误差小于0.25度。该方法将用于对TAIGA实验中使用的几种类型的检测器的数据进行多模式分析。

## 材料科学(cond-mat.mtrl-sci:Materials Science)

该领域共有 3 篇论文

### Explainable Multimodal Machine Learning for Revealing Structure-Property Relationships in Carbon Nanotube Fibers 
[[arxiv](https://arxiv.org/abs/2502.07400)] [[cool](https://papers.cool/arxiv/2502.07400)] [[pdf](https://arxiv.org/pdf/2502.07400)]
> **Authors**: Daisuke Kimura,Naoko Tajima,Toshiya Okazaki,Shun Muroga
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 33 pages, 9 figures
- **标题**: 可解释的多模式的机器学习，用于揭示碳纳米管纤维中的结构特质关系
- **领域**: 材料科学,软凝聚态物质,人工智能,机器学习,数据分析、统计和概率
- **摘要**: 在这项研究中，我们提出了可解释的多模式机器学习（EMML），该学习将碳纳米管（CNT）纤维制成的碳纳米管（CNT）纤维进行特征提取的因子分析，将各种数据类型（多模式数据）的分析整合在一起。该方法是一种强大的方法，可以阐明有关材料特性的机制，在该机制中，多阶段的制造条件和多尺度结构具有复杂的影响。因此，在我们的情况下，这种方法有助于我们了解各种规模的不同处理步骤和结构如何影响CNT纤维的最终特性。该分析针对从纳米级到宏观的靶向结构，包括CNT分散体的聚集大小分布和CNT的有效长度。此外，由于某些类型的数据很难使用标准方法来解释，因此使用负矩阵分解（NMF）分析了提取确定结果的关键特征的挑战到解释分布数据。 Shapley加性解释（SHAP）的贡献分析表明，小的，均匀分布的骨料对于提高断裂强度至关重要，而有效长度的CNT是提高电导率的重要因素。该分析还确定了这些关键因素的阈值和趋势，以帮助定义优化CNT纤维性能所需的条件。 EMML不限于CNT纤维，而可以应用于源自纳米材料的其他材料的设计，使其成为开发各种高级材料的有用工具。这种方法为推进数据驱动的材料研究提供了基础。

### Towards an automated workflow in materials science for combining multi-modal simulative and experimental information using data mining and large language models 
[[arxiv](https://arxiv.org/abs/2502.14904)] [[cool](https://papers.cool/arxiv/2502.14904)] [[pdf](https://arxiv.org/pdf/2502.14904)]
> **Authors**: Balduin Katzer,Steffen Klinder,Katrin Schulz
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 迈向材料科学的自动化工作流，用于使用数据挖掘和大型语言模型组合多模式模拟和实验信息
- **领域**: 材料科学,机器学习
- **摘要**: 为了检索和比较材料科学中的模拟和实验的科学数据，必须易于访问和机器可读数据，以使和量化各种材料科学现象。开放科学的最新进展利用了数据的可访问性。但是，大多数信息都是在科学文档中编码的，这些信息限制了找到合适文献和材料特性的能力。该手稿展示了一个自动化的工作流，该工作流程使用自然语言处理和语言，从科学文献到机器可读的文本，图，表格，方程式和元数据的可读数据结构，以及视觉变压器模型，以生成机器可读数据库。可以用本地数据丰富机器可读数据库，例如未发表或私人材料数据，导致知识综合。该研究表明，这种自动化工作流程从多模式输入数据中提取信息检索，近端的上下文检测和材料特性，示例性地显示了面部中心立方单晶的微观结构分析的研究领域。最终，基于检索的演示生成（RAG）大语言模型（LLM）使一个快速有效的问题回答聊天机器人。

### Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization 
[[arxiv](https://arxiv.org/abs/2502.18604)] [[cool](https://papers.cool/arxiv/2502.18604)] [[pdf](https://arxiv.org/pdf/2502.18604)]
> **Authors**: Grace Guinan,Addison Salvador,Michelle A. Smeaton,Andrew Glaws,Hilary Egan,Brian C. Wyatt,Babak Anasori,Kevin R. Fiedler,Matthew J. Olszta,Steven R. Spurgeon
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 33 pages, 6 figures
- **标题**: 注意差距：桥接AI愿望与自主表征的现实之间的鸿沟
- **领域**: 材料科学,人工智能
- **摘要**: 材料科学在“人工智能时代”是什么样的？每个材料域的合成，表征和建模都以独特的挑战和约束为动机，这是一个不同的答案。这项工作着重于电子显微镜中自主表征的巨大潜力。我们介绍了在开发域感知的多模式模型的最新进步，以描述复杂的原子系统。然后，我们解决了自主显微镜的理论承诺与当前的实际局限性之间的关键差距，展示了最近的成功，同时突出了必要的发展，以实现强大的现实世界自治。

## 人工智能(cs.AI:Artificial Intelligence)

该领域共有 48 篇论文

### Imitation Game for Adversarial Disillusion with Multimodal Generative Chain-of-Thought Role-Play 
[[arxiv](https://arxiv.org/abs/2501.19143)] [[cool](https://papers.cool/arxiv/2501.19143)] [[pdf](https://arxiv.org/pdf/2501.19143)]
> **Authors**: Ching-Chun Chang,Fan-Yun Chen,Shih-Hong Gu,Kai Gao,Hanrui Wang,Isao Echizen
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: 模仿幻觉的模仿游戏，具有多模式生成链的角色扮演
- **领域**: 人工智能,密码学和安全,计算机视觉和模式识别
- **摘要**: 作为人工智能的基石，机器的感知面临着对抗性幻想带来的基本威胁。这些对抗性攻击以两种主要形式表现出来：演绎幻觉，其中特定的刺激是基于受害者模型的一般决策逻辑和归纳幻觉的，在受害者模型的一般决策逻辑中是由特定的刺激塑造的。前者利用模型的决策边界来创建一个刺激，当应用时，该刺激会干扰其决策过程。后者在模型中加强了条件反射，在学习阶段嵌入后门，该后门在刺激触发时会引起异常行为。对抗性幻觉的多方面性质要求建立统一的防御框架，以解决各种形式的攻击方面的脆弱性。在这项研究中，我们根据模仿游戏的概念提出了一个幻灭范式。模仿游戏的核心是多模式的生成剂，在经过思考的推理的指导下，观察，内部和重建了样本的语义本质，从经典的追求将样本倒流到其原始状态。作为概念证明，我们使用多模式生成对话代理进行实验模拟，并在各种攻击方案下评估方法。

### Zero-Shot Warning Generation for Misinformative Multimodal Content 
[[arxiv](https://arxiv.org/abs/2502.00752)] [[cool](https://papers.cool/arxiv/2502.00752)] [[pdf](https://arxiv.org/pdf/2502.00752)]
> **Authors**: Giovanni Pio Delvecchio,Huy Hong Nguyen,Isao Echizen
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 零拍的警告生成错误的多模式内容
- **领域**: 人工智能,计算语言学,信息检索
- **摘要**: 错误信息的普遍流行引起了重大的社会问题。在真实的图像与虚假文本配对的情况下，外观上的错误信息特别具有欺骗性，很容易误导观众。大多数现有检测方法主要评估图像文本的一致性，但通常缺乏足够的解释，这对于有效揭露错误信息至关重要。我们提出了一个模型，该模型通过交叉模式一致性检查检测多模式错误信息，需要最小的训练时间。此外，我们提出了一个轻巧的模型，该模型仅使用参数的三分之一实现竞争性能。我们还引入了一项双重用途的零照片学习任务，以生成上下文化警告，实现自动启动并增强用户理解。对产生的警告的定性和人类评估既凸显了我们方法的潜力和局限性。

### MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.00698)] [[cool](https://papers.cool/arxiv/2502.00698)] [[pdf](https://arxiv.org/pdf/2502.00698)]
> **Authors**: Huanqia Cai,Yijun Yang,Winston Hu
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: MM-IQ：在多模型中基准类似人类的抽象和推理
- **领域**: 人工智能,计算机视觉和模式识别
- **摘要**: 智商测试已成为评估人类认知能力的基础方法，故意将评估与语言背景，语言水平或特定领域的知识分离，以隔离抽象和推理方面的核心能力。然而，目前，人工智能研究缺乏系统的基准来量化多模式系统中这些关键认知维度。为了解决这个关键的差距，我们提出了MM-IQ，这是一个全面的评估框架，其中包括2,710个精心策划的测试项目，涵盖了8种不同的推理范式。通过对领先的开源和专有多模式的系统评估，我们的基准揭示了惊人的局限性：即使是最先进的架构也只能达到比随机机会的略有优势（27.49％和25％的基线精度）。这种实质性的鸿沟强调了当前多模式系统在近似基本的人类推理能力方面的不足，强调了对范式转移进步以弥合这种认知鸿沟的需求。

### Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach 
[[arxiv](https://arxiv.org/abs/2502.00577)] [[cool](https://papers.cool/arxiv/2502.00577)] [[pdf](https://arxiv.org/pdf/2502.00577)]
> **Authors**: Changdae Oh,Zhen Fang,Shawn Im,Xuefeng Du,Yixuan Li
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 了解分配变化的多模式LLM：一种信息理论方法
- **领域**: 人工智能,计算语言学,机器学习
- **摘要**: 多模式的大语言模型（MLLM）显示出有希望的功能，但在分配变化中挣扎，评估数据与说明调谐分布不同。尽管以前的作品提供了经验评估，但我们认为，建立一个可以表征和量化MLLM风险的正式框架对于确保MLLM在现实世界中的安全和可靠应用是必要的。通过采用信息理论的观点，我们提出了第一个理论框架，该框架可以量化分布偏移的最大MLLM风险。我们框架的核心是引入有效的共同信息（EMI），这是一个原则上的指标，可量化输入查询和模型响应之间的相关性。我们得出了分布（ID）和分布（OOD）数据之间的EMI差异的上限，并将其连接到视觉和文本分布差异。在实际基准数据集上进行的广泛实验，跨越61个偏移方案，从经验上验证了我们的理论见解。

### SensorChat: Answering Qualitative and Quantitative Questions during Long-Term Multimodal Sensor Interactions 
[[arxiv](https://arxiv.org/abs/2502.02883)] [[cool](https://papers.cool/arxiv/2502.02883)] [[pdf](https://arxiv.org/pdf/2502.02883)]
> **Authors**: Xiaofan Yu,Lanxiang Hu,Benjamin Reichman,Dylan Chu,Rushil Chandrupatla,Xiyuan Zhang,Larry Heck,Tajana Rosing
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: Under review
- **标题**: 传感器：在长期多模式相互作用中回答定性和定量问题
- **领域**: 人工智能,人机交互
- **摘要**: 与传感系统的自然语言互动对于使所有用户能够理解传感器数据及其对日常生活的影响至关重要。但是，通常以问答方式（QA）操作的现有系统在其可以处理的传感器数据的持续时间和复杂性方面受到了显着限制。在这项工作中，我们介绍了Sensorchat，这是第一个端到端QA系统，旨在使用多模式和高维数据（包括时间序列）进行长期传感器监视。 Sensorchat有效地回答了定性（需要高级推理）和定量（需要从传感器数据中得出的准确响应）问题。为了实现这一目标，Sensorchat使用创新的三阶段管道，其中包括问题分解，传感器数据查询和答案组件。第一阶段和第三阶段利用大型语言模型（LLM）进行直观的人类互动，并指导传感器数据查询过程。与现有的多模式LLM不同，Sensorchat结合了一个明确的查询阶段，以精确地从长期传感器数据中提取事实信息。我们实现了传感器，并演示了其在云服务器上实时交互的能力，同时也能够在量化后完全在边缘平台上运行。全面的质量检查评估表明，在定量问题上，SensorChat比最先进的系统达到了高达26％的答案准确性。此外，一项具有八名志愿者的用户研究强调了Sensorchat在处理定性和开放式问题方面的有效性。

### Secure & Personalized Music-to-Video Generation via CHARCHA 
[[arxiv](https://arxiv.org/abs/2502.02610)] [[cool](https://papers.cool/arxiv/2502.02610)] [[pdf](https://arxiv.org/pdf/2502.02610)]
> **Authors**: Mehul Agarwal,Gauri Agarwal,Santiago Benoit,Andrew Lippman,Jean Oh
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-05
> **comment**: NeurIPS 2024 Creative AI Track
- **标题**: 通过Charcha安全和个性化的音乐与视频发电
- **领域**: 人工智能,计算机视觉和模式识别,人机交互,多媒体
- **摘要**: 音乐是一种深刻的个人体验，我们的目的是通过完全自动化的管道来进行个性化音乐视频的生成来增强这种体验。我们的作品使听众不仅可以通过音乐视频，还可以根据音乐中的歌词，节奏和情感创建个性化，一致和上下文驱动的视觉效果来成为消费者。管道结合了多模式翻译和发电技术，并利用听众的图像中的低级别适应来创建反映音乐和个人的沉浸式音乐视频。为了确保用户身份的道德使用，我们还引入了Charcha（专利申请），这是一种面部身份验证协议，可保护人们免受未经授权使用其面部的使用，同时又收集了从用户个性化视频的授权图像。因此，本文提供了一个安全，创新的框架，用于创建深厚的个性化音乐视频。

### PerPO: Perceptual Preference Optimization via Discriminative Rewarding 
[[arxiv](https://arxiv.org/abs/2502.04371)] [[cool](https://papers.cool/arxiv/2502.04371)] [[pdf](https://arxiv.org/pdf/2502.04371)]
> **Authors**: Zining Zhu,Liang Zhao,Kangheng Lin,Jinze Yang,En Yu,Chenglong Liu,Haoran Wei,Jianjian Sun,Zheng Ge,Xiangyu Zhang
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: Perpo：通过判别奖励优化感知偏好优化
- **领域**: 人工智能,计算语言学,机器学习
- **摘要**: 本文介绍了感知偏好优化（PERPO），这是一种旨在应对生成预训练的多模式大型语言模型（MLLM）中视觉歧视挑战的感知对准方法。为了使MLLM与人类的视觉感知过程保持一致，Perpo采用歧视性奖励来收集各种负面样本，然后进行列表偏好优化以对它们进行排名。通过利用奖励作为排名的定量余量，我们的方法有效地桥接了生成性优先优化和歧视性的经验风险最小化。 Perpo显着增强了MLLM的视觉歧视能力，同时保持其生成优势，减轻图像无关奖励黑客入侵，并确保在视觉任务中保持一致的性能。这项工作标志着朝着更感知和多功能的MLLM迈出的至关重要的一步。我们还希望Perpo鼓励社区重新考虑MLLM对齐策略。

### LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning 
[[arxiv](https://arxiv.org/abs/2502.05453)] [[cool](https://papers.cool/arxiv/2502.05453)] [[pdf](https://arxiv.org/pdf/2502.05453)]
> **Authors**: Hanqing Yang,Jingdi Chen,Marie Siew,Tania Lorido-Botran,Carlee Joe-Wong
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 具有自适应分层知识图的LLM驱动分散生成代理用于合作计划
- **领域**: 人工智能,多代理系统
- **摘要**: 在动态开放世界情景中开发智能代理商进行长期合作是多代理系统的主要挑战。传统的多代理增强学习（MARL）框架等集中式培训分散执行（CTDE）挣扎着可伸缩性和灵活性。他们需要集中的长期计划，这在没有自定义奖励功能的情况下很难，并且在处理多模式数据时面临挑战。 CTDE方法还采用固定的合作策略，使其在代理需要独立适应和计划的动态环境中不切实际。为了解决分散的多代理合作，我们建议在新型的多代理手工艺环境中分散的自适应知识记忆和结构化通信系统（DAMC）。通过大型语言模型（LLM）驱动的我们的生成代理，通过利用外部知识和语言来长期计划和推理，比传统的MARL代理更可扩展。 DAMC并没有完全共享所有过去经验的信息，而是引入了一个多模式的内存系统，该系统作为层次知识图和结构化通信协议组织，以优化代理合作。这使代理可以从过去的交互中推理并有效地共享相关信息。关于新型多代理开放世界任务的实验表明，在任务效率和协作方面，DAMC在任务效率和协作方面的表现都优于MAL和LLM基准。与单一代理方案相比，两种代理的方案以少63％的步骤实现了相同的目标，而六个代理情景却少了74％，突出了自适应记忆和结构化沟通在实现长期目标方面的重要性。我们在以下网址公开发布我们的项目：https：//happyeureka.github.io/damcs。

### Can We Trust AI Benchmarks? An Interdisciplinary Review of Current Issues in AI Evaluation 
[[arxiv](https://arxiv.org/abs/2502.06559)] [[cool](https://papers.cool/arxiv/2502.06559)] [[pdf](https://arxiv.org/pdf/2502.06559)]
> **Authors**: Maria Eriksson,Erasmo Purificato,Arman Noroozian,Joao Vinagre,Guillaume Chaslot,Emilia Gomez,David Fernandez-Llorca
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: Submitted to ACM Conference on Fairness, Accountability, and Transparency (FAccT) 2025
- **标题**: 我们可以信任AI基准吗？ AI评估中当前问题的跨学科审查
- **领域**: 人工智能
- **摘要**: 定量人工智能（AI）基准已成为评估AI模型和系统的性能，能力和安全性的基本工具。目前，它们塑造了AI开发的方向，并在监管框架中起着越来越重要的作用。然而，随着影响力的增长，他们对如何以及对他们评估高度敏感主题（包括高影响力能力，安全性和系统性风险）等影响的影响也感到担忧。本文介绍了大约100项研究的跨学科元评估，讨论了过去10年中发表的定量基准测试实践中的缺点。它在基准的设计和应用中汇集了许多细粒度问题（例如，创建数据集的偏见，文档不足，数据污染以及无法区分信号与噪声的不区分信号的偏见），具有更广泛的社会技术（例如，对单个测试逻辑进行了基于文本的AI模型的越来越多，无法对其他型号进行互动和越来越多地进行竞争的竞争方式，而越来越多地进行了竞争，该模型越来越多地进行了竞争，而越来越多地进行了竞争的方式。我们的审查还重点介绍了当前基准测试实践中的一系列系统缺陷，例如未对准激励措施，构建有效性问题，未知的未知数以及基准结果游戏的问题。此外，它强调了基准实践从根本上是由文化，商业和竞争性动态来塑造的，这些动态通常以更广泛的社会关注为代价来优先考虑最先进的表现。通过提供与现有基准测试程序相关的风险概述，我们将基准中的不成比例信任提出问题，并为改善现实世界情景复杂性中定量AI基准的责任感和相关性的持续努力做出贡献。

### Universal Adversarial Attack on Aligned Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.07987)] [[cool](https://papers.cool/arxiv/2502.07987)] [[pdf](https://arxiv.org/pdf/2502.07987)]
> **Authors**: Temurbek Rahmatullaev,Polina Druzhinina,Matvey Mikhalchuk,Andrey Kuznetsov,Anton Razzhigaev
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: Added an affiliation
- **标题**: 对对齐的多模式LLM的通用对抗性攻击
- **领域**: 人工智能
- **摘要**: 我们提出了对多模式大语言模型（LLMS）的通用对抗性攻击，该模型利用单个优化的图像来覆盖各种查询甚至多个模型的对齐保障。通过通过视觉编码器和语言主题进行反向传播，我们制作了一个合成图像，迫使模型以有针对性的短语响应（例如，'当然，这里是''）或其他不安全的内容，即使是有害提示。在SafeBench基准测试的实验中，我们的方法比现有基准的攻击成功率明显高得多，包括仅文本通用提示（例如，在某些型号上最多可达93％）。我们进一步通过对几个多模式LLMS的训练并在看不见的体系结构上进行测试，进一步证明了跨模型的可传递性。此外，我们方法的多回答变体还会产生更自然的（但仍然是恶意的）响应。这些发现强调了当前的多模式对齐中的关键漏洞，并要求更强大的对抗性防御。我们将在Apache-2.0许可证下发布代码和数据集。警告：本文中多模式LLM产生的某些内容可能对某些读者有冒犯性。

### Recursive Inference Scaling: A Winning Path to Scalable Inference in Language and Multimodal Systems 
[[arxiv](https://arxiv.org/abs/2502.07503)] [[cool](https://papers.cool/arxiv/2502.07503)] [[pdf](https://arxiv.org/pdf/2502.07503)]
> **Authors**: Ibrahim Alabdulmohsin,Xiaohua Zhai
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 18 pages, 9 figures
- **标题**: 递归推理缩放：在语言和多模式系统中可扩展推理的获胜途径
- **领域**: 人工智能,机器学习
- **摘要**: 语言建模的最新研究揭示了两种缩放效果：众所周知的训练计算的改进，以及应用更复杂或计算密集的推理方法而鲜为人知的提升。受到语言分形几何形状的最新发现的启发，我们将递归推理缩放（RINS）作为一种互补的，用于缩放推理时间的插件配方。对于给定的固定模型体系结构和培训计算预算，RIN可大大提高语言建模性能。它还概括了纯语言任务，从而在多模式系统中带来了增长，包括Siglip-B/16的0-Shot Imagenet精度提高了2％。此外，通过得出数据扩展定律，我们表明RINS可以改善渐近性能限制和缩放指数。与最先进的递归技术相比，即使在移动LLM中的“重复全部”策略（RAO）策略相比，这些优势也可以保持。最后，随机RINS不仅可以进一步提高性能，而且还为可选的放弃在测试时间增加推理计算而提供的灵活性会以最小的性能降级。

### EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges 
[[arxiv](https://arxiv.org/abs/2502.08859)] [[cool](https://papers.cool/arxiv/2502.08859)] [[pdf](https://arxiv.org/pdf/2502.08859)]
> **Authors**: Clinton J. Wang,Dean Lee,Cristina Menghini,Johannes Mols,Jack Doughty,Adam Khoja,Jayson Lynch,Sean Hendryx,Summer Yue,Dan Hendrycks
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: Enigmaeval：长期多模式推理挑战的基准
- **领域**: 人工智能,计算语言学
- **摘要**: 随着语言模型掌握现有推理基准，我们需要新的挑战来评估其认知前沿。解决难题的事件是富有挑战的多模式问题的丰富存储库，这些问题测试了广泛的高级推理和知识能力，使它们成为评估边境语言模型的独特测试。我们介绍了Enigmaeval，这是一个来自拼图竞争和事件的问题和解决方案的数据集，这些问题和解决方案探讨了模型执行隐式知识合成和多步推论推理的能力。与现有的推理和知识基准不同，难题解决挑战模型，以发现看似无关的信息之间的隐藏连接以发现解决方案路径。该基准包括1184个各种复杂性的难题 - 每个通常都需要熟练的求解器团队数小时到几天才能完成 - 具有明确的，可验证的解决方案，以实现有效的评估。最先进的语言模型在这些难题上的准确性极低，甚至比其他困难的基准（例如人类的最后考试）要低，在面对需要非结构化和横向推理的问题的挑战时揭示了模型的缺点。

### AutoS$^2$earch: Unlocking the Reasoning Potential of Large Models for Web-based Source Search 
[[arxiv](https://arxiv.org/abs/2502.09913)] [[cool](https://papers.cool/arxiv/2502.09913)] [[pdf](https://arxiv.org/pdf/2502.09913)]
> **Authors**: Zhengqiu Zhu,Yatai Ji,Jiaheng Huang,Yong Zhao,Sihang Qiu,Rusheng Ju
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: Autos $^2 $ earch：解锁大型模型的推理潜力用于基于Web的源搜索
- **领域**: 人工智能,人机交互
- **摘要**: 基于Web的管理系统已被广泛用于风险控​​制和工业安全。但是，有效地将源搜索功能集成到这些系统中，以使决策者能够找到和解决危害（例如，气体泄漏检测）仍然是一个挑战。尽管先前的努力使用网络众包和AI算法来探索来源搜索决策支持，但这些方法在招募人类参与者和时间敏感的情况下的响应时间缓慢而遭受了间接费用。为了解决这个问题，我们介绍了Autos $^2 $ earch，这是一个新颖的框架，利用大型型号在Web应用程序中进行零拍源搜索。 Autos $^2 $ earch在通过基于Web的显示器投影的简化视觉环境中运行，利用旨在模拟人类推理的经过经过经过经过经过经过经验的促进及。多模式大型语言模型（MLLM）将视觉观测转换为语言描述，使LLM能够在四个方向选择上执行语言推理。广泛的实验表明，汽车$^2 $ earch的性能几乎等同于人类协作源搜索，同时消除了对众包劳动的依赖。我们的工作为使用Web工程设计在其他工业应用中设计此类自主系统提供了宝贵的见解。

### Artificial Intelligence in Spectroscopy: Advancing Chemistry from Prediction to Generation and Beyond 
[[arxiv](https://arxiv.org/abs/2502.09897)] [[cool](https://papers.cool/arxiv/2502.09897)] [[pdf](https://arxiv.org/pdf/2502.09897)]
> **Authors**: Kehan Guo,Yili Shen,Gisela Abigail Gonzalez-Montiel,Yue Huang,Yujun Zhou,Mihir Surve,Zhichun Guo,Prayel Das,Nitesh V Chawla,Olaf Wiest,Xiangliang Zhang
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 光谱法中的人工智能：从预测到世代及以后的化学反应前进
- **领域**: 人工智能,机器学习
- **摘要**: 机器学习（ML）和人工智能（AI）的快速出现已经催化了化学的重大转化，但是这些方法将这些方法应用于光谱和光谱数据，称为光谱机器学习（SpectRAML），它仍然相对尚未被逐渐流行。现代光谱技术（MS，NMR，IR，Raman，UV-VIS）产生了不断增长的高维数据，超出了传统的基于专家的工作流程，对自动化和智能分析产生了迫切需求。在这项调查中，我们提供了统一的综述，对远期任务（分子到光谱预测）和倒数任务（频谱到 - 分子推断）的谱系进行了系统检查的最新方法。我们追踪了光谱中ML的历史演变，从早期模式识别到能够进行高级推理的最新基础模型，并提供了代表性神经体系结构的分类学，包括基于图基和基于变压器的方法。在应对数据质量，多模式集成和计算可扩展性等关键挑战时，我们突出了新兴方向，例如合成数据生成，大规模预处理和少数或零摄像的学习。为了培养可重复的研究，我们还发布了一个开源存储库，其中包含最近的论文及其相应的策划数据集（https://github.com/mine-lab-nd/spectrumml_survey_papers）。我们的调查是研究人员的路线图，并指导光谱和AI交集的进步。

### MuDoC: An Interactive Multimodal Document-grounded Conversational AI System 
[[arxiv](https://arxiv.org/abs/2502.09843)] [[cool](https://papers.cool/arxiv/2502.09843)] [[pdf](https://arxiv.org/pdf/2502.09843)]
> **Authors**: Karan Taneja,Ashok K. Goel
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: 5 pages, 3 figures, AAAI-MAKE 2025
- **标题**: MUDOC：互动的多模式的互动文档接地AI系统
- **领域**: 人工智能,人机交互,多媒体
- **摘要**: 多模式AI是建立有效工具以利用人类交流中多种方式的重要一步。构建一个多模式文档的AI系统与长文档进行交互是一个挑战。我们的工作旨在填补直接利用文档中的接地视觉效果以及文档中的文本内容进行响应生成的研究空白。我们提出了基于GPT-4O的交互式对话AI代理“ mudoc”，以通过交织的文本和数字生成文档接地的响应。 Mudoc的智能教科书接口促进了信任度，并通过允许即时导航来源文本和文档中的数字来验证系统响应。我们还基于MUDOC响应强调其优势和局限性讨论定性观察。

### EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents 
[[arxiv](https://arxiv.org/abs/2502.09560)] [[cool](https://papers.cool/arxiv/2502.09560)] [[pdf](https://arxiv.org/pdf/2502.09560)]
> **Authors**: Rui Yang,Hanyang Chen,Junyu Zhang,Mark Zhao,Cheng Qian,Kangrui Wang,Qineng Wang,Teja Venkat Koripella,Marziyeh Movahedi,Manling Li,Heng Ji,Huan Zhang,Tong Zhang
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: 52 pages
- **标题**: 体现台面：全面的基准测试视觉驱动的代理的多模式大语言模型
- **领域**: 人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 利用多模式的大语言模型（MLLM）创建具体的代理，为解决现实世界任务提供了有希望的途径。虽然以语言为中心的体现药物引起了很大的关注，但由于缺乏全面的评估框架，基于MLLM的体现药物仍未被忽视。为了弥合这一差距，我们引入了体现，这是一种广泛的基准测试，旨在评估视觉驱动的体现剂。体现Bench特征：（1）在四个环境中进行的一组1,128个测试任务，从高级语义任务（例如家庭）到涉及原子动作（例如导航和操纵）的低级任务范围； （2）六个精心策划的子集评估基本药物的能力，例如常识性推理，复杂的教学理解，空间意识，视觉感知和长期计划。通过广泛的实验，我们评估了体现膨化板内的19个领先的专有和开源MLLM。我们的发现表明：MLLM在高级任务上表现出色，但在低级操纵中挣扎，最佳模型GPT-4O平均得分仅为28.9％。体现Bench提供了一个多面的标准化评估平台，不仅强调了现有的挑战，而且还提供了有价值的见解，以推动基于MLLM的体现代理。我们的代码可从https://embodiedbench.github.io获得。

### From large language models to multimodal AI: A scoping review on the potential of generative AI in medicine 
[[arxiv](https://arxiv.org/abs/2502.09242)] [[cool](https://papers.cool/arxiv/2502.09242)] [[pdf](https://arxiv.org/pdf/2502.09242)]
> **Authors**: Lukas Buess,Matthias Keicher,Nassir Navab,Andreas Maier,Soroosh Tayebi Arasteh
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 从大型语言模型到多模式AI：关于生成AI潜力的范围评论
- **领域**: 人工智能
- **摘要**: 生成人工智能（AI）模型，例如扩散模型和OpenAI的Chatgpt，正在通过提高诊断准确性和自动化临床工作流程来改变医学。该领域已经迅速发展，从临床文档和决策支持等任务的仅文本大型语言模型发展到能够在单个模型中整合各种数据模式的多模式AI系统，包括成像，文本和结构化数据。这些技术的各种景观以及兴趣不断上升，强调了对其应用和潜力进行全面审查的必要性。该范围审查探讨了多模式AI的演变，突出了其在临床环境中的方法，应用，数据集和评估。遵守PRISMA-SCR指南，我们系统地查询PubMed，IEEE Xplore和Web of Science，优先考虑到2024年底发表的最新研究。在严格的筛选后，包括144篇论文，揭示了这个动态领域中的关键趋势和挑战。我们的发现强调了从单峰方式转变为多模式方法，推动了诊断支持，医疗报告生成，药物发现和对话性AI的创新。但是，仍然存在关键挑战，包括整合异质数据类型，改善模型的解释性，解决道德问题以及在现实世界中临床环境中验证AI系统。这篇综述总结了当前的最新状态，确定了关键差距，并提供了见解，以指导医疗保健中可扩展，可信赖和临床影响力的多模式AI解决方案的开发。

### Visual Graph Question Answering with ASP and LLMs for Language Parsing 
[[arxiv](https://arxiv.org/abs/2502.09211)] [[cool](https://papers.cool/arxiv/2502.09211)] [[pdf](https://arxiv.org/pdf/2502.09211)]
> **Authors**: Jakob Johannes Bauer,Thomas Eiter,Nelson Higuera Ruiz,Johannes Oetsch
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: In Proceedings ICLP 2024, arXiv:2502.08453. This work was partially funded from the Bosch Center for AI
- **标题**: 用ASP和LLM回答语言解析的视觉图询问
- **领域**: 人工智能,计算机视觉和模式识别,计算机科学中的逻辑
- **摘要**: 视觉问题回答（VQA）是一个具有挑战性的问题，需要处理多模式输入。答案集编程（ASP）在这方面表现出很大的潜力，可以增加模块化VQA体系结构的解释性和解释性。在这项工作中，我们解决了如何将ASP与视觉和自然语言处理的模块集成在一起的问题，以求解与图形图像有关的新的，苛刻的VQA变体（而不是符号形式的图形）。包含基于图的结构的图像是一种无处不在的可视化形式。在这里，我们处理了受到公交网络启发的图表的特定问题，并介绍了一个新颖的数据集，该数据集通过添加类似于地铁线的图像来修改现有图表。我们的模块化神经符号方法结合了用于图形解析的光学图识别，这是用于解析标签的验证的光学特征识别神经网络，用于语言处理的大语言模型（LLMS）以及用于推理的ASP。此方法是第一个基线，并且在数据集上达到了73％的总体平均精度。我们的评估提供了进一步的证据，证明了模块化神经符号系统的潜力，尤其是预验证的模型，这些模型不涉及任何进一步的培训和逻辑编程以解决复杂的VQA任务。

### FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading 
[[arxiv](https://arxiv.org/abs/2502.11433)] [[cool](https://papers.cool/arxiv/2502.11433)] [[pdf](https://arxiv.org/pdf/2502.11433)]
> **Authors**: Guojun Xiong,Zhiyang Deng,Keyi Wang,Yupeng Cao,Haohang Li,Yangyang Yu,Xueqing Peng,Mingquan Lin,Kaleb E Smith,Xiao-Yang Liu,Jimin Huang,Sophia Ananiadou,Qianqian Xie
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 旗帜交易：Fusion LLM-Antent具有基于梯度的加固学习用于金融交易
- **领域**: 人工智能,计算工程、金融和科学,交易和市场微观结构
- **摘要**: 对多模式财务数据进行了微调的大型语言模型（LLMS）在各种财务任务中表现出令人印象深刻的推理能力。但是，他们经常在交易式金融市场（例如交易）中与多步，面向目标的方案斗争，在这种情况下，需要复杂的代理方法来改善决策。为了解决这个问题，我们提出了\ textsc {flag-trader}，这是一种统一的体系结构，将语言处理（通过LLMS）与梯度驱动的增强学习（RL）策略优化相结合，其中部分精细的LLM充当策略网络，利用策略网络，在适应性地通过参数效率进行预先培训的知识，并通过参数效率高效地效率高效。通过交易奖励驱动的政策梯度优化，我们的框架不仅可以提高交易中的LLM绩效，还可以提高其他财务域任务的结果。我们提供了广泛的经验证据来验证这些增强能力。

### TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents 
[[arxiv](https://arxiv.org/abs/2502.11418)] [[cool](https://papers.cool/arxiv/2502.11418)] [[pdf](https://arxiv.org/pdf/2502.11418)]
> **Authors**: Geon Lee,Wenchao Yu,Kijung Shin,Wei Cheng,Haifeng Chen
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: AAAI 2025
- **标题**: 时间表：学习与大语言模型代理的上下文化，增强和预测时间序列事件
- **领域**: 人工智能,机器学习
- **摘要**: 时间序列数据在各种应用程序中至关重要，包括气候建模，医疗保健监测和财务分析。了解与实际时间序列数据相关的上下文信息通常对于准确可靠的事件预测至关重要。在本文中，我们介绍了时间序列处理框架，该框架创造性地采用了大型语言模型（LLM）作为时间序列数据的上下文化器，从而扩展了其典型的用法作为预测因素。时间表合并了两个独立的LLM代理：一个人生成一个文本摘要来捕获时间序列的上下文，而另一个则使用此丰富的摘要来做出更明智的预测。此外，时间卡还采用了与LLM代理协同作用的多模式编码器，通过使用封闭式示例的输入相互增加来增强预测性能。现实世界数据集的实验结果表明，时间表优于时间序列事件预测的最先进方法，包括利用LLMS作为预测变量的方法，在F1分数中平均提高了28.75％。

### Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents 
[[arxiv](https://arxiv.org/abs/2502.11357)] [[cool](https://papers.cool/arxiv/2502.11357)] [[pdf](https://arxiv.org/pdf/2502.11357)]
> **Authors**: Vardaan Pahuja,Yadong Lu,Corby Rosset,Boyu Gou,Arindam Mitra,Spencer Whitehead,Yu Su,Ahmed Awadallah
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: 24 pages, 7 figures
- **标题**: 资源管理器：缩放探索驱动的Web轨迹合成多模式Web代理
- **领域**: 人工智能,人机交互
- **摘要**: 在大型多模型模型（LMM）中，最近的成功引发了能够自主完成复杂Web任务的代理的有希望的应用。尽管开源LMM代理在离线评估基准方面取得了重大进步，但在更现实的在线环境中，它们的性能仍然大大低于人类水平的功能。一个关键的瓶颈是缺乏各个域中的多样化和大规模轨迹级别的数据集，这些数据集很昂贵。在本文中，我们通过开发可扩展的配方来解决这一挑战，以合成迄今为止最大，最多样化的轨迹级数据集，其中包含超过94K成功的多模式Web轨迹，涵盖了49k独特的URL，720k屏幕截图和33M Web元素。特别是，我们利用广泛的Web探索和完善来获得各种任务意图。每项成功轨迹的平均成本为28美分，使社区中广泛的用户负担得起。利用此数据集，我们训练Explorer，这是一种多模式Web代理，并在离线和在线Web代理基准（例如Mind2Web-live，Multopodal-Mind2Web和Miniiwob ++）上展示了强劲的性能。此外，我们的实验将数据扩展为提高Web代理功能的关键驱动力。我们希望这项研究能使最先进的LMM代理研究更易于访问。

### Leveraging Multimodal-LLMs Assisted by Instance Segmentation for Intelligent Traffic Monitoring 
[[arxiv](https://arxiv.org/abs/2502.11304)] [[cool](https://papers.cool/arxiv/2502.11304)] [[pdf](https://arxiv.org/pdf/2502.11304)]
> **Authors**: Murat Arda Onsu,Poonam Lohan,Burak Kantarci,Aisha Syed,Matthew Andrews,Sean Kennedy
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: 6 pages, 7 figures, submitted to 30th IEEE International Symposium on Computers and Communications (ISCC) 2025
- **标题**: 利用实例细分辅助的多模态llms进行智能交通监控
- **领域**: 人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 强大而有效的交通监控系统对于智能城市和智能运输系统（ITS），使用传感器和摄像机来跟踪车辆运动，优化交通流量，减少拥堵，增强道路安全并实现实时自适应交通控制至关重要。交通监控模型必须全面了解动态的城市条件，并为有效管理提供直观的用户界面。这项研究利用LLAVA视觉接地多模式大型语言模型（LLM）在实时Quanser Interactive Lab Simulation平台上进行交通监视任务，涵盖了诸如交叉点，拥塞和碰撞之类的场景。放置在多个城市位置的摄像机从模拟中收集实时图像，这些图像被带入LLAVA模型中，并通过查询进行分析。集成到相机中的实例分割模型突出了关键要素，例如车辆和行人，从而增强了训练和吞吐量。该系统在识别车辆位置方面的准确性为84.3％，在确定转向方向，优于传统模型方面，该系统的准确性为76.4％。

### The Philosophical Foundations of Growing AI Like A Child 
[[arxiv](https://arxiv.org/abs/2502.10742)] [[cool](https://papers.cool/arxiv/2502.10742)] [[pdf](https://arxiv.org/pdf/2502.10742)]
> **Authors**: Dezhi Luo,Yijiang Li,Hokin Deng
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 像孩子一样成长AI的哲学基础
- **领域**: 人工智能
- **摘要**: 尽管在高水平的推理方面表现出色，但当前的语言模型在现实情况下缺乏鲁棒性，并且在对人类直观的基本解决问题的基本任务上表现不佳。本文认为，这两种挑战源于人类和机器认知发展之间的核心差异。尽管这两种系统都依赖于增加代表权，但人类出现语言模型中缺乏核心知识基础的认知结构，从而开发了可靠的，可推广的能力，在这些能力中，复杂的技能在其各自领域内的简单技能基础上基于更简单。它探讨了人类核心知识的经验证据，分析了语言模型为什么无法获取它，并认为这种限制不是固有的建筑约束。最后，它概述了通过认知原型制定策略大规模生成的合成训练数据，可以将核心知识系统地整合到未来的多模式模型中。

### Demographic User Modeling for Social Robotics with Multimodal Pre-trained Models 
[[arxiv](https://arxiv.org/abs/2502.10642)] [[cool](https://papers.cool/arxiv/2502.10642)] [[pdf](https://arxiv.org/pdf/2502.10642)]
> **Authors**: Hamed Rahimi,Mouad Abrini,Mahdi Khoramshahi,Mohamed Chetouani
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 具有多模式预训练模型的社交机器人技术的人口统计用户建模
- **领域**: 人工智能,计算机视觉和模式识别
- **摘要**: 本文研究了基于视觉语言人口统计数据的用户分析任务中多模式预训练模型的性能。这些模型对于适应社会机器人技术中人类用户的需求和偏好至关重要，从而提供个性化的响应并提高互动质量。首先，我们介绍了两个专门策划的数据集，以表示从用户面部图像衍生的人口统计学特征。接下来，我们在这些数据集上（无论是在开箱即用的状态和微调之后）评估了这些数据集上突出的对比度多模式预训练的模型的性能。初始结果表明，剪辑在不进行微调的情况下将图像与人口统计学描述匹配时进行了次优。尽管微调显着提高了其预测能力，但该模型在有效地概括了细微的人口细微差别时仍表现出局限性。为了解决这个问题，我们建议采用蒙版的图像建模策略来改善概括并更好地捕获微妙的人口属性。这种方法为增强多模式用户建模任务中的人口敏感性提供了一种途径。

### USER-VLM 360: Personalized Vision Language Models with User-aware Tuning for Social Human-Robot Interactions 
[[arxiv](https://arxiv.org/abs/2502.10636)] [[cool](https://papers.cool/arxiv/2502.10636)] [[pdf](https://arxiv.org/pdf/2502.10636)]
> **Authors**: Hamed Rahimi,Adil Bahaj,Mouad Abrini,Mahdi Khoramshahi,Mounir Ghogho,Mohamed Chetouani
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 用户-VLM 360：个性化视觉语言模型，具有用户意识调整的社交人类机器人互动
- **领域**: 人工智能,人机交互,机器人技术
- **摘要**: 将视觉模型的整合到机器人系统中构成了使机器以更直观的方式与周围环境相互作用的重大进步。虽然VLM提供丰富的多模式推理，但现有方法缺乏特定于用户的适应性，通常依靠无法说明个体行为，上下文或社会情感细微差别的通用交互范式。尝试自定义时，用户数据中的偏见产生了道德问题，冒着排除或不公平治疗的风险。为了解决这些双重挑战，我们提出了用户-VLM 360°，这是一个整体框架，将多模式用户建模与偏见感知优化整合在一起。我们的方法功能：（1）使用视觉语言信号实时适应交互作用的用户意识调整； （2）通过偏好优化偏置缓解； （3）以人口统计学，情感和关系元数据注释的360°社会情感互动数据集。跨八个基准测试的评估表明了最新的结果：个性化VQA中的 +35.3％F1，面部特征理解中的 +47.5％F1，降低15％的偏置和30倍的速度，比基线速度加速30倍。消融研究证实了组件功效，而在胡椒机器人上的部署可以验证各种用户的实时适应性。我们开放源参数有效的3B/10B模型和负责调整的道德验证框架。

### POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning 
[[arxiv](https://arxiv.org/abs/2502.10038)] [[cool](https://papers.cool/arxiv/2502.10038)] [[pdf](https://arxiv.org/pdf/2502.10038)]
> **Authors**: Jiawei Cheng,Jingyuan Wang,Yichuan Zhang,Jiahao Ji,Yuanshao Zhu,Zhibo Zhang,Xiangyu Zhao
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: AAAI 25
- **标题**: Poi-Enhancer：POI表示学习的基于LLM的语义增强框架
- **领域**: 人工智能
- **摘要**: POI表示学习在处理与用户移动性数据相关的任务中起着至关重要的作用。最近的研究表明，通过多模式信息丰富POI表示可以显着提高其任务绩效。以前，包含在POI表示形式中的文本信息通常仅涉及POI类别或登机内容，从而导致现有方法中的文本特征相对较弱。相比之下，已经发现接受广泛文本数据培训的大型语言模型（LLM）具有丰富的文本知识。但是，利用这种知识来增强POI表示学习提出了两个关键挑战：首先，如何有效地从LLM中提取与POI相关的知识，其次，如何整合提取的信息以增强POI表示。为了应对这些挑战，我们提出了Poi-Enhancer，这是一个便携式框架，利用LLMS改善经典POI学习模型产生的POI表示。我们首先设计了三个专门提示，以有效地从LLM中提取语义信息。然后，双功能对齐模块增强了提取信息的质量，而语义特征融合模块则保持其完整性。然后，交叉注意的融合模块将这种高质量信息完全自适应地整合到POI表示中，多视图对比度学习进一步将人类理解的语义信息注入了这些表示形式。在三个现实世界数据集上进行的广泛实验证明了我们框架的有效性，显示了所有基线表示的显着改善。

### A Survey on Bridging EEG Signals and Generative AI: From Image and Text to Beyond 
[[arxiv](https://arxiv.org/abs/2502.12048)] [[cool](https://papers.cool/arxiv/2502.12048)] [[pdf](https://arxiv.org/pdf/2502.12048)]
> **Authors**: Shreya Shukla,Jose Torres,Abhijit Mishra,Jacek Gwizdka,Shounak Roychowdhury
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 一项有关桥接脑电图和生成AI的调查：从图像和文本到超越
- **领域**: 人工智能,人机交互,机器学习
- **摘要**: 大脑计算机界面（BCIS）和生成人工智能（Genai）的整合已经开放了大脑信号解码，促进辅助通信，神经表示学习和多模式整合的新边界。 BCI，尤其是那些利用脑电图（EEG）的人，提供了一种将神经活动转化为有意义的输出的非侵入性手段。深度学习的最新进展，包括生成的对抗性网络（GAN）和基于变形金刚的大型语言模型（LLMS），已显着改善了基于脑电图的图像，文本和语音的产生。本文提供了对基于EEG的多模式生成的最先进的文献综述，重点介绍了（i）通过GAN，变异自动编码器（VAE）和扩散模型以及（ii）eeg-to-to-toxt Text生成杠杆化杠杆杠杆化的语言模型和损坏学习方法。此外，我们讨论了EEG到语音综合的新兴领域，这是一种不断发展的多模式前沿。我们重点介绍关键数据集，用例，挑战和EEG功能编码方法，这些方法是生成方法的基础。通过提供基于EEG的生成AI的结构化概述，该调查旨在为研究人员和从业人员提供洞察力，以推动神经解码，增强辅助技术并扩大脑部计算机相互作用的前沿。

### GRAPHGPT-O: Synergistic Multimodal Comprehension and Generation on Graphs 
[[arxiv](https://arxiv.org/abs/2502.11925)] [[cool](https://papers.cool/arxiv/2502.11925)] [[pdf](https://arxiv.org/pdf/2502.11925)]
> **Authors**: Yi Fang,Bowen Jin,Jiacheng Shen,Sirui Ding,Qiaoyu Tan,Jiawei Han
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: GraphGpt-o：图形上的协同多模式理解和生成
- **领域**: 人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 多模式大语言模型（MLLM）的快速发展使大型语言模型（LLM）框架内的多种模式（包括文本和图像）的集成。但是，文本和图像通常是互连的，形成多模式归因图（MMAG）。毫无疑问，mllms如何在此类图上将关系信息（\ textit {i.e。}，图形结构）和语义信息（\ textIt {i.e。，}文本和图像）合并到多模式理解和生成的此类图上。在本文中，我们提出了graphgpt-o，它支持在mmags上的omni-multimodal理解和创建。我们首先全面研究线性化变体，以转化语义和结构信息作为MLLM的输入。然后，我们提出了一个层次对准器，该层次对准器可以启用深图编码，弥合MMAGS和MLLM之间的间隙。最后，我们探讨了推理选择，将MLLM适应图形方案中的交错文本和图像生成。来自不同领域的三个数据集上的广泛实验证明了我们提出的方法的有效性。接受后，数据集和代码将在接受后开源。

### HintsOfTruth: A Multimodal Checkworthiness Detection Dataset with Real and Synthetic Claims 
[[arxiv](https://arxiv.org/abs/2502.11753)] [[cool](https://papers.cool/arxiv/2502.11753)] [[pdf](https://arxiv.org/pdf/2502.11753)]
> **Authors**: Michiel van der Meer,Pavel Korshunov,Sébastien Marcel,Lonneke van der Plas
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 提示图：具有真实和合成主张的多模式检查数据集
- **领域**: 人工智能
- **摘要**: 错误的信息可以通过事实检查来反驳，但是该过程代价高昂且缓慢。确定Checkworthy主张是第一步，自动化可以帮助扩展事实检查者的努力。但是，检测方法与内容为1）多模式，2）来自不同领域的内容和3）合成。我们介绍了Thintsoftruth，这是一个公共数据集，用于$ 27 $ K现实世界和合成图像/索赔对，用于多模式的检查范围检测。实际和合成数据的组合使该数据集具有独特性，并且是基准测试检测方法的理想选择。我们比较微调和促使大型语言模型（LLMS）。我们发现，配置良好的轻巧基于文本的编码器与多模型模型相当，但第一个唯一的重点是识别非寻求的内容。多模式LLM可以更准确，但具有巨大的计算成本，使其在大规模应用中不切实际。当面对合成数据时，多模式模型的性能更加牢固

### A Survey of Automatic Prompt Engineering: An Optimization Perspective 
[[arxiv](https://arxiv.org/abs/2502.11560)] [[cool](https://papers.cool/arxiv/2502.11560)] [[pdf](https://arxiv.org/pdf/2502.11560)]
> **Authors**: Wenwu Li,Xiangfeng Wang,Wenhao Li,Bo Jin
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 19 pages, 4 figures
- **标题**: 自动及时工程的调查：优化视角
- **领域**: 人工智能,机器学习
- **摘要**: 基础模型的兴起已将重点从资源密集的微调转变为促使工程设计，这是一种通过输入设计而不是重量更新来指导模型行为的范式。手动及时工程虽然面临可伸缩性，适应性和跨模式对齐的限制，自动化方法，基于基础模型（FM）的优化，进化方法，基于梯度的优化和强化学习，但提供了有希望的解决方案。但是，现有的调查仍然跨越模态和方法论。本文通过统一的优化理论镜头介绍了对自动化及时工程的首次全面调查。我们将迅速优化的正式优化作为一个最大化问题，而不是离散，连续和混合提示空间，通过其优化变量（指令，软提示，示例），特定于任务的目标和计算框架来系统地组织方法。通过将理论表述与跨文本，视觉和多模式领域的实际实现相结合，该调查为研究人员和从业人员建立了一个基本框架，同时突出了在约束优化和面向代理的及时设计方面未经驱动的边界。

### Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding 
[[arxiv](https://arxiv.org/abs/2502.11492)] [[cool](https://papers.cool/arxiv/2502.11492)] [[pdf](https://arxiv.org/pdf/2502.11492)]
> **Authors**: Kung-Hsiang Huang,Can Qin,Haoyi Qiu,Philippe Laban,Shafiq Joty,Caiming Xiong,Chien-Sheng Wu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Code and data are available at https://github.com/SalesforceAIResearch/CogAlign
- **标题**: 为什么视觉语言模型在视觉算术中挣扎？迈向增强图表和几何理解
- **领域**: 人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 视觉语言模型（VLM）在多模式任务中取得了显着的进步，但是它们经常在视觉算术，看似简单的功能（例如对象计数或长度比较）上挣扎，这对于相关的复杂任务（例如图表理解和几何理解和几何推理）至关重要。在这项工作中，我们首先通过一系列探测任务，重点关注基本的视觉算术。我们的分析表明，尽管预训练的视力编码通常会捕获足够的信息，但文本解码器通常无法正确解码算术推理。为了解决这个问题，我们提出了Cogalign，这是一种受Piaget的认知发展理论启发的新型培训策略。 Cogalign训练VLM，以识别视觉转换下的不变属性。我们证明，这种方法可显着提高我们提出的探测任务上三种不同VLM的性能。此外，Cogalign的巧克力平均增长了4.6％，数学视频的2.9％，表现优于或匹配的监督微调方法，同时仅需要少60％的培训数据。这些结果突出了Cogalign在改善基本视觉算术能力及其转移到下游任务方面的有效性和概括性。

### MatterChat: A Multi-Modal LLM for Material Science 
[[arxiv](https://arxiv.org/abs/2502.13107)] [[cool](https://papers.cool/arxiv/2502.13107)] [[pdf](https://arxiv.org/pdf/2502.13107)]
> **Authors**: Yingheng Tang,Wenbin Xu,Jie Cao,Jianzhu Ma,Weilu Gao,Steve Farrell,Benjamin Erichson,Michael W. Mahoney,Andy Nonaka,Zhi Yao
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: Matterchat：材料科学的多模式LLM
- **领域**: 人工智能,机器学习
- **摘要**: 理解和预测无机材料的特性对于加速材料科学和驱动能源，电子设备及以后的应用的进步至关重要。通过多模式的大语言模型（LLM）将材料结构数据与基于语言的信息集成在一起，通过增强人类互动来支持这些努力。但是，一个关键的挑战在于将原子结构完全分辨到LLMS中。在这项工作中，我们介绍了MatterChat，这是一种多功能结构感知的多模式LLM，将材料结构数据和文本输入统一为单个粘性模型。 MatterChat采用桥接模块来有效地将验证的机器学习间潜力与预处理的LLM保持一致，从而降低培训成本并提高灵活性。我们的结果表明，MatterChat显着提高了材料性质预测和人类相互作用的性能，超过了GPT-4等通用LLM。我们还证明了它在更先进的科学推理和逐步材料合成等应用中的有用性。

### A consensus set for the aggregation of partial rankings: the case of the Optimal Set of Bucket Orders Problem 
[[arxiv](https://arxiv.org/abs/2502.13769)] [[cool](https://papers.cool/arxiv/2502.13769)] [[pdf](https://arxiv.org/pdf/2502.13769)]
> **Authors**: Juan A. Aledo,José A. Gámez,Alejandro Rosete
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 26 pages, 2 figures
- **标题**: 部分排名集合的共识设置：最佳订单订单问题的情况
- **领域**: 人工智能
- **摘要**: 在等级聚合问题（RAP）中，解决方案通常是共识排名，可以推广一组输入顺序。有不同的变体不仅在用作输入和输出的排名类型方面有所不同，而且在评估所需输出排名质量的目标函数方面也有所不同。相比之下，在某些机器学习任务（例如亚组发现）或多模式优化任务中，注意专门用于获取多种模型/结果，以说明输入数据或搜索景观的多样性。因此，在本文中，我们建议将作为RAP的解决方案提供一组排名，以更好地解释输入顺序中表达的偏好。我们通过最佳的存储订单问题（OBOP）来体现我们的建议，该RAP在于找到单个共识排名（带有纽带），该排名概括为一组输入排名编码为优先级矩阵。为了解决这个问题，我们介绍了最佳的存储订单问题集（OSBOP），这是OBOP的概括，旨在产生单个排名作为输出，而是一组共识排名。提出了实验结果来说明这一建议，表明通过提供一组共识排名，解决方案的适应性如何显着相对于原始的OBOP之一，而不会失去可理解性。

### Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework 
[[arxiv](https://arxiv.org/abs/2502.14864)] [[cool](https://papers.cool/arxiv/2502.14864)] [[pdf](https://arxiv.org/pdf/2502.14864)]
> **Authors**: Yuming Yang,Jiang Zhong,Li Jin,Jingwang Huang,Jingpeng Gao,Qing Liu,Yang Bai,Jingyuan Zhang,Rui Jiang,Kaiwen Wei
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 通过基于图表的文档提问生成框架对多模式抹布进行基准测试
- **领域**: 人工智能,计算机视觉和模式识别
- **摘要**: 多模式检索效果生成（MRAG）通过整合外部知识来增强推理能力。但是，现有的基准主要集中于简单的图像文本交互，俯瞰复杂的视觉格式，例如在现实世界应用中普遍存在的图表。在这项工作中，我们介绍了一个基于图表的MRAG的新任务，以解决此限制。为了半自动地生成高质量的评估样本，我们提出了基于图表的文档问题缠绕的生成（电荷），该框架是通过结构化关键点提取，跨模式验证和基于密钥的生成来产生评估数据的框架。通过将电荷与专家验证相结合，我们构建了Chart-MRAG基准，这是用于基于图表的MRAG评估的全面基准，其中包含来自现实世界文档的8个域中的4,738对撤退对。 Our evaluation reveals three critical limitations in current approaches: (1) unified multimodal embedding retrieval methods struggles in chart-based scenarios, (2) even with ground-truth retrieval, state-of-the-art MLLMs achieve only 58.19% Correctness and 73.87% Coverage scores, and (3) MLLMs demonstrate consistent text-over-visual modality bias during Chart-based MRAG reasoning.费用和图表-MRAG台在https://github.com/nomothings/carge.git上发布。

### Chitrarth: Bridging Vision and Language for a Billion People 
[[arxiv](https://arxiv.org/abs/2502.15392)] [[cool](https://papers.cool/arxiv/2502.15392)] [[pdf](https://arxiv.org/pdf/2502.15392)]
> **Authors**: Shaharukh Khan,Ayush Tarun,Abhinav Ravi,Ali Faraz,Akshat Patidar,Praveen Kumar Pokala,Anagha Bhangare,Raja Kolla,Chandra Khatri,Shubham Agarwal
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: Chitrarth：十亿人的桥接视觉和语言
- **领域**: 人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 最近的多模式基础模型主要是对英语或高资源欧洲语言数据进行培训的，这阻碍了其适用于其他中型和低资源语言。为了解决这一限制，我们介绍了包容性视觉语言模型（VLM）的Chitrarth（Chitra：Image; Artha：含义），专门针对10种突出的印度语言的丰富语言多样性和视觉推理。我们的模型有效地集成了一个最先进的（SOTA）多语言大语模型（LLM），并主要在多语言图像文本数据上训练。此外，我们还推出了Bharatbench，这是一个综合框架，用于评估各种印度语言的VLM，最终为更多样化和有效的AI系统做出了贡献。我们的模型可在低资源语言的基准中获得SOTA的结果，同时保留其英语效率。通过我们的研究，我们的目标是在多语言媒体能力方面设定新的基准，从而对现有模型提供了实质性的改进，并为促进该领域的未来进步建立了基础。

### How Do Large Language Monkeys Get Their Power (Laws)? 
[[arxiv](https://arxiv.org/abs/2502.17578)] [[cool](https://papers.cool/arxiv/2502.17578)] [[pdf](https://arxiv.org/pdf/2502.17578)]
> **Authors**: Rylan Schaeffer,Joshua Kazdan,John Hughes,Jordan Juravsky,Sara Price,Aengus Lynch,Erik Jones,Robert Kirk,Azalia Mirhoseini,Sanmi Koyejo
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 大型语言猴子如何获得力量（法律）？
- **领域**: 人工智能,机器学习
- **摘要**: 跨数学问题解决，证明助手编程和多模式越狱的最新研究记录了一个惊人的发现：当（多模式）语言模型应对每项任务进行多次尝试的一组任务 - 如果任何尝试是正确的，那么成功率的成功率是平均成功率的负数。在这项工作中，我们确定了一个明显的难题：一个简单的数学计算预测，在每个问题上，故障率应随着尝试的数量而成倍地下降。我们从经验上证实了这一预测，提出了一个问题：聚集多项式缩放从何处出现？然后，我们通过证明单个指数缩放的每个问题的缩放来回答这个问题，如果单击成功概率的分布分布是沉重的尾巴，以使总体成功概率的一小部分任务集体地扭曲了总体成功趋势，即使每个问题范围均为权力范围 - 即使每个问题自身都按指数为指数级别。我们进一步证明，这种分布的观点解释了先前观察到的与幂律缩放的偏差，并提供了一种简单的方法，可以预测功率定律指数的相对误差较低，或等效地，$ {\ sim} 2-4 $降低了推理计算的范围。总体而言，我们的工作有助于更好地理解神经语言模型的表现如何通过缩放推理计算以及对（多模式）语言模型的可预测评估的开发发展。

### Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts 
[[arxiv](https://arxiv.org/abs/2502.17297)] [[cool](https://papers.cool/arxiv/2502.17297)] [[pdf](https://arxiv.org/pdf/2502.17297)]
> **Authors**: Zhenghao Liu,Xingsheng Zhu,Tianshuo Zhou,Xinyi Zhang,Xiaoyuan Yi,Yukun Yan,Yu Gu,Ge Yu,Maosong Sun
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 在多模式上下文中基准测试检索生成
- **领域**: 人工智能
- **摘要**: 本文介绍了多模式检索仪（M^2RAG），这是一种基准测试，旨在评估多模式大语言模型（MLLMS）在利用多模式检索文档的知识中的有效性。基准标准包括四个任务：图像字幕，多模式问答，多模式的事实验证和图像重新管理。所有任务均在开放域设置中设置，需要从多模式文档收集中检索与查询相关的信息，并将其用作抹布建模的输入上下文。为了增强MLLM的上下文利用功能，我们还引入了多模式检索指令调整（MM-rait），这是一种指令调整方法，可在多模式上下文中优化MLLM。我们的实验表明，MM-rait通过使它们能够有效地从多模式上下文中学习来提高抹布系统的性能。所有数据和代码均可在https://github.com/neuir/m2rag上找到。

### Applications of Large Models in Medicine 
[[arxiv](https://arxiv.org/abs/2502.17132)] [[cool](https://papers.cool/arxiv/2502.17132)] [[pdf](https://arxiv.org/pdf/2502.17132)]
> **Authors**: YunHe Su,Zhengyang Lu,Junhui Liu,Ke Pang,Haoran Dai,Sa Liu Yuxin Jia,Lujia Ge,Jing-min Yang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 大型模型在医学中的应用
- **领域**: 人工智能
- **摘要**: 本文探讨了大型模型在医学领域的进步和应用，特别关注医学大型模型（MEDLMS）。这些模型包括大型语言模型（LLM），视觉模型，3D大型模型和多模型模型，正在通过增强疾病预测，诊断援助，个性化治疗计划和药物发现来彻底改变医疗保健。医学知识图和药物发现中图神经网络的整合突出了大图模型（LGM）在理解复杂生物医学关系中的潜力。该研究还强调了视觉模型（VLM）和3D大型模型在医学图像分析，解剖学建模和假体设计中的变革作用。尽管面临挑战，但这些技术仍在医疗创新中树立新的基准，提高了诊断准确性，并为个性化的医疗保健解决方案铺平了道路。本文旨在全面概述大型医学中大型模型的当前状态和未来方向，强调了它们在促进全球健康方面的重要性。

### TabulaTime: A Novel Multimodal Deep Learning Framework for Advancing Acute Coronary Syndrome Prediction through Environmental and Clinical Data Integration 
[[arxiv](https://arxiv.org/abs/2502.17049)] [[cool](https://papers.cool/arxiv/2502.17049)] [[pdf](https://arxiv.org/pdf/2502.17049)]
> **Authors**: Xin Zhang,Liangxiu Han,Stephen White,Saad Hassan,Philip A Kalra,James Ritchie,Carl Diver,Jennie Shorley
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: Tabulatime：通过环境和临床数据整合进行急性冠状动脉综合征预测的新型多模式深度学习框架
- **领域**: 人工智能,机器学习
- **摘要**: 急性冠状动脉综合征（AC），包括ST段升高心肌梗死（STEMI）和非ST段升高升高的心肌梗死（NSTEMI），仍然是全球死亡率的主要原因。传统的心血管风险评分主要依赖于临床数据，通常会忽视环境影响，例如空气污染，从而显着影响心脏健康。此外，将复杂的时间序列环境数据与临床记录相结合是具有挑战性的。我们介绍了Tabulatime，这是一个多模式深度学习框架，通过将临床风险因素与空气污染数据相结合来增强ACS风险预测。 Tabulatime具有三个关键创新：首先，它将时间序列的空气污染数据与临床表格数据集成在一起，以提高预测准确性。其次，其PatchRWKV模块会自动提取复杂的时间模式，克服传统功能工程的局限性，同时保持线性计算复杂性。第三，注意机制通过揭示临床和环境因素之间的相互作用来增强可解释性。实验结果表明，与常规模型（例如Catboost，Random Forest和LightGBM）相比，Tabulatime提高了预测准确性超过20％，仅空气污染数据就会提高10％。特征重要性分析确定了关键预测因子，包括先前的心绞痛，收缩压，PM10和NO2。总体而言，Tabulatime Bridges临床和环境见解，支持个性化的预防策略，并告知公共卫生政策以减轻ACS风险。

### Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation 
[[arxiv](https://arxiv.org/abs/2502.18725)] [[cool](https://papers.cool/arxiv/2502.18725)] [[pdf](https://arxiv.org/pdf/2502.18725)]
> **Authors**: Xin Liu,Ziyue Zhang,Jingxin Nie
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 20 pages, 6 figures
- **标题**: 与大脑交谈：使用大型语言模型作为代理来建模大脑语义表示
- **领域**: 人工智能,计算语言学,神经元和认知
- **摘要**: 利用自然主义刺激的传统心理实验在手动注释和生态有效性中面临挑战。为了解决这个问题，我们引入了一种新颖的范式，利用多模式大语模型（LLMS）作为代理，通过视觉问题答案（VQA）策略从自然主义图像中提取丰富的语义信息，以分析人类的视觉语义表示。 LLM衍生的表示成功地预测了通过fMRI（例如面部，建筑物）衡量的确定的神经活动模式，从而验证了其可行性并揭示了整个皮质区域的分层语义组织。由LLM衍生表示构建的大脑语义网络标识了有意义的群集，反映了功能和上下文关联。这种创新的方法论为通过自然主义刺激，克服传统注释方法的局限性调查大脑语义组织提供了一种强大的解决方案，并为对人类认知的更生态有效探索铺平了道路。

### MindMem: Multimodal for Predicting Advertisement Memorability Using LLMs and Deep Learning 
[[arxiv](https://arxiv.org/abs/2502.18371)] [[cool](https://papers.cool/arxiv/2502.18371)] [[pdf](https://arxiv.org/pdf/2502.18371)]
> **Authors**: Sepehr Asgarian,Qayam Jetha,Jouhyun Jeon
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 7 pages, 5 figures, 4 Tables, AAAI 2025 Economics of Modern ML: Markets, Incentives, and Generative AI Workshop
- **标题**: MindMem：使用LLM和深度学习预测广告记忆的多模式
- **领域**: 人工智能
- **摘要**: 在广告的竞争环境中，成功取决于有效地导航和利用消费者，广告客户和广告平台之间的复杂互动。这些多方面的互动迫使广告客户优化为消费者行为建模，增强品牌召回和调整广告内容的策略。为了应对这些挑战，我们介绍了MindMem，这是广告记忆性的多模式预测模型。通过整合文本，视觉和听觉数据，MindMem可以实现最新的性能，而Spearman的相关系数在Lambda上为0.631，Memento10K数据集上的相关性系数为0.631，始终超过现有方法。此外，我们的分析确定了影响广告记忆力的关键因素，例如视频起搏，场景复杂性和情感共鸣。为此，我们介绍了MindMem-Read（MindMem驱动的重新生成的广告），该广告采用了大型基于语言模型的仿真来优化广告内容和展示位置，从而提高了广告记忆性的74.12％。我们的结果突出了人工智能在广告中的变革潜力，为广告商提供了一种强大的工具，可以提高参与度，增强竞争力并最大程度地在迅速发展的市场中产生影响。

### ChatMotion: A Multimodal Multi-Agent for Human Motion Analysis 
[[arxiv](https://arxiv.org/abs/2502.18180)] [[cool](https://papers.cool/arxiv/2502.18180)] [[pdf](https://arxiv.org/pdf/2502.18180)]
> **Authors**: Lei Li,Sen Jia,Jianhao Wang,Zhaochong An,Jiaang Li,Jenq-Neng Hwang,Serge Belongie
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 小屋：人类运动分析的多模式多模式
- **领域**: 人工智能,多代理系统
- **摘要**: 多模式大语言模型（MLLM）的进步已经提高了人类运动的理解。但是，这些模型仍然受其“仅教学”性质的限制，缺乏互动性和适应性的分析观点。为了应对这些挑战，我们引入了ChatMotion，这是一个多式模式的人类运动分析框架。 ChatMotion动态解释用户意图，将复杂的任务分解为元任务，并激活专门的功能模块以进行运动理解。它整合了多个专业模块，例如运动核，以从各个角度分析人类运动。广泛的实验证明了Chatmotion的精度，适应性和用户参与人的运动理解。

### Agentic Mixture-of-Workflows for Multi-Modal Chemical Search 
[[arxiv](https://arxiv.org/abs/2502.19629)] [[cool](https://papers.cool/arxiv/2502.19629)] [[pdf](https://arxiv.org/pdf/2502.19629)]
> **Authors**: Tiffany J. Callahan,Nathaniel H. Park,Sara Capponi
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: PDF includes supplemental material
- **标题**: 用于多模式化学搜索的工程流量的代理混合物
- **领域**: 人工智能
- **摘要**: 庞大而复杂的材料设计空间需要创新的策略，以整合多学科科学知识并优化材料发现。尽管大型语言模型（LLM）表现出了各个领域的有希望的推理和自动化功能，但由于缺乏基准标准和实践实施框架，它们在材料科学中的应用仍然有限。为了应对这些挑战，我们介绍了工程流的混合物，以进行自校正检索型生成（crag-mow） - 一种新颖的范式，该范式使用开源llms使用不同的crag策略来精心策划多个代理工作流程。与先前的方法不同，Crag-Mow通过编排代理合成了不同的输出，从而可以在同一问题域中直接评估多个LLM。我们在小分子，聚合物和化学反应以及多模式的核磁共振（NMR）光谱检索中基准岩石测试。我们的结果表明，在比较评估中更频繁地首选的岩壁成绩可以实现与GPT-4O相当的性能，强调了结构化检索和多代理合成的优势。通过揭示跨数据类型的性能变化，Crag-Mow提供了一种可扩展，可解释和基准驱动的方法来优化材料发现的AI架构。这些见解在解决基准LLM和自治AI代理的基本差距方面至关重要。

### Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems 
[[arxiv](https://arxiv.org/abs/2502.19596)] [[cool](https://papers.cool/arxiv/2502.19596)] [[pdf](https://arxiv.org/pdf/2502.19596)]
> **Authors**: Nayoung Choi,Grace Byun,Andrew Chung,Ellie S. Paek,Shinsun Lee,Jinho D. Choi
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 值得信赖的答案，Messier数据：在域专家系统中弥合低资源检索生成的差距
- **领域**: 人工智能,信息检索
- **摘要**: RAG已通过减少幻觉来增强LLM的关键技术，尤其是在LLM可能缺乏足够固有知识的领域专家系统中。但是，在低资源设置中开发这些系统引入了几个挑战：（1）处理异质数据源，（2）优化可信赖答案的检索阶段，以及（3）评估各个方面跨不同方面生成的答案。为了解决这些问题，我们介绍了一个数据生成管道，该管道将原始多模式数据转换为结构化语料库和问答对，一个高级重新级别的阶段提高了检索精度，以及一种与参考匹配算法增强答案的可追溯性。应用于汽车工程领域，我们的系统改善了事实正确性（+1.94），信息性（+1.16）和有用性（+1.67）（+1.67）（+1.67），基于LLM法官的1-5比例。这些结果突出了我们在不同方面的有效性，具有强大的答案接地和透明度。

### Repurposing the scientific literature with vision-language models 
[[arxiv](https://arxiv.org/abs/2502.19546)] [[cool](https://papers.cool/arxiv/2502.19546)] [[pdf](https://arxiv.org/pdf/2502.19546)]
> **Authors**: Anton Alyakin,Jaden Stryker,Daniel Alexander Alber,Karl L. Sangwon,Brandon Duderstadt,Akshay Save,David Kurland,Spencer Frome,Shrutika Singh,Jeff Zhang,Eunice Yang,Ki Yun Park,Cordelia Orillac,Aly A. Valliani,Sean Neifert,Albert Liu,Aneek Patel,Christopher Livia,Darryl Lau,Ilya Laufer,Peter A. Rozman,Eveline Teresa Hidalgo,Howard Riina,Rui Feng,Todd Hollon, et al. (6 additional authors not shown)
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 通过视觉语言模型重新利用科学文献
- **领域**: 人工智能,计算语言学,人机交互
- **摘要**: AI在科学方面的研究通常着重于使用AI技术来增强科学过程的组成部分，或者在某些情况下是整个科学方法。科学出版物的AI怎么样？经过同行评审的期刊是专业知识的基本存储库，用特定于学科的语言编写，与用于培训大多数大型语言模型（LLM）和视觉语言模型（VLMS）的一般互联网内容不同。我们假设，通过将科学期刊家族与生成的AI模型相结合，我们可以为科学沟通，教育和临床护理发明新颖的工具。我们将23,000篇文章从神经外科出版物转换为1.34亿个单词和78,000个图像捕获对的多模式数据库-Neuropubs，以开发六个用于构建AI模型的数据集。我们表明，与更广泛的数据集和PubMed相比，神经植物的含量独特地代表了神经外科特异性的临床环境。为了出版，我们采用了通才VLM来自动从文章中生成图形摘要。编辑委员会成员将其中的70％评为准备出版，而无需进一步的编辑。对于教育，我们以ABN书面考试的风格产生了89,587个测试问题，这是学员和教职员工的神经外科医生，发现与真实例子没有54％的时间。我们将这些问题与课程学习过程一起使用，以跟踪知识获取，同时培训我们的340亿参数VLM（CNS-Obsidian）。在一项盲目的，随机对照试验中，我们证明了CNS-Obsidian至GPT-4O（P = 0.1154）作为神经外科服务的诊断副本。我们的发现为AI提供了科学的新基础，并建立了一个框架，以使用最先进的生成人工智能来提升科学交流，同时保持严格的质量标准。

### Opus: A Workflow Intention Framework for Complex Workflow Generation 
[[arxiv](https://arxiv.org/abs/2502.19532)] [[cool](https://papers.cool/arxiv/2502.19532)] [[pdf](https://arxiv.org/pdf/2502.19532)]
> **Authors**: Phillip Kingston,Théo Fagnoni,Mahsun Altin
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-27
> **comment**: 1 Figure, 27 Pages
- **标题**: 作品：复杂工作流程生成的工作流意向框架
- **领域**: 人工智能
- **摘要**: 本文介绍了工作流意图，这是一个新颖的框架，用于在复杂的业务环境中识别和编码过程目标。工作流意图是定义工作流程信号在业务伪像中解释的工作流程的转换目标的输入，过程和输出元素的对齐。它指定了如何处理输入以实现所需的输出，包括质量标准，业务规则，合规要求和约束。我们采用端到端的业务伪像编码器和工作流信号解释方法，涉及四个步骤：特定于模态的编码，模式内关注，模式间融合关注然后意图解码。我们提供培训程序和关键的损失功能定义。在本文中，我们介绍了工作流信号和工作流意向的概念，其中工作流信号分解为输入，过程和输出元素是从业务人工制品解释的，而工作流意图是这些元素的完整三倍。我们介绍了一个数学框架，用于将工作流信号表示为向量，而工作流意向为张量，并将这些对象的属性形式化。最后，我们提出了一个模块化，可扩展，可训练，基于注意力的多模式生成系统，以解决业务伪像的工作流意向。

### TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding 
[[arxiv](https://arxiv.org/abs/2502.19400)] [[cool](https://papers.cool/arxiv/2502.19400)] [[pdf](https://arxiv.org/pdf/2502.19400)]
> **Authors**: Max Ku,Thomas Chong,Jonathan Leung,Krish Shah,Alvin Yu,Wenhu Chen
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 定理：朝向LLM定理理解的多模式解释
- **领域**: 人工智能,计算语言学,计算机视觉和模式识别,多媒体
- **摘要**: 理解特定于域的定理通常不仅需要基于文本的推理。通过结构化的视觉解释有效沟通对于更深入的理解至关重要。尽管大型语言模型（LLMS）在基于文本的定理推理中表现出很强的性能，但它们产生连贯和教学意义的视觉解释的能力仍然是一个开放的挑战。在这项工作中，我们介绍了Theoremememplainagent，这是一种使用Manim Animations生成长格式定理视频（超过5分钟）的代理方法。为了系统地评估多模式定理的解释，我们提出了定理班基，这是一个基准测试，涵盖了跨多个STEM学科的240个定理，以及5个自动评估指标。我们的结果表明，代理计划对于生成详细的长期视频至关重要，而O3-Mini代理的成功率为93.8％，总分为0.77。但是，我们的定量和定性研究表明，大多数视频都会出现视觉元素布局的小问题。此外，多模式的解释暴露了更深的推理缺陷，基于文本的解释无法揭示，突出了多模式解释的重要性。

### Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy 
[[arxiv](https://arxiv.org/abs/2502.19902)] [[cool](https://papers.cool/arxiv/2502.19902)] [[pdf](https://arxiv.org/pdf/2502.19902)]
> **Authors**: Zaijing Li,Yuquan Xie,Rui Shao,Gongwei Chen,Dongmei Jiang,Liqiang Nie
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Accept to CVPR 2025, Project page: https://cybertronagent.github.io/Optimus-2.github.io/
- **标题**: Optimus-2：具有目标攻击条件策略的多模式Minecraft代理
- **领域**: 人工智能
- **摘要**: 建立一个可以模仿人类行为模式以完成各种开放世界任务的代理是一个长期目标。为了使代理商能够有效地学习各种任务的行为模式，关键的挑战在于建模观察，行动和语言之间的复杂关系。为此，我们提出了一种新型的Minecraft代理Optimus-2，该代理将用于高级计划的多模式大语模型（MLLM）以及一个用于低水平控制的目标观察行动条件策略（GOAP）。 GOAP包含（1）一个动作引导的行为编码器，该行为编码器在每个时间步段的观察值和动作之间建模因果关系，然后与历史观察序列进行动态交互，将其整合到固定长度行为代币中，以及（2）MLLM与开放式语言指令相一致的MLLM，以预测开放式语言指令自动进行操作。此外，我们引入了一个高质量的Minecraft目标观察性（MGOA）}数据集，该数据集包含8个原子任务中的25,000个视频，提供了约3000万个目标 - 目标效果对。自动化施工方法以及MGOA数据集，可以为社区训练Minecraft代理的努力做出贡献。广泛的实验结果表明，Optimus-2在Minecraft中跨越原子任务，长途任务和开放式指示任务表现出卓越的性能。请参阅https://cybertronagent.github.io/optimus-2.github.io/的项目页面。

## 计算工程、金融和科学(cs.CE:Computational Engineering, Finance, and Science)

该领域共有 1 篇论文

### Data-Efficient Model for Psychological Resilience Prediction based on Neurological Data 
[[arxiv](https://arxiv.org/abs/2502.01377)] [[cool](https://papers.cool/arxiv/2502.01377)] [[pdf](https://arxiv.org/pdf/2502.01377)]
> **Authors**: Zhi Zhang,Yan Liu,Mengxia Gao,Yu Yang,Jiannong Cao,Wai Kai Hou,Shirley Li,Sonata Yau,Yun Kwok Wing,Tatia M. C. Lee
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 基于神经数据的心理弹性预测的数据有效模型
- **领域**: 计算工程、金融和科学,人工智能
- **摘要**: 心理韧性被定义为从逆境中反弹的能力，对心理健康至关重要。与通过自我报告的问卷进行的传统弹性评估相比，基于神经数据的复原力评估可通过生物学标记提供更多客观的结果，因此可以显着提高信誉。本文提出了一个新的数据效率模型，以解决神经数据的稀缺性。我们采用Neuro Kolmogorov-Arnold网络作为预测模型的结构。在训练阶段，提出了一种具有智能块技术的新特征的多模式表示算法，以学习具有有限数据的共享潜在空间。在测试阶段，提出了一种新的噪声推理算法来解决神经系统数据的信噪比低。提出的模型不仅在公共数据集和自我建构的数据集上显示出令人印象深刻的性能，而且还为未来的研究提供了一些有价值的心理假设。

## 计算语言学(cs.CL:Computation and Language)

该领域共有 117 篇论文

### Do Large Multimodal Models Solve Caption Generation for Scientific Figures? Lessons Learned from SciCap Challenge 2023 
[[arxiv](https://arxiv.org/abs/2501.19353)] [[cool](https://papers.cool/arxiv/2501.19353)] [[pdf](https://arxiv.org/pdf/2501.19353)]
> **Authors**: Ting-Yao E. Hsu,Yi-Li Hsu,Shaurya Rohatgi,Chieh-Yang Huang,Ho Yin Sam Ng,Ryan Rossi,Sungchul Kim,Tong Yu,Lun-Wei Ku,C. Lee Giles,Ting-Hao K. Huang
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: Accepted to TACL 2025
- **标题**: 大型多模型模型是否为科学人物求解了字幕的产生？从SCICAP Challenge 2023中学到的教训
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 自2021年SCICAP数据集启动以来，研究界在为学术文章中的科学数字生成标题方面取得了重大进展。在2023年，第一个SCICAP挑战赛发生了，邀请全球团队使用扩展的SCICAP数据集开发模型，用于在各个学术领域为各种数字类型字幕字幕。同时，文本生成模型迅速发展，许多强大的预训练的大型多模型（LMM）出现了，在各种视觉和语言任务中都表现出令人印象深刻的功能。本文概述了第一个SCICAP挑战的概述，并详细介绍了各种模型在其数据上的性能，从而捕获了字段状态的快照。我们发现，GPT-4V生成的专业编辑绝大多数优选的数字字幕，而不是其他所有模型，甚至是作者编写的原始字幕。在此关键发现之后，我们进行了详细的分析以回答这个问题：高级LMM已解决了为科学人物生成字幕的任务吗？

### Efficient Reasoning with Hidden Thinking 
[[arxiv](https://arxiv.org/abs/2501.19201)] [[cool](https://papers.cool/arxiv/2501.19201)] [[pdf](https://arxiv.org/pdf/2501.19201)]
> **Authors**: Xuan Shen,Yizhou Wang,Xiangxi Shi,Yanzhi Wang,Pu Zhao,Jiuxiang Gu
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: Preprint version
- **标题**: 用隐藏思维有效推理
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 经过思考链（COT）推理已成为改善多模式大语言模型（MLLMS）中复杂解决问题能力的有力框架。但是，文本推理的详细性质引入了重要的低效率。在这项工作中，我们提出了$ \ textbf {heima} $（作为隐藏的骆驼），这是一个有效的推理框架，它利用隐藏的潜在空间来利用cots的推理。我们设计了Heima编码器，将每个中间的婴儿床凝结成一个紧凑，更高级别的隐藏表示形式，使用单个思维令牌，有效地最大程度地减少了冗长的速度，并减少了推理过程中所需的代币数量。同时，我们设计了使用传统大型语言模型（LLM）的对应的Heima解码器，以将隐藏表示形式自适应地解释为可变的长度文本序列，重建与原始COTS非常相似的推理过程。各种推理的实验结果MLLM基准测试表明，Heima模型在维持甚至更好的零击任务准确性的同时达到了更高的产生效率。此外，使用Heima解码器对多模式推理过程的有效重建验证了我们方法的鲁棒性和解释性。

### Calling a Spade a Heart: Gaslighting Multimodal Large Language Models via Negation 
[[arxiv](https://arxiv.org/abs/2501.19017)] [[cool](https://papers.cool/arxiv/2501.19017)] [[pdf](https://arxiv.org/pdf/2501.19017)]
> **Authors**: Bin Zhu,Huiyan Qi,Yinxuan Gui,Jingjing Chen,Chong-Wah Ngo,Ee-Peng Lim
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: None
- **领域**: 计算语言学
- **摘要**: 多模式大语言模型（MLLM）在整合不同的方式，在复杂的理解和发电任务方面表现出色。尽管他们成功了，但MLLM仍然容易受到对话对抗的投入，尤其是否定论点。本文系统地评估了各种基准的最先进的MLLM，当引入否定论点以最初正确的响应时，揭示了绩效下降。值得注意的是，我们介绍了第一个基准GaslightingBench，该基准是专门设计的，以评估MLLMS对否定论点的脆弱性。 GaslightingBench由现有数据集策划的多项选择问题以及20个不同类别的否定提示。在整个广泛的评估过程中，我们发现与Qwen2-VL和Llava（例如Qwen2-VL和Llava）相比，与Gemini-1.5-Flash，GPT-4O和Claude-3.5-Sonnet等专有模型相比，具有更好的弹性。但是，所有评估的MLLM都难以在对话期间在否定论点下保持逻辑一致性。我们的发现提供了关键的见解，以提高MLLM对否定输入的鲁棒性，从而有助于开发更可靠和值得信赖的多模式AI系统。

### FutureVision: A methodology for the investigation of future cognition 
[[arxiv](https://arxiv.org/abs/2502.01597)] [[cool](https://papers.cool/arxiv/2502.01597)] [[pdf](https://arxiv.org/pdf/2502.01597)]
> **Authors**: Tiago Timponi Torrent,Mark Turner,Nicolás Hinrichs,Frederico Belcavello,Igor Lourenço,Arthur Lorenzi Almeida,Marcelo Viridiano,Ely Edison Matos
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: FutureVision：一种调查未来认知的方法论
- **领域**: 计算语言学
- **摘要**: 本文介绍了一种方法，将多式联运语义分析与令人眼花track乱的实验方案相结合，以研究了解未来情景的交流所涉及的认知工作。为了证明该方法，我们进行了一项试点研究，研究了使用便携式眼轨道仪在虚构的广告片中评估价值和反事实时的视觉固定模式如何变化。参与者的眼动在评估刺激并将其描述给对话伙伴的同时记录了眼动。凝视模式与刺激的语义表示和参与者描述一起分析，该描述是根据语言和视觉方式的框架语义注释构建的。初步结果表明，远程和悲观的场景与更长的固定和更不稳定的扫视有关，这支持了以下假设：对未来情景解释的基础空间中的断裂增加了对理解者的认知负担。

### AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding 
[[arxiv](https://arxiv.org/abs/2502.01341)] [[cool](https://papers.cool/arxiv/2502.01341)] [[pdf](https://arxiv.org/pdf/2502.01341)]
> **Authors**: Ahmed Masry,Juan A. Rodriguez,Tianyu Zhang,Suyuchen Wang,Chao Wang,Aarash Feizi,Akshay Kalkunte Suresh,Abhay Puri,Xiangru Jian,Pierre-André Noël,Sathwik Tejaswi Madhusudhan,Marco Pedersoli,Bang Liu,Nicolas Chapados,Yoshua Bengio,Enamul Hoque,Christopher Pal,Issam H. Laradji,David Vazquez,Perouz Taslakian,Spandana Gella,Sai Rajeswar
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: AlignVLM：桥接视觉和语言潜在空间，用于多模式理解
- **领域**: 计算语言学
- **摘要**: 将视觉特征与语言嵌入对齐是视觉模型（VLMS）的关键挑战。此类模型的性能取决于拥有一个良好的连接器，该连接器将视觉编码器生成的视觉特征映射到使用LLM的共享嵌入空间，同时保持语义相似性。现有的连接器（例如多层感知器（MLP））通常会产生分布式或嘈杂的输入，从而导致模态之间的错位。在这项工作中，我们提出了一种新颖的视觉文本对准方法AlignVLM，该方法将视觉特征映射到LLM文本嵌入的加权平均值。我们的方法利用了LLM编码的语言先验，以确保将视觉特征映射到LLM可以有效解释的空间区域。 AlignVLM对于文档理解任务特别有效，必须将扫描的文档图像准确地映射到其文本内容。我们的广泛实验表明，与先前的对齐方法相比，AlignVLM实现了最先进的性能。我们提供了进一步的分析，证明了视力文本的提高特征特征对齐和与噪声的鲁棒性。

### COVE: COntext and VEracity prediction for out-of-context images 
[[arxiv](https://arxiv.org/abs/2502.01194)] [[cool](https://papers.cool/arxiv/2502.01194)] [[pdf](https://arxiv.org/pdf/2502.01194)]
> **Authors**: Jonathan Tonglet,Gabriel Thiem,Iryna Gurevych
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: Camera-ready version accepted to NAACL 2025 Main Conference
- **标题**: Cove：上下文图像的上下文和真实性预测
- **领域**: 计算语言学
- **摘要**: 从上下文中拍摄的图像是多模式错误信息的最普遍形式。揭穿它们需要（1）提供图像的真实上下文以及（2）检查图像标题的真实性。但是，现有的自动化事实检查方法无法明确解决这两个目标。在这项工作中，我们介绍了Cove，这是一种新方法，该方法首先预测图像的真实上下文，然后使用它来预测标题的真实性。 Cove在所有上下文项目上都击败了SOTA上下文预测模型，通常超过5个百分点。它与合成数据的最佳准确性预测模型具有竞争力，并在现实世界数据上胜过它们，这表明依次将这两个任务结合在一起是有益的。最后，我们进行了一项人类研究，揭示了预测的上下文是可重复使用的，可解释的伪像，可以验证同一图像的新的外观字幕。我们的代码和数据可用。

### PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback 
[[arxiv](https://arxiv.org/abs/2502.00988)] [[cool](https://papers.cool/arxiv/2502.00988)] [[pdf](https://arxiv.org/pdf/2502.00988)]
> **Authors**: Kanika Goswami,Puneet Mathur,Ryan Rossi,Franck Dernoncourt
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: Plotgen：通过多模式反馈的多代理LLM的科学数据可视化
- **领域**: 计算语言学,人工智能
- **摘要**: 科学数据可视化是将原始数据转换为可理解的视觉表示的关键，从而实现模式识别，预测和数据驱动的见解的呈现。但是，由于选择适当的工具和掌握可视化技术的复杂性，新手用户通常会遇到困难。大型语言模型（LLMS）最近表现出了协助代码生成的潜力，尽管它们的准确性很难，并且需要迭代调试。在本文中，我们提出了Plotgen，这是一种新型的多代理框架，旨在自动创建精确的科学可视化。 PlotGen orchestrates multiple LLM-based agents, including a Query Planning Agent that breaks down complex user requests into executable steps, a Code Generation Agent that converts pseudocode into executable Python code, and three retrieval feedback agents - a Numeric Feedback Agent, a Lexical Feedback Agent, and a Visual Feedback Agent - that leverage multimodal LLMs to iteratively refine the data accuracy, textual labels, and visual correctness of通过自我反射产生的地块。广泛的实验表明，Plotgen优于强大的基准，在Matplotbench数据集上取得了4-6％的改善，从而增强了对LLM生成的可视化的信任，并提高了新手生产率，因为绘制错误所需的调试时间减少了。

### Weak Supervision Dynamic KL-Weighted Diffusion Models Guided by Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.00826)] [[cool](https://papers.cool/arxiv/2502.00826)] [[pdf](https://arxiv.org/pdf/2502.00826)]
> **Authors**: Julian Perry,Frank Sanders,Carter Scott
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 弱监督动态KL加权扩散模型以大语言模型为指导
- **领域**: 计算语言学
- **摘要**: 在本文中，我们提出了一种新的方法，可以通过将大语言模型（LLM）与扩散模型相结合，这是一种旨在从文本描述中实现图像综合质量和效率的混合方法。我们的方法引入了一种新的动态KL加权策略，以优化扩散过程，并结合了预先训练的LLMS的语义理解以指导生成过程。提出的方法可以显着提高具有文本描述的生成图像的视觉质量和对齐方式，从而解决了诸如计算效率低下，训练中的不稳定性以及对文本变异性的鲁棒性等挑战。我们在可可数据集上评估了我们的方法，并在定量和定性上证明了其优于传统基于GAN的模型。包括消融研究和人类评估在内的广泛实验证实，我们的方法在图像现实主义，与输入文本的相关性以及整体美学质量方面优于现有方法。我们的方法还显示了对其他多模式任务的可伸缩性的希望，这使其成为广泛生成应用的多功能解决方案。

### Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities 
[[arxiv](https://arxiv.org/abs/2502.00451)] [[cool](https://papers.cool/arxiv/2502.00451)] [[pdf](https://arxiv.org/pdf/2502.00451)]
> **Authors**: Aishik Mandal,Tanmoy Chakraborty,Iryna Gurevych
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: 18 pages, 2 figures
- **标题**: 迈向隐私感知心理健康AI模型：进步，挑战和机遇
- **领域**: 计算语言学,人工智能
- **摘要**: 精神疾病是一种广泛而令人衰弱的状况，具有巨大的社会和个人成本。传统的诊断和治疗方法，例如自我报告的问卷和心理治疗课程，通常会对患者和临床医生施加巨大的负担，从而限制了可及性和效率。人工智能（AI）的最新进展，特别是在自然语言处理和多模式技术方面，具有识别和解决诸如抑郁症，焦虑，躁郁症，精神分裂症和创伤后应激障碍等疾病的巨大潜力。但是，隐私问题，包括数据集中敏感数据泄漏和训练有素的模型的风险，仍然是将这些AI系统部署在现实世界中的临床环境中的关键障碍。这些挑战在多模式方法中得到了放大，在这些方法中，可以滥用语音和面部数据等个人标识符。本文对与开发和部署心理健康模型相关的隐私挑战进行了批判性和全面的研究。我们进一步规定了潜在的解决方案，包括数据匿名，合成数据生成和保护隐私模型培训，以加强实际应用中的隐私保护措施。此外，我们讨论了评估框架，以评估这些方法中的隐私 - 实用性权衡。通过解决这些挑战，我们的工作旨在推动开发可靠的，隐私意识的AI工具，以支持临床决策并改善心理健康成果。

### The Impact of Persona-based Political Perspectives on Hateful Content Detection 
[[arxiv](https://arxiv.org/abs/2502.00385)] [[cool](https://papers.cool/arxiv/2502.00385)] [[pdf](https://arxiv.org/pdf/2502.00385)]
> **Authors**: Stefano Civelli,Pietro Bernardelle,Gianluca Demartini
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: Companion Proceedings of the ACM Web Conference 2025 (WWW Companion'25)
- **标题**: 基于角色的政治观点对仇恨内容检测的影响
- **领域**: 计算语言学,人工智能
- **摘要**: 尽管已经显示出具有政治内容的语言模型可以改善下游任务公平性，但这种方法需要许多研究人员和组织通常无法访问的大量计算资源。最近的工作确定，基于角色的提示可以在模型产出中引入政治多样性，而无需额外的培训。但是，尚不清楚这种促使这种促使策略是否可以取得与下游任务的政治预读的结果。我们使用基于角色的促进策略在多模式仇恨言论检测任务中调查了这个问题，特别关注模因中的仇恨言论。我们的分析表明，当将角色映射到政治指南针并衡量角色同意时，固有的政治定位与分类决策的相关性很小。值得注意的是，即使角色被明确注射更强的意识形态描述符，这种缺乏相关性仍然存在。我们的发现表明，尽管LLM可以在对直接政治问题的回答中表现出政治偏见，但这些偏见可能对实际分类任务的影响比以前所假设的要少。这就提出了关于在下游任务中实现公平绩效的计算昂贵政治预审查的必要性的重要问题。

### Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions 
[[arxiv](https://arxiv.org/abs/2502.00339)] [[cool](https://papers.cool/arxiv/2502.00339)] [[pdf](https://arxiv.org/pdf/2502.00339)]
> **Authors**: Jingyuan Yi,Zeqiu Xu,Tianyi Huang,Peiyang Yu
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: LLM驱动的假新闻检测中的挑战和创新：方法和未来方向的综合
- **领域**: 计算语言学,计算机与社会
- **摘要**: 通过社交媒体平台传播假新闻的普遍性为公众，社会稳定和民主制度的信任带来了关键风险。这项挑战要求在检测中采用新颖的方法学，这可以与错误信息的动态和多模式性质保持同步。最近的工作包括使用多模式框架中的大语言模型进步，使用图形的方法和虚假新闻文献中的对抗性培训为检测提供动力。基于可以带来成功的不同方法，将强调一些关键的亮点：通过更先进的语义和跨模式融合以增强LLM-Improves精度，以实现强大的检测。该综述进一步确定了适应动态媒体趋势，实时和跨平台检测能力的关键差距，以及滥用LLM所带来的道德挑战。未来的方向强调了风格不足的模型，跨语性检测框架和强大的政策的开发，以减轻LLM驱动的错误信息。因此，这种综合为那些致力于加强具有并发症的假新闻检测系统的研究人员和从业人员奠定了混凝土基础，并在数字景观中不断增长。

### Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning 
[[arxiv](https://arxiv.org/abs/2502.02871)] [[cool](https://papers.cool/arxiv/2502.02871)] [[pdf](https://arxiv.org/pdf/2502.02871)]
> **Authors**: Yibo Yan,Shen Wang,Jiahao Huo,Jingheng Ye,Zhendong Chu,Xuming Hu,Philip S. Yu,Carla Gomes,Bart Selman,Qingsong Wen
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 位置：多模式大语模型可以显着提高科学推理
- **领域**: 计算语言学,人工智能
- **摘要**: 科学推理，人类采用逻辑，证据和批判性思维的过程来探索和解释科学现象，对于推进各种领域的知识推理至关重要。然而，尽管取得了重大进展，但当前的科学推理模型仍在跨领域的概括中挣扎，并且通常没有多模式的感知。整合文本，图像和其他方式的多模式大型语言模型（MLLM）为克服这些局限性并增强科学推理提供了令人兴奋的机会。因此，该立场论文认为，MLLM可以在数学，物理，化学和生物学等学科中显着提高科学推理。首先，我们提出了科学推理能力的四阶段研究路线图，并在科学推理中强调了MLLM应用的当前状态，并指出了它们在各种数据类型上整合和推理的能力。其次，我们总结了仍然障碍达到MLLM充分潜力的障碍的主要挑战。为了应对这些挑战，我们提出了对未来的可行见解和建议。总体而言，我们的作品为MLLM与科学推理提供了新的观点，为LLM社区提供了实现人工通用智能（AGI）的宝贵愿景。

### SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency 
[[arxiv](https://arxiv.org/abs/2502.02458)] [[cool](https://papers.cool/arxiv/2502.02458)] [[pdf](https://arxiv.org/pdf/2502.02458)]
> **Authors**: Qianhao Yuan,Yanjiang Liu,Yaojie Lu,Hongyu Lin,Ben He,Xianpei Han,Le Sun
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: SAISA：迈向具有培训和推理效率的多模式大语模型
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 多模式的大语言模型（MLLM）主要属于两个架构，每种构建涉及训练和推理效率之间的权衡：在推理过程中拟合空间对齐（例如Llava-1.5）效率低下，而跨注意空间一致性（例如，FLAMINGO）在培训中效率低下。在本文中，我们比较了这两种架构，并确定了构建有效MLLM的关键因素。它们之间的主要区别在于如何将注意力应用于视觉令牌，尤其是它们相互作用。要调查是否需要在视觉令牌之间进行注意，我们提出了一种新的自我注意力机制（\ textbf {n} o \ textbf {a} ttention \ textbf {a} mong \ textbf {a} mong \ textbf {vi} sual \ textbf \ textbf {t} t），以消除这种类型的注意。我们在LLAVA-1.5上的试点实验表明，视觉令牌之间的注意力高度多余。基于这些见解，我们介绍了Saisa（\ TextBf {s} Elf- \ textBf {a} ttention \ textbf {i} nput \ textbf {s} pace \ textbf {a} wextbf {a}木板），一种增强培训和培训的新颖体系结构。 Saisa将视觉特征与Naavit自我发场块的输入空间保持一致，从而减少了自我发挥块和前馈网络（FFN）中的计算开销。 Saisa使用与LLAVA-1.5相同的配置，将推理拖鞋降低66 \％，培训预算降低26 \％，同时在准确性方面取得了出色的性能。全面的消融研究进一步验证了Saisa在各种LLM和视觉编码器中的有效性。该代码和模型将在https://github.com/icip-cas/saisa上公开获取。

### Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking 
[[arxiv](https://arxiv.org/abs/2502.02339)] [[cool](https://papers.cool/arxiv/2502.02339)] [[pdf](https://arxiv.org/pdf/2502.02339)]
> **Authors**: Jinyang Wu,Mingkuan Feng,Shuai Zhang,Ruihan Jin,Feihu Che,Zengqi Wen,Jianhua Tao
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 用MCTS Automated结构化思维来增强多模式推理
- **领域**: 计算语言学
- **摘要**: 多模式大语模型（MLLM）具有令人印象深刻的功能，但在复杂的视觉推理中仍然面临挑战。尽管最近的努力试图通过通过明确的搜索结构或教师指导的蒸馏来纳入类似OpenAi O1的结构化思维来增强MLLM的推理，但他们经常努力平衡性能和效率。一个关键的限制是他们对广泛的数据和搜索空间的严重依赖，从而导致低效率的隐性见解提取和数据利用。为了解决这个问题，我们提出了Astar，这是一种通过Monte Carlo Tree搜索（MCT）进行多模式推理的自动结构化思维范式。 ASTAR使用MCTS驱动的分层结构自动从有限的数据中衍生出高级认知推理模式。在这些明确模式的基础上，我们设计了一个统一的推理框架，该框架无缝整合了模型的内部推理功能和外部推理指南，从而可以通过最小的树迭代效率地推断。这种新颖的范式在性能和效率之间取得了令人信服的平衡。广泛的实验证明了Astar的有效性，具有7B骨架的数学基准测试，在数学基准上实现了卓越的精度（54.0 $ \％$），超过了GPT-4O（50.2 $ \％$ $ $），同时保持实质性数据和计算效率。

### On Fairness of Unified Multimodal Large Language Model for Image Generation 
[[arxiv](https://arxiv.org/abs/2502.03429)] [[cool](https://papers.cool/arxiv/2502.03429)] [[pdf](https://arxiv.org/pdf/2502.03429)]
> **Authors**: Ming Liu,Hao Chen,Jindong Wang,Liwen Wang,Bhiksha Raj Ramakrishnan,Wensheng Zhang
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 关于图像生成的统一多模式大语模型的公平性
- **领域**: 计算语言学,人工智能
- **摘要**: 统一的多模式大型语言模型（U-MLLM）在端到端管道中的视觉理解和产生中表现出了令人印象深刻的表现。与仅一代模型（例如，稳定扩散）相比，U-MLLM可能会引发有关其产出中偏见的新问题，该问题可能会受到其统一功能的影响。考虑到传播有害刻板印象的探索风险不足，这一差距尤其令人担忧。在本文中，我们基准了最新的U-MLLM，发现大多数人的人口偏见，例如性别和种族偏见。为了更好地理解和减轻此问题，我们提出了一个定位的策略，我们在其中审核并展示单个模型组件如何受偏差的影响。我们的分析表明，偏见主要源自语言模型。更有趣的是，我们观察到U-Mllms中的“部分一致性”现象，在这种情况下，理解偏见似乎很少，但是产生偏见仍然很大。因此，我们提出了一种新型的平衡偏好模型，以平衡人口统计学分布与综合数据。实验表明，我们的方法可以减少人口偏见，同时保留语义保真度。我们希望我们的发现强调了将来对U-MLLM的更多整体解释和偏见策略的需求。

### iVISPAR -- An Interactive Visual-Spatial Reasoning Benchmark for VLMs 
[[arxiv](https://arxiv.org/abs/2502.03214)] [[cool](https://papers.cool/arxiv/2502.03214)] [[pdf](https://arxiv.org/pdf/2502.03214)]
> **Authors**: Julius Mayer,Mohamad Ballout,Serwan Jassim,Farbod Nosrat Nezami,Elia Bruni
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: ivispar- VLMS的交互式视觉空间推理基准
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 众所周知，视觉语言模型（VLM）在空间推理和视觉对齐中挣扎。为了帮助克服这些局限性，我们引入了Ivispar，这是一种交互式的多模式基准，旨在评估VLM的空间推理功能，以作用为代理。 Ivispar基于滑动瓷砖拼图的一种经典问题，需要逻辑计划，空间意识和多步骤推理。该基准支持Visual 2D，3D和基于文本的输入方式，从而可以全面评估VLMS的计划和推理能力。我们评估了一系列最先进的开源和封闭源VLM的套件，比较了它们的性能，同时还提供了最佳的路径解决方案和人体基线，以评估任务的复杂性和对人类的可行性。结果表明，尽管某些VLM在简单的空间任务上表现良好，但它们遇到了困难，具有更复杂的配置和问题属性。值得注意的是，尽管与3D或基于文本的表示相比，VLM通常在2D视觉中表现更好，但它们始终没有人类的性能，这说明了视觉对齐的持续挑战。这突出了当前VLM功能的关键差距，突出了它们在实现人类水平认知方面的局限性。

### LLaVAC: Fine-tuning LLaVA as a Multimodal Sentiment Classifier 
[[arxiv](https://arxiv.org/abs/2502.02938)] [[cool](https://papers.cool/arxiv/2502.02938)] [[pdf](https://arxiv.org/pdf/2502.02938)]
> **Authors**: T. Chay-intr,Y. Chen,K. Viriyayudhakorn,T. Theeramunkong
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: llavac：微调llava作为多模式分类器
- **领域**: 计算语言学
- **摘要**: 我们提出了LLAVAC，这是一种构建用于多模式分析的分类器的方法。该方法利用大语言和视觉助手（LLAVA）的微调来预测图像和文本方式上的情感标签。我们的方法涉及设计一个结构化的提示，该提示将单峰和多模式标签纳入微调Llava，从而使其能够有效地执行情感分类。 MVSA单格数据集上的实验表明，LLAVAC在三个数据处理过程中的多模式情感分析中的现有方法优于现有方法。 LLAVAC的实现可在https://github.com/tchayintr/llavac上公开获得。

### EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.04424)] [[cool](https://papers.cool/arxiv/2502.04424)] [[pdf](https://arxiv.org/pdf/2502.04424)]
> **Authors**: He Hu,Yucheng Zhou,Lianzhong You,Hongbo Xu,Qianning Wang,Zheng Lian,Fei Richard Yu,Fei Ma,Laizhong Cui
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: EMOBENCH-M：为多模式大语言模型的基准测试情商
- **领域**: 计算语言学,人工智能
- **摘要**: 随着将多模式的大语言模型（MLLM）整合到机器人系统和各种AI应用中，将情绪智力（EI）功能嵌入这些模型对于使机器人能够有效地解决人类的情绪需求并在现实世界中无缝交互至关重要。现有的静态，基于文本或文本图像基准测试忽略了现实世界相互作用的多模式复杂性，并且无法捕获情感表达的动态，多模式的性质，从而使它们不足以评估MLLMS的EI。基于EI的既定心理理论，我们建立了Emobench-M，这是一种新颖的基准，旨在评估来自三个关键维度的13个评估场景中MLLM的EI能力：基本的情感识别，对话情感理解和社会复杂的情绪分析。对EMOBENCE-M上开源和封闭源MLLM的评估揭示了它们与人之间的显着性能差距，这强调了进一步提高其EI能力的必要性。所有基准资源（包括代码和数据集）均在https://emo-gml.github.io/上公开可用。

### Multimodal Medical Code Tokenizer 
[[arxiv](https://arxiv.org/abs/2502.04397)] [[cool](https://papers.cool/arxiv/2502.04397)] [[pdf](https://arxiv.org/pdf/2502.04397)]
> **Authors**: Xiaorui Su,Shvat Messica,Yepeng Huang,Ruth Johnson,Lukas Fesser,Shanghua Gao,Faryad Sahneh,Marinka Zitnik
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: conference
- **标题**: 多模式医学代码令牌
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 对患者电子健康记录（EHR）培训的基金会模型需要将医疗数据化为离散词汇序列的序列。现有的令牌者将来自EHR的医疗法规视为孤立的文本令牌。但是，每个医学法规都由其文本描述，其在本体论等级制度中的地位及其与其他代码的关系（例如疾病共发生和药物治疗协会）所定义。医学词汇包含超过600,000个代码，其中包含用于临床推理的关键信息。我们介绍了Medtok，这是一种使用文本描述和代码的关系上下文的多模式医学代码令牌。 Medtok使用语言模型编码器处理文本，并使用图形编码器编码关系结构。然后，它将两种模态量化为一个统一的令牌空间，从而保留了特定于模态和跨模式信息。我们将MEDTOK集成到五个EHR模型中，并将其评估在患者和门诊数据集的操作和临床任务上，包括结果预测，诊断分类，药物建议和风险分层。将标准的EHR Tokenizers与MEDTOK交换可改善所有EHR模型的AUPRC，MIMIC-III的AUPRC在4.10％中，对MIMIC-IV的4.78％，EHRSHOT的AUPRC提高了4.78％，在药物推荐方面具有最大的收益。除了EHR建模之外，我们还证明了使用Medtok令牌与医疗质量检查系统一起演示。我们的结果表明，Medtok是医疗法规的统一令牌的潜力，从而改善了医疗基础模型的令牌化。

### DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization 
[[arxiv](https://arxiv.org/abs/2502.04370)] [[cool](https://papers.cool/arxiv/2502.04370)] [[pdf](https://arxiv.org/pdf/2502.04370)]
> **Authors**: Zhenglin Zhou,Xiaobo Xia,Fan Ma,Hehe Fan,Yi Yang,Tat-Seng Chua
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-07
> **comment**: 20 pages, 12 figures
- **标题**: Dreamdpo：通过直接偏好优化将文本到3D的一致与人类的偏好结合
- **领域**: 计算语言学,图形,机器学习
- **摘要**: 文本到3D生成从文本描述中自动创建3D内容，该描述在各个领域都具有变革性的潜力。但是，现有的方法通常很难使生成的内容与人类的偏好相结合，从而限制了其适用性和灵活性。为了解决这些局限性，在本文中，我们提出了DreamDPO，这是一个基于优化的框架，该框架通过直接优先优化将人类偏好纳入3D生成过程。实际上，DreamDPO首先构建成对示例，然后使用奖励或大型多模型模型将其对齐与人类的偏好进行比较，最后通过偏好驱动损失函数优化3D表示。通过利用成对比较来反映偏好，DreamDPO降低了对精确的质量评估的依赖，同时通过偏好引导的优化实现了细粒度的可控性。实验表明，DreamDPO取得了竞争成果，并且与现有方法相比提供了更高质量，更可控制的3D内容。代码和型号将是开源的。

### CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements 
[[arxiv](https://arxiv.org/abs/2502.04353)] [[cool](https://papers.cool/arxiv/2502.04353)] [[pdf](https://arxiv.org/pdf/2502.04353)]
> **Authors**: Afshin Khadangi,Amir Sartipi,Igor Tchappi,Gilbert Fridgen
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 认知：自动化艺术分析和解码审美元素的大型语言模型
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 艺术作为一种通用语言，可以通过各种方式来解释，艺术品体现了深刻的含义和细微差别。大型语言模型（LLM）的出现以及多模式大语言模型（MLLM）的可用性提出了一个问题，即如何使用这些变革模型来评估和解释艺术品的艺术要素。据我们所知，尽管在该领域进行了研究，但尚未探讨对使用LLMS的艺术品的技术和表达特征深入详细的理解。在这项研究中，我们研究了正式的艺术分析框架的自动化，以迅速分析大量的艺术品，并检查其模式如何随着时间的流逝而发展。我们探讨了LLM如何解码艺术表达，视觉元素，组成和技术，从而揭示跨时期发展的新兴模式。最后，我们在这种情况下讨论了LLM的优势和局限性，强调了它们处理大量与艺术相关数据并产生有见地的解释的能力。由于结果的详尽和颗粒状性质，我们开发了交互式数据可视化，在线可用https://cognartive.github.io/，以增强理解和可及性。

### MTPChat: A Multimodal Time-Aware Persona Dataset for Conversational Agents 
[[arxiv](https://arxiv.org/abs/2502.05887)] [[cool](https://papers.cool/arxiv/2502.05887)] [[pdf](https://arxiv.org/pdf/2502.05887)]
> **Authors**: Wanqi Yang,Yanda Li,Meng Fang,Ling Chen
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: NAACL 2025 Findings
- **标题**: MTPCHAT：一种多模式的时间感知角色数据集用于对话代理
- **领域**: 计算语言学,人工智能
- **摘要**: 了解时间动态对于对话代理，实现有效的内容分析和明智的决策至关重要。但是，时间感知的数据集，尤其是对于角色界面的对话，仍然有限，这会缩小其范围并减少其复杂性。为了解决这一差距，我们介绍了MTPCHAT，这是一种多模式，时间吸引的角色对话数据集，该数据集将语言，视觉和时间元素集成到对话和角色记忆中。利用MTPCHAT，我们提出了两个时间敏感的任务：时间敏感的下一个响应预测（TNRP）和时间接地记忆预测（TGMP），均旨在评估模型了解隐式时间提示和动态相互作用的能力。此外，我们提出了一个创新的框架，该框架具有自适应时间模块，以有效整合多模式流并捕获时间依赖性。实验结果证明了MTPCHAT所带来的挑战，并证明了我们在多模式时间敏感的情况下框架的有效性。

### Large Multimodal Models for Low-Resource Languages: A Survey 
[[arxiv](https://arxiv.org/abs/2502.05568)] [[cool](https://papers.cool/arxiv/2502.05568)] [[pdf](https://arxiv.org/pdf/2502.05568)]
> **Authors**: Marian Lupascu,Ana-Cristina Rogoz,Mihai Sorin Stupariu,Radu Tudor Ionescu
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 低资源语言的大型多式模型：调查
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 在这项调查中，我们系统地分析了用于调整低资源（LR）语言的大型多模型模型（LMM）的技术，研究了从视觉增强和数据创建到跨模式传输和融合策略的方法。通过对跨75 LR语言的106项研究的全面分析，我们确定了研究人员如何应对有限数据和计算资源的挑战的关键模式。我们发现，视觉信息通常是改善LR设置模型性能的关键桥梁，尽管在幻觉缓解和计算效率等领域仍存在重大挑战。我们旨在为研究人员提供清楚的了解，以使LR（研究研究的）语言更容易使LMM更容易获得挑战。我们通过以下网址提供的开源存储库来补充：https：//github.com/marianlupascu/lmm4lrl-survey。

### Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation 
[[arxiv](https://arxiv.org/abs/2502.05151)] [[cool](https://papers.cool/arxiv/2502.05151)] [[pdf](https://arxiv.org/pdf/2502.05151)]
> **Authors**: Steffen Eger,Yong Cao,Jennifer D'Souza,Andreas Geiger,Christian Greisinger,Stephanie Gross,Yufang Hou,Brigitte Krenn,Anne Lauscher,Yizhi Li,Chenghua Lin,Nafise Sadat Moosavi,Wei Zhao,Tristan Miller
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: Work in progress. Will be updated soon
- **标题**: 通过大型语言模型转化科学：一项有关AI辅助科学发现，实验，内容生成和评估的调查
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 随着大型多模式模型的出现，科学现在正处于基于AI的技术转型的一个门槛上。最近，已经提出了众多新的AI模型和工具，有望使全球研究人员和学者更有效，有效地进行研究。这包括研究周期的所有方面，特别是（1）寻找相关文献； （2）产生研究思想和进行实验；生成（3）基于文本的和（4）多模式内容（例如，科学数字和图）； （5）基于AI的自动同行评审。在这项调查中，我们对这些令人兴奋的最新发展提供了深入的概述，这些发展有望从根本上改变科学研究过程。我们的调查涵盖了上面概述的五个方面，表明相关的数据集，方法和结果（包括评估）以及未来研究的限制和范围。关于这些工具缺点和滥用潜力（假科学，窃，对研究完整性的危害）的道德问题在我们的讨论中特别重要。我们希望我们的调查不仅将成为该领域的新移民的参考指南，而且还将成为“ AI4Science”领域的新计划的催化剂。

### CodeSCM: Causal Analysis for Multi-Modal Code Generation 
[[arxiv](https://arxiv.org/abs/2502.05150)] [[cool](https://papers.cool/arxiv/2502.05150)] [[pdf](https://arxiv.org/pdf/2502.05150)]
> **Authors**: Mukur Gupta,Noopur Bhatt,Suman Jana
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: Accepted to NAACL 2025
- **标题**: CODESCM：多模式代码生成的因果分析
- **领域**: 计算语言学
- **摘要**: 在本文中，我们提出了CodeSCM，即一种结构性因果模型（SCM），用于使用大语言模型（LLMS）分析多模式代码生成。通过将干预措施应用于CODESCM，我们衡量了模型上不同及时模态（例如自然语言，代码和输入输出示例）的因果关系。 CODESCM引入了潜在的调解器变量，以将多模式代码生成提示的代码和自然语言语义分开。使用有关这些介体的因果中介分析的原理，我们量化了代表模型虚假倾向的直接效应。我们发现，除了自然语言指示外，投入输出示例还显着影响代码生成。

### Multimodal Cognitive Reframing Therapy via Multi-hop Psychotherapeutic Reasoning 
[[arxiv](https://arxiv.org/abs/2502.06873)] [[cool](https://papers.cool/arxiv/2502.06873)] [[pdf](https://arxiv.org/pdf/2502.06873)]
> **Authors**: Subin Kim,Hoonrae Kim,Heejin Do,Gary Geunbae Lee
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-11
> **comment**: NAACL 2025 Main
- **标题**: 通过多跳心治疗推理的多模式认知重新构架疗法
- **领域**: 计算语言学,人工智能
- **摘要**: 先前的研究揭示了大型语言模型（LLM）的潜力支持认知重塑疗法。但是，他们的重点主要是基于文本的方法，通常忽略非语言证据在现实生活中至关重要的重要性。为了减轻这一差距，我们将文本认知反映扩展到了多模式，并结合了视觉线索。具体来说，我们提出了一个称为多模态认知支持对话（M2COSC）的新数据集，该数据集将每个GPT-4生成的对话与反映虚拟客户的面部表情的图像配对。为了更好地反映真实的心理治疗，面部表情会导致解释隐性情绪证据，我们提出了一种多跳的心理治疗推理方法，该方法明确识别并结合了微妙的证据。我们对LLM和视觉模型（VLM）进行的全面实验表明，M2COSC数据集通过VLMS作为心理治疗师的表现得到了显着提高。此外，多跳的心理治疗推理方法使VLMS能够提供更周到和善解人意的建议，表现优于标准提示方法。

### Target-Augmented Shared Fusion-based Multimodal Sarcasm Explanation Generation 
[[arxiv](https://arxiv.org/abs/2502.07391)] [[cool](https://papers.cool/arxiv/2502.07391)] [[pdf](https://arxiv.org/pdf/2502.07391)]
> **Authors**: Palaash Goel,Dushyant Singh Chauhan,Md Shad Akhtar
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: 目标增强基于融合融合的多模式讽刺产生
- **领域**: 计算语言学
- **摘要**: 讽刺是一种语言现象，旨在以固有的方式嘲笑目标（例如实体，事件或人）。多模式的讽刺解释（MUSE）旨在使用自然语言解释在讽刺帖子中揭示讽刺的讽刺。尽管很重要，但现有系统忽略了讽刺目标在产生解释方面的重要性。在本文中，我们提出了一个基于融合融合的讽刺解释模型，也就是。涡轮。我们设计了一种新颖的共享融合机制，以利用图像及其标题之间的模式间关系。 Turbo假定了讽刺的目标，并指导了学习预期的解释的复杂性时多模式共享的融合机制。我们在More+数据集上评估了我们提出的涡轮模型。与多个基线和最先进的模型的比较表明，涡轮增压的平均保证金为$+3.3 \％$。此外，我们为我们的任务探索了零和一击设置的LLM，并观察到LLM生成的解释虽然出色，但通常无法捕捉讽刺的关键细微差别。此外，我们对涡轮生成的解释进行了广泛的人类评估，并发现它们比其他系统要好得多。

### Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation 
[[arxiv](https://arxiv.org/abs/2502.08826)] [[cool](https://papers.cool/arxiv/2502.08826)] [[pdf](https://arxiv.org/pdf/2502.08826)]
> **Authors**: Mohammad Mahdi Abootorabi,Amirhosein Zobeiri,Mahdi Dehghani,Mohammadali Mohammadkhani,Bardia Mohammadi,Omid Ghahroodi,Mahdieh Soleymani Baghshah,Ehsaneddin Asgari
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: GitHub repository: https://github.com/llm-lab-org/Multimodal-RAG-Survey
- **标题**: 以任何方式询问：一项关于多式联运的全面调查
- **领域**: 计算语言学,人工智能,信息检索
- **摘要**: 大型语言模型（LLM）由于依赖静态培训数据而与幻觉和过时的知识斗争。通过集成外部动态信息来增强事实和更新的基础，检索增强的生成（RAG）通过整合外部动态信息来减轻这些问题。多模式学习的最新进展导致了多模式抹布的发展，并结合了多种模式，例如文本，图像，音频和视频，以增强生成的输出。但是，跨模式的对齐和推理对多模式抹布引入了独特的挑战，将其与传统的单峰抹布区分开。这项调查提供了对多模式抹布系统的结构化和全面分析，涵盖了检索，融合，增强和一代中的数据集，指标，基准，评估，方法和创新。我们精确地审查了培训策略，鲁棒性增强和损失功能，同时还探索了多种多态的破布场景。此外，我们讨论了支持这个不断发展的领域进步的开放挑战和未来的研究方向。这项调查为开发更有效和可靠的AI系统的基础奠定了基础，这些系统有效地利用了多模式动态外部知识库。资源可在https://github.com/llm-lab-org/multimodal-rag-survey上找到。

### Semantic Role Labeling: A Systematical Survey 
[[arxiv](https://arxiv.org/abs/2502.08660)] [[cool](https://papers.cool/arxiv/2502.08660)] [[pdf](https://arxiv.org/pdf/2502.08660)]
> **Authors**: Huiyao Chen,Meishan Zhang,Jing Li,Min Zhang,Lilja Øvrelid,Jan Hajič,Hao Fei
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 语义角色标签：系统调查
- **领域**: 计算语言学
- **摘要**: 语义角色标签（SRL）是一种中央自然语言处理（NLP）任务，旨在了解文本中的语义角色，从而促进了广泛的下游应用程序。尽管SRL获得了广泛而持久的研究，但目前缺乏全面的调查，可以彻底组织和综合该领域。本文旨在回顾过去二十年来SRL社区的整个研究轨迹。我们首先提供SRL的完整定义。为了提供全面的分类法，我们将SRL方法论分为四个关键角度：模型架构，语法功能建模，应用程序方案和多模式扩展。此外，我们讨论了SRL基准，评估指标和范式建模方法，同时还探索了各个领域的实际应用。最后，我们分析了SRL的未来研究方向，以解决SRL在大语言模型（LLMS）（LLMS）及其对更广泛的NLP景观的潜在影响的不断发展的作用。我们维护一个公共存储库，并始终在以下网址更新相关资源

### Salamandra Technical Report 
[[arxiv](https://arxiv.org/abs/2502.08489)] [[cool](https://papers.cool/arxiv/2502.08489)] [[pdf](https://arxiv.org/pdf/2502.08489)]
> **Authors**: Aitor Gonzalez-Agirre,Marc Pàmies,Joan Llop,Irene Baucells,Severino Da Dalt,Daniel Tamayo,José Javier Saiz,Ferran Espuña,Jaume Prats,Javier Aula-Blasco,Mario Mina,Iñigo Pikabea,Adrián Rubio,Alexander Shvets,Anna Sallés,Iñaki Lacunza,Jorge Palomar,Júlia Falcão,Lucía Tormo,Luis Vasquez-Reina,Montserrat Marimon,Oriol Pareras,Valle Ruiz-Fernández,Marta Villegas
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 萨拉曼德拉技术报告
- **领域**: 计算语言学
- **摘要**: 这项工作介绍了Salamandra，这是一套开源解码器，只有三种不同尺寸的大型大型语言模型：2、7和400亿个参数。这些模型是从划痕的高度多语言数据中培训的，该数据包括35种欧洲语言和代码的文本。我们精心策划的语料库专门由从各种来源汇编的开放访问数据制成。除了基本模型外，还为聊天应用程序发布了通过公共域指令数据进行微调的补充检查点。此外，我们还共享有关多模式的初步实验，这些实验是概念验证，以展示萨拉曼德拉家族的潜在应用。我们对多语言基准测试的广泛评估表明，与类似尺寸的开源模型相比，Salamandra具有很强的能力，具有竞争性能。我们为标准下游任务以及与偏见和安全有关的关键方面提供了全面的评估结果。在此技术报告的情况下，我们打算通过共享我们的设计选择，数据策略策略和评估方法背后的所有细节来促进开放科学。除此之外，我们通过公开访问培训和评估脚本来偏离通常的实践。我们在允许的Apache 2.0许可下发布所有模型，以促进未来的研究并促进商业用途，从而为大型语言模型的开源生态系统做出了贡献。

### Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting 
[[arxiv](https://arxiv.org/abs/2502.08317)] [[cool](https://papers.cool/arxiv/2502.08317)] [[pdf](https://arxiv.org/pdf/2502.08317)]
> **Authors**: Jiarui Wu,Zhuo Liu,Hangfeng He
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: 19 pages, accepted to NAACL Findings
- **标题**: 通过约束意识提示来缓解多模式空间关系中的幻觉
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 空间关系幻觉在大型视觉模型（LVLMS）中构成了持续的挑战，从而导致对图像中对象位置和空间配置产生错误的预测。为了解决这个问题，我们提出了一个限制感知的提示框架，旨在减少空间关系幻觉。具体而言，我们介绍了两种类型的约束：（1）双向约束，确保对成对对象关系的一致性，以及（2）传递性约束，这可以在多个对象之间实施关系依赖性。通过合并这些约束，LVLM可以产生更多的空间相干和一致的输出。我们在三个广泛使用的空间关系数据集上评估了我们的方法，并证明了现有方法的性能提高。此外，对各种双向关系分析选择和传递性参考选择的系统分析突出了我们方法在纳入减轻空间关系幻觉的约束时的更大可能性。

### What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations 
[[arxiv](https://arxiv.org/abs/2502.08279)] [[cool](https://papers.cool/arxiv/2502.08279)] [[pdf](https://arxiv.org/pdf/2502.08279)]
> **Authors**: Dongqi Liu,Chenxi Whitehouse,Xi Yu,Louis Mahon,Rohit Saxena,Zheng Zhao,Yifu Qiu,Mirella Lapata,Vera Demberg
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 那是什么？视频到文本摘要数据集用于科学演示
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 将记录的视频转换为简洁明了的文本摘要是多模式学习的日益增长的挑战。本文介绍了Vista，Vista是一个专门为科学领域中视频到文本摘要而设计的数据集。 Vista包含18,599个记录的AI会议演示文稿，并搭配其相应的纸张摘要。我们基准了最先进的大型模型的性能，并应用基于计划的框架来更好地捕获摘要的结构化性质。人类和自动化的评估都证实，明确的计划提高了总结质量和事实的一致性。但是，模型和人类绩效之间仍然存在很大的差距，强调了科学视频摘要的挑战。

### SARChat-Bench-2M: A Multi-Task Vision-Language Benchmark for SAR Image Interpretation 
[[arxiv](https://arxiv.org/abs/2502.08168)] [[cool](https://papers.cool/arxiv/2502.08168)] [[pdf](https://arxiv.org/pdf/2502.08168)]
> **Authors**: Zhiming Ma,Xiayang Xiao,Sihao Dong,Peidong Wang,HaiPeng Wang,Qingyun Pan
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: Sarchat-Bench-2M：用于SAR图像解释的多任务视觉语言基准
- **领域**: 计算语言学
- **摘要**: 作为一个强大的全天候地球观测工具，合成的孔径雷达（SAR）遥感可以实现重要的军事侦察，海上监视和基础设施监测。尽管视觉语言模型（VLM）在自然语言处理和图像理解方面取得了显着进步，但由于域专业知识不足，它们的应用在专业领域中仍然有限。本文创新提出了第一个用于SAR图像的大规模多模式对话数据集，该图像名为Sarchat-2m，其中包含大约200万个高质量的图像文本对，其中包含带有详细目标注释的各种情况。 This dataset not only supports several key tasks such as visual understanding and object detection tasks, but also has unique innovative aspects: this study develop a visual-language dataset and benchmark for the SAR domain, enabling and evaluating VLMs' capabilities in SAR image interpretation, which provides a paradigmatic framework for constructing multimodal datasets across various remote sensing vertical domains.通过对16个主流VLM的实验，数据集的有效性已得到充分验证。该项目将在https://github.com/jimmymya99/sarchat上发布。

### Efficient Multitask Learning in Small Language Models Through Upside-Down Reinforcement Learning 
[[arxiv](https://arxiv.org/abs/2502.09854)] [[cool](https://papers.cool/arxiv/2502.09854)] [[pdf](https://arxiv.org/pdf/2502.09854)]
> **Authors**: Yu-Chen Lin,Sanat Sharma,Hari Manikandan,Jayant Kumar,Tracy Holloway King,Jing Zheng
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 通过颠倒的强化学习，在小语言模型中有效的多任务学习
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 在这项工作中，我们证明了小语言模型（SLM），特别是100m参数GPT-2模型，可以在多任务及时生成任务中实现竞争性能，而仅需要大语言模型（LLMS）所需的计算资源的一小部分。通过从功能强大的LLM，Llama-3中进行颠倒的增强学习和合成数据蒸馏的新型组合，我们训练一个SLM，该SLM在最先进的模型的5％以内，包括Llama-3，Qwen2和Mistral，尽管高达80倍，但最多适合资源限制和实时的应用程序。这项研究突出了SLM作为多模式设置中有效的多任务学习者的潜力，为LLM提供了有希望的可扩展，低延迟部署的替代方案。

### Large Language Models and Provenance Metadata for Determining the Relevance of Images and Videos in News Stories 
[[arxiv](https://arxiv.org/abs/2502.09689)] [[cool](https://papers.cool/arxiv/2502.09689)] [[pdf](https://arxiv.org/pdf/2502.09689)]
> **Authors**: Tomas Peterka,Matyas Bohacek
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 大型语言模型和出处元数据，用于确定新闻故事中图像和视频的相关性
- **领域**: 计算语言学,计算机视觉和模式识别,计算机与社会
- **摘要**: 最有效的错误信息广告系列是多模式的，通常将文本与图像和视频结合起来，或者完全制造出来，以支持给定的叙述。当代检测错误信息的方法，无论是在深层还是文本文章中，通常都会错过多种方式之间的相互作用。本文提出的系统围绕大型语言模型建立，解决了这些挑战。它分析了文章的文本和包含图像和视频的出处元数据，以确定它们是否相关。我们开源系统原型和交互式Web界面。

### Multi-level Conflict-Aware Network for Multi-modal Sentiment Analysis 
[[arxiv](https://arxiv.org/abs/2502.09675)] [[cool](https://papers.cool/arxiv/2502.09675)] [[pdf](https://arxiv.org/pdf/2502.09675)]
> **Authors**: Yubo Gao,Haotian Wu,Lei Zhang
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: 5 pages, 1 figure
- **标题**: 多模式情绪分析的多级冲突感知网络
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 多模式情感分析（MSA）旨在通过利用文本，声学和视觉方式来识别人类情绪，从而如何充分利用不同方式之间的相互作用是MSA的核心挑战。互动包含一致性和冲突方面。当前的作品主要强调一致性和单峰方式之间的固有差异，从而忽略了双峰组合之间也存在潜在的冲突的事实。此外，基于多任务的基于学习的冲突建模方法通常依赖于不稳定的生成标签。为了应对这些挑战，我们提出了一个新颖的多层次冲突感知网络（MCAN）进行多模式情感分析，该网络逐渐将一致性和冲突成分与单峰和双峰表示形式分离，并进一步利用了冲突成分与冲突组成部分。在冲突建模分支中，我们在表示和预测的输出水平上进行差异约束，避免依赖生成的标签。 CMU-MOSI和CMU-MOSEI数据集的实验结果证明了所提出的MCAN的有效性。

### From No to Know: Taxonomy, Challenges, and Opportunities for Negation Understanding in Multimodal Foundation Models 
[[arxiv](https://arxiv.org/abs/2502.09645)] [[cool](https://papers.cool/arxiv/2502.09645)] [[pdf](https://arxiv.org/pdf/2502.09645)]
> **Authors**: Mayank Vatsa,Aparna Bharati,Surbhi Mittal,Richa Singh
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 从不知道：分类法，挑战和机会在多模式基础模型中进行否定理解
- **领域**: 计算语言学,人工智能
- **摘要**: 否定是一种传达缺席，否认或矛盾的语言结构，对多语言多模式基础模型构成了重大挑战。这些模型在机器翻译，文本引导的生成，图像字幕，音频交互和视频处理等任务中表现出色，但通常很难准确解释跨不同语言和文化背景的否定。从这个角度来看，我们提出了否定结构的全面分类法，说明结构，语义和文化因素如何影响多模式基础模型。我们提出了开放的研究问题，并强调了关键挑战，强调解决这些问题以实现强大的否定处理的重要性。最后，我们倡导专门的基准，特定于语言的令牌化，细粒度的注意机制和高级多模式体系结构。这些策略可以促进更适应性和语义上精确的多模式基础模型，可以更好地浏览和准确地解释多语言多模式环境中的否定复杂性。

### SparQLe: Speech Queries to Text Translation Through LLMs 
[[arxiv](https://arxiv.org/abs/2502.09284)] [[cool](https://papers.cool/arxiv/2502.09284)] [[pdf](https://arxiv.org/pdf/2502.09284)]
> **Authors**: Amirbek Djanibekov,Hanan Aldarmaki
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: Sparqle：语音查询通过LLMS进行文本翻译
- **领域**: 计算语言学,人工智能
- **摘要**: 随着大语言模型（LLM）的不断增长的影响，对将语音表述与它们整合在一起的兴趣越来越多，以实现更多无缝的多模式处理和语音理解。这项研究介绍了一种新颖的方法，该方法通过教学调节的LLM结合语音到文本翻译来利用自我监督的语音表示形式。所提出的方法利用模式适配器使用英语数据将提取的语音特征与指令调整的LLMS对齐。我们的实验表明，该方法有效地保留了输入语音的语义内容，并作为自我监督的语音模型和教学调节的LLM之间的有效桥梁，为各种语音理解应用程序提供了有希望的解决方案。

### Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval 
[[arxiv](https://arxiv.org/abs/2502.11431)] [[cool](https://papers.cool/arxiv/2502.11431)] [[pdf](https://arxiv.org/pdf/2502.11431)]
> **Authors**: Ze Liu,Zhengyang Liang,Junjie Zhou,Zheng Liu,Defu Lian
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 任何信息都值得一个单个屏幕截图：通过可视化信息检索统一搜索
- **领域**: 计算语言学
- **摘要**: 随着多模式技术的普及，它获得了越来越多的兴趣，以获取视觉形式的有用信息。在这项工作中，我们正式定义了一种称为\ textIt {可视化信息检索}或\ textbf {vis-ir}的新兴IR范式，其中多模式信息（例如文本，图像，表格和图表）由称为\ textbf {sexenshots的统一视觉格式共同表示。我们进一步为Vis-Ir做出了三个关键贡献。首先，我们创建\ textbf {vira}（Vis-ir聚合），这是一个大规模数据集，其中包括来自不同来源的大量屏幕截图，经过精心策划为字幕和提问格式。其次，我们开发\ textbf {unise}（通用屏幕截图嵌入式），这是一个检索模型家族，可以使屏幕截图可以查询或在任意数据模态上查询或查询。最后，我们构建\ textbf {mvrb}（大量可视化的IR基准），这是一个涵盖各种任务表格和应用程序方案的全面基准。通过对MVRB的广泛评估，我们强调了现有的多模式猎犬的不足以及木质的实质性改进。我们的工作将与社区分享，为这个新兴领域奠定坚实的基础。

### Do we Really Need Visual Instructions? Towards Visual Instruction-Free Fine-tuning for Large Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.11427)] [[cool](https://papers.cool/arxiv/2502.11427)] [[pdf](https://arxiv.org/pdf/2502.11427)]
> **Authors**: Zikang Liu,Kun Zhou,Wayne Xin Zhao,Dawei Gao,Yaliang Li,Ji-Rong Wen
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: under review
- **标题**: 我们真的需要视觉说明吗？迈向大型视觉模型的无视觉教学微调
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 视觉指导调整已成为引发大型视觉模型（LVLM）多模式解决功能的主要技术。尽管取得了成功，但由于视觉说明需要图像作为输入，它将留下差距从骨干LLMS继承解决任务的功能，并使收集大型数据集的昂贵。为了解决这个问题，我们建议VIFT，这是LVLMS的无视觉指导微调框架。在VIFT中，我们只需要在培训期间单独学习解决任务和视觉感知能力的仅文本说明和图像标题数据。在推断期间，我们提取和组合文本和图像输入的表示形式，以融合这两种能力来完成多模式任务。实验结果表明，VIFT可以在基准之后的几个视觉推理和视觉指导上实现最新的性能，并具有较少的训练数据。我们的代码和数据将公开发布。

### VLDBench: Vision Language Models Disinformation Detection Benchmark 
[[arxiv](https://arxiv.org/abs/2502.11361)] [[cool](https://papers.cool/arxiv/2502.11361)] [[pdf](https://arxiv.org/pdf/2502.11361)]
> **Authors**: Shaina Raza,Ashmal Vayani,Aditya Jain,Aravind Narayanan,Vahid Reza Khazaie,Syed Raza Bashir,Elham Dolatabadi,Gias Uddin,Christos Emmanouilidis,Rizwan Qureshi,Mubarak Shah
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: under review
- **标题**: VLDBENCH：视觉语言模型虚假信息检测基准测试
- **领域**: 计算语言学
- **摘要**: AI生成的内容的迅速上升使检测到虚假信息越来越具有挑战性。特别是，多模式的虚假信息，即包含图像和文本的在线帖子 - 具有伪造信息的文本是专门设计用于欺骗的。尽管现有的AI安全基准主要解决偏见和毒性，但多模式的虚假信息检测仍然在很大程度上没有被忽略。为了应对这一挑战，我们介绍了视觉虚假信息检测基准VLDBENCH，这是第一个综合基准，用于检测单峰（仅文本）和多模式（文本和图像）内容，包括31,000}新闻文章图像对，跨越13个不同的类别，用于强大的评估，包括31,000}新闻文章图像对。 VLDBENCH具有严格的半自动数据策展管道，有22个域专家专用300多个小时}进行注释，达到了强大的通知者一致性（Cohen Kappa = 0.78）。我们广泛评估了最先进的大语模型（LLM）和视觉模型（VLMS），这表明与单模型模型相比，多模式新闻帖子中的文本和视觉提示会提高5-35％的虚假信息检测准确性。 VLDBENCH与AI治理框架（例如《欧盟AI法》，NIST指南和MIT AI风险存储库2024年的一致性，VLDBENCH预计将成为检测在线多模式内容中的虚假信息的基准。我们的代码和数据将公开可用。

### CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships? 
[[arxiv](https://arxiv.org/abs/2502.11300)] [[cool](https://papers.cool/arxiv/2502.11300)] [[pdf](https://arxiv.org/pdf/2502.11300)]
> **Authors**: Aashish Anantha Ramakrishnan,Aadarsh Anantha Ramakrishnan,Dongwon Lee
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: :I.2.7; I.2.10
- **标题**: 亲切：多模式大语模型可以有效地了解连贯的关系吗？
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 多模式的大语言模型（MLLM）以其在各种问题域中的出色指导跟踪和推理能力而闻名。但是，现有的基准主要集中于评估下游任务中的事实和逻辑正确性，而有限的重点是评估MLLM的解释实用线索和模式间关系的能力。为了解决这一差距，我们评估了MLLM在使用相干关系进行多模式话语分析（MDA）方面的能力。我们的基准诚挚的基准包括在不同水平的粒度水平的3个不同的话语领域之间的一系列连贯关系。通过采用不同提示策略的10+ MLLMS的实验，我们表明，即使是Gemini 1.5 Pro和GPT-4O等顶级模型也无法匹配简单基于分类器的基线的性能。这项研究强调需要超越基于相似性的指标，并采用以话语为导向的框架来评估MLLM，从而对其能力进行更细微的评估。基准和代码可在以下网址获得：https：//github.com/aashish2000/cordial。

### A Survey of LLM-based Agents in Medicine: How far are we from Baymax? 
[[arxiv](https://arxiv.org/abs/2502.11211)] [[cool](https://papers.cool/arxiv/2502.11211)] [[pdf](https://arxiv.org/pdf/2502.11211)]
> **Authors**: Wenxuan Wang,Zizhan Ma,Zheng Wang,Chenghan Wu,Wenting Chen,Xiang Li,Yixuan Yuan
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 一项基于LLM的医学代理的调查：我们离Baymax有多远？
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 大型语言模型（LLM）通过开发基于LLM的代理商可以理解，推理和协助医疗任务来改变医疗保健。这项调查提供了对基于LLM的医学代理商的全面审查，研究了其体系结构，应用和挑战。我们分析了医疗代理系统的关键组成部分，包括系统概况，临床计划机制，医疗推理框架和外部能力增强。该调查涵盖了主要的应用程序方案，例如临床决策支持，医疗文献，培训模拟和医疗服务优化。我们讨论用于评估这些代理在医疗保健环境中的表现的评估框架和指标。尽管基于LLM的代理商在增强医疗保健提供方面表现出了希望，但仍有一些挑战，包括幻觉管理，多模式整合，实施障碍和道德考虑。该调查结束了，强调未来的研究方向，包括受LLM体系结构最近发展，与物理系统集成以及培训模拟的改进的医学推理的进步。这项工作为研究人员和从业人员提供了有关医学中LLM代理的现状和未来前景的结构化概述。

### Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.11184)] [[cool](https://papers.cool/arxiv/2502.11184)] [[pdf](https://arxiv.org/pdf/2502.11184)]
> **Authors**: Wenxuan Wang,Xiaoyuan Liu,Kuiyi Gao,Jen-tse Huang,Youliang Yuan,Pinjia He,Shuai Wang,Zhaopeng Tu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 看不到树木的森林：对多模式LLMS的多模式安全意识
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,多媒体
- **摘要**: 多模式大型语言模型（MLLM）通过通过文本和图像启用互动来扩展传统语言模型的功能。但是，确保这些模型的安全仍然是一个重大挑战，特别是在准确确定多模式内容是安全的还是不安全的，我们将其称为安全意识。在本文中，我们介绍了MMSAFEAWARE，这是第一个综合的多模式安全意识基准测试，旨在评估29个安全场景中的MLLM，并使用1500个精心策划的图像推出对。 MMSAFEAWARE包括不安全和过度安全子集，以评估模型能力，以正确识别不安全的内容并避免过度敏感，从而阻碍有用的帮助。使用mmsafeaware评估九种MLLM的九种MLLM表明，当前模型不够安全，而且通常过于敏感。例如，GPT-4V将36.1％的不安全输入误以为是安全的，而59.9％的良性输入则为不安全。我们进一步探讨了三种方法，以改善基于安全意识的方法，视觉对比度解码以及以视觉为中心的推理微调，但没有人可以实现令人满意的性能。我们的发现突出了以强大的安全意识开发MLLM的深刻挑战，强调了该领域进一步研究的必要性。所有代码和数据将公开使用，以促进未来的研究。

### DuplexMamba: Enhancing Real-time Speech Conversations with Duplex and Streaming Capabilities 
[[arxiv](https://arxiv.org/abs/2502.11123)] [[cool](https://papers.cool/arxiv/2502.11123)] [[pdf](https://arxiv.org/pdf/2502.11123)]
> **Authors**: Xiangyu Lu,Wang Xu,Haoyu Wang,Hongyun Zhou,Haiyan Zhao,Conghui Zhu,Tiejun Zhao,Muyun Yang
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: 12 pages, 6 figures
- **标题**: 复合amamba：增强使用双工和流媒体功能的实时演讲对话
- **领域**: 计算语言学
- **摘要**: 实时语音对话对于需要双工和流式传输功能的自然和高效的人机相互作用至关重要。传统的基于变压器的对话聊天机器人以转弯的方式运行，并表现出二次计算复杂性，随着输入尺寸的增加而增长。在本文中，我们提出了Duplexmamba，这是一种基于曼巴am的端到端多模式模型，用于语音到文本对话。 DuplexMamba可以同时进行输入处理和输出生成，并动态调整以支持实时流。具体来说，我们开发了一个基于MAMBA的语音编码器，并使用基于Mamba的语言模型进行调整。此外，我们引入了一种新型的双工解码策略，该策略使双工amamba能够同时处理输入并生成输出。实验结果表明，双工amamba成功实现了双链体和流功能，同时实现了与几种自动语音识别（ASR）任务（ASR）和语音助手基准测试评估的最近开发的基于变压器的模型相当的性能。我们的代码和模型已发布

### Demystifying Hateful Content: Leveraging Large Multimodal Models for Hateful Meme Detection with Explainable Decisions 
[[arxiv](https://arxiv.org/abs/2502.11073)] [[cool](https://papers.cool/arxiv/2502.11073)] [[pdf](https://arxiv.org/pdf/2502.11073)]
> **Authors**: Ming Shan Hee,Roy Ka-Wei Lee
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: Preprint. Accepted at ICWSM'25
- **标题**: 揭开仇恨内容的神秘面纱：利用大型的多模式模型来仇恨模因检测可解释的决定
- **领域**: 计算语言学
- **摘要**: 仇恨的模因检测是由于解释模因中的隐性仇恨信息和上下文提示的复杂性，因此提出了一个重大挑战作为多模式任务。先前的方法具有微调的预训练的视觉模型（PT-VLM），利用它们在训练期间获得的知识及其注意力机制来理解模因含量。但是，这些模型对隐性知识和复杂的注意机制的依赖使他们的决定难以解释，这对于建立对模因分类的信任至关重要。在本文中，我们介绍了Intmeme，这是一个新颖的框架，该框架利用大型多模型（LMMS）进行可恶的模因分类和可解释的决定。 Intmeme解决了提高模因适度精度和解释性的双重挑战。该框架使用LMM来生成类似人类的模因的解释性分析，从而为多模式内容和上下文提供了更深入的见解。此外，它对模因及其解释都使用独立的编码模块，然后将其组合起来以增强分类性能。我们的方法解决了与PT-VLM相关的不透明度和错误分类问题，从而优化了LMM用于仇恨模因检测的使用。我们通过在三个数据集中进行的全面实验来证明Intmeme的有效性，从而展示了其优越性比最先进的模型。

### MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.11051)] [[cool](https://papers.cool/arxiv/2502.11051)] [[pdf](https://arxiv.org/pdf/2502.11051)]
> **Authors**: Jiahao Huo,Yibo Yan,Xu Zheng,Yuanhuiyi Lyu,Xin Zou,Zhihua Wei,Xuming Hu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: mmunlearner：在多模式模型时代重新设计多模式的机器
- **领域**: 计算语言学,人工智能
- **摘要**: 机器学习的最新进展（MU）引入了解决方案，以选择性去除深度神经网络中编码的私人或敏感信息。但是，多模式大语言模型（MLLM）的MU仍处于其新生阶段。因此，我们建议在MLLM时代重新重新重新重新制定多模式MU的任务，该任务仅删除与给定实体相关的视觉模式，同时保留语言模型骨干的原始参数中编码的相应文本知识。此外，我们开发了一种新型的几何梯度下降方法mmunlearner。它通过在学习期间的其余概念和文本知识共同限制的重量显着性图更新了MLLM的权重，从而保留了非目标知识必不可少的参数。广泛的实验表明，Mmunlearner超过了所有评估维度，通过梯度上升（GA）或负偏好优化（NPO）直接使用VQA数据对MLLM进行了捕获。我们的代码将在接受后发布。

### Akan Cinematic Emotions (ACE): A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues 
[[arxiv](https://arxiv.org/abs/2502.10973)] [[cool](https://papers.cool/arxiv/2502.10973)] [[pdf](https://arxiv.org/pdf/2502.10973)]
> **Authors**: David Sasu,Zehui Wu,Ziwei Gong,Run Chen,Pengyuan Shi,Lin Ai,Julia Hirschberg,Natalie Schluter
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: Akan电影情绪（ACE）：在电影对话中识别情感的多模式多方数据集
- **领域**: 计算语言学
- **摘要**: 在本文中，我们介绍了Akan对话情感（ACE）数据集，这是非洲语言的第一个多模式情感对话数据集，以解决情感识别研究中低资源语言缺乏资源。为Akan语言开发的Ace包含385个情感标记的对话，并在音频，视觉和文字方式上进行了6,162个话语，以及单词级别的韵律突出注释。该数据集中的韵律标签的存在也使其成为第一个韵律注释的非洲语言数据集。我们通过实验使用最先进的情感识别方法来证明ACE的质量和实用性，从而为未来的研究建立了坚实的基准。我们希望Ace启发在包容性，语言和文化上多样化的NLP资源上的进一步工作。

### MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models 
[[arxiv](https://arxiv.org/abs/2502.10886)] [[cool](https://papers.cool/arxiv/2502.10886)] [[pdf](https://arxiv.org/pdf/2502.10886)]
> **Authors**: Vanya Cohen,Raymond Mooney
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: Met-Bench：用于评估视觉和推理模型局限的多模式实体跟踪
- **领域**: 计算语言学
- **摘要**: 实体跟踪是自然语言理解中的基本挑战，需要模型维持实体的连贯表示。以前的工作已经在纯粹基于文本的任务中基于实体跟踪性能。我们介绍了Met-Bench，这是一种多模式实体跟踪基准测试，旨在评估视觉模型在跨模式中跟踪实体状态的能力。使用两个结构化域，国际象棋和壳游戏，我们评估当前模型如何整合基于文本和图像的状态更新。我们的发现揭示了基于文本的跟踪和基于图像的跟踪之间存在显着的性能差距，并且这种性能差距源于视觉推理而不是感知的缺陷。我们进一步表明，明确的基于文本的推理策略可以改善绩效，但仍然存在实质性限制，尤其是在长途多模式场景中。我们的结果强调了需要改进的多模式表示和推理技术，以弥合文本和视觉实体跟踪之间的差距。

### MM-RLHF: The Next Step Forward in Multimodal LLM Alignment 
[[arxiv](https://arxiv.org/abs/2502.10391)] [[cool](https://papers.cool/arxiv/2502.10391)] [[pdf](https://arxiv.org/pdf/2502.10391)]
> **Authors**: Yi-Fan Zhang,Tao Yu,Haochen Tian,Chaoyou Fu,Peiyan Li,Jianshu Zeng,Wulin Xie,Yang Shi,Huanyu Zhang,Junkang Wu,Xue Wang,Yibo Hu,Bin Wen,Fan Yang,Zhang Zhang,Tingting Gao,Di Zhang,Liang Wang,Rong Jin,Tieniu Tan
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: Project Page: https://mm-rlhf.github.io/
- **标题**: MM-RLHF：多模式LLM对齐中的下一步
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 尽管多模式大语言模型（MLLM）的显着进步，但大多数最先进的模型尚未与人类偏好进行彻底的一致性。之所以存在这一差距，是因为当前的一致性研究主要在特定领域（例如减少幻觉）取得了进步，而对与人类偏好的比对模型是否可以系统地增强MLLM能力的更广泛的问题仍然在很大程度上没有探索。为此，我们介绍了MM-RLHF，该数据集包含$ \ MATHBF {120K} $细粒度，人类通知比较比较对。该数据集代表了对现有资源的实质性进步，提供了较高的规模，多样性，注释粒度和质量。利用此数据集，我们提出了几项关键创新，以提高奖励模型的质量和对齐算法的效率。值得注意的是，我们引入了基于批评的奖励模型，该模型在分配分数之前会产生模型输出的评论，与传统的标量奖励机制相比，提供了增强的可解释性和更有信息的反馈。此外，我们提出了动态奖励缩放，该方法可根据奖励信号调节每个样本的减肥重量，从而优化使用高质量比较对。我们的方法在$ \ mathbf {10} $不同的尺寸和$ \ mathbf {27} $基准中进行了严格评估，结果表明模型性能的显着和一致的改进。具体来说，带有MM-RLHF和我们的对齐算法的微调LLAVA-OV-7B导致$ \ MathBf {19.5} $％的会话能力和$ \ MATHBF {60} $％的安全性提高。我们已经开源了偏好数据集，奖励模型，培训和评估法，以及奖励建模和安全基准。有关更多详细信息，请访问我们的项目页面：https：//mm-rlhf.github.io。

### VisCon-100K: Leveraging Contextual Web Data for Fine-tuning Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.10250)] [[cool](https://papers.cool/arxiv/2502.10250)] [[pdf](https://arxiv.org/pdf/2502.10250)]
> **Authors**: Gokul Karthik Kumar,Iheb Chaabane,Kebin Wu
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: Accepted at PAKDD 2025
- **标题**: Viscon-100K：利用上下文网络数据进行微调视觉语言模型
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 视觉语言模型（VLM）在各种视觉基准中都表现出色，但通常受到缺乏高质量视觉微调数据的限制。为了应对这一挑战，我们介绍了Viscon-100K，这是一种来自交织的图像文本Web文档的新颖数据集。我们的方法将45K Web文档从质子数据集转换为100k图像对话样本。我们利用GPT-4V生成图像上下文字幕和OpenChat 3.5模型，将这些字幕转换为各种自由形式和多项选择的问答对。集成此数据集以进行微调大大提高了多个基准测试的VLM性能。与仅关注细粒视觉内容的方法不同，我们的方法利用了Web上下文，从而产生了卓越的结果。我们还发现，“漏水模式”，其中的对话样本包含可以从图像及其上下文标题中回答的问题，它的表现优于标题和Q＆A对的非裸露组合。 VisCon-100K数据集通过两种流行的VLM方法显示出很强的性能：使用图像标题数据（ShareGPT4V-7B）与视觉编码器排列的仅文本大型语言模型（LLM）和使用交织织物的图像图形数据数据来对齐。除了释放VisCon-100K数据集外，我们还提供了在该数据集上培训的上下文标题者，从​​而促进可扩展的微调数据生成，以实现未来的研究和开源应用程序。使用同一条管道，但将我们训练的上下文标题用GPT-4V代替，我们还释放了较大的Viscon-1M数据集。

### A Preliminary Exploration with GPT-4o Voice Mode 
[[arxiv](https://arxiv.org/abs/2502.09940)] [[cool](https://papers.cool/arxiv/2502.09940)] [[pdf](https://arxiv.org/pdf/2502.09940)]
> **Authors**: Yu-Xiang Lin,Chih-Kai Yang,Wei-Chih Chen,Chen-An Li,Chien-yu Huang,Xuanjun Chen,Hung-yi Lee
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: Work in progress
- **标题**: GPT-4O语音模式的初步探索
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 随着多模式大语言模型的兴起，GPT-4O作为开创性模型脱颖而出，推动我们评估其功能。该报告评估了各种任务中的GPT-4O，以分析其音频处理和推理能力。我们发现，GPT-4O在音频，语音和音乐理解方面表现出很强的知识，在意图分类，口语命令分类，语义和语法推理，多语言语音识别和歌唱分析等任务中表现良好。与其他大型音频语言模型（LALMS）相比，它还显示出对幻觉的鲁棒性。但是，它在音频持续时间预测和仪器分类等任务上挣扎。此外，GPT-4O的安全机制导致其拒绝诸如说话者识别，年龄分类，MOS预测和音频深击检测等任务。值得注意的是，该模型在响应不同数据集上的说话者验证任务时表现出明显不同的拒绝率。这可能是由于随附的说明或输入音频质量的变化所致，这表明其内置保障的敏感性。最后，我们承认模型性能随评估协议而变化。该报告仅是对LALMS现状的初步探索。

### MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition 
[[arxiv](https://arxiv.org/abs/2502.12478)] [[cool](https://papers.cool/arxiv/2502.12478)] [[pdf](https://arxiv.org/pdf/2502.12478)]
> **Authors**: Yang Yang,Xunde Dong,Yupeng Qiang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: MSE-AUPAPTER：轻巧的插件赋予LLMS，具有执行多模式分析和情感识别的能力
- **领域**: 计算语言学
- **摘要**: 基于预训练的语言模型的对话（ERC）方法（ERC）方法中的当前多模式情感分析（MSA）和情绪识别表现出两个主要局限性：1）一旦接受过MSA和ERC任务的培训，这些预训练的语言模型将失去其原始的广义能力。 2）他们需要大量的计算资源。随着预训练的语言模型的规模不断增长，使用以前的方法培训更大的多模式分析模型可能会导致不必要的计算成本。为了应对这一挑战，我们提出了\ textbf {m} ult-Imodal \ textbf {s} intiments Anallys和\ textbf {e}运动识别\ textbf {adapter}（mse-adapter），一个轻量级和适应性的插件。该插件使一个大型语言模型（LLM）能够使用最小的计算开销执行MSA或ERC任务（在6/7B型号上仅引入约260万至280万的可训练参数），同时保留LLM的内在功能。在MSE适配器中，引入了文本指标混合物（TGM）模块，以通过Hadamard产品在非文本和文本模式之间建立明确的连接。这使得非文本模式可以更好地与特征级别的文本方式保持一致，从而促进了高质量的伪代币的产生。使用消费级GPU和开源LLM（QWEN-1.8B，CHATGLM3-6B-BASE和LLAMA2-7B）作为骨干进行了大量实验。结果证明了拟议的插件的有效性。盲目审查后，该代码将在Github上发布。

### On the Robust Approximation of ASR Metrics 
[[arxiv](https://arxiv.org/abs/2502.12408)] [[cool](https://papers.cool/arxiv/2502.12408)] [[pdf](https://arxiv.org/pdf/2502.12408)]
> **Authors**: Abdul Waheed,Hanin Atwany,Rita Singh,Bhiksha Raj
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 25 Pages. Work in Progress
- **标题**: 在ASR指标的稳健近似值上
- **领域**: 计算语言学
- **摘要**: 语音基础模型的最新进展主要是通过扩展模型大小和数据的驱动，从而使它们能够执行各种任务，包括语音识别。传统上，使用单词错误率（WER）和字符错误率（CER）等指标来评估ASR模型，这些指标取决于地面真相标签。由于来自不同领域和测试条件的标记数据有限，这些模型超出标准基准的真实概括能力尚不清楚。此外，标记数据既昂贵又耗时。为了解决这个问题，我们提出了一种新颖的无标签方法，用于近似ASR性能指标，从而消除了对地面真相标签的需求。我们的方法利用统一空间中的多模式嵌入来进行语音和转录表示，并结合了高质量的代理模型来计算代理指标。这些功能用于训练回归模型，以预测诸如单词错误率（WER）和字符错误率（CER）之类的关键ASR指标。我们在代表标准和现有测试条件的14个数据集中尝试了40多个模型。我们的结果表明，在所有实验配置中，我们将指标近似于单位的绝对差异，从而超过50 \％的基线表现优于最新基线。

### How to Upscale Neural Networks with Scaling Law? A Survey and Practical Guidelines 
[[arxiv](https://arxiv.org/abs/2502.12051)] [[cool](https://papers.cool/arxiv/2502.12051)] [[pdf](https://arxiv.org/pdf/2502.12051)]
> **Authors**: Ayan Sengupta,Yash Goel,Tanmoy Chakraborty
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 20 pages, 8 tables, 4 figures
- **标题**: 如何使用缩放法来提高神经网络？调查和实用准则
- **领域**: 计算语言学,机器学习
- **摘要**: 神经缩放定律通过揭示模型大小，数据集量和计算资源之间的可预测关系，彻底改变了大规模AI模型的设计和优化。早期研究在模型绩效中建立了幂律关系，从而导致了最佳的缩放策略。但是，最近的研究强调了它们在架构，方式和部署环境之间的局限性。稀疏的模型，专家的混合物，检索式学习和多模型通常偏离传统的缩放模式。此外，缩放行为在视觉，增强学习和微调等领域各不相同，强调了对更细微的方法的需求。在这项调查中，我们综合了50多项研究的见解，研究了缩放定律的理论基础，经验发现和实际含义。我们还探讨了关键挑战，包括数据效率，推理缩放和特定于体系结构的约束，并提倡针对现实世界应用量身定制的自适应缩放策略。我们建议，尽管扩展法律提供了有用的指南，但它们并不总是在所有架构和培训策略中概括。

### Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction 
[[arxiv](https://arxiv.org/abs/2502.11946)] [[cool](https://papers.cool/arxiv/2502.11946)] [[pdf](https://arxiv.org/pdf/2502.11946)]
> **Authors**: Ailin Huang,Boyong Wu,Bruce Wang,Chao Yan,Chen Hu,Chengli Feng,Fei Tian,Feiyu Shen,Jingbei Li,Mingrui Chen,Peng Liu,Ruihang Miao,Wang You,Xi Chen,Xuerui Yang,Yechang Huang,Yuxiang Zhang,Zheng Gong,Zixin Zhang,Hongyu Zhou,Jianjian Sun,Brian Li,Chengting Feng,Changyi Wan,Hanpeng Hu, et al. (120 additional authors not shown)
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: Step-Adio：智能语音互动中的统一理解和产生
- **领域**: 计算语言学,人工智能,人机交互,声音,音频和语音处理
- **摘要**: 实时语音互动是人机合作的基本接口，具有巨大的潜力。但是，当前的开源模型面临着诸如语音数据收集的高成本，动态控制中的弱点和智力有限的限制。为了应对这些挑战，本文介绍了Step-Audio，这是第一个可以生产的开源解决方案。主要贡献包括：1）130B参数统一的语音文本多模式模型，该模型具有开源的Step-Audio-Chat版本，可以实现统一的理解和生成； 2）一种生成的语音数据引擎，该数据引擎建立了负担得起的语音克隆框架，并通过蒸馏生产开源的轻巧的步骤ADIO-TTS-3B模型； 3）指令驱动的精细控制系统，实现了方言，情感，唱歌和说唱的动态调整； 4）增强的认知体系结构增强了工具呼叫和角色扮演能力，以有效地管理复杂的任务。根据我们新的Stepeval-Audio-360评估基准，Step-Audio在人类评估中实现了最先进的表现，尤其是在以下教学方面。在诸如Llama问题之类的开源基准上，表明平均绩效提高了9.3％，这表明我们致力于推进开源多模式语言技术的发展。我们的代码和模型可在https://github.com/stepfun-ai/step-audio上找到。

### EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.11916)] [[cool](https://papers.cool/arxiv/2502.11916)] [[pdf](https://arxiv.org/pdf/2502.11916)]
> **Authors**: Jiamin Su,Yibo Yan,Fangteng Fu,Han Zhang,Jingheng Ye,Xiang Liu,Jiahao Huo,Huiyu Zhou,Xuming Hu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: JS and YY are co-first authors. XH is the corresponding author
- **标题**: ESSAYJUDGE：用于评估多模式大语模型的自动论文评分能力的多个基准测试
- **领域**: 计算语言学,人工智能
- **摘要**: 自动论文评分（AES）通过提供对写作任务的可扩展和一致的评估，在教育评估中起着至关重要的作用。但是，传统的AES系统面临三个主要挑战：（1）依赖限制可推广性的手工特征，（2）难以捕获连贯性和论证等细粒度的性状，以及（3）无法处理多模式上下文。在多模式大语言模型（MLLM）时代，我们提出了EssayJudge，这是第一个评估跨词汇，句子和话语级特征的AES功能的多模式基准。通过利用MLLM在特定特定的评分和多模式上下文理解中的优势，EssayJudge的目标是提供精确的，上下文丰富的评估，而无需手动功能工程，以解决长期存在的AES限制。与人类评估相比，我们对18个代表性MLLM的实验揭示了AES性能的差距，尤其是在话语级的特征中，强调了基于MLLM的AES研究中进一步进步的必要性。我们的数据集和代码将在接受后提供。

### MMRC: A Large-Scale Benchmark for Understanding Multimodal Large Language Model in Real-World Conversation 
[[arxiv](https://arxiv.org/abs/2502.11903)] [[cool](https://papers.cool/arxiv/2502.11903)] [[pdf](https://arxiv.org/pdf/2502.11903)]
> **Authors**: Haochen Xue,Feilong Tang,Ming Hu,Yexin Liu,Qidong Huang,Yulong Li,Chengzhi Liu,Zhongxing Xu,Chong Zhang,Chun-Mei Feng,Yutong Xie,Imran Razzak,Zongyuan Ge,Jionglong Su,Junjun He,Yu Qiao
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: MMRC：一个大规模的基准，用于理解现实世界中的多模式大语言模型
- **领域**: 计算语言学
- **摘要**: 最近的多模式大语模型（MLLM）在开放式对话中表现出了巨大的潜力，产生了更准确和个性化的响应。但是，他们在现实世界中持续相互作用中记住，回忆和理性的能力仍然没有被淘汰。本文介绍了MMRC，这是一种多模式现实世界对话基准，用于评估MLLM的六个核心开放式能力：信息提取，多转变推理，信息更新，图像管理，内存召回和答案拒绝。通过从实际情况收集的数据，MMRC包括5,120次对话和28,720个相应的手动标记问题，对现有MLLM构成了重大挑战。对MMRC中20个MLLM的评估表明在开放式相互作用期间的准确性下降。我们确定了四种常见的故障模式：长期记忆力下降，更新事实知识的不足，累积的错误传播假设以及不愿说否。为了减轻这些问题，我们提出了一种简单而有效的笔记策略，该策略可以记录对话中的关键信息，并在其响应过程中提醒该模型，从而增强对话能力。跨六个MLLM的实验显示出显着的性能改善。

### Code-Vision: Evaluating Multimodal LLMs Logic Understanding and Code Generation Capabilities 
[[arxiv](https://arxiv.org/abs/2502.11829)] [[cool](https://papers.cool/arxiv/2502.11829)] [[pdf](https://arxiv.org/pdf/2502.11829)]
> **Authors**: Hanbin Wang,Xiaoxuan Zhou,Zhipeng Xu,Keyuan Cheng,Yuxin Zuo,Kai Tian,Jingwei Song,Junting Lu,Wenhui Hu,Xueyang Liu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 15 pages
- **标题**: 代码视频：评估多模式LLMS逻辑理解和代码生成功能
- **领域**: 计算语言学,人工智能,软件工程
- **摘要**: 本文介绍了Code-Vision，这是一种基准测试，旨在评估多模式大语言模型（MLLM）的逻辑理解和代码生成功能。它挑战MLLM的生成正确的程序，该程序根据给定流程图满足特定功能要求，该程序在视觉上代表所需的算法或过程。 Code-Vision包括三个子集：HumaneVal-V，算法和数学，它们评估了MLLMS在基本编程，算法和数学解决问题的域上的编码能力。我们的实验评估了12 mllms code-Vision。实验结果表明，专有和开源模型之间的性能差异很大。在严重问题上，GPT-4O可以达到79.3％的通过@1，但最好的开源模型只能达到15％。进一步的实验表明，与其他多模式推理基准MMCODE和MATHVISTA相比，代码视图可能构成独特的挑战。我们还探讨了开源模型性能不佳的原因。所有数据和代码均可在https://github.com/wanghanbinpanda/codevision上找到。

### InfiR : Crafting Effective Small Language Models and Multimodal Small Language Models in Reasoning 
[[arxiv](https://arxiv.org/abs/2502.11573)] [[cool](https://papers.cool/arxiv/2502.11573)] [[pdf](https://arxiv.org/pdf/2502.11573)]
> **Authors**: Congkai Xie,Shuo Cai,Wenjun Wang,Pengxiang Li,Zhijie Sang,Kejing Yang,Yiming Zhang,Zhen Li,Guanghao Zhu,Zeyu Liu,Yang Yu,Yuhang Liu,Su Lu,Baoyi He,Qi Zhou,Xiaotian Han,Jianbo Yuan,Shengyu Zhang,Fei Wu,Hongxia Yang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: infir：在推理中制定有效的小语言模型和多模式的小语言模型
- **领域**: 计算语言学,人工智能
- **摘要**: 大型语言模型（LLM）和多模式大型语言模型（MLLM）在推理能力方面取得了重大进步。但是，他们仍然面临诸如高度计算需求和隐私问题之类的挑战。本文着重于开发有效的小语言模型（SLM）和保留竞争推理能力的多模式小语言模型（MSLMS）。我们介绍了一条新型的培训管道，该管道增强了推理能力并促进在边缘设备上的部署，从而实现了最先进的性能，同时最大程度地降低了开发成本。 \ Infr〜旨在通过改善推理，减少采用障碍并通过较小的模型大小来解决隐私问题来推进AI系统。资源可在https：// github上找到。 com/reallm-labs/infir。

### Investigating Inference-time Scaling for Chain of Multi-modal Thought: A Preliminary Study 
[[arxiv](https://arxiv.org/abs/2502.11514)] [[cool](https://papers.cool/arxiv/2502.11514)] [[pdf](https://arxiv.org/pdf/2502.11514)]
> **Authors**: Yujie Lin,Ante Wang,Moye Chen,Jingyao Liu,Hao Liu,Jinsong Su,Xinyan Xiao
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 调查多模式思想链的推理时间缩放：初步研究
- **领域**: 计算语言学
- **摘要**: 最近，已经证明了思考链（COT）的推理时间缩放是解决多模式推理任务的一种有前途的方法。尽管现有的研究主要集中在基于文本的思维上，但在推理过程中的视觉和文本方式的整合仍然没有探索。在这项研究中，我们以多模式思想的探索进行了推理时间缩放的探索，旨在弥合这一差距。为了提供全面的分析，我们系统地研究了跨越各个域的10个挑战性任务，基于流行的采样和基于树搜索的推理时间缩放方法。此外，我们统一地采用了一致性增强的验证器，以确保对不同思想范式的两种方法有效指导。结果表明，多模式思想可以促进与常规文本思想相比，将两种类型的思想融合在一起，促进了更多样化的思维。尽管有这些优势，但多模式的思想仍需要更高的令牌消费来处理更丰富的视觉投入，这引起了实际应用的关注。我们希望我们对该研究线的优点和缺点的发现将激发该领域的未来作品。

### Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem? 
[[arxiv](https://arxiv.org/abs/2502.11501)] [[cool](https://papers.cool/arxiv/2502.11501)] [[pdf](https://arxiv.org/pdf/2502.11501)]
> **Authors**: Zichen Wen,Yifeng Gao,Weijia Li,Conghui He,Linfeng Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 12 pages, 3 figures
- **标题**: 多模式大语言模型中的象征性修剪：我们是否解决了正确的问题？
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 多模式的大型语言模型（MLLM）表现出出色的性能，可以进行跨模式的理解和产生，但仍处于严重的推理成本。最近，已经提出了大量的工作来解决令牌修剪的问题，该问题可以识别MLLM中的冗余令牌，然后将它们修剪以降低计算和KV存储成本，从而在没有培训的情况下导致明显的加速。尽管这些方法声称提高效率，但有关其基本设计和评估的关键问题仍未得到答案：与天真的随机代币选择相比，为什么许多现有的方法表现不佳？基于注意力的评分足以可靠地识别多余的代币吗？在代币修剪期间，语言信息真的有帮助吗？是什么使令牌重要性与重复之间的良好权衡？当前的评估协议是否全面且无偏见？对这些问题的先前研究的无知阻碍了令牌修剪的长期发展。在本文中，我们一一回答这些问题，提供有关未来令牌修剪方法设计的见解。

### Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More 
[[arxiv](https://arxiv.org/abs/2502.11494)] [[cool](https://papers.cool/arxiv/2502.11494)] [[pdf](https://arxiv.org/pdf/2502.11494)]
> **Authors**: Zichen Wen,Yifeng Gao,Shaobo Wang,Junyuan Zhang,Qintong Zhang,Weijia Li,Conghui He,Linfeng Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 15 pages, 8 figures
- **标题**: 停止在多模式语言模型中寻找重要的令牌：重复很重要
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 多模式大型语言模型中的视觉令牌通常会主导巨大的计算开销，因为与语言方式相比，它们的长度过长。最近的大量方法旨在解决令牌修剪解决这个问题的问题，该方法首先定义了代币的重要性标准，然后在推断过程中修剪了不重要的视觉令牌。但是，在本文中，我们表明，重要性不是决定是否应该修剪令牌的理想指标。令人惊讶的是，它通常会导致性能不优于随机的代币修剪，并导致对有效的注意计算运算符的不兼容。iNSTEAD，我们建议DART（对代币的重复降低令牌）根据其重复与其他代币的重复，从而导致重要的和训练的无效加速。具体地，Dart选择了一小部分枢轴令牌，然后保留对枢轴的重复较低的令牌，从而确保了在代币修剪过程中的最小信息丢失。实验表明，飞镖可以在保持可比性能的同时修剪88.9％的视力令牌，从而导致1.99 $ \ times $和2.99 $ \ times $ \ times $ $ \ times $在总时间和预填充阶段加速阶段，并且具有良好的兼容性，并且与有效的注意操作员有良好的兼容性。我们的代码可在https://github.com/zichenwen1/dart上找到。

### MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification 
[[arxiv](https://arxiv.org/abs/2502.13383)] [[cool](https://papers.cool/arxiv/2502.13383)] [[pdf](https://arxiv.org/pdf/2502.13383)]
> **Authors**: Linzhuang Sun,Hao Liang,Jingxuan Wei,Bihui Yu,Tianpeng Li,Fan Yang,Zenan Zhou,Wentao Zhang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: MM-Verify：通过经过经过经过经过经过经过经过经过经验的验证的增强多模式推理
- **领域**: 计算语言学,计算机视觉和模式识别,机器学习
- **摘要**: 根据测试时间缩放，已经证明了外部缓慢思考与验证机制的集成可以增强大语言模型（LLMS）中的多轮推理。但是，在多模式（mm）域中，仍然缺乏强大的MM佛教剂。在本文中，我们介绍了MM佛教和MM-Reasoner，以通过更长的推理和更强大的验证来增强多模式推理。首先，我们提出了一个两步的MM验证数据合成方法，该方法将基于仿真的树搜索与验证结合在一起，并使用拒绝采样来生成高质量的思想链（COT）数据。然后，该数据用于微调验证模型MM佛教符。此外，我们提出了一种更有效的方法来合成MMCOT数据，从而弥合了基于文本和多模式推理之间的差距。合成的数据用于微调MM-Reasoner。我们的MM佛罗里达人在Mathcheck，Mathvista和Mathverse Benchmarks上的表现优于所有较大模型。此外，MM-Reasoner表现出强大的有效性和可扩展性，随着数据规模的增加，性能的提高。最后，我们的方法在组合MM-Reasoner和MM佛教徒时取得了强大的性能，在Mathvista上的准确度达到65.3，超过GPT-4O（63.8），并进行了12次推广。

### Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection 
[[arxiv](https://arxiv.org/abs/2502.13061)] [[cool](https://papers.cool/arxiv/2502.13061)] [[pdf](https://arxiv.org/pdf/2502.13061)]
> **Authors**: Jingbiao Mei,Jinghong Chen,Guangyu Yang,Weizhe Lin,Bill Byrne
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Preprint. Under Review
- **标题**: 改进了大型多模型模型的微调，用于可恨模因检测
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 可恶的模因已成为互联网上的重大关注，需要强大的自动检测系统。尽管大型多模型模型在各种任务中表现出强烈的概括，但由于与新兴的社会趋势和突发新闻相关的模因的动态性质，它们对可恶的模因检测的概括不佳。最近的工作进一步凸显了在这种情况下，大型多模型模型的常规监督微调的局限性。为了应对这些挑战，我们提出了大型多模型检索引导的对比度学习（LMM-RGCL），这是一个新型的两阶段微调框架，旨在提高域中的准确性和交叉域的概括。六个广泛使用的模因分类数据集的实验结果表明，LMM-RGCL达到了最先进的性能，优于基于代理的系统，例如VPD-PALI-X-55B。此外，我们的方法有效地概括为低资源设置下的跨域模因，超过了诸如GPT-4O之类的模型。

### SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.13059)] [[cool](https://papers.cool/arxiv/2502.13059)] [[pdf](https://arxiv.org/pdf/2502.13059)]
> **Authors**: Xianfu Cheng,Wei Zhang,Shiwei Zhang,Jian Yang,Xiangyuan Guan,Xianjie Wu,Xiang Li,Ge Zhang,Jiaheng Liu,Yuying Mai,Yutao Zeng,Zhoufutu Wen,Ke Jin,Baorui Wang,Weixiao Zhou,Yunhong Lu,Tongliang Li,Wenhao Huang,Zhoujun Li
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: SimpleVQA：多模式大语言模型的多模式事实评估
- **领域**: 计算语言学
- **摘要**: 多模式大语言模型（MLLM）在各个领域的越来越多地介绍了其产出可靠性和准确性的本质，尤其是它们产生以事实信息为基础的内容的能力（例如，普通和特定于领域的知识）。在这项工作中，我们介绍了SimpleVQA，这是第一个评估MLLM回答自然语言简短问题的事实能力的全面多模式基准。 SimpleVQA的特征是六个关键特征：它涵盖了多个任务和多种情况，确保高质量和具有挑战性的查询，保持静态和永恒的参考答案，并且可以简单地进行评估。我们的方法涉及将视觉提问的项目分类为围绕客观事件或常识的9个不同任务，并将其置于9个主题中。实施严格的质量控制过程以确保高质量，简洁和清晰的答案，从而通过LLM-AS-A-A-A-Gudge评分系统促进评估，从而促进评估。使用SimpleVQA，我们对领先的18个MLLM和8个仅文本LLM进行全面评估，通过识别和分析错误情况来深入研究其图像理解和文本生成能力。

### AEIA-MN: Evaluating the Robustness of Multimodal LLM-Powered Mobile Agents Against Active Environmental Injection Attacks 
[[arxiv](https://arxiv.org/abs/2502.13053)] [[cool](https://papers.cool/arxiv/2502.13053)] [[pdf](https://arxiv.org/pdf/2502.13053)]
> **Authors**: Yurun Chen,Xueyu Hu,Keting Yin,Juncheng Li,Shengyu Zhang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: AEIA-MN：评估多模式LLM动力移动剂的鲁棒性针对主动环境注射攻击
- **领域**: 计算语言学
- **摘要**: 随着研究人员不断优化AI代理以在操作系统中更有效地执行任务时，他们经常忽略以解决使这些代理能够在系统中识别“冒名顶替者”的关键需求。通过对代理商的操作环境的分析，我们确定了潜在的威胁：攻击者可以将其攻击方法掩盖为环境因素，将主动干扰注入代理商的执行过程，从而破坏他们的决策。我们将这种类型的攻击定义为主动环境注入攻击（AEIA）。基于此，我们提出了AEIA-MN，这是一种主动环境注射攻击方案，利用移动操作系统中的相互作用漏洞来评估基于MLLM的代理商对此类威胁的鲁棒性。实验结果表明，即使是高级MLLM也很容易受到此攻击的影响，在Androidworld基准中，最大攻击成功率为93％。

### MVL-SIB: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching 
[[arxiv](https://arxiv.org/abs/2502.12852)] [[cool](https://papers.cool/arxiv/2502.12852)] [[pdf](https://arxiv.org/pdf/2502.12852)]
> **Authors**: Fabian David Schmidt,Florian Schneider,Chris Biemann,Goran Glavaš
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: MVL-SIB：跨模式局部匹配的大量多语言视觉语言基准
- **领域**: 计算语言学
- **摘要**: 现有的多语言视觉语言（VL）基准通常仅涵盖少数语言。因此，对大型视力语言模型（LVLM）的评估主要针对高资源语言，强调了低资源语言评估数据的需求。为了解决这一限制，我们介绍了MVL-SIB，这是一种大量多种语言视觉语言基准，可评估205种语言的跨模式和仅文本主题匹配 - 比最多语言现有的VL基准多100多种。然后，我们与MVL-SIB上的GPT-4O（-MINI）一起对一系列开放量LVLM进行了基准测试。我们的结果表明，LVLM在跨模式的主题中挣扎的较低的资源语言匹配，在N'Koo之类的语言上表现不佳。我们的分析进一步表明，LVLMS中的VL支持相对于对低资源语言的文本支持而言，VL的支持不成比例，这可以通过比较跨模式和仅文本主题匹配性能的比较来证明。我们进一步观察到，开放权重的LVLM并不能从代表一个以上图像的主题中受益，这表明这些模型尚未完全有效地处理多图像任务。通过将MVL-SIB的性能与其他多语言VL基准相关联，我们强调，MVL-SIB是对LVLM中多语言VL理解的全面探针。

### An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation 
[[arxiv](https://arxiv.org/abs/2502.12836)] [[cool](https://papers.cool/arxiv/2502.12836)] [[pdf](https://arxiv.org/pdf/2502.12836)]
> **Authors**: Mohammad Feli,Iman Azimi,Pasi Liljeberg,Amir M. Rahmani
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 用于生理数据分析的LLM驱动剂：基于PPG的心率估计的案例研究
- **领域**: 计算语言学
- **摘要**: 大型语言模型（LLM）通过通过互动沟通改善诊断，患者护理和决策支持来彻底改变医疗保健。最近，它们已用于分析生理时间序列，例如可穿戴数据，以进行健康洞察提取。现有方法将原始数值序列直接嵌入到提示中，从而超过令牌限制并增加了计算成本。此外，一些研究集成了从文本提示或应用多模式方法中从时间序列提取的集成特征。但是，由于LLMS在解释连续波形时的分析严格和效率低下，这些方法通常会产生通用和不可靠的输出。在本文中，我们开发了一种以LLM驱动的代理，用于生理时间序列分析，旨在弥合LLM与公认的分析工具的差距。我们的代理商建立在开源LLM驱动的框架的OpenCha上，它具有整合用户互动，数据源和分析工具的编排，以生成准确的健康见解。为了评估其有效性，我们在远程健康监测研究中使用PPG和心电图（ECG）记录的数据集实施了对光摄取图（PPG）信号的心率（HR）估计的案例研究。该代理商的性能是针对OpenAI GPT-4O-Mini和GPT-4O的基准测试的，ECG是人力资源估计的黄金标准。结果表明，我们的代理通过达到较低的错误率和更可靠的人力资源估计来显着优于基准模型。代理实施在GitHub上公开可用。

### Towards Text-Image Interleaved Retrieval 
[[arxiv](https://arxiv.org/abs/2502.12799)] [[cool](https://papers.cool/arxiv/2502.12799)] [[pdf](https://arxiv.org/pdf/2502.12799)]
> **Authors**: Xin Zhang,Ziqi Dai,Yongqi Li,Yanzhao Zhang,Dingkun Long,Pengjun Xie,Meishan Zhang,Jun Yu,Wenjie Li,Min Zhang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: 16 pages, 14 figures
- **标题**: 迈向文本图像交错的检索
- **领域**: 计算语言学,计算机视觉和模式识别,信息检索
- **摘要**: 当前的多模式信息检索研究主要集中于单图像输入，该输入限制了涉及多个图像和文本图像交织内容的现实世界应用。在这项工作中，我们介绍了文本图像交织的检索（TIIR）任务，其中查询和文档是交织的文本图像序列，并且需要模型才能从交织的上下文中理解语义以进行有效检索。我们基于自然交织的Wikihow教程构建了TIIR基准测试，其中特定管道旨在生成交织的查询。为了探索任务，我们适应了几个现成的检索器，并通过交织的多模式大语言模型（MLLM）建立一个密集的基线。然后，我们提出了一种新型的Matryoshka多模式嵌入器（MME），该嵌入者压缩了不同粒度的视觉令牌的数量，以解决基于MLLM的TIIR模型中过度的视觉令牌的挑战。实验表明，现有模型的简单适应不会始终产生有效的结果。我们的MME通过视觉令牌较少而实现了基线的显着改善。我们提供广泛的分析，并将发布数据集和代码以促进未来的研究。

### Mind the Gap: Aligning the Brain with Language Models Requires a Nonlinear and Multimodal Approach 
[[arxiv](https://arxiv.org/abs/2502.12771)] [[cool](https://papers.cool/arxiv/2502.12771)] [[pdf](https://arxiv.org/pdf/2502.12771)]
> **Authors**: Danny Dongyeop Han,Yunju Cho,Jiook Cha,Jay-Yoon Lee
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 注意差距：将大脑与语言模型保持一致需要一种非线性和多模式的方法
- **领域**: 计算语言学,神经元和认知
- **摘要**: 自我监督的语言和音频模型有效地预测了大脑对语音的反应。然而，尽管在语音理解过程中，听觉信号与语言和语义信息跨越了语言和语义信息，但传统的预测模型依赖于单峰特征的线性映射。在这里，我们介绍了一个非线性的多模式预测模型，该模型结合了预训练模型的音频和语言特征（例如Llama，Whisper）。我们的方法比传统的单峰线性模型提高了17.2％和17.9％的预测性能（未归一化和归一化相关性），分别比先前的先前先进模型提高了7.7％和14.4％。这些改进是朝着未来强大的内部测试和改善解码性能的重大步骤。他们还揭示了听觉和语义信息如何融合在运动，体感和高级语义区域，与现有的神经语言理论保持一致。总体而言，我们的工作突出了非线性和多模式的大脑建模方法通常被忽视的潜力，这为将来的研究铺平了道路，以将这些策略纳入自然主义的神经语言学研究中。

### Label Drop for Multi-Aspect Relation Modeling in Universal Information Extraction 
[[arxiv](https://arxiv.org/abs/2502.12614)] [[cool](https://papers.cool/arxiv/2502.12614)] [[pdf](https://arxiv.org/pdf/2502.12614)]
> **Authors**: Lu Yang,Jiajia Li,En Ci,Lefei Zhang,Zuchao Li,Ping Wang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Accepted to NAACL-main 2025
- **标题**: 通用信息提取中的多光值关系建模的标签下降
- **领域**: 计算语言学,人工智能
- **摘要**: 通用信息提取（UIE）由于能够有效解决模型爆炸问题的能力而引起了极大的关注。提取的UIE可以使用相对较小的模型实现强大的性能，从而广泛采用。提取的UIE通常依赖于不同任务的任务说明，包括单目标说明和多目标指令。单目标指令UIE可以一次提取一种类型的关系，从而限制了其建模关系之间相关性的能力，从而限制了其提取复杂关系的能力。尽管多目标指令UIE允许同时提取多个关系，但包含无关的关系会引入决策复杂性并影响提取准确性。因此，对于多关系提取，我们提出了LDNET，该LDNET结合了多光值关系建模和标签下降机制。通过将不同的关系分配给不同层次以理解和决策，我们减少了决策困惑。此外，标签滴机制有效地减轻了无关关系的影响。实验表明，LDNET在9个任务，33个数据集中，在单模式和多模式，很少弹药和零弹药设置中，在9个任务，33个数据集上优于最先进的系统的竞争性能。\ footNote {

### SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings 
[[arxiv](https://arxiv.org/abs/2502.12562)] [[cool](https://papers.cool/arxiv/2502.12562)] [[pdf](https://arxiv.org/pdf/2502.12562)]
> **Authors**: Weikai Lu,Hao Peng,Huiping Zhuang,Cen Chen,Ziqian Zeng
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 海：多模式大语言模型通过合成嵌入的低资源安全对准
- **领域**: 计算语言学,密码学和安全,多媒体
- **摘要**: 多模式大语言模型（MLLM）具有严重的安全性漏洞。虽然使用由文本和其他模式的数据组成的多模式数据集可以有效地增强MLLM的安全性，但构造这些数据集是昂贵的。现有的低资源安全对准方法（包括文本一致性）已被发现与其他模式相比的安全风险挣扎。为了解决这个问题，我们提出了合成嵌入增强安全对齐（SEA）的嵌入，该渐变更新优化了其他模态的嵌入，以扩展文本数据集。即使只有文本数据，这也可以实现多模式的安全对准训练。基于图像，视频和基于音频的MLLM的广泛实验表明，SEA可以在24秒内合成单个RTX3090 GPU上的高质量嵌入。当面临其他方式的威胁时，SEA可显着提高MLLM的安全性。为了评估视频和音频引入的安全风险，我们还引入了一个名为VA-SafetyBench的新基准。多个MLLM的高攻击成功率验证了其挑战。我们的代码和数据将在https://github.com/zeronlp/sea上找到。

### Latent Distribution Decoupling: A Probabilistic Framework for Uncertainty-Aware Multimodal Emotion Recognition 
[[arxiv](https://arxiv.org/abs/2502.13954)] [[cool](https://papers.cool/arxiv/2502.13954)] [[pdf](https://arxiv.org/pdf/2502.13954)]
> **Authors**: Jingwang Huang,Jiang Zhong,Qin Lei,Jinpeng Gao,Yuming Yang,Sirui Wang,Peiguang Li,Kaiwen Wei
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 潜在分布解耦：一种不确定性感知多模式识别的概率框架
- **领域**: 计算语言学,机器学习
- **摘要**: 多模式多标签情绪识别（MMER）旨在确定多模式数据中多种情绪的同时存在。现有研究主要集中于改善融合策略和建模形式与标签依赖性。但是，他们经常忽略\ textbf {areatoric不确定性}的影响，这是多模式数据中固有的噪声，并通过将歧义引入特征表示形式来阻碍模态融合的有效性。为了解决这个问题并有效地模拟了不确定性，本文提出了潜在的情绪分布分解，并从不确定性感知（LDDU）框架中提出了潜在情感空间概率建模的新角度。具体而言，我们在情绪空间内引入了一种对比度脱节的分布机制，以建模多模式数据，从而提取语义特征和不确定性。此外，我们设计了一种不确定性感知的融合多模式方法，该方法解释了不确定性的分布并整合了分布信息。实验结果表明，LDDU在CMU-Mosei和M $^3 $ ED数据集上实现了最先进的性能，从而强调了MMER中不确定性建模的重要性。代码可从https://github.com/201983290498/lddu \_mmer.git获得。

### Beyond Single Frames: Can LMMs Comprehend Temporal and Contextual Narratives in Image Sequences? 
[[arxiv](https://arxiv.org/abs/2502.13925)] [[cool](https://papers.cool/arxiv/2502.13925)] [[pdf](https://arxiv.org/pdf/2502.13925)]
> **Authors**: Xiaochen Wang,Heming Xia,Jialin Song,Longyu Guan,Yixin Yang,Qingxiu Dong,Weiyao Luo,Yifan Pu,Yiru Wang,Xiangdi Meng,Wenjie Li,Zhifang Sui
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 除了单帧：LMM可以理解图像序列中的时间和上下文叙事吗？
- **领域**: 计算语言学
- **摘要**: 大型多模型模型（LMM）在各种视觉语言任务中取得了巨大的成功。但是，现有的基准主要集中在单像理解上，而对图像序列的分析很大程度上没有探索。为了解决此限制，我们引入了Stripcipher，这是一种综合基准，旨在评估LMM的能力，以理解和推理顺序图像。带状卷手包括一个人类通知的数据集和三个具有挑战性的子任务：视觉叙事理解，上下文框架预测和时间叙事重新排序。我们对包括GPT-4O和QWEN2.5VL在内的最先进的LMM的$ 16 $评估显示，与人类能力相比，性能差距有显着的性能差距，尤其是在需要重新排序洗牌的顺序图像的任务中。例如，GPT-4O在重新排序子任务中仅达到23.93％的精度，比人类绩效低56.07％。进一步的定量分析讨论了几个因素，例如图像的输入格式，影响了LLM在顺序理解中的性能，从而强调了LMMS开发中仍然存在的基本挑战。

### GIMMICK -- Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking 
[[arxiv](https://arxiv.org/abs/2502.13766)] [[cool](https://papers.cool/arxiv/2502.13766)] [[pdf](https://arxiv.org/pdf/2502.13766)]
> **Authors**: Florian Schneider,Carolin Holtermann,Chris Biemann,Anne Lauscher
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 头 - 全球包含多模式的多模式文化知识基准测试
- **领域**: 计算语言学
- **摘要**: 大型视觉模型（LVLM）由于其独特的性能和广泛的适用性而引起了人们的关注。虽然以前已经证明，它们在涉及非西方环境的用法场景中的功效短缺，但现有研究的范围有限，仅涵盖了狭窄的文化范围，仅专注于少数文化方面，或者仅评估单个任务上有限选择的模型。为了全球包含的LVLM研究，我们介绍了Gimmick，这是一种广泛的多模式基准，旨在评估144个代表六个全球宏观区域的144个国家 /地区的广泛文化知识。 Gimmick包括六个任务，构建了三个新的数据集，它们跨越了728个独特的文化事件或我们评估了20个LVLM和11个LLM的唯一文化事件或方面，其中包括5个专有和26个所有尺寸的开放权重模型。我们系统地检查（1）区域文化偏见，（2）模型大小，（3）输入方式和（4）外部提示的影响。我们的分析揭示了跨模型和任务对西方文化的强烈偏见，并突出了模型大小和性能之间的强烈相关性，以及多模式输入和外部地理线索的有效性。我们进一步发现，模型比无形方面（例如，食物与仪式）具有更多的有形知识，并且它们在认识到广泛的文化渊源方面表现出色，但以更加细微的理解挣扎。

### Unlocking Multimodal Integration in EHRs: A Prompt Learning Framework for Language and Time Series Fusion 
[[arxiv](https://arxiv.org/abs/2502.13509)] [[cool](https://papers.cool/arxiv/2502.13509)] [[pdf](https://arxiv.org/pdf/2502.13509)]
> **Authors**: Shuai Niu,Jing Ma,Hongzhan Lin,Liang Bai,Zhihua Wang,Wei Bi,Yida Xu,Guo Li,Xian Yang
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 13 pages, 5 figures
- **标题**: 在EHR中解锁多模式集成：语言和时间序列融合的及时学习框架
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 大型语言模型（LLM）在视觉任务中表现出色，但是它们在医学领域的应用仍未得到充满反感，尤其是将结构化时间序列数据与非结构化的临床注释集成在一起。在临床实践中，动态时间序列数据（例如实验室测试结果）捕获了关键的时间模式，而临床注释则提供了丰富的语义环境。由于连续信号和离散文本之间的固有差异，合并这些方式是具有挑战性的。为了弥合这一差距，我们介绍了Promedts，这是一种新型的自我监管的多模式框架，采用迅速引入的学习来统一这些异质数据类型。我们的方法利用轻巧的异常检测来生成用作提示的异常字幕，将原始时间序列数据的编码引导到信息性嵌入中。这些嵌入与共享潜在空间中的文本表示相一致，并在语义洞察力的同时保留细粒度的时间细微差别。此外，我们的框架结合了量身定制的自我监督目标，以增强内部和模式间对准。我们使用现实世界数据集评估了有关疾病诊断任务的Promedt，结果表明，我们的方法始终优于最先进的方法。

### Transferring Textual Preferences to Vision-Language Understanding through Model Merging 
[[arxiv](https://arxiv.org/abs/2502.13487)] [[cool](https://papers.cool/arxiv/2502.13487)] [[pdf](https://arxiv.org/pdf/2502.13487)]
> **Authors**: Chen-An Li,Tzu-Han Lin,Yun-Nung Chen,Hung-yi Lee
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: Preprint. Under Review
- **标题**: 通过模型合并将文本偏好转移到视觉理解中
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 大型视觉模型（LVLM）在各种多模式任务中表现出色。但是，他们评估生成内容的能力仍然有限，并且具有偏好数据的培训视觉奖励模型（VLRMS）在计算上很昂贵。本文通过将基于文本的奖励模型（RMS）与LVLMS合并以创建VLRMS来探讨无培训替代方案。我们的方法表明，整合这些模型会导致LVLMS的评分和基于文本的RMS的性能提高，从而提供了将文本偏好纳入LVLM的有效方法。

### Social Genome: Grounded Social Reasoning Abilities of Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.15109)] [[cool](https://papers.cool/arxiv/2502.15109)] [[pdf](https://arxiv.org/pdf/2502.15109)]
> **Authors**: Leena Mathur,Marian Qian,Paul Pu Liang,Louis-Philippe Morency
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: Under Review, 22 pages
- **标题**: 社会基因组：多模型的基础社会推理能力
- **领域**: 计算语言学,机器学习
- **摘要**: 社会推理能力对于AI系统有效地解释和应对社会环境中多模式的交流和互动至关重要。我们介绍了社会基因组，这是多模型模型的细粒度，基础社会推理能力的第一个基准。社会基因组包含272个与这些相互作用的推论有关的相互作用和1,486个人类宣传的推理痕迹。这些痕迹包含5,777个推理步骤，可参考视觉提示，口头提示，人声线索和外部知识（视频外部知识）的证据。社会基因组也是研究社会推理中外部知识的第一个建模挑战。社会基因组计算指标以整体评估模型生成的社会推理痕迹的语义和结构质量。我们通过实验最先进的模型来证明社会基因组的实用性，确定了绩效差距和未来研究的机会，以提高多模型模型的基础社会推理能力。

### Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation 
[[arxiv](https://arxiv.org/abs/2502.15040)] [[cool](https://papers.cool/arxiv/2502.15040)] [[pdf](https://arxiv.org/pdf/2502.15040)]
> **Authors**: Yun-Wei Chu,Kai Zhang,Christopher Malon,Martin Renqiang Min
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: GenAI4Health - AAAI '25
- **标题**: 通过视觉检索发达的一代减少医学多模式大型语言模型的幻觉
- **领域**: 计算语言学,人工智能
- **摘要**: 多模式的大语言模型（MLLM）在视觉和文本任务中表现出令人印象深刻的表现。但是，幻觉仍然是一个重大挑战，尤其是在细节至关重要的医疗保健领域。在这项工作中，我们展示了如何增强MLLM来支持Visual Rag（V-RAG），这是一种检索型生成框架，该框架合并了来自检索的图像的文本和视觉数据。在模拟CXR胸部X射线报告生成和多层医疗图像字幕生成数据集上，我们表明视觉抹布提高了实体探测的准确性，这询问医疗实体是否以图像为基础。我们表明，这些改进既扩展到了频繁和稀有实体，后者的训练数据可能较少。在下游，我们将V-rag与实体探测一起应用以纠正幻觉并产生更临床准确的X射线报告，从而获得较高的radgraph-f1分数。

### InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback 
[[arxiv](https://arxiv.org/abs/2502.15027)] [[cool](https://papers.cool/arxiv/2502.15027)] [[pdf](https://arxiv.org/pdf/2502.15027)]
> **Authors**: Henry Hengyuan Zhao,Wenqi Pei,Yifei Tao,Haiyang Mei,Mike Zheng Shou
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 18 pages, 10 figures
- **标题**: 反馈：通过人类反馈揭示大型多模型模型的互动智能
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别,人机交互
- **摘要**: 现有的基准测试不会在与人类用户的互动智能上测试大型多模型模型（LMM），这对于开发通用AI助手至关重要。我们设计了一个交互式框架，可以将其应用于任何LMM和数据集以自主评估该功能。最重要的是，我们介绍了Interfeedback基础台，该基础台上使用两个代表性数据集（MMMU-PRO和MATHVERSE）评估交互式智能，以测试10种不同的开源LMM。此外，我们提出了Interfectback-Human，这是一个新收集的数据集，该数据集的120个病例，旨在在诸如OpenAI-O1和Claude-3.5-Sonnet等领先模型中手动测试交互式性能。我们的评估结果表明，即使是最先进的LMM OpenAI-O1，也努力根据人类反馈来完善其反应，平均得分少于50％。我们的发现表明，需要增强LMMS解释和受益于反馈的能力的方法。

### Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.14906)] [[cool](https://papers.cool/arxiv/2502.14906)] [[pdf](https://arxiv.org/pdf/2502.14906)]
> **Authors**: Srishti Yadav,Zhi Zhang,Daniel Hershcovich,Ekaterina Shutova
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-21
> **comment**: ef:NAACL 2025
- **标题**: 超越言语：探索多模型模型中的文化价值敏感性
- **领域**: 计算语言学,人工智能
- **摘要**: 基于文化背景的大语言模型（LLM）中的价值一致性已成为关键的研究领域。但是，在大型视觉模型（VLM）中尚未广泛探索类似的偏见。随着多模式模型的规模不断增长，评估图像是否可以作为文化的可靠代理以及如何通过整合视觉和文本数据来嵌入这些值变得越来越重要。在本文中，我们对不同尺度的多模式模型进行了彻底的评估，重点是它们与文化价值的一致性。我们的发现表明，与LLM一样，VLM对文化价值表现出敏感性，但是它们与这些价值一致性的性能高度依赖于上下文。尽管VLM通过使用图像显示了提高价值理解的潜力，但在多种模型对齐中，这种对齐方式在强调复杂性和毫无创伤的挑战的情况下差异很大。

### ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting 
[[arxiv](https://arxiv.org/abs/2502.14780)] [[cool](https://papers.cool/arxiv/2502.14780)] [[pdf](https://arxiv.org/pdf/2502.14780)]
> **Authors**: Abhijit Mishra,Richard Noh,Hsiang Fu,Mingda Li,Minji Kim
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 12 pages, 7 figures, 3 tables
- **标题**: 修订：用于保护任务的视觉指令的数据集和基线VLM重写
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 高效和保护隐私的多模式互动至关重要，因为AR，VR和具有强大相机的现代智能手机成为人类计算机通信的主要接口。现有强大的大型视觉模型（VLM）实现多模式互动通常依赖于基于云的处理，从而通过将敏感视觉数据传输到服务器以及（2）其实时的实时，磁性可用性来引起（1）视觉隐私。本文探讨了视觉指令重写，这是一种新颖的方法，将多模式指令转换为仅文本命令，从而使轻质的放在设备指令重写器VLMS（250m参数）与现有对话AI系统的无缝集成，并增强视觉数据隐私。为了实现这一目标，我们介绍了14个域中39,000多个示例的数据集，并开发了一个紧凑的VLM，该数据集在图像字幕的数据集中预定，并进行了微调以重写。通过NLG指标进行评估的实验结果，例如BLEU，Meteor和Rouge，以及语义解析分析，表明即使是量化的模型（<500MB存储足迹），甚至可以实现有效的指令重写，从而实现了以隐私为中心的多载AI应用程序。

### Harnessing PDF Data for Improving Japanese Large Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.14778)] [[cool](https://papers.cool/arxiv/2502.14778)] [[pdf](https://arxiv.org/pdf/2502.14778)]
> **Authors**: Jeonghun Baek,Akiko Aizawa,Kiyoharu Aizawa
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 15 pages, 8 figures
- **标题**: 利用PDF数据来改善日本大型多式联运模型
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 大型多模型模型（LMM）在英语方面表现出很强的性能，但是由于缺乏高质量的培训数据，它们在日语中的有效性仍然有限。当前的日本LMM通常依靠翻译的英语数据集，从而限制了它们捕获日本特定文化知识的能力。为了解决这个问题，我们探讨了日本PDF数据作为培训资源的潜力，该领域在很大程度上未被充分利用。我们引入了一条全自动的管道，该管道利用预处理的模型通过布局分析，OCR和视觉 - 语言配对从PDF中提取图像文本对，从而消除了手动注释的需求。此外，我们从提取的图像文本对构建指令数据以丰富培训数据。为了评估PDF衍生数据的有效性，我们培训日本LMM并评估其在日本LMM基准上的性能。我们的结果表明，苍鹭板凳的性能增长范围从3.9％到13.8％。进一步的分析强调了PDF衍生数据对各种因素（例如模型大小和语言模型）的影响，从而增强了其作为日本LMM的多模式资源的价值。我们计划在接受后公开提供源代码和数据。

### HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States 
[[arxiv](https://arxiv.org/abs/2502.14744)] [[cool](https://papers.cool/arxiv/2502.14744)] [[pdf](https://arxiv.org/pdf/2502.14744)]
> **Authors**: Yilei Jiang,Xinyan Gao,Tianshuo Peng,Yingshui Tan,Xiaoyong Zhu,Bo Zheng,Xiangyu Yue
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: HIDDENDETECT：通过监视隐藏状态，检测针对大型视力语言模型的越狱攻击
- **领域**: 计算语言学
- **摘要**: 与仅语言的同行相比，其他方式的整合增加了大型视力语言模型（LVLM）对安全风险（例如越狱攻击）的敏感性。尽管现有的研究主要集中在事后对准技术上，但LVLMS内的潜在安全机制在很大程度上尚未探索。在这项工作中，我们调查了LVLM在推断过程中内部激活中固有地编码与安全相关的信号。我们的发现表明，在处理不安全的提示时，LVLM会表现出不同的激活模式，可以利用这些激活模式来检测和减轻对抗性输入而无需进行广泛的微调。在此洞察力的基础上，我们引入了HiddenDetect，这是一个新颖的无调框架，可利用内部模型激活以增强安全性。实验结果表明，{HiddenDetect}在检测对LVLMS的越狱攻击方面超过了最新方法。通过利用固有的安全感知模式，我们的方法提供了一种有效且可扩展的解决方案，可增强对多模式威胁的LVLM稳健性。我们的代码将在https://github.com/leigest519/hiddendetect公开发布。

### Multiscale Byte Language Models -- A Hierarchical Architecture for Causal Million-Length Sequence Modeling 
[[arxiv](https://arxiv.org/abs/2502.14553)] [[cool](https://papers.cool/arxiv/2502.14553)] [[pdf](https://arxiv.org/pdf/2502.14553)]
> **Authors**: Eric Egli,Matteo Manica,Jannis Born
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: Under Review
- **标题**: 多尺度字体语言模型 - 因果关系序列建模的层次结构
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 字节构成了数字世界的基础，因此是多模式基础模型的有希望的基础。最近，字节语言模型（BLM）已经出现以克服令牌化，但是过度长度的副流需要新的建筑范式。因此，我们提出了多尺度字节语言模型（MBLM），这是一种模型不合时宜的层次解码器堆栈，允许在单个GPU上以$ 5 $ M字节的上下文窗口进行完整模型精度的培训。我们在单峰和多模式任务上彻底检查了MBLM在变压器和MAMBA块中的性能。我们的实验表明，混合体系结构在训练过程中处理极长的字节序列有效，同时达到了接近线性的世代效率。据我们所知，我们在Visual Q \＆A任务上进行了首次评估BLM的评估，并发现尽管序列化图像和没有编码器，但具有纯粹的下一象征预测的MBLM可以匹配定制的CNN-LSTM架构，并指定分类头。我们表明，MBLM在整合包括像素和图像文件的多种数据表示方面表现出强大的适应性，这强调了它们对综合基础模型的潜力。源代码可公开可用：https：//github.com/ai4sd/multiscale-byte-lm

### Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach 
[[arxiv](https://arxiv.org/abs/2502.14285)] [[cool](https://papers.cool/arxiv/2502.14285)] [[pdf](https://arxiv.org/pdf/2502.14285)]
> **Authors**: Yurong Wu,Fangwen Mu,Qiuhong Zhang,Jinjing Zhao,Xinrun Xu,Lingrui Mei,Yang Wu,Lin Shi,Junjie Wang,Zhiming Ding,Yiwei Wang
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 14 pages,8 figures,4 tables
- **标题**: 文本到图像模型促使模板窃取的脆弱性：差分进化方法
- **领域**: 计算语言学
- **摘要**: 近年来，迅速交易已成为一个重要的知识产权问题，供应商在出售可以生成类似图像的及时模板之前通过展示样本图像来吸引用户。这项工作调查了一个关键的安全漏洞：攻击者只能使用有限数量的示例图像窃取及时的模板。为了调查这种威胁，我们介绍了Prism，这是一个迅速偷走的基准测试，该基准由50个模板和450张图像组成，分为简单而困难的水平。为了确定VLM的漏洞提示窃取，我们提出了EvoStealer，这是一种新型的模板窃取方法，该方法通过利用差异进化算法而无需模型进行微调。该系统首先使用基于预定义模式的多模式大语言模型（MLLM）初始化人口集，然后迭代通过MLLM生成增强的后代。在进化过程中，EvoStealer识别了跨后代的共同特征，以推导广义模板。我们跨开源（Intervl2-26b）和封闭源模型（GPT-4O和GPT-4O-MINI）进行的全面评估表明，Evostealer的被盗模板可以复制高度相似的图像，并有效地概括了其他受试者，并显着概括了基线的基线方法，其平均改善超过了10％。此外，我们的成本分析表明，Evostealer通过可忽略的计算费用实现了模板窃取。我们的代码和数据集可从https://github.com/whitepagewu/evostealer获得。

### Are Large Language Models Good Data Preprocessors? 
[[arxiv](https://arxiv.org/abs/2502.16790)] [[cool](https://papers.cool/arxiv/2502.16790)] [[pdf](https://arxiv.org/pdf/2502.16790)]
> **Authors**: Elyas Meguellati,Nardiena Pratama,Shazia Sadiq,Gianluca Demartini
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 大型语言模型是良好的数据预处理器吗？
- **领域**: 计算语言学
- **摘要**: 高质量的文本培训数据对于多模式数据处理任务的成功至关重要，但是来自BLIP和GIT（例如GIT）的图像字幕模型的输出通常包含错误和异常，这些错误和异常很难使用基于规则的方法进行纠正。尽管最近解决此问题的工作主要集中在使用GPT模型对相对简单的公共数据集上进行数据预处理，但有必要探索更广泛的大型语言模型（LLMS），并解决更具挑战性和更具挑战性的数据集。在这项研究中，我们研究了包括Llama 3.1 70B，GPT-4 Turbo和Sonnet 3.5 V2在内的多个LLM的使用，以完善和清洁Blip和Git的文本输出。我们通过比较下游任务（Semeval 2024子任务“模因中的多标记说服检测”）模型来评估LLM辅助数据清洁的影响。虽然我们的实验结果显示使用LLM清洗标题时的改进，但统计测试表明，这些改进大多数并不显着。这表明，尽管LLM有可能增强数据清洁和维修，但它们的有效性可能受到限制，具体取决于其所应用的上下文，任务的复杂性以及文本中的噪声水平。我们的发现强调了需要进一步研究LLM在数据预处理管道中的功能和局限性的需求，尤其是在处理挑战性数据集时，为将LLMS整合到数据预处理管道中的持续讨论中提供了经验证据。

### Beyond Pattern Recognition: Probing Mental Representations of LMs 
[[arxiv](https://arxiv.org/abs/2502.16717)] [[cool](https://papers.cool/arxiv/2502.16717)] [[pdf](https://arxiv.org/pdf/2502.16717)]
> **Authors**: Moritz Miller,Kumar Shridhar
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 超越模式识别：探测LMS的心理表征
- **领域**: 计算语言学
- **摘要**: 语言模型（LMS）在解决复杂的推理任务方面表现出了令人印象深刻的能力，尤其是在提示产生中间解释时。但是，这些中间推理痕迹是一个动态，不断发展的思维过程还是仅反映在大规模预训练中获得的复杂模式识别，仍然是一个空的问题。从人类认知中汲取灵感，因为新信息被吸收并不断更新，因此推理会逐步发展，我们建议深入研究各种LMS的心理模型。我们提出了一种评估LMS心理建模的新方法，在该方法中逐渐为它们提供了问题细节，从而允许每个新数据构建并完善模型对任务的内部表示。我们从系统地将此逐步的心理建模策略与仅文本以及视觉和文本方式的传统完整及时方法进行比较。在不同模型大小和问题复杂性的数学数据集上进行的实验证实，基于文本的LLM和多模式LMS都在努力创建心理表示，质疑其内部认知过程如何工作。

### Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries 
[[arxiv](https://arxiv.org/abs/2502.16636)] [[cool](https://papers.cool/arxiv/2502.16636)] [[pdf](https://arxiv.org/pdf/2502.16636)]
> **Authors**: Yin Wu,Quanyu Long,Jing Li,Jianfei Yu,Wenya Wang
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: 23 pages, 6 figures
- **标题**: 视觉窗口：基准对文本对图像检索的增强生成，以进行视觉知识密集查询
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 检索授权的一代（RAG）是一种通过解决事实并回答知识密集性问题的局限性来增强大语模型（LLM）的流行方法。随着LLM的研究扩展了处理文本以外的输入方式的能力，例如图像，提出了几个多模式的抹布基准。尽管如此，他们主要将文本知识基础作为增强证据的主要来源。仍然缺乏旨在评估图像作为抹布系统中的增强图像以及它们如何利用视觉知识的基准测试。我们提出了Visual-Rag，这是一个回答基准的新颖问题，强调视觉知识密集型问题。与依靠基于文本的证据的先前作品不同，视觉rag需要文本对图像检索和相关线索图像的整合以提取视觉知识作为证据。借助视觉窗格，我们评估了5个开源和3个专有的多模式LLM（MLLM），揭示了图像可以作为抹布中的良好证据。但是，即使SOTA模型也在有效提取和利用视觉知识方面努力

### MemeIntel: Explainable Detection of Propagandistic and Hateful Memes 
[[arxiv](https://arxiv.org/abs/2502.16612)] [[cool](https://papers.cool/arxiv/2502.16612)] [[pdf](https://arxiv.org/pdf/2502.16612)]
> **Authors**: Mohamed Bayan Kmainasi,Abul Hasnat,Md Arid Hasan,Ali Ezzat Shahroor,Firoj Alam
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: disinformation, misinformation, factuality, harmfulness, fake news, propaganda, hateful meme, multimodality, text, images
- **标题**: Memeintel：可解释的宣传和仇恨模因的检测
- **领域**: 计算语言学,人工智能
- **摘要**: 社交媒体上多模式内容的扩散在理解和调节复杂的，依赖上下文的问题（例如误导，仇恨言论和宣传）方面提出了重大挑战。尽管已经努力开发资源并提出了自动检测的新方法，但对标签检测和预测标签的基于解释的理由的产生有限。为了应对这一挑战，我们介绍了Memeintel，这是一个用英语的阿拉伯语和仇恨模因宣传模因的解释增强数据集，这使其成为这些任务的第一个大规模资源。为了解决这些任务，我们提出了一种多阶段优化方法和火车视觉模型（VLMS）。我们的结果表明，这种方法可显着提高\ textbf {label检测}和解释产生的基本模型的性能，从而优于当前的最新技术，而在Armeme的绝对提高约为3％，而仇恨模因却〜7％。对于可重复性和未来的研究，我们旨在使Memeintel数据集和实验资源公开可用。

### Contrastive Learning of English Language and Crystal Graphs for Multimodal Representation of Materials Knowledge 
[[arxiv](https://arxiv.org/abs/2502.16451)] [[cool](https://papers.cool/arxiv/2502.16451)] [[pdf](https://arxiv.org/pdf/2502.16451)]
> **Authors**: Yang Jeong Park,Mayank Kumaran,Chia-Wei Hsu,Elsa Olivetti,Ju Li
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: 24 pages, 14 figure
- **标题**: 对材料知识多模式表示的英语和水晶图的对比度学习
- **领域**: 计算语言学
- **摘要**: 人工智能（AI）越来越多地用于材料的反设计，例如晶体和分子。现有的对分子的AI研究具有分子的化学结构，并具有文本知识以适应复杂的指示。但是，由于研究晶体的有偏分布以及在同行评审的文献中缺乏语义监督的数据稀缺，这种方法是无法实现的。在这项工作中，我们在126K Crystal结构文本对的新合成数据集上介绍了对对比的语言 - 晶体模型（CLAC）。为了证明使用合成数据克服数据稀缺的优势，我们构建了从学术论文中提取的可比数据集。我们通过各种零射击跨模式任务和下游应用程序评估CLAC的概括能力。在实验中，CLAC在理解晶体结构中实现了最先进的零拍概括性能，超过了最新的大型语言模型。

### Chain-of-Description: What I can understand, I can put into words 
[[arxiv](https://arxiv.org/abs/2502.16137)] [[cool](https://papers.cool/arxiv/2502.16137)] [[pdf](https://arxiv.org/pdf/2502.16137)]
> **Authors**: Jiaxin Guo,Daimeng Wei,Zongyao Li,Hengchao Shang,Yuanchang Luo,Hao Yang
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 描述链：我能理解的是，我可以说话
- **领域**: 计算语言学,人工智能
- **摘要**: 在本文中，我们提出了一种定义为描述链（COD）提示的新型策略，该策略是针对多模式大型语言模型量身定制的。这种方法涉及让模型首先提供对多模式输入的详细描述，然后再产生答案。与标准提示方法相比，当应用于QWEN2-AUDIO，QWEN2-VL和QWEN2.5-VL等模型时，COD提示会显着提高性能。这是通过音频基准空气板聊天的语音类别的几乎4 \％改进的证明，而视觉基准mmmu \ _pro的硬级部分的5.3 \％改进。我们的消融研究进一步验证了COD提示的有效性。

### Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models 
[[arxiv](https://arxiv.org/abs/2502.16033)] [[cool](https://papers.cool/arxiv/2502.16033)] [[pdf](https://arxiv.org/pdf/2502.16033)]
> **Authors**: Qianqi Yan,Yue Fan,Hongquan Li,Shan Jiang,Yang Zhao,Xinze Guan,Ching-Chen Kuo,Xin Eric Wang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 多模式不一致推理（MMIR）：多模式推理模型的新基准
- **领域**: 计算语言学,人工智能
- **摘要**: 现有的多模式大型语言模型（MLLMS）主要是对一致的视觉文本输入进行训练和测试的，这使他们打开了是否可以处理现实世界中的不一致之处的问题。为了弥合这一差距，我们提出了多模式不一致推理（MMIR）基准测试，以评估MLLM检测和推理有关网页，演示幻灯片和海报等文物中语义不匹配的能力。 MMIR包括534个具有挑战性的样本，每个样本都包含五个重度推理类别的合成错误：事实矛盾，身份错误贡献，上下文不匹配，定量差异和时间/空间不相互关系。我们评估了六个最先进的MLLM，表明具有专用多模式推理功能的模型（例如O1）大大优于其同行，而开源模型仍然特别容易受到不一致错误的影响。详细的错误分析进一步表明，模型在检测成对不一致的情况下表现出色，但在复杂布局中仅限于单个元素的不一致而挣扎。探测实验表明，单模式的提示，包括经营链（COT）和标记（SOM）方法，可产生边缘增长，揭示了交叉模式推理中的关键瓶颈。我们的发现强调了对高级多模式推理的需求，并指出了对多模式不一致的未来研究。

### Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.15910)] [[cool](https://papers.cool/arxiv/2502.15910)] [[pdf](https://arxiv.org/pdf/2502.15910)]
> **Authors**: Zheyuan Liu,Guangyao Dou,Xiangchi Yuan,Chunhui Zhang,Zhaoxuan Tan,Meng Jiang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 19 pages
- **标题**: 在多模式大语模型中学习的模态感知神经元修剪
- **领域**: 计算语言学
- **摘要**: 在大规模数据集中训练的诸如大语言模型（LLM）和多模式大语言模型（MLLM）之类的生成模型可能会导致他们记住并无意间揭示敏感信息，从而引发道德和隐私问题。尽管一些先前的作品在LLM的背景下探讨了这个问题，但由于跨模式的知识的纠缠性，它给MLLM带来了独特的挑战，这使得全面学习更加困难。为了应对这一挑战，我们提出了模态意识到神经元学习（MANU），这是一个新颖的MLLMS框架，旨在根据其对目标忘记数据的相对重要性选择性地剪辑神经元，以策划不同的方式。具体而言，MANU由两个阶段组成：重要的神经元选择和选择性修剪。第一阶段识别并收集了相对于目标忘记知识的模态性最具影响力的神经元，而第二阶段则致力于修剪那些选定的神经元。 Manu有效地隔离并去除对每种模式中忘记数据做出最大贡献的神经元，同时保留保留知识的完整性。我们在各种MLLM体系结构进行的实验表明，Manu可以在每种模式中实现更平衡，更全面的学习，而不会在很大程度上影响整体模型实用程序。

### Evaluating Multimodal Generative AI with Korean Educational Standards 
[[arxiv](https://arxiv.org/abs/2502.15422)] [[cool](https://papers.cool/arxiv/2502.15422)] [[pdf](https://arxiv.org/pdf/2502.15422)]
> **Authors**: Sanghee Park,Geewook Kim
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 18 pages; To appear at NAACL 2025 Main Conference (Project page: https://github.com/naver-ai/KoNET )
- **标题**: 评估具有韩国教育标准的多模式生成AI
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 本文介绍了韩国国家教育测试基准（KONET），这是一种新的基准测试，旨在使用韩国国家教育测试评估多模式生成的AI系统。 KONET包括四个考试：韩国基础一般教育发展测试（KOEGED），中间（Komged），高（KOHGED）和大学学术能力测试（KOCSAT）。这些考试以其严格的标准和各种问题而闻名，从而促进了对不同教育水平的AI表现的全面分析。通过专注于韩语，Konet提供了对较少探索语言的模型性能的见解。我们通过检查困难，主题多样性和人为错误率来评估一系列模型 - 开源，开放访问和封闭的API。代码和数据集构建器将在https://github.com/naver-ai/konet上完全开源。

### Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews 
[[arxiv](https://arxiv.org/abs/2502.15226)] [[cool](https://papers.cool/arxiv/2502.15226)] [[pdf](https://arxiv.org/pdf/2502.15226)]
> **Authors**: Mengqiao Liu,Tevin Wang,Cassandra A. Cohen,Sarah Li,Chenyan Xiong
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 通过LLM驱动的内在用户体验访谈了解大语模型的用户意见
- **领域**: 计算语言学,人工智能,人机交互
- **摘要**: 哪种大型语言模型（LLM）更好？每个评估都讲述了一个故事，但是用户对当前LLM的真正看法是什么？本文介绍了线索，这是一位由LLM驱动的访调员，在用户与LLMS互动之后，可以进行瞬间的用户体验访谈，并自动从大量访谈日志中获得有关用户意见的见解。我们对数千名用户进行了一项研究，以了解主流LLM的用户意见，招募用户首先与目标LLM聊天，然后接受Clue的采访。我们的实验表明，线索捕获了有趣的用户意见，例如，对显示的DeepSeek-R1的推理过程的双相情感以及对信息新鲜度和多模式的需求。我们收集的聊天浏览日志将发布。

### Can Multimodal LLMs Perform Time Series Anomaly Detection? 
[[arxiv](https://arxiv.org/abs/2502.17812)] [[cool](https://papers.cool/arxiv/2502.17812)] [[pdf](https://arxiv.org/pdf/2502.17812)]
> **Authors**: Xiongxiao Xu,Haoran Wang,Yueqing Liang,Philip S. Yu,Yue Zhao,Kai Shu
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection ofmultimodalLLMs and time series analysis are on the website https://mllm-ts.github.io
- **标题**: 多模式LLM可以执行时间序列异常检测吗？
- **领域**: 计算语言学,机器学习
- **摘要**: 大型语言模型（LLM）已越来越多地用于时间序列分析。然而，时间序列的多模式LLM（MLLM）的潜力（尤其是视觉模型）的潜力在很大程度上仍然很少。人类检测时间序列异常的一种自然方法是通过可视化和文本描述。在此激励的情况下，我们提出了一个关键而实用的研究问题：多模式LLM可以执行时间序列异常检测吗？为了回答这一点，我们提出了VisualTimeanomaly基准测试，以评估时间序列异常检测（TSAD）中的MLLM。我们的方法将时间序列数据转换为图像格式，并将这些图像馈送到各种MLLM中，包括专有模型（GPT-4O和GEMINI-1.5）和开源模型（LLAVA-NEXT和QWEN2-VL），每个模型都具有一个较大和一个较小的变体。总体而言，VisualTimeanomaly包含12.4K时间序列图像，涵盖了3个方案和3种异常粒度，在8 mllms上具有9种异常类型。从单变量案例（点和范围异常）开始，我们将评估扩展到更实际的场景，包括多元和不规则的时间序列方案，以及各种视频异常。我们的研究揭示了几个关键的见解：1）MLLMS比点异常更有效地检测范围和变化异常。 2）MLLM非常健壮至不规则时间序列，即使缺少25％的数据。 3）开源MLLM与TSAD的专有模型相当。尽管开源MLLM在单变量时间序列上表现出色，但专有的MLLM在多变量时间序列上表现出较高的有效性。据我们所知，这是第一项全面研究TSAD的工作，特别是对于多元和不规则时间序列的情况。我们在https://github.com/mllm-ts/visaltimeanomaly上发布数据集和代码，以支持未来的研究。

### Towards Human Cognition: Visual Context Guides Syntactic Priming in Fusion-Encoded Models 
[[arxiv](https://arxiv.org/abs/2502.17669)] [[cool](https://papers.cool/arxiv/2502.17669)] [[pdf](https://arxiv.org/pdf/2502.17669)]
> **Authors**: Bushi Xiao,Michael Bennie,Jayetri Bardhan,Daisy Zhe Wang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 8 pages, 9 figures
- **标题**: 迈向人类认知：视觉上下文指导融合编码模型中的句法启动
- **领域**: 计算语言学
- **摘要**: 我们引入了Prismatic，这是第一个多模式结构启动数据集，并提出了无参考评估指标，该指标评估启动效应而没有预定义的目标句子。使用此指标，我们构建了具有不同多模式编码体系结构（双重编码器和融合编码器）的模型，以研究其结构保存能力。我们的发现表明，具有两种编码方法的模型表明了可比的句法启动效应。但是，只有融合编码的模型在启动效应和视觉相似性之间表现出强大的正相关性，这表明认知过程与人类心理语言模式更加一致。这项工作为评估和了解如何在多模式模型中处理句法信息提供了新的见解。

### MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference 
[[arxiv](https://arxiv.org/abs/2502.17599)] [[cool](https://papers.cool/arxiv/2502.17599)] [[pdf](https://arxiv.org/pdf/2502.17599)]
> **Authors**: Zhongwei Wan,Hui Shen,Xin Wang,Che Liu,Zheda Mai,Mi Zhang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: NAACL 2025 Main
- **标题**: MEDA：动态KV缓存分配，用于有效的多模式长篇小说推断
- **领域**: 计算语言学
- **摘要**: 长篇文本多模式模型（MLLM）结合了长文字图像和文本视频模式，需要大量资源，因为它们的多模式键值（KV）缓存随着输入长度的增加而增长，具有挑战性的推理效率。在纯文本和多模式LLM中，现有的KV缓存压缩方法已忽略了各个层的注意力密度变化，因此通常采用均匀或渐进的减少策略来分配层。在这项工作中，我们提出了MEDA，这是一种动态层的KV缓存分配方法，用于有效的多模式长篇下说推断。作为其核心，MEDA利用跨模式的注意熵来确定每个MLLMS层处的KV缓存尺寸。鉴于每一层动态分配的KV高速缓存大小，MEDA还采用KV对选择方案来确定要选择哪种KV对以及合并所选和非选择的KV对合并策略，以从整个上下文中保留信息。 MEDA可实现高达72％的KV缓存内存减少和2.82倍的解码速度，同时在长篇小说设置中维持或增强各种多模式任务的性能，包括多映射和长时间视频场景。我们的代码在https://github.com/aiot-mlsys-lab/meda上发布。

### Systematic Weight Evaluation for Pruning Large Language Models: Enhancing Performance and Sustainability 
[[arxiv](https://arxiv.org/abs/2502.17071)] [[cool](https://papers.cool/arxiv/2502.17071)] [[pdf](https://arxiv.org/pdf/2502.17071)]
> **Authors**: Ashhadul Islam,Samir Brahim Belhaouari,Amine Bermak
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 修剪大语言模型的系统重量评估：增强性能和可持续性
- **领域**: 计算语言学,人工智能
- **摘要**: 像Chatgpt这样的大型语言模型（LLM）的指数增长彻底改变了人工智能，提供了自然语言处理中前所未有的功能。但是，训练所需的广泛的计算资源具有重大的环境影响，包括高碳排放，能源消耗和用水。这项研究提出了一种新颖的LLM修剪方法，重点是在整个培训过程中对个人体重重要性的系统评估。通过监视参数演变，我们提出了一种有效降低模型大小而不会损害性能的方法。使用缩小的LLM和大型多模式模型进行了广泛的实验表明，中等修剪会提高效率并降低损失，而过度修剪会大大恶化模型性能。这些发现凸显了优化AI模型以确保可持续发展，平衡技术进步与环境责任的关键需求。

### All-in-one: Understanding and Generation in Multimodal Reasoning with the MAIA Benchmark 
[[arxiv](https://arxiv.org/abs/2502.16989)] [[cool](https://papers.cool/arxiv/2502.16989)] [[pdf](https://arxiv.org/pdf/2502.16989)]
> **Authors**: Davide Testa,Giovanni Bonetta,Raffaella Bernardi,Alessandro Bondielli,Alessandro Lenci,Alessio Miaschi,Lucia Passaro,Bernardo Magnini
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 多合一：通过MAIA基准的多模式推理的理解和产生
- **领域**: 计算语言学
- **摘要**: 我们介绍了Maia（多模式AI评估），这是一种本地 - 意大利基准测试，旨在细化视频中视觉语言模型的推理能力。 Maia不同于其他可用的视频基准，其设计，推理类别，使用的指标以及视频的语言和文化。它在两个校准任务上评估视觉语言模型（VLM）：视觉语句验证任务和一个开放式的视觉问题解答任务，均在相同的一组与视频相关的问题上。它考虑了十二个推理类别，旨在通过强调两个单独编码足够的信息来求解任务时，以及短视频的全部丰富性是必不可少的，而不仅仅是其中的一部分时，旨在解散语言和视觉关系。由于其精心授课的设计，它通过汇总度量来评估VLMS的一致性和视觉上的自然语言理解和产生。最后但并非最不重要的一点是，视频集已仔细选择，以反映意大利文化，语言数据是由母语扬声器产生的。

### NUTSHELL: A Dataset for Abstract Generation from Scientific Talks 
[[arxiv](https://arxiv.org/abs/2502.16942)] [[cool](https://papers.cool/arxiv/2502.16942)] [[pdf](https://arxiv.org/pdf/2502.16942)]
> **Authors**: Maike Züfle,Sara Papi,Beatrice Savoldi,Marco Gaido,Luisa Bentivogli,Jan Niehues
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: Nutshell：从科学谈话中进行抽象生成的数据集
- **领域**: 计算语言学
- **摘要**: 科学沟通正在受到自然语言处理的越来越多的关注，尤其是为了帮助研究访问，总结和生成内容。在该领域的一个新兴应用是语音到抽象的产生（SAG），该应用程序旨在自动从记录的科学演示中生成摘要。 SAG使研究人员能够有效地参与会议谈判，但是由于缺乏大规模数据集，进度受到了限制。为了解决这一差距，我们介绍了 *ACL会议对话与相应摘要的新型多式模式数据集Nutshell。我们建立了强大的基准，以使用自动指标和人类判断来评估生成摘要的质量。我们的结果突出了SAG的挑战，并证明了Nutshell培训的好处。通过在开放许可（CC-BY 4.0）下释放简介，我们旨在推进SAG的研究并促进改进的模型和评估方法的开发。

### Sarang at DEFACTIFY 4.0: Detecting AI-Generated Text Using Noised Data and an Ensemble of DeBERTa Models 
[[arxiv](https://arxiv.org/abs/2502.16857)] [[cool](https://papers.cool/arxiv/2502.16857)] [[pdf](https://arxiv.org/pdf/2502.16857)]
> **Authors**: Avinash Trivedi,Sangeetha Sivanesan
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: AAAI-25 DEFACTIFY 4.0 Workshop AI generated text detection (1st Rank)
- **标题**: Sarang在Defactify 4.0：使用列的数据和Deberta模型集合检测AI生成的文本
- **领域**: 计算语言学,人工智能
- **摘要**: 本文提出了一种检测AI生成的文本的有效方法，该方法是为Defactify 4.0共享任务开发的，该任务是在第四个关于多模式事实检查和仇恨言论检测的研讨会上开发的。该任务由两个子任务组成：任务-A，对文本是AI生成还是人为书面的分类，以及Task-B，对生成文本的特定大型语言模型进行分类。我们的团队（Sarang）在两项任务中均获得了F1分别为1.0和0.9531的第一名。该方法涉及在数据集中添加噪声，以改善模型的鲁棒性和概括。我们使用Deberta模型的合奏来有效地捕获文本中的复杂模式。结果表明了我们以噪声为基础和合奏的方法的有效性，在AI生成的文本检测中设定了新标准，并为未来的发展提供指导。

### Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs 
[[arxiv](https://arxiv.org/abs/2502.18791)] [[cool](https://papers.cool/arxiv/2502.18791)] [[pdf](https://arxiv.org/pdf/2502.18791)]
> **Authors**: Jungsoo Park,Junmo Kang,Gabriel Stanovsky,Alan Ritter
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 21 pages, 9 figures
- **标题**: 看到树木的森林：大规模，不断更新Frontier LLMS的荟萃分析
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: LLM研究的激增使他们的发现具有挑战性。荟萃分析可以发现整个研究中的重要趋势，但其使用受到手动数据提取的耗时性的限制。我们的研究提出了一种半自动化的荟萃分析方法，可加速使用LLMS提取数据。它会自动识别相关的ARXIV论文，提取实验结果和相关属性，并将其组织到结构化的数据集中。我们使用自动提取的数据集对Frontier LLM进行了全面的荟萃分析，与手动方法相比，纸张测量和数据提取的努力减少了93％以上。我们通过证明它从最近的手动荟萃分析（COT）重现关键发现来验证我们的数据集，并发现超出数据的新见解，例如，在封闭式示例中有利于多模式任务，但与COT相比提供了数学任务的有限收益。我们自动更新的数据集可通过提取评估研究来促进目标模型的连续跟踪，因为新数据可用。通过我们的科学工件和经验分析，我们为LLM提供了新的见解，同时促进了其行为的持续荟萃分析。

### What are Foundation Models Cooking in the Post-Soviet World? 
[[arxiv](https://arxiv.org/abs/2502.18583)] [[cool](https://papers.cool/arxiv/2502.18583)] [[pdf](https://arxiv.org/pdf/2502.18583)]
> **Authors**: Anton Lavrouk,Tarek Naous,Alan Ritter,Wei Xu
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 后苏联世界的基础模型是什么？
- **领域**: 计算语言学
- **摘要**: 后苏联国家的文化很复杂，它是由动荡的历史继续影响时事的塑造。在这项研究中，我们通过构建Borsch来调查基础模型的后文化食品知识，Borsch是俄罗斯和乌克兰语言中涵盖1147和823菜的多模式数据集，以后苏联地区为中心。 We demonstrate that leading models struggle to correctly identify the origins of dishes from Post-Soviet nations in both text-only and multimodal Question Answering (QA), instead over-predicting countries linked to the language the question is asked in. Through analysis of pretraining data, we show that these results can be explained by misleading dish-origin co-occurrences, along with linguistic phenomena such as Russian-Ukrainian code mixing.最后，为了超越基于质量检查的评估，我们测试了模型的能力，以产生对菜肴的准确视觉描述。这项任务与质量检查之间的弱相关性表明，仅质量保证可能不足以评估文化理解。为了促进进一步的研究，我们将在https://github.com/alavrouk/borsch上公开提供Borsch。

### BottleHumor: Self-Informed Humor Explanation using the Information Bottleneck Principle 
[[arxiv](https://arxiv.org/abs/2502.18331)] [[cool](https://papers.cool/arxiv/2502.18331)] [[pdf](https://arxiv.org/pdf/2502.18331)]
> **Authors**: EunJeong Hwang,Peter West,Vered Shwartz
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 瓶装：使用信息瓶颈原则的自我信息幽默解释
- **领域**: 计算语言学
- **摘要**: 幽默在在线通信中普遍存在，它通常依赖于多种模式（例如卡通和模因）。在多模式环境中解释幽默需要利用各种类型的知识，包括隐喻，社会文化和常识知识。但是，确定最有用的知识仍然是一个悬而未决的问题。我们介绍了\ method {}，这是一种受到信息瓶颈原则启发的方法，该方法从视觉和语言模型中引起了世界知识的相关知识，这是迭代地完善的，用于以一种无聊的方式产生幽默的解释。我们在三个数据集上的实验证实了我们方法比一系列基线的优势。将来，我们的方法可以进一步适应其他任务，这些任务可以从相关的世界知识中引起和调理，并朝着这一方向开放新的研究途径。

### Uncertainty Modeling in Multimodal Speech Analysis Across the Psychosis Spectrum 
[[arxiv](https://arxiv.org/abs/2502.18285)] [[cool](https://papers.cool/arxiv/2502.18285)] [[pdf](https://arxiv.org/pdf/2502.18285)]
> **Authors**: Morteza Rohanian,Roya M. Hüppi,Farhad Nooralahzadeh,Noemi Dannecker,Yves Pauli,Werner Surbeck,Iris Sommer,Wolfram Hinzen,Nicolas Langer,Michael Krauthammer,Philipp Homan
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 在整个精神病谱系中多模式语音分析中的不确定性建模
- **领域**: 计算语言学
- **摘要**: 由于语音模式的固有变化，捕获精神病范围内的细微语音中断是具有挑战性的。这种可变性反映了临床和非临床人群中症状的个体差异和症状的波动。言语数据中的不确定性考虑到预测症状严重程度和提高诊断精度至关重要。精神病的语音中断特征出现在整个范围内，包括非临床个体。我们开发了一个不确定性感知的模型，该模型整合了声学和语言特征，以预测症状严重程度和与精神病相关的特征。量化特定方式的不确定性可以使模型可以解决语音可变性，从而提高预测准确性。我们分析了来自114名参与者的语音数据，其中包括32名患有早期精神病的人和82个患有低或高精神分裂症的人，通过结构化访谈，半结构化自传任务以及德语中的叙事驱动互动收集。该模型提高了预测准确性，降低了RMSE，并在ECE = 4.5E-2的情况下达到了83％的F1分数，在不同的相互作用环境中显示出良好的性能。不确定性估计通过识别语音标记（例如音高变异性，流利度中断和光谱不稳定性）的可靠性差异来改善模型的解释性。该模型对任务结构进行了动态调整，在结构化设置和语言特征中加权声学特征在非结构化上下文中。这种方法在精神病谱研究中加强了早期检测，个性化评估和临床决策。

### Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs 
[[arxiv](https://arxiv.org/abs/2502.18179)] [[cool](https://papers.cool/arxiv/2502.18179)] [[pdf](https://arxiv.org/pdf/2502.18179)]
> **Authors**: Gaye Colakoglu,Gürkan Solmaz,Jonathan Fürst
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 问题解决了吗？使用LLMS的信息提取设计空间富含布局的文档
- **领域**: 计算语言学,人工智能
- **摘要**: 本文使用大型语言模型（LLMS）定义并探讨了来自布局丰富文档的信息提取（IE）的设计空间。 LLM的布局意识到IE的三个核心挑战是1）数据构造，2）模型参与度和3）输出细化。我们的研究深入研究了这些核心挑战中的子问题，例如输入表示，分解，提示和选择LLMS和多模型。它通过新的布局意识IE测试套件来检查不同设计选择的结果，并针对Art-Art（SOA）模型Layoutlmv3进行了基准测试。结果表明，来自一次（OFAT）试验的配置实现了接近最佳的结果，而基线模型的F1得分增益为14.1分，而完整的阶乘探索仅产生的15.1点在大约36倍的标记使用情况下仅略高。我们证明，配置良好的通用LLM可以与专业模型的性能相匹配，从而提供了具有成本效益的替代方案。我们的测试套件可在https://github.com/gayecolakoglu/layie-llm上免费获得。

### NusaAksara: A Multimodal and Multilingual Benchmark for Preserving Indonesian Indigenous Scripts 
[[arxiv](https://arxiv.org/abs/2502.18148)] [[cool](https://papers.cool/arxiv/2502.18148)] [[pdf](https://arxiv.org/pdf/2502.18148)]
> **Authors**: Muhammad Farid Adilazuarda,Musa Izzanardi Wijanarko,Lucky Susanto,Khumaisa Nur'aini,Derry Wijaya,Alham Fikri Aji
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: Nusaaksara：用于保存印度尼西亚土著脚本的多模式和多语言基准
- **领域**: 计算语言学
- **摘要**: 印度尼西亚拥有丰富的语言和脚本。但是，大多数NLP的进度都是使用罗马化文本进行的。在本文中，我们介绍了Nusaaksara，这是印尼语言的新型公共基准，其中包括其原始脚本。我们的基准涵盖了文本和图像方式，并涵盖了各种任务，例如图像分割，OCR，音译，翻译和语言标识。我们的数据是由人类专家通过严格的步骤构建的。 Nusaaksara涵盖了7种语言的8个脚本，包括NLP基准中常见的低资源语言。尽管Unicode不支持，但该数据集包含Lampung脚本。我们在LLM和VLMS（例如GPT-4O，LLAMA 3.2和AYA 23）的几种模型中基准了我们的数据，到诸如PP-OCR和Langid之类的特定任务系统，并表明大多数NLP技术无法处理印度尼西亚本地脚本，许多人的效果接近零。

### LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts 
[[arxiv](https://arxiv.org/abs/2502.19202)] [[cool](https://papers.cool/arxiv/2502.19202)] [[pdf](https://arxiv.org/pdf/2502.19202)]
> **Authors**: Thanh-Phong Le,Trung Le Chi Phan,Nghia Hieu Nguyen,Kiet Van Nguyen
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Accepted at IJDAR
- **标题**: 光：注入布局的生成变压器，用于视觉问题，回答越南收据
- **领域**: 计算语言学
- **摘要**: 文档视觉问题回答（文档VQA）挑战多模式系统，以整体处理文本，布局和视觉方式，以提供适当的答案。由于文档量的增加和对数字化的需求越来越大，VQA近年来已广受欢迎。尽管如此，大多数文档VQA数据集都以高资源语言（例如英语）开发。在本文中，我们介绍了RecoriptVQA（\ textbf {receip} \ textbf {v} isual \ textbf {q} uestion \ textbf {a} nswering），最初的大规模文档VQA数据集在越南人中，越南人vQA数据集，专用于收到，具有高级商业潜力，具有高级商业潜力。该数据集包括\ textbf {9,000+}收据图像和\ textbf {60,000+}手动注释的Question-Asswer对。除了研究外，我们还介绍了Ligt（\ textbf {l} ayout- \ textbf {i} nfused \ textbf {g} forthbf {g} textbf {g} textbf {t} ransformer），布局意识 - 意识到的意识 - 意识到的型号架构旨在驱动语言模型的其他单元模型，以实现运行的单元模型。关于RecoriptVQA的实验表明，与出色的基线相比，我们的体系结构产生了有希望的表现，从而实现了竞争成果。此外，在分析实验结果的过程中，我们发现了明显的模式，与可以产生答案的体系结构相比，采用仅编码模型体系结构具有相当大的缺点。我们还观察到，尽管语言模型的语义理解至关重要，但有必要结合多种方式来解决我们的数据集。我们希望我们的工作将鼓励和促进越南文档VQA的未来发展，从而为越南语言的多样化多模式研究社区做出了贡献。

### Exploring Rewriting Approaches for Different Conversational Tasks 
[[arxiv](https://arxiv.org/abs/2502.18860)] [[cool](https://papers.cool/arxiv/2502.18860)] [[pdf](https://arxiv.org/pdf/2502.18860)]
> **Authors**: Md Mehrab Tanjim,Ryan A. Rossi,Mike Rimer,Xiang Chen,Sungchul Kim,Vaishnavi Muppala,Tong Yu,Zhengmian Hu,Ritwik Sinha,Wei Zhang,Iftikhar Ahamath Burhanuddin,Franck Dernoncourt
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Preprint
- **标题**: 探索针对不同对话任务的重写方法
- **领域**: 计算语言学
- **摘要**: 会话助理通常需要一个问题重写算法，该算法利用过去的交互作用来为用户的问题或请求提供更有意义的（准确）答案。但是，确切的重写方法通常取决于对话助手支持的用例和特定于应用程序的任务以及其他约束。在本文中，我们在两个根本不同的生成任务上，系统地研究了两种不同的方法，称为重写和融合，包括文本到文本生成任务和多模式生成任务，该任务将作为输入文本并生成可视化或数据表，以回答用户的问题。我们的结果表明，特定的重写或融合方法在很大程度上取决于基本用例和生成任务。特别是，我们发现，对于对话式的提问助手，查询重写方法表现最佳，而对于数据分析助手，该助手根据用户与助手的对话生成可视化和数据表，Fusion方法最有效。值得注意的是，我们探索了两个用于数据分析助理用例的数据集，用于简短和长时间的对话，并且发现查询融合的性能总是更好，而对于基于对话的基于文本的问题，查询重写方法的性能最好。

### A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.20504)] [[cool](https://papers.cool/arxiv/2502.20504)] [[pdf](https://arxiv.org/pdf/2502.20504)]
> **Authors**: Julius Broomfield,Kartik Sharma,Srijan Kumar
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 一千个单词或图像：研究角色方式在多模式LLM中的影响
- **领域**: 计算语言学,人工智能,计算机视觉和模式识别
- **摘要**: 大型语言模型（LLMS）最近在体现各种角色，增强其作为对话代理和虚拟助手的有效性方面取得了显着进步。因此，LLM在处理和整合多模式信息方面已取得了重大进步。但是，即使可以在文本和图像中表达人类角色，角色的形式影响LLM的实施方案的程度仍然在很大程度上没有探索。在本文中，我们研究了不同模态在多模式LLM中的表现力如何影响。为此，我们创建了一个新颖的模式平行数据集，该数据集的年龄，性别，职业和位置各不相同。这包括四种方式，可以等效地代表角色：仅图像，仅文本，图像和小文本的组合以及印刷图像，在该图像上进行了视觉风格的文本以传达与角色相关的属性。然后，我们创建一个系统的评估框架，其中有60个问题和相应的指标，以评估LLM在其属性和场景中体现每个角色的表现如何。 $ 5 $多模式LLMS的综合实验表明，由详细文本代表的角色显示出更多的语言习惯，而印刷图像通常显示出与角色的一致性。我们的结果表明，LLM经常忽略通过图像传达的特定于人格特定的细节，突出了基本的局限性，并为将来的研究铺平了这一差距。我们在https://github.com/claws-lab/persona-modality上发布数据和代码。

### Protecting multimodal large language models against misleading visualizations 
[[arxiv](https://arxiv.org/abs/2502.20503)] [[cool](https://papers.cool/arxiv/2502.20503)] [[pdf](https://arxiv.org/pdf/2502.20503)]
> **Authors**: Jonathan Tonglet,Tinne Tuytelaars,Marie-Francine Moens,Iryna Gurevych
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Preprint. Code and data available at https://github.com/UKPLab/arxiv2025-misleading-visualizations
- **标题**: 保护多模式大语模型免受误导性可视化
- **领域**: 计算语言学
- **摘要**: 我们评估了多模式大语言模型误导可视化的脆弱性 - 图表使用诸如截短或倒轴等技术扭曲基础数据的图表，导致读者得出可能支持错误信息或阴谋理论的不准确结论。我们的分析表明，这些扭曲严重损害了多模式的大语言模型，将其提问的准确性降低到随机基线的水平。为了减轻这种脆弱性，我们介绍了六种推理时间方法，以提高MLLM在误导性可视化方面的性能，同时保留其在非误导性方面的准确性。最有效的方法涉及（1）提取基础数据表，以及（2）使用仅文本大型语言模型来回答基于表的问题。这种方法将误导性可视化的性能提高了15.4至19.6个百分点。

### Chitranuvad: Adapting Multi-Lingual LLMs for Multimodal Translation 
[[arxiv](https://arxiv.org/abs/2502.20420)] [[cool](https://papers.cool/arxiv/2502.20420)] [[pdf](https://arxiv.org/pdf/2502.20420)]
> **Authors**: Shaharukh Khan,Ayush Tarun,Ali Faraz,Palash Kamble,Vivek Dahiya,Praveen Pokala,Ashish Kulkarni,Chandra Khatri,Abhinav Ravi,Shubham Agarwal
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: ef:https://aclanthology.org/2024.wmt-1.80/
- **标题**: Chitranuvad：调整多语言LLM用于多模式翻译
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 在这项工作中，我们在亚洲翻译研讨会（WAT2024）的研讨会上提供了作为英语的一部分提交的系统描述。我们介绍了Chitranuvad，这是一种多模型模型，可有效整合多语言LLM和一个视觉模块以进行多模式翻译。我们的方法使用VIT图像编码器将视觉表示形式提取为视觉令牌嵌入，通过适配器层将其投影到LLM空间并以自动回归方式生成翻译。我们参与了指示语言（即英文翻译到印地语，孟加拉语和马利亚拉姆）的所有三个曲目（图像字幕，仅文本和多模式翻译任务），并在挑战集中在挑战集中获得了印地语的SOTA结果，同时在共享任务中对其他语言保持竞争力。

### Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents 
[[arxiv](https://arxiv.org/abs/2502.19917)] [[cool](https://papers.cool/arxiv/2502.19917)] [[pdf](https://arxiv.org/pdf/2502.19917)]
> **Authors**: Zhenyu Liu,Yunxin Li,Baotian Hu,Wenhan Luo,Yaowei Wang,Min Zhang
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 15 pages, 7 figures
- **标题**: 挑选农作物的奶油：以合作代理商的视觉数据选择
- **领域**: 计算语言学
- **摘要**: 为了改善多模式大型语言模型的（MLLM）处理图像和复杂说明的能力，研究人员主要策划大规模的视觉说明调谐数据集，这些数据集是从现有视觉任务中来自现有视觉任务或使用LLMS和图像描述的合成生成的。但是，它们通常会遭受关键缺陷，包括未对准的教学图像对和低质量的图像。此类问题阻碍了训练效率并限制了绩效的提高，因为在嘈杂或无关的数据上浪费资源对整体能力的好处最小。为了解决这个问题，我们通过\ textbf {a} gents Collaboration（Visa）提出了一个以\ textric-centric \ textbf {s}选举方法进行的\ textbf {vi} \ textbf {vi}，该方法以图像质量评估和图像结构相关性评估为中心。具体而言，我们的方法包括1）通过视觉代理协作的图像信息量化方法，以选择具有丰富视觉信息的图像，以及2）以视觉为中心的指令质量评估方法，以选择与高质量图像有关的高质量指导数据。最后，我们从大型开源数据集重新组织了80K指令数据。广泛的实验表明，签证的表现优于七个基准的当前最新模型，仅使用2.5％的原始数据，强调了我们数据选择方法的效率。此外，我们进行消融研究以验证方法的每个组成部分的有效性。该代码可在https://github.com/hitsz-tmg/visa上找到。

### MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge 
[[arxiv](https://arxiv.org/abs/2502.19870)] [[cool](https://papers.cool/arxiv/2502.19870)] [[pdf](https://arxiv.org/pdf/2502.19870)]
> **Authors**: Yuntao Du,Kailin Jiang,Zhi Gao,Chenrui Shi,Zilong Zheng,Siyuan Qi,Qing Li
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Accept to ICLR2025. Project Page: https://mmke-bench-iclr.github.io/
- **标题**: MMKE BENCH：一种多模式编辑基准，用于不同的视觉知识
- **领域**: 计算语言学
- **摘要**: 知识编辑技术已成为更新大型语言模型（LLMS）和多模式模型（LMMS）的事实知识的基本工具，从而使它们可以纠正过时的或不准确的信息，而无需从头开始重新审阅。但是，用于多模式知识编辑的现有基准主要集中于表示为简单三胞胎的实体级别知识，这些知识未能捕获现实世界多模式信息的复杂性。为了解决这个问题，我们介绍了MMKE-Bench，这是一种全面的多模式知识编辑基准，旨在评估LMM在现实世界中编辑各种视觉知识的能力。 MMKE基础通过合并三种类型的编辑任务来解决这些限制：视觉实体编辑，视觉语义编辑和特定于用户的编辑。此外，MMKE-Bench使用自由形式的自然语言来表示和编辑知识，提供更灵活，更有效的格式。该基准由33个广泛类别的2,940张知识和8,363张图像组成，评估问题自动产生和人为验证。我们评估了三种突出的LMM上的五种最先进的知识编辑方法，表明没有任何方法在所有标准上都擅长，并且视觉和用户特定的编辑特别具有挑战性。 MMKE BENCH设定了一个新标准，用于评估多模式知识编辑技术的鲁棒性，从而在这个快速发展的领域中推动了进度。

## 密码学和安全(cs.CR:Cryptography and Security)

该领域共有 7 篇论文

### `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.00735)] [[cool](https://papers.cool/arxiv/2502.00735)] [[pdf](https://arxiv.org/pdf/2502.00735)]
> **Authors**: Chun Wai Chiu,Linghan Huang,Bo Li,Huaming Chen
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: ````做我不像我'''
- **领域**: 密码学和安全,人工智能,软件工程
- **摘要**: 大型语言模型（LLMS）由于处理多种类型的输入数据的能力，包括文本，音频，图像和视频，因此在各个领域看到了广泛的应用程序。尽管LLM在理解和生成不同场景的上下文方面表现出了出色的表现，但它们很容易受到及时的攻击，这些攻击主要是通过文本输入。在本文中，我们介绍了对多模式LLM的首次基于语音的越狱攻击，称为侧翼攻击，该攻击可以同时处理不同类型的输入。我们的工作是由单语语音驱动的大语言模型的最新进步激发的，这些模型引入了传统的LLM基于文本的漏洞以外的新攻击表面。为了调查这些风险，我们检查了最新的多模式LLM，可以通过不同类型的输入（例如音频输入）访问它们，重点是对抗提示如何绕过其防御机制。我们提出了一种新颖的策略，其中不允许的提示侧面是良性，叙事驱动的提示。它集成在侧翼攻击中，该攻击试图使交互环境人性化并通过虚构的环境执行攻击。此外，为了更好地评估攻击性能，我们提出了一个半自动化的自我评估框架，以进行违反政策。我们证明，侧翼攻击能够操纵最先进的LLM，以产生未对准和禁止的输出，这在七个禁止的情况下达到了平均攻击成功率在0.67至0.93之间。

### Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment 
[[arxiv](https://arxiv.org/abs/2502.02438)] [[cool](https://papers.cool/arxiv/2502.02438)] [[pdf](https://arxiv.org/pdf/2502.02438)]
> **Authors**: Yaling Shen,Zhixiong Zhuang,Kun Yuan,Maria-Irina Nicolae,Nassir Navab,Nicolas Padoy,Mario Fritz
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: Accepted at AAAI 2025
- **标题**: 医疗多模型模型通过对抗域对齐窃取攻击
- **领域**: 密码学和安全,人工智能
- **摘要**: 医学多模式大型语言模型（MLLM）正在成为医疗保健系统的重要组成部分，从而帮助医务人员进行决策和结果分析。放射学报告生成的模型能够解释医学图像，从而减少了放射科医生的工作量。由于医疗数据稀缺并受到隐私法规的保护，医疗MLLM代表了宝贵的知识产权。但是，这些资产可能容易受到模型窃取的影响，攻击者旨在通过Black-Box访问复制其功能。到目前为止，为医疗领域窃取模型已集中在分类上。但是，现有攻击对MLLM无效。在本文中，我们引入了对抗域的对准（ADA-Steal），这是对医疗MLLM的第一次偷窃攻击。 Ada Stereal依赖于公开且广泛可用的自然图像，而不是其医疗对应物。我们表明，具有对抗性噪声的数据增加足以克服自然图像与受害者MLLM的域特异性分布之间的数据分布差距。 IU X射线和MIMIC-CXR放射学数据集的实验表明，对抗域的对准使攻击者可以窃取医疗MLLM而无需访问任何医疗数据。

### LLMs in Software Security: A Survey of Vulnerability Detection Techniques and Insights 
[[arxiv](https://arxiv.org/abs/2502.07049)] [[cool](https://papers.cool/arxiv/2502.07049)] [[pdf](https://arxiv.org/pdf/2502.07049)]
> **Authors**: Ze Sheng,Zhicheng Chen,Shuning Gu,Heqing Huang,Guofei Gu,Jeff Huang
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: 33 pages, 12 figures
- **标题**: 软件安全性LLM：漏洞检测技术和见解的调查
- **领域**: 密码学和安全,人工智能
- **摘要**: 大型语言模型（LLM）正在成为用于软件漏洞检测的变革性工具，从而解决了安全域中的关键挑战。传统方法（例如静态和动态分析）通常由于低效率，高误报率以及现代软件系统日益增长的复杂性而动摇。通过利用其分析代码结构的能力，识别模式并生成维修建议，LLM，以GPT，BERT和CODEBERT等模型为例，提出了一种新颖且可扩展的方法来减轻脆弱性。本文提供了脆弱性检测中LLM的详细调查。它研究了关键方面，包括模型架构，应用程序方法，目标语言，微调策略，数据集和评估指标。我们还分析了当前研究问题的范围，突出了现有方法的优势和劣势。此外，我们应对诸如跨语言漏洞检测，多模式数据集成和存储库级分析等挑战。基于这些发现，我们为在低资源场景中的数据集可伸缩性，模型可解释性和应用程序等问题提出了解决方案。我们的贡献是三个方面：（1）对LLM在脆弱性检测中的应用； （2）对研究之间共享模式和差异的分析，并具有理解该领域的统一框架； （3）关键挑战和未来研究方向的摘要。这项工作为推进基于LLM的脆弱性检测提供了宝贵的见解。我们还在https://github.com/owensanzas/llm-for-vulnerability-detection上维护并定期更新最新的选定论文

### X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks 
[[arxiv](https://arxiv.org/abs/2502.10475)] [[cool](https://papers.cool/arxiv/2502.10475)] [[pdf](https://arxiv.org/pdf/2502.10475)]
> **Authors**: Zihang Cheng,Huiping Zhuang,Chun Li,Xin Meng,Ming Li,Fei Richard Yu
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: X-SG $^2 $ S：安全且可概括的高斯分裂X维水印
- **领域**: 密码学和安全,人工智能,计算机视觉和模式识别
- **摘要**: 3D高斯裂（3DG）已被广泛用于3D重建和3D代中。培训以获得3DGS场景通常需要大量时间和资源，甚至需要宝贵的灵感。越来越多的3DGS数字资产给版权保护带来了巨大的挑战。但是，它仍然缺乏针对3DG的深刻探索。在本文中，我们提出了一个新的框架X-SG $^2 $ S，该框架可以同时将最初的3DGS场景保持在几乎没有变化的同时。通常，我们有一个用于同时添加多模式消息的X-SG $^2 $ S喷油器和用于提取它们的提取器。具体来说，我们首先以固定的方式将水印将其分为消息补丁，然后对3DGS点进行排序。自适应门用于挑选合适的位置用于水印。然后，使用XD（多维）提示头将多模式消息添加到排序的3DGS点中。可学习的大门可以识别带有额外消息的位置，XD撤销头可以从可学习门推荐的位置恢复隐藏的消息。广泛的实验表明，所提出的X-SG $^2 $ S可以有效地隐藏多模态消息，而无需更改预验证的3DGS管道或3DGS参数的原始形式。同时，X-SG $^2 $ S具有简单有效的模型结构和高实用性，在隐藏和提取多模式内部结构化或非结构化消息方面仍然显示出良好的性能。 X-SG $^2 $ S是第一个统一3DGS的1至3D水印模型，也是第一个在一个3DG中添加多模式水印的框架，该框架为以后的研究铺平了波浪。

### A Survey of Safety on Large Vision-Language Models: Attacks, Defenses and Evaluations 
[[arxiv](https://arxiv.org/abs/2502.14881)] [[cool](https://papers.cool/arxiv/2502.14881)] [[pdf](https://arxiv.org/pdf/2502.14881)]
> **Authors**: Mang Ye,Xuankun Rong,Wenke Huang,Bo Du,Nenghai Yu,Dacheng Tao
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-21
> **comment**: 22 pages, 2 figures
- **标题**: 大型视觉模型的安全性调查：攻击，防御和评估
- **领域**: 密码学和安全,计算机视觉和模式识别
- **摘要**: 随着大型视觉模型（LVLM）的快速发展，确保其安全性已成为至关重要的研究领域。这项调查提供了对LVLM安全性的全面分析，涵盖了诸如攻击，防御和评估方法之类的关键方面。我们介绍了一个统一的框架，该框架整合了这些相互关联的组件，为LVLM的脆弱性和相应的缓解策略提供了整体观点。通过对LVLM生命周期的分析，我们引入了一个分类框架，该框架区分推理和训练阶段，并提供了进一步的子类别，以提供更深入的见解。此外，我们强调了现有研究的局限性，并概述了未来的方向，旨在增强LVLM的鲁棒性。作为我们研究的一部分，我们对最新的LVLM DeepSeek Janus-Pro进行了一系列安全评估，并对结果进行了理论分析。我们的发现提供了提高LVLM安全性并确保其在高风险，现实世界中的安全部署的战略建议。这项调查旨在作为未来研究的基石，促进了模型的开发，不仅可以突破多模式智能的界限，而且还遵守了最高的安全和道德完整性标准。此外，为了帮助这一领域的研究，我们创建了一个公共存储库，以不断编译和更新有关LVLM安全的最新工作：https：//github.com/xuankunrong/awsome/awsome-lvlm-safety。

### Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM 
[[arxiv](https://arxiv.org/abs/2502.17763)] [[cool](https://papers.cool/arxiv/2502.17763)] [[pdf](https://arxiv.org/pdf/2502.17763)]
> **Authors**: Yuqing Wang,Xiao Yang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 设计和实施分布式安全威胁检测系统集成了联合学习和多模式LLM
- **领域**: 密码学和安全,人工智能,分布式、并行和集群计算,表现
- **摘要**: 传统的安全保护方法难以解决大规模分布式系统中的复杂攻击向量，尤其是在平衡检测准确性与数据隐私问题时。本文介绍了一种新颖的分布式安全威胁检测系统，该系统将联合学习与多模式大语模型（LLMS）集成在一起。我们的系统利用联合学习来确保数据隐私，同时采用多模式LLMS处理异质数据源，包括网络流量，系统日志，图像和传感器数据。对10TB分布式数据集的实验评估表明，我们的方法达到96.4％的检测准确性，表现优于传统基线模型，高于4.1个百分点。该系统分别将假阳性和假负率降低1.8和2.4个百分点。性能分析表明，我们的系统在分布式环境中保持有效的处理能力，需要180秒进行模型培训，在分布式网络上进行威胁检测3.8秒。这些结果表明检测准确性和计算效率有了显着提高，同时保留了数据隐私，这表明在大规模安全系统中实现现实世界部署的强大潜力。

### Steganography Beyond Space-Time With Chain of Multimodal AI Agents 
[[arxiv](https://arxiv.org/abs/2502.18547)] [[cool](https://papers.cool/arxiv/2502.18547)] [[pdf](https://arxiv.org/pdf/2502.18547)]
> **Authors**: Ching-Chun Chang,Isao Echizen
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 具有多模式AI代理链的超越时空的隐肌
- **领域**: 密码学和安全,人工智能,多代理系统,多媒体
- **摘要**: 隐肌是秘密写作的艺术和科学，在网络安全领域中，广泛的应用相互交织。随着人工智能的不断发展，其合成现实内容的能力在试图操纵和歪曲真理的网络犯罪分子的手中成为一种威胁。这样的合成内容引入了覆盖造影目的的微妙变化的非平凡风险。当空间和时间域中的信号容易受到不可预见的覆盖，它要求反思毕竟可能保持不变的内容。这项研究提出了对视听媒体的隐身志范围的范式，其中隐藏在空间和时间域之外的信息。开发了一系列多模式代理，将视听内容解构为封面文本，将消息嵌入语言领域中，然后通过将听觉和视觉方式与结果的Stego文本同步，然后重建视听内容。该消息是通过偏向语言生成模型的单词采样过程来编码的，并通过分析单词选择的概率分布来解码。在零位和多位容量设置下评估消息传输的准确性。通过生物识别和语义相似性来评估富达，捕获了记录的面部和声音的身份，以及通过媒体传达的核心思想。通过封面和Stego文本之间的统计比较来检查保密。在各种情况下都测试了鲁棒性，包括视听压缩，面部扫描，语音粘合及其组合。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 266 篇论文

### PixelWorld: Towards Perceiving Everything as Pixels 
[[arxiv](https://arxiv.org/abs/2501.19339)] [[cool](https://papers.cool/arxiv/2501.19339)] [[pdf](https://arxiv.org/pdf/2501.19339)]
> **Authors**: Zhiheng Lyu,Xueguang Ma,Wenhu Chen
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: None
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 现有的基础模型通常将视觉输入作为像素和文本输入作为代币，这是一种与人类感知形成对比的范式，在这种范式上，这两种方式都是以统一的方式处理的。随着体现和代理AI的兴起，输入主要来自相机像素，对统一感知框架的需求变得越来越明显。在本文中，我们建议将所有模式（文本，表，代码，图表，图像等）统一为像素输入，即“将所有内容视为像素”（PEAP）。我们介绍了PixelWorld，这是一个新颖的评估套件，将所有上述模式统一到像素空间中，以评估现有模型的性能。我们的发现表明，（1）PEAP在多模式数据集中具有基于令牌的输入优于基线，从统一输入中受益于更好的歧义，（2）所有模型的推理和编码能力的显着下降，在处理基于像素的输入时，在处理基于像素的投入时，可以增强基础模型的较大模型，以增强基础模型的较大模型（3）较大的范围（3）较大的范围（3）较大的效果，（3）较大的行为，（3）较大的范围（3）较大的范围（3），3） PHI-3.5-V遭受明显的性能降解，（4）PEAP的注意力模式与文本令牌输入高度一致，（5）PEAP可以通过利用空间稀疏来显着加速PEAP。我们得出的结论是，现有的边界模型具有像素感知能力，但是，仍然有改进的余地。我们的代码，数据集将在接受后发布。

### Transformation trees -- documentation of multimodal image registration 
[[arxiv](https://arxiv.org/abs/2501.19140)] [[cool](https://papers.cool/arxiv/2501.19140)] [[pdf](https://arxiv.org/pdf/2501.19140)]
> **Authors**: Agnieszka Anna Tomaka,Dariusz Pojda,Michał Tarnawski,Leszek Luchowski
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: 18 pages, 11 figures
- **标题**: 转换树 - 多模式图像注册的文档
- **领域**: 计算机视觉和模式识别
- **摘要**: 本文提出了将树结构应用于在与采集设备相关的坐标系中获得的多模式图像的各种登记，并在一个患者特异性坐标系统中注册的多模式图像的各种登记。引入了特殊的文件格式.DPW（数字患者工作区）。 DPVision软件中说明了从正畸分析中产生的不同注册的示例，并显示了树结构使用的主要方面。

### EgoMe: Follow Me via Egocentric View in Real World 
[[arxiv](https://arxiv.org/abs/2501.19061)] [[cool](https://papers.cool/arxiv/2501.19061)] [[pdf](https://arxiv.org/pdf/2501.19061)]
> **Authors**: Heqian Qiu,Zhaofeng Shi,Lanxiao Wang,Huiyu Xiong,Xiang Li,Hongliang Li
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: egome：通过现实世界中的以自我为中心的观点来关注我
- **领域**: 计算机视觉和模式识别
- **摘要**: 当与现实世界互动时，人类经常以自我为中心的（第一人称）视为基准，自然地转移了从异位中心（第三人称）观点观察到的行为。这种认知理论为研究机器人如何更有效地模仿人类行为奠定了基础。但是，当前的研究要么采用多个摄像机，具有不同的观点，要同时着重于同一个人的行为，或者遇到了不符合的自我exo视图方案，因此没有努力在现实世界中充分利用人类的认知行为。为了填补这一空白，在本文中，我们介绍了一种新颖的大型自我中心数据集，称为Egome，该数据集旨在通过现实世界中的以自我为中心的观点来遵循人类模仿学习的过程。我们的数据集包括7902对视频（15804个视频），用于现实情况下的每日行为。对于一对视频，一个视频捕获了观察演示者动作的模仿者的异议观点，而另一个视频则捕获了示威者的动作，而随后遵循这些动作的模仿者的以自我为中心的观点。值得注意的是，我们的数据集还包含Exo-ego眼睛凝视，角速度，加速度，磁强度和其他传感器多模式数据，以帮助建立观察和以下过程之间的相关性。此外，我们还提出了八项具有挑战性的基准任务，以充分利用此数据资源并促进机器人模仿学习能力的研究。与现有数据集相比，广泛的统计分析显示出显着的优势。拟议的Egome数据集和基准将很快发布。

### Contrast-Aware Calibration for Fine-Tuned CLIP: Leveraging Image-Text Alignment 
[[arxiv](https://arxiv.org/abs/2501.19060)] [[cool](https://papers.cool/arxiv/2501.19060)] [[pdf](https://arxiv.org/pdf/2501.19060)]
> **Authors**: Song-Lin Lv,Yu-Yang Chen,Zhi Zhou,Yu-Feng Li,Lan-Zhe Guo
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: 微调剪辑的对比度感知校准：利用图像文本对齐
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 视觉语言模型（VLM）（例如剪辑）已经证明了出色的概括能力，并且可以通过及时的微调快速适应下游任务。不幸的是，在涉及非训练类的分类任务中，被称为开放式视频计师环境，微调的VLM通常过于培训培训课程，从而导致置信度得分与未见类别的实际准确性之间的不对准，从而极大地破坏了实际部署中其可靠性。现有的置信度校准方法通常需要培训参数或分析培训数据集中的功能，从而限制了它们在没有相应的火车数据的情况下概括了看不见的类的能力。此外，VLM特定的校准方法仅依赖于火车类别作为校准指标的文本功能，从而固有地限制了其校准火车类别的能力。为了应对这些挑战，我们提出了一种有效的多模式校准方法对比感知校准（CAC）。基于原始剪辑的零射击适应性以及经验分析的结论，即在看不见的阶级上较差的阶层内和阶层间判别能力是根本原因，我们根据原始和微型调整剪辑之间的对比差来计算校准权重。这种方法不仅可以适应校准看不见的类，而且还克服了无法校准火车类的先前VLM校准方法的局限性。在涉及11种微调方法的11个数据集的实验中，CAC始终在不牺牲准确性和推理速度的情况下对火车和看不见的类别都达到了最佳的校准效果。

### Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models 
[[arxiv](https://arxiv.org/abs/2501.19054)] [[cool](https://papers.cool/arxiv/2501.19054)] [[pdf](https://arxiv.org/pdf/2501.19054)]
> **Authors**: Ruiyu Wang,Yu Yuan,Shizhao Sun,Jiang Bian
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: 通过在大型语言模型中注入视觉反馈的文本到cad生成
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 创建计算机辅助设计（CAD）模型需要大量的专业知识和精力。将文本描述转换为CAD参数序列的文本到cad对于简化此过程至关重要。最近的研究利用了地面真实的参数序列（称为顺序信号）作为实现此目标的监督。但是，CAD模型本质上是多模式，包括参数序列和相应的渲染视觉对象。此外，从参数序列到视觉对象的渲染过程是众多的。因此，顺序和视觉信号对于有效训练至关重要。在这项工作中，我们介绍了CadFusion，该框架使用大型语言模型（LLM）作为骨干和两个训练阶段之间的交替：顺序学习（SL）阶段和视觉反馈（VF）阶段。在SL阶段，我们使用地面真实参数序列训练LLM，从而能够生成逻辑上一致的参数序列。在VF阶段，我们奖励将参数序列呈现为视觉上优先的对象并惩罚那些没有的对象，从而允许LLMS了解如何感知和评估渲染的视觉对象。这两个阶段在整个培训过程中交替出现，以确保平衡学习并保留两个信号的好处。实验表明，循环在定性和定量上都显着提高了性能。

### RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs 
[[arxiv](https://arxiv.org/abs/2501.19036)] [[cool](https://papers.cool/arxiv/2501.19036)] [[pdf](https://arxiv.org/pdf/2501.19036)]
> **Authors**: Hongliang Li,Jiaxin Zhang,Wenhui Liao,Dezhi Peng,Kai Ding,Lianwen Jin
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: 冗余：揭示和利用视觉令牌处理冗余，以进行有效的解码器
- **领域**: 计算机视觉和模式识别
- **摘要**: 当前的多模式大型语言模型（MLLM）体系结构面临性能和效率之间的关键权衡：仅解码器的体系结构可实现更高的性能，但效率较低，而基于跨注意的体系结构可提供更高的效率，但性能较低。关键区别在于如何处理视觉令牌。仅解码器架构对视觉令牌应用自我注意力和FFN操作，而跨注意体系结构跳过了这些计算。为了调查在这个计算昂贵的过程中是否存在冗余，我们提出了一个无训练的框架，用于分析训练有素的MLLM。它由探测激活的动态FFN和空心注意力组成，可调节视觉令牌的计算，以及层排名算法，该算法优先考虑这些降低的层。广泛的实验表明，仅解码器MLLM独有的实质性，结构化和聚类的冗余，为未来的MLLM体系结构设计提供了宝贵的见解。此外，通过利用我们的还原框架作为一种无训练的推理加速方法，我们实现了与最先进的方法相当或更好的性能，同时与它们保持兼容。代码将在https://github.com/l-hugh/redundancylens上公开获取。

### XRF V2: A Dataset for Action Summarization with Wi-Fi Signals, and IMUs in Phones, Watches, Earbuds, and Glasses 
[[arxiv](https://arxiv.org/abs/2501.19034)] [[cool](https://papers.cool/arxiv/2501.19034)] [[pdf](https://arxiv.org/pdf/2501.19034)]
> **Authors**: Bo Lan,Pei Li,Jiaxi Yin,Yunpeng Song,Ge Wang,Han Ding,Jinsong Han,Fei Wang
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: 27 pages, 11 figures, 8 tables
- **标题**: XRF V2：用于使用Wi-Fi信号的数据集，以及手机，手表，耳塞和眼镜的IMU
- **领域**: 计算机视觉和模式识别
- **摘要**: 人类行动识别（HAR）在健康监测，智能家庭自动化和人类计算机互动等应用中起着至关重要的作用。尽管对HAR进行了广泛的研究，但涉及识别和总结持续行动的行动摘要仍然是一项新的任务。本文介绍了新型XRF V2数据集，该数据集旨在室内日常活动时间动作定位（TAL）和动作摘要。 XRF V2集成了来自Wi-Fi信号，IMU传感器（智能手机，智能手表，耳机和智能眼镜）的多模式数据，以及同步视频录像，从而提供来自三个不同环境的16个志愿者的各种室内活动。为了解决TAL和ACTION摘要，我们提出了XRFMAMBA神经网络，该网络擅长捕获未经修剪的感觉序列中的长期依赖性，并胜过最先进的方法，例如Action Former和Wifitad。我们设想XRF V2是推进人类行动本地化研究，行动预测，姿势估算，多模式基础模型预训练，合成数据生成等的宝贵资源。

### LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models 
[[arxiv](https://arxiv.org/abs/2501.18954)] [[cool](https://papers.cool/arxiv/2501.18954)] [[pdf](https://arxiv.org/pdf/2501.18954)]
> **Authors**: Shenghao Fu,Qize Yang,Qijie Mo,Junkai Yan,Xihan Wei,Jingke Meng,Xiaohua Xie,Wei-Shi Zheng
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: LLMDET：在大语言模型的监督下学习强大的开放式视频对象探测器
- **领域**: 计算机视觉和模式识别
- **摘要**: 最近的开放式视频探测器通过丰富的区域级注释数据实现了有希望的性能。在这项工作中，我们表明，通过为每个图像生成图像级详细字幕字幕可以进一步提高性能，与大语言模型共同训练了开放式摄取探测器。为了实现目标，我们首先收集一个数据集，接地cap-1m，其中每个图像都伴随着相关的接地标签和图像级详细的标题。借助此数据集，我们将对开放式摄影仪进行训练，其中包括训练目标，包括标准的接地损失和字幕产生损失。我们利用大型语言模型为每个图像的每个区域和图像级长字幕生成两个区域级的简短字幕。在大语言模型的监督下，由此产生的检测器LLMDET以明确的边缘优于基线，具有卓越的开放式播放能力。此外，我们表明改进的LLMDET可以又可以建立更强大的大型多模式模型，从而实现相互利益。代码，模型和数据集可在https://github.com/isee-laboratory/llmdet上找到。

### TV-Dialogue: Crafting Theme-Aware Video Dialogues with Immersive Interaction 
[[arxiv](https://arxiv.org/abs/2501.18940)] [[cool](https://papers.cool/arxiv/2501.18940)] [[pdf](https://arxiv.org/pdf/2501.18940)]
> **Authors**: Sai Wang,Fan Ma,Xinyi Li,Hehe Fan,Yu Wu
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: None
- **领域**: 计算机视觉和模式识别
- **摘要**: LLM的最新进步加速了跨文本和图像的对话生成的发展，但基于视频的对话生成仍然没有被忽视，并带来了独特的挑战。在本文中，我们介绍了主题感知的视频对话制作（TVDC），这是一项新型任务，旨在生成与视频内容相符并遵守用户指定主题的新对话。我们提出了TV-Dialogue，这是一种新型的多模式代理框架，可确保主题对齐（即对话围绕主题围绕主题）和视觉一致性（即，对话在视频中角色中字符的情感和行为与视频中字符的情感和行为）匹配，从而使视频中的实时互动构成了视频的实时互动，从而使视频内容与新的对话中的对话进行了积极的了解，从而使他们对对话进行了良好的对话。为了评估生成的对话，我们以高准确性，可解释性和可靠性提出了多个跨性评估基准，这证明了电视拨号对自收集数据集的有效性，直接使用现有LLMS而不是直接使用。广泛的实验表明，TV-Dialogue可以在没有培训的情况下以零拍的方式生成任何长度和任何主题的视频对话。我们的发现强调了电视数据元素在各种应用中的潜力，例如视频重新创建，电影配音及其在下游多模式任务中的使用。

### Mitigating Object Hallucinations in Large Vision-Language Models via Attention Calibration 
[[arxiv](https://arxiv.org/abs/2502.01969)] [[cool](https://papers.cool/arxiv/2502.01969)] [[pdf](https://arxiv.org/pdf/2502.01969)]
> **Authors**: Younan Zhu,Linwei Tao,Minjing Dong,Chang Xu
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 通过注意力校准在大视觉模型中缓解对象幻觉
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 大型视觉模型（LVLM）具有令人印象深刻的多模式推理能力，但仍然容易受到对象幻觉的影响，其中模型会产生与视觉内容不符的响应。最近的作品将此问题归因于LVLM的固有偏见，在该问题中，视觉令牌注意图与空间位置具有固定的相关性，并建议通过重新排序视觉令牌来减轻此问题。但是，我们发现不同的LVLM在注意力和空间位置之间表现出不同的相关性，这使得现有的解决方案难以推广到其他LVLM。为了解决这个问题，我们首先引入了无训练的解决方案，即均匀的注意校准（UAC），该解决方案估算了单个毫无意义的输入图像的偏见，并应用了校准矩阵以纠正注意力失衡。为了进一步减轻偏见，我们放宽了UAC中单个毫无意义输入的假设，并引入微调解决方案，动态注意校准（DAC），该解决方案在对象通过插件模块中放置在图像中的任何位置。跨多个基准测试的全面实验表明，UAC和DAC显着减少了物体幻觉，同时改善了一般的多模式比对。我们的方法在各种指标上实现了各种LVLM架构的最新性能。

### MATCNN: Infrared and Visible Image Fusion Method Based on Multi-scale CNN with Attention Transformer 
[[arxiv](https://arxiv.org/abs/2502.01959)] [[cool](https://papers.cool/arxiv/2502.01959)] [[pdf](https://arxiv.org/pdf/2502.01959)]
> **Authors**: Jingjing Liu,Li Zhang,Xiaoyang Zeng,Wanquan Liu,Jianhua Zhang
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: MATCNN：基于多尺度CNN的红外且可见的图像融合方法，带有注意变压器
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管基于注意力的方法在增强图像融合并解决了长期特征依赖性所带来的挑战方面表现出了很大的进步，但由于缺乏多样化的接受场提取技术，它们在捕获局部特征方面的效力受到了损害。为了克服现有融合方法提取多尺度局部特征并保留全球特征的缺点，本文提出了一种基于带有注意变压器（MATCNN）的多尺度卷积神经网络的新型跨模式图像融合方法。 MATCNN利用多尺度融合模块（MSFM）在不同尺度上提取本地特征，并采用全局特征提取模块（GFEM）提取全局特征。结合两者可以减少细节特征的丧失，并提高全局特征表示的能力。同时，信息掩码用于标记图像中的相关细节，旨在增强在融合图像中可见图像中的红外图像和背景纹理中保留重要信息的比例。随后，开发了一种新颖的优化算法，利用掩码来通过内容的集成，结构相似性指数测量和全局特征损失来指导特征提取。定量和定性评估均在各个数据集中进行，表明MATCNN有效地突出了红外的显着目标，可在可见图像中保留其他细节，并在跨模式图像中获得更好的融合结果。 MATCNN的代码将在https://github.com/zhang3849/matcnn.git上找到。

### DAMA: Data- and Model-aware Alignment of Multi-modal LLMs 
[[arxiv](https://arxiv.org/abs/2502.01943)] [[cool](https://papers.cool/arxiv/2502.01943)] [[pdf](https://arxiv.org/pdf/2502.01943)]
> **Authors**: Jinda Lu,Junkang Wu,Jinghan Li,Xiaojun Jia,Shuo Wang,YiFan Zhang,Junfeng Fang,Xiang Wang,Xiangnan He
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: DAMA：多模式LLM的数据和模型意识对齐
- **领域**: 计算机视觉和模式识别
- **摘要**: 直接偏好优化（DPO）显示出在将多模式大型语言模型（MLLM）与人类偏好相结合的有效性。但是，现有方法对变化硬度的数据表现出不平衡的响应能力，倾向于过度贴上易于播放的数据，同时在难以透视的数据上不适合使用。在本文中，我们建议数据和模型感知的DPO（DAMA）从两个关键方面动态调整优化过程：（1）一种包含数据硬度的数据感知策略，以及（2）一种集成实时模型响应的模型感知策略。通过结合两种策略，Dama使该模型能够有效地适应具有不同硬度水平的数据。对五个基准测试的广泛实验表明，达玛不仅显着提高了可信赖性，而且还提高了对一般任务的有效性。例如，在对象半座上，我们的DAMA-7B分别将响应级别和提及的幻觉降低了90.0％和95.3％，超过了GPT-4V的性能。

### Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.01576)] [[cool](https://papers.cool/arxiv/2502.01576)] [[pdf](https://arxiv.org/pdf/2502.01576)]
> **Authors**: Hashmat Shadab Malik,Fahad Shamshad,Muzammal Naseer,Karthik Nandakumar,Fahad Khan,Salman Khan
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: Under Review
- **标题**: 强大的llava：关于多模式大语言模型的大规模强大图像编码器的有效性
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式大型语言模型（MLLM）在视觉任务中出色，但仍然容易受到视觉对抗性扰动的影响，这些扰动可以引起幻觉，操纵响应或绕过安全机制。现有的方法试图通过将受限的对抗微调应用于剪辑视觉编码器上，以确保保留其概括能力，以减轻这些风险。但是，这种有限的对抗训练限制了鲁棒性和更广泛的概括。在这项工作中，我们探讨了一种替代方法，以利用在大规模数据上对对手进行对抗进行对抗的现有视觉分类模型。我们的分析揭示了两个主要贡献：（1）对抗性预训练的广泛规模和多样性使这些模型能够表现出对各种对抗性威胁的优势鲁棒性，从不可察觉的扰动到先进的越狱尝试到先进的越狱尝试，到不需要其他型号的近距离型号，以及（2）与这些型号的构图以及（2）适用于这些型号的型模型，以及（2）适用于这些型模型，（2）在复杂的推理任务上胜过现有的插件方法。通过在视觉提问，图像字幕和狱卒攻击之间进行系统的评估，我们证明了经过这些强大模型训练的MLLM具有出色的对抗性鲁棒性，同时保持有利的清洁性能。我们的框架分别在字幕和VQA任务中达到了2倍和1.5倍的平均鲁棒性增长，并且针对越狱袭击的攻击可提供超过10％的提高。代码和预估计的模型将在https://github.com/hashmatshadab/robust-llava上找到。

### VisTA: Vision-Text Alignment Model with Contrastive Learning using Multimodal Data for Evidence-Driven, Reliable, and Explainable Alzheimer's Disease Diagnosis 
[[arxiv](https://arxiv.org/abs/2502.01535)] [[cool](https://papers.cool/arxiv/2502.01535)] [[pdf](https://arxiv.org/pdf/2502.01535)]
> **Authors**: Duy-Cat Can,Linh D. Dang,Quang-Huy Tang,Dang Minh Ly,Huong Ha,Guillaume Blanc,Oliver Y. Chén,Binh T. Nguyen
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: Vista：使用多模式数据进行循证，可靠和可解释的阿尔茨海默氏病诊断的视觉文本对准模型与对比度学习
- **领域**: 计算机视觉和模式识别,计算语言学,定量方法
- **摘要**: 目的：使用高维放射学图像评估阿尔茨海默氏病（AD）在临床上很重要，但具有挑战性。尽管人工智能（AI）已经进行了AD诊断，但尚不清楚如何设计具有可预测性和解释性的AI模型。在这里，我们提出了Vista，这是一种由对比度学习辅助的多模式语言视觉模型，以优化疾病预测和循证基于循证的，可解释的临床决策解释。方法：我们开发了用于AD诊断的Vista（视觉文本比对模型）。在建筑上，我们从BiomedClip中建造了Vista，并使用对比度学习对其进行了微调，以使图像与经过验证的异常及其描述对齐。为了训练Vista，我们使用了包含图像，异常类型和医学专家验证的描述的构建参考数据集。 Vista产生四个输出：预测的异常类型，与参考案例相似，证据驱动的解释和最终的AD诊断。为了说明Vista的功效，我们报告了异常检索和痴呆预测的准确度指标。为了证明Vista的解释性，我们将其解释与人类专家的解释进行了比较。结果：与用于基线预处理的1500万张图像相比，Vista仅使用170个样品进行微调，并获得了异常检索和痴呆预测的显着改善。对于异常检索，Vista的精度达到74％，AUC为0.87（基线模型分别为26％和0.74）。对于痴呆预测，Vista的精度达到了88％，AUC分别为0.82（基线模型分别为30％和0.57）。生成的解释与人类专家的强烈一致，并提供了对诊断过程的见解。总的来说，Vista优化了预测，临床推理和解释。

### Efficiently Integrate Large Language Models with Visual Perception: A Survey from the Training Paradigm Perspective 
[[arxiv](https://arxiv.org/abs/2502.01524)] [[cool](https://papers.cool/arxiv/2502.01524)] [[pdf](https://arxiv.org/pdf/2502.01524)]
> **Authors**: Xiaorui Ma,Haoran Xie,S. Joe Qin
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: 28 pages, 3 figures
- **标题**: 有效地将大型语言模型与视觉感知整合：从训练范式角度来看的调查
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,机器学习
- **摘要**: 视觉语言方式的整合一直是多模式学习的重点，传统上依赖于视觉识别的模型。但是，随着大型语言模型（LLM）的出现，朝着将LLM与视觉方式结合在一起的明显转变。此后，将视力模式纳入LLM的训练范例已经发展。最初，该方法是通过预处理模态积分器（名为单阶段调整）来整合模式。此后，它已将其分支为重点是增强性能的方法，称为两阶段调整，以及那些优先级参数效率（称为直接适应）。但是，现有的调查主要通过两阶段的调整来解决最新的大语言模型（VLLM），从而差距了解训练范式的演变及其独特的参数有效考虑因素。本文对顶级会议，期刊和高度引用的Arxiv论文进行了分类和审查，从训练范式的角度来调整了参数效率。我们首先介绍LLM和参数效率学习方法的架构，然后讨论视觉编码器和模态积分器的全面分类学。然后，我们回顾了三个培训范例及其效率注意事项，总结了VLLM领域的基准。为了更深入了解其在参数效率上的有效性，我们比较并讨论了代表模型的实验结果，其中重复了直接适应范式的实验。该调查提供了有关最新发展和实际用途的见解，是研究人员和从业者有效地整合到LLM中的重要指南。

### End-to-end Training for Text-to-Image Synthesis using Dual-Text Embeddings 
[[arxiv](https://arxiv.org/abs/2502.01507)] [[cool](https://papers.cool/arxiv/2502.01507)] [[pdf](https://arxiv.org/pdf/2502.01507)]
> **Authors**: Yeruru Asrar Ahmed,Anurag Mittal
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 使用双文本嵌入的文本对图像合成的端到端培训
- **领域**: 计算机视觉和模式识别
- **摘要**: 文本对图像（T2I）合成是一项具有挑战性的任务，需要对两种模态（即文本和图像）之间进行建模复杂的相互作用。在最近的最新方法中采用的一种常见框架来实现这种多模式相互作用，是通过使用对比度损失训练的预训练的图像对准文本嵌入式的学习过程来引导学习过程。此外，这些嵌入通常经过一般训练，并在各种合成模型中重复使用。相比之下，我们探讨了一种学习文本嵌入的方法，专门针对以端到端方式训练的T2i合成网络。此外，我们结合了生成性和对比度训练，并使用两个嵌入式，一种优化以增强生成的图像的照片真实性，另一种是寻求捕获文本对象对齐的。在三个文本到图像基准数据集（Oxford-102，Caltech-UCSD和MS-Coco）上进行的一系列实验表明，与使用共享的方法相比，具有两个单独的嵌入比使用共享的方法相比，具有更好的结果，并且与使用鉴定方法使用预先训练的文本培训的方法相比，这种方法具有良好的作用。最后，我们证明了这种学习的嵌入也可以在其他情况下使用，例如文本对象操作。

### Deep Unfolding Multi-modal Image Fusion Network via Attribution Analysis 
[[arxiv](https://arxiv.org/abs/2502.01467)] [[cool](https://papers.cool/arxiv/2502.01467)] [[pdf](https://arxiv.org/pdf/2502.01467)]
> **Authors**: Haowen Bai,Zixiang Zhao,Jiangshe Zhang,Baisong Jiang,Lilun Deng,Yukun Cui,Shuang Xu,Chunxia Zhang
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: Accepted in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2024
- **标题**: 通过归因分析深入展开多模式图像融合网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式图像融合将来自多个源的信息综合为单个图像，从而促进了下游任务，例如语义分割。当前的方法主要集中于通过复杂的映射在视觉显示层上获取信息融合图像。尽管某些方法试图共同优化图像融合和下游任务，但这些努力通常缺乏直接的指导或相互作用，仅供预定义融合损失。为了解决这个问题，我们提出了一个``展开归因分析融合网络''（UAAFUSION），使用归因分析来更有效地量身定制融合图像以进行语义分割，从而增强了融合和分割之间的相互作用。具体而言，我们利用归因分析技术来探索源图像中语义区域对任务歧视的贡献。同时，我们的融合算法结合了源图像中更有益的特征，从而使分割可以指导融合过程。我们的方法构建了一个模型驱动的展开网络，该网络使用从归因分析得出的优化目标，其归因融合损失是根据分割网络的当前状态计算得出的。我们还为归因分析开发了一个新的途径功能，该功能专门针对我们展开的网络中的融合任务量身定制。在每个网络阶段都集成了归因注意机制，从而使融合网络可以优先考虑领域和像素对于高级识别任务至关重要。此外，为了减轻传统展开网络中的信息损失，还将内存增强模块合并到我们的网络中，以改善各个网络层的信息流。广泛的实验证明了我们的方法在图像融合和对语义分割的适用性方面的优势。

### Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.01419)] [[cool](https://papers.cool/arxiv/2502.01419)] [[pdf](https://arxiv.org/pdf/2502.01419)]
> **Authors**: Mingi Jung,Saehuyng Lee,Eunji Kim,Sungroh Yoon
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: :I.2.7
- **标题**: 视觉关注永远不会逐渐消失：选择性渐进的注意重新校准用于多模式模型中详细图像字幕的详细图像字幕
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 详细的图像字幕对于数据生成和有助于视觉障碍的个体等任务至关重要。高质量的标题需要在精确度和召回之间保持平衡，这对于当前的多模式模型（MLLM）仍然具有挑战性。在这项工作中，我们假设这种限制源于随着响应的延长而减弱和嘈杂的视觉关注。为了解决这个问题，我们提出了SPARC（选择性进行性注意重新校准），这是一种无训练的方法，可增强解码过程中视觉令牌的贡献。 SPARC建立在三个主要观察结果上：（1）增加所有视觉令牌的影响降低了回忆；因此，SPARC有选择地放大视觉令牌。 （2）随着字幕的延长，视觉注意力变得更加嘈杂，因此SPARC通过利用跨时间步骤的注意力差异来识别关键的视觉令牌； （3）随着视觉注意力逐渐减弱，SPARC加强了它以保护其影响。我们的实验均包含自动化和人类评估，表明现有方法以召回成本提高了MLLM的精度。相反，我们提出的方法通过最小的计算开销增强了精度和回忆。

### The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles 
[[arxiv](https://arxiv.org/abs/2502.01081)] [[cool](https://papers.cool/arxiv/2502.01081)] [[pdf](https://arxiv.org/pdf/2502.01081)]
> **Authors**: Vernon Y. H. Toh,Yew Ken Chia,Deepanway Ghosal,Soujanya Poria
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 跳跃推理曲线？跟踪GPT- [N]和O- [N]模型中推理性能的演变
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: OpenAI的O1和O3的释放标志着大型语言模型向高级推理能力的重大范式转变。值得注意的是，在新颖的问题解决和技能获取方面，O3在人工通用智能（ARC-AGI）方面的表现优于人类。但是，该基准限于象征性模式，而人类通常会感知和有关涉及视觉和语言数据的多模式场景的理由。因此，迫切需要研究多模式任务中的高级推理能力。为此，我们跟踪有关挑战多模式难题的GPT- [n]和O- [n]系列模型的演变，需要使用抽象或算法推理进行细粒度的视觉感知。 O1的出色性能是GPT-4O的计算成本的近750倍，这引起了人们对其效率的担忧。我们的结果揭示了跨模型迭代的推理能力的清晰趋势，其性能在GPT系列模型中以及随后在O1中的表现出色。尽管如此，我们观察到O1模型仍然在需要抽象推理的简单多模式难题中挣扎。此外，它在算法难题中的性能仍然很差。我们计划在本文中不断跟踪新模型，并在本文中更新我们的结果。此评估中使用的所有资源均可公开可用https://github.com/declare-lab/llm-puzzletest。

### Mitigating Hallucinations in Large Vision-Language Models with Internal Fact-based Contrastive Decoding 
[[arxiv](https://arxiv.org/abs/2502.01056)] [[cool](https://papers.cool/arxiv/2502.01056)] [[pdf](https://arxiv.org/pdf/2502.01056)]
> **Authors**: Chao Wang,Xuancheng Zhou,Weiwei Fu,Yang Zhou
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 通过基于内部事实对比解码的大型视力语言模型减轻幻觉
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 大型视觉语言模型（LVLM）整合了视觉和语言方式，在各种多模式任务中表现出卓越的性能。然而，LVLM仍然容易受到对象幻觉问题的影响。以前的减轻此问题的努力集中在监督的微调（SFT）或纳入外部知识上，这两者都需要与培训和收购外部数据有关的重大成本。为了应对这些挑战，我们提出了一种新颖的模型不足的方法，称为内部事实对比解码（IFCD），旨在通过利用LVLMS自己的幻觉来减轻和抑制LVLMS的推理过程中的幻觉。 IFCD基于实验观察结果，即对LVLM的内部表示的改变往往会扩大由语言偏见引起的幻觉。通过对比的是干扰分布，IFCD校准了LVLM的输出，并有效地从最终预测中删除了幻觉逻辑。实验结果证明，IFCD显着减轻对象级别和属性级别的幻觉，同时分别与直接解码相比，教皇的平均准确性提高了9％的准确性，而MME对象幻觉子集的准确度提高了8％。

### FCBoost-Net: A Generative Network for Synthesizing Multiple Collocated Outfits via Fashion Compatibility Boosting 
[[arxiv](https://arxiv.org/abs/2502.00992)] [[cool](https://papers.cool/arxiv/2502.00992)] [[pdf](https://arxiv.org/pdf/2502.00992)]
> **Authors**: Dongliang Zhou,Haijun Zhang,Jianghong Ma,Jicong Fan,Zhao Zhang
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: This paper has been accepted for presentation at ACM Multimedia 2023
- **标题**: FCBOOST-NET：一种生成网络，用于通过时尚兼容性增强多个共处服装的生成网络
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 在时尚技术领域，服装的生成是一项艰巨的任务，在该领域中，其目的是创建一组相互融合的时尚项目，以补充一组给定的物品。以前在该领域的研究仅限于根据给定的一组项目生成独特的时尚项目，而无需为用户提供其他选项。缺乏多种选择的选择需要开发更通用的框架。但是，当使用多模式图像到图像翻译方法接近生成共处和多元化服装的任务时，就非对齐的图像翻译而言，它构成了一个具有挑战性的问题，这很难用现有方法来解决。在这项研究中，我们提出了FCBoost-NET，这是一种用于服装生成的新框架，利用了预培训的生成模型的力量生产多个共处和多元化的服装。最初，FCBoost-NET随机合成了多种时尚项目，然后使用新型时尚兼容性助推器在几轮中改进合成集的兼容性。这种方法的灵感来自增强算法，并允许在多个步骤中逐渐提高性能。经验证据表明，提出的策略可以改善随机合成时尚项目的时尚兼容性，并保持其多样性。广泛的实验证实了我们提出的框架在视觉真实性，多样性和时尚兼容性方面的有效性。

### CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling 
[[arxiv](https://arxiv.org/abs/2502.00965)] [[cool](https://papers.cool/arxiv/2502.00965)] [[pdf](https://arxiv.org/pdf/2502.00965)]
> **Authors**: Xinze Wang,Chen Chen,Yinfei Yang,Hong-You Chen,Bowen Zhang,Aditya Pal,Xiangxin Zhu,Xianzhi Du
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 剪辑：一种简单有效的Experts剪辑培训配方，稀疏的升级配方
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 专家（MOE）模型的混合物对于在控制推理成本的同时对缩放模型的容量至关重要。在将MOE集成到诸如夹子之类的多模型中，可以提高性能，但众所周知，培训这些模型具有挑战性且昂贵。我们提出了剪辑剪辑（剪辑），这是一种有效的替代培训策略，可将预先训练的密集夹模型转换为稀疏的MoE体系结构。通过对各种环境和辅助损失进行广泛的实验，我们证明了剪辑可显着降低训练的复杂性和成本。值得注意的是，我们稀疏的剪辑B/16型号，经过剪辑训练，在可可和Flickr30k文本对图像中分别以1个基准测试，其密集量优于其密集的7.2％和6.6％。它甚至超过了此任务上较大的夹子L/14模型，同时仅使用30％的推理失败。我们进一步证明了跨不同尺度的培训配方的普遍性，从而确立了稀疏的升级，作为一种实用且可扩展的方法，用于构建有效的高性能剪辑模型。

### SAM-guided Pseudo Label Enhancement for Multi-modal 3D Semantic Segmentation 
[[arxiv](https://arxiv.org/abs/2502.00960)] [[cool](https://papers.cool/arxiv/2502.00960)] [[pdf](https://arxiv.org/pdf/2502.00960)]
> **Authors**: Mingyu Yang,Jitong Lu,Hun-Seok Kim
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: ICRA 2025
- **标题**: SAM指导的伪标签增强多模式3D语义分割
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式3D语义细分对于诸如自动驾驶和虚拟现实（VR）等应用至关重要。为了有效地在实际情况下部署这些模型，必须采用跨域适应技术来弥合训练数据和现实世界数据之间的差距。最近，使用伪标签的自我训练已成为多模式3D语义分割中跨域适应的主要方法。但是，生成可靠的伪标签需要严格的约束，这通常会导致修剪后稀疏的伪标记。这种稀疏性可能会在适应过程中妨碍性能提高。我们提出了一种图像引导的伪标签增强方法，该方法利用了段中的任何模型（SAM）来介绍更可靠的伪标记，从而提高了域的适应性。具体而言，给定一个3D点云和SAM掩码从其配对的图像数据中，我们收集了每个SAM蒙版覆盖的所有3D点，这些3D点可能属于同一对象。然后，我们的方法分为两个步骤来完善每个SAM蒙版内的伪标签。首先，我们使用多数投票确定每个掩码的类标签，并采用各种约束来过滤不可靠的掩码标签。接下来，我们介绍几何学感知的渐进式传播（GAPP），该传播将蒙版标签传播到SAM蒙版内的所有3D点，同时避免由2d-3d未对准引起的异常值。跨多个数据集和域适应方案进行的实验表明，我们提出的方法显着增加了高质量伪标签的数量，并提高了基线方法的适应性性能。

### Vision and Language Reference Prompt into SAM for Few-shot Segmentation 
[[arxiv](https://arxiv.org/abs/2502.00719)] [[cool](https://papers.cool/arxiv/2502.00719)] [[pdf](https://arxiv.org/pdf/2502.00719)]
> **Authors**: Kosuke Sakurai,Ryotaro Shimizu,Masayuki Goto
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: 8 pages, 2 figures
- **标题**: 视觉和语言参考提示到SAM进行几次分段
- **领域**: 计算机视觉和模式识别
- **摘要**: 细分任何模型（SAM）代表一个大规模的分割模型，该模型可以通过灵活的提示来实现强大的零击功能。虽然SAM可以将任何对象分割为零，但它需要为每个目标图像提供用户提供的提示，并且不会将任何标签信息附加到掩模。很少有射击分段模型通过将带注释的参考图像作为提示来解决这些问题，并可以在没有用户提供的提示的情况下将特定对象分割为特定对象。以前的基于SAM的少数分段模型仅使用带注释的参考图像作为提示，因此由于缺乏参考信息而导致精度有限。在本文中，我们提出了一个新颖的几片分段模型，视觉和语言参考提示提示（VLP-SAM），该模型通过不仅输入图像，还将语言作为参考信息来利用参考图像的视觉信息和文本标签的语义信息。特别是，VLP-SAM是一种简单且可扩展的结构，具有最小的可学习参数，它使用多模式视觉语言模型输入带有视觉信息信息的嵌入到SAM中。为了证明VLP-SAM的有效性，我们在Pascal-5i和CoCo-20i数据集上进行了实验，并在少数弹片分割任务中实现了高性能，以优于先前的最先进模型（分别为MIOU中的6.3％和9.5％）。此外，VLP-SAM在训练数据中未包含的看不见对象中演示了其普遍性。我们的代码可在https://github.com/kosukesakurai1/vlp-sam上找到。

### PhiP-G: Physics-Guided Text-to-3D Compositional Scene Generation 
[[arxiv](https://arxiv.org/abs/2502.00708)] [[cool](https://papers.cool/arxiv/2502.00708)] [[pdf](https://arxiv.org/pdf/2502.00708)]
> **Authors**: Qixuan Li,Chao Wang,Zongjin He,Yan Peng
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: 13 pages.8 figures
- **标题**: PHIP-G：物理引导的文本到3D构图场景生成
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在2D扩散先验的监督下，文本到3D资产的产生已实现了显着优化。但是，在处理构图场景时，现有方法遇到了几个挑战：1）。无法确保复合场景的布局符合物理定律； 2）。难以准确捕获复杂场景描述中描述的资产和关系； 3）。布局方法中有限的自主资产产生能力利用大型语言模型（LLMS）。为了避免这些妥协，我们为构图场景生成PHIP-G提出了一个新颖的框架，该框架将生成技术与基于世界模型的布局指导无缝整合。 PHIP-G利用基于LLM的代理分析了复杂的场景描述以生成场景图，并集成了多模式2D代理和用于创建目标资产的3D高斯生成方法。对于布局的阶段，PHIP-G采用具有粘附功能的物理池和视觉监督代理，为布局预测和计划形成了世界模型。广泛的实验表明，PHIP-G可以显着提高构图场景的产生质量和物理合理性。值得注意的是，PHIP-G在剪辑分数中达到最先进的表现（SOTA），与T $^3 $台式衡量的生成质量的领先方法相均达到了平等，并提高了24倍的效率。

### TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion 
[[arxiv](https://arxiv.org/abs/2502.00695)] [[cool](https://papers.cool/arxiv/2502.00695)] [[pdf](https://arxiv.org/pdf/2502.00695)]
> **Authors**: Linglong Wu,Xuhao Shan,Ruiquan Ge,Ruoyu Liang,Chi Zhang,Yonghong Li,Ahmed Elazab,Huoling Luo,Yunbi Liu,Changmiao Wang
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: 6 pages, 3 figures, accepted by IEEE ISBI 2025
- **标题**: TMI-CLNET：来自成像，临床和放射线数据融合的慢性肝病预后的三模式相互作用网络
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 慢性肝病是全球范围内的重大健康挑战，准确的预后评估对于个性化治疗计划至关重要。最近的证据表明，整合多模式数据，例如计算机断层扫描成像，放射线特征和临床信息，可以提供更全面的预后信息。但是，模态具有固有的异质性，并且结合其他方式可能会加剧异质数据融合的挑战。此外，现有的多模式融合方法通常很难适应更丰富的医学方式，因此很难捕获模式间关系。为了克服这些局限性，我们提出了三模式相互作用慢性肝网络（TMI-CLNET）。具体而言，我们开发一个模式内聚集模块和三模式的交叉意见融合模块，该模块旨在分别消除模式内冗余和提取交叉模式信息。此外，我们设计了一个三模式特征融合损耗函数，以使跨模态的特征表示。肝脏预后数据集的广泛实验表明，我们的方法显着优于现有的最新单峰模型和其他多模式技术。我们的代码可在https://github.com/mysterwll/liver.git上找到。

### Mitigating the Modality Gap: Few-Shot Out-of-Distribution Detection with Multi-modal Prototypes and Image Bias Estimation 
[[arxiv](https://arxiv.org/abs/2502.00662)] [[cool](https://papers.cool/arxiv/2502.00662)] [[pdf](https://arxiv.org/pdf/2502.00662)]
> **Authors**: Yimu Wang,Evelien Riddell,Adrian Chow,Sean Sedwards,Krzysztof Czarnecki
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 减轻模态差距：具有多模式原型和图像偏差估计的几乎没有分布的检测
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 现有的视觉模型（VLM）基于分布的方法（OOD）检测通常取决于输入图像和分布图（ID）文本原型之间的相似性分数。但是，图像和文本之间的方式差距通常会导致高误报率，因为OOD样品可以表现出与ID文本原型的高相似性。为了减轻这种方式差距的影响，我们建议将ID图像原型和ID文本原型合并。我们提供了理论分析和经验证据，表明这种方法在没有任何其他培训的情况下增强了基于VLM的OOD检测性能。为了进一步减少图像和文本之间的差距，我们介绍了一个新颖的几声调谐框架，至尊，包括偏见提示生成（BPG）和图像文本一致性（ITC）模块。 BPG通过根据高斯基于高斯的估计图像域偏置调节ID文本原型来增强图像文本融合并改善概括。 ITC通过最大程度地减少模式内和模式间距离来减少模态差距。此外，受我们的理论和经验发现的启发，我们介绍了一个新颖的OOD评分$ s _ {\ textit {gmp}} $，利用uni-and和cross-dodal的相似性。最后，我们提出了广泛的实验，以证明至尊始终优于现有的基于VLM的OOD检测方法。

### Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions 
[[arxiv](https://arxiv.org/abs/2502.00568)] [[cool](https://papers.cool/arxiv/2502.00568)] [[pdf](https://arxiv.org/pdf/2502.00568)]
> **Authors**: Samiran Dey,Christopher R. S. Banerji,Partha Basuchowdhuri,Sanjoy K. Saha,Deepak Parashar,Tapabrata Chakraborti
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 从癌症组织病理学产生跨模式基因表达可改善多模式AI预测
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 新兴的研究强调，基于人工智能的数字病理和转录组特征的多模式融合可以改善癌症诊断（分级/亚型）和预后（生存风险）预测。但是，在实际临床环境中，这种直接融合的联合决策是不切实际的，因为组织病理学仍然是诊断和转录组检验的黄金标准，至少在公共医疗保健系统中。借助我们新型基于扩散的跨模式生成AI模型途径，我们表明，由数字组织病理学合成的基因组表达共同预测癌症的分级和患者的生存风险，具有高精度（最先进的性能），确定性（通过结构覆盖的保证）和可解释性（通过分布式注意图）。 Pathgen Code可以通过GitHub通过https://github.com/samiran-dey/pathgen进行开放使用。

### Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition 
[[arxiv](https://arxiv.org/abs/2502.00547)] [[cool](https://papers.cool/arxiv/2502.00547)] [[pdf](https://arxiv.org/pdf/2502.00547)]
> **Authors**: Zaitian Wang,Jian He,Yu Liang,Xiyuan Hu,Tianhao Peng,Kaixin Wang,Jiakai Wang,Chenlong Zhang,Weili Zhang,Shuang Niu,Xiaoyang Xie
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 米尔默：一个基于多个实例学习的多模式情感识别的框架
- **领域**: 计算机视觉和模式识别,人工智能,人机交互
- **摘要**: 情绪在人类的行为和决策中起着至关重要的作用，使情感识别成为人类计算机互动（HCI）的关键领域。这项研究通过将面部表达分析与脑电图（EEG）信号整合在一起，从而解决了情绪识别的挑战，并引入了一种新颖的多模式框架。所提出的框架采用基于变压器的融合方法来有效整合视觉和生理方式。它由脑电图预处理模块，面部特征提取和平衡模块以及跨模式融合模块组成。为了增强视觉特征提取，我们在与情绪相关的数据集上微调了预训练的SWIN变压器。此外，引入了跨注意机制，以平衡跨模式的令牌表示，从而确保有效的特征整合。这项工作的关键创新是采用多个实例学习（MIL）方法，该方法随着时间的推移从多个面部表达图像中提取有意义的信息，从而捕获了以前的研究中经常忽略的关键时间动态。在DEAP数据集上进行的广泛实验证明了所提出的框架的优越性，在四类情绪识别任务中达到了96.72％的分类精度。消融研究进一步验证了每个模块的贡献，强调了高级特征提取和融合策略在增强情绪识别性能方面的重要性。我们的代码可在https://github.com/liangyubuaa/milmer上找到。

### TEST-V: TEst-time Support-set Tuning for Zero-shot Video Classification 
[[arxiv](https://arxiv.org/abs/2502.00426)] [[cool](https://papers.cool/arxiv/2502.00426)] [[pdf](https://arxiv.org/pdf/2502.00426)]
> **Authors**: Rui Yan,Jin Wang,Hongyu Qu,Xiaoyu Du,Dong Zhang,Jinhui Tang,Tieniu Tan
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 测试-V：零摄像视频分类的测试时间支持集调整
- **领域**: 计算机视觉和模式识别
- **摘要**: 最近，通过将视觉语言模型（VLMS）调整为零射击视觉分类，通过使用一些提示（测试时间提示调音，TPT）嵌入类别的类别分类或用生成的视觉样本（支持集）替换类名称（support-support-Set）显示出令人鼓舞的结果。但是，TPT无法避免模式之间的语义差距，而支持集不能调整支持集。为此，我们借鉴了彼此的优势，并提出了一个新颖的框架，即用于测试时间支持集对零拍摄视频分类（Test-V）的调整。它首先使用多个提示（多启发支持集扩张，MSD）扩张支持集，然后通过可学习的权重侵蚀支持集，以动态地挖掘密钥线索（时间感知支持集合侵蚀，TSE）。具体而言，i）MSD根据LLM的多个提示来扩展每个类的支持样本，以丰富支持集的多样性。 ii）TSE根据时间预测的一致性以一种自我监督的方式调整了可分解的可学习权重，以挖掘每个班级的关键支持线索。 $ \ textbf {test-v} $在四个基准中实现了最新的结果，并且可以很好地解释支持集的扩张和侵蚀。

### MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization 
[[arxiv](https://arxiv.org/abs/2502.00425)] [[cool](https://papers.cool/arxiv/2502.00425)] [[pdf](https://arxiv.org/pdf/2502.00425)]
> **Authors**: JiangYong Yu,Sifan Zhou,Dawei Yang,Shuo Wang,Shuoyu Li,Xing Hu,Chen Xu,Zukang Xu,Changyong Shu,Zhihang Yuan
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: First quantization solution forMultimodallarge language models applicable to 5 mainstream MLLMs
- **标题**: mquant：通过完整的静态量化释放多模式大语言模型的推理潜力
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式的大型语言模型（MLLM）由于能够理解多模式输入的能力而引起了广泛的关注。但是，它们的较大参数大小和大量的计算需求严重阻碍了其实际部署和应用。虽然量化是减少模型大小和推理潜伏期的有效方法，但其对MLLM的应用仍未得到充分激励。在本文中，我们提出了Mquant，这是一种训练后量化（PTQ）框架，旨在应对多模式大型语言模型（MLLM）的独特挑战。传统的量化通常与MLLM斗争，因为（a）来自大型视觉令牌计数，（b）视觉和文本令牌之间的分布差异，以及（c）基于Hadamard的转换引入的极端异常值。为了解决这些问题，Mquant介绍了：特定于模态的静态量化（MSQ），为视觉和文本令牌分配了不同的静态尺度；注意不变的灵活开关（AIF），重新排序令牌以保持随意的注意，同时消除昂贵的令牌计算；旋转幅度抑制（RMS），减轻在线Hadamard旋转引起的重量异常值。在五个主流MLLM（包括QWEN-VL，Minicpm-V，COGVLM2）上，W4A8下的MQuant达到了接近浮动点的准确性（<1％降解），同时将推断潜伏期降低高达30％，极大地超过了现有的PTQ碱基。我们的Mquant有效地弥合了差距，以在资源约束设备中有效，准确的MLLM推断。代码将发布。

### Embodied Intelligence for 3D Understanding: A Survey on 3D Scene Question Answering 
[[arxiv](https://arxiv.org/abs/2502.00342)] [[cool](https://papers.cool/arxiv/2502.00342)] [[pdf](https://arxiv.org/pdf/2502.00342)]
> **Authors**: Zechuan Li,Hongshan Yu,Yihao Ding,Yan Li,Yong He,Naveed Akhtar
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: Work in progress
- **标题**: 3D理解的体现智能：关于3D场景问题的调查回答
- **领域**: 计算机视觉和模式识别
- **摘要**: 3D场景问题回答（3D SQA）代表了一个跨学科的任务，该任务集成了3D视觉感知和自然语言处理，从而赋予智能代理以理解和与复杂的3D环境进行交互。大型多模式建模的最新进展推动了不同数据集的创建，并刺激了3D SQA的指令调整和零摄像方法的开发。但是，这种快速的进步引入了挑战，尤其是在跨数据集和基线的统一分析和比较时。本文介绍了对3D SQA的首次全面调查，系统地审查了数据集，方法和评估指标，同时着重强调了数据集标准化，多模式融合和任务设计的关键挑战和未来机会。

### INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation 
[[arxiv](https://arxiv.org/abs/2502.00262)] [[cool](https://papers.cool/arxiv/2502.00262)] [[pdf](https://arxiv.org/pdf/2502.00262)]
> **Authors**: Dianwei Chen,Zifan Zhang,Yuchen Liu,Xianfeng Terry Yang
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 洞察力：通过视觉语言模型在上下文感知危险检测和边缘案例评估上增强自主驾驶安全性
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 自主驾驶系统在处理不可预测的边缘场景时面临重大挑战，例如对抗性行人运动，危险的车辆操纵和突然的环境变化。由于传统检测和预测方法的局限性，当前的端到端驾驶模型与对这些罕见事件的概括为努力。为了解决这个问题，我们提出了洞察力（用于广义危害跟踪的语义和视觉输入的集成），这是一个层次视觉模型（VLM）框架，旨在增强危害检测和边缘评估。通过使用多模式数据融合，我们的方法可以整合语义和视觉表示形式，从而可以精确解释驾驶场景并准确预测潜在危险。通过对VLM的监督微调，我们使用基于注意的机制和坐标回归技术来优化空间危害定位。 BDD100K数据集的实验结果表明，与现有模型相比，危险预测的直接性和准确性有很大的改善，从而实现了概括性能的显着提高。这一进步增强了自主驾驶系统的鲁棒性和安全性，确保了在复杂的现实世界情景下的情境意识和潜在决策。

### AIN: The Arabic INclusive Large Multimodal Model 
[[arxiv](https://arxiv.org/abs/2502.00094)] [[cool](https://papers.cool/arxiv/2502.00094)] [[pdf](https://arxiv.org/pdf/2502.00094)]
> **Authors**: Ahmed Heakl,Sara Ghaboura,Omkar Thawkar,Fahad Shahbaz Khan,Hisham Cholakkal,Rao Muhammad Anwer,Salman Khan
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: 20 pages, 16 figures, ACL
- **标题**: AIN：阿拉伯语包含大型多模型
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,人机交互,机器学习
- **摘要**: 在大型语言模型（LLM）及其发展到大型多模型（LMM）中的迅速发展中，已经以英语和中文等高资源语言取得了重大进步。尽管阿拉伯语LLMS已经取得了显着的进步，但阿拉伯语LMM在很大程度上尚未探索，通常狭窄地专注于语言和视觉理解的一些特定方面。为了弥合这一差距，我们介绍了Ain-ain-Arabial包含的多模型设计，该模型设计为在各个领域中出色。 Ain是一种英语双语LMM，旨在用英语和阿拉伯语出色，利用精心构造的360万个高质量的阿拉伯语英语多模式数据样本。 Ain展示了最先进的阿拉伯表现，同时还具有强大的英语视觉能力。在最近的骆驼基准基准中，包括38个子域，包括多图像理解，复杂的视觉感知，手写文档理解，视频理解，医学成像，植物性疾病和基于遥感的土地使用理解，我们的AIN表现出7b的强劲性能，与7B模型的GPT-4O具有强大的3.4％domains和38个domains和38个Subains和38 subains和38 subains的绝对增益。 AIN的出色能力将其定位为迈向使用各种应用程序中先进的多模式AI工具来赋予阿拉伯语扬声器的重要一步。

### CerraData-4MM: A multimodal benchmark dataset on Cerrado for land use and land cover classification 
[[arxiv](https://arxiv.org/abs/2502.00083)] [[cool](https://papers.cool/arxiv/2502.00083)] [[pdf](https://arxiv.org/pdf/2502.00083)]
> **Authors**: Mateus de Souza Miranda,Ronny Hänsch,Valdivino Alexandre de Santiago Júnior,Thales Sehn Körting,Erison Carlos dos Santos Monteiro
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: 9 pages, 13 Figures, 3 tables
- **标题**: Cerradata-4mm：塞拉多（Cerrado
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 尽管诸如类不平衡和视觉上类似的类别等挑战，但Cerrado面临着日益增加的环境压力，需要准确的土地利用和土地覆盖（LULC）映射。为了解决这个问题，我们提出了Cerradata-4mm，这是一种结合Sentinel-1合成孔径雷达（SAR）和Sentinel-2多光谱图（MSI）的多模式数据集（MSI），并具有10M空间分辨率。该数据集分别包括两个分类分类级别，分别为7个和14个类，重点关注多样化的bico do papagaio生态区。我们通过评估标准的U-NET和更复杂的视觉变压器（VIT）模型来强调Cerradata-4MM基准测试高级语义分割技术的能力。 VIT在多模式方案中取得了卓越的性能，最高的宏观F1得分为57.60％，平均相交比联合（MIOU）在第一个分层级别为49.05％。这两种模型都在少数群体中挣扎，尤其是在第二个分层级别，在该水平上，U-NET的性能下降到F1得分为18.16％。阶级平衡改善了代表性不足的班级的代表性，但降低了整体准确性，强调了加权培训的权衡。 Cerradata-4MM为推进深度学习模型来处理类不平衡和多模式数据融合提供了一个具有挑战性的基准。代码，训练有素的模型和数据可在https://github.com/ai4luc/cerradata-4mm上公开获取。

### Deep Learning-Based Facial Expression Recognition for the Elderly: A Systematic Review 
[[arxiv](https://arxiv.org/abs/2502.02618)] [[cool](https://papers.cool/arxiv/2502.02618)] [[pdf](https://arxiv.org/pdf/2502.02618)]
> **Authors**: F. Xavier Gaya-Morey,Jose M. Buades-Rubio,Philippe Palanque,Raquel Lacuesta,Cristina Manresa-Yee
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 基于深度学习的面部表情识别老年人：系统评价
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 全球人口的迅速衰老强调了需要技术支持老年人的必要性，尤其是在医疗保健和情感福祉方面。面部表情识别（FER）系统提供了一种非侵入性手段，用于监测情绪状态，并在辅助生活，心理健康支持和个性化护理中应用。这项研究对基于深度学习的FER系统进行了系统的综述，重点是他们对老年人的应用。经过严格的方法，我们分析了过去十年中发表的31项研究，以解决诸如老年人特异性数据集的稀缺性，阶级失衡以及与年龄相关的面部表达差异的影响。我们的发现表明，卷积神经网络在FER中仍然占主导地位，尤其是针对资源受限环境的轻量级版本。但是，现有数据集通常缺乏年龄表示的多样性，而现实世界的部署仍然有限。此外，隐私问题和对可解释的人工智能的需求作为收养的关键障碍。这篇评论强调了开发包含年龄的数据集，集成多模式解决方案并采用XAI技术来增强系统可用性，可靠性和可信赖性的重要性。最后，我们为将来的研究提供了建议，以弥合学术进步与老年护理实施现实世界之间的差距。

### COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation 
[[arxiv](https://arxiv.org/abs/2502.02589)] [[cool](https://papers.cool/arxiv/2502.02589)] [[pdf](https://arxiv.org/pdf/2502.02589)]
> **Authors**: Xueqing Deng,Qihang Yu,Ali Athar,Chenglin Yang,Linjie Yang,Xiaojie Jin,Xiaohui Shen,Liang-Chieh Chen
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: project website: https://xdeng7.github.io/coconut.github.io/coconut_pancap.html
- **标题**: 椰子粉：联合综合分割和接地标题，用于细粒度的理解和产生
- **领域**: 计算机视觉和模式识别
- **摘要**: 本文介绍了椰子pancap数据集，该数据集是为了增强全景分割和接地图像字幕而创建的。该数据集在带有先进的椰子泛面罩的可可数据集的基础上，旨在克服通常缺乏详细，场景全面描述的现有图像text数据集中的局限性。椰子pancap数据集结合了扎根于全景分割面罩的细粒度的区域级字幕，可确保一致性并改善生成的字幕的细节。通过人类编辑的密集注释的描述，椰子pancap支持改进视觉模型（VLMS）的培训，以了解文本对图像任务的图像理解和生成模型。实验结果表明，椰子 - 果肉可以显着提高跨理解和发电任务的性能，从而为大规模数据集提供互补的益处。该数据集设置了一个新的基准测试，用于评估联合综合分割和接地字幕任务的模型，从而解决了多模式学习中高质量，详细的图像文本注释的需求。

### LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.02406)] [[cool](https://papers.cool/arxiv/2502.02406)] [[pdf](https://arxiv.org/pdf/2502.02406)]
> **Authors**: Tzu-Tao Chang,Shivaram Venkataraman
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: LV-XATTN：多模式模型中长期视觉输入的分布式交叉注意
- **领域**: 计算机视觉和模式识别,人工智能,分布式、并行和集群计算,机器学习
- **摘要**: 跨注意事件通常在多模式大语言模型（MLLMS）中采用，以将视觉信息整合到语言骨架中。但是，在具有大型视觉输入的应用中，例如视频理解，处理跨注意层中大量的视觉令牌会导致高内存需求，并且通常需要在多个GPU上进行分布式计算。现有的分布式注意机制面临着重要的沟通开销，使得跨注意层成为有效训练和推断MLLM的关键瓶颈。为了解决这个问题，我们提出了LV-XATTN，这是一种分布式的，精确的跨注意机制，其开销很少。我们观察到，在涉及大型视觉输入的应用中，查询块的大小通常比键值块的应用小得多。因此，在LV-XATTN中，我们将大型键值块保持在每个GPU上，并在GPU上交换较小的查询块。我们还引入了一种有效的激活重新计算技术，可为更长的视觉上下文提供支持。我们从理论上分析了LV-XATTN的沟通优势，并表明它可以为广泛的模型实现加速。我们对Mplug-Owl3和OpenFlamingo模型进行的评估发现，与现有方法相比，LV-XATTN可实现高达5.58 $ \ times $端到端的速度。

### MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm 
[[arxiv](https://arxiv.org/abs/2502.02358)] [[cool](https://papers.cool/arxiv/2502.02358)] [[pdf](https://arxiv.org/pdf/2502.02358)]
> **Authors**: Ziyan Guo,Zeyu Hu,Na Zhao,De Wen Soh
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: MotionLab：统一的人类运动产生和通过运动条件运动范式编辑
- **领域**: 计算机视觉和模式识别
- **摘要**: 人类运动生成和编辑是计算机图形和视觉的关键组成部分。但是，该领域的当前方法倾向于提供针对特定任务的孤立解决方案，对于现实世界应用，这可能是效率低下且不切实际的。尽管一些努力旨在统一与运动相关的任务，但这些方法只是使用不同的方式作为指导运动的条件。因此，他们缺乏编辑功能，细粒度的控制和无法促进跨任务共享的知识共享。为了解决这些局限性，并提供了能够处理人类运动生成和编辑的多功能，统一的框架，我们引入了一种新颖的范式：运动条件运动，该范式可以通过三个概念：源运动，状况和目标运动来统一对各种任务进行统一的配方。基于此范式，我们提出了一个统一的框架MotionLab，该框架结合了整流的流以学习从源运动到目标运动的映射，并在指定条件下指导。在MotionLab中，我们介绍了1）MotionFlow Transformer，以增强没有任务特定模块的条件生成和编辑； 2）对齐旋转位置编码}，以确保源运动和目标运动之间的时间同步； 3）任务指定的指令调制； 4）运动课程学习，用于有效的多任务学习和跨任务的知识共享。值得注意的是，我们的MotionLAB表现出有希望的概括能力和对人类运动的多个基准的推理效率。我们的代码和其他视频结果可在以下网址获得：https：//diouo.github.io/motionlab.github.io/。

### VerteNet -- A Multi-Context Hybrid CNN Transformer for Accurate Vertebral Landmark Localization in Lateral Spine DXA Images 
[[arxiv](https://arxiv.org/abs/2502.02097)] [[cool](https://papers.cool/arxiv/2502.02097)] [[pdf](https://arxiv.org/pdf/2502.02097)]
> **Authors**: Zaid Ilyas,Arooba Maqsood,Afsah Saleem,Erchuan Zhang,David Suter,Parminder Raina,Jonathan M. Hodgson,John T. Schousboe,William D. Leslie,Joshua R. Lewis,Syed Zulqarnain Gilani
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: 10 pages with 7 figures
- **标题**: vertenet-多种膜下文混合CNN变压器，用于在侧脊柱DXA图像中精确的椎骨位置定位
- **领域**: 计算机视觉和模式识别
- **摘要**: 侧脊柱图像（LSI）分析对于医学诊断，治疗计划和详细的脊柱健康评估很重要。尽管通常使用诸如计算机断层扫描和数字X射线成像之类的模式，但双重能量X射线吸收仪（DXA）通常是由于降低辐射暴露，无缝捕获和成本效益而受到首选的。 LSIS上的准确椎骨位置定位（VLL）对于检测诸如Kyphosis和Lordosis等脊柱条件以及使用椎间间指南（IVGS）评估腹主动脉钙化（AAC）很重要。但是，很少有自动化的VLL方法集中在DXA LSI上。我们提出Vertenet是一种混合CNN转换器模型，该模型具有新型的双分辨率注意机制，在自我和跨注意域中，被称为双分辨率自我注意力（DRSA）和双分辨率交叉注意（DRCA）。这些机制通过在两个不同的特征图分辨率下操作DXA图像中的不同频率。此外，我们设计了一个多文本特征融合块（MCFB），该特征融合块（MCFB）使用DRSA和DRCA有效地集成了功能。我们从各种机器上对620个DXA LSI进行训练，与现有方法相比，取得了卓越的结果。我们还设计了一种算法，该算法利用Vertenet的预测来估计感兴趣区域（ROI）检测潜在的腹主动脉植物，在这种算法中，软组织不足阻碍了钙化评估。此外，我们提出了一项少量概念证明研究，以表明从VLL信息产生的IVG可以改善AAC评分中的阅读器间相关性，以解决专家AAC-24评分方面的两个关键领域：IVG放置和质量控制以进行全腹主动脉aorta评估。可以在https://github.com/zaidilyas89/vertenet上找到这项工作的代码。

### IPO: Iterative Preference Optimization for Text-to-Video Generation 
[[arxiv](https://arxiv.org/abs/2502.02088)] [[cool](https://papers.cool/arxiv/2502.02088)] [[pdf](https://arxiv.org/pdf/2502.02088)]
> **Authors**: Xiaomeng Yang,Zhiyu Tan,Hao Li
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: IPO：文本到视频生成的迭代偏好优化
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 视频基础模型在网络升级和模型扩展的帮助下取得了重大进步。但是，由于发电质量不满意，它们仍然很难满足应用的要求。为了解决这个问题，我们建议从本文的培训后的角度将视频基础模型与人类的偏好相结合。因此，我们引入了一种迭代偏好优化策略，通过结合人类反馈来增强产生的视频质量。具体而言，IPO利用了评论家模型来为视频世代合理地对成对排名合理，就像在Kahneman-Tversky优化中一样，直接偏好优化或点数得分。鉴于此，IPO通过偏好反馈的信号进行指导，优化了视频基础模型，这有助于提高主题一致性，运动平滑度和美学质量等的视频质量。此外，IPO还将评论家模型与多模式大型语言模型结合在一起，这使其能够自动分配优先标签，而无需重试或相关。这样，IPO可以以迭代方式有效地执行多轮偏好优化，而无需繁琐的手动标记。全面的实验表明，所提出的IPO可以有效地提高预验证模型的视频生成质量，并帮助仅使用2B参数的模型超过了5B参数。此外，IPO在VBENCH基准测试中实现了新的最新性能。

### Tell2Reg: Establishing spatial correspondence between images by the same language prompts 
[[arxiv](https://arxiv.org/abs/2502.03118)] [[cool](https://papers.cool/arxiv/2502.03118)] [[pdf](https://arxiv.org/pdf/2502.03118)]
> **Authors**: Wen Yan,Qianye Yang,Shiqi Huang,Yipei Wang,Shonit Punwani,Mark Emberton,Vasilis Stavrinides,Yipeng Hu,Dean Barratt
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: 5 pages, 3 figures, conference paper
- **标题**: Tell2Reg：通过相同的语言提示建立图像之间的空间对应关系
- **领域**: 计算机视觉和模式识别,人工智能,图像和视频处理
- **摘要**: 空间对应关系可以通过成对的分段区域表示，以便图像注册网络的目的是分割相应的区域，而不是预测位移字段或变换参数。在这项工作中，我们表明，使用基于接地迪诺和SAM的预训练的大型多模型模型，可以通过相同的语言提示在两个不同的图像上通过相同的语言提示来预测这种相应的区域对。这使得具有全自动且无训练的注册算法，可能可以推广到广泛的图像注册任务。在本文中，我们使用一个具有挑战性的任务之一提出了实验结果，记录了受试者间前列腺MR图像，该图像涉及患者之间高度可变的强度和形态。 Tell2Reg是无培训的，消除了此注册任务以前所需的昂贵且耗时的数据策展和标签的需求。这种方法的表现优于未经监督的基于学习的注册方法，其性能与弱监督的方法相当。还提出了其他定性结果，以表明，语言语义和空间对应之间首次存在潜在的相关性，包括语言贡献区域的空间不变性以及所获得的本地和全球对应关系之间的语言提示差异。代码可在https://github.com/yanwenci/tell2reg.git上找到。

### Driver Assistance System Based on Multimodal Data Hazard Detection 
[[arxiv](https://arxiv.org/abs/2502.03005)] [[cool](https://papers.cool/arxiv/2502.03005)] [[pdf](https://arxiv.org/pdf/2502.03005)]
> **Authors**: Long Zhouxiang,Ovanes Petrosian
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 基于多模式数据危害检测的驾驶员援助系统
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 自动驾驶技术已经大大提高，但是由于驾驶事件的长尾分布，发现驾驶异常仍然是一个重大挑战。现有方法主要依赖于单模道路状况视频数据，这限制了其捕获罕见且不可预测的驾驶事件的能力。本文提出了一个多模式驾驶员辅助检测系统，该系统集成了道路状况视频，驾驶员面部视频和音频数据，以提高事件识别精度。我们的模型采用了基于注意力的中间融合策略，可以实现端到端学习，而无需单独的特征提取。为了支持这种方法，我们使用驱动模拟器开发了一个新的三模式数据集。实验结果表明，我们的方法有效地捕获了跨模式相关性，减少了错误判断并提高了驾驶安全性。

### Disentangling CLIP Features for Enhanced Localized Understanding 
[[arxiv](https://arxiv.org/abs/2502.02977)] [[cool](https://papers.cool/arxiv/2502.02977)] [[pdf](https://arxiv.org/pdf/2502.02977)]
> **Authors**: Samyak Rawlekar,Yujun Cai,Yiwei Wang,Ming-Hsuan Yang,Narendra Ahuja
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 解开剪辑功能，以增强本地化的理解
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉模型（VLMS）在图像分类和检索等粗粒任务中表现出令人印象深刻的功能。但是，他们在需要局部理解的细粒度任务中挣扎。为了研究这种弱点，我们全面分析了剪辑特征并确定一个重要的问题：语义特征高度相关。具体而言，类的特征编码有关其他类的信息，我们称之为共同特征信息（MFI）。当我们查询特定类别并与目标类别一起激活无关的对象时，此相互信息变得很明显。为了解决这个问题，我们提出了Unmix-CLIP，这是一个旨在减少MFI并改善功能分离的新型框架。我们介绍了MFI损失，该损失通过将文本特征投射到最小化类相似性的空间中明确分开文本特征。为了确保图像特征中相应的分离，我们使用多标签识别（MLR）将图像特征与分离的文本特征对齐。这确保了图像和文本功能均被跨模态分开和对齐，从而改善了下游任务的特征分离。对于可可14个数据集，Unmix-CLIP将相似性降低了24.9％。我们通过广泛评估MLR和Zeroshot语义分割（ZS3）来证明其有效性。在MLR中，我们的方法在VOC2007上进行了竞争性，并使用较少的培训参数超过了可可-14数据集上的SOTA方法。此外，Unmix-CLIP始终优于可可和VOC上现有的ZS3方法

### Color in Visual-Language Models: CLIP deficiencies 
[[arxiv](https://arxiv.org/abs/2502.04470)] [[cool](https://papers.cool/arxiv/2502.04470)] [[pdf](https://arxiv.org/pdf/2502.04470)]
> **Authors**: Guillem Arias,Ramon Baldrich,Maria Vanrell
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: 6 pages, 10 figures, conference, Artificial Intelligence
- **标题**: 视觉语言模型中的颜色：剪辑缺陷
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 这项工作探讨了如何在剪辑（对比语言图像预训练）中编码的颜色，该剪辑目前是人工智能中最具影响力的VML（视觉语言模型）。在为此任务创建的合成数据集执行了不同的实验之后，我们得出结论，剪辑能够将正确的颜色标签归因于彩色的视觉刺激，但是，我们遇到了两个主要缺陷：（a）与颜色概念相关的可观性刺激明显偏见，因此白色，灰色，灰色，黑色和黑色很大程度上分配了彩色实验性的实验性实验室； （b）优先考虑文本而不是其他视觉信息的趋势。在这里，我们证明它通过详尽的Stroop效应测试在颜色标记方面非常重要。为了找到这些颜色缺陷的原因，我们分析了神经元水平的内部表示。我们得出的结论是，剪辑对文本进行了重要的选择性，特别是在网络的最深层中，以及少量的多模式颜色神经元，这可能是正确理解颜色概念的关键。我们的调查强调了在人类理解的神经网络中精炼神经网络中的颜色表示机制的必要性，以增强对颜色的更全面的理解，从而提高了在现实世界情景中剪辑（例如剪辑）等多模型模型的功效和多功能性。

### No Images, No Problem: Retaining Knowledge in Continual VQA with Questions-Only Memory 
[[arxiv](https://arxiv.org/abs/2502.04469)] [[cool](https://papers.cool/arxiv/2502.04469)] [[pdf](https://arxiv.org/pdf/2502.04469)]
> **Authors**: Imad Eddine Marouf,Enzo Tartaglione,Stephane Lathuiliere,Joost van de Weijer
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: 8 pages, in-review
- **标题**: 没有图像，没有问题：保留不断的VQA中的知识，只有问题 - 
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在视觉问题回答中的持续学习（VQACL）需要模型来学习新的视觉语言任务（可塑性），同时从以前的任务中保留知识（稳定性）。 VQACL的多模式性质提出了独特的挑战，要求模型平衡跨视觉和文本域的稳定性，同时保持可塑性以适应新颖的对象和推理任务。现有的方法主要是为单峰任务设计的，通常很难有效地平衡这些需求。在这项工作中，我们介绍了仅通过注意力蒸馏（Quad）来重播的，这是VQACL的一种新颖方法，仅利用过去的任务问题进行正规化，消除了存储视觉数据并解决内存和隐私问题的需求。 Quad通过引入仅提问的重播机制来实现稳定性，该机制有选择地使用以前任务中的问题来防止对当前任务的答案空间过度拟合，从而减轻了解答外设置的问题。在此方面，我们提出了关注一致性蒸馏，该蒸馏既独特地执行了任务之间的模式内和模式间注意一致性，从而保留了基本的视觉语言关联。在VQAV2和Next-QA上进行的广泛实验表明，四边形的表现明显优于最先进的方法，从而在连续的VQA中实现了稳健的性能。

### Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting 
[[arxiv](https://arxiv.org/abs/2502.04395)] [[cool](https://papers.cool/arxiv/2502.04395)] [[pdf](https://arxiv.org/pdf/2502.04395)]
> **Authors**: Siru Zhong,Weilin Ruan,Ming Jin,Huan Li,Qingsong Wen,Yuxuan Liang
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: 19 pages
- **标题**: 时间vlm：探索增强时间序列的多模式视觉模型预测
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 时间序列预测的最新进步已经探索了具有文本或视觉方式的增强模型，以提高准确性。尽管文本提供了上下文理解，但通常缺乏细粒度的时间细节。相反，视觉捕获了复杂的时间模式，但缺乏语义上下文，从而限制了这些方式的互补潜力。为了解决这个问题，我们提出了Time-VLM，这是一种新型的多模式框架，利用预先训练的视觉语言模型（VLMS）来桥接时间，视觉和文本方式，以增强预测。我们的框架包括三个关键组成部分：（1）检索授权的学习者，该学习者通过内存库交互来提取丰富的时间特征； （2）一个具有远见的学习者，将时间序列编码为信息图像； （3）文本启动的学习者，该学习者生成上下文文本描述。这些组件与冷冻预训练的VLMS合作生成多模式嵌入，然后将其与时间特征融合以进行最终预测。跨不同数据集的广泛实验表明，Time-VLM可以实现卓越的性能，尤其是在几次射击和零拍摄的情况下，从而为多模式时间序列序列建立了新的方向。

### Can Large Language Models Capture Video Game Engagement? 
[[arxiv](https://arxiv.org/abs/2502.04379)] [[cool](https://papers.cool/arxiv/2502.04379)] [[pdf](https://arxiv.org/pdf/2502.04379)]
> **Authors**: David Melhart,Matthew Barthet,Georgios N. Yannakakis
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-07
> **comment**: This work has been submitted to the IEEE for possible publication
- **标题**: 大型语言模型可以捕获视频游戏的参与吗？
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,人机交互
- **摘要**: 开箱即用的大型语言模型（LLM）可以在观察视频时成功地检测人类影响吗？为了第一次解决这个问题，我们全面评估了流行的LLMS注释和成功地预测视频的连续注释的能力，并以多模式方式提示一系列文本和视频帧。特别是在本文中，我们测试了LLMS在80分钟内从Gamevibe Corpus的20个第一人称射击游戏中的注释视频游戏镜头中正确标记游戏中参与度更改的能力。我们进行了2,400多个实验，以研究LLM体系结构，模型大小，输入方式，提示策略和地面真相处理方法对参与预测的影响。我们的发现表明，尽管LLMS正确地声称在多个领域声称具有类似人类的表现，但它们通常落后于捕获人类提供的连续体验注释。我们研究了相对较差的总体表现的一些根本原因，突出了LLM超过预期的情况，并为通过LLMS进一步探索自动情绪标签的路线图。

### DILLEMA: Diffusion and Large Language Models for Multi-Modal Augmentation 
[[arxiv](https://arxiv.org/abs/2502.04378)] [[cool](https://papers.cool/arxiv/2502.04378)] [[pdf](https://arxiv.org/pdf/2502.04378)]
> **Authors**: Luciano Baresi,Davide Yi Xian Hu,Muhammad Irfan Mas'udi,Giovanni Quattrocchi
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: Dillema：多模式增强的扩散和大型语言模型
- **领域**: 计算机视觉和模式识别,图形,机器学习,软件工程
- **摘要**: 确保深度学习模型的鲁棒性需要全面和多样化的测试。现有的方法通常基于简单的数据增强技术或生成对抗网络，在产生现实和多样化的测试用例方面受到限制。为了解决这些局限性，我们提出了一个新的框架，用于测试视觉神经网络，该框架利用大型语言模型和控制条件的扩散模型生成合成的高保真测试案例。我们的方法首先使用字幕模型将图像转换为详细的文本描述，从而允许语言模型识别图像的可修改方面并生成反事实描述。然后，这些描述用于通过文本对图像扩散过程产生新的测试图像，该过程保留空间一致性并保持场景的关键要素。我们使用两个数据集证明了我们方法的有效性：Imagenet1k用于图像分类和自主驾驶中语义分割的变化。结果表明，我们的方法可以产生重要的测试案例，以揭示弱点并通过靶向重新培训提高模型的鲁棒性。我们使用机械Turk进行了人类评估以验证生成的图像。参与者的回应证实，在选民之间达成了高度同意，我们的方法会产生有效和现实的图像。

### MapFusion: A Novel BEV Feature Fusion Network for Multi-modal Map Construction 
[[arxiv](https://arxiv.org/abs/2502.04377)] [[cool](https://papers.cool/arxiv/2502.04377)] [[pdf](https://arxiv.org/pdf/2502.04377)]
> **Authors**: Xiaoshuai Hao,Yunfeng Diao,Mengchuan Wei,Yifan Yang,Peng Hao,Rong Yin,Hui Zhang,Weiming Li,Shu Zhao,Yu Liu
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: MapFusion：用于多模式地图构造的新型BEV特征融合网络
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 地图构造任务在提供自主驾驶系统必不可少的精确且全面的静态环境信息方面起着至关重要的作用。主传感器包括相机和LIDAR，基于成本效果的考虑因素，配置在仅相机，仅激光镜或相机范围的融合之间变化。尽管基于融合的方法通常会表现最佳，但现有方法通常会忽略模式相互作用并依赖于简单的融合策略，这些策略遭受了未对准和信息丢失问题的困扰。为了解决这些问题，我们提出了MapFusion，这是一种新型的多模式鸟类视图（BEV）用于地图结构的特征融合方法。具体而言，为了解决摄像机和激光镜头BEV功能之间的语义不对对准问题，我们介绍了跨模式相互作用变换（CIT）模块，从而可以通过自我注意事项机制在两个BEV特征空间之间进行相互作用，并增强功能表示。此外，我们提出了一个有效的双动力融合（DDF）模块，以适应不同方式从不同模式中选择有价值的信息，这可以充分利用不同方式之间的固有信息。此外，MapFusion设计为简单且插件，易于集成到现有管道中。我们在两个地图构造任务上评估MAPFusion，包括高清（HD）地图和BEV地图分段，以显示其多功能性和有效性。与最先进的方法相比，MapFusion分别在Nuscenes数据集上的HD MAP构造和BEV MAP分割任务上实现了3.6％和6.2％的绝对改进，这表明了我们方法的优势。

### WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.04326)] [[cool](https://papers.cool/arxiv/2502.04326)] [[pdf](https://arxiv.org/pdf/2502.04326)]
> **Authors**: Jack Hong,Shilin Yan,Jiayin Cai,Xiaolong Jiang,Yao Hu,Weidi Xie
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: Worldsense：评估多模式LLM的现实世界中的全段理解
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在本文中，我们介绍了WorldSense，这是第一个评估多模式视频理解的基准，同时涵盖了视觉，音频和文本输入。与现有基准相反，我们的WorldSense具有多个功能：（i）Omni-Modosity的协作，我们设计了评估任务以具有强烈的音频和视频耦合，要求模型有效地利用Omni-Mododation的协同感知； （ii）视频和任务的多样性，WorldSense涵盖了1,662个音频视频的多样化集合，系统地分类为8个主要领域和67个细粒子类别，以涵盖广泛的方案，以及3,172个跨26个不同的QA Pairs，以遍及全面的评估； （iii）高质量的注释，所有质量检查对由80个具有多个校正的专家注释者手动标记，以确保质量。根据我们的世界义，我们广泛评估了各种最新模型。实验结果表明，现有模型在理解现实世界情景方面面临重大挑战（48.0％的最佳精度）。我们希望我们的世界义务能够提供一个平台，以评估来自Omni模式的构建和理解连贯环境的能力。

### ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features 
[[arxiv](https://arxiv.org/abs/2502.04320)] [[cool](https://papers.cool/arxiv/2502.04320)] [[pdf](https://arxiv.org/pdf/2502.04320)]
> **Authors**: Alec Helbling,Tuna Han Salih Meral,Ben Hoover,Pinar Yanardag,Duen Horng Chau
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 概念：扩散变压器学习高度可解释的特征
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 多模式扩散变压器（DIT）的丰富表示表现出可增强其可解释性的独特特性吗？我们介绍了一种新颖的方法，它利用DIT注意层的表现力来产生高质量的显着性图，这些图精确地定位了图像中的文本概念。在不需要额外的培训的情况下，概念重新利用了DIT注意层的参数，以产生高度上下文化的概念嵌入，这有助于一个主要发现，即在DIT注意层的输出空间中执行线性预测与常用的交叉意见机制相比，在DIT注意力层的输出空间中产生明显的显着性图。值得注意的是，概念甚至可以在零拍图像分割基准上实现最先进的性能，在成像网段数据集和pascalvoc的单层子集上表现优于其他11种其他零摄像的可解释性方法。我们的工作有助于第一个证据表明，多模式DIT模型（如通量）的表示可以高度转移到诸如细分之类的视觉任务，甚至超过了剪辑（例如剪辑）的多模式基础模型。

### Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion 
[[arxiv](https://arxiv.org/abs/2502.04263)] [[cool](https://papers.cool/arxiv/2502.04263)] [[pdf](https://arxiv.org/pdf/2502.04263)]
> **Authors**: Marco Mistretta,Alberto Baldrati,Lorenzo Agnolucci,Marco Bertini,Andrew D. Bagdanov
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: Accepted for publication at ICLR 2025
- **标题**: 跨越差距：通过模态反演暴露剪辑中模式的未对准
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 预先训练的多模式视觉语言模型（例如剪辑）广泛用于各种应用。在本文中，我们表明，对于这些功能强大的多模式模型的文本或图像编码器单独利用的常见实践对于模式内任务（例如图像到图像检索）而言是高度最佳的。我们认为，这是固有的，这是由于夹子式的模式间对比损失，该损失不会强制执行任何模式内约束，从而导致我们称之为模式内的未对准。为了证明这一点，我们利用了两种基于优化的模态反演技术，这些技术将其从输入模式映射到互补的形式，而无需辅助数据或其他训练有素的适配器。我们从经验上表明，在图像到图像和文本对文本检索的模式内任务中，接近这些任务在超过15个数据集上的模式内基线方面可以显着提高性能。此外，我们证明，接近天然模式间任务（例如，零图像分类）会在模式内降低性能，从而进一步验证我们的发现。最后，我们表明，将模式内术语纳入预训练目标或缩小文本和图像特征嵌入空间之间的模态差距有助于减少模式内未对准。该代码可在以下网址公开获取：https：//github.com/miccunifi/cross-the-gap。

### Keep It Light! Simplifying Image Clustering Via Text-Free Adapters 
[[arxiv](https://arxiv.org/abs/2502.04226)] [[cool](https://papers.cool/arxiv/2502.04226)] [[pdf](https://arxiv.org/pdf/2502.04226)]
> **Authors**: Yicen Li,Haitz Sáez de Ocáriz Borde,Anastasis Kratsios,Paul D. McNicholas
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 保持轻度！通过无文本适配器简化图像聚类
- **领域**: 计算机视觉和模式识别,机器学习,神经和进化计算,计算,机器学习
- **摘要**: 许多竞争性的聚类管道具有多模式设计，利用大型语言模型（LLM）或其他文本编码器以及文本图像对，它们在下游应用程序中通常无法使用。此外，此类框架通常很复杂，需要大量的计算资源，从而使广泛采用具有挑战性。在这项工作中，我们表明，在深度聚类，竞争性能中，可以使用无文本和高度简化的训练管道来实现更复杂的最新方法。特别是，我们的方法是通过预训练模型（SCP）进行简单聚类，仅训练一个小的群集头，同时利用预训练的视觉模型特征表示和正数据对。在包括CIFAR-10，CIFAR-20，CIFAR-100，STL-10，Imagenet-10和Imagenet-DOG等基准数据集上的实验表明，SCP可以实现高度竞争性的性能。此外，我们提供了一个理论上的结果，解释了为什么至少在理想条件下，对于在视力中实现强大的聚类性能可能不是必需的其他基于文本的嵌入。

### PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models? 
[[arxiv](https://arxiv.org/abs/2502.04192)] [[cool](https://papers.cool/arxiv/2502.04192)] [[pdf](https://arxiv.org/pdf/2502.04192)]
> **Authors**: Mennatullah Siam
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: Under Review
- **标题**: PixFoundation：我们是否使用像素级视觉基础模型朝正确的方向前进？
- **领域**: 计算机视觉和模式识别
- **摘要**: 已经出现了多项作品，以将多模式大语言模型（MLLM）的界限推向像素级的理解。这种方法在基准上表现出强烈的性能，用于引用表达细分和扎根的对话生成。像素级MLLM的当前趋势是在大型标记数据上对像素级接地监督进行训练。但是，我们表明，当对最近具有挑战性视力基准的评估时，这种MLLM在视觉问答回答中表现出较弱的能力。令人惊讶的是，其中一些方法甚至降低了从未接受过这种监督训练的MLLM的接地能力。在这项工作中，我们提出了两个新颖的挑战性基准，并表明没有像素级接地监督的MLLM在评估像素级接地和视觉问题的回答时，在此类任务中可以超越此类任务的最新状态。我们提出简单的基线来提取可以插入任何MLLM的接地信息，我们称之为PixFoundation。更重要的是，我们研究了“何时在未接受像素级接地监督训练的MLLM中出现基础的研究问题？”我们表明，接地可以与对象零件或位置/外观信息一致。代码存储库位于https://github.com/msiam/pixfoundation/。

### Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.06130)] [[cool](https://papers.cool/arxiv/2502.06130)] [[pdf](https://arxiv.org/pdf/2502.06130)]
> **Authors**: Ce Zhang,Zifu Wan,Zhehan Kan,Martin Q. Ma,Simon Stepputtis,Deva Ramanan,Russ Salakhutdinov,Louis-Philippe Morency,Katia Sycara,Yaqi Xie
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: Accepted by ICLR 2025. Project page:https://zhangce01.github.io/DeGF/
- **标题**: 自我校正解码，并具有生成反馈，以减轻大型视觉模型中的幻觉
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 尽管最近的大型视力模型（LVLM）在多模式任务中表现出了显着的性能，但它们很容易产生幻觉文本响应，这些响应与给定的视觉输入不符，这限制了它们在现实世界中的实际适用性。在这项工作中，受到观察的启发，即文本对图像生成过程是LVLMS中图像条件的响应生成的倒数，我们探讨了利用文本对图像生成模型的潜力，以帮助减轻LVLMS中的幻觉。我们发现，生成模型可以提供有价值的自我反馈，以减轻响应和令牌水平的幻觉。在这种见识的基础上，我们将自我校正解码与生成反馈（DEGF）一起引入了一种新颖的无培训算法，将从文本到图像生成模型的反馈结合到解码过程中，以有效地减轻LVLM中的幻觉。具体而言，DEGF从LVLMS产生的初始响应中生成图像，该图像充当辅助视觉参考，并提供了自我反馈，以通过互补或对比解码来验证和纠正初始响应。广泛的实验结果证明了我们方法在缓解各种幻觉的有效性，从而始终超过六个基准测试的最新方法。代码可在https://github.com/zhangce01/degf上找到。

### Col-OLHTR: A Novel Framework for Multimodal Online Handwritten Text Recognition 
[[arxiv](https://arxiv.org/abs/2502.06100)] [[cool](https://papers.cool/arxiv/2502.06100)] [[pdf](https://arxiv.org/pdf/2502.06100)]
> **Authors**: Chenyu Liu,Jinshui Hu,Baocai Yin,Jia Pan,Bing Yin,Jun Du,Qingfeng Liu
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: ICASSP 2025
- **标题**: Col-Olhtr：多式联运在线手写文本识别的新颖框架
- **领域**: 计算机视觉和模式识别,信号处理
- **摘要**: 在线手写文本识别（OLHTR）因其各种应用程序范围而引起了极大的关注。当前方法通常将OLHTR视为序列识别任务，该任务采用单个轨迹或图像编码器或多流编码器，并与CTC或基于注意力的识别解码器结合使用。但是，这些方法面临几个缺点：1）单一编码器通常专注于局部轨迹或视觉区域，缺乏在具有挑战性的情况下动态捕获相关全局特征的能力； 2）多流编码器虽然更全面，但却遭受了复杂的结构和推理成本的增加。为了解决这个问题，我们提出了一个基于协作的OLHTR框架，即Col-Olhtr，该框架在训练过程中学习了多模式的特征，同时保持单际推理过程。 Col-Olhtr由轨迹编码器，点对点比对（P2SA）模块和基于注意力的解码器组成。 P2SA模块旨在通过轨迹编码的特征和2D旋转位置嵌入来学习图像级的空间特征。在培训期间，对额外的图像流编码器进行了协作培训，以提供P2SA功能的监督。在推断时，额外的流被丢弃，并且在解码器之前仅使用和合并P2SA模块，在保留高性能的同时简化了该过程。几个OLHTR基准测试的广泛实验结果证明了最先进的（SOTA）性能，证明了我们设计的有效性和鲁棒性。

### Temporal Working Memory: Query-Guided Segment Refinement for Enhanced Multimodal Understanding 
[[arxiv](https://arxiv.org/abs/2502.06020)] [[cool](https://papers.cool/arxiv/2502.06020)] [[pdf](https://arxiv.org/pdf/2502.06020)]
> **Authors**: Xingjian Diao,Chunhui Zhang,Weiyi Wu,Zhongyu Ouyang,Peijun Qing,Ming Cheng,Soroush Vosoughi,Jiang Gui
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: Accepted at NAACL 2025
- **标题**: 时间工作记忆：查询引导的细分细分细节，以增强多模式的理解
- **领域**: 计算机视觉和模式识别,多媒体,声音,音频和语音处理
- **摘要**: 多模式基础模型（MFMS）在视觉字幕，问答和图像文本检索等任务中表现出了巨大的成功。但是，由于其有限的内部能力，这些模型面临固有的局限性，这限制了它们处理扩展时间序列的能力，这是全面视频和音频分析的关键要求。为了克服这些挑战，我们引入了专门的认知模块，时间工作记忆（TWM），旨在增强MFM的时间建模功能。它有选择地保留跨时间维度的任务与任务相关的信息，以确保在整个视频和音频内容的处理过程中保留关键细节。 TWM采用查询引导的注意方法来关注时间序列中最有用的多模式段。通过仅保留最相关的内容，TWM优化了模型有限容量的使用，从而增强了其时间建模能力。该插件模块可以轻松地集成到现有的MFMS中。借助我们的TWM，九种最先进的模型在视频字幕，问答和视频文本检索等任务中表现出重大的性能改进。通过增强时间建模，TWM扩展了MFM有效处理复杂，时间敏感数据的能力。我们的代码可从https://github.com/xid32/naacl_2025_twm获得。

### ClinKD: Cross-Modal Clinical Knowledge Distiller For Multi-Task Medical Images 
[[arxiv](https://arxiv.org/abs/2502.05928)] [[cool](https://papers.cool/arxiv/2502.05928)] [[pdf](https://arxiv.org/pdf/2502.05928)]
> **Authors**: Hongyu Ge,Longkun Hao,Zihui Xu,Zhenxin Lin,Bin Li,Shoujun Zhou,Hongjin Zhao,Yihang Liu
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: Clinkd：多任务医学图像的跨模式临床知识蒸馏器
- **领域**: 计算机视觉和模式识别
- **摘要**: 医学视觉问题回答（MED-VQA）代表了一般VQA域内的关键和挑战性的子任务。尽管在一般视觉问题回答（VQA）中取得了重大进展，但在处理多任务VQA方案时，多模式大语言模型（MLLMS）仍会显示出很大的限制。这些局限性通过错误的空间定位和对医学图像的误解而表现出来，这主要源于两个基本问题：对于专用医疗应用的通用MLLM中的图像文本对齐不足和医学知识不足。为了解决这些问题，我们介绍了跨模式临床知识蒸馏器（Clinkd），这是一个创新的框架，旨在增强图像文本对齐并建立更有效的医学知识适应机制，使MLLM可以适应医学知识。我们广泛的实验评估表明，Clinkd在Med-Grit-270k数据集上实现了最先进的性能，这是一种具有挑战性的医学基准，其中包含精细粒度的多任务QA对。结果表明，我们的方法不仅显着改善了图像文本的对齐方式，还可以有效地使MLLM适应医学知识。 Clinkd的源代码可在以下网址提供：https：//github.com/overloadedhenry/clinkd。

### A 3D Multimodal Feature for Infrastructure Anomaly Detection 
[[arxiv](https://arxiv.org/abs/2502.05779)] [[cool](https://papers.cool/arxiv/2502.05779)] [[pdf](https://arxiv.org/pdf/2502.05779)]
> **Authors**: Yixiong Jing,Wei Lin,Brian Sheil,Sinan Acikgoz
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 用于基础架构异常检测的3D多模式特征
- **领域**: 计算机视觉和模式识别
- **摘要**: 衰老结构需要定期检查以识别结构缺陷。先前的工作已经使用几何扭曲来定位合成砌体桥点云中的裂缝，但一直在努力检测小裂纹。为了解决这一限制，本研究提出了一种新型的3D多模式特征3DMulti-FPFHI，该特征将定制的快速点特征直方图（FPFH）与强度特征相结合。该特征被整合到PatchCore异常检测算法中，并通过统计和参数分析进行评估。使用真实砌体拱桥的点云和混凝土隧道的全尺度实验模型进一步评估该方法。结果表明，3D强度特征通过改善裂纹检测来增强检查质量。它还可以识别引入强度异常的水入口。 3DMulti-FPFHI优于FPFH和最先进的多模式异常检测方法。与基于学习的方法相比，数据的最小要求强调了该方法解决不同基础架构异常检测方案的潜力。代码和相关点云数据集可在https://github.com/jingyixiong/3d-multi-fpfhi上获得。

### Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails 
[[arxiv](https://arxiv.org/abs/2502.05772)] [[cool](https://papers.cool/arxiv/2502.05772)] [[pdf](https://arxiv.org/pdf/2502.05772)]
> **Authors**: Yijun Yang,Lichao Wang,Xiao Yang,Lanqing Hong,Jun Zhu
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 有效的黑盒多面攻击违反视觉大型语言模型护栏
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 视觉大语模型（VLLM）整合了视觉数据处理，扩大了其现实世界的应用程序，同时也增加了产生不安全响应的风险。作为回应，领先的公司已经实施了多层安全防御，包括对齐培训，安全系统提示和内容审核。但是，它们针对复杂的对抗性攻击的有效性在很大程度上尚未探索。在本文中，我们提出了多方面的攻击，这是一个新颖的攻击框架，旨在系统地绕过VLLM中的多层防御。它包括三个互补的攻击方：视觉攻击，利用VLLM的多模式性质以通过图像提示注入有毒系统；对齐打破攻击，该攻击操纵模型的对准机制，以优先考虑对比反应的产生；和对抗性签名，通过在响应结束时将误导性信息策略性地放置，从而欺骗了内容主持人。对八个商业VLLM在黑盒环境中的广泛评估表明，多方面的攻击达到了61.56％的攻击成功率，超过了最新方法，至少超过42.18％。

### The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions 
[[arxiv](https://arxiv.org/abs/2502.05673)] [[cool](https://papers.cool/arxiv/2502.05673)] [[pdf](https://arxiv.org/pdf/2502.05673)]
> **Authors**: Ping Liu,Jiawei Du
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 数据集蒸馏的演变：朝向可扩展的可扩展解决方案
- **领域**: 计算机视觉和模式识别
- **摘要**: 数据集蒸馏将大规模数据集凝结成紧凑的合成表示，已成为有效训练现代深度学习模型的关键解决方案。虽然先前的调查专注于2023年之前的发展，但这项工作对最近的进步进行了全面审查，强调了对大规模数据集（例如Imagenet-1K和Imagenet-21K）的可扩展性。我们将进度分为几种关键方法：轨迹匹配，梯度匹配，分布匹配，可扩展的生成方法和解耦优化机制。作为对最近数据集蒸馏的全面检查，这项调查突出了突破性的创新：SRE2L框架高效有效冷凝的框架，可显着提高模型准确性的软标签策略以及无损蒸馏技术，可在保持性能的同时最大程度地提高压缩性。除了这些方法论的进步之外，我们还应对关键挑战，包括针对对抗和后门攻击的鲁棒性，有效处理非IID数据分布。此外，我们探讨了在视频和音频处理，多模式学习，医学成像和科学计算中的新兴应用程序，突出了其域的多功能性。通过提供广泛的绩效比较和可行的研究方向，这项调查使研究人员和从业人员提供了实用的见解，以提高有效且可推广的数据集蒸馏，为未来的创新铺平了道路。

### Evaluating Vision-Language Models for Emotion Recognition 
[[arxiv](https://arxiv.org/abs/2502.05660)] [[cool](https://papers.cool/arxiv/2502.05660)] [[pdf](https://arxiv.org/pdf/2502.05660)]
> **Authors**: Sree Bhattacharyya,James Z. Wang
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: Accepted to NAACL 2025 Findings
- **标题**: 评估情绪识别的视觉模型
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 大型视觉模型（VLM）在几项客观的多模式推理任务中取得了前所未有的成功。但是，为了进一步增强他们与人类的善解人意和有效沟通的能力，改善VLM的处理和理解情绪至关重要。尽管对提高情感理解的研究很大，但缺乏对与情绪相关任务的VLM的详细评估，这可能有助于为下游的微调工作提供信息。在这项工作中，我们介绍了对VLM的首次全面评估，以识别图像中的唤起情绪。从正确性和鲁棒性的角度来看，我们为唤起情绪识别的任务创建了一个基准，并研究了VLM的性能。通过几个实验，我们证明了情绪识别表现取决于的重要因素，并且还表征了VLM在此过程中造成的各种错误。最后，我们通过人类评估研究来指出错误的潜在原因。我们使用实验结果为VLM的背景下的情感研究未来提供了建议。

### SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation 
[[arxiv](https://arxiv.org/abs/2502.05539)] [[cool](https://papers.cool/arxiv/2502.05539)] [[pdf](https://arxiv.org/pdf/2502.05539)]
> **Authors**: Yixian Shen,Qi Bi,Jia-Hong Huang,Hongyi Zhu,Andy D. Pimentel,Anuj Pathania
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: SSH：通过离散哈特利转换的稀疏频谱改编
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 在微调大型基础模型（LLM）时，低级适应性（LORA）已有效地减少可训练的参数数量。但是，当扩展到较大的模型或解决更复杂的任务适应时，它仍然遇到计算和内存挑战。在这项工作中，我们通过离散的Hartley Transformation（SSH）引入了稀疏频谱适应，这是一种新颖的方法，可显着减少可训练参数的数量，同时增强模型性能。在离散的哈特利转换（DHT）之后，在初始权重的指导下，它选择了所有层中最有用的光谱成分。然后，轻巧的倒数DHT然后将频谱投射回空间域以进行更新。在两个单模式任务中进行的广泛实验，例如语言理解和生成以及多模式的任务，例如视频文本理解，表明SSH的表现优于现有参数有效的微调方法（PEFT）方法，同时实现了计算成本和内存需求的实质性减少。

### Fg-T2M++: LLMs-Augmented Fine-Grained Text Driven Human Motion Generation 
[[arxiv](https://arxiv.org/abs/2502.05534)] [[cool](https://papers.cool/arxiv/2502.05534)] [[pdf](https://arxiv.org/pdf/2502.05534)]
> **Authors**: Yin Wang,Mu Li,Jiapeng Liu,Zhiying Leng,Frederick W. B. Li,Ziyao Zhang,Xiaohui Liang
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: FG-T2M ++：LLMS启动的细颗粒驱动的人类运动产生
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们解决了细粒度驱动的人类运动产生的挑战性问题。现有作品产生的不精确动作无法准确捕获文本中指定的关系，因为：（1）缺乏有效的文本解析，用于有关身体部位的详细语义提示，（2）在单词之间无法完全建模语言结构以全面理解文本。要解决这些局限性，我们提出了一个新颖的细粒框架FG-T2M ++，该框架由：（1）LLMS语义解析模块，以从文本中提取身体部位描述和语义，（（2）双重性文本表示模块将相关信息用于在文本单位之间编码相关信息，以将索引依赖的空间嵌入量身像范围内，并（3）型号filefiper图形，并（3）文本和运动功能。关于HumanML3D和Kit-ML数据集的广泛实验表明，FG-T2M ++的表现优于SOTA方法，从而验证了其准确生成粘附于综合文本语义的动作的能力。

### Evaluation of Vision Transformers for Multimodal Image Classification: A Case Study on Brain, Lung, and Kidney Tumors 
[[arxiv](https://arxiv.org/abs/2502.05517)] [[cool](https://papers.cool/arxiv/2502.05517)] [[pdf](https://arxiv.org/pdf/2502.05517)]
> **Authors**: Óscar A. Martín,Javier Sánchez
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: 13 pages, 3 figures, 8 tables
- **标题**: 评估多模式图像分类的视觉变压器：关于大脑，肺和肾脏肿瘤的案例研究
- **领域**: 计算机视觉和模式识别
- **摘要**: 神经网络已成为医学诊断的标准技术，尤其是在癌症检测和分类方面。这项工作评估了视觉变压器体系结构的性能，包括Swin Transformer和Maxvit，在磁共振成像（MRI）和计算机断层扫描（CT）扫描的几个数据集中。我们使用了三个带有大脑，肺和肾脏肿瘤的训练图。每个数据集都包括不同的分类标签，从脑胶质瘤和脑膜瘤到良性和恶性肺部条件以及肾脏异常，例如囊肿和癌症。这项工作旨在分析每个数据集中神经网络的行为，以及结合不同图像方式和肿瘤类别的好处。我们通过在合并和各个图像方式上微调模型来设计多个实验。结果表明，SWIN变压器提供了很高的精度，用于肾脏肿瘤分类的99.9 \％，在组合数据集中达到99.3 \％的精度。 Maxvit在单个数据集中还提供了出色的结果，但是当数据组合时，效果不佳。这项研究突出了基于变压器模型对各种图像方式和特征的适应性。但是，挑战仍然存在，包括有限的注释数据和解释性问题。未来的工作将通过结合其他图像方式并增强诊断能力来扩展这项研究。在各种数据集中整合这些模型可以标志着精确医学的关键进步，为更高效，更全面的医疗保健解决方案铺平了道路。

### Show-o Turbo: Towards Accelerated Unified Multimodal Understanding and Generation 
[[arxiv](https://arxiv.org/abs/2502.05415)] [[cool](https://papers.cool/arxiv/2502.05415)] [[pdf](https://arxiv.org/pdf/2502.05415)]
> **Authors**: Chenkai Xu,Xu Wang,Zhenyi Liao,Yishun Li,Tianqi Hou,Zhijie Deng
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: Show-o Turbo：迈向加速统一的多模式理解和产生
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在建立统一的多模式理解和生成模型的研究兴趣中，越来越多的研究兴趣，其中show-o是一个著名的代表，这对文本对图像和图像到文本的生成都展现了巨大的希望。 Show-O的推论涉及逐步确定图像令牌和自动汇总解码文本令牌，因此不幸的是，双方都遭受了效率低下的问题。本文介绍了Show-o Turbo来弥合差距。我们首先根据文本令牌的平行解码来确定统一的denoising观点，用于在show-o中生成图像和文本。然后，我们建议将一致性蒸馏（CD）扩展，这是一种缩短扩散模型的去核过程的合格方法，到show-o的多模式denoising轨迹。我们引入了轨迹分割策略和课程学习程序，以改善培训融合。从经验上讲，在文本到图像生成中，show-o涡轮在不使用无分类器指导（CFG）的情况下以4个采样步骤显示了遗传得分为0.625，以优于原始Show-O的8个步骤和CFG的表现；在图像到文本生成中，Show-o Turbo表现出1.5倍的速度，而不会显着牺牲性能。该代码可从https://github.com/zhijie-group/show-o-turbo获得。

### Survey on AI-Generated Media Detection: From Non-MLLM to MLLM 
[[arxiv](https://arxiv.org/abs/2502.05240)] [[cool](https://papers.cool/arxiv/2502.05240)] [[pdf](https://arxiv.org/pdf/2502.05240)]
> **Authors**: Yueying Zou,Peipei Li,Zekun Li,Huaibo Huang,Xing Cui,Xuannan Liu,Chenghanyu Zhang,Ran He
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 对AI生成的媒体检测的调查：从非MLLM到MLLM
- **领域**: 计算机视觉和模式识别
- **摘要**: AI生成的媒体的扩散对信息真实性和社会信任提出了重大挑战，这使得可靠的检测方法高度要求。检测AI生成的培养基的方法已迅速发展，与多模式大型语言模型（MLLM）的进步相似。当前的检测方法可以分为两个主要组：基于非MLLM和基于MLLM的方法。前者采用了由深度学习技术提供动力的高精度，特定于领域的探测器，而后者则利用基于MLLM的通用检测器，这些探测器基于整合真实性验证，解释性和本地化功能的MLLM。尽管在该领域取得了重大进展，但文献中仍然存在有关一项综合调查的差距，该调查研究了从域特异性到通用检测方法的过渡。本文通过对两种方法进行系统的综述，从单模式和多模式的角度分析它们来解决这一差距。我们对这些类别进行了详细的比较分析，研究了它们的方法论上的相似性和差异。通过此分析，我们探讨了潜在的混合方法，并确定了伪造检测中的关键挑战，为将来的研究提供了方向。此外，随着MLLM在检测任务中越来越普遍，道德和安全考虑因素已成为关键的全球关注点。我们研究了各个司法管辖区围绕生成AI（Genai）的监管景观，为该领域的研究人员和从业人员提供了宝贵的见解。

### QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation 
[[arxiv](https://arxiv.org/abs/2502.05178)] [[cool](https://papers.cool/arxiv/2502.05178)] [[pdf](https://arxiv.org/pdf/2502.05178)]
> **Authors**: Yue Zhao,Fuzhao Xue,Scott Reed,Linxi Fan,Yuke Zhu,Jan Kautz,Zhiding Yu,Philipp Krähenbühl,De-An Huang
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: Tech report. Project page: https://nvlabs.github.io/QLIP/
- **标题**: QLIP：文本对准的视觉令牌化统一自动回归多模式的理解和生成
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们介绍了量化的语言图像预处理（QLIP），这是一种视觉令牌化方法，将最先进的重建质量与最新的零拍图像理解结合在一起。 QLIP通过重建和语言图像对准目标训练基于二进制定量的自动编码器。我们是第一个表明这两个目标不需要矛盾的人。我们在训练过程中动态平衡了两个损失项，并表明两阶段的训练管道有效地将图像语言预训练的大批量要求与重建目标施加的内存瓶颈混合在一起。我们通过单个模型来验证QLIP对多模式理解和文本条件形成图像生成的有效性。具体而言，QLIP可作为LLAVA视觉编码器的置换式替代品和Llamagen的图像令牌，具有可比的性能甚至更好的性能。最后，我们证明了QLIP可以使统一的混合模式自动回归模型用于理解和产生。

### Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy 
[[arxiv](https://arxiv.org/abs/2502.05177)] [[cool](https://papers.cool/arxiv/2502.05177)] [[pdf](https://arxiv.org/pdf/2502.05177)]
> **Authors**: Yunhang Shen,Chaoyou Fu,Shaoqi Dong,Xiong Wang,Yi-Fan Zhang,Peixian Chen,Mengdan Zhang,Haoyu Cao,Ke Li,Xiawu Zheng,Yan Zhang,Yiyi Zhou,Ran He,Caifeng Shan,Rongrong Ji,Xing Sun
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: https://github.com/VITA-MLLM/Long-VITA
- **标题**: Longonvita：将大型多模式模型缩放为100万个令牌
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们介绍了Long-Vita，这是一个简单而有效的大型多模式模型，用于长篇小说视觉语言理解任务。它擅长在4K帧或1M令牌上同时处理和分析图像，视频和文本的方式，同时在短篇文本多模式任务上提供高级性能。我们提出了一个有效的多模式训练架构，该模式从大型语言模型开始，并通过视觉对齐，常识学习和长期微调的两个顺序阶段进行。我们进一步实施了上下文 - 并行分布的推理和logits掩盖的语言建模，以扩展长vita到模型推理期间的图像和文本的无限长度输入。关于培训数据，Longy Vita仅基于公共数据集的1700万样本的组合，并证明了各种多模式基准测试的最先进性能，与最近具有内部数据的最先进模型相比。 LongeVita是完全可重现的，并支持NPU和GPU平台进行培训和测试。通过利用我们的推理设计，Longy Vita模型在单个节点中以8 GPU的形式实现了显着的2倍预填充加速和4倍上下文长度延伸。我们希望Longen Vita可以充当竞争性的基准，并为开源社区提供宝贵的见解，以促进长期文化多模式的理解。

### Multitwine: Multi-Object Compositing with Text and Layout Control 
[[arxiv](https://arxiv.org/abs/2502.05165)] [[cool](https://papers.cool/arxiv/2502.05165)] [[pdf](https://arxiv.org/pdf/2502.05165)]
> **Authors**: Gemma Canet Tarrés,Zhe Lin,Zhifei Zhang,He Zhang,Andrew Gilbert,John Collomosse,Soo Ye Kim
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: Multitwine：具有文本和布局控件的多对象组合
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们介绍了第一个能够在文本和布局的指导下同时进行多对象合成的生成模型。我们的模型允许在场景中添加多个对象，从而捕获一系列相互作用，从简单的位置关系（例如，在旁边，在面前）到需要重新安息的复杂动作（例如，拥抱，弹吉他）。当互动意味着其他道具（例如“自拍照”）时，我们的模型会自动生成这些支持对象。通过共同培训组合和主题驱动的生成（也称为定制），我们实现了文本和视觉输入的更加平衡的整合，用于文本驱动对象合成。结果，我们获得了一个多功能模型，在这两个任务中都具有最先进的性能。我们进一步提出了利用视觉和语言模型的数据生成管道，以毫不费力地综合了多模式，对齐的训练数据。

### Hummingbird: High Fidelity Image Generation via Multimodal Context Alignment 
[[arxiv](https://arxiv.org/abs/2502.05153)] [[cool](https://papers.cool/arxiv/2502.05153)] [[pdf](https://arxiv.org/pdf/2502.05153)]
> **Authors**: Minh-Quan Le,Gaurav Mittal,Tianjian Meng,A S M Iftekhar,Vishwas Suryanarayanan,Barun Patra,Dimitris Samaras,Mei Chen
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: Accepted to ICLR 2025. Project page: https://roar-ai.github.io/hummingbird
- **标题**: 蜂鸟：通过多模式上下文对齐的高保真图像生成
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管扩散模型在产生以对象为中心任务的高质量，多样化的合成数据方面具有强大的功能，但现有方法与场景感知的任务（例如视觉询问回答（VQA）（VQA）和人类对象相互作用（HOI）推理）遇到了困难，在此至关重要的是，在与范围的上下文中保持一致的图像，即具有参考图像，这对于保留一致的图像中的场景属性至关重要。为了解决这个问题，我们介绍了蜂鸟，蜂鸟是第一个基于扩散的图像发生器，在给定多模式上下文的情况下，它生成了高度多样化的图像W.R.T.参考图像通过准确保留场景属性（例如对象相互作用和文本指南的空间关系）来确保高保真度。 Hummingbird采用了一种新颖的多模式上下文评估器，同时优化了我们配制的全球​​语义和细粒度的一致性奖励，以确保生成的图像保留有关文本指南的参考图像的场景属性，同时保持多样性。作为解决多模式上下文的第一个确定多样性和忠诚的任务的模型，我们引入了一种新的基准公式，其中包含MME感知和Bongard HOI数据集。基准实验表明，蜂鸟通过在保持多样性的同时实现卓越的保真度来优于所有现有方法，从而在复杂的视觉任务中验证了Hummingbird的潜力作为强大的多模式上下文对立图像发生器。

### Lost in Time: Clock and Calendar Understanding Challenges in Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.05092)] [[cool](https://papers.cool/arxiv/2502.05092)] [[pdf](https://arxiv.org/pdf/2502.05092)]
> **Authors**: Rohit Saxena,Aryo Pradipta Gema,Pasquale Minervini
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: Preprint
- **标题**: 丢失时间：时钟和日历理解多模式LLMS中的挑战
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 了解视觉表示的时间是一种基本的认知技能，但对于多模式大语言模型（MLLM）来说仍然是一个挑战。在这项工作中，我们研究了MLLM在通过模拟时钟和年度日历解释时间和日期的功能。为了促进这一点，我们策划了一个结构化数据集，其中包括两个子集：1）$ \ textit {clockqa} $，其中包括各种类型的时钟样式$  -  $标准，黑色dial，no-second hand，roman Numeral和Arrow Hand Clocks $  - 与时间相关的问题配对； 2）$ \ textit {calendarqa} $，由年度日历图像组成，其问题范围从众所周知的日期（例如，圣诞节，元旦）到计算派生的问题（例如，一年中的第100或153天）。我们旨在分析MLLM在使用与时间相关的视觉数据显示时如何执行视觉识别，数值推理和时间推断。我们的评估表明，尽管有最近的进步，但可靠地了解时间仍然是MLLM的重大挑战。

### ELITE: Enhanced Language-Image Toxicity Evaluation for Safety 
[[arxiv](https://arxiv.org/abs/2502.04757)] [[cool](https://papers.cool/arxiv/2502.04757)] [[pdf](https://arxiv.org/pdf/2502.04757)]
> **Authors**: Wonjun Lee,Doehyeon Lee,Eugene Choi,Sangyoon Yu,Ashkan Yousefpour,Haon Park,Bumsub Ham,Suhyun Kim
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: Elite：安全性的增强语言图像毒性评估
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 当前的视觉语言模型（VLM）仍然容易受到诱发有害产出的恶意提示。 VLMS的现有安全基准主要依赖于自动评估方法，但是这些方法难以检测隐性有害内容或产生不准确的评估。因此，我们发现现有基准的有害性，模棱两可的数据和图像文本对组合的多样性有限。为了解决这些问题，我们提出了Elite Benchmark，这是VLM的高质量安全评估基准，并由我们增强的评估方法（精英评估者）支撑。精英评估者明确纳入了毒性评分，以准确评估多模式环境中的有害性，在该环境中，VLMS通常提供特定，令人信服但不受伤害的图像描述。我们使用Elite评估器从现有基准测试中滤除了模棱两可和低质量的图像对文本对，并生成了安全且不安全的图像文本对的各种组合。我们的实验表明，与先前的自动化方法相比，精英评估者与人类评估相比具有较高的一致性，并且精英基准提供了增强的基准测试质量和多样性。通过介绍精英，我们为更安全，更健壮的VLM铺平了道路，为评估和减轻现实世界应用中的安全风险提供了基本工具。

### MLLM4PUE: Toward Universal Embeddings in Computational Pathology through Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.07221)] [[cool](https://papers.cool/arxiv/2502.07221)] [[pdf](https://arxiv.org/pdf/2502.07221)]
> **Authors**: Qifeng Zhou,Thao M. Dang,Wenliang Zhong,Yuzhi Guo,Hehuan Ma,Saiyang Na,Junzhou Huang
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: MLLM4PUE：通过多模式LLMS在计算病理学中的通用嵌入
- **领域**: 计算机视觉和模式识别
- **摘要**: 病理在诊断多种疾病中起着至关重要的作用，但是现有的方法通常很大程度上依赖于针对广泛，标记良好的数据集训练的特定任务模型。由于病理多样性和数据收集的劳动密集型性质，这些方法面临可持续性挑战。为了解决这些限制，我们强调了可以支持多个下游任务的通用多模式嵌入的需求。以前的方法通常涉及基于剪辑的微调模型，该模型分别处理图像和文本，从而限制了它们捕获复杂的多模式关系的能力。此外，这些模型在不同的数据集中进行了评估，而没有统一的基准测试，用于评估病理中的多模式嵌入。为了应对这些挑战，我们提出了MLLM4PUE，这是一个新型框架，利用多模式大型语言模型（MLLMS）生成病理通用嵌入。 MLLM4PUE框架不仅促进了图像和文本的强大整合，而且还增强了各种任务的理解和融合功能。我们进一步介绍了病理多模式嵌入基准（PMEB），这是一种综合基准，旨在评估病理多模式嵌入的质量。 PMEB包括15个原始任务，这些任务来自14个数据集，分为三个元任务：检索，分类和组成的检索。实验结果证明了MLLM4PUE的优势，说明基于MLLM的模型可以有效地支持广泛的下游任务，并统一病理学基础模型的研究方向。

### Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content 
[[arxiv](https://arxiv.org/abs/2502.07138)] [[cool](https://papers.cool/arxiv/2502.07138)] [[pdf](https://arxiv.org/pdf/2502.07138)]
> **Authors**: Girish A. Koushik,Diptesh Kanojia,Helen Treharne
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: Accepted to the MM4SG Workshop at the WebConf 2025
- **标题**: 迈向多模式仇恨检测的强大框架：一项关于视频与基于图像的内容的研究
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 社交媒体平台使跨不同方式，例如文本，听觉和视觉，需要有效的检测方法来传播可恶的内容。尽管最近的方法在处理各个模式方面表现出了希望，但它们在不同方式组合中的有效性仍未得到探索。本文介绍了基于融合的多模式仇恨检测方法的系统分析，重点介绍了它们在视频和基于图像的内容中的性能。我们的全面评估揭示了特定于模式特定的局限性：虽然简单嵌入融合会在视频内容（HATEMM数据集）上获得最新的性能，而F1得分的提高了9.9％，但它与模因中的复杂图像text关系（仇恨的模因数据集）中挣扎。通过详细的消融研究和错误分析，我们证明了当前的融合方法如何无法捕获细微的跨模式相互作用，尤其是在涉及良性混杂因素的情况下。我们的发现为开发更健壮的仇恨检测系统提供了重要的见解，并强调了对特定于模式的建筑考虑的需求。该代码可在https://github.com/gak97/video-vs-meme-hate上找到。

### AI-Driven HSI: Multimodality, Fusion, Challenges, and the Deep Learning Revolution 
[[arxiv](https://arxiv.org/abs/2502.06894)] [[cool](https://papers.cool/arxiv/2502.06894)] [[pdf](https://arxiv.org/pdf/2502.06894)]
> **Authors**: David S. Bhatti,Yougin Choi,Rahman S M Wahidur,Maleeka Bakhtawar,Sumin Kim,Surin Lee,Yongtae Lee,Heung-No Lee
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-11
> **comment**: 39 Pages, 22 figures, 20 tables
- **标题**: AI驱动的HSI：多模式，融合，挑战和深度学习革命
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 高光谱成像（HSI）捕获了空间和光谱数据，从而可以分析传统系统不可见的特征。该技术在诸如天气监测，食品质量控制，假冒检测，医疗诊断以及延伸到国防，农业和工业自动化等领域至关重要。 HSI随着光谱分辨率，微型化和计算方法的改进而进步。这项研究概述了HSI，其应用，数据融合中的挑战以及深度学习模型在处理HSI数据中的作用。我们讨论了多模式HSI与AI的整合，尤其是在深度学习的情况下如何提高分类准确性和运营效率。深度学习增强了特征提取，变更检测，脱核，降低尺寸降低，地覆盖映射，数据扩展，光谱构建和超级分辨率等领域的HSI分析。新兴的重点是高光谱摄像机与大语言模型（LLMS）的融合，称为高脑LLM，从而可以开发高级应用程序，例如低可见性崩溃检测和面对反稳定性。我们还强调了HSI行业的关键参与者，其复合年增长率和工业意义的增长。目的是为技术和非技术受众提供洞察力，涵盖HSI的图像，趋势和未来方向，同时提供有关HSI数据集和软件库的宝贵信息。

### A New Hybrid Intelligent Approach for Multimodal Detection of Suspected Disinformation on TikTok 
[[arxiv](https://arxiv.org/abs/2502.06893)] [[cool](https://papers.cool/arxiv/2502.06893)] [[pdf](https://arxiv.org/pdf/2502.06893)]
> **Authors**: Jared D. T. Guerrero-Sosa,Andres Montoro-Montarroso,Francisco P. Romero,Jesus Serrano-Guerrero,Jose A. Olivas
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 一种新的混合智能方法，用于多模式检测Tiktok的怀疑虚假信息
- **领域**: 计算机视觉和模式识别,计算语言学,多媒体,符号计算
- **摘要**: 在多媒体内容的快速传播的背景下，识别Tiktok等社交媒体平台上的虚假信息是一个重大挑战。这项研究介绍了一个混合框架，该框架将深度学习的计算能力与模糊逻辑的解释性结合在一起，以检测Tiktok视频中可疑的虚假信息。该方法由两个核心组成部分组成：一种多模式特征分析仪，该分析仪从文本，音频和视频中提取和评估数据；以及基于模糊逻辑的多模式虚假信息检测器。这些系统结合起作用，以评估人们对人体语言，语言模式和文本连贯性等人类行为线索的估计。进行了两个实验：一个专注于上下文特定的虚假信息，另一个集中于跨更广泛主题的模型的可扩展性。对于评估，高质量，全面，结构良好的报告的每个视频，都会生成虚假行为的详细视图。

### EVEv2: Improved Baselines for Encoder-Free Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.06788)] [[cool](https://papers.cool/arxiv/2502.06788)] [[pdf](https://arxiv.org/pdf/2502.06788)]
> **Authors**: Haiwen Diao,Xiaotong Li,Yufeng Cui,Yueze Wang,Haoge Deng,Ting Pan,Wenxuan Wang,Huchuan Lu,Xinlong Wang
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: 19 pages, 9 figures
- **标题**: EVEV2：改进的无编码视觉语言模型的基准
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 现有的无编码视觉语言模型（VLMS）正在用基于编码器的对应物迅速缩小性能差距，从而突出了具有结构性简单和有效部署的统一多模式系统的有希望的潜力。我们使用预先训练的视觉编码器，离散的令牌和简约的视觉层从划痕中系统地阐明VLM之间的性能差距，从而深刻挖掘了无编码器VLM的不足检查的特征。我们为与基于编码器的主流主流的无编码VLM制定有效的策略。经过深入的调查，我们启动了Evev2.0，这是一个新的，改进的无编码器VLM家族。我们表明：（i）在统一模型中正确分解和分层的视觉和语言可以减少方式之间的干扰。 （ii）精心设计的培训策略可以为无编码器VLMS有效优化。通过广泛的评估，我们的EVEV2.0代表了一项详尽的研究，用于跨模态开发仅解码器的体系结构，证明了卓越的数据效率和强大的视力反应能力。代码可公开可用：https：//github.com/baaivision/eve。

### Learning Musical Representations for Music Performance Question Answering 
[[arxiv](https://arxiv.org/abs/2502.06710)] [[cool](https://papers.cool/arxiv/2502.06710)] [[pdf](https://arxiv.org/pdf/2502.06710)]
> **Authors**: Xingjian Diao,Chunhui Zhang,Tingxuan Wu,Ming Cheng,Zhongyu Ouyang,Weiyi Wu,Jiang Gui
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: Accepted at EMNLP 2024
- **标题**: 学习音乐表演问题的音乐表述回答
- **领域**: 计算机视觉和模式识别,多媒体,声音,音频和语音处理
- **摘要**: 音乐表演是视听建模的代表性场景。与稀疏音频的常见场景不同，音乐表演持续涉及密集的音频信号。尽管在音频质量质量质量检查中的现有多模式学习方法在一般情况下表现出令人印象深刻的功能，但它们无法处理音乐表演中的基本问题：它们在性能中的多模式信号之间的相互作用不足，并且无法考虑乐器和音乐的独特特征。因此，现有的方法倾向于回答有关音乐表演的问题。为了弥合上述研究差距，（i）鉴于音乐数据固有的复杂的多模式互连性，我们的主要骨干旨在将多模式的交互在音乐的背景下结合； （ii）为了使模型学习音乐特征，我们在当前的音乐数据集中注释和释放节奏和音乐来源； （iii）对于时间感知到的视听建模，我们将模型的音乐预测与时间维度保持一致。我们的实验显示了对音乐AVQA数据集的最新影响。我们的代码可在https://github.com/xid32/amuse上找到。

### Adaptive Perception for Unified Visual Multi-modal Object Tracking 
[[arxiv](https://arxiv.org/abs/2502.06583)] [[cool](https://papers.cool/arxiv/2502.06583)] [[pdf](https://arxiv.org/pdf/2502.06583)]
> **Authors**: Xiantao Hu,Bineng Zhong,Qihua Liang,Zhiyi Mo,Liangtao Shi,Ying Tai,Jian Yang
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 统一视觉多模式对象跟踪的自适应感知
- **领域**: 计算机视觉和模式识别
- **摘要**: 最近，许多多模式跟踪器将RGB优先考虑为主要模式，将其他模态视为辅助方式，并且分别进行了各种多模式任务。这种不平衡的依赖性不平衡限制了方法在复杂场景中从每种模式中动态使用互补信息的能力，这使得充分感知多模式的优势变得具有挑战性。结果，统一的参数模型通常在各种多模式跟踪任务中表现不佳。为了解决这个问题，我们提出了APTRACK，这是一种专为多模式自适应感知而设计的新型统一跟踪器。与以前的方法不同，Aptrack通过平等的建模策略探索统一的表示。该策略允许模型动态适应各种方式和任务，而无需在不同任务之间进行其他微调。此外，我们的跟踪器集成了自适应模态相互作用（AMI）模块，该模块通过生成可学习的令牌有效地桥接交叉模式相互作用。在五种多种模式数据集（RGBT234，Lasher，Visevent，Depthtrack和dot-RGBD2022）上进行的实验表明，APTRACK不仅超过了现有的最新统一统一的多模式跟踪器，而且超过了用于特定多种模式任务的特定多模式跟踪器。

### UniMoD: Efficient Unified Multimodal Transformers with Mixture-of-Depths 
[[arxiv](https://arxiv.org/abs/2502.06474)] [[cool](https://papers.cool/arxiv/2502.06474)] [[pdf](https://arxiv.org/pdf/2502.06474)]
> **Authors**: Weijia Mao,Zhenheng Yang,Mike Zheng Shou
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: UNIMOD：有效的统一多模式变压器，具有深度混合物
- **领域**: 计算机视觉和模式识别
- **摘要**: 统一的多模式变压器处理共享参数空间内的发电和理解任务，在最近的研究中受到了越来越多的关注。尽管已经提出了各种统一的变压器，但是培训这些模型的昂贵，这是由于多余的令牌和大量的注意计算。过去，对大语言模型的研究表明，令牌修剪方法（例如深度（MOD）的混合物）可以显着提高计算效率。 MOD采用路由器来选择在变压器层中处理的最重要的路由器。但是，直接将基于mod的令牌修剪应用于统一变压器将导致次优性能，因为不同的任务表现出不同级别的令牌冗余。在我们的工作中，我们通过（1）检查注意力重量模式，（2）评估层的重要性和令牌冗余，以​​及（3）分析任务相互作用。我们的发现表明，令牌冗余主要受不同的任务和层次的影响。在这些发现的基础上，我们介绍了Unimod，这是一种任务感知的令牌修剪方法，该方法采用一个单独的路由器来确定应修剪哪个令牌。我们将我们的方法应用于Show-O和EMU3，在Show-O中减少了约15％，在EMU3中减少了40％，同时维持或改善了几种基准的性能。代码将在https://github.com/showlab/unimod上发布。

### CoS: Chain-of-Shot Prompting for Long Video Understanding 
[[arxiv](https://arxiv.org/abs/2502.06428)] [[cool](https://papers.cool/arxiv/2502.06428)] [[pdf](https://arxiv.org/pdf/2502.06428)]
> **Authors**: Jian Hu,Zixu Cheng,Chenyang Si,Wei Li,Shaogang Gong
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: A training-free test-time optimisation approach for long video understanding
- **标题**: cos：射击促使促使视频理解长期理解
- **领域**: 计算机视觉和模式识别
- **摘要**: 由于需要过多的视觉令牌，多模式大型语言模型（MLLM）与长视频斗争。这些令牌大大超过了MLLM的上下文长度，从而填充了冗余的任务射击。如何选择镜头是一个尚未解决的关键问题：稀疏采样风险缺少关键细节，而详尽的采样使模型淹没了无关的内容，从而导致视频误解。为了解决这个问题，我们提出了射击促进链（COS）。关键想法是将镜头选择作为测试时间视觉提示优化，通过优化射击任务对齐方式选择适应视频理解语义任务的镜头。 COS有两个关键部分：（1）执行伪时间基础的二进制视频摘要机制，发现了二进制编码以识别与任务相关的镜头，以及（2）视频共同设计的模块，将二进制编码部署到配对（学习以对齐）与不相关的负负相位的正面镜头。它将优化的镜头选择嵌入到原始视频中，从而促进了对相关上下文的关注，以优化长期的视频理解。三个基线和五个数据集的实验证明了COS的有效性和适应性。 https://lwpyh.github.io/cos中给出的代码。

### Multi-Scale Transformer Architecture for Accurate Medical Image Classification 
[[arxiv](https://arxiv.org/abs/2502.06243)] [[cool](https://papers.cool/arxiv/2502.06243)] [[pdf](https://arxiv.org/pdf/2502.06243)]
> **Authors**: Jiacheng Hu,Yanlin Xiang,Yang Lin,Junliang Du,Hanchao Zhang,Houze Liu
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 用于准确医学图像分类的多尺度变压器体系结构
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 这项研究介绍了基于增强的变压器结构建立的AI驱动的皮肤病变分类算法，从而解决了医学图像分析中准确性和鲁棒性的挑战。通过整合多尺度的特征融合机制并完善自我注意事项过程，该模型有效地提取了全球和局部特征，从而增强了其使用模棱两可的边界和复杂结构检测病变的能力。对ISIC 2017数据集的性能评估表明，改进的变压器超过了建立的AI模型，包括Resnet50，VGG19，Resnext和Vision Transformer在精度，AUC，AUC，F1得分和精度等关键指标上。 Grad-CAM可视化进一步强调了该模型的解释性，展示了该算法的焦点区域和实际病变位点之间的强烈一致性。这项研究强调了先进的AI模型在医学成像中的变革潜力，为更准确和可靠的诊断工具铺平了道路。未来的工作将探讨这种方法对更广泛的医学成像任务的可伸缩性，并研究多模式数据的整合，以增强AI驱动的诊断框架的智能医疗保健。

### Multimodal Task Representation Memory Bank vs. Catastrophic Forgetting in Anomaly Detection 
[[arxiv](https://arxiv.org/abs/2502.06194)] [[cool](https://papers.cool/arxiv/2502.06194)] [[pdf](https://arxiv.org/pdf/2502.06194)]
> **Authors**: You Zhou,Jiangshan Zhao,Deyu Zeng,Zuo Zuo,Weixiang Liu,Zongze Wu
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 多模式任务表示记忆库与灾难性遗忘在异常检测中
- **领域**: 计算机视觉和模式识别
- **摘要**: 无监督的连续异常检测（UCAD）在多任务表示学习中面临重大挑战，现有的方法患有不完整的表示和灾难性的遗忘。与监督的模型不同，无监督的场景缺乏先前的信息，因此难以有效区分冗余和互补的多模式特征。为了解决这个问题，我们通过两项关键技术创新提出了多模式任务表示内存库（MTRMB）方法：一种关键 - 促进型 -  multimodal知识（KPMK）机制，该机制使用简洁的关键提示来指导Bert和Vit之间的交叉模式特征交互。基于结构的对比度学习（RSCL）利用接地恐龙和SAM生成精确的分割口罩，将同一结构区域的特征靠近，同时将不同的结构区域推开。 MVTEC AD和VISA数据集的实验证明了MTRMB的优势，以最低的遗忘速率达到了0.921的平均检测准确性，并且表现明显优于最先进的方法。我们计划在Github上开源。

### MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models 
[[arxiv](https://arxiv.org/abs/2502.08079)] [[cool](https://papers.cool/arxiv/2502.08079)] [[pdf](https://arxiv.org/pdf/2502.08079)]
> **Authors**: Peng-Fei Zhang,Guangdong Bai,Zi Huang
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: MAA：针对视力语言预训练模型的细致对抗性攻击
- **领域**: 计算机视觉和模式识别
- **摘要**: 当前的对抗性攻击是用于评估多模式任务中视力语言预先训练（VLP）模型的鲁棒性的稳健性，这些模型的可传递性有限，在这种情况下，为特定模型制定的攻击通常努力地在不同模型上有效地概括，从而限制了他们的效用，从而更广泛地评估了鲁棒性。这主要归因于对模型特异性特征和区域的过度依赖，尤其是在图像模式中。在本文中，我们提出了一种优雅而高效的方法，称为细致的对抗攻击（MAA），以完全利用单个样本的模型独立特征和脆弱性，从而实现了增强的普遍性和减少模型依赖性。 MAA通过开发一种新颖的调整和滑动作物（RSCROP）技术来强调对对抗性图像的细粒优化，并结合了多粒性相似性破坏（MGSD）策略。跨不同VLP模型，多个基准数据集以及各种下游任务的广泛实验表明，MAA显着提高了对抗性攻击的有效性和可传递性。进行了大量的绩效研究，以产生对各种模型配置有效性的见解，从而指导该领域的未来进步。

### Joint Modelling Histology and Molecular Markers for Cancer Classification 
[[arxiv](https://arxiv.org/abs/2502.07979)] [[cool](https://papers.cool/arxiv/2502.07979)] [[pdf](https://arxiv.org/pdf/2502.07979)]
> **Authors**: Xiaofei Wang,Hanyu Liu,Yupei Zhang,Boyang Zhao,Hao Duan,Wanming Hu,Yonggao Mou,Stephen Price,Chao Li
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: accepted by Medical Image Analysis
- **标题**: 癌症分类的联合模型组织学和分子标记
- **领域**: 计算机视觉和模式识别
- **摘要**: 癌症的特征是异质性和预后多样。准确的癌症分类对于患者分层和临床决策至关重要。尽管数字病理学一直在推进癌症的诊断和预后，但癌症病理学的范式已从纯粹依赖组织学特征转变为纳入分子标记。迫切需要数字病理学方法来满足新范式的需求。我们介绍了一种新型的数字病理学方法，可以共同预测分子标记和组织学特征，并建模其与癌症分类的相互作用。首先，为了减轻跨磁化信息传播的挑战，我们提出了一个多尺度的分离模块，从而使多尺度特征从高磁化（蜂窝级）到低磁化（组织级别）全幻灯片图像提取多尺度特征。此外，根据多尺度特征，我们提出了一个基于注意力的层次多任务多任务多任务学习框架，以同时预测组织学和分子标记。此外，我们提出了一个基于共发生概率的标签相关图网络，以模拟分子标记的共发生。最后，我们设计了一个具有动态置信度损失和跨模式梯度调制策略的跨模式相互作用模块，以模拟组织学和分子标记的相互作用。我们的实验表明，我们的方法在分类神经胶质瘤，组织学特征和分子标记方面优于其他最先进的方法。我们的方法有望促进精确的肿瘤学，并有可能推进生物医学研究和临床应用。该代码可从https://github.com/lhy1007/m3c2获得

### DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities 
[[arxiv](https://arxiv.org/abs/2502.07905)] [[cool](https://papers.cool/arxiv/2502.07905)] [[pdf](https://arxiv.org/pdf/2502.07905)]
> **Authors**: Chashi Mahiul Islam,Samuel Jacob Chacko,Preston Horne,Xiuwen Liu
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 19 pages, 4 figures
- **标题**: 旅行中的DeepSeek：通过表示漏洞引起有针对性的视觉幻觉
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 多模式的大语言模型（MLLM）代表了AI技术的最前沿，DeepSeek模型成为领先的开源替代方案，为封闭形式系统提供竞争性能。尽管这些模型表现出了非凡的功能，但他们的视觉整合机制引入了特定的漏洞。我们对DeepSeek Janus进行了适应的嵌入操作攻击，该攻击通过系统优化的图像嵌入来引起目标视觉幻觉。通过对可可，DALL-E 3和SVIT数据集进行广泛的实验，我们在开放式问题上保持了高视觉保真度（SSIM> 0.88）的幻觉速度高达98.0％（SSIM> 0.88）。我们的分析表明，DeepSeek Janus的1B和7B变体都容易受到这些攻击的影响，与开放式问题相比，封闭形式的评估表明，幻觉率始终更高。我们使用Llama-3.1 8B指示进行了强大的评估，介绍了一种新型的多项式幻觉检测框架。这些发现的含义特别关心DeepSeek的开源性质和广泛的部署潜力。这项研究强调了在MLLM部署管道中嵌入级别的安全措施的关键需求，并有助于对负责人AI实施的更广泛的讨论。

### NanoVLMs: How small can we go and still make coherent Vision Language Models? 
[[arxiv](https://arxiv.org/abs/2502.07838)] [[cool](https://papers.cool/arxiv/2502.07838)] [[pdf](https://arxiv.org/pdf/2502.07838)]
> **Authors**: Mukund Agarwalla,Himanshu Kumar,Raj Dandekar,Rajat Dandekar,Sreedath Panat
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: 11 pages, 8 figures, 3 tables
- **标题**: Nanovlms：我们可以走多小，仍然制作连贯的视觉语言模型？
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 视觉语言模型（VLMS），例如GPT-4V和Llama 3.2 Vision，它因其在多模式任务中利用大语言模型（LLM）的能力而引起了重要的研究关注。但是，它们的潜力受到固有挑战的限制，包括专有限制，实质性的计算需求和有限的可访问性。诸如GIT和BLIP之类的较小模型表现出明显的局限性，即使经过广泛的培训，也无法生成连贯和一致的文本。这强调了一个关键的查询：VLM可以很小而仍然产生流利且一致的文本？从3-4岁的孩子的出色学习过程中汲取灵感，他们严重依赖于理解和交流的视觉提示，我们介绍了两个新颖的数据集：ShortDESC（具有简洁的图像描述）和Longdesc（包含更详细的图像描述）。这些数据集由图像文本对组成，其中文本仅限于通常由幼儿使用的简单词汇和语法，并以缩放模型GPT-4O产生。使用这些数据集，我们证明可以训练明显较小的VLM，比最大的vlms（SOTA）小VLM小10倍，同时保持体系结构的简单性。为了评估输出，我们利用GPT-4O来对文本进行评分，好像由学生编写的故事，关于创造力，有意义和一致性，分配10分的分数。此方法通过适应非结构​​化的输出并提供模型功能的多维评估来解决标准基准测试的限制。我们的发现有助于开发用于资源约束环境的轻质，可访问的多模式模型。

### Captured by Captions: On Memorization and its Mitigation in CLIP Models 
[[arxiv](https://arxiv.org/abs/2502.07830)] [[cool](https://papers.cool/arxiv/2502.07830)] [[pdf](https://arxiv.org/pdf/2502.07830)]
> **Authors**: Wenhao Wang,Adam Dziedzic,Grace C. Kim,Michael Backes,Franziska Boenisch
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: Accepted at ICLR 2025
- **标题**: 由字幕捕获：关于记忆及其在剪辑模型中的缓解
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 多模式模型（例如剪辑）在对齐视觉和文本表示方面表现出了很强的性能，在图像检索和零摄影分类等任务中表现出色。尽管取得了成功，但这些模型利用训练数据的机制，尤其是记忆的作用，尚不清楚。在指导和自我监督的单模式模型中，记忆已被证明对于概括至关重要。但是，尚不清楚这些发现将如何适用于剪辑，这些发现将两者通过字幕提供了与标签相似的监督信号的元素，以及通过对比度目标的自我监督学习的元素。为了弥合理解差距，我们提出了对剪辑（Clipmem）中记忆的形式定义，并使用它来量化剪辑模型中的记忆。我们的结果表明，剪辑的记忆行为落在受监督和自我监督的范式之间，“错误捕获”样本表现出最高水平的记忆。此外，我们发现文本编码器比图像编码器对记忆的贡献更大，这表明缓解策略应集中在文本域上。在这些见解的基础上，我们提出了多种策略来减少记忆，同时改善公用事业 - 在传统学习范式之前没有展示过，在传统学习范式中，降低记忆通常会导致公用事业减少。

### Deep Learning in Automated Power Line Inspection: A Review 
[[arxiv](https://arxiv.org/abs/2502.07826)] [[cool](https://papers.cool/arxiv/2502.07826)] [[pdf](https://arxiv.org/pdf/2502.07826)]
> **Authors**: Md. Ahasan Atick Faisal,Imene Mecheter,Yazan Qiblawey,Javier Hernandez Fernandez,Muhammad E. H. Chowdhury,Serkan Kiranyaz
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: 40 pages, 12 figures
- **标题**: 自动化电源线检查中的深度学习检查：评论
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 近年来，通过朝着计算机视觉驱动的自动化检查迈进，电力线维护范围已经发生了范式的转变。大量视频和图像的利用对于维持电力传输的可靠性，安全性和可持续性至关重要。在最近的研究中，观察到了将深度学习技术应用于增强电力线检查过程的重要重点。本文已经对现有研究进行了全面综述，以帮助研究人员和行业开发改进的基于深度学习的系统，以分析电力线数据。已经检查了电力线检查中数据分析的常规步骤，并且当前研究的主体已系统地分为两个主要领域：组件的检测和故障诊断。已经封装了这些领域中采用的各种方法和技术的详细摘要，从而提供了对其功能和用例的见解。特别注意探索基于深度学习的方法，用于分析电源线检查数据，并阐述了其基本原理和实际应用。此外，已经概述了对未来研究方向的愿景，强调了对Edge-Cloud协作以及多模式分析等进步的需求。因此，本文是研究人员深入学习电力线分析的综合资源，阐明了当前知识的程度和未来研究的潜在领域。

### PDM-SSD: Single-Stage Three-Dimensional Object Detector With Point Dilation 
[[arxiv](https://arxiv.org/abs/2502.07822)] [[cool](https://papers.cool/arxiv/2502.07822)] [[pdf](https://arxiv.org/pdf/2502.07822)]
> **Authors**: Ao Liang,Haiyang Hua,Jian Fang,Wenyu Chen,Huaici Zhao
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: PDM-SSD：单阶段的三维对象检测器随点扩张
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 当前基于点的探测器只能从提供的点学中学习，并且对此类目标的全球学习能力不足。在本文中，我们提出了一种利用这两种表示的单阶段3D检测（PDM-SSD）的新颖点扩张机制。具体而言，我们首先使用PointNet风格的3D主干来进行有效的特征编码。然后，使用点扩张机构（PDM）的颈部扩展特征空间，该空间涉及两个关键步骤：点扩张和特征填充。前者扩大了一个尺寸的网格，以欧几里得空间中的采样点为中心。后者用球形谐波系数和高斯密度函数在方向和规模方面填充了空置网格的功能。接下来，我们将多个扩张中心和融合系数关联，以通过高度压缩获得稀疏的网格特征。最后，我们设计了一个用于关节学习的混合检测头，一方面，预测场景的热图可以补充设定的投票点以提高检测准确性，另一方面，通过功能融合来校准检测到的盒子的目标概率。在具有挑战性的Karlsruhe技术研究所和丰田技术学院（KITTI）数据集上，PDM-SSD在单模式方法中获得了最先进的结果，用于具有68帧的单模式方法。我们还证明了PDM-SSD在通过许多对象级实例中检测稀疏和不完整的对象方面的优势。此外，PDM可以用作辅助网络，以在采样点和对象中心之间建立连接，从而在不牺牲推理速度的情况下提高模型的准确性。我们的代码将在https://github.com/alanliangc/pdm-ssd.git上找到。

### Magic 1-For-1: Generating One Minute Video Clips within One Minute 
[[arxiv](https://arxiv.org/abs/2502.07701)] [[cool](https://papers.cool/arxiv/2502.07701)] [[pdf](https://arxiv.org/pdf/2502.07701)]
> **Authors**: Hongwei Yi,Shitong Shao,Tian Ye,Jiantong Zhao,Qingyu Yin,Michael Lingelbach,Li Yuan,Yonghong Tian,Enze Xie,Daquan Zhou
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: Serious updates are needed
- **标题**: 魔术1对1：在一分钟内生成一分钟的视频剪辑
- **领域**: 计算机视觉和模式识别
- **摘要**: 在此技术报告中，我们提出了魔术1-ther-1（魔术141），这是一种有效的视频生成模型，具有优化的内存消耗和推理潜伏期。关键想法很简单：将文本对视频生成任务分解为两个单独的易于扩散步骤蒸馏的任务，即文本到图像生成和图像到视频生成。我们验证使用相同的优化算法，图像到视频任务确实更容易在文本到视频任务上收敛。我们还探索了一袋优化技巧，以减少从三个方面训练图像到视频（I2V）模型的计算成本：1）通过使用多模式的先验状态注入，模型收敛加速； 2）通过应用对抗步骤蒸馏来提高推理潜伏期的速度，3）推理内存成本优化，参数稀疏。借助这些技术，我们能够在3秒内生成5秒的视频剪辑。通过应用测试时间滑动窗口，我们能够在一分钟内生成一个长时间的视频，并具有显着改善的视觉质量和运动动态，平均生成1秒的视频剪辑的花费不到1秒钟。我们进行了一系列初步探索，以找出扩散步骤蒸馏期间计算成本和视频质量之间的最佳权衡，并希望这可能是开源探索的良好基础模型。代码和型号的权重可在https://github.com/da-group-pku/magic-1-for-1上获得。

### Matrix3D: Large Photogrammetry Model All-in-One 
[[arxiv](https://arxiv.org/abs/2502.07685)] [[cool](https://papers.cool/arxiv/2502.07685)] [[pdf](https://arxiv.org/pdf/2502.07685)]
> **Authors**: Yuanxun Lu,Jingyang Zhang,Tian Fang,Jean-Daniel Nahmias,Yanghai Tsin,Long Quan,Xun Cao,Yao Yao,Shiwei Li
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: Project Page: https://nju-3dv.github.io/projects/matrix3d
- **标题**: Matrix3D：大型摄影测量模型多合一
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们提出了Matrix3d，这是一个执行多个摄影测量子任务的统一模型，包括使用同一模型的姿势估计，深度预测和新型视图合成。 MATRIX3D利用多模式扩散变压器（DIT）来整合几种模态的转换，例如图像，相机参数和深度图。 MATRIX3D大规模多模式训练的关键在于融合面具学习策略。即使通过部分完整的数据，例如图像置态和图像深度对的双模式数据，这也可以实现全模式模型训练，从而大大增加了可用训练数据的库。 MATRIX3D在姿势估计和新型视图综合任务中展示了最先进的性能。此外，它通过多轮交互提供了细粒度的控制，使其成为创建3D内容的创新工具。项目页面：https：//nju-3dv.github.io/projects/matrix3d。

### Scaling Pre-training to One Hundred Billion Data for Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.07617)] [[cool](https://papers.cool/arxiv/2502.07617)] [[pdf](https://arxiv.org/pdf/2502.07617)]
> **Authors**: Xiao Wang,Ibrahim Alabdulmohsin,Daniel Salz,Zhe Li,Keran Rong,Xiaohua Zhai
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: 将预培训缩放到视觉语言模型的十亿数据
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们提供了对训练前视觉模型模型的潜力的实证研究：1000亿个例子。我们发现，模型性能倾向于在许多常见的以西方为中心的分类和检索基准（例如可可标题）上以这种规模饱和。然而，由于其对长尾概念的覆盖范围，文化多样性的任务从1000亿个制度的Web数据中获得了更大的收益。此外，我们还分析了该模型的多语言性，并以低资源语言显示了增长。此外，我们观察到，通过质量过滤器（通常用于增强性能）来减少预训练数据集的大小，甚至可能会无意中降低甚至在大型数据集中所代表的文化多样性。我们的结果表明，尽管传统基准可能无法从嘈杂的原始网络数据扩展到1000亿个示例中受益匪浅，但此数据量表对于构建真正包容性的多模式系统至关重要。

### Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.07601)] [[cool](https://papers.cool/arxiv/2502.07601)] [[pdf](https://arxiv.org/pdf/2502.07601)]
> **Authors**: Jiacong Xu,Shao-Yuan Lo,Bardia Safaei,Vishal M. Patel,Isht Dwivedi
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 19 pages, 10 figures
- **标题**: 通过多模式大语言模型进行零射击异常检测和推理
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 零射异常检测（ZSAD）是新兴的AD范式。与需要大量普通样品训练模型的传统无监督的广告设置不同，ZSAD对于处理限制数据限制的现实世界情景更为实用。最近，多模式的大语言模型（MLLM）在各种视觉任务中显示了革命性的推理能力。但是，由于缺乏相应的数据集和基准，图像异常的推理仍然没有被忽视。为了促进广告和推理的研究，我们建立了第一个视觉教学调谐数据集，Anomaly-Instruct-1125K和评估基准Visa-D＆R。通过使用我们的基准进行研究，我们揭示了当前的MLLM（例如GPT-4O）无法准确检测并描述图像中细粒度的异常细节。为了解决这个问题，我们提出了ZSAD和推理的第一位专业视觉助手，提议异常换情（Anomaly-OV）。受到视觉检查的人类行为的启发，Anomaly-OV利用了外观特征功能匹配（LTFM）机制来适应性地选择和强调异常的视觉令牌。广泛的实验表明，在检测和推理方面，异常-OV对高级通才模型取得了重大改进。提供医学和3D广告的扩展供将来的研究。我们项目页面的链接：https：//xujiacong.github.io/anomaly-ov/

### EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering 
[[arxiv](https://arxiv.org/abs/2502.07411)] [[cool](https://papers.cool/arxiv/2502.07411)] [[pdf](https://arxiv.org/pdf/2502.07411)]
> **Authors**: Sheng Zhou,Junbin Xiao,Qingyun Li,Yicong Li,Xun Yang,Dan Guo,Meng Wang,Tat-Seng Chua,Angela Yao
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: egotextvqa：倾向于以egipentric场景 - 培训意识到视频问题回答
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 我们介绍了Egotextvqa，这是一种小说且严格构建的基准，用于涉及场景文本的Egentric Qa援助。 eGoteXtVQA包含1.5k的自我视频视频和7K场景 - 文本意识到问题，这些问题反映了室外驾驶和室内房屋保管活动中的房地产用户需求。这些问题旨在在以自我为中心和动态的环境中引起场景文本的识别和推理。借助EgotextVQA，我们全面评估了10种突出的多模式模型。目前，所有模型都在努力，最佳结果（Gemini 1.5 Pro）的精度约为33％，强调了这些技术在Egintric QA援助中的严重缺陷。我们的进一步调查表明，精确的时间基础和多框架推理以及高分辨率和辅助场景文本输入是更好的性能的关键。通过彻底的分析和启发式建议，我们希望EgotextVQA可以作为以Egontric Sc​​ene-text Text QA援助的研究为实体测试。

### MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification 
[[arxiv](https://arxiv.org/abs/2502.07409)] [[cool](https://papers.cool/arxiv/2502.07409)] [[pdf](https://arxiv.org/pdf/2502.07409)]
> **Authors**: Anh-Tien Nguyen,Duy Minh Ho Nguyen,Nghiem Tuong Diep,Trung Quoc Nguyen,Nhat Ho,Jacqueline Michelle Metsch,Miriam Cindy Maurer,Daniel Sonntag,Hanibal Bohnenberger,Anne-Christin Hauschild
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: first version
- **标题**: MGPATH：具有多个迅速学习的视觉语言模型，用于几次WSI分类
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 整个幻灯片病理图像分类呈现出由于吉像素图像的大小和有限的注释标签引起的挑战，阻碍了模型的概括。本文介绍了一种及时的学习方法，以适应大型视力模型，以进行几种射击病理分类。我们首先，通过在13亿病理图像图块上预先培训的Prod Gigapath视觉基础模型通过添加适配器并通过923K Image-Text Pairs上的对比度学习将其与医学文本编码器保持一致，并将其与医学文本编码器保持一致。然后，该模型用于从几个带有可学习的及时嵌入的微型注释和微调中提取视觉特征和文本嵌入。与使用前缀嵌入或自我注意力结合提示与冷冻特征的先前方法不同，我们提出了多个粒度关注，以比较可学习的提示与单个图像贴片和它们的组之间的相互作用。这种方法提高了模型捕获细粒细节和更广泛的环境的能力，从而增强了其对跨区域复杂模式的认识。为了进一步提高准确性，我们利用（不平衡）基于最佳的传输视觉距离来确保模型鲁棒性通过减轻数据增强过程中可能发生的扰动。关于肺，肾脏和乳房病理方式的经验实验验证了我们方法的有效性；因此，我们超越了几个最新的竞争对手，并不断提高各种体系结构的性能，包括剪辑，PLIP和Prov-Gigapath Integrated Plip。我们在此MGPATH上发布了我们的实现和预培训模型。

### PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology 
[[arxiv](https://arxiv.org/abs/2502.08916)] [[cool](https://papers.cool/arxiv/2502.08916)] [[pdf](https://arxiv.org/pdf/2502.08916)]
> **Authors**: Fatemeh Ghezloo,Mehmet Saygin Seyfioglu,Rustin Soraki,Wisdom O. Ikezogwo,Beibin Li,Tejoram Vivekanandan,Joann G. Elmore,Ranjay Krishna,Linda Shapiro
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 探路者：一种用于组织病理学的医学诊断决策的多模式多代理系统
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,多代理系统
- **摘要**: 通过组织病理学诊断疾病整个幻灯片图像（WSIS）在现代病理学中是基础的，但受WSIS的千兆像素量表和复杂性挑战。受过训练的组织病理学家通过浏览WSI，寻找相关的补丁，记下笔记并编译它们以产生最终的整体诊断来克服这一挑战。传统的AI方法，例如多个实例学习和基于变压器的模型，失败了这种整体，迭代，多规模诊断程序，从而限制了它们在现实世界中的采用。我们介绍了Pathfinder，这是一种多模式的多代理框架，模仿了专家病理学家的决策过程。 Pathfinder集成了四个AI代理，即分类代理，导航代理，描述代理和诊断代理，它们会协作浏览WSIS，收集证据并提供自然语言解释的全面诊断。分诊代理将WSI归类为良性或风险；如果有风险，导航和描述代理会迭代地集中在重要区域上，从而产生了采样贴片的重要性图和描述性见解。最后，诊断剂综合了发现，以确定患者的诊断分类。我们的实验表明，探路者在皮肤黑色素瘤诊断中的最先进方法的表现高出8％，同时通过自然语言描述诊断相关的斑块提供了固有的解释性。病理学家的定性分析表明，描述代理的输出具有高质量，与GPT-4O相当。 Pathfinder也是第一个基于AI的系统，可以超过这项挑战性黑色素瘤分类任务的病理学家的平均表现，从而为病理学中的有效，准确且可解释的AI辅助诊断创造了新的记录。 https://pathfinder-dx.github.io/可用的数据，代码和模型

### SB-Bench: Stereotype Bias Benchmark for Large Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.08779)] [[cool](https://papers.cool/arxiv/2502.08779)] [[pdf](https://arxiv.org/pdf/2502.08779)]
> **Authors**: Vishal Narnaware,Ashmal Vayani,Rohit Gupta,Sirnam Swetha,Mubarak Shah
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: SB台：大型多模型的刻板印象偏置基准
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型多模型（LMM）中的刻板印象偏见使有害的社会偏见永存，破坏了AI应用的公平性和平等性。随着LMM越来越有影响力，在现实世界情景中，与刻板印象，有害世代和模棱两可的假设有关的固有偏见已经变得至关重要。但是，评估LMM中刻板印象偏见的现有数据集通常缺乏多样性并依赖合成图像，从而在现实世界中的视觉环境中给予偏见评估差距。为了解决这个问题，我们介绍了刻板印象的偏置基准（SB基础），这是迄今为止最全面的框架，用于评估具有非合成图像的九种不同类别的刻板印象偏见。 SB板凳通过精心策划的，视觉接地的场景严格评估LMM，挑战它们准确地推理了视觉刻板印象。它提供了一个可靠的评估框架，其中包含现实世界的视觉样本，图像变化和多项选择问题格式。通过引入视觉扎根的查询，将视觉偏见与文本偏差分离出来，SB板台上可以对模型的推理能力进行精确而细微的评估。通过严格测试最先进的开源和封闭源LMM，SB板凳提供了一种系统的方法来评估关键社会维度的LMMS中的刻板印象偏见。该基准是迈向促进AI系统中公平性并减少有害偏见的重要一步，为更公平且对社会负责的LMM奠定了基础。我们的代码和数据集公开可用。

### PulseCheck457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.08636)] [[cool](https://papers.cool/arxiv/2502.08636)] [[pdf](https://arxiv.org/pdf/2502.08636)]
> **Authors**: Xingrui Wang,Wufei Ma,Tiezheng Zhang,Celso M de Melo,Jieneng Chen,Alan Yuille
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: Pulsecheck457：用于大型多模型6D空间推理的诊断基准
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管大型多模型模型（LMM）在视觉场景的解释和推理中表现出显着的功能，但它们具有复杂而精确的三维空间推理的能力仍然不确定。现有的基准主要集中在2D空间理解上，并且缺乏在各种复杂性之间全面评估6D空间推理的框架。为了解决此限制，我们提出了PulseCheck457，这是一种可扩展且无偏的合成数据集，设计具有4个用于空间推理的关键功能：多对象识别，2D位置，3D位置和3D方向。我们开发了一个级联的评估结构，在5个难度级别上构建了7种问题类型，范围从基本的单个对象识别到我们新提出的复杂6D空间推理任务。我们评估了Pulsecheck457上的各种大型多模型（LMM），观察到任务复杂性的增加，性能的总体下降，尤其是在3D推理和6D空间任务中。为了量化这些挑战，我们引入了相对性能下降率（RPDR），突出了3D推理能力中的关键弱点。利用数据集的无偏属性设计，我们还发现了跨不同属性的预测偏差，在现实世界图像设置中观察到了相似的模式。

### A Novel Approach to for Multimodal Emotion Recognition : Multimodal semantic information fusion 
[[arxiv](https://arxiv.org/abs/2502.08573)] [[cool](https://papers.cool/arxiv/2502.08573)] [[pdf](https://arxiv.org/pdf/2502.08573)]
> **Authors**: Wei Dai,Dequan Zheng,Feng Yu,Yanrong Zhang,Yaohui Hou
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 多模式情感识别的新方法：多模式语义信息融合
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 随着人工智能和计算机视觉技术的发展，多模式情感识别已成为一个著名的研究主题。但是，现有方法面临着诸如异质数据融合和模态相关性的有效利用之类的挑战。本文基于对比度学习和视觉序列压缩的整合，提出了一种新型的多模式识别方法DeepMsi-Mer。所提出的方法通过对比度学习增强了跨模式特征融合，并通过利用视觉序列压缩来降低视觉模态的冗余。在Iemocap和Meld的两个公共数据集上进行的实验结果表明，DeepMsi-Mer显着提高了情绪识别的准确性和鲁棒性，从而验证了多模式特征融合的有效性和拟议的方法。

### Human-Centric Foundation Models: Perception, Generation and Agentic Modeling 
[[arxiv](https://arxiv.org/abs/2502.08556)] [[cool](https://papers.cool/arxiv/2502.08556)] [[pdf](https://arxiv.org/pdf/2502.08556)]
> **Authors**: Shixiang Tang,Yizhou Wang,Lu Chen,Yuan Wang,Sida Peng,Dan Xu,Wanli Ouyang
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: 9 pages
- **标题**: 以人为中心的基础模型：感知，产生和代理建模
- **领域**: 计算机视觉和模式识别,人工智能,机器学习,多媒体
- **摘要**: 人类的理解和产生对于建模数字人类和人形体实施方案至关重要。最近，受通才模型（例如大语言和视觉模型）成功启发的以人为中心的基础模型（HCFM）已出现，将各种以人为中心的任务统一为单个框架，超过了传统的特定任务方法。在这项调查中，我们提出了将当前方法分类为四组的分类法，介绍了HCFMS的全面概述：（1）以人为中心的感知基础模型，该模型捕获了用于多模式2D和3D理解的精细元素特征。 （2）以人为中心的AIGC基金会模型产生高保真性，与人类相关的内容多样化。 （3）整合这些能力以增强人类理解和综合的统一感知和产生模型。 （4）以人类体现的任务学习人类智力和互动行为，超越了人们的感知和产生，超越了人类的智力和互动行为。我们回顾了最新的技术，讨论新兴的挑战和未来的研究方向。这项调查旨在作为研究人员和从业人员的路线图，以更加健壮，多功能和智能的数字人类和实施例建模。

### Referring Remote Sensing Image Segmentation via Bidirectional Alignment Guided Joint Prediction 
[[arxiv](https://arxiv.org/abs/2502.08486)] [[cool](https://papers.cool/arxiv/2502.08486)] [[pdf](https://arxiv.org/pdf/2502.08486)]
> **Authors**: Tianxiang Zhang,Zhaokun Wen,Bo Kong,Kecheng Liu,Yisi Zhang,Peixian Zhuang,Jiangyun Li
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 通过双向对齐指导的关节预测参考遥感图像分割
- **领域**: 计算机视觉和模式识别
- **摘要**: 参考遥感图像分割（RRSIS）对于生态监测，城市规划和灾难管理至关重要，需要在文本描述引导的遥感图像中精确细分对象。由于具有巨大的视觉语言差距，高空间分辨率和具有不同类别和小目标的遥感图像的广泛覆盖以及存在模糊边缘的群集，不清楚的目标，因此这项任务具有巨大的挑战。为了解决这些问题，我们提出了一个新颖的框架，旨在弥合视觉差距，增强多尺度特征交互并改善细粒的对象差异。具体而言，\我们的介绍：（1）双向空间相关（BSC），用于改进视觉语言特征对齐，（2）目标 - 靠地面的双式双座解码器（T-BTD），用于目标与非距离之间的目标之间的精确区别，以及（3）双重模式对象学习策略（3）功能强大的跨度（D）跨度跨度（3）。基准数据集和RRSIS-D上的大量实验表明，\我们可以达到最新的性能。具体而言，\我们的总体IOU（OIOU）分别提高了3.76个百分点（80.57）和1.44个百分点（79.23），分别为两个数据集。此外，它在平均值（MIOU）中的先前方法的表现要优于5.37个百分点（67.95）和1.84个百分点（66.04），从而有效地解决了RRSIS的核心挑战，并具有增强的精度和鲁棒性。

### mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data 
[[arxiv](https://arxiv.org/abs/2502.08468)] [[cool](https://papers.cool/arxiv/2502.08468)] [[pdf](https://arxiv.org/pdf/2502.08468)]
> **Authors**: Haonan Chen,Liang Wang,Nan Yang,Yutao Zhu,Ziliang Zhao,Furu Wei,Zhicheng Dou
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: MME5：通过高质量的合成数据改善多模式多语言嵌入
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 多模式嵌入模型已引起了他们对从不同模式（例如文本和图像）映射数据为统一表示空间的能力的重大关注。但是，有限的标记多模式数据通常会阻碍嵌入性能。最近的方法已利用数据综合来解决此问题，但合成数据的质量仍然是关键的瓶颈。在这项工作中，我们确定了高质量合成多模式数据的三个标准。首先，广泛的范围确保生成的数据涵盖了各种任务和方式，使其适用于各种下游场景。其次，稳健的跨模式比对在语义上具有不同的方式。第三，高保真度确保合成数据维护现实的细节以提高其可靠性。在这些原则的指导下，我们合成了：（1）涵盖多种任务，模态组合和语言，（2）是通过在多模式大型语言模型的单个通过中通过深思熟虑的过程生成的，（3）通过自我vifeelity和自我vresseval和“自我”和“自我”来确保真实的图像，以确保现实世界中的图像。利用这些高质量的合成和标记的数据集，我们训练多模式的多语言E5 Model MME5。广泛的实验表明，MME5在MMEB基准测试和XTD基准上的高级多语言性能上实现了最先进的性能。我们的代码，数据集和模型在https://github.com/haon-chen/mme5中发布。

### Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions 
[[arxiv](https://arxiv.org/abs/2502.08438)] [[cool](https://papers.cool/arxiv/2502.08438)] [[pdf](https://arxiv.org/pdf/2502.08438)]
> **Authors**: Prajwal Gatti,Kshitij Parikh,Dhriti Prasanna Paul,Manish Gupta,Anand Mishra
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: Accepted at AAAI 2024, 9 pages. Project Website: https://vl2g.github.io/projects/cstbir
- **标题**: 复合草图+用于检索具有难以捉摸的名称和复杂交互的对象的文本查询
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,信息检索,多媒体
- **摘要**: 词汇量有限的非本地扬声器通常很难命名特定的物体，例如，澳大利亚以外的人正在寻找Numbats。此外，用户可能需要搜索具有难以辨认交互的难以捉摸的对象，例如，在地面上挖掘了Numbat。在这种常见但复杂的情况下，用户希望接受一个搜索界面，该搜索接口接受复合多模式查询，其中包括难以名称但易于绘制的对象的手绘草图，并描述了难以熟悉但易于进行语言的对象属性或与场景的互动的文本。这个新颖的问题陈述与以前经过精心研究的TBIR（基于文本的图像检索）和SBIR（基于素描的图像检索）问题明显不同。为了研究此不足探索的任务，我们策划了一个数据集CSTBIR（Composite Sketch+基于文本的图像检索），包括大约。 2M查询和108K自然场景图像。此外，作为解决此问题的解决方案，我们提出了一个预处理的基于多模式变压器的基线STNET（Sketch+Text Network），该基线使用手绘草图将相关对象定位在自然场景图像中，并编码文本和图像以执行图像检索。除了对比度学习外，我们还提出了改善模型性能的多个培训目标。广泛的实验表明，我们提出的方法的表现优于几种最新的检索方法，用于仅文本，仅素描和复合查询方式。我们在项目网站上提供数据集和代码。

### UniCoRN: Unified Commented Retrieval Network with LMMs 
[[arxiv](https://arxiv.org/abs/2502.08254)] [[cool](https://papers.cool/arxiv/2502.08254)] [[pdf](https://arxiv.org/pdf/2502.08254)]
> **Authors**: Maximilian Jaritz,Matthieu Guillaumin,Sabine Sternig,Loris Bazzani
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 独角兽：使用LMMS的统一评论的检索网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式检索方法在处理复合物，组成查询中具有局限性，这些查询需要有关查询和检索到的实体的视觉内容的推理。另一方面，大型多模型模型（LMM）可以用语言回答更复杂的视觉问题，但没有固有的能力来检索相关实体以支持其答案。我们旨在通过独角兽（Unicorn）解决这些局限性，Unicorn是一个统一的评论检索网络，结合了构成的多模式检索方法和生成语言方法的优势，而不是检索授权的一代（RAG）。我们介绍一个实体适配器模块，以将检索到的多模式的实体注入LMM中，以便在生成答案和评论的同时参加它们。通过保持基本LMM冷冻，Unicorn保留了其原始功能，同时能够在单个集成框架下同时执行检索和文本生成任务。为了评估这些新能力，我们介绍了评论的检索任务（COR）和相应的数据集，目的是检索准确回答给定问题的图像，并产生额外的文本响应，以提供更多澄清和有关视觉信息的详细信息。我们证明了独角兽对几个数据集的有效性，显示 +4.5％的召回率在构成的多模式检索的状态下， +14.9％的流星 / +18.4％BEM比RAG对COR进行评论。

### Learning Human Skill Generators at Key-Step Levels 
[[arxiv](https://arxiv.org/abs/2502.08234)] [[cool](https://papers.cool/arxiv/2502.08234)] [[pdf](https://arxiv.org/pdf/2502.08234)]
> **Authors**: Yilu Wu,Chenhui Zhu,Shuai Wang,Hanlin Wang,Jing Wang,Zhaoxiang Zhang,Limin Wang
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 在关键水平学习人类技能生成器
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们致力于以键步水平学习人类技能生成器。技能的产生是一项具有挑战性的努力，但是其成功的实施可以极大地促进人类的技能学习，并为具体的智力提供更多的经验。尽管当前的视频生成模型可以简单而原子的人类操作，但由于其复杂的程序过程，它们在人类技能上挣扎。人类技能涉及多步骤，长期的动作和复杂的场景过渡，因此现有的幼稚自动回火方法用于综合长视频无法产生人类技能。为了解决这个问题，我们提出了一项新颖的任务，即关键步骤的生成（KS-GEN），旨在降低生成人类技能视频的复杂性。鉴于初始状态和技能描述，任务是生成关键步骤的视频剪辑以完成技能，而不是全长视频。为了支持此任务，我们引入了精心策划的数据集并定义多个评估指标以评估性能。考虑到KS-Gen的复杂性，我们为这项任务提出了一个新的框架。首先，多模式大语言模型（MLLM）使用检索参数为关键步骤生成描述。随后，我们使用键步图像生成器（KIG）来解决技能视频中的关键步骤之间的不连续性。最后，视频生成模型使用这些描述和键步图像来生成具有高时间一致性的关键步骤的视频剪辑。我们对结果进行了详细的分析，希望为人类技能提供更多见解。所有模型和数据均可在https://github.com/mcg-nju/ks-gen上找到。

### Insect-Foundation: A Foundation Model and Large Multimodal Dataset for Vision-Language Insect Understanding 
[[arxiv](https://arxiv.org/abs/2502.09906)] [[cool](https://papers.cool/arxiv/2502.09906)] [[pdf](https://arxiv.org/pdf/2502.09906)]
> **Authors**: Thanh-Dat Truong,Hoang-Quan Nguyen,Xuan-Bac Nguyen,Ashley Dowling,Xin Li,Khoa Luu
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 昆虫发现：基础模型和大型多模式数据集，用于视觉昆虫理解
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式对话生成的AI通过学习大量文本图像数据在各种视野和语言理解中表现出了令人印象深刻的能力。但是，当前的对话模型仍然缺乏有关视觉昆虫的知识，因为它们经常受到视觉数据的一般知识的训练。同时，了解昆虫是精确农业中的一个基本问题，有助于促进农业的可持续发展。因此，本文提出了一种新型的多模式对话模型，昆虫式的模型，以促进昆虫域知识中的视觉理解。特别是，我们首先引入了一个新的大型多模式昆虫数据集，并具有视觉昆虫指导数据，该数据能够学习多模式基础模型。我们提出的数据集使对话模型能够理解昆虫的视觉和语义特征。其次，我们提出了一种新的昆虫式模型，这是一种新的通用大型语言和视觉昆虫理解的助手。然后，为了增强学习昆虫特征的能力，我们通过引入一种新的微功能自我监督学习的学习来开发昆虫基础模型，并通过贴片相关的注意机制来捕获昆虫图像之间的微妙差异。我们还提出了描述一致性损失，以通过文本描述改善微功能学习。对我们新的视觉昆虫问题回答基准测试的实验结果进行了评估，这说明了我们提出的视觉昆虫理解方法的有效表现，并在与昆虫相关的任务的标准基准上实现最先进的表现。

### On the robustness of multimodal language model towards distractions 
[[arxiv](https://arxiv.org/abs/2502.09818)] [[cool](https://papers.cool/arxiv/2502.09818)] [[pdf](https://arxiv.org/pdf/2502.09818)]
> **Authors**: Ming Liu,Hao Chen,Jindong Wang,Wensheng Zhang
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 关于多峰语言模型的鲁棒性，分散注意力
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管视觉模型（VLM）在各种应用程序（例如视觉质疑）中取得了重大成功，但它们对迅速变化的韧性仍然是一个不足探索的领域。了解分心如何影响VLM对于改善其现实世界的适用性至关重要，因为在许多实际情况下，投入可能会有嘈杂且无关紧要的信息。本文旨在评估VLM在科学问题回答的背景下对视觉和文本干扰的鲁棒性。我们建立在ScienceQA数据集的基础上，开发了一种新的基准测试，该基准在视觉和文本上下文中引入了分心，以评估这些分心的VLM的推理能力。我们的发现表明，包括GPT-4在内的大多数最不合理的VLM都容易受到各种各样的干扰，在面对分心的情况下，在推理能力方面遭受了明显的降解。值得注意的是，诸如Internvl2之类的模型表明，这些分心的鲁棒性更高。我们还发现，模型对文本分心的敏感性比视觉分散。此外，我们探索了各种缓解策略，例如及时的工程，以抵消分心的影响。尽管这些策略提高了解决方案的准确性，但我们的分析表明，还有很大的改进机会。

### ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.09696)] [[cool](https://papers.cool/arxiv/2502.09696)] [[pdf](https://arxiv.org/pdf/2502.09696)]
> **Authors**: Jonathan Roberts,Mohammad Reza Taesiri,Ansh Sharma,Akash Gupta,Samuel Roberts,Ioana Croitoru,Simion-Vlad Bogolin,Jialu Tang,Florian Langer,Vyas Raina,Vatsal Raina,Hanyi Xiong,Vishaal Udandarao,Jingyi Lu,Shiyang Chen,Sam Purkis,Tianshuo Yan,Wenye Lin,Gyungin Shin,Qiaochu Yang,Anh Totti Nguyen,David I. Atkinson,Aaditya Baranwal,Alexandru Coca,Mikah Dang, et al. (9 additional authors not shown)
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: 20 pages, 13 figures
- **标题**: Zerobench：当代大型多模型的不可能的视觉基准
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型多模型模型（LMM）在解释图像时表现出重大缺陷，并且通过某些措施比小孩或动物的空间认知较差。尽管如此，他们还是在许多流行的视觉基准上获得了很高的分数，并且由于持续的模型进度激增而迅速侵蚀了净空。为了解决这个问题，迫切需要困难的基准测试，这些基准仍然相关。我们通过引入Zerobench-轻巧的视觉推理基准将这个想法达到极限，这对于当代边境LMM是完全不可能的。我们的基准包括100个手动策划的问题和334个难度较少的子问题。我们在Zerobench上评估了20个LMM，所有这些LMM的得分为0.0％，并严格分析错误。为了鼓励视觉理解的进步，我们公开发布了Zerobench。

### IMM-MOT: A Novel 3D Multi-object Tracking Framework with Interacting Multiple Model Filter 
[[arxiv](https://arxiv.org/abs/2502.09672)] [[cool](https://papers.cool/arxiv/2502.09672)] [[pdf](https://arxiv.org/pdf/2502.09672)]
> **Authors**: Xiaohong Liu,Xulong Zhao,Gang Liu,Zili Wu,Tao Wang,Lei Meng,Yuhan Wang
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-14
> **comment**: 8 pages,5 figures
- **标题**: IMM-MOT：具有相互作用多个模型过滤器的新型3D多对象跟踪框架
- **领域**: 计算机视觉和模式识别,机器人技术
- **摘要**: 3D多对象跟踪（MOT）提供周围物体的轨迹，协助机器人或车辆进行更智能的路径计划和避免障碍物。现有的3D MOT方法基于逐个检测框架通常使用单个运动模型在整个跟踪过程中跟踪对象。但是，由于周围环境中的变化，对象可能会改变其运动模式。在本文中，我们在IMM-MOT中介绍了相互作用的多个模型过滤器，该滤波器准确地符合单个对象的复杂运动模式，从而克服了现有方法中单模型跟踪的限制。此外，我们将阻尼窗口机理纳入轨迹生命周期管理中，利用轨迹的连续关联状态来控制其创造和终止，从而减少了被忽视的低信心真实目标的发生。此外，我们提出了基于距离的分数增强模块，该模块通过调整检测分数来增强误报和真实阳性之间的差异，从而提高了得分滤波器的有效性。在Nuscenes Val数据集上，IMM-MOT使用3D点云优于大多数其他单模式模型，达到73.8％的AMOTA。我们的项目可在https://github.com/ap01lo/imm-mot上找到。

### MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency 
[[arxiv](https://arxiv.org/abs/2502.09621)] [[cool](https://papers.cool/arxiv/2502.09621)] [[pdf](https://arxiv.org/pdf/2502.09621)]
> **Authors**: Dongzhi Jiang,Renrui Zhang,Ziyu Guo,Yanwei Li,Yu Qi,Xinyan Chen,Liuhui Wang,Jianhan Jin,Claire Guo,Shen Yan,Bo Zhang,Chaoyou Fu,Peng Gao,Hongsheng Li
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: Project Page: https://mmecot.github.io/
- **标题**: mme-cot：大型多模型中的基准测试链，用于推理质量，鲁棒性和效率
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 用思考链（COT）回答问题已大大提高了大语言模型（LLMS）的推理能力，但其对大型多模型模型（LMM）的影响仍然缺乏系统的评估和深入的研究。在本文中，我们介绍了MME-COT，这是一种专门的基准测试，评估了LMM的COT推理性能，跨越了六个领域：数学，科学，OCR，OCR，逻辑，时空和一般场景。作为该领域的首次全面研究，我们提出了一个彻底的评估套件，其中包含了三个新颖的指标，这些指标可以评估优质水平的推理质量，鲁棒性和效率。利用策划的高质量数据和独特的评估策略，我们对最先进的LMM进行了深入的分析，发现了几个关键见解：1）具有反射机制的模型表明了Kimi K1.5的出色COT质量，其表现优于GPT-4O，并证明了最高质量的结果； 2）COT促使COT经常在感知重度任务上降低LMM的表现，这表明可能有害的过度思考行为； 3）尽管COT质量很高，但具有反射反射的LMM在正常反应和自我校正阶段均显示出明显的效率低下。我们希望MME-COT为推进LMM的多模式推理的基础。项目页面：https：//mmecot.github.io/

### Exploring the Potential of Encoder-free Architectures in 3D LMMs 
[[arxiv](https://arxiv.org/abs/2502.09620)] [[cool](https://papers.cool/arxiv/2502.09620)] [[pdf](https://arxiv.org/pdf/2502.09620)]
> **Authors**: Yiwen Tang,Zoey Guo,Zhuhao Wang,Ray Zhang,Qizhi Chen,Junli Liu,Delin Qu,Zhigang Wang,Dong Wang,Xuelong Li,Bin Zhao
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: The code is released at https://github.com/Ivan-Tang-3D/ENEL
- **标题**: 探索3D LMM中无编码器体系结构的潜力
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 无编码器体系结构已在2D视觉域中进行了初步探索，但仍然是一个空旷的问题，是否可以有效地应用于3D理解场景。在本文中，我们对无编码器体系结构的潜力进行了首次全面研究，以克服基于编码的3D大型多模型（LMM）的挑战。这些挑战包括未能适应各种点云的分辨率以及编码器的点特征不满足大语模型（LLMS）的语义需求。我们确定了3D LMM的关键方面，以删除编码器并使LLM能够扮演3D编码器的作用：1）我们在预训练阶段提出了LLM插入的语义编码策略，探索了各种点云自我选择的损失的效果。我们提出了提取高级语义的混合语义损失。 2）我们在教学调整阶段介绍了分层的几何聚集策略。这将电感偏置纳入LLM早期层中，以关注点云的局部细节。到最后，我们提出了第一个无编码器的3D LMM Enel。我们的7B模型可与当前最新模型Shapellm-13b媲美，分别在分类，字幕和VQA任务上分别达到55.0％，50.92％和42.7％。我们的结果表明，无编码器的体系结构对于在3D理解领域中替换基于编码器的体系结构非常有希望。该代码在https://github.com/ivan-tang-3d/enel上发布

### GAIA: A Global, Multi-modal, Multi-scale Vision-Language Dataset for Remote Sensing Image Analysis 
[[arxiv](https://arxiv.org/abs/2502.09598)] [[cool](https://papers.cool/arxiv/2502.09598)] [[pdf](https://arxiv.org/pdf/2502.09598)]
> **Authors**: Angelos Zavras,Dimitrios Michail,Xiao Xiang Zhu,Begüm Demir,Ioannis Papoutsis
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: 22 pages, 13 figures
- **标题**: 盖亚：用于遥感图像分析的全局，多模式，多尺度视觉语言数据集
- **领域**: 计算机视觉和模式识别
- **摘要**: 地球卫星的连续操作产生了遥不可及的遥感（RS）图像档案。自然语言提出了一个直观的界面，用于访问，查询和解释此类档案的数据。然而，现有的视觉模型（VLM）主要是在网络结束的，嘈杂的图像文本数据上训练的，表现出对RS专业领域的有限接触。这种缺陷会导致RS特定任务的性能不佳，因为常用的数据集通常缺乏详细的，科学的准确的文本描述，而仅强调日期和位置等属性。为了弥合这个关键的差距，我们介绍了Gaia，Gaia是一种专为多尺度，多传感器和多模式RS图像分析而设计的新型数据集。盖亚（Gaia）由205,150个精心策划的RS图像文本对组成，代表与不同空间分辨率相关的各种RS模态。与RS中的现有视觉语言数据集不同，Gaia专门专注于捕获各种RS应用程序，提供有关环境变化，自然灾害和其他各种动态现象的独特信息。该数据集提供了一个空间和时间平衡的分布，遍布全球，涵盖了过去25年，观察值的时间分布平衡。盖亚（Gaia）的构造涉及一个两个阶段的过程：（1）来自知名RS相关资源的图像和随附的文本的针对性网络剪贴，以及（2）使用精心制作的提示，生成五个高质量的，科学的基于科学的合成字幕，以利用GPT-4O的高级视觉 - 语言功能。我们的广泛实验（包括夹子和BLIP2模型的微调）表明，Gaia显着提高了RS图像分类，跨模式检索和图像字幕任务的性能。

### Long-Term TalkingFace Generation via Motion-Prior Conditional Diffusion Model 
[[arxiv](https://arxiv.org/abs/2502.09533)] [[cool](https://papers.cool/arxiv/2502.09533)] [[pdf](https://arxiv.org/pdf/2502.09533)]
> **Authors**: Fei Shen,Cong Wang,Junyao Gao,Qin Guo,Jisheng Dang,Jinhui Tang,Tat-Seng Chua
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 长期谈话通过运动 - 条件扩散模型产生
- **领域**: 计算机视觉和模式识别
- **摘要**: 有条件扩散模型的最新进展显示出了产生逼真的说话视频的希望，但挑战在实现一致的头部移动，同步的面部表情以及在延长世代内准确的唇部同步。 To address these, we introduce the \textbf{M}otion-priors \textbf{C}onditional \textbf{D}iffusion \textbf{M}odel (\textbf{MCDM}), which utilizes both archived and current clip motion priors to enhance motion prediction and ensure temporal consistency.该模型由三个关键要素组成：（1）一个存档的CLIP MOTHIOT-PRIOR，其中包含历史框架和一个保留身份和背景的参考框架； （2）捕获多模式因果关系的当前旋转运动 - 扩散模型，以准确预测头部移动，唇部同步和表达； （3）一种记忆有效的时间注意机制，该机制通过动态存储和更新运动功能来减轻误差的积累。我们还发布了\ textbf {TalkingFace-wild}数据集，这是一个超过200个小时的10个语言镜头的多语言集合。实验结果证明了MCDM在长期说话表面生成的身份和运动连续性方面的有效性。代码，模型和数据集将公开可用。

### Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models 
[[arxiv](https://arxiv.org/abs/2502.09356)] [[cool](https://papers.cool/arxiv/2502.09356)] [[pdf](https://arxiv.org/pdf/2502.09356)]
> **Authors**: Gabriel Tseng,Anthony Fuller,Marlena Reil,Henry Herzog,Patrick Beukema,Favyen Bastani,James R. Green,Evan Shelhamer,Hannah Kerner,David Rolnick
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 伽利略：在验证遥感模型中学习全球和本地特征
- **领域**: 计算机视觉和模式识别
- **摘要**: 从作物地图到洪水检测，遥感中的机器学习具有广泛的社会利益应用。这些应用程序中的遥感数据之间的共同点为量身定制的机器学习模型提供了一个机会，该模型量身定制为遥感，以减少解决各个任务所需的标记数据和精力。但是，这样的模型必须是：（i）足够灵活地摄取不同传感器方式和形状的输入数据（即，空间和时间尺寸的不同），以及（ii）能够模拟不同尺度和类型的地面表面现象。为了解决这一差距，我们提出了伽利略，这是一个经过审计的遥感模型，旨在灵活处理多模式遥感数据。我们还介绍了一种新颖且高效的自学学习方法，以学习大型和小规模的特征，这是以前的模型所无法解决的挑战。我们的伽利略模型在各种遥感任务中获得最先进的结果。

### A Benchmark for Crime Surveillance Video Analysis with Large Models 
[[arxiv](https://arxiv.org/abs/2502.09325)] [[cool](https://papers.cool/arxiv/2502.09325)] [[pdf](https://arxiv.org/pdf/2502.09325)]
> **Authors**: Haoran Chen,Dong Yi,Moyan Cao,Chensen Huang,Guibo Zhu,Jinqiao Wang
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 大型模型的犯罪监视视频分析基准
- **领域**: 计算机视觉和模式识别
- **摘要**: 监视视频中的异常分析是计算机视觉中的关键主题。近年来，多模式的大语言模型（MLLM）在各个领域中具有优于特定于任务的模型。尽管MLLM尤其具有通用性，但由于该领域的过时基准，他们无法提供MLLM式的QAS和有效算法来评估模型的开放式文本响应，因此他们理解异常概念和细节的能力不足。为了填补这一空白，我们提出了一个用于犯罪监视视频分析的基准，其中大型模型称为UCVL，包括1,829个视频，并从UCF-Crime和UCF-Crime注释数据集中重新组织注释。我们设计了六种类型的问题，并产生了多样化的质量检查对。然后，我们开发详细的说明，并使用OpenAI的GPT-4O进行准确的评估。我们基准为八个盛行的MLLM，范围从0.5B到40b参数，结果证明了该基准的可靠性。此外，我们在UCVL的训练套装上进行了finetune llava-onevision。改进验证了我们的数据在视频异常分析中的高质量。

### EmoAssist: Emotional Assistant for Visual Impairment Community 
[[arxiv](https://arxiv.org/abs/2502.09285)] [[cool](https://papers.cool/arxiv/2502.09285)] [[pdf](https://arxiv.org/pdf/2502.09285)]
> **Authors**: Xingyu Qi,He Li,Linjie Li,Zhenyu Wu
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 表演者：视觉障碍社区的情感助手
- **领域**: 计算机视觉和模式识别,计算机与社会
- **摘要**: 大型多模式模型（LMM）的快速发展已经显着推动了人工智能融入实际应用中。可以处理包括视觉，文本和音频在内的多模式数据的视觉问题回答（VQA）系统具有巨大的潜力，可以帮助视觉障碍（VI）社区在浏览复杂而动态的现实世界环境中。但是，现有的VI辅助LMM忽略了VI个体的情感需求，而当前的基准缺乏对这些LMM的情感评估。为了解决这些差距，本文介绍了Emoassist Benchmark，这是一个全面的基准测试，旨在评估VI社区LMM的辅助性能。据我们所知，这是将情商智力作为关键考虑因素的第一个基准。此外，我们提出了表达模型，这是一种专门为VI社区设计的情感辅助LMM。表情助手模型利用直接偏好优化（DPO）与人类情感偏好保持一致。实验结果表明，表情训练模型可显着增强对VI使用者的隐性情绪和意图的认识，提供善解人意的响应并提供可行的指导。具体而言，与预先调整的LMM相比，同理心和建议指标的同理心和建议指标的各自改善在同理心和39.7％中的改善，甚至超过了诸如GPT-4O之类的最先进的LLMS。

### Multimodal HIE Lesion Segmentation in Neonates: A Comparative Study of Loss Functions 
[[arxiv](https://arxiv.org/abs/2502.09148)] [[cool](https://papers.cool/arxiv/2502.09148)] [[pdf](https://arxiv.org/pdf/2502.09148)]
> **Authors**: Annayah Usman,Abdul Haseeb,Tahir Syed
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 新生儿中的多模式HIE病变分割：损失功能的比较研究
- **领域**: 计算机视觉和模式识别
- **摘要**: 新生儿MRI中缺氧 - 缺血性脑病（HIE）病变的分割是一项至关重要但具有挑战性的任务，这是由于弥漫性多灶性病变的体积变化和有限的带注定的Hie Hie lesion数据集的可用性。使用BONBID-HIE数据集，我们实施了一个3D U-NET，具有优化的预处理，增强和培训策略来克服数据约束。这项研究的目的是确定专门针对HIE病变细分任务的最佳损失函数。为此，我们评估了各种损失函数，包括骰子，骰子 - 焦点，Tversky，Hausdorff距离（Hausdorffdt）损失，以及两个提出的化合物损失 - 骰子 - 局限性 - 核 -  hausdorffdt和tversky-hausdorffdt--以增强分割性能。结果表明，不同的损失函数可以预测不同的分割面具，复合损失的表现优于独立损失。 Tversky-Hausdorffdt损失达到了最高的骰子和归一化的表面骰子评分，而骰子 - 焦点 - 霍斯多夫德特损失损失最小化的表面距离最小。这项工作强调了特定于任务的损失函数优化的重要性，表明将基于区域和边界感知的损失结合起来也会导致更准确的HIE病变细分，即使培训数据有限。

### From Visuals to Vocabulary: Establishing Equivalence Between Image and Text Token Through Autoregressive Pre-training in MLLMs 
[[arxiv](https://arxiv.org/abs/2502.09093)] [[cool](https://papers.cool/arxiv/2502.09093)] [[pdf](https://arxiv.org/pdf/2502.09093)]
> **Authors**: Mingxiao Li,Fang Qu,Zhanpeng Chen,Na Su,Zhizhou Zhong,Ziyang Chen,Nan Du,Xiaolong Li
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 从视觉到词汇：通过MLLMS中的自回归预培训在图像和文本令牌之间建立等效性
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管MLLM在感知任务上表现良好，但它们缺乏精确的多模式对准，从而限制了性能。为了应对这一挑战，我们提出了视觉动态嵌入引导预处理（VDEP），这是MLLM的混合自动回归训练范式。在视觉编码器之后，通过使用MLP的动态嵌入，此方法监督图像隐藏状态，并将图像令牌集成到自动回归训练中。现有的MLLM主要集中于从文本输入中恢复信息，通常忽略了图像数据的有效处理。相比之下，这项工作的关键改进是将多模式对准的重新解释为从输入数据中恢复信息的过程，特别是重构详细的视觉特征。该方法无缝地集成到标准模型的情况下而没有建筑变化。 13个基准的实验显示VDEP优于基线，超过了现有方法。

### Evolution of Data-driven Single- and Multi-Hazard Susceptibility Mapping and Emergence of Deep Learning Methods 
[[arxiv](https://arxiv.org/abs/2502.09045)] [[cool](https://papers.cool/arxiv/2502.09045)] [[pdf](https://arxiv.org/pdf/2502.09045)]
> **Authors**: Jaya Sreevalsan-Nair,Aswathi Mundayatt
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 数据驱动的单和多危险易感性映射以及深度学习方法的出现的演变
- **领域**: 计算机视觉和模式识别
- **摘要**: 数据驱动的自然危害易感性映射已利用了代表栅格图像的异质源中使用的分类方法的进步。易感映射是朝着任何自然危害的风险评估迈出的重要一步。越来越多的多种危害在空间上，时间或两者兼而有之，这需要对多危险易感性映射进行深入研究。近年来，单危险易感性映射算法已经建立了完善，并已扩展到多危险的敏感性映射。深度学习也成为单危险易感性映射的有前途的方法。在这里，我们讨论了单个危害的方法的演变，它们扩展到多危险图作为决策的后期融合，以及在敏感性映射中使用深度学习方法。我们最终提出了一个愿景，以使多模式深度学习中的数据融合策略适应多危险易感性映射。从易感方法的背景研究中，我们证明了深度学习模型是多危险易感映射的有希望的，未开发的方法。数据融合策略提供了适用于多危险易感映射的深度学习模型的更大空间。

### Exploiting Point-Language Models with Dual-Prompts for 3D Anomaly Detection 
[[arxiv](https://arxiv.org/abs/2502.11307)] [[cool](https://papers.cool/arxiv/2502.11307)] [[pdf](https://arxiv.org/pdf/2502.11307)]
> **Authors**: Jiaxiang Wang,Haote Xu,Xiaolu Chen,Haodi Xu,Yue Huang,Xinghao Ding,Xiaotong Tu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: 10 pages, 7 figures
- **标题**: 使用双键进行剥削点语言模型，用于3D异常检测
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 3D点云中的异常检测（AD）在广泛的工业应用中至关重要，尤其是在各种形式的精确制造中。考虑到可靠3D AD的工业需求，已经开发了几种方法。但是，这些方法中的大多数通常都需要为每个类别进行培训单独的模型，这是记忆密集型并且缺乏灵活性。在本文中，我们提出了一个新型的点语言模型，该模型具有3D异常检测（平面）的双prompts。该方法利用多模式提示将预训练点语言模型（PLMS）的强大概括能力扩展到3D点云AD的域，从而使用单个模型在多个类别中实现了令人印象深刻的检测性能。具体来说，我们提出了一种双提出学习方法，同时结合了文本和点云提示。该方法利用动态提示创建器模块（DPCM）产生特定于样本的动态提示，然后将其与每种模式的特定于类的静态提示集成，从而有效地驱动PLM。此外，基于点云数据的特征，我们提出了一种伪3D异常生成方法（ANO3D），以在无监督的设置中提高模型的检测能力。实验结果表明，在多级模型范式下，所提出的方法与最先进的单级单位模型方法相比，在异常检测和定位性能方面获得了+8.7％/ +17％的增长，用于异常塑性数据集，并获得+4.3％/ +4.1％/ +4.1％的数据。代码将在出版时提供。

### Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding 
[[arxiv](https://arxiv.org/abs/2502.11168)] [[cool](https://papers.cool/arxiv/2502.11168)] [[pdf](https://arxiv.org/pdf/2502.11168)]
> **Authors**: Xin Gu,Yaojie Shen,Chenxi Luo,Tiejian Luo,Yan Huang,Yuewei Lin,Heng Fan,Libo Zhang
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 了解您的目标：目标感知变压器使更好的时空视频接地
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 由于其端到端的管道和有希望的结果，变压器引起了对STVG的越来越多的兴趣。现有的基于变压器的STVG方法通常利用一组对象查询，这些查询仅使用零来初始化，然后通过具有多模式特征的迭代交互逐渐学习目标位置信息，以进行空间和时间定位。尽管简单起见，但由于缺乏目标特异性提示，这些零对象查询很难从复杂场景中与多模式特征的相互作用中学习歧视性目标信息（例如，分散术或遮挡），从而导致降级。在解决此问题时，我们引入了一种新型的STVG（TA-STVG）的目标感知变压器，该变压器试图通过探索给定的视频文本对的目标特异性提示来适应对象查询，以改善STVG。关键在于两个简单但有效的模块，其中包括文本引导的时间抽样（TTS）和属性感知的空间激活（ASA），它们在级联中工作。前者着重于利用整体文本信息从视频中选择与目标相关的时间提示，而后者旨在从先前的目标感知的时间提示中进一步利用对象的细粒度视觉属性信息，该信息适用于对象查询初始化。与利用零定位的查询的现有方法相比，我们的ta-STVG中的对象查询直接从给定的视频文本对生成，自然携带特定于目标的提示，使它们自适应，并更好地与多模态特征相互作用，以学习学习更多歧视性信息以改善STVG。在我们对三个基准测试的实验中，TA-STVG达到了最先进的性能，并显着优于基线，从而验证了其功效。

### Text-promptable Propagation for Referring Medical Image Sequence Segmentation 
[[arxiv](https://arxiv.org/abs/2502.11093)] [[cool](https://papers.cool/arxiv/2502.11093)] [[pdf](https://arxiv.org/pdf/2502.11093)]
> **Authors**: Runtian Yuan,Jilan Xu,Mohan Chen,Qingqiu Li,Yuejie Zhang,Rui Feng,Tao Zhang,Shang Gao
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 用于引用医学图像序列分割的文本倾向的传播
- **领域**: 计算机视觉和模式识别
- **摘要**: 由基于2D视频的考试和3D成像技术生成的医学图像序列，由顺序框架或切片组成，这些帧或切片从多个角度捕获相同的解剖实体（例如器官或病变）。现有的分割研究通常使用2D或3D方法隔离地处理医学图像，通常忽略这些图像之间的固有一致性。此外，交互式分割虽然在临床方案中非常有益，但面临着整合文本的挑战，可以在多模式之间有效提示。为了解决这些问题，我们介绍了一项创新的任务，首次引用了医学图像序列细分，该任务旨在分割与医学文本提示相对应的引用的解剖实体。我们开发了一个强大的基线模型，即启发文本繁殖（TPP），旨在利用顺序图像之间的内在关系及其相关的文本描述。 TPP支持基于跨模式及时融合的任意对象的分割。精心设计的医疗提示被融合并用作查询，以通过三个传播来指导图像序列分割。我们策划了一个庞大而全面的基准测试，涵盖了4种方式以及20种不同的器官和病变。与这些数据集的先前方法相比，实验结果始终证明了我们方法的出色性能。

### Phantom: Subject-consistent video generation via cross-modal alignment 
[[arxiv](https://arxiv.org/abs/2502.11079)] [[cool](https://papers.cool/arxiv/2502.11079)] [[pdf](https://arxiv.org/pdf/2502.11079)]
> **Authors**: Lijie Liu,Tianxiang Ma,Bingchuan Li,Zhuowei Chen,Jiawei Liu,Qian He,Xinglong Wu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 幻影：通过跨模式对齐的主题一致的视频生成
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 视频发电的基础模型的持续开发正在发展为各种应用程序，主题一致的视频生成仍处于探索阶段。我们将其称为主题到视频，该主题从参考图像中提取主题元素，并通过文本说明生成主题一致的视频。我们认为，主题到视频的本质在于平衡文本和图像的双模式提示，从而深入并同时使文本和视觉内容对齐。为此，我们提出了Phantom，这是单个和多主题参考的统一视频生成框架。在现有的文本到视频和图像到视频体系结构的基础上，我们重新设计了联合文本图像注入模型，并通过文本图像 - 视频 - 视频图三重态数据将其驱动以学习跨模式对齐。特别是，我们强调了人类一代的主题一致性，涵盖了现有的ID保存视频生成，同时提供了增强的优势。项目主页在这里https://phantom-video.github.io/phantom/。

### TPCap: Unlocking Zero-Shot Image Captioning with Trigger-Augmented and Multi-Modal Purification Modules 
[[arxiv](https://arxiv.org/abs/2502.11024)] [[cool](https://papers.cool/arxiv/2502.11024)] [[pdf](https://arxiv.org/pdf/2502.11024)]
> **Authors**: Ruoyu Zhang,Lulu Wang,Yi He,Tongling Pan,Zhengtao Yu,Yingna Li
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: TPCAP：用触发器和多模式纯化模块解锁零拍图像字幕
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型语言模型（LLM）的最新进展显着提高了图像字幕的流利性和逻辑连贯性。检索授权的一代（RAG）被广泛采用，以将外部知识纳入LLM；但是，现有的基于抹布的方法依赖于单独的检索库，引入计算开销并限制了LLMS固有的零击功能的利用。为了解决这些局限性，我们提出了TPCAP，这是一个新颖的触发器和多模式纯化框架，用于零拍图像字幕，而无需外部检索库。 TPCAP由两个关键组成部分组成：触发器（TA）生成和多模式纯化（MP）。 TA模块采用触发投影仪，具有冷冻和可学习的预测，以激活LLMS的上下文推理，增强视觉文本对齐并减轻数据偏差。 MP模块通过过滤噪声和增强功能质量，进一步完善了生成的实体相关信息，从而确保更精确且实际一致的字幕。我们评估COCO，NOCAPS，FLICKR30K和WHOOPS数据集的TPCAP。 TPCAP仅在单个NVIDIA RTX 4090 GPU上进行训练参数和培训，可实现与最新模型相当的竞争性能。

### Skillful Nowcasting of Convective Clouds With a Cascade Diffusion Model 
[[arxiv](https://arxiv.org/abs/2502.10957)] [[cool](https://papers.cool/arxiv/2502.10957)] [[pdf](https://arxiv.org/pdf/2502.10957)]
> **Authors**: Haoming Chen,Xiaohui Zhong,Qiang Zhai,Xiaomeng Li,Ying Wa Chan,Pak Wai Chan,Yuanyuan Huang,Hao Li,Xiaoming Shi
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 使用级联扩散模型对流云进行熟练的凝结
- **领域**: 计算机视觉和模式识别,大气和海洋物理
- **摘要**: 从卫星图像中准确对对流云进行准确，对于减轻气象灾害的影响至关重要，尤其是在发展中国家和偏远地区，基于地面的观测值有限。深度学习的最新进展已在视频预测中显示出希望。但是，现有模型经常产生模糊的结果，并在预测物理领域时表现出降低的精度。在这里，我们介绍了SATCAST，这是一个扩散模型，该模型利用级联体系结构和多模式输入用于卫星图像中的云字段。 SATCAST结合了Fuxi预测的物理领域，Fuxi是一种深入学习的天气模型，与过去的卫星观测值一起作为有条件的输入，以生成高质量的未来云场。通过全面的评估，Satcast在多个指标上的表现优于常规方法，证明了其优异的准确性和鲁棒性。消融研究强调了其多模式设计和级联体系结构在实现可靠预测中的重要性。值得注意的是，SATCAST保持了长达24小时的预测技能，强调了其操作启动应用程序的潜力。

### Transformer-Driven Modeling of Variable Frequency Features for Classifying Student Engagement in Online Learning 
[[arxiv](https://arxiv.org/abs/2502.10813)] [[cool](https://papers.cool/arxiv/2502.10813)] [[pdf](https://arxiv.org/pdf/2502.10813)]
> **Authors**: Sandeep Mandia,Kuldeep Singh,Rajendra Mitharwal,Faisel Mushtaq,Dimpal Janu
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: 22 pages, 5 figures, and 6 tables
- **标题**: 变压器驱动的可变频率功能的建模，用于分类学生参与在线学习
- **领域**: 计算机视觉和模式识别
- **摘要**: Covid-19的大流行和互联网的可用性最近促进了在线学习。但是，监测在线学习中的参与对于教师来说是一项艰巨的任务。在这种情况下，及时的自动学生参与分类可以帮助教师进行自适应调整以满足学生的需求。本文建议使用视频模式进行参与分类，这是一种基于变压器的架构，具有序列池。所提出的体系结构从输入视频中计算三个视图，并使用变压器编码并并行处理它们。然后，全局编码器处理每个编码器的表示形式，最后，多层perceptron（MLP）预测参与度级别。从现有的开源数据库中策划了一个以学习为中心的情感状态数据集。 The proposed method achieved an accuracy of 63.9%, 56.73%, 99.16%, 65.67%, and 74.89% on Dataset for Affective States in E-Environments (DAiSEE), Bahcesehir University Multimodal Affective Database-1 (BAUM-1), Yawning Detection Dataset (YawDD), University of Texas at Arlington Real-Life Drowsiness Dataset （UTA-RLDD），分别以学习为中心的情感状态数据集。在Baum-1，Daisee和Yawdd数据集上实现的结果表明了最先进的性能，这表明所提出的模型在准确地对这些数据集上的情感状态进行分类时的优越性。此外，在涉及两级分类的UTA-RLDD数据集中获得的结果是未来研究的基准。这些结果为进一步的研究奠定了基础，并作为未来工作的参考点，以比较和改进。

### Distraction is All You Need for Multimodal Large Language Model Jailbreaking 
[[arxiv](https://arxiv.org/abs/2502.10794)] [[cool](https://papers.cool/arxiv/2502.10794)] [[pdf](https://arxiv.org/pdf/2502.10794)]
> **Authors**: Zuopeng Yang,Jiluan Fan,Anli Yan,Erdun Gao,Xin Lin,Tao Li,Kanghua mo,Changyu Dong
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 分散注意力是多模式大语言模型越狱所需的
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式大语言模型（MLLMS）弥合了视觉和文本数据之间的差距，从而实现了一系列高级应用程序。但是，视觉元素之间的复杂内部互动及其与文本的一致性可能引入漏洞，可以利用这些漏洞来绕过安全机制。为了解决这个问题，我们分析图像内容与任务之间的关系，发现子图像的复杂性而不是其内容是关键。在这种见解的基础上，我们提出了分散注意力的假设，其次是一个新颖的框架，称为对比鲜明的分散注意力越狱（CS-DJ），以通过多级分心策略破坏MLLM的对准来实现越狱。 CS-DJ由两个组成部分组成：结构化的分散注意力，通过查询分解实现，通过将有害的提示分解为子征服和视觉增强的分心，从而诱导分布转移，从而通过构建对比的子图像来破坏模型中视觉元素之间的相互作用。这种双重策略散布了模型的注意力，从而降低了其检测和减轻有害内容的能力。在五个代表性场景和四个流行的封闭源MLLM中进行了广泛的实验，包括GPT-4O-MINI，GPT-4O，GPT-4V和GEMINI-1.5-FLASH，表明CS-DJ可实现攻击成功率的52.40％的平均成功率，而攻击成功率为74.10％。这些结果揭示了基于分心的方法利用和绕过MLLM的防御能力的潜力，从而为攻击策略提供了新的见解。

### Learning semantical dynamics and spatiotemporal collaboration for human pose estimation in video 
[[arxiv](https://arxiv.org/abs/2502.10616)] [[cool](https://papers.cool/arxiv/2502.10616)] [[pdf](https://arxiv.org/pdf/2502.10616)]
> **Authors**: Runyang Feng,Haoming Chen
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 在视频中学习人姿势估计的语义动力和时空协作
- **领域**: 计算机视觉和模式识别
- **摘要**: 时间建模和时空协作是基于视频的人类姿势估计的关键技术。大多数最先进的方法采用光流或时间差异，在像素级别上学习跨帧的本地视觉内容对应，以捕获运动动力学。但是，这样的范式基本上依赖于局部像素到像素的相似性，这忽略了框架之间的语义相关性，并且容易受到图像质量降解的影响（例如遮挡或模糊）。此外，现有方法通常通过简单的串联或求和结合运动和空间（外观）特征，从而在充分利用这些独特的方式方面面临实际挑战。在本文中，我们提出了一个新颖的框架，该框架学习了多帧人姿势估计的多级语义动力学和密集的时空协作。具体而言，我们首先使用多掩盖上下文和姿势重建策略设计多层语义运动编码器。该策略通过逐步掩盖（贴片）立方体和框架的特征来刺激模型，以探索框架之间多范围的时空语义关系。我们进一步引入了一个空间运动相互学习模块，该模块密集地传播和巩固了从空间和运动特征的上下文信息，以增强模型的能力。广泛的实验表明，我们的方法在三个基准数据集（Posetrack2017，posteTrack2018和posetrack21）上设置了新的最新结果。

### PolyPath: Adapting a Large Multimodal Model for Multi-slide Pathology Report Generation 
[[arxiv](https://arxiv.org/abs/2502.10536)] [[cool](https://papers.cool/arxiv/2502.10536)] [[pdf](https://arxiv.org/pdf/2502.10536)]
> **Authors**: Faruk Ahmed,Lin Yang,Tiam Jaroensri,Andrew Sellergren,Yossi Matias,Avinatan Hassidim,Greg S. Corrado,Dale R. Webster,Shravya Shetty,Shruthi Prabhakara,Yun Liu,Daniel Golden,Ellery Wulczyn,David F. Steiner
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: 8 main pages, 21 pages in total
- **标题**: 息肉路线：适应多模型多模型多模型的多型病理报告生成
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 组织病理病例病例的解释是医学中许多重要的诊断和治疗决定的基础。值得注意的是，此过程通常要求病理学家每个情况跨多个幻灯片整合和总结发现。到目前为止，计算病理学中现有的视力语言能力在很大程度上仅限于感兴趣的小区域，较大的放大倍率或单个全滑动图像（WSIS）。这限制了对跨多个WSI的多个高磁化区域的发现的解释。通过使用Gemini 1.5 Flash，这是一种具有100万令牌上下文窗口的大型多式联运模型（LMM），我们演示了从多个WSIS从多个WSIS产生底线诊断的能力。这相当于1 fps的最多11个小时的视频。专家病理学家评估表明，生成的报告文本在临床上是准确的，相当于或优先于原始报告，其中最多5个幻灯片的示例为68％（95％CI：[60％，76％]）。虽然效果降低了6个或更多幻灯片的示例，但本研究表明了利用现代LMM的长期文化功能来实现医疗报告生成的独特挑战性任务，每个案例都可以包含数千个图像补丁。

### RAMer: Reconstruction-based Adversarial Model for Multi-party Multi-modal Multi-label Emotion Recognition 
[[arxiv](https://arxiv.org/abs/2502.10435)] [[cool](https://papers.cool/arxiv/2502.10435)] [[pdf](https://arxiv.org/pdf/2502.10435)]
> **Authors**: Xudong Yang,Yizhang Zhu,Nan Tang,Yuyu Luo
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-17
> **comment**: 9 pages
- **标题**: 拉默：基于重建的对抗模型，用于多方多模式多标签情绪识别
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 传统的多模式多标签情绪识别（MME）通常从视频中充分利用视觉，文字和声学方式。但是，实际的多方设置经常违反这一假设，因为非说话者经常缺乏声学和文本输入，从而导致模型性能的显着降级。现有的方法还倾向于将异质方式统一为单一表示形式，从而忽略了每种模式的独特特征。为了应对这些挑战，我们提出了Ramer（基于重建的对抗性模型，用于情绪识别），该模型通过通过对比学习增强的重建功能来探索模态性和特异性来利用对抗性学习来完善多模式表示。拉默还介绍了一个个性辅助任务，以使用模态级别的关注来补充缺失的方式，从而改善情绪推理。为了进一步增强模型捕获标签和模态相互依存的能力，我们提出了一种堆栈洗牌策略，以丰富标签和特定于模态特征之间的相关性。在三个基准测试的实验，即备忘录，CMU-Mosei和$ M^3 $ ED上，证明Ramer在二甲和多方Mermer方案中实现了最先进的表现。

### Interpretable Concept-based Deep Learning Framework for Multimodal Human Behavior Modeling 
[[arxiv](https://arxiv.org/abs/2502.10145)] [[cool](https://papers.cool/arxiv/2502.10145)] [[pdf](https://arxiv.org/pdf/2502.10145)]
> **Authors**: Xinyu Li,Marwa Mahmoud
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 多模式人类行为建模的基于概念的可解释的深度学习框架
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 在当代的智能连通性时代，情感计算（AC），使系统能够识别，解释和响应人类行为状态，已成为许多AI系统的综合组成部分。作为所有以人为中心的系统中负责人AI和可信赖性的最关键组成部分之一，解释性在AC中一直是一个主要问题。特别是，最近发布的欧盟一般数据保护法规要求任何高风险的AI系统都具有足够的解释，包括基于生物识别的系统和情感计算领域广泛使用的情感识别系统。现有的可解释方法通常会在解释性和绩效之间妥协。他们中的大多数仅着重于突出关键网络参数，而没有向利益相关者提供有意义的特定领域的解释。此外，他们还在有效共同学习和解释多模式数据源的见解方面面临挑战。为了解决这些局限性，我们提出了一个新颖且可推广的框架，即注意引导概念模型（AGCM），该模型（AGCM）通过确定导致预测及其观察的概念来提供可学习的概念解释。 AGCM可通过多模式概念对准和共同学习扩展到任何空间和时间信号，从而使利益相关者能够深入了解该模型的决策过程。我们验证了AGCM对公认的面部表达识别基准数据集的效率，同时还证明了其对更复杂的现实世界人类行为理解应用的概括性。

### V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.09980)] [[cool](https://papers.cool/arxiv/2502.09980)] [[pdf](https://arxiv.org/pdf/2502.09980)]
> **Authors**: Hsu-kuang Chiu,Ryo Hachiuma,Chien-Yi Wang,Stephen F. Smith,Yu-Chiang Frank Wang,Min-Hung Chen
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: Our project website: https://eddyhkchiu.github.io/v2vllm.github.io/
- **标题**: V2V-LLM：使用多模式大型语言模型的车辆到车辆合作自动驾驶
- **领域**: 计算机视觉和模式识别,机器人技术
- **摘要**: 当前的自动驾驶车辆主要依靠其单个传感器来了解周围的场景并计划未来的轨迹，当传感器发生故障或遮挡时，这可能是不可靠的。为了解决这个问题，已经提出了通过车辆到车辆（V2V）通信的合作感知方法，但它们倾向于专注于检测和跟踪。这些方法如何有助于整体合作计划绩效。受到最新进度使用大型语言模型（LLM）来构建自动驾驶系统的启发，我们提出了一个新的问题设置，将LLM集成到合作的自主驾驶中，并与拟议的车辆到车辆的问题驱动器（V2V-QA）数据集和基准标准。我们还建议使用LLM融合来自多个连接的自动驾驶汽车（CAVS）（CAVS）和回答与驾驶相关的问题的基线方法（V2V-LLM），该方法使用LLM融合感知信息：接地，著名的对象识别和计划。实验结果表明，我们提出的V2V-LLM可以是一种有希望的统一模型结构，用于在合作自主驾驶中执行各种任务，并且要优于使用其他融合方法的其他基线方法。我们的工作还创建了一个新的研究方向，可以提高未来自动驾驶系统的安全性。我们的项目网站：https：//eddyhkchiu.github.io/v2vllm.github.io/。

### Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence 
[[arxiv](https://arxiv.org/abs/2502.09927)] [[cool](https://papers.cool/arxiv/2502.09927)] [[pdf](https://arxiv.org/pdf/2502.09927)]
> **Authors**: Granite Vision Team,Leonid Karlinsky,Assaf Arbelle,Abraham Daniels,Ahmed Nassar,Amit Alfassi,Bo Wu,Eli Schwartz,Dhiraj Joshi,Jovana Kondic,Nimrod Shabtay,Pengyuan Li,Roei Herzig,Shafiq Abedin,Shaked Perek,Sivan Harary,Udi Barzelay,Adi Raz Goldfarb,Aude Oliva,Ben Wieles,Bishwaranjan Bhattacharjee,Brandon Huang,Christoph Auer,Dan Gutfreund,David Beymer, et al. (38 additional authors not shown)
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 花岗岩视觉：一种轻巧的开源多模型，用于企业智能
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 我们介绍了带有视觉功能的轻型大语言模型的花岗岩愿景，专门设计用于在企业用例中出色，尤其是在视觉文档理解中。我们的模型经过了综合指令遵循数据集的培训，包括与文档相关的任务，例如从表，图表，图表，图表，草图和信息图表中提取内容，以及一般图像任务。花岗岩视觉的架构以视觉方式对齐为中心，只有一个仅解码器，20亿个参数花岗岩大语言模型。此外，我们在测试时间中引入了专用的安全分类方法，该方法利用了一组稀疏的注意向量来识别潜在的有害输入。尽管具有轻巧的体系结构，但Granite Vision在与视觉文档理解以及LiveXiv基准测试中相关的标准基准测试中取得了良好的结果，该基准旨在通过使用不断更新的最近发表的Arxiv论文的语料库来避免测试集污染。我们正在根据Apache-2许可证发布该模型，同时允许研究和商业用途，同时为培训数据和其他相关细节提供完整的可见性。有关模型权重，请参见https://huggingface.co/ibm-granite/。

### TaskGalaxy: Scaling Multi-modal Instruction Fine-tuning with Tens of Thousands Vision Task Types 
[[arxiv](https://arxiv.org/abs/2502.09925)] [[cool](https://papers.cool/arxiv/2502.09925)] [[pdf](https://arxiv.org/pdf/2502.09925)]
> **Authors**: Jiankang Chen,Tianke Zhang,Changyi Liu,Haojie Ding,Yaya Shi,Feng Cheng,Huihui Xiao,Bin Wen,Fan Yang,Tingting Gao,Di Zhang
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: TaskGalaxy：使用数以万计的视觉任务类型来缩放多模式指令微调
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式的视觉语言模型在开放世界应用程序中获得了突出性，这是由模型体系结构，培训技术和高质量数据的进步驱动的。但是，它们的性能通常受到特定于任务的数据不足的限制，导致概括和产出偏差。劳动密集型的手动任务标签过程妨碍了现有的努力来增加微调数据集中的任务多样性，该过程通常仅生产几百个任务类型。为了解决这个问题，我们提出了TaskGalaxy，这是一个大规模的多模式指令微型数据集，其中包括19,227个分层任务类型和413,648个样本。 TaskGalaxy利用GPT-4O来通过从一小部分手动定义的任务扩展来丰富任务多样性，剪辑和GPT-4O过滤那些最能匹配开源映像的人，并生成相关的问答 - 答案对。采用多种型号来确保样本质量。这个自动化过程增强了任务多样性和数据质量，从而减少了手动干预。将TaskGalaxy纳入LLAVA-V1.5和Internvl-Chat-V1.0模型中显示了16个基准的绩效改进，这表明了任务多样性的重要性。 TaskGalaxy在https://github.com/kwai-yuanqi/taskgalaxy上公开发布。

### SafeEraser: Enhancing Safety in Multimodal Large Language Models through Multimodal Machine Unlearning 
[[arxiv](https://arxiv.org/abs/2502.12520)] [[cool](https://papers.cool/arxiv/2502.12520)] [[pdf](https://arxiv.org/pdf/2502.12520)]
> **Authors**: Junkai Chen,Zhijie Deng,Kening Zheng,Yibo Yan,Shuliang Liu,PeiJun Wu,Peijie Jiang,Jia Liu,Xuming Hu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: Safeeraser：通过多模式的机器在多模式大型语言模型中增强安全性
- **领域**: 计算机视觉和模式识别
- **摘要**: 随着多模式大语言模型（MLLM）的发展，它们的潜在安全问题变得越来越突出。机器 - 学习（MU）是忘记培训数据中特定知识的有效策略，已被广泛用于隐私保护。但是，在MLLM中的安全性尚未得到充分探索。为了解决这个问题，我们建议使用Safeeraser，这是MLLM的安全性基准，由3,000张图像和28.8k VQA Pairs组成。我们从两个角度全面评估了学习方法：忘记质量和模型实用程序。我们的发现表明，现有的MU方法在实施忘记操作的同时努力保持模型性能，并且经常遭受过度遗忘。因此，我们介绍了迅速的decouple（PD）损失，以减轻在未学习过程中通过Decouple提示过度遗忘的。为了定量测量通过PD损失缓解过度遗忘，我们提出了一个称为安全答案拒绝率（SARR）的新指标。实验结果表明，将PD损失与现有的未学习方法相结合可以有效防止过度遗漏，并在LLAVA-7B和LLAVA-13B的SARR指标中降低79.5％，同时保持忘记的质量和模型效用。我们的代码和数据集将在接受后发布。警告：本文包含有害语言和图像的示例，建议读者酌情决定。

### RealSyn: An Effective and Scalable Multimodal Interleaved Document Transformation Paradigm 
[[arxiv](https://arxiv.org/abs/2502.12513)] [[cool](https://papers.cool/arxiv/2502.12513)] [[pdf](https://arxiv.org/pdf/2502.12513)]
> **Authors**: Tiancheng Gu,Kaicheng Yang,Chaoyi Zhang,Yin Xie,Xiang An,Ziyong Feng,Dongnan Liu,Weidong Cai,Jiankang Deng
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 16 pages, 12 figures, Webpage: https://garygutc.github.io/RealSyn
- **标题**: Rearsyn：有效且可扩展的多模式交织文档转换范式
- **领域**: 计算机视觉和模式识别
- **摘要**: 在广泛的图像文本对进行预训练之后，对比的语言图像预训练（剪辑）在各种基准上都表现出有希望的表现。但是，大量的非生产数据（例如多模式交错文档）仍未用于视觉表示学习。为了充分利用这些未配对的文档，我们最初建立了一个现实世界中的数据提取管道来提取高质量的图像和文本。然后，我们设计了一种分层检索方法，以有效地将每个图像与多个语义相关的现实文本相关联。为了进一步增强细粒度的视觉信息，我们提出了一个图像语义增强生成模块，用于合成文本生产。此外，我们采用语义平衡采样策略来改善数据集多样性，从而更好地学习长尾概念。基于这些创新，我们构建了一个结合现实和合成文本的数据集，分为三个尺度：15m，30m和100m。广泛的实验表明，Rearsyn有效地提高了视觉表示的学习并表现出强大的可扩展性。在Realsyn进行的预训练的模型在多个下游任务上实现了最新的性能。为了促进未来的研究，Rearsyn数据集和预训练的模型权重在https://github.com/deepglint/realsyn上发布。

### Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning 
[[arxiv](https://arxiv.org/abs/2502.12488)] [[cool](https://papers.cool/arxiv/2502.12488)] [[pdf](https://arxiv.org/pdf/2502.12488)]
> **Authors**: Xiang He,Dongcheng Zhao,Yiting Dong,Guobin Shen,Xin Yang,Yi Zeng
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: The manuscript is under review and the code is available https://github.com/Brain-Cog-Lab/S-CMRL
- **标题**: 通过语义一致性和跨模式残差学习来增强视听尖峰神经网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 人类通过整合来自视觉和听力等多种方式的感官信息来解释和感知世界。作为脑启发的计算模型，尖峰神经网络（SNN）在模拟大脑的信息处理机制方面具有独特的优势。但是，现有的SNN模型主要集中于单峰处理和缺乏有效的跨模式信息融合，从而限制了它们在现实世界多模式方案中的有效性。为了应对这一挑战，我们提出了一个语义对准跨模式残差学习（S-CMRL）框架，这是一种基于变压器的多模式SNN体系结构，旨在有效的视听整合。 S-CMRL利用时空尖峰注意机制来提取跨模态的互补特征，并结合了跨模式的残留学习策略来增强特征整合。此外，引入语义对齐优化机制是在共享的语义空间内对齐的跨模式特征，从而提高了它们的一致性和互补性。在三个基准数据集Crema-D，Urbansound8K-AV和MnistDVS-NTIDiDigits上进行的大量实验表明，S-CMRL明显胜过现有的多模式SNN方法，从而实现了最先进的性能。该代码可在https://github.com/brain-cog-lab/s-cmrl上公开获取。

### Benchmarking Zero-Shot Facial Emotion Annotation with Large Language Models: A Multi-Class and Multi-Frame Approach in DailyLife 
[[arxiv](https://arxiv.org/abs/2502.12454)] [[cool](https://papers.cool/arxiv/2502.12454)] [[pdf](https://arxiv.org/pdf/2502.12454)]
> **Authors**: He Zhang,Xinyi Fu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 10 pages
- **标题**: 使用大语言模型进行基准测试零镜头的面部情感注释：Dailylife中的多级和多框架方法
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 这项研究调查了使用大语言模型（LLM）在日常情况下自动注释人类情绪的可行性和性能。我们采用了GPT-4O-MINI模型，对从视频段中提取的关键帧进行快速，零拍标记，对公开可用的Ferv39K数据集进行了实验。在七级情感分类法（“愤怒”，“厌恶”，“恐惧”，“快乐”，“中立”，“悲伤”，“惊喜”）下，LLM的平均精度约为50％。相反，如果仅限于三元情绪分类（负/中性/阳性）时，平均精度增加到约64％。此外，我们探索了一种策略，该策略在1-2秒的视频片段中集成了多个帧，以提高标签性能并降低成本。结果表明，这种方法可以稍微提高注释精度。总体而言，我们的初步发现突出了零摄影LLM在人面部情感注释任务中的潜在应用，提供了降低标签成本并扩大LLM在复杂多模式环境中的适用性的新途径。

### Robust Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning 
[[arxiv](https://arxiv.org/abs/2502.12425)] [[cool](https://papers.cool/arxiv/2502.12425)] [[pdf](https://arxiv.org/pdf/2502.12425)]
> **Authors**: Mengshi Qi,Changsheng Lv,Huadong Ma
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 强大的分解反事实学习，用于物理视觉常识性推理
- **领域**: 计算机视觉和模式识别
- **摘要**: 在本文中，我们提出了一种新的鲁棒性分解反事实学习（RDCL）方法，用于物理视听常识性推理。该任务旨在根据视频和音频输入来推断对象的物理通心，主要挑战是如何模仿人类的推理能力，即使在缺失模态的情况下也是如此。当前的大多数方法都无法在多模式数据中充分利用不同的特征，并且在模型中缺乏因果推理能力阻碍了隐式物理知识推断的进展。为了解决这些问题，我们提出的RDCL方法将视频分解为潜在空间中的静态（时间不变）和动态（时变）因素，该因素通过分离的顺序编码器，通过差异自动编码器（VAE）采用差异损失函数，从而最大程度地发挥了差异性信息。此外，我们引入了反事实学习模块，以通过对反事实干预下的不同对象之间的物理知识关系进行建模，以增强模型的推理能力。为了减轻不完整的模式数据问题，我们引入了一种强大的多模式学习方法，通过分解共享功能和特定于模型的功能来恢复缺失的数据。我们提出的方法是一个插件模块，可以将其合并到包括VLM在内的任何基线中。在实验中，我们表明我们提出的方法提高了基线方法的推理准确性和鲁棒性，并实现了最新的性能。

### Duo Streamers: A Streaming Gesture Recognition Framework 
[[arxiv](https://arxiv.org/abs/2502.12297)] [[cool](https://papers.cool/arxiv/2502.12297)] [[pdf](https://arxiv.org/pdf/2502.12297)]
> **Authors**: Boxuan Zhu,Sicheng Yang,Zhuo Wang,Haining Liang,Junxiao Shen
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 10 pages, 4 figures
- **标题**: 二人流媒体：流媒体识别框架
- **领域**: 计算机视觉和模式识别
- **摘要**: 在资源受限的方案中的手势识别在实现高准确性和低潜伏期方面面临重大挑战。本文提出的流媒体识别框架二重奏流媒体通过三阶段的稀疏识别机制，具有外部隐藏状态的RNN-Lite模型以及专门的培训和后处理管道来解决这些挑战，从而在实时性能和轻量级设计方面取得了创新的进步。实验结果表明，二人流媒体与精度指标的主流方法匹配，同时将实时因子降低了约92.3％，即提供了近13倍的速度。此外，与主流模型相比，该框架将参数计数缩小为1/38（空闲状态）和1/9（繁忙状态）。总而言之，二人流媒体不仅提供了一种有效且实用的解决方案，用于在资源约束设备中流式识别识别手势，而且还为在多模式和多样化场景中扩展应用程序奠定了坚实的基础。

### HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation 
[[arxiv](https://arxiv.org/abs/2502.12148)] [[cool](https://papers.cool/arxiv/2502.12148)] [[pdf](https://arxiv.org/pdf/2502.12148)]
> **Authors**: Ling Yang,Xinchen Zhang,Ye Tian,Chenming Shang,Minghao Xu,Wentao Zhang,Bin Cui
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Code: https://github.com/Gen-Verse/HermesFlow
- **标题**: HERMESFLOW：无缝缩小多模式理解和发电的差距
- **领域**: 计算机视觉和模式识别
- **摘要**: 自回旋范式的显着成功在多模式大语言模型（MLLMS）方面取得了重大进步，诸如Show-O，输血和EMU3之类的强大模型在统一的图像理解和产生方面取得了显着进步。我们第一次发现了一个共同的现象：MLLM的理解能力通常比其生成能力强，两者之间存在显着差距。在这个见解的基础上，我们提出了Hermesflow，这是一个简单而通用的框架，旨在无缝地弥合MLLM中理解与产生之间的差距。具体而言，我们将同源数据作为输入来策划理解和发电的同源偏好数据。通过配对和自我播放迭代优化，Hermesflow使用同源偏好数据有效地对齐了多模式的理解和生成。广泛的实验证明了我们的方法比先前方法的显着优越性，尤其是在缩小多模式理解与产生之间的差距方面。这些发现突出了Hermesflow作为下一代多模式模型的一般对齐框架的潜力。代码：https：//github.com/gen-verse/hermesflow

### PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection 
[[arxiv](https://arxiv.org/abs/2502.12119)] [[cool](https://papers.cool/arxiv/2502.12119)] [[pdf](https://arxiv.org/pdf/2502.12119)]
> **Authors**: Jinhe Bi,Yifan Wang,Danqi Yan,Xun Xiao,Artur Hecker,Volker Tresp,Yunpu Ma
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 棱镜：用于无训练多模式数据选择的自我灌输内在选择方法
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 视觉指导调整完善了预先训练的多模式大型语言模型（MLLM），以增强其实际任务性能。但是，视觉指导数据集的快速扩展引入了重要的数据冗余，导致了过度的计算成本。现有的数据选择方法主要依赖于代理模型或基于损失的指标，这两个指标都施加了大量的计算开销，这是由于模型推理和反向传播的必要性。为了应对这一挑战，我们提出了Prism，这是一种无培训的新方法，用于有效的多模式数据选择。与现有方法不同，Prism消除了对代理模型的依赖，热身预处理和基于梯度的优化。取而代之的是，它利用Pearson相关分析来量化MLLM的内在视觉编码属性，计算特定于任务的相关得分以识别高价值实例。这不仅可以提供数据有效的选择，而且可以保持原始性能。跨多个MLLM的经验评估表明，PRISM将视觉指导调整和数据选择所需的整体时间缩短为30％的常规方法，同时超过了八个多模式和三种语言理解基准的完全微调模型，在最终绩效中实现了101.7％的相对改善。

### Intuitive physics understanding emerges from self-supervised pretraining on natural videos 
[[arxiv](https://arxiv.org/abs/2502.11831)] [[cool](https://papers.cool/arxiv/2502.11831)] [[pdf](https://arxiv.org/pdf/2502.11831)]
> **Authors**: Quentin Garrido,Nicolas Ballas,Mahmoud Assran,Adrien Bardes,Laurent Najman,Michael Rabbat,Emmanuel Dupoux,Yann LeCun
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 24 pages,14 figures, 5 tables
- **标题**: 直观的物理理解来自自然视频的自我监督预读
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 我们研究了直觉物理学理解的出现，以通用的深度神经网络模型，该模型训练有素，可以预测自然视频中的掩盖区域。利用违反预测框架，我们发现经过训练的视频预测模型，可以预测在学会的表示空间中的结果，这表明了对各种直观物理属性的理解，例如对象的永久性和形状一致性。相比之下，像素空间和多模式大语言模型中的视频预测，这些模型通过文本进行推理，可以实现更接近机会的性能。我们对这些体系结构的比较表明，共同学习抽象表示空间，同时预测感觉输入的缺失部分，类似于预测性编码，足以获得对直觉物理学的理解，甚至在一周的独特视频中训练的模型上都可以实现上面的偶然性表现。这挑战了以下观点：核心知识（一组固有的系统来帮助了解世界）需要熟练地建立对直觉物理学的理解。

### video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model 
[[arxiv](https://arxiv.org/abs/2502.11775)] [[cool](https://papers.cool/arxiv/2502.11775)] [[pdf](https://arxiv.org/pdf/2502.11775)]
> **Authors**: Guangzhi Sun,Yudong Yang,Jimin Zhuang,Changli Tang,Yixuan Li,Wei Li,Zejun MA,Chao Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: Video-Salmonn-O1：推理增强音频 - 视听大语模型
- **领域**: 计算机视觉和模式识别
- **摘要**: While recent advancements in reasoning optimization have significantly enhanced the capabilities of large language models (LLMs), existing efforts to improve reasoning have been limited to solving mathematical problems and focusing on visual graphical inputs, neglecting broader applications in general video understanding.This paper proposes video-SALMONN-o1, the first open-source reasoning-enhanced audio-visual LLM designed for general video understanding tasks.为了增强其推理能力，我们开发了一个通过逐步解决方案具有挑战性的视听问题的推理密集型数据集。我们还提出了直接偏好优化（PDPO）的过程，该过程利用对比度的步骤选择，以实现针对多模式输入量身定制的有效级别奖励建模。此外，我们介绍了Rivabench，这是第一个由推理密集的视频理解基准，其中包括4,000多个高质量，专家策划的问题 - 答案对，例如站立喜剧，学术演示和合成视频检测。 Video-Salmonn-O1在不同的视频推理基准测试基准方面，比Llava-onevision基线的精度提高了3-8％。此外，与Rivabench上的监督微调模型相比，PDPO的提高了6-8％。增强的推理启用视频 -  salmonn-O1零射击合成视频检测功能。

### Language Models Can See Better: Visual Contrastive Decoding For LLM Multimodal Reasoning 
[[arxiv](https://arxiv.org/abs/2502.11751)] [[cool](https://papers.cool/arxiv/2502.11751)] [[pdf](https://arxiv.org/pdf/2502.11751)]
> **Authors**: Yuqi Pang,Bowen Yang,Haoqin Tu,Yun Cao,Zeyu Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Accepted to ICASSP 2025
- **标题**: 语言模型可以更好地看到：LLM多模式推理的视觉对比度解码
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 尽管大型语言模型（LLM）在推理和生成语言任务方面表现出色，但并不是专门为多模式挑战而设计的。但是，培训多模式大型语言模型（MLLM）是资源密集的，受各种培训限制的限制。在本文中，我们提出了基于模块化的视觉对比解码（MVCD）框架来移动这一障碍。我们的框架利用了LLMS的内在学习（ICL）功能和所提出的视觉对比度解码（CED），该解码（CED）是专门针对此框架量身定制的，而无需任何其他培训。通过将视觉信号转换为文本并关注解码过程中的对比输出分布，我们可以强调通过上下文示例引入的新信息，探索其连接，并避免过度依赖对先前编码的知识。 MVCD增强了LLMS的视觉感知，以使其在输入视觉效果上看到并推理。为了证明MVCD的有效性，我们在五个问答数据集中对四个LLM进行实验。我们的结果不仅显示出模型准确性的一致性提高，还可以很好地解释我们解码策略中的有效组件。我们的代码将在https://github.com/pbhgit/mvcd上找到。

### Incomplete Modality Disentangled Representation for Ophthalmic Disease Grading and Diagnosis 
[[arxiv](https://arxiv.org/abs/2502.11724)] [[cool](https://papers.cool/arxiv/2502.11724)] [[pdf](https://arxiv.org/pdf/2502.11724)]
> **Authors**: Chengzhi Liu,Zile Huang,Zhe Chen,Feilong Tang,Yu Tian,Zhongxing Xu,Zihong Luo,Yalin Zheng,Yanda Meng
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 7 Pages, 6 figures
- **标题**: 不完整的方式分解眼科疾病分级和诊断的表示形式
- **领域**: 计算机视觉和模式识别
- **摘要**: 眼科医生通常需要多模式数据源，以提高临床决策的诊断准确性。但是，由于医疗设备短缺，低质量的数据和数据隐私问题，在现实世界中，缺少数据模式很常见。现有的深度学习方法倾向于通过学习不同方式组合的隐式潜在子空间表示来解决它。我们确定了这些方法的两个重要局限性：（1）隐式表示的约束，这些限制阻碍了模型捕获特定于模态信息的能力以及（2）模态异质性，从而在特征表示中导致分布差距和冗余。为了解决这些问题，我们提出了一种不完整的模态释放表示（IMDR）策略，该策略通过指导相互信息，将特征分解为明确的独立模态 - 模态和模态特异性特征，从而使信息知识提炼并促进其重建有价值的丢失语义，并产生强大的多模型表示。此外，我们引入了一个联合代理学习模块，该模块通过从每个类中利用提取的代理来帮助消除模式内冗余。四个眼科多模式数据集的实验表明，所提出的IMDR的表现明显优于最新方法。

### MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression 
[[arxiv](https://arxiv.org/abs/2502.11651)] [[cool](https://papers.cool/arxiv/2502.11651)] [[pdf](https://arxiv.org/pdf/2502.11651)]
> **Authors**: Linjie Mu,Zhongzhen Huang,Shengqian Qin,Yakun Zhu,Shaoting Zhang,Xiaofan Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: MMXU：一种多模式和多X射线了解疾病进展的数据集
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 大型视觉模型（LVLM）在医疗应用中表现出了巨大的希望，尤其是在视觉问题答案（MEDVQA）和医学图像诊断中。但是，现有的数据集和模型通常无法考虑医学诊断的关键方面，例如历史记录的整合以及随着时间的推移分析疾病进展。在本文中，我们引入了MMXU（多模式和多射线理解），这是一种用于MEDVQA的新型数据集，重点是识别两个患者访问之间特定区域的变化。与以前主要解决单片问题的数据集不同，MMXU启用了多图像问题，并纳入了当前和历史患者数据。我们证明了当前LVLM在MMXU- \ textit {test}上识别疾病进展的局限性，即使是在传统基准上表现良好的疾病。为了解决这个问题，我们提出了一种杂种杰出的一代（MAG）方法，并纳入了全球和区域历史记录。我们的实验表明，整合历史记录至少可以显着提高诊断准确性20 \％，从而弥合了当前LVLM和人类专家绩效之间的差距。此外，我们在mmxu- \ textit {dev}上使用MAG微调了模型，该模型证明了显着的改进。我们希望这项工作可以通过强调历史环境在解释医学图像中的重要性来阐明在医学诊断中推进使用LVLM的途径。我们的数据集在\ href {https://github.com/linjiemu/mmxu} {https://github.com/linjiemu/mmxu}中发布。

### Semantically Robust Unsupervised Image Translation for Paired Remote Sensing Images 
[[arxiv](https://arxiv.org/abs/2502.11468)] [[cool](https://papers.cool/arxiv/2502.11468)] [[pdf](https://arxiv.org/pdf/2502.11468)]
> **Authors**: Sheng Fang,Kaiyu Li,Zhe Li,Jianli Zhao,Xingli Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 配对遥感图像的语义上强大的无监督图像翻译
- **领域**: 计算机视觉和模式识别
- **摘要**: 用于变更检测或分类的图像翻译是唯一的。尽管它可以获取配对的图像，但仍然无监督。此外，始终需要在翻译中进行严格的语义保存，而不是多模式输出。为了应对这些问题，本文提出了一种新方法，即SRUIT（语义上强大的无监督图像到图像翻译），该方法可确保语义上强大的翻译并产生确定性的输出。受到以前的作品的启发，该方法探讨了双期遥感图像的基本特征，并设计了相应的网络。首先，我们假设双向遥感图像共享相同的潜在空间，因为它们总是从同一土地位置获得。因此，SRUIT使发电机共享其高级层，该约束将迫使两个域映射落入相同的潜在空间。其次，考虑到双暂时图像的土地覆盖物可以相互发展，Sruit利用了交叉周期符合的对抗网络将其从一个转换为另一个并恢复它们。实验结果表明，共享权重和跨周期一致性的限制使翻译的图像具有良好的感知图像质量和语义保存，从而有很大的差异。

### SNN-Driven Multimodal Human Action Recognition via Event Camera and Skeleton Data Fusion 
[[arxiv](https://arxiv.org/abs/2502.13385)] [[cool](https://papers.cool/arxiv/2502.13385)] [[pdf](https://arxiv.org/pdf/2502.13385)]
> **Authors**: Naichuan Zheng,Hailun Xia
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 通过事件摄像机和骨架数据融合，SNN驱动的多模式人类动作识别
- **领域**: 计算机视觉和模式识别
- **摘要**: 基于RGB和骨骼数据融合的多模式人类作用识别虽然有效，但受到重大局限性的限制，例如高计算复杂性，过度记忆力消耗和实质性能量需求，尤其是在用人工神经网络（ANN）实施时。这些限制限制了其在资源约束方案中的适用性。为了应对这些挑战，我们提出了一个新型的尖峰神经网络（SNN）驱动的框架，用于使用事件摄像机和骨架数据，用于多模式人类动作识别。我们的框架集中在两个关键的创新上：（1）一种新型的多模式SNN体系结构，为每种模态采用不同的骨干网络 - 用于事件摄像机数据的基于AN SNN的MAMBA和尖峰图形卷积网络（SGN），用于与尖峰的语义萃取模块捕获深层语义语义表述，以构成骨架数据融合； （2）基于SNN的开创性离散信息瓶颈机制，用于模态融合，有效地平衡了特定于模态语义的保存和有效的信息压缩。为了验证我们的方法，我们提出了一种新的方法，用于构建一个集成事件摄像机和骨架数据的多模式数据集，从而实现全面的评估。广泛的实验表明，我们的方法在识别准确性和能源效率方面都能达到卓越的性能，从而为实用应用提供了有希望的解决方案。

### Pretrained Image-Text Models are Secretly Video Captioners 
[[arxiv](https://arxiv.org/abs/2502.13363)] [[cool](https://papers.cool/arxiv/2502.13363)] [[pdf](https://arxiv.org/pdf/2502.13363)]
> **Authors**: Chunhui Zhang,Yiren Jian,Zhongyu Ouyang,Soroush Vosoughi
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Accepted to the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025). The first two authors contributed equally and were listed in random order
- **标题**: 预验证的图像文本模型是秘密的视频标题
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 开发视频字幕模型在计算上很昂贵。视频的动态性质也使可以有效地标题这些序列的多模型模型的设计变得复杂。但是，我们发现，通过使用最小的计算资源并且没有复杂的修改来解决视频动态，可以重新使用基于图像的模型以胜过几个专业的视频字幕系统。我们的改编模型在主要基准测试中表明了顶级性能，在MSRVTT和MSVD上排名第二，而Vatex上的第三级表现。我们通过训练典型的图像字幕模型Blip2，将其转换为竞争性的视频字幕，仅使用6,000个视频文本对，并且简单地将框架串联（比其他方法少得多），该帧使用2.5至1.44亿对。从资源优化的角度来看，此视频字幕研究的重点是三个基本因素：优化模型量表，最大化数据效率并纳入强化学习。这项广泛的研究表明，基于图像的适应性策略可以与最先进的视频字幕系统相匹配，为低资源场景提供实用的解决方案。

### Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization 
[[arxiv](https://arxiv.org/abs/2502.13146)] [[cool](https://papers.cool/arxiv/2502.13146)] [[pdf](https://arxiv.org/pdf/2502.13146)]
> **Authors**: Shuo Xing,Yuping Wang,Peiran Li,Ruizheng Bai,Yueqi Wang,Chengxuan Qian,Huaxiu Yao,Zhengzhong Tu
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: 15 pages
- **标题**: 重新调整：通过检索授权的直接偏好优化对齐视觉语言模型
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 大型视觉语言模型（VLM）的出现通过集成视觉方式扩大了单模式大语言模型（LLM）的范围和功能，从而在各种现实世界中解锁了变换的跨模式应用程序。尽管表现令人印象深刻，但VLM却容易出现重大幻觉，尤其是以跨模式不一致的形式。基于从人类反馈（RLHF）对齐LLM中的增强学习成功的基础，最近的进步重点是在精心策划的数据集中应用直接优先优化（DPO）来减轻这些问题。然而，这种方法通常以野蛮的方式引入偏好信号，从而忽略了视觉信息在对齐过程中的关键作用。在本文中，我们介绍了重新对齐，这是一个新颖的对齐框架，利用图像检索来构建双重偏好数据集，从而有效地包含了文本和视觉偏好信号。我们进一步介绍了RDPO，这是标准直接偏好优化的扩展，该扩展在微调过程中包含了附加的视觉偏好目标。我们的实验结果表明，重新调整不仅比以前的方法更有效地减轻幻觉，而且在一般视觉问题避开（VQA）任务中会产生显着的性能增长。此外，我们表明，重新平衡在各种VLM尺寸和体系结构上保持稳健性和可扩展性。这项工作代表了对齐多模式LLM的重要一步，为更可靠和有效的跨模式应用铺平了道路。我们在https://github.com/taco-group/re-align中发布所有代码。

### Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation 
[[arxiv](https://arxiv.org/abs/2502.13145)] [[cool](https://papers.cool/arxiv/2502.13145)] [[pdf](https://arxiv.org/pdf/2502.13145)]
> **Authors**: Bencheng Liao,Hongyuan Tao,Qian Zhang,Tianheng Cheng,Yingyue Li,Haoran Yin,Wenyu Liu,Xinggang Wang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Code and model are available at https://github.com/hustvl/mmMamba
- **标题**: 多模式MAMBA：仅解码器多模式空间模型通过二次蒸馏
- **领域**: 计算机视觉和模式识别
- **摘要**: 最近的多模式大型语言模型（MLLM）取得了出色的性能，但由于其二次计算复杂性，增长的键值缓存要求以及依赖单独的视觉编码器而面临部署挑战。我们提出了Mmmamba，这是一个框架，用于通过使用中等学术计算资源从现有的MLLM进行逐步蒸馏来开发线性复杂性本地多模式空间模型。我们的方法使只有训练有素的单位MLLM直接转换为线性复杂体系结构，而无需进行预先训练的RNN LLM或视觉编码器。我们提出了一种播种策略，以从训练有素的变压器和三阶段的蒸馏配方中雕刻曼巴，该配方可以有效地将知识从变压器转移到Mamba，同时保留多模式功能。我们的方法还支持灵活的混合体系结构，这些架构将变压器和MAMBA层相结合，以实现可定制的效率 - 性能权衡。 MMMAMBA线性从基于变压器的仅解码器的Hovle蒸馏出来，可以针对现有的线性和二次复杂性VLMS实现竞争性能，而Mmmamba-Hybrid则进一步提高了性能，从而接近Hovle的功能。与Hovle相比，Mmmamba-linear在103K令牌时表现出20.6 $ \ times $速度和75.8％的GPU内存减少，而Mmmamba-Hybrid则获得13.5 $ \ times $速度和60.2％的内存储蓄。代码和型号在https://github.com/hustvl/mmmamba上发布

### Magma: A Foundation Model for Multimodal AI Agents 
[[arxiv](https://arxiv.org/abs/2502.13130)] [[cool](https://papers.cool/arxiv/2502.13130)] [[pdf](https://arxiv.org/pdf/2502.13130)]
> **Authors**: Jianwei Yang,Reuben Tan,Qianhui Wu,Ruijie Zheng,Baolin Peng,Yongyuan Liang,Yu Gu,Mu Cai,Seonghyeon Ye,Joel Jang,Yuquan Deng,Lars Liden,Jianfeng Gao
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: 29 pages, 16 figures, technical report from MSR
- **标题**: 岩浆：多模式AI代理的基础模型
- **领域**: 计算机视觉和模式识别,人工智能,人机交互,机器学习,机器人技术
- **摘要**: 我们提出了岩浆，这是一个基础模型，可在数字世界和物理世界中提供多模式AI代理任务。岩浆是视觉语言（VL）模型的显着扩展，因为它不仅保留了后者的VL理解能力（语言智能），而且还具有在视觉空间世界（空间 - 周期性的智能）中计划和行动的能力，并且从UI导航到机器人操作。 To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action计划。广泛的实验表明，SOM和Tom达到了良好的协同作用，并促进了我们的岩浆模型获得时空智能的获取，这对于多种任务至关重要，如图1所示。特别是，岩浆在UI导航和机器人操纵任务上创建了新的最新结果，超过了专门针对这些任务量身定制的先前模型。在图像和视频相关的多模式任务上，岩浆还与受到大量较大数据集训练的流行大型多模型相比。我们在https://microsoft.github.io/magma上将模型和代码公开为可重复性。

### Understanding and Rectifying Safety Perception Distortion in VLMs 
[[arxiv](https://arxiv.org/abs/2502.13095)] [[cool](https://papers.cool/arxiv/2502.13095)] [[pdf](https://arxiv.org/pdf/2502.13095)]
> **Authors**: Xiaohan Zou,Jian Kang,George Kesidis,Lu Lin
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 了解和纠正VLM中的安全感知失真
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 最近的研究表明，视觉模型（VLMS）在整合视觉方式后，更容易受到有害要求和越狱攻击的影响，比仅使用文本LLM的骨架表现出更大的脆弱性。为了揭示这种现象的根本原因，我们进行了深入的分析并确定一个关键问题：多模式输入引入了模态引起的激活转移向“更安全的”方向相比，与他们的文本相比相比，VLMS导致系统地超过了有害输入的安全性。我们将此问题称为安全感知失真。为了减轻这种失真，我们提出了激活移位分离和校准（ShiftDC），这是一种无训练的方法，可分解和校准模态引起的激活转移，以减少模态对安全性的影响。通过隔离和删除与安全相关的组件，ShiftDC可以恢复LLM主链的固有安全对准，同时保留VLM的视觉语言功能。经验结果表明，ShiftDC显着提高了安全基准上的对齐性能，而不会损害模型效用。

### RobuRCDet: Enhancing Robustness of Radar-Camera Fusion in Bird's Eye View for 3D Object Detection 
[[arxiv](https://arxiv.org/abs/2502.13071)] [[cool](https://papers.cool/arxiv/2502.13071)] [[pdf](https://arxiv.org/pdf/2502.13071)]
> **Authors**: Jingtong Yue,Zhiwei Lin,Xin Lin,Xiaoyu Zhou,Xiangtai Li,Lu Qi,Yongtao Wang,Ming-Hsuan Yang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Accepted by ICLR2025
- **标题**: RoburCDET：增强雷达相机融合的鲁棒性在鸟类的眼景中以检测3D对象
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管最近的低成本雷达相机方法在多模式3D对象检测中显示出有希望的结果，但两个传感器都面临着环境和内在干扰所面临的挑战。照明或不利天气条件降低了相机的性能，而雷达则遭受噪音和位置歧义。实现强大的雷达相机3D对象检测需要在各种条件下保持一致的性能，这一主题尚未得到充分探讨。在这项工作中，我们首先在五种噪声上对雷达相机检测中的鲁棒性进行系统分析，并提出RoburcDet，RoburcDet是BEV中的强大对象检测模型。具体而言，我们设计一个3D高斯膨胀（3DGE）模块，以减轻雷达点的不准确性，包括位置，雷达横截面（RCS）和速度。 3DGE使用RCS和速度先验来生成可变形的内核图和内核大小调节和值分布的方差。此外，我们引入了一个天气自适应的融合模块，该模块可以根据相机信号信心自适应地融合雷达和摄像头功能。对流行基准Nuscenes的广泛实验表明，我们的模型可以在常规和嘈杂的条件下实现竞争性。

### Corrupted but Not Broken: Rethinking the Impact of Corrupted Data in Visual Instruction Tuning 
[[arxiv](https://arxiv.org/abs/2502.12635)] [[cool](https://papers.cool/arxiv/2502.12635)] [[pdf](https://arxiv.org/pdf/2502.12635)]
> **Authors**: Yunhao Gou,Hansi Yang,Zhili Liu,Kai Chen,Yihan Zeng,Lanqing Hong,Zhenguo Li,Qun Liu,James T. Kwok,Yu Zhang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 损坏但没有破坏：重新思考视觉指令调整中损坏数据的影响
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉指导调整（VIT）增强了多模式大型语言模型（MLLM），但由于包含幻觉内容，不正确响应和OCR质量差的损坏数据集所阻碍。虽然先前的工作通过高质量的数据收集或基于规则的过滤专注于数据集的细化，但它们昂贵或限于特定类型的损坏。为了深入了解损坏的数据如何影响MLLM，我们会系统地研究此问题，并发现虽然损坏的数据降低了MLLM的性能，但其效果在很大程度上是肤浅的，因为MLLM的性能可以在很大程度上通过少量的参数子集或在少量的清洁数据进行训练或培训后恢复。此外，损坏的MLLM具有提高的能力，可以将干净样本与损坏的样本区分开，从而无需外部帮助即可清洁数据集。基于这些见解，我们提出了结合自然验证和训练后的腐败训练范式，这极大地胜过现有的缓解腐败策略。

### S2C: Learning Noise-Resistant Differences for Unsupervised Change Detection in Multimodal Remote Sensing Images 
[[arxiv](https://arxiv.org/abs/2502.12604)] [[cool](https://papers.cool/arxiv/2502.12604)] [[pdf](https://arxiv.org/pdf/2502.12604)]
> **Authors**: Lei Ding,Xibing Zuo,Danfeng Hong,Haitao Guo,Jun Lu,Zhihui Gong,Lorenzo Bruzzone
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: S2C：多模式遥感图像中无监督变更检测的学习抗噪声差异
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式遥感（RS）图像中无监督的变化检测（UCD）仍然是一个困难的挑战，这是由于数据中固有的时空复杂性以及由不同的成像传感器引起的异质性。受视觉基础模型（VFM）和对比度学习（CL）方法的最新进展的启发，该研究旨在开发CL方法，以将VFM中的隐式知识转化为变化表示形式，从而消除了对显式监督的需求。为此，我们在同质和多模式RS图像中为UCD引入了语义变化（S2C）学习框架。与通常专注于学习多阶段相似性的现有CL方法不同，我们引入了一种新颖的三胞胎学习策略，该策略明确地模拟了时间差异，这对于CD任务至关重要。此外，在训练期间引入了随机的空间和光谱扰动，以增强对时间噪声的鲁棒性。另外，将网格稀疏正规化定义为抑制微不足道的变化，并开发了IOU匹配算法以完善CD结果。四个基准CD数据集的实验表明，所提出的S2C学习框架的准确性显着提高，分别超过了31 \％，9 \％，23 \％和15 \％的当前最新技术。它还表明了鲁棒性和样本效率，适用于训练和适应各种视觉基础模型（VFM）或骨干神经网络。相关代码将在以下网址提供：github.com/dinglei14/s2c。

### CutPaste&Find: Efficient Multimodal Hallucination Detector with Visual-aid Knowledge Base 
[[arxiv](https://arxiv.org/abs/2502.12591)] [[cool](https://papers.cool/arxiv/2502.12591)] [[pdf](https://arxiv.org/pdf/2502.12591)]
> **Authors**: Cong-Duy Nguyen,Xiaobao Wu,Duc Anh Vu,Shuai Zhao,Thong Nguyen,Anh Tuan Luu
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: Cutpaste＆Find：具有视觉援助知识库的有效多模式幻觉探测器
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 大型视觉模型（LVLM）表现出了令人印象深刻的多模式推理能力，但它们仍然容易受到幻觉的影响，尤其是对物体幻觉，在生成的描述中不存在任何物体或不正确的属性。现有的检测方法实现了强劲的性能，但在很大程度上依赖于昂贵的API调用和基于LVLM的迭代验证，这使得它们对于大规模或离线使用不切实际。为了解决这些限制，我们提出了Cutpaste \＆Find，这是一个轻巧且无训练的框架，用于检测LVLM生成的输出中的幻觉。我们的方法利用现成的视觉和语言模块有效地执行多步验证，而无需LVLM推断。我们框架的核心是一个视觉援助知识库，该知识库编码丰富的实体 - 属性关系和相关的图像表示。我们引入了一个缩放因子，以优化相似性得分，即使是地面图像文本对，也可以减轻次优对准值的问题。对包括教皇和R型台的基准数据集进行了全面评估，这表明Cutpaste \＆Find可以实现竞争性的幻觉检测性能，同时比以前的方法更有效和更具成本效益。

### MomentSeeker: A Comprehensive Benchmark and A Strong Baseline For Moment Retrieval Within Long Videos 
[[arxiv](https://arxiv.org/abs/2502.12558)] [[cool](https://papers.cool/arxiv/2502.12558)] [[pdf](https://arxiv.org/pdf/2502.12558)]
> **Authors**: Huaying Yuan,Jian Ni,Yueze Wang,Junjie Zhou,Zhengyang Liang,Zheng Liu,Zhao Cao,Zhicheng Dou,Ji-Rong Wen
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: MomentSeeker：在长视频中的全面基准和强大的基线检索
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 检索增强发电（RAG）在解决与长期视频理解相关的挑战方面具有巨大的希望。这些方法从长视频中获取了有用的时刻，以完成其呈现的任务，从而使多模式大型语言模型（MLLMS）以具有成本效益的方式生成高质量的答案。在这项工作中，我们介绍了MomentSeeker，这是一个全面的基准，旨在评估检索模型在处理一般远程Video Moment检索（LVMR）任务时的性能。 MomentSeeker提供了三个关键优势。首先，它平均包含了长500秒的长视频，这使其成为第一个专门用于Longvideo Moment检索的基准。其次，它涵盖了广泛的任务类别（包括矩搜索，字幕对齐，图像条件的时刻搜索以及视频条件的时刻搜索）以及各种应用程序场景（例如，体育，电影，卡通和自我），使其成为评估检索模型的一般LVMR的全面工具。此外，评估任务是通过人类注释仔细策划的，从而确保了评估的可靠性。我们进一步调整了基于MLLM的LVMR回收师的合成数据，这表明我们的基准表现出强烈的性能。我们基于我们的基准，对各种流行的多模式检索器进行了广泛的实验，其结果突出了LVMR的挑战以及现有方法的局限性。我们创建的资源将与社区共享，以推进该领域的未来研究。

### Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition 
[[arxiv](https://arxiv.org/abs/2502.14195)] [[cool](https://papers.cool/arxiv/2502.14195)] [[pdf](https://arxiv.org/pdf/2502.14195)]
> **Authors**: Tianyi Shang,Zhenyu Li,Pengjie Xu,Jinwei Qiao,Gang Chen,Zihan Ruan,Weijun Hu
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 8 pages, 4 figures, conference
- **标题**: 桥接文本和视觉：跨模式位置识别的多视文本文本注册方法
- **领域**: 计算机视觉和模式识别,机器人技术
- **摘要**: 移动机器人需要先进的自然语言理解能力才能准确识别位置并执行诸如包装交付之类的任务。但是，传统的视觉位置识别（VPR）方法仅依赖于单视觉信息，无法解释人类语言描述。为了克服这一挑战，我们通过提出多视图（周围环境的360°视图）文本视觉注册方法称为text4vpr进行位置识别任务，这是第一个专门利用文本描述以匹配图像数据库的方法。 Text4VPR采用冷冻的T5语言模型来提取全局文本嵌入。此外，它利用具有温度系数的sindhorn算法将局部令牌分配给其各自的簇，从而从图像中汇总了视觉描述符。在训练阶段，Text4VPR强调了单个文本图像对之间的对齐，以进行精确的文本描述。在推理阶段，Text4VPR使用级联的跨意义余弦对齐（CCCA）来解决文本和图像组之间的内部不匹配。随后，Text4VPR根据文本图像组的描述精确地执行匹配。在我们创建的Image VPR数据集的第一个文本中，Text4VPR建立了强大的基线，在测试集上的5米半径内实现了前1个领先的前1个精度，在5米半径内达到了92％的前1个精确度，这表明从文本描述到图像的本地化并不是可行的1，从而表明了一项可能性的影响1，因此在5米的半径内实现了前10％的准确性，从而有可能出现。

### Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.14191)] [[cool](https://papers.cool/arxiv/2502.14191)] [[pdf](https://arxiv.org/pdf/2502.14191)]
> **Authors**: Michihiro Yasunaga,Luke Zettlemoyer,Marjan Ghazvininejad
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: Dataset available at https://github.com/facebookresearch/multimodal_rewardbench
- **标题**: 多模式奖励基地：视觉语言模型的奖励模型的整体评估
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 奖励模型通过评估输出质量以使与人类偏好保持一致，在训练视觉模型（VLM）中起着至关重要的作用。尽管它们的重要性，但研究界缺乏用于评估VLM中多模式奖励模型的全面开放基准。为了解决这一差距，我们介绍了多模式奖励基地，这是一个专家宣布的基准测试，涵盖了六个领域：一般性正确，偏好，知识，推理，安全和视觉提问。我们的数据集包括从各种VLM收集的5,211个注释（提示，选择的响应，拒绝响应）的三胞胎。在评估一系列VLM法官时，我们发现即使是表现最好的模型，Gemini 1.5 Pro和Claude 3.5十四行诗也只能达到72％的总体精度。值得注意的是，大多数模型在推理和安全域中挣扎。这些发现表明，多模式奖励基地为推进跨多个领域的奖励模型开发提供了挑战性的测试床。我们在https://github.com/facebookresearch/multimodal_rewardbench上发布基准。

### Object-centric Binding in Contrastive Language-Image Pretraining 
[[arxiv](https://arxiv.org/abs/2502.14113)] [[cool](https://papers.cool/arxiv/2502.14113)] [[pdf](https://arxiv.org/pdf/2502.14113)]
> **Authors**: Rim Assouel,Pietro Astolfi,Florian Bordes,Michal Drozdzal,Adriana Romero-Soriano
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 以对比语言图像预处理中以对象为中心的绑定
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 视觉语言模型（VLM）的最新进展是由诸如剪辑之类的对比模型驱动的，这些模型学会将视觉信息与相应的文本描述相关联。但是，这些模型在理解涉及多个对象及其空间关系的复杂组成场景方面有局限性。为了应对这些挑战，我们提出了一种新颖的方法，该方法与常用策略不同，该方法依赖于硬性增强的设计。取而代之的是，我们的工作着重于将归纳偏见整合到类似培训的夹子样模型中，以改善其组成理解，而无需使用任何其他硬性阴性。为此，我们引入了一个结合模块，该模块连接一个从文本描述中得出的场景图，并带有插槽结构的图像表示，从而促进了两种模式之间的结构化相似性评估。我们还利用关系作为文本条件的视觉约束，从而更有效地捕获对象之间的复杂相互作用。我们最终的模型不仅增强了基于夹的模型在多对象组成理解中的性能，而且还为复杂场景的更准确和样品效率的图像文本匹配铺平了道路。

### PedDet: Adaptive Spectral Optimization for Multimodal Pedestrian Detection 
[[arxiv](https://arxiv.org/abs/2502.14063)] [[cool](https://papers.cool/arxiv/2502.14063)] [[pdf](https://arxiv.org/pdf/2502.14063)]
> **Authors**: Rui Zhao,Zeyu Zhang,Yi Xu,Yi Yao,Yan Huang,Wenxin Zhang,Zirui Song,Xiuying Chen,Yang Zhao
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: PEDDET：用于多模式行人检测的自适应光谱优化
- **领域**: 计算机视觉和模式识别
- **摘要**: 智能运输系统中的行人检测取得了重大进展，但面临两个关键挑战：（1）可见光和红外光谱之间互补信息的融合不足，尤其是在复杂的情况下，以及（2）对照明变化的敏感性，例如低光或过度暴露的条件，导致绩效降级。为了解决这些问题，我们提出了PEDDET，这是一种自适应光谱优化互补性框架，专门增强和优化，可用于多光谱的行人检测。 PEDDET引入了多尺度光谱特征感知模块（MSFPM），以适应可见和红外功能，增强了功能提取方面的鲁棒性和灵活性。此外，通过解耦行人和背景特征，照明鲁棒性特征解耦模块（IRFDM）在不同的照明下提高了检测稳定性。我们进一步设计了对比度对齐，以增强联运特征歧视。 LLVIP和MSDS数据集的实验表明，PEDDET可以实现最先进的性能，即使在弱光条件下，也以卓越的检测精度提高了地图，这标志着道路安全的重要一步。代码将在https://github.com/aigeeksgroup/peddet上找到。

### Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data 
[[arxiv](https://arxiv.org/abs/2502.14044)] [[cool](https://papers.cool/arxiv/2502.14044)] [[pdf](https://arxiv.org/pdf/2502.14044)]
> **Authors**: Yucheng Shi,Quanzheng Li,Jin Sun,Xiang Li,Ninghao Liu
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: Accepted by ICLR 2025. Code: https://github.com/sycny/SelfSynthX
- **标题**: 通过自同一数据增强多模式基础模型的认知和解释性
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 大型多模型（LMM）或视觉语言模型（VLMS）在各种视觉任务中显示出令人印象深刻的功能。但是，他们经常在细粒度的视觉推理上挣扎，无法识别特定领域的目标，并为其预测提供了合理的解释。为了应对上述挑战，我们提出了一个新颖的视觉拒绝抽样框架，以使用自合成的数据来提高LMM的认知和解释。具体而言，视觉微调需要图像，查询和目标答案。我们的方法首先综合了包括可验证的视觉特征的可解释答案。这些功能基于专家定义的概念，并根据其与图像内容的对齐方式进行了精心选择。每一轮微调后，我们都会应用无奖励模型的过滤机制，为下一轮调整选择最高质量的可解释答案。合成数据生成和微调的这种迭代过程逐渐提高了模型生成准确和合理的解释的能力。实验结果证明了我们方法在提高专业视觉分类任务的准确性和解释性方面的有效性。

### A Chain-of-Thought Subspace Meta-Learning for Few-shot Image Captioning with Large Vision and Language Models 
[[arxiv](https://arxiv.org/abs/2502.13942)] [[cool](https://papers.cool/arxiv/2502.13942)] [[pdf](https://arxiv.org/pdf/2502.13942)]
> **Authors**: Hao Huang,Shuaihang Yuan,Yu Hao,Congcong Wen,Yi Fang
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 11 pages, 3 figures, 5 tables
- **标题**: 经过思想链的子空间元学习，用于与大型视觉和语言模型的几片图像字幕
- **领域**: 计算机视觉和模式识别
- **摘要**: 已在大量数据上鉴定的大规模视觉和语言模型编码视觉和语言的先验，这使得生成更自然和现实的图像和语言变得更加容易。尽管如此，视觉和语言的方式之间仍然存在显着的领域差距，尤其是在训练数据中稀缺的情况下，只有非常有限的数据可用于培训。为了减轻此问题，已经提出了一个多模式的元学习框架，以通过引入一个可调的提示，连接这两个大型模型，以弥合两个冷冻验证的大型视力和语言模型之间的差距。对于几个图像字幕，现有的多模型元学习框架利用一个步骤提示方案来积累输入图像的视觉特征来指导语言模型，该模型努力使用只有几个培训样本来生成准确的图像描述。取而代之的是，我们提出了一个经过思考链（COT）元学习方案作为多步图像字幕程序，以更好地模仿人类如何描述图像。此外，我们进一步建议学习与不同子空间中每个COT步骤相对应的模型的不同元参数，以避免干扰。我们在几个射击设置下评估了三个常用的图像字幕数据集（即MSCOCO，FLICKR8K和FLICKR30K）上的方法。我们的实验结果表明，在不同指标测量的不同数据集的性能方面，我们经过深思熟虑的子空间元学习策略优于基准。

### Multi-view Video-Pose Pretraining for Operating Room Surgical Activity Recognition 
[[arxiv](https://arxiv.org/abs/2502.13883)] [[cool](https://papers.cool/arxiv/2502.13883)] [[pdf](https://arxiv.org/pdf/2502.13883)]
> **Authors**: Idris Hamoud,Vinkle Srivastav,Muhammad Abdullah Jamal,Didier Mutter,Omid Mohareri,Nicolas Padoy
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 手术室活动识别的多视频视频预处理
- **领域**: 计算机视觉和模式识别
- **摘要**: 了解复杂手术室中手术程序的工作流程需要深入了解临床医生与其环境之间的相互作用。手术活动识别（SAR）是一个关键的计算机视觉任务，可检测来自多视摄像机录音的活动或阶段。现有的SAR模型通常无法说明细粒度的临床医生运动和多视图知识，或者它们需要校准的多视图摄像头设置和高级点云处理以获得更好的结果。在这项工作中，我们提出了一种新颖的无校准多视图多模式预处理框架，称为视频姿势手术活动识别的多视图预处理，该框架将2D姿势和视觉嵌入跨相机视图对齐。我们的模型遵循剪辑式双编码器架构：一个编码器过程可视化特征，而另一个编码人类姿势嵌入。为了处理连续的2D人姿势坐标，我们引入了一个令牌化的离散表示形式，以将连续的2D姿势坐标转换为离散的姿势嵌入，从而在双编码器框架内实现了有效的集成。为了弥合这两种方式之间的差距，我们提出了几个在嵌入空间内的交叉​​和内模型几何约束的预处理目标，并结合了蒙版的姿势姿势令牌预测策略以增强表示表示。广泛的实验和消融研究表明了对强基础的改进，而在两个不同的手术室数据集上的数据效率实验进一步突出了我们方法的有效性。我们强调了我们在多视图和单视图中识别手术活动识别方法的好处，从而在复杂的手术环境中展示了其实际适用性。代码将在以下网址提供：https：//github.com/camma-public/previps。

### MEX: Memory-efficient Approach to Referring Multi-Object Tracking 
[[arxiv](https://arxiv.org/abs/2502.13875)] [[cool](https://papers.cool/arxiv/2502.13875)] [[pdf](https://arxiv.org/pdf/2502.13875)]
> **Authors**: Huu-Thien Tran,Phuoc-Sang Pham,Thai-Son Tran,Khoa Luu
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 6 pages, 6 figures, 2024 International Conference on Advanced Technologies for Communications (ATC), Signal Processing Track
- **标题**: MEX：引用多对象跟踪的内存效率方法
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 参考多对象跟踪（RMOT）是一个相对较新的概念，在计算机视觉和自然语言处理的交集中，已迅速获得了吸引力的研究方向。与传统的多对象跟踪不同，RMOT可以识别和跟踪对象，并结合对象类名称的文本描述，从而使该方法更加直观。已经提出了各种技术来解决这个具有挑战性的问题；但是，由于其端到端性质，大多数人都需要对整个网络进行培训。在这些方法中，Ikun已成为一种特别有希望的解决方案。因此，我们进一步探索了它的管道并提高其性能。在本文中，我们介绍了一个实用的模块，称为记忆有效的跨模式-MEX。这种记忆效率的技术可以直接应用于Ikun等现成的跟踪器，从而实现重大的建筑改进。我们的方法在推断具有4 GB内存的单个GPU期间有效。在各种基准测试中，Refer-Kitti数据集提供具有相关语言表达式的各种自动驾驶场景，对于研究此问题特别有用。从经验上讲，我们的方法证明了HOTA跟踪得分的有效性和效率，从而大大提高了内存分配和处理速度。

### Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge 
[[arxiv](https://arxiv.org/abs/2502.13818)] [[cool](https://papers.cool/arxiv/2502.13818)] [[pdf](https://arxiv.org/pdf/2502.13818)]
> **Authors**: Nikolaos Dionelis,Nicolas Longépé,Alessandra Feliciotti,Mattia Marconcini,Devis Peressutti,Nika Oman Kadunc,JaeWan Park,Hagai Raja Sinulingga,Steve Andreas Immanuel,Ba Tran,Caroline Arnold
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: 6 pages, 12 figures
- **标题**: 建筑年龄估计：一种新的多模式基准数据集和社区挑战
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 估计建筑物的建设年对于可持续性至关重要。可持续建筑将能源消耗降至最低，是负责任和可持续的城市规划和发展的关键部分，以有效地打击气候变化。通过使用人工智能（AI）和最近提出的变压器模型，我们能够从多模式数据集中估算建筑物的构建时代。在本文中，我们介绍了一个新的基准多模式数据集，即您的城市数据集（MYCD）的地图，其中包含顶级图像非常高的分辨率（VHR）图像，地球观察（EO）多光谱数据，来自copernicus-sentinel-2卫星星座和许多不同的城市的cosity and cose and cose in ofer of Cose in Cose in ocior in Cose in Cose of Ouloe，以及欧洲的街道图像的多光谱数据，以及时代。我们评估了在培训中被认为的新/以前看不见的城市的EO泛化性能，并且仅在推理期间出现。在这项工作中，我们介绍了基于MYCD组织的基于社区的数据挑战。 ESA AI4EO挑战Mapyourcity于2024年开放了4个月。在这里，我们介绍了前4个性能模型和主要评估结果。在推断期间，使用所有三种输入模式的模型性能，并且仅检查了两种顶级视图模式，即没有街道视图图像。评估结果表明，这些模型是有效的，并且可以在估计建筑物时代，即使在以前看不见的城市中，甚至仅在推理过程中仅使用两种顶级视图模式（即VHR和Sentinel-2），可以实现良好的性能。

### From Correctness to Comprehension: AI Agents for Personalized Error Diagnosis in Education 
[[arxiv](https://arxiv.org/abs/2502.13789)] [[cool](https://papers.cool/arxiv/2502.13789)] [[pdf](https://arxiv.org/pdf/2502.13789)]
> **Authors**: Yi-Fan Zhang,Hang Li,Dingjie Song,Lichao Sun,Tianlong Xu,Qingsong Wen
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 从正确性到理解：用于教育中个性化错误诊断的AI代理
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型语言模型（LLMS），例如GPT-4，已经显示出令人印象深刻的数学推理能力，在GSM8K等基准上实现了几乎完美的性能。但是，由于对错误诊断和反馈产生的正确性，他们在个性化教育中的应用仍然受到限制。当前的模型无法提供对学生错误原因的有意义的见解，从而限制了他们在教育环境中的效用。为了应对这些挑战，我们提出了三个关键贡献。首先，我们介绍\ textbf {Mathccs}（数学分类和建设性建议），这是一种用于系统错误分析和量身定制反馈的多模式基准测试。 MATHCC包括现实世界中的问题，专家注册的错误类别和纵向学生数据。对最新模型的评估，包括\ textIt {qwen2-vl}，\ textit {llava-ov}，\ textit {claude-3.5-sonnet}和\ textit {gpt-4o}人类水平的表现。其次，我们开发了一个顺序错误分析框架，该框架利用历史数据来跟踪趋势并提高诊断精度。最后，我们提出了一个多代理协作框架，该框架结合了一个用于历史分析的时间序列代理和用于实时改进的MLLM代理，从而增强了错误分类和反馈生成。这些贡献共同为推进个性化的教育提供了一个强大的平台，弥合了当前的AI功能与现实世界教学的需求之间的差距。

### CardiacMamba: A Multimodal RGB-RF Fusion Framework with State Space Models for Remote Physiological Measurement 
[[arxiv](https://arxiv.org/abs/2502.13624)] [[cool](https://papers.cool/arxiv/2502.13624)] [[pdf](https://arxiv.org/pdf/2502.13624)]
> **Authors**: Zheng Wu,Yiping Xie,Bo Zhao,Jiguang He,Fei Luo,Ning Deng,Zitong Yu
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: CardiacMamba：一种具有远程生理测量状态空间模型的多模式RGB-RF融合框架
- **领域**: 计算机视觉和模式识别
- **摘要**: 通过远程照相体积学（RPPG）进行心率（HR）估计，提供了一种无创的解决方案，用于健康监测。但是，由于照明变化，运动文物和肤色偏见，传统的单模式方法（RGB或射频（RF））在平衡鲁棒性和准确性方面面临挑战。在本文中，我们提出了CardiacMamba，这是一种多模式RGB-RF融合框架，利用这两种方式的互补优势。它引入了时间差异MAMBA模块（TDMM），以使用帧之间的时序差异捕获RF信号的动态变化，从而增强了本地和全局特征的提取。此外，CardiacMamba还采用双向SSM进行跨模式对齐和渠道快速傅立叶变换（CFFT），以有效捕获和完善RGB和RF信号的频域特征，最终提高心率估计的精度和周期性检测。在该数据集上进行的广泛实验表明了最先进的性能，从而取得了明显的准确性和鲁棒性的提高。 CardiacMamba大大减轻了肤色偏差，降低了人口统计组的性能差异，并在缺失模式的情况下保持了弹性。通过解决公平，适应性和精确性的关键挑战，该框架将RPPG技术朝着医疗保健中可靠的现实世界部署而发展。这些代码可在以下网址提供：https：//github.com/wuzheng42/cardiacmamba。

### UrbanSAM: Learning Invariance-Inspired Adapters for Segment Anything Models in Urban Construction 
[[arxiv](https://arxiv.org/abs/2502.15199)] [[cool](https://papers.cool/arxiv/2502.15199)] [[pdf](https://arxiv.org/pdf/2502.15199)]
> **Authors**: Chenyu Li,Danfeng Hong,Bing Zhang,Yuxuan Li,Gustau Camps-Valls,Xiao Xiang Zhu,Jocelyn Chanussot
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: Urbansam：学习不变性的适配器，用于分段的城市建筑中的任何模型
- **领域**: 计算机视觉和模式识别
- **摘要**: 遥感（RS）图像中的对象提取和细分是城市环境监测中的一项至关重要但具有挑战性的任务。城市形态本质上是复杂的，具有不同形状和不同尺度的不规则对象。这些挑战通过RS数据源（包括传感器，平台和方式）之间的异质性和规模差异扩大，从而使准确的对象细分特别要求。尽管该细分市场模型（SAM）在分割复杂场景方面显示出很大的潜力，但由于手动相互作用的提示，其处理形式变化的对象的性能仍然有限。为此，我们提出了Urbansam，这是一种专门设计的SAM定制版本，旨在分析复杂的城市环境，同时从远程感知的观察结果中解决缩放效果。受多分辨率分析（MRA）理论的启发，Urbansam融合了一个具有USCALING-ADAPTER的新型可学习的求职者，该提示可以遵守不变性标准，使该模型能够捕获对象的多尺度上下文信息并适应了与理论保证的任意规模变化。此外，USCALING-APAPTER和TRUNK编码器的功能通过掩盖的跨注意操作对齐，从而使中继编码器可以继承适配器的多尺度聚合能力。这种协同作用增强了细分性能，从而在学识渊博的适配器支持下产生了更强大和准确的输出。广泛的实验结果表明，拟议的Urbansam在全球规模的数据集上的灵活性和出色的细分性能，包括建筑物，道路和水等规模变化的城市对象。

### Methods and Trends in Detecting Generated Images: A Comprehensive Review 
[[arxiv](https://arxiv.org/abs/2502.15176)] [[cool](https://papers.cool/arxiv/2502.15176)] [[pdf](https://arxiv.org/pdf/2502.15176)]
> **Authors**: Arpan Mahara,Naphtali Rishe
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 30 pages, 4 Figures, 10 Tables
- **标题**: 检测生成图像的方法和趋势：全面评论
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 生成模型的扩散，例如生成对抗网络（GAN），扩散模型和变异自动编码器（VAE），已使高质量的多媒体数据合成。但是，这些进步还引起了人们对对抗攻击，不道德使用和社会伤害的重大关注。认识到这些挑战，研究人员越来越专注于开发方法，以有效地检测合成数据，以减轻潜在的风险。先前的评论主要集中在深泡检测上，并且通常缺乏对合成图像检测的最新进展的覆盖范围，尤其是利用多模式框架的方法来改进法医分析。为了解决这一差距，本调查对检测和分类由高级生成AI模型生成的合成图像进行了全面综述。这篇综述系统地检查了核心检测方法，确定了方法之间的共同点，并将其分类为有意义的分类法。此外，鉴于大规模数据集在该领域的关键作用，我们介绍了公开可用数据集的概述，这些数据集促进了合成数据检测的进一步研究和基准测试。

### M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment 
[[arxiv](https://arxiv.org/abs/2502.15167)] [[cool](https://papers.cool/arxiv/2502.15167)] [[pdf](https://arxiv.org/pdf/2502.15167)]
> **Authors**: Chuan Cui,Kejiang Chen,Zhihua Wei,Wen Shen,Weiming Zhang,Nenghai Yu
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 14 pages, 5 figures. This work has been submitted to the IEEE for possible publication
- **标题**: M3-agiqa：多模式，多轮，多镜头AI生成的图像质量评估
- **领域**: 计算机视觉和模式识别
- **摘要**: AI生成的图像（AGI）模型的快速发展在评估其质量方面引入了重大挑战，这需要考虑多个维度，例如感知质量，及时的对应关系和真实性。为了应对这些挑战，我们提出了M3-Agiqa，这是一个多模式，多场和多方面的AGI质量评估的综合框架。我们的方法通过通过低级别适应（LORA）微调来利用多模式大语模型（MLLM）作为联合文本和图像编码器的功能，并提取高级字幕功能。该框架包括一个结构化的多轮评估机制，其中生成中间图像描述以提供对质量，对应性和真实性方面的更深入见解。为了使预测与人类的感知判断相一致，由XLSTM构建的预测指标和回归头构建，以处理顺序逻辑并预测平均意见分数（MOSS）。在多个基准数据集上进行的广泛实验表明，M3-Agiqa实现了最先进的性能，有效地捕获了AGI质量的细微方面。此外，跨数据集验证证实了其强大的普遍性。该代码可在https://github.com/strawhatboy/m3-agiqa上找到。

### TransMamba: Fast Universal Architecture Adaption from Transformers to Mamba 
[[arxiv](https://arxiv.org/abs/2502.15130)] [[cool](https://papers.cool/arxiv/2502.15130)] [[pdf](https://arxiv.org/pdf/2502.15130)]
> **Authors**: Xiuwei Chen,Sihao Lin,Xiao Dong,Zisheng Chen,Meng Cao,Jianhua Han,Hang Xu,Xiaodan Liang
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: TransMamba：Fast Universal Architecture从变形金刚到Mamba的改编
- **领域**: 计算机视觉和模式识别
- **摘要**: 在注意模块中的灵活可扩展性中，变形金刚在单模式和多模式基础模型中都受到青睐。因此，公开可用的许多预训练的变压器模型，例如Llava，Clip和Deit。最近的研究引入了诸如Mamba之类的亚二次体系结构，这使全球意识具有线性复杂性。然而，从头开始培训专门的次级体系结构，既是资源密集型又耗时。作为动力，我们探索了跨架构培训，以将现有变压器模型中的现成知识转移到称为TransMamba的替代体系结构Mamba。我们的方法采用了两阶段的战略来加快培训新的Mamba模型，从而确保跨单模和跨模式任务的有效性。关于架构差异，我们将中间特征投射到转移知识之前，将中间特征投射到一个对齐的潜在空间。最重要的是，引入了重量亚克隆和自适应双向蒸馏方法（WSAB）以进行知识转移，而不会限制各个层计数。对于跨模式学习，我们提出了一个跨孟买模块，该模块将语言意识整合到Mamba的视觉特征中，从而增强了Mamba体系结构的跨模式互动功能。尽管通常需要从头开始培训的培训数据的少于75％，但TransMamba在各种网络架构和下游任务中的性能大大提高，包括图像分类，视觉问题答案和文本视频检索。该代码将公开可用。

### CrossOver: 3D Scene Cross-Modal Alignment 
[[arxiv](https://arxiv.org/abs/2502.15011)] [[cool](https://papers.cool/arxiv/2502.15011)] [[pdf](https://arxiv.org/pdf/2502.15011)]
> **Authors**: Sayan Deb Sarkar,Ondrej Miksik,Marc Pollefeys,Daniel Barath,Iro Armeni
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: Project Page: sayands.github.io/crossover/
- **标题**: 跨界：3D场景交叉模式对齐
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式3D对象的理解已引起了很大的关注，但是当前的方法通常会在所有模式中都具有完整的数据可用性和严格的对准。我们展示了Crossover，这是一个通过灵活的场景级别对齐方式进行跨模式3D场景理解的新颖框架。与每个对象实例需要对齐模态数据的传统方法不同，跨界可以通过对齐方式来学习一个统一的，情态的嵌入空间，用于场景 -  RGB图像，点云，CAD模型，平面图，地板图表和文本描述 - 具有放松的约束，没有明确的对象对象语义。利用特定维度的编码器，多阶段训练管道以及紧急的跨模式行为，即使缺少模态，跨模式都支持强大的场景检索和对象定位。对扫描仪和3RSCAN数据集的评估显示了其在不同指标中的出色性能，从而突出了3D场景理解中对现实世界应用程序的适应性。

### LAVID: An Agentic LVLM Framework for Diffusion-Generated Video Detection 
[[arxiv](https://arxiv.org/abs/2502.14994)] [[cool](https://papers.cool/arxiv/2502.14994)] [[pdf](https://arxiv.org/pdf/2502.14994)]
> **Authors**: Qingyuan Liu,Yun-Yun Tsai,Ruijian Zha,Victoria Li,Pengyuan Shi,Chengzhi Mao,Junfeng Yang
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: LAVID：扩散生成视频检测的代理LVLM框架
- **领域**: 计算机视觉和模式识别
- **摘要**: 生成模型在创建高质量视频中的令人印象深刻的成就引起了人们对数字完整性和隐私脆弱性的关注。 AI生成的内容检测的最新作品已在图像字段（例如DeepFake）中进行了广泛研究，但视频字段尚未探索。大型视觉语言模型（LVLM）已成为AI生成的内容检测的新兴工具，其强大的推理和多模式能力。它打破了传统的基于深度学习的局限性，即缺乏透明度和无法识别新的人工制品。在此激励的情况下，我们提出了Lavid，这是一种基于LVLMS的新型AI生成的视频检测，并具有明确的知识增强。我们的见解列表如下：（1）领先的LVLM可以调用外部工具来提取有用的信息以促进其自己的视频检测任务； （2）构造提示可以影响LVLM在视频内容中解释信息的推理能力。我们提出的管道会自动选择一组显式知识工具进行检测，然后通过自我练习自适应地调整结构提示。与训练其他检测器的先前SOTA不同，我们的方法是无训练的，仅需要推断LVLM进行检测。为了促进我们的研究，我们还创建了一个新的基准\ vidfor，并使用由多种视频生成工具产生的高质量视频。评估结果表明，LAVID在四个SOTA LVLMS上的数据集上的最高基准比F1分数提高了6.2％，至30.2％。

### Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning 
[[arxiv](https://arxiv.org/abs/2502.14917)] [[cool](https://papers.cool/arxiv/2502.14917)] [[pdf](https://arxiv.org/pdf/2502.14917)]
> **Authors**: Rui Zhao,Qirui Yuan,Jinyu Li,Haofeng Hu,Yun Li,Chengyuan Zheng,Fei Gao
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: SCE2DRIVEX：用于场景到驱动学习的广义MLLM框架
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 端到端的自主驾驶直接将原始传感器输入映射到低级车辆控件，是体现AI的重要组成部分。尽管成功地将多模式大语模型（MLLM）用于高级交通语义的语义理解，但将这些概念语义的理解有效地转化为低级运动控制命令并实现跨景点驾驶中的概括和共识仍然是一项挑战。我们介绍了SCE2DRIVEX，这是一种类似人类的驾驶链链（COT）推理MLLM框架。 SCE2DRIVEX利用本地场景视频和全球BEV地图的多模式联合学习来深入了解长期时空关系和道路拓扑，从而增强了其在3D动态/静态场景中的全面看法和推理能力，并实现跨场景的推动概括。在此基础上，它重建了人类驾驶中固有的隐性认知链，涵盖了场景的理解，元行动推理，行为解释分析，运动计划和控制，从而进一步弥合了自主驾驶和人类思维过程之间的差距。为了提升模型性能，我们开发了第一个广泛的视觉问题回答（VQA）驾驶指令数据集，该数据集量身定制，用于3D空间理解和长轴任务推理。广泛的实验表明，SCE2Drivex从场景理解到端到端驾驶以及Carla Bench2Drive基准的强大概括从场景的理解到最新的表现。

### What Is a Good Caption? A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Coverage of MLLMs 
[[arxiv](https://arxiv.org/abs/2502.14914)] [[cool](https://papers.cool/arxiv/2502.14914)] [[pdf](https://arxiv.org/pdf/2502.14914)]
> **Authors**: Zhihang Liu,Chen-Wei Xie,Bin Wen,Feiwu Yu,Jixuan Chen,Boqiang Zhang,Nianzu Yang,Pandeng Li,Yun Zheng,Hongtao Xie
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-21
> **comment**: Work in progress
- **标题**: 什么是好的标题？用于评估MLLM的正确性和覆盖率的全面视觉标题基准
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 多模式大语言模型（MLLM）的最新进展使传统的视觉字幕基准过时，因为它们主要用过时的指标评估简短的描述。尽管最近的基准测试通过将字幕分解为视觉元素并采用基于模型的评估来解决这些局限性，但它们仍然是不完整的关键方面，同时提供了模糊的，非解释性的分数。为了弥合这一差距，我们提出了CV-Capbench，这是一个全面的视觉标题基准，该基准在6个视图和13个维度上系统地评估标题质量。 CV-CAPBENCH介绍了每个维度的精度，召回率和命中率指标，从而唯一评估了正确性和覆盖范围。在领先的MLLM上进行的实验揭示了明显的功能差距，尤其是在动态和知识密集的维度上。这些发现为未来的研究提供了可行的见解。代码和数据将发布。

### KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.14908)] [[cool](https://papers.cool/arxiv/2502.14908)] [[pdf](https://arxiv.org/pdf/2502.14908)]
> **Authors**: Peter Carragher,Nikitha Rao,Abhinand Jha,R Raghav,Kathleen M. Carley
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: Koala：视觉语言模型中鲁棒性的知识冲突增强
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,机器学习
- **摘要**: 大型语言模型（LLM）在单形态问题答案系统中针对知识冲突的鲁棒性得到了很好的研究。但是，尚未探索信息源中的冲突对视觉语言模型（VLM）在多模式设置中的影响。在这项工作中，我们提出了\ segsub，该框架将有针对性的扰动应用于图像来源来研究和改善VLMS对三种不同类型的知识冲突的鲁棒性，即参数，源和反事实冲突。与先前的发现相反，该发现表明LLMS对文本扰动引起的参数冲突敏感，我们发现VLM在很大程度上对图像扰动是强大的。另一方面，VLM在反事实示例（<30％的精度）上的表现较差，并且无法推理源冲突（<1％的精度）。我们还发现了幻觉和图像上下文之间的联系，当带有高度背景化的反事实示例时，GPT-4O容易幻觉。尽管挑战与来源冲突有关，但填充模型可显着改善反事实样本的推理。我们的发现突出了对VLM培训方法的需求，以增强其推理能力，尤其是在解决多模式来源之间复杂的知识冲突时。

### NOTA: Multimodal Music Notation Understanding for Visual Large Language Model 
[[arxiv](https://arxiv.org/abs/2502.14893)] [[cool](https://papers.cool/arxiv/2502.14893)] [[pdf](https://arxiv.org/pdf/2502.14893)]
> **Authors**: Mingni Tang,Jiajia Li,Lu Yang,Zhiqiang Zhang,Jinghao Tian,Zuchao Li,Lefei Zhang,Ping Wang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: NOTA：视觉大语言模型的多模式音乐符号理解
- **领域**: 计算机视觉和模式识别,人工智能,机器学习,声音,音频和语音处理
- **摘要**: 符号音乐以两种不同的形式表示：二维，视觉直观的分数图像和一维标准化的文本注释序列。尽管大型语言模型在音乐中表现出极大的潜力，但当前的研究主要集中在单峰符号序列文本上。现有的通用视觉语言模型仍然缺乏音乐符号理解的能力。认识到这一差距，我们提出了Nota，这是第一个大型综合多模式符号数据集。它由来自世界3个地区的1,019,237个记录组成，并包含3个任务。基于数据集，我们训练了宣布，这是一种音乐符号视觉大语言模型。具体而言，我们涉及一个预先对准训练阶段，以在音乐得分图像中描述的音符与ABC符号中的文字表示之间进行跨模式对齐。随后的培训阶段着重于基础音乐信息提取，然后进行音乐符号分析培训。实验结果表明，我们的宣传-7B在音乐理解上取得了重大改进，展示了NOTA和训练管道的有效性。我们的数据集在https://huggingface.co/datasets/myth-lab/nota-dataset上进行开源。

### EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild 
[[arxiv](https://arxiv.org/abs/2502.14892)] [[cool](https://papers.cool/arxiv/2502.14892)] [[pdf](https://arxiv.org/pdf/2502.14892)]
> **Authors**: Junhyeok Kim,Min Soo Kim,Jiwan Chung,Jungbin Cho,Jisoo Kim,Sungwoong Kim,Gyeongbo Sim,Youngjae Yu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-21
> **comment**: NAACL 2025 Findings. Project page at https://jun297.github.io/EgoSpeak/
- **标题**: Egoskeak：学习何时为野外以自我为中心的对话代理人说话
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 预测何时在现实世界环境中发言仍然是对会话代理的基本挑战。我们介绍了EgoSeak，这是一个新颖的框架，用于以自我为中心的流媒体视频中实时语音启动预测。通过从说话者的第一人称观点中对对话进行建模，Egoskeak是针对类似人类的互动而定制的，在这种互动中，对话代理必须不断观察其环境并动态决定何时进行交谈。我们的方法通过整合四个关键功能来弥合简化的实验设置与复杂的自然对话之间的差距：（1）第一人称视角，（2）RGB处理，（3）在线处理和（4）未修剪的视频处理。我们还提出了YT转换，这是来自YouTube的野外对话视频的多样化集合，作为大规模预处理的资源。对EasyCom和EGO4D的实验表明，Egoseak实时超过了随机和沉默的基准。我们的结果还强调了多模式输入和上下文长度在有效决定何时讲话时的重要性。

### Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability 
[[arxiv](https://arxiv.org/abs/2502.14889)] [[cool](https://papers.cool/arxiv/2502.14889)] [[pdf](https://arxiv.org/pdf/2502.14889)]
> **Authors**: Zhiyu Zhu,Zhibo Jin,Jiayu Zhang,Nan Yang,Jiahao Huang,Jianlong Zhou,Fang Chen
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-21
> **comment**: Accepted by ICLR 2025
- **标题**: 狭窄的信息瓶颈理论用于多模式图像文本表示可解释性
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 识别多模式图像文本表示的任务吸引了人们越来越多的关注，尤其是在诸如剪辑（对比性语言图像训练的模型）之类的模型中，这些模型在学习图像和文本之间的学习复杂关联时表现出了出色的表现。尽管取得了这些进步，但确保此类模型的可解释性对于它们在医疗保健等实际应用中的安全部署至关重要。尽管已经为单峰任务开发了许多可解释性方法，但由于表示结构的固有差异，这些方法通常无法有效地转移到多模式上下文。在信息理论中建立良好的瓶颈方法已应用于增强剪辑的解释性。但是，通常会受到强烈的假设或固有的随机性的阻碍。为了克服这些挑战，我们提出了狭窄的信息瓶颈理论，这是一个从根本上重新定义传统瓶颈方法的新颖框架。该理论专门设计用于满足当代归因公理，为改善多模型的可解释性提供了一种更健壮和可靠的解决方案。在我们的实验中，与最先进的方法相比，我们的方法平均可以增强图像可解释性9％，文本可解释性平均提高58.83％，并使处理速度的加速速度升高63.95％。我们的代码可在https://github.com/lmbtough/nib上公开访问。

### The Multi-Faceted Monosemanticity in Multimodal Representations 
[[arxiv](https://arxiv.org/abs/2502.14888)] [[cool](https://papers.cool/arxiv/2502.14888)] [[pdf](https://arxiv.org/pdf/2502.14888)]
> **Authors**: Hanqi Yan,Xiangxiang Cui,Lu Yin,Paul Pu Liang,Yulan He,Yifei Wang
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 多模式表示中的多方面的单体气质
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在本文中，我们利用特征单体性的最新进步来从深层多模型中提取可解释的特征，从而提供了对模式差距的数据驱动的理解。具体而言，我们研究了剪辑（对比性语言图像预处理），这是一种在广泛的图像文本对中训练的突出的视觉表示模型。在为单模模型开发的可解释性工具的基础上，我们扩展了这些方法，以评估夹子特征的多模式可解释性。此外，我们将模式优势得分（MDS）介绍为将每个特征的解释性归因于其各自的模态。接下来，我们将剪辑功能转换为更容易解释的空间，使我们能够将它们分为三个不同的类：视觉功能（单模式），语言特征（单模式）和视觉语言特征（交叉模式）。我们的发现表明，这种分类与人类对不同方式的认知理解紧密相符。我们还展示了这种特定于模态特征的重要用例，包括检测性别偏见，对抗性攻击防御和文本对图像模型编辑。这些结果表明，配备了任务不可解释性工具的大规模多模型模型为关键连接和不同模态之间的区别提供了宝贵的见解。

### Vision-Enhanced Time Series Forecasting via Latent Diffusion Models 
[[arxiv](https://arxiv.org/abs/2502.14887)] [[cool](https://papers.cool/arxiv/2502.14887)] [[pdf](https://arxiv.org/pdf/2502.14887)]
> **Authors**: Weilin Ruan,Siru Zhong,Haomin Wen,Yuxuan Liang
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 通过潜在扩散模型预测视觉增强的时间序列
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 扩散模型最近已成为生成高质量图像的强大框架。尽管最近的研究探讨了他们对时间序列预测的应用，但这些方法在跨模式建模和有效转换视觉信息以捕获时间模式方面面临着巨大的挑战。在本文中，我们提出了LDM4TS，这是一个新颖的框架，利用潜在扩散模型的强大图像重建功能来预测视觉增强时间序列。我们不是引入外部视觉数据，而是第一个使用互补转换技术将时间序列转换为多视觉视觉表示的人，从而使模型可以利用预训练的视觉编码器的丰富特征提取能力。随后，这些表示是使用具有跨模式调节机制和融合模块的潜扩散模型重建的。实验结果表明，LDM4TS在时间序列预测任务方面的表现优于各种专业预测模型。

### From 16-Bit to 1-Bit: Visual KV Cache Quantization for Memory-Efficient Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.14882)] [[cool](https://papers.cool/arxiv/2502.14882)] [[pdf](https://arxiv.org/pdf/2502.14882)]
> **Authors**: Zeliang Zhang,Yifan Zhu,Susan Liang,Zhiyuan Wang,Jiani Liu,Haiting Lin,Mingjie Zhao,Chenliang Xu,Kun Wan,Wentian Zhao
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 从16位到1位：用于存储效率多模式模型的视觉KV缓存量化
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式大语言模型（MLLM）在各种应用程序中都取得了巨大的成功，但是部署期间其计算开销仍然是一个关键的挑战。尽管密钥值（KV）缓存通过将存储器用于计算来提高推理效率，但存储的内存足迹不断增长，而存储广泛的KV caches可减少吞吐量，并限制了具有约束GPU内存的设备上的长期执行。现有方法主要集中于降低不重要的令牌以减少KV缓存大小，从而以潜在的信息损失为代价减轻内存约束。相比之下，我们提出了一种简单而有效的视觉量化策略，该策略可保留所有视觉令牌，同时大大减少记忆消耗。为了实现极端量化比，即1位量化，我们提出了以KV缓存的固有模式进行的，我们提出了特定于小组的量化和基于分位数的量化方法。我们的方法是插件，使无缝集成到各种MLLM中，以提高内存效率而无需进行体系结构修改。广泛的实验表明，我们的方法有效地减少了内存开销，同时保持计算效率并保留多模式性能。

### Time Travel: A Comprehensive Benchmark to Evaluate LMMs on Historical and Cultural Artifacts 
[[arxiv](https://arxiv.org/abs/2502.14865)] [[cool](https://papers.cool/arxiv/2502.14865)] [[pdf](https://arxiv.org/pdf/2502.14865)]
> **Authors**: Sara Ghaboura,Ketan More,Ritesh Thawkar,Wafa Alghallabi,Omkar Thawakar,Fahad Shahbaz Khan,Hisham Cholakkal,Salman Khan,Rao Muhammad Anwer
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 4 pages, 6 figures
- **标题**: 时间旅行：评估LMM的全面基准
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 了解历史和文化伪像需要人类的专业知识和先进的计算技术，但是该过程仍然很复杂且耗时。尽管大型多模式模型提供了有希望的支持，但他们的评估和改进需要标准化的基准测试。为了解决这个问题，我们介绍了Timetravel，这是10,250个专家验证的样品的基准，这些样本跨越了10个主要历史区域。 Timetravel设计用于手稿，艺术品，铭文和考古发现的分析，Timetravel提供了一个结构化的数据集和强大的评估框架，以评估AI模型在分类，解释和历史理解中的能力。通过将AI与历史研究相结合，Timetravel为历史学家，考古学家，研究人员和文化游客提供了AI驱动的工具，以提取宝贵的见解，同时确保技术对历史发现和文化遗产保护有意义贡献。我们在Timetravel上评估了当代的AI模型，强调了它们的优势并确定了改进领域。我们的目标是建立AI作为保护文化遗产的可靠合作伙伴，以确保技术进步对历史发现有意义贡献。我们的代码可在：\ url {https://github.com/mbzuai-oryx/timetravel}中获得。

### Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation 
[[arxiv](https://arxiv.org/abs/2502.14846)] [[cool](https://papers.cool/arxiv/2502.14846)] [[pdf](https://arxiv.org/pdf/2502.14846)]
> **Authors**: Yue Yang,Ajay Patel,Matt Deitke,Tanmay Gupta,Luca Weihs,Andrew Head,Mark Yatskar,Chris Callison-Burch,Ranjay Krishna,Aniruddha Kembhavi,Christopher Clark
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 20 pages, 19 figures, 9 tables, website: https://yueyang1996.github.io/cosyn/
- **标题**: 通过代码引导的合成多模式数据生成缩放文本丰富的图像理解
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 关于图像具有丰富文本的推理，例如图表和文档，是视觉模型（VLM）的关键应用。但是，由于多种文本丰富的视觉语言数据的稀缺性，VLM经常在这些领域中挣扎。为了应对这一挑战，我们提出了Cosyn，该框架利用了仅文本大型语言模型（LLMS）的编码功能来自动创建合成文本丰富的多模式数据。给定的输入文本描述了目标域（例如“营养事实标签”），COSYN提示LLM生成代码（Python，HTML，乳胶等），以渲染合成图像。以基础代码为合成图像的文本表示形式，Cosyn可以生成高质量的指令数据，再次依靠仅使用文本LLM。使用COSYN，我们构建了一个包含400K图像和270万视觉语言指令数据的数据集。对七个基准测试的综合实验表明，经过合成数据训练的模型在竞争性开源模型中实现了最先进的性能，包括Llama 3.2，以及超越专有模型，例如GPT-4V和Gemini 1.5 1.5 Flash。此外，Cosyn可以产生合成指向数据，从而使VLMS能够在输入图像中接地信息，从而展示了其开发能够在现实世界环境中起作用的多模式剂的潜力。

### Exploring Advanced Techniques for Visual Question Answering: A Comprehensive Comparison 
[[arxiv](https://arxiv.org/abs/2502.14827)] [[cool](https://papers.cool/arxiv/2502.14827)] [[pdf](https://arxiv.org/pdf/2502.14827)]
> **Authors**: Aiswarya Baby,Tintu Thankom Koshy
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 8 pages, No figures
- **标题**: 探索视觉问题的先进技术回答：全面比较
- **领域**: 计算机视觉和模式识别,人工智能,新兴技术,机器学习
- **摘要**: 视觉问题回答（VQA）已成为计算机视觉和自然语言处理交集的关键任务，要求模型理解和理论有关自然语言问题的视觉内容。分析VQA数据集对于开发可以处理多模式推理复杂性的强大模型至关重要。已经开发了几种方法来检查这些数据集，每个数据集都提供了有关问题多样性，答案分布和视觉文本相关性的不同观点。尽管取得了重大进展，但现有的VQA模型仍面临与数据集偏见，有限的模型复杂性，常识性推理差距，严格的评估方法以及对现实世界情景的概括有关的挑战。本文提供了对原始VQA数据集，基线模型和方法的详细研究，以及对五个高级VQA模型的比较研究，即ABC-CNN，Kicnle，Kicnle，Masked Vision和语言建模，BLIP-2和OFA，每种都采用独特的方法来应对这些持续的挑战。

### AVD2: Accident Video Diffusion for Accident Video Description 
[[arxiv](https://arxiv.org/abs/2502.14801)] [[cool](https://papers.cool/arxiv/2502.14801)] [[pdf](https://arxiv.org/pdf/2502.14801)]
> **Authors**: Cheng Li,Keyuan Zhou,Tong Liu,Yu Wang,Mingqiao Zhuang,Huan-ang Gao,Bu Jin,Hao Zhao
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: ICRA 2025, Project Page: https://an-answer-tree.github.io/
- **标题**: AVD2：事故视频扩散的事故视频描述
- **领域**: 计算机视觉和模式识别
- **摘要**: 交通事故对自动驾驶提出了复杂的挑战，通常具有不可预测的场景，从而阻碍了准确的系统解释和响应。尽管如此，由于缺乏事故情景特定的训练数据，阐明事故的原因并提出了预防措施时，普遍的方法学院缺乏。在这项工作中，我们介绍了AVD2（事故视频描述的事故视频扩散），这是一个新颖的框架，通过生成与详细的自然语言描述和推理相符的事故视频来增强事故现场的理解，从而导致了EMM-AU（增强的多模式事故事故视频理解）。经验结果表明，EMM-AU数据集的整合在自动指标和人类评估中建立了最先进的性能，从而显着推进了事故分析和预防的领域。项目资源可从https://an-answer-tree.github.io获得。

### PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.14504)] [[cool](https://papers.cool/arxiv/2502.14504)] [[pdf](https://arxiv.org/pdf/2502.14504)]
> **Authors**: Yu Meng,Kaiyuan Li,Chenran Huang,Chen Gao,Xinlei Chen,Yong Li,Xiaoping Zhang
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 12 pages, 8 figures
- **标题**: PLPHP：每层每个头视觉令牌图案用于有效的大型视力模型
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 大型视觉模型（LVLM）在一系列多模式任务中表现出显着的功能。但是，它们的推论效率受到解码过程中处理的大量视觉令牌的限制。为了应对这一挑战，我们提出了每层视觉图令牌（PLPHP），这是一种两级细粒度的修剪方法，包括层级保留率分配和头部级视觉图令牌修剪。由跨解码器层的视觉令牌重新注意事件的动机，我们按一层动态调整令牌保留率。对视觉信息的更加关注的层次可以保留更多的视力令牌，而视力较低的层则积极进行。此外，PLPHP在注意力头级上进行修剪，使同一层中的不同头能够独立保留关键上下文。多个基准测试的实验表明，PLPHP的解码速度速度更快18％，并将钥匙值高速缓存（KV CACHE）的大小降低了50％以上，所有这些速度的平均性能下降了0.46％，同时也可以显着提高多图像任务的性能。这些结果突出了细粒度的修剪的有效性，并有助于提高LVLM的效率和可扩展性。我们的源代码将公开可用。

### Integrating Extra Modality Helps Segmentor Find Camouflaged Objects Well 
[[arxiv](https://arxiv.org/abs/2502.14471)] [[cool](https://papers.cool/arxiv/2502.14471)] [[pdf](https://arxiv.org/pdf/2502.14471)]
> **Authors**: Chengyu Fang,Chunming He,Longxiang Tang,Yuelin Zhang,Chenyang Zhu,Yuqi Shen,Chubin Chen,Guoxia Xu,Xiu Li
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: 12 pages, 5 figures, 6 tables
- **标题**: 整合额外的方式有助于分割伪装的对象
- **领域**: 计算机视觉和模式识别
- **摘要**: 伪装的物体分割（COS）仍然是一个具有挑战性的问题，因为伪装的对象和背景之间存在细微的视觉差异。由于可见光频谱可获得的视觉提示极为有限，因此以前的RGB单模式方法通常难以实现令人满意的结果，从而促使探索多模式数据以提高检测准确性。在这项工作中，我们提出了Unicos，这是一个新颖的框架，可有效利用各种数据方式来改善细分性能。 UNICOS包括两个关键组成部分：多式联运分段，蛋白介和一个跨模式知识学习模块，Unirearner。教官使用状态空间融合机制来将跨模式特征整合在统一的状态空间中，从而增强上下文理解并改善异质数据集成的鲁棒性。此外，它包括一种促进特征提取的融合反馈机制。 Unilearner利用与COS任务无关的多模式数据，以通过生成伪模式含量和跨模式语义关联来提高COS模型的分割能力。广泛的实验表明，无论是真实的还是伪型cos cos数据，教uniseg的表现都优于现有的多模式COS（MCOS）分段。此外，在多模式COS数据无法使用但多模式非COS数据的情况下，Unireter有效利用这些数据来增强细分性能。我们的代码将在\ href {https://github.com/cnyvfang/unicos} {github}上公开提供。

### SwimVG: Step-wise Multimodal Fusion and Adaption for Visual Grounding 
[[arxiv](https://arxiv.org/abs/2502.16786)] [[cool](https://papers.cool/arxiv/2502.16786)] [[pdf](https://arxiv.org/pdf/2502.16786)]
> **Authors**: Liangtao Shi,Ting Liu,Xiantao Hu,Yue Hu,Quanjun Yin,Richang Hong
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: 12 pages, 7 figures
- **标题**: 秋千：逐步多模式融合和视觉接地的适应
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉接地旨在通过自然语言将图像区域扎根，这在很大程度上依赖于跨模式对齐。大多数现有方法通过完全微调的单模式预训练模型分别传输视觉/语言知识，然后是简单的视觉变压器，用于多峰融合。但是，这些方法不仅限制了视觉和语言环境之间的足够相互作用，还限制了巨大的计算成本。因此，为了解决这些问题，我们探索了一个逐步的多模式融合和适应框架，即Swimvg。具体而言，SwimVG提出了逐步的多模式提示（SWIP）和跨模式交互式适配器（CIA），以进行视觉接地，以取代繁琐的变压器堆栈进行多模式融合。 SWIP可以以令牌级的融合方式逐步改善视觉和语言表示之间的对齐。此外，重量级CIA进一步通过跨模式相互作用促进了多模式融合。 SWIP和CIA都是参数有效的范式，它们逐渐融合了从浅层到深层的跨模式特征。四个广泛使用基准的实验结果表明，在效率方面，SwimVG具有出色的能力和相当大的好处。我们的代码可在https://github.com/liuting20/swimvg上找到。

### AeroReformer: Aerial Referring Transformer for UAV-based Referring Image Segmentation 
[[arxiv](https://arxiv.org/abs/2502.16680)] [[cool](https://papers.cool/arxiv/2502.16680)] [[pdf](https://arxiv.org/pdf/2502.16680)]
> **Authors**: Rui Li,Xiaowei Zhao
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: Aeroreformer：用于基于无人机的参考图像分割的航空引用变压器
- **领域**: 计算机视觉和模式识别
- **摘要**: 作为一项新颖而充满挑战的任务，引用分割将计算机视觉和自然语言处理结合在一起，以基于文本描述进行本地化和细分对象。虽然引用图像分割（RIS）在自然图像中进行了广泛的研究，但很少关注航空影像，尤其是从无人机（无人机）（UAVS）中。无人机图像的独特挑战，包括复杂的空间尺度，遮挡和不同的对象取向，使现有的RIS方法无效。一个关键的限制是缺乏特定于特定的数据集，因为手动注释的像素级面具和生成文本描述是劳动密集型且耗时的。为了解决此差距，我们设计了一个自动标记管道，该管道利用了预先存在的无人机分段数据集和多模式大语言模型（MLLM）来生成文本描述。此外，我们提出了空中参考变压器（AeroreFormer），这是无人机引用图像分割（UAV-RIS）的新型框架，其中包含一个视觉语言跨道模块（VLCAM），用于有效的跨模式理解和旋转意识到的多尺度融合（RAMSF）的解码器以增强裂缝场景，以增强序列的序列。在两个新开发的数据集上进行的广泛实验证明了Aerorformer优于现有方法，并为UAV-RIS建立了新的基准。数据集和代码将在以下网址公开可用：https：//github.com/lironui/aeroreformer。

### Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines 
[[arxiv](https://arxiv.org/abs/2502.16641)] [[cool](https://papers.cool/arxiv/2502.16641)] [[pdf](https://arxiv.org/pdf/2502.16641)]
> **Authors**: Xinwei Long,Zhiyuan Ma,Ermo Hua,Kaiyan Zhang,Biqing Qi,Bowen Zhou
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: AAAI-25
- **标题**: 通过内置自动回归搜索引擎回答检索启动的视觉问题
- **领域**: 计算机视觉和模式识别,计算语言学,信息检索
- **摘要**: 检索增强的一代（RAG）已出现，以解决知识密集的视觉问题回答（VQA）任务。当前方法主要采用单独的检索和发电模块来获取外部知识并产生答案。我们提出了Reause，这是基于知识的VQA任务的先前抹布模型的替代方法，该模型将知识回收者无缝地集成到生成的多模式大语言模型中，并用作内置的搜索引擎。具体而言，我们的模型既可以作为生成试器和准确的答案生成器发挥作用。它不仅通过为每个文档生成标识符来帮助从知识库中检索文档，而且还根据检索到的文档回答视觉问题。此外，我们提出了一个从相关反馈的加强检索校准模块，以提高检索性能并与偏好保持准确的答案生成。与强质基线相比，对两个代表性OKVQA和A-OKVQA数据集进行了广泛的实验表明，所有评估指标的显着改善范围从2.9 \％到9.6 \％。

### VidLBEval: Benchmarking and Mitigating Language Bias in Video-Involved LVLMs 
[[arxiv](https://arxiv.org/abs/2502.16602)] [[cool](https://papers.cool/arxiv/2502.16602)] [[pdf](https://arxiv.org/pdf/2502.16602)]
> **Authors**: Yiming Yang,Yangyang Guo,Hui Lu,Yan Wang
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: vidlbeval：视频涉及的LVLMS中的基准测试和缓解语言偏见
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 最近，大型视觉模型（LVLM）在各种多模式任务和基准方面取得了重大步骤。本文揭示了从现有视频涉及的LVLMS（语言偏见）中探索的问题，其中模型倾向于将语言优先于视频而不是视频，从而导致响应不正确。为了解决这一研究差距，我们首先收集视频语言偏见评估基准，该基准是专门设计的，旨在通过两个关键任务来评估视频涉及的LVLM的语言偏见：模棱两可的视频对比和疑问性问题探测。因此，我们设计的伴随的评估指标旨在惩罚语言偏见的LVLM。此外，我们还提出了多分支对比解码（MCD），向两个专家分支引入了两个专家分支，以抵消业余文本分支可能产生的语言偏见。我们的实验表明，i）现有的视频涉及的LVLM，包括专有和开源，在很大程度上受到语言偏见问题的限制； ii）我们的MCD可以有效地减轻此问题，并在各种视频涉及的LVLM中保持通用功能，而无需对模型体系结构进行任何其他重新培训或更改。

### Multimodal Large Language Models for Text-rich Image Understanding: A Comprehensive Review 
[[arxiv](https://arxiv.org/abs/2502.16586)] [[cool](https://papers.cool/arxiv/2502.16586)] [[pdf](https://arxiv.org/pdf/2502.16586)]
> **Authors**: Pei Fu,Tongkun Guan,Zining Wang,Zhentao Guo,Chen Duan,Hao Sun,Boming Chen,Jiayao Ma,Qianyi Jiang,Kai Zhou,Junfeng Luo
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 多模式的大型语言模型，用于文本丰富的图像理解：全面评论
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式大语言模型（MLLM）的最新出现引入了文本丰富的图像理解（TIU）领域的新维度，模型表现出令人印象深刻且令人鼓舞的性能。但是，他们的快速发展和广泛采用使得跟上最新进步变得越来越具有挑战性。为了解决这个问题，我们提出了一项系统的全面调查，以促进对TIU MLLM的进一步研究。最初，我们概述了几乎所有TIU MLLM的时间轴，架构和管道。然后，我们回顾主流基准上选定模型的性能。最后，我们探讨了该领域内有希望的方向，挑战和局限性。

### MQADet: A Plug-and-Play Paradigm for Enhancing Open-Vocabulary Object Detection via Multimodal Question Answering 
[[arxiv](https://arxiv.org/abs/2502.16486)] [[cool](https://papers.cool/arxiv/2502.16486)] [[pdf](https://arxiv.org/pdf/2502.16486)]
> **Authors**: Caixiong Li,Xiongwei Zhao,Jinhang Zhang,Xing Zhang,Qihao Sun,Zhou Wu
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: MQADET：通过多模式问题回答增强开放式摄制对象检测的插件范式
- **领域**: 计算机视觉和模式识别
- **摘要**: 开放式视频检测（OVD）是一项具有挑战性的任务，可以从一组无限制的类别（包括培训期间看不见的类别）中对对象进行分类。现有的开放式视频探测器受到复杂的视觉文本未对准和长尾类别的不平衡的限制，导致在挑战性的情况下表现出色。为了解决这些局限性，我们引入了MQADET，这是一种通用范式，用于通过利用多模式大语言模型（MLLMS）的跨模式推理能力来增强现有的开放式摄氏探测器。 MQADET充当插件解决方案，可与预训练的对象检测器无缝集成，而无需大量额外的培训成本。具体来说，我们设计了一种新颖的三阶段多模式问答（MQA）管道，以指导MLLM精确定位复杂的文本和视觉目标，同时有效地增强了现有对象检测器对相关对象的焦点。为了验证我们的方法，我们提出了一个新的基准测试，用于评估四个挑战性的开放式数据集的范式，使用三个最先进的对象探测器作为基准。实验结果表明，我们提出的范式显着改善了现有探测器的性能，尤其是在看不见的复杂类别中，在各种和具有挑战性的情况下。为了促进未来的研究，我们将公开发布我们的代码。

### Cross-domain Few-shot Object Detection with Multi-modal Textual Enrichment 
[[arxiv](https://arxiv.org/abs/2502.16469)] [[cool](https://papers.cool/arxiv/2502.16469)] [[pdf](https://arxiv.org/pdf/2502.16469)]
> **Authors**: Zeyu Shangguan,Daniel Seita,Mohammad Rostami
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: arXiv admin note: substantial text overlap with arXiv:2403.16188
- **标题**: 跨域几射击对象检测，具有多模式的文本富集
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在几次学习任务中，跨模式特征提取和集成的进步可以显着提高性能。但是，当遇到实质域移动时，当前的多模式对象检测（MM-OD）方法通常会出现显着的性能降解。我们建议，合并丰富的文本信息可以使模型能够在视觉实例及其相应的语言描述之间建立更强大的知识关系，从而减轻域转移的挑战。具体而言，我们专注于跨域多模式的几射击对象检测（CDMM-FSOD）的问题，并引入了一个基于元学习的框架，旨在利用丰富的文本语义作为辅助模式，以实现有效的域适应性。我们的新体系结构包含了两个关键组成部分：（i）多模式特征聚合模块，该模块将视觉和语言特征嵌入对齐，以确保跨模态的凝聚力整合。 （ii）一种丰富的文本语义整流模块，该模块采用双向文本特征生成来完善多模式特征对齐，从而增强对语言及其在对象检测中的应用的理解。我们在常见的跨域对象检测基准上评估了提出的方法，并证明它显着超过了现有的少数射击对象检测方法。

### VisFactor: Benchmarking Fundamental Visual Cognition in Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.16435)] [[cool](https://papers.cool/arxiv/2502.16435)] [[pdf](https://arxiv.org/pdf/2502.16435)]
> **Authors**: Jen-Tse Huang,Dasen Dai,Jen-Yuan Huang,Youliang Yuan,Xiaoyuan Liu,Wenxuan Wang,Wenxiang Jiao,Pinjia He,Zhaopeng Tu
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: Working in Progress
- **标题**: 粘性器：多模式模型中基本的视觉认知基准测试
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 多模式的大语言模型（MLLM）在多模式理解方面表现出显着的进步。但是，它们的基本视觉认知能力在很大程度上仍然没有得到充实。为了弥合这一差距，我们介绍了粘性，这是一种从因子引用的认知检验（FRCT）得出的新基准，这是对人类认知的完善心理测量评估。粘性器将与视觉相关的FRCT子测验数字化，以系统地评估基本视觉认知任务的MLLM，包括空间推理，感知速度和模式识别。我们对最先进的MLLM进行了全面评估，例如GPT-4O，Gemini-Pro和Qwen-Vl，使用粘性器在各种促使策略等促使链条和多代理辩论之类的促使策略中进行了辩论。我们的发现揭示了当前MLLM的基本视觉认知的缺陷，即使使用高级提示技术，性能也经常接近随机猜测，并且仅显示边缘改进。这些结果强调了重点研究的关键需求，以增强MLLM的核心视觉推理能力。为了促进该领域的进一步调查，我们在https://github.com/cuhk-arise/disfactor上发布了粘性基准测试。

### Visual Reasoning Evaluation of Grok, Deepseek Janus, Gemini, Qwen, Mistral, and ChatGPT 
[[arxiv](https://arxiv.org/abs/2502.16428)] [[cool](https://papers.cool/arxiv/2502.16428)] [[pdf](https://arxiv.org/pdf/2502.16428)]
> **Authors**: Nidhal Jegham,Marwan Abdelatti,Abdeltawab Hendawi
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: Grok，DeepSeek Janus，Gemini，Qwen，Mistral和Chatgpt的视觉推理评估
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式大语言模型（LLM）的传统评估受到关注单像推理的限制，无法评估关键方面，例如上下文理解，推理稳定性和不确定性校准。这项研究通过引入一个新的基准测试来解决这些局限性，该基准将多图像推理任务与基于拒绝的评估和位置偏见检测相结合。为了评估这些维度，我们进一步引入熵作为一种新的度量标准，用于量化重新排序的答案变体的推理一致性。我们将此基准测试用于评估Grok 3，Chatgpt-4O，Chatgpt-O1，Gemini 2.0 Flash实验，DeepSeek Janus模型，QWEN2.5-VL-72B-Instruct，QVQ-72B-Preview和Pixtral 12B，包括八个视觉推理任务，包括差异差异和图形。我们的发现揭示了CHATGPT-O1的总体准确性（82.5 \％）和排斥准确性（70.0 \％），紧随其后的是Gemini 2.0 Flash实验性（70.8 \％）。 QVQ-72B-preiview表现出较高的排斥准确性（85.5 \％）。值得注意的是，PixTral 12b（51.7 \％）在特定领域显示出希望，而Janus模型在偏置和不确定性校准方面表现出挑战，反映在低排斥准确性和高熵分数中。 Janus模型（Janus 7b：0.8392，Janus 1b：0.787）的高熵得分强调了它们对位置偏见和不稳定推理的敏感性，与低熵和Chatgpt模型的稳健推理形成鲜明对比。该研究进一步表明，模型大小不是性能的唯一决定因素，尽管Grok 3的表现不佳，尽管其大量参数计数。通过采用多图像上下文，拒绝机制和基于熵的一致性指标，该基准标准为评估多模式LLMS设定了新的标准，从而实现了对下一代AI系统的更强大和可靠的评估。

### A Survey on Industrial Anomalies Synthesis 
[[arxiv](https://arxiv.org/abs/2502.16412)] [[cool](https://papers.cool/arxiv/2502.16412)] [[pdf](https://arxiv.org/pdf/2502.16412)]
> **Authors**: Xichen Xu,Yanshu Wang,Yawen Huang,Jiaqi Liu,Xiaoning Lei,Guoyang Xie,Guannan Jiang,Zhichao Lu
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 一项关于工业异常合成的调查
- **领域**: 计算机视觉和模式识别,计算工程、金融和科学
- **摘要**: 本文全面回顾了异常合成方法。现有的调查专注于有限的技术，缺少整体现场视图和理解方法互连。相比之下，我们的研究提供了统一的综述，涵盖了基于手工制作的，基于分布的生成模型（GM）的基于手工制作的，基于视觉模型和基于视觉模型（VLM）基于基于的综合的代表性方法。我们介绍了第一个工业异常合成（IAS）分类法。先前的工作缺乏正式分类或使用简单的分类法，妨碍结构化的比较和趋势识别。我们的分类法提供了一个精细的框架，反映了方法论进步和实际含义，从而扎根未来的研究。此外，我们探讨了跨模式合成和大规模VLM。先前的调查忽略了异常合成中的多模式数据和VLM，将见解限制在其优势中。我们的调查分析了它们的整合，收益，挑战和前景，提供了通过多模式学习来提高IAS的路线图。可以在https://github.com/m-3lab/awesome-anomaly-synthesis中获得更多资源。

### Audio Visual Segmentation Through Text Embeddings 
[[arxiv](https://arxiv.org/abs/2502.16359)] [[cool](https://papers.cool/arxiv/2502.16359)] [[pdf](https://arxiv.org/pdf/2502.16359)]
> **Authors**: Kyungbok Lee,You Zhang,Zhiyao Duan
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 通过文本嵌入的视觉视觉分割
- **领域**: 计算机视觉和模式识别,人工智能,多媒体
- **摘要**: 视听细分（AVS）的目的是从视频帧本地化和分割源源对象。从事AV的研究人员遭受数据集有限的数据集，因为手工制作的注释很昂贵。最近的工作试图通过利用细分基础模型SAM来克服有限数据的挑战，并提示音频以增强其分割源源对象的能力。尽管这种方法通过利用对SAM的预先培训的知识来减轻模型理解视觉方式的负担，但它并未解决有限数据集的基本挑战，以学习视听关系。为了解决这些限制，我们提出\ textbf {av2t-sam}，这是一个新颖的框架，它将音频特征带入预先训练的文本启动的SAM的文本嵌入空间。我们的方法利用了从丰富的文本图像配对数据集中学到的多模式对应关系，以增强视听对齐。此外，我们介绍了一个新颖的功能，$ \ mathbf {\ textIt {\ textbf {f}} _ {clip} \ odot \ textIt {\ textbf {f textbf {f}} _ {f} _ {clap}}} $，这强调了在噪音中分享的音频和视觉模态的共享语义，同时又有噪音。 AVSBench数据集的实验显示了AVSBENCE的两个数据集上的最先进性能。我们的方法通过有效利用验证的分割模型和跨模式的语义一致性来超过现有方法。

### Mojito: LLM-Aided Motion Instructor with Jitter-Reduced Inertial Tokens 
[[arxiv](https://arxiv.org/abs/2502.16175)] [[cool](https://papers.cool/arxiv/2502.16175)] [[pdf](https://arxiv.org/pdf/2502.16175)]
> **Authors**: Ziwei Shan,Yaoyu He,Chengfeng Zhao,Jiashen Du,Jingyan Zhang,Qixuan Zhang,Jingyi Yu,Lan Xu
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: First three authors contribute equally. Project page: https://koyui.github.io/mojito/
- **标题**: 莫吉托：LLM辅助运动讲师，以及抖动的惯性令牌
- **领域**: 计算机视觉和模式识别,人工智能,图形
- **摘要**: 人的身体运动传达了对行动意图和认知过程的批判性见解，但现有的多模式系统主要集中于通过语言，视觉和音频来理解人类运动，这些系统努力捕获3D运动中固有的动态力量和扭矩。惯性测量单元（IMU）提出了一种有希望的替代方案，可轻巧，可穿戴和隐私意识感测。但是，流媒体IMU数据的处理面临着诸如无线传输不稳定性，传感器噪声和漂移等挑战，限制了其长期实时运动捕获（MOCAP）的效用，更重要的是在线运动分析。为了应对这些挑战，我们介绍了莫吉托（Mojito），这是一种智能运动代理，将惯性传感与大语言模型（LLM）集成在一起，以进行交互式运动捕获和行为分析。

### OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing and Its Generality to Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.16161)] [[cool](https://papers.cool/arxiv/2502.16161)] [[pdf](https://arxiv.org/pdf/2502.16161)]
> **Authors**: Wenwen Yu,Zhibo Yang,Jianqiang Wan,Sibo Song,Jun Tang,Wenqing Cheng,Yuliang Liu,Xiang Bai
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: omn​​iparser v2：统一视觉文本解析的结构化点及其对多模式大语言模型的一般性
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 视觉上的文本解析（VSTP）最近看到了显着的进步，这是由于对自动文档理解的需求不断增长以及能够处理基于文档的问题的大型语言模型的出现。尽管已经提出了各种方法来解决VSTP的复杂性，但现有解决方案通常依赖于特定于任务的架构和目标来解决各个任务。由于目标多样化和异构模式，这导致了模态隔离和复杂的工作流程。在本文中，我们将OmniparSer V2介绍为统一VSTP典型任务的通用模型，包括文本斑点，密钥信息提取，表识别和布局分析，将其用于统一的框架。我们方法的核心是提出的结构化重点（点）提示图式模式，该图案通过利用统一的编码器架构，目标和输入\＆输出表示来提高各种情况的模型性能。 SPOT消除了对特定于任务的体系结构和损失功能的需求，从而大大简化了处理管道。我们对八个不同数据集的四个任务进行了广泛的评估表明，Omniparser V2在VSTP中实现了最先进或竞争性的结果。此外，我们探讨了斑点在多模式大语言模型结构中的集成，进一步增强了文本本地化和识别能力，从而确认了斑点提示技术的一般性。该代码可在\ href {https://github.com/alibabaresearch/advancedliteratemachinery} {AdvancessLiterateMachinery}中获得。

### FeatSharp: Your Vision Model Features, Sharper 
[[arxiv](https://arxiv.org/abs/2502.16025)] [[cool](https://papers.cool/arxiv/2502.16025)] [[pdf](https://arxiv.org/pdf/2502.16025)]
> **Authors**: Mike Ranzinger,Greg Heinrich,Pavlo Molchanov,Jan Kautz,Bryan Catanzaro,Andrew Tao
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 壮举Sharp：您的视觉模型功能，更清晰
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉编码器的特征图是无数现代AI任务的基础，从核心感知算法（例如语义分割，对象检测，深度感知等）到视觉模型（VLMS）中的现代多模态理解。目前，在计算机视觉中，通用视觉骨架的前沿是视觉变压器（VIT），通常使用对比度损失（例如剪辑）训练。大多数现成的VIT，尤其是剪辑的关键问题是这些模型的分辨率较低。大多数以224x224px运行，而“高分辨率”版本约为378-448px，但仍然不灵活。我们介绍了一种新颖的方法，可将低分辨率视觉编码器的特征图均匀地示例置于同时拾取细粒细节，否则由于分辨率而丢失的细节。我们证明了这种方法对核心感知任务以及聚集模型（无线电）培训中的有效性，作为提供更丰富的蒸馏目标的一种方式。

### Multi-Agent Multimodal Models for Multicultural Text to Image Generation 
[[arxiv](https://arxiv.org/abs/2502.15972)] [[cool](https://papers.cool/arxiv/2502.15972)] [[pdf](https://arxiv.org/pdf/2502.15972)]
> **Authors**: Parth Bhalerao,Mounika Yalamarty,Brian Trinh,Oana Ignat
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 多元文化文本以形象生成多元文化的多模型模型
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 大型语言模型（LLMS）在各种多模式任务中表现出令人印象深刻的性能。但是，由于现有数据和模型的主要以西方为中心的性质，它们在跨文化环境中的有效性仍然有限。同时，多代理模型在解决复杂的任务方面表现出很强的功能。在本文中，我们评估了LLM在多元文化图像生成的新任务中在多代理相互作用环境中的性能。我们的主要贡献是：（1）我们介绍了Mosaig，这是一个多代理框架，通过利用具有不同文化角色的LLM来增强多元文化形象的产生； （2）我们提供了一个跨越五个国家，三个年龄段，两个性别，25个历史地标和五种语言的9,000个多元文化图像的数据集； （3）我们证明，多代理相互作用的表现优于多个评估指标的简单，无代理的模型，为未来的研究提供了宝贵的见解。我们的数据集和模型可在https://github.com/oanaignat/mosaig上找到。

### Forgotten Polygons: Multimodal Large Language Models are Shape-Blind 
[[arxiv](https://arxiv.org/abs/2502.15969)] [[cool](https://papers.cool/arxiv/2502.15969)] [[pdf](https://arxiv.org/pdf/2502.15969)]
> **Authors**: William Rudman,Michal Golovanesky,Amir Bar,Vedant Palit,Yann LeCun,Carsten Eickhoff,Ritambhara Singh
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 遗忘的多边形：多模式的大语言模型是形状盲
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 尽管在视觉任务上表现出色，但多模式大语言模型（MLLM）与数学解决问题解决方案斗争，开源和最先进的模型都无法在视觉记录基准测试中人类的性能。为了系统地检查MLLM中的视觉数学推理，我们（1）评估了他们对几何原始素的理解，（2）测试多步电推理，以及（3）探索潜在的解决方案，以提高视觉推理能力。我们的发现表明了形状识别的基本缺陷，顶级模型在识别常规多边形方面的准确性低于50％。我们通过双流程理论的角度分析了这些故障，并表明MLLM依赖于系统1（直观，记忆的关联），而不是系统2（故意推理）。因此，MLLM无法计算熟悉的形状和新颖形状的方面，这表明他们既没有学会侧面的概念，也没有有效地处理视觉输入。最后，我们提出了视觉提示的思想链（VC-COT）提示，该提示通过在图表中明确引用视觉注释来增强多步数学推理，从而在不规则的Polygon副核对任务上提高了GPT-4O的准确性，从而从7％提高了7％至93％。我们的发现表明，MLLM中的系统2推理仍然是一个空旷的问题，并且视觉引导的提示对于成功吸引视觉推理至关重要。可用的代码，网址为：https：//github.com/rsinghlab/shape-blind。

### Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis 
[[arxiv](https://arxiv.org/abs/2502.15635)] [[cool](https://papers.cool/arxiv/2502.15635)] [[pdf](https://arxiv.org/pdf/2502.15635)]
> **Authors**: Ziqian Ni,Sicong Du,Zhenghua Hou,Chenming Wu,Sheng Yang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: Accepted by International Conference on 3D Vision (3DV) 2025
- **标题**: Para-lane：多车道数据集注册平行扫描以进行基准测试新视图合成
- **领域**: 计算机视觉和模式识别
- **摘要**: 为了评估端到端的自主驾驶系统，基于新型视图合成（NVS）技术的模拟环境至关重要，它综合了来自先前记录的新车辆姿势下的照片真实图像和点云，尤其是在跨界场景中。因此，必须开发多车道数据集和基准。尽管最近基于合成场景的NVS数据集已准备好用于跨车道基准测试，但它们仍然缺乏捕获的图像和点云的现实主义。为了进一步评估基于NERF和3DG的现有方法的性能，我们介绍了第一个专门用于新型驾驶视图合成数据集的登记并行扫描的多车道数据集，该数据集衍生自现实世界扫描，包括25组相关序列，包括16,000个前视图，包括64,000个周围环境图像和16,000 Lidar frames frames frames frames frames frame scans。所有帧都标记为将移动对象与静态元素区分开。使用此数据集，我们评估了不同车道和距离的各种测试方案中现有方法的性能。此外，我们的方法还提供了解决和评估多模式数据一致性的多传感器姿势质量的解决方案，以在现实世界中策划此类数据集。我们计划不断添加新序列，以测试不同情况下现有方法的概括。该数据集在项目页面公开发布：https：//nizqleo.github.io/paralane-dataset/。

### Memory Helps, but Confabulation Misleads: Understanding Streaming Events in Videos with MLLMs 
[[arxiv](https://arxiv.org/abs/2502.15457)] [[cool](https://papers.cool/arxiv/2502.15457)] [[pdf](https://arxiv.org/pdf/2502.15457)]
> **Authors**: Gengyuan Zhang,Mingcong Ding,Tong Liu,Yao Zhang,Volker Tresp
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: Short paper (5 pages)
- **标题**: 内存有帮助，但造型误导：与MLLM的视频中了解流媒体事件
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式的大语言模型（MLLM）在整体上了解视频方面表现出了很强的表现，但是它们处理流视频视频的能力被视为一系列视觉事件捕获序列，却没有被驱动。直观地，利用过去的事件作为内存可以丰富对当前事件的上下文和时间的理解。在本文中，我们表明，将记忆作为上下文有助于MLLM更好地理解视频事件。但是，由于这样的记忆依赖于前事件的预测，因此它们可能包含错误信息，导致串联和降级性能。为了解决这个问题，我们提出了一种综合感知的内存修改方法，该方法减轻了对内存增强事件的理解的库存记忆。

### MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition 
[[arxiv](https://arxiv.org/abs/2502.15448)] [[cool](https://papers.cool/arxiv/2502.15448)] [[pdf](https://arxiv.org/pdf/2502.15448)]
> **Authors**: Paul Koch,Marian Schlüter,Jörg Krüger
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: Accepted to IMPROVE 2025
- **标题**: MVIP-用于以应用程序为导向的多视图和多模式工业零件识别的数据集和方法
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 我们提出了MVIP，这是一种新型数据集，用于多模式和多视图，面向应用程序的工业部分识别。在这里，我们是第一个将校准的RGBD多视图数据集与其他对象上下文（例如物理属性，自然语言和超级类）相结合的人。当前可用数据集的投资组合提供了广泛的表示和基准相关方法的表示。与现有的分类挑战相反，工业识别应用程序提供了受控的多模式环境，但同时有与传统的2D/3D分类挑战不同的问题。通常，工业应用必须处理少量或增加数量的培训数据，视觉上相似的零件以及变化的物体大小，同时需要在成本和时间限制下达到可靠的100％前5个准确性。当前的方法可以单独应对此类挑战，但是在工业应用中直接采用这些方法很复杂，需要进一步研究。 MVIP的主要目标是研究和推动相关下游任务中各种最新方法的可转移性，以有效地部署工业分类器。此外，我们打算对MVIP研究进行有关几种模态融合主题，（自动化）合成数据生成和复杂数据采样的研究 - 在单个面向应用程序的基准中合并。

### LongCaptioning: Unlocking the Power of Long Video Caption Generation in Large Multimodal Models 
[[arxiv](https://arxiv.org/abs/2502.15393)] [[cool](https://papers.cool/arxiv/2502.15393)] [[pdf](https://arxiv.org/pdf/2502.15393)]
> **Authors**: Hongchen Wei,Zhihong Tan,Yaosi Hu,Chang Wen Chen,Zhenzhong Chen
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 长距离训练：在大型多模型中解锁长视频字幕生成的力量
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型多模型模型（LMM）在视频字幕任务中表现出了出色的性能，尤其是对于简短的视频。但是，随着视频的长度的增加，生成长而详细的字幕成为一个重大挑战。在本文中，我们研究了LMM在为长视频生成长字幕时的局限性。我们的分析表明，开源LMM难以始终如一地产生超过300个单词的输出，从而导致视觉内容的不完整或过于简洁的描述。这种限制阻碍了LMM为长时间视频提供全面和详细的字幕的能力，最终缺少重要的视觉信息。通过受控实验，我们发现在训练过程中长期限制的配对示例的稀缺是限制模型输出长度的主要因素。然而，手动注释长期视频的长期示例是耗时且昂贵的。为了克服注释瓶颈，我们提出了长束代理，该框架通过层次的语义聚合综合了长字幕数据。 ％汇总的多层次描述。我们使用远程代理，我们策划了一个新的长束缚数据集，longcaption-10k。我们还开发了Longcaption Bench，这是一种基准测试，旨在全面评估LMMS产生的长字幕的质量。通过将Longcaption-10k纳入培训中，我们使LMMS能够为长形式视频产生超过1,000个字的字幕，同时保持高输出质量。在Longcaption Bench中，我们的模型达到了最先进的性能，甚至超过了GPT4O等较大的专有模型。

### MOVE: A Mixture-of-Vision-Encoders Approach for Domain-Focused Vision-Language Processing 
[[arxiv](https://arxiv.org/abs/2502.15381)] [[cool](https://papers.cool/arxiv/2502.15381)] [[pdf](https://arxiv.org/pdf/2502.15381)]
> **Authors**: Matvey Skripkin,Elizaveta Goncharova,Dmitrii Tarasov,Andrey Kuznetsov
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 10 pages, 6 figures, 4 tables
- **标题**: 移动：以域名为中心的视觉处理的视频编码器方法
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式语言模型（MLMS）通过通过特定适配器将视觉编码器与大语言模型耦合来整合视觉和文本信息。尽管现有方法通常依赖于单个预训练的视觉编码器，但专门编码器的差异很大，可以提高模型在不同域中的性能。在这项工作中，我们提出了移动（视觉编码器的混合）一种简单而有效的方法，可以利用多个预训练的编码器来进行专门的多模式任务。在Unichat，InternVit和Scemify等候选人中自动将输入的路由输入到最合适的编码器，从而提高了各种基准测试的性能，包括ChartQA，MMBENCH和MMMU。实验结果表明，移动可以达到竞争精度，而不会产生高分辨率图像的图像切片的复杂性。

### PFSD: A Multi-Modal Pedestrian-Focus Scene Dataset for Rich Tasks in Semi-Structured Environments 
[[arxiv](https://arxiv.org/abs/2502.15342)] [[cool](https://papers.cool/arxiv/2502.15342)] [[pdf](https://arxiv.org/pdf/2502.15342)]
> **Authors**: Yueting Liu,Hanshi Wang,Zhengjun Zha,Weiming Hu,Jin Gao
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: PFSD：在半结构化环境中用于丰富任务的多模式行人对焦场景数据集
- **领域**: 计算机视觉和模式识别
- **摘要**: 自主驾驶感知的最新进步揭示了以车辆交通为主的结构化环境中的出色能力。但是，当前的感知模型在半结构化环境中表现出显着的局限性，在半结构化的环境中，动态行人的运动不规则运动和遮挡占据了多样性。我们将这一缺点归因于半结构化场景中高质量数据集的稀缺性，尤其是关于行人的感知和预测。在这项工作中，我们介绍了以Nuscenes格式在半结构化场景中严格注释的多模式行人专注的场景数据集（PFSD）。 PFSD提供了全面的多模式数据注释，其中包括点云分割，检测和跟踪对象ID。它包括在各种情况下捕获的13万多个行人实例，并具有不同的密度，运动模式和遮挡。此外，为了证明应对更多样化和复杂的半结构化环境所带来的挑战的重要性，我们提出了一种新型混合多尺度融合网络（HMFN）。具体而言，为了检测人口稠密且遮挡的场景中的行人，我们的方法使用精心设计的混合框架有效地捕获和融合了多规模的特征，该特征集成了稀疏和香草卷积。 PFSD的广泛实验表明，HMFN比现有方法的平均平均精度（MAP）取得了改善，从而强调了其在解决复杂半结构化环境中3D行人检测的挑战方面的功效。可以使用编码和基准。

### Research advances on fish feeding behavior recognition and intensity quantification methods in aquaculture 
[[arxiv](https://arxiv.org/abs/2502.15311)] [[cool](https://papers.cool/arxiv/2502.15311)] [[pdf](https://arxiv.org/pdf/2502.15311)]
> **Authors**: Shulong Zhang,Daoliang Li,Jiayin Zhao,Mingyuan Yao,Yingyi Chen,Yukang Huo,Xiao Liu,Haihua Wang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 22 pages, 4 figures,
- **标题**: 水产养殖中鱼类进食行为识别和强度定量方法的研究进展
- **领域**: 计算机视觉和模式识别,新兴技术
- **摘要**: 作为水产养殖管理的关键部分，鱼类喂养行为识别和强度量化一直是研究人员非常关注的热门领域，并且在监测鱼类健康，指导诱饵工作和提高水产养殖效率方面起着至关重要的作用。为了更好地完成相关工作，本文首先回顾了基于单个模式中计算机视觉，声学和传感器的鱼类喂养行为识别和强度定量方法的研究进展。然后，阐述了当前新出现的多模式融合在鱼类进食行为识别和强度定量方法中的应用。最后，比较和分析了各种技术的优势和缺点，并设想了未来的研究方向。

### AutoMR: A Universal Time Series Motion Recognition Pipeline 
[[arxiv](https://arxiv.org/abs/2502.15228)] [[cool](https://papers.cool/arxiv/2502.15228)] [[pdf](https://arxiv.org/pdf/2502.15228)]
> **Authors**: Likun Zhang,Sicheng Yang,Zhuo Wang,Haining Liang,Junxiao Shen
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 5 figures
- **标题**: Automr：通用时间序列运动识别管道
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在本文中，我们提出了为多模式数据集设计的端到端自动运动识别（AUTOMR）管道。提出的框架无缝整合了数据预处理，模型训练，高参数调整和评估，从而在各种情况下实现了稳健的性能。我们的方法解决了两个主要挑战：1）传感器数据格式和数据集的参数的可变性，传统上需要特定于任务的机器学习实现，以及2）超参数调整的复杂性和时间消耗以实现最佳模型性能。我们的图书馆提供了一种将石英NET作为核心模型，自动超参数调整和全面指标跟踪的多合一解决方案。广泛的实验证明了其在10个不同数据集上的有效性，从而实现了最新的性能。这项工作为在各种现实世界应用程序中部署运动捕获解决方案奠定了坚实的基础。

### Easy-Poly: A Easy Polyhedral Framework For 3D Multi-Object Tracking 
[[arxiv](https://arxiv.org/abs/2502.17822)] [[cool](https://papers.cool/arxiv/2502.17822)] [[pdf](https://arxiv.org/pdf/2502.17822)]
> **Authors**: Peng Zhang,Xin Li,Xin Lin,Liang He
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 8 pages, 3 figures, 5 tables
- **标题**: Easy-Poly：用于3D多对象跟踪的简单多面体框架
- **领域**: 计算机视觉和模式识别
- **摘要**: 3D多对象跟踪（3D MOT）的最新进展主要依赖于逐个检测管道。但是，这些方法通常会忽略3D检测过程中的潜在增强，从而导致高误报（FP），遗漏检测（FN）和身份转换（IDS），尤其是在挑战性的情况下，例如拥挤的场景，小物体配置和不利天气条件。此外，数据预处理，关联机制，运动建模和生命周期管理的局限性阻碍了整体跟踪鲁棒性。为了解决这些问题，我们提出了Easy-Poly，这是一个实时的，基于过滤器的3D MOT框架，用于多个对象类别。我们的贡献包括：（1）利用多模式数据增强和精制SPCONV操作的增强提案生成器，可显着改善Nuscenes的地图和NDS； （2）一种动态轨道（DTO）数据关联算法，该算法通过最佳分配和多个假设处理来有效地管理不确定性和遮挡； （3）一种动态运动建模（DMM），结合了置信加权的卡尔曼过滤器和自适应噪声协方差，在具有挑战性的条件下增强了MOTA和AMOTA； （4）具有调节阈值的扩展生命周期管理系统，以减少ID开关和错误终止。实验结果表明，Easy Poly优于最先进的方法，例如Poly-Mot和Fast-Poly，在MAP中取得了显着的增长（例如，使用groundkernel3d）和Amota（例如，从63.30％到64.96％）和AMOTA（例如，从73.1％到74.5％），而同时进行实时运行。这些发现突出显示了在不同情况下轻松的适应性和鲁棒性，使其成为自动驾驶和相关3D MOT应用的引人注目的选择。本文的源代码将在接受后发表。

### Contrastive Visual Data Augmentation 
[[arxiv](https://arxiv.org/abs/2502.17709)] [[cool](https://papers.cool/arxiv/2502.17709)] [[pdf](https://arxiv.org/pdf/2502.17709)]
> **Authors**: Yu Zhou,Bingxuan Li,Mohan Tang,Xiaomeng Jin,Te-Lin Wu,Kuan-Hao Huang,Heng Ji,Kai-Wei Chang,Nanyun Peng
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 对比性视觉数据增加
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,机器学习,多媒体
- **摘要**: 大型多模型模型（LMM）通常很难识别新颖的概念，因为它们依赖于预训练的知识，并且能够捕获微妙的视觉细节的能力。培训中特定于领域的知识差距也使它们容易混淆视觉相似，通常歪曲或低资源概念。为了帮助LMM更好地使细微差别的视觉特征与语言保持一致，提高了他们识别和理由的新颖或稀有概念的能力，我们提出了一种对比性视觉数据增强（CODA）策略。 CODA提取目标概念的关键对比文本和视觉特征与已知的概念被误认为为已知概念，然后使用多模式生成模型生成目标的合成数据。如人类注释者所验证的那样，实施了提取的功能和增强图像的自动过滤，以确保其质量。我们显示了尾声对低资源概念以及包括Inaturalist和Sun在内的各种场景识别数据集的有效性和效率。我们还收集了小说《小说》，这是一个由新发现的动物物种组成的基准数据集，这些动物物种保证不会被LMMS看到。 LLAVA-1.6这三个数据集上的1-shot更新结果表明，CODA显着将SOTA视觉数据增强策略提高了12.3％（小说类），5.1％（Sun）和6.0％（inatat）的准确性（INAT）绝对获得。

### METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling 
[[arxiv](https://arxiv.org/abs/2502.17651)] [[cool](https://papers.cool/arxiv/2502.17651)] [[pdf](https://arxiv.org/pdf/2502.17651)]
> **Authors**: Bingxuan Li,Yiwei Wang,Jiuxiang Gu,Kai-Wei Chang,Nanyun Peng
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 金属：通过测试时间缩放的图表生成的多代理框架
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 图表生成旨在生成代码以产生满足所需的视觉属性的图表，例如文本，布局，颜色和类型。它具有巨大的潜力，可以在财务分析，研究表现，教育和医疗保健中赋予自动专业报告的能力。在这项工作中，我们建立了一个基于视觉语言模型（VLM）的多代理框架，以生成有效的自动图表。生成高质量的图表需要强大的视觉设计技能和精确的编码功能，将所需的视觉属性嵌入代码中。对于直接提示VLM，很难进行如此复杂的多模式推理过程。为了解决这些挑战，我们提出了金属，这是一个多代理框架，将图表生成的任务分解为专业代理之间的迭代协作。金属在图表生成任务中的最佳结果比当前的最佳结果提高了5.2％。金属框架表现出测试时间缩放的现象：随着对数计算预算从512个令牌增长到8192代币，其性能会单调增加。此外，我们发现在金属的批评过程中分离不同的方式可以增强VLM在多模式背景下的自我纠正能力。

### PosterSum: A Multimodal Benchmark for Scientific Poster Summarization 
[[arxiv](https://arxiv.org/abs/2502.17540)] [[cool](https://papers.cool/arxiv/2502.17540)] [[pdf](https://arxiv.org/pdf/2502.17540)]
> **Authors**: Rohit Saxena,Pasquale Minervini,Frank Keller
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: This paper includes a dataset of research posters with abstracts. We provide two cited examples ( arXiv:2211.11880 and arXiv:2210.07571 ) to illustrate reference summaries
- **标题**: 海报：科学海报摘要的多模式基准
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 从多模式文档中生成准确而简洁的文本摘要是具有挑战性的，尤其是在处理诸如科学海报之类的视觉上复杂内容时。我们介绍了海报，这是一种新颖的基准测试，旨在推动视力模型的发展，这些模型可以理解并将科学海报总结到研究论文摘要中。我们的数据集包含16,305次会议海报，并将其相应的摘要与摘要配对。每个海报以图像格式提供，并提出各种视觉理解挑战，例如复杂的布局，密集的文本区域，表格和数字。我们在海报上基准了最先进的多模式大型语言模型（MLLM），并证明它们难以准确解释和总结科学海报。我们提出了段并总结，这是一种层次结构方法，在自动指标上的当前MLLM胜过，在Rouge-L中获得了3.14％的增长。这将是关于海报摘要的未来研究的起点。

### Introducing Visual Perception Token into Multimodal Large Language Model 
[[arxiv](https://arxiv.org/abs/2502.17425)] [[cool](https://papers.cool/arxiv/2502.17425)] [[pdf](https://arxiv.org/pdf/2502.17425)]
> **Authors**: Runpeng Yu,Xinyin Ma,Xinchao Wang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 将视觉感知令牌引入多模式大语言模型
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 为了利用视觉信息，多模式大语言模型（MLLM）依赖于其视觉编码器的感知过程。视觉感知的完整性和准确性显着影响空间推理，细粒度理解和其他任务的精度。但是，MLLM仍然缺乏控制自己的视觉感知过程的自主能力，例如，选择性地审查图像的特定区域或专注于与特定对象类别相关的信息。在这项工作中，我们提出了视觉感知令牌的概念，旨在增强MLLM的能力，以控制其视觉感知过程。我们设计了两种类型的视觉感知令牌，称为区域选择令牌和视力重新编码令牌。 MLLMS自主生成这些令牌，就像它们生成文本一样，并使用它们来触发其他视觉感知动作。区域选择令牌明确标识需要进一步感知的图像中的特定区域，而视觉重新编码令牌则使用其隐藏状态作为控制信号来指导其他视觉感知过程。广泛的实验证明了这些令牌在处理空间推理，改善细粒度的理解和其他任务方面的优势。平均而言，视觉感知令牌的引入将2B模型的性能提高了23.6 \％，将其得分从0.572提高到0.708，甚至超过7B参数模型的分数提高了13.4 \％（从0.624起）。请查看我们的回购

### MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.17422)] [[cool](https://papers.cool/arxiv/2502.17422)] [[pdf](https://arxiv.org/pdf/2502.17422)]
> **Authors**: Jiarui Zhang,Mahyar Khayatkhoei,Prateek Chhikara,Filip Ilievski
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: Published as a conference paper at ICLR 2025. Code at: https://github.com/saccharomycetes/mllms_know
- **标题**: MLLM知道在哪里看：使用多模式LLM的小视觉细节的无训练感知
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 近年来，多模式大型语言模型（MLLM）在视觉识别任务方面经历了快速的进步。鉴于它们的潜在集成到许多关键应用中，因此重要的是要了解其视觉感知的局限性。在这项工作中，我们研究MLLM在回答有关图像的问题时是否可以像大型视觉细节一样有效地感知小型视觉细节。我们观察到它们的性能对问题的视觉主题的大小非常敏感，并进一步表明，这种效果实际上是通过进行干预研究而因果关系。接下来，我们在回答视觉问题时研究MLLM的注意力模式，并有趣地发现，即使他们提供了错误的答案，他们也一贯知道在哪里看。基于这些发现，我们建议采用无训练的视觉干预方法，以以注意力和梯度图的形式利用任何MLLM本身的内部知识，以增强其对小型视觉细节的看法。我们在两个广泛使用的MLLM和七个视觉问题上评估了我们提出的方法回答基准，并表明它们可以显着提高MLLM的精度而无需进行任何培训。我们的结果阐明了将MLLM应用于有关小细节的视觉识别任务的风险，并指出使用模型内部状态的视觉干预是减轻这种风险的有希望的方向。

### Parameter Efficient Merging for Multimodal Large Language Models with Complementary Parameter Adaptation 
[[arxiv](https://arxiv.org/abs/2502.17159)] [[cool](https://papers.cool/arxiv/2502.17159)] [[pdf](https://arxiv.org/pdf/2502.17159)]
> **Authors**: Fanhu Zeng,Haiyang Guo,Fei Zhu,Li Shen,Hao Tang
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 具有互补参数适应的多模式大语言模型的参数有效合并
- **领域**: 计算机视觉和模式识别
- **摘要**: 通过自定义数据进行微调预训练的模型可为特定任务提供众多专家模型。将模型合并为一个通用模型，以赋予避免数据泄漏的多任务能力能力的能力。随着数据和模型大小的扩展，参数有效调整成为有效地获得特定于任务模型的常见实践。但是，我们观察到，在有效调整下设计旨在进行全面微调合并失败的现有方法。为了解决这些问题，我们分析了低排放分解，并揭示保持方向和补偿奇异值之间的差距对于有效的模型合并至关重要。因此，我们提出了Copa-Mersing，这是一种具有互补参数适应的无训练参数有效合并方法。具体而言，我们（1）从参数之间的关系中进行修剪参数和构造缩放系数，以补偿任务干扰的性能下降，并且（2）执行交叉任务归一化以增强看不见的任务概括。我们建立了一个由多种多模式任务组成的基准，我们在其上进行了实验，以证明我们方法的出色性能和概括性。其他研究和广泛的分析进一步展示了有效性。

### Shakti-VLMs: Scalable Vision-Language Models for Enterprise AI 
[[arxiv](https://arxiv.org/abs/2502.17092)] [[cool](https://papers.cool/arxiv/2502.17092)] [[pdf](https://arxiv.org/pdf/2502.17092)]
> **Authors**: Syed Abdul Gaffar Shakhadri,Kruthika KR,Kartik Basavaraj Angadi
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: shakti-vlms：企业AI的可扩展视觉语言模型
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们介绍了Shakti VLM，这是一个旨在解决多模式学习中数据效率挑战的1B和4B参数的视觉模型家族。尽管最近的VLM通过广泛的培训数据实现了强劲的绩效，但Shakti模型利用建筑创新来获得竞争成果，以减少令牌。关键进步包括QK范围的注意力稳定性，混合归一化技术和增强的位置编码。三阶段的培训策略进一步优化了学习效率。评估表明，在文档理解，视觉推理，OCR提取和一般的多模式推理方面，Shakti-Shakti-vlm-1b和Shakti-VLM-4B Excel在Excel中进行了Excel。我们的结果强调，可以通过模型设计和培训策略而不是纯粹的数据量来实现高性能，从而使Shakti成为企业规模多模式任务的有效解决方案。

### DUNIA: Pixel-Sized Embeddings via Cross-Modal Alignment for Earth Observation Applications 
[[arxiv](https://arxiv.org/abs/2502.17066)] [[cool](https://papers.cool/arxiv/2502.17066)] [[pdf](https://arxiv.org/pdf/2502.17066)]
> **Authors**: Ibrahim Fayad,Max Zimmer,Martin Schwartz,Philippe Ciais,Fabian Gieseke,Gabriel Belouze,Sarah Brood,Aurelien De Truchis,Alexandre d'Aspremont
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 26 pages, 8 figures
- **标题**: DUNIA：通过跨模式比对以地球观测应用的像素大小的嵌入
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 已经致力于调整自我监督的多模式学习来进行地球观察应用。但是，现有方法会产生粗糙的斑块大小嵌入，从而限制了它们的有效性和与LIDAR等其他模态的整合。为了缩小这一差距，我们提出了Dunia，这是一种通过图像和全波形激光雷达数据之间的跨模式对齐来学习像素大小的嵌入的方法。由于模型以对比方式进行了训练，因此可以在零拍设置中的各种环境监视任务的背景下直接利用嵌入。在我们的实验中，我们证明了嵌入对于七个此类任务的有效性（冠层高度映射，分数盖覆盖，土地覆盖映射，树种物种识别，植物区域指数，农作物类型分类和每个像素波形的垂直垂直结构映射）。结果表明，即使在低数据制度中，嵌入式以及零摄像分类器也通常超过了专业监督模型。在微调环境中，我们在六个任务中的五个任务中表现出强大的低弹能功能，其表现接近或更好。

### Exploring Causes and Mitigation of Hallucinations in Large Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.16842)] [[cool](https://papers.cool/arxiv/2502.16842)] [[pdf](https://arxiv.org/pdf/2502.16842)]
> **Authors**: Yaqi Sun,Kyohei Atarashi,Koh Takeuchi,Hisashi Kashima
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 19 pages, 6 figures
- **标题**: 在大型视觉语言模型中探索原因和缓解幻觉
- **领域**: 计算机视觉和模式识别
- **摘要**: 大型视觉模型（LVLM）将图像编码器与大语言模型（LLMS）集成在一起，以处理多模式输入并执行复杂的视觉任务。但是，它们通常通过描述不存在的对象或属性来产生幻觉，从而损害其可靠性。这项研究分析了图像字幕中的幻觉模式，表明并非生成过程中的所有令牌都受图像输入的影响，并且图像依赖性可以作为幻觉检测的有用信号。为了解决这个问题，我们开发了一条自动管道来识别幻觉对象，并使用并行推理中的隐藏表示形式训练令牌级别的分类器，而没有图像输入。利用此分类器，我们引入了一种解码策略，该策略可有效控制推理时图像字幕中的幻觉率。

### Diffusion Models for conditional MRI generation 
[[arxiv](https://arxiv.org/abs/2502.18620)] [[cool](https://papers.cool/arxiv/2502.18620)] [[pdf](https://arxiv.org/pdf/2502.18620)]
> **Authors**: Miguel Herencia García del Castillo,Ricardo Moya Garcia,Manuel Jesús Cerezo Mazón,Ekaitz Arriola Garcia,Pablo Menéndez Fernández-Miranda
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 有条件MRI的扩散模型
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 在本文中，我们提出了一个潜在的扩散模型（LDM），用于生成脑磁共振成像（MRI），根据病理学（健康，胶质母细胞瘤，硬化，痴呆症）和获取方式（T1W，T1CE，T1CE，T1CE，T2W，FLAIR，FLAIR，PD）来调节其生成。为了评估生成的图像的质量，采用了Fréchet成立距离（FID）和多尺度结构相似性指数（MS-SSSIM）指标。结果表明，该模型生成的图像具有类似于真实图像，从而保持视觉保真度和多样性之间的平衡。此外，该模型展示了外推能力，从而使训练数据中不存在的配置产生。结果证明了该模型在临床数据集中增加样品数量，平衡代表性不足的类别以及评估医学中的AI模型的潜力，这有助于开发放射学诊断工具而不损害患者隐私。

### OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference 
[[arxiv](https://arxiv.org/abs/2502.18411)] [[cool](https://papers.cool/arxiv/2502.18411)] [[pdf](https://arxiv.org/pdf/2502.18411)]
> **Authors**: Xiangyu Zhao,Shengyuan Ding,Zicheng Zhang,Haian Huang,Maosong Cao,Weiyun Wang,Jiaqi Wang,Xinyu Fang,Wenhai Wang,Guangtao Zhai,Haodong Duan,Hua Yang,Kai Chen
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: Omnialign-V：增强MLLM与人类偏爱的对齐
- **领域**: 计算机视觉和模式识别
- **摘要**: 开源多模式大型语言模型（MLLM）的最新进展主要集中在增强基础能力上，从而在人类的偏好一致性方面存在很大的差距。本文介绍了Omnialign-V，这是一个全面的数据集，该数据集的200K高质量培训样本具有各种图像，复杂的问题和各种响应格式，以改善MLLM与人类偏好的一致性。我们还提出了MM-Alignbench，这是一种专门旨在评估MLLM与人类价值的对齐的人类宣传的基准。实验结果表明，使用有监督的微调（SFT）或直接偏好优化（DPO）将MLLM与Omnialign-V进行芬太尼，在维持标准VQA基准上保持或增强性能，可显着增强人类偏好比对，从而保持其基本功能。我们的数据集，基准，代码和检查点已在https://github.com/phoenixz810/omnialign-v上发布。

### UASTrack: A Unified Adaptive Selection Framework with Modality-Customization in Single Object Tracking 
[[arxiv](https://arxiv.org/abs/2502.18220)] [[cool](https://papers.cool/arxiv/2502.18220)] [[pdf](https://arxiv.org/pdf/2502.18220)]
> **Authors**: He Wang,Tianyang Xu,Zhangyong Tang,Xiao-Jun Wu,Josef Kittler
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: UASTRACK：一个统一的自适应选择框架，在单个对象跟踪中具有模态定位
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式跟踪在单对象跟踪（SOT）中至关重要，因为不同的传感器类型为克服对象外观变化引起的挑战提供了独特的功能。但是，现有的统一RGB-X跟踪器（X表示深度，事件或热模式）要么依赖于单个RGB-X图像对的特定任务训练策略，要么无法解决现实世界应用中模态自适应感知的重要重要性。在这项工作中，我们提出了UASTRACK，这是一个统一的自适应选择框架，可促进模型和参数统一，以及各种多模式跟踪任务的自适应模态歧视。为了在关节RGB-X对中实现模态自适应感知，我们设计了能够识别模态标签的判别性自动选择器（DAS），从而区分了辅助方式的数据分布。此外，我们提出了一个针对潜在空间各种方式量身定制的任务的优化适配器（TCOA）。该策略有效地过滤了噪声冗余，并根据每种模式的特定特征来减轻背景干扰。在包括Lasher，GTOT，RGBT234，Visevent和Depthtrack在内的五个基准上进行的广泛比较，涵盖RGB-T，RGB-E和RGB-D跟踪场景，这表明我们仅通过引入1.87m和1.95g的其他培训参数来实现我们创新的方法，从而实现了比较性能。该代码将在https://github.com/wanghe/uastrack上找到。

### CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification 
[[arxiv](https://arxiv.org/abs/2502.18176)] [[cool](https://papers.cool/arxiv/2502.18176)] [[pdf](https://arxiv.org/pdf/2502.18176)]
> **Authors**: Mingkun Zhang,Keping Bi,Wei Chen,Jiafeng Guo,Xueqi Cheng
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: accepted by ICLR 2025
- **标题**: 剪辑：通过剪辑在潜在空间中纯化，以进行对抗稳健的零击分类
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在本文中，我们旨在构建一个对抗性稳健的零击图像分类器。我们将工作基于剪辑，这是一种视觉语言预训练的编码器模型，可以通过将图像与文本提示匹配``<class-name>>''''的图像来执行零弹片分类。纯化是我们选择的路径，因为它不需要对特定攻击类型的对抗训练，因此可以应对任何预测的攻击。然后，我们提出纯化风险，因为纯化的纯化过程的关节分布之间的KL差异是通过双向随机微分方程（SDES）通过双向随机差异方程（SDES）向良性样品添加扰动的攻击过程。最终的衍生结果激发了我们在剪辑的多模式潜在空间中探索纯化。我们为我们的剪辑方法提出了两个变体：剪贴画式 - 模拟图像的潜在载体与Dalle-2中的扩散模块的潜在矢量的可能性（建模剪贴画的潜在载体的生成过程），以及剪贴画的生成过程，将图像与图像和图像之间相似之处相似的可能性建模。据我们所知，剪贴画是多模式潜在空间中的第一种纯化方法，而夹子cos是第一种不基于生成模型的纯化方法，它显着提高了防御效率。我们在CIFAR-10，ImageNet和13个数据集上进行了广泛的实验，这些实验以前基于夹子的防御方法用于评估零弹药分类的鲁棒性。结果表明，夹子将SOTA鲁棒性提高了一个大幅度，例如，CIFAR10的固定性从39.7％降低到91.1％，ImageNet上的稳健性从59.6％升至72.6％，而先前SOTA的13个数据集的平均鲁棒性相对提高了108％。该代码可在https://github.com/tmlresearchgroup-cas/clipure上找到。

### LightFC-X: Lightweight Convolutional Tracker for RGB-X Tracking 
[[arxiv](https://arxiv.org/abs/2502.18143)] [[cool](https://papers.cool/arxiv/2502.18143)] [[pdf](https://arxiv.org/pdf/2502.18143)]
> **Authors**: Yunfeng Li,Bo Wang,Ye Li
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: LightFC-X：用于RGB-X跟踪的轻量级卷积跟踪器
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管在多模式跟踪方面取得了长足的进步，但对于资源受限的设备而言，这些跟踪器仍然太重且昂贵。为了减轻这个问题，我们提出了LightFC-X，这是一个轻巧的卷积RGB-X跟踪器，探索了轻巧的多模式跟踪的统一卷积体系结构。我们的核心思想是实现对多模式特征的轻质跨模式建模和关节改进以及目标的时空外观特征。具体而言，我们提出了一个新型有效的跨意义模块（ECAM）和一个新型时空模板聚集模块（StAM）。 ECAM仅具有208m参数的模板搜索区域集成特征的轻巧跨模式相互作用。 StAM通过模块微调范式增强了模型对时间信息的利用。全面的实验表明，我们的LightFC-X实现了最先进的性能以及参数，性能和速度之间的最佳平衡。例如，LightFC-T-ST在Lasher基准上的SR和PR的表现优于CMD 4.3％和5.7％，其参数降低了2.6倍，而2.7倍的速度降低了2.6倍。它以22 fps的速度实时在CPU上实时运行。该代码可在https://github.com/liyunfenglyf/lightfc-x上找到。

### PromptMID: Modal Invariant Descriptors Based on Diffusion and Vision Foundation Models for Optical-SAR Image Matching 
[[arxiv](https://arxiv.org/abs/2502.18104)] [[cool](https://papers.cool/arxiv/2502.18104)] [[pdf](https://arxiv.org/pdf/2502.18104)]
> **Authors**: Han Nie,Bin Luo,Jun Liu,Zhitao Fu,Huan Zhou,Shuo Zhang,Weixing Liu
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 15 pages, 8 figures
- **标题**: 提示：基于扩散和视觉基础模型的模态描述符
- **领域**: 计算机视觉和模式识别
- **摘要**: 图像匹配的理想目标是在看不见的域中实现稳定有效的性能。但是，许多现有的基于学习的光SAR图像匹配方法在特定方案中有效，但仍表现出有限的概括，并且难以适应实际应用。反复培训或微调匹配模型以解决域差异不仅不够优雅，而且还引入了其他计算开销和数据生产成本。近年来，一般基础模型显示出增强概括的巨大潜力。但是，自然和遥感图像之间的视觉域的差异为其直接应用带来了挑战。因此，有效利用基础模型来改善光 -  SAR图像匹配的概括仍然是挑战。为了应对上述挑战，我们提出了提示，这是一种新颖的方法，该方法使用基于土地使用分类的文本提示来构建模式不变的描述符，作为光学和SAR图像匹配的先验信息。提示提取物通过利用预训练的扩散模型和视觉基础模型（VFM）来多尺度的模态不变特征，而专门设计的特征聚合模块有效地构成了不同粒度范围内的功能。来自四个不同区域的光SAR图像数据集的广泛实验表明，促使敏捷的表现优于最先进的匹配方法，在可见和看不见的域中取得了卓越的结果，并且表现出强大的跨域泛化功能。源代码将公开提供https://github.com/hanniewhu/promptmid。

### Detecting Offensive Memes with Social Biases in Singapore Context Using Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.18101)] [[cool](https://papers.cool/arxiv/2502.18101)] [[pdf](https://arxiv.org/pdf/2502.18101)]
> **Authors**: Cao Yuxuan,Wu Jiayang,Alistair Cheong Liang Chuen,Bryan Shan Guanrong,Theodore Lee Chong Jen,Sherman Chann Zhi Shen
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: Accepted at 3rd Workshop on Cross-Cultural Considerations in NLP (C3NLP), co-located with NAACL 2025. This is an extended version with some appendix moved to the main body
- **标题**: 使用多模式模型检测新加坡环境中具有社会偏见的进攻模因
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 传统的在线内容审核系统努力对现代多模式的交流手段进行分类，例如模因，一种非常细微的和信息密集的媒介。在新加坡等文化多样化的社会中，使用低资源语言，需要在本地环境中进行广泛的知识来解释在线内容，这项任务尤其困难。我们策划了由GPT-4V标记的112K模因，用于微调VLM，以对新加坡环境中的进攻模因进行分类。我们在数据集上显示了微调VLM的有效性，并提出了一条包含OCR，翻译和70亿个参数级VLM的管道。我们的解决方案达到80.62％的精度和0.8192 AUROC，在持有的测试集中，可以极大地帮助人类调节在线内容。数据集，代码和模型权重已在https://github.com/aliencaocao/vlm-for-memes-aisg上开源。

### A Fusion Model for Artwork Identification Based on Convolutional Neural Networks and Transformers 
[[arxiv](https://arxiv.org/abs/2502.18083)] [[cool](https://papers.cool/arxiv/2502.18083)] [[pdf](https://arxiv.org/pdf/2502.18083)]
> **Authors**: Zhenyu Wang,Heng Song
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 基于卷积神经网络和变压器的艺术品识别的融合模型
- **领域**: 计算机视觉和模式识别
- **摘要**: 在文化遗产保护，艺术市场分析和历史研究等领域，艺术品的识别至关重要。随着深度学习的发展，卷积神经网络（CNN）和变压器模型已成为图像分类的关键工具。尽管CNN在本地功能提取方面表现出色，但它们在全球环境中挣扎，而变形金刚在捕获全球依赖性方面表现强大，但在细粒度的本地细节方面却很弱。为了应对这些挑战，本文提出了将CNN和变压器组合用于艺术品识别的融合模型。该模型首先使用CNN提取本地特征，然后使用变压器捕获全局上下文，然后使用特征融合机制来提高分类精度。有关中国和油画数据集的实验显示，融合模型的表现优于单个CNN和变压器模型，将分类精度分别提高了9.7％和7.1％，并将F1分数提高了0.06和0.05。结果证明了模型的有效性和未来改进的潜力，例如多模式集成和架构优化。

### Progressive Local Alignment for Medical Multimodal Pre-training 
[[arxiv](https://arxiv.org/abs/2502.18047)] [[cool](https://papers.cool/arxiv/2502.18047)] [[pdf](https://arxiv.org/pdf/2502.18047)]
> **Authors**: Huimin Yan,Xian Yang,Liang Bai,Jiye Liang
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 进行医学多模式预训练的逐步局部对齐
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 医学图像和文本之间的局部对齐对于准确的诊断至关重要，尽管由于没有天然局部配对以及刚性区域识别方法的局限性，它仍然具有挑战性。传统方法依赖于硬边界，这引起了不确定性，而医学成像则需要灵活的软区域识别来处理不规则的结构。为了克服这些挑战，我们提出了渐进式的本地对齐网络（PLAN），该网络设计了一种新型的基于对比的学习方法，以建立有意义的单词像素关系，并引入渐进的学习策略，以迭代地完善这些关系，增强对准精度和鲁棒性。通过结合这些技术，计划有效地改善了软区域的识别，同时抑制了噪声干扰。在多个医疗数据集上进行的广泛实验表明，计划超过短语接地，图像文本检索，对象检测和零照片分类中的最新方法，为医疗图像文本对齐设定了新的基准测试。

### VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion 
[[arxiv](https://arxiv.org/abs/2502.18042)] [[cool](https://papers.cool/arxiv/2502.18042)] [[pdf](https://arxiv.org/pdf/2502.18042)]
> **Authors**: Pei Liu,Haipeng Liu,Haichao Liu,Xin Liu,Jinxin Ni,Jun Ma
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: VLM-E2E：通过多模式驾驶员注意融合增强端到端自动驾驶
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 人类驾驶员通过利用丰富的注意语义来熟练地在复杂的场景中导航，但是当前的自主系统努力复制这种能力，因为他们在将2D观测转换为3D空间时通常会失去关键的语义信息。从这个意义上讲，它阻碍了他们在动态和复杂环境中的有效部署。利用视觉模型（VLMS）的优越场景的理解和推理能力，我们提出了VLM-E2E，这是一个新颖的框架，该框架使用VLMS通过提供注意力提示来增强训练。我们的方法将文本表示形式集成到语义监督的Bird's-eye-View（BEV）功能中，这使该模型能够学习更丰富的功能表示，以明确捕获驾驶员的注意语义。通过专注于注意语义，VLM-E2E可以更好地与类似人类的驾驶行为保持一致，这对于导航动态和复杂环境至关重要。此外，我们引入了BEV文本可学习的加权融合策略，以解决融合多模式信息的模式重要性不平衡问题。这种方法动态平衡了BEV和文本功能的贡献，以确保有效地利用了来自视觉和文本方式的互补信息。通过明确解决多模式融合中的不平衡，我们的方法促进了对驾驶环境的更全面和强大的表示。我们在Nuscenes数据集上评估了VLM-E2E，并证明了其优于最先进的方法，并展示了性能的显着改善。

### ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents 
[[arxiv](https://arxiv.org/abs/2502.18017)] [[cool](https://papers.cool/arxiv/2502.18017)] [[pdf](https://arxiv.org/pdf/2502.18017)]
> **Authors**: Qiuchen Wang,Ruixue Ding,Zehui Chen,Weiqi Wu,Shihang Wang,Pengjun Xie,Feng Zhao
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: vidorag：通过动态迭代推理剂的视觉文档检索启动生成
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,信息检索
- **摘要**: 从视觉上富裕的文档中了解信息仍然是传统检索型生成（RAG）方法的重大挑战。现有基准主要集中在基于图像的问题答案（QA）上，忽视了密集的视觉文档中有效检索，理解和推理的基本挑战。为了弥合这一差距，我们介绍了Vidoseek，这是一个新颖的数据集，旨在评估需要复杂推理的视觉丰富文档的抹布性能。基于它，我们确定了当前抹布方法中的关键局限性：（i）纯粹的视觉检索方法难以有效地整合文本和视觉特征，以及（ii）以前的方法通常分配不足的推理令牌，从而限制了它们的有效性。为了应对这些挑战，我们提出了Vidorag，这是一个新型的多代理RAG框架，该框架量身定制了跨视觉文档的复杂推理。 Vidorag采用高斯混合模型（GMM）的混合策略来有效处理多模式检索。为了进一步引起该模型的推理能力，我们引入了一个迭代代理工作流，其中包含了探索，摘要和反思，提供了一个框架，用于研究RAG域中的测试时间缩放。关于Vidoseek的广泛实验验证了我们方法的有效性和概括。值得注意的是，Vidorag在竞争性Vidoseek基准上优于现有方法超过10％。

### UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting 
[[arxiv](https://arxiv.org/abs/2502.17860)] [[cool](https://papers.cool/arxiv/2502.17860)] [[pdf](https://arxiv.org/pdf/2502.17860)]
> **Authors**: Haoyuan Li,Yanpeng Zhou,Tao Tang,Jifei Song,Yihan Zeng,Michael Kampffmeyer,Hang Xu,Xiaodan Liang
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: ICLR 2025; Corrected citation of Uni3D;
- **标题**: Unigs：统一的语言图像-3D预处理高斯裂开
- **领域**: 计算机视觉和模式识别
- **摘要**: 多模式3D预训练方法的最新进展显示出在学习文本，图像和点云的联合表示方面有希望的功效。但是，作为3D表示，采用点云无法完全捕获3D世界的复杂性，并在离散点和图像的密集2D像素之间表现出明显的差距。为了解决这个问题，我们提出了单格，将3D高斯（3DG）集成到多模式的预训练中以增强3D表示。我们首先依靠3DGS表示将3D世界建模为具有颜色和不透明度的3D高斯人的集合，并结合了3D场景的所有信息，同时与2D图像建立了牢固的联系。然后，为了实现与语言图像3D相关的语言图像，Unig从预先训练的视觉语言模型开始，以通过广泛的现实世界图像文本对建立共享的视觉和文本空间。随后，Unigs使用3D编码器将优化的3DG与语言图像表示一致，以学习统一的多模式表示。为了促进3D编码器提取全球显式3D特征并实现更好的跨模式对齐，我们还引入了一种新颖的高斯感知指导模块，该模块指导3D域的细粒度表示。通过对OBJAVERSE，ABO，MVIMGNET和SUN RGBD数据集进行零拍，文本驱动的检索和开放世界理解任务的广泛实验，我们证明了Unig在学习更一般和更强大的更坚固和更强的一致性多模式表示方面的有效性。具体而言，Unigs在不同的3D任务中取得了领先的结果，比以前的SOTA，UNI3D取得了显着改进，包括零摄像分类（+9.36％），文本驱动的检索（+4.3％）和开放世界的理解（+7.92％）。

### M-LLM Based Video Frame Selection for Efficient Video Understanding 
[[arxiv](https://arxiv.org/abs/2502.19680)] [[cool](https://papers.cool/arxiv/2502.19680)] [[pdf](https://arxiv.org/pdf/2502.19680)]
> **Authors**: Kai Hu,Feng Gao,Xiaohan Nie,Peng Zhou,Son Tran,Tal Neiman,Lingyun Wang,Mubarak Shah,Raffay Hamid,Bing Yin,Trishul Chilimbi
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 基于M-LLM的视频框架选择，以进行有效的视频理解
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式大语言模型（M-LLS）的最新进展在视频推理中显示出令人鼓舞的结果。流行的多模式大型语言模型（M-LLM）框架通常应用幼稚的统一抽样，以减少被送入M-LLM的视频帧数量，尤其是对于长上下文视频。但是，在视频的某些时期，它可能会失去关键环境，因此下游M-LLM可能没有足够的视觉信息来回答问题。为了攻击这个疼痛点，我们提出了一种基于M-LLM的轻质框架选择方法，该方法适应性地选择了与用户查询更相关的帧。为了训练提出的框架选择器，我们引入了两个监督信号（i）空间信号，其中单帧的重要性得分通过提示m-llm； （ii）时间信号，其中通过所有框架候选者的标题提示大型语言模型（LLM）来选择多个帧。然后，通过冷冻的下游视频M-LLM来消化所选的帧，以进行视觉推理和问答。经验结果表明，所提出的M-LLM视频框架选择器改善了跨介质（ActivityNet，Next-QA）和Long（Egoschema，longVideObench）上下文视频的各种下游视频大型语言模型（视频-LLM）的表演。

### MICINet: Multi-Level Inter-Class Confusing Information Removal for Reliable Multimodal Classification 
[[arxiv](https://arxiv.org/abs/2502.19674)] [[cool](https://papers.cool/arxiv/2502.19674)] [[pdf](https://arxiv.org/pdf/2502.19674)]
> **Authors**: Tong Zhang,Shu Shen,C. L. Philip Chen
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 12 pages, 7 figures
- **标题**: MICINET：多层间令人困惑的信息删除，以进行可靠的多模式分类
- **领域**: 计算机视觉和模式识别
- **摘要**: 在存在嘈杂数据的情况下，可靠的多模式学习是一个广泛关注的问题，尤其是在关键安全应用中。许多可靠的多模式方法研究了解决模式特异性或跨模式噪声。但是，他们无法有效处理两种类型的噪声的共存。此外，缺乏对全球和个人级别噪声的全面考虑限制了其可靠性。为了解决这些问题，提出了一种可靠的多模式分类方法，称为多层阶层间令人困惑的信息删除网络（MICINET）。 Micinet通过将两种类型的噪声统一到阶层间令人困惑的信息（\ textit {ici}）的概念中来实现可靠的去除，并在全球和个体级别上都消除了噪声。具体而言，MiCinet首先可靠地通过所提出的\ textbf {\ textit {global \ textbf {ici}学习模块}}来可靠地学习全局\ textit {ici}分布。然后，它将\ textbf {\ textIt {全局引导的样本ICI学习模块}}从示例功能中有效地删除全局级别\ textit {ici}，利用示例功能，利用学习的global \ textit {ici {ici}分布。 Subsequently, the \textbf{\textit{Sample-adaptive Cross-modality Information Compensation module}} is designed to remove individual-level \textit{ICI} from each sample reliably.这是通过基于歧视特征与\ textit {ici}之间的互补关系以及对相对歧视能力引入的模态相对质量的感知来实现的。在四个数据集上的实验表明，在各种噪声条件下，MICINET优于其他最先进的可靠多模式分类方法。

### Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack 
[[arxiv](https://arxiv.org/abs/2502.19672)] [[cool](https://papers.cool/arxiv/2502.19672)] [[pdf](https://arxiv.org/pdf/2502.19672)]
> **Authors**: Chenhe Gu,Jindong Gu,Andong Hua,Yao Qin
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: arXiv admin note: text overlap with arXiv:2403.09766
- **标题**: 通过动态视觉语言对准攻击提高MLLM中的对抗性可传递性
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 基于LLM的多模式大语言模型（MLLM）最近因其在图像识别和理解方面的能力而引起了人们的关注。但是，尽管MLLM容易受到对抗性攻击的影响，但这些攻击在不同模型中的转移性仍然有限，尤其是在目标攻击设置下。现有方法主要集中于特定视觉的扰动，但与视觉方式一致性的复杂性质斗争。在这项工作中，我们介绍了动态视觉语言对准（DynVLA）攻击，这是一种新颖的方法，将动态扰动注入视觉语言连接器中，以增强不同模型的各种视觉平行的概括。我们的实验结果表明，Dynvla显着提高了对抗性示例在各种MLLM中的转移性，包括Blip2，TenderchBlip，Minigpt4，Llava，Llava和Gemini等封闭式模型。

### Ev-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras 
[[arxiv](https://arxiv.org/abs/2502.19630)] [[cool](https://papers.cool/arxiv/2502.19630)] [[pdf](https://arxiv.org/pdf/2502.19630)]
> **Authors**: Hoonhee Cho,Jae-young Kang,Youngho Kim,Kuk-Jin Yoon
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Accepted by CVPR2025
- **标题**: EV-3DOD：使用事件摄像机推动3D对象检测的时间边界
- **领域**: 计算机视觉和模式识别
- **摘要**: 检测点云中的3D对象在自主驾驶系统中起着至关重要的作用。最近，结合相机信息的高级多模式方法已取得了显着的性能。对于安全有效的自主驾驶系统，不仅在准确性，而且速度和潜伏期低的算法都是必不可少的。但是，由于固定帧速率传感器的延迟和带宽的限制，例如LIDAR和相机，现有算法无法满足这些要求。为了解决此限制，我们首次将异步事件摄像机引入3D对象检测中。我们利用它们的高时间分辨率和低带宽来实现高速3D对象检测。我们的方法通过通过事件摄像机检索以前的3D信息，即使在同步数据不可用时也可以在框架间间隔内进行检测。此外，我们介绍了第一个基于事件的3D对象检测数据集DSEC-3DOD，其中包括100 fps的接地图3D边界框，为基于事件的3D检测器建立了第一个基准。代码和数据集可在https://github.com/mickeykang16/ev3dod上获得。

### CLIP-Optimized Multimodal Image Enhancement via ISP-CNN Fusion for Coal Mine IoVT under Uneven Illumination 
[[arxiv](https://arxiv.org/abs/2502.19450)] [[cool](https://papers.cool/arxiv/2502.19450)] [[pdf](https://arxiv.org/pdf/2502.19450)]
> **Authors**: Shuai Wang,Shihao Zhang,Jiaqi Wu,Zijian Tian,Wei Chen,Tongzhu Jin,Miaomiao Xue,Zehua Wang,Fei Richard Yu,Victor C. M. Leung
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 通过ISP-CNN Fusion在不均匀照明下通过ISP-CNN融合来增强夹夹的多模式图像
- **领域**: 计算机视觉和模式识别
- **摘要**: 清晰的监视图像对于煤矿视频事物（IOVT）系统的安全操作至关重要。但是，在地下环境中，低照明和不均匀的亮度显着降低了图像质量，对通常依赖于难以实现的配对参考图像的增强方法提出了挑战。此外，在IOVT系统内的边缘设备上的增强性能与计算效率之间存在权衡。为了解决这些问题，我们提出了一种针对煤矿IOVT量身定制的多模式图像增强方法，该方法利用ISP-CNN融合体系结构优化了用于不均匀照明的ISP-CNN融合体系结构。这种两阶段的策略结合了全球增强和细节优化，有效地提高了图像质量，尤其是在光线不足的领域。基于夹的多模式迭代优化可以无监督的增强算法训练。 By integrating traditional image signal processing (ISP) with convolutional neural networks (CNN), our approach reduces computational complexity while maintaining high performance, making it suitable for real-time deployment on edge devices.Experimental results demonstrate that our method effectively mitigates uneven brightness and enhances key image quality metrics, with PSNR improvements of 2.9%-4.9%, SSIM by 4.3%-11.4%, and VIF by 4.9％-17.8％，而七种最先进的算法。模拟煤矿监控方案验证了我们方法平衡性能和计算需求的能力，促进实时增强和支持更安全的采矿操作。

### ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.19409)] [[cool](https://papers.cool/arxiv/2502.19409)] [[pdf](https://arxiv.org/pdf/2502.19409)]
> **Authors**: Danae Sánchez Villegas,Ingo Ziegler,Desmond Elliott
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Code, dataset, and checkpoints are publicly available at https://github.com/danaesavi/ImageChain
- **标题**: ImageChain：在多模式大语言模型中推进顺序图像到文本推理
- **领域**: 计算机视觉和模式识别,计算语言学,机器学习
- **摘要**: 对图像序列的推理仍然是多模式大语言模型（MLLM）的挑战。尽管最近的模型在预训练期间包含了多图像数据，但他们仍然很难识别顺序结构，通常会独立处理图像。这项工作介绍了ImageChain，该框架通过将视觉序列作为多转向对话建模，从而增强了MLLM的顺序推理能力，以超过图像数据。在ImageChain中，图像与相应的文本描述交织在一起，以形成一个受控的对话，该对话明确捕获了时间依赖性和叙事进展。我们的方法优化了隔壁描述的任务，其中该模型基于前面的视觉和文本提示生成了即将到来的场景的上下文感知描述。我们证明我们的方法提高了下一场景描述任务的性能 - 平均将Simmrate的平均提高从3.7％提高到19％，该指标量化了与人类宣布的地面真理的语义相似性。此外，ImageChain在从漫画到机器人技术等的应用中实现了强劲的零射击外域性能。广泛的实验验证了多模式多圈对话设计中的指导调节是弥合静态图像理解与时间感知推理之间差距的关键。

### Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions 
[[arxiv](https://arxiv.org/abs/2502.19293)] [[cool](https://papers.cool/arxiv/2502.19293)] [[pdf](https://arxiv.org/pdf/2502.19293)]
> **Authors**: Ruben T. Lucassen,Sander P. J. Moonemans,Tijn van de Luijtgaarden,Gerben E. Breimer,Willeke A. M. Blokx,Mitko Veta
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 11 pages, 2 figures. arXiv admin note: text overlap with arXiv:2502.19285
- **标题**: 皮肤黑素细胞病变的病理报告生成和多模式表示学习
- **领域**: 计算机视觉和模式识别
- **摘要**: 每年病理学家检查数百万个黑素细胞皮肤病变，其中大多数涉及常见的NEVI（即普通痣）。尽管这些病变中的大多数可以在几秒钟内诊断，但编写相应的病理报告耗时要耗时。因此，报告写作的一部分可以减轻病理学家的工作量不断增加。在这项工作中，我们开发了专门针对皮肤黑素细胞病变病理结构域的视觉模型。该模型遵循对比字幕框架，并使用42,512 H＆e染色的整个幻灯片图像和19,645个相应病理学报告的黑素细胞病变数据集进行了训练和评估。我们的结果表明，模型生成的报告的质量得分与一名专家病理学家在读者研究中评估的常见NEVI的病理学家所写的报告相当。虽然报告的产生显示稀有黑素细胞病变亚型更困难，但这些病例的跨模式检索性能要好得多。

### On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation 
[[arxiv](https://arxiv.org/abs/2502.19285)] [[cool](https://papers.cool/arxiv/2502.19285)] [[pdf](https://arxiv.org/pdf/2502.19285)]
> **Authors**: Ruben T. Lucassen,Tijn van de Luijtgaarden,Sander P. J. Moonemans,Gerben E. Breimer,Willeke A. M. Blokx,Mitko Veta
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 11 pages, 1 figure
- **标题**: 关于文本预处理对多模式表示学习和病理报告生成的重要性
- **领域**: 计算机视觉和模式识别
- **摘要**: 病理学中的视觉模型可实现多模式病例检索和自动报告的生成。然而，到目前为止，许多模型都已在病理报告上进行了培训，这些报告包括无法从配对的整个幻灯片图像（例如患者历史记录）中推断出的信息，这可能会导致生成报告中的幻觉句子。为此，我们研究了视觉模型的病理报告中信息的选择如何影响多模式表示和生成报告的质量。更具体地说，我们比较了一个在完整报告中训练的模型与经过预处理报告训练的模型，该模型仅包括根据H＆E染色幻灯片描述细胞和组织外观的句子。在实验中，我们建立在BLIP-2框架上，并使用了42,433 H＆e染色的整个幻灯片图像和19,636个相应病理学报告的皮肤黑素细胞病变数据集。使用图像到文本和文本对图像检索评估模型性能，并对专家病理学家对生成的报告进行定性评估。我们的结果表明，文本预处理可阻止报告生成中的幻觉。尽管生成的报告的质量有所提高，但在完整报告中培训视觉模型表现出更好的跨模式检索性能。

### Neural Antidote: Class-Wise Prompt Tuning for Purifying Backdoors in Pre-trained Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.19269)] [[cool](https://papers.cool/arxiv/2502.19269)] [[pdf](https://arxiv.org/pdf/2502.19269)]
> **Authors**: Jiawei Kong,Hao Fang,Sihang Guo,Chenxi Qing,Bin Chen,Bin Wang,Shu-Tao Xia
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 神经解毒剂：课堂及时调整，以在预训练的视觉模型中净化后门
- **领域**: 计算机视觉和模式识别
- **摘要**: 虽然预训练的视觉模型（VLM）（例如剪辑）具有多模式数据的出色代表性能力，但最近的研究表明它们容易受到后门攻击的影响。为了减轻威胁，现有的防御策略主要集中于对整个可疑模型进行微调，但仅提供对最先进攻击的边际阻力，并且通常会导致清洁准确性降低，尤其是在数据限制的情况下。它们的故障可能归因于VLM中的微调数据和大量参数之间的不匹配。为了应对这一挑战，我们建议通过课堂后门提示（CBPT）防御，这是一种有效的方法，可在文本上运行，提示间接净化中毒的VLM。具体来说，我们首先通过精心制作的积极和负面样本采用先进的对比度学习，以有效地颠倒攻击者潜在采用的后门触发器。建立虚拟触发器后，我们将利用有效的及时调整技术来优化这些类别的文本提示，以修改模型的决策边界，以进一步重新分类后门触发器的特征区域。广泛的实验表明，CBPT可以大大减轻后门威胁，同时保存模型实用程序，例如在七个主流后门攻击中，平均清洁准确性（CA）为58.86 \％，攻击成功率（ASR）为0.39 \％。这些结果强调了我们及时净化设计的优势，可以增强模型鲁棒性，以防止后门攻击。

### ProxyTransformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding 
[[arxiv](https://arxiv.org/abs/2502.19247)] [[cool](https://papers.cool/arxiv/2502.19247)] [[pdf](https://arxiv.org/pdf/2502.19247)]
> **Authors**: Qihang Peng,Henry Zheng,Gao Huang
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 12 pages, 3 figures. Accepted by CVPR2025
- **标题**: 偏置形式：前塑形点云流形，具有3D视觉接地的替代关注
- **领域**: 计算机视觉和模式识别
- **摘要**: 体现的智能要求代理根据语言说明实时与3D环境进行交互。该领域的基本任务是以自我为中心的3D视觉接地。但是，从RGB-D图像呈现的点云保留了大量的冗余背景数据和固有的噪声，这两者都可以干扰目标区域的歧管结构。现有的点云增强方法通常需要一个乏味的过程来改善流形，这不适合实时任务。我们提出适用于多模式任务的代理转换，以有效地改善点云流形。我们的方法首先利用可变形点聚类来识别目标区域中的点云子序列。然后，我们提出了一个使用多模式代理来指导点云转换的代理注意模块。基于代理人的关注，我们设计了一个子手机转换生成模块，其中文本信息全球指导了不同子手机的翻译向量，从而优化了目标区域的相对空间关系。同时，图像信息指导每个子手法中的线性变换，从而完善了目标区域的局部点云流形。广泛的实验表明，代理转换显着胜过所有现有方法，在易于目标方面取得了令人印象深刻的提高7.49％，而硬目标的4.60％，同时将注意力阻滞的计算开销降低了40.6％。这些结果在以自我为中心的3D视觉接地中建立了新的SOTA，展示了我们方法的有效性和鲁棒性。

### A Dual-Purpose Framework for Backdoor Defense and Backdoor Amplification in Diffusion Models 
[[arxiv](https://arxiv.org/abs/2502.19047)] [[cool](https://papers.cool/arxiv/2502.19047)] [[pdf](https://arxiv.org/pdf/2502.19047)]
> **Authors**: Vu Tuan Truong,Long Bao Le
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 扩散模型中用于后门防御和后门扩增的双重用途框架
- **领域**: 计算机视觉和模式识别
- **摘要**: 扩散模型已成为最先进的生成框架，在生产高质量的多模式样本方面表现出色。但是，最近的研究揭示了它们对后门攻击的脆弱性，当将预定义的触发器嵌入到其输入中时，后门模型会产生特定的，不受欢迎的输出称为后门目标（例如，有害图像）。在本文中，我们提出了纯净的填充，这是一个双重用途框架，同时扮演着两个对比角色：后门防御和后门攻击放大。为了防御，我们引入了两个新型损失功能，以倒入嵌入扩散模型中的后门触发器。第一个利用触发诱导的分布在扩散过程的多个时间步上移动，而第二个则在激活后门时利用了去索的一致性效果。一旦实现了准确的触发反转，我们就开发了一种后门检测方法，该方法分析了倒置触发器和生成的后门目标，以识别后门攻击。就攻击者的角色而言，我们描述了如何使用触发反转算法来加强嵌入在后置扩散模型中的原始触发器。这大大提高了攻击性能，同时减少了所需的后门训练时间。实验结果表明，纯化源达到接近完美的检测准确性，超过了现有的防御能力，尤其是针对复杂的触发模式。此外，在攻击情况下，我们的攻击放大方法将现有后门攻击的攻击成功率（ASR）提高到近100 \％，同时将训练时间降低到20倍。

### Towards General Visual-Linguistic Face Forgery Detection(V2) 
[[arxiv](https://arxiv.org/abs/2502.20698)] [[cool](https://papers.cool/arxiv/2502.20698)] [[pdf](https://arxiv.org/pdf/2502.20698)]
> **Authors**: Ke Sun,Shen Chen,Taiping Yao,Ziyin Zhou,Jiayi Ji,Xiaoshuai Sun,Chia-Wen Lin,Rongrong Ji
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 8 pages, 5 figures, Accpet by CVPR2025
- **标题**: 迈向一般的视觉语言伪造检测（V2）
- **领域**: 计算机视觉和模式识别
- **摘要**: 面部操纵技术已取得了重大进步，对安全和社会信任提出了严重的挑战。最近的著作表明，利用多模式模型可以增强面部伪造检测的概括和解释性。但是，现有的注释方法，无论是通过人类标签还是直接多模式大语模型（MLLM）产生，通常会遭受幻觉问题的困扰，导致文本描述不准确，尤其是对于高质量的伪造。为了解决这个问题，我们提出了面部伪造文本生成器（FFTG），这是一种新颖的注释管道，通过利用伪造掩码的初始区域和类型识别来生成准确的文本描述，然后采取全面的提示策略来指导MLLMS减少幻觉。我们通过通过三个分支机构培训框架进行微调来验证我们的方法，从而结合了单峰和多模式目标，并将MLLM与我们的结构化注释相结合。实验结果表明，我们的方法不仅具有较高的区域识别精度的更准确的注释，而且还可以改善各种伪造检测基准的模型性能。我们的代码可在https://github.com/skjack/vlffd.git中找到。

### Interpreting CLIP with Hierarchical Sparse Autoencoders 
[[arxiv](https://arxiv.org/abs/2502.20578)] [[cool](https://papers.cool/arxiv/2502.20578)] [[pdf](https://arxiv.org/pdf/2502.20578)]
> **Authors**: Vladimir Zaigrajew,Hubert Baniecki,Przemyslaw Biecek
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 用分层稀疏自动编码器解释剪辑
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 稀疏的自动编码器（SAE）可用于在神经网络中检测和转向可解释的特征，具有理解复杂的多模式表示的潜力。鉴于它们能够发现可解释的功能，SAE对于分析大规模视觉模型（例如剪辑和siglip）特别有价值，这些模型是现代系统中的基本构建基础，但仍具有挑战性的解释和控制。但是，当前的SAE方法通过同时优化重建质量和稀疏性而受到限制，因为它们依赖于激活抑制或刚性稀疏性约束。为此，我们介绍了Matryoshka Sae（MSAE），这是一种新的体系结构，同时以多种粒度学习层次结构表示，从而可以直接优化两个指标而没有妥协。 MSAE在重建质量和稀疏性之间建立了新的最新帕累托前沿，达到0.99余弦相似性，而差异的差异小于0.1个方差的比例小于0.1，同时保持〜80％的稀疏性。最后，我们通过从其表示中提取超过120个语义概念来表明MSAE作为解释和控制剪辑的工具，以在Celeba等下游任务中执行基于概念的相似性搜索和偏见分析。

### Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection 
[[arxiv](https://arxiv.org/abs/2502.20573)] [[cool](https://papers.cool/arxiv/2502.20573)] [[pdf](https://arxiv.org/pdf/2502.20573)]
> **Authors**: Sari Masri,Huthaifa I. Ashqar,Mohammed Elhenawy
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 城市交叉点的视觉推理：用于交通冲突检测的GPT-4O的鉴定
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 由于复杂性，频繁的冲突和盲点，未信号的城市交汇处中的交通控制提出了重大挑战。这项研究探讨了利用多模式大型语言模型（MLLM）（例如GPT-4O）的能力，通过直接使用四足相交的鸟类视频视频来提供逻辑和视觉推理。在这种提出的方​​法中，GPT-4O充当智能系统，可检测冲突并为驾驶员提供解释和建议。微调模型的准确度为77.14％，而对微型GPT-4O的真实预测值的手动评估显示，模型生成的解释的精确度为89.9％，对于推荐的下一步动作，对模型生成的解释的精度为92.3％。这些结果强调了使用视频作为输入进行实时流量管理的可行性，从而为交叉点流量管理和操作提供了可扩展且可操作的见解。本研究中使用的代码可在https://github.com/sarimasri3/traffic-intersection-conflict-detection-using-images.git上获得。

### VideoA11y: Method and Dataset for Accessible Video Description 
[[arxiv](https://arxiv.org/abs/2502.20480)] [[cool](https://papers.cool/arxiv/2502.20480)] [[pdf](https://arxiv.org/pdf/2502.20480)]
> **Authors**: Chaoyu Li,Sid Padmanabhuni,Maryam Cheema,Hasti Seifi,Pooyan Fazli
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: ACM CHI 2025
- **标题**: Videoa11y：用于可访问的视频说明的方法和数据集
- **领域**: 计算机视觉和模式识别,人机交互
- **摘要**: 视频描述对于盲目和低视力（BLV）用户访问视觉内容至关重要。但是，由于培训数据集中人类注释质量的限制，当前用于生成描述的人工智能模型通常会缺乏，从而导致描述无法完全满足BLV用户的需求。为了解决这一差距，我们介绍了VideoA11Y，该方法利用多模式模型（MLLM）和视频可访问性指南来生成针对BLV个体量身定制的描述。使用此方法，我们策划了VideoA11Y-40K，这是针对BLV用户描述的40,000个视频的最大，最全面的数据集。在15个视频类别中进行了严格的实验，其中涉及347名观察参与者，40名BLV参与者和7个专业描述者，表明VideoA11Y描述的表现优于新手人体注释，并且与训练有素的人类注释有关，以清晰，准确性，客观性，描述性和用户满意度具有训练的人类注释。我们使用标准标准和自定义指标评估了VideoA11Y-40K上的模型，表明该数据集上的MLLMS微调可产生高质量的可访问描述。代码和数据集可在https://people-robots.github.io/videoa11y上找到。

### M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging 
[[arxiv](https://arxiv.org/abs/2502.20301)] [[cool](https://papers.cool/arxiv/2502.20301)] [[pdf](https://arxiv.org/pdf/2502.20301)]
> **Authors**: Jinghao Feng,Qiaoyu Zheng,Chaoyi Wu,Ziheng Zhao,Ya Zhang,Yanfeng Wang,Weidi Xie
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 38 pages, 7 figures
- **标题**: M^3Builder：一种用于医学成像中自动化机器学习的多代理系统
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 代理AI系统因其自主执行复杂任务的能力而引起了极大的关注。但是，他们依赖精心准备的工具限制了其在医疗领域的适用性，这需要培训专业模型。在本文中，我们做出了三个贡献：（i）我们提出了M3Builder，这是一种旨在在医学成像中自动化机器学习（ML）的新型多机构系统。 M3Builder以其核心采用了四个专门代理商，可以合作解决复杂的多步骤医疗ML工作流程，从自动数据处理和环境配置到独立的自动调试和模型培训。这些代理在医学成像ML工作区中运行，这是一个结构化的环境，旨在为代理提供数据集，培训代码和交互工具的自由文本描述，从而实现无缝通信和任务执行。 （ii）为了评估自动化成像ML的进度，我们提出了M3Bench，这是一种基准，其中包括14个培训数据集的四个一般任务，涉及五种解剖和三种成像方式，涵盖了2D和3D数据。 （iii）我们尝试七个最先进的大语言模型，这些模型是我们系统的代理核心，例如Claude系列，GPT-4O和DeepSeek-V3。与现有的ML代理设计相比，M3Builder在完成医学成像中的ML任务方面表现出了卓越的性能，使用Claude-3.7-Sonnet作为代理核心达到94.29％的成功率，在医疗成像中显示出巨大的自动化机器学习潜力。

### Explainable, Multi-modal Wound Infection Classification from Images Augmented with Generated Captions 
[[arxiv](https://arxiv.org/abs/2502.20277)] [[cool](https://papers.cool/arxiv/2502.20277)] [[pdf](https://arxiv.org/pdf/2502.20277)]
> **Authors**: Palawat Busaranuvong,Emmanuel Agu,Reza Saadati Fard,Deepak Kumar,Shefalika Gautam,Bengisu Tulu,Diane Strong
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 可解释的，可解释的多模式伤口感染分类，该分类来自带有生成标题的图像
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 糖尿病足溃疡（DFU）的感染会引起严重的并发症，包括组织死亡和肢体截肢，突出了需要准确，及时诊断的需求。以前的机器学习方法已重点是通过单独分析伤口图像来识别感染，而无需使用其他元数据，例如医疗笔记。在这项研究中，我们旨在通过引入合成标题增强检索伤口感染检测（Scarwid）来改善感染检测，这是一个新型的深度学习框架，利用合成的文本描述来增强DFU图像。 Scarwid由两个组成部分组成：（1）伤口薄片，一种在GPT-4O生成的描述上微调的视觉模型（VLM），以合成图像中一致的字幕； （2）使用交叉注意的图像文本融合模块从图像及其相应的伤口抹布标题中提取交叉模式嵌入。感染状态是通过从标记的支持集中检索TOP-K相似项目来确定的。为了增强训练数据的多样性，我们使用了潜在扩散模型来生成其他伤口图像。结果，对于伤口感染分类，Scarwid的表现优于最先进的模型，分别达到平均灵敏度，特异性和0.85、0.78和0.81。显示生成的字幕与伤口图像和感染检测结果一起增强了可解释性和信任，使护士能够使Scarwid的输出与其医学知识相结合。当不可用或协助新手护士时，他们可能很难识别伤口感染的视觉属性时，这尤其有价值。

### Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think 
[[arxiv](https://arxiv.org/abs/2502.20172)] [[cool](https://papers.cool/arxiv/2502.20172)] [[pdf](https://arxiv.org/pdf/2502.20172)]
> **Authors**: Liang Chen,Shuai Bai,Wenhao Chai,Weichu Xie,Haozhe Zhao,Leon Vinci,Junyang Lin,Baobao Chang
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 13 pages, 9 figures, codebase in https://github.com/chenllliang/DreamEngine
- **标题**: 图像生成的多模式表示对准：文本图像交织的控制比您想象的要容易
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 高级文本到图像生成的领域正在见证将功能强大的文本编码器（例如夹子和T5）与扩散变压器骨架相结合的统一框架的出现。尽管已经努力控制具有其他条件的输出图像，例如Canny和Depth Map，但仍缺乏用于任意文本图像交错控制的综合框架。当试图从生成过程中的多个图像中合并概念或视觉元素时，这一差距尤为明显。为了减轻差距，我们进行了初步实验，表明大型多模型（LMMS）提供了一个有效的共享表示空间，可以很好地对图像和文本进行良好的一致性，以作为外部扩散模型的条件。基于这一发现，我们提出了Dream Engine，这是一个高效且统一的框架，旨在在图像生成模型中进行任意文本图像交错控制。在强大的文本到图像模型（如SD3.5）的基础上，我们通过合并多种模式信息编码器（例如qwenvl）来替换原始的仅文本编码器。我们的方法利用了两个阶段的训练范式，包括联合文本图像对齐和多模式交织教学调整。我们的实验表明，这种训练方法是有效的，在Geneval基准上取得了0.69的总分，并匹配了最先进的文本对图像模型（如SD3.5和Flux）的性能。

### Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion 
[[arxiv](https://arxiv.org/abs/2502.20120)] [[cool](https://papers.cool/arxiv/2502.20120)] [[pdf](https://arxiv.org/pdf/2502.20120)]
> **Authors**: QingYuan Jiang,Longfei Huang,Yang Yang
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 从缓解分类能力的角度重新思考多模式学习
- **领域**: 计算机视觉和模式识别
- **摘要**: 尽管多模式学习〜（MML）取得了显着的进步，但模态失衡的存在阻碍了多模式学习，从而无法在实践中实现其比单峰模型的预期优势。为了克服这个问题，主流多模式学习方法更加强调平衡学习过程。但是，这些方法并不能明确提高弱模态的分类能力，从而导致绩效促进有限。通过设计持续的增强算法，我们提出了一种新型的多模式学习方法，以动态平衡弱和强态的分类能力。具体而言，我们首先使用设计的可配置分类器模块同时优化分类和残差错误，从而在多模式学习中提出了一种持续的增强算法。然后，我们提出了一种自适应分类器分配策略，以动态地促进弱模态的分类性能。为此，强度和弱模态的分类能力预计将保持平衡，从而减轻不平衡问题。广泛使用的数据集上的经验实验通过与各种最新的（SOTA）多模式学习基线进行比较来揭示我们方法的优势。

### SegLocNet: Multimodal Localization Network for Autonomous Driving via Bird's-Eye-View Segmentation 
[[arxiv](https://arxiv.org/abs/2502.20077)] [[cool](https://papers.cool/arxiv/2502.20077)] [[pdf](https://arxiv.org/pdf/2502.20077)]
> **Authors**: Zijie Zhou,Zhangshuo Qi,Luqi Cheng,Guangming Xiong
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: seglocnet：通过鸟类视图分段自动驾驶的多模式定位网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 强大而准确的本地化对于自主驾驶至关重要。传统的基于GNSS的本地化方法在城市环境中具有信号阻塞和多径效应。同时，依靠高清图（HD）地图的方法受到与高清图的构建和维护相关的高成本的限制。另一方面，基于标准定义（SD）基于地图的方法通常表现出不满意的性能或由于过度拟合而导致的概括能力不佳。为了应对这些挑战，我们提出了Seglocnet，这是一种多模式GNSS的本地化网络，使用Bird's-eye-View（BEV）语义分割实现精确的定位。 Seglocnet采用BEV分割网络来从多个传感器输入中生成语义图，然后进行详尽的匹配过程来估算车辆的自我姿势。这种方法避免了基于回归的姿势估计的局限性，并保持高解释性和泛化。通过引入统一的地图表示形式，我们的方法可以应用于HD和SD地图，而无需对网络体系结构进行任何修改，从而平衡了本地化精度和区域覆盖率。在Nuscenes和Argoverse数据集上进行的广泛实验表明，我们的方法的表现优于当前的最新方法，并且我们的方法可以在不依赖GNS的情况下准确地估计城市环境中的自我姿势，同时保持强大的泛化能力。我们的代码和预培训模型将公开发布。

### AsymLoRA: Harmonizing Data Conflicts and Commonalities in MLLMs 
[[arxiv](https://arxiv.org/abs/2502.20035)] [[cool](https://papers.cool/arxiv/2502.20035)] [[pdf](https://arxiv.org/pdf/2502.20035)]
> **Authors**: Xuyang Wei,Chunlin Tian,Li Li
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: ASYMLORA：MLLM中的数据冲突和共同点
- **领域**: 计算机视觉和模式识别
- **摘要**: 对各种图像text数据集进行的有效指令进行微调，对于开发多种模式大型语言模型（MLLM）至关重要，其中数据集组成决定了跨多模式任务的模型的适应性。但是，复杂的数据集通常包含固有的冲突（源于特定于模式的优化目标）以及能够启用交叉任务传输的潜在共同点，而大多数现有方法分别处理。为了弥合这一差距，我们介绍了Asymlora，这是一个参数有效的调谐框架，通过不对称的Lora统一知识模块化和跨模式协调：特定于任务的低级别投影（Matrix b），保留了独特的适应途径，用于相互矛盾的目标，以及共享的预测（Matrix A），以构成合理性的共同点。广泛的评估表明，Asymlora始终超过仅捕获共同点的Vanilla Lora，而Lora-Moe仅关注冲突，从而实现了各种不同的模型性能和系统效率基准。

### Joint Fusion and Encoding: Advancing Multimodal Retrieval from the Ground Up 
[[arxiv](https://arxiv.org/abs/2502.20008)] [[cool](https://papers.cool/arxiv/2502.20008)] [[pdf](https://arxiv.org/pdf/2502.20008)]
> **Authors**: Lang Huang,Qiyu Wu,Zhongtao Miao,Toshihiko Yamasaki
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 联合融合和编码：从头开始推进多模式检索
- **领域**: 计算机视觉和模式识别
- **摘要**: 对于当今的互联网应用程序，信息检索是必不可少的，但是传统的语义匹配技术通常在捕获复杂查询所需的细粒跨模式相互作用方面通常不足。尽管晚融合的两位型体系结构试图通过独立编码视觉数据和文本数据来弥合这一差距，但它们经常忽略对于全面理解所必需的微妙相互作用。在这项工作中，我们严格评估这些局限性，并引入一个统一的检索框架，该框架从头开始融合视觉和文本线索，从而使早期的跨模式相互作用能够增强上下文解释。通过两阶段的训练过程 - 复制后训练后的适应，然后进行指导调整 - 我们使用简单的一个塔式体系结构将MLLMS作为检索器。我们的方法的表现优于各种检索方案的传统方法，尤其是在处理复杂的多模式输入时。值得注意的是，与不相比的任务相比，联合融合编码器对需要模态融合的任务有了更大的改进，强调了早期整合策略的变革潜力，并指向有希望的方向，以实现上下文有效的信息检索。

### Can Large Language Models Unveil the Mysteries? An Exploration of Their Ability to Unlock Information in Complex Scenarios 
[[arxiv](https://arxiv.org/abs/2502.19973)] [[cool](https://papers.cool/arxiv/2502.19973)] [[pdf](https://arxiv.org/pdf/2502.19973)]
> **Authors**: Chao Wang,Luning Zhang,Zheng Wang,Yang Zhou
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 11pages
- **标题**: 大型语言模型可以揭开奥秘吗？探索他们在复杂方案中解锁信息的能力
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 在复杂的情况下，将多个感知输入和执行组合推理结合在一起是人类的复杂认知功能。随着多模式大语言模型的进步，最近的基准倾向于评估跨多个图像的视觉理解。但是，他们经常忽略跨多个感知信息的组合推理的必要性。为了探索高级模型在复杂场景中集成多个感知输入以进行组合推理的能力，我们介绍了两个基准：线索 - 视觉问题答案（CVQA），以及三种任务类型，以评估视觉理解和合成，以及密码 - 视觉询问答案的线索以及两种任务类型的范围，并应用了两个任务类型。对于我们的基准测试，我们提出了三种插件方法：利用模型输入来推理，通过随机性生成的最小余量解码来增强推理，并检索与语义相关的视觉信息以进行有效的数据集成。组合结果表明，当前模型在组合推理基准上的性能差，甚至最先进的（SOTA）闭合源模型在CVQA上仅达到33.04％的准确性，CPVQA的精度仅达到7.38％。值得注意的是，我们的方法提高了模型在组合推理上的性能，而在SOTA闭合源模型上，CVQA增长了22.17％，CPVQA上的CPVQA增强了9.40％，这表明了其在复杂场景中具有多个感知输入的组合推理方面的有效性。该代码将公开可用。

### ReCon: Enhancing True Correspondence Discrimination through Relation Consistency for Robust Noisy Correspondence Learning 
[[arxiv](https://arxiv.org/abs/2502.19962)] [[cool](https://papers.cool/arxiv/2502.19962)] [[pdf](https://arxiv.org/pdf/2502.19962)]
> **Authors**: Quanxing Zha,Xin Liu,Shu-Juan Peng,Yiu-ming Cheung,Xing Xu,Nannan Wang
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 10 pages, 4 figures, Accepted by CVPR2025
- **标题**: 侦察：通过强大嘈杂对应学习的关系一致性增强真实的对应歧视
- **领域**: 计算机视觉和模式识别,信息检索
- **摘要**: 我们可以准确地从包含不匹配数据对的多模式数据集中确定真实的对应关系吗？现有方法主要强调跨模态对象表示之间的相似性匹配，从而有可能忽略模式中关键关系的一致性，这对于区分真实和错误的对应关系特别重要。这样的遗漏通常会冒着将否定性误认为是积极因素的风险，从而导致意外的绩效退化。为了解决这个问题，我们提出了一个一般关系一致性学习框架，即重新侦察，以准确区分多模式数据之间的真实对应关系，从而有效地减轻了不匹配引起的不良影响。具体而言，侦察利用了一种新颖的关系一致性学习，以确保不同方式与模态内模式之间的跨模式关系一致性分别确保双模式关系的一致性。得益于对关系的这种双重约束，侦察大大提高了其对真实对应歧视的有效性，因此可靠地滤除了错配对的成对，以减轻错误监督的风险。在三个广泛使用的基准数据集上进行了广泛的实验，包括Flickr30k，MS-Coco和概念标题，以证明与其他SOTA相比，侦察的有效性和优越性。该代码可在以下网址提供：https：//github.com/qxzha/recon。

### ChatReID: Open-ended Interactive Person Retrieval via Hierarchical Progressive Tuning for Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.19958)] [[cool](https://papers.cool/arxiv/2502.19958)] [[pdf](https://arxiv.org/pdf/2502.19958)]
> **Authors**: Ke Niu,Haiyang Yu,Mengyang Zhao,Teng Fu,Siyang Yi,Wei Lu,Bin Li,Xuelin Qian,Xiangyang Xue
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: chatreid：通过层次渐进式调整视觉语言模型的开放式互动人检索
- **领域**: 计算机视觉和模式识别
- **摘要**: 人重新识别（RE-ID）是以人为中心的智能系统的关键任务，可以使用多模式查询信息始终如一地识别各个相机视图的个体。最近的研究已成功地将LVLM与Re-ID相结合，从而产生了令人鼓舞的结果。但是，现有的基于LVLM的方法面临几个局限性。他们依靠从固定模板中提取文本嵌入，这些嵌入方式被用作图像表示形式的中间功能，也可以用作特定领域特定任务中的及时调整。此外，他们无法采用VQA推理格式，从而大大限制了其更广泛的适用性。在本文中，我们提出了一个小说，多才多艺的，一对一的人重新框架框架，Chatreid。我们的方法引入了层次渐进式调整（HPT）策略，该策略通过逐步完善模型区分行人身份的能力来确保精细的身份水平检索。广泛的实验表明，我们的方法在四个不同的重新ID设置中的十个基准测试中的SOTA方法的表现优于SOTA方法，从而提供了增强的灵活性和用户友好性。 ChatReid为现实世界中的重新ID应用提供了可扩展的实用解决方案，从而实现了有效的多模式互动和细粒度的身份歧视。

### One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion 
[[arxiv](https://arxiv.org/abs/2502.19854)] [[cool](https://papers.cool/arxiv/2502.19854)] [[pdf](https://arxiv.org/pdf/2502.19854)]
> **Authors**: Chunyang Cheng,Tianyang Xu,Zhenhua Feng,Xiaojun Wu,ZhangyongTang,Hui Li,Zeyang Zhang,Sara Atito,Muhammad Awais,Josef Kittler
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Accepted by CVPR 2025 v2
- **标题**: 所有人的一个模型：低级任务互动是任务不可能的图像融合的关键
- **领域**: 计算机视觉和模式识别
- **摘要**: 高级图像融合方法主要优先考虑高级任务，其中任务交互在语义间隙中挣扎，需要复杂的桥接机制。相比之下，我们建议利用数字摄影融合中的低级视觉任务，从而通过像素级的监督进行有效的功能交互。这种新的范式为无监督的多模式融合提供了强有力的指导，而无需依靠抽象的语义，增强了任务共享的功能学习以提高更广泛的适用性。拟议中的GIFNET拥有混合图像特征和增强的通用表示，支持各种融合任务，通过单个模型在可见和看不见的情况下实现高性能。独特的实验结果表明，我们的框架还支持单模性增强，为实用应用提供了卓越的灵活性。我们的代码将在https://github.com/awcxv/gifnet上找到。

## 计算机与社会(cs.CY:Computers and Society)

该领域共有 1 篇论文

### DeepSeek reshaping healthcare in China's tertiary hospitals 
[[arxiv](https://arxiv.org/abs/2502.16732)] [[cool](https://papers.cool/arxiv/2502.16732)] [[pdf](https://arxiv.org/pdf/2502.16732)]
> **Authors**: Jishizhan Chen,Qingzeng Zhang
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: DeepSeek在中国的三级医院重塑医疗保健
- **领域**: 计算机与社会,人工智能
- **摘要**: 人工智能（AI）迅速整合到医疗保健中正在改变临床决策和医院运营。自2025年1月以来，DeepSeek已成为领先的AI系统，该系统已广泛部署在中国的三级医院中。最初在上海的主要医疗机构中实施，此后在全国范围内扩大了扩展，增强了诊断精度，简化工作流程和改善患者的管理。 AI驱动的病理学，成像分析和临床决策支持系统在优化医疗过程和减轻医疗保健专业人员的认知负担方面具有巨大的潜力。但是，医疗保健中AI的广泛采用提出了关键的监管和道德挑战，尤其是在AI辅助诊断和自动化偏见的风险方面。缺乏明确定义的责任框架强调了确保AI作为辅助工具而不是自主决策者的政策的需求。随着技术的持续进步，预计AI将整合多模式数据源，例如基因组学和放射线学，为精确医学和个性化治疗策略铺平了道路。医疗保健中AI的未来取决于透明的监管结构，行业协作和适应性治理框架的发展，这些框架将创新与责任保持平衡，确保公平有效的AI-drien驱动医疗服务。

## 数据库(cs.DB:Databases)

该领域共有 1 篇论文

### Personalized Top-k Set Queries Over Predicted Scores 
[[arxiv](https://arxiv.org/abs/2502.12998)] [[cool](https://papers.cool/arxiv/2502.12998)] [[pdf](https://arxiv.org/pdf/2502.12998)]
> **Authors**: Sohrab Namazi Nia,Subhodeep Ghosh,Senjuti Basu Roy,Sihem Amer-Yahia
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 预测分数的个性化TOP-K设置查询
- **领域**: 数据库,人工智能,机器学习
- **摘要**: 这项工作研究了昂贵的外神座的适用性，例如大语言模型在回答预测分数的Top-K查询中。用户定义的功能会产生此类分数，以回答多模式数据的个性化查询。我们提出了一个通用的计算框架，只要可以将功能分解为构造，该框架可以处理基于设置的评分函数，每个功能都将其发送到Oracle（在我们的情况下是LLM）以预测部分分数。在给定的时间点，该框架假设一组响应及其部分预测的分数，并且保持了可能是真正的顶级K的可能集合。由于呼叫甲骨文是昂贵的，因此我们的框架明智地识别下一个构造，即问甲骨文的下一个最佳问题，以最大程度地提高识别真正的顶级K的可能性。我们提出了一种量化可能性的原则性概率模型。我们研究设计算法的效率机会。我们使用三个大型数据集，评分功能和基准进行评估。实验表明了我们的框架的功效，因为它在需要LLM调用的同时确保结果准确性时，它比基准的数量级改善了。可伸缩性实验进一步表明我们的框架可以用于大规模应用中。

## 分布式、并行和集群计算(cs.DC:Distributed, Parallel, and Cluster Computing)

该领域共有 2 篇论文

### Towards Efficient Large Multimodal Model Serving 
[[arxiv](https://arxiv.org/abs/2502.00937)] [[cool](https://papers.cool/arxiv/2502.00937)] [[pdf](https://arxiv.org/pdf/2502.00937)]
> **Authors**: Haoran Qiu,Anish Biswas,Zihan Zhao,Jayashree Mohan,Alind Khare,Esha Choukse,Íñigo Goiri,Zeyu Zhang,Haiying Shen,Chetan Bansal,Ramachandran Ramjee,Rodrigo Fonseca
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 迈向有效的大型多模型服务
- **领域**: 分布式、并行和集群计算,人工智能
- **摘要**: 生成AI的最新进展导致了大型多模式模型（LMM），能够同时处理各种模式的输入，例如文本，图像，视频和音频。尽管这些模型表现出令人印象深刻的功能，但由于其复杂的架构和异质资源需求，在生产环境中有效地为生产环境提供了巨大的挑战。我们在六个代表性的开源模型上介绍了两种突出的LMM体系结构，仅解码和交叉注意的两个综合系统分析。我们研究了他们的多阶段推理管道和资源利用模式，这些模式导致独特的系统设计含义。我们还对生产LMM推理轨迹进行了深入的分析，发现了独特的工作负载特征，包括可变，重尾请求分布，各种模态组合和爆发的交通模式。我们的主要发现表明，不同的LMM推理阶段表现出高度异质性的性能特征和资源需求，而跨模态的并发请求会导致绩效的重大干扰。为了应对这些挑战，我们提出了一个脱钩的服务体系结构，可以为每个阶段进行独立的资源分配和自适应扩展。我们进一步提出了优化，例如阶段托管，以最大程度地提高吞吐量和资源利用率，同时达到延迟目标。

### Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach 
[[arxiv](https://arxiv.org/abs/2502.06355)] [[cool](https://papers.cool/arxiv/2502.06355)] [[pdf](https://arxiv.org/pdf/2502.06355)]
> **Authors**: Timo Fudala,Vasileios Tsouvalas,Nirvana Meratnia
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: 10 pages, 4 figures, submitted to IJCAI 2025
- **标题**: 边缘的微调多模式变压器：平行分开学习方法
- **领域**: 分布式、并行和集群计算,机器学习
- **摘要**: 多模式变压器集成了图像，音频和文本等多样化的数据类型，例如视听理解和图像文本检索等任务；然而，它们的高参数化限制了在资源受限的边缘设备上的部署。 Split Learning（SL）在指定的切割机上划分了模型，以将计算密集型操作卸载到服务器上，但为多模式变压器的分布式培训提供了有前途的方法，尽管其应用程序仍未得到充满激发。我们提出了MPSL，这是一种以分布式方式对多模式变压器进行计算有效微调的平行SL方法，同时消除了标签共享，客户端同步和每个客户的子模型管理。 MPSL采用轻巧的客户端引物器和统一的模态 - 不合理的编码器，从而可以灵活地适应特定于任务的需求。我们对7个多模式数据集进行的评估表明，MPSL匹配或胜过联合学习，将客户端计算减少250倍，并在模型增长中实现较高的通信成本可扩展性。通过广泛的分析，我们重点介绍了MPSL擅长的任务适用性，权衡和场景，从而激发了进一步的探索。

## 人机交互(cs.HC:Human-Computer Interaction)

该领域共有 5 篇论文

### Multimodal Brain-Computer Interfaces: AI-powered Decoding Methodologies 
[[arxiv](https://arxiv.org/abs/2502.02830)] [[cool](https://papers.cool/arxiv/2502.02830)] [[pdf](https://arxiv.org/pdf/2502.02830)]
> **Authors**: Siyang Li,Hongbin Wang,Xiaoqing Chen,Dongrui Wu
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 多模式的脑部计算机接口：AI驱动的解码方法
- **领域**: 人机交互,机器学习,神经元和认知
- **摘要**: 大脑计算机界面（BCIS）可以在大脑和外部设备之间进行直接通信。这篇综述强调了核心解码算法，使多模式BCI，包括对元素的解剖，对多元化方法的统一观点以及对该领域状态的全面分析。我们强调了跨模式映射，顺序建模的算法进步，除了经典的多模式融合外，还说明了这些新型AI方法如何增强大脑数据的解码。全面探讨了BCI应用于视觉，语音和情感解码的目前文献。展望未来，我们引起人们对多模式变压器等新兴体系结构的影响，并讨论诸如大脑数据异质性和常见错误之类的挑战。这篇评论还可以作为神经科学背景和研究AI专家的专家的跨学科领域的桥梁，旨在为AI驱动的多模式BCI提供全面的理解。

### SensPS: Sensing Personal Space Comfortable Distance between Human-Human Using Multimodal Sensors 
[[arxiv](https://arxiv.org/abs/2502.07441)] [[cool](https://papers.cool/arxiv/2502.07441)] [[pdf](https://arxiv.org/pdf/2502.07441)]
> **Authors**: Ko Watanabe,Nico Förster,Shoya Ishimaru
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: Sensp：使用多模式传感器在人类之间传感个人空间舒适的距离
- **领域**: 人机交互,人工智能
- **摘要**: 个人空间，也称为个人空间，在人类的社会互动中至关重要，影响了舒适，沟通和社会压力。估计和尊重个人空间对于增强人类计算机互动（HCI）和智能环境至关重要。个人空间偏好因个人特征，文化背景和上下文因素而异。高级多模式传感技术（包括眼球跟踪和腕带传感器）提供了开发自适应系统，以动态适应用户舒适度的水平。整合生理和行为数据可以更深入地了解空间相互作用。这项研究开发了一个基于传感器的模型，以估计舒适的个人空间并确定影响空间偏好的关键特征。我们的发现表明，多模式传感器，尤其是眼睛跟踪和生理腕带数据，可以有效地预测个人空间偏好，而眼球跟踪的数据起着更为重要的作用。一项涉及受控人类相互作用的实验研究表明，基于变压器的模型可实现估计个人空间的最高预测精度（F1分数：0.87）。凝视点和瞳孔直径等眼球追踪特征是最重要的预测因子，而腕带传感器的生理信号略有贡献。这些结果突出了在自适应环境中对社会空间进行AI驱动的个性化的潜力，这表明可以利用多模式感应来开发智能系统，以优化工作场所，教育机构和公共环境中的空间安排。未来的工作应探索较大的数据集，现实世界应用和其他生理标记，以增强模型鲁棒性。

### Enhancing Higher Education with Generative AI: A Multimodal Approach for Personalised Learning 
[[arxiv](https://arxiv.org/abs/2502.07401)] [[cool](https://papers.cool/arxiv/2502.07401)] [[pdf](https://arxiv.org/pdf/2502.07401)]
> **Authors**: Johnny Chan,Yuming Li
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 9 pages, 4 figures, accepted and presented in the 2025 6th International Conference on Advances in Education and Information Technology (AEIT)
- **标题**: 通过生成AI增强高等教育：一种个性化学习的多模式方法
- **领域**: 人机交互,人工智能
- **摘要**: 这项研究通过设计和开发多模式聊天机器人为本科课程探索了高等教育领域中生成的AI（Genai）的机会。利用Chatgpt API进行细微的基于文本的互动和Google Bard进行高级图像分析和图表到代码转换，我们展示了Genai在解决广泛的教育查询方面的潜力。此外，聊天机器人为教育工作者提供了基于文件的分析仪，通过情感和情感分析为学生反馈提供了深入的见解，并用关键指标总结了课程评估。这些组合突出了多模式对话AI在增强教学过程中的关键作用，有望在教育适应性，参与度和反馈分析方面取得重大进步。通过展示实用的Web应用程序，这项研究强调了整合Genai技术以促进更具动态和响应迅速的教育环境的必要性，最终有助于改善教育成果和教学策略。

### ZIA: A Theoretical Framework for Zero-Input AI 
[[arxiv](https://arxiv.org/abs/2502.16124)] [[cool](https://papers.cool/arxiv/2502.16124)] [[pdf](https://arxiv.org/pdf/2502.16124)]
> **Authors**: Aditi De,NeuroBits Labs
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: Zia：零输入AI的理论框架
- **领域**: 人机交互,机器学习
- **摘要**: 零输入AI（ZIA）通过在没有明确的用户命令的情况下实现主动的意图预测来引入人机交互的新框架。它将目光跟踪，生物信号（脑电图，心率）和上下文数据（时间，位置，用法历史记录）集成到实时推理的多模式模型中，针对<100 ms的延迟。所提出的体系结构采用了一个基于变压器的模型，具有交叉模式的注意力，跨模式的贝叶斯推断对不确定性估计，以及对自适应优化的增强学习。为了支持边缘设备（CPU，TPU，NPU）上的部署，Zia利用量化，权重修剪和线性注意力，以降低从二次到线性的复杂性，并具有序列长度。理论分析建立了有关预测误差的信息理论，并证明了多模式融合如何提高单模式方法的准确性。预期的性能表明EEG集成和60-100毫秒的推断潜伏期的准确性为85-90％。 Zia为可访问性，医疗保健和消费者应用提供了一个可扩展的，保护隐私的框架，将AI推向了预期智能。

### M2LADS Demo: A System for Generating Multimodal Learning Analytics Dashboards 
[[arxiv](https://arxiv.org/abs/2502.15363)] [[cool](https://papers.cool/arxiv/2502.15363)] [[pdf](https://arxiv.org/pdf/2502.15363)]
> **Authors**: Alvaro Becerra,Roberto Daza,Ruth Cobos,Aythami Morales,Julian Fierrez
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: Published in the Workshop on Innovation and Responsibility in AI-Supported Education (iRAISE25) at AAAI 2025
- **标题**: M2LADS演示：生成多模式学习分析仪表板的系统
- **领域**: 人机交互,计算机视觉和模式识别
- **摘要**: 我们介绍了一个名为M2LADS的基于Web的系统（“用于生成多模式学习分析仪表板的系统”），该系统旨在集成，同步，可视化和分析在基于计算机的学习过程中记录的多模态数据与生物传感器。该系统在基于Web的仪表板上介绍了一系列的生物识别和行为数据，并为各种基于生理和活动的指标提供了详细的见解。可视化的多模式数据包括用于评估注意力和大脑活动的脑电图（EEG）数据，心率指标，眼睛跟踪数据以衡量视觉注意力，网络摄像头视频记录以及受监视任务的活动日志。 M2LADS的目的是通过两种关键方式来协助数据科学家：（1）通过提供参与者的经验的全面视图，显示参与者参与的活动所分类的所有数据，以及（2）通过同步所有的生物信号和视频，促进如果任何活动信息包含错误，则可以促进更轻松的数据重新显示。

## 信息检索(cs.IR:Information Retrieval)

该领域共有 24 篇论文

### VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos 
[[arxiv](https://arxiv.org/abs/2502.01549)] [[cool](https://papers.cool/arxiv/2502.01549)] [[pdf](https://arxiv.org/pdf/2502.01549)]
> **Authors**: Xubin Ren,Lingrui Xu,Long Xia,Shuaiqiang Wang,Dawei Yin,Chao Huang
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: Videorag：带有极端长篇小说视频的检索演示一代
- **领域**: 信息检索,人工智能,计算机视觉和模式识别
- **摘要**: 通过外部知识整合增强大语言模型（LLM）在增强大型语言模型（LLM）方面取得了显着成功，但其应用主要集中在文本内容上，使多模式视频知识的丰富领域主要没有探索。本文介绍了Videorag，这是第一个专门为处理和理解极其长篇小说视频而设计的检索生成框架。我们的核心创新在于其双通道体系结构，该体系结构无缝集成（i）基于图形的文本知识接地，用于捕获跨Video语义关系，以及（ii）多模式上下文编码以有效保留视觉特征。这项新颖的设计通过构造跨越多个视频的精确知识图，同时通过专门的多模式检索范式来维持语义依赖性，从而使Videorag能够处理无限长度的视频。通过对我们拟议的Longervideos基准进行的全面经验评估，超过160个视频，在整个讲座，纪录片和娱乐类别中总计134小时以上，与现有的抹布替代方案和长期视频理解方法相比，纪录片和娱乐类别类别表现出了实质性的性能。 Videorag实现和基准数据集的源代码可公开可用：https：//github.com/hkuds/videorag。

### MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling 
[[arxiv](https://arxiv.org/abs/2502.00321)] [[cool](https://papers.cool/arxiv/2502.00321)] [[pdf](https://arxiv.org/pdf/2502.00321)]
> **Authors**: Bencheng Yan,Si Chen,Shichang Jia,Jianyu Liu,Yueran Liu,Chenghan Fu,Wanxian Guan,Hui Zhao,Xiang Zhang,Kai Zhang,Wenbo Su,Pengjie Wang,Jian Xu,Bo Zheng,Baolin Liu
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: MIM：用于用户行为建模的多模式内容兴趣建模范例
- **领域**: 信息检索,人工智能
- **摘要**: 点击率（CTR）预测是推荐系统，在线搜索和广告平台中的至关重要任务，在该计划中，准确捕获用户对内容的真正兴趣对于性能至关重要。但是，现有方法在很大程度上依赖于ID嵌入，这些嵌入方式无法反映用户对图像和标题等内容的真正偏好。在冷启动和长尾场景中，这种限制变得尤为明显，在这种情况下，传统方法难以实现有效的结果。为了应对这些挑战，我们提出了一种新型的多模式内容兴趣建模范式（MIM），该范围包括三个关键阶段：培训前，内容含义的有监督的微调（C-SFT）和内容与内容相关性的UBM（CIUBM）。训练阶段的基础模型适应特定于域的数据，从而可以提取高质量的多模式嵌入。 C-SFT阶段通过利用用户行为信号指导嵌入与用户偏好的嵌入对齐方式来弥合内容和用户兴趣之间的语义差距。最后，CIUBM阶段将多模式的嵌入式和基于ID的协作过滤信号集成到统一的框架中。在世界上最大的电子商务平台之一The Toobao上进行的全面离线实验和在线A/B测试证明了MIM方法的有效性和效率。该方法已成功地在线部署，在CTR中实现了 +14.14％的显着增长，RPM +4.12％，展示了其工业适用性，并对平台性能产生了重大影响。为了促进进一步的研究，我们已在https://pan.quark.cn/s/8fc8ec3e74f3上公开发布了代码和数据集。

### Contrastive Learning for Cold Start Recommendation with Adaptive Feature Fusion 
[[arxiv](https://arxiv.org/abs/2502.03664)] [[cool](https://papers.cool/arxiv/2502.03664)] [[pdf](https://arxiv.org/pdf/2502.03664)]
> **Authors**: Jiacheng Hu,Tai An,Zidong Yu,Junliang Du,Yuanshuai Luo
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 与自适应功能融合的冷启动推荐的对比度学习
- **领域**: 信息检索,机器学习
- **摘要**: 本文提出了一个冷启动推荐模型，该模型集成了对比度学习，旨在解决由于用户和项目交互数据的稀缺性，因此在冷启动方案中推荐系统的性能降解问题。该模型通过自适应特征选择模块动态调节关键特征的权重，并通过组合多模式特征融合机制，有效地整合用户属性，项目元信息和上下文特征，从而提高建议性能。此外，该模型还引入了一种对比度学习机制，以通过构建正和负样本对来增强特征表示的鲁棒性和概括能力。实验是在Movielens-1M数据集上进行的。结果表明，所提出的模型在HR，NDCG，MRR和召回率方面显着优于主流建议方法，例如矩阵分解，LightGBM，DEEPFM和AUTOREC，尤其是在冷启动场景中。消融实验进一步验证了每个模块在改善模型性能中的关键作用，并且学习率敏感性分析表明，中等学习率对于模型的优化效果至关重要。这项研究不仅为冷启动问题提供了新的解决方案，而且还为在推荐系统中应用对比度学习提供了重要的参考。将来，该模型有望在更广泛的方案中发挥作用，例如实时建议和跨域建议。

### Intent Alignment between Interaction and Language Spaces for Recommendation 
[[arxiv](https://arxiv.org/abs/2502.03307)] [[cool](https://papers.cool/arxiv/2502.03307)] [[pdf](https://arxiv.org/pdf/2502.03307)]
> **Authors**: Yu Wang,Lei Sang,Yi Zhang,Yiwen Zhang
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: 11 pages, 8 figures
- **标题**: 互动和语言空间之间的意图对齐
- **领域**: 信息检索
- **摘要**: 基于意图的推荐系统已引起了揭示潜在细粒偏好的极大关注。意图，作为相互作用的潜在因素，对于改善建议解释性至关重要。大多数方法将意图定义为与交互一起更新的可学习参数。但是，现有的框架通常会忽略文本信息（例如用户评论，项目描述），这对于减轻互动意图的稀疏至关重要。探索这些多模式意图，尤其是表示空间中的固有差异，提出了两个关键挑战：i）如何使多模式意图保持一致并有效缓解噪声问题； ii）如何提取和匹配跨模态的潜在密钥意图。为了应对这些挑战，我们提出了一个模型不足的框架，意图表示使用大语言模型（IRLLREC），该学习利用大型语言模型（LLMS）来构建多模式意图并增强建议。具体而言，iRLLREC采用双较高的体系结构来学习多模式意图表示。接下来，我们提出成对和翻译对准，以消除模式间差异并增强对噪声输入特征的鲁棒性。最后，为了更好地匹配基于文本和互动的意图，我们采用动量蒸馏来对融合意图表示进行教师学习。三个数据集的经验评估表明，我们的IRLLREC框架的表现优于基准。

### Large Language Models Are Universal Recommendation Learners 
[[arxiv](https://arxiv.org/abs/2502.03041)] [[cool](https://papers.cool/arxiv/2502.03041)] [[pdf](https://arxiv.org/pdf/2502.03041)]
> **Authors**: Junguang Jiang,Yanwen Huang,Bin Liu,Xiaoyu Kong,Ziru Xu,Han Zhu,Jian Xu,Bo Zheng
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 大型语言模型是普遍推荐学习者
- **领域**: 信息检索,机器学习
- **摘要**: 在实际推荐系统中，通常使用具有精心设计的模型架构的特定任务数据集对不同的任务进行处理。我们证明，大型语言模型（LLM）可以充当通用建议学习者，能够在统一输入输出框架中处理多个任务，从而消除了对专业模型设计的需求。为了提高LLMS的建议性能，我们引入了用于项目表示形式的多模式融合模块，以及一种用于有效候选生成的序列中的方法。当应用于工业规模的数据时，我们的LLM通过精心设计的专家模型来实现竞争成果。此外，我们的分析表明，推荐结果对文本输入非常敏感，强调了迅速工程在优化工业规模推荐系统方面的潜力。

### HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads 
[[arxiv](https://arxiv.org/abs/2502.05822)] [[cool](https://papers.cool/arxiv/2502.05822)] [[pdf](https://arxiv.org/pdf/2502.05822)]
> **Authors**: Guobing Gan,Kaiming Gao,Li Wang,Shen Jiang,Peng Jiang
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: Accepted by WWW 2025 (Industry Track)
- **标题**: HCMRM：搜索广告的高稳态多模式相关模型
- **领域**: 信息检索
- **摘要**: 搜索广告对于商人在短视频平台上访问目标用户至关重要。与用户搜索意图对齐的简短视频广告通过相关匹配和出价排名机制显示。本文着重于改进查询与视频相关性匹配，以提高广告系统中排名的有效性。最近的视觉训练预训练模型已经在各种多模式任务中表现出了希望。但是，它们对下游查询视频相关任务的贡献是有限的，因为这对视​​觉信号和文本之间的对齐与查询，视觉信号和视频文本的三重率建模不同。此外，我们以前的相关性模型提供了有限的排名功能，这在很大程度上是由于二进制跨循环微调目标与排名目标之间的差异。为了解决这些局限性，我们设计了高稳定的多模式相关模型（HCMRM）。它利用一种简单但有效的方法来增强培训和相关任务之间的一致性。具体而言，在训练阶段，以及对齐视觉信号和视频文本的一致性，从视频文本中提取了几个关键字作为伪Queries，以执行三重态相关性建模。对于微调阶段，我们引入了层次软效果丢失，该损失使该模型能够在标签中学习订单，同时最大程度地提高正面样品和负样本之间的区别。这促进了随后排名阶段的相关性和竞标的融合排名。所提出的方法已在Kuaishou搜索广告系统中部署了一年多，导致无关广告比例降低了6.1％，广告收入增加了1.4％。

### Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information 
[[arxiv](https://arxiv.org/abs/2502.08071)] [[cool](https://papers.cool/arxiv/2502.08071)] [[pdf](https://arxiv.org/pdf/2502.08071)]
> **Authors**: Yunhang He,Cong Xu,Jun Wang,Wei Zhang
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: 协作过滤符合频谱偏移：将用户项目交互与图形结构的侧面信息连接
- **领域**: 信息检索
- **摘要**: 图形神经网络（GNN）在协作过滤中证明了它们的优势，其中用户项目（U-I）相互作用双方图作为基本数据格式。但是，当将图形结构的侧面信息（例如，多模式相似性图或社交网络）集成到U-I双方图中时，现有的图形协作滤波方法无法实现令人满意的性能。我们从光谱的角度定量分析了这个问题。回想一下，两分图在[-1，1]的范围内具有完整的光谱，其频率最高可在-1处达到-1，最低频率在1；但是，我们观察到随着更多的侧面信息的合并，增强邻接矩阵的最高频率逐渐向右移动。这种频谱转移现象已导致以前为全频谱[-1，1]构建的方法，以将不匹配的重要性分配给不同的频率。为此，我们提出了频谱偏移校正（称为SSC），并结合了变化和缩放因子，以使光谱GNNS适应移位光谱。与以前的利用侧面信息的范式（需要针对各种数据类型的设计量身定制设计），SSC将传统的图形协作过滤连接到任何图形结构的侧面信息。关于社会和多模式建议的实验证明了SSC的有效性，可实现多达23％的相对改善，而不会产生任何其他计算开销。

### An Efficient Large Recommendation Model: Towards a Resource-Optimal Scaling Law 
[[arxiv](https://arxiv.org/abs/2502.09888)] [[cool](https://papers.cool/arxiv/2502.09888)] [[pdf](https://arxiv.org/pdf/2502.09888)]
> **Authors**: Songpei Xu,Shijia Wang,Da Guo,Xianwen Guo,Qiang Xiao,Fangjian Li,Chuanjiang Luo
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 有效的大型推荐模型：迈向一项最佳的规模法律
- **领域**: 信息检索
- **摘要**: 对扩展建议模型的追求会面临扩展模型容量和保持计算障碍性之间的内在张力。虽然先前的研究探讨了推荐系统的规模法律，但对于大多数工业应用，其资源密集型范式通常需要数以万计的A100 GPU小时。这项工作解决了一个关键的差距：在严格的计算预算下实现可持续模型缩放。我们提出了登山者，这是一个资源有效的推荐框架，其中包括两个协同组件：用于算法创新的Astro模型体系结构和用于工程优化的涡轮加速框架。 Astro（适应性伸缩变压器用于推荐）采用了两种核心创新：（1）多尺度序列分区，通过层次块将注意力从O（N^2D）降低到O（N^2D/NB），从而降低了注意力（N^2D/NB），从而实现更有效的缩放尺度的序列； （2）动态温度调制，可自适应地调整由固有多模式分布的注意力评分，该分布由固有的多scenario和多行为相互作用引起。登山者在没有绩效降级的情况下，辅以涡轮（两阶段的统一排名和批处理输出），这是一个共同设计的加速框架，集成了梯度感知功能压缩和记忆效率高的键值缓存。多个数据集上的全面离线实验验证了登山者表现出更理想的缩放曲线。据我们所知，这是第一个公开记录的框架，其中受控的模型缩放驱动了连续的在线度量增长（总升降机为12.19％），而没有资源成本过高。登山者已成功部署在中国最大的音乐流媒体平台之一的Netase Cloud Music上，每天为数千万用户提供服务。

### Multi-Turn Multi-Modal Question Clarification for Enhanced Conversational Understanding 
[[arxiv](https://arxiv.org/abs/2502.11442)] [[cool](https://papers.cool/arxiv/2502.11442)] [[pdf](https://arxiv.org/pdf/2502.11442)]
> **Authors**: Kimia Ramezan,Alireza Amiri Bavandpour,Yifei Yuan,Clemencia Siro,Mohammad Aliannejadi
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 多态多模式问题澄清，以增强对话理解
- **领域**: 信息检索,人工智能,计算语言学,机器学习
- **摘要**: 会话查询澄清使用户可以通过交互式对话来完善其搜索查询，从而提高搜索效果。传统方法取决于基于文本的澄清问题，这些问题通常无法捕获复杂的用户偏好，尤其是涉及视觉属性的偏好。尽管最近的工作探索了与文本以及文本以及图像以及文本并不完全支持用户意图在多个转弯的渐进性的渐进性的单程澄清。在此的激励下，我们介绍了多模式澄清问题（MMCQ）任务，该任务结合了文本和视觉方式，以在多转交谈中完善用户查询。为了促进这项任务，我们创建了一个名为Clarimm的大规模数据集，该数据集包括超过13k的多转交互和33K询问答案对，其中包含多模式澄清的问题。我们提出了Mario，这是一个采用两阶段排名策略的检索框架：与BM25的初始检索，然后是一个多模式生成重新排列模型，该模型集成了来自对话历史记录中的文本和视觉信息。我们的实验表明，多模式的多模式澄清优于单模式和单转弯方法，将MRR提高了12.88％。在较长的相互作用中，收益最为重要，证明了进行复杂查询的进行性完善的价值。

### MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation 
[[arxiv](https://arxiv.org/abs/2502.11246)] [[cool](https://papers.cool/arxiv/2502.11246)] [[pdf](https://arxiv.org/pdf/2502.11246)]
> **Authors**: Sayantan Adak,Somnath Banerjee,Rajarshi Mandal,Avik Halder,Sayan Layek,Rima Hazra,Animesh Mukherjee
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: Code and data available at: https://github.com/sayantan11995/MemeSense
- **标题**: MEMESENSE：社交常识性模因适度的自适应内部文化框架
- **领域**: 信息检索,计算语言学,计算机与社会
- **摘要**: 模因提出了独特的节制挑战，因为它们的图像，文本和社会环境的微妙，多模式相互作用。主要依赖于明确的文本提示的标准系统通常会忽略具有讽刺，象征或文化参考的有害内容。为了解决这一差距，我们介绍了Memesense，这是一个自适应的内在学习框架，将社交常识性推理与视觉和语义相关的参考示例融合在一起。通过将重要的任务信息编码为可学习的认知转移矢量，模因有效地平衡了词汇，视觉和道德考虑因素，从而实现了精确而又具有上下文感知的模因干预措施。对一组策划的一组隐含有害模因的广泛评估表明，模因基本上优于强大的基线，为更安全的在线社区铺平了道路。代码和数据可用：https：//github.com/sayantan11995/memesense

### From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers in Generation, Comprehension, Recommendation, and Information Retrieval 
[[arxiv](https://arxiv.org/abs/2502.12448)] [[cool](https://papers.cool/arxiv/2502.12448)] [[pdf](https://arxiv.org/pdf/2502.12448)]
> **Authors**: Jian Jia,Jingtong Gao,Ben Xue,Junhao Wang,Qingpeng Cai,Quan Chen,Xiangyu Zhao,Peng Jiang,Kun Gai
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 从原则到应用程序：对生成，理解，建议和信息检索的离散引物的全面调查
- **领域**: 信息检索
- **摘要**: 离散的引物器已成为现代机器学习系统中必不可少的组件，尤其是在自回归建模和大型语言模型（LLMS）的背景下。这些令牌器是将原始的，非结构化的数据从不同模态转换为离散令牌的关键接口，使LLMS能够在各种任务中有效运行。尽管在发电，理解和推荐系统中它们的核心作用，但在文献中，一项致力于离散引导者的综合调查仍然显着。本文通过对离散令牌的设计原理，应用和挑战进行系统的审查来解决这一差距。我们首先剖析令牌剂的子模块，并系统地证明其内部机制，以提供对其功能和设计的全面理解。在此基础的基础上，我们合成了最先进的方法，将其分类为多模式生成和理解任务，并进行语义令牌以获得个性化建议。此外，我们批判性地分析了现有的引导者的局限性，并概述了未来研究的有希望的方向。通过提出一个统一的框架来理解离散的引物器，该调查旨在指导研究人员和从业人员应对公开挑战并推进该领域，最终有助于开发更强大和多功能的AI系统。

### REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark 
[[arxiv](https://arxiv.org/abs/2502.12342)] [[cool](https://papers.cool/arxiv/2502.12342)] [[pdf](https://arxiv.org/pdf/2502.12342)]
> **Authors**: Navve Wasserman,Roi Pony,Oshri Naparstek,Adi Raz Goldfarb,Eli Schwartz,Udi Barzelay,Leonid Karlinsky
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: Real-MM-rag：现实世界中的多模式检索基准
- **领域**: 信息检索,计算机视觉和模式识别
- **摘要**: 准确的多模式文档检索对于检索型发电（RAG）至关重要，但是现有的基准测试并不能以目前的设计完全捕捉现实世界中的挑战。我们介绍了Real-MM-rag，这是一种自动生成的基准测试，旨在解决现实检索所必需的四个关键属性：（i）多模式文档，（ii）增强的难度，（iii）现实的rag查询和（iv）精确的标签。此外，我们基于查询重新设计提出了一个多缺陷级方案，以评估模型超出关键字匹配的语义理解。我们的基准揭示了重要的模型弱点，尤其是在处理桌面繁重的文档和鲁棒性时以查询重新设计。为了减轻这些缺点，我们策划了一个改头换面的培训集，并引入了一个新的以金融为中心的台式数据集。这些数据集上的微调使模型能够在Real-MM-Rag基准测试上实现最新的检索性能。我们的工作提供了一种更好的方法来评估和改善多模式抹布系统的检索，同时还提供了解决当前限制的培训数据和模型。

### Open-Ended and Knowledge-Intensive Video Question Answering 
[[arxiv](https://arxiv.org/abs/2502.11747)] [[cool](https://papers.cool/arxiv/2502.11747)] [[pdf](https://arxiv.org/pdf/2502.11747)]
> **Authors**: Md Zarif Ul Alam,Hamed Zamani
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 开放式和知识密集的视频问题回答
- **领域**: 信息检索
- **摘要**: 视频问题回答需要外部知识以外的视觉内容，这在AI系统中仍然是一个重大挑战。尽管模型可以根据直接的视觉观察有效地回答问题，但在面对需要更广泛的上下文知识的问题时，它们通常会步履蹒跚。为了解决这一局限性，我们通过多模式检索仪的镜头研究知识密集型视频问题回答（KI-Videoqa），特别着眼于处理开放式问题，而不仅仅是多选择格式。我们的全面分析使用尖端的检索和视觉语言模型研究了各种检索增强方法，测试了零击和微调配置。我们研究了几个关键维度：不同信息源和模式之间的相互作用，整合多种模式环境的策略以及查询配方与检索结果利用之间的动态。我们的发现表明，尽管检索增强表明在改善模型性能方面有希望，但其成功在很大程度上取决于所选的方式和检索方法。该研究还强调了查询构建和检索深度优化在有效知识整合中的关键作用。通过我们提出的方法，我们在知识VQA数据集中的多项选择问题上取得了17.5％的准确性提高，建立了新的最先进的绩效水平。

### TALKPLAY: Multimodal Music Recommendation with Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.13713)] [[cool](https://papers.cool/arxiv/2502.13713)] [[pdf](https://arxiv.org/pdf/2502.13713)]
> **Authors**: Seungheon Doh,Keunwoo Choi,Juhan Nam
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 谈话：大语模型的多模式音乐推荐
- **领域**: 信息检索,声音,音频和语音处理
- **摘要**: 我们介绍Talkplay，这是一种多模式的音乐推荐系统，将建议任务重新设计为大语言模型代币产生。 Talkplay通过扩展的令牌词汇代表音乐，该词汇编码多种模式 - 音频，歌词，元数据，语义标签和播放列表共发生。使用这些丰富的表示，该模型学会通过对音乐推荐对话进行的下一步预测来生成建议，这需要学习协会自然语言查询和响应以及音乐项目。换句话说，该公式将音乐推荐转化为自然语言理解任务，该任务预测对话代币的能力直接优化了查询项目的相关性。我们的方法消除了传统的建议管道的复杂性，从而使查询意识到的音乐建议的端到端学习。在实验中，谈话是成功训练的，并且在各个方面都超过了基线方法，这表明了强烈的上下文理解是对话音乐的推荐者。

### A Survey of Model Architectures in Information Retrieval 
[[arxiv](https://arxiv.org/abs/2502.14822)] [[cool](https://papers.cool/arxiv/2502.14822)] [[pdf](https://arxiv.org/pdf/2502.14822)]
> **Authors**: Zhichao Xu,Fengran Mo,Zhiqi Huang,Crystina Zhang,Puxuan Yu,Bei Wang,Jimmy Lin,Vivek Srikumar
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 信息检索中模型体系结构的调查
- **领域**: 信息检索
- **摘要**: 这项调查研究了信息检索（IR）中模型架构的演变，重点介绍了两个关键方面：用于特征提取的骨干模型和相关性估计的端到端系统体系结构。该评论有意将建筑考虑与培训方法分开，以对IR系统中的结构创新进行重点分析。我们追踪从传统的基于术语的方法到现代神经方法的发展，尤其是突出了基于变压器的模型和随后的大型语言模型（LLMS）的影响。我们通过讨论新出现的挑战和未来方向，包括针对性能和可扩展性的建筑优化，多模式，多语言数据的处理以及对传统搜索范式以外的新型应用领域的适应。

### Semantic Gaussian Mixture Variational Autoencoder for Sequential Recommendation 
[[arxiv](https://arxiv.org/abs/2502.16140)] [[cool](https://papers.cool/arxiv/2502.16140)] [[pdf](https://arxiv.org/pdf/2502.16140)]
> **Authors**: Beibei Li,Tao Xiang,Beihong Jin,Yiyuan Zheng,Rui Zhao
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: Accepted by DASFAA 2025
- **标题**: 语义高斯混合物变化自动编码器用于顺序建议
- **领域**: 信息检索
- **摘要**: 序列推荐（SR）的变异自动编码器（VAE），该（SR）学习了每个用户 - 项目交互序列而不是确定的嵌入，它是可靠的，可以抵抗数据缺陷并实现明显的性能。但是，现有的基于VAE的SR模型假设单峰高斯分布是序列表示的先前分布，从而导致在用户具有多个兴趣时捕获复杂用户兴趣并限制建议性能的能力受到限制。因此，用户具有多种不同的兴趣是很常见的，我们认为在SR方案中建立多模式的先验分布更合理，而不是单峰型。因此，在本文中，我们提出了一种名为Sigma的新型基于VAE的SR模型。 Sigma假定序列表示的先验表示符合高斯混合物分布，其中分布的每个组成部分在语义上对应于多个兴趣之一。为了进行多种利益启发，Sigma包括一个概率的多功能提取模块，该模块根据隐式项目超类别来学习每个兴趣的单峰高斯分布。此外，为了将多模式兴趣纳入序列表示学习中，Sigma构建了多个感知的Elbo，它与高斯混合物兼容。公共数据集的广泛实验证明了Sigma的有效性。该代码可从https://github.com/libeibei95/sigma获得。

### ESANS: Effective and Semantic-Aware Negative Sampling for Large-Scale Retrieval Systems 
[[arxiv](https://arxiv.org/abs/2502.16077)] [[cool](https://papers.cool/arxiv/2502.16077)] [[pdf](https://arxiv.org/pdf/2502.16077)]
> **Authors**: Haibo Xing,Kanefumi Matsuyama,Hao Deng,Jinxin Hu,Yu Zhang,Xiaoyi Zeng
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 10 pages, 6 figures, Proceedings of the ACM Web Conference 2025
- **标题**: ESANS：大规模检索系统的有效和语义意识的负抽样
- **领域**: 信息检索
- **摘要**: 工业推荐系统通常涉及一个两个阶段的过程：检索和排名，旨在与数百万个项目相匹配。在检索阶段，经典的基于嵌入的检索（EBR）方法取决于有效的负抽样技术，以提高性能和效率。但是，现有技术通常会遭受虚假负面影响，确保采样质量和语义信息缺乏的高成本。为了解决这些局限性，我们提出了有效和语义意识的负抽样（ESAN），该采样（ESAN）整合了两个关键组成部分：有效密集的插值策略（EDIS）和多模式语义感知聚类（MSAC）。 EDIS在低维嵌入空间内生成虚拟样本，以提高采样分布的多样性和密度，同时最大程度地减少计算成本。 MSAC通过基于多模式信息（视觉，文本，行为）的层次聚类项表示，通过层次聚类的项目表示来完善负面采样分布，从而确保语义一致性并降低虚假负面。广泛的离线和在线实验证明了ESAN的效率和性能的卓越效率和性能。

### Joint Similarity Item Exploration and Overlapped User Guidance for Multi-Modal Cross-Domain Recommendation 
[[arxiv](https://arxiv.org/abs/2502.16068)] [[cool](https://papers.cool/arxiv/2502.16068)] [[pdf](https://arxiv.org/pdf/2502.16068)]
> **Authors**: Weiming Liu,Chaochao Chen,Jiahe Xu,Xinting Liao,Fan Wang,Xiaolin Zheng,Zhihui Fu,Ruiguang Pei,Jun Wang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 联合相似性项目探索和多模式跨域建议的重叠用户指南
- **领域**: 信息检索
- **摘要**: 跨域建议（CDR）已广泛研究，用于通过跨域的知识共享解决长期存在的数据稀疏问题。在本文中，我们专注于多模式跨域推荐（MMCDR）问题，其中不同的项目具有多模式信息，而很少有用户在范围内重叠。 MMCDR在两个方面尤其具有挑战性：完全利用每个域内的多种多态信息，并利用跨领域的有用的知识转移。但是，以前的方法无法群集具有相似特征的项目，同时滤除了不同方式以不同方式的噪声，从而刺激了模型性能。更糟糕的是，传统的CDR模型主要依赖于重叠的用户进行域适应，使他们能够处理大多数用户不被拼写的方案。为了填补这一空白，我们建议联合相似性项目探索和用户指导（SIEOG）解决MMCDR问题。 Sieoug首先提出了相似性项目探索模块，该模块不仅可以获得成对和小组的项目 - 项目图形知识，而且还减少了多模式建模的无关噪声。然后，Sieoug提出了用户项目协作过滤模块，以汇总用户/项目的嵌入方式，并使用关注机制进行协作过滤。最终，Sieoug提出了与最佳用户匹配的重叠用户指南模块，以跨域进行知识共享。我们对亚马逊数据集的实证研究具有多个不同的任务，这表明，Sieoug在MMCDR设置下的最先进模型大大优于最先进的模型。

### Visual Zero-Shot E-Commerce Product Attribute Value Extraction 
[[arxiv](https://arxiv.org/abs/2502.15979)] [[cool](https://papers.cool/arxiv/2502.15979)] [[pdf](https://arxiv.org/pdf/2502.15979)]
> **Authors**: Jiaying Gong,Ming Cheng,Hongda Shen,Pierre-Yves Vandenbussche,Janet Jenq,Hoda Eldardiry
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 10 pages, 4 figures, accepted for publication in NAACL 2025 Industry Track
- **标题**: 视觉零击电子产品属性值提取
- **领域**: 信息检索,计算机视觉和模式识别
- **摘要**: 电子商务行业中现有的零击产品属性值（方面）提取方法依赖于单模式或多模式模型，要求卖方为产品提供详细的文本输入（产品说明）。但是，手动提供（键入）产品描述对卖家来说是耗时的和令人沮丧的。因此，我们提出了基于夹子的跨模式零摄像属性生成框架（VIOC-AG），该框架仅需要产品图像作为输入。 vioc-ag遵循仅文本培训过程，其中通过冻结的剪辑文本编码器对任务注定的文本解码器进行了培训，以减轻模态差距和任务断开连接。在零弹性推理期间，产品方面是由与训练有素的任务注重文本解码器连接的冷冻夹图像编码器生成的。 OCR令牌和来自冷冻提示的LLM的OCR代币和输出校正了分域属性值的解码输出。实验表明，VIOC-AG明显胜过其他微调视觉模型，用于零击属性值提取。

### A Survey on Multimodal Recommender Systems: Recent Advances and Future Directions 
[[arxiv](https://arxiv.org/abs/2502.15711)] [[cool](https://papers.cool/arxiv/2502.15711)] [[pdf](https://arxiv.org/pdf/2502.15711)]
> **Authors**: Jinfeng Xu,Zheyu Chen,Shuo Yang,Jinze Li,Wei Wang,Xiping Hu,Steven Hoi,Edith Ngai
> **First submission**: 2025-01-22
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 一项关于多模式推荐系统的调查：最新进展和未来方向
- **领域**: 信息检索,多媒体
- **摘要**: 从互联网上快速扩展的信息中获取有价值的数据已成为一个重大问题，并且推荐系统已成为一种广泛使用且有效的工具，可帮助用户发现感兴趣的项目。推荐系统的本质在于它们能够预测用户评级或各种项目的偏好，并随后根据历史互动数据和公开可用信息推荐最相关的评分。随着多种多媒体服务的出现，包括文本，图像，视频和音频，人类可以通过多种方式感知世界。因此，能够理解和解释不同模态数据的推荐系统可以更有效地指出个体偏好。多模式推荐系统（MRS）不仅捕获了多种模态的隐式交互信息，而且有可能发现这些模式之间隐藏的关系。这项调查的主要目的是全面回顾MRS的最新研究进步，并从技术角度分析模型。具体而言，我们旨在从技术角度总结太太的一般过程和主要挑战。然后，我们通过将其分类为四个关键领域来介绍现有的MRS模型：特征提取，编码器，多模式融合和损耗功能。最后，我们进一步讨论了开发和增强MRS的潜在未来方向。这项调查是MRS Field Mrs Field的研究人员和从业人员的综合指南，提供了有关MRS技术现状的见解，并确定了未来研究的领域。我们希望有助于开发更复杂和有效的多模式推荐系统。为了访问本文的更多详细信息，我们开源一个存储库：https：//github.com/jinfeng-xu/awesome-multimodal-recommender-systems。

### Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations 
[[arxiv](https://arxiv.org/abs/2502.15542)] [[cool](https://papers.cool/arxiv/2502.15542)] [[pdf](https://arxiv.org/pdf/2502.15542)]
> **Authors**: Wenyu Zhang,Jie Luo,Xinming Zhang,Yuan Fang
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 审计的多峰模型和建议之间的桥接域间隙
- **领域**: 信息检索,人工智能
- **摘要**: 随着多模式含量在线的爆炸性增长，预训练的视觉语言模型已显示出多模式推荐的巨大潜力。但是，尽管这些模型在以冷冻方式应用时实现了不错的性能，但令人惊讶的是，由于较大的领域差距（例如，特征分配差异和任务目标未对准）在预训练和个性化建议之间，采用联合培训方法会导致比基线差的绩效。现有方法要么依赖简单的功能提取，要么需要计算昂贵的完整模型微调，以平衡有效性和效率。为了应对这些挑战，我们提出\ textbf {p}芳香级\ textbf {t}对\ textbf {m} ult-imodal \ textbf {rec} ommendation（\ textbf {\ textbf {ptmrec}），通过新颖的框架进行跨模型和domain模型之间的建议，参数有效培训策略。该框架不仅消除了对昂贵的额外预训练的需求，而且还可以灵活地适应各种参数有效的调整方法。

### Multimodal Search in Chemical Documents and Reactions 
[[arxiv](https://arxiv.org/abs/2502.16865)] [[cool](https://papers.cool/arxiv/2502.16865)] [[pdf](https://arxiv.org/pdf/2502.16865)]
> **Authors**: Ayush Kumar Shah,Abhisek Dey,Leo Luo,Bryan Amador,Patrick Philippy,Ming Zhong,Siru Ouyang,David Mark Friday,David Bianchi,Nick Jackson,Richard Zanibbi,Jiawei Han
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: 4 pages, 2 figures, SIGIR 2025 Demonstration Submission
- **标题**: 化学文档和反应中的多模式搜索
- **领域**: 信息检索
- **摘要**: 我们提出了一种多模式搜索工具，可促进从科学文献中检索化学反应，分子结构和相关文本的检索。查询可以结合分子图，文本描述和反应数据，从而使用户可以连接化学信息的不同表示。为了支持这一点，索引过程包括化学图提取和解析，从表格形式中提取反应数据以及图表的跨模式链接及其在文本中的提及。我们描述了系统的架构，关键功能和检索过程，以及对系统的专家评估。该演示强调了搜索系统的工作流程和技术组件。

### MDE: Modality Discrimination Enhancement for Multi-modal Recommendation 
[[arxiv](https://arxiv.org/abs/2502.18481)] [[cool](https://papers.cool/arxiv/2502.18481)] [[pdf](https://arxiv.org/pdf/2502.18481)]
> **Authors**: Hang Zhou,Yucheng Wang,Huijing Zhan
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: MDE：多模式建议的模态歧视增强
- **领域**: 信息检索,人工智能
- **摘要**: 多模式推荐系统旨在通过将项目的内容功能与用户行为数据集成在一起，从而提高性能。有效利用来自不同方式的特征需要解决两个挑战：在模态跨模态（模态共享）之间保存语义共同点，并为每种模态（特定于模态）捕获独特的特征。大多数现有方法都集中在跨模态的特征空间上，这有助于表示模态共享的特征。但是，通常会忽略特定于模式的区别，尤其是在模态之间存在显着的语义变化时。为了解决这个问题，我们提出了一种模式独特性增强（MDE）框架，该框架优先提取特定于模式的信息，以提高建议精度，同时保持共同的功能。 MDE通过新型的多模式融合模块增强了跨模态的差异，并引入了节点级的权衡机制，以平衡跨模式的对准和分化。在三个公共数据集上进行的广泛实验表明，我们的方法极大地胜过其他最先进的方法，证明了共同考虑模式共享和特定于模态特征的有效性。

### CS-PaperSum: A Large-Scale Dataset of AI-Generated Summaries for Scientific Papers 
[[arxiv](https://arxiv.org/abs/2502.20582)] [[cool](https://papers.cool/arxiv/2502.20582)] [[pdf](https://arxiv.org/pdf/2502.20582)]
> **Authors**: Javin Liu,Aryan Vats,Zihao He
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: CS-Papersum：AI生成的科学论文摘要的大规模数据集
- **领域**: 信息检索
- **摘要**: 计算机科学中科学文献的快速扩展在跟踪研究趋势和提取关键见解方面提出了挑战。现有数据集提供元数据，但缺乏捕获核心贡献和方法论的结构化摘要。我们介绍了CS-Papersum，这是一个来自31个顶级计算机科学会议的91,919篇论文的大规模数据集，并使用Chatgpt富含AI生成的结构化摘要。为了评估摘要质量，我们进行了嵌入对准分析和关键字重叠分析，证明了关键概念的强烈保存。我们进一步介绍了有关AI研究趋势的案例研究，强调了方法论和跨学科跨界的转变，包括自我监督学习的兴起，检索效果的一代和多模式AI。我们的数据集可实现自动文献分析，研究趋势预测以及AI驱动的科学发现，为研究人员，政策制定者和科学信息检索系统提供了宝贵的资源。

## 信息论(cs.IT:Information Theory)

该领域共有 1 篇论文

### Token Communications: A Unified Framework for Cross-modal Context-aware Semantic Communications 
[[arxiv](https://arxiv.org/abs/2502.12096)] [[cool](https://papers.cool/arxiv/2502.12096)] [[pdf](https://arxiv.org/pdf/2502.12096)]
> **Authors**: Li Qiao,Mahdi Boloursaz Mashhadi,Zhen Gao,Rahim Tafazolli,Mehdi Bennis,Dusit Niyato
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 令牌通信：跨模式上下文感知语义通信的统一框架
- **领域**: 信息论,计算机视觉和模式识别,多媒体,信号处理
- **摘要**: 在本文中，我们介绍了代币通信（Tokcom），这是一个统一的框架，用于利用生成语义通信（GENSC）中的跨模式上下文信息。 Tokcom是一个新的范式，是由生成基础模型和多模式大语模型（GFM/MLLM）的最新成功的促进的，在该模型中，通信单元是令牌，从而在发射器和接收器上实现了有效的基于变压器的代币处理。在本文中，我们介绍了在GENSC中利用环境的潜在机会和挑战，探讨了如何将基于GFM/MLLMS的令牌处理整合到语义通信系统中，以有效利用跨模式环境，介绍在未来无线网络中各个层的有效Tokcom的关键原则。我们在GENSC设置中证明了相应的TOKCOM优势用于图像，利用跨模式上下文信息，这将带宽效率提高了70.8％，而语义/感知质量的丧失却可以忽略不计。最后，确定了潜在的研究方向，以促进未来无线网络中Tokcom的采用。

## 机器学习(cs.LG:Machine Learning)

该领域共有 105 篇论文

### CAAT-EHR: Cross-Attentional Autoregressive Transformer for Multimodal Electronic Health Record Embeddings 
[[arxiv](https://arxiv.org/abs/2501.18891)] [[cool](https://papers.cool/arxiv/2501.18891)] [[pdf](https://arxiv.org/pdf/2501.18891)]
> **Authors**: Mohammad Al Olaimat,Serdar Bozdag
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: CAAT-EHR：多模式电子健康记录嵌入的跨注意自动回旋变压器
- **领域**: 机器学习
- **摘要**: 电子健康记录（EHRS）提供了纵向患者数据的全面来源，包括结构化模式，例如实验室结果，成像数据和生命体征以及非结构化的临床注释。这些数据集在必要的预处理以清洁和格式化数据以进行分析之后，通常保持其原始EHR形式，代表数值或分类值，而无需进一步转换为任务不合时宜的嵌入。尽管这种原始EHR数据可以实现预测性建模，但其对手动功能工程或下游任务特定优化的依赖限制了其用于通用应用程序的实用性。深度学习（DL）技术，例如复发性神经网络（RNN）和变压器，具有促进疾病进展和诊断预测等预测任务。但是，这些方法通常很难完全利用EHR数据中固有的时间和多模式依赖性，因为它们依赖预处理但未转换的原始EHR输入。在这项研究中，我们介绍了CAAT-EHR，这是一种新型的架构，旨在通过从RAW EHR数据中产生坚固的，任务不可能的纵向嵌入来弥合这一差距。 CAAT-EHR利用其编码器中的自我和跨注意机制来整合多种模态的时间和上下文关系，将数据转换为捕获复杂依赖性的丰富嵌入。自回归解码器通过预测预训练期间的未来时间点数据来补充编码器，从而确保所得的嵌入保持时间一致性和对齐。 CAAT-EHR消除了对手动功能工程的需求，并在不同的下游任务中实现了无缝的可传递性。对基准数据集进行了广泛的评估，证明了CAAT-EHR生成的嵌入比预处理的原始EHR数据和其他基线方法的优越性。

### MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving 
[[arxiv](https://arxiv.org/abs/2502.01960)] [[cool](https://papers.cool/arxiv/2502.01960)] [[pdf](https://arxiv.org/pdf/2502.01960)]
> **Authors**: Shiju Zhao,Junhao Hu,Rongxiao Huang,Jiaqi Zheng,Guihai Chen
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: 14 pages, 11 figures, the first version
- **标题**: MPIC：与位置无关的多模式上下文缓存系统，用于有效的MLLM服务
- **领域**: 机器学习
- **摘要**: 上下文缓存技术用于通过当前流行的服务平台来加速多模式大语言模型（MLLM）。但是，这种方法仅重用迅速的初始序列的键值（KV）缓存，即使前缀略有不同，也会导致完整的KV缓存重新组件。这在交织的文本和图像以及多模式检索的生成中尤其低。本文提出与位置无关的缓存，作为多模式信息管理的更有效方法。我们已经设计并实施了一个名为MPIC的缓存系统，以解决系统级和算法级别的挑战。接收多模式数据时，MPIC将KV缓存存储在本地或远程磁盘上，并在推断过程中并行计算和加载KV缓存。为了减轻准确性降解，我们将集成的重用并重新计算了系统中的机制。实验结果表明，与现有上下文缓存系统相比，MPIC可以减少多达54％的响应时间，同时保持可忽略不计或准确性损失。

### Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection 
[[arxiv](https://arxiv.org/abs/2502.01699)] [[cool](https://papers.cool/arxiv/2502.01699)] [[pdf](https://arxiv.org/pdf/2502.01699)]
> **Authors**: Tianlin Zhang,En Yu,Yi Shao,Shuai Li,Sujuan Hou,Jiande Sun
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 多模式反向注意网络具有固有的判别功能开发用于假新闻检测
- **领域**: 机器学习,计算语言学,计算机视觉和模式识别,信息检索,多媒体
- **摘要**: 多式联运的假新闻检测因其对社会保障的深刻影响而引起了人们的重大关注。尽管现有的方法有助于理解跨模式的一致性，但它们通常无法利用模态特定的表示和明确的差异功能。为了解决这些限制，我们提出了一个多模式反向注意网络（Mian），这是一个新颖的框架，该框架探索了基于新闻内容的固有判别特征，以推动假新闻检测。具体而言，Mian引入了一个分层学习模块，该模块通过局部到全球和本地互动来捕获多样化的模式内关系，从而产生增强的单峰表示形式，以改善模式内假新闻的识别。此外，跨模式相互作用模块还采用共同注意机制来建立和建模精制的单峰表示之间的依赖性，从而促进了跨模态的无缝语义整合。为了明确提取不一致的特征，我们提出了一种反向注意机制，该机制有效地强调了假新闻在内部和模式间中引入的矛盾模式和语义偏差。基准数据集的广泛实验表明，Mian的表现明显胜过最先进的方法，强调了其通过增强的多模式假新闻检测来推进社会安全的关键贡献。

### Position: Empowering Time Series Reasoning with Multimodal LLMs 
[[arxiv](https://arxiv.org/abs/2502.01477)] [[cool](https://papers.cool/arxiv/2502.01477)] [[pdf](https://arxiv.org/pdf/2502.01477)]
> **Authors**: Yaxuan Kong,Yiyuan Yang,Shiyu Wang,Chenghao Liu,Yuxuan Liang,Ming Jin,Stefan Zohren,Dan Pei,Yan Liu,Qingsong Wen
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 位置：使用多模式LLM的授权时间序列推理
- **领域**: 机器学习,人工智能
- **摘要**: 了解时间序列数据对于多个现实世界应用至关重要。尽管大型语言模型（LLMS）在时间序列任务中表现出希望，但当前的方法通常仅依靠数值数据，忽略了与时间相关信息的多模式性质，例如文本说明，视觉数据和音频信号。此外，这些方法不足以LLMS的推理能力，将分析限制为表面级别的解释，而不是更深的时间和多模式推理。在该立场论文中，我们认为多模式LLM（MLLM）可以为时间序列分析提供更强大和灵活的推理，增强决策和现实世界的应用程序。我们呼吁研究人员和从业人员通过制定优先级信任，解释性和强大推理的策略来利用这一潜力。最后，我们重点介绍了关键的研究方向，包括新的推理范式，建筑创新和特定于领域的应用程序，以推进MLLM的时间序列推理。

### MIND: Modality-Informed Knowledge Distillation Framework for Multimodal Clinical Prediction Tasks 
[[arxiv](https://arxiv.org/abs/2502.01158)] [[cool](https://papers.cool/arxiv/2502.01158)] [[pdf](https://arxiv.org/pdf/2502.01158)]
> **Authors**: Alejandro Guerra-Manzanares,Farah E. Shamout
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: Published in Transactions on Machine Learning Research (01/2025), https://openreview.net/forum?id=BhOJreYmur&noteId=ymnAhncuez
- **标题**: 思维：多模式临床预测任务的模态知识蒸馏框架
- **领域**: 机器学习,人工智能,计算机视觉和模式识别
- **摘要**: 多模式融合利用跨模式的信息来学习更好的功能表示，目的是改善基于融合的任务的性能。但是，多模式数据集，尤其是在医疗设置中，通常比单峰同行小，这可能会阻碍多模型的性能。此外，模式数量的增加通常与多模式网络大小的总体增加有关，这在医疗用例中可能是不希望的。利用较小的单峰编码器可能会导致次优性能，尤其是在处理高维临床数据时。在本文中，我们提出了基于知识蒸馏的多模型压缩方法的模式知识知识蒸馏（思维）框架，该方法将知识从预训练的不同大小的预训练的深神经网络转移到较小的多模式学生中。教师模型由单峰网络组成，使学生可以从各种表示形式中学习。 Mind采用多头关节融合模型，而不是单头模型，在单模型样品的情况下，可以使用单峰编码器，而无需插补或掩盖缺乏模态。结果，Mind生成了优化的多模式模型，增强了多模式和单峰表示。在培训期间，它也可以利用以平衡多模式学习。我们使用时间序列数据和胸部X射线图像评估二进制和多标签临床预测任务的思维。此外，我们评估了三个非医学多模式多类数据集上思维框架的普遍性。实验结果表明，与最先进的基线相比，心灵在所有五个任务以及各种融合方法和多模式体系结构中提高了较小的多模式网络的性能。

### Continuous Autoregressive Modeling with Stochastic Monotonic Alignment for Speech Synthesis 
[[arxiv](https://arxiv.org/abs/2502.01084)] [[cool](https://papers.cool/arxiv/2502.01084)] [[pdf](https://arxiv.org/pdf/2502.01084)]
> **Authors**: Weiwei Lin,Chenghan He
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: ICLR 2025
- **标题**: 连续自回归建模与随机单调对准语音合成
- **领域**: 机器学习,声音,音频和语音处理
- **摘要**: 我们提出了一种新型的自动建模方法，用于语音合成，将各种自动编码器（VAE）与多模式潜在空间和使用高斯混合模型（GMM）作为条件概率分布相结合。与以前依赖残留向量量化的方法不同，我们的模型利用了VAE潜在空间的连续语音表示，从而大大简化了训练和推理管道。我们还引入了一种随机的单调对准机制来强制执行严格的单调对准。我们的方法在主观和客观评估中大大优于最先进的自回归模型VALL-E，仅用Vall-E参数的10.3％实现这些结果。这证明了连续语言模型作为现有基于量化语音语言模型的更有效替代方案的潜力。可以在https://tinyurl.com/gmm-lm-tts上找到示例音频。

### UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs 
[[arxiv](https://arxiv.org/abs/2502.00806)] [[cool](https://papers.cool/arxiv/2502.00806)] [[pdf](https://arxiv.org/pdf/2502.00806)]
> **Authors**: Yufei He,Yuan Sui,Xiaoxin He,Yue Liu,Yifei Sun,Bryan Hooi
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: WWW 2025
- **标题**: Unigraph2：学习一个统一的嵌入空间以绑定多模式图
- **领域**: 机器学习
- **摘要**: 现有的基础模型（例如剪辑）旨在学习用于多模式数据的统一嵌入空间，从而实现了诸如搜索，建议和内容分类等广泛的下游基于Web的应用程序。但是，这些模型通常忽略多模式数据集中的固有图形结构，在该数据集中，实体及其关系至关重要。多模式图（MMG）代表这样的图形，其中每个节点与不同模态的特征相关联，而边缘捕获这些实体之间的关系。另一方面，现有的图形基础模型主要集中在文本属性图（TAG）上，并非旨在处理MMG的复杂性。为了解决这些局限性，我们提出了Unigraph2，这是一种新型的跨域图基础模型，可实现对MMG的一般表示，提供统一的嵌入空间。 Unigraph2与图形神经网络（GNN）一起使用特定于模态的编码器，以学习统一的低维嵌入空间，该空间同时捕获了多模式信息和基础图结构。我们提出了一种大规模的新的跨域多绘图预训练算法，以确保在不同的图形域和模态之间进行有效的转移学习。此外，我们采用专家（MOE）组件的混合物来对齐来自不同领域和模态的特征，从而确保连贯且稳健的嵌入，从而统一跨模态的信息。对各种多模式图任务进行的广泛实验表明，Unigraph2在诸如表示学习，转移学习和多模式生成任务之类的任务中的最先进模型大大优于最先进的模型，为MMGS提供了可扩展和灵活的解决方案。

### Understanding and Mitigating the High Computational Cost in Path Data Diffusion 
[[arxiv](https://arxiv.org/abs/2502.00725)] [[cool](https://papers.cool/arxiv/2502.00725)] [[pdf](https://arxiv.org/pdf/2502.00725)]
> **Authors**: Dingyuan Shi,Lulu Zhang,Yongxin Tong,Ke Xu
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: 16 pages
- **标题**: 了解和减轻路径数据扩散中的高计算成本
- **领域**: 机器学习
- **摘要**: 移动服务，导航系统和智能运输技术的进步使收集大量路径数据成为可能。建模该路径数据的分布（称为路径生成（PG）问题）对于理解城市流动性模式和开发智能运输系统至关重要。最近的研究探索了使用扩散模型来解决PG问题，因为它们能够捕获多模式分布和支持条件产生。最近的工作在图形空间中明确设计了一个扩散过程，并实现了最新的性能。但是，此方法在时间和内存方面都遭受高计算成本，这禁止其应用。在本文中，我们在理论上和实验上分析了这种方法，发现其高计算成本的主要罪魁祸首是其在图形空间中扩散过程的明确设计。为了提高效率，我们设计了一个潜在空间路径扩散（LPD）模型，该模型在潜在空间而不是图形空间中运行。我们的LPD将时间和记忆成本大幅降低多达82.8％和83.1％。尽管有这些减少，但我们的方法并未遭受性能降解的困扰。在大多数情况下，它的表现要优于最先进的方法24.5％〜34.0％。

### "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models 
[[arxiv](https://arxiv.org/abs/2502.00718)] [[cool](https://papers.cool/arxiv/2502.00718)] [[pdf](https://arxiv.org/pdf/2502.00718)]
> **Authors**: Isha Gupta,David Khachaturov,Robert Mullins
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: “我很糟糕”：在音频模型中解释隐形，通用和强大的音频越狱
- **领域**: 机器学习,声音,音频和语音处理
- **摘要**: 多模式大语言模型的兴起引入了创新的人机相互作用范式，但在机器学习安全方面也面临重大挑战。由于口语交流的直观性质，音频模型（ALM）特别相关，但对它们的故障模式知之甚少。本文探讨了针对施舍的音频越狱，重点是绕过对齐机制的能力。我们构建了对抗性扰动，这些扰动跨越提示，任务甚至基本音频样本，证明了音频方式中的第一个通用越狱，并表明这些在模拟现实世界中仍然有效。除了证明攻击可行性外，我们还分析了施舍如何解释这些音频对抗性示例，并揭示它们以编码不可察觉的第一人称有毒语音 - 这表明最有效的扰动是在音频信号中引起特定嵌入语言特征的有毒输出。这些结果对于理解多模式模型中不同方式之间的相互作用具有重要意义，并提供了可行的见解，以增强防御对抗性音频攻击的防御能力。

### Generic Multimodal Spatially Graph Network for Spatially Embedded Network Representation Learning 
[[arxiv](https://arxiv.org/abs/2502.00530)] [[cool](https://papers.cool/arxiv/2502.00530)] [[pdf](https://arxiv.org/pdf/2502.00530)]
> **Authors**: Xudong Fan,Jürgen Hackl
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 通用多模式的空间图网络，用于空间嵌入式网络表示学习
- **领域**: 机器学习,人工智能,社交和信息网络
- **摘要**: 空间嵌入式网络（SENS）代表一种特殊类型的复杂图，其拓扑受到网络嵌入式空间环境的约束。因此，此类网络的图表示受节点和边缘的嵌入式空间特征的影响。图形结构和图形特征的准确网络表示是各种与图形相关任务的基本任务。在这项研究中，开发了通用多模式的空间图卷积网络（GMU-SGCN），以有效地表示空间嵌入式网络。开发的GMU-SGCN模型具有通过多模式节点和边缘功能学习节点连接模式的能力。为了评估开发的模型，河网络数据集和电源网络数据集已被用作测试床。河网络代表自然发展的感官，而电源网络代表人造网络。两种类型的网络都受到自然界的空间环境和不确定性的严重限制。全面的评估分析表明，与仅考虑节点网络测试床中节点的位置特征相比，开发的GMU-SGCN可以提高边缘存在预测任务的准确性37.1 \％。我们的模型展示了考虑空间嵌入式网络表示的多维空间特征的重要性。

### Mordal: Automated Pretrained Model Selection for Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.00241)] [[cool](https://papers.cool/arxiv/2502.00241)] [[pdf](https://arxiv.org/pdf/2502.00241)]
> **Authors**: Shiqi He,Insu Jang,Mosharaf Chowdhury
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: Mordal：视觉语言模型的自动预验证模型选择
- **领域**: 机器学习,人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 将多种模式纳入大语言模型（LLMS）是增强他们对非文本数据的理解，使他们能够执行多模式任务的有力方法。视觉语言模型（VLM）构成了多模型增长最快的类别，因为它们的许多实际用例，包括医疗保健，机器人技术和可访问性。不幸的是，尽管文献中不同的VLM在不同的基准测试中表现出令人印象深刻的视觉功能，但它们还是由人类专家手工制作的。没有自动化的框架来创建特定于任务的多模式模型。我们介绍了Mordal，这是一种自动多模型搜索框架，可有效找到无需手动干预即可提供用户定义任务的最佳VLM。 Mordal通过减少在搜索过程中要考虑的候选人的数量以及最大程度地减少评估每个剩余候选人所需的时间来实现这一点。我们的评估表明，Mordal可以使用最高$ 8.9 \ times $  -  $ 11.6 \ times $ $ gpu小时比GRID搜索找到最佳的VLM。在我们的评估过程中，我们还发现了新的VLM，以优于其最先进的VLM。

### Mol-LLM: Generalist Molecular LLM with Improved Graph Utilization 
[[arxiv](https://arxiv.org/abs/2502.02810)] [[cool](https://papers.cool/arxiv/2502.02810)] [[pdf](https://arxiv.org/pdf/2502.02810)]
> **Authors**: Chanhui Lee,Yuheon Song,YongJun Jeong,Hanbum Ko,Rodrigo Hormazabal,Sehui Han,Kyunghoon Bae,Sungbin Lim,Sungwoong Kim
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: mol-llm：通才分子LLM具有改进的图形利用率
- **领域**: 机器学习,人工智能,化学物理,生物分子
- **摘要**: 大型语言模型（LLM）的最新进展激发了分子任务的一般LLM的发展。尽管几项研究表明，微调的LLM可以实现令人印象深刻的基准表现，但由于缺乏对分子结构的基本了解，它们远非真正的通才分子LLM。具体而言，当鉴于分子任务指示时，接受了天真的下一步预测训练训练的LLM为原始和负损坏的分子分配了类似的似然评分，这表明他们缺乏分子结构的理解，这对于可靠和一般的分子LLM至关重要。为了克服这一局限性并获得真正的通才分子LLM，我们基于彻底的多模式教学调整以及所选图和拒绝图之间的分子结构偏好优化，引入了一种新型的多模式训练方法。在各种分子基准上，所提出的称为Mol-LLM的通才分子LLM在大多数任务上在大多数任务中都达到了最先进的表现，同时超过或与最先进的专家LLM相当。此外，mol-llm在反应预测任务中还显示出了出色的泛化性能，这证明了分子结构理解对概括的透视的影响。

### Federated Low-Rank Tensor Estimation for Multimodal Image Reconstruction 
[[arxiv](https://arxiv.org/abs/2502.02761)] [[cool](https://papers.cool/arxiv/2502.02761)] [[pdf](https://arxiv.org/pdf/2502.02761)]
> **Authors**: Anh Van Nguyen,Diego Klabjan,Minseok Ryu,Kibaek Kim,Zichao Di
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 多模式图像重建的联合低级张量估计
- **领域**: 机器学习,计算机视觉和模式识别,分布式、并行和集群计算
- **摘要**: 低量张量估计提供了一种有力的方法来解决高维数据挑战，并可以实质上改善解决方案的解决方案，例如在噪声或不足采样条件下的图像重建方面。同时，由于其在利用潜在空间结构及其提高沟通效率的能力方面的有效性，张量分解在联邦学习（FL）方面已获得突出。在本文中，我们提出了一种联合图像重建方法，该方法应用了塔克分解，结合了关节分解和随机素描来管理大规模的多模式数据。我们的方法避免重建全尺寸张量并支持异质等级，从而使客户可以根据先验知识或沟通能力选择个性化的分解等级。数值结果表明，与现有方法相比，我们的方法实现了优越的重建质量和通信压缩，从而突出了其在FL环境中多模式反问题的潜力。

### Vision-Language Model Dialog Games for Self-Improvement 
[[arxiv](https://arxiv.org/abs/2502.02740)] [[cool](https://papers.cool/arxiv/2502.02740)] [[pdf](https://arxiv.org/pdf/2502.02740)]
> **Authors**: Ksenia Konyushkova,Christos Kaplanis,Serkan Cabi,Misha Denil
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 自我完善的视觉语言模型对话框游戏
- **领域**: 机器学习,人工智能
- **摘要**: 对高质量，多样化的培训数据的需求不断增长，在推进视觉模型（VLMS）方面构成了重要的瓶颈。本文介绍了VLM对话游戏，这是VLM的新颖且可扩展的自我完善框架。我们的方法利用了以目标标识为中心的两名代理商之间的自我玩法。通过过滤成功的游戏交互，我们将自动策划一个相互交织的图像和文本的高质量数据集。我们证明，对此合成数据进行微调会导致在数据集各个数据集的下游任务和一般性上的性能增长。此外，随着模型的改进会导致更好的游戏玩法，可以迭代地应用此过程。这项工作为自我改善的VLM铺平了道路，在各种现实世界中的潜在应用，尤其是当稀缺的高质量多模式数据时。

### MedRAX: Medical Reasoning Agent for Chest X-ray 
[[arxiv](https://arxiv.org/abs/2502.02673)] [[cool](https://papers.cool/arxiv/2502.02673)] [[pdf](https://arxiv.org/pdf/2502.02673)]
> **Authors**: Adibvafa Fallahpour,Jun Ma,Alif Munim,Hongwei Lyu,Bo Wang
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: 11 pages, 4 figures, 2 tables
- **标题**: Medrax：胸部X射线的医疗推理剂
- **领域**: 机器学习,人工智能,多代理系统
- **摘要**: 胸部X射线（CXR）在推动疾病管理和患者护理方面的关键决策中起着不可或缺的作用。尽管最近的创新导致了针对各种CXR解释任务的专门模型，但这些解决方案通常是孤立地运行的，从而限制了它们在临床实践中的实际实用性。我们提出了MedRax，这是第一个无缝将最先进的CXR分析工具和多模式大语言模型集成到统一框架中的多功能AI代理。 Medrax动态地利用这些模型来解决复杂的医疗查询，而无需进行额外的培训。为了严格评估其功能，我们介绍了ChestAgentBench，这是一个全面的基准测试，其中包含7种不同类别的2,500个复杂的医疗查询。我们的实验表明，与开源和专有模型相比，MEDRAX实现了最先进的性能，这代表了自动化CXR解释系统的实际部署迈出的重要一步。数据和代码已在https://github.com/bowang-lab/medrax上公开获取

### Efficient Domain Adaptation of Multimodal Embeddings using Constrastive Learning 
[[arxiv](https://arxiv.org/abs/2502.02048)] [[cool](https://papers.cool/arxiv/2502.02048)] [[pdf](https://arxiv.org/pdf/2502.02048)]
> **Authors**: Georgios Margaritis,Periklis Petridis,Dimitris J. Bertsimas
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 使用对比度学习对多模式嵌入的有效域适应
- **领域**: 机器学习,计算语言学,计算机视觉和模式识别
- **摘要**: 机器学习（ML），自然语言处理（NLP）和基础模型的最新进展已显示出对关键（尽管是Compute consute consute conconconconconconconconconconconconte consute consute consute consute consute consute consute consute consute consute consute consearthe shealthcare诸如医疗保健）的希望。在这样的领域，将基础模型与监督的ML相结合，可以自动化诊断和治疗计划等任务，但是在有效应用这些技术之前，现场计算资源的可用性有限地提出了巨大的挑战：当前的方法要么在使用特定任务适应的情况下使用预审计的模型，或者需要针对特定​​的计算资源，或者需要实质性的计算资源来实现较高的态度，而这些模型通常会进入较高的环境，因此可以进行此类调整。这使它们在性能和质量标准很高的应用中无法访问，但计算资源很少。为了弥合一流的性能和可访问性之间的差距，我们提出了一种新颖的方法，可以将基础，多模式嵌入到下游任务，而无需昂贵的微调过程。我们的方法利用了大型语言模型（LLM）和视觉模型的冷冻嵌入，并使用对比度学习来训练可以在下游任务中使用的小型，特定于任务的非线性投影，而无需微调原始的基础模型。我们表明，这种有效的过程会导致各种下游任务的大大改进，并且更重要的是，使用最少的计算开销，为在资源受限设置中使用高级基础ML模型提供了一种实用的解决方案。

### How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies 
[[arxiv](https://arxiv.org/abs/2502.03698)] [[cool](https://papers.cool/arxiv/2502.03698)] [[pdf](https://arxiv.org/pdf/2502.03698)]
> **Authors**: Basavasagar Patil,Akansha Kalra,Guanhong Tao,Daniel S. Brown
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 我的政策有多脆弱？对现代行为克隆政策的对抗性攻击
- **领域**: 机器学习,密码学和安全,机器人技术
- **摘要**: 从示范中学习（LFD）算法在机器人操纵任务中显示出令人鼓舞的结果，但是它们对对抗性攻击的脆弱性仍然没有得到充实。本文介绍了对经典和最近提出的算法的对抗性攻击的全面研究，包括行为克隆（BC），LSTM-GMM，隐式行为克隆（IBC），扩散策略（DP）和VQ-Behavior Transformer（VQ-BET）。我们研究了这些方法对不靶向，有针对性和普遍的对抗扰动的脆弱性。尽管可以以与标准计算机视觉模型相同的方式攻击诸如BC，LSTM-GMM和VQ-BET之类的显式政策，但我们发现，隐式和DeNoising策略模型的攻击是细微的，需要开发新颖的攻击方法。我们对几个模拟机器人操纵任务的实验表明，当前大多数方法都非常容易受到对抗扰动的影响。我们还表明，这些攻击是可以在算法，架构和任务中转移的，从而提高了有关使用白框威胁模型的安全漏洞的提高。此外，我们测试了一种随机平滑，一种广泛使用的对抗防御技术的功效，并强调了其在捍卫对复杂和多模式的动作分布攻击中的限制，在复杂控制任务中常见。总而言之，我们的发现突出了现代卑诗省算法的脆弱性，铺平了为解决此类局限的未来工作的方式。

### DocMIA: Document-Level Membership Inference Attacks against DocVQA Models 
[[arxiv](https://arxiv.org/abs/2502.03692)] [[cool](https://papers.cool/arxiv/2502.03692)] [[pdf](https://arxiv.org/pdf/2502.03692)]
> **Authors**: Khanh Nguyen,Raouf Kerkouche,Mario Fritz,Dimosthenis Karatzas
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: ICLR 2025
- **标题**: DOCMIA：对DOCVQA模型的文档级会员推理攻击
- **领域**: 机器学习,计算语言学,密码学和安全
- **摘要**: 文档视觉问题回答（DOCVQA）引入了一种新的范式，以了解端到端文档的理解，并迅速成为多模式LLM的标准基准之一。由DOCVQA模型驱动的文档处理工作流程自动化的工作流程为许多业务领域带来了巨大的潜力。但是，文件倾向于包含高度敏感的信息，从而引起了对培训此类DOCVQA模型相关的隐私风险的担忧。会员推理攻击所利用的一个重要隐私脆弱性是对手可以确定特定记录是否是模型培训数据的一部分。在本文中，我们介绍了专门针对DOCVQA模型量身定制的两项新型会员推理攻击。这些攻击是针对两个不同的对抗场景设计的：一个白色框设置，攻击者可以完全访问模型体系结构和参数，以及一个黑框设置，其中只有模型的输出可用。值得注意的是，我们的攻击假设对手缺乏获得辅助数据集的访问，这在实践中更现实，但更具挑战性。我们无监督的方法优于各种DOCVQA模型和数据集的现有最新成员推理攻击，证明了它们的有效性并突出了该域中的隐私风险。

### Scaling laws in wearable human activity recognition 
[[arxiv](https://arxiv.org/abs/2502.03364)] [[cool](https://papers.cool/arxiv/2502.03364)] [[pdf](https://arxiv.org/pdf/2502.03364)]
> **Authors**: Tom Hoddes,Alex Bijamov,Saket Joshi,Daniel Roggen,Ali Etemad,Robert Harle,David Racz
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: 可穿戴人类活动识别中的缩放法律
- **领域**: 机器学习
- **摘要**: 已经提出了许多深层建筑和自我监督的预训练技术，以通过可穿戴多模式传感器的人类活动识别（HAR）。缩放定律有可能通过将模型容量与培训前数据量联系起来，从而帮助采取更有原则的设计。然而，尚未以与语言和愿景相同的程度建立缩放法律。通过对培训前数据和变压器体系结构进行详尽的网格搜索，我们为HAR建立了第一个已知的缩放定律。我们表明，与数据和参数数量的功率法关系具有与数据集数量的量相关的训练损失量表，并且数据集中的用户数量的增加会导致性能的提高比每位用户增加数据的幅度更大，这表明预训练数据的多样性很重要，这与自我耐受性的HAR中一些先前报道的发现相反。我们表明，这些扩展定律转化为三个HAR基准数据集的下游性能改进，运动方式和日常生活的活动：UCI HAR和WISDM Phone以及Wisdm Watch。最后，我们建议根据这些规模定律具有更高的模型能力，应重新审视一些先前发表的作品。

### CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements 
[[arxiv](https://arxiv.org/abs/2502.04592)] [[cool](https://papers.cool/arxiv/2502.04592)] [[pdf](https://arxiv.org/pdf/2502.04592)]
> **Authors**: Yang Zhang,Wenbo Yang,Jun Wang,Qiang Ma,Jie Xiong
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: COAMF：通过整合时间序列模式和显着的宏观经济公告，以因果关系驱动事件驱动的多模式驱动的财务预测
- **领域**: 机器学习,人工智能,计算工程、金融和科学
- **摘要**: 准确地预测宏观经济事件的影响对于投资者和政策制定者至关重要。诸如货币政策决策和就业报告之类的显着事件通常通过塑造对经济增长和风险的期望来触发市场的发展，从而在事件与市场行为之间建立因果关系。现有的预测方法通常集中于文本分析或时间序列建模，但无法捕获金融市场的多模式性质以及事件和价格变动之间的因果关系。为了解决这些差距，我们提出了COMEF（因果关系多模式事件驱动的财务预测），这是一个多模式框架，可有效地将文本和时间序列数据与因果学习机制以及基于LLM的基于LLM的相对事件增强技术，以实现Causal-Exhance-Exhance-Enallance-Enally Hanthancanced Finance Financed Finance Felinessing。我们的贡献包括：（1）一个多模式框架，该框架捕获了政策文本和历史价格数据之间的因果关系； （2）一个新的金融数据集，该数据集具有从2008年到2024年4月的六种宏观经济发行，以及五个主要的美国金融资产的高频实际交易数据； （3）基于LLM的反事实事件增强策略。我们将CAREF与最先进的变压器时间序列和多模式基线进行比较，并进行消融研究以验证因果学习机制和事件类型的有效性。

### Transforming Multimodal Models into Action Models for Radiotherapy 
[[arxiv](https://arxiv.org/abs/2502.04408)] [[cool](https://papers.cool/arxiv/2502.04408)] [[pdf](https://arxiv.org/pdf/2502.04408)]
> **Authors**: Matteo Ferrante,Alessandra Carosi,Rolando Maria D Angelillo,Nicola Toschi
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 将多模式转换为放射疗法的动作模型
- **领域**: 机器学习,人工智能
- **摘要**: 放射治疗是一种至关重要的癌症治疗方法，需要精确的计划平衡消除肿瘤和保存健康组织。传统的治疗计划（TP）是迭代，耗时的，并且依赖于人类专业知识，这可能会引入可变性和效率低下。我们提出了一个新颖的框架，以使用一些射击强化学习（RL）方法将大型多模式基础模型（MLM）转变为TP的动作模型。我们的方法利用了MLM广泛的物理，放射和解剖学知识，从而通过几次学习过程来增强它。这允许模型使用蒙特卡洛模拟器迭代改进治疗计划。我们的结果表明，这种方法在质量和效率方面都优于基于RL的常规方法，在前列腺癌数据的模拟中获得了更高的奖励分数和更优化的剂量分布。该概念验证表明，将高级AI模型集成到临床工作流程中，有可能提高放射治疗计划的速度，质量和标准化。

### Adaptive Prototype Knowledge Transfer for Federated Learning with Mixed Modalities and Heterogeneous Tasks 
[[arxiv](https://arxiv.org/abs/2502.04400)] [[cool](https://papers.cool/arxiv/2502.04400)] [[pdf](https://arxiv.org/pdf/2502.04400)]
> **Authors**: Keke Gai,Mohan Wang,Jing Yu,Dongjue Wang,Qi Wu
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 具有混合方式和异质任务的联合学习的自适应原型知识转移
- **领域**: 机器学习,人工智能,密码学和安全,多媒体
- **摘要**: 多模式联合学习（MFL）使多个客户能够在确保客户的隐私的同时协作培训多模式数据的模型。但是，模式和任务异质性阻碍了客户学习统一表示，削弱了本地模型的概括，尤其是在MFL中，在具有混合方式的MFL中，只有某些客户只有多模式数据。在这项工作中，我们提出了一个基于自适应原型的多模式联合学习（APROMFL）框架，以解决混合方式和异质任务，以解决上述问题。我们的APOMFL通过没有以前的公共数据集的自适应构造的原型传递知识。客户端适应与任务一致的原型构建方法；服务器将客户端原型转换为统一的多模式原型，并将其汇总以形成全局原型，避免客户保留统一标签。我们将模型分为各种模块，仅汇总映射模块，以减少通信和计算开销。为了解决异质性的聚合问题，我们开发了一个基于客户关系图的方案，以动态调整聚合权重。代表性数据集的广泛实验APROMFL的证据有效性。

### MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation 
[[arxiv](https://arxiv.org/abs/2502.04176)] [[cool](https://papers.cool/arxiv/2502.04176)] [[pdf](https://arxiv.org/pdf/2502.04176)]
> **Authors**: Qinhan Yu,Zhiyou Xiao,Binghui Li,Zhengren Wang,Chong Chen,Wentao Zhang
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: 11 pages
- **标题**: MRAMG-BENCH：多模式检索多模式生成的超越文本基准
- **领域**: 机器学习,信息检索
- **摘要**: 通过将外部知识整合到生成模型中，在提高响应准确性和相关性方面表现出了出色的性能，在提高响应准确性和相关性方面表现出色。但是，现有的抹布方法主要集中于提供仅文本的答案，即使在多模式检索的生成场景中也是如此。在这项工作中，我们介绍了多模式检索仪的多模式生成（MRAMG）任务，该任务旨在生成结合文本和图像的答案，并完全利用语料库中的多模式数据。尽管这项任务很重要，但仍有明显的缺乏全面的基准来有效评估MRAMG性能。为了弥合这一差距，我们介绍了Mramg-Bench，这是一个经过精心策划的人类宣传的数据集，其中包括4,346个文档，14,190张图像和4,800个QA对，来自三个类别：网络数据，学术报纸，学术报纸和生活方式。数据集结合了各种难度级别和复杂的多图像场景，为评估多模式生成任务提供了强大的基础。为了促进严格的评估，我们的MRAMG-BENCH结合了统计和基于LLM的指标的全面套件，从而可以对MRAMG任务中流行生成模型的性能进行详尽的分析。此外，我们提出了一个有效的多模式答案生成框架，该框架利用LLM和MLLM来生成多模式响应。我们的数据集可在以下网址找到：https：//huggingface.co/mramg。

### Innovative Framework for Early Estimation of Mental Disorder Scores to Enable Timely Interventions 
[[arxiv](https://arxiv.org/abs/2502.03965)] [[cool](https://papers.cool/arxiv/2502.03965)] [[pdf](https://arxiv.org/pdf/2502.03965)]
> **Authors**: Himanshi Singh,Sadhana Tiwari,Sonali Agarwal,Ritesh Chandra,Sanjay Kumar Sonbhadra,Vrijendra Singh
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 创新框架以早期估算精神障碍得分以及时干预
- **领域**: 机器学习
- **摘要**: 个人的一般福祉受到心理健康状况的极大影响，包括抑郁症和创伤后应激障碍（PTSD），强调了早期检测的重要性和精确诊断的重要性，以促进迅速的临床干预。本文介绍了用于PTSD和抑郁症自动分类的先进的多模式深度学习系统。该方法利用临床访谈数据集中的文本和音频数据结合了从模式中获得的特征，通过结合LSTM（长期短期记忆）和Bilstm（双向长期短期记忆）的结构。音频具有捕获人声性状，包括节奏，音调和音调。这种方式的结合增强了模型确定与心理健康状况相关的微小模式的能力。使用测试数据集，所提出的方法可实现抑郁症的分类精度，而PTSD的分类精度为92％，表现优于传统的单峰方法，并证明其准确性和鲁棒性。

### Bridging the inference gap in Mutimodal Variational Autoencoders 
[[arxiv](https://arxiv.org/abs/2502.03952)] [[cool](https://papers.cool/arxiv/2502.03952)] [[pdf](https://arxiv.org/pdf/2502.03952)]
> **Authors**: Agathe Senellart,Stéphanie Allassonnière
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 桥接Mutimodal变异自动编码器中的推理差距
- **领域**: 机器学习,机器学习
- **摘要**: 从医疗诊断到自动驾驶汽车，关键应用依赖于多种异构数据模式的整合。多模式变分自动编码器提供了多功能和可扩展的方法，用于从观察到的模态产生未观察到的方式。最新的模型使用Experts聚集的混合物构成了理论上的限制，这些局限性限制了它们在复杂数据集上的发电质量。在本文中，我们提出了一个可解释的新型模型，能够同时学习关节和条件分布，而无需引入混合物聚合。我们的模型遵循一个多阶段训练过程：首先使用变异推理对关节分布进行建模，然后用标准化流量对条件分布进行建模以更好地近似真实的后代。重要的是，我们还建议提取和利用模式之间共享的信息，以改善生成的样品的条件连贯性。我们的方法在几个基准数据集上实现了最新的结果。

### Multimodal Data-Driven Classification of Mental Disorders: A Comprehensive Approach to Diagnosing Depression, Anxiety, and Schizophrenia 
[[arxiv](https://arxiv.org/abs/2502.03943)] [[cool](https://papers.cool/arxiv/2502.03943)] [[pdf](https://arxiv.org/pdf/2502.03943)]
> **Authors**: Himanshi Singh,Sadhana Tiwari,Sonali Agarwal,Ritesh Chandra,Sanjay Kumar Sonbhadra,Vrijendra Singh
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 多模式数据驱动的精神障碍分类：诊断抑郁症，焦虑和精神分裂症的全面方法
- **领域**: 机器学习
- **摘要**: 这项研究调查了多模式数据整合的潜力，该数据将脑电图（EEG）数据与社会人口统计学特征（例如年龄，性别，教育和智力）（IQ）（IQ）结合在一起，以诊断精神疾病，例如精神分裂症，抑郁症和焦虑。使用Apache Spark和卷积神经网络（CNN），为大数据环境开发了数据驱动的分类管道，以有效地分析大型数据集。为了评估与精神障碍相关的大脑活动和连接模式，检查了脑电图参数，例如功率谱密度（PSD）和连贯性。比较分析强调了连贯性特征的重要性，这表明分类准确性和鲁棒性显着提高。这项研究强调了整体方法通过整合各种数据源而对有效诊断工具的重要性。这些发现通过证明利用大数据，复杂的深度学习方法和多模式数据集的潜力来增强对心理健康诊断的精确性，可用性和理解性，以证明利用大数据，复杂的深度学习方法和多模式数据集的潜力，为治疗精神疾病打开了大门。

### Graph Neural Network-Driven Hierarchical Mining for Complex Imbalanced Data 
[[arxiv](https://arxiv.org/abs/2502.03803)] [[cool](https://papers.cool/arxiv/2502.03803)] [[pdf](https://arxiv.org/pdf/2502.03803)]
> **Authors**: Yijiashun Qi,Quanchao Lu,Shiyu Dou,Xiaoxuan Sun,Muqing Li,Yankaiqi Li
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 图形神经网络驱动的分层挖掘，用于复杂的不平衡数据
- **领域**: 机器学习
- **摘要**: 这项研究为高维不平衡的数据提供了一个分层挖掘框架，利用深度图模型来解决处理复杂，高维数据分布的常规方法的固有性能限制，并具有不平衡的样本表示。通过构建数据集的结构化图表并集成图形神经网络（GNN）嵌入，该方法有效地捕获了样本之间的全局相互依赖性。此外，采用层次结构策略来增强少数族类特征模式的表征和提取，从而促进精确且强大的数据挖掘。跨多种实验场景的经验评估验证了所提出的方法的功效，证明了对关键绩效指标的传统方法的实质性改进，包括模式发现数，平均支持和少数族裔班级覆盖范围。值得注意的是，该方法在少数级特征提取和模式相关分析中表现出卓越的能力。这些发现强调了深度图模型的潜力以及分层挖掘策略，以显着提高数据分析的效率和准确性。这项研究为高维复杂数据处理提供了一个新颖的计算框架，并为将来扩展到动态发展的不平衡数据和多模式数据应用奠定了基础，从而将高级数据挖掘方法的适用性扩展到更复杂的分析领域。

### A Multimodal PDE Foundation Model for Prediction and Scientific Text Descriptions 
[[arxiv](https://arxiv.org/abs/2502.06026)] [[cool](https://papers.cool/arxiv/2502.06026)] [[pdf](https://arxiv.org/pdf/2502.06026)]
> **Authors**: Elisa Negrini,Yuxuan Liu,Liu Yang,Stanley J. Osher,Hayden Schaeffer
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 多模式PDE基础模型，用于预测和科学文本说明
- **领域**: 机器学习,数值分析
- **摘要**: 神经网络是近似于科学计算任务（例如替代建模，实时预测和最佳控制）中使用的非线性微分方程的工具。 PDE基础模型利用神经网络同时训练近似值到多个微分方程，因此是可以适应下游任务的通用求解器。当前的PDE基础模型专注于学习通用解决方案运算符和/或方程式管理系统，因此仅处理数值或符号模式。但是，现实世界的应用程序可能需要更灵活的数据方式，例如文本分析或描述性输出。为了解决这一差距，我们提出了一种新型的多模式深度学习方法，该方法利用基于变压器的架构近似于解决方案操作员的各种ODE和PDE。我们的方法集成了数值输入，例如方程参数和初始条件，以及物理过程或系统动力学的文本描述。这使我们的模型能够处理符号表示可能不完整或不可用的设置。除了提供准确的数值预测外，我们的方法还会产生可解释的科学文本描述，从而更深入地了解基本动力学和解决方案属性。数值实验表明，我们的模型提供了分布数据（平均相对误差小于3.3％）的准确解决方案，并且分布数据（平均相对误差小于7.8％）以及精确的文本说明（具有正确的描述生成了100％次）。在某些测试中，该模型还显示能够及时推断解决方案。

### Redefining Robot Generalization Through Interactive Intelligence 
[[arxiv](https://arxiv.org/abs/2502.05963)] [[cool](https://papers.cool/arxiv/2502.05963)] [[pdf](https://arxiv.org/pdf/2502.05963)]
> **Authors**: Sharmita Dey
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 通过互动智能重新定义机器人概括
- **领域**: 机器学习,人工智能,机器人技术
- **摘要**: 大规模机器学习的最新进展已经产生了能够适应一系列下游任务的高容量基础模型。尽管这种模型对机器人技术有着巨大的希望，但现行的范式仍然将机器人描绘成单身自主决策者，执行操纵和导航等任务，人类的参与度有限。然而，包括可穿戴机器人技术（例如，假体，矫形器，外骨骼），近视和神经界面在内的大量现实世界机器人系统是半自动的，需要与人类伙伴进行持续的互动协调，这具有挑战性的单一主管假设。在该立场论文中，我们认为机器人基础模型必须演变为交互式多机构观点，以处理实时人类机器人共同适应的复杂性。 We propose a generalizable, neuroscience-inspired architecture encompassing four modules: (1) a multimodal sensing module informed by sensorimotor integration principles, (2) an ad-hoc teamwork model reminiscent of joint-action frameworks in cognitive science, (3) a predictive world belief model grounded in internal model theories of motor control, and (4) a memory/feedback mechanism that echoes concepts of基于Hebbian和基于增强的可塑性。尽管通过半机械系统的镜头进行了说明，但在可穿戴设备和人类生理学的情况下，该框架密不可分地交织，但提出的框架广泛适用于在半自治或交互式环境中运行的机器人。通过超越单一代理设计，我们的立场强调了机器人技术中的基础模型如何实现更强大，个性化和预期的性能水平。

### Predictive Crash Analytics for Traffic Safety using Deep Learning 
[[arxiv](https://arxiv.org/abs/2502.05777)] [[cool](https://papers.cool/arxiv/2502.05777)] [[pdf](https://arxiv.org/pdf/2502.05777)]
> **Authors**: Karthik Sivakoti
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 使用深度学习的交通安全性进行预测性崩溃分析
- **领域**: 机器学习,人工智能
- **摘要**: 传统的自动化崩溃分析系统在很大程度上依赖静态统计模型和历史数据，需要大量的手动解释和缺乏实时预测能力。这项研究通过整合集合学习方法和多模式数据融合来实现实时崩溃风险评估和预测，从而提出了一种创新的交通安全性分析方法。我们的主要贡献在于开发一个层次严重性分类系统，该系统将空间 - 周期性崩溃模式与环境条件相结合，从而比传统统计方法取得了重大改进。该系统的平均平均精度（MAP）为0.893，比当前最新方法提高了15％（基线图：0.776）。我们介绍了一种新颖的功能工程技术，该技术将崩溃位置数据与事件报告和天气状况相结合，在风险预测中达到92.4％的精度，而热点识别的精度为89.7％。通过使用500,000个初始崩溃记录过滤到59,496个高质量样本的广泛验证，我们的解决方案显示了预测准确性和计算效率的明显提高。关键创新包括强大的数据清洁管道，自适应功能生成以及可扩展的实时预测系统，能够处理1,000个并发请求的峰值负载，同时保持低于100ms的响应时间。

### Analog and Multi-modal Manufacturing Datasets Acquired on the Future Factories Platform V2 
[[arxiv](https://arxiv.org/abs/2502.05020)] [[cool](https://papers.cool/arxiv/2502.05020)] [[pdf](https://arxiv.org/pdf/2502.05020)]
> **Authors**: Ramy Harik,Fadi El Kalach,Jad Samaha,Philip Samaha,Devon Clark,Drew Sander,Liam Burns,Ibrahim Yousif,Victor Gadow,Ahmed Mahmoud,Thorsten Wuest
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 在未来工厂平台上获得的模拟和多模式制造数据集V2
- **领域**: 机器学习
- **摘要**: 本文介绍了在南卡罗来纳大学未来工厂实验室的制造装配线连续运行的8小时连续运行中捕获的两个行业级数据集，该数据集于08/13/2024。数据集遵守行业标准，涵盖了通信协议，执行器，控制机制，传感器，传感器和相机。数据收集利用了整个实验室的集成和外部传感器，包括嵌入执行器和外部安装设备中的传感器。此外，高性能摄像机捕获了操作的关键方面。在先前的实验[1]中，进行了3​​0小时连续运行，在此期间记录了所有异常。随后实施了维护程序，以减少潜在的错误和操作中断。这两个数据集包括：（1）时间序列模拟数据集，以及（2）包含同步系统数据和图像的多模式时序列数据集。这些数据集旨在通过提供一个用于测试新算法的平台而无需重新创建物理制造环境的平台来支持未来的研究。此外，数据集是开源的，旨在促进人工智能模型的培训，通过为各种应用程序和项目提供全面的，即用的资源来简化研究。

### G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion 
[[arxiv](https://arxiv.org/abs/2502.04684)] [[cool](https://papers.cool/arxiv/2502.04684)] [[pdf](https://arxiv.org/pdf/2502.04684)]
> **Authors**: Mengdi Liu,Zhangyang Gao,Hong Chang,Stan Z. Li,Shiguang Shan,Xilin Chen
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: G2PDiffusion：通过进化扩散的跨物种基因型对表型预测
- **领域**: 机器学习,人工智能
- **摘要**: 了解基因如何影响跨物种的表型是基因工程中的一个基本挑战，这将促进各种领域的进步，例如作物育种，保护生物学和个性化医学。但是，当前的表型预测模型仅限于单个物种和昂贵的表型标记过程，这使得基因型到表型预测高度依赖于域依赖性和数据筛选问题。为此，我们建议以大规模多模式预处理促进跨物种的概括，以促进跨物种的概括。我们提出了第一个基因型到表型扩散模型（G2PDiffusion），该模型（G2PDiffusion）从DNA产生形态学图像，考虑到两个关键进化信号，即多个序列比对（MSA）和环境环境。该模型包含三个新组件：1）MSA检索引擎，可识别保守和共同进化模式； 2）一种环境意识的MSA条件编码器，该编码器有效地模拟了复杂的基因型 - 环境相互作用； 3）适应性现象比对模块，以提高基因型 - 表型一致性。广泛的实验表明，将进化信号与环境环境相结合，丰富了模型对整个物种表型变异性的理解，从而对先进的AI辅助基因组分析提供了有价值的探索。

### Exploring Neural Network Pruning with Screening Methods 
[[arxiv](https://arxiv.org/abs/2502.07189)] [[cool](https://papers.cool/arxiv/2502.07189)] [[pdf](https://arxiv.org/pdf/2502.07189)]
> **Authors**: Mingyuan Wang,Yangzi Guo,Sida Liu,Yanwen Xiao
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: This work has been submitted to the IEEE for possible publication
- **标题**: 通过筛选方法探索神经网络修剪
- **领域**: 机器学习,机器学习
- **摘要**: 深层神经网络（DNN），例如用于视觉任务的卷积神经网络（CNN），用于序列数据的复发神经网络（RNN）以及用于丰富语言或多模式任务的变压器模型，在各种任务上都实现了前所未有的绩效。现代DNN的令人印象深刻的表现部分归因于其纯粹的规模。最新的深度学习模型具有数万到数亿到数亿个参数，从而使推理过程资源密集。这些网络的高计算复杂性阻止了它们在资源有限的设备（例如移动平台，IoT设备和边缘计算系统）上的部署，因为这些设备需要节能且实时的处理功能。本文提出并评估了一个网络修剪框架，该框架基于对分类类别的网络组件意义的统计分析消除了非必需参数。所提出的方法使用筛选方法与加权方案相结合，以评估非结构化和结构化修剪的连接和通道贡献，从而可以消除不必要的网络元素，而不会显着降低模型性能。对完全连接的神经网络（FNN）和CNN的现实世界视觉数据集进行了广泛的实验验证，这表明，与原始网络相比，所提出的框架会产生竞争性的精益网络。此外，提议的框架在三种情况下的两种中都优于最先进的网络修剪方法。

### Early Risk Prediction of Pediatric Cardiac Arrest from Electronic Health Records via Multimodal Fused Transformer 
[[arxiv](https://arxiv.org/abs/2502.07158)] [[cool](https://papers.cool/arxiv/2502.07158)] [[pdf](https://arxiv.org/pdf/2502.07158)]
> **Authors**: Jiaying Lu,Stephanie R. Brown,Songyuan Liu,Shifan Zhao,Kejun Dong,Del Bold,Michael Fundora,Alaa Aljiffry,Alex Fedorov,Jocelyn Grunwell,Xiao Hu
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 通过多模式变压器从电子健康记录中对小儿心脏骤停的早期风险预测
- **领域**: 机器学习,人工智能
- **摘要**: 小儿心脏骤停（CA）的早期预测对于及时干预高风险重症监护环境至关重要。我们介绍了PEDCA-FT，这是一种基于变压器的新型框架，将EHR的表格视图与EHR的派生文本视图融合在一起，以完全释放高维风险因素及其动态的相互作用。通过为每个模态视图采用专用的变压器模块，Pedca-FT捕获了复杂的时间和上下文模式，以产生强大的CA风险估计。通过Choa-CICU数据库进行了精选的小儿队列评估，我们的方法在五个关键的绩效指标上的其他十个人工智能模型都优于其他十个人工智能模型，并确定了临床上有意义的风险因素。这些发现强调了多模式融合技术增强早期CA检测并改善患者护理的潜力。

### Conditional Distribution Quantization in Machine Learning 
[[arxiv](https://arxiv.org/abs/2502.07151)] [[cool](https://papers.cool/arxiv/2502.07151)] [[pdf](https://arxiv.org/pdf/2502.07151)]
> **Authors**: Blaise Delattre,Sylvain Delattre,Alexandre Vérine,Alexandre Allauzen
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 机器学习中的条件分配量化
- **领域**: 机器学习
- **摘要**: 有条件的期望\ Mathbb {e}（y \ mid x）通常无法捕获多模式条件分布的复杂性\ Mathcal {l}（y \ mid x）。为了解决这个问题，我们建议使用n点条件量化 - 可通过梯度下降来学习的X的函数映射 - 至近似\ Mathcal {l}（y \ mid x）。这种方法适应了针对有条件分布的竞争性学习矢量量化（CLVQ）。通过提供更好地反映多模式结构的多个代表性点，它超越了单值预测。它实现了在瓦斯尔斯坦距离内的真实条件定律的近似。最终的框架在理论上是基础的，对于不确定性定量和多模式数据生成任务有用。例如，在计算机视觉介绍任务中，对于相同的部分观察到的输入图像X，可能存在多个合理的重建。我们通过对合成和现实世界数据集的实验来证明我们的方法的有效性。

### A Deep Learning Framework Integrating CNN and BiLSTM for Financial Systemic Risk Analysis and Prediction 
[[arxiv](https://arxiv.org/abs/2502.06847)] [[cool](https://papers.cool/arxiv/2502.06847)] [[pdf](https://arxiv.org/pdf/2502.06847)]
> **Authors**: Yu Cheng,Zhen Xu,Yuan Chen,Yuhan Wang,Zhenghao Lin,Jinsong Liu
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 一个深入学习框架，将CNN和BilstM整合进行财务系统风险分析和预测
- **领域**: 机器学习,计算工程、金融和科学
- **摘要**: 这项研究提出了一个基于卷积神经网络（CNN）和双向长期记忆网络（BILSTM）组合的深度学习模型，用于判别财务系统风险。该模型首先使用CNN提取金融市场多维特征的本地模式，然后通过Bilstm建模时间序列的双向依赖性，以全面地表征空间特征和时间动态中系统性风险定律的变化定律。该实验基于真实的财务数据集。结果表明，该模型在准确性，回忆和F1分数方面比传统的单个模型（例如Bilstm，CNN，Transformer和TCN）优于。 F1得分达到0.88，显示出极高的判别能力。这表明，结合CNN和Bilstm的联合策略不仅可以完全捕获市场数据的复杂模式，而且还可以有效地解决时间序列数据中的长期依赖性问题。此外，这项研究还探讨了该模型在处理数据噪声和处理高维数据方面的鲁棒性，从而为智能财务风险管理提供了大力支持。将来，研究将进一步优化模型结构，引入强化学习和多模式数据分析等方法，并提高模型应对更复杂的财务环境的效率和概括能力。

### Prot2Chat: Protein LLM with Early Fusion of Sequence and Structure 
[[arxiv](https://arxiv.org/abs/2502.06846)] [[cool](https://papers.cool/arxiv/2502.06846)] [[pdf](https://arxiv.org/pdf/2502.06846)]
> **Authors**: Zhicong Wang,Zicheng Ma,Ziqiang Cao,Changlong Zhou,Jun Zhang,Yiqin Gao
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-11
> **comment**: 9 pages, 2 figures
- **标题**: PROT2CHAT：蛋白LLM具有序列和结构的早期融合
- **领域**: 机器学习,人工智能,生物分子
- **摘要**: 蛋白质在生物体中起关键作用，但是了解其功能提出了重大挑战，包括基于分类的方法的灵活性有限，无法有效利用空间结构信息以及缺乏针对蛋白质Q＆A系统的系统评估指标。为了解决这些局限性，我们提出了Prot2Chat，Prot2Chat是一个新颖的框架，通过统一模块将多模式蛋白质表示与自然语言集成，从而使大型语言模型（LLM）驱动的答案生成。我们的模型结合了修改的蛋白质编码器，该编码器以统一的方式编码蛋白质序列和结构信息，具有跨注意机制的蛋白质文本适配器和Llama3解码器。为了优化培训效率，我们将编码器冻结并为解码器采用Lora技术。我们在两个数据集上进行了实验，即自动指标和专家评估证明了我们的模型的出色性能。此外，零拍的预测结果突出了其强大的概括能力。该框架为桥接蛋白质领域知识提供了一种有希望的解决方案，并具有自然语言的理解，为蛋白质相关研究的变革性进步铺平了道路。

### CAST: Cross Attention based multimodal fusion of Structure and Text for materials property prediction 
[[arxiv](https://arxiv.org/abs/2502.06836)] [[cool](https://papers.cool/arxiv/2502.06836)] [[pdf](https://arxiv.org/pdf/2502.06836)]
> **Authors**: Jaewan Lee,Changyoung Park,Hongjun Yang,Sungbin Lim,Sehui Han
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-11
> **comment**: 10 pages, 3 figures
- **标题**: 铸件：基于跨注意的材料财产预测的结构和文本的多模式融合
- **领域**: 机器学习,材料科学,人工智能
- **摘要**: AI的最新进展彻底改变了材料科学的财产预测，并加速了材料发现。图形神经网络（GNN）由于能够表示晶体结构作为图形，有效地捕获局部相互作用并提供出色的预测，因此脱颖而出。但是，这些方法通常会丢失关键的全球信息，例如晶体系统和重复的单元连接。为了解决这个问题，我们提出了Cast是一种基于跨注意的多模式融合模型，该模型集成了图形和文本模式以保留基本材料信息。 CAST使用交叉注意机制结合了节点和令牌级别的特征，超过了以前依赖于材料级嵌入（如图形平均值或[CLS]令牌）的方法。掩盖的节点预测预处理策略进一步增强了原子级信息的整合。与Crysmmnet和Multimat（例如Crysmmnet和Multimat）相比，我们的方法在包括带隙在内的四个晶体性能（包括带隙）的性质预测方面达到了22.9％的提高。预处理是对齐节点和文本嵌入的关键，注意地图确认了其在捕获节点和令牌之间关系的有效性。这项研究强调了材料科学中多模式学习的潜力，为更强大的预测模型铺平了道路，这些模型既包含了本地和全球信息。

### No Location Left Behind: Measuring and Improving the Fairness of Implicit Representations for Earth Data 
[[arxiv](https://arxiv.org/abs/2502.06831)] [[cool](https://papers.cool/arxiv/2502.06831)] [[pdf](https://arxiv.org/pdf/2502.06831)]
> **Authors**: Daniel Cai,Randall Balestriero
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 没有留下的位置：测量和改善地球数据的隐性表示的公平性
- **领域**: 机器学习,人工智能
- **摘要**: 隐式神经表示（INRS）在应对地球表示挑战方面表现出越来越多的希望，从排放监测到气候建模。但是，现有方法不成比例地优先考虑全球平均绩效，而从业者则需要细粒度的见解来了解这些模型中的偏见和变化。为了弥合这一差距，我们介绍了公平地球：一个明确制作的，旨在检查和挑战地球表示不平等现象。公平地球包括各种高分辨率的地球信号和沿着诸如陆地大小和人口密度等各个分​​层的广泛元数据，以评估模型的公平性。在公平地球的各种方式上评估最新的INR，我们发现了惊人的绩效差异。某些亚组，尤其是与高频信号相关的亚组（例如，岛屿，海岸线），始终以现有方法建模。作为回应，我们提出了基于先前的空间编码研究的球形小波编码。利用小波的多分辨率功能，我们的编码在各种尺度和位置产生一致的性能，从而提供了有偏见的亚组的更准确，更健壮的表示。这些开源贡献是朝着公平评估和地球部署的关键步骤。

### CTR-Driven Advertising Image Generation with Multimodal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.06823)] [[cool](https://papers.cool/arxiv/2502.06823)] [[pdf](https://arxiv.org/pdf/2502.06823)]
> **Authors**: Xingye Chen,Wei Feng,Zhenbang Du,Weizhen Wang,Yanyin Chen,Haohan Wang,Linkai Liu,Yaoyu Li,Jinyuan Zhao,Yu Li,Zheng Zhang,Jingjing Lv,Junjie Shen,Zhangang Lin,Jingping Shao,Yuanjie Shao,Xinge You,Changxin Gao,Nong Sang
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-11
> **comment**: Accepted to WWW 2025
- **标题**: CTR驱动的广告图像生成具有多模式大语模型
- **领域**: 机器学习,计算机视觉和模式识别,图形,信息检索
- **摘要**: 在Web数据中，广告图像对于吸引用户关注和提高广告效率至关重要。大多数现有方法都会为产品的背景产生主要关注的美学质量，这可能无法实现令人满意的在线性能。为了解决这一限制，我们探讨了多模式大语言模型（MLLM）的使用来通过优化点击率（CTR）作为主要目标来生成广告图像。首先，我们构建了有针对性的预训练任务，并利用大型电子商务多模式数据集为MLLM提供了广告图像生成任务的初始功能。为了进一步改善生成的图像的CTR，我们提出了一个新颖的奖励模型，通过增强学习（RL）对预训练的MLLM进行微调，该模型可以共同利用多模式功能并准确反映用户点击偏好。同时，制定了以产品为中心的优先优化策略，以确保生成的背景内容与微调后的产品特性保持一致，从而增强了广告图像的整体相关性和有效性。广泛的实验表明，我们的方法在在线和离线指标中都达到了最先进的性能。我们的代码和预培训模型可在以下网址公开获取：https：//github.com/chenguoz/caig。

### DiffListener: Discrete Diffusion Model for Listener Generation 
[[arxiv](https://arxiv.org/abs/2502.06822)] [[cool](https://papers.cool/arxiv/2502.06822)] [[pdf](https://arxiv.org/pdf/2502.06822)]
> **Authors**: Siyeol Jung,Taehwan Kim
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-11
> **comment**: Accepted at ICASSP 2025
- **标题**: DIFFLISTENER：听众生成的离散扩散模型
- **领域**: 机器学习,计算语言学,图形
- **摘要**: 听众头部（LHG）任务旨在根据演讲者的多模式提示产生天然的非语言听众响应。虽然先前的工作要么依赖有限的方式（例如音频和面部信息），要么采用具有限制的自回归方法，例如累积预测错误。为了解决这些局限性，我们提出了Difflistener，这是一种基于自动回归倾听者头部生成的基于离散扩散的方法。我们的模型将说话者的面部信息，音频和文本作为输入，还将面部差异信息包含以表示表达式和运动的时间动态。通过对面部动力学的这种明确建模，差异器可以以非自动回旋方式产生相干反应序列。通过全面的实验，Difflistener在定量和定性评估中都表明了最先进的表现。用户研究表明，difflistener会生成自然上下文感知的听众反应，这些反应与扬声器很好地同步。代码和演示视频可在https://siyeoljung.github.io/difflistener中找到

### Diffusion Instruction Tuning 
[[arxiv](https://arxiv.org/abs/2502.06814)] [[cool](https://papers.cool/arxiv/2502.06814)] [[pdf](https://arxiv.org/pdf/2502.06814)]
> **Authors**: Chen Jin,Ryutaro Tanno,Amrutha Saseendran,Tom Diethe,Philip Teare
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-11
> **comment**: Project page at https://astrazeneca.github.io/vlm/
- **标题**: 扩散说明调整
- **领域**: 机器学习,人工智能,图形
- **摘要**: 我们引入了薰衣草，这是一种简单的监督微调（SFT）方法，它通过利用最先进的图像生成模型（例如稳定的扩散）来提高先进视觉模型（VLM）的性能。具体而言，薰衣草将VLM变压器中的文本视频关注与SFT期间稳定扩散使用的等效物相结合，而不是对单独的编码器进行调整。这种一致性丰富了模型的视觉理解，并显着提高了整个分布任务的性能。薰衣草只需要有10.3万次培训示例，典型的大型SFT数据集的2.5％，以及一日标准硬件（8 GPU）的微型。它一致地改善了最先进的开源多模式LLM（例如Llama-3.2-11b，minicpm-llama3-v2.5），可实现多达30％的增长，并提高68％的增长，并提高挑战性的挑战性过失范围内的质量检查质量检查。通过有效地将图像发生器的视觉专业知识通过最小的监督传输，薰衣草为更准确的视觉语言系统提供了可扩展的解决方案。所有代码，培训数据和模型将在https://astrazeneca.github.io/vlm/上共享。

### Microcanonical Langevin Ensembles: Advancing the Sampling of Bayesian Neural Networks 
[[arxiv](https://arxiv.org/abs/2502.06335)] [[cool](https://papers.cool/arxiv/2502.06335)] [[pdf](https://arxiv.org/pdf/2502.06335)]
> **Authors**: Emanuel Sommer,Jakob Robnik,Giorgi Nozadze,Uros Seljak,David Rügamer
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 微型范围Langevin合奏：推进贝叶斯神经网络的采样
- **领域**: 机器学习
- **摘要**: 尽管最近进步，但基于抽样的贝叶斯神经网络（BNN）的推断仍然是概率深度学习的重大挑战。虽然基于抽样的方法不需要各种分布假设，但是当前的最新采样器仍然难以浏览BNN的复杂且高度多模式的后代。结果，尽管最近在使软件实施更有效地取得了进步方面的进步，但即使对于小型神经网络，抽样仍需要比非bayesian方法更长的推理时间。除了找到高概率区域的困难外，采样器提供对这些区域的足够探索的时间仍然是不可预测的。为了应对这些挑战，我们介绍了一种结合方法，该方法利用优化的策略以及最近提出的称为微域的采样器，称为MicroCanonical Langevin Monte Carlo（MCLMC），以提高高效，可靠且可预测的采样性能。与基于最先进的No-U-Turn采样器的方法相比，我们的方法可提供最大的速度至数量级，同时保持或改善各种任务和数据模式的预测性能和不确定性量化。建议的微型人道兰格文合奏和对MCLMC的修改另外增强了该方法在资源需求中的可预测性，从而促进了更容易的并行化。总而言之，提出的方法为BNN的实用，可扩展的推断提供了有希望的方向。

### ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources 
[[arxiv](https://arxiv.org/abs/2502.07862)] [[cool](https://papers.cool/arxiv/2502.07862)] [[pdf](https://arxiv.org/pdf/2502.07862)]
> **Authors**: Jason Wu,Kang Yang,Lance Kaplan,Mani Srivastava
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: ADMN：层面的自适应多模式网络，用于动态输入噪声和计算资源
- **领域**: 机器学习,人工智能,计算机视觉和模式识别
- **摘要**: 由于多种传感方式提供了鲁棒性，因此在动态场景中部署了多模式深度学习系统。然而，他们在不同的计算资源可用性（由于多租户，设备异质性等）以及输入质量的波动（来自传感器饲料饲料腐败，环境噪声等）方面挣扎。当前的多模式系统采用静态资源提供，并且当计算资源随时间变化时，无法轻易适应。此外，他们对使用固定特征提取器处理传感器数据的依赖不足以处理方式质量的变化。因此，诸如噪音高的非信息模式（例如那些具有高噪音的方式）可以更好地分配资源，以分配给其他方式。我们提出了ADMN，这是一个能够应对这两个挑战的层次自适应深度多模式网络 - 它调整了所有模式中的活动层的总数以满足计算资源限制，并根据其模态质量不断地跨输入方式重新分层。我们的评估展示了ADMN可以匹配最先进的网络的准确性，同时降低了其浮点操作的75％。

### Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative 
[[arxiv](https://arxiv.org/abs/2502.08942)] [[cool](https://papers.cool/arxiv/2502.08942)] [[pdf](https://arxiv.org/pdf/2502.08942)]
> **Authors**: Zihao Li,Xiao Lin,Zhining Liu,Jiaru Zou,Ziwei Wu,Lecheng Zheng,Dongqi Fu,Yada Zhu,Hendrik Hamann,Hanghang Tong,Jingrui He
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: Preprint, 37 pages
- **标题**: 时间流的语言：时间序列配对的文本编织成统一的时间叙事
- **领域**: 机器学习,人工智能
- **摘要**: 尽管时间序列模型的许多进展仅关注数值数据，但多模式时间序列的研究，尤其是涉及现实世界情景中通常遇到的上下文文本信息的研究序列，仍然处于起步阶段。因此，有效整合文本方式仍然具有挑战性。在这项工作中，我们重点介绍了现有作品所忽略的直观而重要的观察：时间序列配对的文本表现出定期属性，这些属性与原始时间序列的属性非常相似。在这个见解的基础上，我们提出了一个新颖的框架，文本为时间序列（TAT），该文本认为时间序列的文本是时间序列的辅助变量。可以将TAT插入任何现有的仅数值时间序列模型中，并使他们能够有效地处理时间序列数据。通过对具有各种现有时间序列模型的基准数据集的多模式时间序列预测和插入任务的广泛实验，我们证明TAT可以增强预测性能并实现超越性能而无需修改模型体系结构。

### Monge SAM: Robust Reparameterization-Invariant Sharpness-Aware Minimization Based on Loss Geometry 
[[arxiv](https://arxiv.org/abs/2502.08448)] [[cool](https://papers.cool/arxiv/2502.08448)] [[pdf](https://arxiv.org/pdf/2502.08448)]
> **Authors**: Albert Kjøller Jacobsen,Georgios Arvanitidis
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: Monge SAM：基于损失几何形状的鲁棒重新聚集不变的清晰度最小化
- **领域**: 机器学习,机器学习
- **摘要**: 关于深神经网络的最新研究表明，损失景观的平坦最小值与改善的概括相关。清晰度感知的最小化（SAM）通过根据对抗性扰动的梯度更新参数来有效地找到平坦区域。扰动取决于欧几里得度量，使SAM在重新构度下不变，从而模糊了清晰度和概括。我们提出了Monge SAM（M-SAM），这是SAM的重新配合不变版本，通过考虑由损耗表面自然引起的参数空间中的Riemannian度量。与以前的方法相比，M-SAM在任何建模选择下工作，仅依赖于轻度假设，而与SAM的计算高效”。从理论上讲，我们认为M-SAM在SAM和梯度下降（GD）之间有所不同，这增加了对超参数选择的鲁棒性，并将吸引力像鞍点那样降低到次优平衡中。我们在理论上和经验上都在多模式表示一致性任务上证明了这种行为。

### Quality over Quantity: Boosting Data Efficiency Through Ensembled Multimodal Data Curation 
[[arxiv](https://arxiv.org/abs/2502.08211)] [[cool](https://papers.cool/arxiv/2502.08211)] [[pdf](https://arxiv.org/pdf/2502.08211)]
> **Authors**: Jinda Xu,Yuhao Song,Daming Wang,Weiwei Zhao,Minghua Chen,Kangliang Chen,Qinya Li
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 质量优于数量：通过结合的多模式数据策划提高数据效率
- **领域**: 机器学习,人工智能
- **摘要**: 在一个被大量数据淹没的时代，Web-Crawl数据集的有效策划对于优化模型性能至关重要。本文解决了与此类数据集的非结构化和异构性质相关的挑战。传统的启发式策展方法通常不足地捕获复杂的特征，从而导致偏见和排除相关数据。我们介绍了一种先进的学习驱动方法，通过多模式运算符（Ecodatum）对数据进行整体策划，并结合了一种新型的质量指导的重复数据删除方法，以确保平衡的特征分布。 Ecodatum从策略性地将各种单峰和多模式数据策划运算符集成到弱监督集成框架中，并利用自动化优化有效地对每个数据点进行评分。 Ecodatum显着提高了数据策展质量和效率，优于现有的最新技术（SOTA）技术，在Datacomp排行榜上排名第一，在38个不同的评估数据集中，平均性能得分为0.182。这比DataComp基线方法提高了28％，证明了其在提高数据集策划和模型培训效率方面的有效性。

### Variational Rectified Flow Matching 
[[arxiv](https://arxiv.org/abs/2502.09616)] [[cool](https://papers.cool/arxiv/2502.09616)] [[pdf](https://arxiv.org/pdf/2502.09616)]
> **Authors**: Pengsheng Guo,Alexander G. Schwing
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 变分的整流流匹配
- **领域**: 机器学习,计算机视觉和模式识别
- **摘要**: 我们研究了差异整流的流量匹配，该框架通过对多模式速度矢量场进行建模来增强经典的整流流匹配。在推理时，经典的整流流匹配“移动”样品通过沿速度向量场的集成求解普通的微分方程，从源分布到目标分布。在训练时，通过线性插值从源来绘制的耦合样品和一个随机从目标分布中绘制的耦合样品，从而学习了速度矢量场。这导致“地面真相”'速度矢量场在同一位置的不同方向上指向，即速度矢量场是多模式/模棱两可的。但是，由于训练使用了标准的于点误差损失，因此学习的速度矢量场平均值“地面真相”方向，而不是多模式。相比之下，从多模式流方向的变异整流流匹配学习和样品。我们在合成数据，MNIST，CIFAR-10和ImageNet上显示，变异整流的流量匹配导致了令人信服的结果。

### UNITE-FND: Reframing Multimodal Fake News Detection through Unimodal Scene Translation 
[[arxiv](https://arxiv.org/abs/2502.11132)] [[cool](https://papers.cool/arxiv/2502.11132)] [[pdf](https://arxiv.org/pdf/2502.11132)]
> **Authors**: Arka Mukherjee,Shreya Ghosh
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: 28 pages, 16 figures
- **标题**: UNITE-FND：通过单峰场景翻译重新构架多式联运假新闻检测
- **领域**: 机器学习,人工智能
- **摘要**: 多模式假新闻检测通常需要复杂的架构和大量的计算资源，从而在现实世界中提出了部署挑战。我们介绍了Unite-FND，这是一个新颖的框架，将多模式的假新闻检测重新构架为单峰文本分类任务。我们建议使用Gemini 1.5 Pro提出六个专业提示策略，将视觉内容转换为结构化的文本描述，并启用有效的仅文本模型以保留关键的视觉信息。为了基准我们的方法，我们介绍了Uni-Fakeddit-55K，这是一个由55,000个样本组成的策划数据集家族，每个家族都通过我们的多模式到杀式翻译框架进行处理。实验结果表明，Unite-FND在二元分类中达到92.52％的精度，超过了多模型，同时将计算成本降低了10倍（Tinybert变体：14.50万参数，而SOTA模型中的250m+）。此外，我们提出了一套综合的五个新型指标，以评估图像到文本转换质量，从而确保最佳信息保存。我们的结果表明，基于结构的文本表示可以替换直接的多模式处理，而准确性的损失最小，这使得Unite-Fnd成为资源约束环境的实用且可扩展的替代方案。

### Collaborative Deterministic-Diffusion Model for Probabilistic Urban Spatiotemporal Prediction 
[[arxiv](https://arxiv.org/abs/2502.11013)] [[cool](https://papers.cool/arxiv/2502.11013)] [[pdf](https://arxiv.org/pdf/2502.11013)]
> **Authors**: Zhi Sheng,Yuan Yuan,Yudi Zhang,Depeng Jin,Yong Li
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 概率城市时空预测的协作确定性扩散模型
- **领域**: 机器学习,人工智能
- **摘要**: 对城市时空动态的准确预测对于增强城市管理和决策至关重要。现有的时空预测模型主要是确定性的，重点是主要时空模式。但是，这些动态非常复杂，表现出多模式分布，这些分布对于确定性模型捕获具有挑战性。在本文中，我们强调了概率预测在捕获时空数据中固有的不确定性和复杂性方面的关键作用。尽管主流概率模型可以捕获不确定性，但它们在准确地学习基本模式方面挣扎，并且经常遭受计算效率低下的困扰。为了应对这些挑战，我们提出了成本，该成本协作确定性和概率模型，以提高预测精度和处理不确定性的能力。为了实现这一目标，我们设计了一个均值分解框架，其中平均值是由确定性模型建模的，并且残留变化是通过概率模型（特别是扩散模型）学习的。此外，我们引入了一个量表感知的扩散过程，该过程更好地说明了不同区域之间空间异质动力学。对八个现实世界数据集进行的广泛实验表明，成本在确定性和概率指标中的现有方法均显着优于现有方法，并以低计算成本提高了20％。成本桥梁确定性精度和概率不确定性之间的差距，在城市时空预测领域取得了重大进步。

### Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings 
[[arxiv](https://arxiv.org/abs/2502.11007)] [[cool](https://papers.cool/arxiv/2502.11007)] [[pdf](https://arxiv.org/pdf/2502.11007)]
> **Authors**: Liangqi Yuan,Dong-Jun Han,Shiqiang Wang,Christopher G. Brinton
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 在多模式，多任务，多代码设置
- **领域**: 机器学习,分布式、并行和集群计算
- **摘要**: 与传统的机器学习模型相比，最近的大型语言模型（LLMS）可以通过多个对话和多模式数据源表现出多任务解决功能。 LLM的这些独特特征超出了它们的规模，使其在推理阶段的部署更具挑战性。具体而言，（i）在本地设备上部署LLMS面临计算，内存和能源资源问题，而（ii）将它们部署在云中无法保证实时服务并造成通信/使用成本。在本文中，我们设计了一个本地云LLM推理卸载（LCIO）系统，其中（i）可以处理多模式数据源的大规模云LLM，以及（ii）轻巧的本地LLM，可以高速处理简单的任务。 LCIO采用资源受限的强化学习（RCRL）来确定在何处进行推理（即本地与云），以及用于每个对话/任务的多模式数据源，旨在最大程度地提高长期奖励（将响应质量，延迟和用法成本纳入资源约束，以最大程度地提高响应质量，延迟和用法成本）。我们还提出了M4A1，这是一个新的数据集，该数据集说明了多模式，多任务，多任务和多LLLM特征，以调查LLMS在各种实际情况下的功能。我们证明了与基线相比，LCIO的有效性，在达到令人满意的响应质量的同时，显示出大量节省的潜伏期和成本。

### BalanceBenchmark: A Survey for Multimodal Imbalance Learning 
[[arxiv](https://arxiv.org/abs/2502.10816)] [[cool](https://papers.cool/arxiv/2502.10816)] [[pdf](https://arxiv.org/pdf/2502.10816)]
> **Authors**: Shaoxuan Xu,Menglu Cui,Chengxiang Huang,Hongfa Wang,Di Hu
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: 9 pages, 3 figures
- **标题**: BalanceBenchmark：多模式失衡学习的调查
- **领域**: 机器学习,人工智能
- **摘要**: 多模式学习已引起关注，因为它可以整合来自不同模式的信息的能力。但是，多模式的失衡问题通常会阻碍它，在这些问题中，某些模态占主导地位，而其他模式仍然不足。尽管最近的研究提出了各种方法来减轻此问题，但它们缺乏全面且公平的比较。在本文中，我们根据他们采用减轻失衡的策略进行了系统地将各种主流多模式不平衡算法分为四组。为了促进对这些方法的全面评估，我们引入了BalanceBenchmark，这是一个基准，包括多个使用多维数据集和评估指标，从三个角度：性能，不平衡程度和复杂性。为了确保公平的比较，我们开发了一个模块化和可扩展的工具包，该工具包标准化了不同方法的实验工作流程。基于使用BalanceBenchmark的实验，我们已经确定了有关性能，平衡程度和计算复杂性方面不同方法组的特征和优势的几个关键见解。我们希望这种分析可以激发更有效的方法来解决未来的失衡问题以及基础模型。该工具包的代码可在https://github.com/gewu-lab/balancebenchmark上获得。

### Artificial intelligence-enabled detection and assessment of Parkinson's disease using multimodal data: A survey 
[[arxiv](https://arxiv.org/abs/2502.10703)] [[cool](https://papers.cool/arxiv/2502.10703)] [[pdf](https://arxiv.org/pdf/2502.10703)]
> **Authors**: Aite Zhao,Yongcan Liu,Xinglin Yu,Xinyue Xing
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 使用多模式数据对帕金森氏病的启用人工智能检测和评估：一项调查
- **领域**: 机器学习,声音
- **摘要**: 高度适应性和可重复使用的人工智能（AI）模型的快速出现将彻底改变医疗领域，尤其是在帕金森氏病（PD）的诊断和管理中。当前，尚无有效的生物标志物来诊断PD，评估其严重性或跟踪其进展。现在，许多AI算法用于PD诊断和治疗，能够基于多模式和异构疾病症状数据执行各种分类任务，例如步态，手动运动和PD患者的语音模式。它们提供表达的反馈，包括预测PD的潜在可能性，评估个体或多种症状的严重性，有助于早期检测以及评估康复和治疗效果，从而证明先进的医学诊断能力。因此，这项工作通过生物识别症状识别进行了对有关PD检测和评估的最新作品的汇编，重点是机器学习和深度学习方法，强调其收益，揭露其弱点，以及它们在开放新的研究道路上的影响。此外，它还对用于解决相关约束的数据集，方法和架构进行了分类和表征。此外，本文探讨了数据驱动的AI技术在PD诊断中带来的潜在机会和挑战。

### I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models 
[[arxiv](https://arxiv.org/abs/2502.10458)] [[cool](https://papers.cool/arxiv/2502.10458)] [[pdf](https://arxiv.org/pdf/2502.10458)]
> **Authors**: Zhenxing Mi,Kuan-Chieh Wang,Guocheng Qian,Hanrong Ye,Runtao Liu,Sergey Tulyakov,Kfir Aberman,Dan Xu
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-17
> **comment**: Project page: https://mizhenxing.github.io/ThinkDiff, 19 pages, 14 figures
- **标题**: 我认为，因此，我扩散：在扩散模型中启用多模式的内部上下文推理
- **领域**: 机器学习,人工智能
- **摘要**: 本文介绍了ThinkDiff，这是一种新颖的对齐范式，通过整合视觉模型（VLMS）的优势，可以将文本对图扩散模型与多模式的理解和推理能力赋予能力。现有的多模式扩散式芬太尼方法主要集中于像素级重建，而不是内在的推理，并且受到基于推理的数据集的复杂性和有限的可用性的限制。 ThinkDiff通过利用视觉语言培训作为代理任务来解决这些挑战，将VLM与编码器模型大型语言模型（LLM）而不是扩散解码器的解码器保持一致。此代理任务基于这样的观察，即$ \ textbf {llm dododer} $与$ \ textbf {diffusion decoders} $共享相同的输入功能空间，该{fiffusion decoders} $使用相应的$ \ textbf {llm encoder} $来提示嵌入。结果，可以通过与LLM解码器对齐来简化与扩散解码器的对齐VLM。如果没有复杂的培训和数据集，ThinkDiff可以有效地释放理解，推理和构成扩散模型中的功能。实验表明，在具有挑战性的COBSAT基准中，ThinkDiff的准确性从19.2％提高到46.3％，用于多模式中上下文推理，仅在4个A100 GPU上进行了5个小时的培训。此外，ThinkDiff在将多个图像和文本撰写为逻辑上的图像中表现出了出色的性能。项目页面：https：//mizhenxing.github.io/thinkdiff。

### E2LVLM:Evidence-Enhanced Large Vision-Language Model for Multimodal Out-of-Context Misinformation Detection 
[[arxiv](https://arxiv.org/abs/2502.10455)] [[cool](https://papers.cool/arxiv/2502.10455)] [[pdf](https://arxiv.org/pdf/2502.10455)]
> **Authors**: Junjie Wu,Yumeng Fu,Nan Yu,Guohong Fu
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: E2LVLM：多模式外观误导性检测的证据增强大型视觉模型
- **领域**: 机器学习,多媒体
- **摘要**: 在大型视觉模型（LVLM）中的最新研究表明，多模式的脱离（OOC）错误信息传染性检测方面取得了令人印象深刻的进步，从而辨别出索赔中是否错误地使用了真实的图像。尽管它们成功了，但从反向搜索中检索到的真实图像的文字证据仍直接传输到LVLM，从而导致决策阶段不准确或虚假信息。为此，我们提出了E2LVLM，这是一种新颖的证据增强的大型视力模型，通过在两个层面上调整文本证据。首先，由于外部工具提供的文本证据而努力与LVLMS输入保持一致，我们设计了一种重新依赖和重写策略来产生连贯且上下文上的讨论内容，从而推动了与真实图像相关的LVLM的一致性和有效行为。其次，为了通过判断和解释解决新闻域数据集的稀缺性，我们通过提示具有信息内容的LVLM来获取可靠的解释，从而生成一种新颖的OOC多模式指令跟随数据集。此外，我们通过令人信服的解释来制定多模式指导策略。该方案有助于用于多模式OOC错误信息检测和解释的E2LVLM。许多实验表明，E2LVLM的性能比最先进的方法具有出色的性能，并且还为判断提供了令人信服的理由。

### Evaluating and Explaining Earthquake-Induced Liquefaction Potential through Multi-Modal Transformers 
[[arxiv](https://arxiv.org/abs/2502.10446)] [[cool](https://papers.cool/arxiv/2502.10446)] [[pdf](https://arxiv.org/pdf/2502.10446)]
> **Authors**: Sompote Youwai,Tipok Kitkobsin,Sutat Leelataviwat,Pornkasem Jongpradist
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 通过多模式变压器评估和解释地震引起的液化电位
- **领域**: 机器学习,地球物理学
- **摘要**: 这项研究为土壤液化预测提供了可解释的平行变压器架构，该结构集成了三个不同的数据流：光谱地震编码，土壤地层象征化令牌和特定地点特异性特征。该体系结构从11个主要地震中的165个案例历史进行了研究，采用快速的傅立叶变换来进行地震波形编码和大型语言模型的原理，以进行土壤层的令牌化。通过Shapley加性解释（SHAP）实现了可解释性，该解释将预测分解为来自地震特征，土壤特性和现场条件的个人贡献。该模型在跨区域验证集上实现了93.75％的预测准确性，并通过对地面运动强度和土壤抗性参数的敏感性分析来证明性能良好。值得注意的是，针对2024年Noto Peninsula地震以前看不见的地面运动数据的验证证实了该模型的概括能力和实用性。作为公开访问的Web应用程序实施可以同时快速评估多个站点。这种方法在岩土技术深度学习中建立了一个新的框架，其中复杂的多模式分析通过定量解释和可访问的部署符合实用的工程要求。

### DiOpt: Self-supervised Diffusion for Constrained Optimization 
[[arxiv](https://arxiv.org/abs/2502.10330)] [[cool](https://papers.cool/arxiv/2502.10330)] [[pdf](https://arxiv.org/pdf/2502.10330)]
> **Authors**: Shutong Ding,Yimiao Zhou,Ke Hu,Xi Yao,Junchi Yan,Xiaoying Tang,Ye Shi
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: DIOPT：自我监督的扩散以进行约束优化
- **领域**: 机器学习
- **摘要**: 扩散模型的最新进展通过利用其多模式采样能力来逃避局部优势来显示出有希望的基于学习优化的潜力。但是，现有的基于扩散的优化方法通常依赖于监督培训，缺乏确保严格限制满意度的机制，这在现实世界应用中通常需要。最终的观察结果是分布未对准，即生成的溶液分布通常表现出与可行域的小重叠。在本文中，我们提出了DIOPT，这是一种新型的扩散范式，该范式通过迭代自我训练系统地学习了近乎最佳的可行解决方案分布。我们的框架引入了几种关键创新：专门设计的目标分布，旨在最大程度地与受约束的解决方案歧管重叠；一种自动训练机制，该机制根据约束违规和最佳差距的严重程度适应候选解决方案；以及动态的内存缓冲区，通过在训练迭代术中保留高质量的解决方案来加速收敛。据我们所知，DIOPT代表了自我监督的扩散和硬约束满意度的首次成功整合。对包括电网控制，运动重新定位，无线分配在内的各种任务的评估表明了其优势在最佳和约束满意度方面。

### AttenGluco: Multimodal Transformer-Based Blood Glucose Forecasting on AI-READI Dataset 
[[arxiv](https://arxiv.org/abs/2502.09919)] [[cool](https://papers.cool/arxiv/2502.09919)] [[pdf](https://arxiv.org/pdf/2502.09919)]
> **Authors**: Ebrahim Farahmand,Reza Rahimi Azghan,Nooshin Taheri Chatrudi,Eric Kim,Gautham Krishna Gudur,Edison Thomaz,Giulia Pedrielli,Pavan Turaga,Hassan Ghasemzadeh
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: pertengluco：基于多模式变压器的血糖预测在AI-Readi数据集上
- **领域**: 机器学习,人工智能
- **摘要**: 糖尿病是一种慢性代谢疾病，其特征是持续的高血糖水平（BGL），导致严重的并发症，例如心血管疾病，神经病和视网膜病变。预测BGLS使患者能够在安全范围内维持葡萄糖水平，并使护理人员能够通过生活方式修改采取积极的措施。连续的葡萄糖监测（CGM）系统提供实时跟踪，为监测BGL提供了有价值的工具。但是，由于体育锻炼，饮食和其他因素，由于波动而导致的波动率准确地预测BGL仍然具有挑战性。最近的深度学习模型显示了改善BGL预测的希望。尽管如此，从长期预测范围内的多模式，不规则采样数据中准确地预测BGL仍然是一个具有挑战性的研究问题。在本文中，我们提出了Pastengluco，这是一种基于多模式变压器的长期血糖预测的框架。 Pentengluco采用交叉注意来有效整合CGM和活动数据，以解决与不同采样率融合的挑战。此外，它采用多尺度的关注来捕获时间数据中的长期依赖性，从而提高了预测准确性。为了评估时候出顾问的性能，我们在最近发布的Aireadi数据集上进行了预测实验，分析了其在包括健康个体，患有糖尿病前期患者和2型糖尿病患者在内的不同受试者队列中的预测准确性。此外，由于引入了新的队列，我们​​研究了其绩效的改进和忘记行为。我们的评估表明，与多模态LSTM模型相比，左右的左右改善了所有误差指标，例如均方根误差（RMSE），平均绝对误差（MAE）和相关性。在RMSE和MAE方面，PertengLuco的基线模型的表现分别高约10％和15％。

### On Creating a Causally Grounded Usable Rating Method for Assessing the Robustness of Foundation Models Supporting Time Series 
[[arxiv](https://arxiv.org/abs/2502.12226)] [[cool](https://papers.cool/arxiv/2502.12226)] [[pdf](https://arxiv.org/pdf/2502.12226)]
> **Authors**: Kausik Lakkaraju,Rachneet Kaur,Parisa Zehtabi,Sunandita Patra,Siva Likitha Valluru,Zhen Zeng,Biplav Srivastava,Marco Valtorta
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 关于创建一种因果基础的可用评级方法，用于评估支持时间序列的基础模型的鲁棒性
- **领域**: 机器学习,人工智能
- **摘要**: 基金会模型（FMS）改善了在金融等各个领域的时间序列预测，但是他们输入干扰的脆弱性可能会阻碍利益相关者（例如投资者和分析师）的收养。为了解决这个问题，我们提出了一个因果关系框架，以研究有关输入扰动的时间序列（FMT）的基础模型的鲁棒性。我们评估了我们对股票价格预测问题的方法，这是一个易于访问的公共数据的一个充分研究的问题，评估了六个跨越三个行业的六个杰出股票的最先进（一些多模式）FMT。我们的框架提出的评级有效地评估了FMT的鲁棒性，还为模型选择和部署提供了可行的见解。在我们的研究范围内，我们发现（1）多模式FMT与单峰版本相比具有更好的鲁棒性和准确性，并且（2）FMT进行了预先训练的时间序列预测任务，与多种多样的设置相比，预测任务具有更好的鲁棒性和更好的鲁棒性和预测准确性。此外，为了验证我们的框架的可用性，我们进行了一项用户研究，展示了FMTS预测错误以及计算评级。该研究证实，我们的评分减少了用户比较不同系统鲁棒性的困难。

### AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors 
[[arxiv](https://arxiv.org/abs/2502.12191)] [[cool](https://papers.cool/arxiv/2502.12191)] [[pdf](https://arxiv.org/pdf/2502.12191)]
> **Authors**: Ruoxuan Feng,Jiangyu Hu,Wenke Xia,Tianci Gao,Ao Shen,Yuhao Sun,Bin Fang,Di Hu
> **First submission**: 2025-02-15
> **First announcement**: 2025-02-18
> **comment**: Accepted by ICLR 2025
- **标题**: AnyTouch：在多个Visuo tactile传感器上学习统一的静态动态表示
- **领域**: 机器学习,计算机视觉和模式识别,机器人技术
- **摘要**: Visuo-Tactile传感器旨在模仿人类的触觉感知，使机器人能够精确理解和操纵物体。随着时间的流逝，许多精心设计的Visuo-Tactile传感器已集成到机器人系统中，有助于完成各种任务。但是，这些低标准的视觉触诊传感器的独特数据特性阻碍了强大的触觉感知系统的建立。我们认为解决此问题的关键在于学习统一的多传感器表示，从而整合了传感器并促进它们之间的触觉知识转移。为了实现这种性质的统一表示，我们介绍了Tacquad，这是来自四个不同的Visuo-Tactile传感器的一个对齐多模式的多模式多传感器触觉数据集，从而使各种传感器的显式集成。认识到人类通过获取诸如纹理和压力变化之类的多种触觉信息来感知物理环境，我们进一步建议从静态和动态的角度学习统一的多传感器表示。通过集成触觉图像和视频，我们提出了AnyTouch，这是一种统一的静态动态多传感器表示学习框架，并具有多层结构，旨在增强全面的感知能力和实现有效的跨传感器传输。这种多层次体系结构通过掩盖建模从触觉数据中捕获了像素级的细节，并通过学习语义级传感器 - 敏捷功能来增强感知和可传递性，通过多模式对齐和跨传感器匹配。我们对多传感器可传递性进行了全面的分析，并在各种数据集和现实世界浇筑任务中验证我们的方法。实验结果表明，我们的方法优于现有方法，在各种传感器上具有出色的静态和动态感知能力。

### CAMEL: Continuous Action Masking Enabled by Large Language Models for Reinforcement Learning 
[[arxiv](https://arxiv.org/abs/2502.11896)] [[cool](https://papers.cool/arxiv/2502.11896)] [[pdf](https://arxiv.org/pdf/2502.11896)]
> **Authors**: Yanxiao Zhao,Yangge Qian,Jingyang Shan,Xiaolin Qin
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Accepted at RLDM 2025
- **标题**: 骆驼：通过大型语言模型启用了加强学习的连续动作掩蔽
- **领域**: 机器学习,人工智能
- **摘要**: 连续作用空间中的强化学习（RL）遇到了持续的挑战，例如效率低下和融合到次优的解决方案。为了解决这些局限性，我们提出了一个新颖的框架，这是一个新颖的框架，将LLM生成的次优政策整合到RL训练管道中。骆驼利用动态动作掩盖和适应性的Epsilon掩蔽机制来指导早期训练阶段的探索，同时逐渐使代理人能够独立优化政策。骆驼的核心是基于环境描述和任务目标生成的LLMS生成的Python-Cycutable次优政策的整合。尽管这些政策简单明了，但这些政策为RL代理提供了宝贵的初始指导。为了有效利用这些先验，骆驼采用掩饰感知的优化来动态限制基于LLM输出的动作空间。此外，Epsilon掩盖逐渐减少了对LLM生成的指导的依赖，从而使代理商能够从受约束的探索过渡到自主政策的完善。实验性验证对体育馆的穆乔科环境证明了骆驼的有效性。在Hopper-V4和ANT-V4中，LLM生成的政策显着提高了样本效率，从而实现了可与或超过专家掩盖基准相当的性能。对于Walker2D-V4，LLM难以准确建模双足步态动力学，Camel在没有明显退化的情况下保持了强大的RL性能，突出了该框架在各种任务中的适应性。骆驼在提高样本效率和缓解收敛挑战方面表现出希望，但这些问题仍在开放以进行进一步研究。未来的工作旨在将骆驼推广到多模式LLMS，以进行更广泛的观察空间并自动化政策评估，从而减少人类干预并提高RL培训管道中的可伸缩性。

### Mitigating Visual Knowledge Forgetting in MLLM Instruction-tuning via Modality-decoupled Gradient Descent 
[[arxiv](https://arxiv.org/abs/2502.11740)] [[cool](https://papers.cool/arxiv/2502.11740)] [[pdf](https://arxiv.org/pdf/2502.11740)]
> **Authors**: Junda Wu,Yuxin Xiong,Xintong Li,Yu Xia,Ruoyu Wang,Yu Wang,Tong Yu,Sungchul Kim,Ryan A. Rossi,Lina Yao,Jingbo Shang,Julian McAuley
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 9 pages
- **标题**: 通过模态 - 偶联的梯度下降来减轻MLLM指导调查中的视觉知识忘记
- **领域**: 机器学习,计算机视觉和模式识别
- **摘要**: 最近的MLLM在大规模多模式数据集中进行了预训练后，已经显示出新兴的视觉理解和推理能力。与预训练的情况下，MLLM会获得丰富的视觉文本对齐方式，教学调节通常是通过文本驱动的，视觉监督较弱，从而导致预训练的视觉理解的降低并引起视觉遗忘。现有的方法，例如直接的微调和持续学习方法，无法明确解决此问题，通常会压缩视觉表示形式，并优先考虑任务对准而不是视觉保留，这进一步使视觉遗忘了。为了克服这一局限性，我们引入了一种新颖的视角，利用有效的等级来量化视觉表示丰富度的降级，并通过信息瓶颈原理解释这种降级为过度压缩，从而导致至关重要的预先获得的视觉知识的降低。在此视图的基础上，我们提出了一种模态耦合梯度下降（MDGD）方法，该方法调节梯度更新以保持视觉表示的有效等级，同时减轻信息瓶颈描述的过度压缩效果。通过明确删除特定于任务对齐的视觉理解的优化，MDGD可以保留预先训练的视觉知识，同时实现有效的任务适应。为了启用轻质指令调整，我们进一步使用梯度遮罩进一步开发了一种记忆有效的微调方法，该方法有选择地更新模型参数的子集以启用参数有效的微调（PEFT），从而减少了计算架设，从而在保留丰富的视觉表示的同时还要减少计算开销。跨各种下游任务和骨干MLLM的广泛实验表明，MDGD有效地减轻了从预训练的任务中遗忘的视觉遗忘，同时可以强大适应新任务。

### Maximum Entropy Reinforcement Learning with Diffusion Policy 
[[arxiv](https://arxiv.org/abs/2502.11612)] [[cool](https://papers.cool/arxiv/2502.11612)] [[pdf](https://arxiv.org/pdf/2502.11612)]
> **Authors**: Xiaoyi Dong,Jian Cheng,Xi Sheryl Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 21 pages, 7 figures
- **标题**: 通过扩散政策的最大熵增强学习
- **领域**: 机器学习,人工智能
- **摘要**: 具有高斯政策的软演员 - 批评（SAC）算法已成为实现最大熵增强学习（Maxent RL）目标的主流实施，该目标结合了熵最大化，以鼓励探索并增强政策鲁棒性。尽管高斯政策在更简单的任务上表现良好，但其探索能力和在复杂的多进球RL环境中的潜在性能受到其固有的单模式的限制。在本文中，我们采用了扩散模型，这是一个强大的生成模型，能够捕获复杂的多模式分布，作为实现Maxent RL目标的策略表示形式，开发了一种具有扩散策略（MAXENTDP）的名为Maxent RL的方法。我们的方法实现了有效的探索，并使政策更接近最佳最大政策。 Mujoco基准测试的实验结果表明，MaxEntDP在Maxent RL框架中优于高斯策略和其他生成模型，并且与其他基于其他基于最新扩散的在线RL算法相当地执行。我们的代码可在https://github.com/diffusionyes/maxentdp上找到。

### Connector-S: A Survey of Connectors in Multi-modal Large Language Models 
[[arxiv](https://arxiv.org/abs/2502.11453)] [[cool](https://papers.cool/arxiv/2502.11453)] [[pdf](https://arxiv.org/pdf/2502.11453)]
> **Authors**: Xun Zhu,Zheng Zhang,Xi Chen,Yiming Shi,Miao Li,Ji Wu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 连接器-S：多模式大语言模型中连接器的调查
- **领域**: 机器学习,人工智能
- **摘要**: 随着多模式大语言模型（MLLM）的快速发展，连接器在桥接多种方式和增强模型性能方面起着关键作用。但是，连接器的设计和演变尚未进行全面分析，在了解这些组件如何运行并阻碍更强大的连接器的开发方面留下了差距。在这项调查中，我们系统地检查了MLLM中连接器的当前进度，并提出了结构化的分类法，将连接器分类为原子操作（映射，压缩，专家的混合物）和整体设计（多层，多层编码器，多模式场景），以突出显示其技术和进步。此外，我们讨论了一些有希望的研究前沿和挑战，包括高分辨率输入，动态压缩，指南信息选择，组合策略和解释性。该调查旨在作为研究人员的基础参考和明确的路线图，为下一代连接器的设计和优化提供宝贵的见解，以增强MLLM的性能和适应性。

### HyperGCL: Multi-Modal Graph Contrastive Learning via Learnable Hypergraph Views 
[[arxiv](https://arxiv.org/abs/2502.13277)] [[cool](https://papers.cool/arxiv/2502.13277)] [[pdf](https://arxiv.org/pdf/2502.13277)]
> **Authors**: Khaled Mohammed Saifuddin,Shihao Ji,Esra Akbas
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: 9 pages, 2 figures
- **标题**: HyperGCL：通过可学习的超图视图的多模式图对比度学习
- **领域**: 机器学习,人工智能
- **摘要**: 图形对比度学习（GCL）的最新进步表明，在改善图表表示方面具有出色的有效性。但是，依靠预定义的增强（例如，节点下降，边缘扰动，属性掩蔽）可能会导致与任务相关的信息丢失，并且缺乏对各种输入数据的适应性。此外，否定样品的选择仍然很少探索。在本文中，我们介绍了HyperGCL，这是一种新型的多模式GCL框架，从超图的角度来看。 HyperGCL通过共同利用输入图的结构和属性来构建三个不同的超图观点，从而使多种模态在对比学习中的全面整合。可学习的自适应拓扑增强技术通过保留重要关系并滤除噪声来增强这些观点。特定于视图的编码器从每种视图中捕获基本特征，而网络感知的对比损失则利用基本拓扑来有效地定义正面和负样本。基准数据集的大量实验表明，HyperGCL达到了最新的节点分类性能。

### An improved wind power prediction via a novel wind ramp identification algorithm 
[[arxiv](https://arxiv.org/abs/2502.12807)] [[cool](https://papers.cool/arxiv/2502.12807)] [[pdf](https://arxiv.org/pdf/2502.12807)]
> **Authors**: Yifan Xu
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 通过新颖的风力坡道识别算法改进的风力预测
- **领域**: 机器学习
- **摘要**: 作者：Yifan Xu摘要：常规风力预测方法通常难以在风速和功率输出突然变化的情况下提供准确可靠的预测。为了应对这一挑战，本研究提出了一种结合风速突变识别算法的综合算法，优化的相似时期匹配算法和风能预测算法。通过利用气象事件的收敛特性，该方法显着提高了在突然气象变化下风能预测的准确性。首先，开发了一种基于变异模式分解的新型自适应模型，即VMD-IC模型，用于识别和标记历史风能数据中的关键转弯点，代表突然的气象环境。同时，本文提出了坡道因子（RF）指标和风速相似性系数，以优化当前风力坡道事件（WPRE）的定义算法。在创新了攀登和降级算法的定义之后，本文使用告密者深度学习算法来输出前两个模型以及多模式数据，例如NWP数值天气预报，以实现准确的风向预测。消融研究的实验结果证实了所提出的风坡识别方法的有效性和可靠性。与现有方法相比，提出的模型表现出色，并为电力系统的安全和成本效益的操作提供了宝贵的指导。

### SleepGMUformer: A gated multimodal temporal neural network for sleep staging 
[[arxiv](https://arxiv.org/abs/2502.14227)] [[cool](https://papers.cool/arxiv/2502.14227)] [[pdf](https://arxiv.org/pdf/2502.14227)]
> **Authors**: Chenjun Zhao,Xuesen Niu,Xinglin Yu,Long Chen,Na Lv,Huiyu Zhou,Aite Zhao
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: Sleepgmuformer：一个封闭的多模式时间神经网络，用于睡眠阶段
- **领域**: 机器学习,人工智能
- **摘要**: 睡眠分期是评估睡眠质量和诊断睡眠障碍的关键方法。但是，当前的深度学习方法面临挑战：1）灌注后技术忽略了不同方式的不同贡献； 2）未经处理的睡眠数据可能会干扰频域信息。为了解决这些问题，本文提出了用于多域睡眠数据的封闭式多模式的时间神经网络，包括心率，运动，步骤，脑电图（FPZ-CZ，PZ-oz）和wristhr-Motion-Sleep和Sleepedf-78的EOG。该模型集成：1）用于特征对齐，缺失值处理和脑电图下降的预处理模块； 2）在时间维度中用于复杂睡眠特征的特征提取模块； 3）用于实时模态加权的动态融合模块。示例显示SleepedF-78的分类精度为85.03％，Wristhr-Motion-Sleep数据集的分类精度为94.54％。该模型处理异质数据集，并胜过最先进的模型1.00％-4.00％。

### Quantifying Memorization and Retriever Performance in Retrieval-Augmented Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.13836)] [[cool](https://papers.cool/arxiv/2502.13836)] [[pdf](https://arxiv.org/pdf/2502.13836)]
> **Authors**: Peter Carragher,Abhinand Jha,R Raghav,Kathleen M. Carley
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 量化记忆和检索器的表现
- **领域**: 机器学习,人工智能
- **摘要**: 大型语言模型（LLMS）表现出了出色的回答功能（QA），但是评估其对记忆与检索的依赖的指标仍然不发达。此外，虽然填充模型是封闭域任务的最新模型，但GPT-4O（例如GPT-4O）的通用模型表现出强烈的零拍性能。这引发了有关记忆，概括和检索之间权衡的问题。在这项工作中，我们分析了与基线VLM相比，多式联运检索的VLMS记住培训数据的程度。使用WebQA基准测试，我们将鉴定模型与基线VLMS对比对多台面检索和问题答案进行了对比，从而研究了芬太尼对数据记忆的影响。为了量化端到端检索系统和质量检查系统中的记忆，我们通过调查质量请QA成功的实例提出了几个代理指标。我们的结果揭示了固定模型依赖记忆的程度。相比之下，以准确性为代价（WebQA测试集为72％vs 52％），以检索功能的VLM的记忆分数较低。因此，我们的措施对未来的工作构成了挑战，以调和开放域质量检查和联合检索任务中的记忆和概括。

### Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model 
[[arxiv](https://arxiv.org/abs/2502.13449)] [[cool](https://papers.cool/arxiv/2502.13449)] [[pdf](https://arxiv.org/pdf/2502.13449)]
> **Authors**: Dongki Kim,Wonbin Lee,Sung Ju Hwang
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: Project Page: https://mol-llama.github.io/
- **标题**: Mol-Lalama：在大分子语言模型中对分子的一般理解
- **领域**: 机器学习,化学物理
- **摘要**: 理解分子是理解生物体并推动药物发现的进步，需要跨化学和生物学的跨学科知识的关键。尽管大型分子语言模型在解释分子结构方面取得了显着的成功，但它们的指导数据集仅限于以任务为导向的数据集中的特定知识，并且不能完全涵盖分子的基本特征，从而阻碍了它们作为通用分子助手的能力。为了解决这个问题，我们提出了Mol-llama，这是一种大型分子语言模型，该模型通过多模式教学调整掌握了以分子为中心的通用知识。为此，我们设计了涵盖分子基本特征的关键数据类型，并结合了分子结构的基本知识。此外，为了提高对分子特征的理解，我们引入了一个模块，该模块整合了来自不同分子编码器的互补信息，从而利用了不同分子表示的明显优势。我们的实验结果表明，Mol-llama能够理解分子的一般特征，并通过详细的解释对用户的查询产生相关响应，这意味着其作为分子分析的通用助手的潜力。我们的项目页面位于https://mol-llama.github.io/。

### EigenShield: Causal Subspace Filtering via Random Matrix Theory for Adversarially Robust Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.14976)] [[cool](https://papers.cool/arxiv/2502.14976)] [[pdf](https://arxiv.org/pdf/2502.14976)]
> **Authors**: Nastaran Darabi,Devashri Naik,Sina Tayebati,Dinithi Jayasuriya,Ranganath Krishnan,Amit Ranjan Trivedi
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 特征柴菲尔德：通过随机矩阵理论的因果子空间过滤，以实现对抗性强大的视觉模型
- **领域**: 机器学习,密码学和安全,计算机视觉和模式识别
- **摘要**: 视觉模型（VLM）继承了大语言模型（LLMS）的对抗性漏洞，这些模型被其多模式的性质进一步加剧。现有的防御措施，包括对抗性训练，输入转换和启发式检测，在计算上是昂贵，依赖建筑的且针对自适应攻击的脆弱的。我们介绍了特征性的特征，这是一种推理时间防御，利用随机矩阵理论来量化高维VLM表示中的对抗性破坏。与依赖经验启发式方法的先前方法不同，特征希尔德采用尖刺的协方差模型来检测结构的光谱偏差。使用基于鲁棒性的非符号评分（RBN）和基于分位数的阈值，它将因果特征向量分开，该因素特征向量将语义信息与对对抗性伪影易感的相关特征向量进行编码。通过将嵌入到因果子空间上，特征柴场可以过滤对抗噪声，而无需修改模型参数或需要对抗训练。这种与建筑无关的，攻击无形的方法大大降低了攻击成功率，建立了光谱分析作为常规防御的原则替代方案。我们的结果表明，特征希尔德始终优于所有现有防御，包括对抗性训练，Uniguard和苹果酒。

### Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective 
[[arxiv](https://arxiv.org/abs/2502.14770)] [[cool](https://papers.cool/arxiv/2502.14770)] [[pdf](https://arxiv.org/pdf/2502.14770)]
> **Authors**: Weizhong Huang,Yuxin Zhang,Xiawu Zheng,Fei Chao,Rongrong Ji
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 通过理论观点确定大语言模型的层次稀疏性
- **领域**: 机器学习
- **摘要**: 在本文中，我们解决了通过理论观点确定大语言模型（LLM）的层次稀疏率的挑战。具体来说，我们确定了现有LLMS稀疏方法中的“ $ \ textbf {重建错误爆炸} $'”的关键问题。这是指在整个滥用过程中重建误差的累积效应，在此过程中，早期层中的误差在后续层中传播并放大。结果，总体重建误差大大增加，导致模型性能的实质性下降。通过理论分析，我们得出了一种简单而有效的方法来减轻层次分配，从而减轻了这个问题。我们的方法使用单调增加的算术进程，从而降低了多层确定稀疏率的过程，以确定单个共同差异超参数。值得注意的是，这只能通过几次试验来确定最佳的层次稀疏率。我们的理论分析和实验结果都表明，这种稀疏性分配方案几乎是最佳的。广泛的实验表明，我们的方法显着提高了各种体系结构稀疏LLM的性能，从而超过了现有的层次稀疏方法。此外，它增强了各种压缩技术的性能，并且适用于视觉和多模型模型。 Notably, our method achieves a reduction of 52.10 in perplexity for the 70$\%$ sparse LLaMA2-7B model obtained via Wanda, improves average zero-shot accuracy by 10.50$\%$, and delivers speedups of 2.63$\times$ and 2.23$\times$ on CPU and GPU, respectively.

### Composable Strategy Framework with Integrated Video-Text based Large Language Models for Heart Failure Assessment 
[[arxiv](https://arxiv.org/abs/2502.16548)] [[cool](https://papers.cool/arxiv/2502.16548)] [[pdf](https://arxiv.org/pdf/2502.16548)]
> **Authors**: Jianzhou Chen,Xiumei Wang,Jinyang Sun,Xi Chen,Heyu Chu,Guo Song,Yuji Luo,Xingping Zhou,Rong Gu
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 可组合策略框架与集成视频文本的大型语言模型用于心力衰竭评估
- **领域**: 机器学习,人工智能,计算机视觉和模式识别
- **摘要**: 根据世界卫生组织（WHO）和其他公共卫生机构的数据，心力衰竭是全世界死亡的主要原因之一，每年米隆死亡。尽管在心力衰竭领域取得了重大进展，导致生存率提高和射血分数的提高，但由于复杂性和多因素特征，仍然存在很大的未满足需求。因此，我们提出了一个可组合的策略框架，以评估和治疗在心力衰竭中优化。该框架模拟了医生咨询过程，并利用多模式算法来分析一系列数据，包括视频，体格检查，文本结果以及病史。通过整合这些各种数据源，我们的框架为患者提供了更全面的评估和优化的治疗计划。我们的结果表明，这种多模式方法在心力衰竭（HF）预测的准确性方面优于单模式人工智能（AI）算法。通过这种方法，我们可以进一步评估各种病理指标对HF预后的影响，从而提供更全面的评估。

### A Split-Window Transformer for Multi-Model Sequence Spammer Detection using Multi-Model Variational Autoencoder 
[[arxiv](https://arxiv.org/abs/2502.16483)] [[cool](https://papers.cool/arxiv/2502.16483)] [[pdf](https://arxiv.org/pdf/2502.16483)]
> **Authors**: Zhou Yang,Yucai Pang,Hongbo Yin,Yunpeng Xiao
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 使用多模型变量自动编码器的多模型序列垃圾邮件启动器检测的拆分窗口变压器
- **领域**: 机器学习,人工智能,多媒体,社交和信息网络
- **摘要**: 本文介绍了一种新的变压器，称为MS $^2 $ DFORMER，可以用作多模式序列垃圾邮件器检测的通用骨干。垃圾邮件发送者检测是一项复杂的多模式任务，因此应用变压器的挑战是两个方面。首先，有关用户的复杂多模式嘈杂信息可以干扰功能挖掘。其次，用户的历史行为的长序列也给注意力计算带来了巨大的GPU记忆压力。为了解决这些问题，我们首先根据多模式变异自动编码器（MVAE）设计了用户行为令牌化算法。随后，提出了分层分裂窗口多头注意（SW/W-MHA）机制。分裂窗口策略将超长的序列从层次上转化为短期内部和窗口间的整体关注的组合。 MS $^2 $ DFORMER的性能远远超过了以前的最新状态，在公共数据集中进行了预训练。实验证明了MS $^2 $ DFORMER充当骨干的能力。

### MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra 
[[arxiv](https://arxiv.org/abs/2502.16284)] [[cool](https://papers.cool/arxiv/2502.16284)] [[pdf](https://arxiv.org/pdf/2502.16284)]
> **Authors**: Liang Wang,Shaozhen Liu,Yu Rong,Deli Zhao,Qiang Liu,Shu Wu,Liang Wang
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: Accepted by ICLR 2025
- **标题**: Molspectra：具有多模式能光谱的训练前3D分子表示
- **领域**: 机器学习,人工智能,计算工程、金融和科学,化学物理
- **摘要**: 事实证明，建立3D结构与分子系统的能量状态之间的关系是学习3D分子表示的一种有希望的方法。但是，现有方法仅限于对经典力学的分子能状态进行建模。这种局限性导致对量子机械效应的显着监督，例如量化（离散的）能级结构，这些结构可提供更准确的分子能量估计，并且可以通过能量光谱实验测量。在本文中，我们建议利用能量光谱来增强3D分子表示（Molspectra）的预训练，从而将量子力学知识融合到分子表示中。具体而言，我们提出了SpecFormer，这是一种通过掩盖贴片重建来编码分子光谱的多光谱编码器。通过使用对比度目标从3D编码器和光谱编码器中进一步对齐输出，我们增强了3D编码器对分子的理解。对公共基准测试的评估表明，我们的预训练表示超过了预测分子特性和建模动态的现有方法。

### Understanding the Emergence of Multimodal Representation Alignment 
[[arxiv](https://arxiv.org/abs/2502.16282)] [[cool](https://papers.cool/arxiv/2502.16282)] [[pdf](https://arxiv.org/pdf/2502.16282)]
> **Authors**: Megan Tjandrasuwita,Chanakya Ekbote,Liu Ziyin,Paul Pu Liang
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-24
> **comment**: 21 pages, 22 figures, 3 tables
- **标题**: 了解多模式表示对准的出现
- **领域**: 机器学习,人工智能
- **摘要**: 多模式表示学习从根本上讲是将无与伦比的模式转化为可比表示的。虽然先前的研究主要致力于通过针对性的学习目标和模型架构明确调整这些表示形式，但最近的一项工作发现，具有独立训练的量表和性能的独立训练的单峰模型可以彼此隐含地对齐。这些发现引发了有关多模式学习中对齐表示的出现的基本问题。具体：（1）何时以及为什么对齐会隐式出现？ （2）对齐是可靠的性能指标吗？通过全面的实证研究，我们证明了一致性的出现及其与任务绩效的关系取决于几个关键数据特征。这些包括但不一定限于它们为任务提供的冗余和独特信息之间的相似程度与平衡之间的相似程度。我们的发现表明，一致性可能不是普遍有益的。相反，其对性能的影响因数据集和任务而异。这些见解可以帮助实践者确定模式之间的一致性是否有利，或者在某些情况下对实现最佳性能有害。代码在https://github.com/megantj/multimodal_alignment上发布。

### Directional Gradient Projection for Robust Fine-Tuning of Foundation Models 
[[arxiv](https://arxiv.org/abs/2502.15895)] [[cool](https://papers.cool/arxiv/2502.15895)] [[pdf](https://arxiv.org/pdf/2502.15895)]
> **Authors**: Chengyue Huang,Junjiao Tian,Brisa Maneechotesuwan,Shivang Chopra,Zsolt Kira
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: Accepted to ICLR 2025
- **标题**: 方向性梯度投影，用于强大的基础模型微调
- **领域**: 机器学习,人工智能,计算语言学,计算机视觉和模式识别
- **摘要**: 强大的微调旨在使大型基础模型适应下游任务，同时保留其对分配变化的稳健性。现有方法主要集中在基于微调和预训练的权重之间的幅度上，将当前模型限制为预先训练的初始化，这通常需要广泛的高参数调整，有时可能导致适合。在这项工作中，我们提出了方向性梯度投影（DIGRAP），这是一种新颖的可训练方法，它结合了从梯度到桥梁正则化和多目标优化的定向信息。除了展示我们的图像分类方法之外，作为另一个贡献，我们将该领域推广到多模式评估设置以进行健壮的微调。具体而言，我们首先通过对图像分类重新构图的视觉质量答案（VQA）基准进行分析来弥合单模式和多模式差距，并通过分布移位类型和程度（即接近vess versus versus versus versus versus versus vess）进行进一步对十个隔离（OOD）VQA数据集进行进一步分类。实验结果表明，DIGRAP始终在图像分类和VQA任务上均超过现有的基线，并具有歧视性和生成性骨架，从而改善了分布（ID）概括和OOD稳健性。

### FedMobile: Enabling Knowledge Contribution-aware Multi-modal Federated Learning with Incomplete Modalities 
[[arxiv](https://arxiv.org/abs/2502.15839)] [[cool](https://papers.cool/arxiv/2502.15839)] [[pdf](https://arxiv.org/pdf/2502.15839)]
> **Authors**: Yi Liu,Cong Wang,Xingliang Yuan
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-24
> **comment**: The Web Conference 2025
- **标题**: FEDMOBILE：以不完整的方式启用知识贡献的多模式联合学习
- **领域**: 机器学习,人工智能
- **摘要**: 事物的网络（WOT）增强了跨基于Web的和普遍存在的计算平台的互操作性，同时补充了现有的物联网标准。引入了多模式联合学习（FL）范式，以通过在保留隐私的同时融合多源移动传感数据来增强WOT。但是，使用多模式FL的移动传感系统中的一个关键挑战是模态不完整，其中某些模态可能是不可用的或仅部分捕获的，可能会降低系统的性能和可靠性。当前的多模式FL框架通常会训练多个单型号FL子系统或在节点侧应用插值技术以近似丢失的方式。但是，这些方法忽略了不同节点的不完整方式之间共享的潜在特征空间，并且无法区分低质量的节点。为了解决这一差距，我们提出了FedMobile，这是一种新的知识贡献的多模式FL框架，尽管缺失了模式，却是为了健壮学习而设计的。 FedMobile优先考虑局部到全球知识转移，利用跨节点多模式信息来重建缺失的特征。它还通过严格的节点贡献评估和知识贡献 - 意识到的聚合规则来增强系统性能和对方式异质性的韧性。对五个公认的多模式基准数据集进行的经验评估表明，即使缺少多达90％的模态信息，FedMobile仍保持强大的学习，或者当来自两个模态的数据随机丢失时，超过了最先进的盆地。

### Challenges of Multi-Modal Coreset Selection for Depth Prediction 
[[arxiv](https://arxiv.org/abs/2502.15834)] [[cool](https://papers.cool/arxiv/2502.15834)] [[pdf](https://arxiv.org/pdf/2502.15834)]
> **Authors**: Viktor Moskvoretskii,Narek Alvandian
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 多模式核心选择深度预测的挑战
- **领域**: 机器学习,机器学习
- **摘要**: 核心选择方法可有效加速训练和减少记忆要求，但在应用的多模式设置中仍未探索。我们将最新的（SOTA）核心选择技术适应多模式数据，重点是深度预测任务。我们通过嵌入聚合和降低降低方法的实验揭示了将单峰算法扩展到多模式场景的挑战，强调了对更好地捕获模式间关系的专业方法的需求。

### InsightVision: A Comprehensive, Multi-Level Chinese-based Benchmark for Evaluating Implicit Visual Semantics in Large Vision Language Models 
[[arxiv](https://arxiv.org/abs/2502.15812)] [[cool](https://papers.cool/arxiv/2502.15812)] [[pdf](https://arxiv.org/pdf/2502.15812)]
> **Authors**: Xiaofei Yin,Yijie Hong,Ya Guo,Yi Tu,Weiqiang Wang,Gongshen Liu,Huijia zhu
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-24
> **comment**: 19 pages, 10 figures
- **标题**: InsightVision：一种全面的，基于中文的基于中文的基准，用于评估大型视觉语言模型中的隐性视觉语义
- **领域**: 机器学习,人工智能
- **摘要**: 在多模式模型不断发展的景观中，了解通过视觉提示传达的细微含义（例如讽刺，侮辱或批评）仍然是一个重大挑战。现有的评估基准主要集中于图像字幕之类的直接任务，或仅限于狭窄的类别，例如幽默或讽刺，以进行深入的语义理解。为了解决这一差距，我们首次介绍了一种全面的，基于中文的基于中文的基准，专门用于评估对图像中隐性含义的理解。该基准系统被系统地分为四个子任务：表面级内容的理解，象征意义解释，背景知识理解和隐式意义理解。我们提出了一种创新的半自动方法，用于构建数据集，并遵守已建立的施工协议。使用此基准，我们评估了15种开源大型视觉语言模型（LVLM）和GPT-4O，这表明即使是人类绩效落后于人类绩效的最佳模型也落在了近14％的落后于14％方面。我们的发现强调了当前LVLM在掌握细微差别的视觉语义方面所面临的内在挑战，从而强调了该领域未来研究和发展的重要机会。我们将公开发布我们的InsightVision数据集，即接受论文后的代码。

### Megrez-Omni Technical Report 
[[arxiv](https://arxiv.org/abs/2502.15803)] [[cool](https://papers.cool/arxiv/2502.15803)] [[pdf](https://arxiv.org/pdf/2502.15803)]
> **Authors**: Boxun Li,Yadong Li,Zhiyuan Li,Congyi Liu,Weilin Liu,Guowei Niu,Zheyue Tan,Haiyang Xu,Zhuyu Yao,Tao Yuan,Dong Zhou,Yueqing Zhuang,Shengen Yan,Guohao Dai,Yu Wang
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: Megrez-Omni技术报告
- **领域**: 机器学习,计算语言学
- **摘要**: 在这项工作中，我们介绍了包括语言模型（Megrez-3b-Instruct）和多模式模型（Megrez-3b-3b-Omni）的Megrez模型。这些模型旨在通过软件硬件共同设计方法提供快速的推理，紧凑性和鲁棒的边缘智能。 Megrez-3b-Instruct提供了几种优势，包括高精度，高速，易用性和广泛的应用。 Megrez-3b-omni在Megrez-3b-Instruct的基础上是一个支持图像，文本和音频分析的设备多模式理解LLM。它在所有三种方式上都达到了最新的精度，并表现出强烈的多功能性和鲁棒性，为多模式AI模型树立了新的基准。

### Anomaly Detection in Smart Power Grids with Graph-Regularized MS-SVDD: a Multimodal Subspace Learning Approach 
[[arxiv](https://arxiv.org/abs/2502.15793)] [[cool](https://papers.cool/arxiv/2502.15793)] [[pdf](https://arxiv.org/pdf/2502.15793)]
> **Authors**: Thomas Debelle,Fahad Sohrab,Pekka Abrahamsson,Moncef Gabbouj
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-24
> **comment**: 20 pages, 5 figures, supplementary material
- **标题**: 具有图形调节的MS-SVDD中的智能电网中的异常检测：一种多模式子空间学习方法
- **领域**: 机器学习,系统与控制
- **摘要**: 在本文中，我们使用多模式子空间支持向量数据描述（MS-SVDD）解决了智能电网中的异常检测问题。这种方法旨在通过将数据视为来自不同方式的数据来利用更好的功能关系。这些数据被投影到共享的低维子空间中，旨在保留其内部特征。为了补充有关此主题的先前工作，我们介绍了新型的多模式图形的正规化器，以利用每种方式的图形信息来增强训练过程，并且我们考虑了改进的训练方程式，使我们可以根据指定的标准最大化或最小化每个模态。在将MS-SVDD算法概括为任何数量的模态之后，我们将此正则化的图形模型应用于3模式数据集。为了设置我们的应用程序，我们提出了一个整个预处理程序，以从时期的事件时间序列中提取一级分类培训实例，该实例用于评估我们模型的可靠性和初步性以进行事件检测。

### Detecting Content Rating Violations in Android Applications: A Vision-Language Approach 
[[arxiv](https://arxiv.org/abs/2502.15739)] [[cool](https://papers.cool/arxiv/2502.15739)] [[pdf](https://arxiv.org/pdf/2502.15739)]
> **Authors**: D. Denipitiyage,B. Silva,S. Seneviratne,A. Seneviratne,S. Chawla
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-24
> **comment**: 11 pages, 8 figures
- **标题**: 在Android应用中检测违反内容违反内容：一种视觉语言方法
- **领域**: 机器学习,计算机视觉和模式识别,多媒体
- **摘要**: 尽管监管努力为移动应用程序建立可靠的内容评级指南，但在Google Play商店中分配内容评分的过程仍由应用程序开发人员自我调节。由于量表压倒性量表，或者由于解释文本和视觉数据并将其与内容评分相关联的挑战性问题，因此没有直接的方法可以手动验证开发人员分配的内容额定值。我们建议使用流行的Android游戏的元数据数据集预测和评估一种视觉语言方法，以预测移动游戏应用程序的内容评分并检测违反内容评分。与在多模式设置中的最先进的剪贴式调整模型相比，我们的方法相对准确度优于6％。在野外应用分类器时，我们发现了70多个可能的违反内容评级案件，包括九个实例，其中包括“老师批准”徽章。此外，我们的发现表明，我们分类器确定为违反内容评分的应用程序中有34.5％从Play商店中删除了。相比之下，正确分类应用程序的去除率仅为27％。这种差异突出了分类器在识别可能会根据用户投诉删除的应用程序的实际有效性。

### The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning 
[[arxiv](https://arxiv.org/abs/2502.15214)] [[cool](https://papers.cool/arxiv/2502.15214)] [[pdf](https://arxiv.org/pdf/2502.15214)]
> **Authors**: Sheila Schoepp,Masoud Jafaripour,Yingyue Cao,Tianpei Yang,Fatemeh Abdollahi,Shadan Golestan,Zahin Sufiyan,Osmar R. Zaiane,Matthew E. Taylor
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 9 pages, 4 figures
- **标题**: LLM和VLM综合增强学习的不断发展的景观
- **领域**: 机器学习,人工智能,计算语言学
- **摘要**: 强化学习（RL）在顺序决策任务中显示出令人印象深刻的结果。同时，出现了大型语言模型（LLM）和视觉模型（VLM），在多模式理解和推理方面表现出了令人印象深刻的能力。这些进步导致了将LLM和VLMS整合到RL的研究激增。在这项调查中，我们审查了代表性的作品，其中LLM和VLM被用来克服RL中的关键挑战，例如缺乏先验知识，长期学习计划和奖励设计。我们提出了一种分类法，将这些LLM/VLM辅助RL方法分为三个角色：代理，计划者和奖励。我们结论是探索开放问题，包括基础，缓解偏见，改进的表示和行动建议。通过巩固现有的研究并确定未来的方向，该调查建立了将LLM和VLM集成到RL中的框架，并提高了将自然语言和视觉理解与顺序决策统一的方法。

### MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks 
[[arxiv](https://arxiv.org/abs/2502.17832)] [[cool](https://papers.cool/arxiv/2502.17832)] [[pdf](https://arxiv.org/pdf/2502.17832)]
> **Authors**: Hyeonjeong Ha,Qiusi Zhan,Jeonghwan Kim,Dimitrios Bralios,Saikrishna Sanniboina,Nanyun Peng,Kai-Wei Chang,Daniel Kang,Heng Ji
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG
- **标题**: MM-PoisonRag：用局部和全球中毒攻击破坏多模式抹布
- **领域**: 机器学习,人工智能,密码学和安全,计算机视觉和模式识别
- **摘要**: 配备有检索增强生成（RAG）的多模式大语言模型（MLLM）都利用了他们丰富的参数知识和动态，外部知识在诸如问题回答之类的任务中表现出色。尽管RAG通过将与查询相关的外部知识的响应接地来增强MLLM，但这种依赖构成了一个关键但毫无疑问的安全风险：知识中毒攻击，其中错误的信息或无关的知识被故意注入外部知识基础，以操纵模型输出，以使模型输出不正确，甚至有害。为了在多模式抹布中暴露这种漏洞，我们提出了MM-PoisonRag，这是一种新型的知识中毒攻击框架，具有两种攻击策略：局部中毒攻击（LPA），该攻击（LPA）在有针对性的操纵中注入了特定的文本和图像，以涉及全球性中毒攻击（GPA），以涉足MM，以在MM中提供了较高的指导。我们评估了跨多个任务，模型和访问设置的攻击，表明LPA成功地操纵了MLLM以生成攻击者控制的答案，而多模态的成功率高达56％。此外，GPA仅通过一次不相关的知识注入将模型生成限制为0％的准确性。我们的结果强调了迫切需要防御知识中毒以保护多模式抹布框架的必要性。

### Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations 
[[arxiv](https://arxiv.org/abs/2502.17618)] [[cool](https://papers.cool/arxiv/2502.17618)] [[pdf](https://arxiv.org/pdf/2502.17618)]
> **Authors**: Sangwon Seo,Vaibhav Unhelkar
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: Extended version of an identically-titled paper accepted at AAMAS 2025
- **标题**: 从异质示范中学习团队行为的分层模仿
- **领域**: 机器学习,人工智能,多代理系统
- **摘要**: 成功的协作要求团队成员保持一致，尤其是在复杂的顺序任务中。团队成员必须动态协调要执行哪些子任务以及按什么顺序进行。但是，诸如部分可观察性和有限的沟通带宽等现实世界的约束通常会导致次优协作。即使在专家团队中，也可以多种方式执行相同的任务。为了开发用于此类任务的多代理系统和人类AI团队，我们对数据驱动的多模式团队行为的学习感兴趣。多机构模仿学习（MAIL）为数据驱动的团队行为学习提供了一个有希望的框架，但是现有的方法与异质示威的斗争，因为他们认为所有示威活动均来自单个团队策略。因此，在这项工作中，我们介绍了DTIL：一种层次邮件算法，旨在在复杂的顺序任务中学习多模式团队行为。 DTIL代表每个团队成员的分层政策，并以偏向的方式从异质团队示威中学习这些政策。通过采用分配匹配方法，DTIL减轻了将复合错误和尺度有效地缩小到长度的状态和连续状态表示形式。实验结果表明，DTIL的表现优于邮件基线，并在各种协作场景中准确地对团队行为进行建模。

### A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models 
[[arxiv](https://arxiv.org/abs/2502.17516)] [[cool](https://papers.cool/arxiv/2502.17516)] [[pdf](https://arxiv.org/pdf/2502.17516)]
> **Authors**: Zihao Lin,Samyadeep Basu,Mohammad Beigi,Varun Manjunatha,Ryan A. Rossi,Zichao Wang,Yufan Zhou,Sriram Balasubramanian,Arman Zarei,Keivan Rezaei,Ying Shen,Barry Menglong Yao,Zhiyang Xu,Qin Liu,Yuxiang Zhang,Yan Sun,Shilong Liu,Li Shen,Hongxuan Li,Soheil Feizi,Lifu Huang
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-25
> **comment**: 30 pages, 4 Figures, 10 Tables
- **标题**: 关于多模式基础模型的机械解释性的调查
- **领域**: 机器学习,人工智能
- **摘要**: 基础模型的兴起已经改变了机器学习研究，促使努力揭示其内部运作，并开发更高效，更可靠的应用程序以更好地控制。尽管在解释大语言模型（LLM），多模式基础模型（MMFMS）（例如对比视觉模型，生成视觉语言模型和文本对图像模型）方面取得了重大进展，并在非兴趣框架之外提出了独特的解释性挑战。尽管进行了初步研究，但LLM和MMFM的可解释性之间仍然存在很大的差距。该调查探讨了两个关键方面：（1）LLM可解释性方法对多模型模型的适应，以及（2）了解单峰语言模型和跨模式系统之间的机械差异。通过系统地审查当前的MMFM分析技术，我们提出了一种结构化分类法的解释性分类法，比较单峰和多模式架构之间的见解，并突出关键的研究差距。

### SAE-V: Interpreting Multimodal Models for Enhanced Alignment 
[[arxiv](https://arxiv.org/abs/2502.17514)] [[cool](https://papers.cool/arxiv/2502.17514)] [[pdf](https://arxiv.org/pdf/2502.17514)]
> **Authors**: Hantao Lou,Changye Li,Jiaming Ji,Yaodong Yang
> **First submission**: 2025-02-22
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: SAE-V：解释多模型以增强对齐方式
- **领域**: 机器学习,人工智能,计算语言学
- **摘要**: 随着图像方式的整合，多模式大语言模型（MLLM）的语义空间比仅文本模型更为复杂，这使得它们的可解释性更具挑战性和对齐方式降低，尤其是对低质量数据的影响，这可能会导致模态，幻觉，幻觉，幻觉和偏见的输出之间的不一致性。结果，开发MLLM的可解释性方法对于提高对齐质量和效率至关重要。在仅使用文本的LLM中，稀疏的自动编码器（SAE）因其解释潜在表示的能力而引起了人们的关注。但是，将SAE扩展到多模式设置，由于模态融合和隔离跨模式表示的难度而提出了新的挑战。为了应对这些挑战，我们引入了SAE-V，这是一种机械性解释性框架，将SAE范式扩展到MLLM。通过识别和分析可解释的特征及其相应的数据，SAE-V可以对模型行为和数据质量进行细粒度的解释，从而更深入地了解跨模式相互作用和对齐动态。此外，通过利用跨模式特征加权，SAE-V提供了一种固有的数据滤波机制来增强模型对齐方式，而无需其他模型。具体而言，当应用于MLLM的对齐过程时，基于SAE-V的数据过滤方法可以实现超过110％的性能，而数据少于50％。我们的结果突出了SAE-V增强MLLM中的可解释性和一致性的能力，从而提供了对其内部机制的见解。

### Towards Hierarchical Rectified Flow 
[[arxiv](https://arxiv.org/abs/2502.17436)] [[cool](https://papers.cool/arxiv/2502.17436)] [[pdf](https://arxiv.org/pdf/2502.17436)]
> **Authors**: Yichi Zhang,Yici Yan,Alex Schwing,Zhizhen Zhao
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: ICLR 2025; Project Page: https://riccizz.github.io/HRF/
- **标题**: 朝向分层整流流
- **领域**: 机器学习,计算机视觉和模式识别
- **摘要**: 我们制定了分层整流的流程，以建模数据分布。它从层次上耦合了多个普通微分方程（ODE），并定义了一个时间差异的随机过程，该过程从已知的源分布中生成数据分布。每种极类似于在经典的整流流中解决的颂歌，但在其域中有所不同，即位置，速度，加速度等。与经典的整流流程配方不同，它在位置域中的单个驱动器不同，仅捕获了预期的速度字段（足以捕获一个多型模型数据分布），该速度的随机模型）速度字段，加速场等。随机速度场的这种更忠实的建模可以使整合路径在数据生成过程中求解底层ode时相交。相交的路径反过来导致集成轨迹比在经典整流的流动公式中获得的轨迹更直，在整合路径无法相交的情况下。这导致具有更少神经功能评估的数据分布建模。我们在合成的1D和2D数据以及MNIST，CIFAR-10和Imagenet-32数据上进行经验验证这一点。我们的代码可在以下网址提供：https：//riccizz.github.io/hrf/。

### Diffusion Models for Tabular Data: Challenges, Current Progress, and Future Directions 
[[arxiv](https://arxiv.org/abs/2502.17119)] [[cool](https://papers.cool/arxiv/2502.17119)] [[pdf](https://arxiv.org/pdf/2502.17119)]
> **Authors**: Zhong Li,Qi Huang,Lincen Yang,Jiayang Shi,Zhao Yang,Niki van Stein,Thomas Bäck,Matthijs van Leeuwen
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 表格数据的扩散模型：挑战，当前进度和未来方向
- **领域**: 机器学习,人工智能
- **摘要**: 近年来，生成模型在各种应用程序中取得了出色的性能，包括图像生成，文本综合，音频创建，视频生成和数据增强。扩散模型已成为生成对抗网络（GAN）和变化自动编码器（VAE）的优越替代品，例如解决它们的局限性，例如训练不稳定性，模式崩溃和多模式分布的表示不良。这一成功激发了广泛的研究兴趣。在表格数据的域中，扩散模型已经开始展示与gan和vaes相似的优势，从而实现了重大的性能突破，并展示了它们在表格数据建模中的独特挑战的潜力。但是，尽管图像和时间序列之类的域有许多调查总结了扩散模型中的进步，但文献中对于表格数据仍然存在显着差距。尽管对表格数据的扩散模型的兴趣越来越大，但系统地审查和总结了这些发展的努力很少。缺乏专门的调查限制了对这个关键领域中的挑战，进步和未来方向的清晰了解。该调查通过对表格数据的扩散模型进行全面审查来解决这一差距。涵盖2015年6月的工作，当出现扩散模型到2024年12月时，我们分析了几乎所有相关研究，并在\ href {https://github.com/diffusion-model-leiden/awsome-leiden/awsome-diffusy-models-models-models-for-tabular-data}中维护了更新。假设读者具有统计和扩散模型的基本知识，我们采用数学公式来提供严格而详细的审查，旨在促进这一新兴和令人兴奋的领域的发展。

### Distributional Vision-Language Alignment by Cauchy-Schwarz Divergence 
[[arxiv](https://arxiv.org/abs/2502.17028)] [[cool](https://papers.cool/arxiv/2502.17028)] [[pdf](https://arxiv.org/pdf/2502.17028)]
> **Authors**: Wenzhe Yin,Zehao Xiao,Pan Zhou,Shujian Yu,Jiayi Shen,Jan-Jakob Sonke,Efstratios Gavves
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: Cauchy-Schwarz的分布视觉对齐差异
- **领域**: 机器学习,人工智能
- **摘要**: 多模式比对对于诸如跨模式生成和检索等各种下游任务至关重要。先前的多模式方法（例如剪辑）诸如剪辑之类的方法主要通过跨模态对齐样品，同时忽略分布差异，从而最大程度地提高了相互信息。在本文中，为了克服限制，我们提出了CS-Aligner，这是一个新颖而直接的框架，通过将Cauchy-Schwarz（CS）差异与相互信息整合在一起，通过将Cauchy-Schwarz（CS）差异整合来执行分布视觉的路线。在拟议的框架中，我们发现CS差异和相互信息在多模式对齐中起互补的作用，从而捕获了每种模态的全局分布信息和成对的语义关系，从而产生了更严格，更精确的对准。此外，CS-Eligher启用了从未配对的数据和令牌级表示中的其他信息，从而在实践中增强了灵活和细粒度的对齐方式。关于文本形象生成和跨模式检索任务的实验证明了我们方法对视觉对齐的有效性。

### M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance 
[[arxiv](https://arxiv.org/abs/2502.18778)] [[cool](https://papers.cool/arxiv/2502.18778)] [[pdf](https://arxiv.org/pdf/2502.18778)]
> **Authors**: Qingpei Guo,Kaiyou Song,Zipeng Feng,Ziping Ma,Qinglong Zhang,Sirui Gao,Xuzheng Yu,Yunxiao Sun,Tai-Wei Chang,Jingdong Chen,Ming Yang,Jun Zhou
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: M2-omni：推进Omni-Mllm，以获得竞争性能的全面方式支持
- **领域**: 机器学习,人工智能,计算语言学
- **摘要**: 我们提出了M2-omni，这是一种尖端的开源Omni-Mllm，可实现GPT-4O的竞争性能。 M2-OMNI采用统一的多模式序列建模框架，该框架授权大型语言模型（LLMS）获得综合的跨模式理解和发电能力。具体而言，M2-OMNI可以处理音频，视频，图像和文本模式的任意组合，作为输入，生成与音频，图像或文本输出相互交织的多模式序列，从而启用高级和交互式的实时体验。这种OMNI-MLLM的培训受到跨模式的数据数量和收敛率的显着差异的挑战。为了应对这些挑战，我们在预培训期间提出了一个步骤平衡策略，以处理特定于模态数据中的数量差异。此外，在教学调整阶段中引入了动态自适应平衡策略，以同步模态训练进度，从而确保最佳收敛。值得注意的是，我们优先考虑在纯文本任务上保持强大的绩效，以保持M2-omni语言理解能力在整个培训过程中的稳健性。据我们所知，M2-OMNI目前是GPT-4O的非常有竞争力的开源模型，其特征是其全面的方式和任务支持及其出色的性能。我们预计M2-OMNI将推进Omni-Mllms的发展，从而促进该领域的未来研究。

### Cross-Modality Investigation on WESAD Stress Classification 
[[arxiv](https://arxiv.org/abs/2502.18733)] [[cool](https://papers.cool/arxiv/2502.18733)] [[pdf](https://arxiv.org/pdf/2502.18733)]
> **Authors**: Eric Oliver,Sagnik Dakshit
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: Wesad应力分类的跨模式调查
- **领域**: 机器学习,人工智能
- **摘要**: 深度学习的越来越流行驱动了其在医疗保健中的广泛使用，其中AI和传感器的进步增强了诊断，治疗和监测。在移动健康中，AI驱动的工具可以早期诊断并持续监测压力等条件。可穿戴技术和多模式生理数据使压力检测越来越可行，但模型功效取决于数据质量，数量和方式。这项研究开发了使用WESAD数据集，心电图训练（ECG），电肌活动（EDA），肌电图（EMG），呼吸率（RESS），温度（温度）和3轴加速度计（ACC）信号的变压器模型。结果表明，单模式变压器在分析生理信号中的有效性，以准确性，精度和召回率在$ 99.73 \％$至$ 99.95 \％\％$ $ 99.95 \％$ $ $ $ $ $ $ $ $ $ $ $中，以实现最先进的性能。此外，这项研究探讨了跨模式的性能，并使用基于数据差异的学习嵌入空间和定量分析的2D可视化也解释了这一点。尽管在压力检测和监测方面进行了大量工作，但尚未探索这些模型的鲁棒性和概括性。这项研究是解释嵌入空间以进行压力检测的最初努力之一，提供了有关跨模式性能的宝贵信息。

### Patient Trajectory Prediction: Integrating Clinical Notes with Transformers 
[[arxiv](https://arxiv.org/abs/2502.18009)] [[cool](https://papers.cool/arxiv/2502.18009)] [[pdf](https://arxiv.org/pdf/2502.18009)]
> **Authors**: Sifal Klioui,Sana Sellami,Youssef Trardi
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 患者轨迹预测：将临床笔记与变压器整合
- **领域**: 机器学习
- **摘要**: 从电子健康记录（EHRS）中预测疾病轨迹是一项复杂的任务，这是由于主要挑战，例如数据非平稳性，医疗法规的高粒度以及多模式数据的整合。 EHR既包含结构化数据，例如诊断代码，也包含非结构化数据，例如临床注释，这些数据通常会忽略基本信息。当前的模型主要基于结构化数据，难以捕获患者的完整医疗环境，从而导致损失有价值的信息。为了解决这个问题，我们提出了一种方法，将非结构化临床注释整合到基于变压器的深度学习模型中，以进行顺序疾病预测。这种整合丰富了患者病史的表示，从而提高了诊断预测的准确性。对模拟物IV数据集的实验表明，所提出的方法的表现优于仅依赖结构化数据的传统模型。

### Broadening Discovery through Structural Models: Multimodal Combination of Local and Structural Properties for Predicting Chemical Features 
[[arxiv](https://arxiv.org/abs/2502.17986)] [[cool](https://papers.cool/arxiv/2502.17986)] [[pdf](https://arxiv.org/pdf/2502.17986)]
> **Authors**: Nikolai Rekut,Alexey Orlov,Klea Ziu,Elizaveta Starykh,Martin Takac,Aleksandr Beznosikov
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 通过结构模型扩大发现：局部和结构特性的多模式组合，用于预测化学特征
- **领域**: 机器学习,人工智能
- **摘要**: 近年来，机器学习已深刻地重塑了化学领域，从而促进了各种应用的显着进步，包括对分子特性的预测和分子结构的产生。语言模型和基于图的模型在该域中广泛使用，始终在一系列任务中实现最新结果。但是，以微笑格式表示化学化合物的主要实践（大多数数据集和许多语言模型都使用）将显着的限制作为培训数据格式。相比之下，化学指纹提供了更明智的化合物表示，从而增强了其对模型训练的适用性。这项研究旨在开发一种专门针对指纹训练的语言模型。此外，我们介绍了将该语言模型与图形模型集成的双峰体系结构。我们提出的方法将这些方法综合了这些方法，利用罗伯塔作为语言模型，并采用图形同构网络（GIN），图形卷积网络（GCN）和图形机器作为图形模型。与定量结构 - 活性关系（QSAR）和核磁共振（NMR）光谱等的常规策略相比，这种整合导致预测性能的显着改善。

### Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs 
[[arxiv](https://arxiv.org/abs/2502.17900)] [[cool](https://papers.cool/arxiv/2502.17900)] [[pdf](https://arxiv.org/pdf/2502.17900)]
> **Authors**: Che Liu,Cheng Ouyang,Zhongwei Wan,Haozhe Wang,Wenjia Bai,Rossella Arcucci
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 知识增强的多模式心电图表示与任意铅输入的学习
- **领域**: 机器学习,人工智能
- **摘要**: 多模式心电图表示学习中心的最新进展与配对的自由文本报告对齐ECG信号。但是，由于医学语言的复杂性和对完整的12铅设置的依赖，次优路线持续存在，在资源不足的设置中通常无法使用。为了解决这些问题，我们建议** k-merl **，这是一个知识增强的多模式心电图表示框架。 ** k-merl **利用大型语言模型从自由文本报告中提取结构化知识，并使用带有动态铅掩蔽的铅感应ECG编码器来容纳任意铅输入。对六个外部ECG数据集的评估表明，** k-merl **在零摄像分类和线性探测任务中实现了最先进的性能，同时提供了平均** 16％** AUC对部分潜在铅零发射分类的现有方法的改进。

### Unlocking Multi-Modal Potentials for Dynamic Text-Attributed Graph Representation 
[[arxiv](https://arxiv.org/abs/2502.19651)] [[cool](https://papers.cool/arxiv/2502.19651)] [[pdf](https://arxiv.org/pdf/2502.19651)]
> **Authors**: Yuanyuan Xu,Wenjie Zhang,Ying Zhang,Xuemin Lin,Xiwei Xu
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 解锁动态文本属性图表示的多模式电势
- **领域**: 机器学习,计算语言学
- **摘要**: 动态文本属性图（dytags）是一种新颖的图形范式，可捕获不断发展的时间边缘与丰富的文本属性。代表dytags的先前方法利用预先训练的语言模型编码文本属性，然后将其集成到动态图模型中。但是，它遵循以边缘为中心的建模，就像在动态图学习中一样，在局部结构中受到限制，无法利用dytag的独特特征，从而导致次优性能。我们观察到，Dytag固有地包括三种不同的方式，即暂时性，文本和结构性的表现出分散甚至是正交分布，前两个在现有研究中很大程度上被忽略了。在这个洞察力的基础上，我们建议时刻，这是一个模型不合时宜的多模式框架，可以与动态图形模型无缝集成以进行结构模态学习。核心思想是从中心中心转向以节点为中心的建模，完全利用了三种模式用于节点表示。具体而言，Moment基于注意机制呈现非共享节点的编码器，以从时间和文本方式捕获全局时间和语义上下文以及本地结构学习，从而生成特定于模态的令牌。为了防止脱节潜在空间，我们提出了一种对称对准损失，这是一个辅助目标，它使时间和文本令牌保持一致，从而确保全球时间的语义一致性与理论保证。最后，我们设计了一个轻巧的适配器来融合这些令牌，从而产生全面和凝聚力的节点表示。从理论上讲，我们证明了时刻增强了对以边缘为中心建模的判别能力。在七个数据集和两个下游任务上进行的广泛实验表明，使用四个动态图模型，相对于基线，最大提高了33.62％。

### Random Similarity Isolation Forests 
[[arxiv](https://arxiv.org/abs/2502.19122)] [[cool](https://papers.cool/arxiv/2502.19122)] [[pdf](https://arxiv.org/pdf/2502.19122)]
> **Authors**: Sebastian Chwilczyński,Dariusz Brzezinski
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 随机相似性隔离林
- **领域**: 机器学习
- **摘要**: 随着预测模型的流行，公司正在扩大他们收集的数据类型。结果，收集的数据集不仅由简单的数值特征，还包括更复杂的对象，例如时间序列，图像或图形。这样的多模式数据有可能提高预测任务中的性能，例如离群检测，其目标是识别偏离主要数据分布的对象。但是，当前的异常检测算法专用于各个类型的数据。因此，使用混合类型的数据需要融合多个数据特定模型或将所有表示形式转换为单个格式，这两种形式都可以阻碍预测性能。在本文中，我们提出了一种称为随机相似性隔离林的多模式离群检测算法。我们的方法结合了隔离和基于相似性的投影的概念，以处理数据集，并具有任意数据类型的特征的混合物。在47个基准数据集上进行的实验表明，随机相似性隔离森林的表现优于五个最先进的竞争对手。我们的研究表明，使用多种模式确实可以改善异常的检测，并突出了针对多模式算法量身定制的新离群检测基准的需求。

### TabGLM: Tabular Graph Language Model for Learning Transferable Representations Through Multi-Modal Consistency Minimization 
[[arxiv](https://arxiv.org/abs/2502.18847)] [[cool](https://papers.cool/arxiv/2502.18847)] [[pdf](https://arxiv.org/pdf/2502.18847)]
> **Authors**: Anay Majee,Maria Xenochristou,Wei-Peng Chen
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Accepted to AAAI 2025
- **标题**: TABGLM：通过多模式一致性最小化学习可转移表示的表格图形语言模型
- **领域**: 机器学习
- **摘要**: 在表格数据集中处理异质数据对深度学习模型构成了重大挑战。尽管基于注意力的体系结构和自我监督的学习取得了显着的成功，但它们对表格数据的应用在线性和基于树的模型中的效率仍然不那么效率。尽管模型已经实现了几个突破，这些模型将表转换为图像，语言和图形等单模式变换，但在存在特征异质性的情况下，这些模型通常表现不佳。为了解决此差距，我们介绍了TABGLM（表格图语言模型），这是一种新型的多模式体系结构，旨在模拟表格中的结构和语义信息。 TABGLM将表的每一行转换为完全连接的图形和序列化文本，然后分别使用图神经网络（GNN）和文本编码对其进行编码。通过通过联合，多模式，自我监督的学习目标对齐这些表示形式，TABGLM利用这两种模式的互补信息，从而增强了特征学习。 TABGLM的灵活的图形管道有效地处理异质数据集，而在现有的深度学习方法中，参数较少。 25个基准数据集的评估证明了大量的性能提高，而TABGLM的平均AUC-ROC提高了高达5.56％的AUC-ROC（SOTA）表格学习方法。

### Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Inference 
[[arxiv](https://arxiv.org/abs/2502.20587)] [[cool](https://papers.cool/arxiv/2502.20587)] [[pdf](https://arxiv.org/pdf/2502.20587)]
> **Authors**: Mingyuan Wu,Jize Jiang,Haozhen Zheng,Meitang Li,Zhaoheng Li,Beitong Tian,Bo Chen,Yongjoo Park,Minjia Zhang,Chengxiang Zhai,Klara Nahrstedt
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Mingyuan, Jize, and Haozhen contributed equally, while Minjia, Chengxiang, and Klara advised equally
- **标题**: 经过思考：具有成本效益的视觉语言模型推理的大师批准框架
- **领域**: 机器学习
- **摘要**: 视觉语言模型（VLM）在增加复杂性和尺度的广泛视力应用中取得了显着的成功，但选择正确的VLM模型大小涉及响应质量和成本之间的权衡。虽然较小的VLM可以运行便宜，但它们通常比在MMMU等基准上的随机猜测要好得多。在本文中，我们提出了思想的缓存（COT），这是一个硕士学徒框架，用于大型和小VLM之间的协作推断。 COT管理高质量查询是由缓存中的大VLM（主）产生的，然后通过新型的多模态检索和秘密学习来选择该问题，以帮助小型VLM（学徒）的表现。我们对各种公认和具有挑战性的一般VQA基准进行了广泛评估COT，并表明COT在相同的预算下将总体VQA绩效提高了高达7.7％，并特别使学徒VLMS的绩效提高了36.6％。

### Data Distributional Properties As Inductive Bias for Systematic Generalization 
[[arxiv](https://arxiv.org/abs/2502.20499)] [[cool](https://papers.cool/arxiv/2502.20499)] [[pdf](https://arxiv.org/pdf/2502.20499)]
> **Authors**: Felipe del Río,Alain Raymond-Sáez,Daniel Florea,Rodrigo Toro Icarte,Julio Hurtado,Cristián Buc Calderón,Álvaro Soto
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 数据分布性能作为系统概括的电感偏差
- **领域**: 机器学习
- **摘要**: 深度神经网络（DNNS）在系统概括（SG）上挣扎。几项研究评估了通过新的结构，损失功能或训练方法提出的提议来促进SG的可能性。然而，很少有研究集中在培训数据特性在促进SG中的作用。在这项工作中，我们研究了某些数据分布属性的影响，作为多模式语言模型SG能力的归纳偏见。为此，我们研究了三种不同的属性。首先，数据多样性是将训练分布中潜在特性的可能值提高而实例化的。其次，爆发性，我们可能会在训练过程中限制特定输入的潜在因素的可能值数量。第三，潜在干预，其中特定的潜在因素在训练过程中随机改变。我们发现，这三个因素都显着增强了SG，多样性在受影响最大的财产中的准确性绝对增加了89％。通过一系列实验，我们测试了各种假设，以了解这些特性为何促进SG。最后，我们发现训练分布中的潜在属性之间的归一化互信息（NMI）强烈预测分布式概括。我们发现，较低NMI诱导SG的机制是表示形式的几何形状。特别是，我们发现NMI在模型的神经表示（即，在平行神经向量中编码的输入特征）中诱导更多的并行性，这是一种与类比的推理能力相关的属性。

### R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts 
[[arxiv](https://arxiv.org/abs/2502.20395)] [[cool](https://papers.cool/arxiv/2502.20395)] [[pdf](https://arxiv.org/pdf/2502.20395)]
> **Authors**: Zhongyang Li,Ziyue Li,Tianyi Zhou
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: R2-T2：在测试时重新穿线，用于多峰混合物
- **领域**: 机器学习
- **摘要**: 在大型多模式模型（LMM）中，对非语言模式（例如，视觉表示）的感知通常与大语言模型（LLMS）的“强大推理能力相提并论），这阻止了LMMS在具有挑战性的下游任务上的表现。最近，通过用专家的混合物（MOE）代替视觉编码器，从而提供了这种弱点，该混合物提供了丰富的，多跨性的性能和多样化的下游任务所需的多种表示。多模式MOE的性能在很大程度上取决于其路由器，该路由器重量并混合了每个输入的不同专家的表示。但是，我们发现，端到端训练的路由器并不总是为每个测试样品产生最佳路由权重。为了弥合差距，我们提出了一种新颖而有效的方法“在测试时间（R2-T2）中重新布置”，该方法通过将其移动到测试样品附近正确预测的样品的矢量来局部优化测试时间中的路由权重的向量。我们提出了三种具有不同优化目标和邻居搜索空间的R2-T2策略。 R2-T2始终如一，大大提高了最先进的LMM在不同任务的基准方面的表现，而无需培训任何基本模型参数。

### Walking the Web of Concept-Class Relationships in Incrementally Trained Interpretable Models 
[[arxiv](https://arxiv.org/abs/2502.20393)] [[cool](https://papers.cool/arxiv/2502.20393)] [[pdf](https://arxiv.org/pdf/2502.20393)]
> **Authors**: Susmit Agrawal,Deepika Vemuri,Sri Siddarth Chakaravarthy P,Vineeth N. Balasubramanian
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 8 pages of main text, 6 figures in main text, 11 pages of Appendix, published in AAAI 2025
- **标题**: 以渐进训练的可解释模型走上概念类关系的网络
- **领域**: 机器学习,人工智能,计算机视觉和模式识别
- **摘要**: 基于概念的方法已成为在标准监督设置中开发可解释的神经网络的有希望的方向。但是，大多数在渐进环境中研究它们的工作都假定在所有经验中设置的静态概念，或者假设每种体验都依赖于一组不同的概念。在这项工作中，我们在更现实，动态的环境中研究了基于概念的模型，除了引入新概念本身外，新类还可能依赖旧概念。我们表明，概念和班级形成了一个复杂的关系网络，该网络容易降解，需要在经验中保留和增强。我们介绍了新的指标，以表明现有的基于概念的模型也无法保留这些关系，即使使用方法训练以防止灾难性遗忘，因为它们无法同时处理概念，班级和概念类关系水平。为了解决这些问题，我们提出了一种新颖的方法-Mucil-，该方法使用多模式概念来执行分类，而无需增加体验跨体验的可训练参数的数量。多模式概念与自然语言提供的概念保持一致，从而可以通过设计来解释。通过广泛的实验，我们表明我们的方法与其他基于概念的模型相比获得了最新的分类性能，在某些情况下，分类性能达到了2美元以上的分类性能。我们还研究了模型对概念进行干预措施的能力，并表明它可以在输入图像中定位视觉概念，从而提供事后解释。

### Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription 
[[arxiv](https://arxiv.org/abs/2502.20295)] [[cool](https://papers.cool/arxiv/2502.20295)] [[pdf](https://arxiv.org/pdf/2502.20295)]
> **Authors**: Benjamin Gutteridge,Matthew Thomas Jackson,Toni Kukurin,Xiaowen Dong
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: 11 pages (including references and appendix), 14 figures, accepted at AAAI-25 Workshop on Document Understanding and Intelligence, non-archival
- **标题**: 根据其封面来判断一本书：调查多页面手写文档转录的多模式LLM
- **领域**: 机器学习,人工智能,计算机视觉和模式识别
- **摘要**: 手写文本识别（HTR）仍然是一项具有挑战性的任务，特别是对于多页文档，页面具有共同的格式和上下文特征。虽然现代的光学特征识别（OCR）发动机精通印刷文本，但其手写的性能是有限的，通常需要昂贵的标签数据以进行微调。在本文中，我们探讨了多模式大型语言模型（MLLM）的使用，以零拍设置转录多页手写文档。我们研究了商用OCR发动机和MLLM的各种配置，将后者均用作端到端转录器和后处理器，并具有图像组件。我们提出了一种新颖的方法“+第一页”，该方法通过提供整个文档的OCR输出以及仅首页图像来增强MLLM转录。这种方法利用共享的文档功能，而不会产生处理所有图像的高成本。在IAM手写数据库的多页版本上进行的实验表明，“+第一页”提高了转录精度，平衡成本与性能，甚至通过外推格式格式和OCR误差模式从单个页面中提高了样本外文本的结果。

### Mixture of Experts for Recognizing Depression from Interview and Reading Tasks 
[[arxiv](https://arxiv.org/abs/2502.20213)] [[cool](https://papers.cool/arxiv/2502.20213)] [[pdf](https://arxiv.org/pdf/2502.20213)]
> **Authors**: Loukas Ilias,Dimitris Askounis
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: No comments
- **标题**: 专家识别面试和阅读任务的抑郁症的混合物
- **领域**: 机器学习,计算机与社会
- **摘要**: 抑郁症是一种精神障碍，可能导致多种症状，包括心理，身体和社交。言语已被证明是早期认识抑郁症的客观标记。因此，已经开发了许多研究，旨在通过语音识别抑郁症。但是，现有方法仅依赖于通过阅读语音获得的自发语音忽略信息的使用，使用经常难以获得（手动）或具有高单词率率（自动）的成绩单，并且不关注输入条件计算方法。为了解决这些局限性，这是在抑郁识别任务中的第一个研究，可以利用多模式融合方法获得自发和读取语音的表示，并在单个深层神经网络中使用专家（MOE）模型的混合物。具体来说，我们使用与采访和读取任务相对应的音频文件，然后将每个音频文件转换为log-mel频谱图，delta和delta-delta。接下来，这两个任务的图像表示通过共享的Alexnet模型。 Alexnet模型的输出作为多模式融合方法的输入。结果向量通过MOE模块。在这项研究中，我们采用了基于分解的三种MOE，即稀疏门控的MOE和多线性MOE。调查结果表明，我们提出的方法在Androids语料库上得出的准确性和F1评分分别为87.00％和86.66％。

### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion 
[[arxiv](https://arxiv.org/abs/2502.19834)] [[cool](https://papers.cool/arxiv/2502.19834)] [[pdf](https://arxiv.org/pdf/2502.19834)]
> **Authors**: Guanzhou Ke,Shengfeng He,Xiao Li Wang,Bo Wang,Guoqing Chao,Yuanyang Zhang,Yi Xie,HeXing Su
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Accepted to CVPR 2025
- **标题**: 知识布里奇：迈向无培训的丢失多模式完成
- **领域**: 机器学习,计算机视觉和模式识别,多媒体
- **摘要**: 以前的成功方法完成的方法取决于精心设计的融合技术和对完整数据的大量预培训，这可能会限制其在室外（OOD）方案中的普遍性。在这项研究中，我们提出了一个新的挑战：我们能否开发一个缺失的模式完成模型，既资源效率又强大？为了解决这个问题，我们提出了一个无训练的框架，用于缺少模式完成，该框架利用大型多模型（LMMS）。我们的方法称为“知识布里奇”，是模态敏捷的，并整合了缺失模态的产生和排名。通过定义特定领域的先验，我们的方法自动从可用模式中提取结构化信息以构建知识图。这些提取的图将缺失的模态生成和通过LMM进行排名，从而导致缺失模态的高质量归档。一般和医疗领域的实验结果表明，我们的方法始终优于竞争方法，包括在OOD概括中。此外，我们的知识驱动的生成和排名技术表明，与直接采用LMM进行生成和排名的变体相比，提供了对其他领域应用程序可能有价值的见解。

## 多代理系统(cs.MA:Multiagent Systems)

该领域共有 4 篇论文

### Group Trip Planning Query Problem with Multimodal Journey 
[[arxiv](https://arxiv.org/abs/2502.03144)] [[cool](https://papers.cool/arxiv/2502.03144)] [[pdf](https://arxiv.org/pdf/2502.03144)]
> **Authors**: Dildar Ali,Suman Banerjee,Yamuna Prasad
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: 11 Pages
- **标题**: 小组旅行计划查询问题与多模式旅程
- **领域**: 多代理系统,数据库,数据结构和算法
- **摘要**: 在小组旅行计划（GTP）查询问题中，为我们提供了一个城市路网络，其中有许多兴趣点（POI）被标记为各自类别（例如，自助餐厅，公园，电影院等）。一群代理商想从各自的起始地点访问每个类别的一个POI，并且一旦完成，他们希望到达各自的目的地。这个问题询问应选择每个类别的POI，以便将小组的汇总旅行成本最小化。在过去的十年中，已经对这个问题进行了广泛的研究，并提出了几种解决方案方法。但是，据我们所知，现有的研究都没有考虑过旅程的不同方式，这使问题更加实用。为了弥合这一差距，我们在本文中介绍并研究了多模式旅程的GTP查询问题。除了GTP查询问题的其他输入外，我们还获得了旅途的不同方式及其各自的成本。现在，问题不仅是从各个类别中选择毒药，而且是选择旅程的方式。对于这个问题，我们提出了一种有效的解决方案方法，该方法已进行了分析以了解其时间和空间要求。已经使用现实生活数据集进行了大量实验，并报告了结果。从结果来看，我们观察到，提议的解决方案方法推荐的POI和旅程方式导致的时间和成本要比基线方法少得多。

### A survey about perceptions of mobility to inform an agent-based simulator of subjective modal choice 
[[arxiv](https://arxiv.org/abs/2502.12058)] [[cool](https://papers.cool/arxiv/2502.12058)] [[pdf](https://arxiv.org/pdf/2502.12058)]
> **Authors**: Carole Adam,Benoit Gaudou
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: arXiv admin note: substantial text overlap with arXiv:2406.02063
- **标题**: 一项关于对流动性的看法的调查，以告知基于代理的模拟器主观模态选择
- **领域**: 多代理系统,计算机与社会
- **摘要**: 为了适应气候变化和公共卫生的问题，城市政策试图鼓励软化流动性，但汽车的份额仍然很大。除了已知的限制之外，我们在这里研究了感知偏见对单个选择的影响。我们设计了一个多标准决策模型，整合了习惯和偏见的影响。然后，我们进行了一项在线调查，该调查收到了650个回复。我们使用这些来计算现实的移动感知值，以便初始化Netlogo中实现的模态选择模拟器的环境和种群。这使我们能够可视化模态分布对城市规划发展的反应的适应，这取决于我们是否激活了个体推理中的偏见和习惯。这是在JFSMA-JFMS 2024上以法语发表的演示论文的扩展和翻译版本，“ UN SIMULATER MULTI-AGENT DE CHOIX MODAL MODAL MOCYIF”

### Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review 
[[arxiv](https://arxiv.org/abs/2502.11518)] [[cool](https://papers.cool/arxiv/2502.11518)] [[pdf](https://arxiv.org/pdf/2502.11518)]
> **Authors**: Di Wu,Xian Wei,Guang Chen,Hao Shen,Xiangfeng Wang,Wenhao Li,Bo Jin
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 18 pages
- **标题**: 体现AI的生成多代理协作：系统评价
- **领域**: 多代理系统,人工智能,机器学习
- **摘要**: 体现的多代理系统（EMAS）引起了人们越来越多的关注，因为它们在物流和机器人技术等领域中应对复杂的现实世界挑战的潜力。基础模型的最新进展为能够富裕的沟通和适应性问题解决的生成代理铺平了道路。这项调查提供了对EMA如何从这些生成能力中受益的系统检查。我们提出了一种分类学，将EMAS按系统架构和实施方式进行分类，并强调协作如何跨越物理和虚拟环境。然后分析中央构建块，感知，计划，沟通和反馈，以说明生成技术如何增强系统的鲁棒性和灵活性。通过具体示例，我们证明了将基础模型整合到体现的多代理框架中的变革效应。最后，我们讨论了挑战和未来的方向，强调了EMA重塑AI驱动协作的景观的重大希望。

### Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems 
[[arxiv](https://arxiv.org/abs/2502.14321)] [[cool](https://papers.cool/arxiv/2502.14321)] [[pdf](https://arxiv.org/pdf/2502.14321)]
> **Authors**: Bingyu Yan,Xiaoming Zhang,Litian Zhang,Lian Zhang,Ziyi Zhou,Dezhuang Miao,Chaozhuo Li
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 超越自我对话：以沟通为中心的基于LLM的多代理系统的调查
- **领域**: 多代理系统,计算语言学
- **摘要**: 大型语言模型（LLMS）最近在推理，计划和决策中表现出了显着的功能。在这些优势的基础上，研究人员已开始将LLMS纳入多代理系统（MAS），在该系统中，代理商通过自然语言互动进行合作或竞争，以解决超出单代理设置范围的任务。在这项调查中，我们介绍了基于LLM的多代理系统的以通信为中心的观点，研究了关键的系统级特征，例如体系结构设计和通信目标，以及内部机制，例如通信策略，范式，范式，对象和内容。我们说明了这些沟通元素如何相互作用以实现集体智慧和灵活的协作。此外，我们讨论了杰出的挑战，包括可伸缩性，安全性和多模式集成，并为将来的工作提出了方向，以推动该新兴领域的研究。最终，这项调查是进一步创新的催化剂，促进了各种应用领域的更强大，可扩展和智能的多机构系统。

## 多媒体(cs.MM:Multimedia)

该领域共有 2 篇论文

### UniForm: A Unified Diffusion Transformer for Audio-Video Generation 
[[arxiv](https://arxiv.org/abs/2502.03897)] [[cool](https://papers.cool/arxiv/2502.03897)] [[pdf](https://arxiv.org/pdf/2502.03897)]
> **Authors**: Lei Zhao,Linfeng Feng,Dongxu Ge,Fangqiu Yi,Chi Zhang,Xiao-Lei Zhang,Xuelong Li
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: Our demos are available at https://uniform-t2av.github.io/
- **标题**: 统一：统一的扩散变压器，用于音频发电
- **领域**: 多媒体,人工智能,计算机视觉和模式识别,声音,音频和语音处理
- **摘要**: 作为天然的多模式内容，Audible视频提供了沉浸式的感官体验。因此，音频视频生成系统具有巨大的潜力。但是，现有基于扩散的研究主要采用相对独立的模块来产生每种模态，而这种模式缺乏共享重量生成模块的探索。这种方法可能低估了音频和视觉方式之间的固有相关性，从而可能导致次优的生成质量。为了解决这个问题，我们提出了统一，这是一种统一扩散变压器，旨在增强跨模式的一致性。通过串联听觉和视觉信息，统一学会了在统一的潜在空间中同时生成音频和视频，从而促进了高质量且良好的视听对的创建。广泛的实验表明，我们方法在联合音频录像生成，音频引导的视频生成和视频引导的音频生成任务中的出色性能。我们的演示可在https://uniform-t2av.github.io/上找到。

### A Comprehensive Survey on Composed Image Retrieval 
[[arxiv](https://arxiv.org/abs/2502.18495)] [[cool](https://papers.cool/arxiv/2502.18495)] [[pdf](https://arxiv.org/pdf/2502.18495)]
> **Authors**: Xuemeng Song,Haoqiang Lin,Haokun Wen,Bohan Hou,Mingzhu Xu,Liqiang Nie
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-26
> **comment**: No comments
- **标题**: 一项关于构成图像检索的全面调查
- **领域**: 多媒体,人工智能,计算机视觉和模式识别,信息检索
- **摘要**: 组成的图像检索（CIR）是一项新兴但具有挑战性的任务，允许用户使用多模式查询搜索目标图像，包括参考图像和修改文本，指定用户对参考图像的所需更改。鉴于其具有巨大的学术和实践价值，CIR已成为计算机视觉和机器学习社区中迅速增长的兴趣领域，尤其是在深度学习方面的进步。据我们所知，目前尚无对CIR的全面审查，可以及时概述该领域。因此，我们综合了包括ACM TOIS，Sigir和CVPR在内的顶级会议和期刊中120多个出版物的见解，我们使用精细的分类法系统地将现有监督的CIR和零摄入CIR模型分类。为了进行全面综述，我们还简要讨论了与CIR密切相关的任务的方法，例如基于属性的CIR和基于对话的CIR。此外，我们通过比较多个数据集的实验结果来总结基准数据集，以评估和分析现有的监督和零射击CIR方法。此外，我们提出了这一领域有希望的未来方向，为对进一步探索感兴趣的研究人员提供了实践见解。在https://github.com/haokunwen/awesome-composed-image-retrieval中维护并不断更新相关工作的策划集合。

## 神经和进化计算(cs.NE:Neural and Evolutionary Computing)

该领域共有 2 篇论文

### Spiking Neural Network Feature Discrimination Boosts Modality Fusion 
[[arxiv](https://arxiv.org/abs/2502.10423)] [[cool](https://papers.cool/arxiv/2502.10423)] [[pdf](https://arxiv.org/pdf/2502.10423)]
> **Authors**: Katerina Maria Oikonomou,Ioannis Kansizoglou,Antonios Gasteratos
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 尖峰神经网络特征歧视增强了方式融合
- **领域**: 神经和进化计算,计算机视觉和模式识别,机器学习,图像和视频处理
- **摘要**: 特征歧视是神经网络设计的关键方面，因为它直接影响了网络区分类别和跨越不同数据集的类别的能力。实现高质量特征表示的实现可确保阶层内的可分离性高，并提出了最具挑战性的研究方向之一。传统的深度神经网络（DNNS）依靠复杂的转换和非常深的网络来提出有意义的特征表示，但它们通常需要数天的培训并消耗大量能量。为此，尖峰神经网络（SNN）提供了一种有希望的替代方案。 SNN捕获时间和空间依赖性的能力使它们特别适合复杂的任务，在需要多模式数据的情况下。在本文中，我们提出了一种使用SNN进行多模式学习的功能歧视方法，重点是视听数据。我们对视觉方式处理进行深度尖峰残差学习，并为听觉方式处理一个简单而有效的尖峰网络。最后，我们将尖峰的多层感知器部署以进行模态融合。我们介绍了我们的发现，并评估我们的方法与分类挑战领域的类似作品。据我们所知，这是调查SNN中特征歧视的第一项工作。

### A Hybrid Swarm Intelligence Approach for Optimizing Multimodal Large Language Models Deployment in Edge-Cloud-based Federated Learning Environments 
[[arxiv](https://arxiv.org/abs/2502.10419)] [[cool](https://papers.cool/arxiv/2502.10419)] [[pdf](https://arxiv.org/pdf/2502.10419)]
> **Authors**: Gaith Rjouba,Hanae Elmekki,Saidul Islam,Jamal Bentahar,Rachida Dssouli
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 一种用于优化基于边缘云的联合学习环境中多模式大语言模型部署的混合群智能方法
- **领域**: 神经和进化计算,人工智能,机器学习
- **摘要**: 联合学习（FL），多模式大语言模型（MLLM）和Edge-Cloud Computing的组合可以实现分布式和实时数据处理，同时在边缘设备和云基础架构上保留隐私。但是，在具有资源受限的边缘设备的FL环境中部署MLLM面临着重大挑战，包括资源管理，通信开销和非IID数据。为了应对这些挑战，我们提出了一个新型混合框架，其中将MLLM部署在配备有足够资源和电池寿命的边缘设备上，而大多数培训发生在云中。为了确定适合部署的边缘设备，我们采用了粒子群优化（PSO），并利用蚂蚁集菌的优化（ACO）来优化边缘和云节点之间模型更新的传输。该提出的基于群体智能的框架旨在通过在云中进行广泛的培训并在边缘进行微调，从而降低能耗和通信成本，从而提高MLLM培训的效率。我们的实验结果表明，所提出的方法可显着提高系统性能，达到92％的精度，将通信成本降低30％，并与传统的FL方法相比提高了客户的参与。这些结果使所提出的方法非常适合大规模的边缘云计算系统。

## 网络和互联网架构(cs.NI:Networking and Internet Architecture)

该领域共有 1 篇论文

### A Survey on Video Analytics in Cloud-Edge-Terminal Collaborative Systems 
[[arxiv](https://arxiv.org/abs/2502.06581)] [[cool](https://papers.cool/arxiv/2502.06581)] [[pdf](https://arxiv.org/pdf/2502.06581)]
> **Authors**: Linxiao Gong,Hao Yang,Gaoyun Fang,Bobo Ju,Juncen Guo,Xiaoguang Zhu,Xiping Hu,Yan Wang,Peng Sun,Azzedine Boukerche
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 一项关于云边缘末端协作系统中视频分析的调查
- **领域**: 网络和互联网架构,计算机视觉和模式识别,机器学习
- **摘要**: 视频数据的爆炸性增长推动了云边缘末端协作（CETC）系统中分布式视频分析的开发，从而实现了有效的视频处理，实时推理和隐私保护分析。在多个优点中，CETC系统可以分发视频处理任务，并在云，边缘和终端设备上启用自适应分析，从而在视频监视，自动驾驶和智能城市中取得突破。在这项调查中，我们首先分析了基本建筑组件，包括等级，分布式和混合框架，以及边缘计算平台和资源管理机制。在这些基础的基础上，以边缘为中心的方法强调设备处理，边缘辅助下载和边缘智能，而以云为中心的方法则利用强大的计算能力来用于复杂的视频理解和模型培训。我们的调查还涵盖了混合视频分析，其中包含自适应任务卸载和资源感知的调度技术，可优化整个系统的性能。除了传统的方法之外，大语言模型和多模式集成的最新进展既揭示了平台可伸缩性，数据保护和系统可靠性的机会和挑战。未来的方向还包括可解释的系统，有效的处理机制和高级视频分析，为这个动态领域的研究人员和从业人员提供了宝贵的见解。

## 机器人技术(cs.RO:Robotics)

该领域共有 26 篇论文

### Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors 
[[arxiv](https://arxiv.org/abs/2501.19042)] [[cool](https://papers.cool/arxiv/2501.19042)] [[pdf](https://arxiv.org/pdf/2501.19042)]
> **Authors**: Simon Idoko,B. Bhanu Teja,K. Madhava Krishna,Arun Kumar Singh
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: Submitted to RAL
- **标题**: 蜂群：快速生成各种可行的群体行为
- **领域**: 机器人技术,人工智能
- **摘要**: 机器人群中的协调行为本质上是本质上的多模式。也就是说，有多种方式可以通过多种方式避免跨性别的碰撞并实现各自的目标。但是，以可扩展方式产生多样化和可行的群体行为的问题在很大程度上仍未得到解决。在本文中，我们通过将生成模型与安全过滤器（SF）相结合来填补这一空白。具体而言，我们从学习的生成模型中采样了不同的轨迹，后来使用SF投射到可行的集合上。我们对生成模型进行了两种选择，即：条件变异自动编码器（CVAE）和矢量定量的变分自动编码器（VQ-VAE）。我们强调了这两个模型在计算时间和轨迹多样性方面提供的权衡。我们为我们的SF开发自定义求解器，并为其配备一个可预测特定于上下文初始化的神经网络。利用SF求解器的可不同性，以一种自制的方式对临界网络进行了训练。我们提供两组经验结果。首先，我们证明我们可以生成大量的多模式，可行的轨迹，在几十毫秒内模拟各种群体行为。其次，我们表明我们的初始化网络提供了我们的SF求解器相对于其他替代启发式方法的更快收敛。

### HeRCULES: Heterogeneous Radar Dataset in Complex Urban Environment for Multi-session Radar SLAM 
[[arxiv](https://arxiv.org/abs/2502.01946)] [[cool](https://papers.cool/arxiv/2502.01946)] [[pdf](https://arxiv.org/pdf/2502.01946)]
> **Authors**: Hanjun Kim,Minwoo Jung,Chiyun Noh,Sangwoo Jung,Hyunho Song,Wooseong Yang,Hyesu Jang,Ayoung Kim
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: 2025 IEEE International Conference on Robotics and Automation (ICRA 2025)
- **标题**: 大力神：复杂城市环境中的异质雷达数据集用于多课程雷达大满贯
- **领域**: 机器人技术,计算机视觉和模式识别
- **摘要**: 最近，雷达在机器人技术中广泛出现，因为它们在充满挑战的天气条件下的稳健性。两种常用的雷达类型是旋转雷达和梯级阵列雷达，每个雷达都提供不同的传感器特性。现有数据集通常仅具有单一类型的雷达，从而导致算法的开发仅限于该特定类型。在这项工作中，我们强调说，将不同的雷达类型组合提供了互补的优势，可以通过异质雷达数据集利用它们。此外，这个新的数据集促进了多项式和多机器人方案的研究，其中机器人配备了不同类型的雷达。在这种情况下，我们介绍了Hercules数据集，这是一个具有异质雷达，FMCW激光雷达，IMU，GPS和相机的全面的多模式数据集。这是第一个将4D雷达和旋转雷达与FMCW LIDAR一起旋转的数据集，提供无与伦比的本地化，映射和位置识别功能。数据集涵盖了各种天气和照明条件以及一系列城市交通情况，从而在各种环境中进行了全面的分析。每个传感器都有多个重新访问和地面真理的序列路径提高了其对位置识别研究的适用性。我们期望大力神数据集促进进程，映射，放置识别和传感器融合研究。数据集和开发工具可从https://sites.google.com/view/herculesdataset获得。

### Composite Gaussian Processes Flows for Learning Discontinuous Multimodal Policies 
[[arxiv](https://arxiv.org/abs/2502.01913)] [[cool](https://papers.cool/arxiv/2502.01913)] [[pdf](https://arxiv.org/pdf/2502.01913)]
> **Authors**: Shu-yuan Wang,Hikaru Sasaki,Takamitsu Matsubara
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 复合高斯流程流动以学习不连续的多模式政策
- **领域**: 机器人技术,机器学习
- **摘要**: 对现实世界机器人任务的学习控制政策通常涉及诸如多模态，本地不连续性以及对计算效率的需求等挑战。这些挑战来自机器人环境的复杂性，其中多种解决方案可能共存。为了解决这些问题，我们提出了复合高斯流程流（CGP-Flows），这是一种新型的机器人政策半参数模型。 CGP流与连续归一化流（CNF）集成了高斯过程（OMGP）的重叠混合物，从而使它们能够对解决多模式和局部不连续性的复杂策略进行建模。这种混合方法保留了OMGP的计算效率，同时结合了CNF的灵活性。在模拟和实际机器人任务中进行的实验表明，CGP-FLOW可显着提高建模控制策略的性能。在模拟任务中，我们确认与基线方法相比，CGP-Flows的成功率更高，并且GCP-Flow的成功率与卡方检验中其他基线的成功率显着差异。

### From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment 
[[arxiv](https://arxiv.org/abs/2502.01828)] [[cool](https://papers.cool/arxiv/2502.01828)] [[pdf](https://arxiv.org/pdf/2502.01828)]
> **Authors**: Yilin Wu,Ran Tian,Gokul Swamy,Andrea Bajcsy
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 从远见到前瞻性：通过潜在一致性的VLM-In-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-in-inop政策策略转向
- **领域**: 机器人技术,机器学习
- **摘要**: 尽管生成的机器人政策在学习复杂的学习中表现出了巨大的潜力，但示范中的多模式行为仍表现出多种模式的行为，但它们在部署时间时仍会表现出多种失败。政策转向提供了一种优雅的解决方案，可以通过使用外部验证器从不完美的生成策略提出的低级动作中进行选择来减少失败的机会。在这里，人们可能希望将视觉语言模型（VLM）用作验证者，以利用其开放的推理能力。但是，现成的VLM努力地了解低级机器人行动的后果，因为它们的代表与文本和图像进行了不同，并对VLM进行了培训。作为回应，我们提出了一种新颖的框架，这是一个新颖的框架，旨在释放VLM作为开放式唱片验证器的潜力，用于运行时策略转向。我们的关键思想是将VLM从评估（预见）中预测行动结果（预见）的负担。对于远见而言，我们利用潜在的世界模式来想象未来的潜在国家鉴于各种低级行动计划。对于有前途的，我们将VLM与这些预测的潜在国家保持一致，以推理其本地代表性（天然语言）中行动的后果，并有效地过滤了拟议的计划。我们在各种机器人操纵任务上验证了我们的框架，表明了其弥合代表性差距并提供可靠，可推广的政策转向的能力。可以在项目网站上找到视频：https：//yilin-wu98.github.io/forewarn/。

### VILP: Imitation Learning with Latent Video Planning 
[[arxiv](https://arxiv.org/abs/2502.01784)] [[cool](https://papers.cool/arxiv/2502.01784)] [[pdf](https://arxiv.org/pdf/2502.01784)]
> **Authors**: Zhengtong Xu,Qiang Qiu,Yu She
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: VILP：通过潜在视频计划模仿学习
- **领域**: 机器人技术,计算机视觉和模式识别
- **摘要**: 在生成AI的时代，将视频生成模型整合到机器人技术中为通用机器人代理打开了新的可能性。本文通过潜在的视频计划（VILP）介绍了模仿学习。我们提出了一个潜在的视频扩散模型，以生成在良好程度上遵守时间一致性的预测机器人视频。我们的方法能够从多个视图中生成高度时期的视频，这对于机器人政策学习至关重要。我们的视频生成模型高度时间效率。例如，它可以从两个不同的角度生成视频，每个角度由六个帧组成，分辨率为96x160像素，速率为5 Hz。在实验中，我们证明VILP的表现优于几个指标的现有视频生成机器人策略：培训成本，推理速度，生成视频的时间一致性以及策略的性能。我们还将我们的方法与其他模仿学习方法进行了比较。我们的发现表明，VILP可以较少依赖广泛的高质量特定任务机器人动作数据，同时仍保持稳健的性能。此外，VILP在表示多模式作用分布方面具有强大的功能。我们的论文提供了一个实用的例子，说明如何有效地将视频生成模型整合到机器人策略中，并可能为相关领域和方向提供见解。有关更多详细信息，请参阅我们的开源存储库https://github.com/zhengtongxu/vilp。

### VertiFormer: A Data-Efficient Multi-Task Transformer for Off-Road Robot Mobility 
[[arxiv](https://arxiv.org/abs/2502.00543)] [[cool](https://papers.cool/arxiv/2502.00543)] [[pdf](https://arxiv.org/pdf/2502.00543)]
> **Authors**: Mohammad Nazeri,Anuj Pokhrel,Alexandyr Card,Aniket Datar,Garrett Warnell,Xuesu Xiao
> **First submission**: 2025-02-01
> **First announcement**: 2025-02-04
> **comment**: 9 figures, url: https://github.com/mhnazeri/VertiFormer
- **标题**: Vertiformer：越野机器人移动性的数据效率多任务变压器
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 精致的学习体系结构，例如变形金刚为机器人提供了一个独特的机会，可以使自己了解复杂的车辆 - 远处动力学互动以实现越野移动性。尽管互联网规模的数据可用于自然语言处理（NLP）和计算机视觉（CV）任务以训练变形金刚，但使用物理机器人在越野地形中导航的实体机器人很难获得现实世界中的移动性数据。此外，专门设计用于处理NLP和CV中文本和图像数据的培训技术可能不适用于机器人移动性。在本文中，我们提出了Vertiformer，这是一种新型的数据有效的多任务变压器模型，该模型仅使用一个小时的数据训练，以解决将变压器体系结构应用于机器人移动性在极其坚固，垂直挑战，越野地形上的挑战。具体而言，Vertiformer采用新的可学习的蒙版建模和下一代币预测范式来预测下一个姿势，动作和地形贴片，以同时启用各种越野移动性任务，例如前进和逆动力学动力学建模。非自动进取的设计减轻了与自回归模型相关的计算瓶颈和错误传播。 Vertriformer的统一模态表示还增强了对各种时间映射和状态表示的学习，这些时间与多个目标函数相结合，进一步改善了模型的概括。我们的实验提供了有效使用有限数据的越野机器人移动性的洞察力，并证明了我们有效训练的变压器可以促进物理移动机器人的多个越野移动任务。

### Intelligent Sensing-to-Action for Robust Autonomy at the Edge: Opportunities and Challenges 
[[arxiv](https://arxiv.org/abs/2502.02692)] [[cool](https://papers.cool/arxiv/2502.02692)] [[pdf](https://arxiv.org/pdf/2502.02692)]
> **Authors**: Amit Ranjan Trivedi,Sina Tayebati,Hemant Kumawat,Nastaran Darabi,Divake Kumar,Adarsh Kumar Kosta,Yeshwanth Venkatesha,Dinithi Jayasuriya,Nethmi Jayasinghe,Priyadarshini Panda,Saibal Mukhopadhyay,Kaushik Roy
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 智能感知到边缘强大的自治：机遇和挑战
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 机器人技术，智能城市和自动驾驶汽车中的自主边缘计算依赖于在动态环境中实时决策的无缝集成。从本质上讲，感应到作用循环，它迭代地将传感器输入与计算模型相一致，以驱动自适应控制策略。这些循环可以适应超本地条件，提高资源效率和响应能力，但也面临挑战，例如资源限制，多模式数据融合中的同步延迟以及反馈循环中级联错误的风险。本文探讨了积极，上下文感知的感应到操作和对动作感应的适应能够通过基于任务需求动态调整感应和计算来提高效率，例如感测非常有限的环境部分并预测其余部分。通过通过控制操作引导感测，对动作感应途径可以改善任务相关性和资源的使用，但它们还需要强大的监视以防止级联错误并保持可靠性。多机构传感循环通过跨分布式代理的协调感应和动作进一步扩展了这些功能，从而通过协作优化资源使用。此外，受生物系统启发的神经形态计算为基于尖峰的，事件驱动的处理提供了一个有效的框架，该框架可以节省能量，减少潜伏期并支持层次结构控制 - 将其设置为多代理优化的理想选择。本文强调了端到端共同设计策略的重要性，该策略将算法模型与硬件和环境动态保持一致，并改善跨层相互依存，以提高吞吐量，精度和适应性的复杂环境中能节能边缘自主权。

### SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification 
[[arxiv](https://arxiv.org/abs/2502.02657)] [[cool](https://papers.cool/arxiv/2502.02657)] [[pdf](https://arxiv.org/pdf/2502.02657)]
> **Authors**: Yifu Tao,Maurice Fallon
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: webpage: https://dynamic.robots.ox.ac.uk/projects/silvr/
- **标题**: SILVR：可扩展的LIDAR-VISUAL-VISUAL-VISUAL RADIANCE FIELD重建具有不确定性定量
- **领域**: 机器人技术,计算机视觉和模式识别
- **摘要**: 我们提出了一个基于神经辐射场（NERF）的大规模重建系统，该系统融合了激光雷达和视力数据，以生成几何准确的高质量重建，并捕获光真逼真的纹理。我们的系统采用最先进的NERF代表性来纳入LIDAR。添加LiDAR数据在深度和表面正态上增加了强大的几何约束，这在建模包含模棱两可的视觉重建提示的均匀纹理表面时特别有用。此外，我们将重建的认知不确定性视为辐射场中每个点位置的空间差异，鉴于相机和激光镜头的传感器观察结果。这样可以识别每个传感器模式可靠重建的区域，从而可以根据估计的不确定性过滤地图。我们的系统还可以利用在线映射期间实时姿势盖雷达激光雷达系统产生的轨迹，以进行自举 - （后处理）结构（SFM）重建程序，将SFM训练时间缩短了高达70％。它还有助于正确限制整体度量标准，这对于激光雷达深度损失至关重要。然后，可以将光谱聚类一起使用与共同可见图像的组集一起将全球一致的轨迹分为子膜。与基于距离的分区相比，这种沉积方法更适合视觉重建。根据点的不确定性估计值对每个子扣进行过滤，并合并以获得最终的大规模3D重建。我们在涉及机器人安装和手持式扫描的实验中使用多摄像机的LIDAR传感器套件来演示重建系统。我们的测试数据集涵盖了超过20,000平方米的总面积，包括多个大学建筑物和对多层楼的航空调查。

### VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation 
[[arxiv](https://arxiv.org/abs/2502.02175)] [[cool](https://papers.cool/arxiv/2502.02175)] [[pdf](https://arxiv.org/pdf/2502.02175)]
> **Authors**: Siyu Xu,Yunke Wang,Chenghao Xia,Dihao Zhu,Tao Huang,Chang Xu
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: VLA-CACHE：通过机器人操纵中的自适应令牌缓存迈向有效的视觉语言行动模型
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 视觉语言动作（VLA）模型可以处理指令和视觉感知，从而由于其强大的多模式推理能力而直接以端到端方式生成动作作为输出。尽管VLA模型的性能是有希望的，但它们的计算成本可能很大。这引发了将它们应用于机器人技术任务的挑战，这需要实时决策才能快速响应环境变化。由于机器人控制涉及顺序决策，因此视觉输入通常在连续步骤之间表现出最小的变化。一个自然的想法是从最后一步重复使用不变的视觉令牌的计算结果。在这个想法的激励下，我们提出了VLA-CACHE，这是一个有效的视觉语言操作模型。 VLA-CACHE结合了一个令牌选择机制，将每个步骤的视觉输入与上一步的输入进行比较，从而自适应地识别具有最小变化的视觉令牌。然后，通过KV-CACHE在后续步骤中重复使用这些不变令牌的计算结果，从而显着提高了VLA-CACHE模型的效率。对模拟（例如Libero基准和更简单）和现实世界机器人有效VLA-CACHE的实验结果都可以在成功率上以最小的牺牲来实现实用的加速度。

### PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map 
[[arxiv](https://arxiv.org/abs/2502.05752)] [[cool](https://papers.cool/arxiv/2502.05752)] [[pdf](https://arxiv.org/pdf/2502.05752)]
> **Authors**: Yue Pan,Xingguang Zhong,Liren Jin,Louis Wiesmann,Marija Popović,Jens Behley,Cyrill Stachniss
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: 14 pages, 8 figures
- **标题**: ping：高斯裂缝符合基于点的隐式神经图内的距离场
- **领域**: 机器人技术,计算机视觉和模式识别,图形
- **摘要**: 机器人需要对其环境的高保真重建才能有效操作。这样的场景表示应该是几何准确和逼真的，以支持下游任务。虽然可以通过从摄像机的范围传感器和辐射场构建距离字段来实现这一目标，但两个字段的可扩展增量映射始终且同时具有高质量的范围仍然具有挑战性。在本文中，我们提出了一个新的图表表示，该表示在基于弹性且紧凑的基于点的隐式神经图内统一了连续的签名距离场和高斯弹性辐射场。通过在这些领域之间执行几何一致性，我们通过利用这两种方式来实现相互改进。我们使用建议的地图表示，设计了一个称为ping的LiDAR-Visual SLAM系统，并在几个具有挑战性的大规模数据集上对其进行评估。实验结果表明，PINGS可以逐步建立用紧凑的神经点编码的全球一致距离和辐射场。与最先进的方法相比，PINGS通过利用距离场的约束来实现新型视图的优质光度和几何渲染。此外，通过利用辐射场的密集光度线索和多视图一致性，PINGS会产生更准确的距离场，从而改善了探光估计和网状重建。

### Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs 
[[arxiv](https://arxiv.org/abs/2502.05641)] [[cool](https://papers.cool/arxiv/2502.05641)] [[pdf](https://arxiv.org/pdf/2502.05641)]
> **Authors**: Aayam Shrestha,Pan Liu,German Ros,Kai Yuan,Alan Fern
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: ef:The European Conference on Computer Vision (ECCV), 2024
- **标题**: 从多模式输入中产生物理逼真的和可直接的人类动作
- **领域**: 机器人技术,人工智能
- **摘要**: 这项工作着重于从多模式输入中生成现实的，基于物理的人类行为，这可能只能部分指定所需的运动。例如，输入可能来自提供手臂运动和身体速度的VR控制器，部分关键点动画，应用于视频的计算机视觉，甚至更高级别的运动目标。这需要一个多功能的低级类人动物控制器，该控制器可以处理如此稀疏，指定不足的指导，在技能之间无缝切换并从失败中恢复。当前从演示数据中学习类人生物控制器的方法捕获了其中一些特征，但没有一个全部实现。为此，我们介绍了遮罩的人形控制器（MHC），这是一种新的方法，该方法将多目标模仿学习应用于增强和选择性地掩盖的运动演示。训练方法产生了MHC，该MHC表现出与异步输入命令相关的关键功能，结合了多个运动序列的元素，并从稀疏的多模式输入中完成了不指定的运动部分。我们证明了MHC在87种不同技能的数据集中学习的这些关键功能，并展示了不同的多模式用例，包括与计划框架集成，以突出MHC解决无需任何填充而解决新的用户定义任务的能力。

### Vision-Ultrasound Robotic System based on Deep Learning for Gas and Arc Hazard Detection in Manufacturing 
[[arxiv](https://arxiv.org/abs/2502.05500)] [[cool](https://papers.cool/arxiv/2502.05500)] [[pdf](https://arxiv.org/pdf/2502.05500)]
> **Authors**: Jin-Hee Lee,Dahyun Nam,Robin Inho Kee,YoungKey Kim,Seok-Jun Buu
> **First submission**: 2025-02-08
> **First announcement**: 2025-02-10
> **comment**: Submitted to Engineering Applications of Artificial Intelligence
- **标题**: 视觉启动机器人系统基于对制造中气体和弧危险检测的深度学习
- **领域**: 机器人技术,人工智能
- **摘要**: 气体泄漏和电弧排放在工业环境中带来了重大风险，需要强大的检测系统以确保安全和操作效率。这项研究受到将视觉识别与声学验证相结合的人类协议的启发，提出了一种基于学习的机器人系统，用于自主检测和分类制造环境中的气体泄漏和电弧排放。该系统旨在完全在机器人上执行所有实验任务。该系统利用以96 kHz的采样率操作的112通道原声摄像机来捕获超声波频率，该系统处理在各种工业场景中记录的现实世界数据集。这些数据集包括在不同的环境噪声条件下的多种气体泄漏配置（例如，针孔，开放端）和部分放电类型（电晕，表面，浮动）。提出的系统集成了视觉检测和波束形成增强的声学分析管道。使用STFT对信号进行转换，并通过伽马校正进行完善，从而实现鲁棒的特征提取。 Inception启发的CNN进一步对危害进行了分类，达到99％的气体泄漏检测准确性。该系统不仅检测到单个危害来源，而且还通过融合视觉传感器和声传感器的多模式数据来增强分类的可靠性。当在混响和噪音提示的环境中进行测试时，该系统的表现优于常规模型，最多44％P，并精心设计实验任务，以确保公平性和可重复性。此外，该系统将用于实时部署，并在移动机器人平台上保持2.1秒的推理时间。通过模拟类似人类的检查协议并将视觉与声学方式整合在一起，本研究为工业自动化提供了有效的解决方案，可显着提高安全性和操作可靠性。

### End-to-End Predictive Planner for Autonomous Driving with Consistency Models 
[[arxiv](https://arxiv.org/abs/2502.08033)] [[cool](https://papers.cool/arxiv/2502.08033)] [[pdf](https://arxiv.org/pdf/2502.08033)]
> **Authors**: Anjian Li,Sangjae Bae,David Isele,Ryne Beeson,Faizan M. Tariq
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: 使用一致性模型的自动驾驶的端到端预测计划者
- **领域**: 机器人技术,机器学习
- **摘要**: 轨迹预测和计划是自动驾驶汽车在动态环境中安全有效导航的基本组件。传统上，这些组件通常被视为单独的模块，从而限制了执行互动计划的能力并导致多代理场景中的计算效率低下。在本文中，我们提出了一个新颖的统一和数据驱动的框架，该框架将预测和计划与单个一致性模型集成在一起。我们的一致性模型受到现实世界中的人类驾驶数据集的培训，从自我和多个周围代理的高维，多模式的联合轨迹分布中生成样本，从而实现了端到端的预测计划。它有效地产生了互动行为，例如主动促进和屈服，以确保与其他道路使用者的安全有效互动。为了在自我车辆上纳入其他计划限制，我们提出了一种交替的方向方法，用于在线指导采样中多目标指导。与扩散模型相比，我们的一致性模型通过较少的采样步骤实现更好的性能，使其更适合实时部署。与各种现有方法相比，Waymo开放运动数据集（WOMD）的实验结果证明了我们方法在轨迹质量，约束满意度和互动行为方面的优越性。

### RoboBERT: An End-to-end Multimodal Robotic Manipulation Model 
[[arxiv](https://arxiv.org/abs/2502.07837)] [[cool](https://papers.cool/arxiv/2502.07837)] [[pdf](https://arxiv.org/pdf/2502.07837)]
> **Authors**: Sicheng Wang,Jianhua Shan,Jianwei Zhang,Haozhang Gao,Hailiang Han,Yipeng Chen,Kang Wei,Chengkun Zhang,Kairos Wong,Jie Zhao,Lei Zhao,Bin Fang
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: No comments
- **标题**: Robobert：端到端的多模式机器人操纵模型
- **领域**: 机器人技术,机器学习
- **摘要**: 体现的智能会整合多种方式，使代理能够同时了解图像，语言和动作。但是，现有模型始终取决于其他数据集或大量预培训，以最大程度地提高性能，消耗丰富的培训时间和昂贵的硬件成本。为了解决这个问题，我们介绍了Robobert，这是一种新颖的端到端机器人操纵模型，该模型与独特的培训策略集成在一起。该模型利用基于CNN的扩散策略，通过将训练过程分开不同方式来增强和稳定该模型的有效性。它还强调了数据增强的重要性，验证了各种技术以显着提高性能。与依赖额外数据或大型基础模型的模型不同，Robobert仅使用语言标记的专家演示并保持相对较小的模型大小，取得了高度竞争的成功率。具体而言，Robobert的平均长度为\（ABCD \ rightarrow d \）任务的Calvin基准测试，设置了一个新的最先进（SOTA）记录。此外，在对真实机器人进行测试时，该模型表现出卓越的性能，比其他使用相同数据训练的方法获得了更高的成功率。我们建议，这些Robobert的这些概念和方法表现出广泛的多功能性和兼容性，这显着有助于轻巧的多模式机器人模型的发展。可以在https://github.com/peterwangsicheng/robobert上访问该代码

### 3D-Grounded Vision-Language Framework for Robotic Task Planning: Automated Prompt Synthesis and Supervised Reasoning 
[[arxiv](https://arxiv.org/abs/2502.08903)] [[cool](https://papers.cool/arxiv/2502.08903)] [[pdf](https://arxiv.org/pdf/2502.08903)]
> **Authors**: Guoqin Tang,Qingxuan Jia,Zeyuan Huang,Gang Chen,Ning Ji,Zhipeng Yao
> **First submission**: 2025-02-12
> **First announcement**: 2025-02-13
> **comment**: No comments
- **标题**: 机器人任务计划的3D接地视觉框架：自动及时合成和监督推理
- **领域**: 机器人技术,人工智能
- **摘要**: 视觉语言模型（VLM）在场景理解和感知任务中取得了巨大的成功，使机器人能够在动态环境中自适应地计划和执行操作。但是，大多数多模式的大型语言模型都缺乏强大的3D场景本地化功能，从而限制了它们在细粒度的机器人操作中的有效性。此外，诸如低识别精度，低效率，可传递性差和可靠性之类的挑战阻碍了他们在精确任务中的使用。为了解决这些限制，我们提出了一个新颖的框架，该框架通过将2D图像映射到点云，集成了2D提示合成模块，并结合了一个小语言模型（SLM），以监督VLM输出。 2D提示合成模块可以使VLM在2D图像和文本上进行训练，可以自主提取精确的3D空间信息，而无需手动干预，从而显着增强了3D场景的理解。同时，SLM监督VLM输出，减轻幻觉并确保可靠的可执行机器人控制代码生成。我们的框架消除了在新环境中进行重新培训的需求，从而提高了成本效率和运营稳健性。提议的框架获得了96.0 \％的任务成功率（TSR）的实验结果，表现优于其他方法。消融研究证明了2D提示合成模块和输出监督模块的关键作用（当删除时，引起了67 \％TSR下降）。这些发现证明了该框架在改善3D识别，任务计划和机器人任务执行方面的有效性。

### TRIFFID: Autonomous Robotic Aid For Increasing First Responders Efficiency 
[[arxiv](https://arxiv.org/abs/2502.09379)] [[cool](https://papers.cool/arxiv/2502.09379)] [[pdf](https://arxiv.org/pdf/2502.09379)]
> **Authors**: Jorgen Cani,Panagiotis Koletsis,Konstantinos Foteinos,Ioannis Kefaloukos,Lampros Argyriou,Manolis Falelakis,Iván Del Pino,Angel Santamaria-Navarro,Martin Čech,Ondřej Severa,Alessandro Umbrico,Francesca Fracasso,AndreA Orlandini,Dimitrios Drakoulis,Evangelos Markakis,Iraklis Varlamis,Georgios Th. Papadopoulos
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: TRIFFID：提高第一响应者效率的自动机器人辅助
- **领域**: 机器人技术,人工智能
- **摘要**: 自然灾害事件的复杂性日益复杂，需要创新的技术解决方案，以支持急救人员的努力。本文介绍了Triffid System，这是一个全面的技术框架，将无人机和航空车与先进的人工智能功能相结合，以增强野火，城市洪水以及地球后搜索和救援任务的灾难反应能力。通过利用最新的自动导航，语义感知和人类机器人交互技术，Triffid提供了一个由以下关键组件组成的复杂系统：混合机器人平台，集中式地面站，定制通信基础设施和智能手机应用程序。定义的研发活动表明，深度神经网络，知识图和多模式信息融合可以使机器人能够自主浏览和分析灾难环境，降低人员风险并加速响应时间。拟议的系统通过提供高级任务计划，安全监控和自适应任务执行功能来增强应急小组。此外，它确保在复杂和风险的情况下实时情境意识和运营支持，从而促进快速，精确的信息收集和协调的行动。

### IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation 
[[arxiv](https://arxiv.org/abs/2502.12371)] [[cool](https://papers.cool/arxiv/2502.12371)] [[pdf](https://arxiv.org/pdf/2502.12371)]
> **Authors**: Krishan Rana,Robert Lee,David Pershouse,Niko Suenderhauf
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Videos and code are available at https://imle-policy.github.io/
- **标题**: IMLE政策：通过隐式最大似然估计快速和样本有效的视觉运动策略学习
- **领域**: 机器人技术,人工智能,机器学习
- **摘要**: 模仿学习的最新进展，尤其是使用生成建模技术，例如扩散，使策略能够捕获复杂的多模式作用分布。但是，这些方法通常需要大型数据集和动作生成的多个推理步骤，在机器人技术中提出挑战，在这些机器人技术中，数据收集成本很高，计算资源受到限制。为了解决这个问题，我们介绍了Imle Policy，这是一种基于隐式最大似然估计（IMLE）的新型行为克隆方法。 Imle政策在低数据制度中擅长，从最小的演示中有效地学习，并需要38％的数据降低数据，以匹配学习复杂的多模式行为的基线方法的性能。与扩散策略相比，其简单的基于生成器的体系结构可以生成单步操作，将推理速度提高了97.3 \％，同时表现优于单步流匹配。我们在模拟和现实世界环境中跨越各种操纵任务的方法验证了我们的方法，从而展示了其在数据约束下捕获复杂行为的能力。视频和代码在我们的项目页面上提供：https：//imle-policy.github.io/。

### RHINO: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations 
[[arxiv](https://arxiv.org/abs/2502.13134)] [[cool](https://papers.cool/arxiv/2502.13134)] [[pdf](https://arxiv.org/pdf/2502.13134)]
> **Authors**: Jingxiao Chen,Xinyao Li,Jiahang Cao,Zhengbang Zhu,Wentao Dong,Minghuan Liu,Ying Wen,Yong Yu,Liqing Zhang,Weinan Zhang
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: Project website: https://humanoid-interaction.github.io/
- **标题**: 犀牛：从人类示范中学习实时的人形 - 对象相互作用
- **领域**: 机器人技术,人机交互,机器学习
- **摘要**: 人形机器人在运动和操纵方面表现出成功。尽管有这些基本能力，仍需要类人动物快速理解人类的指示并根据人类互动信号做出反应，以成为人类日常生活中有价值的助手。不幸的是，大多数现有作品仅专注于多阶段的交互，分别处理每个任务并忽略实时反馈。在这项工作中，我们旨在使人形机器人具有实时反应能力，以实现各种任务，允许人类随时打断机器人，并使机器人立即对人类做出反应。为了支持这种能力，我们提出了一个普通的人形 - 人类对象相互作用框架，称为犀牛，即实时人形类人 - 人类相互作用和对象操纵。 Rhino在多种人类信号方式（例如语言，图像和动作）上提供了反应性运动，基于指导的操作和安全问题的统一观点。犀牛是一个分层学习框架，使类人动物能够从人类对象的演示和远程操作数据中学习反应技能。特别是，它将相互作用过程分为两个级别：1）一名高级计划者从实时人类行为中推断人类意图； 2）基于预测的意图，低级控制器实现了反应性运动行为和对象操纵技能。我们在真实的类人机器人上评估了提议的框架，并在各种情况下证明了其有效性，灵活性和安全性。

### CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.15119)] [[cool](https://papers.cool/arxiv/2502.15119)] [[pdf](https://arxiv.org/pdf/2502.15119)]
> **Authors**: Zihao Sheng,Zilin Huang,Yansong Qu,Yue Leng,Sruthi Bhavanam,Sikai Chen
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: curricuvlm：通过视觉模型通过个性化安全 - 关键课程学习进行安全自动驾驶
- **领域**: 机器人技术,人工智能,计算机视觉和模式识别
- **摘要**: 确保自主驾驶系统的安全仍然是一个至关重要的挑战，尤其是在处理罕见但潜在的灾难性安全至关重要的情况下。尽管现有研究探讨了为自动驾驶汽车（AV）测试生成安全关键方案，但有效地将这些情景纳入政策学习以增强安全性方面的工作有限。此外，开发适应AV不断发展的行为模式和性能瓶颈的培训课程在很大程度上没有探索。为了应对这些挑战，我们提出了Curricuvlm，这是一个新颖的框架，利用视觉模型（VLM）为自主驾驶剂提供个性化的课程学习。我们的方法独特利用了VLMS的多模式理解能力，以分析代理行为，识别性能弱点并动态生成量身定制的培训方案，以适应课程。通过对具有叙事描述不安全的驾驶情况的全面分析，Curricuvlm执行了深入的推理，以评估AV的能力并确定关键的行为模式。然后，该框架综合了针对这些确定的限制的定制培训方案，从而实现了有效和个性化的课程学习。 Waymo Open Motion数据集的广泛实验表明，在常规和安全至关重要的情况下，Curricuvlm的表现都超过了最先进的基线，在导航成功，推动效率和安全指标方面取得了卓越的性能。进一步的分析表明，Curricuvlm是一种通用方法，可以与各种RL算法集成以增强自主驾驶系统。代码和演示视频可在以下网址获得：https：//zihaosheng.github.io/curricuvlm/。

### ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model 
[[arxiv](https://arxiv.org/abs/2502.14420)] [[cool](https://papers.cool/arxiv/2502.14420)] [[pdf](https://arxiv.org/pdf/2502.14420)]
> **Authors**: Zhongyi Zhou,Yichen Zhu,Minjie Zhu,Junjie Wen,Ning Liu,Zhiyuan Xu,Weibin Meng,Ran Cheng,Yaxin Peng,Chaomin Shen,Feifei Feng
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: CHATVLA：统一的多模式理解和具有视觉语言模型的机器人控制
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 人类具有统一的认知能力，可以感知，理解和与物理世界互动。为什么大型语言模型不能复制这种整体理解？通过对视觉语言行动模型（VLA）中现有的训练范式的系统分析，我们确定了两个关键挑战：虚假遗忘，机器人培训覆盖关键的视觉识别和任务干扰，在这种情况下，竞争控制和理解任务在训练时竞争性的任务降低了绩效。为了克服这些局限性，我们提出了Chatvla，这是一个具有分阶段对齐训练的新型框架，该框架在初始控制掌握后会逐步整合多模式数据，并且一种混合架构可以将架构结构结构最小化。 Chatvla在视觉提问数据集上展示了竞争性能，并且在多模式理解基准方面显着超过了最先进的视觉语言动作（VLA）方法。值得注意的是，它在MMMU上的性能高出六倍，并且在MMSTAR上得分47.2％，其参数效率高于ECOT。此外，与现有的VLA方法（如OpenVLA）相比，Chatvla在25个现实世界机器人操纵任务上表现出了卓越的性能。我们的发现突出了我们统一框架获得稳健多模式理解和有效机器人控制的潜力。

### NatSGLD: A Dataset with Speech, Gesture, Logic, and Demonstration for Robot Learning in Natural Human-Robot Interaction 
[[arxiv](https://arxiv.org/abs/2502.16718)] [[cool](https://papers.cool/arxiv/2502.16718)] [[pdf](https://arxiv.org/pdf/2502.16718)]
> **Authors**: Snehesh Shrestha,Yantian Zha,Saketh Banagiri,Ge Gao,Yiannis Aloimonos,Cornelia Fermüller
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: arXiv admin note: substantial text overlap with arXiv:2403.02274
- **标题**: Natsgld：具有语音，手势，逻辑和演示的数据集，用于自然人类机器人互动中的机器人学习
- **领域**: 机器人技术,人工智能
- **摘要**: 多模式人类机器人相互作用（HRI）数据集的最新进展强调了语音和手势的整合，从而使机器人能够吸收明确的知识和默认的理解。但是，现有数据集主要关注对象指向和推动等基本任务，从而将其适用性限制在复杂域。他们优先考虑更简单的人类命令数据，但较少强调培训机器人以正确解释任务并做出适当响应。为了解决这些差距，我们介绍了使用绿野仙踪（WOZ）方法收集的NatsGLD数据集，其中参与者与他们认为是自治的机器人进行了互动。 Natsgld记录了人类的多模式命令（语音和手势），每个命令与演示轨迹和线性时间逻辑（LTL）公式配对，该公式提供了对命令任务的基本真实解释。该数据集是HRI和机器学习交集的研究基础。通过提供多模式输入和详细注释，Natsgld可以从示范中的诸如多模式指导，计划识别和可鉴定的强化学习等领域进行探索。我们在https://www.snehesn.com/natsgld/上发布MIT许可证下的数据集和代码，以支持未来的HRI研究。

### Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions 
[[arxiv](https://arxiv.org/abs/2502.15336)] [[cool](https://papers.cool/arxiv/2502.15336)] [[pdf](https://arxiv.org/pdf/2502.15336)]
> **Authors**: Shoubin Chen,Zehao Wu,Kai Zhang,Chunyu Li,Baiyang Zhang,Fei Ma,Fei Richard Yu,Qingquan Li
> **First submission**: 2025-02-21
> **First announcement**: 2025-02-24
> **comment**: 81 pages, submitted to a journal for review
- **标题**: 探索具体的多模式大型模型：开发，数据集和未来方向
- **领域**: 机器人技术,人工智能
- **摘要**: 体现的多模式大型模型（EMLM）近年来因其在复杂的现实世界环境中弥合感知，认知和作用之间的差距而引起了极大的关注。这项全面的评论探讨了此类模型的发展，包括大语言模型（LLM），大型视觉模型（LVM）和其他模型，同时还研究了其他新兴体系结构。我们讨论EMLM的演变，重点是体现的感知，导航，相互作用和仿真。此外，该评论对用于培训和评估这些模型的数据集提供了详细的分析，强调了多种高质量数据的重要性，以进行有效学习。本文还确定了EMLM面临的关键挑战，包括可扩展性，概括和实时决策问题。最后，我们概述了未来的方向，强调了多模式感应，推理和行动的整合，以推动日益自主系统的发展。通过对最新方法进行深入分析并确定关键差距，本文旨在激发EMLMS及其在不同领域的应用中的未来进步。

### CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems 
[[arxiv](https://arxiv.org/abs/2502.17821)] [[cool](https://papers.cool/arxiv/2502.17821)] [[pdf](https://arxiv.org/pdf/2502.17821)]
> **Authors**: Rui Liu,Yu Shen,Peng Gao,Pratap Tokekar,Ming Lin
> **First submission**: 2025-02-24
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: CAML：多代理系统的协作辅助模态学习
- **领域**: 机器人技术,人工智能,机器学习
- **摘要**: 多模式学习已成为改善跨自主驾驶，机器人技术和感知系统等域的机器学习应用程序的至关重要技术。尽管辅助模态学习（AML）等现有框架在训练过程中有效地利用了多个数据源，并可以通过减少的模态来推断，但它们主要在单一代理上下文中运行。在动态环境（例如连接的自动驾驶汽车（CAV））中，这种限制尤其重要，在该环境中，数据覆盖不完整会导致决策盲点。为了应对这些挑战，我们提出了协作辅助模态学习（$ \ textbf {CAML} $），这是一个新型的多机构多模式框架，使代理商在培训过程中可以在训练过程中协作和共享多模式数据，同时允许在测试过程中降低每个代理的态度减少。我们从不确定性降低和数据覆盖范围的角度系统地分析了$ \ textbf {caml} $的有效性，从而为其优势提供了比AML的优势。在容易发事故的情况下为CAV进行协作决策的实验结果表明，\我们的〜实现了$ {\ bf 58.13} \％$ $ $ $ $ $改善事故检测。此外，我们在现实世界中的空中机器人数据上验证$ \ textbf {caml} $用于协作语义细分，达到了MIOU的$ {\ bf 10.61} \％$的改进。

### A Real-time Spatio-Temporal Trajectory Planner for Autonomous Vehicles with Semantic Graph Optimization 
[[arxiv](https://arxiv.org/abs/2502.18151)] [[cool](https://papers.cool/arxiv/2502.18151)] [[pdf](https://arxiv.org/pdf/2502.18151)]
> **Authors**: Shan He,Yalong Ma,Tao Song,Yongzhi Jiang,Xinkai Wu
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: This work has been accepted for publication in IEEE Robotics and Automation Letters (RA-L). The final published version is available in IEEE Xplore (DOI: 10.1109/LRA.2024.3504239)
- **标题**: 具有语义图优化的自动驾驶汽车的实时时空轨迹计划器
- **领域**: 机器人技术,人工智能
- **摘要**: 通过在复杂的城市环境中充分利用感知信息，为自动驾驶汽车计划安全且可行的轨迹具有挑战性。在本文中，我们提出了一种基于图形优化的时空轨迹计划方法。它通过通过静态和动态障碍物的分离处理来构建语义时空 - 周期性图来有效提取感知模块的多模式信息，然后通过基于语义时空时空超时性超弹力来快速通过稀疏的图形优化生成可行的轨迹。广泛的实验证明，所提出的方法可以有效地处理复杂的城市公共道路情景并实时执行。我们还将发布我们的代码，以适应研究社区的基准测试

### Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation 
[[arxiv](https://arxiv.org/abs/2502.18842)] [[cool](https://papers.cool/arxiv/2502.18842)] [[pdf](https://arxiv.org/pdf/2502.18842)]
> **Authors**: Muhammad A. Muttaqien,Tomohiro Motoda,Ryo Hanai,Domae Yukiyasu
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 夹子和SAM的注意力指导集成在机器人操作中精确对象掩盖
- **领域**: 机器人技术,人工智能,计算机视觉和模式识别
- **摘要**: 本文介绍了一条新型管道，以增强便利店中掩盖产品特定域内机器人操作的对象掩盖的精度。该方法集成了两个先进的AI模型，即剪辑和SAM，重点是它们的协同组合以及有效使用多模式数据（图像和文本）。重点是利用基于梯度的注意机制和定制数据集进行微调性能。在建立夹子，SAM和Grad-cam的组件时，它们在该结构化管道中的集成代表了对该领域的重要贡献。通过这种组合方法生成的产生的分段面具可以有效地用作机器人系统的输入，从而在便利店产品的背景下实现更精确和自适应的对象操作。

### CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving 
[[arxiv](https://arxiv.org/abs/2502.19908)] [[cool](https://papers.cool/arxiv/2502.19908)] [[pdf](https://arxiv.org/pdf/2502.19908)]
> **Authors**: Dongkun Zhang,Jiaming Liang,Ke Guo,Sha Lu,Qi Wang,Rong Xiong,Zhenwei Miao,Yue Wang
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: CVPR 2025
- **标题**: Carplanner：自动驾驶中大规模增强学习的一致自动回归轨迹计划
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 轨迹计划对于自动驾驶至关重要，可确保在复杂环境中安全有效的航行。尽管最近基于学习的方法，尤其是强化学习（RL），在特定方案中表现出了希望，但RL计划者在培训效率低下和管理大型现实世界驾驶场景方面挣扎。在本文中，我们介绍了\ textbf {carplanner}，a \ textbf {c} onsistent \ textbf {a} uto- \ textbf {r} egrescement \ textbf {planner}使用rl来生成多摩座轨迹。自动回归结构可实现有效的大规模RL培训，而一致性的结合可以通过在时间步骤中保持连贯的时间一致性来确保稳定的政策学习。此外，Carplanner采用了具有专家指导的奖励功能和不变视图模块的生成选择框架，从而简化了RL培训并增强了政策性能。广泛的分析表明，我们提出的RL框架有效地解决了训练效率和绩效提高的挑战，将Carplanner定位为自主驾驶中轨迹计划的有前途的解决方案。据我们所知，我们是第一个证明基于RL的计划者可以超过IL和规则的最先进（SOTA）（SOTA）的人。我们提出的Carplanner超过了此苛刻数据集中的RL-，IL-和基于规则的SOTA方法。

## 声音(cs.SD:Sound)

该领域共有 7 篇论文

### Emotional Face-to-Speech 
[[arxiv](https://arxiv.org/abs/2502.01046)] [[cool](https://papers.cool/arxiv/2502.01046)] [[pdf](https://arxiv.org/pdf/2502.01046)]
> **Authors**: Jiaxin Ye,Boyuan Cao,Hongming Shan
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 情绪化面对语音
- **领域**: 声音,计算机视觉和模式识别,音频和语音处理
- **摘要**: 我们只能从富有表现力的面孔中推断出一种情感声音多少？这个有趣的问题具有巨大的潜力，例如虚拟角色配音和具有富有表现力障碍的个人。现有的面对语音方法在捕获身份特征方面提供了巨大的希望，但努力以情感表达产生多种声音。在本文中，我们探索了一项新任务，称为情感上的面对语音，旨在直接从表现力的面部提示中综合情感语音。为此，我们介绍了Demoface，这是一种新颖的生成框架，它利用基于多层神经音频编解码器的课程学习利用离散扩散变压器（DIT）。具体而言，我们提出了多模式DIT块以动态地对齐文本和语音，同时根据面部情感和身份量身定制人声风格。为了提高训练效率和发电质量，我们进一步引入了用于多级代币处理的粗到精细课程学习算法。此外，我们开发了一个增强的无预测指导，以处理各种条件方案，使多条件生成并有效地解开复杂属性。广泛的实验结果表明，与基线相比，Demoface甚至超过语音驱动的方法会产生更自然和一致的语音。演示显示在https://demoface-ai.github.io/上。

### Metis: A Foundation Speech Generation Model with Masked Generative Pre-training 
[[arxiv](https://arxiv.org/abs/2502.03128)] [[cool](https://papers.cool/arxiv/2502.03128)] [[pdf](https://arxiv.org/pdf/2502.03128)]
> **Authors**: Yuancheng Wang,Jiachen Zheng,Junan Zhang,Xueyao Zhang,Huan Liao,Zhizheng Wu
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: METIS：具有掩盖生成预训练的基础语音生成模型
- **领域**: 声音,人工智能,机器学习,音频和语音处理,信号处理
- **摘要**: 我们介绍了Metis，这是统一语音生成的基础模型。与以前的特定任务或多任务模型不同，METIS遵循预训练和微调范式。它是使用掩盖的生成建模的大规模无标记的语音数据进行预训练的，然后进行了微调以适应各种语音生成任务。具体而言，1）METIS利用两个离散的语音表示：SSL令牌是从语音自学学习（SSL）特征中得出的，而声音令牌则直接从波形进行量化。 2）METIS利用30万小时的不同语音数据对SSL代币进行了掩盖的生成预训练，没有任何其他条件。 3）通过对特定于任务的条件进行微调，METIS即使使用有限的数据和可训练的参数，METIS在支持多模式输入的同时，可以有效地适应各种语音生成任务。实验表明，METIS可以作为统一语音生成的基础模型：METIS在五个语音生成任务中优于最先进的任务特异性或多任务系统，包括零弹奏的文本到文本到语音，语音转换，目标扬声器提取，语音增强，语音增强和lip to-toech，甚至少于20m可训练的参数训练或300次训练数据，甚至更少。音频样本可在https://metis-demo.github.io/上找到。

### Synthetic Audio Helps for Cognitive State Tasks 
[[arxiv](https://arxiv.org/abs/2502.06922)] [[cool](https://papers.cool/arxiv/2502.06922)] [[pdf](https://arxiv.org/pdf/2502.06922)]
> **Authors**: Adil Soubki,John Murzaku,Peter Zeng,Owen Rambow
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: John Murzaku and Adil Soubki contributed equally to this work
- **标题**: 合成音频有助于认知状态任务
- **领域**: 声音,人工智能,计算语言学,机器学习
- **摘要**: NLP社区广泛地关注了认知状态任务的纯文本方法，但是音频可以通过韵律来提供重要的缺失线索。我们认为，文本到语音模型学会了跟踪认知状态的各个方面以产生自然主义的音频，并且信号音频模型隐含地识别与语言模型开发的信息正交。我们提出了合成音频数据微调（SAD），该框架表明，在文本和零摄像的合成音频数据上，从现成的TTS系统上获得了与认知状态建模有关的7个任务。在将合成音频数据添加到仅文本语料库时，我们对仅文本模式显示了改进。此外，与文本和金音频相比，在包含黄金音频的任务和语料库中，我们显示的悲伤框架可以通过文本和合成音频实现竞争性能。

### JamendoMaxCaps: A Large Scale Music-caption Dataset with Imputed Metadata 
[[arxiv](https://arxiv.org/abs/2502.07461)] [[cool](https://papers.cool/arxiv/2502.07461)] [[pdf](https://arxiv.org/pdf/2502.07461)]
> **Authors**: Abhinaba Roy,Renhang Liu,Tongyu Lu,Dorien Herremans
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-12
> **comment**: 8 pages, 5 figures
- **标题**: jamendomaxcaps：带有元数据的大型音乐捕获数据集
- **领域**: 声音,人工智能
- **摘要**: 我们介绍了JamendomaxCaps，这是一个大规模的音乐捕获数据集，该数据集具有来自著名的Jamendo平台的200,000多个免费许可的乐器曲目。该数据集包括由最新字幕模型生成的字幕，并通过估算的元数据增强。我们还引入了一个检索系统，该系统利用音乐功能和元数据来识别类似的歌曲，然后使用当地的大语言模型（LLLM）填充缺失的元数据。这种方法使我们能够为从事音乐语言理解任务的研究人员提供更全面和信息丰富的数据集。我们通过五个不同的测量值定量地验证了这种方法。通过公开提供JamendomAxcaps数据集，我们提供了高质量的资源来推进音乐语言理解任务的研究，例如音乐检索，多模式表示学习和生成音乐模型。

### Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives 
[[arxiv](https://arxiv.org/abs/2502.11858)] [[cool](https://papers.cool/arxiv/2502.11858)] [[pdf](https://arxiv.org/pdf/2502.11858)]
> **Authors**: Zeliang Zhang,Susan Liang,Daiki Shimada,Chenliang Xu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: Accepted by ICLR 2025
- **标题**: 从时间和方式观点重新思考视听对抗脆弱性
- **领域**: 声音,计算机视觉和模式识别
- **摘要**: 虽然视听学习通过利用多种感觉方式使模型对现实世界的了解更丰富，但这种整合也引入了对抗性攻击的新脆弱性。在本文中，我们介绍了视听模型的对抗性鲁棒性，考虑到时间和方式特定的脆弱性。我们提出了两次强大的对抗性攻击：1）一种时间不变性攻击，该攻击利用了连续时间段的固有的时间冗余性，以及2）一种模态错位攻击，引入了音频和视觉方式之间的不一致。这些攻击旨在彻底评估视听模型对不同威胁的鲁棒性。此外，为了防止这种攻击，我们引入了一个新颖的视听对手训练框架。该框架通过合并了针对多模式数据和对抗性课程策略量身定制的有效的对抗扰动手工制作，解决了香草对抗训练的关键挑战。动力学数据集中的广泛实验表明，我们提出的基于时间和模态在降低模型性能中的攻击可以实现最新的性能，而我们的对抗性训练防御在很大程度上可以提高对抗性的鲁棒性以及对抗性训练效率。

### DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning 
[[arxiv](https://arxiv.org/abs/2502.12623)] [[cool](https://papers.cool/arxiv/2502.12623)] [[pdf](https://arxiv.org/pdf/2502.12623)]
> **Authors**: Zhuoyuan Mao,Mengjie Zhao,Qiyu Wu,Hiromi Wakaki,Yuki Mitsufuji
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: 深差异：通过以音乐为中心的多路教学来增强多模式的理解
- **领域**: 声音,人工智能,计算语言学,多媒体,音频和语音处理
- **摘要**: 音乐大语言模型（LLM）的最新进展已大大改善了音乐理解任务，这涉及该模型分析和解释各种音乐元素的能力。这些改进主要集中于整合音乐和文本输入。但是，结合其他方式，例如图像，视频和文本音乐功能来增强音乐理解的潜力仍未得到探索。为了弥合这一差距，我们提出了Deepresonance，这是一种多模式的音乐理解LLM通过多路说明调谐，并通过多路对齐的音乐，文本，图像和视频数据进行调整。为此，我们构建了Music4Way-Mi2T，Music4Way-MV2T和Music4Way-any2t，这是三个四向培训和评估数据集，旨在使深度分辨能够集成视觉和文本音乐功能内容。我们还引入了多样采样的imageBind嵌入式和预先对齐的变压器，以在输入文本LLM之前增强模态融合，从而为多路说明调整量身定制深差。我们的模型在六个音乐理解任务中实现了最先进的表演，突出了辅助方式的好处和深度分辨的结构优势。我们计划开放模型和新建的数据集。

### DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models 
[[arxiv](https://arxiv.org/abs/2502.19924)] [[cool](https://papers.cool/arxiv/2502.19924)] [[pdf](https://arxiv.org/pdf/2502.19924)]
> **Authors**: Weihao wu,Zhiwei Lin,Yixuan Zhou,Jingbei Li,Rui Niu,Qinghua Wu,Songjun Cao,Long Ma,Zhiyong Wu
> **First submission**: 2025-02-27
> **First announcement**: 2025-02-28
> **comment**: Accepted by ICASSP 2025
- **标题**: DIFFCSS：通过扩散模型的多样和表达对话语音综合
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 会话言语综合（CSS）旨在综合上下文适当和表现力的语音，并为增强对对话环境的理解做出了巨大的努力。但是，现有的CSS系统仅限于确定性预测，忽略了潜在响应的多样性。此外，他们很少采用基于语言模型（LM）的TTS骨架，从而限制了综合语音的自然性和质量。为了解决这些问题，在本文中，我们提出了一个创新的CSS框架，该框架利用扩散模型和基于LM的TTS骨架来生成多样的，表现力和上下文相干的语音。提出了基于扩散的上下文感知韵律预测因子，以样本以多模式对话环境为条件的各种韵律嵌入。然后开发了可控制的LM基于LM的TTS主链，以与采样的韵律嵌入合成高质量的语音。实验结果表明，与现有的CSS系统相比，来自DIFFCS的综合语音更多样化，上下文相干和表现力

## 软件工程(cs.SE:Software Engineering)

该领域共有 1 篇论文

### Every Software as an Agent: Blueprint and Case Study 
[[arxiv](https://arxiv.org/abs/2502.04747)] [[cool](https://papers.cool/arxiv/2502.04747)] [[pdf](https://arxiv.org/pdf/2502.04747)]
> **Authors**: Mengwei Xu
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 每个作为代理的软件：蓝图和案例研究
- **领域**: 软件工程,人工智能
- **摘要**: （多模式）大语言模型（LLMS）的兴起已经阐明了软件代理 - 软件可以在其中理解和遵循自然语言的用户说明。但是，在准确性和效率方面，基于API和GUI的代理等现有方法远非令人满意。取而代之的是，我们主张将LLMS授予访问软件内部（源代码和运行时上下文）的访问权限，并授予动态注入生成代码的权限，以进行执行。在这样的白框设置中，可以更好地利用软件上下文和LLM的编码能力。然后，我们就两个流行的基于Web的桌面应用程序介绍了总体设计体系结构和案例研究。我们还深入讨论了挑战和未来的方向。我们认为，这种新的范式有可能从根本上推翻现有的软件代理设计，并最终创建一个数字世界，在该世界中，软件可以理解，操作，协作甚至思考以满足复杂的用户需求。

## 音频和语音处理(eess.AS:Audio and Speech Processing)

该领域共有 4 篇论文

### mWhisper-Flamingo for Multilingual Audio-Visual Noise-Robust Speech Recognition 
[[arxiv](https://arxiv.org/abs/2502.01547)] [[cool](https://papers.cool/arxiv/2502.01547)] [[pdf](https://arxiv.org/pdf/2502.01547)]
> **Authors**: Andrew Rouditchenko,Samuel Thomas,Hilde Kuehne,Rogerio Feris,James Glass
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: Mwhisper-Flamingo用于多语言音频噪音的语音识别
- **领域**: 音频和语音处理,计算机视觉和模式识别,声音
- **摘要**: 视听语音识别（AVSR）结合了基于唇部的视频与音频，可以提高噪声的性能，但大多数方法仅在英语数据上接受培训。一个限制是缺乏大规模的多语言视频数据，这使得很难从头开始训练模型。在这项工作中，我们为多语言AVSR提出了Mwhisper-Flamingo，它结合了预训练的音频模型（Whisper）和视频模型（AV-Hubert）的优势。为了启用更好的多模式集成并改善嘈杂的多语言性能，我们介绍了解码器模态辍学，其中模型在配对的音频输入和单独的音频/视觉输入上都经过训练。 Mwhisper-Flamingo在Muavic上实现了最先进的方法，这是一种9种语言的AVSR数据集。在嘈杂的条件下，视听的Mwhisper-Flamingo始终在所有语言上都胜过一声的音频。

### SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation 
[[arxiv](https://arxiv.org/abs/2502.02603)] [[cool](https://papers.cool/arxiv/2502.02603)] [[pdf](https://arxiv.org/pdf/2502.02603)]
> **Authors**: Chunyu Sun,Bingyu Liu,Zhichao Cui,Anbin Qi,Tian-hao Zhang,Dinghao Zhou,Lewei Lu
> **First submission**: 2025-01-26
> **First announcement**: 2025-02-05
> **comment**: No comments
- **标题**: 印章：语音嵌入语音学习的对齐方式大语言模型，带有检索
- **领域**: 音频和语音处理,计算语言学,声音
- **摘要**: 基于嵌入的检索模型已在用于文本和多模式大语言模型（LLMS）应用程序的检索效果（RAG）技术方面取得了长足的进步。但是，当涉及语音语言模型（SLLMS）时，这些方法仅限于两个阶段的过程，其中自动语音识别（ASR）与基于文本的检索结合使用。该顺序体系结构遭受了高潜伏期和错误传播。为了解决这些限制，我们提出了一个统一的嵌入框架，以消除对中间文本表示的需求。具体而言，该框架包括单独的语音和文本编码器，然后是共享缩放层，将这两种模态映射到一个通用的嵌入空间中。与传统的两阶段方法相比，我们的模型可将管道潜伏期降低50 \％，同时获得更高的检索精度。我们还提供了端到端语音检索中固有的挑战的理论分析，并介绍建筑原理以进行有效的语音到文档匹配。广泛的实验证明了我们在各种声学条件和说话者变化中的方法的鲁棒性，为多模式SLLMS检索系统的新范式铺平了道路。

### A Comprehensive Survey on Generative AI for Video-to-Music Generation 
[[arxiv](https://arxiv.org/abs/2502.12489)] [[cool](https://papers.cool/arxiv/2502.12489)] [[pdf](https://arxiv.org/pdf/2502.12489)]
> **Authors**: Shulei Ji,Songruoyao Wu,Zihao Wang,Shuyu Li,Kejun Zhang
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 对视频到音乐的生成AI的全面调查
- **领域**: 音频和语音处理,人工智能,多媒体
- **摘要**: 视频到音乐生成的迅速增长可以归因于多模式生成模型的上升。但是，缺乏在该领域的工作中全面梳理的文献。为了填补这一空白，本文使用深层生成的AI技术对视频到音乐的一生进行了全面评论，重点关注三个关键组成部分：视觉功能提取，音乐生成框架和调理机制。我们根据每个组件的设计对现有方法进行分类，从而阐明了不同策略的作用。在此之前，我们提供了视频和音乐方式的细粒度分类，说明了不同类别如何影响一代管道中组件的设计。此外，我们总结了可用的多模式数据集和评估指标，同时突出了该领域的持续挑战。

### Gesture-Aware Zero-Shot Speech Recognition for Patients with Language Disorders 
[[arxiv](https://arxiv.org/abs/2502.13983)] [[cool](https://papers.cool/arxiv/2502.13983)] [[pdf](https://arxiv.org/pdf/2502.13983)]
> **Authors**: Seungbae Kim,Daeun Lee,Brielle Stark,Jinyoung Han
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 语言障碍患者的手势感知的零摄声识别
- **领域**: 音频和语音处理,人工智能
- **摘要**: 语言障碍的人通常由于语言处理和理解能力有限而面临重大沟通挑战，这也影响了他们与声音辅助系统的互动，这些系统主要依赖于自动语音识别（ASR）。尽管ASR的进步解决了这种疾病，但几乎没有关注整合非语言交流方法，例如手势，而语言障碍的人实际上依赖于这些方法来补充他们的交流。认识到需要解释不单独语音捕获的视觉信息的潜在含义，我们提出了一种手势感知的ASR系统，该系统利用多模型的多模型模型对具有零摄的学习的多式模型，对有零的语音障碍。我们的实验结果和分析表明，包括手势信息可以显着增强语义理解。这项研究可以帮助开发有效的沟通技术，该技术专门设计，以满足具有语言障碍的个人的独特需求。

## 图像和视频处理(eess.IV:Image and Video Processing)

该领域共有 19 篇论文

### Augmented Intelligence for Multimodal Virtual Biopsy in Breast Cancer Using Generative Artificial Intelligence 
[[arxiv](https://arxiv.org/abs/2501.19176)] [[cool](https://papers.cool/arxiv/2501.19176)] [[pdf](https://arxiv.org/pdf/2501.19176)]
> **Authors**: Aurora Rofena,Claudia Lucia Piccolo,Bruno Beomonte Zobel,Paolo Soda,Valerio Guarrasi
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-03
> **comment**: No comments
- **标题**: 使用生成人工智能的增强智力的乳腺癌多模式虚拟活检
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 全场数字乳房X线摄影（FFDM）是常规乳腺癌筛查的主要成像方式；但是，其有效性受到密集的乳房组织或纤维囊性疾病的患者的限制。对比增强光谱乳房X线摄影（CESM）是一种二级成像技术，可提高肿瘤检测的精度。但是，由于越来越多的辐射暴露，使用造影剂的使用以及有限的可访问性，其应用受到限制。结果，尽管CESM的诊断性能较高，但CESM通常保留用于某些病例，但许多患者仅依靠FFDM。虽然活检仍然是确定诊断的金标准，但它是一种侵入性手术，可能会给患者带来不适。我们介绍了一种用于虚拟活检的多模式，多视图深度学习方法，将FFDM和CESM模态整合在颅和中外侧倾斜的视图中，以将病变分类为恶性或良性。为了应对缺少CESM数据的挑战，我们利用生成人工智能从FFDM扫描中估算CESM图像。实验结果表明，合并CESM模态对于增强虚拟活检的性能至关重要。当缺少真实的CESM数据时，合成的CESM图像被证明有效，胜过单独使用FFDM，尤其是在结合FFDM和CESM模态的多模式配置中。拟议的方法有可能改善诊断工作流程，从而为临床医生提供增强的情报工具，以提高诊断准确性和患者护理。此外，作为对研究社区的贡献，我们公开释放了实验中使用的数据集，从而促进了该领域的进一步进步。

### Registration-Enhanced Segmentation Method for Prostate Cancer in Ultrasound Images 
[[arxiv](https://arxiv.org/abs/2502.00712)] [[cool](https://papers.cool/arxiv/2502.00712)] [[pdf](https://arxiv.org/pdf/2502.00712)]
> **Authors**: Shengtian Sang,Hassan Jahanandish,Cynthia Xinran Li,Indrani Bhattachary,Jeong Hoon Lee,Lichun Zhang,Sulaiman Vesal,Pejman Ghanouni,Richard Fan,Geoffrey A. Sonn,Mirabela Rusu
> **First submission**: 2025-02-02
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 超声图像中前列腺癌的注册增强分割方法
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 前列腺癌是男性与癌症相关的死亡的主要原因，在这种男性中，早期发现大大提高了生存率。尽管MRI-TRUS Fusion活检通过将MRI的详细可视化与TRUS的实时指导相结合，从而提供了卓越的准确性，但它是一种复杂且耗时的程序，在很大程度上依赖于手动注释，从而导致潜在的错误。为了应对这些挑战，我们提出了一种完全自动的MRI TRUS融合分割方法，该方法可以直接在TRUS图像中识别前列腺肿瘤而无需手动注释。与依赖于幼稚数据串联的传统多模式融合方法不同，我们的方法集成了一个注册分割框架，以在MRI和TRUS模态之间对齐和利用空间信息。这种对齐可提高细分精度，并降低对手动努力的依赖。我们的方法在来自斯坦福医院的1,747名患者的数据集上得到了验证，平均骰子系数为0.212，效果超过仅TRUS（0.117）和Naive MRI-TRUS Fusion（0.132）方法，并具有显着改善（p $ <$ 0.01）。该框架展示了降低前列腺癌诊断复杂性的潜力，并提供了适用于其他多模式医学成像任务的灵活建筑。

### Multimodal MRI-Ultrasound AI for Prostate Cancer Detection Outperforms Radiologist MRI Interpretation: A Multi-Center Study 
[[arxiv](https://arxiv.org/abs/2502.00146)] [[cool](https://papers.cool/arxiv/2502.00146)] [[pdf](https://arxiv.org/pdf/2502.00146)]
> **Authors**: Hassan Jahanandish,Shengtian Sang,Cynthia Xinran Li,Sulaiman Vesal,Indrani Bhattacharya,Jeong Hoon Lee,Richard Fan,Geoffrey A. Sonna,Mirabela Rusu
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 用于前列腺癌检测的多模式MRI-ultrasound AI优于放射科医生MRI解释：一项多中心研究
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 生物前磁共振成像（MRI）越来越多地用于靶向可疑的前列腺病变。这导致了人工智能（AI）应用，改善了基于MRI的临床意义前列腺癌（CSPCA）的检测。但是，在活检过程中，MRI检测的病变仍必须映射到转直肠超声（TRUS）图像，从而导致CSPCA缺失。这项研究系统地评估了多模式AI框架，该框架整合了MRI和TRUS图像序列以增强CSPCA识别。这项研究包括三个接受了前列腺活检的机构的三名同类群体的3110名患者。基于3D UNET体系结构的拟议框架对1700个测试用例进行了评估，将性能与单独使用MRI或TRU的单峰AI模型进行了比较。此外，将提出的模型与110例患者队列中的放射科医生进行了比较。与单峰MRI（73％，30％）和TRUS模型（49％，27％）相比，多模式AI方法具有较高的灵敏度（80％）和病变骰子（42％）。与放射科医生相比，多模式模型显示出更高的特异性（88％vs. 78％）和病变骰子（38％vs. 33％），具有等效敏感性（79％）。我们的发现表明，多模式AI在活检和治疗计划过程中改善CSPCA病变靶向的潜力，超过了当前的单峰模型和放射科医生。最终改善了前列腺癌患者的预后。

### Deep Ensembling with Multimodal Image Fusion for Efficient Classification of Lung Cancer 
[[arxiv](https://arxiv.org/abs/2502.00078)] [[cool](https://papers.cool/arxiv/2502.00078)] [[pdf](https://arxiv.org/pdf/2502.00078)]
> **Authors**: Surochita Pal,Sushmita Mitra
> **First submission**: 2025-01-31
> **First announcement**: 2025-02-04
> **comment**: No comments
- **标题**: 与多模式图像融合的深度结合，以有效地分类肺癌
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 这项研究的重点是从多模式肺图像中癌和健康切片的分类。研究中使用的数据包括计算机断层扫描（CT）和正电子发射断层扫描（PET）图像。提出的策略通过利用主成分分析（PCA）和自动编码器来实现PET和CT图像的融合。随后，一个新的基于整体的分类器开发了深层结合的多模式融合（DEMF），采用多数投票来对正在检查的样本图像进行分类。用于可视化受癌症影响图像的分类准确性的梯度加权类激活映射（GRAD-CAM）。鉴于样本量有限，在训练阶段采用的随机图像增强策略。 DEMF网络有助于减轻计算机辅助医学图像分析中稀缺数据的挑战。与三个公开数据集中的最新网络相比，提出的网络与最先进的网络相比。该网络根据指标的表现优于其他人 - 准确性，F1得分，精度和召回率。调查结果突出了提出的网络的有效性。

### AAD-DCE: An Aggregated Multimodal Attention Mechanism for Early and Late Dynamic Contrast Enhanced Prostate MRI Synthesis 
[[arxiv](https://arxiv.org/abs/2502.02555)] [[cool](https://papers.cool/arxiv/2502.02555)] [[pdf](https://arxiv.org/pdf/2502.02555)]
> **Authors**: Divya Bharti,Sriprabha Ramanarayanan,Sadhana S,Kishore Kumar M,Keerthi Ram,Harsh Agarwal,Ramesh Venkatesan,Mohanasankar Sivaprakasam
> **First submission**: 2025-02-04
> **First announcement**: 2025-02-05
> **comment**: Accepted at ICASSP 2025
- **标题**: AAD-DCE：早期和晚期动态对比的汇总多模式注意机制增强了前列腺MRI合成
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 动态对比增强的磁共振成像（DCE-MRI）是一种医学成像技术，在异常病变中的详细可视化和组织灌注和放射学建议中起着至关重要的作用。但是，DCE-MRI涉及给予gadolinium基（GAD）对比剂，这与体内毒性的风险有关。以前的深度学习方法合成DCE-MR图像采用单峰非对比度或低剂量对比度图像的MRI图像，缺乏关注感兴趣的解剖结构内的局部灌注信息。我们提出了AAD-DCE，这是一种生成对抗网络（GAN），其集合的注意歧视器模块由全球和局部歧视者组成。鉴别器提供了一个空间嵌入的注意图，以驱动发电机合成早期和晚期响应DCE-MRI图像。我们的方法采用多模式输入-T2加权（T2W），明显的扩散系数（ADC）和T1预用图像合成。有关Prostatex数据集的广泛比较和消融研究表明，我们的模型（i）对各种生成器基准不可知论，并且（ii）优于其他DCE -MRI合成方法，改进率+0.64 db psnr，+0.0518 SSIM，-0.0.015 MAE，-0.015 MAE，+0.0.015 MAE，+0.0.015 MAE，+0.1 DB， SSIM，-0.021 MAE用于延迟响应，（ii）强调关注的重要性。我们的代码可在https://github.com/bhartidivya/aad-dce上找到。

### A Self-supervised Multimodal Deep Learning Approach to Differentiate Post-radiotherapy Progression from Pseudoprogression in Glioblastoma 
[[arxiv](https://arxiv.org/abs/2502.03999)] [[cool](https://papers.cool/arxiv/2502.03999)] [[pdf](https://arxiv.org/pdf/2502.03999)]
> **Authors**: Ahmed Gomaa,Yixing Huang,Pluvio Stephan,Katharina Breininger,Benjamin Frey,Arnd Dörfler,Oliver Schnell,Daniel Delev,Roland Coras,Charlotte Schmitter,Jenny Stritzelberger,Sabine Semrau,Andreas Maier,Siming Bayer,Stephan Schönecker,Dieter H Heiland,Peter Hau,Udo S. Gaipl,Christoph Bert,Rainer Fietkau,Manuel A. Schmidt,Florian Putz
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 一种自我监管的多模式深度学习方法，以区分后放射治疗的进展与胶质母细胞瘤中的假孕症
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 胶质母细胞瘤（GBM）患者放疗（RT）后，伪孕育（PSP）与真实进展（TP）的准确分化对于最佳治疗计划至关重要。但是，由于PSP和TP的重叠成像特性，此任务仍然具有挑战性。因此，这项研究提出了一种多模式深度学习方法，利用常规解剖学MR图像，临床参数和RT治疗计划信息的补充信息，以提高预测精度。该方法利用自我监督的视觉变压器（VIT）编码多序列MR脑量，从高维输入中有效地捕获全球和局部环境。该编码器在开放式BRATS2021，UPENN-GBM和UCSF-PDGM数据集的无标记的神经胶质瘤MRI数据集中接受了自我监督的上游任务进行训练，以生成来自FLAIR和T1后对抗性序列的临床相关表示的紧凑型，临床相关的表示。然后，这些编码的MR输入与临床数据和RT治疗计划信息集成在一起，通过引导的跨模式注意，从而提高了进度分类的准确性。这项工作是使用来自不同中心的两个数据集开发的：用于培训和验证的Burdenko胶质母细胞瘤进度数据集（n = 59），以及来自大学医院Erlangen（UKER）的GlioCMV进展数据集（uker）（n = 20）进行测试。所提出的方法达到的AUC为75.3％，表现优于当前的最新数据驱动方法。重要的是，提出的方法依赖于容易获得的解剖学MRI序列，临床数据和RT治疗计划信息，从而增强了其临床可行性。提出的方法解决了PSP和TP分化的数据可用性有限的挑战，并可以改善GBM患者的临床决策和优化治疗计划。

### Multi-modal Data Fusion and Deep Ensemble Learning for Accurate Crop Yield Prediction 
[[arxiv](https://arxiv.org/abs/2502.06062)] [[cool](https://papers.cool/arxiv/2502.06062)] [[pdf](https://arxiv.org/pdf/2502.06062)]
> **Authors**: Akshay Dagadu Yewle,Laman Mirzayeva,Oktay Karakuş
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: 28 pages, 7 figures and 5 tables
- **标题**: 多模式数据融合和深度集合学习，以进行准确的作物产量预测
- **领域**: 图像和视频处理,人工智能
- **摘要**: 这项研究介绍了Ricens-Net，这是一种新型的深层集合模型，旨在通过通过多模式数据融合技术整合多种数据源来预测作物产量。该研究专门针对使用合成孔径雷达（SAR），前哨1、2和3卫星的光学遥感数据以及气象测量值，例如表面温度和降雨。该研究的初始现场数据是通过Ernst＆Young（EY）开放科学挑战挑战获得的。主要目的是通过开发能够处理复杂环境数据的机器学习框架来提高作物产量预测的精度。采用了全面的数据工程过程，从100多个潜在预测指标中选择最有用的功能，从而将该集合从5种不同的方式中降低到15个功能。此步骤减轻了``维数的诅咒''并增强了模型性能。Ricens-net体系结构将多个机器学习算法结合在深层整体框架中，整合每种技术的优势以提高预测性准确性。实验结果表明，Ricens-net在341 kg（MAE）中的平均值（MAE）的平均值（MAE）（MAE）的平均值（MAE）（MAE）（MAE）（MAY HA）（Mody）（Mody）（Mody Mody）（Mody Mody）（Mody Mody）（Mody Mody）（Mody Mover）（Mody Mover）（Mody Mover）（Mody Mody）（Mody Mover）complose comploy。区域），大大超过了先前最新模型的性能，包括在EY挑战期间开发的模型。

### A Generative Framework for Bidirectional Image-Report Understanding in Chest Radiography 
[[arxiv](https://arxiv.org/abs/2502.05926)] [[cool](https://papers.cool/arxiv/2502.05926)] [[pdf](https://arxiv.org/pdf/2502.05926)]
> **Authors**: Nicholas Evans,Stephen Baker,Miles Reed
> **First submission**: 2025-02-09
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 双向图像报告理解的生成框架胸部射线照相
- **领域**: 图像和视频处理,计算语言学,计算机视觉和模式识别
- **摘要**: 大语言模型（LLMS）的快速进步已解锁了它们进行多模式任务的潜力，在这些任务中，文本和视觉数据是共同处理的。但是，将LLMS应用于医学成像，特别是对于胸部X射线（CXR），由于需要精确的视觉文本比对并保留关键诊断细节，因此提出了重大挑战。在本文中，我们提出了多阶段自适应视觉语言调整（Mavilt），这是一个新型框架，旨在增强多模式推理和生成CXR理解。 Mavilt结合了临床梯度加权的令牌化过程和层次的微调策略，使其能够生成准确的放射学报告，从文本中综合现实的CXR，并基于回答视觉的临床问题。我们在两个基准数据集（Mimic-CXR和印第安纳大学CXR）上评估了Mavilt，在所有任务中都取得了最新的结果。人类评估进一步验证了毛病的临床相关性和实用性，使其成为现实世界中医疗应用的强大工具。这项工作证明了利用LLM进行多模式医学成像的可行性，同时解决了视觉整合中的关键挑战。

### C2GM: Cascading Conditional Generation of Multi-scale Maps from Remote Sensing Images Constrained by Geographic Features 
[[arxiv](https://arxiv.org/abs/2502.04991)] [[cool](https://papers.cool/arxiv/2502.04991)] [[pdf](https://arxiv.org/pdf/2502.04991)]
> **Authors**: Chenxing Sun,Yongyang Xu,Xuwei Xu,Xixi Fan,Jing Bai,Xiechun Lu,Zhanlong Chen
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: C2GM：从遥感图像中有条件生成的多尺度地图，这些图像受地理特征的约束
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 多尺度地图是测量和制图结果的重要表示，是地理服务的基本组成部分。当前的图像生成网络可以从遥感图像中快速产生地图图块。但是，专为自然图像设计的生成模型通常集中在纹理特征上，忽略了遥感特征的独特特征和瓷砖地图的比例属性。生成模型中的这种局限性会损害地理信息的准确表示，而瓷砖地图生成的质量仍需要改进。扩散模型在各种图像生成任务中取得了巨大的成功，突出了它们解决这一挑战的潜力。本文介绍了C2GM，这是一个新颖的框架，用于通过有条件的引导扩散和多尺度级联产生生成多尺度的瓷砖图。具体而言，我们实现了条件特征融合编码器，以从遥感图像和级联参考双分支输入中提取对象先验，从而确保复杂特征的准确表示。低水平产生的瓷砖是高级地图生成的约束，增强了视觉连续性。此外，我们使用剪辑结合了地图刻度模式信息，以模拟图块地图中地图量表与制图概括之间的关系。广泛的实验评估表明，C2GM始终在所有指标上实现最先进的（SOTA）性能，从而促进了快速有效地产生多尺度大型大型图，以进行紧急响应和远程映射应用。

### MedMimic: Physician-Inspired Multimodal Fusion for Early Diagnosis of Fever of Unknown Origin 
[[arxiv](https://arxiv.org/abs/2502.04794)] [[cool](https://papers.cool/arxiv/2502.04794)] [[pdf](https://arxiv.org/pdf/2502.04794)]
> **Authors**: Minrui Chen,Yi Zhou,Huidong Jiang,Yuhan Zhu,Guanjie Zou,Minqi Chen,Rong Tian,Hiroto Saigo
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: MedMimic：医师启发的多模式融合，用于早期诊断未知来源的发烧
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 未知来源的发烧仍然是诊断挑战。 MedMimic被引入为多模式框架，灵感来自现实世界诊断过程。它使用预验证的模型，例如Dinov2，Vision Transformer和Resnet-18，将高维的18F-FDG PET/CT成像转换为低维，语义上有意义的特征。然后，一个可学习的基于自我注意力的融合网络将这些成像特征与临床数据集成在一起进行分类。从2017年到2023年，使用416个FUO患者病例，多模式融合分类网络MFCN在七个任务中获得了从0.8654到0.9291的宏观分类网络，超过了传统的机器学习和单模式深度学习方法。消融研究和五倍的交叉验证进一步验证了其有效性。通过结合预处理的大型模型和深度学习的优势，MedMimic为疾病分类提供了有希望的解决方案。

### Universal Vessel Segmentation for Multi-Modality Retinal Images 
[[arxiv](https://arxiv.org/abs/2502.06987)] [[cool](https://papers.cool/arxiv/2502.06987)] [[pdf](https://arxiv.org/pdf/2502.06987)]
> **Authors**: Bo Wen,Anna Heinke,Akshay Agnihotri,Dirk-Uwe Bartsch,William Freeman,Truong Nguyen,Cheolhong An
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: No comments
- **标题**: 多模式视网膜图像的通用血管分割
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 我们确定了有关视网膜血管分割的现有研究的两个主要局限性：（1）大多数现有作品仅限于一种模态，即颜色眼底（CF）。但是，在视网膜和视网膜疾病的研究中，每天都使用多模式的视网膜图像，对其他方式的血管分割研究很少。 （2）即使少量作品将实验扩展到有限的新模式，例如多色扫描激光眼镜检查（MC），这些作品仍然需要对新模式的单独模型进行填充。填充将需要额外的培训数据，这很难获得。在这项工作中，我们为多模式视网膜图像提供了一个基础通用血管分割模型（UVSM）。我们不仅以更广泛的方式进行研究，而且我们还提出了一个通用模型，以将所有这些常用方式的血管分割。尽管与现有方法相比，我们的通用模型与现有方法相比更为多，但与最先进的FineTuntoned方法相当的性能。据我们所知，这是第一项实现交叉模式视网膜血管分割的工作，也是研究视网膜血管分割以某些新型方式研究的第一项工作。

### Color Universal Design Neural Network for the Color Vision Deficiencies 
[[arxiv](https://arxiv.org/abs/2502.08671)] [[cool](https://papers.cool/arxiv/2502.08671)] [[pdf](https://arxiv.org/pdf/2502.08671)]
> **Authors**: Sunyong Seo,Jinho Park
> **First submission**: 2025-02-11
> **First announcement**: 2025-02-13
> **comment**: 12 pages, 10 figures
- **标题**: 色觉缺陷的色彩通用设计神经网络
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 有关图像的信息，任何人都应在视觉上理解有关图像的信息，包括那些有颜色缺乏的人。但是，如果似乎扭曲为颜色缺陷的颜色符合相邻对象，则此类信息是无法识别的。本文的目的是提出一个称为CUD-NET的色彩通用设计网络，该网络生成具有色彩缺乏的个人在视觉上可以理解的图像。 CUD-NET是一个卷积深神经网络，可以通过回归分段线性函数的节点点并使用每个图像的特定过滤器来保留颜色并区分输入图像的颜色。为了生成颜色缺陷的CUD图像，我们遵循四步过程。首先，我们根据颜色专家的特定标准来完善CUD数据集。其次，我们通过专门为颜色缺乏视觉的预处理扩展输入图像信息。第三，我们采用多模式融合体系结构来结合特征并处理扩展的图像。最后，我们根据模型的预测图像的组成提出了共轭损耗函数，以解决数据集引起的一对多问题。我们的方法能够产生高质量的CUD图像，以保持颜色和对比度稳定性。 CUD-NET的代码可在GitHub存储库中获得

### Towards Patient-Specific Surgical Planning for Bicuspid Aortic Valve Repair: Fully Automated Segmentation of the Aortic Valve in 4D CT 
[[arxiv](https://arxiv.org/abs/2502.09805)] [[cool](https://papers.cool/arxiv/2502.09805)] [[pdf](https://arxiv.org/pdf/2502.09805)]
> **Authors**: Zaiyang Guo,Ningjun J Dong,Harold Litt,Natalie Yushkevich,Melanie Freas,Jessica Nunez,Victor Ferrari,Jilei Hao,Shir Goldfinger,Matthew A. Jolley,Joseph Bavaria,Nimesh Desai,Alison M. Pouch
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-14
> **comment**: No comments
- **标题**: 迈向双休uspid主动脉瓣修复患者特定的手术计划：4D CT中主动脉瓣的完全自动分割
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 双刺主动脉瓣（BAV）是最普遍的先天性心脏缺陷，可能需要进行狭窄，反流和主动脉症等并发症的手术。 BAV修复手术是有效的，但由于BAV形态的异质性而具有挑战性。可以采用多种成像方式来协助对BAV进行手术计划的定量评估。对比增强的4D计算机断层扫描（CT）产生具有出色的对比度和空间分辨率的体积时间序列。在这些图像中对主动脉尖和根的分割是创建特定于患者的可视化和定量模型的重要步骤。尽管基于深度学习的方法能够完全自动化分段，但不存在BAV特定模型。在阀门分割研究中，对分割结果的临床可用性的定量评估有限。在这项工作中，我们开发了基于NNU-NET的全自动多标签BAV分割管道。预测的分割用于进行手术相关的形态测量值，包括几何尖端高度，连击角和直径，并将结果与​​手动分割进行了比较。自动分割的平均骰子得分超过0.7，对对称的平均距离和对称的平均距离低于0.7 mm，而三个主动脉尖和根壁的平均骰子得分低于0.7 mm。临床上相关的基准测试在手动和预测分段之间表现出良好的一致性。总体而言，4D CT中3D帧的全自动BAV分割可以产生临床上可用的测量方法，以进行手术风险分层，但是需要改善分割的时间一致性。

### ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis 
[[arxiv](https://arxiv.org/abs/2502.12180)] [[cool](https://papers.cool/arxiv/2502.12180)] [[pdf](https://arxiv.org/pdf/2502.12180)]
> **Authors**: Xinpeng Wang,Rong Zhou,Han Xie,Xiaoying Tang,Lifang He,Carl Yang
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: ClusMFL：用于模态的群集增强框架，包括脑成像分析中的多模式联合学习
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别,机器学习
- **摘要**: 多模式联合学习（MFL）已成为一种有前途的方法，用于跨分布式客户，尤其是在医疗保健领域的培训多模型。在大脑成像分析的背景下，模态不完整提出了一个重大挑战，在某些机构由于隐私问题，设备限制或数据可用性问题而导致的一些机构可能缺乏特定的成像方式（例如PET，MRI或CT）。尽管现有工作通常假定模式完整性或过度简化缺失的模式方案，但我们通过在这项研究中考虑客户级和实例级别的方式不完整来模拟更现实的设置。在这种现实的模拟的基础上，我们提出了ClusMFL，这是一种新型的MFL框架，该框架利用聚类以在模态不完整的情况下用于跨机构的脑成像分析。具体而言，ClusMFL利用Finch算法为每个模态标签对的特征嵌入式构造聚类中心池，从而有效地捕获了细粒度的数据分布。然后，这些群集中心通过监督的对比学习将每个模式中的特征对齐方式使用，同时也充当缺失模态的代理，从而允许跨模式知识传递。此外，ClusMFL采用了一种情态感知的聚合策略，进一步增强了模型在严重的方式不完整的情况下的表现。我们利用结构MRI和PET扫描评估了ADNI数据集上提出的框架。广泛的实验结果表明，与不同水平的模态不完整的各种基线方法相比，CLUSMFL可以实现最先进的性能，从而为跨机构的脑成像分析提供了可扩展的解决方案。

### FetalCLIP: A Visual-Language Foundation Model for Fetal Ultrasound Image Analysis 
[[arxiv](https://arxiv.org/abs/2502.14807)] [[cool](https://papers.cool/arxiv/2502.14807)] [[pdf](https://arxiv.org/pdf/2502.14807)]
> **Authors**: Fadillah Maani,Numan Saeed,Tausifa Saleem,Zaid Farooq,Hussain Alasmawi,Werner Diehl,Ameera Mohammad,Gareth Waring,Saudabi Valappi,Leanne Bricker,Mohammad Yaqub
> **First submission**: 2025-02-20
> **First announcement**: 2025-02-21
> **comment**: No comments
- **标题**: 胎儿计：胎儿超声图像分析的视觉语言基础模型
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 基础模型在医疗领域变得越来越有效，在大型数据集上提供了预培训的模型，这些模型很容易适应下游任务。尽管取得了进展，但由于其固有的复杂性，胎儿超声图像对于基础模型仍然是一个具有挑战性的领域，由于配对多模式数据的稀缺，通常需要大量的额外训练和面临局限性。为了克服这些挑战，我们在这里介绍了胎儿clip，这是一种视觉基础模型，能够产生胎儿超声图像的普遍表示。使用多模式学习方法在210,035个胎儿超声图像与文本配对的多种模式学习方法上进行预训练。这代表了迄今为止用于基础模型开发的最大配对数据集。这种独特的训练方法使胎儿能够有效学习胎儿超声图像中存在的复杂解剖特征，从而产生可靠的表示，可用于各种下游应用。在一系列关键的胎儿超声应用方面进行广泛的基准测试，包括分类，胎龄估计，先天性心脏缺陷（CHD）检测和胎儿结构分割，胎儿clip胜过所有基础线，同时证明了明显的普遍性和强大的性能，即使具有有限的标记数据。我们计划公开发布胎儿模型，以造福更广泛的科学界的利益。

### Subclass Classification of Gliomas Using MRI Fusion Technique 
[[arxiv](https://arxiv.org/abs/2502.18775)] [[cool](https://papers.cool/arxiv/2502.18775)] [[pdf](https://arxiv.org/pdf/2502.18775)]
> **Authors**: Kiranmayee Janardhan,Christy Bobby Thomas
> **First submission**: 2025-02-25
> **First announcement**: 2025-02-26
> **comment**: 15 pages, 7 figures, 1 algorithm, 4 tables, journal paper
- **标题**: 使用MRI融合技术对神经胶质瘤的子类分类
- **领域**: 图像和视频处理,计算机视觉和模式识别,机器学习
- **摘要**: 神经瘤是普遍的原发性脑肿瘤，表现出不同的侵略性水平和预后。神经胶质瘤的精确分类对于治疗计划和预测预后至关重要。这项研究旨在开发一种算法，以融合T1，T2，T1CE和流体衰减的反转恢复（FLAIR）序列的MRI图像，以增强神经胶质瘤亚类分类为无肿瘤，坏死性核心，周围肿瘤，肿瘤性水肿和增强肿瘤的功效。这项工作使用了来自Brats数据集的MRI图像。使用Max-Min归一化对图像进行预处理，以确保跨不同图像的像素强度值的一致性。使用UNET结构分别在2D和3D图像上进行了坏死核，周围肿瘤和增强肿瘤的分割。此外，使用加权平均技术融合了来自多模式MRI图像的分段区域。通过捕获切片中的肿瘤形状，边界和强度分布，同时还提供了详细的视图，可以全面地了解大脑体积中的空间范围，形状，质地和定位，从而整合2D和3D分段输出可以提高分类精度。融合图像用作用于神经胶质瘤亚类分类的预训练的RESNET50模型的输入。该网络接受了80％的培训，并在20％的数据上进行了验证。所提出的方法的准确性分类为99.25％，精度为99.30％，召回99.10，F1得分为99.19％，交集超过84.49％，特异性为99.76，其性能表现出比现有技术的表现明显更高。这些发现强调了神经胶质瘤分割和分类在协助准确诊断中的重要性。

### Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis 
[[arxiv](https://arxiv.org/abs/2502.19390)] [[cool](https://papers.cool/arxiv/2502.19390)] [[pdf](https://arxiv.org/pdf/2502.19390)]
> **Authors**: Minjoo Lim,Bogyeong Kang,Tae-Eui Kam
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 肿瘤特异性缺失综合的多模式对比度学习
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 多模式磁共振成像（MRI）对于提供有关脑解剖和病理的互补信息至关重要，从而导致更准确的诊断。但是，由于时间限制，高成本和患者运动伪像等因素，在临床环境中获得高质量的多模式MRI很难。为了克服这一困难，人们对开发可以从可用源源图像合成缺少目标模态图像的生成模型引起了人们的兴趣。因此，我们为缺少MRI设计了一种生成模型，该模型将多模式对比度学习集中在关键的肿瘤区域。具体而言，我们整合了针对多种源方式定制的多模式对比度学习，并通过在对比度学习过程中根据熵选择特征来提高其有效性。此外，我们的网络不仅生成缺失的目标模态图像，而且还可以同时预测分段输出。这种方法提高了发电机精确生成肿瘤区域的能力，最终改善了下游细分任务的性能。通过利用对比度，分割和其他自我代理损失的组合，我们的模型有效地反映了特定目标的信息并产生高质量的目标图像。因此，我们在大脑MR图像合成挑战中的结果表明，所提出的模型在产生缺失的方式方面表现出色。

### Deep learning and classical computer vision techniques in medical image analysis: Case studies on brain MRI tissue segmentation, lung CT COPD registration, and skin lesion classification 
[[arxiv](https://arxiv.org/abs/2502.19258)] [[cool](https://papers.cool/arxiv/2502.19258)] [[pdf](https://arxiv.org/pdf/2502.19258)]
> **Authors**: Anyimadu Daniel Tweneboah,Suleiman Taofik Ahmed,Hossain Mohammad Imran
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 27 pages, 18 figures
- **标题**: 医学图像分析中的深度学习和经典计算机视觉技术：脑MRI组织分割，肺CT COPD注册和皮肤病变分类的案例研究
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 医学成像涵盖了各种任务和方式，在疾病诊断，治疗计划和监测中起着关键作用。这项研究提出了一种新颖的探索，是第一个系统地评估跨多个成像方式的分割，注册和分类任务的探索。从皮肤镜面图像中整合经典和深度学习（DL）方法（DL）方法，以解决脑MRI组织分割，肺CT图像注册和皮肤病变分类，我们证明了这些方法在多样化应用中的互补优势。对于脑组织分割，3D DL模型的表现优于2D和基于贴片的模型，特别是NNU-NET的骰子为0.9397，在Resnet34骨架上具有3D U-NET模型，可提供竞争性的结果，其骰子效果为0.8946。多ATLAS方法为DL方法不可行的情况提供了可靠的替代方法，其平均骰子为0.7267。在肺部CT注册中，基于经典的弹性方法的最小目标注册误差（TRE）为6.68毫米，突出了参数调整的有效性。 HighResnet在DL型号中表现最好，TRE为7.40毫米。对于皮肤病变分类，诸如InceptionResnetv2和Resnet50之类的DL模型的合奏表现出色，达到了90.44％，分别用于二进制和多类分类的93.62％精度。同样，采用单VS-ALL方法，DL获得了94.64％（MEL vs.他人），95.35％，95.35％（BCC vs.他人）和96.93％（SCC vs.他人），而ML模型，而MLSCRAUDTION PERCEPTRON（MLP）在手工制作的特征上提供的替代方案和较高的替代方案，以确定为85.04％的替代品，二进制级任务的83.27％。可应要求提供指向源代码的链接。

### InternVQA: Advancing Compressed Video Quality Assessment with Distilling Large Foundation Model 
[[arxiv](https://arxiv.org/abs/2502.19026)] [[cool](https://papers.cool/arxiv/2502.19026)] [[pdf](https://arxiv.org/pdf/2502.19026)]
> **Authors**: Fengbin Guan,Zihao Yu,Yiting Lu,Xin Li,Zhibo Chen
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: Accepted by ISCAS 2025(Lecture)
- **标题**: InternVQA：通过蒸馏大型基础模型来推进压缩视频质量评估
- **领域**: 图像和视频处理,人工智能,计算机视觉和模式识别
- **摘要**: 视频质量评估任务在很大程度上取决于视频理解所需的丰富功能，例如语义信息，纹理和时间运动。现有的视频基础模型InternVideo2由于其较大的参数大小和大规模的多模式数据，在视频理解任务中表现出了强大的潜力。在此基础上，我们探讨了在压缩方案下internVideo2对视频质量评估的可传递性。为了设计适合此任务的轻质模型，我们提出了一种蒸馏方法，以配备较小的模型，并配备丰富的压缩质量先验。此外，我们在蒸馏过程中检查了不同骨干的性能。结果表明，与其他方法相比，我们从Intervideo2蒸馏出的轻量级模型在压缩视频质量评估中取得了出色的性能。

## 信号处理(eess.SP:Signal Processing)

该领域共有 9 篇论文

### A MIMO Wireless Channel Foundation Model via CIR-CSI Consistency 
[[arxiv](https://arxiv.org/abs/2502.11965)] [[cool](https://papers.cool/arxiv/2502.11965)] [[pdf](https://arxiv.org/pdf/2502.11965)]
> **Authors**: Jun Jiang,Wenjun Yu,Yunfan Li,Yuan Gao,Shugong Xu
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: 6 pages, 2025 ICMLCN accepted
- **标题**: 通过CIR-CSI一致性的MIMO无线通道基础模型
- **领域**: 信号处理,人工智能
- **摘要**: 在人工智能领域，自我监督的学习通过利用大规模的未标记数据集进行预处理，从而证明了卓越的概括能力，这对于无线通信模型适应各种情况至关重要。本文创新地将通道状态信息（CSI）和通道脉冲响应（CIR）视为自然对齐的多模式数据，并提出了第一个名为CSI-CLIP的MIMO无线通道基础模型。通过有效捕获CIR和CSI的联合表示，CSI-CLIP在场景中表现出显着的适应性，并且功能可靠的提取能力。实验结果表明，在定位任务中，CSI-CLIP将平均误差距离降低了22％。在光束管理任务中，与传统监督方法以及频道识别任务相比，它的准确性提高了1％。这些改进不仅强调了CSI-CLIP在整合传感和交流中的潜力和价值，而且还表明了其与现有技术相比的重要优势。此外，将CSI和CIR视为多模式对以及无线通道基础模型的对比度学习，在MIMO无线通信领域打开了新的研究方向。

### Generative Video Semantic Communication via Multimodal Semantic Fusion with Large Model 
[[arxiv](https://arxiv.org/abs/2502.13838)] [[cool](https://papers.cool/arxiv/2502.13838)] [[pdf](https://arxiv.org/pdf/2502.13838)]
> **Authors**: Hang Yin,Li Qiao,Yu Ma,Shuo Sun,Kan Li,Zhen Gao,Dusit Niyato
> **First submission**: 2025-02-19
> **First announcement**: 2025-02-20
> **comment**: No comments
- **标题**: 通过多模式语义融合与大型模型的生成视频语义交流
- **领域**: 信号处理,计算机视觉和模式识别,信息论,图像和视频处理
- **摘要**: 尽管基于香农理论的传统句法交流取得了重大进步，但这些方法努力满足6G沉浸式沟通的要求，尤其是在挑战性的传播条件下。随着生成人工智能（Genai）的发展，在使用高级语义信息重建视频中取得了进展。在本文中，我们提出了一个可扩展的生成视频语义通信框架，该框架提取和传输语义信息以实现高质量的视频重建。具体而言，在发射机，描述和其他条件信号（例如，第一帧，草图等）中分别从源视频中提取，分别充当文本和结构语义。在接收器中，基于扩散的Genai大型模型被用来融合多种模态的语义来重建视频。仿真结果表明，在超低通道带宽比（CBR）下，我们的方案有效地捕获了语义信息，以重建与人类感知在不同信噪比下对齐的视频。值得注意的是，提出的``第一帧+desc。''方案始终达到SNR> 0 dB时CBR = 0.0057时的剪辑得分超过0.92。这证明了即使在低SNR条件下，它也证明了其强大的性能。

### Multimodal Bearing Fault Classification Under Variable Conditions: A 1D CNN with Transfer Learning 
[[arxiv](https://arxiv.org/abs/2502.17524)] [[cool](https://papers.cool/arxiv/2502.17524)] [[pdf](https://arxiv.org/pdf/2502.17524)]
> **Authors**: Tasfiq E. Alam,Md Manjurul Ahsan,Shivakumar Raman
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 多模式轴承故障分类在可变条件下：一维CNN带有转移学习
- **领域**: 信号处理,人工智能,机器学习
- **摘要**: 轴承在确保旋转机械的可靠性和效率方面起着不可或缺的作用 - 减少摩擦和处理临界负载。构成多达90％机械故障的90％的轴承故障突出了对可靠状态监测和故障检测的必要性。这项研究提出了一种多模式轴承断层分类方法，该方法依赖于一维卷积神经网络（1D CNN）框架内的振动和运动相电流信号。该方法融合了来自多个信号的特征，以增强故障检测的准确性。在基线条件下（1,500 rpm，0.7 nm载荷扭矩和1,000 N径向力），该模型在添加L2正则化时达到了96％的精度。与非规范化模型相比，这显着改善了2％。此外，该模型通过采用转移学习（TL）策略来证明在三个不同的操作条件上的稳健性能。在测试的TL变体中，将参数保留到第一个最大池层，然后调整后续层的方法可实现最高的性能。尽管这种方法在各种条件下达到了极好的精度，但由于其可训练的参数数量越大，它需要更多的计算时间。为了解决资源限制，较少的计算密集型模型提供了可行的权衡，尽管其精确成本很小。总体而言，这种具有较晚融合和TL策略的多模式1D CNN框架为在具有可变操作条件的工业环境中提供了更准确，适应性和高效的轴承故障分类的基础。

### Multimodal Sleep Stage and Sleep Apnea Classification Using Vision Transformer: A Multitask Explainable Learning Approach 
[[arxiv](https://arxiv.org/abs/2502.17486)] [[cool](https://papers.cool/arxiv/2502.17486)] [[pdf](https://arxiv.org/pdf/2502.17486)]
> **Authors**: Kianoosh Kazemi,Iman Azimi,Michelle Khine,Rami N. Khayat,Amir M. Rahmani,Pasi Liljeberg
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: 使用视觉变压器的多式联运睡眠阶段和睡眠呼吸暂停分类：多任务解释方法
- **领域**: 信号处理,机器学习
- **摘要**: 睡眠是人类生理学的重要组成部分，对整体健康和生活质量做出了重大贡献。准确的睡眠分期和障碍检测对于评估睡眠质量至关重要。文献研究提出了利用单模式信号的基于PSG的方法和机器学习方法。但是，现有方法通常缺乏多模式，多标签框架，并且分别解决睡眠阶段和疾病分类。在本文中，我们提出了一个1D视频变压器，用于同时对睡眠阶段和睡眠障碍进行分类。我们的方法利用了睡眠障碍与特定睡眠阶段模式的相关性，并同时识别睡眠阶段和睡眠障碍。该模型是使用多模式 - 多型式感觉数据（包括光绘制图，呼吸流和呼吸努力信号）训练和测试的。提出的方法显示，五阶段睡眠分类的总体准确性（Cohen's Kappa）为78％（0.66），睡眠呼吸暂停分类为74％（0.58）。此外，我们分析了编码器注意权重，以阐明我们的模型的预测并研究了不同特征对模型输出的影响。结果表明，确定的模式（例如呼吸道和峰）对最终分类过程做出了更高的贡献。

### Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework 
[[arxiv](https://arxiv.org/abs/2502.17481)] [[cool](https://papers.cool/arxiv/2502.17481)] [[pdf](https://arxiv.org/pdf/2502.17481)]
> **Authors**: Cheol-Hui Lee,Hakseung Kim,Byung C. Yoon,Dong-Joo Kim
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-25
> **comment**: 18 pages, 5 figures
- **标题**: 使用多模式混合自我监督学习框架进行睡眠分析的基础模型
- **领域**: 信号处理,人工智能,机器学习
- **摘要**: 睡眠对于维持人类健康和生活质量至关重要。在睡眠期间分析生理信号对于评估睡眠质量和诊断睡眠障碍至关重要。但是，临床医生的手动诊断是时间密集型和主观的。尽管深度学习的进步增强了自动化，但这些方法仍然在很大程度上取决于标记的大规模数据集。这项研究介绍了Synthsleepnet，这是一种旨在分析多模式的自我监督的学习框架，旨在分析多型多摄影术（PSG）数据。合成器有效地整合了掩盖的预测和对比度学习，以利用多种模态的互补特征，包括脑电图（EEG），电摄影（EOG）（EOG），肌电图（EMG）和心电图（ECG）。这种方法使模型能够学习PSG数据的高度表达性表示。此外，开发了基于MAMBA的时间上下文模块，以有效地捕获跨信号的上下文信息。与三个下游任务中的最新方法相比，合成螺丝夹的性能卓越：睡眠阶段分类，呼吸暂停检测和呼吸症检测，精度分别为89.89％，99.75％和89.60％。该模型在半监督的学习环境中表现出良好的表现，标签有限，在同一任务中实现了87.98％，99.37％和77.52％的准确性。这些结果强调了该模型作为PSG数据综合分析的基础工具的潜力。与其他方法相比，Synthsleepnet在多个下游任务中表现出了全面的卓越性能，因此预计将为睡眠障碍监测和诊断系统设定新的标准。

### ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis 
[[arxiv](https://arxiv.org/abs/2502.17475)] [[cool](https://papers.cool/arxiv/2502.17475)] [[pdf](https://arxiv.org/pdf/2502.17475)]
> **Authors**: Xu Wang,Jiaju Kang,Puyu Han
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: ECG-Expert-QA：用于评估心脏病诊断中医学大语言模型的基准
- **领域**: 信号处理,人工智能,计算语言学,机器学习
- **摘要**: 我们提出了ECG-Expert-QA，这是一个综合的多模式数据集，旨在评估ECG解释中的诊断能力，将实际临床数据与系统生成的合成病例整合在一起。该数据集包括六个基本诊断任务，包括47,211个精心策划的问题解答对，这些问题跨越了临床方案，从基本节奏分析到复杂的病例解释。通过通过严格的医学知识引导的过程来模拟挑战性临床病例，ECG-Expert-QA不仅可以增强带注释的诊断数据的可用性，而且还可以显着提高临床表现的复杂性和多样性，包括罕见的心脏条件和时间进展模式。该设计可以全面评估多个维度的医学语言模型，包括诊断准确性，临床推理和知识整合。为了促进全球研究合作，ECG-Expert-QA有中文和英语版本，具有严格的质量控制，可确保语言和临床一致性。该数据集的挑战性诊断任务包括对复杂心律不齐的解释，微妙的缺血性变化的识别以及临床环境的整合，将其确立为推进AI-AI-ASS辅助ECG解释的有效基准，并突破当前诊断模型的界限。我们的数据集是开源的，可在https://github.com/zaozzz/ecg-expert-qa上找到

### MC2SleepNet: Multi-modal Cross-masking with Contrastive Learning for Sleep Stage Classification 
[[arxiv](https://arxiv.org/abs/2502.17470)] [[cool](https://papers.cool/arxiv/2502.17470)] [[pdf](https://arxiv.org/pdf/2502.17470)]
> **Authors**: Younghoon Na,Hyun Keun Ahn,Hyun-Kyung Lee,Yoongeol Lee,Seung Hun Oh,Hongkwon Kim,Jeong-Gun Lee
> **First submission**: 2025-02-13
> **First announcement**: 2025-02-25
> **comment**: No comments
- **标题**: MC2Sleepnet：多模式的交叉掩模与对比度学习的睡眠阶段分类
- **领域**: 信号处理,人工智能
- **摘要**: 睡眠深刻影响我们的健康，睡眠不足或疾病会引起身体和精神问题。尽管从先前的研究中进行了重大发现，但挑战仍然在优化深度学习模型方面，尤其是在多模式学习中进行高准确的睡眠阶段分类。我们的研究介绍了MC2Sleepnet（用于睡眠阶段分类网络的对比度学习的多模式交叉掩模）。它的目的是促进卷积神经网络（CNN）和变压器体系结构之间的有效合作，以借助对比度学习和交叉掩盖，以进行多模式培训。原始的单个通道脑电图和相应的频谱图为多模式学习提供了不同特征的方式。我们的MC2Sleepnet在SleepEDF-78和Sleep Heart Health研究（SHHS）上的精度为84.6％，精确度为84.6％。这些结果证明了我们在小型和大型数据集中提出的网络的有效概括。

### SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning 
[[arxiv](https://arxiv.org/abs/2502.19668)] [[cool](https://papers.cool/arxiv/2502.19668)] [[pdf](https://arxiv.org/pdf/2502.19668)]
> **Authors**: Mingsheng Cai,Jiuming Jiang,Wenhao Huang,Che Liu,Rossella Arcucci
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 至高无上：多模式心电图表示学习的有监督的训练框架
- **领域**: 信号处理,人工智能,计算语言学,机器学习
- **摘要**: 心血管疾病是全球死亡和残疾的主要原因。心电图（ECG）记录对于诊断和监测心脏健康至关重要，但是获得大规模注释的ECG数据集是劳动密集型且耗时的。最近的ECG自我监督学习（ESSL）方法通过学习功能而没有广泛的标签来减轻这种方法，但无法捕获细粒度的临床语义，并且需要广泛的特定于任务的微调。为了应对这些挑战，我们建议$ \ textbf {supreme} $，a $ \ textbf {su} $ pervised $ \ textbf {pre} $  -  $ \ textbf {m} $ textbf {m} $ ultimopimodal $ \ textbf {e textbf {e} $ cg cg表示的培训框架。 Supreme应用大型语言模型（LLMS）从自由文本ECG报告中提取结构化的临床实体，过滤噪声和无关紧要的内容，增强临床表示学习，并构建高质量的，高质量的标记数据集。通过使用基于文本的心脏查询而不是传统的分类标签，Supreme可以在不进行其他微调的情况下对未见疾病进行零射击分类。我们在涵盖127个心脏条件的六个下游数据集上评估了至高无上的评估，从而超过了1.96 \％以上的零零AUC AUC性能和多模式方法。结果证明了至高无上在利用结构化的，临床上相关的知识来实现​​高质量心电图表示方面的有效性。所有代码和数据将在接受后发布。

### Integrating Biological and Machine Intelligence: Attention Mechanisms in Brain-Computer Interfaces 
[[arxiv](https://arxiv.org/abs/2502.19281)] [[cool](https://papers.cool/arxiv/2502.19281)] [[pdf](https://arxiv.org/pdf/2502.19281)]
> **Authors**: Jiyuan Wang,Weishan Ye,Jialin He,Li Zhang,Gan Huang,Zhuliang Yu,Zhen Liang
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: No comments
- **标题**: 整合生物学和机器智能：脑部计算机界面中的注意机制
- **领域**: 信号处理,人工智能,机器学习
- **摘要**: 随着深度学习的快速发展，注意机制在脑电图（EEG）信号分析中变得必不可少，从而显着增强了脑部计算机界面（BCI）应用。本文对基于传统和变压器的注意机制，其嵌入策略及其在基于EEG的BCI中的应用进行了全面综述，并特别着重于多模式数据融合。通过捕获跨时间，频率和空间通道的脑电图变化，注意机制可改善特征提取，表示学习和模型鲁棒性。这些方法可以大致分为传统的注意机制，这些方法通常与卷积和经常性网络集成在一起，以及基于变压器的多头自我注意力，这些自我注意力符合捕获长期依赖性。除了单模式分析外，注意机制还增强了多模式的脑电图应用，从而促进了脑电图与其他生理或感觉数据之间的有效融合。最后，我们讨论了基于注意力的脑电图建模中的现有挑战和新兴趋势，突出了未来推进BCI技术的方向。这篇综述旨在为寻求利用注意力机制的研究人员提供宝贵的见解，以改善脑电图的解释和应用。

## 优化与控制(math.OC:Optimization and Control)

该领域共有 1 篇论文

### Diffusion at Absolute Zero: Langevin Sampling Using Successive Moreau Envelopes 
[[arxiv](https://arxiv.org/abs/2502.01358)] [[cool](https://papers.cool/arxiv/2502.01358)] [[pdf](https://arxiv.org/pdf/2502.01358)]
> **Authors**: Andreas Habring,Alexander Falk,Thomas Pock
> **First submission**: 2025-02-03
> **First announcement**: 2025-02-04
> **comment**: :65C40; 65C05; 68U10; 65C60ACM Class:G.3; G.1.6
- **标题**: 绝对零的扩散：使用连续的Moreau信封的Langevin采样
- **领域**: 优化与控制,计算机视觉和模式识别,数值分析
- **摘要**: 在本文中，我们提出了一种从$π（x）\ propto \ exp（-u（x））$的Gibbs分布中采样的新方法，并具有潜在的$ u（x）$。特别是，受扩散模型的启发，我们建议考虑一个序列$（π^{t_k}）_ k $的目标密度近似值，$π^{t_k} \ your $ k $ for $ k $ small，另一方面，另一方面，$π^{t_k} $ for $ k $ k $ k sampling $ k sampling $ k samplable。该序列是通过用Moreau信封代替潜在的$ U $部分来获得的。采样是在退火的Langevin类型过程中执行的，也就是说，从$π^{t_k} $中依次采样，以减少$ k $，从而有效地指导样本从简单的起始密度到更复杂的目标。除了进行理论分析外，我们还显示了实验结果，以增加收敛速度和适用于多模式密度$π$的收敛速度和适用性。

## 生物分子(q-bio.BM:Biomolecules)

该领域共有 1 篇论文

### CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening 
[[arxiv](https://arxiv.org/abs/2502.11001)] [[cool](https://papers.cool/arxiv/2502.11001)] [[pdf](https://arxiv.org/pdf/2502.11001)]
> **Authors**: Gen Zhou,Sugitha Janarthanan,Yutong Lu,Pingzhao Hu
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-17
> **comment**: Gen Zhou and Sugitha Janarthanan contributed equally; Accepted at ICLR 2025
- **标题**: CL-MFAP：基于对比度学习的分子特性预测和抗生素筛选的多模式基础模型
- **领域**: 生物分子,人工智能,机器学习,定量方法
- **摘要**: 由于抗菌素耐药性的升高，鉴定具有抗生素潜力的新型化合物对于解决这个全球健康问题至关重要。但是，传统的药物开发方法昂贵且效率低下。认识到对更有效解决方案的紧迫需求，研究人员已转向机器学习技术，以简化新型抗生素化合物的预测和开发。尽管基础模型在抗生素发现中表现出了希望，但当前的主流努力仍然没有完全利用多模式分子数据的潜力。最近的研究表明，利用多模式数据的对比学习框架在各个领域的表示学习中表现出色。在此基础上，我们介绍了CL-MFAP，这是一种无监督的对比度学习（CL）的多模式基础（MF）模型，专门针对使用三种分子数据来发现具有潜在抗生素特性（AP）的小分子。该模型采用160万种生物活性分子，具有从Chembl数据集的药物样性能到共同预处理三个编码器：（1）基于变压器的编码器，具有旋转位置嵌入用于处理微笑字符串的旋转位置； （2）另一个基于变压器的编码器，结合了一种新型的双层路由注意机制来处理分子图表示； （3）使用多层感知器的摩根指纹编码器，以实现对比度学习目的。 CL-MFAP通过有效利用不同的分子模态来优于抗生素性能预测中的基线模型，并在微调抗生素相关性属性预测任务时表现出优质的域特异性性能。

## 基因组学(q-bio.GN:Genomics)

该领域共有 1 篇论文

### Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning 
[[arxiv](https://arxiv.org/abs/2502.03499)] [[cool](https://papers.cool/arxiv/2502.03499)] [[pdf](https://arxiv.org/pdf/2502.03499)]
> **Authors**: Zehui Li,Vallijah Subasri,Yifei Shen,Dongsheng Li,Yiren Zhao,Guy-Bart Stan,Caihua Shan
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-06
> **comment**: No comments
- **标题**: Omni-DNA：统一的跨模式和多任务学习的基因组基础模型
- **领域**: 基因组学,人工智能,机器学习
- **摘要**: 大型语言模型（LLMS）表现出各种任务之间的显着普遍性，但是基因组基础模型（GFMS）仍然需要为每个下游应用程序单独进行填充，随着模型尺寸的增长而产生了明显的开销。此外，现有的GFM受刚性输出格式的限制，从而将其适用性限制在各种基因组任务中。在这项工作中，我们重新访问了基于变压器的自动回归模型，并引入了Omni-DNA，Omni-DNA是一个跨模式多任务模型，范围从2000万到10亿个参数。我们的方法由两个阶段组成：（i）对DNA序列进行预处理，并具有下一个令牌预测目标，以及（ii）为多个下游任务的多模式特定任务特异性令牌和填充扩展。当对核苷酸变压器和GB基准测试进行评估时，Omni-DNA可以在26个任务中的18个任务中实现最先进的性能。通过多任务登录，Omni-DNA立即解决10个乙酰化和甲基化任务，超过了对每个任务进行训练的模型。最后，我们设计了两个复杂的基因组任务，即DNA2功能和针中的DNA，它们分别将DNA序列映射到文本功能描述和图像，表明Omni-DNA的跨模式功能以扩大基因组应用的范围。所有模型均可通过https://huggingface.co/collections/zehui127获得

## 神经元和认知(q-bio.NC:Neurons and Cognition)

该领域共有 4 篇论文

### Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning 
[[arxiv](https://arxiv.org/abs/2502.10425)] [[cool](https://papers.cool/arxiv/2502.10425)] [[pdf](https://arxiv.org/pdf/2502.10425)]
> **Authors**: Wei Wu,Can Liao,Zizhen Deng,Zhengrui Guo,Jinzhuo Wang
> **First submission**: 2025-02-05
> **First announcement**: 2025-02-17
> **comment**: Accepted by ICLR'2025
- **标题**: 使用对比度学习的动力学来自动力学的神经元柏拉金固有表示
- **领域**: 神经元和认知,人工智能,神经和进化计算
- **摘要**: 柏拉图表示假设提出了不同数据方式背后的通用，独立于模态的现实表示。受此启发，我们将每个神经元视为一个系统，并在各种外围条件下检测其多段活动数据。我们假设同一神经元具有时间不变的表示，反映了其内在特性，例如分子谱，位置和形态。获得这些固有神经元表示的目的具有两个标准：（i）来自同一神经元的片段应该比来自不同神经元的片段具有更相似的表示； （ii）表示形式必须很好地推广到室外数据。为了满足这些，我们提出了神经元（神经柏拉图内在表示）框架。它使用对比度学习，来自与正对相同的神经元的片段，而来自不同神经元的神经元作为负对。在实施中，我们使用VICREG，该VICREG专注于正对并通过正规化分开不同的样本。我们测试了有关IZHikeVich模型模拟的神经元种群动力学数据的方法。结果准确地基于预设超参数鉴定了神经元类型。我们还将其应用于两个现实世界的神经元动力学数据集，该数据集具有来自空间转录组学和神经元位置的神经元类型注释。我们的模型学到的表示形式准确地预测了神经元的类型和位置，并且在室外数据（来自看不见的动物）上是强大的。这表明了我们理解神经元系统和未来神经科学研究的潜力。

### Noumenal Labs White Paper: How To Build A Brain 
[[arxiv](https://arxiv.org/abs/2502.13161)] [[cool](https://papers.cool/arxiv/2502.13161)] [[pdf](https://arxiv.org/pdf/2502.13161)]
> **Authors**: Maxwell J. D. Ramstead,Candice Pattisapu,Jason Fox,Jeff Beck
> **First submission**: 2025-02-16
> **First announcement**: 2025-02-19
> **comment**: No comments
- **标题**: Noumenal Labs白皮书：如何建立大脑
- **领域**: 神经元和认知,人工智能
- **摘要**: 这份白皮书描述了人造或机器智能的一些设计原理，这些设计原理指导了Noumenal Labs的工作。这些原则是从自然和我们来代表和理解它的手段中得出的。该领域的研发的最终目标应该是设计机器智能，以增强我们对世界的理解并增强我们在不取代我们的情况下采取行动的能力。在前两个部分中，我们研究了我们方法的核心动机：解决基础问题。我们认为，解决基础问题的解决方案取决于我们所居住的世界上基于的模型的设计，而不是单词模型。能够显着增强我们对人类世界的理解的机器超级智能必须像我们所做的那样代表世界，并能够建立我们已经知道的新知识。换句话说，必须适当地扎根并明确设计用于以科学方法为基础的理性，经验探究。该设计原则的主要含义是代理必须能够自主参与因果物理发现。我们讨论了这种方法的务实含义，尤其是现实的3D世界建模和多模式，多维时间序列分析中的用例。

### Category-Selective Neurons in Deep Networks: Comparing Purely Visual and Visual-Language Models 
[[arxiv](https://arxiv.org/abs/2502.16456)] [[cool](https://papers.cool/arxiv/2502.16456)] [[pdf](https://arxiv.org/pdf/2502.16456)]
> **Authors**: Zitong Lu,Yuxin Wang
> **First submission**: 2025-02-23
> **First announcement**: 2025-02-24
> **comment**: No comments
- **标题**: 深网中的类别选择性神经元：比较纯粹的视觉和视觉模型
- **领域**: 神经元和认知,计算机视觉和模式识别
- **摘要**: 人脑中的类别选择区域，例如梭形面积（FFA），外体外区域（EBA），Parahampocampal位置区域（PPA）和视觉单词形式区域（VWFA），在高级视觉处理中起着至关重要的作用。在这里，我们研究了人工神经网络（ANN）是否表现出相似的类别选择性神经元以及这些神经元在模型层之间以及纯粹的视觉和视觉模型之间的变化。受FMRI功能性本地化实验的启发，我们向深层网络展示了来自不同类别（面，身体，场景，单词，拼命的场景和拼写的单词）的图像，并使用统计标准确定了类别选择性神经元。比较Resnet和基于结构控制的基于RESNET的剪辑模型，我们发现两个模型都包含类别选择性神经元，其比例在各个层之间的增加，反映了高级视觉脑区域的类别选择性。但是，与Resnet相比，夹显示出更高的比例但类别选择性神经元的特异性。此外，剪辑的类别选择性神经元在特征图中更均匀分布，并且在整个层之间表现出更大的代表性一致性。这些发现表明，语言学习增加了类别选择性神经元的数量，同时降低了其选择性强度，从而重塑了深层网络中的视觉表示。我们的研究提供了有关ANN如何反映生物学视觉以及多模式学习如何影响类别选择性表示的见解。

### Deciphering Functions of Neurons in Vision-Language Models 
[[arxiv](https://arxiv.org/abs/2502.18485)] [[cool](https://papers.cool/arxiv/2502.18485)] [[pdf](https://arxiv.org/pdf/2502.18485)]
> **Authors**: Jiaqi Xu,Cuiling Lan,Xuejin Chen,Yan Lu
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-26
> **comment**: 22 pages, 23 figures
- **标题**: 视觉模型中神经元的解密功能
- **领域**: 神经元和认知,计算机视觉和模式识别
- **摘要**: 开源视力语言模型（VLM）的迅速增长促进了跨不同领域的大量应用。确保这些模型的透明度和解释性对于培养值得信赖和负责任的AI系统至关重要。在这项研究中，我们的目标是深入研究VLM的内部，以解释单个神经元的功能。我们观察到有关输入视觉令牌和文本令牌的神经元的激活，并揭示了一些有趣的发现。尤其是，我们发现分别仅负责视觉或文本信息，或两者兼有神经元，分别将它们称为视觉神经元，文本神经元和多模式神经元。我们建立一个框架，该框架可以自动使用GPT-4O助手对神经元的解释。同时，对于视觉神经元，我们提出了一个激活模拟器来评估视觉神经元解释的可靠性。在LLAVA的一个代表性VLM之上进行系统统计分析，发现了不同类别的神经元的行为/特征。

## 定量方法(q-bio.QM:Quantitative Methods)

该领域共有 2 篇论文

### Advancing Precision Oncology Through Modeling of Longitudinal and Multimodal Data 
[[arxiv](https://arxiv.org/abs/2502.07836)] [[cool](https://papers.cool/arxiv/2502.07836)] [[pdf](https://arxiv.org/pdf/2502.07836)]
> **Authors**: Luoting Zhuang,Stephen H. Park,Steven J. Skates,Ashley E. Prosper,Denise R. Aberle,William Hsu
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-12
> **comment**: This work has been submitted to the IEEE RBME for potential publication
- **标题**: 通过建模纵向和多模式数据来推进精度肿瘤学
- **领域**: 定量方法,机器学习
- **摘要**: 癌症通过遗传，表观遗传，微环境和表型变化的复杂相互作用而不断发展。这种动态行为驱动了不受控制的细胞生长，转移，免疫逃避和耐药性，对有效监测和治疗提出了挑战。但是，当今的数据驱动肿瘤学研究主要集中于横截面分析，使用单个模态的数据限制了完全表征和解释疾病动态异质性的能力。现在，多尺度数据收集和计算方法的进步使得可以发现纵向多模式生物标志物以进行精确肿瘤学。纵向数据揭示了疾病进展和治疗反应的模式，这些模式从单个时间点数据中看不出，从而及时异常检测和动态治疗适应。多模式数据集成提供来自不同来源的互补信息，以进行更精确的风险评估和癌症治疗的靶向。在这篇综述中，我们调查了纵向和多模式建模的方法，强调了它们的协同作用，以提供针对患者癌症独特特征的个性化护理的多方面见解。我们总结了纵向多模式分析的当前挑战和未来方向，以提高精度肿瘤学。

### Towards Quantum Tensor Decomposition in Biomedical Applications 
[[arxiv](https://arxiv.org/abs/2502.13140)] [[cool](https://papers.cool/arxiv/2502.13140)] [[pdf](https://arxiv.org/pdf/2502.13140)]
> **Authors**: Myson Burch,Jiasen Zhang,Gideon Idumah,Hakan Doga,Richard Lartey,Lamis Yehia,Mingrui Yang,Murat Yildirim,Mihriban Karaayvaz,Omar Shehab,Weihong Guo,Ying Ni,Laxmi Parida,Xiaojuan Li,Aritra Bose
> **First submission**: 2025-02-18
> **First announcement**: 2025-02-19
> **comment**: 31 pages, 7 figures
- **标题**: 在生物医学应用中朝量子张量分解
- **领域**: 定量方法,机器学习
- **摘要**: 张量分解已成为多模式生物医学数据中特征提取的有力框架。在这篇综述中，我们对张量分解方法进行了全面分析，例如Tucker，Candecomp/Parafac，尖刺张量分解等及其在生物医学领域（例如成像，多组学和空间转录组学）跨生物医学领域的不同应用。为了系统地研究文献，我们采用了一种基于主题建模的方法，该方法识别和组在使用张量分解的生物医学中不同的主题亚地区，从而揭示了关键趋势和研究方向。我们评估了与潜在空间的可伸缩性相关的挑战，以及获得张量的最佳等级，这通常会阻碍从日益大而复杂的数据集中提取有意义的特征。此外，我们讨论了张量分解的量子算法的最新进展，并探讨如何利用量子计算来应对这些挑战。我们的研究包括针对量子计算平台的初步资源估计分析，并检查了在近期量子设备上实施量子增强张量分解方法的可行性。总的来说，这篇综述不仅综合了在生物医学分析中张量分解的当前应用和挑战，而且还概述了有希望的量子计算策略，以增强其对从复杂生物医学数据的可行见解的影响。

## 统计金融(q-fin.ST:Statistical Finance)

该领域共有 1 篇论文

### Multimodal Stock Price Prediction 
[[arxiv](https://arxiv.org/abs/2502.05186)] [[cool](https://papers.cool/arxiv/2502.05186)] [[pdf](https://arxiv.org/pdf/2502.05186)]
> **Authors**: Furkan Karadaş,Bahaeddin Eravcı,Ahmet Murat Özbayoğlu
> **First submission**: 2025-01-23
> **First announcement**: 2025-02-10
> **comment**: 9 pages, 6 table
- **标题**: 多模式股票价格预测
- **领域**: 统计金融,人工智能,机器学习
- **摘要**: 在金融市场受到许多静态和动态因素影响的时代，将各种数据源与机器学习仔细整合以进行准确的股价预测变得越来越重要。本文通过结合包括传统财务指标，推文和新闻文章在内的各种来源的数据来探讨一种用于股票价格预测的多模式机器学习方法。我们通过使用Chatgpt-4O和Finbert模型对这些文本数据进行情感分析来捕获实时市场动态和投资者情绪。我们研究这些集成数据流如何通过标准的长期记忆（LSTM模型）来增强预测，以说明性能提高的程度。我们的研究结果表明，合并上述数据源大大提高了参考模型的预测有效性高达5％。我们还提供了对这些方式的个体和组合预测能力的见解，突出了将推文和新闻文章中的情感分析纳入情感分析的重大影响。这项研究提供了一个系统而有效的框架，用于在财务时间序列中应用多模式数据分析技术，该技术为投资者提供了新的看法，以利用数据进行决策。

## 机器学习(stat.ML:Machine Learning)

该领域共有 8 篇论文

### Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and beyond 
[[arxiv](https://arxiv.org/abs/2502.04575)] [[cool](https://papers.cool/arxiv/2502.04575)] [[pdf](https://arxiv.org/pdf/2502.04575)]
> **Authors**: Wei Guo,Molei Tao,Yongxin Chen
> **First submission**: 2025-02-06
> **First announcement**: 2025-02-07
> **comment**: No comments
- **标题**: 正常估计的复杂性分析：从jarzynski平等到退火重要性采样及以后
- **领域**: 机器学习,机器学习,数值分析,计算物理,计算
- **摘要**: 给定一个不当的概率密度$π\ propto \ mathrm {e}^{ -  v} $，估算其正常化常数$ z = \ int _ {\ mathbb {r}^d}^d} \ mathrm {e}贝叶斯统计，统计力学和机器学习。这是具有挑战性的，尤其是在高尺寸或$π$多模式时。为了减轻常规重要性抽样估计量的较大差异，通常采用基于退火的方法，例如Jarzynski平等和退火重要性采样，但是它们的定量复杂性保证在很大程度上尚未得到探索。我们迈出了对退火重要性采样的非反应分析的第一步。特别是，我们得出了$ \ widetilde {o} \ left（\ frac {\ frac {dβ^2 {\ mathcal {\ nathcal {a}}}^2} {\ varepsilon^4} \ right）$在$中，$ q $ pmotify $ pmooth $ quartive $ pmooth $ pmote $ \ MATHCAL {A} $表示概率曲线的操作，可插值$π$和可处理的参考分布。我们的分析利用Girsanov定理和最佳运输，并不明确需要对目标分布的等速化假设。最后，为了解决广泛使用的概率分布的几何插值的大作用，我们提出了一种基于反向扩散采样器的新归一化恒定估计算法，并建立了一个分析其复杂性的框架。

### Gradient-based Explanations for Deep Learning Survival Models 
[[arxiv](https://arxiv.org/abs/2502.04970)] [[cool](https://papers.cool/arxiv/2502.04970)] [[pdf](https://arxiv.org/pdf/2502.04970)]
> **Authors**: Sophie Hanna Langbein,Niklas Koenen,Marvin N. Wright
> **First submission**: 2025-02-07
> **First announcement**: 2025-02-10
> **comment**: No comments
- **标题**: 深度学习生存模型的基于梯度的解释
- **领域**: 机器学习,机器学习
- **摘要**: 深度学习生存模型通常在事实上的预测中，尤其是在个性化医学中的经典方法，但它们的“黑匣子”自然可以阻碍更广泛的采用。我们为基于梯度的解释方法量身定制了针对生存神经网络的框架，从而扩展了它们的使用超出回归和分类。我们分析了其理论假设对生存环境中时间依赖性解释的含义，并提出了结合时间维度的有效可视化。合成数据的实验表明，基于梯度的方法捕获了局部和全局特征效应的大小和方向，包括时间依赖性。我们介绍了基于梯度的gradshap（t），这是一种基于梯度的对应物（t），在计算速度与准确性权衡方面，它的表现优于survhap（t）和survlime。最后，我们将这些方法应用于具有多模式输入的医疗数据，揭示了相关的表格特征和视觉模式及其时间动力学。

### Generative Distribution Prediction: A Unified Approach to Multimodal Learning 
[[arxiv](https://arxiv.org/abs/2502.07090)] [[cool](https://papers.cool/arxiv/2502.07090)] [[pdf](https://arxiv.org/pdf/2502.07090)]
> **Authors**: Xinyu Tian,Xiaotong Shen
> **First submission**: 2025-02-10
> **First announcement**: 2025-02-11
> **comment**: 31 pages 4 figures
- **标题**: 生成分配预测：一种统一的多模式学习方法
- **领域**: 机器学习,人工智能,机器学习
- **摘要**: 具有多模式数据包含表格，文本和视觉输入或输出的准确预测，这是推进不同应用程序域中分析基础的基础。传统方法通常很难整合异质数据类型，同时保持高预测精度。我们引入了生成分布预测（GDP），这是一个新型框架，利用多模式合成数据生成，例如条件扩散模型，以增强结构化和非结构化方式的预测性能。 GDP是​​模型不合时宜的，与任何高保真生成模型兼容，并支持针对域适应的传输学习。我们为GDP建立了严格的理论基础，在使用扩散模型作为生成骨架时，可以根据其预测准确性提供统计保证。通过估计数据生成分布并适应各种损失函数以最小化风险，GDP可以在多模式设置之间进行准确的点预测。我们从经验上验证了GDP，以四个监督的学习任务 - 尾巴数据预测，问答，图像字幕和自适应分位数回归，以证明其在不同领域的多功能性和有效性。

### Generative Adversarial Networks for High-Dimensional Item Factor Analysis: A Deep Adversarial Learning Algorithm 
[[arxiv](https://arxiv.org/abs/2502.10650)] [[cool](https://papers.cool/arxiv/2502.10650)] [[pdf](https://arxiv.org/pdf/2502.10650)]
> **Authors**: Nanyu Luo,Feng Ji
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 用于高维项目因素分析的生成对抗网络：深层对抗性学习算法
- **领域**: 机器学习,机器学习,应用领域,计算,方法论
- **摘要**: 深度学习和表示学习的进步通过实现更有效和准确的参数估计来改变项目响应理论（IRT）文献中的项目因素分析（IFA）。在这种情况下，变异自动编码器（VAE）一直是对高维潜在变量进行建模的最有影响力的技术之一。但是，基于传统VAE的推论模型的表现力有限仍然会阻碍估计性能。我们引入了对抗性变分贝叶斯（AVB）算法，以改善IFA的VAE，具有提高的灵活性和准确性。通过桥接VAE和生成对抗网络（GAN）的优势，AVB结合了辅助歧视者网络，以将估计过程重新构架为两人对抗游戏，并消除推理模型中标准正常分布的限制性假设。从理论上讲，与VAE相比，AVB可以实现相似或更高的可能性。提出了进一步增强的算法，重要性加权的对抗性贝叶斯（IWAVB），并将其与重要性加权自动编码器（IWAE）进行了比较。在对经验数据的探索性分析中，IWAVB通过与IWAE相比具有更高的可能性，证明了较高的表现力。在使用模拟数据的确认性分析中，IWAVB在始终达到更高的可能性的同时，达到了与IWAE相似的均方误差结果。当潜在变量遵循多模式分布时，IWAVB的表现优于IWAE。凭借其创新的gans使用，IWAVB被证明具有扩展IFA来处理大规模数据的潜力，从而促进了心理图和多模式数据分析的潜在整合。

### Weighted quantization using MMD: From mean field to mean shift via gradient flows 
[[arxiv](https://arxiv.org/abs/2502.10600)] [[cool](https://papers.cool/arxiv/2502.10600)] [[pdf](https://arxiv.org/pdf/2502.10600)]
> **Authors**: Ayoub Belhadji,Daniel Sharp,Youssef Marzouk
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 使用MMD加权量化：从平均场到平均偏移通过梯度流动
- **领域**: 机器学习,机器学习,数值分析
- **摘要**: 使用一组粒子近似概率分布是机器学习和统计数据中的一个基本问题，其应用包括聚类和量化。正式地，我们寻求有限的加权混合物的狄拉克度量，以最能近似目标分布。尽管许多现有的工作依赖于瓦斯汀距离来量化近似误差，但最大平均差异（MMD）的关注较少，尤其是在允许可变粒子重量的情况下。我们从Wasserstein-Fisher-Rao（WFR）几何形状中通过梯度流量最小化MMD的角度研究了量化问题。该梯度流得出一个ode系统，我们从中进一步得出了一种称为平均移位相互作用粒子（MSIP）的定点算法。我们表明，MSIP扩展了（非相互作用）平均移位算法，该算法广泛用于识别内核密度估计中的模式。此外，我们表明MSIP可以解释为预处理的梯度下降，并且它可以放松劳埃德的聚类算法。我们的数值实验表明，MSIP和WFR ODE的表现优于量化多模式和高维靶标的其他算法。

### Generalised Parallel Tempering: Flexible Replica Exchange via Flows and Diffusions 
[[arxiv](https://arxiv.org/abs/2502.10328)] [[cool](https://papers.cool/arxiv/2502.10328)] [[pdf](https://arxiv.org/pdf/2502.10328)]
> **Authors**: Leo Zhang,Peter Potaptchik,Arnaud Doucet,Hai-Dang Dau,Saifuddin Syed
> **First submission**: 2025-02-14
> **First announcement**: 2025-02-17
> **comment**: No comments
- **标题**: 广义平行回火：通过流和扩散的灵活复制品交换
- **领域**: 机器学习,机器学习
- **摘要**: 并行回火（PT）是一种经典的MCMC算法，旨在利用并行计算通过退火从高维，多模态或其他复杂分布中有效采样。 PT标准公式的一个局限性是通过有效的样本量或往返速率来衡量产生高质量样本所需的计算资源的增长，以衡量越来越具有挑战性的分布。为了解决这个问题，我们提出了一个框架：广义平行回火（GEPT），该框架允许在平行回火中纳入现代生成建模的最新进展，例如正常的流量和扩散模型，同时保持与基于MCMC的方法相同的理论保证。例如，我们表明，这使我们能够以并行的方式利用扩散模型，绕过大量步骤的通常计算成本来生成质量样本。此外，我们从经验上证明，GEPT可以提高样本质量并减少处理经典算法所需的计算资源的增长。

### Neural Guided Diffusion Bridges 
[[arxiv](https://arxiv.org/abs/2502.11909)] [[cool](https://papers.cool/arxiv/2502.11909)] [[pdf](https://arxiv.org/pdf/2502.11909)]
> **Authors**: Gefan Yang,Frank van der Meulen,Stefan Sommer
> **First submission**: 2025-02-17
> **First announcement**: 2025-02-18
> **comment**: No comments
- **标题**: 神经引导扩散桥
- **领域**: 机器学习,机器学习
- **摘要**: 我们提出了一种新的方法，用于模拟欧几里得空间中的条件扩散过程（扩散桥）。通过训练神经网络以近似桥梁的动力学，我们的方法消除了对计算密集的马尔可夫链蒙特卡洛（MCMC）方法或反处理建模的需求。与现有方法相比，它在各种扩散规格和条件方案中提供了更大的鲁棒性。这特别适用于罕见事件和多模式分布，这对基于得分学习和基于MCMC的方法构成了挑战。我们提出了一个灵活的变分家族，用于近似于神经网络部分指定的扩散桥路径。训练后，它可以以可与无条件（正向）过程相当的成本进行有效的独立采样。

### Enhancing Gradient-based Discrete Sampling via Parallel Tempering 
[[arxiv](https://arxiv.org/abs/2502.19240)] [[cool](https://papers.cool/arxiv/2502.19240)] [[pdf](https://arxiv.org/pdf/2502.19240)]
> **Authors**: Luxu Liang,Yuhang Jia,Feng Zhou
> **First submission**: 2025-02-26
> **First announcement**: 2025-02-27
> **comment**: 24 pages, 5 figures. arXiv admin note: text overlap with arXiv:2402.17699 by other authors
- **标题**: 通过平行回火增强基于梯度的离散抽样
- **领域**: 机器学习,机器学习,应用领域
- **摘要**: 尽管基于梯度的离散采样器可有效地从复杂分布中取样，但由于这些景观中固有的不连续性，它们容易被困在局部最小值中，尤其是在高维，多模式离散分布中。为了解决这个问题，我们将平行回火（也称为副本交换）与离散的langevin提案相结合，并开发出平行的回火增强了离散的Langevin提案（PTDLP），这些提案在一系列温度下进行了模拟。显着的能量差异促使样品掉期受到大都市标准的约束，专门为离散抽样设计，以确保保持详细的平衡。此外，我们引入了一种自动方案，以确定最佳温度时间表和链数，从而确保以最小的调整来确保各种任务的适应性。从理论上讲，我们确定我们的算法与单个链相比，与目标能相比，算法非偶然地收敛，并且表现出更快的混合。经验结果进一步强调了我们的方法在复杂的多模式离散分布（包括合成问题，受限的玻尔兹曼机器和基于深度能量的模型）中的优势。

## 其他论文

共有 71 篇其他论文

- [GO: The Great Outdoors Multimodal Dataset](https://arxiv.org/abs/2501.19274)
  - **标题**: GO：户外大型数据集
  - **Filtered Reason**: none of cs.RO in whitelist
- [MemPal: Leveraging Multimodal AI and LLMs for Voice-Activated Object Retrieval in Homes of Older Adults](https://arxiv.org/abs/2502.01801)
  - **标题**: 备忘录：利用多模式AI和LLMS在老年人家庭中进行语音激活的物体检索
  - **Filtered Reason**: none of cs.HC in whitelist
- [DietGlance: Dietary Monitoring and Personalized Analysis at a Glance with Knowledge-Empowered AI Assistant](https://arxiv.org/abs/2502.01317)
  - **标题**: Dietglance：饮食监测和个性化分析一眼就具有知识授权的AI助手
  - **Filtered Reason**: none of cs.HC,cs.CY in whitelist
- [Towards Robust Multimodal Large Language Models Against Jailbreak Attacks](https://arxiv.org/abs/2502.00653)
  - **标题**: 采取强大的多模式大语模型，以防止越狱攻击
  - **Filtered Reason**: none of cs.CR in whitelist
- [Assessment of ChatGPT for Engineering Statics Analysis](https://arxiv.org/abs/2502.00562)
  - **标题**: 评估工程静态分析的CHATGPT
  - **Filtered Reason**: none of cs.CE in whitelist
- [Enhancing Psychotherapeutic Alliance in College: When and How to Integrate Multimodal Large Language Models in Psychotherapy](https://arxiv.org/abs/2502.00229)
  - **标题**: 增强大学心理治疗联盟：何时以及如何整合心理治疗中的多模式模型
  - **Filtered Reason**: none of cs.HC in whitelist
- [Where Do Passengers Gaze? Impact of Passengers' Personality Traits on Their Gaze Pattern Toward Pedestrians During APMV-Pedestrian Interactions with Diverse eHMIs](https://arxiv.org/abs/2502.02792)
  - **标题**: 乘客在哪里凝视？乘客人格特质对APMV-Pedestrian与多样的EHMIS相互作用期间对行人的目光影响的影响
  - **Filtered Reason**: none of cs.HC in whitelist
- [The Design of On-Body Robots for Older Adults](https://arxiv.org/abs/2502.02725)
  - **标题**: 老年人的体内机器人的设计
  - **Filtered Reason**: none of cs.RO,cs.HC in whitelist
- [From Accidents to Insights: Leveraging Multimodal Data for Scenario-Driven ADS Testing](https://arxiv.org/abs/2502.02025)
  - **标题**: 从事故到见解：利用多模式数据进行方案驱动的广告测试
  - **Filtered Reason**: none of cs.SE in whitelist
- [More Modality, More AI: Exploring Design Opportunities of AI-Based Multi-modal Remote Monitoring Technologies for Early Detection of Mental Health Sequelae in Youth Concussion Patients](https://arxiv.org/abs/2502.03732)
  - **标题**: 更多模式，更多的AI：探索基于AI的多模式远程监测技术的设计机会，以早日检测到青年脑震荡患者的心理健康后遗症
  - **Filtered Reason**: none of cs.HC in whitelist
- [Towards Scalable Defenses against Intimate Partner Infiltrations](https://arxiv.org/abs/2502.03682)
  - **标题**: 采取可扩展的防御措施，以防止亲密伴侣渗透
  - **Filtered Reason**: none of cs.HC,cs.CR in whitelist
- [EnVisionVR: A Scene Interpretation Tool for Visual Accessibility in Virtual Reality](https://arxiv.org/abs/2502.03564)
  - **标题**: EnvisionVR：用于虚拟现实中视觉可访问性的场景解释工具
  - **Filtered Reason**: none of cs.HC in whitelist
- [Designing LLM-simulated Immersive Spaces to Enhance Autistic Children's Social Affordances Understanding](https://arxiv.org/abs/2502.03447)
  - **标题**: 设计LLM模拟的沉浸式空间，以增强自闭症儿童的社交能力理解
  - **Filtered Reason**: none of cs.HC in whitelist
- [Inverse Mixed Strategy Games with Generative Trajectory Models](https://arxiv.org/abs/2502.03356)
  - **标题**: 与生成轨迹模型相反的混合策略游戏
  - **Filtered Reason**: none of cs.RO in whitelist
- [Cognitive AI framework: advances in the simulation of human thought](https://arxiv.org/abs/2502.04259)
  - **标题**: 认知AI框架：模拟人类思想的进步
  - **Filtered Reason**: none of cs.HC,cs.CY in whitelist
- [User-Friendly Game-Theoretic Modeling and Analysis of Multi-Modal Transportation Systems](https://arxiv.org/abs/2502.04155)
  - **标题**: 用户友好的游戏理论建模和多模式运输系统的分析
  - **Filtered Reason**: none of math.OC,cs.CY in whitelist
- [Enhancing Deliberativeness: Evaluating the Impact of Multimodal Reflection Nudges](https://arxiv.org/abs/2502.03862)
  - **标题**: 增强故意性：评估多模式反射的影响
  - **Filtered Reason**: none of cs.HC in whitelist
- [Modeling and Beamforming Optimization for Pinching-Antenna Systems](https://arxiv.org/abs/2502.05917)
  - **标题**: 捏合 - 安特纳系统的建模和波束形成优化
  - **Filtered Reason**: none of cs.IT in whitelist
- [EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks](https://arxiv.org/abs/2502.05907)
  - **标题**: Evoagent：代理自动进化，具有连续的世界模型，用于长途任务
  - **Filtered Reason**: none of cs.RO in whitelist
- [StreamDCIM: A Tile-based Streaming Digital CIM Accelerator with Mixed-stationary Cross-forwarding Dataflow for Multimodal Transformer](https://arxiv.org/abs/2502.05798)
  - **标题**: StreamDCIM：一个基于瓷砖的流数字CIM加速器，具有多模式变压器的混合式跨前方数据流
  - **Filtered Reason**: none of cs.AR in whitelist
- [Audio-Visual Representation Learning via Knowledge Distillation from Speech Foundation Models](https://arxiv.org/abs/2502.05766)
  - **标题**: 通过语音基础模型通过知识蒸馏学习的视听表示学习
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [IllusionCAPTCHA: A CAPTCHA based on Visual Illusion](https://arxiv.org/abs/2502.05461)
  - **标题**: 幻觉验证码：基于视觉幻觉的验证码
  - **Filtered Reason**: none of cs.CR in whitelist
- [Usability Issues With Mobile Applications: Insights From Practitioners and Future Research Directions](https://arxiv.org/abs/2502.05120)
  - **标题**: 移动应用程序的可用性问题：从业者和未来的研究方向的见解
  - **Filtered Reason**: none of cs.HC in whitelist
- [REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly](https://arxiv.org/abs/2502.05086)
  - **标题**: 重新组装：用于接触富含机器人组装和拆卸的多模式数据集
  - **Filtered Reason**: none of cs.RO in whitelist
- [Towards Multimodal Empathetic Response Generation: A Rich Text-Speech-Vision Avatar-based Benchmark](https://arxiv.org/abs/2502.04976)
  - **标题**: 迈向多模式的同理响应生成：基于丰富的文本语音视觉化头像基准
  - **Filtered Reason**: none of cs.MM in whitelist
- [Mobile Network-specialized Large Language Models for 6G: Architectures, Innovations, Challenges, and Future Trends](https://arxiv.org/abs/2502.04933)
  - **标题**: 6G移动网络专业的大语言模型：建筑，创新，挑战和未来趋势
  - **Filtered Reason**: none of cs.NI in whitelist
- [Multimodal Search on a Line](https://arxiv.org/abs/2502.07000)
  - **标题**: 一行多模式搜索
  - **Filtered Reason**: none of cs.DM in whitelist
- [Actual Achieved Gain and Optimal Perceived Gain: Modeling Human Take-over Decisions Towards Automated Vehicles' Suggestions](https://arxiv.org/abs/2502.06179)
  - **标题**: 实际实现的收益和最佳感知收益：对人类对自动化车辆建议的决策进行建模
  - **Filtered Reason**: none of cs.HC in whitelist
- [DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove](https://arxiv.org/abs/2502.07730)
  - **标题**: Doglove：具有低成本开源触觉反馈手套的灵巧操作
  - **Filtered Reason**: none of cs.RO in whitelist
- [Towards spatial computing: recent advances in multimodal natural interaction for XR headsets](https://arxiv.org/abs/2502.07598)
  - **标题**: 迈向空间计算：XR耳机的多模式自然相互作用的最新进展
  - **Filtered Reason**: none of cs.HC in whitelist
- [FixDrive: Automatically Repairing Autonomous Vehicle Driving Behaviour for $0.08 per Violation](https://arxiv.org/abs/2502.08260)
  - **标题**: FIXDRIVE：自动修理自动驾驶驾驶行为，每次违规$ 0.08
  - **Filtered Reason**: none of cs.SE in whitelist
- [A Deep Learning Approach to Interface Color Quality Assessment in HCI](https://arxiv.org/abs/2502.09914)
  - **标题**: HCI中界面颜色质量评估的深度学习方法
  - **Filtered Reason**: none of cs.HC in whitelist
- [FontCraft: Multimodal Font Design Using Interactive Bayesian Optimization](https://arxiv.org/abs/2502.11399)
  - **标题**: 字体：使用交互式贝叶斯优化的多模式字体设计
  - **Filtered Reason**: none of cs.HC in whitelist
- [Prevalence, Sharing Patterns, and Spreaders of Multimodal AI-Generated Content on X during the 2024 U.S. Presidential Election](https://arxiv.org/abs/2502.11248)
  - **标题**: 在2024年美国总统大选期间，X上X上的多模式AI生成内容的流行，共享模式和播种者
  - **Filtered Reason**: none of cs.CY,cs.SI in whitelist
- [AudioSpa: Spatializing Sound Events with Text](https://arxiv.org/abs/2502.11219)
  - **标题**: Audiospa：带有文字的空间声音事件
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [Large Model Empowered Metaverse: State-of-the-Art, Challenges and Opportunities](https://arxiv.org/abs/2502.10397)
  - **标题**: 大型模型授权元元：最新，挑战和机遇
  - **Filtered Reason**: none of cs.CY in whitelist
- [CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](https://arxiv.org/abs/2502.10362)
  - **标题**: 夹具3：跨未对齐方式和看不见的语言的通用音乐信息检索
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [RoadFed: A Multimodal Federated Learning System for Improving Road Safety](https://arxiv.org/abs/2502.09978)
  - **标题**: ROADFED：一种用于改善道路安全的多模式联合学习系统
  - **Filtered Reason**: none of cs.CE in whitelist
- [AnimAlte:Designing AI-Infused Cartoon Videos to Improve Preschoolers' Language Learning with Family Engagement at Home](https://arxiv.org/abs/2502.12526)
  - **标题**: Animalte：设计注入AI的卡通视频，以在家中的家庭参与度提高学龄前儿童的语言学习
  - **Filtered Reason**: none of cs.HC in whitelist
- [Learning in a Multifield Coherent Ising Machine](https://arxiv.org/abs/2502.12020)
  - **标题**: 在多场连贯的ISING机器中学习
  - **Filtered Reason**: none of cond-mat.mes-hall,nlin.AO,cs.ET,cond-mat.dis-nn,cs.NE in whitelist
- [Robotic CBCT Meets Robotic Ultrasound](https://arxiv.org/abs/2502.12019)
  - **标题**: 机器人CBCT符合机器人超声
  - **Filtered Reason**: none of cs.RO,eess.IV in whitelist
- [Assessing the impacts of tradable credit schemes through agent-based simulation](https://arxiv.org/abs/2502.11822)
  - **标题**: 通过基于代理的模拟评估可交易信贷计划的影响
  - **Filtered Reason**: none of cs.GT,stat.ML,cs.SE in whitelist
- [Early Detection of Human Handover Intentions in Human-Robot Collaboration: Comparing EEG, Gaze, and Hand Motion](https://arxiv.org/abs/2502.11752)
  - **标题**: 人类机器人协作中人类移交意图的早期发现：比较脑电图，注视和手动运动
  - **Filtered Reason**: none of cs.RO,cs.HC in whitelist
- [Virtual Encounters of the Haptic Kind: Towards a Multi-User VR System for Real-Time Social Touch](https://arxiv.org/abs/2502.13421)
  - **标题**: 触觉类型的虚拟相遇：迈向实时社交触摸的多用户VR系统
  - **Filtered Reason**: none of cs.HC,cs.ET in whitelist
- [An Attention-Assisted Multi-Modal Data Fusion Model for Real-Time Estimation of Underwater Sound Velocity](https://arxiv.org/abs/2502.12817)
  - **标题**: 注意力辅助多模式数据融合模型，用于实时估计水下声音速度
  - **Filtered Reason**: none of cs.SD,eess.SP in whitelist
- [Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control](https://arxiv.org/abs/2502.12724)
  - **标题**: 响应噪声避免的扩散政策：响应迅速，有效的视觉控制
  - **Filtered Reason**: none of cs.RO in whitelist
- [REFLEX Dataset: A Multimodal Dataset of Human Reactions to Robot Failures and Explanations](https://arxiv.org/abs/2502.14185)
  - **标题**: 反射数据集：人类对机器人失败和解释的多模式数据集
  - **Filtered Reason**: none of cs.RO in whitelist
- ["It Brought the Model to Life": Exploring the Embodiment of Multimodal I3Ms for People who are Blind or have Low Vision](https://arxiv.org/abs/2502.14163)
  - **标题**: “它使模型栩栩如生”：探索盲人或视力低下的人的多模式I3ms的体现
  - **Filtered Reason**: none of cs.HC in whitelist
- [ArtMentor: AI-Assisted Evaluation of Artworks to Explore Multimodal Large Language Models Capabilities](https://arxiv.org/abs/2502.13832)
  - **标题**: Artmentor：AI辅助评估艺术品以探索多模式模型功能
  - **Filtered Reason**: none of cs.HC in whitelist
- [Cascading CMA-ES Instances for Generating Input-diverse Solution Batches](https://arxiv.org/abs/2502.13730)
  - **标题**: 级联的CMA-ES实例生成输入 - 多样性解决方案批处理
  - **Filtered Reason**: none of cs.NE in whitelist
- [VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation](https://arxiv.org/abs/2502.13508)
  - **标题**: VLAS：带有语音说明的视觉语言动作模型，用于定制机器人操纵
  - **Filtered Reason**: none of cs.RO in whitelist
- [An Enhancement of Cuckoo Search Algorithm for Optimal Earthquake Evacuation Space Allocation in Intramuros, Manila City](https://arxiv.org/abs/2502.13477)
  - **标题**: 马尼拉市Intramuros的最佳地震疏散空间分配的杜鹃搜索算法的增强
  - **Filtered Reason**: none of cs.NE in whitelist
- [MATS: An Audio Language Model under Text-only Supervision](https://arxiv.org/abs/2502.13433)
  - **标题**: MATS：仅在文本监督下的音频语言模型
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [FIP: Endowing Robust Motion Capture on Daily Garment by Fusing Flex and Inertial Sensors](https://arxiv.org/abs/2502.15058)
  - **标题**: FIP：通过融合弹性和惯性传感器来赋予每日服装的强大运动捕获
  - **Filtered Reason**: none of cs.HC in whitelist
- [DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories](https://arxiv.org/abs/2502.15043)
  - **标题**: DDAT：实施动态可接受的机器人轨迹的扩散政策
  - **Filtered Reason**: none of eess.SY,cs.RO in whitelist
- [Watch Out E-scooter Coming Through: Multimodal Sensing of Mixed Traffic Use and Conflicts Through Riders Ego-centric Views](https://arxiv.org/abs/2502.16755)
  - **标题**: 当心通过电子示波器进行：混合交通使用的多模式感应和通过骑手以自我为中心的视图
  - **Filtered Reason**: none of cs.CY in whitelist
- [M4SC: An MLLM-based Multi-modal, Multi-task and Multi-user Semantic Communication System](https://arxiv.org/abs/2502.16418)
  - **标题**: M4SC：基于MLLM的多模式，多任务和多用户语义通信系统
  - **Filtered Reason**: none of cs.IT in whitelist
- [Beyond Visual Perception: Insights from Smartphone Interaction of Visually Impaired Users with Large Multimodal Models](https://arxiv.org/abs/2502.16098)
  - **标题**: 除了视觉感知之外：智能手机互动的见解与视觉障碍用户与大型多模型的互动
  - **Filtered Reason**: none of cs.HC in whitelist
- [Community Detection in Multimodal Data: A Similarity Network Perspective](https://arxiv.org/abs/2502.15993)
  - **标题**: 多模式数据中的社区检测：相似性网络观点
  - **Filtered Reason**: none of cs.SI in whitelist
- [BSODiag: A Global Diagnosis Framework for Batch Servers Outage in Large-scale Cloud Infrastructure Systems](https://arxiv.org/abs/2502.15728)
  - **标题**: BSODIAG：大规模云基础设施系统中批处服务器中断的全球诊断框架
  - **Filtered Reason**: none of cs.DC in whitelist
- [FaultGPT: Industrial Fault Diagnosis Question Answering System by Vision Language Models](https://arxiv.org/abs/2502.15481)
  - **标题**: 故障：工业故障诊断问题答案系统通过视觉语言模型
  - **Filtered Reason**: none of eess.SP,cs.ET in whitelist
- [Multimodal Graph-Based Variational Mixture of Experts Network for Zero-Shot Multimodal Information Extraction](https://arxiv.org/abs/2502.15290)
  - **标题**: 专家网络的基于多模式的基于图的变分混合物，用于零击多模式信息提取
  - **Filtered Reason**: none of cs.MM in whitelist
- [CPVis: Evidence-based Multimodal Learning Analytics for Evaluation in Collaborative Programming](https://arxiv.org/abs/2502.17835)
  - **标题**: CPVIS：基于证据的多模式学习分析，用于协作编程中的评估
  - **Filtered Reason**: none of cs.HC in whitelist
- [Multi-modal and Metadata Capture Model for Micro Video Popularity Prediction](https://arxiv.org/abs/2502.17038)
  - **标题**: 微型视频流行度预测的多模式和元数据捕获模型
  - **Filtered Reason**: none of cs.MM in whitelist
- [CommGPT: A Graph and Retrieval-Augmented Multimodal Communication Foundation Model](https://arxiv.org/abs/2502.18763)
  - **标题**: Commgpt：图形和检索仪的多模式通信基础模型
  - **Filtered Reason**: none of cs.IT in whitelist
- [MulChain: Enabling Advanced Cross-Modal Queries in Hybrid-Storage Blockchains](https://arxiv.org/abs/2502.18258)
  - **标题**: Mulchain：在混合存储区块链中启用高级跨模式查询
  - **Filtered Reason**: none of cs.DB,cs.SE in whitelist
- [Multimodal Interaction and Intention Communication for Industrial Robots](https://arxiv.org/abs/2502.17971)
  - **标题**: 工业机器人的多模式互动和意图交流
  - **Filtered Reason**: none of cs.RO,cs.HC in whitelist
- [Sequential Exchange Monte Carlo: Sampling Method for Multimodal Distribution without Parameter Tuning](https://arxiv.org/abs/2502.17858)
  - **标题**: 顺序交换蒙特卡洛：无参数调整的多模式分布的采样方法
  - **Filtered Reason**: none of cs.IT in whitelist
- [On Adversarial Attacks In Acoustic Drone Localization](https://arxiv.org/abs/2502.20325)
  - **标题**: 关于声学无人机本地化的对抗攻击
  - **Filtered Reason**: none of cs.RO,cs.SD,eess.AS in whitelist
- [Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps](https://arxiv.org/abs/2502.20054)
  - **标题**: 夜行者：对象图中的一致，有效的夜间视觉效果估计
  - **Filtered Reason**: none of eess.SY,cs.RO in whitelist
- [Towards Multimodal Large-Language Models for Parent-Child Interaction: A Focus on Joint Attention](https://arxiv.org/abs/2502.19877)
  - **标题**: 迈向多模式的大型互动模型：关注联合注意力
  - **Filtered Reason**: none of cs.HC in whitelist
