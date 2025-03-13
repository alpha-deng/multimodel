# 2021-01 月度论文分类汇总

共有80篇相关领域论文, 另有18篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 8 篇论文

### KM-BART: Knowledge Enhanced Multimodal BART for Visual Commonsense Generation 
[[arxiv](https://arxiv.org/abs/2101.00419)] [[cool](https://papers.cool/arxiv/2101.00419)] [[pdf](https://arxiv.org/pdf/2101.00419)]
> **Authors**: Yiran Xing,Zai Shi,Zhao Meng,Gerhard Lakemeyer,Yunpu Ma,Roger Wattenhofer
> **First submission**: 2021-01-02
> **First announcement**: 2021-01-04
> **comment**: ACL-IJCNLP 2021 main conference. The first three authors contribute equally to this work
- **标题**: KM-BART：知识增强了视觉常识生成的多模式BART
- **领域**: 计算语言学
- **摘要**: 我们提出了知识增强的多模式巴特（KM-BART），这是一个基于变压器的序列到序列模型，能够从图像和文本的多模式输入中推理常识性知识。我们将生成性的BART体系结构调整为具有视觉和文本输入的多模式模型。我们进一步开发了新颖的预读任务，以提高视觉常识生成（VCG）任务的模型性能。特别是，我们对基于知识的常识生成（KCG）的训练任务通过利用在外部常识性知识图上预先介绍的大语言模型中利用常识性知识来提高VCG任务的模型性能。据我们所知，我们是第一个提出一项专门的任务，以改善VCG任务上的模型性能。实验结果表明，我们的模型通过应用这些新颖的预处理任务来达到VCG任务的最新性能。

### Latent Alignment of Procedural Concepts in Multimodal Recipes 
[[arxiv](https://arxiv.org/abs/2101.04727)] [[cool](https://papers.cool/arxiv/2101.04727)] [[pdf](https://arxiv.org/pdf/2101.04727)]
> **Authors**: Hossein Rajaby Faghihi,Roshanak Mirzaee,Sudarshan Paliwal,Parisa Kordjamshidi
> **First submission**: 2021-01-12
> **First announcement**: 2021-01-13
> **comment**: Published in ALVR 2020, a workshop in ACL 2020
- **标题**: 多模式食谱中程序概念的潜在对齐
- **领域**: 计算语言学,人工智能
- **摘要**: 我们提出了一种新型的对齐机制，以在新发布的多模式质量模式数据集（名为repipeqa）上处理程序推理。我们的模型正在解决文本披肩任务，这是包含图像和说明的食谱上的阅读理解。我们利用注意力网络，跨模式表示的力量以及指令和候选人答案之间的潜在对齐空间以解决问题。我们介绍了受限的最大流动，从而优化了对齐矩阵上的最大流动操作，以在模型的输出之间施加不相交的约束。我们的评估结果表明基准的改善19 \％。

### Narration Generation for Cartoon Videos 
[[arxiv](https://arxiv.org/abs/2101.06803)] [[cool](https://papers.cool/arxiv/2101.06803)] [[pdf](https://arxiv.org/pdf/2101.06803)]
> **Authors**: Nikos Papasarantopoulos,Shay B. Cohen
> **First submission**: 2021-01-17
> **First announcement**: 2021-01-18
> **comment**: No comments
- **标题**: 卡通视频的叙述产生
- **领域**: 计算语言学
- **摘要**: 多模式输入的文本生成的研究主要集中在静态图像上，而较少于视频数据。在本文中，我们提出了一项新的任务，即叙述一代，它正在用叙事文本进行补充，这些视频应在多个地方插入。叙述是视频的一部分，有助于故事情节的发展。此外，它们是上下文信息的，因为它们包含适用于它们涵盖的视频时间范围的信息，而且不需要像标题那样包含输入场景中显示的所有细节。我们从动画电视连续剧Peppa Pig收集了一个新数据集。此外，我们将叙述生成的任务形式化为包括两个独立的任务，即时机和内容生成，并在新任务上介绍了一组模型。

### Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning 
[[arxiv](https://arxiv.org/abs/2101.07393)] [[cool](https://papers.cool/arxiv/2101.07393)] [[pdf](https://arxiv.org/pdf/2101.07393)]
> **Authors**: Austin W. Hanjie,Victor Zhong,Karthik Narasimhan
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: Accepted to ICML 2021. Note author list and name changes from previous version
- **标题**: 将语言与实体和动力进行加强学习的动力
- **领域**: 计算语言学,人工智能,机器学习
- **摘要**: 我们研究了自然语言来推动控制策略的概括，并使用描述环境动态的自由格式文本手册介绍了新的多任务环境使者。与以前的工作不同，Messenger不假定将文本和状态观测值连接的先验知识$  -  $  -  $控制策略必须同时将游戏手册与环境中的实体符号和动态联系起来。我们开发了一个新的模型，艾玛（Emma）（具有多模式注意的实体映射器），该模型使用实体条件的注意模块，该模块可以选择性地关注环境中每个实体的相关描述。艾玛（Emma）是端到端的可区分，并了解了仅使用环境奖励的实体和动态的潜在基础。艾玛（Emma）取得了成功的零球概括，以新的动力学效果，与多个基线相比，获胜率提高了40％。但是，在Messenger最困难的阶段的获胜率仍然很低（10％），这表明需要在这个方向上进行其他工作。

### MONAH: Multi-Modal Narratives for Humans to analyze conversations 
[[arxiv](https://arxiv.org/abs/2101.07339)] [[cool](https://papers.cool/arxiv/2101.07339)] [[pdf](https://arxiv.org/pdf/2101.07339)]
> **Authors**: Joshua Y. Kim,Greyson Y. Kim,Chunfeng Liu,Rafael A. Calvo,Silas C. R. Taylor,Kalina Yacef
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: 14 pages
- **标题**: 莫纳：人类分析对话的多模式叙事
- **领域**: 计算语言学
- **摘要**: 在对话分析中，人类将多模式信息手动编织到笔录中，这非常耗时。我们介绍了一个系统，该系统会自动扩展使用多模式数据流的视频录制对话的逐字记录。该系统使用一组预处理规则将多模式注释编织到逐字记录并促进解释性。我们的功能工程贡献是两个方面：首先，我们确定与检测融洽关系建设相关的多模式功能的范围；其次，我们扩展了多模式注释的范围，并表明扩展会导致检测融洽关系建设的统计显着改善。

### Towards Natural Language Question Answering over Earth Observation Linked Data using Attention-based Neural Machine Translation 
[[arxiv](https://arxiv.org/abs/2101.09427)] [[cool](https://papers.cool/arxiv/2101.09427)] [[pdf](https://arxiv.org/pdf/2101.09427)]
> **Authors**: Abhishek V. Potnis,Rajat C. Shinde,Surya S. Durbha
> **First submission**: 2021-01-23
> **First announcement**: 2021-01-25
> **comment**: Accepted at IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2020
- **标题**: 使用基于注意力的神经机器翻译来回答地球观测的自然语言问题
- **领域**: 计算语言学,人工智能
- **摘要**: 随着地理空间链接的开放数据的增加和通过网络发布，有必要开发直观的接口和系统，以对如此丰富的异质多模式数据集进行无缝有效的探索性分析。这项工作旨在通过开发自然语言界面以促进查询来改善地球观察（EO）链接数据的探索过程。通过地球观察链接的数据提出的问题具有固有的时空维度，可以使用GEOSPARQL表示。本文旨在研究和分析基于RNN的神经机器翻译的使用，并注意将自然语言问题转化为GeoSparql查询。具体而言，它旨在评估神经方法在自然语言中识别和映射空间谓词的可行性，以识别Geosparql的拓扑词汇扩展，包括 -  egenhofer和RCC8关系。然后可以通过三重商店执行查询，以产生自然语言问题的答案。由从自然语言问题到Corine Land Cover（CLC）链接数据的Geosparql查询的映射的数据集是为了训练和验证深神经网络的。从我们的实验中，很明显，关注的神经机器翻译是将自然语言问题中的空间谓词转换为GeoSparql查询的一种有前途的方法。

### PAWLS: PDF Annotation With Labels and Structure 
[[arxiv](https://arxiv.org/abs/2101.10281)] [[cool](https://papers.cool/arxiv/2101.10281)] [[pdf](https://arxiv.org/pdf/2101.10281)]
> **Authors**: Mark Neumann,Zejiang Shen,Sam Skjonsberg
> **First submission**: 2021-01-25
> **First announcement**: 2021-01-26
> **comment**: No comments
- **标题**: 爪子：带标签和结构的PDF注释
- **领域**: 计算语言学
- **摘要**: Adobe的便携式文档格式（PDF）是一种使用丰富的视觉标记分发仅视图文档的流行方式。这对希望使用PDF文档中包含的信息进行培训模型或数据分析的NLP从业人员提出了挑战，因为对这些文档进行注释很困难。在本文中，我们使用标签和结构（PAWLS）介绍PDF注释，这是一种专门为PDF文档格式设计的新注释工具。爪子特别适合混合模式注释和场景，其中注释者需要扩展上下文才能准确注释。 PAWLS支持基于跨度的文本注释，n- ary关系和自由形式，非文本边界框，所有这些都可以以方便的格式导出，以培训多模式机器学习模型。只需阅读的爪网服务器，请访问https://pawls.apps.allenai.org/，源代码可在https://github.com/allenai/pawls上找到。

### Cross-lingual Visual Pre-training for Multimodal Machine Translation 
[[arxiv](https://arxiv.org/abs/2101.10044)] [[cool](https://papers.cool/arxiv/2101.10044)] [[pdf](https://arxiv.org/pdf/2101.10044)]
> **Authors**: Ozan Caglayan,Menekse Kuyu,Mustafa Sercan Amac,Pranava Madhyastha,Erkut Erdem,Aykut Erdem,Lucia Specia
> **First submission**: 2021-01-25
> **First announcement**: 2021-01-26
> **comment**: Accepted to EACL 2021 (Camera-ready version)
- **标题**: 多模式翻译的跨语性视觉预训练
- **领域**: 计算语言学,计算机视觉和模式识别
- **摘要**: 预训练的语言模型已被证明可以大大提高许多自然语言任务的性能。尽管这种模型的早期重点是单语言预训练，但最近的进步导致了跨语言和视觉预训练方法。在本文中，我们将这两种方法结合在一起，以学习视觉上的跨语性表示。具体而言，我们通过掩盖区域分类扩展了翻译语言建模（Lample and Conneau，2019年），并通过三向平行视觉和语言语料库进行预训练。我们表明，当对多模式机器翻译进行微调时，这些模型会获得最新的性能。我们还提供了有关学到的基础表示的有用性的定性见解。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 40 篇论文

### Depth as Attention for Face Representation Learning 
[[arxiv](https://arxiv.org/abs/2101.00652)] [[cool](https://papers.cool/arxiv/2101.00652)] [[pdf](https://arxiv.org/pdf/2101.00652)]
> **Authors**: Hardik Uppal,Alireza Sepas-Moghaddam,Michael Greenspan,Ali Etemad
> **First submission**: 2021-01-03
> **First announcement**: 2021-01-04
> **comment**: 16 pages, 11 figures, Accepted to IEEE Transactions on Information Forensics and Security 2021
- **标题**: 深度作为面部表示学习的关注
- **领域**: 计算机视觉和模式识别
- **摘要**: 面部表示学习解决方案最近在诸如验证和识别等各种应用方面取得了巨大的成功。但是，纯粹基于RGB图像的面部识别方法仅依赖于强度信息，因此对面部变化更敏感，尤其是姿势，遮挡和环境变化，例如照明和背景。提出了一种新型的深度引导的注意机制，用于使用低成本RGB-D传感器进行深层多模式面部识别。我们的新型注意力机制通过使用卷积神经网络（CNN）提取的深度特征将网络的注意力集中在RGB图像中的深层网络中。深度特征有助于网络关注RGB图像中包含更为突出人士的信息的面部区域。然后，我们的注意机制利用这种相关性从CNN提取的深度特征生成了RGB图像的注意力图。我们在四个公共数据集上测试我们的网络，表明我们提出的解决方案获得的功能在Lock3Dface，Curtinfaces，IIIT-D RGB-D和Kasparov数据集上产生更好的结果，其中包括姿势，闭合，照明，照明，表达和时间段的挑战性变化。我们的解决方案达到了87.3 \％（+5.0 \％），99.1 \％（+0.9 \％），99.7 \％（+0.6 \％）和95.3 \％（+0.5 \％）的平均精度（+5.0 \％）的平均精度分别为四个数据集，从而改善了整体状态。我们还使用热图像进行了其他实验，而不是深度图像，显示了解决方案的高概括能力，用于指导注意机制而不是深度信息。

### A Multi-modal Deep Learning Model for Video Thumbnail Selection 
[[arxiv](https://arxiv.org/abs/2101.00073)] [[cool](https://papers.cool/arxiv/2101.00073)] [[pdf](https://arxiv.org/pdf/2101.00073)]
> **Authors**: Zhifeng Yu,Nanchun Shi
> **First submission**: 2020-12-31
> **First announcement**: 2021-01-04
> **comment**: No comments
- **标题**: 视频缩略图选择的多模式深度学习模型
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 缩略图是在线视频的面孔。数量和多样性视频的爆炸性增长构成了良好的缩略图的重要性，因为它可以节省潜在的观众时间选择视频，甚至吸引他们点击它们。良好的缩略图应该是最能代表视频内容的框架，同时引起观众的注意。但是，过去的技术和模型仅关注视频中的框架，我们认为如此狭窄的焦点丢弃了许多有用的信息，这些信息是视频的一部分。在本文中，我们将内容的定义扩展到包括视频的标题，描述和音频，并利用我们选择模型中这些模式提供的信息。具体而言，我们的模型将首先均匀地示例帧，并通过双柱卷积神经网络以最高的美学得分返回该子集中的前1000个帧，以避免处理下游任务中所有帧的计算负担。然后，该模型结合了从VGG16提取的框架功能，Electra的文本功能以及Trill的音频功能。这些模型之所以被选择，是因为它们在流行的数据集上的结果及其竞争性能。在特征提取后，时间序列特征，帧和音频将被馈入变压器编码层中，以返回代表其相应模态的向量。四个功能（帧，标题，描述，音频）中的每一个都将在串联之前通过上下文门控层。最后，我们的模型将在潜在空间中生成一个向量，并选择与潜在空间中该向量最相似的帧。据我们所知，我们是第一个提出一个多模式深度学习模型来选择视频缩略图的人，该模型超过了以前最新模型的结果。

### Transformers in Vision: A Survey 
[[arxiv](https://arxiv.org/abs/2101.01169)] [[cool](https://papers.cool/arxiv/2101.01169)] [[pdf](https://arxiv.org/pdf/2101.01169)]
> **Authors**: Salman Khan,Muzammal Naseer,Munawar Hayat,Syed Waqas Zamir,Fahad Shahbaz Khan,Mubarak Shah
> **First submission**: 2021-01-04
> **First announcement**: 2021-01-05
> **comment**: 30 pages (Accepted in ACM Computing Surveys December 2021)
- **标题**: 视觉中的变压器：调查
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 自然语言任务上的变压器模型令人惊讶的结果吸引了视觉社区研究其在计算机视觉问题上的应用。在它们的显着益处中，变压器可以在输入序列元素和支持序列的并行处理之间与复发网络（例如长期短期内存（LSTM））进行建模。与卷积网络不同，变压器需要最小的电感偏差来设计其设计，并且自然适合于设定功能。此外，变形金刚的直接设计允许使用类似的处理块处理多种方式（例如，图像，视频，文本和语音），并为非常大的容量网络和庞大的数据集展示了出色的可扩展性。这些优势使使用变压器网络的许多视觉任务取得了令人兴奋的进步。该调查旨在为计算机视觉学科中的变压器模型提供全面的概述。我们首先介绍了变压器成功的基本概念，即自我注意力，大规模的预训练和双向编码。然后，我们涵盖了视觉中变形金刚的广泛应用，包括流行的识别任务（例如，图像分类，对象检测，动作识别和细分），生成模型，多模式任务（例如，视觉问题，视觉响应，视觉推理和视觉接地），视频处理（例如，视频识别，视频识别，视频预测），图像效果（E.着色）和3D分析（例如，点云分类和分割）。我们比较了在建筑设计及其实验价值方面的流行技术的各个优势和局限性。最后，我们对开放研究方向和可能的未来工作进行了分析。

### Deep Class-Specific Affinity-Guided Convolutional Network for Multimodal Unpaired Image Segmentation 
[[arxiv](https://arxiv.org/abs/2101.01513)] [[cool](https://papers.cool/arxiv/2101.01513)] [[pdf](https://arxiv.org/pdf/2101.01513)]
> **Authors**: Jingkun Chen,Wenqi Li,Hongwei Li,Jianguo Zhang
> **First submission**: 2021-01-05
> **First announcement**: 2021-01-06
> **comment**: No comments
- **标题**: 多模式未配对的图像分段的深层特异性亲和力引导的卷积网络
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 多模式医学图像分割在临床诊断中起着至关重要的作用。它仍然具有挑战性，因为输入方式通常在空间上不符合。现有的基于学习的方法主要考虑跨模式共享可训练的层，并最大程度地减少视觉特征差异。虽然问题通常作为联合监督功能学习表达，但尚未探索多个规模的特征和特定于班级的表示。在本文中，我们提出了一个亲和力引导的全卷积网络，以进行多模式图像分割。为了学习有效的表示形式，我们设计了特定类的亲和力矩阵，以编码层次特征推理的知识，以及共享的卷积层，以确保交叉模式概括。我们的亲和力矩阵不取决于视觉特征的空间比对，因此使我们能够使用未配对的多模式输入进行训练。我们在两个公共多模式基准数据集和跑赢大盘方法上广泛评估了我们的方法。

### MSD: Saliency-aware Knowledge Distillation for Multimodal Understanding 
[[arxiv](https://arxiv.org/abs/2101.01881)] [[cool](https://papers.cool/arxiv/2101.01881)] [[pdf](https://arxiv.org/pdf/2101.01881)]
> **Authors**: Woojeong Jin,Maziar Sanjabi,Shaoliang Nie,Liang Tan,Xiang Ren,Hamed Firooz
> **First submission**: 2021-01-06
> **First announcement**: 2021-01-07
> **comment**: Accepted to EMNLP 2021 Findings
- **标题**: MSD：显着意识的知识蒸馏，用于多模式理解
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 为了降低模型大小但保留性能，我们通常依靠知识蒸馏（KD），将知识从大型“老师”模型转移到较小的“学生”模型。但是，在多模式数据集（例如视觉语言任务）上的KD相对尚未探索，并且消化多模式信息是具有挑战性的，因为不同的模式呈现出不同类型的信息。在本文中，我们进行了一项大规模的经验研究，以研究每种方式在知识蒸馏中的重要性和影响。此外，我们引入了多模式知识蒸馏框架，特定于模式的蒸馏（MSD），以通过在每种方式中学习教师的行为来从教师那里转移知识。这个想法旨在通过引入每种方式引入辅助损失条款来模仿教师的模式特定的预测。此外，由于每种方式对预测都有不同的显着性，因此我们定义了每种方式的显着性得分，并研究了辅助损失的基于显着性的加权方案。我们进一步研究一种体重学习方法，以学习这些损失条款的最佳权重。在我们的经验分析中，我们研究了KD中每种模式的显着性，证明了MSD中加权方案的有效性，并表明它在四个多模式数据集中的性能比KD更好。

### On-Device Document Classification using multimodal features 
[[arxiv](https://arxiv.org/abs/2101.01880)] [[cool](https://papers.cool/arxiv/2101.01880)] [[pdf](https://arxiv.org/pdf/2101.01880)]
> **Authors**: Sugam Garg,Harichandana,Sumit Kumar
> **First submission**: 2021-01-06
> **First announcement**: 2021-01-07
> **comment**: 8th ACM IKDD CODS and 26th COMAD 2-4 January 2021
- **标题**: 使用多模式功能的设备文档分类
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 从小型屏幕截图到大型视频，文档在现代智能手机中占用了大部分空间。手机中的文档可以从各种来源积累，并且随着手机的高存储能力，在短时间内积累了数百个文档。但是，搜索或管理文档仍然是一项繁重的任务，因为大多数搜索方法取决于元信息或文档中的文本。在本文中，我们展示了一种单一模式不足以分类，并提出了一条新颖的管道来对文档进行分类，从而阻止了任何私人用户数据传输到服务器。对于此任务，我们集成了一个用于光学特征识别（OCR）的开源库和管道中的新型模型体系结构。我们优化了大小的模型，这是设备推理的必要指标。我们使用标准的多模式数据集食物101基准分类模型，并以先前的最新状态展示竞争结果，并具有30％的模型压缩。

### Multi-Stage Residual Hiding for Image-into-Audio Steganography 
[[arxiv](https://arxiv.org/abs/2101.01872)] [[cool](https://papers.cool/arxiv/2101.01872)] [[pdf](https://arxiv.org/pdf/2101.01872)]
> **Authors**: Wenxue Cui,Shaohui Liu,Feng Jiang,Yongliang Liu,Debin Zhao
> **First submission**: 2021-01-06
> **First announcement**: 2021-01-07
> **comment**: ICASSP 2020
- **标题**: 多个阶段的残留藏图图像into-audio隐志
- **领域**: 计算机视觉和模式识别,密码学和安全
- **摘要**: 音频通信技术的广泛应用已加快了整个Internet上流动的音频数据，这使其成为秘密通信的流行运营商。在本文中，我们提出了一种将图像内容隐藏在音频载体中的跨模式隐肌，同时保留了盖音频的感知忠诚度。在我们的框架中，设计了两个多阶段网络：第一个网络与相应的阶段子网络中的不同音频子序列内部编码降低的多级残余误差，而第二个网络则用相应的阶段子网络从修改后的载体中解释了从修改后的载体中的残差错误，以产生最终揭示的结果。提议的框架的多阶段设计不仅使有效载荷能力的控制更加灵活，而且由于残留错误的逐渐稀疏特征而使隐藏更加容易。定性实验表明，对载体的修改是人类听众无法说明的，并且解码的图像高度可理解。

### MSED: a multi-modal sleep event detection model for clinical sleep analysis 
[[arxiv](https://arxiv.org/abs/2101.02530)] [[cool](https://papers.cool/arxiv/2101.02530)] [[pdf](https://arxiv.org/pdf/2101.02530)]
> **Authors**: Alexander Neergaard Olesen,Poul Jennum,Emmanuel Mignot,Helge B. D. Sorensen
> **First submission**: 2021-01-07
> **First announcement**: 2021-01-08
> **comment**: 20 pages, 6 figures
- **标题**: MSED：用于临床睡眠分析的多模式睡眠事件检测模型
- **领域**: 计算机视觉和模式识别,机器学习,信号处理,应用领域,机器学习
- **摘要**: 研究目标：临床睡眠分析需要对睡眠模式进行手动分析，以正确诊断睡眠障碍。几项研究表明评分离散睡眠事件的差异很大。我们希望研究是否可以使用一种自动方法来检测唤醒（AR），腿部运动（LM）和睡眠无序呼吸（SDB）事件，以及这些事件的联合检测是否比拥有三个独立模型的效果更好。方法：我们设计了一个深层神经网络体系结构，以共同检测多个学术图中的睡眠事件。我们在1653个个人录音中训练了该模型，并在1000个单独的录音中测试了优化模型。通过使用Pearson的相关系数将模型的性能量化，并通过F1，精度和召回分数和召回分数与临床值相关。结果：AR，LM和SDB的优化模型的F1分别为0.70、0.63和0.62。与相应的单事件模型相比，当检测事件共同检测事件时，性能较高。从检测事件计算得出的索引值与手动注释良好相关（$ r^2 $ = 0.73，$ r^2 $ = 0.77，$ r^2 $ = 0.78）。结论：共同检测唤醒，腿部运动和睡眠失调的呼吸事件是可能的，计算的指数值与人类注释良好相关。

### Multimodal Gait Recognition for Neurodegenerative Diseases 
[[arxiv](https://arxiv.org/abs/2101.02469)] [[cool](https://papers.cool/arxiv/2101.02469)] [[pdf](https://arxiv.org/pdf/2101.02469)]
> **Authors**: Aite Zhao,Jianbo Li,Junyu Dong,Lin Qi,Qianni Zhang,Ning Li,Xin Wang,Huiyu Zhou
> **First submission**: 2021-01-07
> **First announcement**: 2021-01-08
> **comment**: No comments
- **标题**: 神经退行性疾病的多模式步态识别
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 近年来，在医学图像或其他感觉数据的分析中已经广泛探索了基于单态的步态识别，并且人们认识到每种已建立的方法都有不同的优势和劣势。作为重要的运动症状，步态障碍通常用于诊断和评估疾病。此外，对患者步行模式的多模式分析的使用弥补了仅在单个测量维度中学习步态变化的单态步态识别方法的单面性。多个测量资源的融合表明，在识别与个体疾病相关的步态模式中表现出了有希望的表现。在本文中，作为一种有用的工具，我们提出了一种新型的混合模型，以通过融合和汇总多个传感器的数据来学习三种神经退行性疾病之间的步态差异，帕金森氏病严重程度不同，健康个体和患者之间的步态差异。空间特征提取器（SFE）用于生成图像或信号的代表性特征。为了从两个模式数据中捕获时间信息，新的相关记忆神经网络（Corrmnn）体系结构设计用于提取时间特征。之后，我们嵌入了一个多开关判别器，将观测值与单个状态估计相关联。与几种最先进的技术相比，我们提出的框架显示出更准确的分类结果。

### Associated Spatio-Temporal Capsule Network for Gait Recognition 
[[arxiv](https://arxiv.org/abs/2101.02458)] [[cool](https://papers.cool/arxiv/2101.02458)] [[pdf](https://arxiv.org/pdf/2101.02458)]
> **Authors**: Aite Zhao,Junyu Dong,Jianbo Li,Lin Qi,Huiyu Zhou
> **First submission**: 2021-01-07
> **First announcement**: 2021-01-08
> **comment**: No comments
- **标题**: 相关的时空胶囊网络用于步态识别
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 根据自己的步态模式来识别一个人是一项艰巨的任务。最先进的方法取决于步态的时间或空间特征的分析，并且步态识别通常在单个模态数据（例如图像，骨架关节坐标或力信号）上进行。证据表明，使用多模式数据更有利于步态研究。因此，我们在这里建立了一个自动化学习系统，并通过在多传感器数据集上培训的相关时空胶囊网络（ASTCAPSNET），以分析多模式信息以进行步态识别。具体而言，我们首先设计了一个低级特征提取器和一个高级特征提取器，用于使用新颖的复发记忆单元和一个关系层的步态提取时空特征提取。随后，采用了贝叶斯模型进行班级标签的决策。与几种最新方法相比，在几个公共数据集（正常和异常步态）上进行了广泛的实验验证了拟议的ASTCAPSNET的有效性。

### MAAS: Multi-modal Assignation for Active Speaker Detection 
[[arxiv](https://arxiv.org/abs/2101.03682)] [[cool](https://papers.cool/arxiv/2101.03682)] [[pdf](https://arxiv.org/pdf/2101.03682)]
> **Authors**: Juan León-Alcázar,Fabian Caba Heilbron,Ali Thabet,Bernard Ghanem
> **First submission**: 2021-01-10
> **First announcement**: 2021-01-11
> **comment**: No comments
- **标题**: MAA：主动扬声器检测的多模式分配
- **领域**: 计算机视觉和模式识别
- **摘要**: 主动扬声器检测需要多模式线索的可靠整合。尽管单个模式可以近似解决方案，但只能通过明确融合音频和视觉特征并建模其时间进展来实现准确的预测。尽管具有固有的muti模式性质，但当前的方法仍然集中在框架级别上的单个扬声器的短期视听特征建模和融合。在本文中，我们提出了一种新型的主动扬声器检测方法，该方法直接解决了问题的多模式性质，并提供了一种直接的策略，其中将现场潜在扬声器的独立视觉特征分配给了先前检测到的语音事件。我们的实验表明，一种由单个帧构建的小图数据结构允许近似瞬时视听分配问题。此外，此初始图的时间扩展可在Ava-Activespeaker数据集上获得新的最新最先进，地图为88.8 \％。

### Target Detection and Segmentation in Circular-Scan Synthetic-Aperture-Sonar Images using Semi-Supervised Convolutional Encoder-Decoders 
[[arxiv](https://arxiv.org/abs/2101.03603)] [[cool](https://papers.cool/arxiv/2101.03603)] [[pdf](https://arxiv.org/pdf/2101.03603)]
> **Authors**: Isaac J. Sledge,Matthew S. Emigh,Jonathan L. King,Denton L. Woods,J. Tory Cobb,Jose C. Principe
> **First submission**: 2021-01-10
> **First announcement**: 2021-01-11
> **comment**: Submitted to IEEE Journal of Oceanic Engineering
- **标题**: 使用半监督卷积编码器中的圆形扫描合成孔径图像中的目标检测和分割
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 我们提出了一个基于显着性的，多目标检测的框架，并分割了循环扫描，合成孔径（CSAS）图像的框架。我们的框架依赖于多分支，卷积编码器 - 码头网络（MB-CEDN）。 MB-CEDN的编码部分提取了CSAS图像的视觉对比功能。这些特征被馈入双重解码器，这些解码器对掩盖目标执行像素级分割。每个解码器都提供了有关构成主要目标的不同观点。这些意见被汇总并层叠成一个深层的网络，以完善细分。我们使用由五个广泛的目标类别组成的现实世界中的CSA图像来评估我们的框架。我们将计算机视觉文献的现有方法进行比较。我们表明，我们的框架优于监督专为自然图像而设计的深度工作网络。它极大地超过了为自然图像开发的无监督的显着性方法。这说明了基于自然图像的模型可能需要更改以有效地对这种成像模式有效。

### Multimodal Engagement Analysis from Facial Videos in the Classroom 
[[arxiv](https://arxiv.org/abs/2101.04215)] [[cool](https://papers.cool/arxiv/2101.04215)] [[pdf](https://arxiv.org/pdf/2101.04215)]
> **Authors**: Ömer Sümer,Patricia Goldberg,Sidney D'Mello,Peter Gerjets,Ulrich Trautwein,Enkelejda Kasneci
> **First submission**: 2021-01-11
> **First announcement**: 2021-01-12
> **comment**: This work has been submitted to the IEEE for possible publication
- **标题**: 从课堂上的面部视频进行的多模式参与分析
- **领域**: 计算机视觉和模式识别,多媒体
- **摘要**: 学生参与是学习和教学的关键结构。尽管大多数文献探讨了有关基于计算机的设置的学生参与分析，但本文将重点放在课堂教学上。为了最好地检查教室中的学生视觉参与，我们在一个半月的一个半月内使用了一所中学上的班级的视听记录进行了一项研究，并在重复的会议上获得了每个学生（n = 15）的持续互动标签，并探索了计算机视觉方法以从教室的面孔分类互动水平。我们训练了深入的嵌入，以了解注意力和情感特征，训练注意网络以进行头部姿势估计，并进行面部表达识别。我们还培训了不同的参与分类器，包括支持向量机，随机森林，多层感知器和长期记忆的长期记忆。表现最好的参与分类器在8和12年级的AUC分别为.620和.720。我们进一步研究了融合策略，发现得分级融合可以改善参与分类器，或者与最佳性能方式相当。我们还研究了个性化的效果，发现仅使用60秒的人特异性数据，从基本分类器的边距不确定性选择的情况下，平均AUC提高了0.084。 4.我们这项工作的主要目的是提供技术手段，以促进有关教学质量和教师培训的研究中课堂视频的手动数据分析。

### The Multimodal Driver Monitoring Database: A Naturalistic Corpus to Study Driver Attention 
[[arxiv](https://arxiv.org/abs/2101.04639)] [[cool](https://papers.cool/arxiv/2101.04639)] [[pdf](https://arxiv.org/pdf/2101.04639)]
> **Authors**: Sumit Jha,Mohamed F. Marzban,Tiancheng Hu,Mohamed H. Mahmoud,Naofal Al-Dhahir,Carlos Busso
> **First submission**: 2020-12-23
> **First announcement**: 2021-01-13
> **comment**: 14 pages, 12 Figures, 3 tables
- **标题**: 多模式驱动程序监视数据库：研究驾驶员注意力的自然主义语料库
- **领域**: 计算机视觉和模式识别
- **摘要**: 智能车辆应该能够监视人类驾驶员的行为和行为，以提供关键警告或必要时进行干预。深度学习和计算机视觉的最新进展在监测人类的行为和活动方面表现出了巨大的希望。尽管这些算法在受控的环境中效果很好，但自然主义的驾驶条件增加了新的挑战，例如照明变化，遮挡和极端的头部姿势。需要大量的内域数据来训练在预测相关任务以有效监控驾驶员的行为和行为方面提供高性能的模型。为了构建所需的基础架构，本文介绍了多模式驱动程序监视（MDM）数据集，该数据集由59个记录的主题收集，这些主题执行各种任务。我们使用使用基金标记不断跟踪驾驶员头部移动的限制设备，从而提供基于框架的注释来训练在自然主义驾驶条件下训练头部姿势算法。我们要求驾驶员查看预定的凝视位置，以获得驾驶员的面部图像和视觉注意力之间的准确相关性。当驾驶员执行常见的次要活动（例如使用智能手机导航并操作车内信息娱乐系统）时，我们还会收集数据。所有驾驶员的活动都记录在高清RGB相机和飞行时间深度摄像头。我们还记录了控制器区域网络总车（CAN-BUS），从而提取了重要信息。这些高质量的记录是培训各种有效算法以监视驾驶员的理想资源，从而在车载安全系统领域提供了进一步的进步。

### Context Matters: Self-Attention for Sign Language Recognition 
[[arxiv](https://arxiv.org/abs/2101.04632)] [[cool](https://papers.cool/arxiv/2101.04632)] [[pdf](https://arxiv.org/pdf/2101.04632)]
> **Authors**: Fares Ben Slimane,Mohamed Bouguessa
> **First submission**: 2021-01-12
> **First announcement**: 2021-01-13
> **comment**: No comments
- **标题**: 上下文事项：手语识别的自我注意力
- **领域**: 计算机视觉和模式识别,人工智能,机器学习
- **摘要**: 本文提出了一个注意力网络，以实现连续手语识别的任务。所提出的方法利用了共同依赖数据流的数据来对手语模式进行建模。这些不同的信息渠道可以彼此共享一个复杂的时间结构。因此，我们将注意力应用于同步并有助于捕获不同手语组成部分之间的纠缠依赖关系。即使手语是多渠道，握手也代表着标志解释中的中心实体。在其正确上下文中看到手绘定义了标志的含义。考虑到这一点，我们利用注意力机制有效地将手部特征及其适当的时空上下文汇总，以更好地识别标志。我们发现，通过这样做，该模型能够识别围绕主要手和面部区域的基本手语组成部分。我们在2014年基准数据集RWTH-PHOENIX-WEATHER上测试了我们的模型，从而产生了竞争成果。

### A Multimodal Eye Movement Dataset and a Multimodal Eye Movement Segmentation Analysis 
[[arxiv](https://arxiv.org/abs/2101.04318)] [[cool](https://papers.cool/arxiv/2101.04318)] [[pdf](https://arxiv.org/pdf/2101.04318)]
> **Authors**: Wolfgang Fuhl,Enkelejda Kasneci
> **First submission**: 2021-01-12
> **First announcement**: 2021-01-13
> **comment**: No comments
- **标题**: 多模式的眼动数据集和多模式的眼动分段分析
- **领域**: 计算机视觉和模式识别
- **摘要**: 我们提出一个带有注释的眼动的新数据集。该数据集由在现实世界和模拟器中的汽车骑行期间记录的80万目光。总体而言，注释了19名受试者的眼睛运动。在此数据集中，有几个数据源，例如眼睑闭合，瞳孔中心，光学矢量和矢量从眼角的中心开始进入学生中心。这些不同的数据源对其适合眼动分类的好处进行了分析和评估。这些结果将有助于实时系统和算法的开发人员找到适合其应用程序的最佳数据源。此外，可以在此数据集上对新算法进行培训和评估。数据和MATLAB代码可以在此处下载https://atreus.informatik.uni-tuebingen.de/seafile/d/8e2ab8c3fd4444e1a1a1a1a135/?p=%2FA%2FA%2FA%20Multimodal%20Multimodal%20eyee；

### Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding 
[[arxiv](https://arxiv.org/abs/2101.05208)] [[cool](https://papers.cool/arxiv/2101.05208)] [[pdf](https://arxiv.org/pdf/2101.05208)]
> **Authors**: Dexin Wang,Deyi Xiong
> **First submission**: 2020-12-18
> **First announcement**: 2021-01-14
> **comment**: No comments
- **标题**: 多模式机器翻译的有效对象级的视觉上下文建模：掩盖无关的物体有助于接地
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学
- **摘要**: 视觉上下文为多模式机器翻译（MMT）提供了接地信息。但是，以前的MMT模型和对视觉特征的探测研究表明，在MMT中探索视觉信息的较少，因为它通常是文本信息的多余的。在本文中，我们提出了一个对象级的视觉上下文建模框架（OVC），以有效捕获和探索多模式机器翻译的视觉信息。通过检测到的对象，提出的OVC通过掩盖视觉方式中的无关物体来鼓励MMT对所需的视觉对象进行地面翻译。我们为提出的额外的对象掩盖损失装备以实现这一目标。对象掩盖损失是根据掩盖对象和源文本之间的相似性估算的，以鼓励掩盖源源 -  iRrelevant对象。此外，为了产生视觉符合目标单词，我们进一步提出了OVC的视力加权翻译损失。 MMT数据集的实验表明，所提出的OVC模型优于最先进的MMT模型，并且分析表明，掩盖无关的物体有助于接地MMT。

### Piano Skills Assessment 
[[arxiv](https://arxiv.org/abs/2101.04884)] [[cool](https://papers.cool/arxiv/2101.04884)] [[pdf](https://arxiv.org/pdf/2101.04884)]
> **Authors**: Paritosh Parmar,Jaiden Reddy,Brendan Morris
> **First submission**: 2021-01-13
> **First announcement**: 2021-01-14
> **comment**: Dataset is available from: https://github.com/ParitoshParmar/Piano-Skills-Assessment
- **标题**: 钢琴技能评估
- **领域**: 计算机视觉和模式识别,机器学习,多媒体,声音,音频和语音处理
- **摘要**: 计算机可以确定钢琴演奏者的技能水平吗？最好将此评估基于对玩家表现的视觉分析，或者我们应该在眼睛上信任耳朵吗？由于当前的CNN难以处理长时间的视频视频，因此如何对较短的剪辑进行采样以最好地反映球员的技能水平？在这项工作中，我们收集并发布了一个首先的数据集，以用于多模式技能评估，重点是评估钢琴演奏者的技能水平，回答问问题，启动对钢琴弹弹技巧的自动评估工作，并为未来的工作提供基准。数据集可从：https：//github.com/paritoshparmar/piano-skills-assessment获得。

### Exploring Adversarial Robustness of Multi-Sensor Perception Systems in Self Driving 
[[arxiv](https://arxiv.org/abs/2101.06784)] [[cool](https://papers.cool/arxiv/2101.06784)] [[pdf](https://arxiv.org/pdf/2101.06784)]
> **Authors**: James Tu,Huichen Li,Xinchen Yan,Mengye Ren,Yun Chen,Ming Liang,Eilyan Bitar,Ersin Yumer,Raquel Urtasun
> **First submission**: 2021-01-17
> **First announcement**: 2021-01-18
> **comment**: No comments
- **标题**: 探索自动驾驶中多传感器感知系统的对抗性鲁棒性
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 现代的自动驾驶感知系统已被证明可以在处理互补输入（例如带有图像的LiDAR）时得到改善。孤立地，发现2D图像非常容易受到对抗攻击的影响。然而，关于多模型模型的对抗性鲁棒性的研究有限，该模型将激光雷达与图像特征融合在一起。此外，现有作品不考虑在输入模式中保持一致的物理上可实现的扰动。在本文中，我们通过将对抗物体放置在宿主车上来展示多传感器检测的实用敏感性。我们专注于在实践中可以执行可行的物理上可实现和输入的攻击，并表明单个通用对手可以从最先进的多模式探测器中隐藏不同的主机车辆。我们的实验表明，成功的攻击主要是由易于损坏的图像特征引起的。此外，我们发现，在现代传感器融合方法中，项目图像特征到3D，对抗攻击可以利用投影过程，以在3D中遥远地区生成误报。为了实现更强大的多模式感知系统，我们表明，具有特征降级的对抗性训练可以显着提高这种攻击性的鲁棒性。但是，我们发现标准的对抗防御仍然难以防止假阳性，这也是由于3D激光雷达点和2D像素之间不准确的关联引起的。

### End-to-end Interpretable Neural Motion Planner 
[[arxiv](https://arxiv.org/abs/2101.06679)] [[cool](https://papers.cool/arxiv/2101.06679)] [[pdf](https://arxiv.org/pdf/2101.06679)]
> **Authors**: Wenyuan Zeng,Wenjie Luo,Simon Suo,Abbas Sadat,Bin Yang,Sergio Casas,Raquel Urtasun
> **First submission**: 2021-01-17
> **First announcement**: 2021-01-18
> **comment**: CVPR 2019 (Oral)
- **标题**: 端到端可解释的神经运动计划者
- **领域**: 计算机视觉和模式识别,机器人技术
- **摘要**: 在本文中，我们提出了一个神经运动计划者（NMP），用于学习在复杂的城市场景中自主行驶，其中包括交通照明处理，屈服和与多个路用户的互动。为了实现这一目标，我们设计了一个整体模型，该模型将作为输入原始LIDAR数据和高清图，并以3D检测的形式及其未来轨迹产生可解释的中间表示，以及定义自动驾驶汽车可以在计划范围内采用的每个位置的好处的成本量。然后，我们采样一组不同的物理可能轨迹，并选择具有最低学习成本的轨迹。重要的是，我们的成本量能够自然捕获多模式。我们证明了我们的方法在北美几个城市捕获的现实世界驱动数据中的有效性。我们的实验表明，与所有基准相比，学习的成本量可以产生更安全的计划。

### Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation 
[[arxiv](https://arxiv.org/abs/2101.07253)] [[cool](https://papers.cool/arxiv/2101.07253)] [[pdf](https://arxiv.org/pdf/2101.07253)]
> **Authors**: Maximilian Jaritz,Tuan-Hung Vu,Raoul de Charette,Émilie Wirbel,Patrick Pérez
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: TPAMI 2022
- **标题**: 3D语义分割中域适应的跨模式学习
- **领域**: 计算机视觉和模式识别
- **摘要**: 当标签稀缺时，域的适应性是使学习能够学习的重要任务。尽管大多数作品仅着眼于图像模式，但有许多重要的多模式数据集。为了利用多模式的域适应性，我们提出了跨模式学习，在这种学习中，我们通过相互模仿的两种模式的预测在两种模式的预测之间执行一致性。我们限制了我们的网络以对未标记的目标域数据进行正确的数据和跨模态的一致预测做出正确的预测。在无监督和半监督的域适应设置中的实验证明了这种新型域适应策略的有效性。具体而言，我们从2D图像，3D点云或两者兼而有之的3D语义分割的任务。我们利用最近的驾驶数据集生成各种域的适应场景，包括场景布局，照明，传感器设置和天气的变化，以及合成到现实的设置。我们的方法在所有适应方案上都显着改善了以前的Uni-Modal适应基线。我们的代码可在https://github.com/valeoai/xmuda_journal上公开获取

### Hyperspectral Image Denoising via Multi-modal and Double-weighted Tensor Nuclear Norm 
[[arxiv](https://arxiv.org/abs/2101.07681)] [[cool](https://papers.cool/arxiv/2101.07681)] [[pdf](https://arxiv.org/pdf/2101.07681)]
> **Authors**: Sheng Liu,Xiaozhen Xie,Wenfeng Kong
> **First submission**: 2021-01-19
> **First announcement**: 2021-01-20
> **comment**: arXiv admin note: text overlap with arXiv:2106.12489
- **标题**: 高光谱图像通过多模式和双重加权核定标准降级
- **领域**: 计算机视觉和模式识别
- **摘要**: 高光谱图像（HSIS）通常遭受不同类型的污染。这严重降低了HSI的质量，并限制了随后的处理任务的准确性。 HSI DENOISING可以建模为低级张量降解问题。张量奇异值分解引起的张量核定标（TNN）在此问题中起重要作用。在这封信中，我们首先重新考虑了TNN中的三个不起眼但至关重要的现象。在HSI的傅立叶变换域中，不同的频率切片（FS）包含不同的信息。每个FS的不同单数值（SV）也代表不同的信息。这两个物理现象不仅处于光谱模式，而且位于空间模式下。然后，基于它们，我们提出了一个多模式和双重加权的TNN。它可以根据所有HSIS的身体含义适应FS和SVS。在乘数的交替方向方法的框架中，我们设计了一种有效的交替迭代策略来优化我们所提出的模型。对合成和真实HSI数据集进行了剥落的实验证明了它们在相关方法上的优势。

### AXM-Net: Implicit Cross-Modal Feature Alignment for Person Re-identification 
[[arxiv](https://arxiv.org/abs/2101.08238)] [[cool](https://papers.cool/arxiv/2101.08238)] [[pdf](https://arxiv.org/pdf/2101.08238)]
> **Authors**: Ammarah Farooq,Muhammad Awais,Josef Kittler,Syed Safwan Khalid
> **First submission**: 2021-01-19
> **First announcement**: 2021-01-21
> **comment**: AAAI-2022 (Oral Paper)
- **标题**: Axm-NET：人重新识别的隐式跨模式特征对齐
- **领域**: 计算机视觉和模式识别,机器学习
- **摘要**: 跨模式的人重新识别（RE-ID）对于现代视频监视系统至关重要。关键挑战是与一个人有关的语义信息引起的跨模式表示，而忽略背景信息。这项工作提出了一种新型的基于卷积神经网络（CNN）的体系结构，旨在学习语义上的跨模式视觉和文本表示。基础构建块，名为Axm-Block，是一个统一的多层网络，它可以动态利用多尺度知识从模态中，并根据共享语义重新校准每种模式。为了补充卷积设计，在文本分支中应用上下文注意来操纵长期依赖。此外，我们提出了一种独特的设计，以增强基于视觉零件的特征连贯性和局部性信息。我们的框架具有新颖的能力，可以在功能学习阶段隐式学习模式之间的一致语义。统一特征学习有效地利用文本数据作为视觉表示学习的超级注释信号，并自动拒绝无关的信息。整个AXM-NET经过Cuhk-Pedes数据的端到端训练。我们报告了两个任务的结果，即人搜索和跨模式重新ID。 AXM-NET优于当前最新方法（SOTA）方法，并在Cuhk-Pedes测试集上获得64.44 \％等级@1。在Crossre-ID和Cuhk-Sysu数据集中，它还胜过竞争对手的竞争对手$> $ 10 \％。

### Video Relation Detection with Trajectory-aware Multi-modal Features 
[[arxiv](https://arxiv.org/abs/2101.08165)] [[cool](https://papers.cool/arxiv/2101.08165)] [[pdf](https://arxiv.org/pdf/2101.08165)]
> **Authors**: Wentao Xie,Guanghui Ren,Si Liu
> **First submission**: 2021-01-20
> **First announcement**: 2021-01-21
> **comment**: No comments
- **标题**: 带有轨迹感知多模式特征的视频关系检测
- **领域**: 计算机视觉和模式识别
- **摘要**: 视频关系检测问题是指视频中不同对象之间的关系（例如空间关系和动作关系）的检测。在本文中，我们使用轨迹感知的多模式特征介绍了视频关系检测，以解决此任务。考虑到视频中进行视觉关系检测的复杂性，我们将此任务分解为三个子任务：对象检测，轨迹建议和关系预测。我们使用最新的对象检测方法来确保对象轨迹检测和多模式特征表示的准确性，以帮助对象之间的关系预测。我们的方法赢得了视频关系检测任务的第一名，这是视频关系理解ACM Multimedia 2020中的大挑战的第一名。

### AS-Net: Fast Photoacoustic Reconstruction with Multi-feature Fusion from Sparse Data 
[[arxiv](https://arxiv.org/abs/2101.08934)] [[cool](https://papers.cool/arxiv/2101.08934)] [[pdf](https://arxiv.org/pdf/2101.08934)]
> **Authors**: Mengjie Guo,Hengrong Lan,Changchun Yang,Fei Gao
> **First submission**: 2021-01-21
> **First announcement**: 2021-01-22
> **comment**: No comments
- **标题**: AS-NET：与稀疏数据的多功能融合的快速光声重建
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 光声（PA）成像是一种生物医学成像方式，能够在深度上获得高对比度的光吸收图像，远大于传统的光学成像技术。但是，实际仪器和几何形状限制了成像目标周围可用的声传感器的数量，从而导致传感器数据的稀疏性。传统的PA图像重建方法将直接应用于稀疏PA数据时会产生严重的伪影。在本文中，我们首先建议采用一种新型的信号处理方法，使稀疏PA原始数据更适合神经网络，同时加快图像重建。然后，我们提出了使用多功能融合的PA重建的注意力转向网络（AS-NET）。 AS-NET在不同的数据集上进行了验证，包括来自眼底脉管系统幻像的模拟光声数据以及来自体内鱼和小鼠的实验数据。值得注意的是，该方法还能够消除体内数据的地面真相中存在的一些伪影。结果表明，我们的方法以更快的速度提供了卓越的重建。

### A Person Re-identification Data Augmentation Method with Adversarial Defense Effect 
[[arxiv](https://arxiv.org/abs/2101.08783)] [[cool](https://papers.cool/arxiv/2101.08783)] [[pdf](https://arxiv.org/pdf/2101.08783)]
> **Authors**: Yunpeng Gong,Zhiyong Zeng,Liwen Chen,Yifan Luo,Bin Weng,Feng Ye
> **First submission**: 2021-01-21
> **First announcement**: 2021-01-22
> **comment**: arXiv admin note: text overlap with arXiv:2101.08533
- **标题**: 一个人重新识别数据增强方法具有对抗防御效应
- **领域**: 计算机视觉和模式识别
- **摘要**: 人重新识别（REID）模型的安全性在REID的应用中起决定性作用。但是，深度神经网络已被证明是脆弱的，并且在清洁图像中添加未发现的对抗性扰动可以欺骗在干净的图像中表现良好的深神经网络。我们提出了一种具有对抗防御效果的REID多模式数据增强方法：1）灰度贴片替换，它由局部灰度贴片替换（LGPR）和全球灰度贴片替换（GGPR）组成。这种方法不仅可以提高模型的准确性，而且还可以帮助模型防御对抗性例子。 2）多模式防御，它整合了可见，灰度和草图的三个均匀模态图像，并进一步增强了模型的防御能力。这些方法融合了不同均质图像的不同方式以丰富输入样品品种，样品的变化将减少REID模型过度拟合以染色变化，并使数据集的对抗空间变得难以对齐，因此模型的准确性得到了改善，并且攻击效应大大降低了。融合了模态均匀图像越多，防御能力就越强。所提出的方法在多个数据集上表现良好，并成功地捍卫了CVPR2020对REID [10]提出的MS-SSIM的攻击，并将准确性提高了467倍（0.2％至93.3％）。该代码可在https://github.com/github.com/finger-monkey/reid-reid_adversarialial_defense上获得。

### AI Choreographer: Music Conditioned 3D Dance Generation with AIST++ 
[[arxiv](https://arxiv.org/abs/2101.08779)] [[cool](https://papers.cool/arxiv/2101.08779)] [[pdf](https://arxiv.org/pdf/2101.08779)]
> **Authors**: Ruilong Li,Shan Yang,David A. Ross,Angjoo Kanazawa
> **First submission**: 2021-01-21
> **First announcement**: 2021-01-22
> **comment**: Project page: https://google.github.io/aichoreographer/; Dataset page: https://google.github.io/aistplusplus_dataset/
- **标题**: AI编排者：与AIST ++的音乐条件3D舞蹈一代
- **领域**: 计算机视觉和模式识别,图形,多媒体
- **摘要**: 我们介绍了AIST ++，这是一个新的3D舞蹈运动和音乐的多模式数据集，以及事实，这是一个完整的跨模式变压器网络，用于生成以音乐为条件的3D舞蹈运动。拟议的AIST ++数据集包含1408序列中的3D舞蹈运动的5.2小时，涵盖了10个舞蹈类型，其中包含带有已知相机姿势的多视频视频 - 据我们所知，这是此类最大的数据集。我们表明，将序列模型（例如变压器）应用于该数据集的序列模型以进行音乐条件3D运动的任务不会产生与输入音乐相关的令人满意的3D运动。我们通过引入其架构设计和监督的关键变化来克服这些缺点：事实模型涉及一个深厚的跨模式变压器块，并进行了全面注意，该块经过培训，可以预测$ n $ n $ Future Motions。我们从经验上表明，这些变化是产生长时间逼真的舞蹈运动序列的关键因素，这些因素对输入音乐充分了。我们通过用户研究对AIST ++进行了广泛的实验，我们的方法在定性和定量上都优于最新的最新方法。

### Eliminate Deviation with Deviation for Data Augmentation and a General Multi-modal Data Learning Method 
[[arxiv](https://arxiv.org/abs/2101.08533)] [[cool](https://papers.cool/arxiv/2101.08533)] [[pdf](https://arxiv.org/pdf/2101.08533)]
> **Authors**: Yunpeng Gong,Liqing Huang,Lifei Chen
> **First submission**: 2021-01-21
> **First announcement**: 2021-01-22
> **comment**: No comments
- **标题**: 消除偏差，以增加数据增强和一般的多模式数据学习方法
- **领域**: 计算机视觉和模式识别
- **摘要**: 计算机视觉的挑战之一是它需要适应可变环境中的颜色偏差。因此，将颜色偏差对预测的不利影响最小化是视觉任务的主要目标之一。当前的解决方案着重于使用生成模型增强训练数据以增强输入变化的不变性。但是，这种方法通常会引入新的噪声，从而限制了生成数据的增益。为此，本文提出了一种消除偏差偏差的策略，该偏差称为随机颜色辍学（RCD）。我们的假设是，如果查询图像和画廊图像之间存在颜色偏差，那么在忽略颜色信息后，一些示例的检索结果会更好。具体而言，该策略通过在训练数据中辍学的部分颜色信息来平衡神经网络中颜色特征和无关的特征之间的权重，以克服颜色devitaion的效果。提出的RCD可以与各种现有的REID模型相结合，而无需更改学习策略，并且可以应用于其他计算机视野字段，例如对象检测。在几个REID基线和三个常见的大规模数据集（例如Market1501，Dukemtmc和MSMT17）上进行的实验已验证了该方法的有效性。跨域测试的实验表明，这种策略显着消除了域间隙。此外，为了了解RCD的工作机制，我们从分类的角度分析了该策略的有效性，这表明在具有较强域变化的视觉任务中使用许多颜色信息可能更好。

### Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking 
[[arxiv](https://arxiv.org/abs/2101.08466)] [[cool](https://papers.cool/arxiv/2101.08466)] [[pdf](https://arxiv.org/pdf/2101.08466)]
> **Authors**: Nan Jiang,Kuiran Wang,Xiaoke Peng,Xuehui Yu,Qiang Wang,Junliang Xing,Guorong Li,Jian Zhao,Guodong Guo,Zhenjun Han
> **First submission**: 2021-01-21
> **First announcement**: 2021-01-22
> **comment**: 13 pages, 8 figures, submitted to IEEE T-MM
- **标题**: 反UAV：用于无人机跟踪的大型多模式基准
- **领域**: 计算机视觉和模式识别
- **摘要**: 无人机（UAV）在商业和娱乐中都提供许多应用。这样，监视无人机的操作状态至关重要。在这项工作中，我们考虑跟踪无人机，提供丰富信息（例如位置和轨迹）的任务。为了促进有关此主题的研究，我们提出了一个数据集Anti-UAV，其中300多个视频对包含超过580k手动注释的边界框。在研究跟踪无人机的研究中，发布这样的大规模数据集可能是有用的第一步。此外，应对反UAV的研究挑战的进步可以帮助设计反UAV系统，从而更好地监视无人机。此外，提出了一种名为Dual-Flow语义一致性（DFSC）的新颖方法用于无人机跟踪。跟踪器通过视频序列的语义流进行调节，学习了更强大的类级语义信息，并获得了更具歧视性的实例级特征。实验结果表明，反UAV非常具有挑战性，所提出的方法可以有效地改善跟踪器的性能。 https://github.com/ucas-vg/anti-uav将公开获得反UAV基准和拟议方法的代码。

### A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric 
[[arxiv](https://arxiv.org/abs/2101.09028)] [[cool](https://papers.cool/arxiv/2101.09028)] [[pdf](https://arxiv.org/pdf/2101.09028)]
> **Authors**: Yitian Yuan,Xiaohan Lan,Xin Wang,Long Chen,Zhi Wang,Wenwu Zhu
> **First submission**: 2021-01-22
> **First announcement**: 2021-01-25
> **comment**: No comments
- **标题**: 仔细观察视频中的时间句子接地：数据集和指标
- **领域**: 计算机视觉和模式识别
- **摘要**: 视频（TSGV）中的暂时句子基础，即基础自然语言句子，这表明在长期且未经修剪的视频序列中表明人类的复杂活动，在过去的几年中一直受到前所未有的关注。尽管每种新提出的方法可能比以前的方法更高，但当前的TSGV模型仍然倾向于捕获时刻注释偏见，并且无法充分利用多模式输入。更令人难以置信的是，几个没有训练的极其简单的基线也可以实现最先进的表现。在本文中，我们仔细研究了TSGV的现有评估协议，并发现盛行的数据集拆分和评估指标都是引起不可靠基准测试的魔鬼。为此，我们建议重新组织两个广泛使用的TSGV基准（ActivityNet标题和Charades-STA）。具体而言，我们故意使训练和测试分裂（即分布式分布（OOD）测试）的基础真实力矩分布不同。同时，我们引入了一个新的评估度量DR@n，iou@m，通过对受偏见的时刻预测进行惩罚，并减轻数据集注释偏见引起的膨胀评估，例如越过的基本瞬间。根据我们的新评估协议，我们对八种最先进的TSGV方法进行了广泛的实验和消融研究。所有结果表明，重组的数据集拆分和新指标可以更好地监视TSGV的进度。我们的重组数据集可在https://github.com/yytzsy/grounding_changing_distribution上找到。

### Visual Question Answering based on Local-Scene-Aware Referring Expression Generation 
[[arxiv](https://arxiv.org/abs/2101.08978)] [[cool](https://papers.cool/arxiv/2101.08978)] [[pdf](https://arxiv.org/pdf/2101.08978)]
> **Authors**: Jung-Jun Kim,Dong-Gyu Lee,Jialin Wu,Hong-Gyu Jung,Seong-Whan Lee
> **First submission**: 2021-01-22
> **First announcement**: 2021-01-25
> **comment**: 32 pages, 8 figures
- **标题**: 视觉问题基于本地场景 - 参考表达生成
- **领域**: 计算机视觉和模式识别
- **摘要**: 视觉问题回答需要对图像和自然语言有深入的了解。但是，大多数方法主要集中于视觉概念。例如各种对象之间的关系。对象类别的使用有限使用与其关系或简单的问题嵌入不足以表示复杂的场景和解释决策。为了解决此限制，我们建议使用为图像生成的文本表达式的使用，因为这样的表达式几乎没有结构性约束，并且可以提供图像的更丰富的描述。生成的表达式可以与视觉特征和嵌入问题融合在一起，以获得与问题相关的答案。还提出了一个联合插入多头注意网络，以模拟与共同注意的三种不同信息方式。我们对VQA V2数据集进行了定量和定性评估所提出的方法，并将其与答案预测的最新方法进行了比较。还在reccoco，refcoco+和reccocog数据集上评估了生成表达式的质量。实验结果证明了该方法的有效性，并表明它在定量和定性结果方面都优于所有竞争方法。

### Anytime 3D Object Reconstruction using Multi-modal Variational Autoencoder 
[[arxiv](https://arxiv.org/abs/2101.10391)] [[cool](https://papers.cool/arxiv/2101.10391)] [[pdf](https://arxiv.org/pdf/2101.10391)]
> **Authors**: Hyeonwoo Yu,Jean Oh
> **First submission**: 2021-01-25
> **First announcement**: 2021-01-26
> **comment**: IEEE Robotics and Automation Letters (accepted with ICRA2022 options)
- **标题**: 随时使用多模式变量自动编码器进行3D对象重建
- **领域**: 计算机视觉和模式识别
- **摘要**: 对于有效的人类机器人组合，对于机器人来说，能够与人类操作员分享他们的视觉感知很重要。在苛刻的远程协作环境中，可以利用诸如自动编码器之类的数据压缩技术以紧凑的形式以潜在变量来获取和传输数据。此外，为了确保即使在不稳定的环境下进行实时运行时性能，都需要采用任何时间估算方法，可以从不完整的信息中重新构建完整内容。在这种情况下，我们提出了一种插入潜在变量的方法，其元素部分丢失了。为了实现只有几个变量维度的任何时间属性，利用类别级别的先验信息至关重要。无论每个训练数据点的标签如何，都认为变异自动编码器中使用的先前分布是各向同性高斯。这种类型的扁平先验使得很难从类别级别的分布中执行插补。我们通过利用潜在空间中特定于类别的多模式先验分布来克服这一限制。可以根据其余元素找到特定的模态来采样部分传输数据的缺失元素。由于该方法旨在将部分元素用于任何时间估计，因此它也可以用于数据过度压缩。基于模型网和Pascal3D数据集的实验，提出的方法始终显示出优于自动编码器和多种自动编码器，高达70％的数据丢失。

### Deep Video Inpainting Detection 
[[arxiv](https://arxiv.org/abs/2101.11080)] [[cool](https://papers.cool/arxiv/2101.11080)] [[pdf](https://arxiv.org/pdf/2101.11080)]
> **Authors**: Peng Zhou,Ning Yu,Zuxuan Wu,Larry S. Davis,Abhinav Shrivastava,Ser-Nam Lim
> **First submission**: 2021-01-26
> **First announcement**: 2021-01-27
> **comment**: No comments
- **标题**: 深视频介绍检测
- **领域**: 计算机视觉和模式识别
- **摘要**: 本文研究了视频介绍检测，该视频在空间和时间上都定位了视频中的涂有漆区。特别是，我们介绍了Vidnet，视频介绍检测网络，该网络包含带有注意模块的两流编码器架构。为了揭示在压缩中编码的伪影，Vidnet还将错误级别分析框架吸收以增强RGB框架，并使用编码器在不同级别上产生多模式特征。探索空间和时间关系，这些特征被卷积的LSTM进一步解码，以预测涂层区域的面具。此外，当检测像素是否为覆盖时，我们会提出一个四方向的局部注意模块，该模块从四个方向从其周围像素中借用信息。进行了广泛的实验以验证我们的方法。我们证明，除其他外，Vidnet不仅优于清晰的替代介绍检测方法，而且在训练过程中看不见的新型视频中也很好地概括了。

### Global-Local Propagation Network for RGB-D Semantic Segmentation 
[[arxiv](https://arxiv.org/abs/2101.10801)] [[cool](https://papers.cool/arxiv/2101.10801)] [[pdf](https://arxiv.org/pdf/2101.10801)]
> **Authors**: Sihan Chen,Xinxin Zhu,Wei Liu,Xingjian He,Jing Liu
> **First submission**: 2021-01-26
> **First announcement**: 2021-01-27
> **comment**: No comments
- **标题**: RGB-D语义细分的全局本地传播网络
- **领域**: 计算机视觉和模式识别
- **摘要**: 深度信息在RGB-D语义分段任务中很重要，可为彩色图像提供其他几何信息。大多数现有方法利用多阶段融合策略来传播RGB分支的深度特征。但是，在非常深的阶段，以简单的元素加法方式传播无法完全利用深度信息。我们提出了全局本地传播网络（GLPNET）来解决此问题。具体而言，引入了局部上下文融合模块（L-CFM），以在元素融合之前动态地对齐两种模态，并引入了全局上下文融合模块（G-CFM），以通过共同对多模式全局上下文特征进行建模，将深度信息传播到RGB分支。广泛的实验证明了所提出的融合模块的有效性和互补性。我们的GLPNET将两个融合模块嵌入了两流编码器结构中，在两个具有挑战性的室内场景细分数据集（即NYU-DEPTH V2和SUN-RGBD数据集）上实现了新的最新性能。

### Towards Universal Physical Attacks On Cascaded Camera-Lidar 3D Object Detection Models 
[[arxiv](https://arxiv.org/abs/2101.10747)] [[cool](https://papers.cool/arxiv/2101.10747)] [[pdf](https://arxiv.org/pdf/2101.10747)]
> **Authors**: Mazen Abdelfattah,Kaiwen Yuan,Z. Jane Wang,Rabab Ward
> **First submission**: 2021-01-26
> **First announcement**: 2021-01-27
> **comment**: ef:2021 IEEE International Conference on Image Processing (ICIP)
- **标题**: 朝着级联的摄像头3D对象检测模型的通用物理攻击
- **领域**: 计算机视觉和模式识别,图像和视频处理
- **摘要**: 在自动驾驶汽车的背景下，我们提出了对级联的多模式深度学习网络（DNN）的通用和物理上可实现的对抗性攻击。 DNN在3D对象检测中取得了高性能，但已知它们容易受到对抗性攻击的影响。这些攻击已经在RGB图像域中进行了大量研究，最近在点云域中进行了大量研究，但很少同时在两个域中 - 在本文中填补的差距。我们使用单个3D网格和可区分的渲染来探索网格的几何形状和纹理的扰动如何降低DNN对对抗性攻击的鲁棒性。我们攻击了突出的级联多模式DNN，即Frustum-Pointnet模型。使用流行的KITTI基准测试，我们表明拟议的通用多模式攻击成功地将模型检测到汽车的能力降低了近73％。这项工作可以帮助理解级联的RGB点云DNN所学的内容及其对对抗性攻击的脆弱性。

### DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents 
[[arxiv](https://arxiv.org/abs/2101.11796)] [[cool](https://papers.cool/arxiv/2101.11796)] [[pdf](https://arxiv.org/pdf/2101.11796)]
> **Authors**: Tsu-Jui Fu,William Yang Wang,Daniel McDuff,Yale Song
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: AAAI'22
- **标题**: DOC2PPT：自动演示文稿从科学文档中幻灯
- **领域**: 计算机视觉和模式识别
- **摘要**: 创建演示材料需要复杂的多模式推理技能，以总结关键概念并以逻辑和视觉上令人愉悦的方式安排它们。机器可以学会模仿这个费力的过程吗？我们提出了一项新颖的任务和文档至扫描生成的方法。解决此问题涉及文档摘要，图像和文本检索，幻灯片结构和布局预测，以适合演示的形式安排关键元素。我们提出了一种分层序列到序列的方法，以端到端的方式处理我们的任务。我们的方法利用了文档和幻灯片中的固有结构，并结合了释义和布局预测模块以生成幻灯片。为了帮助加速该域中的研究，我们发布了一个数据集，该数据集大约是我们实验中使用的6K配对文档和幻灯片。我们表明，我们的方法表现优于强大的基线，并产生具有丰富内容和一致图像的幻灯片。

### Multi-Modal Aesthetic Assessment for MObile Gaming Image 
[[arxiv](https://arxiv.org/abs/2101.11700)] [[cool](https://papers.cool/arxiv/2101.11700)] [[pdf](https://arxiv.org/pdf/2101.11700)]
> **Authors**: Zhenyu Lei,Yejing Xie,Suiyi Ling,Andreas Pastor,Junle Wang,Patrick Le Callet
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: 5 pages
- **标题**: 手机游戏图像的多模式审美评估
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 随着各种游戏技术，服务，游戏风格和平台的扩散，对游戏内容的多维美学评估对游戏行业越来越重要。根据多元化的游戏玩家，游戏设计师，图形开发人员等的各种需求。在特别的条件下，需要多模式美学评估来考虑不同的美学维度/观点。由于不同的美学维度之间存在不同的潜在关系，例如，“彩色”和“颜色和谐”之间存在不同的关系，因此利用在多个相关维度中附加的有效信息可能是有利的。为此，我们通过多任务学习解决了这个问题。我们的倾向是寻求和学习不同审美相关维度之间的相关性，以进一步提高预测所有美学维度的概括性能。因此，可以通过利用其他维度的互补来源（即，通过跨维度共享培训信息）间接增强培训数据的互补来源来删除使用一个单个维度的有限数据获得良好预测的“瓶颈”。根据实验结果，提出的模型在预测四个游戏美学维度方面胜过最先进的美学指标。

### Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network 
[[arxiv](https://arxiv.org/abs/2101.11562)] [[cool](https://papers.cool/arxiv/2101.11562)] [[pdf](https://arxiv.org/pdf/2101.11562)]
> **Authors**: Yehao Li,Yingwei Pan,Ting Yao,Jingwen Chen,Tao Mei
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: AAAI 2021; Code is publicly available at: https://github.com/YehLi/TDEN
- **标题**: 通过脱钩编码器网络进行视觉预测中的计划采样
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 尽管对基于BERT的编码器进行了令人印象深刻的视觉语言（VL）以进行VL理解，但对VL理解和发电的通用编码器进行预处理，仍然具有挑战性。困难源于两个学科的本质上不同的特殊性，例如，VL理解任务利用跨模式传递的无限制消息，而生成任务仅采用视觉到文本消息传递。在本文中，我们从编码器解码器结构的两流解耦设计开始，其中涉及两个解耦的跨模式编码器和解码器，以分别执行每种类型的代理任务，以同时了解VL理解和生成预处理。此外，对于VL进行了预处理，主要的方法是用面具令牌替换一些输入视觉/单词令牌，并强制执行多模式编码器/解码器以重建原始令牌，但是在下游任务上进行微调时，不涉及蒙版令牌。作为替代方案，我们提出了一种主要的计划采样策略，该策略通过以两次通用方式通过预处理编码器来优雅地减轻这种差异。广泛的实验表明，通过对四个VL理解和下游任务进行微调，我们预处理的编码码头描述器具有令人信服的普遍性。源代码可在\ url {https://github.com/yehli/tden}中获得。

### Multi-Threshold Attention U-Net (MTAU) based Model for Multimodal Brain Tumor Segmentation in MRI scans 
[[arxiv](https://arxiv.org/abs/2101.12404)] [[cool](https://papers.cool/arxiv/2101.12404)] [[pdf](https://arxiv.org/pdf/2101.12404)]
> **Authors**: Navchetan Awasthi,Rohit Pardasani,Swati Gupta
> **First submission**: 2021-01-28
> **First announcement**: 2021-01-29
> **comment**: No comments
- **标题**: 基于多模式脑肿瘤分割的MRI扫描中的多峰脑肿瘤分割的多模式的多峰值U-NET（MTAU）
- **领域**: 计算机视觉和模式识别,人工智能,图像和视频处理,医学物理
- **摘要**: 神经胶质瘤是最常见的脑肿瘤之一，分为高级和低级神经胶质瘤。分割各个区域（例如肿瘤核心，增强肿瘤等）在确定严重程度和预后中起着重要作用。在这里，我们开发了一个基于注意力NET的多阈值模型，用于鉴定磁共振成像中肿瘤的各个区域（MRI）。我们提出了一个多路段细分，并为感兴趣的不同区域建立了三个单独的模型。提出的模型在训练数据集上分别达到了增强肿瘤，整个肿瘤和肿瘤核心的平均骰子系数为0.59、0.72和0.61。相同的模型在验证数据集上给出了平均骰子系数为0.57、0.73和0.61，测试数据集的平均骰子系数为0.59、0.72和0.57。

### VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs 
[[arxiv](https://arxiv.org/abs/2101.12059)] [[cool](https://papers.cool/arxiv/2101.12059)] [[pdf](https://arxiv.org/pdf/2101.12059)]
> **Authors**: Xudong Lin,Gedas Bertasius,Jue Wang,Shih-Fu Chang,Devi Parikh,Lorenzo Torresani
> **First submission**: 2021-01-28
> **First announcement**: 2021-01-29
> **comment**: Work in progress
- **标题**: VX2TEXT：从多模式输入中对基于视频的文本生成的端到端学习
- **领域**: 计算机视觉和模式识别,计算语言学
- **摘要**: 我们提出\ textsc {vx2text}，这是一个由视频以及文本，语音或音频组成的多模式输入的文本生成框架。为了利用已被证明在建模语言上有效的变压器网络，首先将每种模式转换为可学习的令牌器转换为一组语言嵌入。这使我们的方法可以在语言空间中执行多模式融合，从而消除了对临时跨模式融合模块的需求。为了解决连续输入（例如视频或音频）上令牌化的非差异性，我们利用一种放松方案来实现端到端培训。此外，与以前的仅编码模型不同，我们的网络包括自动回归解码器，以从语言编码器融合的多模式嵌入中生成开放式文本。这使我们的方法充分生成，使其直接适用于不同的“视频+$ x $”文本问题”问题，而无需为每个任务设计专门的网络头。所提出的框架不仅在概念上很简单，而且非常有效：实验表明，我们基于单个体系结构的方法优于三个基于视频的文本生成任务的最先进 - 字幕，问题答案和视听场景意识到的对话框。

## 信息检索(cs.IR:Information Retrieval)

该领域共有 1 篇论文

### Personalization and Recommendation Technologies for MaaS 
[[arxiv](https://arxiv.org/abs/2101.12335)] [[cool](https://papers.cool/arxiv/2101.12335)] [[pdf](https://arxiv.org/pdf/2101.12335)]
> **Authors**: K. Arnaoutaki,E. Bothos,B. Magoutas,G. Mentzas
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-29
> **comment**: 13 pages
- **标题**: MAAS的个性化和建议技术
- **领域**: 信息检索
- **摘要**: 在过去的几年中，MAA经过广泛的研究和发展为提供多种移动性服务，这些服务不断增加，从替代汽车或自行车共享模式到自动驾驶汽车，渴望成为这一新型生态系统的一部分。 MAA为最终用户提供了多模式，集成和数字移动性解决方案，包括许多不同的选择，能够以个性化的方式满足用户的特定需求。这实际上导致了一系列新型的MAAS产品，这些产品可能具有复杂的结构，并且将它们与用户偏好和需求相匹配的挑战，以便可以向最终用户提供合适的产品。此外，在日常使用MAA时，旅行者需要支持以确定符合其个人喜好的目的地的路线，并与他们购买的MAAS产品保持一致。本文通过利用个性化和推荐系统领域的最新技术来应对这两个以用户为中心的挑战，并将它们集成到MAAS平台和路线计划应用程序中。

## 机器学习(cs.LG:Machine Learning)

该领域共有 14 篇论文

### Combining Graph Neural Networks and Spatio-temporal Disease Models to Predict COVID-19 Cases in Germany 
[[arxiv](https://arxiv.org/abs/2101.00661)] [[cool](https://papers.cool/arxiv/2101.00661)] [[pdf](https://arxiv.org/pdf/2101.00661)]
> **Authors**: Cornelius Fritz,Emilio Dorigatti,David Rügamer
> **First submission**: 2021-01-03
> **First announcement**: 2021-01-04
> **comment**: No comments
- **标题**: 将图形神经网络和时空疾病模型结合在一起，以预测德国的共同病例
- **领域**: 机器学习,应用领域
- **摘要**: 在2020年，来自不同研究领域的许多学者研究了Covid-19的感染率。在这种情况下，对疾病事件的可靠预测是政策制定者管理医疗资源的重要工具。几位专家呼吁有必要解释人类流动性来解释Covid-19的传播。现有的方法通常是采用各自研究领域的标准模型。但是，这种习惯通常会伴随着某些限制。例如，大多数统计或流行病学模型无法直接合并非结构化的数据源，包括可能编码人类移动性的关系数据。相比之下，机器学习方法可以通过利用这些数据结构来产生更好的预测，但由于通常被归类为黑盒模型，因此缺乏直观的解释性。我们提出了两个研究方向之间的权衡，并提出了一种多模式学习方法，该方法结合了统计回归和机器学习模型的优势，用于预测德国本地Covid-19案例。这种新颖的方法可以使用更丰富的数据类型，包括移动性流量和托管概率，并在我们的基准研究中获得整个观察期的MSE得分最低。结果证实了包括移动性数据并展示我们方法的灵活性和解释性的必要性。

### OAAE: Adversarial Autoencoders for Novelty Detection in Multi-modal Normality Case via Orthogonalized Latent Space 
[[arxiv](https://arxiv.org/abs/2101.02358)] [[cool](https://papers.cool/arxiv/2101.02358)] [[pdf](https://arxiv.org/pdf/2101.02358)]
> **Authors**: Sungkwon An,Jeonghoon Kim,Myungjoo Kang,Shahbaz Razaei,Xin Liu
> **First submission**: 2021-01-06
> **First announcement**: 2021-01-07
> **comment**: Accepted to AAAI 2021 Workshop: Towards Robust, Secure and Efficient Machine Learning
- **标题**: OAAE：通过正交的潜在空间在多模式正态案例中进行新颖性检测的对抗性自动编码器
- **领域**: 机器学习,计算机视觉和模式识别,机器学习
- **摘要**: 使用深层生成模型（例如自动编码器）的新颖性检测，生成的对抗网络主要将图像重建误差作为新颖性得分函数。但是，图像数据（较高的维度）包含许多不同的功能，除了类信息以外，这些功能使模型难以检测新颖性数据。在多模式正态案例中，问题越来越困难。为了应对这一挑战，我们提出了一种使用正交的潜在空间来衡量多模式正态案例中新颖性评分的新方法。具体而言，我们在潜在空间中采用正交的低级别嵌入，以使用共同的类信息来解开潜在空间中的特征。随着正交的潜在空间，新颖性得分是由每个潜在向量的变化来定义的。使用GAN（例如Rapp and Ocgan）将提出的算法与最新的新颖性检测算法进行了比较，实验结果表明，我们的表现优于这些算法。

### Cauchy-Schwarz Regularized Autoencoder 
[[arxiv](https://arxiv.org/abs/2101.02149)] [[cool](https://papers.cool/arxiv/2101.02149)] [[pdf](https://arxiv.org/pdf/2101.02149)]
> **Authors**: Linh Tran,Maja Pantic,Marc Peter Deisenroth
> **First submission**: 2021-01-06
> **First announcement**: 2021-01-07
> **comment**: No comments
- **标题**: Cauchy-Schwarz正规自动编码器
- **领域**: 机器学习,计算机视觉和模式识别
- **摘要**: 无监督学习的最新工作集中在潜在变量模型中的有效推理和学习上。通过最大化证据（边际可能性）来训练这些模型通常是棘手的。因此，一个共同的近似是最大化证据下限（ELBO）。变性自动编码器（VAE）是一种功能强大且广泛使用的生成模型类，可为大型数据集有效地优化ELBO。但是，VAE对先前的默认高斯选择对其代表真正后部的能力施加了强烈的限制，从而降低了整体表现。高斯混合模型（GMM）将是更丰富的先验，但由于kullback-leibler的差异对于GMM的差异很难，因此无法在VAE框架内有效处理。我们偏离普通的VAE框架，有利于先验的高斯混合物的分析解决方案。为了对GMM先验进行有效的推断，我们基于Cauchy-Schwarz Divergence引入了一个新的约束目标，可以通过分析GMM进行分析计算。这个新的目标使我们能够将更丰富的多模式先验纳入自动编码框架中。我们提供了有关一系列数据集的经验研究，并表明我们的目标改善了密度估计，无监督聚类，半监督学习和面部分析的各种自动编码模型。

### Learning Intuitive Physics with Multimodal Generative Models 
[[arxiv](https://arxiv.org/abs/2101.04454)] [[cool](https://papers.cool/arxiv/2101.04454)] [[pdf](https://arxiv.org/pdf/2101.04454)]
> **Authors**: Sahand Rezaei-Shoshtari,Francois Robert Hogan,Michael Jenkin,David Meger,Gregory Dudek
> **First submission**: 2021-01-12
> **First announcement**: 2021-01-13
> **comment**: AAAI 2021
- **标题**: 使用多模式生成模型学习直观物理
- **领域**: 机器学习,人工智能
- **摘要**: 预测对象与环境接触时的未来相互作用是自主代理采取智能和预期行动的关键。本文提出了一个感知框架，该框架融合了视觉和触觉反馈，以对动态场景中对象的预期运动进行预测。视觉信息捕获了对象属性，例如3D形状和位置，而触觉信息在与环境接触时提供了有关相互作用力的关键提示和结果对象运动。利用一种新型的透明皮肤（STS）传感器，该传感器提供了高分辨率的多模式感应接触表面，我们的系统既捕获了视觉外观和对象的触觉特性。我们使用多模式变异自动编码器（MVAE）来解释传感器的双流信号，从而使我们能够捕获接触对象的两个方式，并开发从视觉到触觉交互的映射，反之亦然。此外，感知系统可用于推断未来的物理相互作用的结果，我们通过模拟和现实世界实验验证，这些实验可以从给定初始条件从给定对象的静止状态进行预测。

### Multi-Source Anomaly Detection in Distributed IT Systems 
[[arxiv](https://arxiv.org/abs/2101.04977)] [[cool](https://papers.cool/arxiv/2101.04977)] [[pdf](https://arxiv.org/pdf/2101.04977)]
> **Authors**: Jasmin Bogatinovski,Sasho Nedelkoski
> **First submission**: 2021-01-13
> **First announcement**: 2021-01-14
> **comment**: 12 pages. Presented at AIOPS 2020 workshop
- **标题**: 分布式IT系统中的多源异常检测
- **领域**: 机器学习,分布式、并行和集群计算,符号计算,软件工程
- **摘要**: 分布式系统生成的多源数据提供了对系统的整体描述。通过学习模型利用不同方式的联合分布可能对维护分布式系统的关键应用有益。这么重要的任务是我们有兴趣检测系统当前行为与理论上预期的当前行为的偏差的任务。在这项工作中，我们利用分布式轨迹和系统日志数据的联合表示，用于分布式系统中异常检测的任务。我们证明，与单个模态异常检测方法相比，痕迹和对数的联合利用产生了更好的结果。此外，我们将学习任务形式化 - 下一个模板预测NTP，该预测被用作对数和分布式迹线的异常检测的概括。最后，我们证明这种形式化允许学习迹线和日志的模板嵌入。联合嵌入可以在其他应用中重复使用，作为跨度和日志的良好初始化。

### Joint Dimensionality Reduction for Separable Embedding Estimation 
[[arxiv](https://arxiv.org/abs/2101.05500)] [[cool](https://papers.cool/arxiv/2101.05500)] [[pdf](https://arxiv.org/pdf/2101.05500)]
> **Authors**: Yanjun Li,Bihan Wen,Hao Cheng,Yoram Bresler
> **First submission**: 2021-01-14
> **First announcement**: 2021-01-15
> **comment**: No comments
- **标题**: 可分离嵌入估计的关节尺寸降低
- **领域**: 机器学习
- **摘要**: 来自不同源数据的低维嵌入在多模式机器学习，多媒体信息检索和生物信息学中起着关键作用。在本文中，我们提出了一种监督维度缩小方法，该方法将与两个特征向量共同学习，代表不同方式的数据或来自不同类型实体的数据的数据。我们还提出了一种有效的特征选择方法，可以在我们的关节降低方法降低方法之前对补充，并且可以应用。假设这些特征存在真正的线性嵌入，我们对学习线性嵌入中误差的分析提供了理论上的保证，即当满足某些技术条件并且样本数量足够大时，降低降低方法可以准确估算真正的嵌入。得出的样品复杂性结果通过数值实验回荡。我们将提出的维度降低方法应用于基因 - 疾病酶的关联，并在降低降低特征矢量上使用内核回归预测未知关联。我们的方法与其他降低方法相比，与预测基因疾病关联的双线性回归方法的最新方法相比。

### In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning 
[[arxiv](https://arxiv.org/abs/2101.06329)] [[cool](https://papers.cool/arxiv/2101.06329)] [[pdf](https://arxiv.org/pdf/2101.06329)]
> **Authors**: Mamshad Nayeem Rizve,Kevin Duarte,Yogesh S Rawat,Mubarak Shah
> **First submission**: 2021-01-15
> **First announcement**: 2021-01-18
> **comment**: ICLR 2021
- **标题**: 为伪标记的辩护：半监督学习的不确定性伪造标签选择框架
- **领域**: 机器学习,计算机视觉和模式识别
- **摘要**: 半监督学习（SSL）的最新研究主要由基于一致性正规化的方法主导，这些方法实现了强大的性能。但是，它们在很大程度上依赖于特定于域的数据增强，这对于所有数据模式都不容易生成。伪标记（PL）是一种通用的SSL方法，没有这种约束，但其原始表述的性能相对较差。我们认为，由于校准较差的模型的错误高置信度预测，PL表现不佳。这些预测会产生许多不正确的伪标记，从而导致嘈杂的训练。我们提出了一个不确定性感知的伪标签选择（UPS）框架，该框架通过大大减少训练过程中遇到的噪声量来提高伪标记精度。此外，UPS概括了伪标记的过程，从而允许产生负伪标签。这些负伪标签可用于多标签分类以及负面学习，以改善单标签分类。与CIFAR-10和CIFAR-100数据集上的SSL方法相比，我们的性能很强。另外，我们在视频数据集UCF-101和多标签数据集Pascal VOC上演示了方法的多功能性。

### Multimodal Variational Autoencoders for Semi-Supervised Learning: In Defense of Product-of-Experts 
[[arxiv](https://arxiv.org/abs/2101.07240)] [[cool](https://papers.cool/arxiv/2101.07240)] [[pdf](https://arxiv.org/pdf/2101.07240)]
> **Authors**: Svetlana Kutuzova,Oswin Krause,Douglas McCloskey,Mads Nielsen,Christian Igel
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: No comments
- **标题**: 半监督学习的多模式变异自动编码器：防御专家产品
- **领域**: 机器学习,人工智能
- **摘要**: 多模式生成模型应该能够学习有意义的潜在表示，该表示可以使所有模态的连贯共同生成（例如，图像和文本）。许多应用程序还需要能够准确采样以观察模式子集的观测来调节的能力。通常，对于所有培训数据点，并不是所有的方式都可以观察到，因此应该进行半监督学习。在这项研究中，我们提出了具有这些所需特性的新型专家产品（POE）的变异自动编码器。我们将其基准为Experts（MOE）方法和将模态与其他编码器网络相结合的方法进行基准测试。经验评估表明，基于POE的模型可以胜过对比模型。我们的实验支持直觉，即POE模型更适合于形态的结合组合。

### Collaborative Federated Learning For Healthcare: Multi-Modal COVID-19 Diagnosis at the Edge 
[[arxiv](https://arxiv.org/abs/2101.07511)] [[cool](https://papers.cool/arxiv/2101.07511)] [[pdf](https://arxiv.org/pdf/2101.07511)]
> **Authors**: Adnan Qayyum,Kashif Ahmad,Muhammad Ahtazaz Ahsan,Ala Al-Fuqaha,Junaid Qadir
> **First submission**: 2021-01-19
> **First announcement**: 2021-01-20
> **comment**: preprint version
- **标题**: 医疗保健协作联盟学习：边缘的多模式Covid-19诊断
- **领域**: 机器学习,分布式、并行和集群计算
- **摘要**: 尽管在过去几年中取得了重大改善，但基于云的医疗保健应用程序由于限制了严格的安全性，隐私和服务质量要求（例如低延迟），因此仍在采用较差。边缘计算趋势以及用于分布式机器学习（例如联合学习）的技术，在这种情况下已成为可行的解决方案。在本文中，我们通过分析和评估在边缘上智能处理临床视觉数据的潜力，从而使远程医疗保健中心（缺乏先进的诊断设施）可以安全地从多模式数据中受益，从而利用了医学中边缘计算的能力。为此，我们利用了聚集联邦学习（CFL）的新兴概念来自动诊断Covid-19。这种自动化系统可以帮助减轻自2019年末Covid-19大流行以来一直承受着很大压力的全球医疗保健系统的负担。我们评估了在两个基准数据集中不同的实验设置下所提出的框架的性能。在两个数据集上都获得了有希望的结果，从而与中心基线获得了可比的结果，在该基线中，专门模型（即每种模型在特定类型的covid-19图像上）都经过中央数据的培训，并且在多模式模型培训中，在X-Ray的多模型模型培训中，已实现了16 \％和11 \％的改进，在整体F1分数中获得了X-Ray的X-Roods and X-Ray，并在X-Ray上进行了效率。我们还详细讨论了可用于在这种隐私和延迟敏感应用程序中部署ML的相关挑战，技术，工具和技术。

### Ensemble manifold based regularized multi-modal graph convolutional network for cognitive ability prediction 
[[arxiv](https://arxiv.org/abs/2101.08316)] [[cool](https://papers.cool/arxiv/2101.08316)] [[pdf](https://arxiv.org/pdf/2101.08316)]
> **Authors**: Gang Qu,Li Xiao,Wenxing Hu,Kun Zhang,Vince D. Calhoun,Yu-Ping Wang
> **First submission**: 2021-01-20
> **First announcement**: 2021-01-21
> **comment**: No comments
- **标题**: 基于集合歧管的正规化多模式图卷积网络，用于认知能力预测
- **领域**: 机器学习
- **摘要**: 目的：可以使用多模式功能磁共振成像（fMRI）来预测基于大脑连通性网络的个人行为和认知性状。方法：为了利用来自多模式fMRI的互补信息，我们提出了一个可解释的多模式图卷积网络（MGCN）模型，该模型结合了每对大脑区域之间的fMRI时间序列和功能连接性（FC）。具体而言，我们的模型从从多模式数据得出的单个大脑网络中学习了图形嵌入。然后，强制执行基于多方面的正规化项，以考虑模式内和模式之间的受试者的关系。此外，我们提出了梯度加权的回归激活映射（Grad-RAM）和Edge Mask学习以解释模型，该模型用于识别与认知相关的重要生物标志物。结果：我们在费城神经发育队列上验证了我们的MGCN模型，以预测单个广泛的成就测试（WRAT）得分。我们的模型通过单个模式和其他竞争方法获得了优于GCN的优越的预测性能。鉴定出的生物标志物是从不同的方法进行交叉验证的。结论和意义：本文为认知能力预测开发了一个新的可解释的图形深度学习框架，并有可能克服几种当前数据融合模型的局限性。结果表明，MGCN在分析多模式fMRI和发现人脑研究的重要生物标志物方面具有力量。

### Annealed Stein Variational Gradient Descent 
[[arxiv](https://arxiv.org/abs/2101.09815)] [[cool](https://papers.cool/arxiv/2101.09815)] [[pdf](https://arxiv.org/pdf/2101.09815)]
> **Authors**: Francesco D'Angelo,Vincent Fortuin
> **First submission**: 2021-01-24
> **First announcement**: 2021-01-25
> **comment**: No comments
- **标题**: 退火的Stein变分梯度下降
- **领域**: 机器学习
- **摘要**: 基于粒子的优化算法最近被开发为采样方法，迭代地更新一组粒子以近似目标分布。特别是，斯坦因变异梯度下降在近似推理文献中引起了人们的灵活性和准确性的关注。我们从经验上探讨了该方法从多模式分布进行采样的能力，并关注两个重要问题：（i）粒子无法逃脱局部模式，以及（ii）再现不同区域密度的效率低下。我们提出了一个退火时间表，以解决这些问题，并通过各种实验表明这种简单的解决方案如何导致模式覆盖率的显着改善，而不会使原始算法的任何理论特性无效。

### A Case Study of Deep Learning Based Multi-Modal Methods for Predicting the Age-Suitability Rating of Movie Trailers 
[[arxiv](https://arxiv.org/abs/2101.11704)] [[cool](https://papers.cool/arxiv/2101.11704)] [[pdf](https://arxiv.org/pdf/2101.11704)]
> **Authors**: Mahsa Shafaei,Christos Smailis,Ioannis A. Kakadiaris,Thamar Solorio
> **First submission**: 2021-01-26
> **First announcement**: 2021-01-28
> **comment**: No comments
- **标题**: 基于深度学习的多模式方法的案例研究，以预测电影预告片的年龄介绍评级
- **领域**: 机器学习,多媒体,声音,音频和语音处理,图像和视频处理
- **摘要**: 在这项工作中，我们探索了不同的方法，以结合电影预告片的自动化年龄评级问题的方式。首先，我们介绍了一个新的数据集，其中包含从IMDB和YouTube下载的英文电影预告片的视频，以及它们相应的年龄介绍评级标签。其次，我们提出了一条多模式深度学习管道，以解决电影预告片年龄适合性评级问题。这是将视频，音频和语音信息结合到此问题的第一次尝试，我们的实验结果表明，多模式方法在此任务中大大优于最佳的单声道和双峰模型。

### Learning Abstract Representations through Lossy Compression of Multi-Modal Signals 
[[arxiv](https://arxiv.org/abs/2101.11376)] [[cool](https://papers.cool/arxiv/2101.11376)] [[pdf](https://arxiv.org/pdf/2101.11376)]
> **Authors**: Charles Wilmot,Gianluca Baldassarre,Jochen Triesch
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: No comments
- **标题**: 通过多模式信号的有损压缩来学习抽象表示
- **领域**: 机器学习,人工智能,多媒体
- **摘要**: 开放式学习的关键能力是形成日益抽象的表示，可用于推动复杂行为。抽象表示忽略了特定的细节，并促进概括。在这里，我们考虑以两种或多种输入方式的多模式设置中的抽象表示。我们将问题视为一个有损的压缩问题，并表明多模式感觉输入的通用有损压缩自然提取抽象表示，这些表示倾向于剥离模态特定的特定细节，并优先保留在不同模态上共享的信息。此外，我们提出了一个体系结构来学习抽象表示，通过仅识别和保留跨多种模式共享的信息，同时丢弃任何模式的特定信息。

### A Machine Learning Challenge for Prognostic Modelling in Head and Neck Cancer Using Multi-modal Data 
[[arxiv](https://arxiv.org/abs/2101.11935)] [[cool](https://papers.cool/arxiv/2101.11935)] [[pdf](https://arxiv.org/pdf/2101.11935)]
> **Authors**: Michal Kazmierski,Mattea Welch,Sejin Kim,Chris McIntosh,Princess Margaret Head,Neck Cancer Group,Katrina Rey-McIntyre,Shao Hui Huang,Tirth Patel,Tony Tadic,Michael Milosevic,Fei-Fei Liu,Andrew Hope,Scott Bratman,Benjamin Haibe-Kains
> **First submission**: 2021-01-28
> **First announcement**: 2021-01-29
> **comment**: 27 pages, 7 figures, under review
- **标题**: 使用多模式数据的头和颈癌预后建模的机器学习挑战
- **领域**: 机器学习,图像和视频处理
- **摘要**: 单个患者的准确预后是精度肿瘤学的关键组成部分。机器学习的最新进展使使用更广泛的数据（包括成像）可以开发模型。放射素学旨在从常规的医学成像中提取定量预测性和预后生物标志物，但是计算机断层扫描放射组学的预后证据仍然尚无定论。我们已经进行了机构学习挑战，以使用从电子病历和治疗前放射学图像中脱颖而出的临床数据来开发一个准确的头颈癌存活预测模型，并评估头部和颈部癌症对放射素的真正额外益处。使用2,552例患者的大型回顾性数据集和一个严格的评估框架，我们使用成像和临床数据分别或组合比较了12种不同的提交。获胜方法在临床数据和肿瘤体积上使用了非线性多任务学习，在2年和终生生存预测中实现了高预后的准确性，并且超越了仅依靠临床数据，工程性放射线学和深度学习的模型。合并模型中的所有提交结合，从而提高了准确性，并从基于图像的深度学习模型中获得最高增长。我们的结果表明，机器学习的潜力和简单，信息丰富的预后因素与大型数据集结合在一起，作为指导个性化癌症护理的工具。

## 多媒体(cs.MM:Multimedia)

该领域共有 1 篇论文

### The Multimodal Sentiment Analysis in Car Reviews (MuSe-CaR) Dataset: Collection, Insights and Improvements 
[[arxiv](https://arxiv.org/abs/2101.06053)] [[cool](https://papers.cool/arxiv/2101.06053)] [[pdf](https://arxiv.org/pdf/2101.06053)]
> **Authors**: Lukas Stappen,Alice Baird,Lea Schumann,Björn Schuller
> **First submission**: 2021-01-15
> **First announcement**: 2021-01-18
> **comment**: accepted version
- **标题**: 汽车评论（Muse-Car）数据集中的多模式情感分析：收集，洞察力和改进
- **领域**: 多媒体,计算语言学
- **摘要**: 真正的现实生活数据为情感和情感研究带来了强烈但令人兴奋的挑战。在建立强大的机器学习模型方面，种类繁多的“野外”属性使大型数据集（例如这些数据集）都是必不可少的。在这种情况下，尚未提供足够数量的数据，这些数据涵盖了每种模式的挑战，以迫使所有模式相互作用的探索性分析。在这项贡献中，我们介绍了穆塞尔汽车，这是同类多模式数据集中的第一个。这些数据是公开可用的，因为它最近作为第一次多模式分析挑战的测试床，并专注于情感，情感目标参与的任务以及通过全面整合音频视频和语言方式的信任识别。此外，我们在收集和注释方面对数据集进行了详尽的概述，包括在今年的穆斯2020年未使用的注释层。此外，对于一个子挑战者之一而言，对于一个子挑战者之一 - 可以预测参与者的基线模型，因此我们在基线模型上没有一个简单但高效的远程量，因此我们超过了。超过0.超过0的跨度额度，该网络超过了。 （几乎改善了50％）。

## 机器人技术(cs.RO:Robotics)

该领域共有 5 篇论文

### A Hybrid Learner for Simultaneous Localization and Mapping 
[[arxiv](https://arxiv.org/abs/2101.01158)] [[cool](https://papers.cool/arxiv/2101.01158)] [[pdf](https://arxiv.org/pdf/2101.01158)]
> **Authors**: Thangarajah Akilan,Edna Johnson,Japneet Sandhu,Ritika Chadha,Gaurav Taluja
> **First submission**: 2021-01-04
> **First announcement**: 2021-01-05
> **comment**: No comments
- **标题**: 用于同时本地化和映射的混合学习者
- **领域**: 机器人技术,机器学习
- **摘要**: 同时定位和映射（SLAM）用于根据位置坐标和物理环境的精确映射预测移动平台的动态运动路径。 SLAM在增强现实（AR），自动驾驶汽车中具有巨大潜力。自动驾驶汽车，无人机，自主导航机器人（ANR）。这项工作介绍了一种混合学习模型，该模型探讨了功能融合的超出特征，并进行了多模式的缝纫策略，以改善基线SLAM算法的性能。它通过不同深网的顶层突变来实现猛击的前端特征提取器的重量。同时，将独立训练模型的轨迹预测合并以完善位置细节。因此，在混合学习框架下，上述早期和晚期融合技术的整合最小化了SLAM模型的翻译和旋转误差。这项研究利用了一些众所周知的深度学习（DL）结构，包括RESNET18，RESNET34，RESNET50，RESNET101，VGG16，VGG16，VGG19和ALEXNET进行实验分析。一项广泛的实验分析证明，混合学习者（HL）取得的结果明显好于具有早期或晚融合策略的单峰方法和多模式方法。因此，发现这项工作中采用的Apolloscape数据集从未在文献中使用融合技术来实现，这使这项工作变得独特而有见地。

### Investigating the Effect of Sensor Modalities in Multi-Sensor Detection-Prediction Models 
[[arxiv](https://arxiv.org/abs/2101.03279)] [[cool](https://papers.cool/arxiv/2101.03279)] [[pdf](https://arxiv.org/pdf/2101.03279)]
> **Authors**: Abhishek Mohta,Fang-Chieh Chou,Brian C. Becker,Carlos Vallespi-Gonzalez,Nemanja Djuric
> **First submission**: 2021-01-08
> **First announcement**: 2021-01-11
> **comment**: No comments
- **标题**: 研究传感器模式在多传感器检测预测模型中的影响
- **领域**: 机器人技术,计算机视觉和模式识别,机器学习
- **摘要**: 检测周围物体及其运动预测是自动驾驶系统的关键组成部分。最近提出的共同解决这些任务的模型依赖于许多传感器来实现最新性能。但是，这增加了系统的复杂性，并可能导致一个脆性模型，该模型在忽略其他传感器的同时过度适合任何单个传感器模式，从而减少了概括。我们专注于这个重要的问题，并分析传感器方式对模型性能的贡献。此外，我们研究了传感器辍学的使用来减轻上述问题，从而导致对现实世界驾驶数据的更强大，更有表现的模型。

### droidlet: modular, heterogenous, multi-modal agents 
[[arxiv](https://arxiv.org/abs/2101.10384)] [[cool](https://papers.cool/arxiv/2101.10384)] [[pdf](https://arxiv.org/pdf/2101.10384)]
> **Authors**: Anurag Pratik,Soumith Chintala,Kavya Srinet,Dhiraj Gandhi,Rebecca Qian,Yuxuan Sun,Ryan Drew,Sara Elkafrawy,Anoushka Tiwari,Tucker Hart,Mary Williamson,Abhinav Gupta,Arthur Szlam
> **First submission**: 2021-01-25
> **First announcement**: 2021-01-26
> **comment**: No comments
- **标题**: 机器人：模块化，异质，多峰剂
- **领域**: 机器人技术,人工智能
- **摘要**: 近年来，在大规模学习的端到端机器学习（ML）系统方面取得了重大进展。但是这些系统中的大多数是：（a）孤立（仅感知，语音或语言）； （b）在静态数据集上训练。另一方面，在机器人技术领域，大规模学习一直很困难。监督很难收集，现实世界的身体互动很昂贵。在这项工作中，我们介绍和开源机器人，这是一个模块化的，异构的代理体系结构和平台。它使我们能够在感知和语言中利用大规模的静态数据集，并且经常用于机器人技术；并提供交互式注释的工具。此外，它将感知，语言和行动汇集到一个平台上，为代理提供了从现实世界互动的丰富性中学习的道路。

### Autonomous Off-road Navigation over Extreme Terrains with Perceptually-challenging Conditions 
[[arxiv](https://arxiv.org/abs/2101.11110)] [[cool](https://papers.cool/arxiv/2101.11110)] [[pdf](https://arxiv.org/pdf/2101.11110)]
> **Authors**: Rohan Thakker,Nikhilesh Alatur,David D. Fan,Jesus Tordesillas,Michael Paton,Kyohei Otsu,Olivier Toupet,Ali-akbar Agha-mohammadi
> **First submission**: 2021-01-26
> **First announcement**: 2021-01-27
> **comment**: 12 Pages, 7 Figures, 2020 International Symposium on Experimental Robotics (ISER 2020)
- **标题**: 在具有感知挑战条件的极端地形上的自动越野航行
- **领域**: 机器人技术,人工智能,系统与控制
- **摘要**: 我们提出了一个框架，用于在感知挑战的未知环境中具有弹性的自主导航框架，其中具有巨大的运动元素，例如带有岩石和巨石的不平衡表面，陡峭的斜坡，悬崖和孔等负面障碍以及狭窄的通道。环境受到全科医生的约束和感知的衰落，并从黑暗到点燃和晦涩的剂（灰尘，雾，烟）的可变照明。缺乏先前的地图和退化的通信消除了事先或外部计算或操作员干预的可能性。这需要使用嘈杂的传感器数据实时计算实时计算。为了应对这些挑战，我们提出了一种有弹性的体系结构，该体系结构利用感应方式中的冗余和异质性。通过在失败时触发恢复行为来实现进一步的弹性。我们提出了一种快速的沉降算法，以实时生成鲁棒的多效率遍历性估计。拟议的方法部署在多个物理系统上，包括Skid-Steer和Tracked Robots，高速RC汽车和腿部机器人，这是Team Costar向DARPA Subterranean挑战所努力的一部分，该团队分别赢得了隧道和隧道和城市路线的第二名和第一名。

### Enabling Robots to Draw and Tell: Towards Visually Grounded Multimodal Description Generation 
[[arxiv](https://arxiv.org/abs/2101.12338)] [[cool](https://papers.cool/arxiv/2101.12338)] [[pdf](https://arxiv.org/pdf/2101.12338)]
> **Authors**: Ting Han,Sina Zarrieß
> **First submission**: 2021-01-14
> **First announcement**: 2021-01-29
> **comment**: The 2nd Workshop on NLG for HRI colocated with The 13th International Conference on Natural Language Generation
- **标题**: 使机器人能够绘制和讲述：迈向视觉接地的多模式描述生成
- **领域**: 机器人技术,人工智能
- **摘要**: 具有社会能力的机器人应具有感知周围世界并以人类方式进行交流的世界的能力。表现出这种能力的代表性技能包括生成图像描述和视觉扎根的参考表达式。在NLG社区中，这些一代任务在不相互互动和仅使用语言的环境中进行了很大的研究。但是，在面对面的互动中，人类经常部署多种方式进行交流，形成自然语言，手势和其他方式（如草图）的无缝集成。为了使机器人能够描述他们对语音和草图/手势的看法，我们建议建模生成自然语言的任务，以及自由手素描/手势，以描述视觉场景和现实生活对象，即视觉上，视觉上的多模式描述生成。在本文中，我们讨论了该任务的挑战和评估指标，以及该任务如何从自然语言处理和计算机视觉领域中取得的进展中受益，这些主题（例如视觉上接地的NLG，分布式语义和基于照片的素描生成）进行了广泛的研究。

## 社交和信息网络(cs.SI:Social and Information Networks)

该领域共有 1 篇论文

### A multi-modal approach towards mining social media data during natural disasters -- a case study of Hurricane Irma 
[[arxiv](https://arxiv.org/abs/2101.00480)] [[cool](https://papers.cool/arxiv/2101.00480)] [[pdf](https://arxiv.org/pdf/2101.00480)]
> **Authors**: Somya D. Mohanty,Brown Biggers,Saed Sayedahmed,Nastaran Pourebrahim,Evan B. Goldstein,Rick Bunch,Guangqing Chi,Fereidoon Sadri,Tom P. McCoy,Arthur Cosby
> **First submission**: 2021-01-02
> **First announcement**: 2021-01-04
> **comment**: 46 pages, 11 Figures
- **标题**: 自然灾害期间采矿社交媒体数据的一种多模式方法 - 飓风Irma的案例研究
- **领域**: 社交和信息网络,信息检索,机器学习
- **摘要**: 流媒体社交媒体可实时了解极端天气的影响。但是，流媒体数据的数量使采矿信息成为应急管理者，政策制定者和纪律科学家的挑战。在这里，我们探讨了从美国佛罗里达州飓风Irma登陆的流媒体社交媒体数据中学习和过滤信息的数据的有效性。 We use 54,383 Twitter messages (out of 784K geolocated messages) from 16,598 users from Sept. 10 - 12, 2017 to develop 4 independent models to filter data for relevance: 1) a geospatial model based on forcing conditions at the place and time of each tweet, 2) an image classification model for tweets that include images, 3) a user model to predict the reliability of the tweeter, and 4) a text model to determine if the text is与飓风Irma有关。所有四个模型均经过独立测试，并且可以合并以快速过滤并根据每个子模型的用户定义的阈值进行过滤和可视化推文。我们设想，这种类型的过滤和可视化例程可作为来自Twitter等嘈杂来源的数据捕获的基本模型。然后，数据随后可以由政策制定者，环境经理，应急管理人员和领域科学家使用，有兴趣在灾难的不同阶段（例如，准备，准备，响应和恢复）或进行详细研究。

## 图像和视频处理(eess.IV:Image and Video Processing)

该领域共有 7 篇论文

### Brain Tumor Segmentation and Survival Prediction using Automatic Hard mining in 3D CNN Architecture 
[[arxiv](https://arxiv.org/abs/2101.01546)] [[cool](https://papers.cool/arxiv/2101.01546)] [[pdf](https://arxiv.org/pdf/2101.01546)]
> **Authors**: Vikas Kumar Anand,Sanjeev Grampurohit,Pranav Aurangabadkar,Avinash Kori,Mahendra Khened,Raghavendra S Bhat,Ganapathy Krishnamurthi
> **First submission**: 2021-01-05
> **First announcement**: 2021-01-06
> **comment**: 11 pages, 4 Figures
- **标题**: 在3D CNN体系结构中使用自动硬采矿的脑肿瘤分割和生存预测
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 我们利用3-D完全卷积神经网络（CNN）从多模式磁共振图像（MRI）中进行分段神经胶质瘤及其成分。该体系结构使用密集的连接模式来减少权重和剩余连接的数量，并通过使用Brats 2018数据集训练该模型获得的权重初始化。在训练期间，通过增加骰子相似性系数（DSC）阈值来训练在训练期间进行训练，以训练难以进行分割任务的困难案例。在BRATS2020验证数据（n = 125）上，该体系结构的肿瘤核，整个肿瘤和活性肿瘤骰子分别为0.744，0.876，0.714。在测试数据集上，我们在肿瘤核心的DSC和活性肿瘤的DSC中增加了约7％。在DSC方面，我们在BRAT 2020测试数据上的网络性能分别为0.775、0.815和0.85，用于增强肿瘤，肿瘤核心和整个肿瘤。使用传统的机器学习从使用生成的分割面罩获得的常规机器学习来确定受试者的总生存。我们的方法已达到0.448和0.452作为验证和测试数据集的精度。

### A Unified Conditional Disentanglement Framework for Multimodal Brain MR Image Translation 
[[arxiv](https://arxiv.org/abs/2101.05434)] [[cool](https://papers.cool/arxiv/2101.05434)] [[pdf](https://arxiv.org/pdf/2101.05434)]
> **Authors**: Xiaofeng Liu,Fangxu Xing,Georges El Fakhri,Jonghye Woo
> **First submission**: 2021-01-13
> **First announcement**: 2021-01-14
> **comment**: Published in IEEE International Symposium on Biomedical Imaging (ISBI) 2021 for Oral presentation
- **标题**: 多模式大脑MR图像翻译的统一条件分解框架
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 多模式MRI为探测组织状况和表征各种疾病提供互补和临床相关的信息。但是，由于研究计划的限制，通常很难从同一主题中获得足够多的方式，而仍然需要定量分析。在这项工作中，我们提出了一个统一的条件解开框架，以从输入方式中综合任何任意模式。我们的框架取决于循环受限的条件对抗训练方法，在这种方法中，它可以使用模态 - 诺斯替编码器提取模态不变的解剖学特征，并使用条件解码器生成目标模态。我们从BRATS'18数据库中验证了四种MRI模式的框架，包括T1加权，T1对比增强，T2加权和FLAIR MRI，在合成质量上表现出优于比较方法的表现。此外，我们报告了对使用合成数据执行的肿瘤分割任务实验的结果。

### VoxelHop: Successive Subspace Learning for ALS Disease Classification Using Structural MRI 
[[arxiv](https://arxiv.org/abs/2101.05131)] [[cool](https://papers.cool/arxiv/2101.05131)] [[pdf](https://arxiv.org/pdf/2101.05131)]
> **Authors**: Xiaofeng Liu,Fangxu Xing,Chao Yang,C. -C. Jay Kuo,Suma Babu,Georges El Fakhri,Thomas Jenkins,Jonghye Woo
> **First submission**: 2021-01-13
> **First announcement**: 2021-01-14
> **comment**: No comments
- **标题**: VoxelHop：使用结构MRI进行ALS疾病分类的连续子空间学习
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 深度学习具有通过医学成像数据准确检测和分类疾病的巨大潜力，但是表现通常受到培训数据集和内存要求的限制。此外，许多深度学习模型被认为是“黑框”，因此通常会限制其在临床应用中的采用。为了解决这个问题，我们提出了一种称为VoxelHop的连续子空间学习模型，以使用T2加权结构MRI数据对肌萎缩性侧面硬化症（ALS）进行准确分类。与流行的卷积神经网络（CNN）结构相比，VoxelHop具有模块化和透明的结构，参数较少而没有任何反向传播，因此非常适合小型数据集大小和3D成像数据。我们的VoxelHop具有四个关键组件，包括（1）用于多通道3D数据的接近到FAR社区的连续扩展； （2）无监督尺寸降低的子空间近似； （3）标记辅助回归以减小监督维度； （4）对照与患者之间的特征和分类的串联。我们的实验结果表明，我们使用20个对照和26名患者的框架的准确度为93.48 $ \％$，而在对照组中分开的患者的AUC得分为0.9394，即使数据集相对较少，其稳健性和有效性也相对较少。我们的详尽评估还表明了其对最先进的3D CNN分类方法的有效性和优越性。我们的框架可以轻松地使用不同的成像方式将其推广到其他分类任务。

### Symmetric-Constrained Irregular Structure Inpainting for Brain MRI Registration with Tumor Pathology 
[[arxiv](https://arxiv.org/abs/2101.06775)] [[cool](https://papers.cool/arxiv/2101.06775)] [[pdf](https://arxiv.org/pdf/2101.06775)]
> **Authors**: Xiaofeng Liu,Fangxu Xing,Chao Yang,C. -C. Jay Kuo,Georges ElFakhri,Jonghye Woo
> **First submission**: 2021-01-17
> **First announcement**: 2021-01-18
> **comment**: Published at MICCAI Brainles 2020
- **标题**: 与肿瘤病理学的对称约束的不规则结构介绍脑MRI注册
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 脑肿瘤患者和健康受试者之间的磁共振图像的可变形注册已成为通过位置比对指定肿瘤几何形状并促进病理分析的重要工具。由于肿瘤区域与任何普通的脑组织不匹配，因此很难将患者的大脑畸形为正常。许多患者图像与不规则分布的病变有关，导致正常组织结构进一步失真，并使注册的相似性度量变得复杂。在这项工作中，我们遵循多步中的情境感知图像介绍框架，以在肿瘤区域产生合成组织强度。粗图像到图像翻译用于对缺失部分的粗略推断。然后，应用功能级匹配匹配改进模块，以通过对贴片功能之间的语义相关性进行建模来完善细节。进一步提出了反映大脑中大量解剖对称性的对称约束，以获得更好的结构理解。在原始的患者图像和正常的大脑之间应用可变形的注册，最终将所得的变形场用于变形原始患者数据以进行最终排列。该方法应用于多模式脑肿瘤分割（BRATS）2018挑战数据库，并与三种现有的涂鸦方法进行了比较。提出的方法得出了峰值信噪比增加，结构相似性指数，建立评分和L1误差的增加，从而导致成功的患者到正常的大脑图像注册。

### Feature Fusion of Raman Chemical Imaging and Digital Histopathology using Machine Learning for Prostate Cancer Detection 
[[arxiv](https://arxiv.org/abs/2101.07342)] [[cool](https://papers.cool/arxiv/2101.07342)] [[pdf](https://arxiv.org/pdf/2101.07342)]
> **Authors**: Trevor Doherty,Susan McKeever,Nebras Al-Attar,Tiarnan Murphy,Claudia Aura,Arman Rahman,Amanda O'Neill,Stephen P Finn,Elaine Kay,William M. Gallagher,R. William G. Watson,Aoife Gowen,Patrick Jackman
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: 19 pages, 8 tables, 18 figures
- **标题**: 使用机器学习前列腺癌检测的拉曼化学成像和数字组织病理学的特征融合
- **领域**: 图像和视频处理,机器学习,定量方法
- **摘要**: 由于其表现的异质性，前列腺癌的诊断是具有挑战性的，导致过度诊断和治疗非临床重要的疾病。准确的诊断可以直接受益于患者的生活质量和预后。为了解决这个问题，我们提出了一种自动鉴定前列腺癌的学习模型。尽管许多前列腺癌研究已经采用了拉曼光谱法，但没有人利用拉曼化学成像（RCI）和其他成像方式的组合。这项研究使用由染色数字组织病理学（DP）和未染色的RCI形成的多模式图像。该方法是在来自32名患者的178个临床样本中开发和测试的，其中包含一系列非癌性，Gleason 3级（G3）和4级（G4）组织微阵列样品。对于每个组织学样本，都有一个病理学家标记为DP -RCI图像对。测试的假设是多模式模型是否可以在诊断准确性方面胜过单一模态基线模型。研究了二进制非癌/癌症模型和更具挑战性的G3/G4分化。关于G3/G4分类，多模式方法的灵敏度为73.8％，特异性为88.1％，而基线DP模型的灵敏度分别为54.1％和84.7％。多模式方法表明，比基线具有统计学意义的12.7％AUC优势，值为85.8％，而73.1％的AUC优势也仅基于RCI和中间拉曼光谱，却超过了模型。 DP和RCI的特征融合并不能改善肿瘤鉴定的更琐碎的任务，而是在G3/G4歧视中具有观察到的优势。在这些有希望的发现的基础上，未来的工作可能包括获取较大的数据集以增强模型概括。

### Comparing Deep Learning strategies for paired but unregistered multimodal segmentation of the liver in T1 and T2-weighted MRI 
[[arxiv](https://arxiv.org/abs/2101.06979)] [[cool](https://papers.cool/arxiv/2101.06979)] [[pdf](https://arxiv.org/pdf/2101.06979)]
> **Authors**: Vincent Couteaux,Mathilde Trintignac,Olivier Nempont,Guillaume Pizaine,Anna Sesilia Vlachomitrou,Pierre-Jean Valette,Laurent Milot,Isabelle Bloch
> **First submission**: 2021-01-18
> **First announcement**: 2021-01-19
> **comment**: 4 pages, 3 figures and 3 tables. Conference paper
- **标题**: 比较T1和T2加权MRI中肝脏的配对但未注册的多模式分割的深度学习策略
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 我们解决了配对但未注册的T1和T2加权MR图像中多模式肝分割的问题。我们比较文献中描述的几种策略，无论是否有多任务培训，有或没有预注册。我们还比较了不同的损失函数（横向倾向，骰子损失和三个对抗损失）。除多任务设置外，所有方法均可达到可比的性能，该设置同时执行两种分割，表现不佳。

### Automatic Segmentation of Gross Target Volume of Nasopharynx Cancer using Ensemble of Multiscale Deep Neural Networks with Spatial Attention 
[[arxiv](https://arxiv.org/abs/2101.11254)] [[cool](https://papers.cool/arxiv/2101.11254)] [[pdf](https://arxiv.org/pdf/2101.11254)]
> **Authors**: Haochen Mei,Wenhui Lei,Ran Gu,Shan Ye,Zhengwentai Sun,Shichuan Zhang,Guotai Wang
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: No comments
- **标题**: 使用多尺度的深神经网络合奏对鼻咽癌的总目标体积自动分割，并引起空间的关注
- **领域**: 图像和视频处理,计算机视觉和模式识别
- **摘要**: 放疗是鼻咽癌的主要治疗方式。从CT和MRI图像等医学图像中描述总目标体积（GTV）是放射疗法的先决条件。由于手动描述是耗时且费力的，因此GTV的自动分割具有改善此过程的潜力。当前，GTV的大多数基于深度学习的自动描述方法主要在CT图像之类的医学图像上执行。然而，这是由于病理区域与周围软组织，小目标区域以及临床CT图像的各向异性分辨率之间的低对比度所挑战。为了解决这些问题，我们提出了一个2.5D卷积神经网络（CNN），以处理平面内和平面分辨率的差异。此外，我们提出了一个空间注意模块，以使网络能够专注于小目标，并使用渠道注意力来进一步提高分割性能。此外，我们使用多尺度抽样方法进行训练，以便网络可以在不同尺度上学习功能，这些功能与多模型集合方法结合使用，以提高分割结果的鲁棒性。我们还根据我们的模型集合估算了分割结果的不确定性，这对于表明放射治疗计划的自动分割结果的可靠性至关重要。

## 信号处理(eess.SP:Signal Processing)

该领域共有 1 篇论文

### A Novel Multi-Stage Training Approach for Human Activity Recognition from Multimodal Wearable Sensor Data Using Deep Neural Network 
[[arxiv](https://arxiv.org/abs/2101.00702)] [[cool](https://papers.cool/arxiv/2101.00702)] [[pdf](https://arxiv.org/pdf/2101.00702)]
> **Authors**: Tanvir Mahmud,A. Q. M. Sazzad Sayyed,Shaikh Anowarul Fattah,Sun-Yuan Kung
> **First submission**: 2021-01-03
> **First announcement**: 2021-01-04
> **comment**: 12 Pages, 7 Figures. This article has been published in IEEE Sensors Journal
- **标题**: 使用深神经网络从多模式可穿戴传感器数据中识别人类活动的新型多阶段训练方法
- **领域**: 信号处理,机器学习
- **摘要**: 深度神经网络是一种有效的选择，可以使用来自各种可穿戴传感器的数据自动识别人类行动。这些网络完全依赖于数据自动化特征提取过程。但是，时间序列数据中的各种噪音在传感器之间具有复杂的模式间关系，使该过程更加复杂。在本文中，我们提出了一种新型的多阶段训练方法，该方法在此特征提取过程中增加了多样性，以通过结合从不同角度提取的特征来准确地识别动作。最初，在时间序列数据中使用了许多转换，而不是使用单一类型的转换，以获取原始数据中编码的功能的杂色表示。提出了一个有效的深CNN体系结构，可以单独训练，以从不同变换的空间中提取特征。后来，将这些CNN特征提取器合并为最佳架构，通过合并的训练阶段或多个顺序训练阶段进行了精心调整，以优化多样化的提取特征。这种方法提供了机会，可以利用具有多种观察窗口的原始传感器数据中的编码功能，并具有巨大的范围，以有效地选择最终收敛的功能。在三个公开可用的数据集中进行了广泛的实验，这些数据集在UCI HAR数据库上提供了出色的性能，平均五倍的交叉验证精度为99.29％，USC HAR数据库的99.02％，在SKODA数据库上超过了其他州的其他州的数据库，为97.21％。

## 地球物理学(physics.geo-ph:Geophysics)

该领域共有 1 篇论文

### HypoSVI: Hypocenter inversion with Stein variational inference and Physics Informed Neural Networks 
[[arxiv](https://arxiv.org/abs/2101.03271)] [[cool](https://papers.cool/arxiv/2101.03271)] [[pdf](https://arxiv.org/pdf/2101.03271)]
> **Authors**: Jonathan D. Smith,Zachary E. Ross,Kamyar Azizzadenesheli,Jack B. Muir
> **First submission**: 2021-01-08
> **First announcement**: 2021-01-11
> **comment**: Updating to accepted version of the paper
- **标题**: Hybosvi：使用Stein变异推理和物理学知情神经网络的偏置反转
- **领域**: 地球物理学,机器学习
- **摘要**: 我们介绍了一种使用Stein变异推断的概率低中心反转的方案。我们的方法以物理知情的神经网络的形式使用了可区分的远期模型，我们训练该模型以解决Eikonal方程。这允许通过迭代优化粒子的集合来快速近似后部，以二键率优化粒子的集合。我们表明，该方法具有良好的能力处理高度多模式后分布，这在次心逆问题中很常见。进行了一套实验，以检查各种超参数的影响。一旦受过培训，该方法对于研究区域内的任何地震网络几何形状有效，而无需构建旅行时间表。我们表明，计算需求随着差异时间的数量有效地扩展，因此它非常适合分布式声传感等大N传感技术。本手稿中概述的技术不仅具有射线追踪过程，还具有相当大的含义，其工作流程适用于其他具有计算昂贵的反转过程（例如Full Wove倒置）的字段。

## 定量方法(q-bio.QM:Quantitative Methods)

该领域共有 1 篇论文

### G-MIND: An End-to-End Multimodal Imaging-Genetics Framework for Biomarker Identification and Disease Classification 
[[arxiv](https://arxiv.org/abs/2101.11656)] [[cool](https://papers.cool/arxiv/2101.11656)] [[pdf](https://arxiv.org/pdf/2101.11656)]
> **Authors**: Sayan Ghosal,Qiang Chen,Giulio Pergola,Aaron L. Goldman,William Ulrich,Karen F. Berman,Giuseppe Blasi,Leonardo Fazio,Antonio Rampino,Alessandro Bertolino,Daniel R. Weinberger,Venkata S. Mattay,Archana Venkataraman
> **First submission**: 2021-01-27
> **First announcement**: 2021-01-28
> **comment**: No comments
- **标题**: G-Mind：用于生物标记和疾病分类的端到端多模式成像基因框架
- **领域**: 定量方法,机器学习,图像和视频处理
- **摘要**: 我们提出了一种新型的深层神经网络结构，以在诊断为指导下整合成像和遗传学数据，该数据提供了可解释的生物标志物。我们的模型由编码器，解码器和分类器组成。编码器学习输入数据模式之间共享的非线性子空间。分类器和解码器充当正规化器，以确保低维编码捕获患者和对照之间的预测差异。我们使用可学习的辍学层从数据中提取可解释的生物标志物，我们独特的培训策略可以轻松适应跨科目中缺少的数据方式。我们已经评估了包括两个功能性MRI（fMRI）范式和单核苷酸多态性（SNP）数据的精神分裂症的人群研究模型。使用10倍的交叉验证，我们证明我们的模型比基线方法实现了更好的分类精度，并且该性能将其推广到在其他站点上收集的第二个数据集。在探索性分析中，我们进一步表明，我们的模型确定的生物标志物与精神分裂症中有据可查的缺陷密切相关。

## 其他论文

共有 18 篇其他论文

- [Robot Adaptation for Generating Consistent Navigational Behaviors over Unstructured Off-Road Terrain](https://arxiv.org/abs/2101.00290)
  - **标题**: 机器人改编，以通过非结构化的越野地形产生一致的导航行为
  - **Filtered Reason**: none of cs.RO in whitelist
- [Towards Cross-Modal Forgery Detection and Localization on Live Surveillance Videos](https://arxiv.org/abs/2101.00848)
  - **标题**: 在实时监视视频中进行跨模式伪造的检测和本地化
  - **Filtered Reason**: none of cs.CR in whitelist
- [A method for nonlinear modal analysis and synthesis: Application to harmonically forced and self-excited mechanical systems](https://arxiv.org/abs/2101.01804)
  - **标题**: 一种非线性模态分析和综合的方法：应用于和谐强迫和自启动的机械系统
  - **Filtered Reason**: none of math.NA,cs.CE in whitelist
- [Playing with Food: Learning Food Item Representations through Interactive Exploration](https://arxiv.org/abs/2101.02252)
  - **标题**: 与食物一起玩：通过互动探索学习食品的表示形式
  - **Filtered Reason**: none of cs.RO in whitelist
- [Incentive Design and Profit Sharing in Multi-modal Transportation Network](https://arxiv.org/abs/2101.03297)
  - **标题**: 在多模式运输网络中的激励设计和利润共享
  - **Filtered Reason**: none of eess.SY,cs.GT in whitelist
- [A Rewriting Logic Approach to Specification, Proof-search, and Meta-proofs in Sequent Systems](https://arxiv.org/abs/2101.03113)
  - **标题**: 重写的逻辑方法，用于序列系统中的规范，证明搜索和元数据范围
  - **Filtered Reason**: none of cs.LO,math.LO in whitelist
- [Structural Analysis of Multimode DAE Systems: summary of results](https://arxiv.org/abs/2101.05702)
  - **标题**: 多模DAE系统的结构分析：结果摘要
  - **Filtered Reason**: none of cs.PL in whitelist
- [Data@Hand: Fostering Visual Exploration of Personal Data on Smartphones Leveraging Speech and Touch Interaction](https://arxiv.org/abs/2101.06283)
  - **标题**: 数据@Hand：促进有关智能手机的个人数据的视觉探索和触摸互动
  - **Filtered Reason**: none of cs.HC in whitelist
- [AMFFCN: Attentional Multi-layer Feature Fusion Convolution Network for Audio-visual Speech Enhancement](https://arxiv.org/abs/2101.06268)
  - **标题**: AMFFCN：用于视听语音增强的注意力多层功能融合网络
  - **Filtered Reason**: none of cs.SD,eess.AS in whitelist
- [Multi-layer Feature Fusion Convolution Network for Audio-visual Speech Enhancement](https://arxiv.org/abs/2101.05975)
  - **标题**: 多层功能融合卷积网络，用于视听语音增强
  - **Filtered Reason**: none of cs.SD,eess.AS,eess.IV in whitelist
- [Multimodality in VR: A survey](https://arxiv.org/abs/2101.07906)
  - **标题**: VR中的多模式：调查
  - **Filtered Reason**: none of cs.GR,cs.HC in whitelist
- [Integrated Visualization Editing via Parameterized Declarative Templates](https://arxiv.org/abs/2101.07902)
  - **标题**: 通过参数化声明模板进行集成可视化编辑
  - **Filtered Reason**: none of cs.HC in whitelist
- [VoterFraud2020: a Multi-modal Dataset of Election Fraud Claims on Twitter](https://arxiv.org/abs/2101.08210)
  - **标题**: 选民Fraud2020：在Twitter上的选举欺诈索赔数据集
  - **Filtered Reason**: none of cs.SI in whitelist
- [AirWare: Utilizing Embedded Audio and Infrared Signals for In-Air Hand-Gesture Recognition](https://arxiv.org/abs/2101.10245)
  - **标题**: 气管：利用嵌入式音频和红外信号进行空中手持识别
  - **Filtered Reason**: none of cs.HC in whitelist
- [Rapid mixing in unimodal landscapes and efficient simulatedannealing for multimodal distributions](https://arxiv.org/abs/2101.10004)
  - **标题**: 在单峰景观中快速混合和有效的模拟摩擦，以进行多模态分布
  - **Filtered Reason**: none of math.PR,cs.DM in whitelist
- [Using Angle of Arrival for Improving Indoor Localization](https://arxiv.org/abs/2101.09904)
  - **标题**: 使用到达角度改善室内定位
  - **Filtered Reason**: none of cs.SD,eess.AS,cs.NI in whitelist
- [Toward Personalized Affect-Aware Socially Assistive Robot Tutors in Long-Term Interventions for Children with Autism](https://arxiv.org/abs/2101.10580)
  - **标题**: 在长期干预自闭症儿童的长期干预措施中，采取个性化情感感知感受的社会辅助机器人教师
  - **Filtered Reason**: none of cs.RO,cs.HC in whitelist
- [HEMVIP: Human Evaluation of Multiple Videos in Parallel](https://arxiv.org/abs/2101.11898)
  - **标题**: Hemvip：并行对多个视频的人类评估
  - **Filtered Reason**: none of cs.HC in whitelist
