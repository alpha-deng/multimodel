# 2024-08 月度论文分类汇总

共有29篇相关领域论文, 另有16篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 6 篇论文

### wav2graph: A Framework for Supervised Learning Knowledge Graph from Speech 
[[arxiv](https://arxiv.org/abs/2408.04174)] [[cool](https://papers.cool/arxiv/2408.04174)] [[pdf](https://arxiv.org/pdf/2408.04174)]
> **Authors**: Khai Le-Duc,Quy-Anh Dang,Tan-Hanh Pham,Truong-Son Hy
> **First submission**: 2024-08-07
> **First announcement**: 2024-08-08
> **comment**: Preprint, 32 pages
- **标题**: Wav2Graph：从语音中监督学习知识图的框架
- **领域**: 计算语言学,人工智能,信息检索,机器学习,声音,音频和语音处理
- **摘要**: 知识图（kgs）通过提供结构化的，互连的数据来改善推理和上下文意识，从而增强了大语言模型（LLM）和搜索引擎的性能。但是，KGS仅关注文本数据，从而忽略了其他模式，例如语音。在这项工作中，我们介绍了Wav2Graph，这是从语音数据中监督学习知识图的第一个框架。我们的管道很直接：（1）基于转录的口语和指定的实体数据库构建kg，（2）将kg转换为嵌入向量，以及（3）训练图形神经网络（GNNS）进行节点分类和链接预测任务。通过使用最先进的GNN模型在归纳和转导学习环境中进行的广泛实验，我们为节点分类和链接人类转录本和自动语音识别（ASR）的链接预测任务提供了基线结果和错误分析，包括使用基于编码和解码器的Node Elbed模型进行评估，以及多个模型，以及多元素 - 以及综合效果。所有相关的代码，数据和模型均在线发布。

### Human Speech Perception in Noise: Can Large Language Models Paraphrase to Improve It? 
[[arxiv](https://arxiv.org/abs/2408.04029)] [[cool](https://papers.cool/arxiv/2408.04029)] [[pdf](https://arxiv.org/pdf/2408.04029)]
> **Authors**: Anupama Chingacham,Miaoran Zhang,Vera Demberg,Dietrich Klakow
> **First submission**: 2024-08-07
> **First announcement**: 2024-08-08
> **comment**: Accepted at HuCLLM @ ACL 2024
- **标题**: 人类的语音感知：大语言模型可以释义改善它吗？
- **领域**: 计算语言学
- **摘要**: 大型语言模型（LLMS）可以通过传输样式属性（如形式）来生成文本，从而产生正式或非正式文本。但是，指示LLMS生成文本时，在声学困难的环境中使用时更可理解的文本是一个不足的主题。我们进行了第一项研究，以评估LLM，以产生声学上可理解的释义，以更好地人类的噪声语音感知。我们在英语中的实验表明，通过标准提示，LLM努力控制非文本属性，即声学清晰度，同时有效地捕获了诸如语义等效性之类的所需文本属性。为了解决这个问题，我们提出了一种简单的提示方法，即提示和选择，该方法通过在文本生成管道中解开所需的文本和非文本属性来生成释义。我们的方法通过释义在听力条件下高度扭曲的话语，并以信噪比（SNR）-5 dB的babble噪声高度扭曲的话语，从而导致了40％的人类语音感知的相对改善。这项研究揭示了LLM在捕获非文本属性中的局限性，我们提出的方法展示了使用LLMS在噪声中更好地人类语音感知的潜力。

### LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition 
[[arxiv](https://arxiv.org/abs/2408.05769)] [[cool](https://papers.cool/arxiv/2408.05769)] [[pdf](https://arxiv.org/pdf/2408.05769)]
> **Authors**: Eunseop Yoon,Hee Suk Yoon,John Harvill,Mark Hasegawa-Johnson,Chang D. Yoo
> **First submission**: 2024-08-11
> **First announcement**: 2024-08-12
> **comment**: INTERSPEECH 2024
- **标题**: Li-TTA：语言知情的测试时间适应自动语音识别
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 测试时间适应（TTA）已成为对域转移挑战的关键解决方案，其中目标环境与原始训练环境不同。主要的示例是自动语音识别（ASR）的TTA，它通过利用输出预测熵最小化作为自学信号来增强模型性能。但是，这种自我判断的关键局限性在于其主要关注声学特征，对输入的语言特性的关注最少。为了解决这一差距，我们提出了语言知情的测试时间适应（LI-TTA），该差距在TTA期间为ASR纳入了语言见解。 Li-TTA通过将校正中的CTC损失与标准TTA损失一起最大程度地减少CTC损失，从而整合了从外部语言模型与声学信息合并的校正。通过广泛的实验，我们表明LI-TTA有效地提高了在各种分配转移情况下ASR的TTA性能。

### An Investigation Into Explainable Audio Hate Speech Detection 
[[arxiv](https://arxiv.org/abs/2408.06065)] [[cool](https://papers.cool/arxiv/2408.06065)] [[pdf](https://arxiv.org/pdf/2408.06065)]
> **Authors**: Jinmyeong An,Wonjun Lee,Yejin Jeon,Jungseul Ok,Yunsu Kim,Gary Geunbae Lee
> **First submission**: 2024-08-12
> **First announcement**: 2024-08-13
> **comment**: Accepted to SIGDIAL 2024
- **标题**: 对可解释的音频仇恨言语检测的调查
- **领域**: 计算语言学,人工智能,声音,音频和语音处理
- **摘要**: 仇恨言论的研究主要围绕文本输入的检测和解释，而口头内容在很大程度上没有探索。尽管在口头声音输入中对仇恨言论检测的探索有限，但可解释性的方面被忽略了。因此，我们介绍了可解释的音频仇恨言语检测的新任务。具体而言，我们旨在确定所谓的确切时间间隔，称为音频框架级别的理由，这些时间间隔是仇恨言语分类的证据。为此，我们提出了两种不同的方法：级联和端到端（E2E）。级联方法最初将音频转换为成绩单，标识这些成绩单中的仇恨言论，然后找到相应的音频时间框架。相反，E2E方法直接处理音频话语，这使其可以在特定的时间范围内指出仇恨言论。此外，由于缺乏包括音频框架级别原理在内的可解释的音频仇恨语音数据集，我们策划了一个合成音频数据集来训练我们的模型。我们进一步验证了这些模型在实际的人类语音话语上，发现E2E方法的表现优于级联方法，而不是辅助（IOU）指标的音频框架交集。此外，我们观察到，包括框架级别的理由可以显着提高E2E方法的仇恨言语检测准确性。 \ textbf {免责声明}读者可能会遇到令人讨厌或可恨的本性的内容。但是，鉴于工作的性质，这是无法避免的。

### A layer-wise analysis of Mandarin and English suprasegmentals in SSL speech models 
[[arxiv](https://arxiv.org/abs/2408.13678)] [[cool](https://papers.cool/arxiv/2408.13678)] [[pdf](https://arxiv.org/pdf/2408.13678)]
> **Authors**: Antón de la Fuente,Dan Jurafsky
> **First submission**: 2024-08-24
> **First announcement**: 2024-08-26
> **comment**: 4 pages, 3 figures, to be published in Interspeech 2024 proceedings
- **标题**: 在SSL语音模型中对普通话和英语上段的层分析
- **领域**: 计算语言学
- **摘要**: 这项研究询问了自我监督的语音模型如何代表上等类别，例如普通话词法音调，英语词汇压力和英语短语口音。通过一系列探测任务，我们对英语和普通话12层单语模型进行了层次比较。我们的发现表明，1）英语和普通话wav2Vec 2.0模型学习了在网络中部三分之一中最强的抽象上段类别的上下文表示。 2）模型更擅长表示其训练数据语言中存在的特征，并且这种差异是由变压器块中丰富的上下文驱动的，而不是本地声学表示。 3）微调WAV2VEC 2.0与主要训练的模型相比，主要针对词汇对比的特征（如音调和压力），4）Hubert和Wavlm学习与WAV2VEC 2.0相似的表示形式，主要在以后的一层性能中不同。我们的结果扩展了对模型如何代表上段的理解，并为这些表示的语言特异性和上下文性质提供了新的见解。

### A Functional Trade-off between Prosodic and Semantic Cues in Conveying Sarcasm 
[[arxiv](https://arxiv.org/abs/2408.14892)] [[cool](https://papers.cool/arxiv/2408.14892)] [[pdf](https://arxiv.org/pdf/2408.14892)]
> **Authors**: Zhu Li,Xiyuan Gao,Yuqing Zhang,Shekhar Nayak,Matt Coler
> **First submission**: 2024-08-27
> **First announcement**: 2024-08-28
> **comment**: accepted at Interspeech 2024
- **标题**: 韵律和语义提示之间的功能权衡，传达讽刺
- **领域**: 计算语言学,声音,音频和语音处理
- **摘要**: 这项研究调查了讽刺的声学特征，并解散了讽刺使用的话语倾向与韵律提示信号讽刺之间的相互作用。使用电视节目中汇编的讽刺性话语的数据集，我们分析了属于三种不同的讽刺类别（嵌入，命题和iLlociquicary）的话语和关键短语中的韵律特征，这些特征在语义线索的程度上不同，并将其与中性的表达相提并论。结果表明，在语义中讽刺意义是显着的短语中，韵律提示与从语义中不明显的讽刺意义相比，韵律含义的相关性不大，这表明在短语水平上讽刺和语义提示之间的权衡。这些发现突出了对语义致密讽刺表达中韵律调制的依赖，并塑造了塑造讽刺意图的交流的细微相互作用。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 7 篇论文

### Evaluation of Segment Anything Model 2: The Role of SAM2 in the Underwater Environment 
[[arxiv](https://arxiv.org/abs/2408.02924)] [[cool](https://papers.cool/arxiv/2408.02924)] [[pdf](https://arxiv.org/pdf/2408.02924)]
> **Authors**: Shijie Lian,Hua Li
> **First submission**: 2024-08-05
> **First announcement**: 2024-08-06
> **comment**: No comments
- **标题**: 评估任何模型2：SAM2在水下环境中的作用
- **领域**: 计算机视觉和模式识别
- **摘要**: 随着大规模建模的突破，该细分市场（SAM）及其扩展已尝试在海洋科学的各种水下可视化任务中进行应用，并对学术界产生了重大影响。最近，Meta进一步开发了任何模型2（SAM2）的细分市场，与其前身相比，它显着提高了运行速度和分割精度。该报告旨在通过在水下实例分割基准数据集UII和USIS10K上评估SAM2在海洋科学中的潜力。实验表明，SAM2的性能极为取决于用户提供的提示的类型。当使用地面真相边界框作为提示时，SAM2在水下实例细分域中表现出色。但是，在自动模式下运行时，SAM2具有点提示和分段水下实例的能力会大大退化。希望本文能够激发研究人员进一步探索水下领域中的SAM模型家族。本文中的结果和评估代码可在https://github.com/liamlian0727/underwatersam2eval上找到。

### U-DECN: End-to-End Underwater Object Detection ConvNet with Improved DeNoising Training 
[[arxiv](https://arxiv.org/abs/2408.05780)] [[cool](https://papers.cool/arxiv/2408.05780)] [[pdf](https://arxiv.org/pdf/2408.05780)]
> **Authors**: Zhuoyan Liu,Bo Wang,Ye Li
> **First submission**: 2024-08-11
> **First announcement**: 2024-08-12
> **comment**: No comments
- **标题**: U-DECN：端到端的水下对象检测Convnet，并改进
- **领域**: 计算机视觉和模式识别
- **摘要**: 由于其特定的环境挑战，水下对象检测对检测器的运行速度和部署效率的要求更高。基于查询的端到端对象检测器的两阶段对象检测器和变压器体系结构的NMS不利于在水下嵌入式设备上部署具有有限的处理能力。至于水下颜色铸造噪声的有害效果，最近的水下对象探测器使网络架构或训练综合体使其在水下车辆平台上的应用和部署也妨碍了。在本文中，我们通过改进的DeNoising Training（U-DECN）提出了水下Deco，该培训（U-DECN）是基于查询的端到端对象检测器（带有Convnet Encoder-Decoder架构），用于解决以上问题的水下颜色铸造噪声。我们将高级技术从DETR变体集成到专门针对Convnet体系结构的装饰和设计优化方法中，包括单独的对比度降解向前和SIM中的可变形卷积。为了解决水下颜色铸造噪声问题，我们提出了一个水下颜色denoising查询，以通过不同的颜色铸造噪声来改善偏见对象特征信息的模型的概括。 Our U-DECN, with ResNet-50 backbone, achieves 61.4 AP (50 epochs), 63.3 AP (72 epochs), 64.0 AP (100 epochs) on DUO, and 21 FPS (5 times faster than Deformable DETR and DINO 4 FPS) on NVIDIA AGX Orin by TensorRT FP16, outperforming the other state-of-the-art query-based end-to-end object探测器。该代码可在https://github.com/lefteyex/u-decn上找到。

### VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers using eXplainable AI and Language Models 
[[arxiv](https://arxiv.org/abs/2408.12808)] [[cool](https://papers.cool/arxiv/2408.12808)] [[pdf](https://arxiv.org/pdf/2408.12808)]
> **Authors**: Purushothaman Natarajan,Athira Nambiar
> **First submission**: 2024-08-22
> **First announcement**: 2024-08-23
> **comment**: 15 pages, 10 tables, 3 figures
- **标题**: 谷：使用可解释的AI和语言模型的图像分类器的多模式的视觉和语言解释框架
- **领域**: 计算机视觉和模式识别,人工智能,计算语言学,机器学习
- **摘要**: 深度神经网络（DNNS）通过实现任务自动化并减少人为错误，彻底改变了各个领域。但是，由于它们的黑匣子性质，他们的内部运作和决策过程仍然晦涩难懂。因此，缺乏解释性限制了这些模型在高风险场景中的应用。为了解决这个问题，可解释的人工智能（XAI）的新兴领域旨在解释和解释DNN的内部工作。尽管取得了进步，Xai仍面临挑战，例如机器和人类理解之间的语义差距，可解释性和绩效之间的权衡以及需要特定于上下文的解释。为了克服这些局限性，我们提出了一个新颖的多模式框架，名为Vale Visual和语言解释。 Vale将可解释的AI技术与先进的语言模型相结合，以提供全面的解释。该框架利用了XAI工具，高级零摄像图分割模型和视觉语言模型的视觉说明来生成相应的文本说明。通过结合视觉和文本解释，Vale桥接了机器输出与人类解释之间的语义差距，从而提供了用户更理解的结果。在本文中，我们对图像分类任务的Vale框架进行了试点研究。具体而言，Shapley添加说明（SHAP）用于识别分类图像中最具影响力的区域。然后，使用段的任何模型（SAM）提取感兴趣的对象，并使用最先进的预训练的视觉模型（VLM）生成解释。在两个数据集上进行了广泛的实验研究：Imagenet数据集和一个自定义的水下声纳图像数据集，证明了水下图像分类中的Vales真实世界的适用性。

### Semantic Communication based on Large Language Model for Underwater Image Transmission 
[[arxiv](https://arxiv.org/abs/2408.12616)] [[cool](https://papers.cool/arxiv/2408.12616)] [[pdf](https://arxiv.org/pdf/2408.12616)]
> **Authors**: Weilong Chen,Wenxuan Xu,Haoran Chen,Xinran Zhang,Zhijin Qin,Yanru Zhang,Zhu Han
> **First submission**: 2024-08-08
> **First announcement**: 2024-08-23
> **comment**: No comments
- **标题**: None
- **领域**: 计算机视觉和模式识别,人工智能
- **摘要**: 水下通信对于环境监测，海洋生物学研究和水下勘探至关重要。传统的水下沟通面临诸如低带宽，高潜伏期和对噪声的敏感性之类的局限性，而语义通信（SC）则通过着重于语义的交换而不是符号或位来提供有前途的解决方案。但是，SC在水下环境中遇到挑战，包括语义信息不匹配和难以准确识别和传输与水下应用不同要求相符的关键信息。为了应对这些挑战，我们提出了一个基于大语言模型（LLM）的新型语义交流（SC）框架。我们的框架利用Visual LLMS根据用户的查询来执行水下图像数据的语义压缩和优先级。通过识别和编码图像中的关键语义元素，系统选择性地传输高优先级信息，同时将较高的压缩率应用于较低的关键区域。在接收方面，基于LLM的恢复机制以及全球视觉控制网络和关键区域控制网络网络有助于重建图像，从而提高了通信效率和鲁棒性。我们的框架将整体数据大小减少到原始数据的0.8％。实验结果表明，我们的方法显着优于现有方法，以确保高质量，语义准确的图像重建。

### Underwater SONAR Image Classification and Analysis using LIME-based Explainable Artificial Intelligence 
[[arxiv](https://arxiv.org/abs/2408.12837)] [[cool](https://papers.cool/arxiv/2408.12837)] [[pdf](https://arxiv.org/pdf/2408.12837)]
> **Authors**: Purushothaman Natarajan,Athira Nambiar
> **First submission**: 2024-08-23
> **First announcement**: 2024-08-26
> **comment**: 55 pages, 9 tables, 18 figures
- **标题**: 使用基于石灰的可解释人工智能的水下声纳图像分类和分析
- **领域**: 计算机视觉和模式识别,人工智能,人机交互,机器学习
- **摘要**: 深度学习技术通过模仿人类认知并自动化复杂的决策过程来彻底改变图像分类。但是，由于缺乏该模型的解释性，AI系统在野外的部署，尤其是在防御等高安全域中的部署。为此，可解释的AI（XAI）是一个新兴领域，旨在探索深神经网络的无法解释的隐藏黑匣子。本文探讨了可解释的人工智能（XAI）工具来解释水下图像分类结果，这是我们所知的最佳域中的第一批作品之一。我们的研究使用源自不同来源的自定义数据集来研究声纳图像分类的领域，包括海床对象KLSG数据集，相机声纳数据集，矿山声纳图像数据集和SCTD数据集。使用基准卷积神经网络（CNN）体系结构（例如VGG16，Resnet50，InceptionV3，Densenet121等）对图像分类进行的转移学习技术进行了广泛的分析。在此分类模型之上，后XAI技术，即。通过局部可解释的模型 - 不足解释（LIME）合并，以通过在本地扰动输入数据以查看预测如何变化来为模型的决策提供透明的理由。此外，还广泛研究了基于supsodular picks的图像的酸橙（Sp-lime）的酸橙版本（SP-lime）。为此，将两种suppodular优化算法，即快速置和简单的线性迭代聚类（SLIC）杠杆式选择。对XAI技术的广泛分析以更符合人类的方式突出了结果的解释性，从而提高了我们的信心和可靠性。

### S3Simulator: A benchmarking Side Scan Sonar Simulator dataset for Underwater Image Analysis 
[[arxiv](https://arxiv.org/abs/2408.12833)] [[cool](https://papers.cool/arxiv/2408.12833)] [[pdf](https://arxiv.org/pdf/2408.12833)]
> **Authors**: Kamal Basha S,Athira Nambiar
> **First submission**: 2024-08-23
> **First announcement**: 2024-08-26
> **comment**: :68T45; 68T05; 65D18ACM Class:I.4.8; I.4.9; I.2.10; J.2
- **标题**: S3Simulator：用于水下图像分析的基准测试扫描声纳模拟器数据集
- **领域**: 计算机视觉和模式识别
- **摘要**: 声音声纳成像系统被广泛用于平民和军事部门的水下监视。但是，购买用于培训人工智能（AI）模型的高质量声纳数据集面临诸如有限的数据可用性，财务限制和数据机密性等挑战。为了克服这些挑战，我们提出了一个新颖的基准数据集的模拟侧面声纳图像，我们将其称为“ S3Simulator数据集”。我们的数据集创建利用先进的模拟技术来准确复制水下条件并产生多样化的合成声纳成像。特别是，尖端的AI分割工具，即将任何模型（SAM）的任何模型（SAM）从真实场景中最佳隔离和分割对象图像（例如船舶和飞机）。此外，采用了高级计算机辅助设计工具，即诸如凉亭之类的自我电卡和仿真软件，以创建3D模型并分别在现实环境中最佳可视化。此外，采用了一系列计算成像技术来提高数据质量，从而使AI模型用于分析声纳图像。对S3Simulator以及真实的声纳数据集进行了广泛的分析，以验证AI模型进行水下对象分类的性能。我们的实验结果强调，S3Simulator数据集将是用于研究水下图像分析的有希望的基准数据集。 https://github.com/bashakamal/s3simulator。

### Global-Local Distillation Network-Based Audio-Visual Speaker Tracking with Incomplete Modalities 
[[arxiv](https://arxiv.org/abs/2408.14585)] [[cool](https://papers.cool/arxiv/2408.14585)] [[pdf](https://arxiv.org/pdf/2408.14585)]
> **Authors**: Yidi Li,Yihan Li,Yixin Guo,Bin Ren,Zhenhuan Xu,Hao Guo,Hong Liu,Nicu Sebe
> **First submission**: 2024-08-26
> **First announcement**: 2024-08-27
> **comment**: We request to withdraw our paper from arXiv due to unresolved author disagreements about the data interpretation and study conclusions. To maintain scientific integrity, we believe withdrawing the paper is necessary. We regret any confusion caused
- **标题**: 基于全球 - 本地蒸馏网络基于网络的视听扬声器跟踪，具有不完整的方式
- **领域**: 计算机视觉和模式识别,声音,音频和语音处理
- **摘要**: 在扬声器跟踪研究中，整合和补充多模式数据是提高跟踪系统准确性和鲁棒性的关键策略。但是，由于遮挡，声音噪声和传感器故障引起的嘈杂观察结果，以不完整的方式跟踪仍然是一个具有挑战性的问题。尤其是当多种模式中缺少数据时，现有的多模式融合方法的性能往往会降低。为此，我们为强大的视听扬声器跟踪提出了一个基于全部本地蒸馏器的跟踪器（GLDTRACKER）。 GldTracker是由教师学生蒸馏模型驱动的，从而使每种模式的不完整信息的灵活融合。教师网络处理由摄像机和麦克风阵列捕获的全球信号，学生网络会处理视觉阻塞和缺少音频频道的本地信息。通过将知识从老师转移到学生，学生网络可以更好地适应复杂的动态场景，并以不完整的观察。在学生网络中，基于生成对抗网络的全局功能重建模块构建是为了从嵌入功能嵌入而没有本地信息的功能中重建全局功能。此外，引入了多模式的多级融合注意力，以整合不完整的功能和重建功能，利用视听和全球本地特征的互补性和一致性。 AV16.3数据集的实验结果表明，所提出的GLDTRACKER的表现优于现有的最先进的视听跟踪器，并且在标准和不完整的模态数据集中达到了领先的性能，突出了其在复杂条件下的优势和鲁棒性。代码和模型将可用。

## 机器学习(cs.LG:Machine Learning)

该领域共有 4 篇论文

### TinyChirp: Bird Song Recognition Using TinyML Models on Low-power Wireless Acoustic Sensors 
[[arxiv](https://arxiv.org/abs/2407.21453)] [[cool](https://papers.cool/arxiv/2407.21453)] [[pdf](https://arxiv.org/pdf/2407.21453)]
> **Authors**: Zhaolan Huang,Adrien Tousnakhoff,Polina Kozyr,Roman Rehausen,Felix Bießmann,Robert Lachlan,Cedric Adjih,Emmanuel Baccelli
> **First submission**: 2024-07-31
> **First announcement**: 2024-08-01
> **comment**: No comments
- **标题**: TinyChirp：在低功率无线声音传感器上使用Tinyml模型识别鸟类歌曲
- **领域**: 机器学习,人工智能,声音,音频和语音处理,信号处理
- **摘要**: 大规模监测生物多样性是具有挑战性的。在细颗粒分类法中检测和识别物种需要高度准确的机器学习（ML）方法。培训此类模型需要大量的高质量数据集。将这些模型部署到低功率设备需要新颖的压缩技术和模型体系结构。尽管物种分类方法从新颖的数据集和ML方法（尤其是神经网络）的进步中获利，但将这些最先进的模型部署到低功率设备仍然很困难。在这里，我们介绍了各种Tinyml神经网络体系结构和物种分类的压缩技术的全面经验比较。我们专注于鸟类歌曲检测的示例，更具体地是一个用于研究玉米束鸟类的数据集。数据集与本研究的所有代码和实验一起发布。在我们的实验中，我们比较了基于经典谱图方法的预测性能，记忆和时间复杂性以及在原始音频信号上运行的最新方法。我们的结果表明，可以用相对简单的体系结构可牢固地检测到单个鸟类，这些结构可以容易部署到低功率设备上。

### Multimodal Gender Fairness in Depression Prediction: Insights on Data from the USA & China 
[[arxiv](https://arxiv.org/abs/2408.04026)] [[cool](https://papers.cool/arxiv/2408.04026)] [[pdf](https://arxiv.org/pdf/2408.04026)]
> **Authors**: Joseph Cameron,Jiaee Cheong,Micol Spitale,Hatice Gunes
> **First submission**: 2024-08-07
> **First announcement**: 2024-08-08
> **comment**: 9 Pages, 7 Tables. To be published and indexed in the IEEE Xplore Digital Library under the ACII 2024 Workshop Proceedings
- **标题**: 抑郁预测中的多模式性别公平：美国和中国数据的见解
- **领域**: 机器学习,人工智能,机器人技术
- **摘要**: 社会代理商和机器人越来越多地用于福利环境中。但是，一个关键的挑战是，这些代理商和机器人通常依靠机器学习（ML）算法来检测和分析个人的心理健康。 ML算法中偏见和公平性的问题正在成为越来越多的关注来源。同时，现有文献还表明，在各个性别和文化中，心理健康状况的表现都不同。我们假设特征（声学，文本和视觉）及其模式间关系的表示在不同文化和性别的受试者之间会有所不同，从而影响了各种ML模型的性能和公平性。我们通过对来自美国和中国的两个不同数据集进行研究，对抑郁症表现的多模式性别公平性进行了首次评估。我们进行彻底的统计和ML实验，并重复几种不同算法的实验，以确保结果不依赖算法。我们的发现表明，尽管两个数据集之间存在差异，但这不是由于抑郁症表现为所假定的或其他外部因素（例如数据收集方法的差异）尚无定论。我们的发现进一步促使人们呼吁建立更一致和具有文化意识的数据收集过程，以解决抑郁症检测中ML偏见的问题，并促进更公平的代理商和机器人的福祉。

### Inferring Underwater Topography with FINN 
[[arxiv](https://arxiv.org/abs/2408.10649)] [[cool](https://papers.cool/arxiv/2408.10649)] [[pdf](https://arxiv.org/pdf/2408.10649)]
> **Authors**: Coşku Can Horuz,Matthias Karlbauer,Timothy Praditia,Sergey Oladyshkin,Wolfgang Nowak,Sebastian Otte
> **First submission**: 2024-08-20
> **First announcement**: 2024-08-21
> **comment**: No comments
- **标题**: 用芬兰推断水下地形
- **领域**: 机器学习,人工智能,大气和海洋物理,计算物理,流体动力学
- **摘要**: 时空部分微分方程（PDE）在各个科学和工程领域找到广泛的应用。尽管物理学和机器学习（ML）社区都出现了许多模型，但逐渐趋于整合这些方法来开发称为物理学的机器学习模型的混合体系结构。其中，有限的体积神经网络（FINN）已成为最近的增加。事实证明，Finn在发现数据中的潜在结构方面特别有效。在这项研究中，我们探讨了Finn在应对浅水方程方面的能力，该方程模拟了沿海地区的波动动态。具体而言，我们研究了Finn根据这些特定波动方程重建水下地形的功效。我们的发现表明，芬兰人表现出仅仅从波动力学来推断地形的显着能力，将自己与常规ML和物理意识的ML模型区分开来。我们的结果强调了FINN在促进我们对时空现象的理解和增强相关领域参数化能力方面的潜力。

### Integrating Audio, Visual, and Semantic Information for Enhanced Multimodal Speaker Diarization 
[[arxiv](https://arxiv.org/abs/2408.12102)] [[cool](https://papers.cool/arxiv/2408.12102)] [[pdf](https://arxiv.org/pdf/2408.12102)]
> **Authors**: Luyao Cheng,Hui Wang,Siqi Zheng,Yafeng Chen,Rongjie Huang,Qinglin Zhang,Qian Chen,Xihao Li
> **First submission**: 2024-08-21
> **First announcement**: 2024-08-22
> **comment**: No comments
- **标题**: 整合音频，视觉和语义信息，以进行增强的多模式扬声器诊断
- **领域**: 机器学习,计算机视觉和模式识别,声音,音频和语音处理
- **摘要**: 说话者诊断，将音频流或基于说话者身份的统一分区分割为同质分区的过程，在人类言语的解释和分析中起着至关重要的作用。大多数现有的扬声器诊断系统仅依赖于单峰声信息，这使得由于音频信号的先天歧义而尤其具有挑战性。最近的研究为视听或音频语义建模做出了巨大的努力，以提高性能。但是，即使是多达两种方式的结合也常常在解决自发和非结构化对话的复杂性方面也很短。为了利用更有意义的对话模式，我们提出了一种新型的多模式方法，该方法共同利用音频，视觉和语义提示来增强说话者的诊断。我们的方法优雅地将多模型建模作为约束优化问题。首先，我们建立了对活动扬声器之间的视觉连接以及口语内容中的语义交互的见解，从而建立了丰富的成对约束。然后，我们根据这些视觉和语义约束将群集扬声器引入群集扬声器。这种整合有效地利用了不同方式的互补优势，从而完善了单个说话者嵌入之间的亲和力估计。在多个多模式数据集上进行的广泛实验表明，我们的方法始终优于最先进的说话者诊断方法。

## 多媒体(cs.MM:Multimedia)

该领域共有 2 篇论文

### Open-Vocabulary Audio-Visual Semantic Segmentation 
[[arxiv](https://arxiv.org/abs/2407.21721)] [[cool](https://papers.cool/arxiv/2407.21721)] [[pdf](https://arxiv.org/pdf/2407.21721)]
> **Authors**: Ruohao Guo,Liao Qu,Dantong Niu,Yanyu Qi,Wenzhen Yue,Ji Shi,Bowei Xing,Xianghua Ying
> **First submission**: 2024-07-31
> **First announcement**: 2024-08-01
> **comment**: Accepted by ACM MM 2024 (Oral)
- **标题**: 开放式视听语义分段
- **领域**: 多媒体,人工智能
- **摘要**: 视听语义细分（AVSS）旨在在带有声音提示的视频中对声音进行分割和分类。但是，大多数方法都在近距离假设上运行，并且仅从培训数据中识别预定义的类别，缺乏在实际应用中检测新类别的概括能力。在本文中，我们介绍了一项新任务：开放式音频语义分段，将AVSS任务扩展到超出带注释的标签空间之外的开放世界情景。这是一项更具挑战性的任务，需要识别所有类别，即使是那些在培训期间从未见过或听到过的类别。此外，我们提出了第一个开放式唱片库AVSS框架OV-AVSS，主要由两个部分组成：1）通用声音源定位模块，以执行视听融合并找到所有潜在的声音对象，并找到所有潜在的声音对象，2）一个开放式杂货分类模块，以通过大型稳定的视觉来预测类别，可预测类别，从大型的可识别视觉上进行启用。为了正确评估开放式摄影库AVSS，我们根据AVSBench-Smantic基准测试（即Avsbench-ov）对零射击训练和测试子集进行了分配。广泛的实验表明，我们的模型在所有类别上的强烈分割和零击概括能力。在AVSBENCH-ov数据集上，OV-AVSS在基本类别上达到55.43％的MIOU，而新型类别的29.14％MIOU超过了41.88％/20.61％的最先进的零摄像方法，而开放式vocabulary方法则以10.2％/11.6％的速度开放式vocabulary方法。该代码可从https://github.com/ruohaoguo/ovavss获得。

### Out-Of-Distribution Detection for Audio-visual Generalized Zero-Shot Learning: A General Framework 
[[arxiv](https://arxiv.org/abs/2408.01284)] [[cool](https://papers.cool/arxiv/2408.01284)] [[pdf](https://arxiv.org/pdf/2408.01284)]
> **Authors**: Liuyuan Wen
> **First submission**: 2024-08-02
> **First announcement**: 2024-08-05
> **comment**: No comments
- **标题**: 视听广义的零拍学习的分布式检测：一般框架
- **领域**: 多媒体,计算机视觉和模式识别,声音,音频和语音处理,图像和视频处理
- **摘要**: 广义零射学习（GZSL）是一项具有挑战性的任务，需要对可见和看不见的类进行准确的分类。在该领域内，鉴于将视觉和声学特征纳入多模式输入，视听GZSL是一项极其令人兴奋但艰巨的任务。该领域的现有工作主要利用基于嵌入的或基于生成的方法。但是，生成训练是困难且不稳定的，而基于嵌入的方法通常会遇到域转移问题。因此，我们发现有望将这两种方法整合到一个统一的框架中，以利用其优势，同时减轻各自的缺点。我们的研究介绍了采用分布外（OOD）检测的一般框架，旨在利用两种方法的优势。我们首先采用生成的对抗网络来综合看不见的功能，从而使OOD检测器与分类器一起培训可见和看不见的类。该检测器确定测试功能是属于所见类还是看不见的类，然后分类使用每个功能类型的单独分类器。我们在三个流行的视听数据集上测试了我们的框架，并观察到与现有最新作品相比的重大改进。代码可以在https://github.com/liuyuan-wen/av-ood-gzsl中找到。

## 声音(cs.SD:Sound)

该领域共有 4 篇论文

### AcousAF: Acoustic Sensing-Based Atrial Fibrillation Detection System for Mobile Phones 
[[arxiv](https://arxiv.org/abs/2408.04912)] [[cool](https://papers.cool/arxiv/2408.04912)] [[pdf](https://arxiv.org/pdf/2408.04912)]
> **Authors**: Xuanyu Liu,Haoxian Liu,Jiao Li,Zongqi Yang,Yi Huang,Jin Zhang
> **First submission**: 2024-08-09
> **First announcement**: 2024-08-12
> **comment**: Accepted for publication in Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp Companion '24)
- **标题**: Acousaf：手机基于声学传感的心房颤动检测系统
- **领域**: 声音,计算工程、金融和科学,新兴技术,机器学习,音频和语音处理
- **摘要**: 心房颤动（AF）的特征是源自心房的不规则电脉冲，这可能导致严重的并发症甚至死亡。由于AF的间歇性质，对AF的早期和及时​​监测对于防止病情进一步加剧至关重要。尽管门诊ECG Holter监视器提供了准确的监控，但这些设备的高成本阻碍了其更广泛的采用。当前基于移动的AF检测系统提供便携式解决方案。但是，这些系统存在各种适用性问题，例如很容易受到环境因素的影响并需要大量的用户努力。为了克服上述局限性，我们提出了Acousaf，这是一种基于智能手机声传感器的新型AF检测系统。特别是，我们使用智能手机扬声器和麦克风从手腕探索脉搏波的潜力。此外，我们提出了一个精心设计的框架，该框架由脉搏波探测，脉搏波提取和AF检测组成，以确保准确可靠的AF检测。我们从智能手机上利用我们的自定义数据收集应用程序的20名参与者收集数据。广泛的实验结果表明了我们系统的高性能，精度为92.8％，精度为86.9％，召回率为87.4％，F1得分为87.1％。

### Audio Enhancement for Computer Audition -- An Iterative Training Paradigm Using Sample Importance 
[[arxiv](https://arxiv.org/abs/2408.06264)] [[cool](https://papers.cool/arxiv/2408.06264)] [[pdf](https://arxiv.org/pdf/2408.06264)]
> **Authors**: Manuel Milling,Shuo Liu,Andreas Triantafyllopoulos,Ilhan Aslan,Björn W. Schuller
> **First submission**: 2024-08-12
> **First announcement**: 2024-08-13
> **comment**: No comments
- **标题**: 计算机试镜的音频增强 - 使用样本重要性的迭代训练范式
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 用于音频任务的神经网络模型，例如自动语音识别（ASR）和声学场景分类（ASC），对现实生活中的噪声污染易受噪声污染。为了提高音频质量，可以独立开发的增强模块在目标音频应用的前端明确使用。在本文中，我们提出了一种端到端的学习解决方案，以共同优化音频增强模型（AE）和后续应用程序。为了指导AE模块针对目标应用的优化，尤其是为了克服困难的样本，我们利用样本的性能度量作为样本重要性的指示。在实验中，我们考虑了四个代表性应用来评估我们的培训范式，即ASR，语音命令识别（SCR），语音情感识别（SER）和ASC。这些应用与有关语义和非语义特征，瞬态和全球信息的语音和非语音任务有关，实验结果表明，我们所提出的方法可以大大提高模型的噪声鲁棒性，尤其是在低信噪比（SNRS）的情况下，对于日常生活中的各种计算机试镜任务。

### Optimising MFCC parameters for the automatic detection of respiratory diseases 
[[arxiv](https://arxiv.org/abs/2408.07522)] [[cool](https://papers.cool/arxiv/2408.07522)] [[pdf](https://arxiv.org/pdf/2408.07522)]
> **Authors**: Yuyang Yan,Sami O. Simons,Loes van Bemmel,Lauren Reinders,Frits M. E. Franssen,Visara Urovi
> **First submission**: 2024-08-14
> **First announcement**: 2024-08-15
> **comment**: No comments
- **标题**: 优化自动检测呼吸道疾病的MFCC参数
- **领域**: 声音,机器学习,音频和语音处理
- **摘要**: 源自呼吸道的语音信号被用作诊断和评估呼吸道疾病的有价值的声学生物标志物。在使用的声学特征中，MEL频率曲线系数（MFCC）被广泛用于自动分析，MFCC提取通常依赖于默认参数。但是，尚无综合研究系统地研究了MFCC提取参数对呼吸道疾病诊断的影响。在这项研究中，我们通过检查关键参数的影响，即系数，框架长度和帧之间的跳长的数量在呼吸状态检查中，解决了这一差距。我们的调查使用四个数据集：Cambridge Covid-19声音数据库，Coswara数据集，Saarbrucken语音疾病（SVD）数据库和一个Tacticas数据集。鉴于其广泛的采用和功效，支持向量机（SVM）被用作分类器。我们的发现表明，MFCC的准确性随着跳跃的长度的增加而降低，并且观察到最佳的系数数量约为30。MFCC的性能随数据集的帧长度而变化：对于COVID-19的数据集，MFCC的性能变化（cambridge-19 （从50毫秒到500毫秒）。此外，我们研究了这些参数的优化组合，并观察到准确性的实质性增强。与最糟糕的组合相比，SVM模型的精度为81.1％，80.6％和71.7％，剑桥Covid-19声音数据库，COSWARA DATASET和SVD数据集分别提高了19.6％，16.10％和14.90％。

### Disentangled Training with Adversarial Examples For Robust Small-footprint Keyword Spotting 
[[arxiv](https://arxiv.org/abs/2408.13355)] [[cool](https://papers.cool/arxiv/2408.13355)] [[pdf](https://arxiv.org/pdf/2408.13355)]
> **Authors**: Zhenyu Wang,Li Wan,Biqiao Zhang,Yiteng Huang,Shang-Wen Li,Ming Sun,Xin Lei,Zhaojun Yang
> **First submission**: 2024-08-23
> **First announcement**: 2024-08-26
> **comment**: ef:ICASSP 2023
- **标题**: 通过对抗性示例进行分解的培训
- **领域**: 声音,人工智能,音频和语音处理
- **摘要**: 在设备上连续运行的关键字发现（KWS）引擎会暴露于以前通常看不到的各种语音信号。在不同的声学环境下建立具有鲁棒性的小英尺印记和高性能的KWS模型是一个具有挑战性的问题。在本文中，我们探讨了如何有效地应用对抗性示例以提高KWS的鲁棒性。我们建议使用对抗性示例进行数据源信息，以减少原始数据和对抗性数据之间的不匹配以及原始培训数据源的不匹配。 KWS模型体系结构基于深度可分离的卷积和简单的注意模块。实验结果表明，提出的学习策略将虚假的拒绝利率提高了$ 40.31％$，$ 1％$ $ $ $ $ $ $ $ $，$ 1％$ false false接受率，而没有使用对抗性示例的最强基线。我们表现​​最佳的系统在Google Speech Commands V1数据集上实现了$ 98.06％的精度。

## 音频和语音处理(eess.AS:Audio and Speech Processing)

该领域共有 6 篇论文

### BSS-CFFMA: Cross-Domain Feature Fusion and Multi-Attention Speech Enhancement Network based on Self-Supervised Embedding 
[[arxiv](https://arxiv.org/abs/2408.06851)] [[cool](https://papers.cool/arxiv/2408.06851)] [[pdf](https://arxiv.org/pdf/2408.06851)]
> **Authors**: Alimjan Mattursun,Liejun Wang,Yinfeng Yu
> **First submission**: 2024-08-13
> **First announcement**: 2024-08-14
> **comment**: Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2024
- **标题**: BSS-CFFMA：基于自我监督的嵌入的跨域特征融合和多发音语音增强网络
- **领域**: 音频和语音处理,人工智能
- **摘要**: 语音自学学习（SSL）代表在多个下游任务中实现了最先进的表现（SOTA）。但是，其在语音增强（SE）任务中的应用仍然不成熟，提供了改进的机会。在这项研究中，我们介绍了一种新型的跨域特征融合和多发的语音增强网络，称为BSS-CFFMA，该网络利用了自我监督的嵌入。 BSS-CFFMA包括一个多尺度的跨域特征融合（MSCFF）块和残留的混合多重注意（RHMA）块。 MSCFF块有效地整合了跨域特征，从而促进了丰富的声学信息的提取。 RHMA块作为主要的增强模块，利用三个不同的注意模块来捕获各种注意力表示并估计高质量的语音信号。我们通过对语音库数据集的比较和消融研究评估BSS-CFFMA模型的性能，从而实现SOTA结果。此外，我们从WHAMR中选择三种类型的数据！数据集是专门针对语音增强任务设计的集合，用于评估BSS-CFFMA在仅降解，仅消除验证以及同时降解和替代验证的任务中的功能。这项研究标志着探索自我监督的基于嵌入的语音增强方法的有效性的首次尝试，包括复杂的任务，其中包括编织和同时的脱氧和替代性。 BSS-CFFMA的演示实现可在线获得\脚注[2] {https://github.com/alimmat/bss-cffma。 \ label {s1}}。

### ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale 
[[arxiv](https://arxiv.org/abs/2408.08739)] [[cool](https://papers.cool/arxiv/2408.08739)] [[pdf](https://arxiv.org/pdf/2408.08739)]
> **Authors**: Xin Wang,Hector Delgado,Hemlata Tak,Jee-weon Jung,Hye-jin Shim,Massimiliano Todisco,Ivan Kukanov,Xuechen Liu,Md Sahidullah,Tomi Kinnunen,Nicholas Evans,Kong Aik Lee,Junichi Yamagishi
> **First submission**: 2024-08-16
> **First announcement**: 2024-08-19
> **comment**: 8 pages, ASVspoof 5 Workshop (Interspeech2024 Satellite)
- **标题**: ASVSPOOF 5：众包语音数据，深击和对抗性攻击
- **领域**: 音频和语音处理,人工智能,声音
- **摘要**: ASVSPOOF 5是一系列挑战中的第五版，促进了对语音欺骗和深击攻击的研究以及检测解决方案的设计。与以前的挑战相比，ASVSPOOF 5数据库是根据在不同声学条件下从大量扬声器中收集的众包数据构建的。使用替代检测模型生成和测试的攻击也是众包的，而第一次进行了对抗攻击。新的指标支持评估欺骗性的自动扬声器验证（SASV）以及独立检测解决方案，即没有ASV的对策。我们描述了两个挑战轨道，新数据库，评估指标，基准和评估平台，并介绍结果的摘要。攻击大大损害了基线系统，而提交的攻击带来了重大改进。

### Estimated Audio-Caption Correspondences Improve Language-Based Audio Retrieval 
[[arxiv](https://arxiv.org/abs/2408.11641)] [[cool](https://papers.cool/arxiv/2408.11641)] [[pdf](https://arxiv.org/pdf/2408.11641)]
> **Authors**: Paul Primus,Florian Schmid,Gerhard Widmer
> **First submission**: 2024-08-21
> **First announcement**: 2024-08-22
> **comment**: In Proceedings of the 9th Workshop onDetectionandClassificationofAcousticScenes and Events, DCASE, Tokyo, Japan, 2024. Implementation available on GitHub: https://github.com/OptimusPrimus/salsa
- **标题**: 估计的音频启动通信改善了基于语言的音频检索
- **领域**: 音频和语音处理,机器学习,声音
- **摘要**: 基于双编码器的音频检索系统通常在一组匹配和不匹配的音频捕获对上通过对比度学习进行优化。这导致了共享的嵌入空间，其中两种方式的相应项目最终结合在一起。由于音频启动数据集通常仅包含匹配的录制和描述对，因此通过将音频与随机从数据集进行的字幕配对来创建不匹配对的练习已成为常见的做法。这不是理想的选择，因为随机采样的标题只能偶然地或完全描述录音。但是，所有可能对的对应信息对于注释而言是昂贵的，因此通常不可用。因此，我们建议将其替换为估计的对应关系。为此，我们提出了一个两阶段的培训程序，其中首先像往常一样对多个检索模型进行训练，即没有估计的对应关系。在第二阶段，这些模型预测的音频捕获对应关系是预测目标。我们在ClothOv2和AudioCaps基准上评估了我们的方法，并表明它可以提高检索性能，即使在限制单个模型生成的限制自我鉴定设置中，然后从估计的通信中学习。我们进一步表明，我们的方法在ClothOv2基准上的地图@10以1.6pp。MAP的表现优于当前的最新状态。

### Infusing Acoustic Pause Context into Text-Based Dementia Assessment 
[[arxiv](https://arxiv.org/abs/2408.15188)] [[cool](https://papers.cool/arxiv/2408.15188)] [[pdf](https://arxiv.org/pdf/2408.15188)]
> **Authors**: Franziska Braun,Sebastian P. Bayerl,Florian Hönig,Hartmut Lehfeld,Thomas Hillemacher,Tobias Bocklet,Korbinian Riedhammer
> **First submission**: 2024-08-27
> **First announcement**: 2024-08-28
> **comment**: Accepted at INTERSPEECH 2024
- **标题**: 将声学暂停上下文注入基于文本的痴呆评估中
- **领域**: 音频和语音处理,计算语言学,声音
- **摘要**: 语音暂停，以及内容和结构，为检测痴呆症提供了有价值且无创的生物标志物。这项工作研究了在基于变压器的语言模型中暂停增强的转录本的使用，以区分没有认知障碍，轻度认知障碍和阿尔茨海默氏症的痴呆的受试者的认知状态，并根据临床评估。我们解决了三个二进制分类任务：发作，监测和痴呆症排除。通过在德国语言流利度测试和图片描述测试的实验中评估了该性能，并比较了模型在不同语音生产环境中的有效性。从文本基线开始，我们研究了暂停信息和声学环境的融合的效果。我们表明，应根据任务选择测试，同样，词汇暂停信息和声学交叉注意的贡献不同。

### Literary and Colloquial Dialect Identification for Tamil using Acoustic Features 
[[arxiv](https://arxiv.org/abs/2408.14887)] [[cool](https://papers.cool/arxiv/2408.14887)] [[pdf](https://arxiv.org/pdf/2408.14887)]
> **Authors**: M. Nanmalar,P. Vijayalakshmi,T. Nagarajan
> **First submission**: 2024-08-27
> **First announcement**: 2024-08-28
> **comment**: submitted to TENCON 2019
- **标题**: 使用声学特征的文学和口语方言识别
- **领域**: 音频和语音处理,机器学习
- **摘要**: 一种语言的演变和多样性可以从各种方言中可以明显看出。如果在自动语音识别和语音综合等技术进步中未解决各种方言，则这些方言可能会消失。语音技术在保存语言的各种方言中发挥了作用，无法灭绝。为了构建一个解决各种方言的完整自动语音识别系统，需要自动方言识别（ADI）系统作为前端。这类似于语言识别系统如何充当处理多种语言的自动语音识别系统的前端。当前的工作提出了一种识别两种流行且广泛分类的泰米尔语言的方法，即文学和口语泰米尔语。使用声学特征，而不是语音和语音学，从而减轻了依赖语言的语言工具的要求。因此，提出的方法的一个主要优点是它不需要注释的语料库，因此很容易适应其他语言。使用MEL频率Cepstral系数（MFCC）功能的高斯混合模型（GMM）用于执行分类任务。实验的错误率为12％。讨论了元音鼻腔化，这是这种良好表现的原因。 GMM的混合模型数量各不相同，并分析了性能。

### WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling 
[[arxiv](https://arxiv.org/abs/2408.16532)] [[cool](https://papers.cool/arxiv/2408.16532)] [[pdf](https://arxiv.org/pdf/2408.16532)]
> **Authors**: Shengpeng Ji,Ziyue Jiang,Wen Wang,Yifu Chen,Minghui Fang,Jialong Zuo,Qian Yang,Xize Cheng,Zehan Wang,Ruiqi Li,Ziang Zhang,Xiaoda Yang,Rongjie Huang,Yidi Jiang,Qian Chen,Siqi Zheng,Zhou Zhao
> **First submission**: 2024-08-29
> **First announcement**: 2024-08-30
> **comment**: Accepted by ICLR 2025
- **标题**: wavtokenizer：有效的声学离散编解码器，用于音频语言建模
- **领域**: 音频和语音处理,机器学习,多媒体,声音,信号处理
- **摘要**: 语言模型已有效地应用于建模自然信号，例如图像，视频，语音和音频。这些模型的关键组成部分是编解码器令牌，它将高维自然信号压缩为较低维的离散令牌。在本文中，我们介绍了WavTokenizer，它比音频域中的SOTA声学编解码器模型具有多个优点：1）极端压缩。通过压缩量化器的层和离散编解码器的时间维度，一秒钟的24kHz采样率的音频仅需要具有40或75个令牌的单个量化器。 2）提高主观质量。尽管令牌数量减少，但WavTokenizer以出色的UTMOS得分达到了最先进的重建质量，并且本质上包含更丰富的语义信息。具体而言，我们通过设计更广泛的VQ空间，扩展上下文窗口和改进的注意力网络，并引入功能强大的多尺度歧视器和逆傅立叶变换结构来实现这些结果。我们在语音，音频和音乐领域进行了广泛的重建实验。与最先进的模型相比，Wavtokenizer在各种客观和主观指标上表现出很强的性能。我们还测试了语义信息，VQ利用率以及对生成模型的适应性。全面的消融研究证实了Wavtokenizer中每个模块的必要性。相关的代码，演示和预培训模型可在https://github.com/jishengpeng/wavtokenizer上找到。

## 其他论文

共有 16 篇其他论文

- [Interaural time difference loss for binaural target sound extraction](https://arxiv.org/abs/2408.00344)
  - **标题**: 双耳目标萃取的室内时间差损失
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Dual Threats in RIS-Aided RF-UOWC Mixed Networks: Secrecy Performance Analysis under Simultaneous RF and UOWC Eavesdropping](https://arxiv.org/abs/2408.06295)
  - **标题**: RIS辅助RF-UOWC混合网络中的双重威胁：同时RF和UOWC窃听下的保密性能分析
  - **Filtered Reason**: none of cs.IT in whitelist
- [Evaluating Source Code Quality with Large Language Models: a comparative study](https://arxiv.org/abs/2408.07082)
  - **标题**: 用大语言模型评估源代码质量：比较研究
  - **Filtered Reason**: none of cs.SE in whitelist
- [RPLUW/M: Enabling RPL on the Internet of Underwater Things](https://arxiv.org/abs/2408.08607)
  - **标题**: rpluw/m：在水下互联网上启用RPL
  - **Filtered Reason**: none of cs.NI,math.OC in whitelist
- [Dynamic Shaping of Multi-Touch Stimuli by Programmable Acoustic Metamaterial](https://arxiv.org/abs/2408.09829)
  - **标题**: 可编程声学超材料对多点触摸刺激的动态塑形
  - **Filtered Reason**: none of cs.HC,physics.app-ph in whitelist
- [Efficient Area-based and Speaker-Agnostic Source Separation](https://arxiv.org/abs/2408.09810)
  - **标题**: 有效的基于区域和扬声器的源源分离
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [R-STELLAR: A Resilient Synthesizable Signature Attenuation SCA Protection on AES-256 with built-in Attack-on-Countermeasure Detection](https://arxiv.org/abs/2408.12021)
  - **标题**: R-stellar：具有内置攻击式检测的AES-256上的可弹性综合签名衰减SCA保护
  - **Filtered Reason**: none of eess.SP,cs.CR in whitelist
- [Near-Field Signal Processing: Unleashing the Power of Proximity](https://arxiv.org/abs/2408.11434)
  - **标题**: 近场信号处理：释放接近的力量
  - **Filtered Reason**: none of eess.SP,cs.IT,eess.AS,cs.SD in whitelist
- [SonarWatch: Field sensing technique for smartwatches based on ultrasound and motion](https://arxiv.org/abs/2408.12689)
  - **标题**: Sonarwatch：基于超声和运动的智能手表的现场感应技术
  - **Filtered Reason**: none of cs.HC in whitelist
- [Exploring the Role of Audio in Multimodal Misinformation Detection](https://arxiv.org/abs/2408.12558)
  - **标题**: 探索音频在多模式错误信息检测中的作用
  - **Filtered Reason**: none of cs.MM in whitelist
- [The effect of self-motion and room familiarity on sound source localization in virtual environments](https://arxiv.org/abs/2408.13904)
  - **标题**: 自我运动和房间熟悉度对虚拟环境中声源本地化的影响
  - **Filtered Reason**: none of eess.AS,cs.HC,cs.SD in whitelist
- [Physics-Informed Machine Learning For Sound Field Estimation](https://arxiv.org/abs/2408.14731)
  - **标题**: 物理知识的机器学习，用于声音字段估计
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [A Preliminary Case Study on Long-Form In-the-Wild Audio Spoofing Detection](https://arxiv.org/abs/2408.14066)
  - **标题**: 关于长格式内部音频欺骗检测的初步案例研究
  - **Filtered Reason**: none of cs.CR,eess.AS,cs.SD in whitelist
- [Deep learning classification system for coconut maturity levels based on acoustic signals](https://arxiv.org/abs/2408.14910)
  - **标题**: 基于声学信号的椰子成熟度的深度学习分类系统
  - **Filtered Reason**: none of eess.SP,eess.AS,cs.SD in whitelist
- [Leveraging Self-supervised Audio Representations for Data-Efficient Acoustic Scene Classification](https://arxiv.org/abs/2408.14862)
  - **标题**: 利用自我监督的音频表示形式进行数据有效的声学场景分类
  - **Filtered Reason**: none of eess.AS,cs.SD in whitelist
- [Automatic detection of Mild Cognitive Impairment using high-dimensional acoustic features in spontaneous speech](https://arxiv.org/abs/2408.16732)
  - **标题**: 在自发语音中使用高维声学特征自动检测轻度认知障碍
  - **Filtered Reason**: none of q-bio.QM,q-bio.NC,eess.AS,cs.SD in whitelist
