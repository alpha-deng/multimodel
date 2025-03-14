# 2015-01 月度论文分类汇总

共有5篇相关领域论文, 另有2篇其他

## 计算语言学(cs.CL:Computation and Language)

该领域共有 2 篇论文

### Combining Language and Vision with a Multimodal Skip-gram Model 
[[arxiv](https://arxiv.org/abs/1501.02598)] [[cool](https://papers.cool/arxiv/1501.02598)] [[pdf](https://arxiv.org/pdf/1501.02598)]
> **Authors**: Angeliki Lazaridou,Nghia The Pham,Marco Baroni
> **First submission**: 2015-01-12
> **First announcement**: 2015-01-13
> **comment**: accepted at NAACL 2015, camera ready version, 11 pages
- **标题**: 将语言和视觉与多模式跳过模型相结合
- **领域**: 计算语言学,计算机视觉和模式识别,机器学习
- **摘要**: 我们扩展了Mikolov等人的跳过模型。 （2013a）考虑视觉信息。像Skip-gram一样，我们的多模式模型（MMSKIP-gram）通过学习预测文本语言中语言上下文来构建基于向量的单词表示。但是，对于一组受限的单词，这些模型还暴露于它们表示的对象的视觉表示（从自然图像中提取），并且必须共同预测语言和视觉特征。 mmskip-gram模型在各种语义基准上实现了良好的性能。此外，由于它们将视觉信息传播到所有单词中，因此我们使用它们来改善零拍设置中的图像标签和检索，在模型训练中从未见过测试概念。最后，mmskip-gram模型发现了抽象词的有趣的视觉属性，为实现含义理论的现实实现铺平了道路。

### Deep Multimodal Learning for Audio-Visual Speech Recognition 
[[arxiv](https://arxiv.org/abs/1501.05396)] [[cool](https://papers.cool/arxiv/1501.05396)] [[pdf](https://arxiv.org/pdf/1501.05396)]
> **Authors**: Youssef Mroueh,Etienne Marcheret,Vaibhava Goel
> **First submission**: 2015-01-22
> **First announcement**: 2015-01-23
> **comment**: ICASSP 2015
- **标题**: 视听语音识别的深度多模式学习
- **领域**: 计算语言学,机器学习
- **摘要**: 在本文中，我们介绍了深度多模式学习的方法，用于融合语音和视觉方式，以进行视听自动语音识别（AV-ASR）。首先，我们研究了一种单独训练的单模式深网，并将其最终的隐藏层融合以获得另一个深网的联合特征空间。虽然单独的音频网络在IBM大型词汇式音频录音室数据集中单独达到了$ 41 \％$的电话错误率（每）$ 41 \％$，但该融合模型的每 / 35.83美元\％$ $均为$ 35.83 \％$的每 / 35.83美元$ $，在噪声率高的信号中，手机分类中的巨大频道值也可以实现。其次，我们提出了一种新的深层网络体系结构，该架构使用双线性软磁层层来解释模态之间的类别特定相关性。我们表明，将双线性网络的后代与上面提到的融合模型的后代相结合，导致电话错误率的进一步降低，最终$ 34.03 \％$ $。

## 计算机视觉和模式识别(cs.CV:Computer Vision and Pattern Recognition)

该领域共有 1 篇论文

### ModDrop: adaptive multi-modal gesture recognition 
[[arxiv](https://arxiv.org/abs/1501.00102)] [[cool](https://papers.cool/arxiv/1501.00102)] [[pdf](https://arxiv.org/pdf/1501.00102)]
> **Authors**: Natalia Neverova,Christian Wolf,Graham W. Taylor,Florian Nebout
> **First submission**: 2014-12-31
> **First announcement**: 2015-01-02
> **comment**: 14 pages, 7 figures
- **标题**: ModDrop：自适应多模式识别
- **领域**: 计算机视觉和模式识别,人机交互,机器学习
- **摘要**: 我们提出了一种基于多尺度和多模式深度学习的手势检测和本地化的方法。每种视觉模态都以特定的空间尺度（例如上半身或手的运动）捕获空间信息，整个系统以三个时间尺度运行。我们技术的关键是一种利用的培训策略：i）仔细初始化个人方式； ii）逐渐融合，涉及用于学习跨模式相关性的单独通道（称为MODDROP）的随机降低，同时保留每个模式特异性表示的唯一性。我们介绍了Chalearn 2014年的实验，探讨了人们的挑战识别曲目，在该曲目中，我们将17支球队排名第一。在几个空间和时间尺度上融合多种方式会导致识别率显着提高，从而使模型可以补偿单个分类器的错误以及在单独的通道中的噪声。提议的ModDrop培训技术可确保分类器在一个或多个渠道中缺少信号的鲁棒性，以从任何可用的模式中产生有意义的预测。此外，我们通过在使用音频增强的同一数据集中进行的实验来证明拟议的融合方案对任意性质的方式的适用性。

## 机器学习(cs.LG:Machine Learning)

该领域共有 1 篇论文

### Particle swarm optimization for time series motif discovery 
[[arxiv](https://arxiv.org/abs/1501.07399)] [[cool](https://papers.cool/arxiv/1501.07399)] [[pdf](https://arxiv.org/pdf/1501.07399)]
> **Authors**: Joan Serrà,Josep Lluis Arcos
> **First submission**: 2015-01-29
> **First announcement**: 2015-01-30
> **comment**: 12 pages, 9 figures, 2 tables
- **标题**: 时间序列图案发现的粒子群优化
- **领域**: 机器学习,神经和进化计算
- **摘要**: 有效地在时间序列数据中找到类似的细分市场或基序是一项基本任务，由于这些数据的无处不在，在各种域和情况下都存在。因此，已经设计了无数的解决方案，但迄今为止，它们似乎都没有完全令人满意和灵活。在本文中，我们提出了一个创新的角度，并提出了一种解决方案：基于粒子群的时间序列图案发现的任何时间多模式优化算法。通过考虑来自各个域的数据，我们表明，与最先进的方法相比，该解决方案具有极具竞争力，在使用最小内存的时间内获得可比较的主题。此外，我们表明它对不同的实施选择是强大的，并且看到它在任务方面具有前所未有的灵活性。所有这些素质使提出的解决方案成为长期序列流中最著名的候选者之一。此外，我们认为，在进一步的时间序列分析和采矿任务中可以利用所提出的观点，从而扩大了研究范围，并可能产生新的有效解决方案。

## 光学(physics.optics:Optics)

该领域共有 1 篇论文

### Holistic random encoding for imaging through multimode fibers 
[[arxiv](https://arxiv.org/abs/1501.03997)] [[cool](https://papers.cool/arxiv/1501.03997)] [[pdf](https://arxiv.org/pdf/1501.03997)]
> **Authors**: Hwanchol Jang,Changhyeong Yoon,Euiheon Chung,Wonshik Choi,Heung-No Lee
> **First submission**: 2014-12-30
> **First announcement**: 2015-01-13
> **comment**: under review for possible publication in Optics express
- **标题**: 通过多模纤维成像成像的整体随机编码
- **领域**: 光学,计算机视觉和模式识别
- **摘要**: 多模纤维（MMF）的输入数值孔径（Na）可以通过将浊度介质放置在MMF的输入端来有效增加。这为通过MMF进行高分辨率成像提供了潜力。虽然输入NA增加，但MMF中的传播模式的数量，因此输出NA保持不变。这使图像重建过程不确定，并可能限制图像重建的质量。在本文中，我们旨在改善通过MMF成像中图像重建的信号与噪声比（SNR）。我们注意到，放置在MMF输入中的浑浊介质将传入波转换为更好的格式，以进行信息传输和信息提取。我们将这种转换称为整体随机（HR）编码浊度介质。通过利用人力资源编码，我们对图像重建的SNR进行了可观的改进。为了有效利用人力资源编码，我们采用稀疏表示（SR），这是一个相对较新的信号重建框架，当它提供HR编码信号时。这项研究首次表明我们知道利用浊度介质的HR编码在光学不确定的系统中恢复的好处，在光学不确定的系统中，其输出Na小于输入NA，用于通过MMF进行成像。

## 其他论文

共有 2 篇其他论文

- [Sparsity based Efficient Cross-Correlation Techniques in Sensor Networks](https://arxiv.org/abs/1501.06473)
  - **标题**: 传感器网络中基于稀疏性的有效互相关技术
  - **Filtered Reason**: none of cs.OH in whitelist
- [Resource Usage Estimation of Data Stream Processing Workloads in Datacenter Clouds](https://arxiv.org/abs/1501.07020)
  - **标题**: 数据流云中数据流处理工作负载的资源用法估计
  - **Filtered Reason**: none of cs.DB in whitelist
