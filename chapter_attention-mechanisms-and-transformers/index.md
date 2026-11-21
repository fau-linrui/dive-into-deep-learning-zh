# 注意力机制与 Transformer
:label:`chap_attention-and-transformers`


深度学习热潮的最初几年，主要由多层感知机、卷积网络和循环网络架构所取得的成果推动。
值得注意的是，尽管时间已过去近 30 年，支撑 2010 年代深度学习众多突破的模型架构
相对于其前身的变化却小得惊人。
虽然大量新的方法学创新进入了大多数从业者的工具箱——ReLU 激活函数、残差层、
批量规范化、暂退法以及自适应学习率调度等就跃然眼前——
但核心的底层架构显然仍可辨认出是经典思想的放大实现。
尽管有成千上万篇论文提出了替代性的思想，
但类似经典卷积神经网络（:numref:`chap_cnn`）的模型仍在计算机视觉领域保持
*最先进*（state-of-the-art）地位，
而类似 Sepp Hochreiter 最初为 LSTM 循环神经网络（:numref:`sec_lstm`）所做设计的模型，
则主导了自然语言处理领域的大多数应用。
可以说，截至当时，深度学习的迅速崛起似乎主要归因于可用计算资源的变化
（得益于 GPU 并行计算的创新）以及海量数据资源的可得性
（得益于廉价的存储和互联网服务）。
这些因素或许确实仍是这项技术能力不断增强的主要驱动力，
但我们终于也开始目睹主导架构格局的巨变。

目前，几乎所有自然语言处理任务的主导模型都基于 Transformer 架构。
给定任何新的自然语言处理任务，默认的第一轮尝试做法是：
取一个基于 Transformer 的大型预训练模型
（例如 BERT :cite:`Devlin.Chang.Lee.ea.2018`、ELECTRA :cite:`clark2019electra`、
RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` 或 Longformer :cite:`beltagy2020longformer`），
按需调整输出层，再在可用数据上针对下游任务微调该模型。
如果你关注过最近几年围绕 OpenAI 大型语言模型的令人目不暇接的新闻报道，
那么你一直在追踪的其实是一场以 GPT-2 和 GPT-3 这两个基于 Transformer 的模型为核心的讨论
:cite:`Radford.Wu.Child.ea.2019,brown2020language`。
与此同时，视觉 Transformer 已成为各种视觉任务的默认模型，
包括图像识别、目标检测、语义分割和超分辨率
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`。
Transformer 还作为有竞争力的方法出现在语音识别 :cite:`gulati2020conformer`、
强化学习 :cite:`chen2021decision` 和图神经网络 :cite:`dwivedi2020generalization` 中。

Transformer 模型背后的核心思想是*注意力机制*（attention mechanism），
这一创新最初被设想为对应用于序列到序列任务
（例如机器翻译）的编码器-解码器 RNN 的一种增强手段 :cite:`Bahdanau.Cho.Bengio.2014`。
你可能还记得，在最早的机器翻译序列到序列模型 :cite:`Sutskever.Vinyals.Le.2014` 中，
整个输入由编码器压缩成单个固定长度的向量，再送入解码器。
注意力背后的直觉是：与其压缩输入，不如让解码器在每一步重新访问输入序列，这或许更好。
此外，与其总是看到相同的输入表示，可以设想解码器应当在特定的解码步上
选择性地关注输入序列的特定部分。
Bahdanau 的注意力机制提供了一种简单手段，
使解码器能够在每个解码步动态地*关注*输入的不同部分。
其高层思想是：编码器可以产生一个长度与原输入序列相等的表示。
然后，在解码时，解码器可以（通过某种控制机制）接收一个上下文向量作为输入，
该向量由输入在每个时间步上的表示的加权和构成。
直观地说，这些权重决定了每一步的上下文在每个输入词元上的“聚焦”程度，
而关键在于让这个分配权重的过程可微，
以便它能与所有其他神经网络参数一起被学习。

最初，这一思想是对当时已主导机器翻译应用的循环神经网络的极其成功的增强。
这些模型的表现优于最初的编码器-解码器序列到序列架构。
此外，研究者注意到，观察注意力权重的模式有时会带来一些有益的定性洞见。
在翻译任务中，注意力模型在生成目标语言中的对应词时，
往往会给跨语言同义词分配较高的注意力权重。
例如，在把句子“my feet hurt”翻译成“j'ai mal au pieds”时，
神经网络在生成对应的法语词“pieds”时可能会给“feet”的表示分配较高的注意力权重。
这些洞见引发了这样的说法：注意力模型带来了“可解释性”。
不过，注意力权重究竟意味着什么——即它们应当如何被*解读*（如果真能解读的话）——
仍然是一个模糊的研究课题。

然而，注意力机制的意义很快就超出了“作为编码器-解码器循环神经网络的增强手段”
以及“挑选显著输入的假定用途”，成为更重要的问题。
:citet:`Vaswani.Shazeer.Parmar.ea.2017` 为机器翻译提出了 Transformer 架构，
完全舍弃了循环连接，转而依靠巧妙安排的注意力机制来捕获
输入与输出词元之间的全部关系。
该架构的表现极为出色，到 2018 年，Transformer 开始出现在
大多数最先进的自然语言处理系统中。
不仅如此，与此同时，自然语言处理领域的主流做法也变成：
在庞大的通用背景语料库上预训练大规模模型，以优化某种自监督的预训练目标，
然后利用可用的下游数据对这些模型进行微调。
在这一预训练范式下，Transformer 与传统架构之间的差距变得尤其大，
因此 Transformer 的崛起与这类大规模预训练模型的崛起同时发生，
这类模型如今有时被称为*基础模型*（foundation models） :cite:`bommasani2021opportunities`。


在本章中，我们将介绍注意力模型，
从最基本的直觉和该思想最简单的实例开始，
然后逐步推进到 Transformer 架构、视觉 Transformer，
以及现代基于 Transformer 的预训练模型的图景。

```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```
