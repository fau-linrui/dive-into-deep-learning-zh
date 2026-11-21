# 现代卷积神经网络
:label:`chap_modern_cnn`

既然我们已经理解了把 CNN 连接起来的基础知识，就让我们来
巡览现代 CNN 架构。由于不断涌现的
令人兴奋的新设计实在太多，这次巡览
必然是不完整的。它们的重要性在于：
它们不仅可以直接用于视觉任务，
而且还可以作为基础特征生成器，服务于更高级的任务，如跟踪
:cite:`Zhang.Sun.Jiang.ea.2021`、分割 :cite:`Long.Shelhamer.Darrell.2015`、目标
检测 :cite:`Redmon.Farhadi.2018` 或样式变换
:cite:`Gatys.Ecker.Bethge.2016`。在本章中，大多数小节
对应于一个重要的 CNN 架构，它曾在某个时期
（或目前）是许多研究项目和
已部署系统所基于的基础模型。这些网络中的每一个
都曾一度是主导架构，其中许多在
[ImageNet 竞赛](https://www.image-net.org/challenges/LSVRC/)
中获胜或获得亚军——自2010年以来，该竞赛
一直是计算机视觉中监督学习进展的晴雨表。直到最近，
Transformer 才开始取代 CNN，这一趋势从 :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` 开始，
随后是 Swin Transformer :cite:`liu2021swin`。我们将在
:numref:`chap_attention-and-transformers` 中稍后介绍这一发展。

虽然*深度*神经网络的思想相当简单（把
一堆层堆叠在一起），但不同
架构和超参数选择下的性能可能千差万别。本章
描述的神经网络是直觉、少量
数学洞见以及大量试错的产物。我们按
时间顺序介绍这些模型，部分是为了传达一种
历史感，以便你能对领域的发展方向
形成自己的直觉，或许还能设计出自己的架构。例如，
本章描述的批量规范化和残差连接
为训练和设计深度模型提供了两种流行的思想，
此后这两者都被应用于计算机
视觉之外的架构中。

我们从 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` 开始现代 CNN 的巡览，
它是第一个在大规模视觉挑战赛上击败传统计算机
视觉方法的大规模网络；VGG 网络
:cite:`Simonyan.Zisserman.2014`，它使用了若干
重复的元素块；Network in Network（NiN），它
用整个神经网络在输入上逐图像块地进行卷积
:cite:`Lin.Chen.Yan.2013`；GoogLeNet，它使用具有
多分支卷积的网络 :cite:`Szegedy.Liu.Jia.ea.2015`；残差
网络（ResNet）:cite:`He.Zhang.Ren.ea.2016`，它至今仍是
计算机视觉中最流行的现成架构之一；
用于更稀疏连接的 ResNeXt 块 :cite:`Xie.Girshick.Dollar.ea.2017`；
以及用于残差架构推广的 DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。随着时间的推移，人们为高效
网络开发了许多专门的优化，例如坐标移位（ShiftNet）:cite:`wu2018shift`。这
最终促成了对高效架构的自动搜索，例如
MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`。它还包括
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020`
的半自动设计探索，由此产生了我们将在本章后面讨论的 RegNetX/Y。
这项工作具有启发性，因为它提供了一条将暴力计算与
实验者的聪明才智结合起来、以搜索高效设计空间的路径。同样值得注意的是
:citet:`liu2022convnet` 的工作，因为它表明训练技术（例如优化器、数据增广和正则化）
在提高准确率方面起着关键作用。它还表明，随着计算量和
数据的增加，一些长期持有的假设（例如
卷积窗口的大小）可能需要重新审视。我们将在本章中适时涵盖这一点以及更多问题。

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```
