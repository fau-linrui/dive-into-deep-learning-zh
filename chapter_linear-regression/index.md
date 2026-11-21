# 用于回归的线性神经网络
:label:`chap_regression`

在考虑如何把神经网络做深之前，先实现一些浅层网络会很有帮助，
因为这类网络的输入直接连接到输出。
这样做有几个重要原因。
首先，我们不必被复杂的架构分散注意力，
而可以专注于神经网络训练的基础知识，
包括参数化输出层、处理数据、
指定损失函数以及训练模型。
其次，这类浅层网络恰好
构成了线性模型这一集合，
它涵盖了许多经典的统计预测方法，
包括线性回归和 softmax 回归。
理解这些经典工具至关重要，
因为它们在许多场景中被广泛使用，
而且在论证使用更花哨的架构时，
我们常常需要将它们用作基准。
本章将集中讨论线性回归，
下一章则通过开发用于分类的线性神经网络，
来扩展我们的建模工具箱。

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```
