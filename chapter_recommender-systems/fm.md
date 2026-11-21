# 因子分解机

因子分解机（factorization machine，FM）由 :citet:`Rendle.2010` 提出，是一种监督学习算法，可用于分类、回归和排序任务。它很快引起关注，并成为预测和推荐中流行且有影响力的方法。特别地，它是线性回归模型和矩阵分解模型的推广。此外，它还让人联想到使用多项式核的支持向量机。与线性回归和矩阵分解相比，因子分解机的优势在于：(1) 它可以对 $\chi$ 阶变量交互建模，其中 $\chi$ 是多项式阶数，通常设为 2；(2) 与因子分解机相关的快速优化算法可以将多项式计算时间降至线性复杂度，使其尤其在高维稀疏输入下极为高效。基于这些原因，因子分解机被广泛应用于现代广告和产品推荐中。下面介绍其技术细节与实现。


## 二阶因子分解机

形式化地，设 $x \in \mathbb{R}^d$ 表示一个样本的特征向量，$y$ 表示相应的标签，它可以是实值标签，也可以是诸如二分类“点击/未点击”的类别标签。二阶因子分解机的模型定义如下：

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

其中 $\mathbf{w}_0 \in \mathbb{R}$ 是全局偏置；$\mathbf{w} \in \mathbb{R}^d$ 表示第 i 个变量的权重；$\mathbf{V} \in \mathbb{R}^{d\times k}$ 表示特征嵌入；$\mathbf{v}_i$ 表示 $\mathbf{V}$ 的第 $i^\textrm{th}$ 行；$k$ 是潜在因子的维数；$\langle\cdot, \cdot \rangle$ 是两个向量的点积。$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 对第 $i^\textrm{th}$ 个特征与第 $j^\textrm{th}$ 个特征之间的交互建模。有些特征交互很容易理解，因此可以由专家设计。然而，大多数其他特征交互隐藏在数据之中，难以识别。因此，自动地对特征交互建模可以大大减少特征工程的工作量。显然，前两项对应于线性回归模型，而最后一项是矩阵分解模型的扩展。如果特征 $i$ 表示物品、特征 $j$ 表示用户，那么第三项恰好是用户嵌入与物品嵌入之间的点积。值得注意的是，FM 也可以推广到更高的阶数（阶数 > 2）。不过，数值稳定性可能会削弱这种推广。


## 一种高效的优化准则

如果以直接的方式优化因子分解机，则复杂度为 $\mathcal{O}(kd^2)$，因为需要计算所有成对交互。为了解决这一低效问题，我们可以重新组织 FM 的第三项，从而大幅降低计算成本，得到线性时间复杂度（$\mathcal{O}(kd)$）。成对交互项的重新表述如下：

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

通过这一重新表述，模型复杂度大大降低。此外，对于稀疏特征，只需计算非零元素，因此总体复杂度与非零特征的数量成线性关系。

为了学习 FM 模型，我们可以对回归任务使用 MSE 损失，对分类任务使用交叉熵损失，对排序任务使用 BPR 损失。随机梯度下降和 Adam 等标准优化器均可用于优化。

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## 模型实现
下面的代码实现了因子分解机。可以清楚地看到，FM 由一个线性回归块和一个高效的特征交互块组成。由于我们把点击率预测视为分类任务，因此对最终得分施加 sigmoid 函数。

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## 加载广告数据集
我们使用上一节的 CTR 数据包装器来加载在线广告数据集。

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 训练模型
随后，我们训练模型。学习率设为 0.02，嵌入大小默认设为 20。模型训练使用 `Adam` 优化器和 `SigmoidBinaryCrossEntropyLoss` 损失。

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 小结

* FM 是一个通用框架，可以应用于回归、分类和排序等多种任务。
* 特征交互/交叉对预测任务很重要，而二阶交互可以用 FM 高效地建模。

## 练习

* 你能在其他数据集（如 Avazu、MovieLens 和 Criteo 数据集）上测试 FM 吗？
* 改变嵌入大小以考察其对性能的影响，你能观察到与矩阵分解类似的变化模式吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/406)
:end_tab:
