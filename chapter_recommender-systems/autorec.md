# AutoRec：用自编码器进行评分预测

尽管矩阵分解模型在评分预测任务上取得了不错的性能，但它在本质上是一个线性模型。因此，这类模型无法捕捉那些可能预示用户偏好的复杂非线性关系和错综复杂的关联。在本节中，我们介绍一种非线性神经网络协同过滤模型——AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`。它把协同过滤（CF）与自编码器架构相结合，旨在在显式反馈的基础上把非线性变换融入协同过滤。神经网络已被证明能够逼近任意连续函数，因此适合用来解决矩阵分解的局限性，并丰富矩阵分解的表达能力。

一方面，AutoRec 的结构与自编码器相同，由输入层、隐藏层和重构（输出）层组成。自编码器是一种学习把输入复制到输出的神经网络，其目的是把输入编码为隐藏的（通常是低维的）表示。在 AutoRec 中，它不把用户/物品显式嵌入低维空间，而是使用交互矩阵的列/行作为输入，然后在输出层中重构交互矩阵。

另一方面，AutoRec 与传统的自编码器不同：它不是学习隐藏表示，而是侧重于学习/重构输出层。它使用部分观测的交互矩阵作为输入，旨在重构一个完整的评分矩阵。同时，输入的缺失项通过输出层的重构被填充，以实现推荐的目的。

AutoRec 有两种变体：基于用户的与基于物品的。为简洁起见，这里我们只介绍基于物品的 AutoRec。基于用户的 AutoRec 可以相应地推导出来。


## 模型

令 $\mathbf{R}_{*i}$ 表示评分矩阵的第 $i^\textrm{th}$ 列，其中未知评分默认设为零。神经架构定义为：

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

其中 $f(\cdot)$ 和 $g(\cdot)$ 表示激活函数，$\mathbf{W}$ 和 $\mathbf{V}$ 是权重矩阵，$\mu$ 和 $b$ 是偏置。令 $h( \cdot )$ 表示 AutoRec 的整个网络。输出 $h(\mathbf{R}_{*i})$ 是评分矩阵第 $i^\textrm{th}$ 列的重构。

下面的目标函数旨在最小化重构误差：

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

其中 $\| \cdot \|_{\mathcal{O}}$ 表示只考虑已观测评分的贡献，也就是说，在反向传播期间只更新与已观测输入相关联的权重。

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## 实现模型

典型的自编码器由编码器和解码器组成。编码器把输入投影为隐藏表示，解码器把隐藏层映射到重构层。我们遵循这种做法，用全连接层创建编码器和解码器。编码器的激活函数默认设为 `sigmoid`，解码器不应用激活函数。在编码变换之后加入暂退法（dropout），以减少过拟合。未观测输入的梯度被掩码，以确保只有已观测评分对模型学习过程有贡献。

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 重新实现评估器

由于输入和输出已经改变，我们需要重新实现评估函数，不过我们仍然使用 RMSE 作为衡量准确性的指标。

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 训练和评估模型

现在，让我们在 MovieLens 数据集上训练并评估 AutoRec。我们可以清楚地看到，测试 RMSE 低于矩阵分解模型，这证实了神经网络在评分预测任务中的有效性。

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## 小结

* 我们可以用自编码器来构建矩阵分解算法，同时融入非线性层和暂退法正则化。
* 在 MovieLens 100K 数据集上的实验表明，AutoRec 取得了优于矩阵分解的性能。



## 练习

* 改变 AutoRec 的隐藏维度，观察它对模型性能的影响。
* 尝试加入更多隐藏层。这有助于提升模型性能吗？
* 你能找到更好的解码器和编码器激活函数组合吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/401)
:end_tab:
