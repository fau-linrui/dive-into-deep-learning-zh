```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 批量规范化
:label:`sec_batch_norm`

训练深度神经网络是困难的。
让它们在合理的时间内收敛可能很棘手。
在本节中，我们介绍*批量规范化*（batch normalization），这是一种流行且有效的技术，
它能稳定地加速深度网络的收敛 :cite:`Ioffe.Szegedy.2015`。
批量规范化与残差块一起——后者将在 :numref:`sec_resnet` 中介绍——使得从业者能够常规地训练超过 100 层的网络。
批量规范化的一个次要（偶然的）好处在于其固有的正则化作用。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## 训练深层网络

在处理数据时，我们经常在训练前进行预处理。
关于数据预处理的选择往往会对最终结果产生巨大差异。
回想我们将多层感知机应用于房价预测（:numref:`sec_kaggle_house`）。
处理真实数据时的第一步
是将输入特征标准化，使其在多个观测上具有
零均值 $\boldsymbol{\mu} = 0$ 和单位方差 $\boldsymbol{\Sigma} = \boldsymbol{1}$ :cite:`friedman1987exploratory`，我们常常还会重新缩放后者，使对角线为 1，即 $\Sigma_{ii} = 1$。
另一种策略是将向量重新缩放到单位长度，且可能使*每个观测*的均值为零。
例如，对于空间传感器数据，这可以很好地奏效。这些预处理技术以及许多其他技术
都有助于使估计问题保持可控。
关于特征选择和提取的综述，例如可以参见 :citet:`guyon2008feature` 的文章。
向量标准化还有一个有益的副作用，即约束作用于其上的函数的复杂度。例如，支持向量机中著名的半径-间隔界 :cite:`Vapnik95` 以及感知机收敛定理 :cite:`Novikoff62` 都依赖于有界范数的输入。

直观地说，这种标准化与我们的优化器配合得很好，
因为它使参数*先验地*具有相似的尺度。
因此，自然会问：在深度网络*内部*进行相应的规范化步骤
是否也可能有益。虽然这并不是导致批量规范化 :cite:`Ioffe.Szegedy.2015` 发明的推理，但它是在统一框架下理解批量规范化及其同类层规范化 :cite:`Ba.Kiros.Hinton.2016` 的有用方式。

其次，对于典型的 MLP 或 CNN，在训练过程中，
中间层中的变量
（例如 MLP 中仿射变换的输出）
可能取具有差异巨大的量级的数值：
无论是在从输入到输出的各层之间、同一层内的各个单元之间，
还是由于我们对模型参数进行更新而随时间变化。
批量规范化的发明者非正式地假设，
这种变量分布的漂移可能会妨碍网络的收敛。
直观地说，我们可能会猜想，如果一层
的变量激活值是另一层的 100 倍，
这可能需要对学习率进行补偿性调整。诸如 AdaGrad :cite:`Duchi.Hazan.Singer.2011`、Adam :cite:`Kingma.Ba.2014`、Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` 或 Distributed Shampoo :cite:`anil2020scalable` 之类的自适应求解器旨在从优化的角度解决这个问题，例如通过加入二阶方法的要素。
另一种替代方案是通过自适应规范化直接防止问题发生。

第三，更深的网络很复杂，往往更容易过拟合。
这意味着正则化变得更加关键。一种常见的正则化技术是噪声注入。
这一点早已为人所知，例如关于对输入的噪声注入 :cite:`Bishop.1995`。
它也构成了 :numref:`sec_dropout` 中暂退法的基础。事实证明，非常偶然地，批量规范化带来了所有三种好处：预处理、数值稳定性和正则化。

批量规范化应用于个别层，或者可选地应用于所有层：
在每个训练迭代中，
我们首先通过减去均值
并除以标准差
来规范化（批量规范化的）输入，
其中均值和标准差都基于当前小批量的统计量来估计。
接下来，我们应用一个缩放系数和一个偏移量，以恢复失去的自由度。
正是由于这种基于*批量*统计量的*规范化*，
*批量规范化*才得名。

请注意，如果我们尝试对大小为 1 的小批量应用批量规范化，
我们将无法学到任何东西。
这是因为减去均值之后，
每个隐藏单元的值都将为 0。
正如你可能猜到的，既然我们用整整一节来介绍批量规范化，
那么在小批量足够大时，该方法被证明是有效且稳定的。
这里的一个要点是，在应用批量规范化时，
批量大小的选择
甚至比不使用批量规范化时更为重要，或者说，
我们可能需要在小批量大小调整时进行适当的校准。

用 $\mathcal{B}$ 表示一个小批量，并令 $\mathbf{x} \in \mathcal{B}$ 为
批量规范化（$\textrm{BN}$）的一个输入。在这种情况下，批量规范化定义如下：

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在 :eqref:`eq_batchnorm` 中，
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ 是样本均值，
$\hat{\boldsymbol{\sigma}}_\mathcal{B}$ 是小批量 $\mathcal{B}$ 的样本标准差。
应用标准化之后，
得到的小批量
具有零均值和单位方差。
选择单位方差
（而不是其他某个魔法数字）是任意的。我们通过包含与 $\mathbf{x}$ 形状相同的
逐元素*缩放参数* $\boldsymbol{\gamma}$ 和*偏移参数* $\boldsymbol{\beta}$
来恢复这个自由度。二者都是
作为模型训练的一部分需要学习的参数。

在训练期间，中间层的变量量级
不会发散，
因为批量规范化会主动将它们中心化并重新缩放
到一个给定的均值和大小（通过 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$）。
实践经验证实，正如在讨论特征重缩放时提到的，批量规范化似乎允许使用更激进的学习率。
我们按如下方式计算 :eqref:`eq_batchnorm` 中的 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$：

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

请注意，我们在方差估计中
加入一个小的常数 $\epsilon > 0$，
以确保我们永远不会尝试除以零，
即使在经验方差估计可能非常小甚至消失的情况下也是如此。
估计值 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ 利用均值和方差的有噪声估计
来抵消缩放问题。
你可能认为这种噪声应该是个问题。
恰恰相反，它实际上是有益的。

事实证明，这是深度学习中反复出现的一个主题。
由于尚未在理论上得到很好刻画的原因，
优化中的各种噪声源
往往会导致训练更快、过拟合更少：
这种变化似乎起到了一种正则化的作用。
:citet:`Teye.Azizpour.Smith.2018` 和 :citet:`Luo.Wang.Shao.ea.2018`
分别将批量规范化的性质与贝叶斯先验和惩罚联系起来。
特别是，这为如下谜题提供了一些线索：
为什么批量规范化在 50 到 100 的中等小批量大小下效果最好。
这种特定大小的小批量似乎为每一层注入了"恰到好处"的噪声，无论是在通过 $\hat{\boldsymbol{\sigma}}$ 体现的尺度方面，还是在通过 $\hat{\boldsymbol{\mu}}$ 体现的偏移方面：
较大的小批量由于估计更稳定而正则化更少，而微小的小批量
则由于高方差而破坏有用的信号。沿着这个方向进一步探索，考虑其他类型的
预处理和滤波，或许还会带来其他有效的正则化类型。

对于训练好的模型，你可能会认为
我们更愿意使用整个数据集
来估计均值和方差。
一旦训练完成，为什么我们希望
同一张图像
根据它恰好所在的那一批而被分类为不同结果呢？
在训练期间，这种精确计算是不可行的，
因为所有数据样例的中间变量
在我们每次更新模型时都会改变。
然而，一旦模型训练完成，
我们就可以基于整个数据集
计算每一层变量的均值和方差。
事实上，对于使用批量规范化的模型来说，
这是标准做法；
因此批量规范化层在*训练模式*下（用小批量统计量规范化）
与在*预测模式*下（用数据集统计量规范化）的行为不同。
在这种形式下，它们与 :numref:`sec_dropout` 中暂退法正则化的行为非常相似，
其中噪声仅在训练期间注入。


## 批量规范化层

用于全连接层
和卷积层的批量规范化实现在细节上略有不同。
批量规范化与其他层之间的一个关键区别
在于：由于前者一次作用于整个小批量，
我们不能像之前介绍其他层时那样
直接忽略批量维度。

### 全连接层

当把批量规范化应用于全连接层时，
:citet:`Ioffe.Szegedy.2015` 在他们最初的论文中，将批量规范化插入在仿射变换之后、
非线性激活函数*之前*。后来的应用则尝试
将批量规范化插入在激活函数*之后*。
用 $\mathbf{x}$ 表示全连接层的输入，
用 $\mathbf{W}\mathbf{x} + \mathbf{b}$ 表示仿射变换（其中权重参数为 $\mathbf{W}$，偏置参数为 $\mathbf{b}$），
用 $\phi$ 表示激活函数，
我们可以将启用批量规范化的、
全连接层输出 $\mathbf{h}$ 的计算表示如下：

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值与方差是在
应用变换的*同一个*小批量上
计算的。

### 卷积层

同样，对于卷积层，
我们可以在卷积之后、
非线性激活函数之前应用批量规范化。与全连接层中批量规范化的
关键区别在于，我们是在*所有位置*上
按通道应用该操作。这与我们导出卷积时所依据的平移
不变性假设是相容的：我们假设图像中
模式的特定位置对于理解的目的并不关键。

假设我们的小批量包含 $m$ 个样例，
并且对于每个通道，
卷积的输出具有高度 $p$ 和宽度 $q$。
对于卷积层，我们在每个输出通道的 $m \cdot p \cdot q$ 个元素上同时执行批量规范化。
因此，在计算均值和方差时，
我们汇集所有空间位置上的值，
从而
在给定通道内使用相同的均值和方差
来规范化每个空间位置上的值。
每个通道都有自己的缩放参数和偏移参数，
二者都是标量。

### 层规范化
:label:`subsec_layer-normalization-in-bn`

请注意，在卷积的语境下，即使小批量大小为 1，
批量规范化也是定义良好的：毕竟，我们有图像中的所有位置可供平均。因此，
即使只是在单个观测内，均值和方差也是定义良好的。这一考虑
促使 :citet:`Ba.Kiros.Hinton.2016` 引入了*层规范化*（layer normalization）的概念。它的工作方式与
批量规范化一样，只是它一次应用于一个观测。因此，偏移量和缩放因子都是标量。对于一个 $n$ 维向量 $\mathbf{x}$，层规范化由下式给出：

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

其中缩放和偏移是逐系数应用的，
并由下式给出：

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

和之前一样，我们加入一个小的偏移量 $\epsilon > 0$ 以防止除以零。使用层规范化的主要好处之一是它可以防止发散。毕竟，忽略 $\epsilon$，
层规范化的输出是与尺度无关的。也就是说，对于任意 $\alpha \neq 0$ 的选择，都有 $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$。当 $|\alpha| \to \infty$ 时，这变为等式（近似相等是由于方差的偏移量 $\epsilon$ 造成的）。

层规范化的另一个优点是它不依赖于小批量大小。它也独立于我们处于训练状态还是测试状态。换句话说，它只是一个确定性的变换，将激活值标准化到给定的尺度。这在防止优化过程中发散方面可能非常有益。我们跳过进一步的细节，并建议感兴趣的读者查阅原论文。

### 预测过程中的批量规范化

正如我们之前提到的，批量规范化在训练模式下的行为通常
与预测模式下不同。
首先，在小批量上估计样本均值和样本方差
所产生的噪声，
在我们训练好模型之后就不再需要了。
其次，我们可能没有
逐批计算规范化统计量的奢侈条件。
例如，
我们可能需要每次只应用模型做一次预测。

通常，训练之后，我们使用整个数据集
来计算变量统计量的稳定估计，
然后在预测时固定它们。
因此，批量规范化在训练期间的行为与测试时不同。
回想一下，暂退法也具有这一特征。

## (**从零开始实现**)

为了了解批量规范化在实践中如何工作，我们在下面从零开始实现一个。

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

现在我们可以[**创建一个合适的 `BatchNorm` 层。**]
我们的层将维护适当的
缩放参数 `gamma` 和偏移参数 `beta`，
二者都会在训练过程中更新。
此外，我们的层将维护
均值和方差的移动平均，
供模型预测时后续使用。

暂且不谈算法细节，
请注意我们实现该层所依据的设计模式。
通常，我们在一个单独的函数中定义数学部分，比如 `batch_norm`。
然后我们将这一功能集成到一个自定义层中，
其代码主要处理簿记事务，
例如将数据移动到正确的设备上下文、
分配并初始化所需的变量、
跟踪移动平均（这里是均值和方差）等。
这种模式使数学部分与样板代码得以清晰分离。
还要注意，为方便起见，
我们这里没有考虑自动推断输入形状；
因此我们需要始终指定特征数。
到目前为止，所有现代深度学习框架都在
高层批量规范化 API 中提供尺寸和形状的自动检测（实践中我们将改用后者）。

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

我们使用 `momentum` 来控制对过去均值和方差估计的聚合。这在一定程度上是个误称，因为它与优化中的*动量*项毫无关系。尽管如此，它是这一术语被普遍采用的名称，并且遵照 API 命名惯例，我们在代码中使用了相同的变量名。

## [**使用批量规范化的 LeNet**]

为了了解如何在具体情境中应用 `BatchNorm`，
我们在下面将其应用于一个传统的 LeNet 模型（:numref:`sec_lenet`）。
回想一下，批量规范化应用在卷积层或全连接层
之后，但在相应的激活函数之前。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
由于 `BatchNorm` 层需要计算批量统计量
（均值和方差），Flax 会跟踪 `batch_stats` 字典，并在
每个小批量时更新它们。像 `batch_stats` 这样的集合可以
作为属性存储在 `TrainState` 对象（在 :numref:`oo-design-training` 中定义的
`d2l.Trainer` 类中）里，在模型的前向传播过程中，
应将这些集合传给 `mutable` 参数，以便 Flax 返回被修改的
变量。
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

和之前一样，我们将[**在 Fashion-MNIST 数据集上训练我们的网络**]。
这段代码与我们最初训练 LeNet 时几乎完全相同。

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

让我们来看看从第一个批量规范化层学到的
[**缩放参数 `gamma`
和偏移参数 `beta`**]。

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**简洁实现**]

与我们刚刚自己定义的 `BatchNorm` 类相比，
我们可以直接使用深度学习框架高层 API 中定义的 `BatchNorm` 类。
代码看起来与我们上面的实现几乎完全相同，
只是我们不再需要提供额外的参数来让它获得正确的维数。

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

下面，我们[**使用相同的超参数来训练我们的模型。**]
请注意，和往常一样，高层 API 版本运行得快得多，
因为它的代码已被编译为 C++ 或 CUDA，
而我们的自定义实现必须由 Python 解释执行。

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## 讨论

直观地说，批量规范化被认为
能使优化地形更加平滑。
然而，我们必须小心区分
推测性的直觉与对我们训练深度模型时所观察到的现象的
真正解释。
回想一下，我们甚至不知道更简单的
深度神经网络（MLP 和传统 CNN）
为何首先能很好地泛化。
即使有暂退法和权重衰减，
它们仍然如此灵活，以至于它们泛化到未见数据的能力
很可能需要精细得多的学习理论泛化保证。

提出批量规范化的原始论文 :cite:`Ioffe.Szegedy.2015` 除了引入一个强大而有用的工具外，
还给出了它为何有效的解释：
通过减少*内部协变量偏移*（internal covariate shift）。
大概地，*内部协变量偏移*意指
上面所表达的直观认识——变量值的分布
在训练过程中会发生变化。
然而，这一解释存在两个问题：
i）这种漂移与*协变量偏移*非常不同，
使得这个名称名不副实。如果说有什么相似之处，它更接近概念漂移。
ii）这一解释提供了一种不够具体的直觉，
但把*为什么恰恰是这种技术有效*这一问题
留作一个有待严格解释的开放问题。
在整本书中，我们旨在传达从业者
用来指导其深度神经网络开发的直觉。
然而，我们认为，将这些指导性直觉
与已确立的科学事实区分开来很重要。
最终，当你掌握了这些材料
并开始撰写自己的研究论文时，
你会希望清楚地区分
技术性论断和直觉。

随着批量规范化的成功，
其基于*内部协变量偏移*的解释
在技术文献的争论中，以及关于如何呈现机器学习研究的
更广泛讨论中反复出现。
在 2017 年 NeurIPS 会议上接受“时间检验奖”（Test of Time Award）
时发表的一篇令人难忘的演讲中，
Ali Rahimi 将*内部协变量偏移*
作为论据的焦点，把现代深度学习的实践
比作炼金术。
随后，这个例子在一篇概述机器学习中
令人不安的趋势的立场论文 :cite:`Lipton.Steinhardt.2018` 中被详细重新审视。
其他作者
也对批量规范化的成功提出了替代解释，
其中一些 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
声称批量规范化取得了成功，尽管它表现出在某些方面
与原始论文所声称的相反的行为。


我们注意到，*内部协变量偏移*
并不比机器学习技术文献中
每年提出的成千上万个同样含糊的论断更值得批评。
它之所以能作为这些争论的焦点引起共鸣，
很可能是因为它对目标受众具有广泛的辨识度。
批量规范化已被证明是一种不可或缺的方法，
应用于几乎所有部署的图像分类器中，
使引入该技术的论文获得了
数万次引用。不过，我们猜想，通过噪声注入实现正则化、
通过重新缩放实现加速以及预处理这些指导原则，
很可能在未来带来更多的层和技术的发明。

从更实际的角度来看，关于批量规范化有若干方面值得记住：

* 在模型训练期间，批量规范化利用小批量的均值和标准差不断调整
  网络的中间输出，从而使整个神经网络中每一层
  中间输出的值更加稳定。
* 批量规范化在全连接层和卷积层中略有不同。事实上，
  对于卷积层，有时可以使用层规范化作为替代方案。
* 与暂退法层一样，批量规范化层在训练模式下的行为
  与预测模式下不同。
* 批量规范化有助于正则化和改善优化中的收敛。相比之下，
  减少内部协变量偏移这一原始动机似乎并不是一个有效的解释。
* 对于对输入扰动不那么敏感的更稳健模型，可以考虑移除批量规范化 :cite:`wang2022removing`。

## 练习

1. 我们是否应该在批量规范化之前移除全连接层或卷积层中的偏置参数？为什么？
1. 比较使用和不使用批量规范化的 LeNet 的学习率。
    1. 绘制验证准确率上升的曲线。
    1. 在两种情况下，学习率最大能取多大而优化仍不失败？
1. 我们需要在每一层都使用批量规范化吗？请进行实验。
1. 实现一个只移除均值的“精简”版批量规范化，或者实现一个
   只移除方差的版本。它的表现如何？
1. 固定参数 `beta` 和 `gamma`。观察并分析结果。
1. 能否用批量规范化替代暂退法？行为会如何变化？
1. 研究思路：想一想你还可以应用哪些其他的规范化变换：
    1. 你能应用概率积分变换吗？
    1. 你能使用满秩协方差估计吗？为什么你很可能不应该那样做？
    1. 你能使用其他紧凑矩阵变体（块对角、低位移秩、Monarch 等）吗？
    1. 稀疏化压缩会起到正则化作用吗？
    1. 还有其他可以使用的投影吗（例如凸锥、对称群特定变换）？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18005)
:end_tab:
