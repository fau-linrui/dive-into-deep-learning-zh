```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 从零开始实现 softmax 回归
:label:`sec_softmax_scratch`

由于 softmax 回归如此基础，
我们认为你应该知道
如何自己实现它。
这里，我们只限于定义
模型中 softmax 特有的部分，
并复用线性回归一节中的其他组件，
包括训练循环。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
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
import jax
from jax import numpy as jnp
from functools import partial
```

## softmax 运算

让我们从最重要的部分开始：
从标量到概率的映射。
作为复习，回想一下 :numref:`subsec_lin-alg-reduction`
和 :numref:`subsec_lin-alg-non-reduction` 中讨论的
求和运算符沿张量特定维度的运算。
[**给定一个矩阵`X`，我们可以对所有元素（默认）求和，
也可以只对同一轴上的元素求和。**]
`axis`变量让我们可以计算按行和与按列和：

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

计算 softmax 需要三个步骤：
(i) 对每一项取指数；
(ii) 对每一行求和，计算每个样例的规范化常数；
(iii) 把每一行除以其规范化常数，
确保结果之和为 1：

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

分母（的对数）称为（对数）*配分函数*（partition function）。
它最初在[统计物理](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))中引入，
用于对热力学系综中所有可能的状态求和。
实现很直接：

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

对于任意输入`X`，[**我们把每个元素
变成非负数，
并且每一行之和为 1，**]
这正是概率所要求的。
注意：上面的代码对非常大或非常小的参数*并不*鲁棒。
虽然它足以说明正在发生什么，
但在任何严肃的用途中，你都*不应*原样使用这段代码。
深度学习框架内置了这类保护措施，今后我们将使用内置的 softmax。

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## 模型

现在我们已经具备了实现
[**softmax 回归模型**]所需的一切。
就像线性回归的例子一样，
每个实例将由一个固定长度的向量表示。
由于这里的原始数据由
$28 \times 28$ 像素的图像组成，
[**我们把每幅图像展平，
将它们视为长度为 784 的向量。**]
在后面的章节中，我们将介绍
卷积神经网络，
它能以更令人满意的方式利用空间结构。

在 softmax 回归中，
网络的输出个数
应当等于类别的个数。
(**由于我们的数据集有 10 个类别，
因此网络的输出维数为 10。**)
因此，我们的权重构成一个 $784 \times 10$ 矩阵，
偏置则是一个 $1 \times 10$ 的行向量。
与线性回归一样，
我们用高斯噪声初始化权重`W`。
偏置初始化为零。

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

下面的代码定义了网络
如何把每个输入映射到输出。
请注意，在把数据传入模型之前，
我们用`reshape`把批次中每幅 $28 \times 28$ 像素的图像
展平为一个向量。

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## 交叉熵损失

接下来我们需要实现交叉熵损失函数
（在 :numref:`subsec_softmax-regression-loss-func` 中介绍）。
这可能是整个深度学习中最常见的损失函数。
目前，很容易转化为分类问题的深度学习应用
在数量上远超那些更适合作为回归问题处理的应用。

回想一下，交叉熵取的是
赋予真实标签的预测概率的负对数似然。
为了提高效率，我们避免使用 Python 的 for 循环，而是改用索引。
具体来说，$\mathbf{y}$ 中的独热编码
使我们能够选出 $\hat{\mathbf{y}}$ 中对应的项。

为了看看实际效果，我们[**创建样本数据`y_hat`，
其中包含 2 个样例在 3 个类别上的预测概率，以及它们对应的标签`y`。**]
正确的标签分别是 $0$ 和 $2$（即第一个和第三个类别）。
[**把`y`用作`y_hat`中概率的索引，**]
我们就能高效地选出相应的项。

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
现在，我们可以通过对所选概率的对数取平均，
来(**实现交叉熵损失函数**)。
:end_tab:

:begin_tab:`jax`
现在，我们可以通过对所选概率的对数取平均，
来(**实现交叉熵损失函数**)。

请注意，为了利用`jax.jit`加速 JAX 实现，
并确保`loss`是一个纯函数，
我们在`loss`内部重新定义了`cross_entropy`函数，
以避免使用任何可能使`loss`函数不纯的全局变量或函数。
感兴趣的读者可以参阅关于`jax.jit`和纯函数的 [JAX 文档](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return cross_entropy(y_hat, y), {}
```

## 训练

我们复用 :numref:`sec_linear_scratch` 中定义的`fit`方法，
来[**用 10 轮训练该模型。**]
请注意，轮数（`max_epochs`）、
小批量大小（`batch_size`）
和学习率（`lr`）
都是可调的超参数。
这意味着，虽然这些值
不会在我们的主要训练循环中被学习，
但它们仍然会影响
我们模型的性能，
无论是对训练性能还是泛化性能而言都是如此。
在实践中，你会希望根据数据的*验证*划分来选择这些值，
然后最终在*测试*划分上评估你的最终模型。
正如 :numref:`subsec_generalization-model-selection` 中所讨论的，
我们将把 Fashion-MNIST 的测试数据
视为验证集，从而
在该划分上报告验证损失和验证准确率。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 预测

既然训练已经完成，
我们的模型就可以[**对一些图像进行分类**]了。

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

我们更感兴趣的是那些被我们*错误*标记的图像。我们通过比较它们的实际标签
（文本输出的第一行）
与模型的预测
（文本输出的第二行）来把它们可视化。

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## 小结

到目前为止，我们开始积累了一些
解决线性回归和分类问题的经验。
凭借这些，我们已经达到了可以说
是 20 世纪 60—70 年代统计建模的最先进水平。
在下一节中，我们将向你展示如何利用
深度学习框架来更加高效地实现这个模型。

## 练习

1. 本节中，我们根据 softmax 运算的数学定义直接实现了 softmax 函数。正如 :numref:`sec_softmax` 中所讨论的，这可能导致数值不稳定。
    1. 测试如果某个输入的值为 $100$，`softmax`是否仍能正确工作。
    1. 测试如果所有输入中最大的值都小于 $-100$，`softmax`是否仍能正确工作。
    1. 通过考察相对于参数中最大元素的值来实现一种修正。
1. 实现一个遵循交叉熵损失函数定义 $\sum_i y_i \log \hat{y}_i$ 的`cross_entropy`函数。
    1. 在本节的代码示例中试用它。
    1. 你认为它为什么运行得更慢？
    1. 你应该使用它吗？什么情况下使用它才有意义？
    1. 你需要注意什么？提示：考虑对数的定义域。
1. 返回最可能的标签总是好主意吗？例如，在医学诊断中你会这样做吗？你会如何设法解决这个问题？
1. 假设我们想用 softmax 回归根据某些特征预测下一个单词。词表很大时可能出现哪些问题？
1. 对本节代码的超参数进行实验。特别是：
    1. 绘制验证损失随学习率变化的曲线。
    1. 改变小批量大小时，验证损失和训练损失会变化吗？需要把批量大小调到多大或多小才能看到效果？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17982)
:end_tab:
