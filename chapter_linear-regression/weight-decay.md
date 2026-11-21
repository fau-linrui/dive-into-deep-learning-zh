```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 权重衰减
:label:`sec_weight_decay`

既然我们已经刻画了过拟合问题，
现在就可以介绍我们的第一个*正则化*（regularization）技术。
回想一下，我们总是可以通过收集更多训练数据来缓解过拟合。
然而，这样做可能代价高昂、耗时，
或者完全不在我们的掌控之中，
从而在短期内无法实现。
目前，我们可以假定自己已经拥有了
资源条件允许的尽可能多的高质量数据，
并在把数据集看作给定条件的前提下，
把注意力集中在手头可用的工具上。

回想一下，在多项式回归的例子
（:numref:`subsec_polynomial-curve-fitting`）中，
我们可以通过调整所拟合多项式的阶数
来限制模型的容量。
的确，限制特征的数量
是缓解过拟合的一种流行技术。
然而，仅仅把特征抛到一边，
可能是一种过于生硬的手段。
还是以多项式回归为例，
考虑高维输入时可能发生什么情况。
多项式向多变量数据的自然推广
称为*单项式*（monomials），
它们就是变量各次幂的乘积。
单项式的次数就是各幂次之和。
例如，$x_1^2 x_2$ 和 $x_3 x_5^2$
都是 3 次单项式。

请注意，次数为 $d$ 的项数
会随着 $d$ 增大而迅速膨胀。
给定 $k$ 个变量，次数为 $d$ 的
单项式个数为 ${k - 1 + d} \choose {k - 1}$。
即使次数只有很小的变化，比如从 $2$ 变为 $3$，
也会显著增加模型的复杂度。
因此，我们往往需要一种更精细的工具
来调节函数的复杂度。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## 范数与权重衰减

(**与直接操纵参数数量不同，*权重衰减*（weight decay）
通过限制参数可以取的值来发挥作用。**)
在深度学习圈子之外，当用小批量随机梯度下降进行优化时，
它更常被称为 $\ell_2$ 正则化；
权重衰减可能是正则化参数化机器学习模型时
使用最广泛的技术。
这项技术的动机来自一个基本直觉：
在所有函数 $f$ 中，
函数 $f = 0$
（对所有输入都赋值为 $0$）
在某种意义上是*最简单的*，
并且我们可以用函数参数到零的距离
来衡量函数的复杂度。
但我们应该如何精确地度量
函数与零之间的距离呢？
对此并没有唯一的正确答案。
事实上，数学的整个分支，
包括泛函分析的一部分以及巴拿赫空间理论，
都在致力于处理这类问题。

一种简单的解释可能是，
用线性函数 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
的权重向量的某种范数，例如 $\| \mathbf{w} \|^2$，
来衡量该函数的复杂度。
回想一下，我们在 :numref:`subsec_lin-algebra-norms` 中
介绍了 $\ell_2$ 范数和 $\ell_1$ 范数，
它们是更一般的 $\ell_p$ 范数的特例。
确保权重向量较小的最常见方法，
是将其范数作为惩罚项
加入最小化损失的问题中。
于是，我们用新的目标
——*最小化预测损失与惩罚项之和*——
取代原来的目标
——*最小化训练标签上的预测损失*。
现在，如果权重向量增长得太大，
我们的学习算法就可能把重点放在
最小化权重范数 $\| \mathbf{w} \|^2$ 上，
而不是最小化训练误差。
这正是我们想要的。
为了用代码说明，
我们重新使用 :numref:`sec_linear_regression` 中
关于线性回归的先前例子。
在那里，我们的损失由下式给出：

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

回想一下，$\mathbf{x}^{(i)}$ 是特征，
$y^{(i)}$ 是任意数据样例 $i$ 的标签，
而 $(\mathbf{w}, b)$
分别是权重和偏置参数。
为了惩罚权重向量的大小，
我们必须设法把 $\| \mathbf{w} \|^2$
加入损失函数，
但模型应当如何在标准损失
与这个新的加性惩罚之间权衡呢？
在实践中，我们通过*正则化常数*（regularization constant）$\lambda$
来刻画这种权衡，
它是一个非负超参数，
我们用验证数据来拟合它：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$

当 $\lambda = 0$ 时，我们恢复了原来的损失函数。
当 $\lambda > 0$ 时，我们限制了 $\| \mathbf{w} \|$ 的大小。
按照惯例除以 $2$：
当我们对二次函数求导时，
$2$ 和 $1/2$ 会相互抵消，
从而保证更新表达式看起来简洁美观。
敏锐的读者可能会问，
为什么我们使用平方范数，
而不是标准范数（即欧几里得距离）。
我们这样做是为了计算上的方便。
通过对 $\ell_2$ 范数求平方，
我们去掉平方根，
只剩下权重向量各分量平方之和。
这使得惩罚项的导数很容易计算：
导数之和等于和的导数。

此外，你可能还会问，
我们为什么一开始就使用 $\ell_2$ 范数，
而不是比如 $\ell_1$ 范数。
事实上，其他选择也是有效且在统计学中流行的。
$\ell_2$ 正则化的线性模型构成了
经典的*岭回归*（ridge regression）算法，
而 $\ell_1$ 正则化的线性回归
则是统计学中同样基础的方法，
俗称*lasso 回归*（lasso regression）。
使用 $\ell_2$ 范数的一个原因是，
它对权重向量的大分量
施加了格外大的惩罚。
这使得我们的学习算法
偏向于在更多特征上均匀分配权重的模型。
在实践中，这可能使这些模型
对单个变量中的测量误差更加鲁棒。
相比之下，$\ell_1$ 惩罚会导致模型
把权重集中在少量特征上，
而把其他权重清零。
这为我们提供了一种有效的*特征选择*（feature selection）方法，
而特征选择可能出于其他原因而需要。
例如，如果我们的模型只依赖少数几个特征，
那么我们可能不需要为其他（被丢弃的）特征
收集、存储或传输数据。

采用 :eqref:`eq_linreg_batch_update` 中的相同记号，
$\ell_2$ 正则化回归的小批量随机梯度下降
更新如下：

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

和之前一样，我们根据
估计值与观测值之间的差异大小
来更新 $\mathbf{w}$。
然而，我们还会把 $\mathbf{w}$ 的大小向零收缩。
这就是该方法有时被称为“权重衰减”的原因：
仅考虑惩罚项时，
我们的优化算法会在训练的每一步*衰减*权重。
与特征选择相比，
权重衰减为我们提供了一种连续调节函数复杂度的机制。
$\lambda$ 取值越小，
对应约束越少的 $\mathbf{w}$；
而 $\lambda$ 取值越大，
对 $\mathbf{w}$ 的约束就越强。
是否加入相应的偏置惩罚 $b^2$
因实现而异，
也可能随神经网络的不同层而异。
我们通常不对偏置项做正则化。
此外，虽然对其他优化算法而言
$\ell_2$ 正则化可能不等价于权重衰减，
但通过收缩权重大小
来实现正则化的思想依然成立。

## 高维线性回归

我们可以通过一个简单的合成示例
来说明权重衰减的好处。

首先，我们[**像之前一样生成一些数据**]：

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

在这个合成数据集中，我们的标签
由输入的某个底层线性函数给出，
并受到均值为零、标准差为 0.01 的高斯噪声干扰。
为了便于说明，
我们可以通过把问题的维数提高到 $d = 200$，
并使用仅含 20 个样例的小训练集，
让过拟合的影响变得明显。

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## 从零开始实现

现在，让我们尝试从零开始实现权重衰减。
由于我们的优化器是小批量随机梯度下降，
我们只需把平方 $\ell_2$ 惩罚
加入原来的损失函数。

### (**定义 $\ell_2$ 范数惩罚**)

实现这一惩罚最方便的方式，
也许是把所有项就地平方后求和。

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### 定义模型

在最终模型中，
线性回归和平方损失自 :numref:`sec_linear_scratch` 以来没有变化，
因此我们将只定义`d2l.LinearRegressionScratch`的一个子类。
这里唯一的变化是我们的损失现在包含了惩罚项。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

下面的代码在包含 20 个样例的训练集上拟合我们的模型，
并在包含 100 个样例的验证集上评估它。

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**不使用正则化的训练**]

现在我们用`lambd = 0`运行这段代码，
即禁用权重衰减。
请注意，我们出现了严重的过拟合：
训练误差下降了，验证误差却没有下降——
这是过拟合的典型教科书案例。

```{.python .input}
%%tab all
train_scratch(0)
```

### [**使用权重衰减**]

下面，我们用较大的权重衰减来运行。
请注意，训练误差增大了，
但验证误差减小了。
这正是我们期望正则化产生的效果。

```{.python .input}
%%tab all
train_scratch(3)
```

## [**简洁实现**]

由于权重衰减在神经网络优化中无处不在，
深度学习框架让它变得特别方便：
框架把权重衰减集成到优化算法本身之中，
以便与任何损失函数轻松组合使用。
此外，这种集成还带来计算上的好处，
允许通过实现技巧把权重衰减加入算法，
而不产生任何额外的计算开销。
由于更新中的权重衰减部分
只依赖于每个参数的当前值，
优化器无论如何都必须访问每个参数一次。

:begin_tab:`mxnet`
下面，我们在实例化`Trainer`时
直接通过`wd`指定
权重衰减超参数。
默认情况下，Gluon 会同时
衰减权重和偏置。
请注意，在更新模型参数时，
超参数`wd`
会乘以`wd_mult`。
因此，如果我们将`wd_mult`设为零，
偏置参数 $b$ 就不会衰减。
:end_tab:

:begin_tab:`pytorch`
下面，我们在实例化优化器时
直接通过`weight_decay`指定
权重衰减超参数。
默认情况下，PyTorch 会同时
衰减权重和偏置，但
我们可以配置优化器，按照不同的策略
处理不同的参数。
这里，我们只为权重设置`weight_decay`（即`net.weight`参数），因此
偏置（即`net.bias`参数）不会衰减。
:end_tab:

:begin_tab:`tensorflow`
下面，我们用权重衰减超参数`wd`
创建一个 $\ell_2$ 正则化器，
并通过`kernel_regularizer`参数
把它应用于该层的权重。
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**绘制的图看起来与我们从零实现权重衰减时相似**]。
然而，这个版本运行得更快，
也更容易实现，
随着你处理更大的问题、
这项工作变得更加常规，
这些好处会变得更加明显。

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

到目前为止，我们已经接触了
关于什么构成简单线性函数的其中一种观念。
然而，即使对于简单的非线性函数，
情况也可能复杂得多。
为了看清这一点，
[再生核希尔伯特空间（RKHS）](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)这一概念
允许人们把为线性函数引入的工具
应用于非线性语境。
不幸的是，基于 RKHS 的算法
往往难以扩展到大型高维数据。
在本书中，我们常常会采用
一种常见的启发式做法，
即对深度网络的所有层都施加权重衰减。

## 小结

正则化是处理过拟合的一种常用方法。经典的正则化技术在损失函数中（在训练时）加入惩罚项，以降低所学模型的复杂度。
一种保持模型简单的特定选择是使用 $\ell_2$ 惩罚。这会在小批量随机梯度下降算法的更新步骤中导致权重衰减。
在实践中，深度学习框架的优化器提供了权重衰减功能。
在同一个训练循环中，不同的参数集合可以有不同的更新行为。

## 练习

1. 在本节的估计问题中试验 $\lambda$ 的取值。绘制训练准确率和验证准确率随 $\lambda$ 变化的曲线。你观察到了什么？
1. 使用验证集找出 $\lambda$ 的最优值。它真的是最优值吗？这有影响吗？
1. 如果我们不使用 $\|\mathbf{w}\|^2$，而是选择 $\sum_i |w_i|$ 作为惩罚（$\ell_1$ 正则化），更新方程会是什么样子？
1. 我们知道 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$。你能找到矩阵的类似等式吗（参见 :numref:`subsec_lin-algebra-norms` 中的 Frobenius 范数）？
1. 回顾训练误差与泛化误差之间的关系。除了权重衰减、增加训练和使用复杂度合适的模型之外，还有哪些方法可能有助于我们应对过拟合？
1. 在贝叶斯统计中，我们通过 $P(w \mid x) \propto P(x \mid w) P(w)$ 用先验与似然的乘积得到后验。你如何把 $P(w)$ 与正则化对应起来？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17979)
:end_tab:
