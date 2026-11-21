```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 合成回归数据
:label:`sec_synthetic-regression-data`


机器学习的一切都在于从数据中提取信息。
因此你可能会问：从合成数据中究竟能学到什么呢？
虽然我们可能从本质上并不关心
自己人为编造的数据生成模型里所嵌入的模式，
但这类数据集仍然具有教学价值，
能帮助我们评估学习算法的性质，
并确认我们的实现是否按预期工作。
例如，如果我们创建的数据，其正确参数是*先验*已知的，
那么就可以检验我们的模型是否真的能恢复它们。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## 生成数据集

为了简洁起见，本例中我们将在低维空间中进行。
下面的代码片段生成1000个样例，
每个样例的二维特征均从
标准正态分布中抽取。
由此得到的设计矩阵$\mathbf{X}$
属于$\mathbb{R}^{1000 \times 2}$。
我们通过一个*真实值*（ground truth）线性函数来生成每个标签，
再用加性噪声$\boldsymbol{\epsilon}$对这些标签进行破坏，
噪声对每个样例独立同分布地抽取：

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

为了方便起见，我们假设$\boldsymbol{\epsilon}$抽取自
均值为$\mu= 0$、
标准差为$\sigma = 0.01$的正态分布。
请注意，为了遵循面向对象的设计，
我们把代码添加到`__init__`方法中，该方法由`d2l.DataModule`的子类定义（该类在 :numref:`oo-design-data` 中介绍）。
允许设置任何额外的超参数是一种好做法。
我们通过`save_hyperparameters()`来实现这一点。
`batch_size`将在稍后确定。

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

下面，我们把真实参数设为$\mathbf{w} = [2, -3.4]^\top$和$b = 4.2$。
稍后，我们就可以对照这些*真实值*来检验估计出的参数。

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**`features`中的每一行是$\mathbb{R}^2$中的一个向量，`labels`中的每一行是一个标量。**] 让我们看看第一个条目。

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## 读取数据集

训练机器学习模型通常需要对数据集进行多遍遍历，
每次抓取一个小批量的样例。
然后用这些数据来更新模型。
为了说明其工作原理，我们
[**实现`get_dataloader`方法，**]
并把它注册到`SyntheticRegressionData`类中，注册方式为`add_to_class`（该函数在 :numref:`oo-design-utilities` 中介绍）。
该方法(**接受批量大小、特征矩阵
和标签向量，并生成大小为`batch_size`的小批量。**)
因此，每个小批量都由一个特征与标签的元组组成。
请注意，我们需要留意当前处于训练模式还是验证模式：
在前一种模式下，我们希望以随机顺序读取数据；
而在后一种模式下，能够按预先定义的顺序读取数据
可能对调试很重要。

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

为了建立一些直观认识，让我们检查第一个小批量的
数据。每个小批量的特征既提供了它的批量大小，也提供了输入特征的维数。
同样，我们的小批量标签也会具有由`batch_size`给出的匹配形状。

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

这一调用看似平淡无奇，
但`iter(data.train_dataloader())`
却展示了Python面向对象设计的强大之处。
请注意，我们为`SyntheticRegressionData`类添加了一个方法，
而这是*在*创建`data`对象*之后*才完成的。
尽管如此，该对象仍然受益于
*事后*（ex post facto）向该类添加的功能。

在整个迭代过程中，我们会获得不同的小批量，
直到整个数据集都被遍历完（不妨一试）。
虽然上面实现的迭代方式在教学上很好，
但它的低效在某些情况下会给处理实际问题带来麻烦。
例如，它要求我们把所有数据都加载到内存中，
并且要执行大量的随机内存访问。
深度学习框架中实现的内置迭代器
要高效得多，而且它们可以处理
存储在文件中的数据、
通过流接收的数据，
以及即时生成或处理的数据等来源。
接下来，让我们尝试用内置迭代器来实现同样的方法。

## 数据加载器的简洁实现

与其自己编写迭代器，
我们可以[**调用框架中已有的API来加载数据。**]
和之前一样，我们需要一个特征为`X`、标签为`y`的数据集。
除此之外，我们在内置数据加载器中设置`batch_size`，
让它高效地处理样例的打乱工作。

:begin_tab:`jax`
JAX的核心是类似NumPy的API，并具有设备加速和函数式变换，
因此至少当前版本并不包含数据加载方法。
其他库已经提供了很好的数据加载器，
JAX建议改用它们。这里我们将采用TensorFlow的数据加载器，
并稍作修改，使其能与JAX配合工作。
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

新的数据加载器与之前的加载器行为相同，只是它更高效，并且提供了一些额外的功能。

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

例如，框架API提供的数据加载器
支持内置的`__len__`方法，
因此我们可以查询它的长度，
即批次的数量。

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## 小结

数据加载器提供了一种便捷的方式，把
加载和操作数据的过程抽象出来。
这样一来，同一个机器学习*算法*
无需修改就能处理许多不同类型和数据源的数据。
数据加载器的一个优点
在于它们可以组合。
例如，我们可能先加载图像，
然后用一个后处理滤波器
对图像进行裁剪或以其他方式修改。
因此，数据加载器可以用来
描述一整个数据处理流水线。

至于模型本身，二维线性模型
大概是我们能遇到的最简单的模型了。
它让我们能够检验回归模型的准确率，
而不必担心数据量不足
或方程组欠定。
我们将在下一节中充分利用这一点。


## 练习

1. 如果样例数不能被批量大小整除，会发生什么？你会如何使用框架的API，通过指定不同的参数来改变这一行为？
1. 假设我们想生成一个巨大的数据集，其中参数向量`w`的大小和样例数`num_examples`都很大。
    1. 如果我们无法把所有数据都保存在内存中，会发生什么？
    1. 如果数据保存在磁盘上，你会如何打乱数据？你的任务是设计一种*高效*算法，它不需要太多随机读写。提示：[伪随机置换生成器](https://en.wikipedia.org/wiki/Pseudorandom_permutation)允许你设计重新洗牌的方法，而无需显式存储置换表 :cite:`Naor.Reingold.1999`。
1. 实现一个数据生成器，每次调用迭代器时都即时生成新数据。
1. 你会如何设计一个随机数据生成器，使其每次被调用时都生成*相同的*数据？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17975)
:end_tab:
