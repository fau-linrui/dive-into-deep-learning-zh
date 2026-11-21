# 小批量随机梯度下降
:label:`sec_minibatch_sgd`

到目前为止，在基于梯度的学习方法中，我们遇到了两个极端：:numref:`sec_gd` 使用整个数据集来计算梯度并更新参数，每次遍历一趟。相反，:numref:`sec_sgd` 每次处理一个训练样例来取得进展。
两者各有其缺点。
当数据非常相似时，梯度下降并不是特别*数据高效*。
随机梯度下降并不是特别*计算高效*，因为 CPU 和 GPU 无法发挥向量化的全部威力。
这表明两者之间可能存在某种折中方案，
事实上，这正是我们迄今为止在讨论的示例中一直在使用的方案。

## 向量化与缓存

使用小批量的核心原因是计算效率。当考虑并行化到多个 GPU 和多个服务器时，这一点最容易理解。在这种情况下，我们需要向每个 GPU 至少发送一张图像。如果每台服务器有 8 个 GPU、共有 16 台服务器，那么我们就已经得到不小于 128 的小批量大小。

对于单个 GPU 甚至 CPU，情况要微妙一些。这些设备具有多种类型的内存，通常还有多种类型的计算单元，以及它们之间不同的带宽约束。
例如，CPU 有少量寄存器，然后还有 L1、L2，有时甚至还有 L3 缓存（后者在不同处理器核心之间共享）。
这些缓存的容量和延迟依次递增（同时带宽依次递减）。
可以说，处理器能够执行的操作远多于主存接口所能提供的操作。

首先，一个 2GHz、16 核且支持 AVX-512 向量化的 CPU 每秒最多可以处理$2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$字节。GPU 的能力很容易超过这个数字 100 倍。另一方面，一台中端服务器处理器的带宽可能不会比 100 GB/s 高多少，即不到满足处理器所需带宽的十分之一。更糟糕的是，并非所有内存访问都是等价的：内存接口通常是 64 位宽或更宽（例如在 GPU 上可达 384 位），因此读取单个字节要付出宽得多的访问代价。

其次，首次访问存在显著的开销，而顺序访问则相对便宜（这通常称为突发读取）。还有更多需要考虑的因素，例如当我们有多个插槽、小芯片（chiplet）以及其他结构时的缓存。
关于更深入的讨论，见这篇[维基百科文章](https://en.wikipedia.org/wiki/Cache_hierarchy)。

缓解这些约束的方法是使用层级化的 CPU 缓存，这些缓存实际上足够快，能够为处理器提供数据。这就是深度学习中批处理背后的*那个*驱动力。为简单起见，考虑矩阵-矩阵乘法，例如$\mathbf{A} = \mathbf{B}\mathbf{C}$。计算$\mathbf{A}$有若干种选择。例如，我们可以尝试以下方式：

1. 我们可以计算$\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$，即通过点积按元素计算它。
1. 我们可以计算$\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$，即一次计算一列。同样，我们可以一次计算$\mathbf{A}$的一行$\mathbf{A}_{i,:}$。
1. 我们可以简单地计算$\mathbf{A} = \mathbf{B} \mathbf{C}$。
1. 我们可以把$\mathbf{B}$和$\mathbf{C}$分解为较小的块矩阵，一次计算$\mathbf{A}$的一个块。

如果我们采用第一种选择，那么每次想要计算元素$\mathbf{A}_{ij}$时，都需要把一行和一个列向量复制到 CPU 中。更糟糕的是，由于矩阵元素是顺序排列的，因此当我们从内存读取数据时，对于两个向量中的一个，我们不得不访问许多不相邻的位置。第二种选择要有利得多。在第二种选择中，我们能够把列向量$\mathbf{C}_{:,j}$保存在 CPU 缓存中，同时不断遍历$\mathbf{B}$。这将内存带宽需求减半，访问速度也相应更快。当然，选项 3 是最理想的。不幸的是，大多数矩阵可能无法完全放入缓存（毕竟这正是我们讨论的问题）。然而，选项 4 提供了一种实用的替代方案：我们可以把矩阵的块移入缓存，并在本地进行乘法运算。优化的库会为我们处理好这些。让我们看看这些操作在实际中的效率如何。

除了计算效率之外，Python 和深度学习框架本身引入的开销也相当可观。回想一下，每次执行一条命令时，Python 解释器都会向 MXNet 引擎发送一条命令，后者需要将其插入计算图并在调度期间处理它。这种开销可能相当有害。简言之，强烈建议尽可能使用向量化（以及矩阵）。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

由于在本书的其余部分我们将频繁地对运行时间进行基准测试，让我们定义一个计时器。

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

按元素赋值就是分别遍历$\mathbf{B}$和$\mathbf{C}$的所有行和列，把值赋给$\mathbf{A}$。

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

一种更快的策略是按列赋值。

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

最后，最有效的方式是在一个块中执行整个操作。
注意，将任意两个矩阵$\mathbf{B} \in \mathbb{R}^{m \times n}$和$\mathbf{C} \in \mathbb{R}^{n \times p}$相乘，大约需要$2mnp$次浮点运算，
其中标量乘法和加法被计为独立的运算（实践中会融合）。
因此，两个$256 \times 256$矩阵相乘
需要$0.03$十亿次浮点运算。
让我们看看这些操作各自的速度如何。

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 小批量

:label:`sec_minibatches`

过去，我们理所当然地认为应当读取*小批量*数据而不是单个观测来更新参数。现在我们对这种做法给出简要的理由。处理单个观测需要我们执行许多单次矩阵-向量（甚至向量-向量）乘法，这相当昂贵，并且会为底层深度学习框架带来显著开销。这既适用于将网络应用于数据（通常称为推理）时的求值，也适用于计算梯度以更新参数时。也就是说，只要执行$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$，就适用这一点，其中

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

我们可以通过一次对一个小批量的观测执行该操作来提高其*计算*效率。也就是说，我们把单个观测上的梯度$\mathbf{g}_t$替换为小批量上的梯度

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

让我们看看这对$\mathbf{g}_t$的统计性质有什么影响：由于$\mathbf{x}_t$以及小批量$\mathcal{B}_t$的所有元素都是从训练集中随机均匀抽取的，梯度的期望保持不变。另一方面，方差显著减小。由于小批量梯度由$b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$个被平均的独立梯度组成，其标准差减小为原来的$b^{-\frac{1}{2}}$倍。这本身是件好事，因为它意味着更新与完整梯度的方向更可靠地对齐。

天真地看，这似乎表明选择较大的小批量$\mathcal{B}_t$在任何情况下都是可取的。唉，在某个点之后，与计算代价的线性增长相比，标准差的额外减小微乎其微。在实践中，我们选择的小批量要足够大，以提供良好的计算效率，同时仍然能放入 GPU 的内存。为了说明这种节省，让我们看一段代码。在这段代码中，我们执行相同的矩阵-矩阵乘法，但这次将其分成每次 64 列的“小批量”。

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

正如我们可以看到的，小批量上的计算与整个矩阵上的计算效率基本相同。需要提醒一句。在 :numref:`sec_batch_norm` 中，我们使用了一种严重依赖小批量中方差大小的正则化。当我们增大批量时，方差会减小，批量规范化带来的噪声注入收益也随之减小。关于如何重新缩放并计算相应的项，详见例如 :citet:`Ioffe.2017`。

## 读取数据集

让我们看看如何从数据中高效地生成小批量。下面我们使用 NASA 开发的一个数据集来测试机翼的[来自不同飞机的噪声](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)，以比较这些优化算法。为方便起见，我们只使用前$1,500$个样例。数据经过白化预处理，即我们去除均值，并将每个坐标的方差重新缩放到$1$。

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 从零开始实现

回想一下 :numref:`sec_linear_scratch` 中的小批量随机梯度下降实现。下面我们提供一个稍微更通用的实现。为方便起见，它与本章后面介绍的其他优化算法具有相同的调用签名。具体来说，我们添加状态输入`states`，并把超参数放入字典`hyperparams`。此外，我们将在训练函数中对每个小批量样例的损失取平均，因此优化算法中的梯度不需要再除以批量大小。

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

接下来，我们实现一个通用的训练函数，以便于使用本章后面介绍的其他优化算法。它初始化一个线性回归模型，并可用于使用小批量随机梯度下降以及随后介绍的其他算法来训练该模型。

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

让我们看看批梯度下降的优化过程。这可以通过将小批量大小设为 1500（即样例总数）来实现。因此，模型参数每轮只更新一次。进展甚微。事实上，6 步之后进展就停滞了。

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

当批量大小等于 1 时，我们使用随机梯度下降进行优化。为便于实现，我们选取了一个常数（尽管很小）学习率。在随机梯度下降中，每处理一个样例就更新一次模型参数。在我们的例子中，这相当于每轮 1500 次更新。正如我们所看到的，目标函数值的下降在一轮之后变慢。尽管两种过程在一轮内都处理了 1500 个样例，但在我们的实验中随机梯度下降比梯度下降消耗更多时间。这是因为随机梯度下降更频繁地更新参数，而且一次处理单个观测的效率较低。

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

最后，当批量大小等于 100 时，我们使用小批量随机梯度下降进行优化。每轮所需的时间比随机梯度下降所需的时间和批梯度下降所需的时间都短。

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

把批量大小减小到 10，每轮的时间反而增加，因为每个批量的工作负载执行效率更低。

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

现在我们可以比较前四个实验的时间-损失曲线。可以看到，尽管就处理的样例数量而言，随机梯度下降比梯度下降收敛得更快，但达到相同损失时它比梯度下降耗费更多时间，因为逐样例计算梯度的效率较低。小批量随机梯度下降能够在收敛速度和计算效率之间取得权衡。小批量大小为 10 时比随机梯度下降更高效；小批量大小为 100 时甚至在运行时间上优于梯度下降。

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 简洁实现

在 Gluon 中，我们可以使用`Trainer`类来调用优化算法。它被用来实现一个通用的训练函数。我们将在本章中自始至终使用它。

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

使用 Gluon 重复上一个实验，会显示出相同的行为。

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## 小结

* 向量化使代码更高效，这既是因为深度学习框架的开销降低了，也是因为 CPU 和 GPU 上的内存局部性和缓存更好。
* 随机梯度下降带来的统计效率与一次处理大批量数据带来的计算效率之间存在权衡。
* 小批量随机梯度下降兼具两者的优点：计算效率和统计效率。
* 在小批量随机梯度下降中，我们处理通过对训练数据随机置换而得到的批量数据（即每个观测每轮只处理一次，尽管顺序是随机的）。
* 建议在训练过程中衰减学习率。
* 一般来说，以实际时钟时间衡量，小批量随机梯度下降在收敛到较小风险方面比随机梯度下降和梯度下降更快。

## 练习

1. 修改批量大小和学习率，观察目标函数值的下降速率以及每轮消耗的时间。
1. 阅读 MXNet 文档，并使用`Trainer`类的`set_learning_rate`函数，在每轮之后将小批量随机梯度下降的学习率降低到先前值的 1/10。
1. 将小批量随机梯度下降与一个真正从训练集中*有放回采样*的变体进行比较。会发生什么？
1. 一个邪恶的精灵在你不知情的情况下复制了你的数据集（即每个观测出现两次，你的数据集增长到原来大小的两倍，但没有人告诉你）。随机梯度下降、小批量随机梯度下降和梯度下降的行为会如何变化？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1069)
:end_tab:
