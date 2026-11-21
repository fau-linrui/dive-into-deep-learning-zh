```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 图像分类数据集
:label:`sec_fashion_mnist`

(~~MNIST 数据集是图像分类中广泛使用的数据集之一，但作为基准数据集它太简单了。我们将使用与之类似但更复杂的 Fashion-MNIST 数据集 ~~)

图像分类中一个广泛使用的数据集是手写数字的 [MNIST 数据集](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`。在它于 20 世纪 90 年代发布之时，它对大多数机器学习算法构成了严峻的挑战：该数据集由 60,000 幅分辨率为 $28 \times 28$ 像素的图像组成（外加 10,000 幅图像的测试数据集）。为了有个直观的比较，1995 年，一台拥有高达 64MB 内存和惊人 5 MFLOPs 算力的 Sun SPARCStation 5，在 AT&T 贝尔实验室被认为是机器学习领域最先进的设备。在数字识别上取得高准确率，是 20 世纪 90 年代美国邮政署（USPS）实现信件分拣自动化的关键组成部分。LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` 等深度网络、具有不变性的支持向量机 :cite:`Scholkopf.Burges.Vapnik.1996` 以及切距离分类器 :cite:`Simard.LeCun.Denker.ea.1998`，都能达到低于 1% 的错误率。

十多年来，MNIST 一直是比较机器学习算法的*那个*基准。
虽然它作为基准数据集有过一段辉煌的历史，
但以今天的标准来看，即使简单模型也能达到 95% 以上的分类准确率，
这使得它不适合用来区分强模型和弱模型。
更有甚者，该数据集允许达到*非常*高的准确率，这在许多分类问题中并不常见。
这使算法的发展偏向于某些能利用干净数据集的特定算法族，例如有效集方法和边界搜索有效集算法。
如今，MNIST 更像是一个健全性检查，而不是一个基准。ImageNet :cite:`Deng.Dong.Socher.ea.2009` 提出了一个相关得多的挑战。
不幸的是，对于本书中的许多示例和图示来说，ImageNet 太大了，
因为训练它需要的时间太长，无法让示例保持交互性。
作为替代，在接下来的几节中，我们将把讨论集中在一个性质相似但小得多的数据集上，
即 2017 年发布的 Fashion-MNIST 数据集 :cite:`Xiao.Rasul.Vollgraf.2017`。
它包含 10 个服装类别的图像，分辨率为 $28 \times 28$ 像素。

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## 加载数据集

由于 Fashion-MNIST 数据集非常有用，所有主流框架都提供了它的预处理版本。我们可以[**使用框架内置的工具将其下载并读入内存。**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST 由来自 10 个类别的图像组成，每个类别
在训练数据集中有 6000 幅图像，在测试数据集中有 1000 幅图像。
*测试数据集*用于评估模型性能（它绝不能用于训练）。
因此，训练集和测试集
分别包含 60,000 幅和 10,000 幅图像。

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

上面的图像是灰度的，并被放大到 $32 \times 32$ 像素的分辨率。这类似于原始的 MNIST 数据集，后者由（二值）黑白图像组成。不过请注意，大多数现代图像数据有三个通道（红、绿、蓝），而高光谱图像可以有超过 100 个通道（HyMap 传感器有 126 个通道）。
按照惯例，我们把一幅图像存储为一个 $c \times h \times w$ 的张量，其中 $c$ 是颜色通道数，$h$ 是高度，$w$ 是宽度。

```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~用于可视化数据集的两个工具函数~~]

Fashion-MNIST 的类别有人们容易理解的名称。
下面的便捷方法在数值标签与其名称之间进行转换。

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## 读取小批量

为了在从训练集和测试集读取数据时让生活更轻松，
我们使用内置的数据迭代器，而不是从头创建一个。
回想一下，在每次迭代时，数据迭代器都会
[**读取一个大小为`batch_size`的小批量数据。**]
我们还对训练数据迭代器中的样例进行随机打乱。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

为了看看它是如何工作的，我们调用`train_dataloader`方法加载一个小批量的图像。它包含 64 幅图像。

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

让我们看看读取这些图像所花费的时间。尽管它是内置的加载器，但速度并没有快得惊人。尽管如此，这已经足够了，因为用深度网络处理图像所需的时间要长得多。因此，它足以保证训练网络不会受 I/O 限制。

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## 可视化

我们将经常使用 Fashion-MNIST 数据集。一个方便的函数`show_images`可用于可视化图像及关联的标签。
跳过实现细节，我们下面只展示接口：对于这类工具函数，我们只需要知道如何调用`d2l.show_images`，而不需要知道它的工作原理。

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

让我们好好利用它。一般来说，可视化并检查你用来训练的数据是个好主意。
人类非常擅长发现异常之处，因此，可视化可以作为实验设计中错误和失误的额外保障。下面是训练数据集中前几个样例的[**图像及其对应的标签**]（文本形式）。

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

现在，我们已准备好在后续各节中使用 Fashion-MNIST 数据集了。

## 小结

我们现在有了一个稍微更真实的分类数据集。Fashion-MNIST 是一个服装分类数据集，由代表 10 个类别的图像组成。我们将在后续各节和章节中使用该数据集来评估各种网络设计，从简单的线性模型到先进的残差网络。正如我们处理图像时常做的那样，我们把它们读取为形状为（批量大小、通道数、高度、宽度）的张量。目前我们只有一个通道，因为图像是灰度的（上面的可视化使用了伪彩色调色板以提高可见性）。

最后，数据迭代器是高效性能的关键组成部分。例如，我们可以使用 GPU 来高效地进行图像解压、视频转码或其他预处理。只要有可能，你就应该依赖那些利用高性能计算的良好实现的数据迭代器，以免拖慢训练循环。


## 练习

1. 减小`batch_size`（例如减到 1）会影响读取性能吗？
1. 数据迭代器的性能很重要。你认为当前的实现足够快吗？探索各种选项来改进它。使用系统性能分析器找出瓶颈所在。
1. 查看框架的在线 API 文档。还有哪些其他数据集可用？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17980)
:end_tab:
