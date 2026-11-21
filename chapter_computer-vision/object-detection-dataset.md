# 目标检测数据集
:label:`sec_object-detection-dataset`

目标检测领域没有像 MNIST 和 Fashion-MNIST 那样的小型数据集。
为了快速演示目标检测模型，
[**我们收集并标注了一个小型数据集**]。
首先，我们拍摄了办公室里的免费香蕉，
并生成了
1000 幅不同旋转角度和尺寸的香蕉图像。
然后，我们把每幅香蕉图像
放到某些背景图像上的随机位置。
最后，我们为图像上的这些香蕉标注了边界框。


## [**下载数据集**]

香蕉检测数据集及其所有图像和
csv 标签文件都可以直接从互联网下载。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 读取数据集

我们将在下面的 `read_data_bananas` 函数中
[**读取香蕉检测数据集**]。
该数据集包含一个 csv 文件，用于记录
目标类别标签
以及左上角和右下角的真实边界框坐标。

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

通过 `read_data_bananas` 函数读取图像和标签之后，
下面的 `BananasDataset` 类
将允许我们[**创建一个定制的 `Dataset` 实例**]，
用于加载香蕉检测数据集。

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最后，我们定义
`load_data_bananas` 函数，[**为训练集和测试集
分别返回两个数据迭代器实例。**]
对于测试数据集，
不需要按随机顺序读取。

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

让我们[**读取一个小批量，并打印该小批量中
图像和标签的形状**]。
图像小批量的形状
（批量大小，通道数，高度，宽度）
看起来似曾相识：
这和我们之前图像分类任务中的情况一样。
标签小批量的形状为
（批量大小，$m$，5），
其中 $m$ 是数据集中任意图像
所拥有的边界框最大可能数量。

尽管按小批量计算更高效，
但它要求所有图像样例
都包含相同数量的边界框，
才能通过连结组成一个小批量。
一般来说，
图像可能具有数量各不相同的边界框；
因此，
边界框数量少于 $m$ 的图像
将被非法边界框填充，
直到达到 $m$。
这样，
每个边界框的标签就表示为一个长度为 5 的数组。
数组的第一个元素是边界框中目标的类别，
其中 -1 表示用于填充的非法边界框。
数组中其余四个元素是
边界框左上角和右下角的
（$x$，$y$）坐标值
（范围在 0 和 1 之间）。
对于香蕉数据集，
由于每幅图像上只有一个边界框，
因此 $m=1$。

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**演示**]

让我们演示十幅带有标注的真实边界框的图像。
我们可以看到，这些图像中香蕉的
旋转角度、尺寸和位置各不相同。
当然，这只是一个简单的人造数据集。
在实践中，现实世界的数据集通常要复杂得多。

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 小结

* 我们收集的香蕉检测数据集可用于演示目标检测模型。
* 目标检测的数据加载与图像分类类似。然而，在目标检测中，标签还包含真实边界框的信息，而图像分类中没有这一信息。


## 练习

1. 演示香蕉检测数据集中其他带有真实边界框的图像。它们在边界框和目标方面有什么不同？
1. 假设我们想对目标检测应用随机裁剪等数据增广。它与图像分类中的数据增广有什么不同？提示：如果裁剪后的图像只包含目标的一小部分，会怎样？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1608)
:end_tab:
