# 多尺度目标检测
:label:`sec_multiscale-object-detection`


在 :numref:`sec_anchor` 中，
我们为输入图像的每个像素都生成了多个以该像素为中心的锚框。
从本质上说，这些锚框
代表了图像不同区域的样本。
然而，
如果为*每个*像素都生成锚框，
需要计算的锚框数量就可能太多了。
考虑一幅 $561 \times 728$ 的输入图像。
如果以每个像素为中心生成
五个形状各异的锚框，
那么整幅图像上就有超过两百万个锚框（$561 \times 728 \times 5$）需要标注和预测。

## 多尺度锚框
:label:`subsec_multiscale-anchor-boxes`

读者或许已经意识到，
减少图像上的锚框数量并不困难。
例如，我们只需
从输入图像中均匀采样一小部分像素，
并以这些像素为中心生成锚框。
此外，
在不同的尺度下
我们可以生成不同数量、
大小不同的锚框。
直观地说，
相比大型目标，
小型目标在图像中出现的可能性更大。
举个例子，
在 $2 \times 2$ 的图像上，
$1 \times 1$、$1 \times 2$ 和 $2 \times 2$ 的目标
分别有 4、2 和 1 种可能出现的方式。
因此，当使用较小的锚框检测较小的目标时，我们可以采样更多的区域；
而对于较大的目标，我们可以采样更少的区域。

为了演示如何生成多尺度的锚框，
让我们读取一幅图像。
它的高度和宽度分别为 561 像素和 728 像素。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

回想一下，在 :numref:`sec_conv_layer` 中，
我们把卷积层输出的二维数组
称为特征图。
通过定义特征图的形状，
我们就可以确定在任意图像上均匀采样的锚框的中心。


`display_anchors` 函数定义如下。
[**我们在特征图（`fmap`）上生成以每个单元（像素）为锚框中心的锚框（`anchors`）。**]
由于锚框（`anchors`）中 $(x, y)$ 轴的坐标值
已经除以特征图（`fmap`）的宽度和高度，
因此这些值介于 0 和 1 之间，
表示锚框在特征图中的相对位置。

由于锚框（`anchors`）的中心
分布在特征图（`fmap`）的所有单元上，
因此就相对空间位置而言，
这些中心在任意输入图像上必须是*均匀*分布的。
更具体地说，
给定特征图的宽度和高度 `fmap_w` 和 `fmap_h`，
下面的函数将在任意输入图像上
*均匀*采样 `fmap_h` 行和 `fmap_w` 列中的像素。
以这些均匀采样的像素为中心，
将生成尺度为 `s`（假设列表 `s` 的长度为 1）、
宽高比不同（`ratios`）的锚框。

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

首先，让我们[**考虑小目标的检测**]。
为了在显示时更容易区分，这里不同中心的锚框互不重叠：
锚框的尺度设为 0.15，
特征图的高度和宽度均设为 4。我们可以看到，
图像上 4 行 4 列锚框的中心是均匀分布的。

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

接下来，我们[**将特征图的高度和宽度减小一半，并使用较大的锚框来检测较大的目标**]。当尺度设为 0.4 时，
一些锚框将彼此重叠。

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

最后，我们[**将特征图的高度和宽度进一步减小一半，并将锚框的尺度增大到 0.8**]。此时锚框的中心就是图像的中心。

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 多尺度检测


既然我们已经生成了多尺度锚框，
接下来就可以利用它们在不同尺度下检测各种大小的目标。
在下面，
我们介绍一种基于 CNN 的多尺度目标检测方法，
并将在 :numref:`sec_ssd` 中实现它。

在某个尺度下，
假设我们有 $c$ 张形状为 $h \times w$ 的特征图。
使用 :numref:`subsec_multiscale-anchor-boxes` 中的方法，
我们生成 $hw$ 组锚框，
其中每组有 $a$ 个中心相同的锚框。
例如，
在 :numref:`subsec_multiscale-anchor-boxes` 的实验中处于第一个尺度时，
给定十张（通道数）$4 \times 4$ 的特征图，
我们生成了 16 组锚框，
其中每组包含 3 个中心相同的锚框。
接下来，根据真实边界框
为每个锚框标注类别和偏移量。在当前尺度下，目标检测模型需要预测输入图像上 $hw$ 组锚框的类别和偏移量，其中不同组具有不同的中心。


假设这里的 $c$ 张特征图
是 CNN 基于输入图像进行前向传播得到的中间输出。
由于每张特征图上有 $hw$ 个不同的空间位置，
因此同一空间位置可以看作
具有 $c$ 个单元。
根据 :numref:`sec_conv_layer` 中感受野的定义，
特征图上同一空间位置处的这 $c$ 个单元
在输入图像上具有相同的感受野：
它们表示同一感受野内的
输入图像信息。
因此，我们可以把特征图上同一空间位置处的
$c$ 个单元
变换为利用该空间位置
生成的 $a$ 个锚框的
类别和偏移量。
从本质上说，
我们利用输入图像在某个感受野内的信息，
来预测输入图像上
与该感受野接近的锚框的
类别和偏移量。


当不同层的特征图
在输入图像上具有不同大小的感受野时，它们就可以用来检测不同大小的目标。
例如，我们可以设计这样一个神经网络：
越靠近输出层的特征图单元
具有越大的感受野，
因此它们可以从输入图像中检测较大的目标。

简而言之，我们可以利用
深度神经网络对图像多个层次的
逐层表示
进行多尺度目标检测。
我们将在 :numref:`sec_ssd` 中
通过一个具体例子展示其工作方式。




## 小结

* 在多个尺度下，我们可以生成不同尺寸的锚框来检测不同尺寸的目标。
* 通过定义特征图的形状，我们可以确定在任意图像上均匀采样的锚框中心。
* 我们利用输入图像在某个感受野内的信息，来预测输入图像上与该感受野接近的锚框的类别和偏移量。
* 通过深度学习，我们可以利用其对图像多个层次的逐层表示来进行多尺度目标检测。


## 练习

1. 根据我们在 :numref:`sec_alexnet` 中的讨论，深度神经网络会为图像学习越来越抽象的层次化特征。在多尺度目标检测中，不同尺度的特征图是否对应不同层次的抽象？为什么？
1. 在 :numref:`subsec_multiscale-anchor-boxes` 实验的第一个尺度（`fmap_w=4, fmap_h=4`）下，生成可能重叠的均匀分布锚框。
1. 给定一个形状为 $1 \times c \times h \times w$ 的特征图变量，其中 $c$、$h$ 和 $w$ 分别是特征图的通道数、高度和宽度。如何将该变量变换为锚框的类别和偏移量？输出的形状是什么？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1607)
:end_tab:
