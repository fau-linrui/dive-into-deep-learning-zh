# 目标检测与边界框
:label:`sec_bbox`


在之前的章节（例如 :numref:`sec_alexnet`--:numref:`sec_googlenet`）中，
我们介绍了用于图像分类的各种模型。
在图像分类任务中，
我们假设图像中只有一个*主要*物体，
并且只关注如何
识别它的类别。
然而，感兴趣的图像中往往有*多个*物体。
我们不仅想知道它们的类别，还想知道它们在图像中的具体位置。
在计算机视觉中，我们把这类任务称为*目标检测*（object detection）（或者*物体识别*（object recognition））。

目标检测已被
广泛应用于许多领域。
例如，自动驾驶需要通过检测
所拍摄视频图像中车辆、行人、道路和障碍物的位置
来规划行驶路线。
此外，
机器人可以在环境导航的整个过程中
使用这种技术
来检测并定位感兴趣的物体。
再者，
安全系统
可能需要检测异常物体，例如入侵者或炸弹。

在接下来的几节中，我们将介绍
几种用于目标检测的深度学习方法。
我们将从介绍
物体的*位置*（或*方位*）开始。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

我们将加载本节要使用的示例图像。可以看到图像左侧有一只狗，右侧有一只猫。
它们是这幅图像中的两个主要物体。

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 边界框


在目标检测中，
我们通常使用一个*边界框*（bounding box）来描述物体的空间位置。
边界框是矩形的，由矩形左上角的 $x$ 和 $y$ 坐标以及右下角的相应坐标确定。
另一种常用的边界框表示是边界框中心的 $(x, y)$ 轴坐标，以及框的宽度和高度。

[**这里我们定义函数，用于在**]这两种(**表示之间进行转换**)：
`box_corner_to_center` 从两角表示转换为中心-宽度-高度表示，
`box_center_to_corner` 则相反。
输入参数 `boxes` 应是形状为（$n$, 4）的二维张量，其中 $n$ 是边界框的数量。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

我们将根据坐标信息[**定义图像中狗和猫的边界框**]。
图像中坐标的原点是图像的左上角，向右和向下分别是
$x$ 轴和 $y$ 轴的正方向。

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

我们可以通过两次转换来验证这两个
边界框转换函数的正确性。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

让我们[**在图像中绘制边界框**]，检查它们是否准确。
在绘制之前，我们将定义一个辅助函数 `bbox_to_rect`。它以 `matplotlib` 包的边界框格式表示边界框。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

在图像上添加边界框之后，
我们可以看到两个物体的主要轮廓基本上都位于这两个框内。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 小结

* 目标检测不仅识别图像中所有感兴趣的物体，还识别它们的位置。位置通常用矩形边界框表示。
* 我们可以在两种常用的边界框表示之间进行转换。

## 练习

1. 另找一张图像，尝试标注一个包含物体的边界框。比较标注边界框和类别：哪一个通常花费的时间更长？
1. 为什么 `box_corner_to_center` 和 `box_center_to_corner` 的输入参数 `boxes` 的最内层维度总是 4？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1527)
:end_tab:
