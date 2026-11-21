```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 数据预处理
:label:`sec_pandas`

到目前为止，我们处理的都是
以现成张量形式出现的合成数据。
然而，要在实际场景中应用深度学习，
我们必须从以任意格式存储的杂乱数据中提取数据，
并对它们进行预处理，以满足我们的需要。
幸运的是，*pandas* [库](https://pandas.pydata.org/)
可以承担其中大部分繁重的工作。
本节虽然无法替代一份正规的*pandas* [教程](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)，
但会为你提供一些最常用例程的速成课程。

## 读取数据集

逗号分隔值（CSV）文件普遍用于
存储表格型（类似电子表格）数据。
在这类文件中，每一行对应一条记录，
由若干个（以逗号分隔的）字段组成，例如，
“Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics”。
为了演示如何用`pandas`加载CSV文件，
我们(**在下面创建一个CSV文件**)`../data/house_tiny.csv`。
该文件表示一个房屋数据集，
其中每一行对应一所不同的房屋，
各列分别对应房间数（`NumRooms`）、
屋顶类型（`RoofType`）和价格（`Price`）。

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

现在，让我们导入`pandas`，并用`read_csv`加载数据集。

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 数据准备

在监督学习中，我们训练模型，
使其在给定一组*输入*值的情况下
预测指定的*目标*值。
处理数据集的第一步，
是把对应输入值的列
与对应目标值的列区分开。
我们可以按名称选择列，
也可以通过基于整数位置的索引（`iloc`）来选择列。

你可能已经注意到，`pandas`把所有取值为`NA`的CSV条目
替换成了一个特殊的`NaN`（*非数*，not a number）值。
只要某个条目为空，也会出现这种情况，
例如，“3,,,270000”。
这些值称为*缺失值*（missing values），
它们是数据科学中的“臭虫”，
是你在整个职业生涯中都会遇到的顽固祸害。
根据具体情境，
缺失值既可以通过*插补*（imputation）处理，
也可以通过*删除*（deletion）处理。
插补用缺失值的估计值来替换缺失值，
而删除则直接丢弃
包含缺失值的那些行或那些列。

下面是一些常见的插补启发式方法。
[**对于类别型输入字段，
我们可以把`NaN`当作一个类别。**]
由于`RoofType`列取值为`Slate`和`NaN`，
`pandas`可以把这一列
转换为两列`RoofType_Slate`和`RoofType_nan`。
屋顶类型为`Slate`的行会把`RoofType_Slate`和`RoofType_nan`
的值分别设为1和0。
对于`RoofType`值缺失的行则相反。

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

对于缺失的数值，
一种常见的启发式方法是
[**用对应列的均值替换`NaN`条目**]。

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## 转换为张量格式

现在，[**`inputs`和`targets`中的所有条目都是数值，
我们可以把它们加载到张量中**]（回想一下 :numref:`sec_ndarray`）。

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## 讨论

现在你已经知道如何划分数据列、
插补缺失变量，
以及把`pandas`数据加载到张量中。
在 :numref:`sec_kaggle_house` 中，你将
学到一些更多的数据处理技巧。
虽然这个速成课程保持简单，
但数据处理可能变得非常棘手。
例如，我们的数据集可能并非以单个CSV文件的形式出现，
而是分散在从关系数据库提取出来的多个文件中。
例如，在电子商务应用中，
客户地址可能存放在一张表里，
而购买数据存放在另一张表里。
此外，实践者还要面对类别型和数值型之外的
众多数据类型，例如，
文本字符串、图像、
音频数据和点云。
通常需要高级工具和高效算法，
才能防止数据处理成为
机器学习流水线中最大的瓶颈。
当我们学到计算机视觉和自然语言处理时，
这些问题就会浮现。
最后，我们必须关注数据质量。
真实世界的数据集往往受到
离群值、传感器错误测量和记录错误的困扰，
这些在把数据送入任何模型之前
都必须加以处理。
诸如[seaborn](https://seaborn.pydata.org/)、
[Bokeh](https://docs.bokeh.org/)或[matplotlib](https://matplotlib.org/)
之类的数据可视化工具，
可以帮助你手动检查数据，
并对你可能需要解决的问题类型
形成直觉。


## 练习

1. 尝试加载数据集，例如从[UCI机器学习仓库](https://archive.ics.uci.edu/ml/datasets)加载Abalone数据集，并检查它们的属性。其中有多大比例的数据含有缺失值？多大比例的变量是数值型、类别型或文本型？
1. 尝试按名称而不是按列号来索引和选择数据列。pandas关于[索引](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)的文档进一步介绍了如何做到这一点。
1. 你认为用这种方式能加载多大的数据集？可能存在哪些限制？提示：考虑读取数据的时间、表示、处理和内存占用。在你的笔记本电脑上试一试。如果在服务器上尝试，会发生什么？
1. 你会如何处理类别数量非常多的数据？如果类别标签全都各不相同，又该怎么办？后者是否应该保留？
1. 你能想到哪些pandas的替代方案？[从文件加载NumPy张量](https://numpy.org/doc/stable/reference/generated/numpy.load.html)怎么样？去看看Python图像处理库[Pillow](https://python-pillow.org/)。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17967)
:end_tab:
