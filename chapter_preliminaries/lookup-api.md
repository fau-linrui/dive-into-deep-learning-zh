```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 文档
:begin_tab:`mxnet`
我们不可能介绍每一个MXNet函数和类
（而且这些信息可能很快就会过时），
但[API文档](https://mxnet.apache.org/versions/1.8.0/api)
以及额外的[教程](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/)和示例
提供了此类文档。
本节提供了一些关于如何探索MXNet API的指导。
:end_tab:

:begin_tab:`pytorch`
我们不可能介绍每一个PyTorch函数和类
（而且这些信息可能很快就会过时），
但[API文档](https://pytorch.org/docs/stable/index.html)以及额外的[教程](https://pytorch.org/tutorials/beginner/basics/intro.html)和示例
提供了此类文档。
本节提供了一些关于如何探索PyTorch API的指导。
:end_tab:

:begin_tab:`tensorflow`
我们不可能介绍每一个TensorFlow函数和类
（而且这些信息可能很快就会过时），
但[API文档](https://www.tensorflow.org/api_docs)以及额外的[教程](https://www.tensorflow.org/tutorials)和示例
提供了此类文档。
本节提供了一些关于如何探索TensorFlow API的指导。
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## 模块中的函数和类

要想知道一个模块中可以调用哪些函数和类，
我们可以调用`dir`函数。例如，我们可以
(**查询模块中用于生成随机数的所有属性**)：

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

通常，我们可以忽略以`__`开头和结尾的函数（Python中的特殊对象），
或者以单个`_`开头的函数（通常是内部函数）。
根据其余的函数名或属性名，
我们或许可以大胆猜测：这个模块提供了
各种生成随机数的方法，
包括从均匀分布（`uniform`）、
正态分布（`normal`）和多项分布（`multinomial`）中采样。

## 具体的函数和类

要想了解某个给定函数或类的具体用法，
我们可以调用`help`函数。例如，让我们
[**探索张量的`ones`函数的使用说明**]。

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

从文档中可以看到，`ones`函数
创建一个具有指定形状的新张量，
并将所有元素的值都设为1。
只要有可能，你就应该(**运行一个快速测试**)
来确认你的理解：

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

在Jupyter 笔记本中，我们可以用`?`在另一个窗口中显示文档。
例如，`list?`将生成与`help(list)`几乎相同的内容，
并在新的浏览器窗口中显示。
此外，如果我们使用两个问号，例如`list??`，
那么实现该函数的Python代码也会显示出来。

官方文档提供了大量超出本书范围的描述和示例。
我们强调那些能让你快速上手解决实际问题的
重要用例，
而不是追求覆盖的完整性。
我们也鼓励你研读这些库的源代码，
看看生产代码中高质量实现的范例。
这样做不仅能让你成为一名更好的科学家，
还能让你成为一名更好的工程师。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17972)
:end_tab:
