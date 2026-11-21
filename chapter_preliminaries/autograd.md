```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 自动微分
:label:`sec_autograd`

回想一下 :numref:`sec_calculus` 中的内容：
计算导数是训练深度网络时
将要使用的所有优化算法中的关键步骤。
虽然这些计算本身很直接，
但手工推导可能既繁琐又容易出错，
而且随着我们的模型变得越来越复杂，
这些问题只会不断加剧。

幸运的是，所有现代深度学习框架
都通过提供*自动微分*
（automatic differentiation，
常简写为*autograd*）
替我们承担了这项工作。
当我们让数据依次通过每一个函数时，
框架会构建一张*计算图*（computational graph），
记录每个值如何依赖于其他值。
为了计算导数，
自动微分沿这张图
反向遍历并应用链式法则。
以这种方式应用链式法则的
计算算法称为*反向传播*（backpropagation）。

虽然autograd库在过去十年里
已成为热门的关注点，
但它们有着悠久的历史。
事实上，关于autograd的最早文献
可以追溯到半个多世纪以前 :cite:`Wengert.1964`。
现代反向传播背后的核心思想
源自1980年的一篇博士论文 :cite:`Speelpenning.1980`，
并在20世纪80年代后期得到进一步发展 :cite:`Griewank.1989`。
虽然反向传播已成为计算梯度
的默认方法，但它并不是唯一的选择。
例如，Julia编程语言采用
前向传播 :cite:`Revels.Lubin.Papamarkou.2016`。
在探索各种方法之前，
让我们先掌握autograd包。

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
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
from jax import numpy as jnp
```

## 一个简单的函数

假设我们感兴趣的是
(**对函数
$y = 2\mathbf{x}^{\top}\mathbf{x}$
关于列向量$\mathbf{x}$求导。**)
首先，我们给`x`赋一个初始值。

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**在计算$y$关于$\mathbf{x}$
的梯度之前，
我们需要一个地方来存储它。**]
一般来说，我们避免每次求导
都分配新内存，
因为深度学习需要对同一组参数
连续计算导数很多很多次，
否则我们可能会面临内存耗尽的风险。
请注意，标量值函数
关于向量$\mathbf{x}$的梯度
是向量值的，
其形状与$\mathbf{x}$相同。
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**现在我们计算关于`x`的函数，
并把结果赋给`y`。**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**现在我们可以求`y`
关于`x`的梯度**]，
具体做法是调用它的`backward`方法。
接下来，我们可以通过`x`的`grad`属性
访问该梯度。
:end_tab:

:begin_tab:`pytorch`
[**现在我们可以求`y`
关于`x`的梯度**]，
具体做法是调用它的`backward`方法。
接下来，我们可以通过`x`的`grad`属性
访问该梯度。
:end_tab:

:begin_tab:`tensorflow`
[**现在我们可以通过调用
`gradient`方法
计算`y`关于`x`的梯度**]。
:end_tab:

:begin_tab:`jax`
[**现在我们可以通过
`grad`变换
来求`y`关于`x`的梯度**]。
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**我们已经知道，函数$y = 2\mathbf{x}^{\top}\mathbf{x}$
关于$\mathbf{x}$的梯度应该是$4\mathbf{x}$。**)
现在我们可以验证自动梯度计算的结果
与预期结果是否完全一致。

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**现在让我们计算
`x`的另一个函数
并求它的梯度。**]
请注意，每当我们记录一个新的梯度时，
MXNet都会重置梯度缓冲区。
:end_tab:

:begin_tab:`pytorch`
[**现在让我们计算
`x`的另一个函数
并求它的梯度。**]
请注意，当我们记录新的梯度时，
PyTorch不会自动
重置梯度缓冲区。
相反，新梯度
会加到已经存储的梯度上。
当我们想优化多个目标函数之和时，
这种行为很方便。
要重置梯度缓冲区，
可以像下面这样调用`x.grad.zero_()`：
:end_tab:

:begin_tab:`tensorflow`
[**现在让我们计算
`x`的另一个函数
并求它的梯度。**]
请注意，每当我们记录一个新的梯度时，
TensorFlow都会重置梯度缓冲区。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## 非标量变量的反向传播

当`y`是向量时，
`y`关于向量`x`的导数
最自然的表示
是一个称为*雅可比矩阵*（Jacobian）的矩阵，
它包含`y`的每个分量
关于`x`的每个分量的偏导数。
同样，对于更高阶的`y`和`x`，
求导的结果可能是更高阶的张量。

虽然雅可比矩阵确实会出现在某些
高级机器学习技术中，
但更常见的是，我们希望把
`y`的每个分量关于整个向量`x`的梯度
加起来，
得到一个与`x`形状相同的向量。
例如，我们常常有一个向量，
它表示在一*批*训练样例中
为每个样例分别计算的损失函数值。
这里，我们只想(**把为每个样例
分别计算的梯度加起来**)。

:begin_tab:`mxnet`
MXNet通过以下方式处理这个问题：在计算梯度之前，
先通过求和把所有张量降维为标量。
换句话说，它返回的不是
雅可比矩阵$\partial_{\mathbf{x}} \mathbf{y}$，
而是和函数的梯度
$\partial_{\mathbf{x}} \sum_i y_i$。
:end_tab:

:begin_tab:`pytorch`
由于不同的深度学习框架
对非标量张量梯度
的解释各不相同，
PyTorch采取了一些措施来避免混淆。
对非标量调用`backward`会引发错误，
除非我们告诉PyTorch如何把该对象降维为标量。
更形式化地说，我们需要提供某个向量$\mathbf{v}$，
使得`backward`计算的是
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$，
而不是$\partial_{\mathbf{x}} \mathbf{y}$。
接下来的这部分可能令人困惑，
但出于后面将会清楚的原因，
这个参数（表示$\mathbf{v}$）被命名为`gradient`。
更详细的描述参见Yang Zhang的
[Medium文章](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29)。
:end_tab:

:begin_tab:`tensorflow`
默认情况下，TensorFlow返回和函数的梯度。
换句话说，它返回的不是
雅可比矩阵$\partial_{\mathbf{x}} \mathbf{y}$，
而是和函数的梯度
$\partial_{\mathbf{x}} \sum_i y_i$。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## 分离计算

有时，我们希望[**把某些计算
移到记录的计算图之外。**]
例如，假设我们利用输入
创建了一些辅助的中间项，
而我们不想为这些项计算梯度。
在这种情况下，我们需要把
相应的计算图从最终结果中*分离*（detach）出来。
下面的玩具示例可以让这一点更清楚：
假设有`z = x * y`和`y = x * x`，
但我们想关注`x`对`z`的*直接*影响，
而不是经由`y`传递的影响。
在这种情况下，我们可以创建一个新变量`u`，
它与`y`取值相同，
但它的*来源*（即它是如何被创建的）
已经被抹去了。
因此`u`在图中没有祖先，
梯度也不会通过`u`流向`x`。
例如，对`z = x * u`求梯度
会得到结果`u`，
（而不是你可能预期的`3 * x * x`，
因为`z = x * x * x`）。

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

请注意，虽然这一过程
把`y`的祖先从通向`z`的图中分离了出去，
但通向`y`的计算图
仍然保留，因此我们仍然可以计算
`y`关于`x`的梯度。

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## 梯度和Python控制流

到目前为止，我们回顾的情形都是：
从输入到输出的路径
可以通过`z = x * x * x`之类的函数明确定义。
编程为我们计算结果的方式
提供了大得多的自由度。
例如，我们可以让结果依赖于辅助变量，
或者根据中间结果来决定选择。
使用自动微分的一个好处是：
[**即使**]要为
(**某个必须穿过由Python控制流构成的迷宫的函数**)
构建计算图
（例如条件语句、循环和任意函数调用），
(**我们仍然可以计算结果变量的梯度。**)
为了说明这一点，考虑下面的代码片段，
其中`while`循环的迭代次数
和`if`语句的判断
都取决于输入`a`的值。

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

下面，我们调用这个函数，
并传入一个随机值作为输入。
由于输入是一个随机变量，
我们不知道
计算图会呈现什么形式。
然而，每当我们在特定输入上
执行`f(a)`时，就会实例化
一张具体的计算图，
随后就可以运行`backward`。

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

尽管为了演示目的，
我们的函数`f`有些刻意构造，
但它对输入的依赖关系相当简单：
它是`a`的一个*线性*函数，
只是缩放系数是分段定义的。
因此，`f(a) / a`是一个各分量
均为常数的向量；而且，
`f(a) / a`必须与
`f(a)`关于`a`的梯度一致。

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

动态控制流在深度学习中非常常见。
例如，在处理文本时，计算图
取决于输入的长度。
在这些情况下，自动微分
对统计建模变得至关重要，
因为无法*先验地*计算梯度。

## 讨论

现在你已经领略了自动微分的能力。
既能自动又能高效地
计算导数的库的发展，
极大地提升了深度学习实践者的生产力，
把他们解放出来，从而能够专注于
不那么琐碎的任务。
此外，autograd让我们能够设计庞大的模型，
对这些模型而言，用纸笔进行梯度计算
将耗时过多而不可行。
有趣的是，虽然我们用autograd来*优化*模型
（从统计意义上说），
但autograd库自身的*优化*
（从计算意义上说）
却是一个内容丰富的课题，
对框架设计者至关重要。
在这里，来自编译器和图处理领域的工具
被用来以最便捷、
最节省内存的方式计算结果。

目前，请试着记住这些基础要点：(i) 为那些我们希望对其求导的变量附加梯度；(ii) 记录目标值的计算过程；(iii) 执行反向传播函数；(iv) 访问得到的梯度。


## 练习

1. 为什么计算二阶导数比计算一阶导数要昂贵得多？
1. 运行反向传播函数之后，立即再次运行它，看看会发生什么。请调查研究。
1. 在计算`d`关于`a`的导数的控制流示例中，如果把变量`a`换成随机向量或矩阵，会发生什么？此时，`f(a)`的计算结果不再是标量。结果会发生什么？我们如何分析这一点？
1. 设$f(x) = \sin(x)$。画出$f$及其导数$f'$的图像。不要利用$f'(x) = \cos(x)$这一事实，而是用自动微分来得到结果。
1. 设$f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$。写出一个依赖图，追踪从$x$到$f(x)$的结果。
1. 使用链式法则计算上述函数的导数$\frac{df}{dx}$，并把每一项放在你之前构造的依赖图上。
1. 给定依赖图和中间导数结果，你在计算梯度时有若干选择。分别从$x$到$f$计算一次结果，再从$f$回溯到$x$计算一次结果。从$x$到$f$的路径通常称为*前向微分*（forward differentiation），而从$f$到$x$的路径称为反向微分。
1. 你在什么时候可能想用前向微分，什么时候想用反向微分？提示：考虑所需的中间数据量、并行化各步骤的能力，以及所涉及的矩阵和向量的大小。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17970)
:end_tab:
