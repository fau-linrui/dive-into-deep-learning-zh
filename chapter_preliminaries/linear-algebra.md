```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 线性代数
:label:`sec_linear-algebra`

到目前为止，我们可以把数据集加载到张量中，
并用基本的数学运算来操作这些张量。
要开始构建复杂的模型，
我们还需要一些线性代数工具。
本节将浅显地介绍最重要的若干概念，
从标量算术开始，
逐步过渡到矩阵乘法。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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

## 标量


日常数学大多由
一次处理一个数的运算组成。
形式上，我们把这样的数值称为*标量*（scalar）。
例如，帕洛阿尔托的气温
是温暖的$72$华氏度。
如果你想把这个温度转换为摄氏度，
就要计算表达式
$c = \frac{5}{9}(f - 32)$，并把$f$设为$72$。
在这个方程中，数值
$5$、$9$和$32$是常量标量。
变量$c$和$f$
一般表示未知的标量。

我们用普通的小写字母
（例如$x$、$y$和$z$）表示标量，
用$\mathbb{R}$表示所有（连续）
*实值*标量的空间。
为简便起见，我们将跳过
对*空间*的严格定义：
只需记住，表达式$x \in \mathbb{R}$
是一种正式说法，表示$x$是一个实值标量。
符号$\in$（读作“in”）
表示属于某个集合。
例如，$x, y \in \{0, 1\}$
表示$x$和$y$是只能
取值$0$或$1$的变量。

(**标量实现为只包含一个元素的张量。**)
下面，我们赋值两个标量，
并执行熟悉的加法、乘法、
除法和求幂运算。

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## 向量

就目前而言，[**你可以把向量看作一个固定长度的标量数组。**]
与代码中对应的对象一样，
我们把这些标量称为向量的*元素*（element），
同义词还有*条目*（entry）和*分量*（component）。
当向量表示来自真实世界数据集的样例时，
它们的数值具有一定的实际意义。
例如，如果我们正在训练一个模型来预测
贷款违约的风险，
我们可能会把每个申请人与一个向量关联起来，
该向量的各个分量对应
他们的收入、工作年限
或过往违约次数等数量。
如果我们正在研究心脏病的风险，
每个向量可能代表一位病人，
其分量可能对应
他们最近的生命体征、胆固醇水平、
每天锻炼的分钟数等。
我们用粗体小写字母
（例如$\mathbf{x}$、$\mathbf{y}$和$\mathbf{z}$）表示向量。

向量实现为$1^{\textrm{st}}$阶张量。
一般来说，这类张量可以有任意长度，
只受内存限制。需注意：与大多数编程语言一样，Python中向量的索引从$0$开始，这也称为*从零开始索引*（zero-based indexing），而在线性代数中，下标从$1$开始（从1开始索引）。

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

我们可以用下标来引用向量的元素。
例如，$x_2$表示$\mathbf{x}$的第二个元素。
由于$x_2$是标量，我们不用粗体表示它。
默认情况下，我们通过
把向量的元素垂直堆叠来可视化向量：

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

这里$x_1, \ldots, x_n$是向量的元素。
后面我们还会区分这样的*列向量*（column vector）
和元素水平堆叠的*行向量*（row vector）。
回想一下，[**我们通过索引访问张量的元素。**]

```{.python .input}
%%tab all
x[2]
```

为了表示一个向量包含$n$个元素，
我们写作$\mathbf{x} \in \mathbb{R}^n$。
形式上，我们把$n$称为向量的*维数*（dimensionality）。
[**在代码中，这对应张量的长度**]，
可以通过Python内置的`len`函数获取。

```{.python .input}
%%tab all
len(x)
```

我们也可以通过`shape`属性访问长度。
形状是一个元组，表示张量沿每个轴的长度。
(**只有一个轴的张量，其形状只有一个元素。**)

```{.python .input}
%%tab all
x.shape
```

人们常常把“维度”（dimension）一词的含义弄混，
既用它表示轴的数量，
又用它表示沿某个特定轴的长度。
为避免这种混淆，
我们用*阶*（order）表示轴的数量，
而把*维数*（dimensionality）专门用来表示
分量的数量。


## 矩阵

正如标量是$0^{\textrm{th}}$阶张量、
向量是$1^{\textrm{st}}$阶张量一样，
矩阵是$2^{\textrm{nd}}$阶张量。
我们用粗体大写字母
（例如$\mathbf{X}$、$\mathbf{Y}$和$\mathbf{Z}$）表示矩阵，
并在代码中用具有两个轴的张量来表示它们。
表达式$\mathbf{A} \in \mathbb{R}^{m \times n}$
表示矩阵$\mathbf{A}$
包含$m \times n$个实值标量，
排列成$m$行$n$列。
当$m = n$时，我们说这个矩阵是*方阵*（square）。
直观上，我们可以把任何矩阵画成一张表格。
要引用单个元素，
我们对行索引和列索引都加下标，例如，
$a_{ij}$是$\mathbf{A}$的
第$i^{\textrm{th}}$行第$j^{\textrm{th}}$列的值：

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


在代码中，我们用形状为($m$, $n$)的$2^{\textrm{nd}}$阶张量
来表示矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$。
[**我们可以把任何尺寸合适的$m \times n$张量
转换为$m \times n$矩阵**]，
只需把期望的形状传给`reshape`：

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

有时我们想翻转坐标轴。
交换矩阵的行和列之后，
得到的结果称为矩阵的*转置*（transpose）。
形式上，我们用$\mathbf{A}^\top$表示矩阵$\mathbf{A}$的转置，
如果$\mathbf{B} = \mathbf{A}^\top$，
那么对所有$i$和$j$都有$b_{ij} = a_{ji}$。
因此，$m \times n$矩阵的转置
是一个$n \times m$矩阵：

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

在代码中，我们可以按如下方式
访问任何(**矩阵的转置**)：

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**对称矩阵是这样一类方阵：
它等于自身的转置，即$\mathbf{A} = \mathbf{A}^\top$。**]
下面的矩阵是对称的：

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

矩阵很适合用来表示数据集。
通常，行对应各个记录，
列对应不同的属性。



## 张量

虽然在机器学习之旅中，
只靠标量、向量和矩阵
你也能走得很远，
但最终你可能需要处理
更高阶的[**张量**]。
张量(**为我们提供了一种通用方式，
用来描述对$n^{\textrm{th}}$阶数组的扩展。**)
我们把*张量类*的软件对象称为“张量”，
正是因为它们同样可以具有任意数量的轴。
虽然把*张量*一词同时用于
数学对象及其在代码中的实现可能令人困惑，
但根据上下文，我们的意思通常是清楚的。
我们用特殊字体的大写字母
（例如$\mathsf{X}$、$\mathsf{Y}$和$\mathsf{Z}$）表示一般张量，
其索引机制
（例如$x_{ijk}$和$[\mathsf{X}]_{1, 2i-1, 3}$）
自然地继承自矩阵的索引机制。

当我们开始处理图像时，
张量会变得更加重要。
每幅图像都以$3^{\textrm{rd}}$阶张量的形式出现，
其各个轴分别对应高度、宽度和*通道*（channel）。
在每个空间位置上，每种颜色
（红、绿、蓝）的强度
沿通道堆叠。
此外，一组图像在代码中
用一个$4^{\textrm{th}}$阶张量表示，
其中不同的图像沿第一个轴索引。
高阶张量的构造方式与向量和矩阵相同，
都是通过增加形状分量的数量来实现的。

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## 张量算法的基本性质

标量、向量、矩阵
和高阶张量
都具有一些方便的性质。
例如，按元素运算
产生的输出与其操作数
具有相同的形状。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

两个矩阵的[**按元素乘积称为它们的*哈达玛积*（Hadamard product）**]
（记作$\odot$）。
我们可以写出两个矩阵
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$的
哈达玛积的各个条目：



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**标量与张量的加法或乘法**]
所产生的结果与原张量具有相同的形状。
这里，张量的每个元素都与该标量相加（或相乘）。

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## 降维
:label:`subsec_lin-alg-reduction`

我们常常希望计算[**张量元素的和。**]
为了表示长度为$n$的向量$\mathbf{x}$中元素的和，
我们写作$\sum_{i=1}^n x_i$。
有一个简单的函数可以完成这件事：

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

为了表示[**任意形状张量的元素之和**]，
我们只需沿其所有轴求和。
例如，$m \times n$矩阵$\mathbf{A}$的元素之和
可以写作$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

默认情况下，调用求和函数
会沿张量的所有轴*降维*，
最终产生一个标量。
我们的库还允许我们[**指定张量应沿哪些轴降维。**]
要沿行（轴0）对所有元素求和，
我们在`sum`中指定`axis=0`。
由于输入矩阵沿轴0降维以生成输出向量，
因此该轴在输出的形状中消失了。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

指定`axis=1`将沿列维度（轴1）降维，即对所有列的元素求和。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

通过求和同时沿行和列对矩阵降维，
等价于对矩阵的所有元素求和。

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**与此相关的一个量是*均值*（mean），也称为*平均数*（average）。**]
我们用元素总和除以元素总数来计算均值。
由于计算均值非常常见，
因此它有专门的库函数，
其用法与`sum`类似。

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同样，用于计算均值的函数
也可以沿特定轴对张量降维。

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## 非降维求和
:label:`subsec_lin-alg-non-reduction`

有时，在调用计算和或均值的函数时，
[**保持轴数不变**]会很有用。
当我们想使用广播机制时，这一点很重要。

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

例如，由于`sum_A`在求每行之和后仍保留它的两个轴，
我们可以(**通过广播用`A`除以`sum_A`**)
来创建一个每行之和为$1$的矩阵。

```{.python .input}
%%tab all
A / sum_A
```

如果我们想计算[**`A`中元素沿某个轴的累积和**]，
比如沿`axis=0`（逐行累加），可以调用`cumsum`函数。
按照设计，该函数不会沿任何轴对输入张量降维。

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## 点积

到目前为止，我们只执行过按元素运算、求和和求平均。
如果这就是我们能做的全部，
线性代数就不值得单独占一节。
幸运的是，从这里开始事情变得更有意思了。
最基本的运算之一是点积。
给定两个向量$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，
它们的*点积*（dot product）$\mathbf{x}^\top \mathbf{y}$（也称为*内积*（inner product），$\langle \mathbf{x}, \mathbf{y}  \rangle$）
是同一位置上元素乘积的和：
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

[~~两个向量的*点积*是同一位置上元素乘积的和~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

等价地，(**我们可以先按元素相乘，再求和，
从而计算两个向量的点积：**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

点积在广泛的场景中都有用。
例如，给定一组由向量$\mathbf{x}  \in \mathbb{R}^n$表示的值，
以及一组由$\mathbf{w} \in \mathbb{R}^n$表示的权重，
$\mathbf{x}$中的值按权重$\mathbf{w}$计算的加权和
可以表示为点积$\mathbf{x}^\top \mathbf{w}$。
当权重非负
且和为$1$时，即$\left(\sum_{i=1}^{n} {w_i} = 1\right)$，
点积表示一个*加权平均*（weighted average）。
把两个向量规范化为单位长度之后，
点积表示它们之间夹角的余弦。
本节后面，我们将正式引入*长度*（length）这一概念。


## 矩阵-向量积

既然我们已经知道如何计算点积，
就可以开始理解
$m \times n$矩阵$\mathbf{A}$
与$n$维向量$\mathbf{x}$之间的*乘积*了。
首先，我们用矩阵的行向量
来表示它

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每个$\mathbf{a}^\top_{i} \in \mathbb{R}^n$
是一个行向量，表示
矩阵$\mathbf{A}$的第$i^\textrm{th}$行。

[**矩阵-向量积$\mathbf{A}\mathbf{x}$
就是一个长度为$m$的列向量，
其第$i^\textrm{th}$个元素是点积
$\mathbf{a}^\top_i \mathbf{x}$：**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

我们可以把与矩阵
$\mathbf{A}\in \mathbb{R}^{m \times n}$相乘
看作一种变换，它把向量
从$\mathbb{R}^{n}$投影到$\mathbb{R}^{m}$。
这些变换非常有用。
例如，我们可以用与某些方阵相乘
来表示旋转。
矩阵-向量积还描述了这样的关键计算：
在神经网络中，根据上一层的输出
计算每一层的输出。

:begin_tab:`mxnet`
为了在代码中表示矩阵-向量积，
我们使用同样的`dot`函数。
具体执行哪种运算
是根据参数的类型推断出来的。
请注意，`A`的列维度
（沿轴1的长度）
必须与`x`的维度（其长度）相同。
:end_tab:

:begin_tab:`pytorch`
为了在代码中表示矩阵-向量积，
我们使用`mv`函数。
请注意，`A`的列维度
（沿轴1的长度）
必须与`x`的维度（其长度）相同。
Python有一个便捷运算符`@`，
既可以执行矩阵-向量积，
也可以执行矩阵-矩阵乘法
（取决于它的参数）。
因此我们可以写`A@x`。
:end_tab:

:begin_tab:`tensorflow`
为了在代码中表示矩阵-向量积，
我们使用`matvec`函数。
请注意，`A`的列维度
（沿轴1的长度）
必须与`x`的维度（其长度）相同。
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## 矩阵-矩阵乘法

一旦你掌握了点积和矩阵-向量积，
*矩阵-矩阵乘法*就应当很容易理解了。

假设我们有两个矩阵
$\mathbf{A} \in \mathbb{R}^{n \times k}$
和$\mathbf{B} \in \mathbb{R}^{k \times m}$：

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


令$\mathbf{a}^\top_{i} \in \mathbb{R}^k$表示
代表矩阵$\mathbf{A}$第$i^\textrm{th}$行的行向量，
并令$\mathbf{b}_{j} \in \mathbb{R}^k$表示
矩阵$\mathbf{B}$第$j^\textrm{th}$列的列向量：

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


为了构成矩阵乘积$\mathbf{C} \in \mathbb{R}^{n \times m}$，
我们只需把每个元素$c_{ij}$
计算为
$\mathbf{A}$的第$i^{\textrm{th}}$行
与$\mathbf{B}$的第$j^{\textrm{th}}$列之间的点积，
即$\mathbf{a}^\top_i \mathbf{b}_j$：

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**我们可以把矩阵-矩阵乘法$\mathbf{AB}$
看作执行$m$次矩阵-向量积
或$m \times n$次点积，
再把结果拼接起来
构成一个$n \times m$矩阵。**]
在下面的代码片段中，
我们对`A`和`B`执行矩阵乘法。
这里，`A`是一个两行三列的矩阵，
而`B`是一个三行四列的矩阵。
相乘之后，我们得到一个两行四列的矩阵。

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

*矩阵-矩阵乘法*一词
常简称为*矩阵乘法*，
不应与哈达玛积混淆。

## 范数
:label:`subsec_lin-algebra-norms`

线性代数中最有用的一些运算符是*范数*（norm）。
直观地说，向量的范数告诉我们它有多*大*。
例如，$\ell_2$范数度量
一个向量的（欧几里得）长度。
这里我们采用的*大小*概念
关注的是向量各分量的量值
（而不是它的维数）。

范数是一个函数$\| \cdot \|$，它把向量
映射为标量，并满足以下三条性质：

1. 给定任意向量$\mathbf{x}$，如果我们把该向量（的所有元素）
   按标量$\alpha \in \mathbb{R}$缩放，它的范数也按相应比例缩放：
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. 对任意向量$\mathbf{x}$和$\mathbf{y}$：
   范数满足三角不等式：
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. 向量的范数非负，并且仅当向量为零向量时才等于零：
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

很多函数都是合法的范数，不同的范数
编码了不同的大小概念。
我们在小学几何中计算直角三角形斜边时
学到的欧几里得范数，
就是向量各元素平方和的平方根。
形式上，这称为[**$\ell_2$ *范数***]，表示为

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

`norm`方法用于计算$\ell_2$范数。

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**$\ell_1$范数**]也很常见，
与之对应的度量称为曼哈顿距离。
按照定义，$\ell_1$范数对
向量各元素的绝对值求和：

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

与$\ell_2$范数相比，它对离群值不那么敏感。
为了计算$\ell_1$范数，
我们把绝对值运算
与求和运算组合起来。

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

$\ell_2$范数和$\ell_1$范数
都是一般的$\ell_p$*范数*的特例：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

对于矩阵，情况更为复杂。
毕竟，矩阵既可以看作各个独立条目的集合，
*也*可以看作对向量进行运算、
把向量变换为其他向量的对象。
例如，我们可以问
矩阵-向量积$\mathbf{X} \mathbf{v}$
相对于$\mathbf{v}$能大多少。
沿着这条思路就得到了所谓的*谱*范数（spectral norm）。
目前，我们先介绍[**弗罗贝尼乌斯范数（Frobenius norm），
它计算起来容易得多**]，
其定义为矩阵各元素平方和的平方根：

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

弗罗贝尼乌斯范数的行为
就好像它是把矩阵形状的向量
当作$\ell_2$范数来度量。
调用下面的函数即可计算
矩阵的弗罗贝尼乌斯范数。

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

虽然我们不想过于超前，
但已经可以就这些概念为什么有用
建立一些直觉。
在深度学习中，我们常常试图求解优化问题：
*最大化*分配给观测数据的概率；
*最大化*与推荐模型相关的收入；
*最小化*预测与真实观测之间的距离；
*最小化*同一个人的照片的表示之间的距离，
同时*最大化*不同人的照片的表示之间的距离。
这些距离构成了深度学习算法的目标，
它们常常用范数来表示。


## 讨论

本节中，我们回顾了
理解现代深度学习很大一部分内容
所需的全部线性代数。
不过，线性代数远不止这些，
其中很多内容对机器学习都很有用。
例如，矩阵可以分解为若干因子，
这些分解可以揭示
真实世界数据集中的低维结构。
机器学习中有一些完整的研究子领域，
专门研究如何利用矩阵分解
及其向高阶张量的推广，
来发现数据集中的结构
并解决预测问题。
但本书的重点是深度学习。
我们相信，一旦你亲自动手
把机器学习应用于真实数据集，
就会更愿意学习更多数学。
因此，虽然我们保留
在后续引入更多数学内容的权利，
但本节到此收尾。

如果你想热切地学习更多线性代数，
有许多优秀的书籍和在线资源。
若想学习更高级的速成课程，可以参考
:citet:`Strang.1993`、:citet:`Kolter.2008` 和 :citet:`Petersen.Pedersen.ea.2008`。

回顾一下：

* 标量、向量、矩阵和张量是
  线性代数中使用的基本数学对象，
  它们分别具有零个、一个、两个和任意数量的轴。
* 张量可以通过索引沿指定轴切片，
  也可以通过`sum`和`mean`等运算沿指定轴降维。
* 按元素乘积称为哈达玛积。
  相比之下，点积、矩阵-向量积和矩阵-矩阵乘法
  都不是按元素运算，它们一般返回的对象
  形状与操作数不同。
* 与哈达玛积相比，矩阵-矩阵乘法
  的计算耗时要长得多（时间复杂度为三次而非二次）。
* 范数刻画了向量（或矩阵）量值的各种概念，
  并且常用于计算两个向量之差，
  以度量它们之间的距离。
* 常见的向量范数包括$\ell_1$范数和$\ell_2$范数，
  常见的矩阵范数包括*谱*范数和*弗罗贝尼乌斯*范数。


## 练习

1. 证明：矩阵转置的转置就是该矩阵本身，即 $(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 给定两个矩阵 $\mathbf{A}$ 和 $\mathbf{B}$，证明求和与转置可交换：$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 给定任意方阵 $\mathbf{A}$，$\mathbf{A} + \mathbf{A}^\top$ 总是对称的吗？能否仅利用前两题的结果给出证明？
1. 本节中我们定义了形状为(2, 3, 4)的张量`X`。`len(X)`的输出是什么？先不运行任何代码写出答案，然后再用代码检验你的答案。
1. 对于任意形状的张量`X`，`len(X)`是否总是对应`X`某个轴的长度？是哪个轴？
1. 运行`A / A.sum(axis=1)`，看看会发生什么。你能分析结果吗？
1. 在曼哈顿市中心两点之间出行时，用坐标（即用大道和街道）表示，你需要走过的距离是多少？你能沿对角线走吗？
1. 考虑一个形状为(2, 3, 4)的张量。沿轴0、轴1和轴2求和后的输出形状分别是什么？
1. 把一个具有三个或更多轴的张量送入`linalg.norm`函数，观察其输出。对于任意形状的张量，这个函数计算的是什么？
1. 考虑三个大矩阵，例如用高斯随机变量初始化的 $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$、$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ 和 $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$。你想计算乘积 $\mathbf{A} \mathbf{B} \mathbf{C}$。计算 $(\mathbf{A} \mathbf{B}) \mathbf{C}$ 与计算 $\mathbf{A} (\mathbf{B} \mathbf{C})$ 在内存占用和速度上是否有差别？为什么？
1. 考虑三个大矩阵，例如 $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$、$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ 和 $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$。计算 $\mathbf{A} \mathbf{B}$ 与计算 $\mathbf{A} \mathbf{C}^\top$ 在速度上是否有差别？为什么？如果不克隆内存而用 $\mathbf{C} = \mathbf{B}^\top$ 初始化，会有什么变化？为什么？
1. 考虑三个矩阵，例如 $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$。通过堆叠 $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$ 构造一个具有三个轴的张量。它的维数是多少？切出第三个轴的第二个坐标，以恢复出 $\mathbf{B}$。检验你的答案是否正确。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17968)
:end_tab:
