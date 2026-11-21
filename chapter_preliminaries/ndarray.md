```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 数据操作
:label:`sec_ndarray`

为了完成任何事情，我们都需要某种方式来存储和操作数据。
一般来说，处理数据时有两件重要的事要做：
(i) 获取数据；
(ii) 数据进入计算机后对其进行处理。
如果没有任何方式存储数据，那么获取数据就毫无意义，
因此作为起点，让我们从$n$维数组入手，
我们也把它称为*张量*（tensor）。
如果你已经了解NumPy科学计算包，
那么这部分内容对你来说将轻而易举。
对于所有现代深度学习框架，
*张量类*（MXNet中的`ndarray`，
PyTorch和TensorFlow中的`Tensor`）
都与NumPy的`ndarray`类似，
但额外增加了几项杀手级特性。
首先，张量类支持自动微分。
其次，它利用GPU加速数值计算，
而NumPy只能在CPU上运行。
这些特性使得神经网络
既易于编写代码，又运行得很快。



## 入门

:begin_tab:`mxnet`
首先，我们从MXNet导入`np`（`numpy`）和
`npx`（`numpy_extension`）模块。
其中，`np`模块包含
NumPy支持的函数，
而`npx`模块包含一组扩展函数，
用于在类NumPy环境中
支持深度学习。
使用张量时，我们几乎总是会
调用`set_np`函数：
这是为了与其他MXNet组件的
张量处理保持兼容。
:end_tab:

:begin_tab:`pytorch`
(**首先，我们导入PyTorch库。
请注意，该包的名称是`torch`。**)
:end_tab:

:begin_tab:`tensorflow`
首先，我们导入`tensorflow`。
为简洁起见，实践者
通常将别名指定为`tf`。
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**张量表示一个由数值组成的（可能是多维的）数组。**]
在一维情形下，即数据只需要一个轴时，
这样的张量称为*向量*（vector）。
有两个轴时，张量称为*矩阵*（matrix）。
有$k > 2$个轴时，我们不再使用这些专门名称，
而直接称该对象为$k^\textrm{th}$-*阶张量*。

:begin_tab:`mxnet`
MXNet提供了多种函数，
用于创建预先填充了数值的新张量。
例如，通过调用`arange(n)`，
我们可以创建一个由均匀间隔数值组成的向量，
从0开始（包含0），
到`n`结束（但不包含在内）。
默认情况下，间隔大小为$1$。
除非另有说明，
新张量存储在内存中，
并指定用于基于CPU的计算。
:end_tab:

:begin_tab:`pytorch`
PyTorch提供了多种函数，
用于创建预先填充了数值的新张量。
例如，通过调用`arange(n)`，
我们可以创建一个由均匀间隔数值组成的向量，
从0开始（包含0），
到`n`结束（但不包含在内）。
默认情况下，间隔大小为$1$。
除非另有说明，
新张量存储在内存中，
并指定用于基于CPU的计算。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow提供了多种函数，
用于创建预先填充了数值的新张量。
例如，通过调用`range(n)`，
我们可以创建一个由均匀间隔数值组成的向量，
从0开始（包含0），
到`n`结束（但不包含在内）。
默认情况下，间隔大小为$1$。
除非另有说明，
新张量存储在内存中，
并指定用于基于CPU的计算。
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
这些数值中的每一个都称为
张量的一个*元素*（element）。
张量`x`包含12个元素。
我们可以通过张量的`size`属性
查看其中元素的总数。
:end_tab:

:begin_tab:`pytorch`
这些数值中的每一个都称为
张量的一个*元素*（element）。
张量`x`包含12个元素。
我们可以通过张量的`numel`方法
查看其中元素的总数。
:end_tab:

:begin_tab:`tensorflow`
这些数值中的每一个都称为
张量的一个*元素*（element）。
张量`x`包含12个元素。
我们可以通过`size`函数
查看张量中元素的总数。
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**我们可以访问张量的*形状***)
（即沿每个轴的长度），
方法是查看它的`shape`属性。
由于这里处理的是一个向量，
因此`shape`只包含一个元素，
并且与大小相同。

```{.python .input}
%%tab all
x.shape
```

我们可以通过调用`reshape`来
[**改变张量的形状，而不改变其大小或数值**]。
例如，我们可以把形状为(12,)的
向量`x`变换为形状为(3, 4)的矩阵`X`。
这个新张量保留了所有元素，
但把它们重新组织成了一个矩阵。
请注意，我们的向量元素
是一次一行地排列的，因此
`x[3] == X[0, 3]`。

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

请注意，为`reshape`指定形状的每一个分量
是多余的。
因为我们已经知道张量的大小，
所以只要给定其余分量，就能算出形状的某一个分量。
例如，给定大小为$n$的张量
和目标形状($h$, $w$)，
我们知道$w = n/h$。
为了自动推断形状的某个分量，
我们可以把需要自动推断的那个形状分量
写成`-1`。
在我们的例子中，除了调用`x.reshape(3, 4)`，
我们还可以等价地调用`x.reshape(-1, 4)`或`x.reshape(3, -1)`。

实践者经常需要处理
初始化为全0或全1的张量。
[**我们可以构造一个所有元素都设为0的张量**] (~~或全为1的张量~~)，
其形状为(2, 3, 4)，
只需调用`zeros`函数。

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

类似地，我们可以调用`ones`
创建一个全为1的张量。

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

我们常常希望从给定的概率分布中
[**随机（且独立地）采样每个元素**]。
例如，神经网络的参数
往往被随机初始化。
下面的代码片段创建一个张量，
其元素从均值为0、
标准差为1的标准高斯（正态）分布中抽取。

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

最后，我们可以通过提供包含数值字面量的
（可能嵌套的）Python列表，
[**为每个元素提供精确的数值**]，
以此构造张量。
这里，我们用一个列表的列表构造一个矩阵，
其中最外层的列表对应轴0，
内层的列表对应轴1。

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 索引与切片

与Python列表一样，
我们可以通过索引访问张量元素
（索引从0开始）。
若要基于元素相对于列表末尾的位置
来访问某个元素，
我们可以使用负索引。
此外，我们可以通过切片
（例如`X[start:stop]`）访问整个索引范围，
返回的值包含第一个索引（`start`）
*但不包含最后一个*（`stop`）。
最后，当对$k^\textrm{th}$阶张量
只指定一个索引（或切片）时，
它作用于轴0。
因此，在下面的代码中，
[**`[-1]`选择最后一行，
`[1:3]`选择第二行和第三行**]。

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
除了读取元素之外，(**我们还可以通过指定索引来*写入*矩阵的元素。**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`Tensors`是不可变的，不能对其赋值。
TensorFlow中的`Variables`是可变的状态容器，支持赋值操作。
请记住，TensorFlow中的梯度不会通过`Variable`赋值反向传播。

除了为整个`Variable`赋值之外，我们还可以通过指定索引
来写入`Variable`的元素。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

如果我们想[**为多个元素赋相同的值，
只需在赋值运算的左侧应用索引即可。**]
例如，`[:2, :]`访问
第一行和第二行，
其中`:`取沿轴1（列）的所有元素。
虽然我们讨论的是矩阵的索引，
但这也适用于向量
以及二维以上的张量。

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## 运算

现在，我们已经知道如何构造张量，
以及如何读写张量的元素，
接下来就可以开始用各种数学运算
来操作它们。
在这些运算中，最有用的一类
是*按元素*（elementwise）运算。
这类运算把标准的标量运算
作用于张量的每一个元素。
对于以两个张量为输入的函数，
按元素运算会对每一对对应的元素
应用某种标准的二元运算符。
我们可以从任何由标量映射到标量的函数出发，
构造一个按元素函数。

在数学记号中，我们用签名
$f: \mathbb{R} \rightarrow \mathbb{R}$
来表示这类*一元*标量运算符（只接受一个输入）。
这只说明该函数把任意实数
映射为另一个实数。
大多数标准运算符，包括像$e^x$这样的一元运算符，都可以按元素应用。

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

同样，我们用签名
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$
来表示*二元*标量运算符，
它把一对实数
映射为一个（单个）实数。
给定任意两个*形状相同*的向量$\mathbf{u}$
和$\mathbf{v}$，
再给定一个二元运算符$f$，我们就能得到向量
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$，
只需令$c_i \gets f(u_i, v_i)$对所有$i$成立，
其中$c_i, u_i$和$v_i$分别是
向量$\mathbf{c}, \mathbf{u}$和$\mathbf{v}$的第$i^\textrm{th}$个元素。
这里，我们通过把标量函数*提升*为
按元素的向量运算，
得到了向量值函数
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$。
常见的标准算术运算符——
加法（`+`）、减法（`-`）、
乘法（`*`）、除法（`/`）
和求幂（`**`）——
都已被*提升*为适用于任意形状、
形状相同张量的按元素运算。

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

除按元素计算之外，
我们还可以执行线性代数运算，
例如点积和矩阵乘法。
我们将在 :numref:`sec_linear-algebra` 中
详细讨论这些内容。

我们还可以[***连结*多个张量**]，
把它们首尾相接地堆叠起来，形成一个更大的张量。
我们只需提供一个张量列表，
并告诉系统沿哪个轴连结。
下面的例子展示了沿行（轴0）
而不是沿列（轴1）连结两个矩阵时会发生什么。
我们可以看到，第一个输出的轴0长度（$6$）
是两个输入张量轴0长度之和（$3 + 3$）；
而第二个输出的轴1长度（$8$）
是两个输入张量轴1长度之和（$4 + 4$）。

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

有时，我们希望
[**通过*逻辑语句*构造一个二元张量。**]
以`X == Y`为例。
对于每个位置`i, j`，如果`X[i, j]`与`Y[i, j]`相等，
那么结果中对应条目的取值为`1`，
否则取值为`0`。

```{.python .input}
%%tab all
X == Y
```

[**对张量中所有元素求和**]会得到一个只有一个元素的张量。

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## 广播
:label:`subsec_broadcasting`

到现在为止，你已经知道如何
对两个形状相同的张量
执行按元素的二元运算。
在某些条件下，
即使形状不同，
我们仍然可以[**通过调用*广播机制*（broadcasting）
来执行按元素的二元运算。**]
广播按照以下两步流程进行：
(i) 沿长度为1的轴复制元素，
扩展一个或两个数组，
使得这一变换之后
两个张量具有相同的形状；
(ii) 对得到的数组
执行按元素运算。

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

由于`a`和`b`分别是$3\times1$
和$1\times2$的矩阵，
它们的形状并不匹配。
广播会先沿列复制矩阵`a`、
沿行复制矩阵`b`，
然后按元素相加，从而产生一个更大的$3\times2$矩阵。

```{.python .input}
%%tab all
a + b
```

## 节省内存

[**运行运算可能会导致为存放结果而
分配新的内存。**]
例如，如果我们写`Y = X + Y`，
就会解除`Y`原先指向的张量，
转而让`Y`指向新分配的内存。
我们可以用Python的`id()`函数演示这一问题，
该函数给出所引用对象在内存中的确切地址。
请注意，在运行`Y = Y + X`之后，
`id(Y)`指向了不同的位置。
这是因为Python先计算`Y + X`，
为结果分配新的内存，
然后再让`Y`指向内存中的这个新位置。

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

这种做法不可取，原因有二。
首先，我们不希望总是到处
不必要地分配内存。
在机器学习中，我们往往有
数百兆字节的参数，
并且每秒要多次更新它们。
只要可能，我们就希望*原地*执行这些更新。
其次，我们可能从多个变量
指向同一组参数。
如果不是原地更新，
我们就必须小心地更新所有这些引用，
以免引起内存泄漏，
或者无意中引用了过期的参数。

:begin_tab:`mxnet, pytorch`
幸运的是，(**执行原地运算**)很容易。
我们可以把运算结果赋给
之前已经分配好的数组`Y`，
也就是使用切片记号：`Y[:] = <expression>`。
为了说明这一概念，
我们先覆盖张量`Z`的值，
即用`zeros_like`把它初始化为
与`Y`具有相同的形状。
:end_tab:

:begin_tab:`tensorflow`
`Variables`是TensorFlow中可变的状态容器。它们提供
了一种存储模型参数的方式。
我们可以用`assign`把运算结果赋给一个`Variable`。
为了说明这一概念，
我们先覆盖`Variable` `Z`的值，
即用`zeros_like`把它初始化为
与`Y`具有相同的形状。
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**如果`X`的值在后续计算中不再使用，
我们还可以用`X[:] = X + Y`或`X += Y`
来减少该运算的内存开销。**]
:end_tab:

:begin_tab:`tensorflow`
即使你已经把状态持久地存储在`Variable`中，
你可能还希望进一步减少内存占用，
办法是避免为非模型参数的张量分配多余的内存。
由于TensorFlow的`Tensors`是不可变的，
而且梯度不会通过`Variable`赋值流动，
因此TensorFlow没有提供显式地
原地执行单个运算的方式。

不过，TensorFlow提供了`tf.function`装饰器，
可以把计算包装进一张TensorFlow图，
在运行前对这张图进行编译和优化。
这使得TensorFlow能够剪除未使用的值，
并复用不再需要的先前分配。
这就把TensorFlow计算的内存开销降至最低。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 转换为其他Python对象

:begin_tab:`mxnet, tensorflow`
[**转换为NumPy张量（`ndarray`）**]，或者反过来转换，都很容易。
转换后的结果不共享内存。
这个小小的不便其实相当重要：
当你在CPU或GPU上执行运算时，
你不希望暂停计算，去等待Python的NumPy包
是否可能想对同一块内存做别的事情。
:end_tab:

:begin_tab:`pytorch`
[**转换为NumPy张量（`ndarray`）**]，或者反过来转换，都很容易。
torch张量与NumPy数组
会共享底层内存，
通过原地运算修改其中一个，
另一个也会随之改变。
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

为了(**把大小为1的张量转换为Python标量**)，
我们可以调用`item`函数或Python的内置函数。

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## 小结

张量类是深度学习库中用于存储和操作数据的主要接口。
张量提供了多种功能，包括构造例程；索引与切片；基本数学运算；广播；节省内存的赋值；以及与其他Python对象之间的相互转换。


## 练习

1. 运行本节中的代码。把条件语句`X == Y`改为`X < Y`或`X > Y`，然后看看你能得到什么样的张量。
1. 把广播机制中按元素运算的两个张量替换为其他形状，例如三维张量。结果是否与预期相同？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17966)
:end_tab:
