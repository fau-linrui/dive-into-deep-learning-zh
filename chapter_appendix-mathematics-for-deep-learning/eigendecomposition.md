# 特征分解
:label:`sec_eigendecompositions`

特征值往往是我们在学习线性代数时会遇到的最有用的概念之一，然而作为初学者，很容易忽视它们的重要性。下面我们介绍特征分解，并尝试传达它为何如此重要。

假设我们有一个元素如下的矩阵 $A$：

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

如果把 $A$ 作用于任意向量 $\mathbf{v} = [x, y]^\top$，我们会得到向量 $\mathbf{A}\mathbf{v} = [2x, -y]^\top$。这有一个直观的解释：在 $x$ 方向上把向量拉伸为原来的两倍宽，然后在 $y$ 方向上翻转它。

然而，有*一些*向量会保持某种不变。具体来说，$[1, 0]^\top$ 被映射到 $[2, 0]^\top$，$[0, 1]^\top$ 被映射到 $[0, -1]^\top$。这些向量仍位于同一条直线上，唯一的改变是矩阵分别将它们拉伸了 $2$ 倍和 $-1$ 倍。我们称这类向量为*特征向量*（eigenvector），称被拉伸的倍数*特征值*（eigenvalue）。

一般而言，如果我们能找到数 $\lambda$ 和向量 $\mathbf{v}$，使得

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

我们就说 $\mathbf{v}$ 是 $A$ 的特征向量，$\lambda$ 是特征值。

## 求特征值
我们来看看如何求出它们。从等式两边减去 $\lambda \mathbf{v}$，然后提取出向量因子，可以看出上式等价于：

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

要使 :eqref:`eq_eigvalue_der` 成立，$(\mathbf{A} - \lambda \mathbf{I})$ 必须把某个方向压缩为零，因此它是不可逆的，从而行列式为零。于是，我们可以通过求出使 $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ 成立的 $\lambda$ 来得到*特征值*。一旦求出特征值，就可以求解 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ 来找到相应的*特征向量*。

### 一个示例
我们用一个更具挑战性的矩阵来看看

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

考虑 $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$，可以看出它等价于多项式方程 $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$。因此，两个特征值是 $4$ 和 $1$。为了求出相应的向量，我们需要求解

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

分别用向量 $[1, -1]^\top$ 和 $[1, 2]^\top$ 即可解出。

我们可以用内置的 `numpy.linalg.eig` 例程在代码中验证这一点。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

请注意，`numpy` 会把特征向量规范化为单位长度，而我们取的是任意长度。此外，符号的选择也是任意的。不过，计算出的向量与我们手工求出的、对应相同特征值的向量是平行的。

## 分解矩阵
让我们把上一个例子再推进一步。设

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

为以矩阵 $\mathbf{A}$ 的特征向量为列的矩阵。设

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

为对角线上元素是相应特征值的矩阵。于是，特征值和特征向量的定义告诉我们

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

矩阵 $W$ 是可逆的，因此我们可以在等式两边右乘 $W^{-1}$，于是可以写成

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

下一节我们将看到它的一些良好推论，但目前我们只需知道：只要我们能够找到一组完整的线性无关的特征向量（从而使 $W$ 可逆），这样的分解就会存在。

## 特征分解上的运算
特征分解 :eqref:`eq_eig_decomp` 的一个好处是，我们可以把通常遇到的许多运算用特征分解简洁地表示出来。作为第一个例子，考虑：

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

这告诉我们，对于矩阵的任意正整数次幂，其特征分解只需把特征值取相同的幂即可得到。负整数次幂也可照此证明，因此如果我们想求矩阵的逆，只需考虑

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

换言之，只需对每个特征值取倒数。只要每个特征值非零，这样做就是可行的，因此我们看到：可逆等价于不存在零特征值。

事实上，进一步的工作可以证明：如果 $\lambda_1, \ldots, \lambda_n$ 是矩阵的特征值，那么该矩阵的行列式为

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

即所有特征值的乘积。这在直觉上是合理的，因为无论 $\mathbf{W}$ 产生了怎样的拉伸，$W^{-1}$ 都会将其抵消，所以最终发生的拉伸只是乘以对角矩阵 $\boldsymbol{\Sigma}$，而它按对角元素的乘积来缩放体积。

最后，回想秩是矩阵中线性无关列的最大数目。仔细考察特征分解，可以看出秩等于 $\mathbf{A}$ 的非零特征值的个数。

这样的例子还可以继续举下去，但希望要点已经清楚：特征分解可以简化许多线性代数计算，并且是许多数值算法以及我们在本书中做的大量线性代数分析背后的基本运算。

## 对称矩阵的特征分解
并不总是能够找到足够多的线性无关特征向量来使上述过程奏效。例如矩阵

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

只有一个特征向量，即 $(1, 0)^\top$。为了处理这类矩阵，我们需要本书无法涵盖的更高级技术（例如 Jordan 标准型或奇异值分解）。我们往往需要把注意力限制在那些能够保证存在完整特征向量集的矩阵上。

最常遇到的一类是*对称矩阵*（symmetric matrix），即满足 $\mathbf{A} = \mathbf{A}^\top$ 的矩阵。在这种情况下，我们可以取 $W$ 为*正交矩阵*（orthogonal matrix）——即各列都是长度为 1 且彼此正交的向量的矩阵，其中 $\mathbf{W}^\top = \mathbf{W}^{-1}$——并且所有特征值都是实数。因此，在这种特殊情况下，我们可以把 :eqref:`eq_eig_decomp` 写成

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Gershgorin 圆盘定理
特征值往往难以凭直觉来推理。给定一个任意矩阵，如果不进行计算，几乎无法说明其特征值是什么。不过，有一个定理可以在矩阵对角线上的值最大时，让我们容易地得到较好的近似。

设 $\mathbf{A} = (a_{ij})$ 是任意方阵（$n\times n$）。我们定义 $r_i = \sum_{j \neq i} |a_{ij}|$。设 $\mathcal{D}_i$ 表示复平面上以 $a_{ii}$ 为圆心、以 $r_i$ 为半径的圆盘。那么，$\mathbf{A}$ 的每个特征值都包含在某个 $\mathcal{D}_i$ 之中。

这个定理可能需要稍微消化一下，我们来看一个例子。考虑矩阵：

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

我们有 $r_1 = 0.3$、$r_2 = 0.6$、$r_3 = 0.8$ 和 $r_4 = 0.9$。该矩阵是对称矩阵，因此所有特征值都是实数。这意味着我们所有的特征值都会落在下列区间之一中

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


进行数值计算可知，特征值约为 $0.99$、$2.97$、$4.95$、$9.08$，它们都稳稳地落在上述区间之内。

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

通过这种方式可以近似特征值，并且在矩阵对角线元素显著大于其他所有元素的情况下，这些近似会相当准确。

这虽然是个小技巧，但对于特征分解这样复杂而微妙的话题，凡能获得的直观理解都是有益的。

## 一个有用的应用：迭代映射的增长

既然我们已经从原理上理解了特征向量是什么，接下来看看如何利用它们来深入理解神经网络行为中的一个核心问题：恰当的权重初始化。

### 特征向量与长期行为

对深度神经网络初始化进行完整的数学研究超出了本书的范围，但我们可以在这里看一个简化的玩具版本，以理解特征值如何帮助我们看清这些模型的工作原理。我们知道，神经网络通过把线性变换层与非线性操作交替穿插来运作。为简单起见，这里我们假设不存在非线性，且变换是单一的重复矩阵运算 $A$，从而模型的输出为

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

这些模型初始化时，$A$ 被取为一个元素服从高斯分布的随机矩阵，所以我们来生成一个这样的矩阵。具体来说，我们从一个服从均值为零、方差为一的高斯分布的 $5 \times 5$ 矩阵开始。

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### 在随机数据上的行为
为简化我们的玩具模型，假设输入的数据向量 $\mathbf{v}_{in}$ 是一个随机的五维高斯向量。让我们思考一下希望发生什么。作为背景，考虑一个一般的机器学习问题：我们试图把输入数据（例如一张图像）变成预测（例如该图像是猫的概率）。如果反复应用 $\mathbf{A}$ 会把随机向量拉伸得很长，那么输入的微小变化会被放大为输出的巨大变化——对输入图像的细微改动就会导致截然不同的预测。这看起来并不合理！

反过来说，如果 $\mathbf{A}$ 把随机向量缩短，那么经过许多层之后，向量本质上会收缩为零，输出将不再依赖于输入。这显然同样不合理！

我们需要在增长与衰减之间走一条窄路，以确保输出会随输入而变化，但又变化不大！

我们来看看反复用矩阵 $\mathbf{A}$ 乘以一个随机输入向量，并跟踪其范数会发生什么。

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

范数在不受控制地增长！事实上，如果我们取相邻范数的商组成的列表，就会看到一个规律。

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

如果观察上述计算的最后一部分，我们会看到随机向量被拉伸了 `1.974459321485[...]` 倍，其中末尾的数字会略有变动，但拉伸倍数保持稳定。

### 回到特征向量

我们已经看到，特征向量和特征值对应于某物被拉伸的程度，但那是针对特定向量和特定拉伸而言的。我们来看看对于 $\mathbf{A}$ 而言它们是什么。这里需要提醒一点：结果表明，要看到全部特征值，我们需要进入复数领域。你可以把复数看作拉伸与旋转的结合。通过取复数的模（实部与虚部平方和的平方根），可以度量这种拉伸倍数。我们顺便把它们排序。

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### 一个观察

我们在这里看到了一点出乎意料的现象：之前我们确定的、矩阵 $\mathbf{A}$ 作用于随机向量时的长期拉伸倍数，*恰好*（精确到小数点后十三位！）就是 $\mathbf{A}$ 的最大特征值。这显然不是巧合！

但是，如果我们现在从几何角度来思考发生了什么，这就开始说得通了。考虑一个随机向量。这个随机向量在每个方向上都有分量，因此特别地，它至少在某种程度上指向与 $\mathbf{A}$ 最大特征值对应的特征向量相同的方向。这一点如此重要，以至于它被称为*主特征值*（principal eigenvalue）和*主特征向量*（principal eigenvector）。应用 $\mathbf{A}$ 之后，我们的随机向量在每个可能的方向上都被拉伸（每个可能的方向都与某个特征向量相关联），但它在与主特征向量对应的方向上被拉伸得最多。这意味着，应用 $A$ 之后，我们的随机向量变得更长，且其指向更接近与主特征向量对齐。多次应用该矩阵后，与主特征向量的对齐程度会越来越高，直到在实际意义上，我们的随机向量已经变成了主特征向量！事实上，这一算法正是所谓的*幂迭代*（power iteration）的基础，用于求矩阵的最大特征值和特征向量。细节可参见例如 :cite:`Golub.Van-Loan.1996`。

### 修正规范化

现在，从上面的讨论中我们得出结论：我们根本不想让随机向量被拉伸或挤压，而是希望随机向量在整个过程中保持大致相同的大小。为此，我们现在用这个主特征值重新缩放矩阵，使最大的特征值变为 1。我们来看看在这种情况下会发生什么。

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

我们也可以像之前那样画出相邻范数之间的比值，并看到它确实稳定了下来。

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 讨论

现在，我们看到了正是所期望的结果！在用主特征值对矩阵进行规范化之后，随机数据不再像之前那样爆炸，而是最终趋于某个特定的值。如果我们能够从第一性原理出发做到这些就好了，而结果表明，如果深入考察其中的数学，我们会发现：对于一个元素独立服从均值为零、方差为一的高斯分布的大随机矩阵，其最大特征值平均约为 $\sqrt{n}$，在我们的例子中就是 $\sqrt{5} \approx 2.2$，这源于一个被称为*圆律*（circular law）的迷人事实 :cite:`Ginibre.1965`。随机矩阵的特征值（以及一个称为奇异值的相关对象）之间的关系已被证明与神经网络的恰当初始化有着深刻的联系，这在 :citet:`Pennington.Schoenholz.Ganguli.2017` 及其后续工作中有所讨论。

## 小结
* 特征向量是被矩阵拉伸但不改变方向的向量。
* 特征值是应用矩阵后特征向量被拉伸的倍数。
* 矩阵的特征分解可以使许多运算简化为对特征值的运算。
* Gershgorin 圆盘定理可以为矩阵的特征值提供近似值。
* 迭代矩阵幂的行为主要取决于最大特征值的大小。这一理解在神经网络初始化理论中有许多应用。

## 练习
1. 求下面矩阵的特征值和特征向量
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1. 下面矩阵的特征值和特征向量是什么？与上一个例子相比，这个例子有什么奇怪之处？
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. 在不计算特征值的情况下，下面矩阵的最小特征值是否可能小于 $0.5$？*注意*：这道题可以心算完成。
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1087)
:end_tab:
