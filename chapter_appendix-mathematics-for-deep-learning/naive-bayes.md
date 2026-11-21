# 朴素贝叶斯
:label:`sec_naive_bayes`

在前面几节中，我们学习了概率论与随机变量的理论。为了让这一理论发挥作用，让我们引入*朴素贝叶斯*（naive Bayes）分类器。它只使用概率论的基本原理，就能让我们对数字进行分类。

学习本质上就是在做假设。如果我们想对一个从未见过的新数据样例进行分类，就必须对哪些数据样例彼此相似作出一些假设。朴素贝叶斯分类器是一个流行且非常清晰的算法，它假设所有特征彼此独立，以简化计算。本节中，我们将把这个模型应用于图像中的字符识别。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 光学字符识别

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` 是广泛使用的数据集之一。它包含 60,000 张训练图像和 10,000 张验证图像。每张图像都包含一个从 0 到 9 的手写数字。任务是把每张图像分类到相应的数字。

Gluon在`data.vision`模块中提供了一个`MNIST`类，
用于从互联网自动获取数据集。
随后，Gluon将使用已下载的本地副本。
我们通过将参数`train`的值分别设置为`True`或`False`，
来指定请求的是训练集还是测试集。
每张图像都是宽高均为$28$的灰度图像，形状为（$28$,$28$,$1$）。我们使用自定义变换来去除最后一个通道维度。此外，数据集用无符号$8$位整数表示每个像素。为简化问题，我们将其量化为二值特征。

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

我们可以访问某个特定的样例，其中包含图像及对应的标签。

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

我们的样例存放在变量`image`中，它对应一张高和宽均为$28$像素的图像。

```{.python .input}
#@tab all
image.shape, image.dtype
```

我们的代码把每张图像的标签存储为一个标量。其类型是$32$位整数。

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

我们也可以同时访问多个样例。

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

让我们把这些样例可视化。

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 分类的概率模型

在分类任务中，我们把一个样例映射到一个类别。这里的样例是一张灰度$28\times 28$图像，类别是一个数字。（更详细的解释参见 :numref:`sec_softmax`。）
表达分类任务的一种自然方式是提出概率问题：给定特征（即图像像素）时，最可能的标签是什么？记$\mathbf x\in\mathbb R^d$为样例的特征，$y\in\mathbb R$为标签。这里的特征是图像像素，我们可以把$2$维图像重塑为向量，使得$d=28^2=784$，而标签是数字。
给定特征时标签的概率为$p(y  \mid  \mathbf{x})$。如果我们能计算出这些概率（在我们的例子中即$y=0, \ldots,9$时的$p(y  \mid  \mathbf{x})$），那么分类器将输出由下式给出的预测$\hat{y}$：

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

遗憾的是，这要求我们对每个$\mathbf{x} = x_1, ..., x_d$的取值都估计$p(y  \mid  \mathbf{x})$。想象每个特征都可以取$2$个值之一。例如，特征$x_1 = 1$可能表示某个给定文档中出现了单词 apple，而$x_1 = 0$则表示没有出现。如果我们有$30$个这样的二值特征，那就意味着我们需要准备好对输入向量$\mathbf{x}$的$2^{30}$（超过 10 亿！）种可能取值中的任意一种进行分类。

此外，学习在哪里呢？如果我们需要看到每一个可能的样例才能预测相应的标签，那我们就不是在真正学习某种模式，而只是在记住数据集。

## 朴素贝叶斯分类器

幸运的是，通过对条件独立性作出一些假设，我们可以引入某种归纳偏置（inductive bias），并构建一个能够从相对适量的训练样例中泛化的模型。首先，让我们用贝叶斯定理把分类器表达为

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

注意分母是规范化项$p(\mathbf{x})$，它不依赖于标签$y$的取值。因此，我们只需要比较不同$y$取值下的分子。即使计算分母变得难以处理，只要我们能计算分子，就可以忽略它。幸运的是，即使我们想恢复规范化常数，也做得到。既然$\sum_y p(y  \mid  \mathbf{x}) = 1$，我们总能恢复规范化项。

现在，让我们关注$p( \mathbf{x}  \mid  y)$。利用概率的链式法则，我们可以把$p( \mathbf{x}  \mid  y)$这一项表示为

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

这个表达式本身并不能让我们更进一步。我们仍然必须估计大约$2^d$个参数。然而，如果我们假设*在给定标签的条件下，各特征彼此条件独立*，那么情况突然就好多了，因为这一项简化为$\prod_i p(x_i  \mid  y)$，从而得到预测器

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

如果我们能对每个$i$和$y$估计$p(x_i=1  \mid  y)$，并把它的值保存在$P_{xy}[i, y]$中（这里$P_{xy}$是一个$d\times n$矩阵，$n$为类别数，$y\in\{1, \ldots, n\}$），那么我们也可以用这个值来估计$p(x_i = 0 \mid y)$，即

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

此外，我们对每个$y$估计$p(y)$，并把它保存在$P_y[y]$中，其中$P_y$是一个长度为$n$的向量。于是，对于任何新样例$\mathbf t = (t_1, t_2, \ldots, t_d)$，我们可以计算

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

对任意$y$成立。因此，条件独立性假设把模型的复杂度从随特征数呈指数依赖$\mathcal{O}(2^dn)$降为线性依赖，即$\mathcal{O}(dn)$。

## 训练

现在的问题是我们并不知道$P_{xy}$和$P_y$。因此，我们首先需要根据一些训练数据来估计它们的值。这就是*训练*模型。估计$P_y$并不太难。由于我们只处理$10$个类别，我们可以统计每个数字的出现次数$n_y$，并用它除以数据总量$n$。例如，如果数字 8 出现了$n_8 = 5,800$次，而我们的图像总数为$n = 60,000$，那么概率估计为$p(y=8) = 0.0967$。

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

现在来处理稍难一些的$P_{xy}$。由于我们选取的是黑白图像，$p(x_i  \mid  y)$表示对类别$y$而言像素$i$被点亮（黑色）的概率。和之前一样，我们可以去统计事件发生的次数$n_{iy}$，并用它除以$y$出现的总次数，即$n_y$。但有一点小麻烦：某些像素可能永远不会是黑色（例如，对于裁剪良好的图像，角落像素可能始终是白色）。统计学家处理这个问题的一种简便方法是给所有计数加上伪计数。因此，我们不使用$n_{iy}$，而使用$n_{iy}+1$；不使用$n_y$，而使用$n_{y}+2$（因为像素$i$可以取两个可能值——它要么是黑色，要么是白色）。这也称为*拉普拉斯平滑*（Laplace smoothing）。它看起来可能有些临时凑合，不过可以从贝叶斯角度用 Beta-二项模型来为它提供依据。

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

通过可视化这些$10\times 28\times 28$个概率（每个类别对应每个像素），我们可以得到一些看起来像“平均”数字的图像。

现在我们可以使用 :eqref:`eq_naive_bayes_estimation` 来预测一张新图像。给定$\mathbf x$，下面的函数对每个$y$计算$p(\mathbf x \mid y)p(y)$。

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

这下可彻底出错了！为了弄清原因，让我们看看逐像素的概率。它们通常是介于$0.001$和$1$之间的数。而我们要把$784$个这样的数相乘。此时值得一提的是，我们是在计算机上计算这些数，因而指数有固定范围。结果就是我们会遇到*数值下溢*（numerical underflow），即把所有小数相乘会得到更小的数，直到它被舍入为零。我们在 :numref:`sec_maximum_likelihood` 中曾把它作为一个理论问题讨论过，但在这里我们清楚地看到了这一现象在实践中的表现。

正如那一节所讨论的，我们利用$\log a b = \log a + \log b$这一事实来解决这个问题，即改为对对数求和。
即使$a$和$b$都是很小的数，对数值也应当处于一个合适的范围内。

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

由于对数函数是递增的，我们可以把 :eqref:`eq_naive_bayes_estimation` 重写为

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

我们可以实现如下稳定版本：

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

现在我们可以检查预测是否正确。

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

如果我们现在预测几张验证图像，就可以看到贝叶斯分类器工作得相当不错。

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

最后，让我们计算分类器的整体准确率。

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

现代深度网络能达到低于$0.01$的错误率。这里相对较差的性能源于我们在模型中所作的错误统计假设：我们假设每一个像素都是*独立*生成的，只取决于标签。这显然不是人类书写数字的方式，而这个错误假设导致我们这个过于朴素的（贝叶斯）分类器败下阵来。

## 小结
* 利用贝叶斯定理，可以通过假设所有观测特征相互独立来构造一个分类器。
* 该分类器可以通过统计标签与像素值组合的出现次数在数据集上训练。
* 数十年来，该分类器在垃圾邮件检测等任务上一直是黄金标准。

## 练习
1. 考虑数据集$[[0,0], [0,1], [1,0], [1,1]]$，其标签由两个元素的异或给出，即$[0,1,1,0]$。在该数据集上构建的朴素贝叶斯分类器，各概率是多少？它能成功分类我们的点吗？如果不能，是哪些假设被违反了？
1. 假设我们在估计概率时没有使用拉普拉斯平滑，而在测试时到来一个数据样例，它含有训练中从未观测到的取值。模型会输出什么？
1. 朴素贝叶斯分类器是贝叶斯网络的一个具体例子，其中随机变量之间的依赖关系用图结构来编码。完整的理论超出了本节的讨论范围（详见 :citet:`Koller.Friedman.2009`），请解释：在异或模型中允许两个输入变量之间存在显式依赖，为什么就能构造出一个成功的分类器？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1101)
:end_tab:
