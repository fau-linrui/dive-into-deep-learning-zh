# 图像增广
:label:`sec_image_augmentation`

在 :numref:`sec_alexnet` 中，
我们提到，大型数据集
是深度神经网络
在各种应用中取得成功的
先决条件。
*图像增广*（image augmentation）
在对训练图像进行一系列随机变化之后，
生成相似但不同的训练样例，从而扩大训练集的大小。
或者，
图像增广的动机也可以来自这样一个事实：
对训练样例进行随机扰动
可以让模型减少对某些属性的依赖，
从而提高其泛化能力。
例如，我们可以用不同的方式裁剪图像，使感兴趣的物体出现在不同的位置，从而降低模型对物体位置的依赖。
我们还可以调整亮度和颜色等因素，以降低模型对颜色的敏感度。
当时 AlexNet 的成功很可能离不开图像增广。
本节我们将讨论计算机视觉中这一被广泛使用的技术。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## 常用的图像增广方法

在考察常用的图像增广方法时，我们将以下面的 $400\times 500$ 图像为例。

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

大多数图像增广方法都具有一定的随机性。为了便于我们观察图像增广的效果，下面我们定义一个辅助函数 `apply`。该函数在输入图像 `img` 上多次运行图像增广方法 `aug`，并展示所有结果。

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### 翻转与裁剪

:begin_tab:`mxnet`
[**左右翻转图像**]通常不会改变物体的类别。
这是最早、也是使用最广泛的图像增广方法之一。
接下来，我们使用 `transforms` 模块创建 `RandomFlipLeftRight` 实例，它以 50% 的概率
对图像进行左右翻转。
:end_tab:

:begin_tab:`pytorch`
[**左右翻转图像**]通常不会改变物体的类别。
这是最早、也是使用最广泛的图像增广方法之一。
接下来，我们使用 `transforms` 模块创建 `RandomHorizontalFlip` 实例，它以 50% 的概率
对图像进行左右翻转。
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**上下翻转图像**]不如左右翻转常见。但至少对于这个示例图像，上下翻转不会妨碍识别。
接下来，我们创建 `RandomFlipTopBottom` 实例，以 50% 的概率
对图像进行上下翻转。
:end_tab:

:begin_tab:`pytorch`
[**上下翻转图像**]不如左右翻转常见。但至少对于这个示例图像，上下翻转不会妨碍识别。
接下来，我们创建 `RandomVerticalFlip` 实例，以 50% 的概率
对图像进行上下翻转。
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

在我们使用的示例图像中，猫位于图像中央，但一般情况下可能并非如此。
在 :numref:`sec_pooling` 中，我们解释了汇聚层可以降低卷积层对目标位置的敏感度。
此外，我们还可以随机裁剪图像，使物体以不同的尺度出现在图像中的不同位置，这同样可以降低模型对目标位置的敏感度。

在下面的代码中，我们[**随机裁剪**]一个面积为原始面积 $10\% \sim 100\%$ 的区域，且该区域的宽高比从 $0.5 \sim 2$ 中随机选取。然后，该区域的宽度和高度都被缩放到 200 像素。
除非另有说明，本节中 $a$ 和 $b$ 之间的随机数是指从区间 $[a, b]$ 中随机均匀采样得到的连续值。

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 改变颜色

另一种增广方法是改变颜色。我们可以改变图像颜色的四个方面：亮度、对比度、饱和度和色调。在下面的例子中，我们[**随机改变图像的亮度**]，将其变为原始图像的 50%（$1-0.5$）到 150%（$1+0.5$）之间的值。

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

同样，我们可以[**随机改变图像的色调**]。

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

我们还可以创建 `RandomColorJitter` 实例，并设置如何[**同时随机改变图像的 `brightness`、`contrast`、`saturation` 和 `hue`**]。

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 组合多种图像增广方法

在实践中，我们会[**组合多种图像增广方法**]。
例如，
我们可以把上面定义的不同图像增广方法组合起来，并通过 `Compose` 实例将它们应用到每张图像上。

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**使用图像增广进行训练**]

让我们使用图像增广来训练一个模型。
这里我们使用 CIFAR-10 数据集，而不是之前使用的 Fashion-MNIST 数据集。
这是因为 Fashion-MNIST 数据集中物体的位置和尺寸已经被规范化过，而 CIFAR-10 数据集中物体的颜色和大小差异更为显著。
CIFAR-10 数据集中的前 32 张训练图像如下所示。

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

为了在预测时得到确定的结果，我们通常只对训练样例应用图像增广，而在预测时不使用带有随机操作的图像增广。
[**这里我们只使用最简单的随机左右翻转方法**]。
此外，我们使用 `ToTensor` 实例把一个小批量的图像转换为深度学习框架所需的格式，即形状为（批量大小，通道数，高度，宽度）、取值在 0 到 1 之间的 32 位浮点数。

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
接下来，我们定义一个辅助函数，以便于读取图像并应用图像增广。
Gluon 的 datasets 提供的 `transform_first` 函数会把图像增广应用到每个训练样例的第一个元素（图像和标签），即图像上。
关于 `DataLoader` 的详细介绍，请参阅 :numref:`sec_fashion_mnist`。
:end_tab:

:begin_tab:`pytorch`
接下来，我们[**定义一个辅助函数，以便于读取图像并应用图像增广**]。
PyTorch 的 dataset 提供的 `transform` 参数会应用增广来变换图像。
关于 `DataLoader` 的详细介绍，请参阅 :numref:`sec_fashion_mnist`。
:end_tab:

```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### 多 GPU 训练

我们在 CIFAR-10 数据集上训练 :numref:`sec_resnet` 中的 ResNet-18 模型。
回想一下 :numref:`sec_multi_gpu_concise` 中对多 GPU 训练的介绍。
下面，[**我们定义一个使用多个 GPU 训练和评估模型的函数**]。

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

现在我们可以[**定义 `train_with_data_aug` 函数来使用图像增广训练模型**]。
该函数获取所有可用的 GPU，使用 Adam 作为优化算法，对训练数据集应用图像增广，最后调用刚刚定义的 `train_ch13` 函数来训练和评估模型。

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

让我们[**训练模型**]，训练时基于随机左右翻转进行图像增广。

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## 小结

* 图像增广基于现有训练数据生成随机图像，以提高模型的泛化能力。
* 为了在预测时得到确定的结果，我们通常只对训练样例应用图像增广，而在预测时不使用带有随机操作的图像增广。
* 深度学习框架提供了许多不同的图像增广方法，这些方法可以同时应用。


## 练习

1. 不使用图像增广训练模型：`train_with_data_aug(test_augs, test_augs)`。比较使用和不使用图像增广时的训练准确率和测试准确率。这个对比实验能否支持“图像增广可以缓解过拟合”这一论点？为什么？
1. 在 CIFAR-10 数据集上的模型训练中组合多种不同的图像增广方法。它能提高测试准确率吗？
1. 查阅深度学习框架的在线文档。它还提供了哪些其他图像增广方法？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1404)
:end_tab:
