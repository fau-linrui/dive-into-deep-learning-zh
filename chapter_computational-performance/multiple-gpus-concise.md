# 多 GPU 训练的简洁实现
:label:`sec_multi_gpu_concise`

为每个新模型都从头实现并行化毫无乐趣可言。此外，针对高性能优化同步工具也能带来可观的收益。下面我们将展示如何使用深度学习框架的高级 API 来完成这件事。
其中的数学原理和算法与 :numref:`sec_multi_gpu` 中相同。
毫不意外，运行本节代码至少需要两块 GPU。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**一个玩具网络**]

让我们使用一个比 :numref:`sec_multi_gpu` 中的 LeNet 稍有意义、但仍然足够容易和快速训练的网络。
我们选择一个 ResNet-18 变体 :cite:`He.Zhang.Ren.ea.2016`。由于输入图像很小，我们对其稍作修改。特别是，与 :numref:`sec_resnet` 相比，我们在开头使用了更小的卷积核、步幅和填充。
此外，我们删除了最大汇聚层。

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## 网络初始化

:begin_tab:`mxnet`
`initialize` 函数允许我们在自己选择的设备上初始化参数。
关于初始化方法的回顾见 :numref:`sec_numerical_stability`。特别方便的是，它还允许我们同时在*多个*设备上初始化网络。让我们在实践中试试它是如何工作的。
:end_tab:

:begin_tab:`pytorch`
我们将在训练循环内部初始化网络。
关于初始化方法的回顾见 :numref:`sec_numerical_stability`。
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
使用 :numref:`sec_multi_gpu` 中介绍的 `split_and_load` 函数，我们可以划分一个小批量数据，并把各部分复制到 `devices` 变量提供的设备列表中。网络实例会*自动*使用合适的 GPU 来计算前向传播的值。这里我们生成 4 个观测，并把它们拆分到各个 GPU 上。
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
一旦数据通过网络，相应的参数就会在*数据经过的设备上*初始化。
这意味着初始化是逐设备进行的。由于我们选择了 GPU 0 和 GPU 1 进行初始化，网络只在它们上面初始化，而不会在 CPU 上初始化。事实上，参数甚至根本不存在于 CPU 上。我们可以通过打印参数并观察可能出现的错误来验证这一点。
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
接下来，让我们把[**评估准确率**]的代码替换为可以(**在多个设备上并行**)工作的版本。它用来替代 :numref:`sec_lenet` 中的 `evaluate_accuracy_gpu` 函数。主要区别在于，我们在调用网络之前先拆分小批量。其他部分基本相同。
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**训练**]

和之前一样，训练代码需要完成几个基本功能，才能实现高效的并行：

* 网络参数需要在所有设备上初始化。
* 在遍历数据集时，小批量需要划分到所有设备上。
* 我们在各设备上并行计算损失及其梯度。
* 梯度被聚合，参数相应地更新。

最后，我们（同样并行地）计算准确率，以报告网络的最终性能。训练流程与此前章节中的实现非常相似，只是我们需要拆分和聚合数据。

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

让我们看看它在实践中效果如何。作为热身，我们[**在单个 GPU 上训练网络**]。

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

接下来我们[**使用 2 个 GPU 进行训练**]。与 :numref:`sec_multi_gpu` 中评估的 LeNet 相比，
ResNet-18 模型要复杂得多。这正是并行化发挥优势的地方。计算所需的时间明显大于同步参数所需的时间。由于并行化的开销变得不那么重要，这改善了可扩展性。

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 小结

:begin_tab:`mxnet`
* Gluon 通过提供上下文列表，为跨多个设备的模型初始化提供了原语。
:end_tab:
* 数据会自动在能够找到数据的设备上进行计算。
* 在尝试访问某个设备上的参数之前，请注意先在该设备上初始化网络。否则你会遇到错误。
* 优化算法会自动在多个 GPU 上聚合。



## 练习

:begin_tab:`mxnet`
1. 本节使用 ResNet-18。尝试不同的轮数、批量大小和学习率。使用更多 GPU 进行计算。如果用 16 个 GPU 尝试（例如在 AWS p2.16xlarge 实例上）会发生什么？
1. 有时，不同设备提供的算力不同。我们可以同时使用 GPU 和 CPU。应该如何划分工作？这样做值得吗？为什么？为什么不？
1. 如果我们去掉 `npx.waitall()` 会发生什么？你会如何修改训练，使并行时最多有两个步骤的重叠？
:end_tab:

:begin_tab:`pytorch`
1. 本节使用 ResNet-18。尝试不同的轮数、批量大小和学习率。使用更多 GPU 进行计算。如果用 16 个 GPU 尝试（例如在 AWS p2.16xlarge 实例上）会发生什么？
1. 有时，不同设备提供的算力不同。我们可以同时使用 GPU 和 CPU。应该如何划分工作？这样做值得吗？为什么？为什么不？
:end_tab:



:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1403)
:end_tab:
