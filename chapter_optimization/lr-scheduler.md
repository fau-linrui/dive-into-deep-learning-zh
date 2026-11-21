# 学习率调度
:label:`sec_scheduler`

到目前为止，我们主要关注的是如何更新权重向量的优化*算法*，而不是更新权重向量的*速率*。然而，调整学习率往往与实际采用的算法同样重要。需要考虑以下几个方面的因素：

* 最明显的是学习率的*大小*至关重要。如果学习率太大，优化会发散；如果太小，训练会耗时太久，或者我们最终只能得到次优的结果。前面我们看到问题的条件数也有影响（详见 :numref:`sec_momentum`）。直观地说，它是最不敏感方向上的变化量与最敏感方向上的变化量之比。
* 其次，衰减速率同样重要。如果学习率一直保持较大，我们可能只是在最小值附近来回震荡，从而无法达到最优。:numref:`sec_minibatch_sgd` 对此作过一些详细讨论，我们也曾在 :numref:`sec_sgd` 中分析了性能保证。简而言之，我们希望学习率衰减，但可能要比 $\mathcal{O}(t^{-\frac{1}{2}})$ 更慢，后者对凸问题来说是一个不错的选择。
* 同样重要的另一个方面是*初始化*。这既涉及参数最初如何设置（详见 :numref:`sec_numerical_stability`），也涉及它们最初如何演化。这属于*预热*（warmup）的范畴，即我们一开始以多快的速度朝解的方向移动。一开始就迈出很大的步子可能并无益处，尤其是因为初始参数集是随机的。初始的更新方向也可能相当没有意义。
* 最后，还有一些执行循环式学习率调整的优化变体。这超出了本章的范围。我们建议读者查阅 :citet:`Izmailov.Podoprikhin.Garipov.ea.2018` 中的细节，例如如何通过对整条参数*路径*求平均来获得更好的解。

鉴于管理学习率需要大量细节，大多数深度学习框架都提供了自动处理这一问题的工具。本章我们将回顾不同调度方案对准确率的影响，并展示如何通过*学习率调度器*来高效地管理学习率。

## 玩具问题

我们从一个玩具问题开始，它计算起来足够便宜，但又足够不平凡，可以说明一些关键方面。为此，我们选取 LeNet 的一个略作现代化的版本（用 `relu` 激活而非 `sigmoid` 激活，用最大汇聚而非平均汇聚），并将其应用于 Fashion-MNIST。此外，我们还对网络进行混合式编程以提高性能。由于大部分代码都是标准的，我们只介绍基本内容，不再作详细讨论。如有需要，请参阅 :numref:`chap_cnn` 复习。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

让我们看看用默认设置调用该算法会发生什么，例如学习率为 $0.3$、训练 $30$ 个迭代。注意训练准确率持续上升，而测试准确率的进展则在某一点之后停滞。两条曲线之间的差距表明存在过拟合。

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## 调度器

调整学习率的一种方法是每一步都显式地设定它。用 `set_learning_rate` 方法可以方便地做到这一点。我们可以每轮之后（甚至每个小批量之后）把它调低，例如以动态方式响应优化的进展。

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

更一般地，我们希望定义一个调度器。当以更新次数为参数调用它时，它返回相应的学习率值。让我们定义一个简单的调度器，把学习率设为 $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$。

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

让我们在一定的取值范围内绘制它的行为。

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

现在我们来看看在 Fashion-MNIST 上训练时这个调度器表现如何。我们只需把调度器作为额外的参数传给训练算法。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

这比之前的做法好得多。有两点很突出：曲线比之前平滑得多；其次，过拟合也更少。遗憾的是，某些策略为何会在*理论上*导致更少的过拟合，目前还没有明确的定论。有一种说法是，较小的步长会使参数更接近零，因而更简单。然而，这并不能完全解释这一现象，因为我们其实并没有真正提前停止，而只是温和地降低了学习率。

## 策略

虽然我们不可能涵盖学习率调度器的全部种类，但下面我们尝试简要概述一些流行的策略。常见的选择有多项式衰减和分段常数调度。此外，余弦学习率调度在某些问题上已被经验证实效果良好。最后，在某些问题上，在使用大学习率之前先预热优化器是有益的。

### 因子调度器

多项式衰减的一种替代方案是乘法衰减，即 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$，其中 $\alpha \in (0, 1)$。为了防止学习率衰减到合理的下界以下，更新方程通常修改为 $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$。

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

这也可以通过 MXNet 中内置的调度器 `lr_scheduler.FactorScheduler` 对象来实现。它接受更多一些参数，例如预热期、预热模式（线性或常数）、期望的最大更新次数等；往后我们将酌情使用内置调度器，这里只解释它们的功能。如例所示，必要时构建自己的调度器是相当直截了当的。

### 多因子调度器

训练深度网络的一种常见策略是让学习率保持分段常数，并每隔一段时间将其减小一个给定的量。也就是说，给定一组需要降低学习率的时刻，例如 $s = \{5, 10, 20\}$，只要 $t \in s$ 就令 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$。假设每一步都减半，我们可以这样实现。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

这种分段常数学习率调度背后的直觉是：先让优化进行到权重向量的分布达到一个驻点，然后（也只有在这之后）再降低学习率，从而获得对良好局部最小值更高质量的近似。下面的例子展示了这如何逐渐产生略微更好的解。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 余弦调度器

:citet:`Loshchilov.Hutter.2016` 提出了一种相当令人困惑的启发式方法。它基于这样一个观察：我们可能不希望在开始时过快地把学习率降下来；此外，我们可能希望在最后用很小的学习率来“精炼”解。这导致了余弦式调度：在 $t \in [0, T]$ 范围内，学习率具有如下函数形式。

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


这里 $\eta_0$ 是初始学习率，$\eta_T$ 是在时刻 $T$ 的目标学习率。此外，对于 $t > T$，我们直接把取值固定为 $\eta_T$，不再重新增大。在下面的例子中，我们把最大更新步数设为 $T = 20$。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

在计算机视觉的语境下，这种调度*可以*带来更好的结果。不过请注意，这种改进并没有保证（如下所示）。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 预热

在某些情况下，初始化参数不足以保证得到好的解。这对一些可能导致优化不稳定的高级网络设计来说尤其成问题。我们可以通过选择足够小的学习率来避免一开始就发散，从而解决这个问题。但遗憾的是，这意味着进展缓慢。反之，一开始就用大的学习率又会导致发散。

对于这一两难困境，一个相当简单的解决办法是使用预热期：在此期间学习率*增大*到初始的最大值，然后在优化过程结束前逐渐冷却下来。为简单起见，通常采用线性增长。这就产生了如下所示的调度形式。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

注意网络在初始阶段收敛得更好（尤其要观察前 5 轮的表现）。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

预热可以应用于任何调度器（不只是余弦调度）。关于学习率调度更详细的讨论以及更多实验，另见 :cite:`Gotmare.Keskar.Xiong.ea.2018`。特别地，他们发现预热阶段限制了非常深的网络中参数的发散程度。这在直觉上是合理的：对于网络中最开始需要最长时间才能取得进展的那些部分，由于随机初始化，我们会预期出现显著的发散。

## 小结

* 在训练过程中降低学习率可以提高准确率，并且（最令人困惑的是）减少模型的过拟合。
* 每当进展停滞时就对学习率进行分段式降低，这在实践中是有效的。从本质上讲，这确保我们高效地收敛到一个合适的解，然后才通过降低学习率来减小参数固有的方差。
* 余弦调度器在某些计算机视觉问题上很受欢迎。此类调度器的细节可参见 [GluonCV](http://gluon-cv.mxnet.io)。
* 在优化前设置预热期可以防止发散。
* 优化在深度学习中有多种用途。除了最小化训练目标之外，不同的优化算法和学习率调度选择还会导致在测试集上相当不同的泛化程度和过拟合程度（即使训练误差相同）。

## 练习

1. 用给定的固定学习率实验优化的行为。这样你能得到的最佳模型是什么？
1. 如果改变学习率下降的指数，收敛会如何变化？为方便实验，可以使用 `PolyScheduler`。
1. 把余弦调度器应用于大型计算机视觉问题，例如训练 ImageNet。与其他调度器相比，它对性能有何影响？
1. 预热应该持续多久？
1. 你能把优化与采样联系起来吗？可以先使用 :citet:`Welling.Teh.2011` 关于随机梯度 Langevin 动力学的结果。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1081)
:end_tab:
