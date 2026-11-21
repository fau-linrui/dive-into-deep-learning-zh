# 自动并行
:label:`sec_auto_para`


深度学习框架（例如 MXNet 和 PyTorch）会在后端自动构建计算图。借助计算图，系统知道所有的依赖关系，
并可以选择性地并行执行多个互不依赖的任务，以提高速度。例如，:numref:`sec_async` 中的 :numref:`fig_asyncgraph` 独立地初始化了两个变量。因此，系统可以选择并行执行它们。


通常，单个运算符会使用所有 CPU 上的全部计算资源，或单个 GPU 上的全部计算资源。例如，`dot` 运算符会使用所有 CPU 上的所有核心（和线程），即使一台机器上有多个 CPU 处理器也是如此。单个 GPU 也是如此。因此，对于单设备计算机来说，并行化并不那么有用。有多个设备时，情况就重要得多了。虽然并行化通常与多个 GPU 的关系最密切，但加入本地 CPU 也会略微提高性能。例如，参见 :citet:`Hadjis.Zhang.Mitliagkas.ea.2016`，该工作关注结合 GPU 和 CPU 来训练计算机视觉模型。借助自动并行框架的便利，我们可以用几行 Python 代码实现同样的目标。更广泛地说，我们对自动并行计算的讨论聚焦于同时使用 CPU 和 GPU 的并行计算，以及计算与通信的并行化。

请注意，运行本节的实验至少需要两个 GPU。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU 上的并行计算

让我们先定义一个用于测试的参考工作负载：下面的 `run` 函数在我们选定的设备上执行 10 次矩阵-矩阵乘法，所用数据分别分配在两个变量 `x_gpu1` 和 `x_gpu2` 中。

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
现在我们把该函数作用于数据。为了确保缓存不会影响结果，我们在测量之前先在任一设备上执行一遍，以此对设备进行预热。
:end_tab:

:begin_tab:`pytorch`
现在我们把该函数作用于数据。为了确保缓存不会影响结果，我们在测量之前先在任一设备上执行一遍，以此对设备进行预热。`torch.cuda.synchronize()` 会等待 CUDA 设备上所有流中的所有内核完成。它接受一个 `device` 参数，即我们需要同步的设备。如果设备参数为 `None`（默认值），则使用由 `current_device()` 给出的当前设备。
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
如果我们去掉两个任务之间的 `waitall` 语句，系统就可以自由地在两个设备上自动并行计算。
:end_tab:

:begin_tab:`pytorch`
如果我们去掉两个任务之间的 `synchronize` 语句，系统就可以自由地在两个设备上自动并行计算。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

在上面的情形中，总执行时间小于各部分时间之和，因为深度学习框架会自动在两个 GPU 设备上调度计算，用户无须编写复杂的代码。



## 并行计算与通信

在许多情况下，我们需要在不同设备之间移动数据，比如在 CPU 和 GPU 之间，或在不同的 GPU 之间。
例如，
当我们想要执行分布式优化、需要跨多张加速卡聚合梯度时，就会发生这种情况。让我们先在 GPU 上计算，然后把结果复制回 CPU，以此进行模拟。

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
这有点低效。请注意，我们其实可以在列表其余部分仍在计算的同时，就开始把 `y` 的一部分复制到 CPU。例如，当我们在一个小批量上计算梯度时，就会出现这种情况。某些参数的梯度会比其他参数的梯度更早可用。因此，在 GPU 仍在运行的同时就开始使用 PCI-Express 总线带宽对我们是有利的。去掉两部分之间的 `waitall` 就能模拟这种场景。
:end_tab:

:begin_tab:`pytorch`
这有点低效。请注意，我们其实可以在列表其余部分仍在计算的同时，就开始把 `y` 的一部分复制到 CPU。例如，当我们在一个小批量上计算（反向传播的）梯度时，就会出现这种情况。某些参数的梯度会比其他参数的梯度更早可用。因此，在 GPU 仍在运行的同时就开始使用 PCI-Express 总线带宽对我们是有利的。在 PyTorch 中，`to()` 和 `copy_()` 等几个函数接受一个显式的 `non_blocking` 参数，让调用者可以在不需要同步时绕过同步。设置 `non_blocking=True` 就能模拟这种场景。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

两项操作所需的总时间（如预期的那样）小于各部分时间之和。
请注意，这个任务与并行计算不同，因为它使用的是不同的资源：CPU 与 GPU 之间的总线。事实上，我们可以同时在两个设备上计算并进行通信。如上所述，计算与通信之间存在依赖关系：必须先计算出 `y[i]`，才能把它复制到 CPU。幸运的是，系统可以在计算 `y[i]` 的同时复制 `y[i-1]`，以减少总运行时间。

最后，我们用一个简单的两层 MLP 在 CPU 和两个 GPU 上训练时的情况，来说明计算图及其依赖关系，如图 :numref:`fig_twogpu` 所示。手动调度由此产生的并行程序会相当痛苦。这正是拥有基于图的计算后端来做优化的优势所在。

![CPU 和两个 GPU 上两层 MLP 的计算图及其依赖关系。](../img/twogpu.svg)
:label:`fig_twogpu`


## 小结

* 现代系统拥有各种各样的设备，例如多个 GPU 和 CPU。它们可以并行、异步地使用。
* 现代系统还拥有各种用于通信的资源，例如 PCI Express、存储（通常是固态硬盘或通过网络访问的存储）以及网络带宽。为了达到峰值效率，这些资源可以并行使用。
* 后端可以通过自动的并行计算与通信来提高性能。

## 练习

1. 本节定义的 `run` 函数中执行了八次操作。它们之间没有依赖关系。设计一个实验，看看深度学习框架是否会自动并行执行它们。
1. 当单个运算符的工作负载足够小时，即使在单个 CPU 或 GPU 上，并行化也能有所帮助。设计一个实验来验证这一点。
1. 设计一个实验，在 CPU、GPU 上使用并行计算，并在两个设备之间进行通信。
1. 使用 NVIDIA 的 [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) 之类的调试器，验证你的代码是高效的。
1. 设计包含更复杂数据依赖关系的计算任务，并通过实验看看能否在提高性能的同时得到正确的结果。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1681)
:end_tab:
