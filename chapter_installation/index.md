# 安装
:label:`chap_installation`

为了顺利开始，我们需要一个环境来运行Python、Jupyter 笔记本、相关库，以及运行本书自身所需的代码。

## 安装Miniconda

最简单的选择是安装[Miniconda](https://conda.io/en/latest/miniconda.html)。请注意，需要Python 3.x版本。如果你的机器已经安装了conda，可以跳过以下步骤。

访问Miniconda网站，根据你的Python 3.x版本和机器架构确定适合你系统的版本。假设你的Python版本是3.9（我们测试过的版本）。如果你使用的是macOS，就下载名称中包含字符串“MacOSX”的bash脚本，进入下载位置，并按如下方式执行安装（以Intel Mac为例）：

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux用户则下载名称中包含字符串“Linux”的文件，并在下载位置执行以下命令：

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows用户按照Miniconda的[在线说明](https://conda.io/en/latest/miniconda.html)下载并安装即可。在Windows上，你可以搜索`cmd`来打开命令提示符（命令行解释器）以运行命令。

接下来，初始化shell，以便我们可以直接运行`conda`。

```bash
~/miniconda3/bin/conda init
```


然后关闭并重新打开当前的shell。现在你应该能够按如下方式创建一个新环境：

```bash
conda create --name d2l python=3.9 -y
```


现在我们可以激活`d2l`环境：

```bash
conda activate d2l
```


## 安装深度学习框架和`d2l`包

在安装任何深度学习框架之前，请先检查你的机器上是否有合适的GPU（标准笔记本电脑上用于驱动显示器的GPU不在本书考虑之列）。例如，如果你的计算机装有NVIDIA GPU并且已安装[CUDA](https://developer.nvidia.com/cuda-downloads)，那就万事俱备了。如果你的机器没有任何GPU，也暂时不必担心。你的CPU所提供的算力足以支撑你读完前几章。只是要记住，在运行较大的模型之前，你会希望用上GPU。


:begin_tab:`mxnet`

要安装支持GPU的MXNet版本，我们需要查明你安装的CUDA版本。你可以通过运行`nvcc --version`或`cat /usr/local/cuda/version.txt`来检查。假设你已安装CUDA 11.2，那么执行以下命令：

```bash
# For macOS and Linux users
pip install mxnet-cu112==1.9.1

# For Windows users
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


你可以根据你的CUDA版本更改最后几位数字，例如CUDA 10.1对应`cu101`，CUDA 9.0对应`cu90`。


如果你的机器没有NVIDIA GPU或CUDA，可以按如下方式安装CPU版本：

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

你可以按如下方式安装支持CPU或GPU的PyTorch（指定的版本在撰写本书时已经过测试）：

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
你可以按如下方式安装支持CPU或GPU的TensorFlow：

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
你可以按如下方式安装支持CPU或GPU的JAX和Flax：

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


如果你的机器没有NVIDIA GPU或CUDA，可以按如下方式安装CPU版本：

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


下一步是安装我们开发的`d2l`包，它封装了全书中经常使用的函数和类：

```bash
pip install d2l==1.0.3
```


## 下载并运行代码

接下来，你需要下载笔记本，以便运行书中每一个代码块。只需点击[D2L.ai网站](https://d2l.ai/)上任意HTML页面顶部的“Notebooks”选项卡即可下载代码，然后解压。或者，你也可以按如下方式从命令行获取这些笔记本：

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:


:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

如果你还没有安装`unzip`，请先运行`sudo apt-get install unzip`。现在，我们可以通过运行以下命令来启动Jupyter 笔记本服务器：

```bash
jupyter notebook
```


此时，你可以在网页浏览器中打开http://localhost:8888（它可能已经自动打开）。然后，我们就可以运行书中每一节的代码了。每当新打开一个命令行窗口时，你都需要执行`conda activate d2l`来激活运行环境，然后才能运行D2L笔记本或更新你的包（无论是深度学习框架还是`d2l`包）。要退出该环境，请运行`conda deactivate`。


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17964)
:end_tab:
