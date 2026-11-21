# 计算机视觉
:label:`chap_cv`

无论是医学诊断、自动驾驶汽车、摄像头监控，还是智能滤镜，计算机视觉领域的许多应用都与我们现在和未来的生活密切相关。
近年来，深度学习一直是推动计算机视觉系统性能提升的变革性力量。
可以说，最先进的计算机视觉应用几乎离不开深度学习。
鉴于此，本章将聚焦计算机视觉领域，探讨近年来在学术界和工业界颇具影响力的方法与应用。


在 :numref:`chap_cnn` 和 :numref:`chap_modern_cnn` 中，我们研究了计算机视觉中常用的各种卷积神经网络，并把它们应用于简单的图像分类任务。
本章开头，我们将介绍两种可能提高模型泛化能力的方法，即*图像增广*（image augmentation）和*微调*（fine-tuning），
并把它们应用于图像分类。
由于深度神经网络能够有效地在多个层级上表示图像，
这种逐层表示已被成功用于各种计算机视觉任务，例如*目标检测*（object detection）、*语义分割*（semantic segmentation）和*样式迁移*（style transfer）。
遵循在计算机视觉中利用逐层表示这一核心思想，
我们将从目标检测的主要组件和技术开始。接下来，我们将展示如何使用*全卷积网络*（fully convolutional network）对图像进行语义分割。然后我们将解释如何使用样式迁移技术生成类似本书封面的图像。
最后，我们通过把本章及此前若干章的材料应用到两个流行的计算机视觉基准数据集上，来结束本章。

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```
