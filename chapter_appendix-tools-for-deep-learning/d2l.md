# `d2l` API 文档
:label:`sec_d2l`

本节展示`d2l`包中的类和函数（按字母顺序排列），并标明它们在书中的定义位置，以便你找到更详细的实现和讲解。
另请参阅 [GitHub 仓库](https://github.com/d2l-ai/d2l-en/tree/master/d2l)上的源代码。

:begin_tab:`pytorch`

```eval_rst

.. currentmodule:: d2l.torch

```

:begin_tab:`mxnet`

```eval_rst

.. currentmodule:: d2l.mxnet

```

:end_tab:


:begin_tab:`tensorflow`

```eval_rst

.. currentmodule:: d2l.torch

```


:end_tab:

## 类

```eval_rst 

.. autoclass:: AdditiveAttention
   :members:
   
.. autoclass:: AddNorm
   :members:

.. autoclass:: AttentionDecoder
   :members: 

.. autoclass:: Classifier
   :members: 
   
.. autoclass:: DataModule
   :members: 
   
.. autoclass:: Decoder
   :members: 
   
.. autoclass:: DotProductAttention
   :members:
   
.. autoclass:: Encoder
   :members:
   
.. autoclass:: EncoderDecoder
   :members:
   
.. autoclass:: FashionMNIST
   :members: 
   
.. autoclass:: GRU
   :members: 
   
.. autoclass:: HyperParameters
   :members: 
   
.. autoclass:: LeNet
   :members: 
   
.. autoclass:: LinearRegression
   :members: 
   
.. autoclass:: LinearRegressionScratch
   :members: 
   
.. autoclass:: Module
   :members: 
   
.. autoclass:: MTFraEng
   :members: 
   
.. autoclass:: MultiHeadAttention
   :members:
   
.. autoclass:: PositionalEncoding
   :members:
   
.. autoclass:: PositionWiseFFN
   :members:
   
.. autoclass:: ProgressBoard
   :members: 
   
.. autoclass:: Residual
   :members: 
   
.. autoclass:: ResNeXtBlock
   :members:
   
.. autoclass:: RNN
   :members: 
   
.. autoclass:: RNNLM
   :members:
   
.. autoclass:: RNNLMScratch
   :members:
   
.. autoclass:: RNNScratch
   :members: 
   
.. autoclass:: Seq2Seq
   :members:  
   
.. autoclass:: Seq2SeqEncoder
   :members:
   
.. autoclass:: SGD
   :members: 
   
.. autoclass:: SoftmaxRegression
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: TimeMachine
   :members: 

.. autoclass:: Trainer
   :members: 

.. autoclass:: TransformerEncoder 
   :members:

.. autoclass:: TransformerEncoderBlock
   :members:

.. autoclass:: Vocab
   :members: 
```


## 函数

```eval_rst 

.. autofunction:: add_to_class

.. autofunction:: bleu

.. autofunction:: check_len

.. autofunction:: check_shape

.. autofunction:: corr2d

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: init_cnn

.. autofunction:: init_seq2seq

.. autofunction:: masked_softmax

.. autofunction:: num_gpus

.. autofunction:: plot

.. autofunction:: set_axes

.. autofunction:: set_figsize

.. autofunction:: show_heatmaps

.. autofunction:: show_list_len_pair_hist

.. autofunction:: try_all_gpus

.. autofunction:: try_gpu

.. autofunction:: use_svg_display

```

