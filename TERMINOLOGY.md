# 《动手学深度学习》中英术语对照表

本表是本次中文全译本（底本：英文版 `d2l-ai/d2l-en` 提交 `23d7a5aecceee57d1292c56e90cce307f183bb0a`，release 1.0.3）的**全书统一译名依据**。

使用约定：

1. 重要术语首次引入时写作"中文译名（English original）"，例如"**暂退法**（dropout）"；其后只用中文译名。
2. 表中"备注"列说明上下文限定、易混概念或保留英文的情形。
3. 框架名、类名、函数名、参数名、API 标识、模型/数据集专名一律保留英文，不进入本表（见文末"保留原文的专名"一节）。
4. 本表在官方中文版《动手学深度学习》术语表（仅作译名核对）基础上，按本次英文底本的实际用词扩充、校正；两者不一致时以本表为准，并在译本中保持全表一致。
5. 译者如遇表中未收录术语，按中文深度学习文献通行译法处理，并反馈合入本表。

## 一、核心概念（A–L）

| 英文 | 中文 | 备注 |
|---|---|---|
| accuracy | 准确率 | 分类任务正确率；与 precision（查准率）区分 |
| action | 动作 | 强化学习 |
| activation function | 激活函数 | |
| AdaGrad | AdaGrad 算法 | 算法名保留 |
| Adam | Adam 算法 | 算法名保留 |
| affine transformation | 仿射变换 | |
| agent | 智能体 | 强化学习 |
| anchor | 锚框 | 目标检测 |
| approximation | 近似 | |
| architecture | 架构 | 网络结构 |
| attention | 注意力 | 机制名译"注意力" |
| attention cue | 注意力提示 | |
| attention mechanism | 注意力机制 | |
| attention model | 注意力模型 | |
| attention pooling | 注意力汇聚 | |
| attention scoring function | 注意力评分函数 | |
| autoencoder | 自编码器 | |
| autograd | 自动微分 | 也可指 `autograd` 包，包名保留代码格式 |
| autoregressive model | 自回归模型 | |
| axis | 轴 | |
| backpropagation | 反向传播 | 缩写 BP |
| backward | 反向 | 如 backward 函数 |
| bag-of-words | 词袋 | |
| baseline | 基准 | |
| batch | 批量 | 一批数据；`batch size` 批量大小 |
| batch normalization | 批量规范化 | 通称"批量归一化"；全书统一用"批量规范化" |
| beam search | 束搜索 | |
| benchmark | 基准测试 | |
| bias | 偏置 | 模型参数；统计语境（bias-variance）译"偏差" |
| bigram | 二元语法 | 也作"二元"；全书统一"二元语法" |
| binary classification | 二元分类 | |
| block | 块 | 网络模块（d2l 的 `Block` 类名保留代码格式） |
| bounding box | 边界框 | |
| broadcast | 广播 | |
| candidate | 候选 | 如候选框、候选词 |
| channel | 通道 | 图像的通道 |
| class | 类 | 分类类别；代码中的 class 保留 |
| classification | 分类 | |
| classifier | 分类器 | |
| click-through rate (CTR) | 点击率 | 推荐系统 |
| collaborative filtering | 协同过滤 | |
| concatenate | 连结 | 官方译法；动词可用"连结" |
| conditional probability | 条件概率 | |
| context variable | 上下文变量 | RNN 编码器输出 |
| context window | 上下文窗口 | |
| context word | 上下文词 | |
| continuous bag-of-words (CBOW) | 连续词袋 | |
| convergence | 收敛 | 反义 diverge 发散 |
| convex | 凸 | 凸集、凸函数、凸优化 |
| convex optimization | 凸优化 | |
| convolution | 卷积 | |
| convolutional kernel | 卷积核 | |
| convolutional layer | 卷积层 | |
| convolutional neural network | 卷积神经网络 | CNN |
| corpus | 语料库 | |
| cost | 成本 | 最优化语境；`cost function` 成本函数 |
| covariance | 协方差 | |
| cross-correlation | 互相关 | 深度学习中的"卷积"实为互相关，需点明 |
| cross-entropy | 交叉熵 | |
| cumulative distribution function | 累积分布函数 | CDF |
| curse of dimensionality | 维数灾难 | |
| custom layer | 自定义层 | |
| data augmentation | 数据增广 | |
| data example | 数据样例 | 与 data instance 区分 |
| data instance | 数据实例 | 官方旧版作"数据示例"；本译本统一"数据实例" |
| data point | 数据点 | |
| dataset | 数据集 | |
| decision boundary | 决策边界 | |
| decoder | 解码器 | |
| deep learning | 深度学习 | |
| dense | 稠密 | 反义 sparse 稀疏 |
| derivative | 导数 | |
| deterministic | 确定性的 | |
| differentiable | 可微的 | |
| discriminator | 判别器 | GAN |
| distribution | 分布 | |
| distribution shift | 分布偏移 | |
| divergence | 散度 | KL divergence 等 |
| diverge | 发散 | |
| dropout | 暂退法 | 官方译法；不用"丢弃法" |
| dynamical programming | 动态规划 | 强化学习 |
| early stopping | 早停 | |
| eigendecomposition | 特征分解 | |
| eigenvalue | 特征值 | |
| eigenvector | 特征向量 | |
| element-wise | 按元素 | 逐元素运算 |
| embedding | 嵌入 | |
| empirical risk minimization | 经验风险最小化 | |
| encoder | 编码器 | |
| end-to-end | 端到端 | |
| ensemble | 集成 | |
| entropy | 熵 | |
| environment | 环境 | 强化学习 |
| epoch | 轮 | 如"训练 5 轮" |
| error | 误差 | |
| estimator | 估计量 | |
| expectation | 期望 | |
| explicit feedback | 显式反馈 | 推荐系统 |
| exponential family | 指数族 | 分布族 |
| expressive power | 表达能力 | |
| factorization machine | 因子分解机 | |
| feature | 特征 | |
| feature map | 特征图 | |
| filter | 滤波器 | 卷积核的同义语 |
| fine-tuning | 微调 | |
| fully connected layer | 全连接层 | |
| function | 函数 | 全书不用"方法"指函数 |
| generalization | 泛化 | |
| generalization error | 泛化误差 | |
| generative adversarial network | 生成对抗网络 | GAN |
| generative model | 生成模型 | |
| generator | 生成器 | GAN |
| gradient | 梯度 | |
| gradient clipping | 梯度裁剪 | |
| gradient descent | 梯度下降 | |
| gradient exploding | 梯度爆炸 | |
| gradient vanishing | 梯度消失 | |
| Gram matrix | 格拉姆矩阵 | |
| greedy search | 贪心搜索 | |
| hidden layer | 隐藏层 | |
| hidden state | 隐藏状态 | |
| hidden variable | 隐藏变量 | |
| hierarchical softmax | 层序 softmax | 官方译法 |
| hyperparameter | 超参数 | 与 parameter 区分 |
| hyperparameter optimization | 超参数优化 | |
| hypothesis | 假设 | 统计/学习理论语境；假设空间 hypothesis space |
| identity mapping | 恒等映射 | |
| i.i.d. (independent and identically distributed) | 独立同分布 | |
| image classification | 图像分类 | |
| implicit feedback | 隐式反馈 | 推荐系统 |
| inference | 推理 | 也译"预测"；模型推理 |
| information theory | 信息论 | |
| initialization | 初始化 | |
| inner product | 内积 | |
| instance | 实例 | |
| integral | 积分 | |
| intersection over union (IoU) | 交并比 | 目标检测 |
| iteration | 迭代 | |
| kernel | 核 | 卷积核；高斯过程语境"核函数" |
| kernel regression | 核回归 | |
| KL divergence | KL 散度 | |
| label | 标签 | |
| Laplace smoothing | 拉普拉斯平滑 | |
| large language model | 大语言模型 | |
| latent variable | 潜在变量 | |
| layer | 层 | |
| layer normalization | 层规范化 | |
| lazy initialization | 延迟初始化 | |
| learning rate | 学习率 | |
| learning rate scheduler | 学习率调度器 | |
| least squares | 最小二乘 | |
| likelihood | 似然 | |
| linear model | 线性模型 | |
| linear regression | 线性回归 | |
| linear transformation | 线性变换 | |
| local minimum | 局部最小值 | |
| logistic regression | 逻辑回归 | 如正文出现 |
| long short-term memory (LSTM) | 长短期记忆网络 | |
| loss | 损失 | |
| loss function | 损失函数 | |

## 二、核心概念（M–Z）

| 英文 | 中文 | 备注 |
|---|---|---|
| machine learning | 机器学习 | |
| machine translation | 机器翻译 | |
| margin | 间隔 | 如分类间隔 |
| marginal likelihood | 边际似然 | 高斯过程 |
| Markov chain | 马尔可夫链 | |
| Markov decision process (MDP) | 马尔可夫决策过程 | 强化学习 |
| matrix | 矩阵 | |
| matrix factorization | 矩阵分解 | 推荐系统 |
| maximum likelihood estimation | 最大似然估计 | |
| mean | 均值 | |
| mean squared error | 均方误差 | MSE |
| metric | 指标 | 评价指标 |
| minibatch | 小批量 | `minibatch SGD` 小批量随机梯度下降 |
| model | 模型 | |
| model construction | 模型构造 | |
| momentum | 动量 | 动量法 |
| multi-head attention | 多头注意力 | |
| multilayer perceptron | 多层感知机 | MLP |
| multivariate calculus | 多元微积分 | |
| mutual information | 互信息 | |
| naive Bayes | 朴素贝叶斯 | |
| natural language inference | 自然语言推断 | |
| negative sampling | 负采样 | |
| neural network | 神经网络 | |
| neural style transfer | 神经样式迁移 | |
| neuron | 神经元 | |
| next sentence prediction | 下一句预测 | BERT |
| node | 节点 | 计算图/神经网络 |
| notebook | 笔记本 | Jupyter notebook 译"Jupyter 笔记本"；不用"记事本" |
| noise | 噪声 | |
| non-maximum suppression | 非极大值抑制 | NMS |
| norm | 范数 | |
| normalization | 规范化 | 全书统一；动词 normalize 规范化 |
| objective function | 目标函数 | |
| offset | 偏移量 | 与 bias 互称时用"偏移量" |
| one-hot encoding | 独热编码 | |
| optimization | 优化 | |
| optimizer | 优化器 | |
| orthogonal | 正交 | |
| out-of-sample | 样本外 | |
| output layer | 输出层 | |
| overfitting | 过拟合 | |
| padding | 填充 | 卷积/序列 |
| parameter | 参数 | 与超参数区分 |
| parameter server | 参数服务器 | |
| partial derivative | 偏导数 | |
| perceptron | 感知机 | |
| perplexity | 困惑度 | |
| policy | 策略 | 强化学习 |
| pooling layer | 汇聚层 | 官方译法；不用"池化层" |
| positional encoding | 位置编码 | |
| posterior | 后验 | |
| precision | 查准率 | 与 accuracy（准确率）区分 |
| prediction | 预测 | |
| pretraining | 预训练 | |
| prior | 先验 | |
| probabilistic model | 概率模型 | |
| probability | 概率 | |
| probability density function | 概率密度函数 | PDF |
| property | 属性 | |
| Q-learning | Q 学习 | 官方译法；不写"Q学习"以外的变体 |
| query | 查询 | 注意力机制；数据库语境亦为"查询" |
| random sampling | 随机采样 | |
| random variable | 随机变量 | |
| ranking | 排序 | 推荐系统 |
| recall | 查全率 | 指标 |
| receptive field | 感受野 | |
| recommendation | 推荐 | |
| recommender system | 推荐系统 | |
| rectified linear unit (ReLU) | 修正线性单元 | ReLU 保留 |
| recurrent neural network (RNN) | 循环神经网络 | |
| regression | 回归 | |
| regularization | 正则化 | |
| reinforcement learning | 强化学习 | |
| residual connection | 残差连接 | |
| residual network | 残差网络 | ResNet 保留 |
| reward | 奖励 | 强化学习 |
| sample | 样本 | 动词 sample 译"采样" |
| sampling | 采样 | |
| self-attention | 自注意力 | |
| semantic segmentation | 语义分割 | |
| sentiment analysis | 情感分析 | |
| sequence | 序列 | |
| sequence model | 序列模型 | |
| sequence-to-sequence | 序列到序列 | seq2seq |
| sequential partitioning | 顺序分区 | |
| skip-gram model | 跳元模型 | 官方译法 |
| softmax | softmax | 函数名保留；`softmax regression` softmax 回归 |
| sparse | 稀疏 | |
| squared loss | 平方损失 | |
| state | 状态 | |
| statistical learning theory | 统计学习理论 | |
| step size | 步长 | |
| stochastic gradient descent (SGD) | 随机梯度下降 | |
| stride | 步幅 | |
| subsample | 子采样 | |
| supervised learning | 监督学习 | |
| support vector machine | 支持向量机 | |
| symbolic programming | 符号式编程 | 混合式编程语境 |
| target | 目标值 | 也作"目标"；与 label 对应时"目标值" |
| teacher forcing | 教师强制 | 序列生成训练 |
| tensor | 张量 | |
| test dataset | 测试数据集 | |
| threshold | 阈值 | |
| time step | 时间步 | |
| token | 词元 | 官方译法；不用"标记" |
| tokenization | 词元化 | |
| tokenizer | 词元分析器 | |
| training dataset | 训练数据集 | |
| training error | 训练误差 | |
| transfer learning | 迁移学习 | |
| transform | 变换 | |
| Transformer | Transformer | 模型名保留原文 |
| transposed convolution | 转置卷积 | 反卷积 |
| trigram | 三元语法 | |
| tuning | 调参 | `tune hyperparameter` 调参 |
| underfitting | 欠拟合 | |
| unigram | 一元语法 | |
| unbiased estimate | 无偏估计 | |
| uniform sampling | 均匀采样 | |
| unknown token | 未知词元 | |
| unsupervised learning | 无监督学习 | |
| update | 更新 | 参数更新 |
| upsampling | 上采样 | |
| validation dataset | 验证数据集 | |
| value | 值 | 注意力机制/字典语境；价值观语境按上下文 |
| variance | 方差 | bias-variance 偏差-方差 |
| vector | 向量 | |
| visualization | 可视化 | |
| weight | 权重 | |
| word embedding | 词嵌入 | |
| word vector | 词向量 | |
| word2vec | word2vec | 模型名保留 |
| vocabulary | 词表 | |
| z-score | z 分数 | 标准化 |

## 三、数学与统计（附录常用）

| 英文 | 中文 | 备注 |
|---|---|---|
| Bayes' theorem | 贝叶斯定理 | |
| Bernoulli distribution | 伯努利分布 | |
| binomial distribution | 二项分布 | |
| central limit theorem | 中心极限定理 | |
| chain rule | 链式法则 | |
| Cholesky decomposition | Cholesky 分解 | |
| confidence interval | 置信区间 | |
| constraint | 约束 | |
| convex set | 凸集 | |
| correlation | 相关性 | |
| covariance function | 协方差函数 | 高斯过程 |
| dual problem | 对偶问题 | |
| empirical distribution | 经验分布 | |
| expectation | 期望 | |
| exponential distribution | 指数分布 | |
| Gaussian distribution | 高斯分布 | 正态分布 |
| Gaussian process | 高斯过程 | |
| Hessian matrix | 黑塞矩阵 | |
| hypothesis test | 假设检验 | |
| independence | 独立性 | |
| inequality | 不等式 | |
| integral calculus | 积分学 | |
| Jacobian matrix | 雅可比矩阵 | |
| Lagrange multiplier | 拉格朗日乘子 | |
| limit | 极限 | |
| marginal probability | 边缘概率 | |
| maximum a posteriori | 最大后验 | |
| Monte Carlo | 蒙特卡洛 | |
| multinomial distribution | 多项分布 | |
| negative log-likelihood | 负对数似然 | |
| orthogonal matrix | 正交矩阵 | |
| p-value | p 值 | |
| Poisson distribution | 泊松分布 | |
| positive definite | 正定 | |
| posterior distribution | 后验分布 | |
| prior distribution | 先验分布 | |
| probability mass function | 概率质量函数 | PMF |
| quantile | 分位数 | |
| saddle point | 鞍点 | |
| singularity | 奇异性 | |
| singular value decomposition | 奇异值分解 | |
| subgradient | 次梯度 | |
| Taylor expansion | 泰勒展开 | |
| trace | 迹 | 矩阵的迹 |
| transpose | 转置 | |
| unbiasedness | 无偏性 | |
| variance-covariance matrix | 协方差矩阵 | |

## 四、自然语言处理与注意力

| 英文 | 中文 | 备注 |
|---|---|---|
| additive attention | 加性注意力 | |
| attention weight | 注意力权重 | |
| Bahdanau attention | Bahdanau 注意力 | 保留人名 |
| bidirectional recurrent neural network | 双向循环神经网络 | BiRNN |
| byte pair encoding (BPE) | 字节对编码 | |
| co-occurrence | 共现 | |
| context vector | 上下文向量 | |
| decoder hidden state | 解码器隐藏状态 | |
| dot product attention | 点积注意力 | |
| encoder-decoder | 编码器-解码器 | |
| gated recurrent unit (GRU) | 门控循环单元 | |
| GloVe | GloVe | 保留 |
| head (attention) | 头 | 多头注意力的"头" |
| key | 键 | 注意力机制 |
| language model | 语言模型 | |
| layer normalization | 层规范化 | |
| masked language model | 掩码语言模型 | |
| multi-head attention | 多头注意力 | |
| natural language processing | 自然语言处理 | |
| negative sampling | 负采样 | |
| next sentence prediction | 下一句预测 | |
| positional encoding | 位置编码 | |
| query | 查询 | |
| scaled dot-product attention | 缩放点积注意力 | |
| self-attention | 自注意力 | |
| subword embedding | 子词嵌入 | |
| Transformer | Transformer | 保留 |
| value | 值 | |
| vision transformer | 视觉 Transformer | ViT 保留 |
| word embedding | 词嵌入 | |
| word vector | 词向量 | |

## 五、计算机视觉

| 英文 | 中文 | 备注 |
|---|---|---|
| anchor box | 锚框 | |
| bounding box | 边界框 | |
| class prediction | 类别预测 | |
| deconvolution | 反卷积 | 转置卷积的别称 |
| feature pyramid | 特征金字塔 | |
| fine-tuning | 微调 | |
| fully convolutional network | 全卷积网络 | FCN |
| image augmentation | 图像增广 | |
| image classification | 图像分类 | |
| instance segmentation | 实例分割 | |
| intersection over union | 交并比 | IoU |
| mean average precision | 平均精度均值 | mAP |
| multi-scale object detection | 多尺度目标检测 | |
| non-maximum suppression | 非极大值抑制 | |
| object detection | 目标检测 | |
| offset | 偏移量 | 锚框偏移 |
| region proposal network | 区域提议网络 | RPN |
| region-based CNN | 基于区域的卷积神经网络 | R-CNN |
| semantic segmentation | 语义分割 | |
| single shot multibox detector | 单发多框检测器 | SSD 保留 |
| transposed convolution | 转置卷积 | |
| upsampling | 上采样 | |

## 六、强化学习、推荐系统、高斯过程与超参数优化

| 英文 | 中文 | 备注 |
|---|---|---|
| action-value function | 动作价值函数 | |
| autorec | 自编码器推荐模型 | AutoRec 保留 |
| Bayesian personalized ranking | 贝叶斯个性化排序 | BPR |
| Bellman equation | 贝尔曼方程 | |
| click-through rate prediction | 点击率预测 | |
| deep factorization machine | 深度因子分解机 | DeepFM 保留 |
| discount factor | 折扣因子 | |
| dynamic programming | 动态规划 | |
| episode | 回合 | |
| exploration | 探索 | |
| exploitation | 利用 | |
| factorization machine | 因子分解机 | FM 保留 |
| hyperparameter optimization | 超参数优化 | |
| likelihood | 似然 | |
| matrix factorization | 矩阵分解 | |
| neural collaborative filtering | 神经协同过滤 | NeuMF 保留 |
| policy | 策略 | |
| posterior | 后验 | |
| prior | 先验 | |
| random search | 随机搜索 | |
| ranking | 排序 | |
| reward | 奖励 | |
| sequence-aware recommendation | 序列感知推荐 | |
| state | 状态 | |
| state-action value | 状态-动作价值 | |
| successive halving | 逐次减半 | |
| user | 用户 | 推荐系统 |
| item | 物品 | 推荐系统 |
| value function | 价值函数 | |
| value iteration | 值迭代 | |

## 七、计算性能、工程与工具

| 英文 | 中文 | 备注 |
|---|---|---|
| asynchronous computation | 异步计算 | |
| auto-parallelism | 自动并行 | |
| computational graph | 计算图 | |
| data parallelism | 数据并行 | |
| deep learning framework | 深度学习框架 | |
| device | 设备 | CPU/GPU |
| distributed training | 分布式训练 | |
| eager execution | 动态图执行 | 命令式执行 |
| model parallelism | 模型并行 | |
| multi-GPU training | 多 GPU 训练 | |
| operator | 运算符 | |
| pipeline | 流水线 | |
| symbolic programming | 符号式编程 | |
| imperative programming | 命令式编程 | |
| hybridization | 混合式编程 | |
| latency | 延迟 | |
| throughput | 吞吐量 | |
| tensor core | 张量核心 | |
| utility function | 工具函数 | |
| hypervisor | 虚拟机管理程序 | 附录 |
| instance type | 实例类型 | 云服务器 |
| spot instance | 竞价实例 | AWS |

## 八、易混词辨析（务必按上下文判断）

| 英文 | 语境 | 译法 |
|---|---|---|
| key | 注意力机制 / 字典 / 数据库 / 密码学 | 键 / 键 / 键 / 密钥 |
| value | 注意力机制、字典 / 数值 / 价值观 | 值 / 数值 / 价值观 |
| bias | 模型参数 / 统计（bias-variance） | 偏置 / 偏差 |
| variance | 统计量 / 模型方差 | 方差 |
| precision | 分类指标 / 一般语境 | 查准率 / 精确 |
| accuracy | 分类指标 | 准确率（不用"精度"以免与查准率混淆） |
| normalization | 数据处理/网络层 | 规范化（batch normalization 批量规范化） |
| example / instance / sample | 数据点 | 样例 / 实例 / 样本 |
| validation / test | 数据集划分 | 验证 / 测试 |
| objective / loss / cost | 优化目标 | 目标函数 / 损失函数 / 成本函数（原文用哪个词就译哪个，不随意互换） |
| converge / diverge | 优化过程 | 收敛 / 发散 |
| kernel | 卷积 / 高斯过程 / SVM | 卷积核（或滤波器）/ 核函数 / 核函数 |
| pool | 网络层 / 内存池 / 连接池 | 汇聚 / 池 / 池 |
| offset | 参数 / 锚框 / 文件 | 偏移量 / 偏移量 / 偏移 |
| state | RNN / 强化学习 / 程序 | 状态 / 状态 / 状态 |
| attention | 机制 / 日常用语 | 注意力 / 关注 |

## 九、保留原文的专名（不翻译）

- **框架/库**：PyTorch、MXNet、TensorFlow、JAX、NumPy、pandas、matplotlib、Gluon、Keras、Flax、spaCy、Jupyter、Colab、SageMaker、AWS、GitHub、Hugging Face。
- **模型/算法/网络名**：AlexNet、VGG、NiN、GoogLeNet、ResNet、DenseNet、LeNet、BERT、GPT、Transformer、ViT、word2vec、GloVe、SSD、R-CNN、FCN、GAN、DCGAN、FM、DeepFM、AutoRec、NeuMF、LSTM、GRU、ReLU、SGD、Adam、RMSProp、AdaGrad、Adadelta、Xavier、He 初始化（若原文为专名）。
- **数据集/基准**：Fashion-MNIST、MNIST、CIFAR-10、ImageNet、MovieLens、IMDb、Penn Tree Bank、WikiText-2、Kaggle、FrozenLake、OpenAI Gym。
- **代码与 API**：所有类名、函数名、参数名、变量名、模块名、文件名、命令行、URL、bib 引用键。
- **人名与机构**：作者姓名、被引论文作者、大学与公司名称。
- **单位与符号**：GB、MB、ms、FLOPS、CPU、GPU、TPU 及数学符号。

> 说明：`Transformer`、`BERT`、`GPT` 等模型名保留英文，但普通名词性的"transformers/注意力"仍按上下文译出；"large transformer" 视语境译"大型 Transformer 模型"。

## 十、补充术语（翻译与审校过程中统一的新增术语）

| 英文 | 中文 | 备注 |
|---|---|---|
| backpropagation through time (BPTT) | 通过时间反向传播 | 全书统一；缩写 BPTT 保留 |
| objective function | 目标函数 | 源文 objective 与 loss 分开时使用 |
| general trainer | 通用训练器 | 深度学习计算章 |
| logit | logit | 保留英文（D2L 惯例） |
| analytic solution | 解析解 | |
| vectorization | 向量化 | |
| standard deviation | 标准差 | |
| joint probability | 联合概率 | |
| marginalize | 边缘化 | 与"边缘概率"同根 |
| consistent estimator | 一致估计量 | |
| frequentist | 频率学派 | |
| Bayesian | 贝叶斯学派 | |
| aleatoric uncertainty | 偶然不确定性 | |
| epistemic uncertainty | 认知不确定性 | |
| empirical risk | 经验风险 | |
| uniform convergence | 一致收敛 | |
| VC dimension | VC 维 | |
| holdout set | 留出集 | |
| test set reuse | 测试集复用 | |
| adaptive overfitting | 自适应过拟合 | |
| multiple hypothesis testing | 多重假设检验 | |
| importance weight | 重要性权重 | |
| over-parametrized | 过参数化的 | |
| interpolation regime | 插值状态 | |
| double descent | 双下降 | |
| inductive bias | 归纳偏置 | |
| neural tangent kernel | 神经正切核 | |
| Rademacher complexity | 拉德马赫复杂度 | |
| patience criterion | 耐心准则 | |
| ridge regularization | 岭正则化 | |
| lasso regularization | lasso 正则化 | |
| Frobenius norm | 弗罗贝尼乌斯范数 | |
| generalization gap | 泛化差距 | |
| model capacity | 模型容量 | |
| confusion matrix | 混淆矩阵 | |
| bandits | 赌博机 | |
| batch learning | 批量学习 | |
| online learning | 在线学习 | |
| co-adaptation | 共适应 | |
| tied parameters | 参数绑定 | 简称"参数共享"处按原文 |
| constant parameter | 常数参数 | |
| error measure | 误差度量 | |
| imprecision / imputation | 插补 | 缺失值补全 |
| surrogate | 替代函数 | 优化语境 |
| forward differentiation | 前向微分 | |
| backward differentiation | 反向微分 | |
| surprisal | 惊奇度 | |
| nat | 纳特 | 信息量单位 |
| partition function | 配分函数 | |
| cumulant function | 累积量函数 | |
| sufficient statistics | 充分统计量 | |
| exponential family | 指数族 | |
| Cauchy distribution | 柯西分布 | |
| Chebyshev's inequality | 切比雪夫不等式 | |
| Fubini's theorem | Fubini 定理 | |
| summary statistic | 汇总统计量 | |
| stem / body / head | 主干 / 主体 / 头部 | CNN 网络结构 |
| design space | 设计空间 | |
| neural architecture search (NAS) | 神经架构搜索 | NAS 保留 |
| bottleneck | 瓶颈 | |
| bottleneck ratio | 瓶颈比率 | |
| grouped convolution | 分组卷积 | |
| residual block | 残差块 | |
| residual mapping | 残差映射 | |
| shortcut connection | 捷径连接 | |
| internal covariate shift | 内部协变量偏移 | |
| foundation model | 基础模型 | |
| masked softmax operation | 掩码 softmax 运算 | |
| subword embedding | 子词嵌入 | |
| byte pair encoding (BPE) | 字节对编码 | BPE 保留 |
| character n-gram | 字符 n 元语法 | |
| noise word | 噪声词 | |
| multiset | 多重集 | |
| multiplicity | 重数 | |
| masked language model | 掩码语言模型 | |
| next sentence prediction | 下一句预测 | |
| natural language inference | 自然语言推断 | |
| textual entailment | 文本蕴含 | |
| neutral | 中性 | NLI 标签 |
| decomposable attention model | 可分解注意力模型 | |
| max-over-time pooling | 最大时间汇聚 | |
| semantics textual similarity | 语义文本相似度 | |
| part-of-speech tagging | 词性标注 | |
| tagging and parsing | 标注与解析 | |
| sequence-aware recommender system | 序列感知推荐系统 | |
| explicit / implicit feedback | 显式/隐式反馈 | |
| click-through rate (CTR) | 点击率 | CTR 保留 |
| normalized discounted cumulative gain (NDCG) | 归一化折损累积增益 | NDCG 保留 |
| factorization machine | 因子分解机 | FM 保留 |
| wide and deep | 宽深模型 | |
| cold start | 冷启动 | |
| hypergradient | 超梯度 | |
| tuner | 调优器 | 超参数优化 |
| trial | 试验 | 超参数优化 |
| rung | 梯级 | 逐次减半 |
| current best configuration | 当前最优配置 | |
| bracketing | 括选 | 超参数优化 |
| worker | 工作节点 | |
| incumbent | 现任最优配置 | |
| generative adversarial network (GAN) | 生成对抗网络 | GAN 保留 |
| discriminator | 判别器 | |
| generator | 生成器 | |
| minimax game | 极小极大博弈 | |
| two-sample test | 双样本检验 | |
| latent factor | 潜在因子 | 推荐系统 |
| pointwise / pairwise / listwise | 逐点 / 成对 / 列表式 | 排序学习 |
| hinge loss | 合页损失 | |
| Markov decision process (MDP) | 马尔可夫决策过程 | MDP 保留 |
| return | 回报 | 强化学习 |
| transition function | 转移函数 | |
| trajectory | 轨迹 | |
| policy evaluation | 策略评估 | |
| deterministic / stochastic policy | 确定性/随机策略 | |
| credible set | 置信集 | 贝叶斯语境 |
| universal approximator | 通用近似器 | |
| length-scale | 长度尺度 | 高斯过程核 |
| amplitude | 振幅 | 高斯过程核 |
| configuration space | 配置空间 | 超参数优化 |
| grid search | 网格搜索 | |
| artificial general intelligence | 通用人工智能 | |
| singularity | 奇点 | |
| end-to-end training | 端到端训练 | |
| feature engineering | 特征工程 | |
| parsimonious model | 简约模型 | |
| stochastic pooling | 随机汇聚 | |
| fractional max-pooling | 分数最大汇聚 | |
| mirror padding | 镜像填充 | |
| general convolution | 常规卷积 | |
| fractionally-strided convolution | 分数步幅卷积 | |
| focal loss | 焦点损失 | |
| region of interest (RoI) | 感兴趣区域 | RoI 保留 |
| pretrained model | 预训练模型 | |
| memory cell | 记忆单元 | RNN |
| input / forget / output gate | 输入门 / 遗忘门 / 输出门 | |
| reset / update gate | 重置门 / 更新门 | |
| candidate hidden state | 候选隐藏状态 | |
| beam size | 束宽 | |
| batch matrix multiplication (BMM) | 批量矩阵乘法 | BMM 保留 |
| convex cone | 凸锥 | |
| preconditionser | 预条件子 | 优化 |
| leaky average | 泄漏平均 | |
| bias correction | 偏差修正 | Adam |
| memory bandwidth | 显存带宽 | |
| false sharing | 伪共享 | |
| systolic array | 脉动阵列 | |
| streaming multiprocessor | 流式多处理器 | |
| burst read | 突发读取 | |
| wear leveling | 磨损均衡 | |
| ring synchronization | 环同步 | |
| commutative reduction | 可交换归约 | |
| key--value store | 键值存储 | |
| instance type | 实例类型 | 云服务 |
| spot instance | 竞价实例 | 云服务 |
| fork | 复刻 | 版本控制语境 |
| pull request | 拉取请求 | 可保留英文缩写 PR |
| reference design | 公版设计 | GPU 硬件 |
| thermal throttling | 过热降频 | |
| notebook instance | 笔记本实例 | SageMaker |
| trainer | 训练器 | 与通用训练器对应 |
| hyperparameter optimization | 超参数优化 | |
| random search | 随机搜索 | |
| successive halving | 逐次减半 | |
| multi-fidelity optimization | 多保真度优化 | |
| Laplacian smoothing | 拉普拉斯平滑 | |
| Zipf's law | 齐普夫定律 | |
| perplexity | 困惑度 | |
| BLEU (Bilingual Evaluation Understudy) | 双语评估替补 | BLEU 保留 |
| beam width | 束宽 | |
| coverage | 覆盖 | 束搜索语境 |
| prefix | 前缀 | 序列生成 |
| dummy token | 哑词元 | |
| padding token | 填充词元 | |
| hand-written / handwritten | 手写的 | |
| self-supervised learning | 自监督学习 | |
| masked autoencoder | 掩码自编码器 | |
| layer-wise | 逐层的 | |
| temperature scaling | 温度缩放 | |
| prompt | 提示 | |
| zero-shot / few-shot | 零样本 / 少样本 | |
| scaling law | 缩放定律 | |
| knowledge distillation | 知识蒸馏 | |
| Jaccard index | Jaccard 指数 | 官方旧版作"杰卡德系数"；本书统一 Jaccard 指数 |
| aspect ratio | 宽高比 | |
| scale ratio | 缩放比 | 目标检测 |
| positive / negative anchor box | 正 / 负锚框 | |


