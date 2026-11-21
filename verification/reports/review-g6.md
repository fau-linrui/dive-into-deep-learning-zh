# 审校组 G6 审校记录（批次 b12、b13）

审校范围：`work/batches/batch-b12.txt`（6 个：多头注意力 / 自注意力与位置编码 / Transformer / 视觉 Transformer / 大模型预训练 / 优化章索引）+ `batch-b13.txt`（7 个：优化与深度学习 / 凸性 / 梯度下降 / 随机梯度下降 / 小批量随机梯度下降 / 动量法 / Adagrad），共 **13 个文件**。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 逐段对照（定义、公式符号说明与推导、因果与限定词、图题、练习、小结必查，数学密集段落逐句核对；代码块、公式、指令、图片路径、URL 只做结构一致性核验，未改动）。只做定向 `edit`，每次修改后 `python work\verify.py --file <path>` 复核，批次结束再整批复验。另用临时脚本做了三类机械核查（脚本已删除）：中英文数字/单位多集比对（含 万/亿/万亿 换算）、`[**`/`:numref:`/`:eqref:`/`:cite:`/`:citet:`/`:label:`/`:eqlabel:`/图片/`:width:`/tab` 计数比对、逐段长度比与段数一一对应检查。

基线：两批 `verify.py --batch` 均为 0 error（b12 0 warning；b13 8 条“内容一致但顺序不同”警告，系中文语序重排）；`align_check.py` 段落级漏译检查两批均为 **0 个段级疑点文件**。审校后两批仍为 6/6、7/7 PASS，0 error，警告数与基线相同。

## 总览

- 审校文件：13 个；通读通过无修改：8 个（self-attention-and-positional-encoding、large-pretraining-transformers、chapter_optimization/index、optimization-intro、convexity、gd、sgd、minibatch-sgd）；有修改：5 个。
- 修改：**5 个文件、6 处**。分类：术语首现未括注英文 4 处（transformer 1、vision-transformer 2、adagrad 1）；术语前后不一致 1 处（momentum“泄漏平均/泄漏平均值”）；表达歧义 1 处（multihead-attention 小结补“的”）。
- 未发现明确误译/漏译（逐段对照后，13 个文件的信息、限定词、因果关系均与底本一致）；本组修改均为术语规范与可读性层面的定向改进。
- 修改后校验：`python work\verify.py --batch work/batches/batch-b12.txt` → **6/6 PASS，0 error，0 warning**；`--batch batch-b13.txt` → **7/7 PASS，0 error，8 warning**（optimization-intro 1、convexity 1、gd 1、sgd 2、momentum 1、adagrad 2，均为“行内公式/行内代码内容一致但顺序与原文不同”，中文语序重排所致，已人工确认无内容缺失，与审校前一致）。

## 修改清单（5 个文件、6 处）

1. `multihead-attention.md`（L387，小结）：“多头注意力通过查询、键和值**不同的**表示子空间……”有歧义（可读成“键和值不同”），补“的”改为“查询、键和值**的**不同表示子空间”。
2. `transformer.md`（L60，模型一节）：“称为**编码器-解码器注意力**”为全书该术语首现（`subsec_transformer-encoder` 等节均引用此处），按“重要术语首现括注英文”补为“编码器-解码器注意力（encoder-decoder attention）”（官方中文版此处亦括注）。
3. `vision-transformer.md`（L143）：ViT 编码器块“**前置规范化/后置规范化**”首现未括注英文，补为“前置规范化（pre-normalization）”“后置规范化（post-normalization）”；`:numref:`、`*之前/之后*` 斜体标记原样保留。
4. `momentum.md`（L19，基础·泄漏平均值小节）：标题作“泄漏平均值”、小结作“泄漏平均值”，正文却作“泄漏平均”，且首现未括注英文，改为“用“泄漏平均值”（leaky average）来替代梯度计算”。
5. `momentum.md`（L132）：同文件“让我们看看**泄漏平均**实际上对更新做了什么”统一为“泄漏平均值”。
6. `adagrad.md`（L28，预处理一节）：“……称为优化问题的**条件数**”首现未括注英文，补为“条件数（condition number）”（官方中文版此处亦括注）。

## 逐文件记录

### chapter_attention-mechanisms-and-transformers/multihead-attention.md
- 1 处表达歧义修改（见上）。
- 公式逐条核对：式 $\mathbf{h}_i=f(\mathbf W_i^{(q)}\mathbf q,\mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v)\in\mathbb R^{p_v}$ 及 $\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$ 等形状说明、输出式 $\mathbf W_o[\mathbf h_1;\ldots;\mathbf h_h]\in\mathbb R^{p_o}$（$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$）、$p_q=p_k=p_v=p_o/h$ 与“输出数量 $p_qh=p_kh=p_vh=p_o$ 才能并行”的因果关系均与底本一致；`transpose_qkv`/`transpose_output` 的说明、图题、练习（剪枝注意力头）核对无误。

### chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md
- 通读通过，无修改。自注意力定义 $\mathbf{y}_i=f(\mathbf{x}_i,(\mathbf{x}_1,\mathbf{x}_1),\ldots,(\mathbf{x}_n,\mathbf{x}_n))$ 及“按 :eqref:`eq_attention_pooling` 定义”；CNN/RNN/自注意力复杂度 $\mathcal{O}(knd^2)$、$\mathcal{O}(nd^2)$、$\mathcal{O}(n^2d)$ 与顺序操作/最大路径长度 $\mathcal{O}(1)$/$\mathcal{O}(n/k)$/$\mathcal{O}(n)$/$\mathcal{O}(1)$ 全部核对一致。
- 位置编码：$p_{i,2j}=\sin(i/10000^{2j/d})$、$p_{i,2j+1}=\cos(\cdot)$ 及“第 $i$ 行、第 $2j$ 或 $2j+1$ 列”的符号说明对应无误；相对位置投影推导（$\omega_j=1/10000^{2j/d}$、$2\times2$ 旋转矩阵、与位置索引 $i$ 无关）逐式核对无误；二进制表示、热图、练习核对无误。
- 疑点见文末第 2 条（positional embedding matrix 的译法）。

### chapter_attention-mechanisms-and-transformers/transformer.md
- 1 处术语首现括注英文（见上）。
- 编码器/解码器结构：残差连接 $\mathbf{x}+\textrm{sublayer}(\mathbf{x})\in\mathbb R^d$ 的要求、层规范化位置、掩码注意力保留自回归性、`dec_valid_lens` 使查询只关注“截至该查询位置”的说明均与底本一致；逐位置前馈网络、`num_hiddens` 跨层一致、噪声/填充词元说明核对无误；训练/可视化两段（编码器自注意力形状、注意力权重逐行展示、解码器自注意力无查询关注未来、输出序列不关注输入填充词元）逐句核对。
- `[**…**]` 高亮 17 处、`:numref:`/`:cite:` 等计数与底本一致。

### chapter_attention-mechanisms-and-transformers/vision-transformer.md
- 1 处术语首现括注英文（见上）。
- 核对：图像块化公式 $m=hw/p^2$、展平向量长度 $cp^2$、`<cls>` 词元与 $m+1$ 个向量、图像块嵌入用卷积实现（核大小=步幅=块大小）、GELU 与暂退法、前置/后置规范化对比、`num_blks`/`ViTBlock`、训练与小结（与 ResNet 对比、ImageNet 1.2M/3 亿图像的两处数字、Swin Transformer 与二次复杂度）均与底本一致。
- 数字核查：底本 “1.2 million images”“300 million images”分别译“120 万幅”“3 亿幅”，换算正确。

### chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md
- 通读通过，无修改。仅解码器/编码器-解码器三种模式、BERT 掩码语言建模与“双向”解释、T5 连续片段预训练（输入/目标序列示例、因果注意力）、GPT/GPT-2/GPT-3 与上下文学习（零/单/少样本）、幂律扩展与可扩展性、大语言模型（Megatron-Turing NLG、Gopher、Chinchilla、PaLM/PaLM 2、Minerva、Galactica、OPT/BLOOM/FALCON、Llama 1/2）、涌现能力、指令微调与 RLHF、思维链提示等全部逐段对照，信息完整。
- 数字/单位专项核查（脚本+人工）：350M→3.5 亿、250B→2500 亿、2000B→2 万亿、1000B→1 万亿、11B→110 亿、4.6B→46 亿、70B→700 亿、100M→1 亿、1.5B→15 亿、300B→3000 亿（2 处）、530B→5300 亿、270B→2700 亿、280B→2800 亿、540B→5400 亿、780B→7800 亿，全部换算正确；PetaFLOP/s-days、40 GB 等保留原样。
- 底本疑似笔误 1 处照译保留，见文末第 4 条。

### chapter_optimization/index.md
- 通读通过，无修改。“黑盒子/咒语（incantations）”的比喻义、非凸为主但凸问题分析有启发性、本章含凸优化入门与凸目标上 SGD 证明等表述与底本一致。

### chapter_optimization/optimization-intro.md
- 通读通过，无修改。优化目标与深度学习目标的区别（训练误差 vs 泛化误差、经验风险 `f`/`g` 与风险）、局部最小值、鞍点与黑塞矩阵特征值判定三条（全正/全负/有正有负）、梯度消失（$f(x)=\tanh x$、$f'(x)=1-\tanh^2x$、$f'(4)=0.0013$）、小结与练习（$d!$ 个等价解、对称随机矩阵特征值分布）全部核对无误。
- 1 条 warning（行内公式顺序）系中文语序重排，可接受。

### chapter_optimization/convexity.md
- 通读通过，无修改。公式与推导逐条核对：凸集定义式、交集/并集证明、凸函数定义式、詹森不等式（含 $E_X[f(X)]\ge f(E_X[X])$ 与变分/对数似然应用）、局部最小值即全局最小值反证（含 $\lambda=1-p/|x^\ast-x'|$ 与三行不等式链）、下水平集 $\mathcal{S}_b$、二阶导数判据（$f''\ge0$、黑塞矩阵半正定 $\nabla^2f\succeq0$）及其完整双向证明（有限差分极限、中值定理、$g(z)=f(z\mathbf x+(1-z)\mathbf y)$ 引理与多维情形）、拉格朗日函数/乘子与鞍点最大化-最小化、惩罚项、投影 $\textrm{Proj}_\mathcal{X}$ 与 $\ell_1/\ell_2$ 球说明、小结与练习全部与底本一致。
- 1 条 warning（行内公式顺序）可接受。

### chapter_optimization/gd.md
- 通读通过，无修改。一维/多元泰勒展开、$\epsilon=-\eta f'(x)$ 代入得 $f(x-\eta f'(x))=f(x)-\eta f'^2(x)+\mathcal{O}(\eta^2f'^2(x))$、学习率过大/过小、非凸函数 $x\cos(cx)$ 的局部最小值；牛顿法（黑塞矩阵 $\mathbf H=\nabla^2f$、$\boldsymbol\epsilon=-\mathbf H^{-1}\nabla f$、$f(x)=\frac12x^2$ 单步收敛、非凸时“除以黑塞矩阵”的致命缺陷、取绝对值/重新引入学习率）；收敛性分析（$e^{(k)}=x^{(k)}-x^\ast$、泰勒展开、$\left|e^{(k+1)}\right|\le c(e^{(k)})^2$ 及各注意事项）；预处理 $\mathbf x\leftarrow\mathbf x-\eta\textrm{diag}(\mathbf H)^{-1}\nabla f(\mathbf x)$、线搜索；小结与练习均核对无误。
- 1 条 warning（行内公式顺序）可接受。

### chapter_optimization/sgd.md
- 通读通过，无修改。目标函数为样例损失平均、梯度计算 $\mathcal{O}(n)$、随机采样索引与无偏估计 $\mathbb E_i\nabla f_i(\mathbf x)=\nabla f(\mathbf x)$、噪声模拟实验、动态学习率三种策略（分段常数/指数衰减/多项式衰减）与收敛对照、凸目标收敛性证明（$R(\mathbf x)=E_{\boldsymbol\xi}[f]$、$\ell_2$ 范数有界 $L$、两个不等式代入、望远镜求和、$\bar{\mathbf x}$ 定义与 $\eta=r/(L\sqrt T)$ 得到 $rL/\sqrt T$ 即 $\mathcal{O}(1/\sqrt T)$）、有放回/无放回采样概率（0.63/0.37）、小结与练习全部核对无误。
- 2 条 warning（行内公式、行内代码顺序）可接受。

### chapter_optimization/minibatch-sgd.md
- 通读通过，无修改。向量化与缓存（寄存器/L1/L2/L3、2GHz 16 核 AVX-512 的 $2\cdot10^9\cdot16\cdot32=10^{12}$ 字节/秒、带宽对比、突发读取）、矩阵乘法四种实现与内存访问分析、$2mnp$ 次浮点运算与 $256\times256$ 矩阵的 $0.03$ 十亿次运算；小批量统计性质（期望不变、标准差降为 $b^{-1/2}$）、批规范化方差权衡、NASA 机翼噪声数据集与白化、从零/简洁实现、批大小 1500/1/100/10 的时间-损失对比、小结与练习均核对无误。

### chapter_optimization/momentum.md
- 2 处术语一致性修改（见上）。
- 公式逐条核对：小批量梯度 $\mathbf g_{t,t-1}$ 与 $\mathbf h_{i,t-1}$ 定义、泄漏平均 $\mathbf v_t=\beta\mathbf v_{t-1}+\mathbf g_{t,t-1}$、展开式 $\mathbf v_t=\sum_{\tau=0}^{t-1}\beta^\tau\mathbf g_{t-\tau,t-\tau-1}$、有效样本权重 $\sum_{\tau=0}^\infty\beta^\tau=\frac{1}{1-\beta}$ 与步长 $\frac{\eta}{1-\beta}$、更新式、病态问题 $f=0.1x_1^2+2x_2^2$ 的轨迹与 $\beta=0.25$ 对照、二次凸函数 $h(\mathbf x)=\frac12\mathbf x^\top\mathbf Q\mathbf x+\mathbf x^\top\mathbf c+b$ 的最小化子/最小值/梯度/特征分解 $\mathbf Q=\mathbf O^\top\boldsymbol\Lambda\mathbf O$ 与变量替换 $\mathbf z=\mathbf O(\mathbf x-\mathbf Q^{-1}\mathbf c)$、谱坐标下 GD/动量分解、标量函数 $\frac{\lambda}{2}x^2$ 的收敛条件与 $2\times2$ 矩阵 $\mathbf R(\beta,\eta,\lambda)$、$0<\eta\lambda<2+2\beta$ 收敛域、小结与练习均与底本一致。
- 1 条 warning（行内公式顺序）可接受；底本公式疑似缺步长 $\eta$ 见文末第 4 条。

### chapter_optimization/adagrad.md
- 1 处术语首现括注英文（见上）。
- 稀疏特征与学习率（$\eta=\eta_0/\sqrt{t+c}$、$\eta_i=\eta_0/\sqrt{s(i,t)+c}$、$s(i,t+1)=s(i,t)+(\partial_if(\mathbf x))^2$）、预处理（$\mathbf Q=\mathbf U^\top\boldsymbol\Lambda\mathbf U$、条件数 $\kappa=\Lambda_1/\Lambda_d$、对角近似 $\tilde{\mathbf Q}=\textrm{diag}^{-1/2}(\mathbf Q)\mathbf Q\textrm{diag}^{-1/2}(\mathbf Q)$）、梯度大小作为黑塞矩阵规模代理的论证（$\partial_{\bar{\mathbf x}}\bar f=\boldsymbol\Lambda(\bar{\mathbf x}-\bar{\mathbf x}_0)$）、算法三式与逐元素运算说明、$\mathcal O(t^{-1/2})$ 学习率、二次凸问题实验（$\eta=0.4$ 与 $2$）、从零/简洁实现、小结与练习（正交不变性、盖尔什戈林圆盘定理）全部核对无误。
- 2 条 warning（行内公式、行内代码顺序）可接受。

## 遗留疑点（未擅改，提请负责人裁决）

1. **术语表未收录（建议合入 TERMINOLOGY.md）**：leaky average 泄漏平均值；condition number 条件数；pre-normalization / post-normalization 前置/后置规范化；encoder-decoder attention 编码器-解码器注意力。另注：b14 的 rmsprop/adam/adadelta（G7 范围）用“泄漏平均”，与本组 momentum“泄漏平均值”存在**跨批次不一致**；官方中文版统一作“泄漏平均值”，G7 报告也建议把 leaky average 合入术语表。建议负责人统一定为“泄漏平均值（leaky average）”并全书替换（本组未越界改 b14 文件）。
2. **位置嵌入/位置编码的取舍**：`self-attention-and-positional-encoding.md` 中底本 “positional embedding matrix $\mathbf{P}$” 译作“位置编码矩阵”，与术语表 positional encoding→位置编码 自洽，但未体现 embedding；是否改为“位置嵌入矩阵”（同时保持“位置编码”为机制名）请负责人定夺（本组未改）。
3. **momentum.md L18 底本用词不严谨**：底本 “we used $\mathbf h_{i,t-1}=\partial_{\mathbf w}f(\mathbf x_i,\mathbf w_{t-1})$ as the stochastic gradient descent for sample $i$” 中 $\mathbf h$ 实为**随机梯度**而非“随机梯度下降”，译文照底本作“对样本 $i$ 的随机梯度下降”（官方中文版译法相同）。是否改为“样本 $i$ 的随机梯度”或加译注，请裁决。
4. **原文疑似笔误（按底本直译保留，未擅改技术内容/公式）**：
   - `large-pretraining-transformers.md` L84：“展示了用 Transformer **编码器**预训练 GPT 的过程”，而同段图注与后文均为“解码器”，底本自身矛盾（英文底本亦作 encoder）。
   - `momentum.md`“标量函数”小节独立公式 $\mathbf z_t=\mathbf z_{t-1}-\boldsymbol\Lambda\mathbf z_{t-1}$ 缺步长 $\eta$（同段前文 GD 式为 $x_{t+1}=x_t-\eta\lambda x_t$；英文底本同样缺失，公式按规范未动）。
   - `gd.md` 收敛分析称该收敛为“*线性*收敛”、“常数收敛速率”，与同段 $\left|e^{(k+1)}\right|\le c(e^{(k)})^2$ 的“二次减小”表述在底本中即易混（忠实照译，未加译注）。
5. **warning 说明**：b13 共 8 条 `verify.py` 警告均为“行内公式/行内代码内容一致但顺序与原文不同”（中文语序重排），人工比对确认无内容缺失，与审校前数量相同，按手册可忽略。
6. **通读中未改的次要生硬点（供参考，不属误译）**：`sgd.md`“每次独立变量迭代的计算代价”、`adagrad.md`“自变量的迭代轨迹”（均与官方中文版同处理）；`vision-transformer.md`“在实证上，从图像中取出 $2\times2$ 的图像块作为输入”句首状语略悬空。以上均可读，按“不过度修改”原则保留。

## 说明

- 仅修改本组负责的 5 个文件、6 处定向替换；未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md` 及其他批次/组文件，未改动任何代码块、行内代码、公式、指令、图片路径、URL、标题层级、表格结构。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b12.txt` = 6/6 PASS（0 error、0 warning）；`--batch work/batches/batch-b13.txt` = 7/7 PASS（0 error、8 warning，与基线一致）；`python work\align_check.py --batch ...` 两批段级疑点均为 0。
