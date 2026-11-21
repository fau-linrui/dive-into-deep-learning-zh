# 审校组 G5 审校记录（批次 b10、b11）

审校范围：`work/batches/batch-b10.txt` + `batch-b11.txt`，共 17 个文件（RNN 从零/简洁实现、BPTT；现代循环网络 7 篇；seq2seq、束搜索；注意力机制 5 篇）。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 逐段对照（重点核对定义、公式符号说明、因果与限定词、图题/表格/练习、术语一致性）；只做定向 `edit`，不改代码块/行内代码/公式/指令/图片路径/URL/标题层级/表格结构；每次修改后 `python work\verify.py --file <path>` 校验，最后整批复验。

基线：两批 `verify.py --batch` 均为 0 error（b10 9 warning、b11 0 warning，均为“内容一致但顺序不同”，可接受）；`align_check.py` 段落级漏译检查 b10 仅 1 条“中文偏少”提示（经核为非漏译），b11 无提示。

## 总览

- 审校文件：17 个；通读通过无修改：5 个（rnn-concise、rnn-scratch、gru、deep-rnn、encoder-decoder）；有修改：12 个。
- 修改问题：约 47 处。分类：**漏译（框架标签页链接文字 `[Discussions]` 未译）21 处**（涉及 6 个文件）；**误译/语义偏差 7 处**；**表达生硬/句法不通 16 处**；**标点、引文位置、连字符等 3 处**。
- 修改后校验：`python work\verify.py --batch work\batches\batch-b10.txt` → **10/10 PASS，0 error，9 warning**；`--batch batch-b11.txt` → **7/7 PASS，0 error，1 warning**。warning 均为“行内公式/行内代码内容一致但顺序与原文不同”（中文语序重排），非内容错误。
- 复跑 `align_check.py`：b10 仍仅 rnn-concise 原提示，b11 段级疑点 0。

## 逐文件记录

### chapter_recurrent-neural-networks/rnn-scratch.md（b10）
- 通读通过，无修改。梯度裁剪的利普希茨连续性推导、$\mathbf{x} \gets \mathbf{x}-\eta\mathbf{g}$ 等公式与符号说明、解码/预热期说明、练习（含 $q(x_t\mid\cdots)\propto P(\cdot)^\alpha$ 采样题）均与底本一致。

### chapter_recurrent-neural-networks/rnn-concise.md（b10）
- 通读通过，无修改。术语（高级 API、隐藏状态、小批量、词元）、标签页、练习均与底本一致。
- align_check 的“段落 11 中文偏少（英文 37 词→中文 34 字）”提示经核为括号内两条 by… 短语的中文压缩表述，信息完整，非漏译。

### chapter_recurrent-neural-networks/bptt.md（b10）
- 误译/生硬修正 4 处：
  - “把和在 $\partial h_{t-\tau}/\partial w_\textrm{h}$ 处终止”→“将求和终止在……处”（原句“把和在”不成词）。
  - “按时间截断反向传播”→“截断的按时间反向传播”（truncated backpropagation through time，原译把 truncated 误挂到“按时间”上）。
  - “适当地赋予过大的权重”→“适当地加大权重”（appropriately overweighted，原译“过大”带贬义，与“适当”矛盾）。
  - “的潜在非常大的幂”→“可能非常大的幂次”（potentially very large powers，原语序生硬）。

### chapter_recurrent-modern/index.md（b10）
- “臭名昭著”→“出了名的”（notorious 的比喻义，原译生硬）。
- “循环边沿”系误译：recurrent edge 指计算图中的“循环边”（原译把 edge 误作“边沿”），改为“沿着权重为 1 的循环边在多个连续时间步上逐级传递”。
- “帮助网络确定：不仅允许哪些输入……而且还确定……”句法冗赘（determine not only the inputs..., but when...），理顺为“确定两件事：哪些输入可以进入记忆状态，以及……”。
- “among a myriad of other tasks”原译“此外还广泛用于无数其他任务”偏离原句结构，改为“序列标注任务以及其他众多任务的主流方法”。

### chapter_recurrent-modern/lstm.md（b10）
- “并且新颖地包含了乘法节点”→“并创新性地加入了乘法节点”（novel inclusion of multiplicative nodes，原译生硬）。
- “再进行一次逐点乘法”→“再进行一次按元素乘法”（point-wise multiplication；与本节“哈达玛（按元素）乘积”、术语表 element-wise→按元素 统一）。
- 其余门控公式、符号说明、图题、jax 标签页说明、练习均核对无误。

### chapter_recurrent-modern/gru.md（b10）
- 通读通过，无修改。重置门/更新门/候选隐藏状态、哈达玛乘积、最终更新公式的符号说明与底本一致；练习 4 的“只实现一部分”表述准确。

### chapter_recurrent-modern/deep-rnn.md（b10）
- 通读通过，无修改。堆叠 RNN 公式 $\mathbf{H}_t^{(l)}$、$\mathbf{W}_{\textrm{xh}}^{(l)}$ 等符号说明、图题、jax 标签页关于 Flax 极简实现的说明、练习均核对无误。

### chapter_recurrent-modern/bi-rnn.md（b10）
- “根据空格之后的内容”（blank）→“根据空白处之后的内容”（blank 指被遮盖的词元空缺，非空格字符）。
- 其余双向公式、$\overrightarrow{}/\overleftarrow{}$ 符号说明、图题、jax 标签页说明、练习均核对无误。

### chapter_recurrent-modern/machine-translation-and-dataset.md（b10）
- 首句“在……中，包括……一项重大进展”句法不通，改为“……重大突破之一，是……的一项重大进展”。
- “返回两个词元列表组成的列表”有歧义（two lists of token lists），改为“返回两个由词元列表组成的列表”。
- 其余预处理步骤、`<eos>`/`<pad>`/`<unk>`/`<bos>` 实体、截断/填充说明、练习均核对无误。

### chapter_recurrent-modern/encoder-decoder.md（b10）
- 通读通过，无修改。leftwards context 按官方译法译“左侧上下文”，术语（编码器/解码器/条件语言模型/编码后的状态）一致。

### chapter_recurrent-modern/seq2seq.md（b11）
- 教师强制示例“<bos>”“Ils”……之间缺少顿号分隔（底本用逗号分隔），补为“<bos>”、“Ils”……。
- “计算出预测分布 $p(\cdot)$，它定义在后续输出词元 $t'+1$ 上”生硬（over 的译法），改为“计算出关于后续输出词元 $t'+1$ 的预测分布 $p(\cdot)$”。
- 4 处框架标签页链接文字 `[Discussions]` 未译，按 b10 与本书通行做法改为 `[讨论]`（URL 不变）。
- BLEU 定义式、“双语评估替补（BLEU）”、$p_n$ 查准率说明、惩罚因子 $\exp(1-6/2)\approx 0.14$ 核对无误；$p_n^{1/2^n}$ 与“原论文使用 $p_n^{1/n}$”括注保留。

### chapter_recurrent-modern/beam-search.md（b11）
- “假设输出字典中有四个词元”→“假设输出词表中有四个词元”（output dictionary 在此即输出词表，且本节前文已用“输出词表”，术语表 vocabulary→词表）。
- 1 处 `[Discussions]` 链接文字未译，改为 `[讨论]`。
- 贪心/穷举/束搜索的定义、$\mathcal{O}(|\mathcal{Y}|^{T'})$ 与 $|\mathcal{Y}|T'$ 成本对比、束搜索评分式与 $L^\alpha$ 惩罚说明、图题、练习均核对无误。

### chapter_attention-mechanisms-and-transformers/index.md（b11）
- 误译修正：“这一创新最初被设想为一种增强手段，用于把编码器-解码器 RNN 应用于序列到序列的应用”——原句 enhancement for encoder--decoder RNNs applied to seq2seq applications 的修饰关系译错，改为“被设想为对应用于序列到序列任务（例如机器翻译）的编码器-解码器 RNN 的一种增强手段”。
- “转而又巧妙地安排注意力机制”中的“又”系误加（instead relying on），改为“转而依靠巧妙安排的注意力机制”。
- 其余章节导言、Transformer/视觉 Transformer/基础模型等表述核对无误。

### chapter_attention-mechanisms-and-transformers/queries-keys-values.md（b11）
- “都关键地依赖于输入”副词硬译，改为“都严重依赖于输入”。
- 数据库例子“例如用对 "Li" 的精确查询（$q$）会返回值 "Mu"”句法残缺，改为“例如对 "Li" 进行精确查询（$q$）会返回值 "Mu"”。
- “如果……那么我们反而会检索到”→“那么我们就会改为检索到”（“反而”不是原文）。
- 4 处 `[Discussions]` 链接文字未译，改为 `[讨论]`。
- 注意力汇聚定义式、凸锥/凸组合/平均汇聚等特例、softmax 规范化式、练习中的 $\nabla_{\mathbf{q}}$ 与协方差等式均核对无误。

### chapter_attention-mechanisms-and-transformers/attention-pooling.md（b11）
- `:cite:parzen1957consistent` 原译被移到了维基百科链接之后（成了给链接的引用），改回紧跟“*Parzen 窗*（Parzen Windows）”。
- “训练特征--标签对为一个键--值对”中文里用了双连字符，改为“训练特征-标签对为一个键-值对”。
- 4 处 `[Discussions]` 链接文字未译，改为 `[讨论]`。
- Nadaraya--Watson 核回归、核宽 $\sigma^2$ 的调整、练习（Parzen 窗等价性、用 SGD 学核宽、单位球面上的点积注意力提示）均核对无误。

### chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md（b11）
- “我们要么使用矩阵 $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ 来解决不匹配问题”表述错位（矩阵指 $\mathbf{M}$），改为“借助一个矩阵，通过 $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ 来解决不匹配问题”。
- 小结中“都关键地依赖于”硬译，改为“都严重依赖”。
- 4 处 `[Discussions]` 链接文字未译，改为 `[讨论]`。
- 点积注意力推导（含 $1/\sqrt{d}$ 缩放、点积方差为 $d$ 的说明）、掩码 softmax、批量矩阵乘法、加性注意力公式与 $\mathbf W_q/\mathbf W_k/\mathbf w_v$ 形状说明、练习均核对无误。

### chapter_attention-mechanisms-and-transformers/bahdanau-attention.md（b11）
- “在键--值对上分配了非均匀的权重”中文里的双连字符，改为“键-值对”。
- 4 处 `[Discussions]` 链接文字未译，改为 `[讨论]`。
- 解码器初始化三点（编码器末层全时间步隐藏状态作为键值、编码器最终步全层隐藏状态用于初始化、编码器有效长度）、注意力公式 $\mathbf{c}_{t'}=\sum_{t}\alpha(\mathbf{s}_{t'-1},\mathbf{h}_{t})\mathbf{h}_{t}$、加性评分函数引用、可视化说明、练习均核对无误。

## 遗留疑点（未擅改，待负责人裁决）

1. **BPTT 译名**：译文在 `bptt.md`、`rnn-scratch.md` 中统一作“按时间反向传播”，中文教材通行（含官方中文版）作“通过时间反向传播”，且 `TERMINOLOGY.md` 未收录该词条。是否全书统一改译，请负责人定夺（本次未改动 15+ 处）。
2. **底本疑似笔误（照译保留）**：`attention-scoring-functions.md` 底本句 “Then the batch matrix multiplication (BMM) computes the elementwise product” 与 BMM 定义不符（BMM 是矩阵乘积，非按元素乘积），译文照译为“计算按元素的乘积”。按手册“原文笔误按底本直译、记录不擅改”处理，建议决定是否加译注。
3. **`[Discussions]` 链接文字**：本次只在 G5 负责的 6 个文件（seq2seq、beam-search、queries-keys-values、attention-pooling、attention-scoring-functions、bahdanau-attention）中改为“讨论”；全书另有数十个文件仍保留英文 `[Discussions]`（属其他组范围），建议负责人统一规范。
4. **术语用词小分歧**：`attention-scoring-functions.md` 用“哑词元”（dummy tokens），而 seq2seq/机器翻译章用“填充词元”（padding tokens）；两者语境略异，是否统一为“占位/填充词元”待定。`deep-rnn.md` 的“非平凡”与全书其他章用法一致，未动。
