# 《动手学深度学习》中文全译本 — 独立一致性与完整性审计报告

- **审计对象**：`d2l-zh-translation/` 全书 192 个书内 Markdown 文件（清单 = `work/batches/done.txt`；不包含 `contrib/`、`static/`、`img/`、`d2l/`、`ci/` 等非书稿目录，也不包含本次新增的 `verification/` 审校记录副本）
- **英文底本**：`d2l-en-src/`（release 1.0.3，commit 23d7a5a）
- **审计方式**：只读。未修改任何译文文件；全部结论基于工具输出 + 人工语境判断，原始命令与数字均可复现
- **审计快照时间**：2026-11-21 14:27–14:30（UTC+8）。审计期间其他审校组在并发收尾（14:17–14:27 窗口内 79/199 个翻译 md 被修改），本报告所有数字均在最终稳定状态（最后一次书内文件改动 14:25:23，`alexnet.md`）上重跑确认；最终构建（14:27:04 完成）的构建副本与当前 192 个书内文件 SHA1 逐字节一致
- **结论口径**：`verify.py --batch`、`verify.py --all`、`termcheck.py`、`check_marks.py`、`align_check.py` 均为项目现有工具；`check_marks.py` 的"源/译计数差"不等于 self-imbalance，本报告分别说明

---

## 0. 总体结论

| 维度 | 结果 |
|---|---|
| 结构完整性（代码块/公式/指令/标题/表格/图片/链接/标签页） | **192/192 文件 PASS，0 error**；全书级 0 error |
| 漏译/增译（段级） | `align_check` 15 个疑点文件逐一复核，**未发现真实漏译**（均为独立公式成段/拆段造成的对齐假阳性，以及英文示例句、文件名、人名列表等合法英文残留） |
| 英文残留 | `verify` 报 13 个文件的"疑似未翻译英文行"，逐条判断**全部为专名、英文示例句、数据样例/路径、POS 标注串、跨行公式**；未发现整段未译正文 |
| 术语 | 任务清单中的 13 类"应收敛变体"**全部为 0 次**（池化/丢弃法/随机失活/隐状态/批归一化/层归一化/位置嵌入/记事本/学习速率/集束搜索/累计/杰卡德/词元标记/精确率）；仅 2 类建议收敛：`拼接`（concatenate 语境 2 处）、`掩蔽`（2 处） |
| 格式标记 self-imbalance | **0**（192 个文件、6 类标记对全部自平衡；alexnet 已修复且不再出现在问题清单） |
| 框架标签页 | 全书 `begin=628 / end=627`，与英文底本**完全一致**；唯一不配对文件 `chapter_appendix-tools-for-deep-learning/d2l.md`（3/2）为**源缺陷**，译文原样保留 |
| 图片/交叉引用/文献 | 相对图片引用 259 个全部可解析；标签 617 个无重复；引用 1097 处（除 1 处源缺陷空引用外）全部可达；610 处文献引用键全部存在 |
| 构建 | `d2lbook build html`（pytorch 标签）**成功**，185 个 HTML 页面；76 条 warning 全部可解释且与译文内容无关 |
| **阻断问题** | **无** |
| **应修问题** | **4 类**：① 22 个文件共 37 处全角 slide 标记残留（构建产物有可见渲染残留）；② 2 处 `(**…**)` 被改成 `[**…**]`（HTML 等价但标记类型不一致）；③ `拼接` 2 处（concatenate 语境）；④ `掩蔽` 2 处 |

---

## 1. 术语一致性

### 1.1 方法

1. 读 `d2l-zh-translation/TERMINOLOGY.md`（330+ 条，含"保留原文的专名"与易混词辨析）；
2. `python work\termcheck.py`（只读）扫描全部 192 个书内文件、**排除代码围栏**；注意：`verification/` 审校记录于 14:26 被加入译文仓库，若直接跑原版脚本会把记录里的引用样例也计入，故审计用同源脚本把 `verification/`、`ref/`、`work/` 加入排除后重跑（结果与加入前一致）；
3. 对任务清单逐项做正则计数 + 上下文人工判断（近义词是否合法）。

### 1.2 任务清单逐项结果（"应收敛变体"出现次数）

| 检查项（应收敛变体） | 出现次数 | 判定 |
|---|---|---|
| 池化 | **0** | 通过（全书 `汇聚` 164 次，`注意力汇聚`/`感兴趣区域汇聚` 等） |
| 丢弃法 | **0** | 通过（`暂退法` 57 次） |
| 随机失活 | **0** | 通过 |
| 隐状态 | **0** | 通过（`隐藏状态`，术语表规定） |
| 批归一化 / 层归一化 | **0 / 0** | 通过（`批量规范化`/`层规范化`） |
| 位置嵌入 | **0** | 通过（`位置编码` 30 次） |
| 记事本 | **0** | 通过（`笔记本` 76 次，Jupyter 笔记本等） |
| 学习速率 | **0** | 通过（`学习率` 199 次） |
| 集束搜索 | **0** | 通过（`束搜索` 15 次） |
| 累计（NDCG 语境） | **0** | 通过（NDCG 统一作"归一化折损**累积**增益"，2 处；"累计"全书 0 次） |
| 杰卡德 | **0** | 通过（统一 `Jaccard 指数` 5 次，与术语表一致） |
| 词元标记 | **0** | 通过（统一 `词元`） |
| 连结（vs 拼接，concatenate 语境） | 连结 80 / **拼接 3** | **2 处建议收敛**，见 1.3-①；1 处合法（英文为 stitching） |
| 查准率（vs 精确率） | 精确率 **0**，查准率 3 | 通过 |

### 1.3 `termcheck.py` 报出的 6 组命中，逐条语境判断

| # | 命中（建议→变体） | 次数/位置 | 人工判定 |
|---|---|---|---|
| ① | 连结 → **拼接** | 3 处：`chapter_natural-language-processing-pretraining/bert-dataset.md:10`、`bert.md:525`、`chapter_preliminaries/linear-algebra.md:880` | **前 2 处应修**：英文分别为 "the **concatenation** of two huge corpora" 与 "pretrained on the **concatenation** of BookCorpus…"，按 TERMINOLOGY（concatenate→连结）应作"连结"。第 3 处英文原文为 "**stitching** the results together"，译"拼接"贴切，**可保留** |
| ② | 规范化 → **归一化** | 2 处：`chapter_recommender-systems/neumf.md:117`、`ranking.md:5` | **可保留**：`归一化折损累积增益`是术语表对 NDCG 的指定全称（normalized＝归一化），非 normalization 术语误用 |
| ③ | 掩码 → **掩蔽** | 2 处：`chapter_recommender-systems/autorec.md:42`（"梯度被掩蔽"）、`chapter_recurrent-modern/seq2seq.md:607`（"用零值掩蔽无关项"） | **建议统一为"掩码"**：全书 mask/masked 的其余 67 处均用"掩码"（含"被掩码的词元""掩码 softmax"）；此处为动词用法。若负责人允许动词用"掩蔽"，则应在 TERMINOLOGY 中补注；否则 2 处收敛 |
| ④ | 暂退法 → **随机丢弃** | 1 处：`chapter_multilayer-perceptrons/dropout.md:334`（"`Dropout` 层会根据指定的暂退概率**随机丢弃**前一层的输出"） | **可保留**：这是对 dropout 机制的白话描述动词，不是术语替代 |
| ⑤ | 概念偏移 → **概念漂移** | 1 处：`chapter_convolutional-modern/batch-norm.md:832` | **可保留**：英文原句为 "more akin to **concept drift**"（不同概念，非 distribution shift 家族术语），"概念漂移"是译界通行译法 |
| ⑥ | 均值 → **平均数** | 1 处：`chapter_preliminaries/linear-algebra.md:558` | **可保留**：原句即"`*均值*（mean），也称为*平均数*（average）`"，是区分 mean/average 的释义句，非不一致 |

> 结论：术语整体收敛良好。任务清单所列变体全部清零；仅需处理 `拼接`（concatenate）2 处与 `掩蔽` 2 处（或补术语表说明）。

---

## 2. 英文残留

### 2.1 `verify.py` 的"疑似未翻译英文行"（13 个文件 / 共 61 行）

逐条核对（行号为实际文件行号）：

| 文件:行 | 内容类别 | 判定 |
|---|---|---|
| `chapter_appendix-mathematics-for-deep-learning/distributions.md:761–790` 附近 3 行 | 跨行行内公式的 LaTeX 续行（`\end{bmatrix}=...`） | 公式行，合法 |
| `chapter_computer-vision/kaggle-dog.md:56–57` | 数据目录清单 `../data/.../labels.csv` 等 | 文件路径，须保留英文，合法 |
| `chapter_convolutional-modern/alexnet.md:176` | `（ImageNet Large Scale Visual Recognition Challenge）` | 竞赛英文全称括注，合法 |
| `chapter_linear-regression/linear-regression.md:306` | `（stochastic gradient descent，SGD）` | 术语英文全称+缩写括注，合法 |
| `chapter_natural-language-processing-applications/finetuning-bert.md:116` | `"NNP (noun, proper singular) NNP POS ..."` | Penn Treebank II 词性标注输出串，合法 |
| `chapter_natural-language-processing-pretraining/glove.md:183` | 表格 `\|$w_k$=\|solid\|gas\|water\|fashion\|` | 语料英文词样例，合法 |
| `chapter_natural-language-processing-pretraining/index.md:23,25` | `"go to the bank to deposit/sit down..."` | 英文示例句（词义随上下文变化），合法 |
| `chapter_natural-language-processing-pretraining/similarity-analogy.md:212,245,257` | `"man"∶"woman"∷"son"∶"daughter"` 等 | 英文词类比示例，合法 |
| `chapter_natural-language-processing-pretraining/word-embedding-dataset.md:42` | `[Penn Tree Bank (PTB)](...)` | 数据集专名，合法 |
| `chapter_preface/index.md:161–201`（41 行） | GitHub 贡献者 ID/姓名列表 | 人名，合法 |
| `chapter_preliminaries/pandas.md:25` | CSV 示例记录 `"Albert Einstein,March 14 1879,..."` | 数据样例，合法 |
| `chapter_recurrent-neural-networks/language-model.md:177–179` | `"It is raining outside"` 等 | 英文示例句，合法 |
| `chapter_recurrent-neural-networks/text-sequence.md:106` | `"Baby needs a new pair of shoes"` | 英文示例句，合法 |

另：不含中文的标题仅 3 个 —— `chapter_computer-vision/rcnn.md` 的 `## Fast R-CNN`、`## Faster R-CNN`、`## Mask R-CNN`（模型专名，合法）。

### 2.2 独立补扫（排除代码/公式/链接后仍无中文的行）

自建扫描（去掉围栏代码、`$$…$$`、行内公式/代码、指令、链接/图片、URL 后，统计 ≥4 个英文词且无汉字的行）共 61 行，**全部落入上表类别**，无遗漏段落。另有 2 项工具级覆盖检查全部通过：`verify` 的"中文/英文词数比"警告为 0 条、"图片替代文本/链接可见文字未翻译"警告为 0 条。

> 结论：未发现真正的未翻译英文内容；`verify` 的 13 条提示均可解释，无需处理。

---

## 3. 格式标记（`(**`/`**)`、`[**`/`**]`、`[~~`/`~~]`、`(~~`/`~~)`）

### 3.1 self-imbalance：0

`python work\check_marks.py`（围栏外、语义等价的 4 类 ASCII 标记对 + 2 类全角变体逐文件自平衡检查）：

- 192 个书内文件，**self-imbalance 文件 0 个**（`[**…**]` 436/436、`[~~…~~]` 4/4、`(**…**)` 71/71、`(~~…~~)` 2/2、全角 `（**…**）` 36/36、全角 `（~~…~~）` 1/1）；
- 即任务要求的"除已修复的 alexnet 外无目标文件自身标记不平衡"**成立**：alexnet 当前不在问题清单，且无任何文件自不平衡。

### 3.2 源/译标记形式差异（23 个文件，需负责人处理）

`check_marks.py` 同时报告 **23 个文件与英文底本的标记计数不一致**（这不是 self-imbalance）。逐项核对源文件后，全部可归因于如下样式改动：

| 类型 | 数量 | 说明 |
|---|---|---|
| `(**text**)` → `（**text**）`（全角括号） | **36 对 / 21 个文件** | 底本 ASCII 的 slide 标记被译成全角括号+粗体 |
| `(~~text~~)` → `（~~text~~）` | **1 对 / 1 个文件**（`chapter_preliminaries/ndarray.md:238`） | 同上 |
| `(**text**)` → `[**text**]`（标记类型改为 D2L 高亮） | 2 处（`chapter_multilayer-perceptrons/mlp.md:287`、`mlp-implementation.md:181`） | HTML 渲染等价，但标记类型与底本不一致（仅 slides 语义不同） |

**受影响文件（全角 `（**…**）`，括号内为对数）**：
`chapter_builders-guide/{custom-layer(2), model-construction(2), parameters(1), read-write(2), use-gpu(4)}`、
`chapter_computational-performance/multiple-gpus-concise(1)`、
`chapter_computer-vision/{anchor(1), bounding-box(1), fine-tuning(1)}`、
`chapter_linear-classification/softmax-regression-scratch(3)`、
`chapter_linear-regression/linear-regression-concise(3)`、
`chapter_multilayer-perceptrons/{dropout(1), kaggle-house-price(4), mlp(1), numerical-stability-and-init(1)}`、
`chapter_preliminaries/ndarray.md`（`（~~…~~）` 1 对）、
`chapter_recurrent-modern/{deep-rnn(1), encoder-decoder(1), gru(1), machine-translation-and-dataset(1)}`、
`chapter_recurrent-neural-networks/{sequence(2), text-sequence(2)}`。
其中 5 处是**标题**：`custom-layer.md:50`、`read-write.md:52`、`numerical-stability-and-init.md:107`、`encoder-decoder.md:56`、`gru.md:147`。

**为什么需要修（实测影响）**：d2lbook 只识别 ASCII 标记对（`d2lbook/slides.py:_match_slide_marks` 的 pairs = `[**`,`(**`,`[~~`,`(~~` 的 ASCII 形式）。全角化后标记不再被剥离，已在本次最终构建（14:27:04）的 HTML 中产生可见残留：

- **22 个 HTML 页面**出现全角括号+粗体残留，例如 `chapter_builders-guide/custom-layer.html` 的章节标题渲染为 `6.5.1. （<strong>不带参数的层</strong>）`，而英文底本构建结果应为无括号、无粗体的 `Layers without Parameters`；
- **1 个页面** `chapter_preliminaries/ndarray.html` 出现**字面残留**：`…所有元素都设为0的张量（[STRIKEOUT:或全为1的张量]）…`（英文底本此处 `(~~or one~~)` 在 HTML 中应整体删除）。

**建议修复方式（机械、低风险，供负责人执行）**：按英文底本将这 22 个文件的标记恢复为 ASCII（`（**`→`(**`、`**）`→`**)`、`（~~`→`(~~`、`~~）`→`~~)`），只改括号字形、不动括号内文字；同时把上述 2 处 `[**…**]` 改回 `(**…**)`。修完 `python work\check_marks.py` 应显示"问题文件 0 个"（self-imbalance 本就为 0），再重跑一次构建即可消除 HTML 残留。审校组 G3/G8 已在各自报告中对同类问题采用同一修复口径（评审记录 `verification/reports/review-g8.md` 第 5 节、`review-g3.md` 第 1–2 条）。

> 说明：`verification/` 目录是 14:26 加入译文仓库的审校记录（非书稿），其中引用的标记样例会干扰整树扫描；建议在交付脚本/复核命令中排除，或说明其非书稿性质。

---

## 4. 章节标题与目录

### 4.1 全部 27 个章节 index.md 的 H1（译文 / 英文底本对照）

| # | 目录 | 译文 H1 | 英文底本 H1 | 判定 |
|---|---|---|---|---|
| 1 | chapter_appendix-mathematics-for-deep-learning | 附录：深度学习数学 | Appendix: Mathematics for Deep Learning | ✓ |
| 2 | chapter_appendix-tools-for-deep-learning | 附录：深度学习工具 | Appendix: Tools for Deep Learning | ✓（与上一附录命名风格一致） |
| 3 | chapter_attention-mechanisms-and-transformers | 注意力机制与 Transformer | Attention Mechanisms and Transformers | ✓ |
| 4 | **chapter_builders-guide** | **深度学习计算** | **Builders' Guide** | ⚠ 语义不完全对应（见 4.2-①） |
| 5 | chapter_computational-performance | 计算性能 | Computational Performance | ✓ |
| 6 | chapter_computer-vision | 计算机视觉 | Computer Vision | ✓ |
| 7 | chapter_convolutional-modern | 现代卷积神经网络 | Modern Convolutional Neural Networks | ✓ |
| 8 | chapter_convolutional-neural-networks | 卷积神经网络 | Convolutional Neural Networks | ✓ |
| 9 | chapter_gaussian-processes | 高斯过程 | Gaussian Processes | ✓ |
| 10 | chapter_generative-adversarial-networks | 生成对抗网络 | Generative Adversarial Networks | ✓ |
| 11 | chapter_hyperparameter-optimization | 超参数优化 | Hyperparameter Optimization | ✓ |
| 12 | chapter_installation | 安装 | Installation | ✓ |
| 13 | chapter_introduction | 引言 | Introduction | ✓ |
| 14 | chapter_linear-classification | 用于分类的线性神经网络 | Linear Neural Networks for Classification | ✓（与 #15 成对统一） |
| 15 | chapter_linear-regression | 用于回归的线性神经网络 | Linear Neural Networks for Regression | ✓（与 #14 成对统一） |
| 16 | chapter_multilayer-perceptrons | 多层感知机 | Multilayer Perceptrons | ✓ |
| 17 | chapter_natural-language-processing-applications | 自然语言处理：应用 | Natural Language Processing: Applications | ✓ |
| 18 | chapter_natural-language-processing-pretraining | 自然语言处理：预训练 | Natural Language Processing: Pretraining | ✓ |
| 19 | chapter_notation | 符号 | Notation | ✓ |
| 20 | chapter_optimization | 优化算法 | Optimization Algorithms | ✓ |
| 21 | chapter_preface | 前言 | Preface | ✓ |
| 22 | chapter_preliminaries | 预备知识 | Preliminaries | ✓ |
| 23 | chapter_recommender-systems | 推荐系统 | Recommender Systems | ✓ |
| 24 | chapter_recurrent-modern | 现代循环神经网络 | Modern Recurrent Neural Networks | ✓ |
| 25 | chapter_recurrent-neural-networks | 循环神经网络 | Recurrent Neural Networks | ✓ |
| 26 | chapter_references | （无 index.md，与底本一致） | （无 index.md） | ✓ |
| 27 | chapter_reinforcement-learning | 强化学习 | Reinforcement Learning | ✓ |

根 `index.md` 用 setext 标题"动手学深度学习"（对应 "Dive into Deep Learning"），无 `#` H1 属正常；根 toc 27 个条目与英文底本**逐项、顺序完全一致**。

### 4.2 发现与判定

1. **`chapter_builders-guide` 章名**：英文底本新标题为 "Builders' Guide"（"构建者指南"），译文沿用官方中文版旧章名"深度学习计算"（`label` 仍为 `chap_computation`；官方中文旧版同名章）。语义上可解释（本章讲模型构造/层/块等"深度学习计算"工具），且与官方中文版一致，**建议可保留**（如追求与英文新标题一一对应，可在章首译注一句）。其余 26 章与英文标题语义对应。
2. **两个附录命名**：当前为"附录：深度学习数学 / 附录：深度学习工具"，平行、简洁，且与英文 "Mathematics for Deep Learning / Tools for Deep Learning" 对应，✓（审计期间该标题经全书统一修正）。
3. **每章 index.md 的 `toc` 条目 vs 目录实际 .md**：25 章完全一致（条目数=文件数，无缺无多）；2 章为**源缺陷/源结构**，非译文问题：
   - `chapter_hyperparameter-optimization/index.md` 的 toc 条目写作 `rs-async.md`（带扩展名，英文底本同样如此）；实际文件 `rs-async.md` 存在，构建正常解析，**可保留**；
   - `chapter_references/` 无 index.md、无 toc（英文底本相同），由根 toc 直接指向 `chapter_references/zreferences`，✓。
4. 章节（含首页/附录）中文命名风格统一：线性网络章成对（用于回归/用于分类）、RNN 两章成对（循环/现代循环）、NLP 两章成对（应用/预训练）、CV/优化/性能等均为常规直译。

---

## 5. 框架标签页（`:begin_tab:` / `:end_tab:`）完整性

### 5.1 总数与配平

| 统计 | 译文 | 英文底本 |
|---|---|---|
| `:begin_tab:` 总数 | **628** | **628** |
| `:end_tab:` 总数 | **627** | **627** |
| 差值 | 1 | 1 |
| 唯一不配对文件 | `chapter_appendix-tools-for-deep-learning/d2l.md`（begin=3 / end=2） | 同左 |

`d2l.md` 的不配对是**英文底本自身缺陷**：L7 `:begin_tab:`pytorch`` 从未闭合（L15 mxnet 的开标签配 L23 的 end，L26 tensorflow 配 L35 的 end）。译文完全照搬该结构（verify 明确提示"英文源本身标签块不配对，译文保持一致即可"），**可保留，无需修改**。

### 5.2 框架名计数与逐文件序列

全书框架名计数（译文与英文底本**完全相等**）：

| 标签名 | 次数 |
|---|---|
| mxnet | 203 |
| pytorch | 187 |
| tensorflow | 125 |
| jax | 93 |
| 组合标签（如 `mxnet, pytorch, tensorflow` 等） | 21（9 种写法，逐一对应一致） |

分章统计（begin 次数，译文=英文底本逐章完全一致；"框架"为该章出现过的独立标签名）：

| 章节 | begin 数 | 框架 | 章节 | begin 数 | 框架 |
|---|---|---|---|---|---|
| appendix-mathematics | 33 | mxnet/pytorch/tensorflow | appendix-tools | 12 | mxnet/pytorch/tensorflow |
| attention-transformers | 30 | 四框架 | builders-guide | 69 | 四框架 |
| computational-performance | 65 | mxnet/pytorch/tensorflow | computer-vision | 42 | mxnet/pytorch |
| conv-modern | 32 | 四框架 | conv-neural-networks | 25 | 四框架 |
| gaussian-processes | 3 | pytorch | GAN | 4 | mxnet/pytorch |
| hyperparameter-optimization | 5 | pytorch | installation | 12 | 四框架 |
| introduction | 0 | — | linear-classification | 24 | 四框架 |
| linear-regression | 53 | 四框架 | multilayer-perceptrons | 25 | 四框架 |
| nlp-applications | 12 | mxnet/pytorch | nlp-pretraining | 14 | mxnet/pytorch |
| notation | 0 | — | optimization | 27 | mxnet/pytorch/tensorflow |
| preface | 8 | 四框架 | preliminaries | 62 | 四框架 |
| recommender-systems | 9 | mxnet | recurrent-modern | 33 | 四框架 |
| recurrent-neural-networks | 26 | 四框架 | references | 0 | — |
| reinforcement-learning | 3 | pytorch | | | |

- `verify.py` 对 192 个文件逐一比对 `:begin_tab:` 名称序列与 `:end_tab:` 数量：**0 error**，即每个文件的标签块名称、顺序、数量与英文底本一一对应；
- 章节级分布差异（如推荐系统章仅 MXNet、GP/HPO/RL 章仅 PyTorch）与英文底本**完全一致**，不是漏译/漏标签。

---

## 6. 图片与引用（含构建验证）

### 6.1 图片

- 译文内**相对图片引用 259 个，全部可在 `d2l-zh-translation/` 下解析（缺失 0）**；
- 远程图片 4 个（均在 `chapter_gaussian-processes/gp-intro.md`，`user-images.githubusercontent.com`），与英文底本完全相同；
- `img/` 目录 292 个文件；构建时 100% 复制成功（构建日志 "copying images ... [100%]"）。

### 6.2 交叉引用 / 文献 / 目录

`python work\verify.py --all`（全书级）：

- 文件 192（与英文底本一一对应，无缺文件）；
- `:label:`/`:eqlabel:` 共 617 个，**无重复**；
- `:numref:`/`:eqref:`/`:ref:` 共 1097 处，除 1 处空引用外全部可解析；
- `:cite:`/`:citet:` 610 处，**全部 bib key 存在于 `d2l.bib`**；
- toc 条目 191 条，均指向存在的文件（`rs-async.md` 条目为源缺陷带扩展名写法，实际文件存在，见 4.2）；
- **0 error；2 条 warning 均可解释为源缺陷**：
  1. `chapter_linear-regression/linear-regression-scratch.md:318`（译文）`:numref:``sec_synthetic-regression-data`` ` 是**双反引号空引用**，英文底本 L328 同样如此 → 源缺陷，可保留；
  2. 全书标签块 628/627 与英文底本一致但自身不配对（见第 5 节，`d2l.md` 源缺陷）。

### 6.3 构建

- 最终构建（14:27:04 完成，`d2lbook build html`，pytorch 标签）**成功**：185 个 HTML 页面；构建副本与当前 192 个书内文件 SHA1 逐字节一致；
- 76 条 warning **全部与译文内容无关**，分类如下：
  - **56 条 autodoc 导入失败**：构建环境未安装 `torch`/`d2l` 运行时，Sphinx autodoc 无法导入 API 类（环境问题，非译文）；
  - **9 条 toctree "nonexisting document"（推荐系统章）**：该章书稿只含 MXNet 标签页（与英文底本一致），按 pytorch 标签构建时被 d2lbook 跳过（"Skip to eval tab pytorch"）→ 属构建口径/底本结构，非译文缺陷；如需在 HTML 中呈现该章，需用 mxnet 标签构建或另行说明；
  - **5 条 undefined label**（`tab_ec2`、`tab_intro_decade`×2、`tab_glove`×2）：这些 `:label:` 挂在表格上，Sphinx 不支持对表格标签的 `:numref:` 解析；`aws.md`/`introduction/index.md`/`glove.md` 的标签与引用在英文底本中**完全相同**（如 `aws.md` 底本 L61/L64）→ 源/构建器行为，可保留；
  - **1 条 favicon 路径**（`_static\x0cavicon.png`，Windows 反斜杠被转义）→ 构建环境问题；
  - 其余为 pkg_resources 弃用等环境提示。
- 未出现任何"未定义交叉引用 / 图片缺失 / 文献键缺失 / toctree 其他章节缺失"等译文质量相关 warning。

---

## 7. 代码/公式抽查

### 7.1 随机抽 10 个文件（覆盖 8 个不同章节）`verify.py --file` 复核

| 文件 | 结果 |
|---|---|
| chapter_preliminaries/ndarray.md | PASS（errors=0, warnings=2） |
| chapter_linear-regression/weight-decay.md | PASS（0, 1） |
| chapter_multilayer-perceptrons/backprop.md | PASS（0, 1） |
| chapter_convolutional-neural-networks/pooling.md | PASS（0, 1） |
| chapter_convolutional-modern/resnet.md | PASS（0, 1） |
| chapter_recurrent-neural-networks/rnn-scratch.md | PASS（0, 2） |
| chapter_attention-mechanisms-and-transformers/transformer.md | PASS（0, 0） |
| chapter_optimization/rmsprop.md | PASS（0, 0） |
| chapter_computer-vision/anchor.md | PASS（0, 2） |
| chapter_recommender-systems/neumf.md | PASS（0, 1） |

10/10 PASS，**0 error**，11 条 warning 全部为"行内公式/行内代码内容一致但顺序不同"（中文语序重排所致，多重集一致，可接受）。

### 7.2 公式上下文人工抽看（2 个文件 × 5 处公式）

**① `chapter_optimization/adam.md`（5 处）**：
- 状态变量公式 `v_t ← β₁v_{t-1}+(1-β₁)g_t`、`s_t ← β₂s_{t-1}+(1-β₂)g_t²`：后文"β₁ 和 β₂ 是非负的权重参数…β₁=0.9，β₂=0.999…方差估计移动比动量项慢得多"——符号解释与公式一致 ✓
- 偏差修正公式 `v̂_t=v_t/(1-β₁^t)`、`ŝ_t=s_t/(1-β₂^t)`：前文解释初始化 v₀=s₀=0 造成初始偏差及 `Σβ^i=(1-β^t)/(1-β)` 的规范化依据——一致 ✓
- 缩放梯度 `g'_t=η v̂_t/(√ŝ_t+ε)`：后文说明"用动量 v̂_t 而非梯度本身""缩放用 1/(√ŝ_t+ε) 而非 1/√(ŝ_t+ε)"——一致 ✓
- 更新式 `x_t ← x_{t-1} - g'_t`：后文"显式学习率 η 允许控制步长"——一致 ✓
- Yogi 更新 `s_t ← s_{t-1}+(1-β₂)(g_t²-s_{t-1})`：后文解释用 `g_t²⊙sgn(g_t²-s_{t-1})` 替代——一致 ✓
- 5 处公式与英文底本逐字相同（`verify` 已全量保证），符号解释无错位。

**② `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md`（5 处）**：
- 高斯核注意力评分 `a(q,k_i)=-½‖q-k_i‖²=q^⊤k_i-½‖k_i‖²-½‖q‖²`：后文逐项解释"最后一项只依赖 q、对所有 (q,k_i) 相同；‖k_i‖ 有界可由批量/层规范化保证"——一致 ✓
- 缩放点积注意力 `a(q,k_i)=q^⊤k_i/√d`：前文完整译出"q、k_i 为零均值单位方差独立同分布 → 点积方差为 d → 按 1/√d 重缩放"——一致 ✓
- softmax 注意力 `α(q,k_i)=softmax(a(q,k_i))=exp(q^⊤k_i/√d)/Σ_j exp(q^⊤k_j/√d)`：前后文"注意力权重仍需规范化，用 softmax 简化"——一致 ✓
- 批量矩阵 `Q=[Q_1,…,Q_n]∈ℝ^{n×a×b}`、`K=…∈ℝ^{n×b×c}`：后文说明 BMM 计算按元素乘积、结果 `ℝ^{n×a×c}`——一致 ✓
- `BMM(Q,K)=[Q_1K_1,…,Q_nK_n]`：与上一条形状/定义对应——一致 ✓
- 公式均与英文底本逐字一致，符号说明与公式对应无错位。

> 补充：`verify` 已对全部 192 个文件的代码块（info+内容）、独立公式 LaTeX、行内公式/代码多重集做逐字比对，**0 error**，故 7.2 的人工抽看主要验证"符号解释文字与公式含义是否一致"，未发现问题。

---

## 8. 需要负责人处理的问题清单

### 阻断（必须修后才能交付）— **无**

### 应修（建议交付前处理）

| ID | 问题 | 位置/规模 | 依据与影响 | 建议 |
|---|---|---|---|---|
| S1 | 全角 slide 标记残留：`（**…**）` | **21 个文件 / 36 对**（清单见 3.2），其中 5 处在标题 | d2lbook 只识别 ASCII 标记；已造成 22 个 HTML 页面出现 `（<strong>…</strong>）` 残留；与英文底本渲染不一致；项目内 G3/G8 已按此口径修复同类问题 | 按底本逐处恢复 ASCII `(**…**)`（只改括号字形），修后 `check_marks.py` 应为 0 个问题文件，重跑构建确认残留消失 |
| S2 | 全角删除线标记：`（~~…~~）` | `chapter_preliminaries/ndarray.md:238`（1 对） | 已在 `ndarray.html` 产生**字面可见残留** `（[STRIKEOUT:或全为1的张量]）`（英文底本此处应整段删除） | 恢复为底本 `(~~or one~~)` 的 ASCII 形式（改后该句按底本语义在 HTML 中删除，或与负责人确认保留策略） |
| S3 | `(**…**)` 被改成 `[**…**]` | 2 处：`mlp.md:287`、`mlp-implementation.md:181` | HTML 渲染等价，但与底本标记类型不一致（仅 slides 语义差异） | 随 S1 一并恢复为 `(**…)` |
| S4 | `拼接` 用于 concatenate 语境 | 2 处：`bert-dataset.md:10`、`bert.md:525` | TERMINOLOGY：concatenate→连结（连结全书 80 处）；此 2 处英文均为 concatenation | 改为"连结而成"（`linear-algebra.md:880` 对应英文 stitching，保留"拼接"） |
| S5 | `掩蔽` 与全书 `掩码` 并存 | 2 处：`autorec.md:42`、`seq2seq.md:607` | 术语表 mask/masked→掩码；全书"掩码"67 处 | 统一为"掩码"；若坚持动词用"掩蔽"，请在 TERMINOLOGY 补注 |

> 合计：S1–S3 为同一类"标记字形"问题（39 处标记对），可一次性批量修复；S4/S5 共 4 处词面替换。

### 可保留（已核实为源缺陷/合法近义/构建环境，无需修改）

| 项 | 说明 |
|---|---|
| `归一化折损累积增益`（2 处） | 术语表对 NDCG 的指定全称，合法 |
| `随机丢弃`（dropout 机制白话描述）、`概念漂移`（源文 concept drift）、`平均数`（源文明确区分 mean/average） | 均为合法近义词/释义，非术语不一致 |
| hyperopt index toc 条目 `rs-async.md`（带扩展名） | 英文底本相同，文件存在、构建正常 |
| `chapter_references` 无 index.md / 无 toc | 英文底本相同，根 toc 直接指向 zreferences |
| `:numref:``sec_synthetic-regression-data`` ` 空引用 | 英文底本相同（双反引号），构建通过 |
| `d2l.md` 标签块 3 begin/2 end | 英文底本缺陷（pytorch 开标签未闭合），译文一致 |
| `chapter_builders-guide` 章名"深度学习计算"vs "Builders' Guide" | 沿用官方中文版旧章名（label `chap_computation`），语义可解释；如追求严格对应可加章首译注 |
| 构建 76 warnings | 56 autodoc 环境导入、9 推荐系统章在 pytorch 标签构建下跳过（书稿仅 MXNet 实现，底本相同）、5 表格标签（Sphinx 不支持，源文相同）、1 favicon 路径；**均与译文质量无关** |

### 附注（流程/交付）

1. **并发编辑提示**：审计期间（14:17–14:27）其他审校组仍在改稿（79/199 个 md 被修改），本报告数字已按最终状态重跑；如交付前再有修改，建议重跑：`python work\verify.py --batch work\batches\done.txt`、`python work\verify.py --all`、`python work\check_marks.py` 并重新构建。
2. **新增 `verification/` 目录**：14:26 起 `d2l-zh-translation/verification/` 含审校报告副本（非书稿）。它不在书稿 toc/构建输入内，但会让遍历整树的脚本（termcheck/check_marks 等）误纳入其引用的标记样例；建议交付说明中注明或让相关脚本排除该目录（本次审计已按 192 个书内文件口径统计）。
3. **复现**：本报告所有检查均只读，未改动任何译文；命令与口径见各节，工具输出中的数字即本报告引用值。

---

### 审计结论

全译本 **结构完整、术语高度收敛、无漏译/无未翻译正文、无标记 self-imbalance、无阻断问题**；剩余需处理的是 **39 处 slide 标记字形**（会体现在构建 HTML 的可见残留）与 **4 处术语词面统一**（拼接/掩蔽），均为低风险机械修改；其余观察项均可解释为英文底本自身缺陷、合法的专名/示例英文，或构建环境因素。完成上述 S1–S5 后，建议重跑一次构建与 `verify --all` 即可交付。
