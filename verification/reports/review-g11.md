# 审校记录：组 G11（批次 b20）

- 审校范围：`work/batches/batch-b20.txt` 全部 13 个文件（超参数优化 4 篇 + 生成对抗网络 3 篇 + 推荐系统 6 篇）：
  - `chapter_hyperparameter-optimization/hyperopt-api.md`、`sh-intro.md`、`sh-async.md`、`rs-async.md`
  - `chapter_generative-adversarial-networks/index.md`、`gan.md`、`dcgan.md`
  - `chapter_recommender-systems/index.md`、`recsys-intro.md`、`movielens.md`、`mf.md`、`autorec.md`、`ranking.md`
- 底本：`d2l-en-src/`（只读）；术语依据：`d2l-zh-translation/TERMINOLOGY.md`；样例依据：`work/EXAMPLES.md`。
- 方法：逐文件通读中文译文；英文底本按“英文段—中文段”逐段对照（公式/定义/因果与限定词密集段落逐句核对，含公式符号解释、变量上下标与维度、图题与图片 alt、练习子题；叙述段通读并抽查）；代码块、独立/行内公式、指令、图片路径、URL、标题层级、表格结构只做一致性核验，未改动。只用 `edit` 做定向修改，每次修改后 `python work\verify.py --file <path>` 复核。辅以 `verify.py --batch`、`align_check.py --batch`（段级漏译）、`termcheck.py`（只读术语变体）、数字/符号多重集比对。

基线/复核：

- 审校前 `python work\verify.py --batch work/batches/batch-b20.txt` = **13/13 PASS，0 error，8 warning**；审校后仍为 **13/13 PASS，0 error，8 warning**，警告文件与条数不变（hyperopt-api 1、sh-intro 1、rs-async 1、gan 1、movielens 1、mf 2、ranking 1，全部为“行内公式/行内代码内容一致但顺序与原文不同”）。8 处已逐条人工确认：均为中文语序重排（如 `LeNet` 与 `:numref:`、`DataLoader` 与 `last_batch`、`\mathbf{P}` 与 `u^\textrm{th}`、u/i/j 的先后），内容多重集与底本完全一致，**非漏译、非误改**。
- `python work\align_check.py --batch work/batches/batch-b20.txt` = **0 个段级疑点文件**（无段落级漏译）。复核后已用全库模式重新生成 `work/reports/align.json`，恢复为全书报告。
- `python work\termcheck.py`（只读）：本批未发现新增术语变体；唯一相关项是 `autorec.md` 的“掩蔽”与全书“掩码”并存（见疑点 3，属全书级取舍，未擅改）。
- 批量核对数字/百分比/维度/轮数等 token 多重集，仅 3 处差异均为中文数词写法（“one trial”→“1 个试验”、“2D”→“二维”、“to one”→“设为 1”），无数字漏译或篡改。

## 修改清单（6 个文件、8 处）

1. `chapter_hyperparameter-optimization/rs-async.md` L24：全角括号后多了一个空格（`（ :numref:`）→ 去掉，改为 `（:numref:`distributed_scheduling` 中的灰色区域）`。
2. `chapter_hyperparameter-optimization/rs-async.md` L42：`在本文中，我们将考察异步随机搜索` → `在本笔记本中，…`。英文为 “In this notebook”，`TERMINOLOGY.md` 定 notebook→“笔记本”，且 `gp-inference.md`/`gp-intro.md` 同书用例均作“本笔记本”，原译“本文”不合术语表。
3. `chapter_hyperparameter-optimization/rs-async.md` L213：`如果我们有掉队进程——…` → `如果我们有掉队进程（straggler）——…`。英文底本此处首次以正文引入 straggler；译文完成报告称已“首次括注”，但正文实际没有英文括注，在此补注，破折号后的释义保留。
4. `chapter_hyperparameter-optimization/sh-async.md` L25：`…这意味着由于掉队进程而导致工作进程空闲时间。` 句子缺谓语中心、读不通（对应 “which implies idle worker time due to stragglers”），改为 `…这意味着掉队进程会造成工作进程的空闲时间。`
5. `chapter_generative-adversarial-networks/dcgan.md` L222：`如果把转置卷积层改为 $4\times 4$ 的核、$1\times 1$ 的步幅和零填充。在输入大小为 $1 \times 1$ 时，…` 前一分句无主句（英文底本亦为残句），合并为 `如果把转置卷积层改为 …，那么在输入大小为 $1 \times 1$ 时，输出的宽度和高度将分别增加 3。`（技术内容、公式逐字未变）
6. `chapter_recommender-systems/recsys-intro.md` L27：`根据隐式反馈为每个用户个人化地对所有物品进行排序` → `…为每个用户分别对所有物品进行排序`。“个人化”非规范译法（personalized 的通行译名为“个性化”，同章已多处使用；此处 personally 意为“逐一/分别”）。
7. `chapter_recommender-systems/mf.md` L3–4：`当时，媒体流媒体和视频租赁公司 Netflix 宣布…` → `当时，流媒体和视频租赁公司 Netflix 宣布…`（“媒体流媒体”语义重复，源文 media-streaming）。
8. `chapter_recommender-systems/autorec.md` L3：`…复杂非线性关系和错综复杂的关系。` → `…复杂非线性关系和错综复杂的关联。`（“关系”重复，源文 “complex nonlinear and intricate relationships”）。

## 逐文件记录

### 超参数优化

- `hyperopt-api.md`：**通读通过，无修改**。核对：searcher/scheduler/tuner 三个决策组件（搜索器、调度器、调优器），`HPOSearcher`/`HPOScheduler`/`HPOTuner` 类名与 `sample_configuration`/`suggest`/`update`/`runtime`/`cumulative_runtime`/`incumbent_trajectory` 标识符；trial→试验、*incumbent* 首现括注（译名见疑点 1）、“any-time performance”→任意时刻性能、“wall-clock time”→实际时间；配置空间与第一个评估配置、`HPOTrainer`/`apply_init` 目标函数、50 次重复的均值/标准差比较、练习 1 的 3 个子题（`num_hiddens_1/2` 的 $[8,1024]$、暂退法 $[0,0.95]$、`batch_size` $[16,384]$）与练习 2 的 `LocalSearcher` 概率分支（两次“与 RandomSearcher.sample_configuration 相同”）均逐句对应。1 条行内代码顺序 warning 为语序重排。
- `sh-intro.md`：**通读通过，无修改**。核对：多保真度设定 $f(\mathbf{x},r)$、$r\in[r_{\mathrm{min}},r_{max}]$、误差随 $r$ 减小/成本随 $r$ 增大；SH 轮次（$N=\eta^K$、保留 $1/\eta$、每轮 $\eta$ 倍预算、仅一个试验到 $r_{max}$）、梯级集合 $\mathcal{R}$、队列与 `prefact`/`K`/`ki`/`ni` 的计算、$r_{\mathrm{min}}=2$、$\eta=2$、$r_{max}=10$ → 梯级 2,4,8,10、两幅图图注（英文底本两处图注相同，照译，见疑点 5）、结尾“工作进程可能空闲”的复杂度说明。1 条 warning 为语序重排。
- `sh-async.md`：**修改 1 处（清单 4）**。其余核对：同步 SH 的同步点与掉队、ASHA“当前梯级收集到至少 $\eta$ 个观测即提升”、Trial-0…Trial-5 与 Worker-0/1 的提升顺序（逐试验核对，η=2 的推理一致）、Syne Tune 的 `metric`/`mode`/`resource_attr`/`max_resource_attr`/`grace_period`/`reduction_factor` 说明、`max_epochs` 固定启动 vs. 提前停止的差别、暂停—恢复与检查点、约 12 分钟运行、两幅图图注（sync_sh/asha）。
- `rs-async.md`：**修改 3 处（清单 1–3）**。其余核对：同步/异步并行 HPO 与灰色区域同步点、线性加速（$K$ 个试验并行则快 $K$ 倍）、Syne Tune 目标函数与 `report` 回调、配置空间传常量（`max_epochs`）、`PythonBackend`、`RandomSearch`、`StoppingCriterion`/`Tuner`、工作进程空闲最小化、练习 1–2（含 4 个 CPU 核心、`n_workers=1/2/4`、`BayesianOptimization`、`LocalSearcher` 与 d2lbook/syne-tune 链接）；图 alt 长句整段译出、URL 原样。

### 生成对抗网络

- `index.md`：**通读通过，无修改**。标题“生成对抗网络”、`:label:` 与 `toc`（gan、dcgan）与底本一致。
- `gan.md`：**通读通过，无修改**。核对：判别式学习/生成式建模、two-sample test 定义与 $X,X'$ 公式、$D(\mathbf x)=1/(1+e^{-o})$、判别器最小化交叉熵、生成器先最大化 $\log(1-D(G(\mathbf z)))$ 后改为最小化 $-\log(D(G(\mathbf z)))$ 的动机（损失接近 0→梯度过小）、极小极大目标、$\mathbf z\sim\mathcal{N}(0,1)$ 潜在变量、$A^TA$ 协方差与均值 $b$、`update_D`/`update_G` 与 “假数据标签 0→1” 说明、5 个独立公式与小结/练习公式逐字、`*即*`/`*例如*` 斜体标记。1 条行内公式顺序 warning 为语序重排（输入 $\mathbf x$ 与输出 $o\in\mathbb R$ 的先后），语义不变。
- `dcgan.md`：**修改 1 处（清单 5）**。其余核对：64×64 调整、`ToTensor` 值域 $[0,1]$ 与 tanh $[-1,1]$ 的 Normalize(0.5,0.5) 匹配、G_block/D_block（转置卷积/卷积 + 批量规范化 + ReLU/leaky ReLU）、转置卷积与判别器两个 `\begin{aligned}` 尺寸公式逐字、生成器四块通道 64×8→64→3、判别器镜像结构 + 输出 1 通道、leaky ReLU 定义与 $\alpha\in[0,1]$ 说明（“dying ReLU”）、Adam $\beta_1$ 0.9→0.5、`n_G`/`n_D` 与三框架代码块、小结 3 条与练习 2 题。

### 推荐系统

- `index.md`：**通读通过，无修改**。作者行（Shuai Zhang / Aston Zhang / Yi Tay，机构保留）、正文两段、`toc` 10 项与底本一致。
- `recsys-intro.md`：**修改 1 处（清单 6）**。其余核对：物品（item 首现括注）、协同过滤术语来源（Tapestry 引文）、基于记忆/基于模型/混合、最近邻与矩阵分解（潜在因子）、显式/隐式反馈（IMDb 十星、YouTube 点赞点踩、含噪性）、推荐任务（评分预测、Top-$n$ 推荐/物品排序、序列感知推荐、点击率预测、冷启动推荐）、小结与 2 题练习。
- `movielens.md`：**通读通过，无修改**。核对：MovieLens 100K（943 用户、1682 电影、10 万条评分、1–5 星、每用户≥20 部）、稀疏度定义与 93.695%、`random`/`seq-aware` 两种划分（时间戳排序、最近一部留作测试）、`read_data_ml100k`/`split_data_ml100k`/`load_data_ml100k`/`split_and_load_ml100k` 与代码注释、“测试集可视为留出的验证集”、`last_batch='rollover'` 说明、小结与练习。1 条行内代码顺序 warning 为语序重排。
- `mf.md`：**修改 1 处（清单 7）**。其余核对：Simon Funk 博客、2006 Netflix 竞赛/Cinematch/10%/百万美元、BellKor's Pragmatic Chaos、交互矩阵与评分矩阵可互换、潜在矩阵 $\mathbf{P}$/$\mathbf{Q}$ 与 $\mathbf{p}_u$/$\mathbf{q}_i$ 含义、$\hat{\mathbf{R}}=\mathbf{PQ}^\top$ 与偏置式、目标函数与 $\lambda$ 正则、$\mathcal{K}$ 集合（`is known` 属公式内容照源保留）、`nn.Embedding` 实现、RMSE 定义与 `mx.metric.RMSE`、$\ell_2$ 损失+权重衰减、潜在因子 30、用户 20/物品 30 预测示例。英文底本 L30–31 行内公式跨行未闭合（底本缺陷），译文保留同样跨行布局，渲染正常且 `verify.py` 0 error；2 条 warning 为公式/代码顺序重排。
- `autorec.md`：**修改 1 处（清单 8）**。其余核对：AutoRec 与 CF 结合、编码器/解码器/重构层、$h(\mathbf{R}_{*i})=f(\mathbf{W}\cdot g(\mathbf{V}\mathbf{R}_{*i}+\mu)+b)$、重构误差目标与 $\|\cdot\|_{\mathcal{O}}$（只更新与已观测输入关联的权重）、dropout=0.05、`pred * np.sign(input)` 掩蔽梯度、重新实现的评估器（RMSE）、MovieLens 训练（隐藏维 500、lr 0.002、25 轮、wd 1e-5）、“测试 RMSE 低于矩阵分解”。
- `ranking.md`：**通读通过，无修改**。核对：逐点/成对/列表式、BPR 后验推导与 BPR-OPT 五个等号逐字、$D$/$I^+_u$/$I$ 集合与 $\hat{y}_{ui}$ 解释、先验 $\mathcal{N}(0,\Sigma_\Theta)$ 与 $\Sigma_\Theta=\lambda_\Theta I$、图 rec-ranking、`BPRLoss` 实现、合页损失式与 $m$（安全间隔）、`HingeLossbRec`、“两种损失可互换”、小结 2 条与练习 2 题。1 条 warning 为公式顺序（u/i/j 语序），语义一致。

## 需负责人裁决的疑点（未擅改）

1. **incumbent 译名与术语表冲突（术语取舍）**：`TERMINOLOGY.md` 定 incumbent→“现任最优配置”，而 `hyperopt-api.md`（L145、L213、L236）与 `rs-async.md`（L176、L225）统一作“当前最优配置（incumbent）”，译文完成报告（`work/reports/b20.md` §3）也把“incumbent 当前最优配置”列为待合入术语。折中方案：若保留现译，请更新术语表备注；若按表强制，本组仅这 2 个文件 5 处需替换（`hyperopt-api.md` 3 处、`rs-async.md` 2 处）。“现任最优配置”较生硬，且与表中另一条 current best configuration→当前最优配置 易混，建议优先考虑前者（更新术语表）。
2. **worker 译名与术语表冲突（术语取舍）**：表定 worker→“工作节点”，而超参数优化章 `sh-intro.md`（4 处）、`sh-async.md`（12 处）、`rs-async.md`（7 处）统一作“工作进程”，`parameterserver.md` 则用“工作节点”。本章 Syne Tune 语境确为同一台机器上的多个 Python 进程（`n_workers`），“工作进程”更准确且章内一致。请裁决：统一为“工作节点”，或在术语表中为 HPO/分布式调度语境加“工作进程”备注（本组意见偏后者）。
3. **“掩蔽”/“掩码”全书变体**：`autorec.md` L42“未观测输入的梯度被掩蔽”（译者刻意“掩蔽梯度”，见 b20 报告），全书其余 60 处用“掩码”（如 masked language model、掩码 softmax），另有 `seq2seq.md` 1 处“掩蔽”。属全书级统一问题且牵涉非本组文件，未改；建议以术语表 masked softmax→“掩码 softmax 运算”为准统一，或在表中补“mask→掩码（名词）/掩蔽（动词）”的用法说明。
4. **straggler 首现括注位置**：本组已在 `rs-async.md` L213 正文首现处补“（straggler）”，但全书该词视觉首现其实是 `rs-async.md` L39 的图片 alt（“在存在掉队进程时…”）。若要求“首次出现即括注”，需在图注补注（图注加英文括注略违常例，故未加，供裁决）。
5. **英文底本自身缺陷（按底本处理，未擅改技术内容）**：
   - `sh-intro.md` L26/L81 两幅图（`samples_lc.svg` 与 `sh.svg`）英文图注完全相同（疑底本复制错误），译文照译。
   - `hyperopt-api.md` “We being by defining…”（应为 begin），译文按“我们首先定义”处理。
   - `gan.md` “…too small to make good progress for the discriminator”（逻辑上疑应为 generator），按底本直译。
   - `dcgan.md` 小结 “GANs composes of two deep networks” 语法错误；L222 条件残句（中文已顺为条件句，未改技术内容）；“given a input shape”等语法瑕疵按意译。
   - `rs-async.md` “asynchronous random search that, where trials…” 多余 that；`sh-async.md` “implementaitons”“ideling” 拼写错误：译文均按正确语义处理。
   - `mf.md` L30–31 行内公式跨行未闭合（底本缺陷），译文保留同样跨行布局，`verify.py` 0 error、渲染正常（处理方式见 `work/reports/b20.md` §4-1）。
6. **`gan.md` L43 estimator 译“估计器”**：术语表 estimator→“估计量”属统计语境，此处“高斯参数估计器”指构造的估计装置/模型，非统计估计量，按上下文处理，未改，供参考。

## 建议合入 `TERMINOLOGY.md` 的新术语（本批已用且内部一致，供负责人参考）

搜索器（searcher）、调度器（scheduler）、任意时刻性能（any-time performance）、实际时间（wall-clock time）、多保真度超参数优化（multi-fidelity HPO，表中已有 multi-fidelity optimization）、减半常数（halving constant）、梯级水平（rung level）、掉队进程（straggler）、潜在矩阵（latent matrix）、交互矩阵（interaction matrix）、评分矩阵（rating matrix）、评分预测（rating prediction）、稀疏度（sparsity）、判别式学习（discriminative learning）、生成式建模（generative modeling）、安全间隔（safety margin）、物品排序（item ranking）。其中 incumbent、worker、掩蔽/掩码三项见疑点 1–3，待裁决后再定稿。
