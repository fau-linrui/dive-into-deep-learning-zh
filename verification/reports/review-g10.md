# 审校组 G10 审校记录（批次 b19）

审校范围：`work/batches/batch-b19.txt` 全部 11 个文件（1 篇自然语言处理应用 + 4 篇强化学习 + 4 篇高斯过程 + 2 篇超参数优化）：

- `chapter_natural-language-processing-applications/natural-language-inference-bert.md`
- `chapter_reinforcement-learning/index.md`、`mdp.md`、`value-iter.md`、`qlearning.md`
- `chapter_gaussian-processes/index.md`、`gp-intro.md`、`gp-priors.md`、`gp-inference.md`
- `chapter_hyperparameter-optimization/index.md`、`hyperopt-intro.md`

说明：任务描述中提到的“NLP 应用索引”实际在 `work/batches/batch-b18.txt`（属其他批次/组），不在本批次清单内，故未处理；本记录只覆盖 batch-b19.txt 的 11 个文件。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照（代码块、独立/行内公式、指令、图片路径、URL、表格结构、标题层级只做一致性核验，未改动）。重点核对强化学习（状态/动作/奖励/策略/价值函数/动作价值函数/贝尔曼方程（动态规划原理）/折扣因子/回报）与高斯过程（先验/后验/核函数/协方差函数/边际似然/预测分布/认知与偶然不确定性）的公式符号解释与推导，以及术语与 `TERMINOLOGY.md` 的一致性。只做定向 `edit`，每次修改后 `python work\verify.py --file <file>` 复核。

基线/复核：审校前 `python work\verify.py --batch work\batches\batch-b19.txt` = 11/11 PASS，0 error，9 warning（均为“内容一致但顺序不同”，中文语序重排所致）；审校后仍为 11/11 PASS，0 error，9 warning，警告文件与条数与基线完全相同。`python work\align_check.py --batch work\batches\batch-b19.txt` 只对 `value-iter.md` 报“结构单元数不同（源 53/译 54）”，经逐段人工比对确认系源文 L65“观察”后紧跟独立公式无空行、译文在两者间补了一个空行（Markdown 渲染不变）造成的对齐错位，**不存在漏译**。`python work\termcheck.py`（只读）未在本批 11 个文件中发现任何术语变体（其报告的变体全部位于其他章节）。

## 修改清单（2 个文件、3 处）

1. `chapter_gaussian-processes/gp-intro.md`（L132，小结）：原句“这使我们能够推理**解的高层性质**”不成句（“推理解”连读、缺谓语），英文为 “which enables us to reason about the high-level properties of the solutions”，改为“这使我们能够**对解的高层性质进行推理**”。
2. `chapter_gaussian-processes/gp-inference.md`（L267）：`noise-free function` 原译“**无声函数**”（soundless function）系误译，同文其余各处均作“无噪声函数”，改为“高斯过程表示的**无噪声函数**加上高斯噪声”。
3. `chapter_gaussian-processes/gp-inference.md`（L224）：英文 “…we are taking two times the square root of our variance estimate **for the noise free function**” 中的限定语漏译，补为“我们是对**无噪声函数的**方差估计取 _两倍_ 的 _平方根_”，与上一段“认知不确定性由潜在无噪声函数的方差刻画”一致。

## 逐文件记录

### 自然语言处理应用

- `natural-language-inference-bert.md`：**通读通过，无修改**。premise/hypothesis 作“前提/假设”（NLI 语境，首现括注英文）；片段标识（segment IDs）、`SNLIBERTDataset` 截断逻辑（`max_len-3`、反复移除较长序列末词元）、`MaskLM`/`NextSentencePred` 陈旧梯度（`ignore_stale_grad=True`）说明均与底本一致；5 处 `[**...**]` 高亮范围与底本一一对应；练习 2 题完整；1 条行内代码顺序 warning 为中文语序重排，与基线相同。

### 强化学习

- `index.md`：**通读通过，无修改**。术语统一：强化学习、决策/动作、马尔可夫决策过程（MDP）、值迭代（value iteration）、Q 学习（Q-learning，首现括注英文）；作者机构“宾夕法尼亚大学与亚马逊”；`toc` 3 节与底本一致。疑点见下（底本 controlled/uncontrolled 矛盾）。
- `mdp.md`：**通读通过，无修改**。逐条核对状态集合 $\mathcal{S}$、动作集合 $\mathcal{A}$、转移函数 $T(s,a,s')=P(s'\mid s,a)$ 及归一化条件、奖励 $r(s,a)$、轨迹 $\tau$、回报（return，首现括注）、折扣回报 $\sum\gamma^t r_t$（$\gamma<1$）、马尔可夫假设（加速度机器人反例、状态取 $(\text{location},\text{velocity})$ 后为马尔可夫）、小结元组 $(\mathcal{S},\mathcal{A},T,r)$；4 个独立公式逐字保留；练习（MountainCar 3 子题、Pong）完整。1 条行内公式顺序 warning 为语序重排。
- `value-iter.md`：**通读通过，无修改**。随机策略 $\pi(a\mid s)$（确定性策略为特例）、价值函数 $V^\pi$ 的两阶段分解与 `eq_dynamic_programming`、动作价值 $Q^\pi$ 及其类似式、最优随机策略 $\pi^*$、动态规划原理（Bellman，1950s，“最优轨迹的剩余部分也是最优”）、值迭代更新与收敛性、策略评估；10 个独立公式逐字保留；$k^{\textrm{th}}$ 保留；FrozenLake 环境描述与底本一致（底本 “down” 对应的 $\rightarrow$ 疑为笔误，按底本逐字保留）。`align_check` 的结构单元差见上文说明，非漏译。
- `qlearning.md`：**通读通过，无修改**。Q 学习更新式、优化问题 `q_learning_optimization_problem`、梯度下降更新与终止状态指示变量、数据集外状态-动作值设为 $-\infty$、探索策略与 $\epsilon$-贪心（首现括注）、softmax 探索与温度 $T$、“自我纠正”性质、DQN 与复兴；6 个独立公式逐字保留；代码块及注释（含底本拼写 “Learing rate”“epsilion gready”）原样保留；不可见 HTML 注释已中译。疑点见下（“上一章”）。

### 高斯过程

- `index.md`：**通读通过，无修改**。GP 的三点重要性（函数空间视角、主动学习/超参数学习/auto-ML/时空回归、GPyTorch 与深度学习互补）、协方差函数/核函数、先验/后验/推断笔记本的导览与底本一致；`toc` 3 节与底本一致。
- `gp-intro.md`：**1 处修改（见上）**。其余核对：GP 定义与联合多元高斯 `eq_gp_prior`；条件后验 $f(x)\mid f(x_{1:n})\sim\mathcal{N}(m,s^2)$ 的 $m$、$s^2$ 公式及 $1\times n$ 向量/$n\times n$ 矩阵解释；RBF 核 $a^2\exp(-\frac{\|x-x'\|^2}{2\ell^2})$ 与振幅/长度尺度含义；认知不确定性、可约不确定性、95% 置信集、无噪声观测与 $\delta_{ij}\sigma^2$ 噪声项；全部独立公式与图片替代文本、图题保留；练习 7 题完整。
- `gp-priors.md`：**通读通过，无修改**。定义（任意有限多个随机变量联合高斯）、`eq_gp-function`（底本无 `$` 的 `f(x)` 按源保持纯文本）、权重空间→函数空间、$m(x)=0$ 与 $k(x,x')=1+xx'$ 的推导、RBF 核由权重空间推导（黎曼和→积分→$\propto k_{\textrm{RBF}}$）、通用近似器、过参数化、平稳/平移不变、神经网络核（Neal 1994/1996、神经正切核、erf 公式）；全部公式逐字保留；小结 4 段、练习 5 题完整。首句个别措辞偏拗口，列为疑点未改。
- `gp-inference.md`：**2 处修改（见上）**。其余核对：观测模型 `eq_gp-regression`、$\mathbf{y}$ 与 $\mathbf{f}_*$ 的联合分布推导、Gaussian identities（Bishop 第 2 章）求条件分布、边际似然与 Occam 剃刀、学习+预测公式（$a_*$、$v_*$）、五点解读（闭式精确推断、线性组合权重、不确定性随远离目标数据增大、拟合项/复杂度项、$\mathcal{O}(n^3)$/$\mathcal{O}(n^2)$ 瓶颈与 Cholesky/条件数/jitter）、从头实现全流程（长度尺度 0.299、噪声标准差 0.24）、认知/偶然不确定性、以两倍平方根构造 95% 置信集、GPyTorch 的 Adam/边际似然流程、平方后的长度尺度与噪声方差 $0.81\approx0.283^2$；全部独立公式逐字保留；练习 6 题完整。多处底本自身前后不一致已列为疑点。

### 超参数优化

- `index.md`：**通读通过，无修改**。超参数控制学习算法或底层统计模型结构、试错/默认值导致次优泛化、“把问题转化为优化问题”、嵌套优化（每次迭代需训练+验证）、廉价代理（proxy）、`toc`（`rs-async.md` 带扩展名照源保留）均与底本一致。
- `hyperopt-intro.md`：**通读通过，无修改**。目标函数 $f:\mathcal{X}\to\mathbb{R}$、$\mathbf{x}_\star\in\argmin f$、超梯度（近似）、噪声观测 $y\sim f(\mathbf{x})+\epsilon$、配置空间/搜索空间、`loguniform`（对数空间 -4 与 -1）、闭有界范围与先验分布、MLP 示例配置空间表格（列数、对齐行、6 个超参数行逐格核对）、层数依赖 $l+1$、随机搜索算法与缺点、网格搜索/维数灾难；练习 1–3（含第 2 题子题编号 1,2,3,5,6 按源保留跳号）完整；2 条 warning（行内公式、行内代码顺序）均为语序重排。

## 遗留疑点（未擅改，提请负责人裁决）

1. **`chapter_reinforcement-learning/index.md` L11 底本自相矛盾**：`uncontrolled variables ... (in RL, these controlled variables are called the environment)`，同一括注先说不可控、再说“这些可控变量”。译文按底本直译“不可控变量（……这些可控变量称为环境）”，未擅自改义。建议定稿前核对英文底本，统一为“不可控变量（……这些变量称为环境）”或在译注中说明。
2. **`qlearning.md` L23 “与上一章一样”**：底本作 “Just like the previous chapter”，但上下文实际指上一节（值迭代一节）的数据收集记法。现按底本直译，建议统一改为“与上一节一样”（属引用对象笔误，非技术内容）。
3. **`gp-inference.md` L24 的 `$x^2$`**：底本 “Then we want to find $x^2$ and $p(\mathbf{f}_*\mid\mathbf{y},X)$” 中 `$x^2$` 疑为 `$\mathbf{f}_*$` 的笔误，按底本逐字保留；建议负责人决定是否修正或加译注。
4. **`gp-inference.md` L43 $S_*$/$S$ 不一致**：条件分布写作 $\mathcal{N}(m_*,S_*)$，但紧接着定义的是 $S$；底本原样，未改。
5. **`gp-intro.md` L118–124 前后点数不一致**：正文“假设我们观测到 $f(x_1), f(x_2)$”，其后公式却用 $x_{1:3}$（三点）；底本原样，未改。
6. **`gp-inference.md` L161 vs 代码 L164–165**：正文说“猜测噪声标准差为 0.75”，代码实际 `post_sig_est = 0.5`；底本原样，未改。
7. **`gp-inference.md` L184 vs L332 数值不一致**：前文学到“噪声标准差为 0.24”，后文对比 GPyTorch 时作“约 0.283”；底本原样（0.81≈0.283² 自洽，但与 0.24 冲突），未改。
8. **`gp-inference.md` L91 $\mathcal{O}(n^2)$ 存储表述**：底本 “$\mathcal{O}(n^2)$ storage for each entry of the kernel (covariance) matrix” 字面有矛盾（应为整个矩阵合计 $\mathcal{O}(n^2)$ 存储）。译文按底本直译为“$\mathcal{O}(n^2)$ 的存储来保存核（协方差）矩阵的每个元素”，如要求技术精确可改译为“保存整个核矩阵需要 $\mathcal{O}(n^2)$ 存储”（属技术内容修正，未擅改）。
9. **credible interval / credible set 译名（术语取舍）**：`gp-intro.md` L23 把 credible interval 译作“置信区间”，而本批其余各处 credible set 统一作“置信集”，且 `gp-inference.md` L222 明确将 confidence interval 与 credible set 对举为“置信区间与置信集”。中文统计文献通常用“置信区间”专指 confidence interval（TERMINOLOGY 也收录 `confidence interval → 置信区间`），建议统一裁决（例如 credible interval → “可信区间”、credible set → “可信集”，或全章统一“置信集/置信区间”并加注）。本组未改，避免与其他章节已定译名冲突。
10. **`gp-priors.md` L8 首句措辞**：“理解高斯过程（GP）对于推理模型构造与泛化、以及……都非常重要”中“推理模型构造”易误读为“推理模型（inference model）”，可改为“对模型构造与泛化的推理”；属表达层面，按“不过度修改”原则保留，供负责人参考。
11. **`gp-inference.md` L332 “GPyTorch 使用 _平方后的_ 长度尺度和观测噪声”**：可读作只修饰长度尺度；下一句已用“噪声方差 = 0.81 ≈ 0.283²”澄清，未改。
12. **建议合入 `TERMINOLOGY.md` 的新术语**（本批已用且内部一致）：回报（return）、转移函数（transition function）、轨迹（trajectory）、随机策略（stochastic policy）、确定性策略（deterministic policy）、策略评估（policy evaluation）、探索参数（exploration parameter）、$\epsilon$-贪心（$\epsilon$-greedy）、自我纠正（self-correcting）、认知不确定性（epistemic uncertainty）、偶然不确定性（aleatoric uncertainty）、可约/不可约不确定性（reducible/irreducible uncertainty）、置信集/置信区间（credible set/interval，待裁决）、振幅（amplitude）、长度尺度（length-scale）、均值函数（mean function）、通用近似器（universal approximator）、平稳性（stationarity）、平移不变性（translation invariance）、过参数化（overparametrized）、配置空间（configuration space）、搜索空间（search space）、神经架构搜索（NAS）、网格搜索（grid search）、超梯度（hypergradient）、点过程（point process）、条件数（conditioning）、抖动（jitter）。

## 说明

- 本组只修改了自己负责的 2 个文件、3 处定向替换；未改动任何代码块、行内代码、独立/行内公式、`:label:`/`:eqlabel:`/`:numref:` 等指令、图片路径、链接 URL、标题层级、表格结构，未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md` 及其他批次/组文件。
- 审校后复核：`python work\verify.py --batch work\batches\batch-b19.txt` = 11/11 PASS（0 error、9 warning，与审校前完全一致）；被改动的 `gp-intro.md`、`gp-inference.md` 单文件复核均 0 error。
