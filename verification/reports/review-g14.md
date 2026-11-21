# 审校组 G14 审校记录（批次 b23）

审校范围：`work/batches/batch-b23.txt`（6 个文件，5 个数学附录 + 1 个工具附录索引，公式密集）：
`chapter_appendix-mathematics-for-deep-learning/` 下的 `maximum-likelihood.md`（最大似然）、`distributions.md`（分布）、`naive-bayes.md`（朴素贝叶斯）、`statistics.md`（统计学）、`information-theory.md`（信息论），以及 `chapter_appendix-tools-for-deep-learning/index.md`（工具附录索引）。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照；本批公式、分布性质与推导极多，数学段落逐句核对，重点为：①公式前后变量/符号解释是否与公式对应（如最大似然的 $\hat{\boldsymbol\theta}$、分布的 $\mu_X/\sigma_X^2$、指数族的 $h/\boldsymbol\eta/T/A$、信息论各量的下标与期望对象）；②推导步骤是否完整（9/13 的求导三解与极值判断、二项→泊松极限、标准化+中心极限定理、偏差-方差六步恒等式、$H(X,Y)=H(X)+H(Y|X)$ 系列、$I(X,Y)$ 的三种等价、CE 与 KL 的关系、二分类/多类最大似然推导）；③限定词与条件（“如果独立”“在极限下”“显然不是”等）是否保留；④术语按 `TERMINOLOGY.md`（似然/先验/后验/熵/交叉熵/KL 散度/互信息/最大似然/拉普拉斯平滑/抽样/采样/估计量/偏差-方差/独立同分布/置信区间等）；⑤图题、图片替代文本、练习子题、小结是否完整；⑥中文是否自然。只做定向 `edit`；公式、代码块、指令、图片路径、URL、标题层级、表格结构只做结构核验、未改动；每次修改后 `python work\verify.py --file <path>` 复核。

基线：`python work\verify.py --batch work/batches/batch-b23.txt` = **6/6 PASS，0 error，8 warning**（5 个文件各 1 条“行内公式内容一致但顺序不同”；distributions 另有“疑似未翻译英文行 3 处”；naive-bayes、statistics 各多 1 条“行内代码顺序不同”）。`python work\align_check.py --batch work/batches/batch-b23.txt` = 1 个段级疑点（distributions 指数族四条列表项），经人工核对为跨行行内公式的 LaTeX 记号被计入英文词数的**假阳性**（四条列表项与底本逐项对应，无漏译）。审校后批次仍为 **6/6 PASS、0 error、8 warning**（警告数与基线一致，均系中文语序重排/跨行公式造成，已逐条确认多重集一致）。

## 修改清单（4 个文件、12 处）

1. `maximum-likelihood.md`（L10）：底本 `the most likely value for the parameters` 原译“如果我想找到模型参数最可能取值”缺结构助词，改为“模型参数的最可能取值”。
2. `maximum-likelihood.md`（L21）：底本 `P(\boldsymbol{\theta}) does not depend on theta either!` 中裸写的 `theta` 原译“同样不依赖于theta”，中英混排不成词；按语义改为“同样不依赖于该参数”（**未**新增 `$\theta$` 行内公式，以免与底本公式多重集不一致；底本此处本身疑为笔误，见遗留疑点 4）。
3. `distributions.md`（L761–763）：指数族“自然参数进入新测度的方式”原译“把这些参数与某个函数 $T(\cdot)$ 作用于 $\mathbf{x}=...$，然后再取指数”——句法不通且漏译 `dot product`（点积）；改为“……与某个函数 $T(\cdot)$ 在 $\mathbf{x}=...$ 处的取值作点积，然后再取指数”。（行内公式一个未动、物理顺序不变。）
4. `distributions.md`（L774）：底本 `univariate variable` 原译“一元变量”语义冗余，改为“单变量”。
5. `distributions.md`（L457）：泊松分布 $\lambda$ 的解释原译“它表示我们在一个单位时间内期望的平均到达数”语序拗口，改为“它表示我们在单位时间内预期会看到的平均到达数”。
6. `statistics.md`（L131）：底本 `Two fundamentally important are ...` 原译“其中两个基本重要的现象是”搭配生硬，改为“其中两个具有根本重要性的现象是”。
7. `statistics.md`（L313）：底本 `with a sample size of only $8$` 原译“只需 $8$ 的样本量”量词误用，改为“只需 $8$ 个样本”。
8. `information-theory.md`（L157）：底本老虎机一例原译“以概率 $\{p_1,\ldots,p_k\}$ 分别独立地发出符号 $\{s_1,\ldots,s_k\}$”语序倒置且丢失“统计独立”义；改为“独立地发出符号 $\{s_1,\ldots,s_k\}$，各符号出现的概率分别为 $\{p_1,\ldots,p_k\}$”。
9. `information-theory.md`（L330）：底本 `principled extension of correlation coefficient` 原译“相关系数的有原则推广”为硬译，改为“相关系数的一种有理论依据的推广”。
10. `information-theory.md`（L397）：互信息消歧应用两处原译“与 Amazon 公司各自具有较大互信息的一组词”语序混乱；改为“找出一组词，其中每个词都与 Amazon 公司（亚马逊雨林）具有较大的互信息”。
11. `information-theory.md`（L501）：原译“预期 $D_{\rm KL}(p\|q_1)$ 与 $D_{\rm KL}(p\|q_2)$ 之间的 KL 散度值相近”重复且不通，改为“预期 $D_{\rm KL}(p\|q_1)$ 与 $D_{\rm KL}(p\|q_2)$ 的取值相近”。
12. `information-theory.md`（L653）：原译“由于在最大似然估计中，我们……。因此，……”为残句（“由于”没有主句），删去“由于”使句法完整，语义不变。

`naive-bayes.md`、`chapter_appendix-tools-for-deep-learning/index.md`：通读对照通过，**无修改**。

## 逐文件记录

- `maximum-likelihood.md`：2 处已改（见上）。逐条核对：贝叶斯式解释（argmax P(θ|X) → 弃 P(X)、无信息先验 → 最大似然估计 :eqref:`eq_max_like`）；9/13 例（$P=\theta^9(1-\theta)^4$、求导得 $\theta^8(1-\theta)^3(9-13\theta)$、三解 0/1/9/13 及“前两个给序列分配零概率故为极小值”的判断）；负对数似然（$\log((1/2)^{10^9})\approx-301029995.6$ 单精度可表示、乘积法则 $n(n-1)$ 乘法+$(n-1)$ 加法 vs 对数化 $n$ 次除法+$n-1$ 次求和）；熵 $H(p)=-\sum p_i\log_2 p_i$ 与交叉熵关系；连续情形 $\epsilon$ 论证（$\epsilon^N\prod p(x_i)$、$-N\log\epsilon$ 与参数无关故可丢弃）。术语：似然/对数似然/负对数似然/无信息先验/贝叶斯法则/熵/交叉熵均按表；“*uninformative prior*”“*likelihood*”等斜体标记保留。公式（独立公式逐字、行内公式多重集）与底本完全一致。
- `distributions.md`：3 处已改（见上）。逐条核对 7 个分布：伯努利 CDF :eqref:`eq_bernoulli_cdf` 与 $\mu=p,\sigma^2=p(1-p)$；离散均匀 :eqref:`eq_discrete_uniform_cdf` 与 $(1+n)/2,(n^2-1)/12$；连续均匀 :eqref:`eq_cont_uniform_pdf`、:eqref:`eq_cont_uniform_cdf` 与 $(a+b)/2,(b-a)^2/12$；二项（组合数 $\binom nk$ 解释、:eqref:`eq_binomial_cdf`、$np,np(1-p)$ 的“期望线性性+独立和方差可加”依据）；泊松（公交车站思想实验、$X^{(1)}$→$X^{(2)}$→$n$ 份切分、$\mu=n(p/n)=p$、$\sigma^2=p(1-p/n)\to p$、“稀有事件定律”、:eqref:`eq_poisson_mass`、:eqref:`eq_poisson_cdf`、速率/形状参数解释）；高斯（固定 $p$ 令 $n\to\infty$ 的动机、标准化 $Y^{(n)}$、中心极限定理、:eqref:`eq_gaussian_pdf`、CDF 无初等闭式与 `erf`、$E[X^4]<\infty$ 条件、最大熵分布性质）；指数族（:eqref:`eq_exp_pdf`、底层测度/基测度、自然参数/典范参数、充分统计量、累积量函数及 $A(\boldsymbol\eta)$ 的积分定义、高斯实例 $h=\frac1{\sqrt{2\pi}}$、$\boldsymbol\eta=[\mu/\sigma^2,\,1/(2\sigma^2)]^\top$、$T=[x,-x^2]^\top$、$A=\frac{\eta_1^2}{4\eta_2}-\frac12\log(2\eta_2)$ —— 与公式逐项一致）。小结/练习完整。verify 的“疑似未翻译英文行 3 处”为跨行公式续行（`\end{bmatrix}` 等），纯 LaTeX、非漏译。
- `naive-bayes.md`：通读通过，无修改。逐条核对：贝叶斯定理 → 条件独立假设 → 预测器 $\mathrm{argmax}_y\prod_i p(x_i|y)p(y)$；$P_{xy}$ 为 $d\times n$ 矩阵、$y\in\{1,\ldots,n\}$、$P_y$ 为长度 $n$ 向量与两种情况公式；:eqref:`eq_naive_bayes_estimation` 两条恒等式；复杂度 $\mathcal O(2^dn)\to\mathcal O(dn)$；训练（$n_y$、$P_y$ 估计 $p(y=8)=0.0967$）；拉普拉斯平滑 $(n_{iy}+1)/(n_y+2)$ 与 Beta-二项模型依据、$10\times28\times28$ 概率与“平均数字”可视化的解释；数值下溢 → 对数化稳定版 :eqref 公式与 $\log ab=\log a+\log b$；小结三条、练习三题完整。术语：朴素贝叶斯、条件独立、归纳偏置（首现括注 inductive bias）、贝叶斯定理/法则、拉普拉斯平滑、数值下溢、规范化项、独立同分布等均一致。
- `statistics.md`：2 处已改（见上）。逐条核对：估计量定义 $\hat\theta_n=\hat f(x_1,\ldots,x_n)$；MSE :eqref:`eq_mse_est`；统计偏差 :eqref:`eq_bias` 与无偏/渐近无偏；标准差 :eqref:`eq_var_est` 与“与 $E(\hat\theta_n)$ 而非真值比较”的解释；偏差-方差分解六步恒等式（$\mathrm{Var}+E^2$ 展开、化为 $(\mathrm{bias})^2+\mathrm{Var}+\mathrm{Var}[\theta]$）、欠拟合/过拟合与不可约误差；代码中 MSE 与偏差实现；模拟 $\mathcal N(1,16)$ 验证（$10{,}000$ 样本）；假设检验（Arbuthnot 起源、原假设/备择假设、化学家例子、试验组/对照组）；统计显著性公式 $1-\alpha=1-P(\text{reject }H_0\mid H_0\text{ true})$ 与第一类错误/假阳性/显著性水平；统计功效公式 $1-\beta$ 与第二类错误；效应量与所需样本数（8 vs 80000）；水过滤器类比；检验统计量；$p$ 值 :eqref 公式、拒绝域；单侧/双侧检验；五步流程；置信区间定义 $P_\theta(C_n\ni\theta)\ge1-\alpha$ 与置信水平/覆盖率、“随机的是区间”的解释、三个谬误；高斯例子（$\hat\mu_n,\hat\sigma^2_n$、$T$ 服从自由度 $n-1$ 的 Student t 分布、$[-1.96,1.96]$、:eqref:`eq_gauss_confidence` 的整理步骤）、代码与 `t_star`。小结/练习完整。术语：估计量、统计偏差、偏差-方差权衡、原假设/备择假设、统计显著性/统计功效、第一/二类错误、假阳性、显著性水平、效应量、检验统计量、拒绝域、置信区间/置信水平/覆盖率、独立同分布（首现括注英文）均按表统一。
- `information-theory.md`：5 处已改（见上）。逐条核对：纸牌思想实验四事件的比特数（0/2/5.7/225.6）；自信息 $I(X)=-\log_2 p$ 与 "0010" 例；本节 $\log=\log_2$ 约定；Shannon 熵三条公理与唯一性/单位规范化；熵定义 :eqref:`eq_ent_def`（离散和、连续积分、微分熵）；三条解释（为何用 log、为何取负、为何取期望）与老虎机例；熵的三条性质（$H\ge0$、$H(X)\le-E_P[\log q]$ 当且仅当 $P=Q$ 取等、均匀分布最大熵及连续情形的支撑集条件）；联合熵 :eqref:`eq_joint_ent_def`、$H(X),H(Y)\le H(X,Y)\le H(X)+H(Y)$；条件熵 :eqref:`eq_cond_ent_def`、$H(Y|X)=H(X,Y)-H(X)$ 与直观解释；互信息 $I(X,Y)=H(X,Y)-H(Y|X)-H(X|Y)$、:eqref:`eq_mut_ent_def` 以及与 $H(X)-H(X|Y)$ 等三个等价式；互信息性质（对称、非负、独立时为 0、可逆函数时 $I=H$）；逐点互信息 :eqref:`eq_pmi_def` 及其正负解释；NL 消歧应用；KL 散度定义 :eqref:`eq_kl_def`、性质（非对称、非负且仅 $P=Q$ 取等、$q(x)=0<p(x)$ 时为 $\infty$、与互信息的三个等价表达式）；对称性玩具例子（3% vs 40%）；交叉熵（二分类最大似然 $l(\theta)$ 推导、:eqref:`eq_ce_def`、$\mathrm{CE}=H(P)+D_{\rm KL}$、三种等价目标、$H(P)$ 为常数时的等价性证明）；多类分类（独热编码、:eqref 交叉熵损失、multinoulli 及其 p.m.f.、对数似然推导、NLL 内置度量对照）；小结/练习（包括 Jensen 不等式提示、猴子/醉排字工/语言模型困惑度、两个高斯间 KL）。术语：自信息、熵、微分熵、联合熵、条件熵、互信息、逐点互信息、KL 散度（相对熵）、交叉熵、困惑度、歧义消解均首现括注英文；公式（独立公式逐字、行内公式多重集）与底本一致。
- `chapter_appendix-tools-for-deep-learning/index.md`：通读通过，无修改。标题“附录：深度学习工具”、`toc` 八项章节文件名与底本逐项一致；正文一句按中文自然语序译出；书名 `*Dive into Deep Learning*` 译作 `*《动手学深度学习》*`（保留斜体标记），可接受。

## 遗留疑点（未擅改，提请负责人裁决）

1. **statistics.md L290 底本自相矛盾（重要）**：英文 “The *statistical significance* measures the probability of erroneously rejecting the null hypothesis, $H_0$, when it should not be rejected, i.e., $\text{statistical significance}=1-\alpha=1-P(\text{reject }H_0\mid H_0\text{ is true})$。” 文字说“度量错误拒绝的概率（即 $\alpha$）”，公式却给 $1-\alpha$。译文按底本直译保留该矛盾（与同节“统计功效 $1-\beta$”的写法一致）。是否加译者注或改写，属技术内容，未擅改。
2. **information-theory.md L452 底本疑似把 X/Y 写反**：第二项 $E_Y\{D_{\rm KL}(P(X\mid Y)\|P(X))\}$ 实际度量“知道 $Y$ 后 $X$ 的不确定性平均降低”，底本却写 “uncertainty about $Y$ ... learning the value of the $X$'s distribution”。译文按底本直译，记录待裁决。
3. **information-theory.md L693 小结“熵是度量不同信号中呈现了多少信息的单位”**：英文 “Entropy is the unit to measure ...”，熵并非“单位”；按底本直译，建议定稿改为“熵是衡量不同信号中呈现多少信息的量（指标）”。
4. **maximum-likelihood.md L21 底本笔误**：“does not depend on theta either” 正文裸写 `theta`（其余处为 $\boldsymbol\theta$）。译文用“该参数”保持语义，且不新增行内公式；如需与底本字形一致可改“theta”，但无技术收益，未改。
5. **术语新增建议（同意 b23 译者报告建议，供术语表合入）**：multinoulli distribution → 多项伯努利分布（注意与 multinomial“多项分布”区分；他组 `chapter_reinforcement-learning/value-iter.md` 将 categorical distribution 译“分类分布”，两者并存需术语表说明）；underlying measure 底层测度；base measure 基测度；natural/canonical parameters 自然参数/典范参数；sufficient statistics 充分统计量；cumulant function 累积量函数；self-information 自信息；differential entropy 微分熵；pointwise mutual information 逐点互信息；relative entropy 相对熵；statistical power 统计功效；type I/II error 第一类/第二类错误；test statistic 检验统计量；region of rejection 拒绝域；coverage 覆盖率；law of rare events 稀有事件定律；ambiguity resolution 歧义消解；support（概率论）支撑集（本批 distributions/information-theory 统一用“支撑集”，与 `chapter_convolutional-modern/cnn-design.md` 一致）。
6. **“Student's t-distribution” 译名**：本批统一“Student t 分布”（保留人名，未用“学生 t 分布”），与 b23 译者报告一致；若负责人决定音译，需全书反向统一。
7. **Bayes' rule / Bayes' theorem 译法**：本批按英文原词分别译“贝叶斯法则”（maximum-likelihood 2 处、naive-bayes 小结 1 处）与“贝叶斯定理”（naive-bayes 正文；`chapter_preliminaries/probability.md` 全书统一用“贝叶斯定理”）。TERMINOLOGY 只收 Bayes' theorem → 贝叶斯定理；建议补注“Bayes' rule → 贝叶斯法则（同义通行译法）”或由负责人裁定是否全书统一为“贝叶斯定理”。
8. **未改的次要直译痕迹**（均可读，按“不过度修改”保留）：information-theory L11 “Information can be encoded in anything with a particular sequence of one or more encoding formats” 译“任何具有一个或多个编码格式的特定序列的事物……”稍拗口；distributions L433 把 $X^{(1)}$ 说成“一分钟时间窗内有一辆公交车到达的概率”（底本如此）；information-theory L395 保留英文标题 “Amazon is on fire”（消歧例子依赖英文词形，合理）。

## 说明

- 仅修改本组负责 6 个文件中的 4 个、共 12 处定向 `edit`；未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md`、`work/batches/`、`work/reports/` 下其他组的报告及其他批次/组文件；未改动任何代码块、行内代码、数学公式、指令、图片路径、URL、标题层级、表格结构（`verify.py` 对公式/代码/指令的逐字校验仍 0 error）。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b23.txt` = **6/6 PASS（0 error、8 warning，与基线一致）**；`python work\align_check.py --batch work/batches/batch-b23.txt` = 1 段级疑点（cross-line 公式造成的伪阳性，已确认无漏译），报告存 `work/reports/align-g14-b23.json`（运行该脚本时同步重生成过默认的 `work/reports/align.json` 全书报告，保持其全书性质）。
