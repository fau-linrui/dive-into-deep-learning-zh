# 审校组 G13 审校记录（批次 b22）

审校范围：`work/batches/batch-b22.txt`（4 个文件，均为数学附录、公式极密集）：
`chapter_appendix-mathematics-for-deep-learning/` 下的 `single-variable-calculus.md`（单变量微积分）、`multivariable-calculus.md`（多变量微积分）、`integral-calculus.md`（积分学）、`random-variables.md`（随机变量）。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照（4 个文件共 647 个结构块全部过一遍）；公式、代码块、指令、图片路径、URL 只做结构一致性核验、未改动。数学段落逐句核对，重点为：①每个公式前后的变量解释、符号与公式是否一致（如 mvc 的 $\nabla_{\mathbf{w}}L$、$\mathbf{H}_f$、雅可比 $D\boldsymbol\phi$，ic 的 $D\phi$ 与 $\left|\det(D\phi)\right|$，rv 的 $E[X]=\sum_i x_ip_i$、$\sigma_{XY}$ 等）；②推导步骤是否完整（和/积/链式法则、微积分基本定理、换元公式、Fubini、黑塞展开、协方差展开）；③限定词（“本质上”“可能”“通常”“无法保证”等）是否保留；④术语按 `TERMINOLOGY.md`（导数/偏导数/梯度/链式法则/黑塞矩阵/雅可比矩阵/积分/期望/方差/标准差/协方差/累积分布函数/边缘分布/柯西分布/切比雪夫不等式等）；⑤图题、图替代文本、练习子题、小结是否完整；⑥中文是否自然。只做定向 `edit`；每次修改后 `python work\verify.py --file <path>` 复核。

基线：`python work\verify.py --batch work/batches/batch-b22.txt` = **4/4 PASS，0 error，4 warning**（每文件 1 条“行内公式内容一致但顺序不同”，系中文语序重排所致）。`python work\align_check.py --batch work/batches/batch-b22.txt` 报告 3 个文件有段级疑点，逐条核对后确认**全部是“独立公式单独成段”造成的段对齐偏移，无漏译**；另用自建的分块对齐脚本（`work/_g13_view.py`）复核：4 个文件均无 EN-only 块（svc 2 处、mvc 1 处 ZH-only 块经人工核对均为公式前后段落拆分，内容对应底本）。审校后批次仍为 **4/4 PASS、0 error、4 warning**（警告数与基线一致）。

## 修改清单（4 个文件、9 处）

1. `single-variable-calculus.md`（L164）：`the output would change by that small amount scaled by the derivative` 原译“那么输出会变化该小量乘以导数”句子不通（“变化该小量”不成句），改为“那么输出的变化量就是该很小的量乘以导数”。
2. `single-variable-calculus.md`（L411）：`while the other terms are all zero` 原译“而其他项的系数都为零”误加“系数”（原文是“其他项（求两次导数后）都为零”，并非系数为零），改为“而对其他项求两次导数时结果都为零”。
3. `multivariable-calculus.md`（L4）：`a loss function of potentially billions of weights` 原译“潜在数十亿个权重”中“潜在”易误解为“潜在的（latent）”，改为“可能多达数十亿个权重”。
4. `multivariable-calculus.md`（L232）：`what variables directly relate to one another` 原译“观察变量之间直接如何相互关联”语序不通，改为“观察变量之间是如何直接相互关联的”。
5. `integral-calculus.md`（L236）：`something much more completely understood` 原译“（导数要完全被理解得多）”不成句，改为“（而导数是被理解得完整得多的东西）”。
6. `integral-calculus.md`（L458）：`this result (called *Fubini's Theorem*)` 原译“（称为*富比尼定理*，Fubini's Theorem）”；`TERMINOLOGY.md` 第十节明确为“Fubini 定理”，改为“（称为*Fubini 定理*）”，与术语表统一。
7. `integral-calculus.md`（L466）：`and say that for $U = [a, b]\times [c, d]$ this is` 原译“并对 $U = ...$ 说这等于”介词结构不通，改为“并说对于 $U = ...$，这就是”。
8. `random-variables.md`（L14）：`measuring a single digit of accuracy, that is to say with bins for 0cm, 1cm, 2cm` 原译“只测到一位小数的精度”与后文区间（整数 cm）矛盾；此处的“single digit”指个位/一位数字，改为“只测到一位数字的精度”（下一段“再往下推一位”作“一位小数”正确，保留）。
9. `random-variables.md`（L757）：`with the expected behavior of 1 meaning most correlated` 原译“其是否符合预期的行为是”误把“符合预期”写成疑问式（“是否”），改为“其行为符合预期：$1$ 表示最相关，$-1$ 表示相关性最小”。

## 逐文件记录

- `single-variable-calculus.md`：2 处已改（见上）。逐条核对了有限差分法、导数定义 :eqref:`eq_der_def`、ε 展开 :eqref:`eq_small_change` 的三步推导、和/积/链式法则的展开（含 $\epsilon^2$ 高阶项与差商）、线性近似、高阶导数与三个二阶导数图题、二次近似、泰勒级数与 $n$ 阶导数记号、$e^x$ 展开；变量解释与公式符号一致（如“分母中的 2 抵消对 $x^2$ 求两次导数得到的 2”“$6=3!=3\times2$ 来自对 $x^3$ 求三次导数的常数”）。图题/alt 文本按底本笔误（“fist derivative”）按语义译出，未加注。小结、习题 4 题完整。
- `multivariable-calculus.md`：2 处已改（见上）。逐条核对多元泰勒展开（:eqref:`eq_part_der` 到 :eqref:`eq_nabla_use`）、点积几何与最速下降方向、临界点、多元链式法则两条路径与网络图 :numref:`fig_chain-1`/`fig_chain-2`、反向传播正向/反向两个步骤与代码输出、黑塞矩阵定义 :eqref:`eq_hess_def` 与混合偏导数对称性、二元二次展开、$xe^{-x^2-y^2}$ 的梯度/黑塞矩阵与 $[-1,0]^\top$ 处的二次近似、矩阵微积分（分母布局、$\mathbf{x}^\top A\mathbf{x}\to(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$、$\|\mathbf{X}-\mathbf{UV}\|_2^2$ 的逐元素推导）——推导步骤完整，符号与变量解释对应。术语按表：偏导数、梯度、最速下降方向、临界点、黑塞矩阵（首现括注 Hessian）、分母布局矩阵导数、幂/链式法则均一致（雅可比矩阵不在本文件）；$U^\top$ 在正文中原文为不带粗体的 $U^\top$（底本如此），未动。
- `integral-calculus.md`：3 处已改（见上）。逐条核对面积的切片近似、微积分基本定理 :eqref:`eq_ftc` 及其展开、常数 $C$ 抵消、换元公式 :eqref:`eq_change_var` 的直觉推导（矩形面积 $\epsilon f(u(x))$ 与 $\epsilon\frac{du}{dx}f(u(x))$）、有向面积与符号约定、多重积分与 Fubini 定理（含反例 $xy(x^2-y^2)/(x^2+y^2)^3$）、高维换元公式与雅可比行列式、极坐标换元 $\int e^{-x^2-y^2}dxdy=\pi$。变量解释与公式一致；“*先验地*”斜体保留；图题/alt 文本完整；习题 4 题完整（底本习题 4 第二式 $f(x,y)=$ 的畸形写法按底本保留）。
- `random-variables.md`：2 处已改（见上）。逐条核对从离散到连续的掷镖思想实验、p.d.f. 定义 :eqref:`eq_pdf_def` 与归一化 :eqref:`eq_pdf_int_one`、c.d.f. 定义与四条性质、均值/方差/标准差定义与性质（:eqref:`eq_exp_def`、:eqref:`eq_var_def`）、$a\pm2$ 三点例子（$\mu_X=a$、$\sigma_X^2=8p$、$\sigma_X=2\sqrt{2p}$）、切比雪夫不等式 :eqref:`eq_chebyshev` 及 $p=1/8$ 的“紧性”讨论、连续情形均值/方差积分、柯西分布均值/方差不存在、联合密度、边缘分布与边缘化、协方差 :eqref:`eq_cov_def` 的 $4p-2$ 推导、相关系数 :eqref:`eq_cor_def` 与 $\rho=\textrm{sign}(a)$、与余弦公式的类比。术语按表：概率密度函数、累积分布函数、汇总统计量、均值、方差、标准差、切比雪夫不等式、柯西分布（均首现括注英文）、联合密度、边缘分布、协方差、相关系数、重尾。`\$10 \pm \$1` 的 `\$` 转义原样保留；小结/习题完整。

## 遗留疑点（未擅改，提请负责人裁决）

1. **mvc 的 forward pass / backwards pass 译名**：L396–397 将 `*forward pass*`、`*backwards pass*` 译为“*正向传播*”“*反向传播*”，与紧邻的算法名“*反向传播*（backpropagation）”字面重合，两个步骤的区分被削弱；而 `chapter_multilayer-perceptrons/backprop.md`（他组文件）把 `forward pass` 译作“前向传递”。建议全书统一为“前向传递/反向传递”（或至少在本处区分），但涉及跨章术语，未擅改。
2. **mvc 底本自相矛盾（已按底本直译）**：英文 “If we now take the partial derivative with respect to say $\beta_1$, ... so we obtain that $\frac{df}{dx_1}=\beta_1$” 前后变量不一致（应对 $x_1$ 求偏导才得 $\beta_1$）；译文照底本保留。另底本 “we will often **counter** throughout this section” 疑为 encounter 笔误，译文按语义译“反复遇到”；行内代码 `f.backwards()` 为底本笔误，按规范逐字保留。
3. **rv 底本两处数学瑕疵（已按底本直译）**：① “if $p=0$, this random variable always takes the value $0$”——按定义 $p=0$ 时恒取 $a$，底本疑误；② 离散 c.d.f. 分段 “$\frac{1}{2}$ & $x<1$” 应为 $0\le x<1$。译文均照底本直译，未自行“修正”，提请确认是否需加译注。
4. **svc 底本表述含混**：“for well behaved functions ... we can write out the infinite number of terms and approximate the exactly same function”，译文“精确地近似同一个函数”保留“近似”与“精确”的张力（底本如此），未改。
5. **Fubini 译名取舍**：本组已按 `TERMINOLOGY.md` 统一为“Fubini 定理”（改掉译稿的“富比尼定理”）；但批次译稿报告 `work/reports/b22.md` 曾建议合入“富比尼定理”，若负责人最终采纳音译，需全书反向统一。
6. **未改的次要生硬点**（均为可读中文，按“不过度修改”保留，供定稿参考）：rv L555“机器学习科学家在定义模型时会让我们通常不必处理这些问题”（稍拗口）；rv “它是否代表了我们能就这个随机变量画出的某种东西？”（直译痕迹）；svc“可疑地接近 $8$”；mvc 两处 `cleaner` 译“干净得多”（与 `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.md`“简洁干净的记号”同章风格一致，未统一为“简洁”）。

## 说明

- 仅修改本组负责的 4 个文件、共 9 处定向替换；未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md`、`work/reports/b22.md` 及其他批次/组文件；未改动任何代码块、行内代码、数学公式、指令、图片路径、URL、标题层级、表格结构（`verify.py` 对公式/代码/指令的逐字校验仍为 0 error）。
- 审校辅助脚本：`work/_g13_view.py`（EN/ZH 分块并排；只读生成对照文本，不参与构建）；对齐检查 JSON：`work/reports/align-g13-b22.json`。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b22.txt` = 4/4 PASS（0 error、4 warning，与基线一致，警告均为“行内公式内容一致但顺序不同”，系中文语序重排，已人工确认多重集一致）。
