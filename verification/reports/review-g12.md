# 审校记录：组 G12（批次 b21）

- 审校范围：`work/batches/batch-b21.txt` 共 8 个文件（推荐系统：NeuMF、序列感知推荐、点击率预测、因子分解机、DeepFM；数学附录：章索引、几何与线性代数运算、特征分解）
- 底本：`d2l-en-src/`（只读）；术语依据：`d2l-zh-translation/TERMINOLOGY.md`；样例依据：`work/EXAMPLES.md`
- 方法：逐文件中译文全文通读；英文底本逐段对照（公式/推导/定义/术语密集段落逐句核对，含公式符号解释、变量上下标、维度一致性、因果与限定词；叙述段通读并抽查）；辅以 `verify.py --batch` 结构校验、`align_check.py --batch` 段级完整性检查、`termcheck.py` 术语变体扫描与数字/符号人工比对。仅对译文做定向 `edit`，未改动代码块、行内代码、公式、指令、图片路径、URL、标题层级与表格结构。

## 逐文件记录

1. `chapter_recommender-systems/neumf.md` — 修改 2 处，其余通读通过。
   - ① "隐层大小恒为 10"→"隐藏层大小恒为 10"（术语表 hidden layer→隐藏层）；
   - ② NDCG"归一化折损累计增益"→"归一化折损累积增益"（术语表作"累积"）。
   - 核对：GMF/MLP 公式、预测层 $\sigma(\mathbf{h}^\top[\mathbf{x},\phi^L])$、Hit@ℓ 与 AUC 公式及符号说明、显式/隐式反馈、BPR/合页损失；$1$ 与 $0$ 的反馈二值化、`seq-aware` 划分表述均与源文一致。
2. `chapter_recommender-systems/seqrec.md` — 通读通过，无修改。
   - 核对：$\mathbf{E}^{(u,t)}$ 维度 $L\times k$、HConv/VConv 输出 $\mathbb{R}^d$ 与 $\mathbb{R}^{kd'}$、$\mathbf{W}\in\mathbb{R}^{k\times(d+kd')}$、预测式 $\hat y_{uit}$、union-level/point-level 模式、$L=5$ 与"三个训练样本"数目关系，均与源文一致。
3. `chapter_recommender-systems/ctr.md` — 通读通过，无修改。
   - 核对：CTR 公式、34 个字段/首列标签、15000/3000 样本、类别特征与独热索引说明；`CTRDataset` 及 Criteo/Avazu 链接原样保留。
4. `chapter_recommender-systems/fm.md` — 通读通过，无修改。
   - 核对：二阶 FM 公式、$\mathcal{O}(kd^2)$→$\mathcal{O}(kd)$ 的成对交互重排推导五个等号、$\mathbf{w}_0/\mathbf{w}/\mathbf{V}/\mathbf{v}_i$ 符号解释、稀疏特征复杂度结论；"第 i 个变量"沿用源文未加公式标记，保持原样。
5. `chapter_recommender-systems/deepfm.md` — 通读通过，无修改。
   - 核对：$\mathbf{z}^{(0)}$ 拼接式、$\mathbf{z}^{(l)}$ 递推式、$\hat y=\sigma(\hat y^{(FM)}+\hat y^{(DNN)})$、DeepFM 与 Wide \& Deep 对比、暂退法/金字塔结构（30-20-10）；$y_{DNN}$ 与 $\hat y^{(DNN)}$、$\mathbf{W}_l/\mathbf{W}^{(l)}$ 的不一致源自底本，按底本保留（见疑点 3）。
6. `chapter_appendix-mathematics-for-deep-learning/index.md` — 修改 2 处，其余通读通过。
   - ① "其中一个关键要素是特征分解基础的展开"（"展开"名词化生硬）→"其中一个关键要素是展开特征分解的基础知识"；
   - ② "把微分学理论发展到……为止"（主谓杂糅）→"把微分学理论发展到足以……的程度"。
   - 作者行、bits-per-character、论域、`toc` 块核对无误。
7. `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.md` — 修改 3 处，其余逐段对照通过。
   - ① "尽管单个向量的默认方向是列向量"→"默认取向"（几何语境下"方向"与 direction 混淆，原文为 orientation）；
   - ② "夹角角度认为较暗的图像是相同的"（"夹角角度"重复、句子不成话）→"从夹角的角度看，较暗的图像被视为是相同的"；
   - ③ "不存在一种可以普遍选定的求和情形"→"求和方式"（原文 summation case）。
   - 核对：点积/范数/夹角与余弦相似度、$\mathbf{v}\cdot\mathbf{w}=0$ 的正交等价、超平面投影长度 $1/\|\mathbf{w}\|=1/\sqrt{5}$、$2x+y=1$、线性组合/线性相关/线性无关、秩的定义与 $B$、$C$ 两例、逆矩阵与 $2\times2$ 逆公式、行列式 $ad-bc$ 与面积/可逆性、张量收缩与爱因斯坦求和/`einsum`，公式与符号解释逐条与英文对应。
8. `chapter_appendix-mathematics-for-deep-learning/eigendecomposition.md` — 修改 4 处，其余逐段对照通过。
   - ① "为对角线上是具有相应特征值的矩阵"（缺"元素"、句子不完整）→"为对角线上元素是相应特征值的矩阵"；
   - ② "从一个均值为零、方差为一的高斯分布 $5\times5$ 矩阵开始"（语序别扭）→"从一个服从均值为零、方差为一的高斯分布的 $5\times5$ 矩阵开始"；
   - ③ "元素为独立、均值为零、方差为一的高斯分布的大随机矩阵"→"元素独立服从均值为零、方差为一的高斯分布的大随机矩阵"；
   - ④ "特征向量归一化为单位长度"→"规范化为单位长度"（术语表 normalization→规范化，termcheck 亦标记）。
   - 核对：特征方程与 $\det(\mathbf{A}-\lambda\mathbf{I})=0$、示例特征值 $4$ 与 $1$ 及对应特征向量、$\mathbf{A}=\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^{-1}$、$\mathbf{A}^n$、$\det=\prod\lambda_i$、对称矩阵 $\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top$、Gershgorin 圆盘与四区间 $[0.7,1.3]$ 等、幂迭代、$\sqrt5\approx2.2$ 与圆律，均与源文一致。

## 汇总

- 审校文件：8 个（全部通读中文并对照英文）
- 修改文件：4 个，修改 11 处（neumf 2、附录 index 2、geometry 3、eigendecomposition 4）
- 未修改文件：4 个（seqrec、ctr、fm、deepfm）
- 主要问题类别：术语不一致 2（隐藏层、NDCG 累积/累计）、表达生硬/句子不完整 7、术语规范化 1（归一化→规范化）、语义歧义 1（方向→取向）；未发现整句漏译、公式符号错位、维度或数值错误、图题/练习漏译、标题层级或表格结构破坏（与 `align_check` 段级检查 0 疑点一致）
- 校验：`python work\verify.py --batch work/batches/batch-b21.txt` → 8/8 PASS，**0 error**，5 warning（均为"行内公式内容一致但顺序与原文不同"，系中文语序调整，逐条确认可接受）；修改后单文件复验均 0 error。

## 遗留疑点（供负责人裁决，未擅自改动）

1. `geometry-linear-algebraic-ops.md` 秩例："可以证明 $\mathbf{C}$ 的秩为 2，因为例如前两列线性无关，而任意四组三列的组合都是线性相关的。"底本原文为 "any of the four collections of three columns are dependent"；该 $4\times5$ 矩阵的三列子集共 $\binom{5}{3}=10$ 个而非 4 个，底本疑有笔误。按手册"照底本直译、不自行修正技术内容"保留"四组"，建议负责人确认是否加译注。
2. 同文件数值问题段：底本 "the inverse will typically have almost every entry non-negative"，由上下文（稀疏矩阵之逆稠密、需存 $1\text{M}^2$ 个元素）判断应为 non-zero；译文作"几乎每个元素都非零"（语义正确），按明显笔误处理，特此记录。
3. `deepfm.md` 底本记号不自洽："Let $y_{DNN}$ denote the output of the prediction" 与公式 $\hat y^{(DNN)}$ 不一致；"$\mathbf{W}_{l}$ 和 $\mathbf{b}_{l}$" 与公式 $\mathbf{W}^{(l)}/\mathbf{b}^{(l)}$ 不一致。均忠实按底本保留，未改动公式。
4. 推荐系统 4 个文件（neumf/seqrec/fm/deepfm）中底本行内公式 `$i^\textrm{th}$`、`$u^\textrm{th}$` 按"行内公式逐字保留"硬性要求未动，中文读作"第 $i^\textrm{th}$ 个/第 $u^\textrm{th}$ 行"略生硬（b21 翻译报告亦已说明）；是否全书统一改写为"第 $i$ 个"，请负责人决定。
5. `neumf.md` 把 NDCG 改为术语表规定的"归一化折损累积增益"；`chapter_recommender-systems/ranking.md`（属 b20，不在本组范围）仍作"归一化折损累计增益"，建议负责人统一协调（`termcheck.py` 同类变体）。
6. `eigendecomposition.md` 底本 "principle eigenvalue/eigenvector" 系 principal 误拼，译文按正确语义译"主特征值/主特征向量"，保留括注 (principal …)，无需改动。
7. 批次清单说明：`work/batches/batch-b21.txt` 第 6 行为数学附录 `index.md`；任务描述中的"推荐系统章索引"（`chapter_recommender-systems/index.md`）实属 b20（G11），本组未改动该文件。
</content>
