# 审校组 G1 报告（b01 + b02 + b03，共 18 个文件）

审校者：r1b（重做；此前 r1 未完成）。审校依据：`work/REVIEW_GUIDE.md`、`d2l-zh-translation/TERMINOLOGY.md`、`work/EXAMPLES.md`、`work/TRANSLATION_GUIDE.md`。
方法：逐文件通读中文全文（跳过逐字保留的围栏代码块），以段级对齐视图逐段对照英文底本；另用自建脚本做机械信号复核（英文大于等于 12 词的段落中的数字缺失、否定词/限定词（may/might/typically 等）缺口、强调标记数、引用指令数、中文占比；`work/_g1_flags.py`），并用 `work/align_check.py`、`work/verify.py`、`work/_g1_markers.py` 复核。
修改原则：只做定向 `edit`（不重写整文件），不改代码块/行内代码/公式/指令/图片路径/URL/标题层级/表格结构。

## 一、总体结果

- 审校文件：**18/18**（b01 6 个、b02 5 个、b03 7 个）。
- 修改文件：**10 个**；定向修改 **42 处**（其中 b01 24 处、b02 3 处、b03 15 处）。
- 结构校验：`python work\verify.py --batch work/batches/batch-b0{1,2,3}.txt` → **18/18 PASS，0 error**；警告 2/7/7 条，均为此前已说明的预期项（贡献者英文名单、语序重排造成的行内公式/代码顺序提示、英文数据样例、英文术语括注），逐条人工确认可接受。
- d2lbook 强调标记：`(**...**)` / `[**...**]` / `***...***` 括号配对检查 **0 问题**；已修复 11 处被写成全角括号的 `(**...**)`（见下），另将 1 处删除线旁的 `（~~应当~~）` 改为 ASCII 与底本一致。
- 段级完整性：`align_check.py --batch` 三批仅 3 条提示（b01 贡献者名单、安装章 1 条中文偏少、引言 1 条），经逐条核对均为英文专名或公式/代码令牌被剔除造成的误报；b02、b03 无提示。

## 二、逐文件审校记录

### b01
- `index.md`：通读通过，无修改（仅书名标题，结构正确）。
- `chapter_preface/index.md`：2 处修改。①"well documented"原译"充分记录"（误译为"被记录"），改为"具有充分的文档记录"；②"interactive resources have popped up sporadically to address a particular topic"原译"偶尔出现，针对某一特定主题作出解答"生硬且有增义，改为"零星出现一些专门探讨特定主题的交互式资源"。贡献者名单按要求保留英文人名；致谢、目标读者、框架依赖四个标签页均逐块核对，无缺漏。
- `chapter_installation/index.md`：通读通过，无修改（Miniconda/CUDA/框架安装四标签页、命令与 URL 均原样）。
- `chapter_notation/index.md`：3 处修改。①"单位矩阵（某个给定维度的）"→"（某个给定维度）"；②"集合减法"→"集合差"（标准数学译名，并与"只包含 A 中不属于 B 的元素"一致）；③"Hadamard（按元素）乘积"→"哈达玛积（Hadamard product，即按元素乘积）"，与 b02 linear-algebra 的"哈达玛积"统一。
- `chapter_introduction/index.md`（重点补译长篇）：**19 处修改**。全文 153 个结构单元与英文底本 1:1 对应，无整节/整段漏译；仅在英文专名（Alexa、Atari、DALL-E 2、Imagen、ResNet-50 等）、代码/citation 处保留英文，符合规范。修改主要为：把 `"it stinks!"` 译出；"To drive the point home"由"把这一点讲透彻"改为"为了强调这一点"；"unrepresented groups"改为"群体……代表性不足"；"disagree on whether..."句、`For reasons that we will get into as the book gets more technical`长前置定语等 2 处欧化句改为自然中文；"motivating example"前后译法统一；"最佳着力点"→"最佳平衡点"（sweet spot）；"因素的皮毛"→"冰山一角"；"国际象棋程序已经领先人类"（competitive 误译为"领先"）改为"具备与人类棋手一较高下的实力"；"开电灯开关"→"打开电灯开关"；"它之所以*深*，正是其模型学习……"补出"因为"；DALL-E/Imagen 句、`(~~should~~)` 相邻句等语序调整。图题、表格（数据集vs内存/算力表）、小结、练习均逐项核对，无漏项。
- `chapter_preliminaries/index.md`：通读通过，无修改。

### b02
- `chapter_preliminaries/ndarray.md`：2 处修改。"施加某种标准的二元运算符"→"应用……"；"这由于两个原因而不可取"→"这种做法不可取，原因有二"。四个框架标签页逐一核对，TensorFlow 可变/不可变、JAX 段落齐全；练习两条与底本一致（其中"三维"对应底本"3-dimensional"，非漏译）。
- `chapter_preliminaries/pandas.md`：通读通过，无修改（CSV 英文数据行、插补/删除、练习链接均核对）。
- `chapter_preliminaries/linear-algebra.md`：1 处修改。"gentle introduction"原译"轻松地介绍"改为"浅显地介绍"。标量/向量/矩阵/张量、降维、点积、矩阵乘法、范数（含弗罗贝尼乌斯范数）、小结、练习全部核对；"阶/维数"的区分与 TERMINOLOGY 一致。
- `chapter_preliminaries/calculus.md`：通读通过，无修改。导数定义、极限、偏导数、梯度、链式法则、乘积/商法则、小结、10 道练习均与底本对应；公式内英文说明（`\textrm{...}`）按硬性规定保留。
- `chapter_preliminaries/autograd.md`：通读通过，无修改。雅可比矩阵、非标量反向传播、detach/来源（provenance）、Python 控制流逐段核对；框架差异段（MXNet 重置梯度缓冲、PyTorch 累加梯度等）无合并、无缺漏。

### b03
- `chapter_preliminaries/probability.md`：通读通过，无修改（2 处底本疑似问题见"四、遗留疑点"）。频率/贝叶斯学派、样本空间/事件、随机变量、联合/条件概率、贝叶斯定理、独立性、HIV 检测例（表格数值）、期望/方差/协方差、偶然/认知不确定性、切比雪夫不等式、练习 1–13 全部核对一致。
- `chapter_preliminaries/lookup-api.md`：3 处修改。2 处 d2lbook 高亮标记误用全角括号（`（**...**）`→`(**...**)`），1 处"Jupyter笔记本"统一为全书多数写法"Jupyter 笔记本"。其余段落（dir/help/`?` 用法、讨论链接）无问题。
- `chapter_linear-regression/index.md`：通读通过，无修改。
- `chapter_linear-regression/linear-regression.md`：5 处修改，全部为高亮标记全角括号修复（`（**我们**）`、`（**把计算向量化…**）`、`（**考虑两种向量相加的方法。**）`、`（**或者…**）`、`（**可视化这些正态分布**）`→ASCII 括号），另将夹在其间的 `（~~应当~~）` 改为 `(~~应当~~)`。术语（训练/测试数据集、标签、特征、权重、偏置、仿射变换、平方误差、最大似然、小批量 SGD、推理/预测辨析）与底本、TERMINOLOGY 一致；"在期望意义下"等统计限定词保留。
- `chapter_linear-regression/oo-design.md`：1 处修改。"Jupyter笔记本"→"Jupyter 笔记本"。Module/DataModule/Trainer、四个框架的基类说明、工具函数、练习均核对；`in deference to TensorBoard`译"为了向……致敬"可接受。
- `chapter_linear-regression/synthetic-regression-data.md`：3 处修改。①高亮标记全角括号修复（`（**接受批量大小…**）`）；②"其特征为从标准正态分布中抽取的二维特征"语义重复，改为"每个样例的二维特征均从标准正态分布中抽取"；③"通过施加一个真实值线性函数"→"通过一个真实值线性函数"。ground truth 首次括注"真实值（ground truth）"、JAX 数据加载差异段、小结、练习均核对。
- `chapter_linear-regression/linear-regression-scratch.md`：3 处修改，均为高亮标记全角括号修复（含跨行块 `（**我们将从零开始实现整个方法…**）`→ASCII）。SGD 类三框架实现、训练循环、`numref` 双反引号写法、练习 1–12 均核对无缺漏。

## 三、修改清单（按类型）

1. **d2lbook 强调标记全角括号**（构建会断言失败的硬性问题）：lookup-api 2 处、linear-regression 5 处、synthetic 1 处、linear-regression-scratch 3 处，共 11 处 → ASCII 成对括号。
2. **术语一致性**：notation "集合减法"→"集合差"；"Hadamard"→"哈达玛积（Hadamard product）"；"Jupyter笔记本"→"Jupyter 笔记本"（book 内多数写法）。
3. **误译/语气修正**：preface "充分记录"→"具有充分的文档记录"；intro "competitive"→"具备与人类棋手一较高下的实力"、"最佳着力点"→"最佳平衡点"、"为了把这一点讲透彻"→"为了强调这一点"、"其它"→"其他"。
4. **中文自然度**：intro 2 处长定语欧化句、preface"interactive resources"句、ndarray 2 处、linear-algebra 1 处、synthetic 2 处。
5. **漏译补齐**：intro 引用的英文顾客评论 `"it stinks!"` 译出。

## 四、遗留疑点（建议负责人裁决/统一）

1. **TERMINOLOGY 内部不一致**：表中 `data instance | 数据示例`，而"易混词辨析"节与全书写法均为"实例"；译本正文（引言"数据"小节）用"数据实例（data instance）"。建议负责人明确 `data instance` 统一译名（倾向"数据实例"）后统一全库或修订术语表。
2. **底本疑似笔误/引用错误（未改，按指南直译并记录）**：
   - `probability.md`："with 10% probability it might provide a 10$\times$ return 10$\times$"（底本重复 10$\times$，译文按底本保留）。
   - `probability.md`："For an empirical review of this fact for large scale language models see :citet:`Revels.Lubin.Papamarkou.2016`"——该文献是 Julia 前向微分类论文，与"大语言模型规模经验综述"不符，疑为底本引用错误；已按底本保留。
3. **`(~~...~~)` 删除线括号写法**：`ndarray.md` 保留 `（~~或全为1的张量~~）`（底本为 `(~~or one~~)`；删除线非 d2lbook 高亮标记，不影响构建）。若要求全库统一 ASCII 括号，可一并处理。另外底本 "(~~or one~~)" 语义断裂（对应前文 "all 0s or 1s"），译文按语义补为"或全为1的张量"，属译者处理。
4. **数字写法全库不一致**：引言/概率等章用 `44000`、`1024 块 GPU`、`1.41倍`，线性回归章用 `10,000维`、`5–10个标签`（千分位逗号 vs 无逗号、`–` vs `-`）。属全书格式统一问题，本组未做批量替换，建议最后统一。
5. **"Jupyter 笔记本"空格写法**：审校前全库 28 处带空格 vs 2 处不带；本组 2 处已统一为带空格。建议负责人做全书扫描统一。
6. **"Discussions" 链接可见文字**：本组全部统一译"讨论"（沿用 b02/b03 译法）；若全书倾向保留 `Discussions`，需全书统一。
7. **引言版式差异**：约第 126 个结构单元之后的行文为"逐短语换行"版式，与前半段换行版式不同（系补译批次差异），内容完整、不影响构建；建议后续统一排版风格（非本次审校可改范围）。
