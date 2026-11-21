# 审校记录：组 G2（批次 b04 + b05）

- 审校范围：`work/batches/batch-b04.txt`、`batch-b05.txt` 共 17 个文件（线性回归泛化/权重衰减 3 篇 + 线性分类 softmax 系列 8 篇 + 多层感知机 6 篇）
- 底本：`d2l-en-src/`（只读）；术语依据：`d2l-zh-translation/TERMINOLOGY.md`
- 方法：逐文件中文本全文通读；英文底本按结构单元对齐后逐段对照（数学/定义/因果与限定词密集段落逐句核对，叙述段抽样，小结/练习逐条核对）；辅以 `verify.py` 结构校验、`align_check.py` 段级完整性检查、数字/否定/限定词自动比对与术语扫描。仅对译文做定向 `edit`，未改动代码块、行内代码、公式、指令、图片路径、URL、标题层级与表格结构。

## 逐文件记录

### 批次 b04

1. `chapter_linear-regression/linear-regression-concise.md` — 通读通过，无修改。
   疑点：原文 "the first generation of such models that found widespread adoption" 中 "such models" 按上下文应指框架，疑为原文笔误；译文照原文译"此类模型"，按规则不自行修正。
2. `chapter_linear-regression/generalization.md` — 通读通过，无修改。
3. `chapter_linear-regression/weight-decay.md` — 通读通过，无修改。
4. `chapter_linear-classification/index.md` — 通读通过，无修改。
5. `chapter_linear-classification/softmax-regression.md` — 修改 1 处。
   "trade off a small amount of accuracy for computational and storage cost" 原译"用少量精度换取计算和存储成本"，"换取"方向含糊；改为"用少量精度换取计算和存储成本的降低"。
6. `chapter_linear-classification/image-classification-dataset.md` — 通读通过，无修改。
7. `chapter_linear-classification/classification.md` — 通读通过，无修改。
8. `chapter_linear-classification/softmax-regression-scratch.md` — 通读通过，无修改。
9. `chapter_linear-classification/softmax-regression-concise.md` — 通读通过，无修改。

### 批次 b05

10. `chapter_linear-classification/generalization-classification.md` — 通读通过，无修改。
    备注：原文小结中 "Alexey Chernovenkis"（正确拼写应为 Chervonenkis）照原文保留。
11. `chapter_linear-classification/environment-and-distribution-shift.md` — 通读通过，无修改。
    备注：原文 "racial sleight of hand" 疑为 racial slight 之笔误，译文"种族上的冒犯"按语义处理，保留。
12. `chapter_multilayer-perceptrons/index.md` — 通读通过，无修改。
13. `chapter_multilayer-perceptrons/mlp.md` — 通读通过，无修改。
14. `chapter_multilayer-perceptrons/mlp-implementation.md` — 修改 2 处。
    ① 错字"例如，设想想在模型的第 42 层……"→"例如，设想一下在模型的第 42 层……"；
    ② "与我们简洁实现 softmax 回归的实现……相比"（"实现"重复、语句冗赘）→"与我们简洁实现 softmax 回归时……相比"。
15. `chapter_multilayer-perceptrons/backprop.md` — 修改 1 处。
    括号内"更常见的叫法是*反向传播*，backpropagation"重复冗余（两个词同译"反向传播"）→"更常见的叫法是 *backpropagation*"。
16. `chapter_multilayer-perceptrons/numerical-stability-and-init.md` — 通读通过，无修改。
17. `chapter_multilayer-perceptrons/generalization-deep.md` — 修改 1 处。
    "（即使某个距离 $d(\mathbf{x}, \mathbf{x}_i')$ 最小的 $k$ 个点）"中"即使"为误译（原文 the $k$ points that minimize some distance），改为"即"。

## 汇总

- 审校文件：17 个（b04：9，b05：8）
- 修改文件：4 个，修改 5 处（softmax-regression 1、mlp-implementation 2、backprop 1、generalization-deep 1）
- 其余 13 个文件通读通过、无修改
- 主要问题类别：误译 1、表达冗余/生硬 3、字词笔误 1；未发现整句漏译、统计/因果限定词丢失、术语表不一致、图题/表格/练习漏译等问题（与 align_check 段级检查 0 疑点一致）

## 遗留疑点（供负责人裁决，未擅自改动）

1. `softmax-regression.md`："softmax 运算保持其各参数之间的顺序"，原文 arguments 指输入变量，译"参数"易与模型参数混淆，可考虑"输入/自变量"，未改。
2. `softmax-regression.md`：Boltzmann 段 "the prevalence of a state of energy" 译"某个能量状态的普遍程度"略生硬，未改。
3. `generalization.md`：*泛化差距*（generalization gap）首次出现未括注英文（同章 *泛化误差* 等均已括注）；`generalization-deep.md` 中已括注。是否统一在首次出现处补括注，请负责人裁决。
4. `linear-regression-concise.md`、`environment-and-distribution-shift.md`、`generalization-classification.md` 中原文笔误（"such models"、racial sleight of hand、Chernovenkis）均照原文直译/保留，未自行修正技术内容。

## 校验结果

- `python work\verify.py --batch work/batches/batch-b04.txt`：9/9 PASS，0 error（个别"内容一致但顺序不同"警告可忽略）
- `python work\verify.py --batch work/batches/batch-b05.txt`：8/8 PASS，0 error
- `python work\align_check.py`（b04、b05）：段级疑点文件 0 个
- 修改后单文件复验均 0 error
