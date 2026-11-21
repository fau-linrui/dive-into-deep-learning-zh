# 审校组 G8 审校记录（批次 b16、b17）

审校范围：`work/batches/batch-b16.txt`（8 个）+ `batch-b17.txt`（10 个），共 **18 个文件**：

- 计算机视觉应用：`multiscale-object-detection.md`、`object-detection-dataset.md`、`ssd.md`、`rcnn.md`、`semantic-segmentation-and-dataset.md`、`transposed-conv.md`、`fcn.md`、`neural-style.md`、`kaggle-cifar10.md`、`kaggle-dog.md`
- NLP 预训练（词嵌入）：`index.md`、`word2vec.md`、`approx-training.md`、`word-embedding-dataset.md`、`word2vec-pretraining.md`、`glove.md`、`subword-embedding.md`、`similarity-analogy.md`

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照（代码块、公式、指令、图片路径、URL、标题层级、表格结构只做一致性核验，未改动；定义、公式符号说明、因果与限定词、图题/表格文字、小结、练习必查，数学密集段落逐句核对）。另用临时脚本做了四类机械核查（脚本用完已删除）：段级结构对齐（align_check）、正文数字多重集比对、`[Discussions]` 链接文字与链接 URL 比对、术语变体（长宽比/宽高比、规范化/标准化及全书用词分布）比对。只做定向 `edit`，每次修改后 `python work\verify.py --file <path>` 复核。

基线：两批 `verify.py --batch` 均为 0 error（b16 8/8 PASS、10 warning；b17 10/10 PASS、13 warning，warning 均为“行内公式/行内代码顺序与原文不同”或英文例句/数据路径/类比词串/公式表头等可接受项）。`align_check.py` 段落级漏译检查 b16 无提示，b17 仅 kaggle-dog 的裸路径列表 1 条（误报）。**审校后两批仍为 8/8、10/10 PASS，0 error，warning 数与基线完全相同（10、13），align_check 结果与基线相同。**

## 总览

- 审校文件 18 个；**通读通过无修改 2 个**（object-detection-dataset、nlp-pretraining/index）；**有修改 16 个，共 34 处**。
- 问题分类：**漏译（框架标签页/文末链接文字 `[Discussions]` 未译）15 处、涉及 9 个文件**；**术语不一致 6 处**（standardize→“规范化”4 处、aspect ratio→“长宽比”2 处）；**重要术语首现未括注英文 6 处（另去重 1 处）**；**句法不通 1 处**；**`（**…**）` 标记形式恢复源文件 ASCII 写法 5 处**。

## 一、修改明细（按类别）

### 1. 漏译：链接可见文字 `[Discussions]` 未译（15 处 / 9 文件）
底本各节末尾的讨论链接在 b16 文件中均译作 `[讨论]`，但 b17/b16 的 Kaggle 文件中 15 处仍为英文 `[Discussions](https://discuss.d2l.ai/t/…)`，属整句层面漏译（URL 原样保留）。统一改为 `[讨论]`：
- `kaggle-cifar10.md` 2 处（/t/379、/t/1479）、`kaggle-dog.md` 2 处（/t/380、/t/1481）
- `word2vec.md` 1 处（/t/381）、`approx-training.md` 1 处（/t/382）、`word-embedding-dataset.md` 2 处（/t/383、/t/1330）、`word2vec-pretraining.md` 2 处（/t/384、/t/1335）、`glove.md` 1 处（/t/385）、`subword-embedding.md` 2 处（/t/386、/t/4587）、`similarity-analogy.md` 2 处（/t/387、/t/1336）
- 复核：18 个文件的 `[讨论]` 链接数与英文 `[Discussions]` 一一对应，URL 全部未变。

### 2. 术语不一致（6 处）
- **standardize → “标准化”（4 处）**：`semantic-segmentation-and-dataset.md` L297、`fcn.md` L353、`neural-style.md` L130/L131 将英文 `standardize(d)` / `standardization` 译作“规范化”。全书同类语境（`fine-tuning.md`、`batch-norm.md`、`kaggle-cifar10.md`、`kaggle-dog.md`、官方中文旧版对应三句）均作“**标准化**”，而“规范化”对应术语表的 normalization；为与全书及同章一致，4 处改为“标准化”（函数名 `normalize_image` 等行内代码不变）。仓库更新后的 `TERMINOLOGY.md` 亦维持 normalization→规范化、“标准化”用于 standardize 的区分。
- **aspect ratio → “宽高比”（2 处）**：`multiscale-object-detection.md` L91、`rcnn.md` L213 原译“长宽比”；`TERMINOLOGY.md` 第十节作“宽高比”，且 `anchor.md`/`fine-tuning.md`/`image-augmentation.md` 均用“宽高比”。已改为“宽高比”。

### 3. 重要术语首现未括注英文（6 处加注 + 1 处去重）
- `approx-training.md` L31：负采样、层序 softmax 是本批首次引入的核心术语，首现句只作 `*负采样*和*层序 softmax*`；改为 `*负采样*（negative sampling）和*层序 softmax*（hierarchical softmax）`，并删去 L149 处重复的 `（hierarchical softmax）` 括注（同术语只首现括注）。
- `approx-training.md` L90：`*噪声词*` 改为 `*噪声词*（noise words）`（该术语已收入 TERMINOLOGY 第十节）。
- `word2vec.md` L67/L71：`*中心词*`→`*中心词*（center word）`、`*上下文词*`→`*上下文词*（context words）`（context word 见术语表）。
- `glove.md` L48/L51：`*多重集*`→`*多重集*（multiset）`、`*重数*`→`*重数*（multiplicity）`（两条均已收入 TERMINOLOGY 第十节）。
- 以上均只加括注，不改斜体标记、不改原句语义。

### 4. 句法不通（1 处）
- `approx-training.md` L14–18：原句“由于 softmax 运算的性质，**由于**上下文词可能是词表 $\mathcal{V}$ 中的任何一个词，:eqref:`eq_skip-gram-log` 的相反数包含……”连用两个“由于”，第一个分句无着落。改为“由于 softmax 运算的性质，上下文词可能是词表 $\mathcal{V}$ 中的任何一个词，**因此** :eqref:`…` 的相反数包含……”，保持底本因果链。

### 5. 标记形式：`（**…**）` 恢复源文件 ASCII 写法（5 处）
底本用 ASCII `(**…**)`，本批 5 处被译成全角括号 `（**…**）`：`ssd.md` L576、`transposed-conv.md` L85、`fcn.md` L127、`kaggle-cifar10.md` L188、`kaggle-dog.md` L4；已恢复为 `(**…**)`（仅括号字形，括号内文字与粗体标记不动）。依据：源文件即 ASCII；全库其余 33 个文件均为 ASCII 写法，且审校期间其他组也在统一该标记（全角写法由 52 处降至 36 处），本组统一后 G8 文件不再有全角写法。

## 二、逐文件记录

### b16
- `multiscale-object-detection.md`：1 处修改（长宽比→宽高比）。多尺度锚框生成、“小目标出现方式数”示例、感受野/feature map 单元解释、多尺度检测与练习均核对无误。
- `object-detection-dataset.md`：**通读通过，无修改**。香蕉数据集、填充用非法边界框（-1）与标签数组长度 5 的说明、演示与练习均与底本一致。
- `ssd.md`：1 处修改（标记括号）。类别/边界框预测层的 $a(q+1)$ 通道对应关系、多尺度预测连结的形状变换、下采样块感受野 $1\times2+(3-1)+(3-1)=6$、尺度列表 0.2–1.05 与 $\sqrt{s_i s_{i+1}}$、5444 个锚框、$\ell_1$ 与平滑 $\ell_1$ 损失、focal loss 公式等逐项核对无误。
- `rcnn.md`：1 处修改（长宽比→宽高比）。选择性搜索、RoI 汇聚（含 $2\times2$ 窗口取最大示例）、faster R-CNN 的 RPN 四个步骤、mask R-CNN 的 RoI 对齐与全卷积网络说明核对无误。
- `semantic-segmentation-and-dataset.md`：1 处修改（standardize 标准化）。图像分割/实例分割辨析、VOC2012 目录结构、固定形状随机裁剪、`VOCSegDataset` 过滤与规范化说明核对无误；底本两处笔误（`On of the most important…`、`` `VOCSegDatase`t ``）按底本语义/原样处理，未擅改。
- `transposed-conv.md`：1 处修改（标记括号）。步幅 1 无填充的展开过程、对输出的填充/步幅语义、与矩阵转置的关系、前向/反向传播交换说明核对无误。
- `fcn.md`：2 处修改（标准化、标记括号）。$1\times1$ 卷积换通道、步幅 32/卷积核 64/填充 16 的尺寸推导、双线性插值核初始化、跨区域预测取平均等核对无误。
- `neural-style.md`：2 处修改（标准化×2）。内容/样式/全变分损失三项、格拉姆矩阵除以 $chw$、`SynthesizedImage` 作为待更新模型参数、训练与练习均核对无误。

### b17
- `kaggle-cifar10.md`：3 处修改（2 处 `[Discussions]`→`[讨论]`、1 处标记括号）。数据组织/验证集按类划分 $\max(\lfloor nr\rfloor,1)$、批大小 32 与完整数据集的说明、学习率每 4 轮乘 0.9 等核对无误。
- `kaggle-dog.md`：3 处修改（2 处链接文字、1 处标记括号）。120 个犬种译名、复用预训练 ResNet-34 输出层输入、仅训练小型自定义输出网络（不重训特征层以省内存）、学习率每 2 轮乘 0.9 等核对无误；底本标题缺 `:label:`、裸数据路径均照底本。
- `index.md`：**通读通过，无修改**。toc 块与“bank”两个英文例句按底本保留英文（b17 报告说明为有意处理）；自监督学习、词元/词表等术语正确。
- `word2vec.md`：3 处修改（1 处链接文字、中心词/上下文词首现括注）。独热向量的余弦相似度为 0、跳元/连续词袋两个模型的条件概率公式与梯度推导、$\mathrm{vec}$ 语境用词均核对无误。
- `approx-training.md`：5 处修改（1 处链接文字、双重“由于”句、负采样/层序 softmax 首现括注、删除重复括注、噪声词首现括注）。负采样事件独立性改写、$K$ 个噪声词的对数损失推导、层序 softmax 二叉树与 $L(w)-1$ 复杂度说明核对无误。
- `word-embedding-dataset.md`：2 处修改（2 处链接文字）。PTB 语料、$<10$ 次替换为 `<unk>`、子采样概率公式与 $t=10^{-4}$、噪声词采样概率 0.75 次幂、掩码/标签变量与 `batchify` 说明核对无误。
- `word2vec-pretraining.md`：2 处修改（2 处链接文字）。嵌入层权重形状、批量矩阵乘法输出形状、二元交叉熵与掩码平均、`embed_size=100` 等核对无误。
- `glove.md`：3 处修改（1 处链接文字、多重集/重数首现括注）。多重集/重数示例（$\mathcal{C}_i=\{j,j,k,k,k,k,l,m\}$）、$x_{ij}$/共现计数改写、权重函数 $h(x)$、对称性与共现概率比值解释、表格数值均逐项核对无误。
- `subword-embedding.md`：2 处修改（2 处链接文字）。fastText 子词和向量、字符 $n$ 元语法与 `<`/`>` 边界符、BPE 合并与 `segment_BPE` 说明核对无误。
- `similarity-analogy.md`：2 处修改（2 处链接文字）。GloVe/fastText 词向量加载、$k$ 近邻、类比完成 $\mathrm{vec}(c)+\mathrm{vec}(b)-\mathrm{vec}(a)$ 均核对无误。

## 三、遗留疑点（未擅改，提请负责人裁决）

1. **“第 $i^\textrm{th}$ 个/行”中文冗余**：`approx-training.md` L164“第 $j^\textrm{th}$ 个节点”、`word-embedding-dataset.md` L337“第 $i^\textrm{th}$ 个样例”、`word2vec-pretraining.md` L79“第 $i^\textrm{th}$ 行”。“第…个”与 th 重复；因不得改动行内公式，无法把 $i^\textrm{th}$ 改为 $i$。官方中文旧版作“$i^\mathrm{th}$个/…节点”（去“第”）。是否统一为“$i^\textrm{th}$ 个”（去“第”）或保留，请负责人定。
2. **倍数表述**：`fcn.md` 中 “increase … by a factor of 32/2/$s$” 译作“增大 32 倍/增大了两倍/增大 $s$ 倍”（严格中文语义为“增大到 32 倍/两倍”）。官方中文旧版同样作“增加32倍/放大2倍”，故未改；后续如全书统一“增大到 N 倍”需统一处理。
3. **`index.md` 中 “bank” 英文例句**：按底本保留英文原句（b17 报告说明避免增译）；官方旧版采用“英文原句（中文括注）”风格。是否全书统一加括注，请负责人定。
4. **类比符号**：`similarity-analogy.md` 类比记号使用全角“∶/∷”（源文件为 ASCII “: / ::”），视觉等价、可读；是否恢复 ASCII 以严格对齐源文件，请负责人定。
5. **kaggle-dog 底本特征**：标题无 `:label:`；`index.md`/`glove.md` 的英文例句、公式表头、裸路径等仍会被 verify 报“疑似未翻译”warning，均为有意保留/底本原样，非漏译。
6. **建议补入 `TERMINOLOGY.md` 的术语**（本组已统一使用，术语表尚未收录）：center word 中心词、approximate training 近似训练、total variation loss 全变分损失、style transfer 样式迁移（现表内只有 neural style transfer）、content image 内容图像、style image 样式图像、content layer 内容层、style layer 样式层、synthesized image 合成图像、one-hot vector 独热向量、single shot multibox detection 单发多框检测（与“检测器”区分）。本组 34 处修改涉及的 multiset/multiplicity/noise word/aspect ratio/RoI/focal loss/fractionally-strided convolution 等已见于术语表第十节。

## 四、说明

- 仅修改本组负责的 16 个文件、共 34 处定向替换；未修改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md` 及其他批次/组文件；未改动任何代码块、行内代码、数学公式、`:label:`/`:numref:` 等指令、图片路径、URL、标题层级、表格结构。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b16.txt` = 8/8 PASS（0 error、10 warning）；`--batch work/batches/batch-b17.txt` = 10/10 PASS（0 error、13 warning）；warning 均为“内容一致但顺序不同”及专名/例句/路径类提示，与审校前一致。`align_check.py` 复核 b16 段级疑点 0、b17 仅上述 kaggle-dog 裸路径 1 条误报。
