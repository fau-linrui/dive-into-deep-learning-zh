# 审校记录：G3（批次 b06 + b07）

- 审校者：r3（审校组 G3）
- 范围：`work/batches/batch-b06.txt`（11 篇）+ `batch-b07.txt`（8 篇），共 19 篇、7235 行、约 13.8 万非空字符
  - 多层感知机：`dropout.md`、`kaggle-house-price.md`
  - 深度学习计算（builders-guide）：index、model-construction、parameters、init-param、lazy-init、custom-layer、read-write、use-gpu
  - 卷积神经网络：index、why-conv、conv-layer、padding-and-strides、channels、pooling、lenet
  - 现代卷积网络：index、alexnet
- 依据：`work/REVIEW_GUIDE.md`、`d2l-zh-translation/TERMINOLOGY.md`、`work/EXAMPLES.md`（官方中文版仅用于核对译名）
- 结论：19 篇全部通读并与 `d2l-en-src/` 逐段对照；**3 篇有修改，共 6 处**；修改后 `python work\verify.py` 逐文件均为 **0 error**；`align_check.py` 无实质漏译疑点。

## 一、修改明细（3 个文件 / 6 处）

| # | 文件 | 位置 | 类别 | 修改前 → 修改后 | 原因 |
|---|---|---|---|---|---|
| 1 | chapter_multilayer-perceptrons/dropout.md | L196 | 标记格式 | 我们可以（**在几个样例上测试 \`dropout_layer\` 函数**）。 → 我们可以[**在几个样例上测试 \`dropout_layer\` 函数**]。 | 原文为 `[**...**]`（D2L 重点高亮语法），原译误作全角括号+粗体；按 EXAMPLES 例 1/6 与 TRANSLATION_GUIDE §9 恢复方括号标记，只动标记不动文字 |
| 2 | chapter_multilayer-perceptrons/dropout.md | L434 | 标记格式 | 接下来，我们（**训练模型**）。 → 接下来，我们[**训练模型**]。 | 同上（原文 `[**train the model**]`） |
| 3 | chapter_builders-guide/custom-layer.md | L134-135 | 表达（重复虚词） | 由于我们处理的是浮点数，／由于量化的原因，我们可能仍会看到／一个非常小的非零数。 → 由于我们处理的是浮点数，受量化影响，／我们可能仍会看到一个非常小的非零数。 | 原译连续两个"由于…由于…"叠用、句子割裂，属"表达生硬"；语义（become of floating point / due to quantization）不变 |
| 4 | chapter_convolutional-neural-networks/why-conv.md | L4-5 | 表达（同义重复） | 时至今日，／我们迄今讨论的模型 → 时至今日，／我们此前讨论的模型 | "时至今日"与"迄今"语义重复，读来拗口；"To this day, the models that we have discussed so far" |
| 5 | chapter_convolutional-neural-networks/why-conv.md | L209 | 用词（增字） | 例如图像结果并不具有平移不变性 → 例如图像并不具有平移不变性 | 原文"if images turned out not to be translation invariant"，"结果"为无据增字 |
| 6 | chapter_convolutional-neural-networks/why-conv.md | L326 | 用词（搭配） | 这些图像报告了许多不同波长的数据。 → 这些图像记录了许多不同波长的数据。 | "report data on many different wavelengths" 中 report 译"报告"不符合中文搭配（指卫星/高光谱图像含数据），改"记录" |

修改均为定向 `edit`，未触碰代码块、行内代码、公式、`:label:`/`:numref:` 等指令、图片路径、URL、标题层级与列表结构。

## 二、逐文件结论

| 文件 | 结果 |
|---|---|
| chapter_multilayer-perceptrons/dropout.md | 已改 2 处（`[**...**]` 标记），其余通读通过 |
| chapter_multilayer-perceptrons/kaggle-house-price.md | 通读通过，无修改 |
| chapter_builders-guide/index.md | 通读通过，无修改 |
| chapter_builders-guide/model-construction.md | 通读通过，无修改 |
| chapter_builders-guide/parameters.md | 通读通过，无修改 |
| chapter_builders-guide/init-param.md | 通读通过，无修改 |
| chapter_builders-guide/lazy-init.md | 通读通过，无修改 |
| chapter_builders-guide/custom-layer.md | 已改 1 处（重复"由于"），其余通读通过 |
| chapter_builders-guide/read-write.md | 通读通过，无修改 |
| chapter_builders-guide/use-gpu.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/index.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/why-conv.md | 已改 3 处（重复"迄今"、"图像结果"、report 搭配），其余通读通过 |
| chapter_convolutional-neural-networks/conv-layer.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/padding-and-strides.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/channels.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/pooling.md | 通读通过，无修改 |
| chapter_convolutional-neural-networks/lenet.md | 通读通过，无修改 |
| chapter_convolutional-modern/index.md | 通读通过，无修改 |
| chapter_convolutional-modern/alexnet.md | 通读通过，无修改 |

## 三、辅助检查记录

- `python work\verify.py --batch work/batches/batch-b06.txt`：11/11 PASS，0 error（6 条 warning 均为"行内公式/行内代码内容一致但顺序与原文不同"，系中文语序调整所致，已人工确认为可接受）。
- `python work\verify.py --batch work/batches/batch-b07.txt`：8/8 PASS，0 error（8 条 warning 同上；alexnet 另有 1 条"疑似未翻译的英文行"，为括注的官方赛事英文全名 "ImageNet Large Scale Visual Recognition Challenge"，属专名，应保留）。
- `python work\align_check.py --batch`（b06、b07）：b06 仅 kaggle-house-price 第 54 段提示"中文偏少"（英文 62 词→中文 55 字）；经人工核对，该段是 Kaggle 提交流程列表（"Submit Predictions"/"Late Submission"/"Upload Submission File"/"Make Submission" 等界面按钮名按惯例保留英文），并非漏译。b07 无提示。
- `python work\termcheck.py`：19 篇中未发现术语变体（"池化/归一化/丢弃法/学习速率"等均无）；唯一命中是 dropout.md 中"随机丢弃前一层的输出"——此处是描述 Dropout 层动作的动词短语（randomly drop out outputs），非术语"暂退法"误用，按语境保留。
- 额外人工核对项：强调标记（`**`/`*`/`[**...**]`）逐文件与英文对齐（仅 dropout 2 处不符已修）；全部图片替代文本均已译；无表格与脚注；数字/单位/专名（Fashion-MNIST、ImageNet、LeNet、AlexNet、NVIDIA、CUDA 等）均按底本保留；否定/条件/限定词（not、unless、rather than、may、typically 等）抽样核对无丢失或反义；各节小结、练习子题数与原文一致。

## 四、遗留疑点（需负责人裁决，未擅自改动）

1. **coarse 的译法**：`chapter_builders-guide/index.md` L12"越来越粗糙的抽象"、L17"粗糙得多的块"，`pooling.md` L14"越来越粗糙的特征图"。官方中文版亦用"粗糙"，故本批保持全书一致、未改；若要改为更准确的"粗粒度（抽象/块/特征图）"需全书统一裁决。
2. **英文底本疑似笔误（按 REVIEW_GUIDE §3 直译保留并记录）**：
   - `padding-and-strides.md` 小结：原文"horizontal stride $s_\textrm{h}$ and vertical stride $s_\textrm{w}$"与正文定义（$s_\textrm{h}$ 高度步幅、$s_\textrm{w}$ 宽度步幅）相反；译文照底本作"水平步幅$s_\textrm{h}$与垂直步幅$s_\textrm{w}$"。
   - `use-gpu.md` 原文"`gpu:0` 和 `gpu` 是等价的"（PyTorch 实际写法为 `cuda:0`/`cuda`），照译未改。
   - `alexnet.md`"AlexNet 的卷积通道数比 LeNet 多十倍"（ten times more；实际实现首层 96 vs LeNet 6，为 16 倍），按底本直译。
   - `lenet.md` 复现说明中的"高斯激活层/高斯解码器"（Gaussian activation layer / Gaussian decoder）表述为底本原文，保留。
   - `why-conv.md` L137"受 $[\mathsf{V}]_{i, j, a, b}$ 加权的 $x$ 中的像素"，底本为小写 $x$（疑为 $\mathbf{X}$ 笔误），符号未改动。
   - `chapter_convolutional-neural-networks/index.md` 底本"Imagnet"拼写笔误，译文按 ImageNet 处理（审校手册允许专名笔误的译者处理）；`why-conv.md` 中"waldo-football""waldoness"等原书用词保留其译法"沃尔多/沃尔多属性"。
3. **version 差异**：本批译文对应 d2l-en 1.0.3 中 `module` 用法（译为"模块"），而官方中文旧版为"块/Block"体系；本批"模块（module）"与 TERMINOLOGY"block→块"并不冲突（`块` 仍指 block），仅作提示，无需改动。

## 五、统计

- 审校文件数：19
- 修改文件数：3（dropout.md、custom-layer.md、why-conv.md）
- 修改处数：6（标记格式 2、表达/用词 4）
- 修改后 verify：19/19 PASS，0 error
- 遗留疑点：4 组（术语裁决 1、底本笔误 1 组 6 条、版本术语提示 1）
