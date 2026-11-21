# 审校组 G4 审校记录（批次 b08、b09）

审校范围：`work/batches/batch-b08.txt` + `batch-b09.txt`，共 12 个文件
（现代卷积网络 7 篇：VGG/NiN/GoogLeNet/批量规范化/ResNet/DenseNet/CNN 设计空间；
循环神经网络基础 5 篇：章节索引/序列模型/文本序列/语言模型/RNN）。

方法：逐文件通读中文译文，与英文底本 `d2l-en-src/<path>` 逐段对照（定义、公式符号说明、因果与限定词、图题、表格、小结、练习必须核对；数学密集段落逐句核对）；只做定向 `edit`，不改代码块/行内代码/公式/指令/图片路径/URL/标题层级/表格结构；每次修改后 `python work\verify.py --file/--batch` 校验。另用临时脚本比对了中英文数字覆盖率与 `[**...**]` 高亮跨度计数（脚本已删除）：
- 数字比对：英文数词按规范转数字（"eight/three"→8/3、"five billion"→"50亿"、"30,000 words"→"三万余词"）均正确，无漏译数字；
- 高亮跨度：发现 densenet.md 一处高亮拆分（已修），其余文件 `[**…**]` 数量与英文一致。

基线：两批 `verify.py --batch` 均为 0 error；`align_check.py` 段级检查 b08 无提示，b09 仅 densenet 段落 19 "中文偏少"、language-model 的英文示例句 2 条提示，经核均为误报（见各文件记录）。

## 逐文件记录

### chapter_convolutional-modern/vgg.md
- 通读通过，无修改。核对：$\log_2 d$ 层数上界论证、$25c^2$ 与 $3\cdot9c^2$ 的参数比较、VGG-11 的 8 卷积层+3 全连接层、块定义、图题、练习 4（上采样 224 维）均与底本一致。

### chapter_convolutional-modern/nin.md
- 修正 2 处生硬句：
  - "很难为将大部分内存花在一个图像分类器上找到正当理由"→"很难有正当理由将大部分内存花在一个图像分类器上"（"为将……找到正当理由"语序拗口）。
  - "选择更少的使用宽核的卷积……它可以为任意给定位置在通道之间提供相当程度的非线性"→"少用宽核卷积、改用 $1 \times 1$ 卷积……在每个给定位置上，它都能在通道之间提供相当程度的非线性"（"的"字叠加、状语语序不顺）。
- 其余（NiN 块/1×1 卷积解释、全局平均汇聚、logit、练习）与底本一致。

### chapter_convolutional-modern/googlenet.md
- 修正 3 处：
  - "这种设计权衡评估网络的成本与误差的降低"→"这种设计在评估网络的成本与误差降低之间进行权衡"（原句成分粘连，缺"在……之间"）。
  - 练习 1 两处 "如 :numref:`sec_batch_norm` 后面所述"→"如后文 :numref:`sec_batch_norm` 所述"；`sec_resnet` 同处理。
- 其余（Inception 块描述、b1–b5 通道数推导、9 个 Inception 块=3 组、讨论节）与底本一致。
- 疑点见文末第 1 条（"并结束了主体的设计"）。

### chapter_convolutional-modern/batch-norm.md
- 修正 2 处表达：
  - "可能*对每个观测*具有零均值"→"且可能使*每个观测*的均值为零"（possibly zero mean per observation 的语序）。
  - "我们可能需要一次应用模型做一个预测"→"我们可能需要每次只应用模型做一次预测"。
- 公式/符号（BN 式、$\hat{\boldsymbol{\mu}}_\mathcal{B}$/$\hat{\boldsymbol{\sigma}}_\mathcal{B}$ 公式与 $\epsilon$ 说明、LN 式、训练/预测模式）、jax `batch_stats` 标签页说明、讨论节（内部协变量偏移、Rahimi 演讲）、练习均核对无误。

### chapter_convolutional-modern/resnet.md
- 通读通过，无修改。核对：函数类/嵌套函数类论证、残差映射 $g(\mathbf{x})=f(\mathbf{x})-\mathbf{x}$、ResNet-18 层数、分组卷积 $O(c_\mathrm{i}c_\mathrm{o}/g)$ 与参数减少 $g$ 倍、ResNeXt 块描述、图题、练习均与底本一致。

### chapter_convolutional-modern/densenet.md
- 修正 1 处格式问题（`verify.py` 不检查此项，靠人工+计数发现）：过渡层示例句的高亮跨度与英文不一致——英文 `[**Apply a transition layer**]` 为**单一**高亮，译文拆成 `[**应用**]一个通道数为10的[**过渡层**]` 两段高亮（译文 `[**` 计数 7，英文 6）。已合并为 `[**应用一个通道数为10的过渡层**]`。
- 其余通读通过；`align_check` 段落 19 "中文偏少"提示为误报（英文 59 词→中文 56 字，信息完整）。

### chapter_convolutional-modern/cnn-design.md
- 通读通过，无修改。核对：AnyNet 主干/主体/头部、17 个设计参数、$2^{17}=131072$、CDF 优超论证、RegNetX32 参数（$k=1,g=16,c_1=32,c_2=80,d_1=4,d_2=6$）、讨论节（scalability trumps inductive biases）、练习均与底本一致。
- 疑点见文末第 4 条（"quarter the spatial resolution"）。

### chapter_recurrent-neural-networks/index.md
- 修正 1 处："长度可变的定长向量序列"→"由定长向量组成的变长序列"（varying-length sequences of fixed-length vectors；原译"可变的定长"字面自相矛盾）。
- 其余（表格型数据、序列任务、RNN 展开、Turing/Transformer 等）与底本一致。

### chapter_recurrent-neural-networks/sequence.md
- 修正 3 处：
  - "按仿佛马尔可夫条件成立的方式运作的模型"→"表现得仿佛马尔可夫条件成立的模型"（"按仿佛"不成词）。
  - "三角函数`sin`函数"→"三角函数`sin`"（"函数"重复）。
  - 练习 1："纳入多于过去四个观测？"→"纳入四个以上的历史观测？"
- 其余（自回归/潜在自回归、马尔可夫条件、解码顺序、误差累积、$k$ 步预测实验与图题）与底本一致。

### chapter_recurrent-neural-networks/text-sequence.md
- 通读通过，无修改。核对：预处理四步、词元/词表、`<unk>`、`build` 方法、齐普夫定律 $n_i\propto 1/i^\alpha$ 与 $\log n_i=-\alpha\log i+c$、停用词与二元/三元语法统计、练习。
- `verify.py` 提示第 30 行 "Baby needs a new pair of shoes" 为**示例句**（讨论 7 词/30 字符词元化），保留英文不算漏译。

### chapter_recurrent-neural-networks/language-model.md
- 修正 1 处："在 :numref:`sec_sequence` 中的那些序列模型分析中"→"在 :numref:`sec_sequence` 介绍的那些序列模型分析中"（"中……中"重复）。
- 其余（马尔可夫与 n 元语法近似、单词频率、拉普拉斯平滑及 $\epsilon_1/\epsilon_2/\epsilon_3$ 说明、困惑度的几何平均解释、序列划分 $d\in[0,n)$ 与 $m=\lfloor(T-d)/n\rfloor$、目标平移一个词元、小结/练习）均与底本一致。
- `verify.py` 提示的 3 行 "It is raining…" 为模型输出示例句，保留英文不算漏译；`align_check` 对应对误报。

### chapter_recurrent-neural-networks/rnn.md
- 修正 1 处：隐藏状态计算"等价于 $\mathbf{X}_t$ 与 $\mathbf{H}_{t-1}$ 的连结和 $\mathbf{W}_{\mathrm{xh}}$ 与 $\mathbf{W}_{\mathrm{hh}}$ 的连结的矩阵乘法"（三个"的"嵌套、成分不清）→"等价于对……的连结和……的连结做矩阵乘法"。
- 其余（隐藏层 vs 隐藏状态、RNN 参数不随时间步增长、字符级语言模型、练习）与底本一致。

## 修改统计
- 审校文件：**12**；修改文件：**8**（nin/googlenet/batch-norm/densenet/index/sequence/language-model/rnn），未修改 4（vgg/resnet/cnn-design/text-sequence）；定向 edit 共 **13** 处。
- 代码块、行内代码、公式、`:label:`/`:cite:` 等指令、图片路径、URL、标题层级、表格结构均未改动；唯一的格式类修改是把 densenet.md 一处被拆分的 `[**…**]` 高亮合并回单个跨度。
- 最终校验：`verify.py --batch batch-b08` 5/5 PASS、`verify.py --batch batch-b09` 7/7 PASS，**0 error**。剩余警告均为"行内公式/行内代码内容一致但顺序与原文不同"（因句子重排，已人工确认无内容缺失）及保留英文的示例句提示。

## 遗留疑点（未改，待负责人裁决）
1. **googlenet.md 第 240 行**："这对应于 Inception 块中的第二个分支，并结束了**主体（body）**的设计。" 按本节定义，主干（stem）= 前两三个卷积（b1 的 $7\times7$ + b2 的 $1\times1$、$3\times3$ 恰为三个卷积），主体（body）= b3–b5 的三个 Inception 组；此处说 b2 "结束了主体的设计"与前后文矛盾，疑为**英文底本笔误（应为 stem）**。译文按底本直译，未改技术内容，建议核对 upstream。官方中文版同样如此，属底本层面的问题。
2. **batch-norm.md "概念漂移"**：`work\termcheck.py` 将其列为"概念偏移"的变体（该脚本按 TERMINOLOGY 约定扫描）。但该处原文为 *concept drift*，中文通行译法即"概念漂移"；TERMINOLOGY 未收录该词条。建议不按 termcheck 机械替换，并考虑把 concept drift→概念漂移 加入术语表/白名单。
3. **中英文间空格不统一（全书层面）**：本组内 vgg/nin/googlenet/batch-norm/resnet 在中文与拉丁词之间统一加空格；densenet.md、cnn-design.md 几乎不加空格（如"ResNet极大改变…""自AlexNet（…）"），index/sequence/text-sequence/language-model/rnn 也存在部分缩写紧贴中文的情况（如"RNN的"）。`TRANSLATION_GUIDE.md` 规定"不强制空格，保持全书观感统一即可"，故未做全书批量替换；若要求统一，需负责人决定统一规则后全书处理。
4. **cnn-design.md "quarter the spatial resolution"**：译作"将空间分辨率缩小为原来的四分之一"，与同段"将分辨率从 $r\times r$ 减半为 $r/2\times r/2$"字面看似矛盾（英文底本如此；"resolution"应为面积/像素数口径，主干与每个阶段把像素数减为 1/4、线性尺寸减半）。保留直译，供负责人知悉。
