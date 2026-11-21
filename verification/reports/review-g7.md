# 审校组 G7 审校记录（批次 b14、b15）

审校范围：`work/batches/batch-b14.txt`（9 个）+ `batch-b15.txt`（8 个），共 17 个文件：
优化器 4 篇（rmsprop / adadelta / adam / lr-scheduler）、计算性能 5 篇（index / hybridize / async-computation / auto-parallelism / hardware）、多 GPU 与参数服务器 3 篇（multiple-gpus / multiple-gpus-concise / parameterserver）、计算机视觉导言与基础 5 篇（cv index / image-augmentation / fine-tuning / bounding-box / anchor）。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照（代码块、公式、指令、图片路径、URL 只做结构一致性核验，未改动）；重点核对漏译/误译、限定词（通常/可能/在期望意义下）、公式符号解释、术语一致性、图题与表格文字、框架标签页、高亮/斜体标记。只做定向 `edit`，每次修改后 `python work\verify.py --file <path>` 复核。

基线：两批 `verify.py --batch` 均为 0 error；b14 有 4 条、b15 有 6 条“内容一致但顺序不同”警告（中文语序重排所致）。`align_check.py` 段落级漏译检查：两批均 0 个疑点文件。审校后两批仍为 17/17 PASS、0 error，警告数与基线相同（10 条，均可接受）。

## 修改清单（5 个文件、6 处）

1. `chapter_optimization/rmsprop.md`（L169，小结）：`coefficient-wise preconditioner` 原译“调整**按系数顺序的**预条件子”误译（coefficient-wise 是“逐系数/按系数”，不是“按系数顺序”），改为“调整**逐系数的**预条件子”。
2. `chapter_computational-performance/hardware.md`（L125）：`Ample memory channels and an L2 cache complement the setup` 中 `memory channels` 原译“存储通道”；同节（L33）及练习（L218）均作“内存通道”，且此处指 GPU 的显存通道，统一改为“充足的**内存通道**”。
3. `chapter_computational-performance/hardware.md`（L167，延迟表 QPI 行）：`QPI hop to a another CPU (per hop)` 原译“跳到另一个 CPU 的 QPI 跳（每次跳）”，中文语义打转（“跳……跳”），改为“到另一个 CPU 的 QPI 跳（每次跳）”。
4. `chapter_computational-performance/hybridize.md`（L73）：`An imperative mode was added to TensorFlow 2.0 and Keras in later revisions.` 原译“TensorFlow 2.0 及后续修订版的 Keras 中加入了命令式模式”把 `in later revisions` 误挂到 Keras 上（成“后续修订版的 Keras”），改为“**在后来的修订版中，TensorFlow 2.0 和 Keras 加入了命令式模式。**”
5. `chapter_computer-vision/anchor.md`（L57）：`(**consider those combinations containing**) $s_1$ or $r_1$` 原译“（考虑 $s_1$ 或 $r_1$ **对应的**组合）”弱化了 containing 的语义，改为“（考虑**包含** $s_1$ 或 $r_1$ 的组合）”（高亮与行内公式结构不变）。
6. `chapter_computational-performance/async-computation.md`（L173）：`... *before* anything else can be done` 的斜体强调在译文中丢失（原译“先……然后才能……”仅存语义、无标记），按项目样式基准（EXAMPLES 例 5：`*before*`→*之前*）给对应词补斜体：“都要求计算图***先***求值出……，然后才能做其他任何事情”。修改后该文件强调标记计数与底本一致。

## 逐文件记录

### b14

- `chapter_optimization/rmsprop.md`：1 处误译已改（见上）。`泄漏平均（leaky average）`、`预条件子（preconditioner）`、半衰期 $\gamma^{-1}$、权重规范化等与底本一致；公式符号逐字核对无误。
- `chapter_optimization/adadelta.md`：通读通过，无修改。`parameter du jour` 意译“当前的参数”可接受；底本练习第 4 题作 “RMS prop”（不规范拼写），译文统一 RMSProp，属合理处理；四个 `$$\begin{aligned}...$$` 与译文一致。
- `chapter_optimization/adam.md`：通读通过，无修改。`exponential weighted moving average`/`泄漏平均`、`bias correction`（偏差修正）、Yogi 更新式、`sgn` 公式均核对无误；Yogi 节各框架代码 `eps` 取值差异系底本原样，未动代码。
- `chapter_optimization/lr-scheduler.md`：通读通过，无修改。`learning rate scheduling` 作“学习率调度”、scheduler 作“调度器”，与 TERMINOLOGY 一致；多因子调度句因中文语序（“只要 $t \in s$ 就令……”）触发 1 条行内公式顺序 warning，多重集一致，可接受；余弦调度、预热、`PolyScheduler` 等述及与底本一致。
- `chapter_computational-performance/index.md`：通读通过，无修改。命令式/符号式编程、异步计算、自动并行、多 GPU 计算等表述与底本一致；`toc` 块原样。
- `chapter_computational-performance/hybridize.md`：1 处误译已改（见上）。`混合式编程（hybridization）`与动词“混合化”的区分、Theano/Chainer/Keras/CNTK/TensorFlow 2.0 历史、`HybridBlock`/`HybridSequential`、序列化各框架段落均核对；底本 TF 段落拼写错误（behavor/cen/direcly）按其语义译出，未另加译注。1 条行内代码顺序 warning 系语句重排，可接受。
- `chapter_computational-performance/async-computation.md`：1 处强调标记已补（见上）。前端/后端、`屏障与阻塞`、隐式阻塞（`print`/`asnumpy`/`item`）、同步与异步 10000 次计算耗时推导（$t_1+10000t_2+t_3$ 及 $10000t_2>9999t_1$ 条件）均与底本一致。
- `chapter_computational-performance/auto-parallelism.md`：通读通过，无修改。2 条 warning 分别为行内代码、引用指令顺序（中文语序重排），多重集一致。**疑点**：底本正文称 `run` 做 “10 matrix-matrix multiplications”、练习第 1 题称 “Eight operations”，而代码为 `range(50)`，三处自相矛盾；译文按底本直译（正文“10 次”、练习“八次”），未擅自统一，留待负责人定夺。
- `chapter_computational-performance/hardware.md`：2 处修改（见上）。两张延迟表逐行核对：行列数、`:---` 对齐行、数字/单位/品牌（Broadwell E5-2690v4、TinyMemBench、DC P3608、MVAPICH2、Omni-Path 等）均保留；表题 `:Common Latency Numbers.` 按 D2L 语法译作 `:常见延迟数字。`；正文 `4 0GB/s` 底本笔误按 40GB/s 处理。**疑点**：PCIe 条目底本 “(up to 32 GB/s on PCIe 4.0 in a 16-lane slot) per lane” 自身矛盾（32 GB/s 为插槽级、非每通道），译文按插槽级 32 GB/s 译出，未加译注。

### b15

- `chapter_computational-performance/multiple-gpus.md`：通读通过，无修改。`模型并行（model parallelism）`首次出现在图题并括注英文，`数据并行`全篇一致；`nvidia-smi`、`allreduce`、`get_params`、`split_and_load`、`train_batch` 等标识与原码保留；底本 “exchanging gradients parameters” 语法有误，按语义译“交换梯度参数”。`[**...**]` 高亮块计数与底本一致。
- `chapter_computational-performance/multiple-gpus-concise.md`：通读通过，无修改。ResNet-18 变体、`initialize`/`split_and_load` 说明、各框架标签页（含只属 mxnet 的小结条目）与底本结构一致；1 条行内代码顺序 warning 为 `evaluate_accuracy_gpu` 与 `:numref:` 语序重排，可接受。
- `chapter_computational-performance/parameterserver.md`：通读通过，无修改。`环同步（ring synchronization）`、`键值存储（key--value store）`、`可交换归约（commutative reduction）`、`工作节点`、`容错（fault tolerance）`等术语与底本及 b15 报告一致；$\mathcal{O}(m)$、$\mathcal{O}(1/n)$、$\mathcal{O}(m/n)$ 与独立公式逐字保留；1 条行内公式顺序 warning（工作节点 $k$ 的 GPU $j$ 语序）可接受。
- `chapter_computer-vision/index.md`：通读通过，无修改。`图像增广（image augmentation）`、`微调（fine-tuning）`、`目标检测（object detection）`、`语义分割（semantic segmentation）`、`样式迁移（style transfer）`、`全卷积网络（fully convolutional network）` 首次出现均括注英文；`toc` 块原样。
- `chapter_computer-vision/image-augmentation.md`：通读通过，无修改。翻转/裁剪/颜色（亮度、对比度、饱和度、色调）/组合多种方法各节与底本逐段一致；`RandomFlipLeftRight`、`RandomHorizontalFlip`、`RandomVerticalFlip`、`ColorJitter`、`Compose`、`ToTensor`、`transform_first` 等 API 保留；1 条行内代码顺序 warning 可接受。
- `chapter_computer-vision/fine-tuning.md`：通读通过，无修改。`[~~数据增广~~]` 删除线结构保留；`*标准化*（standardize）`与术语表 `normalization（规范化）`区分正确；`features/output`、`fc`、`pretrained=True`、`10\eta` 学习率说明与底本一致；全局平均汇聚译法与 b15 一致。
- `chapter_computer-vision/bounding-box.md`：通读通过，无修改。`边界框（bounding box）`、两角表示/中心-宽度-高度表示、`box_corner_to_center`/`box_center_to_corner`、图像坐标系说明均与底本一致；1 条行内代码顺序 warning（练习句语序）可接受。
- `chapter_computer-vision/anchor.md`：1 处修改（见上）。核心术语按 TERMINOLOGY：锚框、真实边界框（ground-truth）、交并比（IoU）、偏移量（offset）、非极大值抑制（NMS）、正/负锚框、宽高比、缩放比；`ASSIGNING` 算法四步骤、偏移量变换公式与 $\mu/\sigma$ 默认值、`multibox_target`/`offset_inverse`/`nms`/`multibox_detection` 说明均核对无误；高亮块内嵌独立公式按 `[** ... $$...$$ **]` 原结构保留；2 条 warning（行内公式、行内代码顺序）均为中文语序所致，多重集一致。

## 遗留疑点（未擅改，提请负责人裁决）

1. **hardware.md PCIe 数值矛盾**：底本 “(up to 32 GB/s on PCIe 4.0 in a 16-lane slot) per lane” 中 32 GB/s 与 `per lane` 不可能同时成立（32 GB/s 是 ×16 插槽的聚合带宽）。现译文按插槽级处理（未译 `per lane`）。建议定稿前确认是保留现状、还是直译并加译注。
2. **hardware.md “4 0GB/s”**：底本显系 “40 GB/s” 笔误，译文按 40GB/s 处理。若要求逐字符忠于底本，可再议。
3. **auto-parallelism.md 底本三处数字不一致**：正文 “10 matrix-matrix multiplications”、练习 “Eight operations”、代码 `range(50)`。现按底本直译，不建议译者擅自统一，请负责人决定是否加译注。
4. **Jaccard index 译名**：anchor.md 全书 3 处均作“Jaccard 指数”，TERMINOLOGY 未收录；官方旧版作“杰卡德系数”。是否统一为“杰卡德系数（Jaccard index）”或“Jaccard 系数”请负责人定夺。
5. **`数据并行`首次出现未括注英文**：multiple-gpus.md 图题/正文均无 “（data parallelism）”，而同图题中“模型并行”已括注。官方旧版亦未括注，故未改；是否按“重要术语首现括注英文”的统一规则补注，请负责人定夺。
6. **建议合入 TERMINOLOGY 的新术语**（本组已用且内部一致）：preconditioner 预条件子；leaky average 泄漏平均；burst read 突发读取；false sharing 伪共享；systolic array 脉动阵列；streaming multiprocessor 流式多处理器；ring synchronization 环同步；commutative reduction 可交换归约；fault tolerance 容错；key--value store 键值存储；aspect ratio 宽高比；scale（锚框）缩放比；positive/negative anchor box 正/负锚框；style transfer 样式迁移（b15 已提出）。
7. **通读中未改的次要生硬点**（供负责人参考，不属误译）：hardware.md “CPUs have between 4 0GB/s...”（已按笔误处理）、“CPUs have 2 and 4 memory channels”；hybridize.md “编程这样的器件对普通深度学习研究者来说被很好地隐藏了起来”被动句式略拗口；adadelta.md “由此得到如下一行代码，调用一下子就紧凑多了”偏口语。以上均为可读中文，按“不过度修改”原则保留。

## 说明

- 仅修改本组负责的 5 个文件、6 处定向替换；未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md` 及其他批次/组文件，未改动任何代码块、行内代码、公式、指令、图片路径、URL、标题层级、表格结构。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b14.txt` = 9/9 PASS（0 error、4 warning）；`--batch work/batches/batch-b15.txt` = 8/8 PASS（0 error、6 warning）。警告均为“内容一致但顺序不同”，与审校前一致。
