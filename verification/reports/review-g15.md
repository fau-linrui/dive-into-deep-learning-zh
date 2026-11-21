# 审校组 G15 审校记录（批次 b24：工具附录 + 参考文献）

审校范围：`work/batches/batch-b24.txt` 全部 9 个文件（工具附录：Jupyter、SageMaker、AWS、Colab、选择服务器与GPU、参与贡献、工具函数 utils、`d2l` 包 API 文档 + 参考文献 zreferences）。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按“英文段—中文段”逐段对照。工具附录属操作说明类，重点核对：①操作步骤/界面路径描述是否准确（Jupyter 菜单项、SageMaker 控制台与实例、EC2 各步骤、Colab 按钮与警告、GitHub 贡献流程、Git 命令）；②界面名/命令/类名/函数名/URL 是否保留英文且未变形；③代码块（`bash`/`python` 围栏）与 `eval_rst` 块是否原样；④术语按 `TERMINOLOGY.md`（复刻/拉取请求/实例类型/笔记本实例/竞价实例/公版设计/过热降频/工具函数/张量核心等）；⑤图题、图片替代文本、表题、表格单元格、小结、练习是否完整；⑥中文是否自然。`utils.md` 约 97% 为代码围栏（1183 行中块外仅 46 行、其中成句散文 12 行），逐行核对块外散文与代码前说明，代码本身由 verify 逐字校验。只做定向 `edit`；未改动代码块、行内代码、公式、指令、图片路径、URL、标题层级、表格结构。

基线：`python work\verify.py --batch work\batches\batch-b24.txt` = **9/9 PASS，0 error，5 warning**（jupyter 行内公式顺序、colab/contributing 行内代码顺序各 1 条，均系中文语序重排；d2l“英文源本身标签块不配对”；zreferences“页面标题已按中文译出”有意例外）。`python work\align_check.py --batch work\batches\batch-b24.txt` = 1 个段级疑点（aws 段落 38@源 L139），人工核对为引号内英文提示语被计入英文词数造成的**假阳性**。

审校后：`python work\verify.py --batch work\batches\batch-b24.txt` 仍为 **9/9 PASS，0 error，5 warning**（警告数与基线完全一致，已逐条确认多重集一致/源本身现象）；单文件复验均 0 error。随后已运行 `python work\align_check.py`（全书）重生成 `work/reports/align.json`，保持其全书报告性质。

## 修改清单（1 个文件、2 处）

1. `chapter_appendix-tools-for-deep-learning/aws.md`（L74）：底本 `For the purpose of illustration, a p2.xlarge will suffice` 原译“为了演示，一块p2.xlarge 就足够了”——`p2.xlarge` 是 EC2 **实例（虚拟机）**而非板卡/GPU，量词“块”误用；改为“为了演示，**一台**p2.xlarge 就足够了”。
2. `chapter_appendix-tools-for-deep-learning/aws.md`（L7）：底本 `Install CUDA (or use an Amazon Machine Image with preinstalled CUDA).` 原译“安装CUDA（或使用预装了CUDA 的亚马逊机器镜像（Amazon Machine Image））。”出现全角括号嵌套（两个“）”连排，且打断序号项可读性）；在语义、词语与首次括注方式完全不变的前提下改为“安装CUDA，**或**使用预装了CUDA 的亚马逊机器镜像（Amazon Machine Image）。”

## 逐文件记录

### 工具附录（`chapter_appendix-tools-for-deep-learning/`）

- `jupyter.md`：**通读通过，无修改**。逐条核对：本地编辑流程（`.ipynb` 后缀、临时 `test.ipynb`、Markdown 单元格与代码单元格、双击编辑、`Ctrl + Enter`、菜单 “Cell”→“Run Cells”、“Kernel”→“Restart & Run All”、“Help”→“Edit Keyboard Shortcuts”）；notedown 插件安装/生成配置文件/`c.NotebookApp.contents_manager_class`、ExecuteTime 插件三条命令、`ssh myserver -L 8888:localhost:8888` 端口转发、“高级选项”整节论证（ipynb 辅助数据使 Git 审阅困难→以 Markdown 原生编辑替代）；小结/练习完整，`*远程*` 斜体标记保留，界面项全部保留英文。1 条 warning 为练习 3 中 $\mathbb{R}^{1024 \times 1024}$ 因中文语序提前到两个运算之前，多重集一致。
- `sagemaker.md`：**通读通过，无修改**。核对：注册（双重身份验证、账单提醒）、控制台搜索 “Amazon SageMaker”、`ml.p3.2xlarge`（1×Tesla V100 + 8 核 CPU）、实例类型链接、创建实例时指定名称/类型、mxnet/pytorch/tensorflow 三个标签页的 GitHub 仓库 URL（`d2l-en-sagemaker` / `d2l-pytorch-sagemaker` / `d2l-tensorflow-sagemaker`）、“Open Jupyter”链接、停止实例以免继续计费、终端更新笔记本（`git reset --hard` / `git pull`）、小结与练习；界面名与命令保留英文，URL 逐一核对无变形。
- `aws.md`：**修改 2 处**（见上）。核对：EC2 面板进入、预设位置（Oregon、首尔/东京、部分数据中心无 GPU 实例）、Limits 配额检查与“Request limit increase”（一个工作日）、Launch Instance、AMI/Ubuntu、`:numref:\`tab_ec2\`` 实例类型表逐格核对（g2/p2/g3/p3/p4/g4 六行，`Grid K520`/`Kepler K80`/`Maxwell M60`/`Volta V100`/`Ampere A100`/`Turing T4`，“老旧/较旧但作为竞价实例通常很便宜/性价比良好/FP16 性能高/针对大规模训练的高性能/针对 FP16/INT8 推理优化”）、p2.xlarge 与 p2.16xlarge 的 GPU 数、密钥对创建与下载、Network settings/Edit、硬盘增至 64 GB、CUDA 占用 4 GB、`chmod 400 D2L_key.pem`、SSH 登录命令与 “Are you sure you want to continue connecting (yes/no)” 提示、CUDA 12.1 官方仓库下载与安装块、`nvidia-smi`、`~/.bashrc` 追加 PATH/LD_LIBRARY_PATH、Miniconda 两条远程提示、SSH 端口转发 `-L 8889:localhost:8888` 与把 URL 端口 8888 改为 8889 的说明、“停止/终止”区别、“Image”→“Create” 制作镜像、“Instance State”→“Terminate”、“My AMIs” 复用镜像、小结与练习；界面名全部保留英文，命令块与底本逐字一致。
- `colab.md`：**通读通过，无修改**。核对：AWS 两节回顾、`Colab` 按钮、首次运行 “RUN ANYWAY” 警告、“需要 GPU 时连接 GPU 实例”、小结与练习；界面名保留英文。1 条行内代码顺序 warning 系 :numref: 与 `Colab` 按钮在中文句中换序，多重集一致。
- `selecting-servers-gpus.md`：**通读通过，无修改**。核对：服务器选择（GPU 与 CPU 性价比、全局解释器锁 GIL 与 4~8 块 GPU 时的单线程性能、6 核 4 GHz vs 8 核 3.5 GHz、电源/机箱/GPU 散热/PCIe 插槽四条准则，350W 预算与*峰值需求*、*公版设计*、过热降频、16$\times$/8$\times$/4$\times$ 带宽）；三档配置建议逐项数字（32/64/128/256 GB DRAM、600W/1000W/1600~2000W、60mm 间距、2+1 冗余电源、10 GBE、1~2 TB NVMe RAID、PLX 复用通道）；GPU 选择（AMD/NVIDIA、GTX/RTX/Tesla、HBM2/GDDR6、FP16/INT8/INT4、TensorCore 加速 5$\times$、显存不小于 4 GB、GUI 占用显存加 2 GB）；两张图的读图结论（GTX 900/1000/RTX 2000 价格-算力、980 Ti vs 1080 Ti、RTX 2000 的*仿射*函数关系；能耗图与 TensorCore 功耗）；斜体术语（*峰值需求*/*公版设计*/*仿射*/*又吵又热*/*物理外形*）均保留；数字、单位、型号逐项核对无误。
- `contributing.md`：**通读通过，无修改**。核对：贡献方式（GitHub 上 Find file → Edit this file → Propose file change → Create pull request）、markdown 格式与 D2L-Book 扩展、`%%tab pytorch/tensorflow/all` 与 `d2lbook`、安装 Git（`apt install git`/Xcode/GitHub 桌面客户端）、`Fork` 按钮与*你的副本*、复刻后 `astonzhang/d2l-en`、克隆仓库（`*你自己的复刻*`）与 `git clone` 命令、`git status` 输出块、`git add/commit/push`、New pull request、审阅与合并反馈、小结与练习；GitHub 界面名保留英文，代码块与底本逐字一致（`# Replace your_github_username with your GitHub username` 注释按项目规范保留英文）。
- `utils.md`：**通读通过，无修改**。本文件 1183 行中代码围栏占绝对多数，逐行核对块外全部散文（标题“工具函数和类”、引言、Hyperparameters./Progress bar.、Add FrozenLake enviroment/Create enviroment、Show value function/Show Q function、Trainer、A bunch of functions that will be deprecated、More for the attention chapter），译文准确、与代码前说明对应；全部围栏（info string + 内容）与底本逐字一致（verify 校验），未改动任何代码/注释/docstring。
- `d2l.md`：**通读通过，无修改**。核对：标题 `d2l` API 文档、两句说明（按字母顺序展示类与函数、标明定义位置、指向 GitHub 源码）、`## 类`/`## 函数`；两个 `eval_rst` 中的 `currentmodule` 与 32 个 `autoclass` + 19 个 `autofunction` 条目与底本逐字一致（类名/函数名/`:members:` 原样）。底本本身 `:begin_tab:`/`:end_tab:` 数量不配对（begin=3、end=2），译文保持一致（verify 基线 warning，非译文问题）。

### 参考文献（`chapter_references/`）

- `zreferences.md`：**通读通过，无修改**。文件仅 `eval_rst` 围栏与 `:bibliography:\`../d2l.bib\`` 指令；按项目约定唯一有意例外是把页面标题 “References” 译为“参考文献”，`.. only:: html` 指令、下划线长度、`:bibliography:` 路径均保持可用；verify 仅给出该有意例外提示，无结构错误。

## 遗留疑点（未擅改，提请负责人裁决）

1. **jupyter.md 底本图题笔误**：正文创建的是 “test.ipynb”，而 :numref:`fig_jupyter01` 的英文图题写作 “text.ipynb”（正文/图题不一致，底本笔误）。译文按底本保留“text.ipynb”，未自行修正；若定稿要求图题与正文一致，可统一为 test.ipynb（或加译注）。
2. **zreferences.md 缩进**：中文标题行与下划线统一缩进 4 个空格（底本为 3 个空格），同一 `.. only:: html` 指令块内缩进一致、RST 可正常渲染；verify 只报“标题已按中文译出”。若要满足“其余围栏逐字一致”的最严读法，可把缩进改回 3 空格（本次未动，因涉及 eval_rst 块）。
3. **“Markdown/markdown”大小写**：`jupyter.md` 正文统一作 “Markdown”（8 处），`contributing.md` 正文统一作 “markdown”（3 处，与底本正文小写一致；底本仅在标题 “Markdown Files in Jupyter” 处大写）。两文件各自内部一致，是否全书统一大小写请裁决（未改）。
4. **机器镜像 vs 机器映像**：`Amazon Machine Image` 译“亚马逊机器镜像”（本批其余 image 统一作“镜像”：为该实例创建镜像、已保存的镜像）。AWS 官方中文文档多用“机器映像”；是否改为“映像”并全书统一，请裁决（未改）。
5. **术语首现英文括注**：`TERMINOLOGY.md` 收录的 instance type/notebook instance/spot instance（实例类型/笔记本实例/竞价实例）在附录首次出现处均未括注英文（底本同样是裸术语，附录为操作说明）；是否按“重要术语首现括注英文”的统一规则补注，请裁决（未改）。
6. **可读但未改的直译痕迹**（按“不过度修改”保留）：①sagemaker.md 三个标签页的“用于在SageMaker 上运行的ipynb 格式的整本书可在 … 获取”前置定语较长；②colab.md“具体来说，如果需要GPU，就会自动请求Colab 连接到GPU 实例”是对底本被动句 “Colab will be automatically requested for connecting to a GPU instance” 的主动化处理，语义一致；③jupyter.md“在最后一行代码之后将各元素乘以2”忠实对应底本同样略含歧义的 “Multiply the elements by 2 after the last line of code”；④aws.md “右键点击 :numref:`fig_connect` 中的示例”中的 “the example” 按底本直译“示例”。

## 说明

- 仅修改本组负责的 1 个文件、共 2 处定向 `edit`（均在 `aws.md`）；未改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md`、`work/batches/`、其他批次/组文件及其他组报告；未改动任何代码块、行内代码、公式、指令、图片路径、URL、标题层级、表格结构。
- 审校后复核：`python work\verify.py --batch work\batches\batch-b24.txt` = **9/9 PASS（0 error、5 warning，与基线一致）**；单文件复验均 0 error。运行 `python work\align_check.py` 时同步重生成 `work/reports/align.json` 全书报告（保持其全书性质）。
