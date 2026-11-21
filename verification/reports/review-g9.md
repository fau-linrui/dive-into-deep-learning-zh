# 审校组 G9 审校记录（批次 b18）

审校范围：`work/batches/batch-b18.txt`，共 10 个文件：
`chapter_natural-language-processing-pretraining/{bert, bert-dataset, bert-pretraining}`、
`chapter_natural-language-processing-applications/{index, sentiment-analysis-and-dataset, sentiment-analysis-rnn, sentiment-analysis-cnn, natural-language-inference-and-dataset, natural-language-inference-attention, finetuning-bert}`。

方法：逐文件通读中文译文，并与英文底本 `d2l-en-src/<path>` 按"英文段—中文段"逐段对照（代码块、公式、指令、图片路径、URL 只做结构一致性核验，未改动）；重点核对漏译/误译、限定词与因果条件、公式符号解释、术语一致性（掩码语言模型、下一句预测、自然语言推断、情感分析、微调、预训练、蕴含/矛盾/中性）、图题与练习。只做定向 `edit`，每次修改后 `python work\verify.py --batch work/batches/batch-b18.txt` 复核。
另用两个只读辅助脚本交叉核查：`work\_g9_nums.py`（英文段—中文段配对，比对数字漂移与残留英文段）、`work\_g9_struct.py`（比对标题层级、图片路径、`:label:`、`:eqlabel:`、有序列表项、`[**...**]` 高亮数、`*` 列表项数）。

基线：审校前 `verify.py --batch` = 10/10 PASS，0 error，9 warning（8 条"行内代码/行内公式内容一致但顺序不同"+1 条"疑似未翻译的英文行"，后者是刻意保留的 Penn Treebank II 词性标记串）。`align_check.py --batch` 基线 1 个疑点文件（bert.md，复核为误报，见下）。审校后仍为 10/10 PASS、0 error、9 warning，警告数与基线完全相同。

## 修改清单（6 个文件、15 处）

1. `chapter_natural-language-processing-pretraining/bert.md`（6 处）
   - L53—54：`GPT ... represents an effort in designing a general task-agnostic model ...` 原译"代表了一种设计／用于上下文敏感表示的通用*任务无关*模型的努力"，"一种设计……的努力"易读成病句；改为"代表了／为上下文敏感表示设计通用*任务无关*模型的一种努力"（`*任务无关*` 标记保留）。
   - L173：`Common in the Transformer encoder, positional embeddings are added at every position...` 原译"Transformer 编码器中常见的是，位置编码被加到……"，主语悬空；改为"与 Transformer 编码器中的常见做法一样，位置编码被加到 BERT 输入序列的每个位置上。"
   - L322：`(e.g., "great" is selected to be masked and predicted in "this movie is great")` 原译"在……中选"great"被掩码并预测"，主动/被动混用；改为"在……中，选择"great"将其掩码并预测"。
   - L440：`Although masked language modeling is able to encode bidirectional context for representing words` 原译"能够为表示词而双向编码上下文"语序生硬；改为"能够双向编码上下文来表示词"。
   - L525：`pretrained on the concatenation of BookCorpus and English Wikipedia` 原译"……的拼接上预训练的"；改为"……拼接而成的语料上预训练的"（`BookCorpus`、`:cite:` 原样）。
   - L596（小结）：`regardless of the context of the word (if any)` 原译"无论词（如果有的话）的上下文如何"，括注被误挂到"词"上；改为"无论词的上下文如何（如果有的话）"。
2. `chapter_natural-language-processing-pretraining/bert-dataset.md`（1 处）
   - L10：同上"两个巨型语料库 BookCorpus 和英文维基百科的拼接上预训练的"改为"……拼接而成的语料上预训练的"。
3. `chapter_natural-language-processing-pretraining/bert-pretraining.md`（4 处）
   - L4—6：`With the BERT model implemented in ... and the pretraining examples generated ... , we will pretrain ...` 原译"在……中实现了 BERT 模型，并且在……中……生成了预训练样本，本节中我们……"，两个分句无主语、与主句衔接断裂；改为"利用……中实现的 BERT 模型，以及……中从 WikiText-2 数据集生成的预训练样本，本节中我们将在 WikiText-2 数据集上预训练 BERT。"（`:numref:` 指令原样）。
   - L282：`the first three elements ... are different from those when the context is different` 原译"……的前三个元素与上下文不同时的前三个元素不同"同语反复；改为"……的前三个元素与上下文不同时有所不同。"
   - L283、L303（2 处）：`This supports that BERT representations are context-sensitive` 原译"这支持了 BERT 表示是上下文敏感的"（英语式"支持了……是"）；改为"这表明 BERT 表示是上下文敏感的"。
4. `chapter_natural-language-processing-applications/sentiment-analysis-rnn.md`（1 处）
   - L5：`Like word similarity and analogy tasks` 原译"与词相似度和类比任务一样"；前一节标题为"词的相似性和类比"（`similarity-analogy.md`），改为"与词的相似性和类比任务一样"。
5. `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md`（1 处）
   - L263：`two inputs ... representing pairs of premises and hypotheses` 原译"分别表示前提和假设对"（"对"字悬置）；改为"分别表示成对的前提和假设"（行内代码 `X[0]`/`X[1]` 原样）。
6. `chapter_natural-language-processing-applications/natural-language-inference-attention.md`（2 处）
   - L14：`Simpler than preserving the order of tokens in premises and hypotheses, we can just align ...` 原译"与保留……的顺序相比，我们可以只需把……对齐"，"相比……可以只需"逻辑不通；改为"与保留前提和假设中词元的顺序相比，更简单的做法是只需把……对齐……"。
   - L175：`we compare a token in one sequence with the other sequence that is softly aligned with that token` 原译"与另一个序列中与其软对齐的序列进行比较"，"序列中……的序列"自指不明；改为"与另一个序列中与其软对齐的表示进行比较"。

## 逐文件记录

### chapter_natural-language-processing-pretraining

- `bert.md`：6 处表达/逻辑修正（见上）。通篇核对：上下文无关/上下文敏感、任务特定/任务无关、`<cls>`/`<sep>` 输入表示、段嵌入、可学习位置编码、`BERTEncoder`/`MaskLM`/`NextSentencePred`/`BERTModel` 各段、掩码语言模型 80%/10%/10% 替换策略（"10% of 15%"处理为"有 10% 的时间（占 15% 的一部分）"）、下一句预测、BookCorpus+Wikipedia 8 亿/25 亿词、两个练习，均与底本一致；110M/340M 一类数字在 `bert-pretraining.md`。6 条行内代码/公式 warning 与基线同数，系中文语序重排，多重集一致。
- `bert-dataset.md`：1 处修改（拼接语料）。核对 WikiText-2 与 PTB 的三点对比、`_read_wiki`/`_get_next_sentence`/`_get_nsp_data_from_paragraph`/`_replace_mlm_tokens`/`_get_mlm_data_from_tokens`/`_pad_bert_inputs`/`_WikiTextDataset`/`load_data_wiki` 的逐句说明（各参数名保留代码格式）、WordPiece 词表 30000、`min_freq=5`、512/64、$10=64\times0.15$、练习（NLTK Punkt、词表大小），均一致；`&lt;pad&gt;`/`&lt;mask&gt;` 等 HTML 实体与 `__getitem__ ` 尾随空格原样保留。
- `bert-pretraining.md`：4 处修改（见上）。核对 base/large 配置（12 层/768/12 头；24 层/1024/16 头；110M/340M）、`_get_batch_loss_bert` 的 MLM/NSP 损失叙述、`train_bert` 以 `num_steps`（迭代步数）替代轮数的说明、`get_bert_encoding` 的"长度为 6""`<cls>` 索引为 0"、单句/句对两组 `crane` 实验与"上下文敏感"结论、练习，均一致。1 条行内代码 warning 系语序重排，可接受。

### chapter_natural-language-processing-applications

- `index.md`：通读通过，无修改。预训练表示→下游任务、"没有预训练"的前文回顾、情感分析/自然语言推断两任务定位、图题、`toc` 块、BERT 微调（序列级/词元级：单文本分类、文本对分类、文本标注、问答）均与底本一致。
- `sentiment-analysis-and-dataset.md`：通读通过，无修改。IMDb 大型电影评论数据集（训练/测试各 25000 条、积极/消极标签均衡——与官方中文版"积极/消极"一致）、`read_imdb`/词表/截断填充到 500/数据迭代器/`load_data_imdb` 逐句说明、图题与练习均一致。
- `sentiment-analysis-rnn.md`：1 处修改（见上）。双向 LSTM 取初始/最终时间步隐藏状态并连结、GloVe 100 维初始化且训练中不更新、`predict_sentiment` 示例句、三条练习（含 spaCy/phrase token 说明）均一致；小结"预训练词向量表示词元""BiRNN 表示文本序列"与底本一致。
- `sentiment-analysis-cnn.md`：通读通过，无修改。一维互相关（含 3 通道与"多输入通道一维互相关 ≡ 单输入通道二维互相关"）、`corr1d`/`corr1d_multi_in`、最大时间汇聚（首现括注 max-over-time pooling）、textCNN 架构三步与 11 词元/6 维/2 与 4 宽度卷积核/9 通道/2 维输出、暂退法、位置编码练习，均与底本一致；图题（含阴影部分公式）与底本逐字对应。
- `natural-language-inference-and-dataset.md`：1 处修改（见上）。自然语言推断定义与蕴含/矛盾/中性三关系（首现均括注英文）、识别文本蕴含、SNLI 500000+ 句对、`read_snli`（labels 0/1/2 对应蕴含/矛盾/中性）、`SNLIDataset` 的 `num_steps` 截断与 `&lt;pad&gt;` 追加、训练集词表用于测试集、550000/10000、128/50、练习（n 元语法匹配、词表大小），均一致。
- `natural-language-inference-attention.md`：2 处修改（见上）。可分解注意力模型（attending/comparing/aggregating = 对齐/比较/聚合）、软/硬对齐、$e_{ij}$、$m+n$ 次 vs $mn$ 次分解技巧、$\beta_i$/$\alpha_j$、$g$/$h$、`Attend`/`Compare`/`Aggregate`/`DecomposableAttention` 逐句说明、100 维 GloVe/200 维隐层、256/50、`split_batch_multi_inputs`、`predict_snli` 与"he is good./he is bad."示例、小结与练习，均一致；2 条 warning 系中文语序，多重集一致。
- `finetuning-bert.md`：通读通过，无修改。序列级/词元级总述、"最小架构改动"即额外全连接层、额外层从头学 + BERT 全参数微调、CoLA、"I should study."示例、文本对分类/回归（语义文本相似性 0—5 序数量表及 3 组样例）、文本标注与 Penn Treebank II 标记串（英文技术数据原样保留）、SQuAD v1.1 阅读理解（段落/问题/文本片段、$s_i$、$p_i$、$e_i$、$s_i+e_j$）、小结与练习（负采样+BERT、语言模型、机器翻译），均一致；样例句保留英文并加中文括注属译者既定处理。

## 遗留疑点（未擅改，提请负责人裁决/知悉）

1. **底本笔误两处，译文按底本直译（未擅改）**：
   - `bert.md` L296—298：底本"special tokens "<cls>" and "<seq>""中 `<seq>` 疑为 `<sep>` 笔误。
   - `finetuning-bert.md` L61—63：底本"the special classification token "<sep>" marks the end..."将其称作"分类词元"，按上下文应为"分隔词元"；译文照译为"特殊分类词元"。若负责人确认，可在全书统一时校正这两处。
2. **attending 小节标题的译法**：本批作"对齐（Attending）"（与正文 attending→"对齐"一致）；官方旧版作"注意（Attending）"。术语表未收录 attending，如全章需与官方提法统一，请负责人定夺（未擅改）。
3. **重要术语首现未括注英文**：本批 `情感分析`（sentiment analysis）、`掩码语言模型`（masked language model）、`下一句预测`（next sentence prediction）、`单文本分类`（single text classification）、`文本对分类`（text pair classification）、`文本标注`（text tagging）、`词性标注`（part-of-speech tagging）、`问答`（question answering）在各自首现处未加英文括注；而 `自然语言推断`、`语义文本相似性`、`前提/假设` 加了。全书对此并不统一（其他章亦有未括注者）。为避免跨组风格冲突，本次未补注，建议负责人统一规则后由专人批量处理。
4. **术语变体报告（termcheck）**：`拼接` 被列为 `连结`（concatenate）的变体，出现在 `bert.md`/`bert-dataset.md` 描述语料拼接处。此处语义为"语料库拼接"，非逐元素 concatenate 运算，故保留"拼接"（已顺带改为"拼接而成的语料上"，读起来更顺）；如需全书统一为"连结"请负责人决定。
5. **`max-over-time pooling` → "最大时间汇聚"、`hidden size` → "隐藏大小"**：均为本批新译（术语表未收录），正文首现已括注英文；建议合入 TERMINOLOGY（见下）。
6. **`ground-truth` 的处理**：`finetuning-bert.md` 练习外正文未出现；`natural-language-inference-and-dataset.md` 练习译"真实译文"（底本 ground-truth translation）。是否改为"参考译文/基准译文"属全书用词口径问题，未擅改。
7. **`align_check.py` 的 1 条疑点**：`bert.md` 段落 38（源 L325 起的 80%/10%/10% 三条替换规则）报"中文偏少：英文 52 词 -> 中文 43 字"。人工复核：三条规则、三个示例、"占 15% 的一部分"说明均完整，英文示例句以原样引号保留，中文较简明所致，属误报。
8. **`finetuning-bert.md` 语义文本相似性样例**：译者保留英文原句并加中文括注（如 `"A plane is taking off."（"一架飞机正在起飞。"）`），属既定处理，未改动；若定稿要求纯中文或纯英文，请负责人统一。

## 建议合入 TERMINOLOGY.md 的新术语（本批已用且内部一致）

- decomposable attention model → 可分解注意力模型
- max-over-time pooling → 最大时间汇聚（textCNN）
- hidden size → 隐藏大小（隐藏单元数）
- ordinal scale → 序数量表
- semantic textual similarity → 语义文本相似性
- text tagging → 文本标注；part-of-speech tagging → 词性标注
- single text classification → 单文本分类；text pair classification / text pair regression → 文本对分类 / 文本对回归
- entailment → 蕴含；contradiction → 矛盾；neutrality → 中性
- segment embedding → 段嵌入
- state of the art → 最优水平（本批统一用法）

## 说明

- 仅修改本组负责的 6 个文件、15 处定向替换；未修改 `d2l-en-src/`、`ref/`、`TERMINOLOGY.md` 及其他批次/组文件；未改动任何代码块、行内代码、公式、`:cite:`/`:numref:`/`:label:` 等指令、图片路径、URL、标题层级、表格结构、`[**...**]` 高亮标记（改动后的高亮与斜体标记计数经专项脚本比对，10 个文件全部与底本一致）。
- 审校后复核：`python work\verify.py --batch work/batches/batch-b18.txt` = 10/10 PASS（0 error、9 warning，与基线同数）；`python work\align_check.py --batch work/batches/batch-b18.txt` = 10 个文件、1 个段级疑点（已复核为误报，见遗留疑点 7）。
