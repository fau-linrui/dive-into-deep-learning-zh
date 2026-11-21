# 译本校验工具

本目录是交付附带的自动化校验脚本（只读对比英文底本与中文译本，不修改任何书稿）。

## 前置条件

- Python 3.8+（仅用标准库）
- 英文底本放在与译本目录**同级的** `d2l-en-src/`（即 d2lbook 官方仓库，固定提交
  `23d7a5aecceee57d1292c56e90cce307f183bb0a`）
- 若底本在别处，可给任一脚本传 `--source-root <path>`；`--target-root` 默认就是本仓库。

## 用法（在本目录下运行）

```bash
python verification/verify.py --batch verification/done-files.txt   # 逐文件结构一致性（代码/公式/指令/标题/表格…）
python verification/verify.py --all                                 # 全书交叉引用、图片、文献、toc 检查
python verification/check_marks.py                                  # d2lbook 强调/删除线标记配对检查
python verification/termcheck.py                                    # 术语变体扫描
python verification/align_check.py                                  # 段落级漏译检查
```

## 交付时的校验结果

- `verify.py`：192/192 个文件通过，0 错误；156 条警告均为中文语序重排或有意保留的英文专名/示例。
- `verify.py --all`：0 错误（617 个标签、1097 处引用、610 处文献引用、191 条 toc 条目全部可解析；
  2 条警告为英文底本自身缺陷的忠实保留）。
- `check_marks.py`：全部文件标记配对平衡。
- `align_check.py`：剩余提示经人工核对均为跨行公式/英文名单等假阳性，无整段漏译。
- 各章审校记录见 `reports/review-g*.md`；机器校验报告见 `reports/verify-*-final.json`。
