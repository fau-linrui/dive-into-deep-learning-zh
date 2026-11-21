#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole-book terminology / consistency scanner for the Chinese translation.

Only scans *prose* (fenced code blocks are stripped). Reports occurrences of
non-standard variants that should be unified to the term in TERMINOLOGY.md.

Usage:
    python work\\termcheck.py
    python work\\termcheck.py --fix    # apply SAFE_FIXES (unambiguous typos/variants)
"""
import argparse
import os
import re
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PACKAGE_ROOT)
TGT = PACKAGE_ROOT
EXCLUDE_DIRS = {".git", "verification", "contrib", "static", "img", "d2l", "graffle", "ci", ".github"}
ROOT_EXCLUDE = {"README.md", "STYLE_GUIDE.md", "INFO.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md",
                "TERMINOLOGY.md", "TRANSLATION_REPORT.md"}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# (推荐写法, [不规范/次选变体...])  — 只列真正需要统一的异体，不含合法的近义词
VARIANTS = [
    ("暂退法", ["丢弃法", "随机失活", "随机丢弃"]),
    ("汇聚", ["池化"]),
    ("连结", ["拼接"]),
    ("词元", ["词元标记"]),
    ("词元化", ["标记化"]),
    ("规范化", ["归一化"]),
    ("掩码", ["掩蔽", "掩模", "遮掩"]),
    ("层规范化", ["层归一化"]),
    ("注意力汇聚", ["注意力池化"]),
    ("学习速率", ["学习速率"]),
    ("束搜索", ["集束搜索"]),
    ("隐藏状态", ["隐状态"]),
    ("阈值", ["阀值"]),
    ("笔记本", ["记事本"]),
    ("批量大小", ["批大小"]),
    ("查准率", ["精确率", "精准率"]),
    ("查全率", ["召回率"]),
    ("词表", ["词汇表"]),
    ("马尔可夫", ["马尔科夫"]),
    ("反向传播", ["后向传播", "反传"]),
    ("通过时间反向传播", ["按时间反向传播", "基于时间的反向传播", "时间反向传播"]),
    ("过拟合", ["过度拟合"]),
    ("欠拟合", ["拟合不足"]),
    ("泛化误差", ["一般化误差"]),
    ("协变量偏移", ["协变量漂移"]),
    ("概念偏移", ["概念漂移"]),
    ("标签偏移", ["标签漂移"]),
    ("分布偏移", ["分布漂移"]),
    ("梯度消失", ["梯度弥散"]),
    ("梯度爆炸", ["梯度激增"]),
    ("注意力权重", ["注意权重"]),
    ("自注意力", ["自我注意力"]),
    ("多头注意力", ["多头部注意力"]),
    ("位置编码", ["位置嵌入"]),
    ("独热编码", ["独热码", "一位有效编码"]),
    ("约束优化", ["带约束优化"]),  # 均可，报告提示
    ("均值", ["平均数"]),
    ("方差", ["变异数"]),
    ("导数", ["微商"]),
    ("积分", ["积分"]),
    ("矩阵", ["矩阵"]),
]

# 明确可安全替换的异体/错字（不会被误伤）
SAFE_FIXES = {
    "丢弃法": "暂退法",
    "池化层": "汇聚层",
    "池化": "汇聚",
    "批量归一化": "批量规范化",
    "批归一化": "批量规范化",
    "层归一化": "层规范化",
    "注意力池化": "注意力汇聚",
    "学习速率": "学习率",
    "集束搜索": "束搜索",
    "隐状态": "隐藏状态",
    "阀值": "阈值",
    "记事本": "笔记本",
    "批大小": "批量大小",
    "精确率": "查准率",
    "召回率": "查全率",
    "词汇表": "词表",
    "马尔科夫": "马尔可夫",
    "后向传播": "反向传播",
    "过度拟合": "过拟合",
    "拟合不足": "欠拟合",
    "梯度弥散": "梯度消失",
    "梯度激增": "梯度爆炸",
}


def book_files():
    out = []
    root = TGT
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if parts and parts[0] in EXCLUDE_DIRS:
            continue
        for name in filenames:
            if name.endswith(".md") and not (not parts and name in ROOT_EXCLUDE):
                out.append("/".join(parts + [name]))
    return sorted(out)


def strip_fences(text):
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(`{3,})(.*)$", lines[i])
        if m:
            fence = m.group(1)
            j = i + 1
            while j < len(lines) and lines[j].strip() != fence:
                j += 1
            out.extend([""] * (j - i + 1))
            i = j + 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    files = book_files()
    content = {}
    for rel in files:
        p = os.path.join(TGT, rel)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                content[rel] = f.read()

    by_pair = {}
    for good, bads in VARIANTS:
        for bad in bads:
            if bad == good or bad in good:
                continue
            for rel, t in content.items():
                prose = strip_fences(t)
                n = prose.count(bad)
                if n:
                    by_pair.setdefault((bad, good), []).append((rel, n))

    if not by_pair:
        print("未发现术语变体使用")
    else:
        print(f"共发现 {len(by_pair)} 组术语变体：")
        for (bad, good), uses in sorted(by_pair.items(), key=lambda kv: -sum(u[1] for u in kv[1])):
            total = sum(u[1] for u in uses)
            print(f"\n  [{bad}] -> [{good}]  出现 {total} 次，涉及 {len(uses)} 个文件")
            for rel, n in sorted(uses, key=lambda x: -x[1])[:5]:
                # show one context snippet
                p = os.path.join(TGT, rel)
                t = content[rel]
                prose = strip_fences(t)
                idx = prose.find(bad)
                snippet = prose[max(0, idx - 30):idx + len(bad) + 30].replace("\n", " ")
                print(f"      {n:>3}x {rel}   例: …{snippet}…")

    # headings without Chinese
    print("\n== 不含中文的标题（专名标题可忽略） ==")
    any_h = False
    for rel, t in content.items():
        for ln, line in enumerate(strip_fences(t).split("\n"), 1):
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m and not CJK_RE.search(m.group(2)) and len(re.findall(r"[A-Za-z]{2,}", m.group(2))) >= 2:
                print(f"  {rel}:{ln}: {line.strip()}")
                any_h = True
    if not any_h:
        print("  无")

    # leftover English paragraphs (strong signal of omissions)
    print("\n== 整段无中文的正文段落（≥10 个英文单词，专名/URL 段可忽略） ==")
    any_p = False
    for rel, t in content.items():
        prose = strip_fences(t)
        prose = re.sub(r":\w+:`[^`]*`", " ", prose)
        prose = re.sub(r"`[^`\n]+`", " ", prose)
        prose = re.sub(r"\$[^$\n]*\$", " ", prose)
        prose = re.sub(r"https?://\S+", " ", prose)
        prose = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", prose)
        for ln, line in enumerate(prose.split("\n"), 1):
            if CJK_RE.search(line) or not line.strip():
                continue
            words = re.findall(r"[A-Za-z][A-Za-z'\-]+", line)
            if len(words) >= 10:
                print(f"  {rel}:{ln}: {line.strip()[:110]}")
                any_p = True
    if not any_p:
        print("  无")

    if args.fix:
        changed = 0
        detail = {}
        for rel, t in content.items():
            nt = t
            for bad, good in SAFE_FIXES.items():
                # do not replace inside fenced code: process by chunks
                parts = nt.split("\n")
                out, i = [], 0
                while i < len(parts):
                    m = re.match(r"^(`{3,})(.*)$", parts[i])
                    if m:
                        fence = m.group(1)
                        j = i + 1
                        while j < len(parts) and parts[j].strip() != fence:
                            j += 1
                        out.extend(parts[i:j + 1])
                        i = j + 1
                    else:
                        if bad in parts[i]:
                            detail[bad] = detail.get(bad, 0) + parts[i].count(bad)
                            parts[i] = parts[i].replace(bad, good)
                        out.append(parts[i])
                        i += 1
                nt = "\n".join(out)
            if nt != t:
                p = os.path.join(TGT, rel)
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    f.write(nt)
                changed += 1
        print(f"\n已对 {changed} 个文件执行安全替换: {detail}")


if __name__ == "__main__":
    main()
