#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paragraph-level completeness checker.

For every translated file, split English and Chinese sources into structural
units (fenced code block / prose paragraph / heading / table), align them, and
flag prose paragraphs whose Chinese text is suspiciously short or long relative
to the English source. This catches omissions and accidental duplication that
the structural verifier (which only checks verbatim items) cannot see.

Usage:
    python work\\align_check.py                 # all translated files
    python work\\align_check.py --batch work/batches/batch-b02.txt
"""
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PACKAGE_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify  # noqa: E402

TGT = PACKAGE_ROOT
SRC = os.path.join(WS, "d2l-en-src")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def units(text):
    """Return list of (kind, payload, line_no) structural units."""
    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^(`{3,})(.*)$", line)
        if m:
            fence = m.group(1)
            j = i + 1
            body = []
            while j < n and lines[j].strip() != fence:
                body.append(lines[j])
                j += 1
            out.append(("fence", m.group(2).strip(), i + 1))
            i = j + 1
            continue
        if not line.strip():
            i += 1
            continue
        # accumulate a prose block until blank line or fence
        block = []
        start = i
        while i < n and lines[i].strip() and not re.match(r"^(`{3,})(.*)$", lines[i]):
            block.append(lines[i])
            i += 1
        txt = "\n".join(block)
        kind = "heading" if re.match(r"^#{1,6}\s", block[0]) else "para"
        out.append((kind, txt, start + 1))
    return out


def clean(txt):
    """Strip verbatim/structural fragments for prose counting."""
    p = txt
    p = verify.MATH_DISPLAY_RE.sub(" ", p)
    p = verify.DIRECTIVE_RE.sub(" ", p)
    p = verify.INLINE_CODE_RE.sub(" ", p)
    p = verify.MATH_INLINE_RE.sub(" ", p)
    p = verify.IMAGE_RE.sub(" ", p)
    p = verify.LINK_RE.sub(lambda m: " " + m.group(1) + " ", p)
    p = re.sub(r"https?://\S+", " ", p)
    p = re.sub(r"<[^>]+>", " ", p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=os.path.join(WS, "work", "reports", "align.json"))
    args = ap.parse_args()

    if args.batch:
        files = verify.load_batch(args.batch)
    else:
        files = [rel for rel in verify.in_scope_files(SRC)
                 if os.path.exists(os.path.join(TGT, rel))
                 and verify.read_text(os.path.join(SRC, rel)) != verify.read_text(os.path.join(TGT, rel))]

    report = []
    flagged = 0
    for rel in files:
        sp = os.path.join(SRC, rel)
        tp = os.path.join(TGT, rel)
        if not os.path.exists(tp):
            continue
        su = units(verify.read_text(sp))
        tu = units(verify.read_text(tp))
        issues = []
        if len(su) != len(tu):
            # align by kind sequence to find structural mismatch
            ks = "".join(k[0] for k in su)
            kt = "".join(k[0] for k in tu)
            sm = SequenceMatcher(None, ks, kt, autojunk=False)
            missing = sum(1 for tag, a1, a2, b1, b2 in sm.get_opcodes() if tag in ("delete", "replace"))
            issues.append(f"结构单元数不同 (源 {len(su)} / 译 {len(tu)}，结构差异段 {missing})")
            # pairwise fallback: compare by index where possible
            pairs = list(zip(su, tu))
        else:
            pairs = list(zip(su, tu))
        for idx, ((sk, spay, sln), (tk, tpay, tln)) in enumerate(pairs):
            if sk != tk:
                issues.append(f"单元 {idx + 1} 类型不同 源[{sk}]@L{sln} vs 译[{tk}]@L{tln}")
                continue
            if sk != "para":
                continue
            sc = clean(spay)
            tc = clean(tpay)
            sw = len(WORD.findall(sc))
            sc_nc = len(sc.replace(" ", ""))
            tc_cjk = len(CJK.findall(tc))
            if sw >= 12:
                # expected ~1.5 CJK per English word; flag suspiciously short/long
                if tc_cjk < max(6, 0.7 * sw):
                    issues.append(
                        f"段落 {idx + 1}@源L{sln} 疑似漏译/过短: 英文 {sw} 词 -> 中文 {tc_cjk} 字 | "
                        f"EN: {spay.strip()[:100]!r} | ZH: {tpay.strip()[:60]!r}")
                elif tc_cjk > 4.2 * sw:
                    issues.append(
                        f"段落 {idx + 1}@译L{tln} 疑似增译/过长: 英文 {sw} 词 -> 中文 {tc_cjk} 字")
                elif sw >= 25 and tc_cjk < 0.95 * sw:
                    issues.append(
                        f"段落 {idx + 1}@源L{sln} 中文偏少: 英文 {sw} 词 -> 中文 {tc_cjk} 字")
            if tc_cjk == 0 and sw >= 6:
                issues.append(f"段落 {idx + 1}@译L{tln} 无中文（英文 {sw} 词）: {tpay.strip()[:80]!r}")
        if issues:
            flagged += 1
            report.append({"file": rel, "issues": issues})

    print(f"检查文件 {len(files)} 个，发现段级疑点文件 {flagged} 个")
    for r in report:
        print(f"\n[{r['file']}]")
        for it in r["issues"][:12]:
            print("  -", it)
        if len(r["issues"]) > 12:
            print(f"  ... 另有 {len(r['issues']) - 12} 条")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n报告: {args.json}")


if __name__ == "__main__":
    main()
