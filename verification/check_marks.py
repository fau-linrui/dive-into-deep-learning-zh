#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check d2lbook slide/highlight marker pairs outside fenced code.

d2lbook's build asserts these pairs are balanced in every markdown cell:
[** ... **] , (** ... **) , [~~ ... ~~] , (~~ ... ~~)
The Chinese translation must keep both counts equal (markers/enclosing ASCII
chars are markup, not prose punctuation).

Usage: python work\\check_marks.py [--list]
"""
import argparse
import os
import re

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PACKAGE_ROOT)
SRC = os.path.join(WS, "d2l-en-src")
TGT = PACKAGE_ROOT
EXCLUDE_DIRS = {".git", "verification", "contrib", "static", "img", "d2l", "graffle", "ci", ".github"}
ROOT_EXCLUDE = {"README.md", "STYLE_GUIDE.md", "INFO.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md",
                "TERMINOLOGY.md", "TRANSLATION_REPORT.md"}
PAIRS = [("[**", "**]"), ("(**", "**)"), ("[~~", "~~]"), ("(~~", "~~)")]


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


def counts(text):
    prose = strip_fences(text)
    return {pair: (prose.count(pair[0]), prose.count(pair[1])) for pair in PAIRS}


def book_files(root, exclude_root=()):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if parts and parts[0] in EXCLUDE_DIRS:
            continue
        for name in filenames:
            if name.endswith(".md") and not (not parts and name in exclude_root):
                out.append("/".join(parts + [name]))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    files = book_files(SRC, ROOT_EXCLUDE)
    bad = []
    for rel in files:
        sp = os.path.join(SRC, rel)
        tp = os.path.join(TGT, rel)
        if not os.path.exists(tp):
            continue
        s = counts(open(sp, encoding="utf-8").read())
        t = counts(open(tp, encoding="utf-8").read())
        issues = []
        for pair in PAIRS:
            if s[pair] != t[pair]:
                issues.append((pair, s[pair], t[pair]))
        # also require each target file itself balanced (for files with slides)
        for pair in PAIRS:
            if t[pair][0] != t[pair][1]:
                issues.append((pair, "self", t[pair]))
        if issues:
            bad.append((rel, issues))
    print(f"检查 {len(files)} 个文件，发现问题文件 {len(bad)} 个")
    for rel, issues in bad:
        print(f"\n[{rel}]")
        for pair, a, b in issues:
            print(f"   {pair[0]}...{pair[1]}: 源 {a} / 译 {b}")
    if args.list:
        for rel, _ in bad:
            print(rel)


if __name__ == "__main__":
    main()
