#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural integrity verifier for the D2L Chinese translation.

Compares each translated file against its English source and checks that
everything that must be preserved verbatim really is:

  * fenced code blocks (info string + body, byte for byte)
  * display math ($$ ... $$) and inline math ($ ... $), byte for byte
  * inline code spans (byte for byte)
  * directives: :label: :eqlabel: :numref: :eqref: :cite: :ref: :width:
  * heading levels, tab directives, table shape, list shape, image paths, link URLs
  * translation coverage (CJK ratio, untranslated lines/paragraphs)

With --all it also resolves cross references, image paths and toc entries across
the whole book.

Usage (run from the workspace root):
  python work\\verify.py --file chapter_preliminaries/ndarray.md
  python work\\verify.py --batch work/batches/batch-b02.txt
  python work\\verify.py --all
  python work\\verify.py --file chapter_preliminaries/ndarray.md --repair
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PACKAGE_ROOT)
DEF_SRC = os.path.join(WS, "d2l-en-src")
DEF_TGT = PACKAGE_ROOT

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
FENCE_RE = re.compile(r"^(`{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)")
DIRECTIVE_RE = re.compile(r":(label|eqlabel|numref|eqref|citet|cite|ref|width):`([^`]*)`")
TAB_BEGIN_RE = re.compile(r":begin_tab:`([^`]*)`")
TAB_END_RE = re.compile(r":end_tab:")
MATH_DISPLAY_RE = re.compile(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", re.S)
MATH_INLINE_RE = re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")

MAX_DETAILS = 6


# ----------------------------------------------------------------------------- io
def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("utf-8", errors="replace")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------------- md parsing
def parse_items(text):
    """Split markdown into ('text', line) and ('fence', info, body) items."""
    lines = text.split("\n")
    items = []
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m:
            fence, info = m.group(1), m.group(2).strip()
            j = i + 1
            body = []
            closed = False
            while j < len(lines):
                if lines[j].strip() == fence:
                    closed = True
                    break
                body.append(lines[j])
                j += 1
            items.append(("fence", info, "\n".join(body), closed, fence))
            i = j + 1 if closed else len(lines)
        else:
            items.append(("text", lines[i]))
            i += 1
    return items


def items_to_text(items):
    out = []
    for it in items:
        if it[0] == "text":
            out.append(it[1])
        else:
            _, info, body, closed, fence = it
            out.append(fence + info)
            if body:
                out.append(body)
            if closed:
                out.append(fence)
    return "\n".join(out)


def md_text_of(items):
    return "\n".join(it[1] for it in items if it[0] == "text")


class Extraction:
    """All comparable features extracted from one markdown file."""

    def __init__(self, text):
        self.text = text
        self.items = parse_items(text)
        self.fences = [(it[1], it[2]) for it in self.items if it[0] == "fence"]
        self.unclosed_fences = sum(1 for it in self.items if it[0] == "fence" and not it[3])
        md = md_text_of(self.items)
        self.md_raw = md
        self.md = md

        # --- pull out verbatim spans so prose checks do not see them
        self.inline_code = [m.group(1) for m in INLINE_CODE_RE.finditer(md)]
        self.display_math = [m.group(1) for m in MATH_DISPLAY_RE.finditer(md)]

        # --- headings
        self.headings = []
        self.paragraphs = []  # non-empty blocks in markdown text
        cur = []
        for line in md.split("\n"):
            hm = HEADING_RE.match(line)
            if hm:
                self.headings.append(len(hm.group(1)))
            if line.strip():
                cur.append(line)
            else:
                if cur:
                    self.paragraphs.append("\n".join(cur))
                    cur = []
        if cur:
            self.paragraphs.append("\n".join(cur))

        # --- directives (ordered)
        self.directives = [(m.group(1), m.group(2)) for m in DIRECTIVE_RE.finditer(md)]
        self.tab_begin = [m.group(1) for m in TAB_BEGIN_RE.finditer(md)]
        self.tab_end_count = len(TAB_END_RE.findall(md))

        # --- images / links (paths only)
        self.image_alts = [m.group(1) for m in IMAGE_RE.finditer(md)]
        self.image_paths = [m.group(2) for m in IMAGE_RE.finditer(md)]
        self.link_urls = [m.group(2) for m in LINK_RE.finditer(md)]
        self.link_texts = [m.group(1) for m in LINK_RE.finditer(md)]

        # --- tables and lists
        self.tables = []
        blocks = []
        cur_t = []
        for line in md.split("\n"):
            s = line.strip()
            if s.startswith("|"):
                cur_t.append(s)
            else:
                if cur_t:
                    blocks.append(cur_t)
                    cur_t = []
        if cur_t:
            blocks.append(cur_t)
        self.tables = blocks
        self.table_shape = [[row.count("|") for row in b] for b in blocks]
        self.list_shape = []
        for line in md.split("\n"):
            lm = LIST_RE.match(line)
            if lm:
                typ = "num" if lm.group(2)[0].isdigit() else "bullet"
                self.list_shape.append((len(lm.group(1)), typ))

        # --- prose used for coverage checks: strip code/math/directives/urls
        p = md
        p = MATH_DISPLAY_RE.sub(" ", p)
        p = DIRECTIVE_RE.sub(" ", p)   # must come before inline-code removal (directives end with backticks)
        p = INLINE_CODE_RE.sub(" ", p)
        p = MATH_INLINE_RE.sub(" ", p)
        p = IMAGE_RE.sub(" ", p)
        p = LINK_RE.sub(lambda m: " " + m.group(1) + " ", p)  # keep visible text only
        p = re.sub(r"https?://\S+", " ", p)
        p = re.sub(r"<[^>]+>", " ", p)
        self.prose = p
        self.prose_lines = []
        for line in p.split("\n"):
            s = line.strip()
            if s:
                self.prose_lines.append(s)
        self.cjk_chars = len(CJK_RE.findall(p))
        self.alpha_words = len(WORD_RE.findall(p))

    def summary_counts(self):
        return {
            "fences": len(self.fences),
            "display_math": len(self.display_math),
            "inline_math": sum(1 for _ in MATH_INLINE_RE.finditer(self.md)),
            "inline_code": len(self.inline_code),
            "directives": len(self.directives),
            "images": len(self.image_paths),
            "tables": len(self.tables),
            "headings": len(self.headings),
        }


MATH_INLINE_RE2 = MATH_INLINE_RE


MATH_INLINE_NL_RE = re.compile(r"(?<!\\)\$(?!\$)([^$]+?)(?<!\\)\$", re.S)


def extract_inline_math(md):
    """Inline math, computed per paragraph with code/display math blocked.

    * escaped dollars (``\\$``) are masked first and restored afterwards, so
      currency inside math (e.g. ``$\\$10 \\pm \\$1$``) does not create phantom
      spans;
    * the scan works at paragraph level and lets a math span cross line breaks
      (some English sources break ``$...$`` across lines, which renders as a
      space); newlines are normalized to single spaces so the comparison is
      insensitive to where the translator wrapped lines.
    """
    tmp = INLINE_CODE_RE.sub(lambda m: "\x00" * len(m.group(0)), md)
    tmp = MATH_DISPLAY_RE.sub(lambda m: "\x00" * len(m.group(0)), tmp)
    tmp = tmp.replace("\\$", "\x02\x02")
    out = []
    for para in re.split(r"\n\s*\n", tmp):
        text = re.sub(r"\s*\n\s*", " ", para)
        for m in MATH_INLINE_NL_RE.finditer(text):
            out.append(m.group(1).replace("\x02\x02", "\\$"))
    return out


# ---------------------------------------------------------------- comparisons
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def clip(s, n=160):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


class FileResult:
    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []
        self.stats = {}

    @property
    def ok(self):
        return not self.errors

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def compare_seq(res, label, src, tgt, describe=lambda x: clip(str(x))):
    if src == tgt:
        return False
    i = first_diff(src, tgt)
    res.err(
        f"{label}: 不一致 (源 {len(src)} 项 / 译 {len(tgt)} 项)"
        + (f"，首个差异在第 {i + 1} 项" if i is not None else "")
        + (f"; 源: {describe(src[i])}" if i is not None and i < len(src) else "")
        + (f"; 译: {describe(tgt[i])}" if i is not None and i < len(tgt) else "")
    )
    return True


def compare_multiset(res, label, src, tgt, describe=lambda x: clip(str(x))):
    """Content must match exactly as a multiset; ordering may legitimately change."""
    cs, ct = Counter(src), Counter(tgt)
    if cs == ct:
        if src != tgt:
            res.warn(f"{label}: 项目内容一致但顺序与原文不同（若因语句重排造成可接受，请人工确认）")
        return False
    only_src = list((cs - ct).items())[:3]
    only_tgt = list((ct - cs).items())[:3]
    res.err(
        f"{label}: 内容不一致 (源 {len(src)} 项 / 译 {len(tgt)} 项)"
        + (f"; 仅源有: {[(describe(k), v) for k, v in only_src]}" if only_src else "")
        + (f"; 仅译有: {[(describe(k), v) for k, v in only_tgt]}" if only_tgt else "")
    )
    return True


def rel_of(src_path, tgt_path):
    """Book-relative path of the translation file (best effort, for exceptions)."""
    p = tgt_path.replace("\\", "/")
    marker = "chapter_references/zreferences.md"
    return marker if p.endswith(marker) else p


def compare_file(src_path, tgt_path):
    res = FileResult(os.path.relpath(tgt_path, WS).replace("\\", "/"))
    if not os.path.exists(tgt_path):
        res.err("译文文件不存在")
        return res
    src_text = read_text(src_path)
    tgt_text = read_text(tgt_path)
    if not tgt_text.strip():
        res.err("译文文件为空")
        return res
    if tgt_text.startswith("\ufeff"):
        res.warn("译文含 BOM，建议去掉")
    if not tgt_text.endswith("\n"):
        res.warn("译文文件末尾缺少换行")

    s = Extraction(src_text)
    t = Extraction(tgt_text)
    res.stats = {
        "src": s.summary_counts(),
        "tgt": t.summary_counts(),
        "src_cjk_ratio": round(s.cjk_chars / max(1, s.alpha_words), 2),
        "tgt_cjk_ratio": round(t.cjk_chars / max(1, t.alpha_words), 2),
    }

    # 1. code fences verbatim
    if s.unclosed_fences:
        res.err(f"英文源文件有 {s.unclosed_fences} 个未闭合代码围栏（源本身异常）")
    if t.unclosed_fences:
        res.err(f"译文有 {t.unclosed_fences} 个未闭合代码围栏")
    # 唯一例外：参考文献页的 eval_rst 块内含渲染用页面标题（"References"→"参考文献"），
    # 这是构建标记而非代码，允许该文件代码块内容不同（仍然校验围栏数量与信息串）。
    if rel_of(src_path, tgt_path) == "chapter_references/zreferences.md":
        if [f[0] for f in s.fences] != [f[0] for f in t.fences] or len(s.fences) != len(t.fences):
            res.err("代码块信息串/数量与原文不一致")
        elif s.fences != t.fences:
            res.warn("参考文献页 eval_rst 中的页面标题已按中文译出（有意例外，其余围栏逐字一致）")
    else:
        compare_seq(res, "代码块(info+内容)", s.fences, t.fences,
                    describe=lambda x: clip(x[0]) + " | " + clip(x[1]))

    # 2. display math verbatim
    compare_seq(res, "独立公式 $$..$$", s.display_math, t.display_math)

    # 3. inline math (content must be identical; order may change when sentences are reordered)
    compare_multiset(res, "行内公式 $..$", extract_inline_math(s.md), extract_inline_math(t.md))

    # 4. inline code (content must be identical; order may change when sentences are reordered)
    compare_multiset(res, "行内代码", s.inline_code, t.inline_code)

    # 5. directives: labels/eqlabels/width are positional (strict);
    #    numref/eqref/ref/cite may be reordered inside a rephrased sentence.
    struct = lambda ds: [(k, v) for (k, v) in ds if k in ("label", "eqlabel", "width")]
    refs = lambda ds: [(k, v) for (k, v) in ds if k not in ("label", "eqlabel", "width")]
    compare_seq(res, "标签/尺寸指令(label/eqlabel/width)", struct(s.directives), struct(t.directives),
                describe=lambda x: f":{x[0]}:`{clip(x[1], 60)}`")
    compare_multiset(res, "引用指令(numref/eqref/ref/cite)", refs(s.directives), refs(t.directives),
                     describe=lambda x: f":{x[0]}:`{clip(x[1], 60)}`")

    # 6. headings levels
    compare_seq(res, "标题层级序列", s.headings, t.headings)

    # 7. tab directives
    compare_seq(res, "框架标签块 :begin_tab:", s.tab_begin, t.tab_begin)
    compare_seq(res, "框架标签块 :end_tab: 数量", [s.tab_end_count], [t.tab_end_count])
    if len(s.tab_begin) != s.tab_end_count:
        res.warn(f"英文源本身标签块不配对 (begin={len(s.tab_begin)}, end={s.tab_end_count})，译文保持一致即可")
    if len(t.tab_begin) != t.tab_end_count and len(s.tab_begin) == s.tab_end_count:
        res.err(f"译文标签块不配对 (begin={len(t.tab_begin)}, end={t.tab_end_count})")

    # 8. images paths / links urls
    compare_seq(res, "图片路径", s.image_paths, t.image_paths, describe=lambda x: x)
    compare_seq(res, "链接 URL", s.link_urls, t.link_urls, describe=lambda x: x)

    # 9. tables / lists
    compare_seq(res, "表格形状(每行竖线数)", s.table_shape, t.table_shape,
                describe=lambda x: clip(str(x), 120))
    if s.list_shape != t.list_shape:
        if Counter(x[1] for x in s.list_shape) != Counter(x[1] for x in t.list_shape):
            res.err(f"列表标记数量不一致 (源 {len(s.list_shape)} / 译 {len(t.list_shape)})")
        else:
            res.warn("列表缩进层级与原文略有差异（请人工确认嵌套未变）")

    # 10. paragraph structure (soft check)
    if len(s.paragraphs) and abs(len(s.paragraphs) - len(t.paragraphs)) > max(2, len(s.paragraphs) // 10):
        res.warn(f"段落块数差异较大 (源 {len(s.paragraphs)} / 译 {len(t.paragraphs)})")

    # 11. coverage: untranslated lines
    bad_lines = []
    for idx, line in enumerate(t.prose_lines, 1):
        if CJK_RE.search(line):
            continue
        words = WORD_RE.findall(line)
        if len(words) >= 4:
            bad_lines.append((idx, clip(line, 100)))
    if bad_lines:
        res.warn(f"疑似未翻译的英文行 {len(bad_lines)} 处（若为专名/引用请忽略）; 例: {bad_lines[:3]}")

    # 12. coverage: CJK ratio (Chinese chars per English prose word; normal ~1.4-2.6)
    if s.alpha_words >= 80:
        ratio = t.cjk_chars / s.alpha_words
        if ratio < 1.0:
            res.warn(f"中文/英文词数比偏低 ({ratio:.2f}，正常约 1.4-2.6)，可能有漏译")
        elif ratio > 3.2:
            res.warn(f"中文/英文词数比偏高 ({ratio:.2f})，可能有增译")
    src_len = len(src_text.strip())
    tgt_len = len(tgt_text.strip())
    if src_len > 2000:
        lr = tgt_len / src_len
        if lr < 0.30 or lr > 1.05:
            res.warn(f"译文字符数/英文字符数={lr:.2f}，偏离常见区间 0.35-0.95，请检查是否漏译或重复")

    # 13. alt text / link text coverage
    for alt, src_alt in zip(t.image_alts, s.image_alts):
        if len(WORD_RE.findall(src_alt)) >= 4 and alt and not CJK_RE.search(alt):
            res.warn(f"图片替代文本疑似未翻译: {clip(alt, 80)}")
    for txt in t.link_texts:
        if len(WORD_RE.findall(txt)) >= 4 and not CJK_RE.search(txt):
            res.warn(f"链接可见文字疑似未翻译: {clip(txt, 80)}")

    return res


# --------------------------------------------------------------------- repair
def fence_spans(text):
    """Character spans of fenced code blocks in text (includes both fences)."""
    lines = text.split("\n")
    starts = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    spans = []
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence = m.group(1)
        j = i + 1
        while j < len(lines) and lines[j].strip() != fence:
            j += 1
        start = starts[i]
        if j < len(lines):
            end = starts[j] + len(lines[j])
        else:
            end = len(text)
        spans.append((start, end))
        i = j + 1
    return spans


def masked(text, spans):
    chars = list(text)
    for start, end in spans:
        for k in range(start, min(end, len(chars))):
            chars[k] = "\x01"
    return "".join(chars)


def replace_spans(text, replacements):
    """replacements: list of (start, end, new_text), non-overlapping, any order."""
    replacements = sorted(replacements, key=lambda r: r[0])
    out = []
    pos = 0
    for start, end, new in replacements:
        if start < pos:
            continue
        out.append(text[pos:start])
        out.append(new)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def repair_file(src_path, tgt_path):
    """Restore code fences, display math and directives verbatim from the source."""
    src_text = read_text(src_path)
    tgt_text = read_text(tgt_path)
    fixed = [0, 0, 0]

    # 1. fenced code blocks
    sf = fence_spans(src_text)
    tf = fence_spans(tgt_text)
    if len(sf) == len(tf):
        reps = []
        for (ss, se), (ts, te) in zip(sf, tf):
            src_block = src_text[ss:se]
            tgt_block = tgt_text[ts:te]
            if src_block != tgt_block:
                reps.append((ts, te, src_block))
                fixed[0] += 1
        if reps:
            tgt_text = replace_spans(tgt_text, reps)

    # 2. display math outside fences
    tf = fence_spans(tgt_text)
    sf = fence_spans(src_text)
    tgt_mask = masked(tgt_text, tf)
    src_mask = masked(src_text, sf)
    tgt_math = list(MATH_DISPLAY_RE.finditer(tgt_mask))
    src_math = [m.group(0) for m in MATH_DISPLAY_RE.finditer(src_mask)]
    if len(tgt_math) == len(src_math):
        reps = []
        for m, src_block in zip(tgt_math, src_math):
            if tgt_text[m.start():m.end()] != src_block:
                reps.append((m.start(), m.end(), src_block))
                fixed[1] += 1
        if reps:
            tgt_text = replace_spans(tgt_text, reps)

    # 3. directives outside fences
    tf = fence_spans(tgt_text)
    sf = fence_spans(src_text)
    src_dirs = [m.group(0) for m in DIRECTIVE_RE.finditer(masked(src_text, sf))]
    tgt_dirs = list(DIRECTIVE_RE.finditer(masked(tgt_text, tf)))
    if len(src_dirs) == len(tgt_dirs):
        reps = []
        for m, src_block in zip(tgt_dirs, src_dirs):
            if tgt_text[m.start():m.end()] != src_block:
                reps.append((m.start(), m.end(), src_block))
                fixed[2] += 1
        if reps:
            tgt_text = replace_spans(tgt_text, reps)

    if any(fixed):
        write_text(tgt_path, tgt_text)
    return fixed


# ----------------------------------------------------------------- global checks
IN_SCOPE_EXCLUDE_DIRS = {".git", "verification", "contrib", "static", "img", "d2l", "graffle", "ci", ".github", "ref", "work"}
ROOT_EXCLUDE = {"README.md", "STYLE_GUIDE.md", "INFO.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md"}


def in_scope_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        dirnames[:] = [d for d in dirnames if d not in IN_SCOPE_EXCLUDE_DIRS]
        if parts and parts[0] in IN_SCOPE_EXCLUDE_DIRS:
            continue
        for name in filenames:
            if not name.endswith(".md"):
                continue
            if not parts and name in ROOT_EXCLUDE:
                continue
            out.append("/".join(parts + [name]))
    return sorted(out)


def toc_entries(text):
    entries = []
    for m in re.finditer(r"(?ms)^```toc\s*$(.*?)^```\s*$", text):
        for line in m.group(1).split("\n"):
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            entries.append(line)
    return entries


def global_checks(src_root, tgt_root):
    errs, warns = [], []
    src_files = in_scope_files(src_root)
    missing = [p for p in src_files if not os.path.exists(os.path.join(tgt_root, p))]
    if missing:
        errs.append(f"缺失译文文件 {len(missing)} 个: {missing[:8]}")

    labels = {}
    refs = []
    cites = []
    toc_entries_all = []
    tabs_begin = tabs_end = 0
    src_tabs_begin = src_tabs_end = 0
    for rel in src_files:
        sp = os.path.join(src_root, rel)
        if os.path.exists(sp):
            stext = read_text(sp)
            src_tabs_begin += len(re.findall(r":begin_tab:`", stext))
            src_tabs_end += len(re.findall(r":end_tab:", stext))
        tp = os.path.join(tgt_root, rel)
        if not os.path.exists(tp):
            continue
        text = read_text(tp)
        for m in re.finditer(r":(label|eqlabel):`([^`]*)`", text):
            name = m.group(2)
            if name in labels:
                errs.append(f"标签重复: {name} ({labels[name]} / {rel})")
            labels[name] = rel
        for m in re.finditer(r":(numref|eqref|ref):`([^`]*)`", text):
            target = m.group(2)
            if " <" in target:
                target = target.split("<", 1)[1].rstrip(">").strip()
            refs.append((m.group(1), target, rel))
        cites.extend(re.findall(r":cite:`([^`]*)`", text))
        cites.extend(re.findall(r":citet:`([^`]*)`", text))
        toc_entries_all.extend((rel, e) for e in toc_entries(text))
        tabs_begin += len(re.findall(r":begin_tab:`", text))
        tabs_end += len(re.findall(r":end_tab:", text))

    # bib keys
    bib_path = os.path.join(tgt_root, "d2l.bib")
    bib_keys = set()
    if os.path.exists(bib_path):
        bib_text = read_text(bib_path)
        bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))
    uncited = set()
    for c in cites:
        for key in c.split(","):
            key = key.strip()
            if key and key not in bib_keys:
                uncited.add(key)
    if uncited:
        errs.append(f"引用键在 d2l.bib 中找不到: {sorted(uncited)[:10]}")

    empty_refs = [(k, f) for (k, t, f) in refs if not t]
    if empty_refs:
        warns.append(f"空引用目标 {len(empty_refs)} 处（多为英文底本自身缺陷）: {empty_refs[:4]}")
    bad_refs = [(k, t, f) for (k, t, f) in refs if t and t not in labels]
    if bad_refs:
        errs.append(f"引用目标标签不存在 {len(bad_refs)} 处: {bad_refs[:6]}")

    bad_imgs = []
    for rel in src_files:
        tp = os.path.join(tgt_root, rel)
        if not os.path.exists(tp):
            continue
        text = read_text(tp)
        for m in IMAGE_RE.finditer(md_text_of(parse_items(text))):
            path = m.group(2)
            if re.match(r"https?://", path):
                continue
            abs_path = os.path.normpath(os.path.join(os.path.dirname(tp), path))
            if not os.path.exists(abs_path):
                bad_imgs.append(f"{rel} -> {path}")
    if bad_imgs:
        errs.append(f"图片文件不存在 {len(bad_imgs)} 处: {bad_imgs[:6]}")

    bad_toc = []
    for rel, entry in toc_entries_all:
        base = os.path.dirname(rel)
        if entry.endswith(".md"):
            cand = os.path.join(tgt_root, base, entry)
        else:
            cand = os.path.join(tgt_root, base, entry + ".md")
        if not os.path.exists(cand):
            bad_toc.append(f"{rel}: {entry}")
    if bad_toc:
        warns.append(f"toc 条目对应文件缺失 {len(bad_toc)} 处: {bad_toc[:6]}")

    if tabs_begin != tabs_end or src_tabs_begin != src_tabs_end:
        if (tabs_begin, tabs_end) == (src_tabs_begin, src_tabs_end):
            warns.append(
                f"全书标签块与英文底本一致但与自身不配对 (begin={tabs_begin}, end={tabs_end}；"
                f"源 begin={src_tabs_begin}, end={src_tabs_end}，为英文底本自身缺陷)")
        else:
            errs.append(
                f"全书标签块不配对: begin={tabs_begin}, end={tabs_end} "
                f"(源 begin={src_tabs_begin}, end={src_tabs_end})")

    return errs, warns, {
        "files": len(src_files),
        "labels": len(labels),
        "refs": len(refs),
        "cites": len(cites),
        "toc_entries": len(toc_entries_all),
        "tabs": tabs_begin,
    }


def find_bib_missing_keys(src_root, tgt_root):
    pass


# ------------------------------------------------------------------------ main
def load_batch(path):
    files = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                files.append(line)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=[])
    ap.add_argument("--batch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source-root", default=DEF_SRC)
    ap.add_argument("--target-root", default=DEF_TGT)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--max-details", type=int, default=MAX_DETAILS)
    args = ap.parse_args()

    files = list(args.file)
    if args.batch:
        files.extend(load_batch(args.batch))
    if not files and not args.all:
        ap.error("请给出 --file / --batch 或 --all")

    results = []
    repaired_total = [0, 0, 0]
    for rel in files:
        src = os.path.join(args.source_root, rel)
        tgt = os.path.join(args.target_root, rel)
        if not os.path.exists(src):
            r = FileResult(rel)
            r.err("英文源文件不存在（路径错误？）")
            results.append(r)
            continue
        if args.repair and os.path.exists(tgt):
            fx = repair_file(src, tgt)
            repaired_total = [a + b for a, b in zip(repaired_total, fx)]
        results.append(compare_file(src, tgt))

    n_err = sum(len(r.errors) for r in results)
    n_warn = sum(len(r.warnings) for r in results)
    n_fail = sum(1 for r in results if r.errors)
    print(f"\n===== 单文件校验：{len(results)} 个文件，{len(results) - n_fail} 通过，{n_fail} 失败；"
          f"错误 {n_err}，警告 {n_warn} =====")
    if args.repair:
        print(f"自动修复：代码块 {repaired_total[0]}，独立公式 {repaired_total[1]}，指令 {repaired_total[2]}")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        if r.errors or r.warnings:
            print(f"\n[{status}] {r.path}  (errors={len(r.errors)}, warnings={len(r.warnings)})")
            for e in r.errors[: args.max_details]:
                print(f"  E {e}")
            if len(r.errors) > args.max_details:
                print(f"  ... 另有 {len(r.errors) - args.max_details} 条错误")
            for w in r.warnings[: args.max_details]:
                print(f"  W {w}")
            if len(r.warnings) > args.max_details:
                print(f"  ... 另有 {len(r.warnings) - args.max_details} 条警告")
        elif len(results) > 1:
            print(f"[PASS] {r.path}")

    report = {"files": [{"path": r.path, "errors": r.errors, "warnings": r.warnings,
                         "stats": r.stats} for r in results]}
    if args.all:
        errs, warns, gstats = global_checks(args.source_root, args.target_root)
        print(f"\n===== 全书检查：文件 {gstats['files']}，标签 {gstats['labels']}，"
              f"引用 {gstats['refs']}，文献引用 {gstats['cites']}，toc 条目 {gstats['toc_entries']}，"
              f"标签块 {gstats['tabs']} =====")
        for e in errs:
            print(f"  E {e}")
        for w in warns:
            print(f"  W {w}")
        if not errs:
            print("  全书交叉引用/图片/文献键检查通过")
        report["global"] = {"errors": errs, "warnings": warns, "stats": gstats}

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"报告已写入 {args.json}")

    ok = n_err == 0 and (not args.all or (not report.get("global", {}).get("errors")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
