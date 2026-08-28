"""Audit caption length across the manuscript's floats.

IEEE convention is a short caption that names what the float shows; the
explanation of how to read it belongs in the body text at the point the float is
referenced. Long self-contained captions are a habit from technical reports, and
they cost page space in a two-column layout where caption text is set small and
unbreakable.

Flags any caption over ``--limit`` words so they can be shortened deliberately
rather than by eye.

    python scripts/audit_captions.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def captions(text: str):
    """Yield (label, caption_text) for every \\caption{...} with brace matching."""
    for m in re.finditer(r"\\caption\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end():i - 1]
        tail = text[i:i + 260]
        lm = re.search(r"\\label\{([^}]+)\}", tail)
        yield (lm.group(1) if lm else "(no label)"), " ".join(body.split())


def clean(c: str) -> str:
    c = re.sub(r"\\emph\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\texttt\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\$[^$]*\$", "X", c)
    c = re.sub(r"\\[a-zA-Z]+\*?", " ", c)
    return " ".join(re.sub(r"[{}]", "", c).split())


def reachable(main_tex: Path) -> list[Path]:
    """The main file plus every file it \\input s, in inclusion order.

    scripts/make_paper_tex.py generates a table for every result CSV, but the
    manuscript only includes the subset it has room for. Auditing the whole
    directory reports captions from files no reader ever sees, so the audit
    follows \\input from the main document instead.
    """
    out = [main_tex]
    text = main_tex.read_text(encoding="utf-8")
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        p = (main_tex.parent / m.group(1))
        p = p if p.suffix else p.with_suffix(".tex")
        if p.exists():
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25,
                    help="words above which a caption is flagged (IEEE norm ~15-25)")
    ap.add_argument("--all", action="store_true",
                    help="also audit generated tables the manuscript does not include")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)

    root = Path("paper/GB_META_v2.tex")
    if not a.paths:
        a.paths = ([str(root)] + [str(p) for p in sorted(Path("paper/tex").glob("*.tex"))]
                   if a.all else [str(p) for p in reachable(root)])

    rows, total_over = [], 0
    for p in a.paths:
        src = Path(p)
        if not src.exists():
            continue
        for label, cap in captions(src.read_text(encoding="utf-8")):
            n = len(clean(cap).split())
            rows.append((src.name, label, n, clean(cap)))
            if n > a.limit:
                total_over += n - a.limit

    w = max((len(r[1]) for r in rows), default=10) + 1
    print(f"=== caption audit (limit {a.limit} words) ===\n")
    for fname, label, n, cap in sorted(rows, key=lambda r: -r[2]):
        flag = "OVER " if n > a.limit else "ok   "
        print(f"  {flag} {label:<{w}} {n:>3}w  {fname}")
        if n > a.limit:
            print(f"        {cap[:150]}{'...' if len(cap) > 150 else ''}")
    over = [r for r in rows if r[2] > a.limit]
    print(f"\n{len(over)}/{len(rows)} captions over limit; "
          f"{total_over} words above budget in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
