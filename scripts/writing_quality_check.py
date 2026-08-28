"""Mechanical writing-quality audit of a LaTeX manuscript.

Implements the checkable parts of the academic-paper skill's
``references/writing_quality_check.md``: flagged-term frequency, em-dash and
semicolon budgets, throat-clearing openers, and paragraph-length uniformity.

Only the mechanical rules live here. Judgement calls -- whether a flagged term
is standard terminology in the target discipline, whether a long paragraph earns
its length -- stay with the author; this script surfaces candidates and counts,
it does not rewrite.

    python scripts/writing_quality_check.py paper/GB_META_v2.tex
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Section A -- terms that appear disproportionately in generated text.
FLAGGED = [
    "delve", "tapestry", "landscape", "pivotal", "crucial", "foster", "showcase",
    "testament", "navigate", "leverage", "realm", "embark", "underscore",
    "multifaceted", "nuanced", "comprehensive", "intricate", "cornerstone",
    "paradigm", "synergy", "holistic", "streamline", "cutting-edge",
    "groundbreaking", "seamless", "meticulous", "profound",
]
#: Standard terminology in this manuscript's field; exempt per the checklist's
#: exception rule. "robust" is a statistics term here (robust scaling, robustness
#: to perturbation), not a vague quality claim.
DISCIPLINE_EXEMPT = {"robust", "robustness"}

THROAT_CLEARING = [
    r"^In this (section|paper|study|work),? (we|I) will",
    r"^It is (important|worth|essential) to note",
    r"^This (section|paper) (will )?(discuss|present|describe)",
    r"^We begin by",
    r"^First(ly)?, it is",
    r"^As mentioned (earlier|above|previously)",
]


def strip_latex(text: str) -> str:
    """Remove preamble, comments, math, and macros so prose is what gets counted."""
    body = text.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\bibliographystyle", 1)[0]
    body = re.sub(r"(?m)(?<!\\)%.*$", "", body)          # comments
    body = re.sub(r"\$[^$]*\$", " MATH ", body)          # inline math
    body = re.sub(r"\\begin\{(equation|tabular|table\*?|figure\*?)\}.*?"
                  r"\\end\{\1\}", " FLOAT ", body, flags=re.S)
    body = re.sub(r"\\(input|includegraphics|label|ref|cite|citep|citet)\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)          # remaining macros
    body = re.sub(r"[{}]", " ", body)
    return body


def paragraphs(prose: str) -> list:
    out = []
    for block in re.split(r"\n\s*\n", prose):
        b = " ".join(block.split())
        if len(b.split()) >= 25:      # skip captions, stubs, stray fragments
            out.append(b)
    return out


def sentences(par: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", par) if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="?", default="paper/GB_META_v2.tex")
    a = ap.parse_args(argv)

    raw = Path(a.tex).read_text(encoding="utf-8")
    prose = strip_latex(raw)
    words = prose.split()
    n_words = len(words)
    pars = paragraphs(prose)

    print(f"=== writing quality check: {a.tex} ===")
    print(f"prose words (excl. floats/refs): {n_words:,} | paragraphs: {len(pars)}\n")

    # --- B. punctuation budgets -------------------------------------------
    # Count in the body source, since --- is the LaTeX em dash.
    body_src = raw.split(r"\begin{document}", 1)[-1].split(r"\bibliographystyle", 1)[0]
    body_src = re.sub(r"(?m)(?<!\\)%.*$", "", body_src)
    em = len(re.findall(r"(?<!-)---(?!-)", body_src))
    # `\;` is a math-mode spacing command, not punctuation. Counting it as a
    # semicolon charges the prose budget for every aligned equation.
    semi = len(re.findall(r"(?<!\\);", prose))
    semi_per_1k = semi / max(n_words / 1000, 1e-9)
    print("-- B. punctuation --")
    print(f"em dashes (---)      : {em:3d}   budget <= 3   "
          f"{'OK' if em <= 3 else 'OVER by ' + str(em - 3)}")
    print(f"semicolons           : {semi:3d}   ({semi_per_1k:.1f} per 1000 words, "
          f"budget <= 2.0)   {'OK' if semi_per_1k <= 2 else 'OVER'}")

    # --- A. flagged terms --------------------------------------------------
    low = prose.lower()
    hits = Counter()
    for t in FLAGGED:
        if t in DISCIPLINE_EXEMPT:
            continue
        n = len(re.findall(rf"\b{re.escape(t)}\w*\b", low))
        if n:
            hits[t] = n
    print("\n-- A. flagged terms --")
    print(f"  {dict(hits) if hits else 'none'}"
          + (f"   (exempt as discipline terms: {sorted(DISCIPLINE_EXEMPT)})"))

    # --- C. throat-clearing openers ---------------------------------------
    tc = [p[:70] for p in pars for pat in THROAT_CLEARING
          if re.match(pat, p, flags=re.I)]
    print("\n-- C. throat-clearing openers --")
    print(f"  {len(tc)} found" + ("".join(f"\n    - {t}..." for t in tc) if tc else ""))

    # --- D. paragraph rhythm ----------------------------------------------
    lens = [len(sentences(p)) for p in pars]
    if lens:
        counts = Counter(lens)
        modal, modal_n = counts.most_common(1)[0]
        share = modal_n / len(lens)
        print("\n-- D. paragraph rhythm (sentences per paragraph) --")
        print(f"  range {min(lens)}-{max(lens)}, mean {sum(lens)/len(lens):.1f}")
        print(f"  distribution: {dict(sorted(counts.items()))}")
        print(f"  modal length {modal} occurs in {share:.0%} of paragraphs "
              f"{'(uniform -- vary it)' if share > 0.4 else '(varied, OK)'}")

        wl = [len(p.split()) for p in pars]
        print(f"  words/paragraph: range {min(wl)}-{max(wl)}, mean {sum(wl)/len(wl):.0f}")

    # --- verdict -----------------------------------------------------------
    problems = []
    if em > 3:
        problems.append(f"em dashes {em} > 3")
    if semi_per_1k > 2:
        problems.append(f"semicolons {semi_per_1k:.1f}/1k > 2")
    if hits:
        problems.append(f"{len(hits)} flagged term(s)")
    if tc:
        problems.append(f"{len(tc)} throat-clearing opener(s)")
    print("\n=== verdict ===")
    print("  PASS" if not problems else "  ISSUES: " + "; ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
