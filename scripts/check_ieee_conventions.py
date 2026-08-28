"""Check an IEEEtran manuscript against IEEE conference conventions.

Covers the rules that are mechanically checkable. Judgement calls (is this claim
supported, is this figure necessary) are out of scope -- this is the copy-edit
pass that catches the things reviewers and IEEE eXpress reject on.

    python scripts/check_ieee_conventions.py paper/GB_META_v2.tex
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ISSUES: list = []


def flag(rule: str, severity: str, detail: str) -> None:
    ISSUES.append((severity, rule, detail))


def body_of(text: str, root: Path | None = None) -> str:
    """Body text with \\input files inlined.

    Table captions and labels live in generated files under paper/tex/, so a
    checker that reads only the main .tex reports every one of those labels as a
    dangling cross-reference.
    """
    b = text.split(r"\begin{document}", 1)[-1]
    b = b.split(r"\bibliographystyle", 1)[0]
    if root is not None:
        def inline(m):
            f = root / (m.group(1) + ".tex")
            return f.read_text(encoding="utf-8") if f.exists() else m.group(0)
        b = re.sub(r"\\input\{([^}]+)\}", inline, b)
    return b


def prose_of(body: str) -> str:
    """Body with floats, math and macros removed."""
    p = re.sub(r"(?m)(?<!\\)%.*$", "", body)
    p = re.sub(r"\\begin\{(figure\*?|table\*?|equation)\}.*?\\end\{\1\}", " ", p, flags=re.S)
    p = re.sub(r"\$[^$]*\$", " X ", p)
    p = re.sub(r"\\[a-zA-Z]+\*?", " ", p)
    return re.sub(r"[{}]", " ", p)


# --------------------------------------------------------------------------
def check_class_and_preamble(text: str) -> None:
    if "\\documentclass[conference]{IEEEtran}" not in text:
        flag("document class", "MAJOR", "expected \\documentclass[conference]{IEEEtran}")
    if "\\usepackage{cite}" not in text:
        flag("cite package", "MINOR",
             "IEEE recommends the cite package so [1], [2], [3] collapses to [1]-[3]")
    for pkg, why in [("subfigure", "deprecated; use subcaption or a single composed image"),
                     ("times", "IEEEtran selects its own fonts"),
                     ("fullpage", "alters IEEE margins"),
                     ("geometry", "alters IEEE margins -- IEEE eXpress rejects this")]:
        if re.search(rf"\\usepackage(\[[^\]]*\])?\{{{pkg}\}}", text):
            flag(f"package {pkg}", "MAJOR", why)


def check_title(text: str) -> None:
    m = re.search(r"\\title\{(.*?)\}\s*\n\s*\\author", text, re.S)
    if not m:
        m = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", text, re.S)
    if not m:
        flag("title", "MAJOR", "no \\title found")
        return
    t = " ".join(m.group(1).split())
    plain = re.sub(r"\\[a-zA-Z]+|[{}]", "", t)
    if plain.rstrip().endswith("."):
        flag("title", "MINOR", "title should not end with a period")
    n = len(plain.split())
    if n > 16:
        flag("title", "MINOR", f"{n} words; IEEE prefers concise titles (<=16)")
    if re.search(r"\b(a|an|the|and|or|of|for|in|on|with|to)\b\s*$", plain.strip(), re.I):
        flag("title", "MINOR", "title ends on a function word")


def check_abstract(text: str) -> None:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not m:
        flag("abstract", "MAJOR", "no abstract")
        return
    a = m.group(1)
    words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", a).split())
    if not 100 <= words <= 250:
        flag("abstract", "MINOR" if words < 300 else "MAJOR",
             f"{words} words; IEEE conference abstracts are typically 150-250")
    if re.search(r"\\cite\{", a):
        flag("abstract", "MAJOR", "abstract must not contain citations")
    if re.search(r"\\ref\{", a):
        flag("abstract", "MAJOR", "abstract must not reference numbered floats/sections")
    if re.search(r"\\begin\{(equation|itemize|enumerate)\}", a):
        flag("abstract", "MAJOR", "abstract must be a single unstructured paragraph")
    if a.count("\n\n") > 1:
        flag("abstract", "MINOR", "abstract appears to contain more than one paragraph")


def check_index_terms(text: str) -> None:
    m = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", text, re.S)
    if not m:
        flag("index terms", "MAJOR", "no IEEEkeywords block")
        return
    terms = [t.strip() for t in re.sub(r"\s+", " ", m.group(1)).split(",") if t.strip()]
    if not 3 <= len(terms) <= 8:
        flag("index terms", "MINOR", f"{len(terms)} terms; IEEE suggests about 3-8")
    lowered = [t.lower() for t in terms]
    if lowered != sorted(lowered):
        order = " | ".join(terms)
        flag("index terms", "MINOR",
             f"not in alphabetical order (IEEE convention). Current: {order}")


def check_float_refs(body: str, prose: str) -> None:
    # IEEE: abbreviate "Fig." in running text, but spell it out at the start of
    # a sentence. Only flag the mid-sentence uses -- a paragraph-initial "Figure"
    # is correct and flagging it sends the author to break a correct rule.
    mid_sentence = []
    for m in re.finditer(r"\bFigure(~?\\ref|\s+\d)", body):
        before = body[max(0, m.start() - 90):m.start()].rstrip()
        # Sentence-initial if preceded by nothing, a blank line, or . ! ?
        if before and not re.search(r"[.!?]\s*$|\n\s*$|\}\s*$", before):
            mid_sentence.append(before[-45:].replace("\n", " ") + " |Figure|")
    if mid_sentence:
        flag("figure reference", "MINOR",
             f"{len(mid_sentence)} mid-sentence use(s) of 'Figure'; IEEE uses 'Fig.' "
             f"there. e.g. ...{mid_sentence[0]}")
    if re.search(r"\bTab\.~?\\ref", body):
        flag("table reference", "MINOR", "IEEE spells out 'Table', never 'Tab.'")

    labels = set(re.findall(r"\\label\{((?:tab|fig|sec|eq):[^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    uncited = {l for l in labels if l not in refs and not l.startswith("sec:")}
    if uncited:
        flag("float citation", "MAJOR",
             f"float(s) never referenced in text: {sorted(uncited)}")
    dangling = {r for r in refs if r not in labels}
    if dangling:
        flag("cross-reference", "MAJOR", f"\\ref to undefined label(s): {sorted(dangling)}")


def check_equations(body: str) -> None:
    for m in re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", body, re.S):
        eq = m.group(1)
        tail = body[m.end():m.end() + 120].lstrip()
        if not re.match(r"[a-z]", tail) and not eq.rstrip().rstrip("}").rstrip().endswith((",", ".")):
            flag("equation punctuation", "MINOR",
                 "equations are part of the sentence and take terminal punctuation")
    if re.search(r"\bEquation~?\\?ref|\bEq\.\s*\\ref", body):
        flag("equation reference", "MINOR",
             "IEEE refers to equations by bare number, e.g. '(1)', not 'Eq. (1)'")


def check_units_and_numbers(prose: str) -> None:
    bad = re.findall(r"\b\d+(?:\.\d+)?(ms|s|GB|MB|kB|Hz|W|J)\b", prose)
    if bad:
        flag("units", "MINOR",
             f"{len(bad)} number(s) with no thin space before the unit; use 3.10\\,ms")
    for m in re.finditer(r"(?m)^\s*(\d+)\s", prose):
        flag("sentence start", "MINOR", f"sentence appears to begin with a numeral: {m.group(1)}")


#: Words ending in -ise/-ize that are not spelling-variant pairs. "size" and
#: "noise" are not the American form of anything, and itemize/footnotesize are
#: LaTeX macro names that survive macro stripping as bare words.
_NOT_VARIANTS = {
    "size", "sizes", "sized", "noise", "wise", "revised", "revises", "precise",
    "concise", "raise", "raised", "praise", "promise", "premise", "surprise",
    "expertise", "otherwise", "likewise", "itemize", "footnotesize", "prize",
    "compromise", "exercise", "advise", "devise", "arise", "arises", "rise",
}


def check_spelling_consistency(prose: str) -> None:
    def variants(pattern):
        return [w for w in re.findall(pattern, prose, re.I)
                if w.lower() not in _NOT_VARIANTS]

    brit = variants(r"\b(\w+is(?:e|ed|es|ing|ation))\b")
    amer = variants(r"\b(\w+iz(?:e|ed|es|ing|ation))\b")
    if brit and amer:
        flag("spelling", "MINOR",
             f"mixed -ise ({len(brit)}: {sorted(set(brit))[:4]}) and "
             f"-ize ({len(amer)}: {sorted(set(amer))[:4]}) forms; pick one variety")
    b2 = len(re.findall(r"\b(behaviour|colour|favour|labour|centre)\b", prose, re.I))
    a2 = len(re.findall(r"\b(behavior|color|favor|labor|center)\b", prose, re.I))
    if b2 and a2:
        flag("spelling", "MINOR", f"mixed British ({b2}) and American ({a2}) spellings")


def check_style(prose: str) -> None:
    for pat, rule, sev, why in [
        (r"\b(don't|won't|can't|isn't|doesn't|it's|we're|that's)\b",
         "contractions", "MINOR", "contractions are informal for IEEE"),
        (r"\b(obviously|clearly|of course|needless to say)\b",
         "hedging", "MINOR", "avoid asserting obviousness"),
        (r"\b(a lot of|lots of|kind of|sort of|pretty much)\b",
         "register", "MINOR", "informal register"),
        (r"\?(\s|$)", "rhetorical question", "MINOR",
         "questions in body text are unusual in IEEE papers"),
        (r"\b(reviewer|referee|rebuttal|as requested)\b",
         "process language", "MAJOR",
         "revision-process language must not appear in the manuscript"),
        (r"\b(will show|we will present|this paper will)\b",
         "tense", "MINOR", "use present tense for what the paper does"),
    ]:
        hits = re.findall(pat, prose, re.I)
        if hits:
            uniq = sorted({h if isinstance(h, str) else h[0] for h in hits})[:5]
            flag(rule, sev, f"{len(hits)} occurrence(s): {uniq}")


def check_references(bib_path: Path, body: str) -> None:
    if not bib_path.exists():
        return
    bib = bib_path.read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    cited = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        cited |= {k.strip() for k in m.group(1).split(",")}
    missing = cited - keys
    if missing:
        flag("references", "MAJOR", f"cited but not in .bib: {sorted(missing)}")
    unused = keys - cited
    if unused:
        flag("references", "INFO", f"in .bib but never cited: {sorted(unused)}")
    if len(cited) < 15:
        flag("references", "MINOR",
             f"only {len(cited)} references cited; conference papers usually cite 15-30")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="?", default="paper/GB_META_v2.tex")
    a = ap.parse_args(argv)
    p = Path(a.tex)
    text = p.read_text(encoding="utf-8")
    body = body_of(text, p.parent)
    prose = prose_of(body)

    check_class_and_preamble(text)
    check_title(text)
    check_abstract(text)
    check_index_terms(text)
    check_float_refs(body, prose)
    check_equations(body)
    check_units_and_numbers(prose)
    check_spelling_consistency(prose)
    check_style(prose)
    check_references(p.parent / "references.bib", body)

    print(f"=== IEEE convention check: {a.tex} ===\n")
    order = {"MAJOR": 0, "MINOR": 1, "INFO": 2}
    for sev, rule, detail in sorted(ISSUES, key=lambda i: order.get(i[0], 3)):
        print(f"  [{sev:<5}] {rule}: {detail}")
    counts = Counter(s for s, _, _ in ISSUES)
    print(f"\n  MAJOR {counts['MAJOR']}   MINOR {counts['MINOR']}   INFO {counts['INFO']}")
    if not ISSUES:
        print("  no issues found")
    return 1 if counts["MAJOR"] else 0


if __name__ == "__main__":
    sys.exit(main())
