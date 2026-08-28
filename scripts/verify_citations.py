"""Verify every reference in a .bib file against the Crossref registry.

Implements the checkable half of the academic-paper skill's citation-compliance
phase (IRON RULE #11: every citation verified via DOI or search). Crossref is the
authoritative registry for DOIs, so a DOI that resolves there with a matching
title is verified; one that 404s is either mistyped or does not exist.

Reports four states per entry:

    VERIFIED   DOI resolves and the registered title matches the .bib title
    MISMATCH   DOI resolves but the title differs -- usually a copied-wrong DOI,
               which is the failure mode that silently attaches a real DOI to the
               wrong paper
    NOT_FOUND  DOI does not resolve; the reference may not exist as cited
    NO_DOI     no DOI in the entry; needs manual verification

It also reports registry metadata (journal, volume, pages, year) so field-level
disagreements are visible rather than assumed correct.

    python scripts/verify_citations.py paper/references.bib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

UA = "gbmeta-citation-check/1.0 (mailto:noreply@example.org)"


def parse_bib(text: str) -> list:
    """Minimal BibTeX reader: enough for well-formed entries, no eval."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        kind, key = m.group(1), m.group(2).strip()
        start = m.end()
        depth, i = 1, m.start(0) + text[m.start(0):].index("{")
        i += 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]

        def field(name):
            fm = re.search(rf"{name}\s*=\s*\{{(.*?)\}},?\s*(?=\n\s*\w+\s*=|\s*$)",
                           body, re.S | re.I)
            if not fm:
                return None
            v = " ".join(fm.group(1).split())
            return re.sub(r"[{}\\]", "", v)

        entries.append({
            "key": key, "type": kind,
            "title": field("title"), "doi": field("doi"),
            "journal": field("journal") or field("booktitle"),
            "volume": field("volume"), "pages": field("pages"), "year": field("year"),
            "flagged": "% CHECK" in text[max(0, m.start() - 400):m.start()],
        })
    return entries


def crossref(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read()).get("message")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib", nargs="?", default="paper/references.bib")
    ap.add_argument("--threshold", type=float, default=0.82)
    a = ap.parse_args(argv)

    entries = parse_bib(Path(a.bib).read_text(encoding="utf-8"))
    print(f"=== citation verification: {a.bib} ({len(entries)} entries) ===\n")

    rows, problems = [], []
    for e in entries:
        if not e["doi"]:
            state, detail = "NO_DOI", "no DOI field -- verify manually"
        else:
            msg = crossref(e["doi"])
            time.sleep(0.4)                       # be polite to the API
            if msg is None:
                state, detail = "NOT_FOUND", "DOI does not resolve at Crossref"
            else:
                reg_title = (msg.get("title") or [""])[0]
                sim = SequenceMatcher(None, norm(e["title"]), norm(reg_title)).ratio()
                reg = {
                    "journal": (msg.get("container-title") or [""])[0],
                    "volume": msg.get("volume"),
                    "pages": msg.get("page"),
                    "year": str((msg.get("issued", {}).get("date-parts") or [[None]])[0][0]),
                }
                # ACM registers proceedings papers under a short title ("XGBoost"
                # for "XGBoost: A Scalable Tree Boosting System"), which tanks the
                # similarity score on a DOI that is actually correct. Treat a
                # registry title that is a prefix of the bib title as a match when
                # the year and first author also agree -- that combination is not
                # something a wrong DOI produces.
                short_form = (
                    len(norm(reg_title)) < len(norm(e["title"]))
                    and norm(e["title"]).startswith(norm(reg_title))
                    and len(norm(reg_title)) >= 4
                )
                first_author = ((msg.get("author") or [{}])[0].get("family") or "").lower()
                author_ok = bool(first_author) and first_author in (e["key"] or "").lower()
                year_ok = bool(e["year"]) and str(e["year"]) == str(reg["year"])

                if sim >= a.threshold or (short_form and year_ok and author_ok):
                    state = "VERIFIED"
                    diffs = [f"{k}: bib={e[k]!r} vs registry={reg[k]!r}"
                             for k in ("volume", "pages", "year")
                             if e[k] and reg[k] and norm(str(e[k])) != norm(str(reg[k]))]
                    detail = "; ".join(diffs) if diffs else "all fields agree"
                    if short_form:
                        detail += "  (registry uses a short title; author+year confirm)"
                    if diffs:
                        state = "VERIFIED*"
                else:
                    state = "MISMATCH"
                    detail = f"registry title: {reg_title[:80]!r} (sim {sim:.2f})"

        rows.append((e["key"], state, detail, e["flagged"]))
        if state not in ("VERIFIED",):
            problems.append((e["key"], state, detail))

    w = max(len(r[0]) for r in rows) + 1
    for key, state, detail, flagged in rows:
        mark = " [was flagged]" if flagged else ""
        print(f"  {key:<{w}} {state:<10} {detail}{mark}")

    print("\n=== summary ===")
    for s in ("VERIFIED", "VERIFIED*", "MISMATCH", "NOT_FOUND", "NO_DOI"):
        n = sum(1 for r in rows if r[1] == s)
        if n:
            print(f"  {s:<10} {n}")
    if problems:
        print("\nNeeds attention before submission:")
        for k, s, d in problems:
            print(f"  - {k}: {s} -- {d}")
    else:
        print("\nAll references verified against Crossref.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
