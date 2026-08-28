"""Look up a work in Crossref by title, or dump the registry record for a DOI.

Companion to ``verify_citations.py``: when that reports MISMATCH or NOT_FOUND,
this finds the correct record so the .bib can be fixed against the registry
rather than against memory.

    python scripts/lookup_citation.py --doi 10.1145/2939672.2939785
    python scripts/lookup_citation.py --title "CST-AFNet dual attention intrusion detection"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

UA = "gbmeta-citation-check/1.0 (mailto:noreply@example.org)"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def show(m: dict, prefix: str = "  ") -> None:
    authors = ", ".join(
        f"{a.get('family','')} {a.get('given','')[:1]}." for a in (m.get("author") or [])[:4]
    ) or "(none listed)"
    print(f"{prefix}title    : {(m.get('title') or [''])[0]}")
    print(f"{prefix}authors  : {authors}")
    print(f"{prefix}container: {(m.get('container-title') or [''])[0]}")
    print(f"{prefix}type     : {m.get('type')}")
    print(f"{prefix}volume   : {m.get('volume')}   issue: {m.get('issue')}   pages: {m.get('page')}")
    print(f"{prefix}year     : {(m.get('issued', {}).get('date-parts') or [[None]])[0][0]}")
    print(f"{prefix}DOI      : {m.get('DOI')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi")
    ap.add_argument("--title")
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args(argv)

    if a.doi:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(a.doi, safe="")
        try:
            show(get(url)["message"])
        except Exception as e:
            print(f"  lookup failed: {e}")
        return 0

    if a.title:
        url = ("https://api.crossref.org/works?query.bibliographic="
               + urllib.parse.quote(a.title) + f"&rows={a.rows}&select="
               "DOI,title,container-title,volume,issue,page,issued,author,type")
        try:
            items = get(url)["message"]["items"]
        except Exception as e:
            print(f"  search failed: {e}")
            return 1
        for i, m in enumerate(items, 1):
            print(f"[{i}]")
            show(m, "    ")
            print()
        return 0

    ap.error("give --doi or --title")


if __name__ == "__main__":
    sys.exit(main())
