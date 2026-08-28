"""Build the comparison against published Edge-IIoTset results.

Published accuracies on this benchmark span roughly 87% to 100%, which is a wider
range than any modelling difference plausibly explains. The table therefore
carries two protocol columns alongside the scores, because a score reported
without global deduplication is not measuring the same quantity as one reported
with it, and the spread is the point rather than the ranking.

Our own row comes from paper/tables/table2_results_edge_iiotset.csv so it cannot
drift from the rest of the manuscript. The published rows are transcribed, each
with its source recorded here; re-confirm them against the papers before
submission.

    python scripts/make_prior_art_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

TAB = ROOT / "paper" / "tables"
OUT = ROOT / "paper" / "tex"

#: (citation key, method, classes, accuracy %, macro-F1 %, dedup, significance)
#: "--" means the work does not report the quantity.
#: Sources:
#:   ferrag2022edgeiiotset  IEEE Access 10:40281-40306, 2022, centralised DNN, 15-class
#:   abdulkareem2024sel     J. Netw. Comput. Appl. 230:103980, 2024, lightweight stacking
#:   asif2025osen           Expert Syst. Appl. 276:127183, 2025, OSEN-IoT
#:   ishtiaq2025cstafnet    Array 27:100501, 2025, CST-AFNet, macro-P/R/F1 all >99.3
#:   gaber2025xcl           Comput. Res. Model. 17(5):799-827, 2025, binary, balanced subsample
#: Ferrag's own centralised DNN baseline (96.8%, 15-class) sits inside the range
#: these four already span and is dropped for space; it is cited in Related Work.
PUBLISHED = [
    ("abdulkareem2024sel", "Lightweight stacking", "15", "87.37", "--", "no", "no"),
    ("asif2025osen", "OSEN-IoT", "15", "99.71", "--", "no", "no"),
    ("ishtiaq2025cstafnet", "CST-AFNet", "15", "99.97", "$>$99.3", "no", "no"),
    ("gaber2025xcl", "Soft vote (GBDT)", "2", "100.0", "--", "no", "no"),
]


def main() -> int:
    edge = pd.read_csv(TAB / "table2_results_edge_iiotset.csv").set_index("model")
    g = edge.loc["GB-META"]

    rows = [f"{m} & \\cite{{{k}}} & {c} & {a} & {f} & {d} & {s} \\\\"
            for k, m, c, a, f, d, s in PUBLISHED]
    rows.append("\\midrule")
    rows.append(f"GB-\\textsc{{Meta}} (this work) & & 15 & {100 * g['accuracy']:.2f} & "
                f"{100 * g['macro_f1']:.2f} & global & yes \\\\")

    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{Published results on Edge-IIoTset. \\emph{Dedup} and \\emph{Sig.} "
           "record whether duplicates were removed globally and whether any significance "
           "test is reported.}\n\\label{tab:priorart}\n"
           "\\resizebox{\\columnwidth}{!}{%\n"
           "\\begin{tabular}{llrrrll}\n\\toprule\n"
           "Method & & Cls. & Acc. & Ma-F1 & Dedup & Sig. \\\\\n\\midrule\n" +
           "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (OUT / "tab_priorart.tex").write_text(tex, encoding="utf-8")
    print("wrote paper/tex/tab_priorart.tex\n")

    accs = [float(a) for _, _, _, a, _, _, _ in PUBLISHED]
    print(f"  published accuracy range: {min(accs)}% to {max(accs)}%  "
          f"(spread {max(accs) - min(accs):.2f} points)")
    print(f"  this work:                {100 * g['accuracy']:.2f}% accuracy, "
          f"{100 * g['macro_f1']:.2f}% macro-F1, globally deduplicated")
    print("\n  NOTE: the published rows are transcribed, not recomputed. Confirm each")
    print("  against its source before submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
