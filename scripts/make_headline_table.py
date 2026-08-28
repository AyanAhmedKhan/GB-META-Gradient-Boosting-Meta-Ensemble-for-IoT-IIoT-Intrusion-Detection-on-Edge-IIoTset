"""Build the headline comparison table on a seed-aware basis.

The earlier form of this table reported a single seed and a within-split
bootstrap interval. Section~\\ref{sec:results} shows seed-to-seed variation on
the same order as the differences being compared, so a single-seed margin
overstates what the study establishes. This version reports the mean and
standard deviation over three seeds and adjudicates with the Nadeau--Bengio
corrected paired t-test, which is the test that accounts for the training-set
overlap between resampled runs.

Sources: paper/tables/table5_seed_variance.csv (mean, sd over seeds) and
paper/tables/table5b_corrected_paired_t.csv (the corrected test).

    python scripts/make_headline_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

TAB = ROOT / "paper" / "tables"
OUT = ROOT / "paper" / "tex"

DISPLAY = {"edge_iiotset": "Edge-IIoTset", "nslkdd": "NSL-KDD",
           "ton_iot": "ToN-IoT", "unsw_nb15": "UNSW-NB15"}
#: table5b names the comparator in lower case; map to the display name used
#: everywhere else in the manuscript.
MODEL = {"catboost": "CatBoost", "lightgbm": "LightGBM", "xgboost": "XGBoost"}


def main() -> int:
    var = pd.read_csv(TAB / "table5_seed_variance.csv")
    nbt = pd.read_csv(TAB / "table5b_corrected_paired_t.csv")

    def stat(dataset, model):
        r = var[(var.dataset == dataset) & (var.model == model)]
        if r.empty:
            return None
        return float(r.iloc[0]["mean"]), float(r.iloc[0]["std"])

    rows = []
    for _, t in nbt.iterrows():
        ds, base = t["dataset"], MODEL.get(t["vs"], t["vs"])
        g, b = stat(ds, "GB-META"), stat(ds, base)
        if g is None or b is None:
            print(f"  missing seed statistics for {ds}/{base}; skipped")
            continue
        # The verdict follows the corrected test, not the sign of the mean.
        if t["significant"]:
            verdict = ("\\textbf{stack wins}" if t["mean_difference"] > 0
                       else "\\textbf{base learner wins}")
        else:
            verdict = "not established"
        rows.append(
            f"{DISPLAY.get(ds, ds)} & ${g[0]:.4f} \\pm {g[1]:.4f}$ & {base} & "
            f"${b[0]:.4f} \\pm {b[1]:.4f}$ & ${t['mean_difference']:+.4f}$ & "
            f"$[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]$ & {t['p']:.3f} & {verdict} \\\\")

    tex = ("\\begin{table*}[t]\n\\centering\n"
           "\\caption{GB-\\textsc{Meta} against the strongest single base learner on each "
           "benchmark, over three seeds. Adjudicated by the Nadeau--Bengio corrected paired "
           "$t$-test (Section~\\ref{sec:eval}).}\n"
           "\\label{tab:headline}\n"
           "\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{lrlrrrrl}\n\\toprule\n"
           "Dataset & GB-\\textsc{Meta} & Best base & Score & $\\Delta$ & 95\\% CI & "
           "$p$ & Verdict \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    (OUT / "tab_headline.tex").write_text(tex, encoding="utf-8")
    print("wrote paper/tex/tab_headline.tex (seed-aware)\n")

    print(f"  {'dataset':<14}{'GB-META':>20}{'best base':>12}{'score':>20}"
          f"{'delta':>10}{'p':>8}  verdict")
    for _, t in nbt.iterrows():
        ds, base = t["dataset"], MODEL.get(t["vs"], t["vs"])
        g, b = stat(ds, "GB-META"), stat(ds, base)
        if g is None or b is None:
            continue
        v = "significant" if t["significant"] else "not established"
        print(f"  {DISPLAY.get(ds, ds):<14}{f'{g[0]:.4f}+/-{g[1]:.4f}':>20}{base:>12}"
              f"{f'{b[0]:.4f}+/-{b[1]:.4f}':>20}{t['mean_difference']:>+10.4f}"
              f"{t['p']:>8.3f}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
