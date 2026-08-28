"""Emit the paper's curated LaTeX tables from the generated CSVs.

The tables in ``paper/tables/*.tex`` are faithful dumps -- every column, raw
header names, no formatting decisions. Those are the reproducible record. The
manuscript needs something else: a small number of tables with chosen columns,
bolded winners, and captions that say what the reader should take away.

Generating them from the same CSVs means the manuscript cannot drift from the
artefacts. Nothing here is typed by hand.

    python scripts/make_paper_tex.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAB = ROOT / "paper" / "tables"
OUT = ROOT / "paper" / "tex"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["edge_iiotset", "nslkdd", "ton_iot", "unsw_nb15"]
DISPLAY = {"edge_iiotset": "Edge-IIoTset", "nslkdd": "NSL-KDD",
           "ton_iot": "ToN-IoT", "unsw_nb15": "UNSW-NB15"}
#: Column order for every model-indexed table.
MODELS = ["logreg", "decision_tree", "random_forest", "mlp", "resattdnn",
          "lightgbm", "xgboost", "catboost", "soft_vote", "weighted_vote", "gbmeta"]
MNAME = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "mlp": "MLP", "resattdnn": "ResAttDNN",
         "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
         "soft_vote": "Soft Vote", "weighted_vote": "Weighted Vote", "gbmeta": "GB-META"}


def esc(s) -> str:
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def wrap(body: str, caption: str, label: str, star: bool = False, note: str = "") -> str:
    env = "table*" if star else "table"
    # A plain footnote paragraph rather than threeparttable's `tablenotes`:
    # that environment is only legal inside a `threeparttable`, and these tables
    # are \input straight into a float.
    notes = (f"\n\\vspace{{2pt}}\n{{\\footnotesize\\raggedright {note}\\par}}"
             if note else "")
    return (f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{body}{notes}\n\\end{{{env}}}\n")


def write(name: str, text: str) -> None:
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"  wrote paper/tex/{name}.tex")


# --------------------------------------------------------------------------
def tab_datasets() -> None:
    """Benchmark characteristics and the leakage audit, in one table."""
    lk = pd.read_csv(TAB / "table1_leakage_audit.csv").set_index("dataset")
    facts = {}
    for d in DATASETS:
        m = json.loads((ROOT / "results" / "paper" / d / "seed42" / "manifest.json")
                       .read_text(encoding="utf-8"))
        facts[d] = (m["dataset"], m["preprocessing"])

    # The memorisation ceiling is folded in rather than given its own float:
    # it is only meaningful next to the overlap rate it bounds.
    ceil_path = TAB / "table1b_memorisation_ceiling.csv"
    ceil = {}
    if ceil_path.exists():
        cdf = pd.read_csv(ceil_path)
        for _, r in cdf.iterrows():
            key = next((k for k, v in DISPLAY.items() if v == r["dataset"]), None)
            if key:
                ceil[key] = r

    rows = []
    for d in DATASETS:
        ds, pre = facts[d]
        r = lk.loc[d]
        c = ceil.get(d)
        extra = (f"{100*float(c['encoded_collapse_rate']):.1f} & "
                 f"{float(c['memorisation_ceiling']):.3f}") if c is not None else "-- & --"
        rows.append(
            f"{DISPLAY[d]} & {ds['raw_rows']:,} & {ds['n_rows']:,} & {ds['n_classes']} & "
            f"{pre['n_features_after_encoding']} & {ds['imbalance_ratio']:.0f} & "
            f"{100*float(r['exact_duplicate_rate']):.1f} & {100*float(r['train_test_overlap']):.1f} & "
            f"{extra} & {float(r['majority_baseline']):.3f} & "
            f"\\texttt{{{esc(r['best_single_feature'])}}} & "
            f"{float(r['best_single_feature_acc']):.3f} \\\\"
        )
    body = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrrrrlr}\n\\toprule\n"
        "& \\multicolumn{5}{c}{Benchmark} & \\multicolumn{7}{c}{Leakage audit} \\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-13}\n"
        "Dataset & Rows & Used & Cls. & Feat. & Imb. & Dup.\\% & Ovl.\\% & Coll.\\% & "
        "Ceiling & Maj. & Strongest single feature & Acc. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}}\n"
    )
    write("tab_datasets", wrap(
        body,
        "Benchmark characteristics and leakage audit. Column definitions are given in "
        "Section~\\ref{sec:data}.",
        "tab:datasets", star=True))


def tab_main() -> None:
    """Macro-F1 per dataset x model, mean over three seeds, best in bold."""
    m = pd.read_csv(TAB / "table4_cross_dataset.csv").set_index("dataset")
    cols = [c for c in MODELS if c in m.columns]
    lines = []
    for d in DATASETS:
        vals = m.loc[d, cols].astype(float)
        best = vals.idxmax()
        cells = [(f"\\textbf{{{vals[c]:.4f}}}" if c == best else f"{vals[c]:.4f}") for c in cols]
        lines.append(f"{DISPLAY[d]} & " + " & ".join(cells) + " \\\\")

    ranks = json.loads((TAB / "cross_dataset_significance.json").read_text(encoding="utf-8"))
    ar = ranks["friedman"]["average_ranks"]
    best_rank = min(ar, key=ar.get)
    rank_cells = [(f"\\textbf{{{ar[c]:.2f}}}" if c == best_rank else f"{ar[c]:.2f}")
                  for c in cols]

    header = " & ".join(MNAME[c] for c in cols)
    body = (
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{l{'r'*len(cols)}}}\n\\toprule\n"
        f"Dataset & {header} \\\\\n\\midrule\n"
        + "\n".join(lines) +
        "\n\\midrule\n"
        "Avg.\\ rank & " + " & ".join(rank_cells) + " \\\\\n"
        "\\bottomrule\n\\end{tabular}}\n"
    )
    fr = ranks["friedman"]
    note = (f"Friedman $\\chi^2={fr['chi2_statistic']:.2f}$, Iman--Davenport "
            f"$F={fr['iman_davenport_F']:.2f}$, $p={fr['iman_davenport_p_value']:.1e}$. "
            f"Nemenyi critical difference $=$ {ranks['nemenyi']['critical_difference']:.2f} "
            f"at $\\alpha=0.05$, separating {ranks['nemenyi']['n_significant']} of "
            f"{len(ranks['nemenyi']['pairs'])} pairs.")
    write("tab_main", wrap(
        body,
        "Macro-F1 on each benchmark, mean of three seeds. Best per row in bold.",
        "tab:main", star=True, note=note))


def tab_headline() -> None:
    """The stack against its own best base learner."""
    h = pd.read_csv(TAB / "table0_headline.csv")
    rows = []
    for _, r in h.iterrows():
        verdict = ("\\textbf{stack wins}" if r["delta"] > 0 and "yes" in str(r["established"])
                   else "\\textbf{stack loses}" if r["delta"] < 0 and "yes" in str(r["established"])
                   else "no difference")
        rows.append(
            f"{DISPLAY[r['dataset']]} & {r['GB-META']:.4f} & {esc(r['best base'])} & "
            f"{r['best base score']:.4f} & ${r['delta']:+.4f}$ & "
            f"$[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]$ & {r['mcnemar_p_holm']:.3f} & {verdict} \\\\")
    body = ("\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lrlrrrrl}\n\\toprule\n"
            "Dataset & GB-META & Best base & Score & $\\Delta$ & 95\\% CI & McNemar $p$ & Verdict \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    write("tab_headline", wrap(
        body,
        "GB-\\textsc{Meta} against the strongest single base learner on each benchmark "
        "(seed 42). Test details are given in Section~\\ref{sec:results}.",
        "tab:headline", star=True))


def tab_ablation() -> None:
    """The three ablation rows that carry the argument, on all four benchmarks."""
    # The raw-probability variant is omitted: it moves nothing on any benchmark,
    # and the finding is one sentence of prose rather than four table rows.
    want = {"soft vote (no meta-learner)": "Soft vote (no meta-learner)"}
    rows = []
    for d in DATASETS:
        a = pd.read_csv(TAB / f"table6_ablation_{d}.csv")
        ref = a[a.kind == "reference"].iloc[0]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{DISPLAY[d]}}} "
                    f"(full stack: {ref['metric_macro_f1']:.4f})}} \\\\")
        sub = a[a.ablation.isin(want)].copy()
        best = a[a.kind == "no-ensemble"]
        for _, r in pd.concat([sub, best]).iterrows():
            label = want.get(r["ablation"], "Best single base learner")
            if label == "Best single base learner":
                label += f" ({esc(r['ablation'].split('(')[-1].rstrip(')'))})"
            mark = "$\\star$" if r["significant"] else ""
            rows.append(
                f"\\quad {label} & {r['metric_macro_f1']:.4f} & ${r['delta']:+.4f}$ & "
                f"$[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]$ & {mark} \\\\")
    # Resized: the CI column overflows a single IEEE column at natural width.
    body = ("\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lrrrc}\n\\toprule\n"
            "Configuration & Macro-F1 & $\\Delta$ vs.\\ stack & 95\\% CI & Sig. \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    write("tab_ablation", wrap(
        body,
        "Ablation of the combination step, measured against the full three-learner stack. "
        "A positive $\\Delta$ means the simpler configuration is \\emph{better}.",
        "tab:ablation", note="$\\star$ = 95\\% bootstrap interval excludes zero."))


def tab_seedvar() -> None:
    """Seed-to-seed spread and the corrected paired t-test, in one table.

    The two belong together: the spread is only interesting next to the test
    that asks whether the difference exceeds it. The full five-model spread stays
    in the released CSV.
    """
    v = pd.read_csv(TAB / "table5_seed_variance.csv")
    nb = pd.read_csv(TAB / "table5b_corrected_paired_t.csv").set_index("dataset")

    lines = []
    for d in DATASETS:
        sub = v[v.dataset == d].set_index("key")

        def cell(k):
            if k not in sub.index:
                return "--"
            return f"{sub.loc[k, 'mean']:.4f}{{\\scriptsize$\\pm${sub.loc[k, 'std']:.4f}}}"

        base_key = nb.loc[d, "vs"] if d in nb.index else None
        cells = [cell("gbmeta"), cell("soft_vote"),
                 (f"{MNAME.get(base_key, base_key)}: {cell(base_key)}"
                  if base_key else "--")]
        if d in nb.index:
            r = nb.loc[d]
            sig = "\\textbf{yes}" if r["significant"] else "no"
            cells += [f"${r['mean_difference']:+.4f}$", f"{r['p']:.4f}", sig]
        else:
            cells += ["--", "--", "--"]
        lines.append(f"{DISPLAY[d]} & " + " & ".join(cells) + " \\\\")

    body = ("\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lrrlrrc}\n\\toprule\n"
            "& \\multicolumn{3}{c}{Macro-F1, mean $\\pm$ sd (3 seeds)} & "
            "\\multicolumn{3}{c}{Nadeau--Bengio} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Dataset & GB-\\textsc{Meta} & Soft Vote & Best base & $\\Delta$ & $p$ & Sig. \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")
    write("tab_seedvar", wrap(
        body,
        "Seed-to-seed variation and the Nadeau--Bengio corrected resampled paired "
        "$t$-test of GB-\\textsc{Meta} against the strongest base learner. The variance "
        "correction $1/n + n_{\\mathrm{test}}/n_{\\mathrm{train}}$ accounts for the "
        "training-set overlap that the naive paired $t$-test ignores. Read the $\\Delta$ "
        "column against the standard deviations to its left.",
        "tab:seedvar"))


def tab_memceiling() -> None:
    """How much of the reported accuracy exact memorisation could account for."""
    p = TAB / "table1b_memorisation_ceiling.csv"
    if not p.exists():
        print("  (skipping tab_memceiling: run scripts/leakage_probe.py first)")
        return
    c = pd.read_csv(p)
    rows = []
    for _, r in c.iterrows():
        rows.append(
            f"{esc(r['dataset'])} & {100*r['raw_duplicate_rate_before_dedup']:.1f} & "
            f"{100*r['encoded_collapse_rate']:.1f} & {100*r['test_rows_with_train_twin']:.1f} & "
            f"{r['majority_baseline']:.3f} & {r['memorisation_ceiling']:.3f} \\\\")
    body = ("\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Dataset & Dup.\\% & Collapse\\% & Twin\\% & Majority & Ceiling \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_memceiling", wrap(
        body,
        "Upper bound on what exact memorisation can explain. \\emph{Dup.} is the "
        "exact-duplicate rate before splitting. \\emph{Collapse} is the fraction of "
        "training rows that share an encoded feature vector with another training row "
        "after the identifier columns are dropped. \\emph{Twin} is the fraction of test "
        "rows with an identical encoded row in training. \\emph{Ceiling} is the accuracy "
        "of a classifier that answers every twinned test row by lookup and every other "
        "row with the majority class. Any accuracy above the ceiling is generalisation.",
        "tab:memceiling"))


def tab_robust_drift() -> None:
    """Perturbation robustness and temporal drift, side by side.

    Both answer the same operational question -- what happens when test-time data
    stops looking like training data -- so they share a float.
    """
    rp = TAB / "table9_robustness.csv"
    if not rp.exists():
        print("  (skipping tab_robust_drift: run make_robustness_drift_hpo.py first)")
        return
    r = pd.read_csv(rp).set_index("model")

    # The ToN-IoT temporal columns are deliberately NOT shown. A chronological
    # split of that benchmark is class-disjoint by construction (see the paper's
    # drift subsection), so the AUT values measure open-set failure rather than
    # drift, and tabulating them beside the perturbation results would invite
    # exactly the misreading the text warns against.
    order = [m for m in ["GB-META", "CatBoost", "XGBoost", "LightGBM", "Decision Tree"]
             if m in r.index]
    rows = []
    for m in order:
        rr = r.loc[m]
        rows.append(f"{m} & {rr['clean']:.4f} & {rr['noise_0.10']:.4f} & "
                    f"{rr['noise_0.50']:.4f} & {rr['noise_1.00']:.4f} & "
                    f"{rr['mask_20pct']:.4f} & {rr['mask_60pct']:.4f} \\\\")

    body = ("\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "& & \\multicolumn{3}{c}{Gaussian noise $\\sigma$} & "
            "\\multicolumn{2}{c}{Features zeroed} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
            "Model & clean & 0.1 & 0.5 & 1.0 & 20\\% & 60\\% \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    write("tab_robust_drift", wrap(
        body,
        "Macro-F1 on Edge-IIoTset under input perturbation: Gaussian noise of the stated "
        "$\\sigma$ in robust-scaled units, and per-row zeroing of the stated fraction of "
        "features. The ordering here is the reverse of the clean-accuracy ordering.",
        "tab:robust"))


def tab_cost() -> None:
    p = TAB / "table7_deployment_cost.csv"
    if not p.exists():
        print("  (skipping tab_cost: table7_deployment_cost.csv not generated yet)")
        return
    c = pd.read_csv(p)
    # Two producers write this file with different headers; normalise so either
    # one renders rather than failing at LaTeX-generation time.
    c = c.rename(columns={
        "p50 ms (batch 1)": "p50_ms_batch1", "p99 ms (batch 1)": "p99_ms_batch1",
        "peak samples/s": "peak_samples_per_s", "size MB (gz)": "size_mb_gzip",
        "trees/params": "n_trees_or_params", "macro-F1": "macro_f1",
    })
    if "train_seconds" not in c.columns:
        c["train_seconds"] = float("nan")

    # The runner records the stack's own fit time as zero, because it is
    # assembled from already-fitted parts. Its real training cost is the sum of
    # its base learners' fits *plus* the out-of-fold generation the meta-learner
    # requires -- which is the larger half. Substituting it here keeps the table
    # from understating the stack by two orders of magnitude.
    stack_seconds = 0.0
    for m in ("lightgbm", "xgboost", "catboost"):
        rec = ROOT / "results" / "paper" / "edge_iiotset" / "seed42" / "models" / f"{m}.json"
        if rec.exists():
            j = json.loads(rec.read_text(encoding="utf-8"))
            stack_seconds += float(j.get("fit_seconds") or 0) + float(j.get("oof_seconds") or 0)
    if stack_seconds > 0:
        c.loc[c["model"].str.lower().str.replace("-", "") == "gbmeta", "train_seconds"] = stack_seconds

    rows = []
    for _, r in c.iterrows():
        size = "--" if pd.isna(r["size_mb_gzip"]) else f"{r['size_mb_gzip']:.3f}"
        nt = "--" if pd.isna(r["n_trees_or_params"]) else f"{int(r['n_trees_or_params']):,}"
        tr = "--" if pd.isna(r["train_seconds"]) else f"{r['train_seconds']:.1f}"
        rows.append(f"{esc(r['model'])} & {r['macro_f1']:.4f} & {r['p50_ms_batch1']:.3f} & "
                    f"{r['p99_ms_batch1']:.3f} & {int(r['peak_samples_per_s']):,} & "
                    f"{size} & {nt} & {tr} \\\\")
    body = ("\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Model & Macro-F1 & p50 (ms) & p99 (ms) & Peak (s$^{-1}$) & Size (MB) & "
            "Trees & Train (s) \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}}\n")
    write("tab_cost", wrap(
        body,
        "Inference cost and model size on Edge-IIoTset. Measurement protocol is given in "
        "Section~\\ref{sec:cost}.",
        "tab:cost"))


def main() -> int:
    print("generating curated LaTeX tables from the result CSVs")
    tab_datasets()
    tab_main()
    tab_headline()
    tab_ablation()
    tab_seedvar()
    tab_robust_drift()
    tab_cost()
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
