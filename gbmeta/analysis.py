"""Turn cached run artefacts into the paper's tables.

Nothing here trains anything. Every function reads the ``.npy`` prediction
matrices and ``.json`` records that :mod:`gbmeta.runner` wrote, which means the
entire results section can be regenerated in seconds -- including after a
reviewer asks for a different metric, a different test, or one more baseline.

Exports to Markdown and LaTeX (``booktabs``) so the tables can go straight into
the manuscript rather than being retyped, which is where transcription errors
come from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALPHA, N_BOOTSTRAP, RESULTS_DIR
from .evaluate import (
    bootstrap_difference, bootstrap_indices, bootstrap_metric,
    confusion_from_labels, metrics_from_confusion,
)
from .stats import (
    build_rank_matrix, corrected_paired_t, friedman_test, nemenyi_posthoc,
    pairwise_mcnemar, wilcoxon_across_datasets,
)
from .utils import LOG, read_json

#: Column order used in every exported table.
MAIN_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "macro_precision", "macro_recall", "mcc")

PRETTY = {
    "logreg": "Logistic Regression", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "catboost": "CatBoost", "mlp": "MLP", "resattdnn": "ResAttDNN",
    "tabtransformer": "TabTransformer", "tabnet": "TabNet",
    "soft_vote": "Soft Vote", "weighted_vote": "Weighted Vote", "gbmeta": "GB-META",
    "accuracy": "Acc.", "balanced_accuracy": "Bal. Acc.", "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted F1", "macro_precision": "Macro Prec.",
    "macro_recall": "Macro Rec.", "mcc": "MCC", "ece": "ECE",
    "roc_auc_ovr_macro": "ROC-AUC", "pr_auc_macro": "PR-AUC",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_run(run_dir) -> dict:
    """Read one completed (dataset, seed) run from disk."""
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no manifest at {run_dir}")
    probas = {p.stem[:-5]: np.load(p) for p in sorted((run_dir / "probas").glob("*_test.npy"))}
    return {
        "dir": run_dir,
        "manifest": read_json(run_dir / "manifest.json"),
        "leakage": read_json(run_dir / "leakage.json") if (run_dir / "leakage.json").exists() else {},
        "classes": read_json(run_dir / "classes.json"),
        "records": {p.stem: read_json(p) for p in sorted((run_dir / "models").glob("*.json"))},
        "test_proba": probas,
        "y_test": np.load(run_dir / "y_test.npy"),
        "y_train": np.load(run_dir / "y_train.npy"),
    }


def collect_runs(tag: str = "main", results_dir=None) -> dict:
    """``{dataset: {seed: run}}`` for every completed run under a tag."""
    root = Path(results_dir or RESULTS_DIR) / tag
    out: dict = {}
    if not root.exists():
        LOG.warning("no results under %s", root)
        return out
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            if not (seed_dir / "manifest.json").exists():
                continue
            seed = int(seed_dir.name.replace("seed", ""))
            out.setdefault(ds_dir.name, {})[seed] = load_run(seed_dir)
    LOG.info("collected %d datasets, %d runs",
             len(out), sum(len(v) for v in out.values()))
    return out


# --------------------------------------------------------------------------
# Per-dataset results table
# --------------------------------------------------------------------------
def results_table(run: dict, metric: str = "macro_f1", B: int = N_BOOTSTRAP,
                  seed: int = 0, alpha: float = ALPHA) -> pd.DataFrame:
    """Headline table for one run: every metric plus a bootstrap CI on ``metric``.

    All models share one set of bootstrap resample indices, so the intervals are
    directly comparable and the paired differences computed elsewhere use the
    same replicas.
    """
    y = run["y_test"]
    C = len(run["classes"])
    idx = bootstrap_indices(len(y), y, B, seed)

    rows = []
    for name, rec in run["records"].items():
        p = run["test_proba"].get(name)
        if p is None:
            continue
        ci = bootstrap_metric(y, y_pred=p.argmax(1), metric=metric, idx=idx,
                              n_classes=C, alpha=alpha)
        row = {"model": pretty(name), "key": name}
        row.update({m: rec["metrics"].get(m, float("nan")) for m in MAIN_METRICS})
        row.update({
            "ece": rec["metrics"].get("ece", float("nan")),
            "roc_auc_ovr_macro": rec["metrics"].get("roc_auc_ovr_macro", float("nan")),
            "pr_auc_macro": rec["metrics"].get("pr_auc_macro", float("nan")),
            f"{metric}_ci_low": ci.lo, f"{metric}_ci_high": ci.hi,
            f"{metric}_ci": ci.fmt(),
            "train_seconds": rec.get("fit_seconds"),
            "oof_seconds": rec.get("oof_seconds"),
            "gpu": rec.get("uses_gpu"),
        })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def significance_table(run: dict, reference: str = "gbmeta", metric: str = "macro_f1",
                       B: int = N_BOOTSTRAP, seed: int = 0) -> pd.DataFrame:
    """Every model against the reference: paired bootstrap CI + McNemar.

    Two tests because they answer different questions. The bootstrap interval
    says how large the metric difference is and how uncertain; McNemar says
    whether the two models disagree systematically on individual rows. A model
    can be significantly different by McNemar while the metric gap is
    negligible -- and on saturated benchmarks that is the usual outcome.
    """
    y = run["y_test"]
    C = len(run["classes"])
    if reference not in run["test_proba"]:
        raise KeyError(f"{reference} not in this run ({sorted(run['test_proba'])})")
    ref = run["test_proba"][reference].argmax(1)
    idx = bootstrap_indices(len(y), y, B, seed)

    preds = {k: v.argmax(1) for k, v in run["test_proba"].items()}
    mc = pairwise_mcnemar(y, {reference: ref, **{k: v for k, v in preds.items() if k != reference}})
    mc_lookup = {}
    for r in mc["pairs"]:
        mc_lookup[(r["model_a"], r["model_b"])] = r
        mc_lookup[(r["model_b"], r["model_a"])] = r

    rows = []
    for name, pred in preds.items():
        if name == reference:
            continue
        d = bootstrap_difference(y, metric=metric, idx=idx, pred_a=ref, pred_b=pred, n_classes=C)
        m = mc_lookup.get((reference, name), {})
        rows.append({
            "model": pretty(name), "key": name,
            f"{metric}": metrics_from_confusion(confusion_from_labels(y, pred, C))[metric],
            "delta_vs_reference": d["difference"],
            "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "ci_excludes_zero": d["excludes_zero"],
            "mcnemar_p": m.get("p_value"),
            "mcnemar_p_holm": m.get("p_adjusted_holm"),
            "mcnemar_significant": m.get("significant"),
            "n_discordant": m.get("n_discordant"),
        })
    return pd.DataFrame(rows).sort_values("delta_vs_reference").reset_index(drop=True)


# --------------------------------------------------------------------------
# Cross-dataset
# --------------------------------------------------------------------------
def cross_dataset_matrix(runs: dict, metric: str = "macro_f1", seed: int | None = None) -> pd.DataFrame:
    """Datasets (rows) x models (columns) for one metric.

    When several seeds exist, the cell is the mean across seeds; use
    :func:`seed_variance_table` for the spread.
    """
    data = {}
    for ds, by_seed in runs.items():
        seeds = [seed] if seed is not None and seed in by_seed else sorted(by_seed)
        per_model: dict = {}
        for s in seeds:
            for name, rec in by_seed[s]["records"].items():
                per_model.setdefault(name, []).append(rec["metrics"].get(metric, np.nan))
        data[ds] = {k: float(np.nanmean(v)) for k, v in per_model.items()}
    df = pd.DataFrame(data).T
    df.index.name = "dataset"
    return df.sort_index()


def cross_dataset_significance(runs: dict, metric: str = "macro_f1",
                               reference: str = "gbmeta", alpha: float = ALPHA) -> dict:
    """Friedman + Iman-Davenport, Nemenyi post-hoc, and Wilcoxon vs the reference.

    Requires a complete matrix; models missing on any dataset are dropped with a
    warning rather than imputed, because an imputed score would invent a rank.
    """
    per_ds = {}
    for ds, by_seed in runs.items():
        s = sorted(by_seed)[0]
        per_ds[ds] = {k: rec["metrics"] for k, rec in by_seed[s]["records"].items()}
    rm = build_rank_matrix(per_ds, metric)

    out = {"metric": metric, "datasets": rm.dataset_names, "models": rm.model_names,
           "scores": rm.scores.tolist(), "friedman": friedman_test(rm, alpha)}
    if "error" not in out["friedman"]:
        out["nemenyi"] = nemenyi_posthoc(rm, alpha)
    if reference in rm.model_names:
        out["wilcoxon"] = wilcoxon_across_datasets(rm, reference, alpha)
    return out


def seed_variance_table(runs: dict, metric: str = "macro_f1") -> pd.DataFrame:
    """Mean +/- std across seeds, per dataset and model.

    This is the table that makes a "+0.03 pp improvement" claim checkable: if
    the seed-to-seed standard deviation is larger than the improvement, the
    improvement is not a result.
    """
    rows = []
    for ds, by_seed in runs.items():
        per_model: dict = {}
        for s, run in sorted(by_seed.items()):
            for name, rec in run["records"].items():
                per_model.setdefault(name, []).append(rec["metrics"].get(metric, np.nan))
        for name, vals in per_model.items():
            v = np.asarray(vals, dtype=float)
            rows.append({
                "dataset": ds, "model": pretty(name), "key": name,
                "n_seeds": int(np.isfinite(v).sum()),
                "mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                "spread": float(np.nanmax(v) - np.nanmin(v)),
            })
    return pd.DataFrame(rows).sort_values(["dataset", "mean"], ascending=[True, False])


def seed_significance(runs: dict, dataset: str, model_a: str, model_b: str,
                      metric: str = "macro_f1") -> dict:
    """Nadeau-Bengio corrected paired t-test across seeds for one dataset."""
    by_seed = runs[dataset]
    a, b = [], []
    for s in sorted(by_seed):
        recs = by_seed[s]["records"]
        if model_a in recs and model_b in recs:
            a.append(recs[model_a]["metrics"][metric])
            b.append(recs[model_b]["metrics"][metric])
    if len(a) < 2:
        return {"error": f"need >=2 seeds with both models (have {len(a)})"}
    sizes = by_seed[sorted(by_seed)[0]]["manifest"]["preprocessing"]["split_sizes"]
    return {
        "dataset": dataset, "model_a": model_a, "model_b": model_b, "metric": metric,
        "scores_a": a, "scores_b": b,
        **corrected_paired_t(a, b, n_train=sizes["train"], n_test=sizes["test"]),
    }


# --------------------------------------------------------------------------
# Leakage summary
# --------------------------------------------------------------------------
def leakage_table(runs: dict) -> pd.DataFrame:
    """One row per dataset: duplicate rate, train/test overlap, strongest single feature.

    The number a reader should see before any accuracy table.
    """
    rows = []
    for ds, by_seed in sorted(runs.items()):
        run = by_seed[sorted(by_seed)[0]]
        lk, man = run["leakage"], run["manifest"]
        probe = (lk.get("top_single_feature_probes") or [{}])[0]
        rows.append({
            "dataset": ds,
            "rows_raw": man["dataset"].get("raw_rows"),
            "rows_used": man["dataset"].get("n_rows"),
            "classes": man["dataset"].get("n_classes"),
            "imbalance_ratio": round(man["dataset"].get("imbalance_ratio", float("nan")), 1),
            "exact_duplicate_rate": lk.get("exact_duplicate_rate_raw"),
            "train_test_overlap": lk.get("train_test_overlap_rate"),
            "majority_baseline": lk.get("majority_class_baseline"),
            "best_single_feature": probe.get("feature"),
            "best_single_feature_acc": probe.get("single_feature_accuracy"),
            "verdict": lk.get("verdict"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}", max_rows: int | None = None) -> str:
    """Markdown table, with a hand-rolled fallback when ``tabulate`` is absent.

    ``DataFrame.to_markdown`` raises ImportError without tabulate, which would
    take down the whole export stage over a formatting nicety.
    """
    d = df if max_rows is None else df.head(max_rows)
    try:
        return d.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        def _fmt(v):
            if isinstance(v, float):
                return "" if v != v else float_fmt.format(v)
            return str(v)
        cols = list(d.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |",
                 "|" + "|".join("---" for _ in cols) + "|"]
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
        return chr(10).join(lines)


def to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.4f",
             columns=None, highlight_max: str | None = None) -> str:
    """LaTeX ``booktabs`` table, ready to \\input into the manuscript."""
    d = df[columns] if columns else df
    d = d.copy()
    if highlight_max and highlight_max in d.columns:
        best = d[highlight_max].astype(float).idxmax()
        d.loc[best, d.columns[0]] = r"\textbf{" + str(d.loc[best, d.columns[0]]) + "}"
    body = d.to_latex(index=False, float_format=float_fmt, escape=False,
                      column_format="l" + "r" * (len(d.columns) - 1))
    body = (body.replace(r"\toprule", r"\toprule").replace("\\begin{tabular}", "\\begin{tabular}"))
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"{body}"
        "}\n\\end{table}\n"
    )


def save_table(df: pd.DataFrame, name: str, out_dir=None, caption: str = "", label: str = "",
               columns=None) -> dict:
    """Write one table as CSV, Markdown and LaTeX."""
    from .config import TAB_DIR

    out_dir = Path(out_dir or TAB_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["csv"] = out_dir / f"{name}.csv"
    df.to_csv(paths["csv"], index=False)
    paths["md"] = out_dir / f"{name}.md"
    paths["md"].write_text(to_markdown(df), encoding="utf-8")
    paths["tex"] = out_dir / f"{name}.tex"
    paths["tex"].write_text(
        to_latex(df, caption or name, label or f"tab:{name}", columns=columns), encoding="utf-8")
    LOG.info("table written: %s (.csv/.md/.tex)", name)
    return {k: str(v) for k, v in paths.items()}
