"""Turn completed runs into every table and figure the paper needs.

Reads only cached artefacts, so it is cheap to re-run after a reviewer asks for
a different metric or one more test. Writes CSV + Markdown + LaTeX (booktabs)
into ``paper/tables/`` and PDF + 600 dpi PNG into ``paper/figures/``.

    python scripts/make_paper_assets.py --tag paper
    python scripts/make_paper_assets.py --tag paper --cost     # + refit for the cost table
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gbmeta.ablation import ablate_stack  # noqa: E402
from gbmeta.analysis import (  # noqa: E402
    collect_runs, cross_dataset_matrix, cross_dataset_significance, leakage_table,
    pretty, results_table, save_table, seed_significance, seed_variance_table,
    significance_table,
)
from gbmeta.config import FIG_DIR, TAB_DIR  # noqa: E402
from gbmeta.evaluate import confusion_from_labels  # noqa: E402
from gbmeta.plots import (  # noqa: E402
    ablation_forest_plot, class_support_vs_f1, confusion_matrix_figure,
    critical_difference_diagram, metric_bars_with_ci, pr_figure, reliability_figure,
    roc_figure, save,
)
from gbmeta.utils import LOG, setup_logging, write_json  # noqa: E402

STACK_BASES = ("lightgbm", "xgboost", "catboost")


def headline(runs, metric, main_seed):
    """The one paragraph a reader needs: does the ensemble actually win?"""
    lines = []
    for d in sorted(runs):
        run = runs[d][main_seed if main_seed in runs[d] else sorted(runs[d])[0]]
        if "gbmeta" not in run["test_proba"]:
            continue
        sig = significance_table(run, reference="gbmeta", metric=metric, B=2000)
        bases = sig[sig["key"].isin(STACK_BASES)]
        if bases.empty:
            continue
        best = bases.sort_values(metric, ascending=False).iloc[0]
        lines.append({
            "dataset": d,
            "GB-META": round(run["records"]["gbmeta"]["metrics"][metric], 5),
            "best base": best["model"],
            "best base score": round(best[metric], 5),
            "delta": round(best["delta_vs_reference"], 5),
            "ci_low": round(best["ci_low"], 5),
            "ci_high": round(best["ci_high"], 5),
            "established": "yes" if best["ci_excludes_zero"] else "no (within noise)",
            "mcnemar_p_holm": (None if best["mcnemar_p_holm"] is None
                               else round(float(best["mcnemar_p_holm"]), 6)),
        })
    return pd.DataFrame(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--main-seed", type=int, default=42)
    ap.add_argument("--cost", action="store_true", help="refit models for the cost table")
    a = ap.parse_args(argv)

    setup_logging()
    runs = collect_runs(a.tag)
    if not runs:
        LOG.error("no completed runs under tag %r", a.tag)
        return 1
    seeds = sorted({s for v in runs.values() for s in v})
    LOG.info("datasets=%s seeds=%s", sorted(runs), seeds)
    main_seed = a.main_seed if a.main_seed in seeds else seeds[0]
    summary = {"tag": a.tag, "metric": a.metric, "datasets": sorted(runs), "seeds": seeds}

    # ---- 1. leakage ------------------------------------------------------
    leak = leakage_table(runs)
    save_table(leak, "table1_leakage_audit",
               caption="Dataset provenance and leakage audit. Duplicate rate is measured "
                       "before splitting; the single-feature probe is a depth-3 stump on "
                       "one feature at a time.",
               label="tab:leakage")
    print("\n=== TABLE 1: leakage audit ===")
    print(leak.drop(columns=["verdict"]).to_string(index=False))

    # ---- 2. per-dataset results -----------------------------------------
    tables = {}
    for d in sorted(runs):
        run = runs[d][main_seed if main_seed in runs[d] else sorted(runs[d])[0]]
        t = results_table(run, metric=a.metric, B=a.boot)
        tables[d] = t
        save_table(t.drop(columns=["key"]), f"table2_results_{d}",
                   caption=f"Model comparison on {d} (seed {main_seed}) with 95\\% "
                           f"percentile bootstrap intervals on {a.metric}.",
                   label=f"tab:results_{d}")
        print(f"\n=== TABLE 2.{d}: results ===")
        cols = ["rank", "model", "accuracy", "balanced_accuracy", a.metric,
                f"{a.metric}_ci", "mcc", "ece"]
        print(t[cols].to_string(index=False))

    # ---- 3. significance -------------------------------------------------
    for d in sorted(runs):
        run = runs[d][main_seed if main_seed in runs[d] else sorted(runs[d])[0]]
        if "gbmeta" not in run["test_proba"]:
            continue
        s = significance_table(run, reference="gbmeta", metric=a.metric, B=a.boot)
        save_table(s.drop(columns=["key"]), f"table3_significance_{d}",
                   caption=f"Every model against GB-META on {d}: paired bootstrap interval "
                           f"on the {a.metric} difference and Holm-corrected McNemar test.",
                   label=f"tab:sig_{d}")

    head = headline(runs, a.metric, main_seed)
    save_table(head, "table0_headline",
               caption="Does the stack beat its best single base learner? Paired bootstrap "
                       "interval on the macro-F1 difference, Holm-corrected McNemar.",
               label="tab:headline")
    print("\n=== TABLE 0: GB-META vs the best single base learner ===")
    print(head.to_string(index=False))
    summary["headline"] = head.to_dict("records")

    # ---- 4. cross-dataset ------------------------------------------------
    matrix = cross_dataset_matrix(runs, metric=a.metric)
    save_table(matrix.reset_index(), "table4_cross_dataset",
               caption=f"{a.metric} across all datasets (mean over {len(seeds)} seeds).",
               label="tab:cross")
    print(f"\n=== TABLE 4: {a.metric} across datasets ===")
    print(matrix.round(4).to_string())

    xstat = cross_dataset_significance(runs, metric=a.metric, reference="gbmeta")
    write_json(TAB_DIR / "cross_dataset_significance.json", xstat)
    summary["cross_dataset"] = xstat
    print("\n=== Friedman ===")
    print({k: v for k, v in xstat["friedman"].items() if k != "average_ranks"})
    if "nemenyi" in xstat:
        n = xstat["nemenyi"]
        print(f"Nemenyi CD = {n['critical_difference']:.3f} | "
              f"{n['n_significant']}/{len(n['pairs'])} pairs separated")
        save(critical_difference_diagram(
            {pretty(k): v for k, v in n["average_ranks"].items()},
            n["critical_difference"], len(xstat["datasets"]),
            f"{pretty(a.metric)} ranks (Nemenyi, alpha=0.05)"), "fig_critical_difference")
    if "wilcoxon" in xstat:
        w = xstat["wilcoxon"]
        print(f"Wilcoxon vs GB-META: {w['n_significant']} significant"
              + (f"  [{w['power_note']}]" if w.get("power_note") else ""))

    # ---- 5. seed variance ------------------------------------------------
    var = seed_variance_table(runs, metric=a.metric)
    save_table(var, "table5_seed_variance",
               caption=f"Seed-to-seed variation of {a.metric} over {len(seeds)} seeds.",
               label="tab:seeds")
    print(f"\n=== TABLE 5: seed variance ({len(seeds)} seeds) ===")
    print(var[var.key.isin(list(STACK_BASES) + ["gbmeta", "soft_vote"])]
          .round(5).to_string(index=False))

    nb_rows = []
    for d in sorted(runs):
        base = (var[(var.dataset == d) & (var.key.isin(STACK_BASES))]
                .sort_values("mean", ascending=False))
        if base.empty:
            continue
        r = seed_significance(runs, d, "gbmeta", base.iloc[0]["key"], metric=a.metric)
        if "error" in r:
            continue
        nb_rows.append({"dataset": d, "vs": base.iloc[0]["key"],
                        "mean_difference": round(r["mean_difference"], 6),
                        "ci_low": round(r["ci_low"], 6), "ci_high": round(r["ci_high"], 6),
                        "t": round(r["t_statistic"], 4), "p": round(r["p_value"], 5),
                        "significant": r["significant"]})
    if nb_rows:
        nb = pd.DataFrame(nb_rows)
        save_table(nb, "table5b_corrected_paired_t",
                   caption="Nadeau-Bengio corrected resampled paired t-test across seeds, "
                           "GB-META against the strongest base learner.",
                   label="tab:nbt")
        print("\n=== TABLE 5b: Nadeau-Bengio corrected paired t (across seeds) ===")
        print(nb.to_string(index=False))
        summary["corrected_paired_t"] = nb_rows

    # ---- 6. ablation -----------------------------------------------------
    for d in sorted(runs):
        run_dir = runs[d][main_seed if main_seed in runs[d] else sorted(runs[d])[0]]["dir"]
        try:
            ab = ablate_stack(run_dir, metric=a.metric, B=a.boot)
        except FileNotFoundError:
            continue
        rows = pd.DataFrame(ab["rows"])
        save_table(rows.drop(columns=["base_models"], errors="ignore"),
                   f"table6_ablation_{d}",
                   caption=f"Component ablation on {d}. Each row is the full stack with one "
                           f"element removed or replaced, with a paired bootstrap interval "
                           f"on the {a.metric} difference.",
                   label=f"tab:abl_{d}")
        save(ablation_forest_plot(ab["rows"], title=f"Component ablation - {d}"),
             f"fig_ablation_{d}")
        print(f"\n=== TABLE 6.{d}: ablation ===\nVERDICT: {ab['verdict']}")
        print(rows[["ablation", f"metric_{a.metric}", "delta", "ci_low", "ci_high",
                    "significant"]].round(5).to_string(index=False))
        summary.setdefault("ablation_verdicts", {})[d] = ab["verdict"]

    # ---- 7. figures ------------------------------------------------------
    for d in sorted(runs):
        run = runs[d][main_seed if main_seed in runs[d] else sorted(runs[d])[0]]
        classes = run["classes"]
        for model in ("gbmeta", "lightgbm"):
            if model not in run["test_proba"]:
                continue
            p = run["test_proba"][model]
            cm = confusion_from_labels(run["y_test"], p.argmax(1), len(classes))
            save(confusion_matrix_figure(cm, classes, f"{pretty(model)} - {d}"),
                 f"fig_confusion_{d}_{model}")
            cpath = run["dir"] / "curves" / f"{model}.json"
            if cpath.exists():
                cur = json.loads(cpath.read_text())
                save(roc_figure(cur["roc"], f"ROC (one-vs-rest) - {d}"), f"fig_roc_{d}_{model}")
                save(pr_figure(cur["pr"], f"Precision-recall - {d}"), f"fig_pr_{d}_{model}")
                save(reliability_figure(cur["reliability"], f"{pretty(model)} - {d}"),
                     f"fig_calibration_{d}_{model}")
            save(class_support_vs_f1(run["records"][model]["per_class"],
                                     f"Per-class F1 vs support - {d}"),
                 f"fig_support_{d}_{model}")
            break

        t = tables[d]
        save(metric_bars_with_ci(
            [{"model": r["model"], "point": r[a.metric],
              "lo": r[f"{a.metric}_ci_low"], "hi": r[f"{a.metric}_ci_high"]}
             for _, r in t.iterrows()],
            metric_label=pretty(a.metric), title=d,
            baseline=run["leakage"].get("majority_class_baseline")),
            f"fig_scores_{d}")

    # ---- 8. optional cost table -----------------------------------------
    if a.cost:
        from dataclasses import replace as _replace
        from gbmeta.config import BUDGET, RunConfig
        from gbmeta.deploy import profile_model
        from gbmeta.ensemble import MetaLearner, StackedEnsemble, compute_oof
        from gbmeta.models.base import ModelContext, available_models, build_model
        from gbmeta.runner import build_data
        from gbmeta.utils import get_device

        d = sorted(runs)[0]
        man = runs[d][main_seed]["manifest"]["config"]["budget"]
        cfg = RunConfig(dataset=d, seed=main_seed, device=get_device("auto"),
                        budget=_replace(BUDGET, **{k: v for k, v in man.items()
                                                   if k in BUDGET.__dataclass_fields__}),
                        tag=a.tag)
        ds, data = build_data(cfg)
        ctx = ModelContext(n_classes=data.n_classes, n_features=data.n_features,
                           seed=main_seed, device=cfg.device, budget=cfg.budget,
                           class_weights=data.class_weights,
                           feature_names=data.feature_names)
        sw = data.sample_weights(data.y_train)
        fitted, profiles = {}, []
        for m in ("decision_tree", "random_forest", "lightgbm", "xgboost", "catboost", "mlp"):
            if m not in available_models():
                continue
            fitted[m] = build_model(m, ctx).fit(data.X_train, data.y_train,
                                                data.X_val, data.y_val, sample_weight=sw)
            profiles.append(profile_model(fitted[m], data.X_test, name=m, n_reps=60))
        keys = [k for k in STACK_BASES if k in fitted]
        if len(keys) >= 2:
            oof = [compute_oof(k, ctx, data.X_train, data.y_train,
                               cfg.budget.n_oof_folds, sw).oof_proba for k in keys]
            comb = MetaLearner(seed=main_seed).fit(oof, data.y_train)
            stack = StackedEnsemble(ctx, {k: fitted[k] for k in keys}, comb)
            profiles.append(profile_model(stack, data.X_test, name="gbmeta", n_reps=60))

        recs = runs[d][main_seed]["records"]
        cost = pd.DataFrame([{
            "model": pretty(p.model),
            a.metric: round(recs.get(p.model, {}).get("metrics", {}).get(a.metric, float("nan")), 4),
            "p50 ms (batch 1)": round(p.latency["latency_batch1_p50_ms"], 4),
            "p99 ms (batch 1)": round(p.latency["latency_batch1_p99_ms"], 4),
            "peak samples/s": int(p.latency["peak_throughput_sps"]),
            "at batch": p.latency["peak_throughput_batch"],
            "size MB (gz)": p.footprint.get("gzip_mb"),
            "trees/params": p.footprint.get("n_trees") or p.footprint.get("n_params"),
        } for p in profiles]).sort_values("p50 ms (batch 1)")
        save_table(cost, "table7_deployment_cost",
                   caption="Inference cost on the measurement host. Latency is batch-1 p50; "
                           "peak throughput is a different operating point.",
                   label="tab:cost")
        print("\n=== TABLE 7: deployment cost ===")
        print(cost.to_string(index=False))
        summary["cost_host"] = profiles[0].environment

    write_json(TAB_DIR / "summary.json", summary)
    print(f"\ntables  -> {TAB_DIR}")
    print(f"figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
