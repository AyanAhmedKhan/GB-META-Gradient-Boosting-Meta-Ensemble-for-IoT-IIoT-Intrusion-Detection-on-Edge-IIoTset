"""Deployment-cost table for a completed study, on the study's own configuration.

Split out from ``make_paper_assets`` because it is the only stage that has to
refit models: latency and model size cannot be recovered from cached
probabilities. Keeping it separate means the statistical tables can be
regenerated in minutes without paying for this again.

    python scripts/make_cost_table.py --tag paper --dataset edge_iiotset
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from gbmeta.analysis import collect_runs, pretty, save_table  # noqa: E402
from gbmeta.config import BUDGET, RunConfig  # noqa: E402
from gbmeta.deploy import profile_model  # noqa: E402
from gbmeta.ensemble import MetaLearner, StackedEnsemble, compute_oof  # noqa: E402
from gbmeta.models.base import ModelContext, available_models, build_model  # noqa: E402
from gbmeta.runner import build_data  # noqa: E402
from gbmeta.utils import LOG, environment_manifest, get_device, setup_logging, write_json  # noqa: E402

STACK_BASES = ("lightgbm", "xgboost", "catboost")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--dataset", default="edge_iiotset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--models", nargs="*",
                    default=["decision_tree", "random_forest", "lightgbm",
                             "xgboost", "catboost", "mlp"])
    a = ap.parse_args(argv)
    setup_logging()

    runs = collect_runs(a.tag)
    if a.dataset not in runs or a.seed not in runs[a.dataset]:
        LOG.error("no completed run for %s seed%d under tag %r", a.dataset, a.seed, a.tag)
        return 1
    run = runs[a.dataset][a.seed]

    # Rebuild the *exact* data the study used, from its own manifest.
    stored = run["manifest"]["config"]["budget"]
    budget = replace(BUDGET, **{k: v for k, v in stored.items()
                                if k in BUDGET.__dataclass_fields__})
    cfg = RunConfig(dataset=a.dataset, seed=a.seed, budget=budget,
                    device=get_device("auto"), tag=a.tag)
    ds, data = build_data(cfg)
    ctx = ModelContext(n_classes=data.n_classes, n_features=data.n_features, seed=a.seed,
                       device=cfg.device, budget=budget,
                       class_weights=data.class_weights, feature_names=data.feature_names)
    sw = data.sample_weights(data.y_train)

    fitted, profiles = {}, []
    for m in a.models:
        if m not in available_models():
            LOG.warning("skipping %s (backend missing)", m)
            continue
        fitted[m] = build_model(m, ctx).fit(data.X_train, data.y_train,
                                            data.X_val, data.y_val, sample_weight=sw)
        profiles.append(profile_model(fitted[m], data.X_test, name=m, n_reps=a.reps))

    # The stack is profiled end to end -- every base model's forward pass plus
    # the meta-learner -- because that is what would actually be deployed.
    keys = [k for k in STACK_BASES if k in fitted]
    if len(keys) >= 2:
        oof = [compute_oof(k, ctx, data.X_train, data.y_train,
                           budget.n_oof_folds, sw).oof_proba for k in keys]
        comb = MetaLearner(seed=a.seed).fit(oof, data.y_train)
        stack = StackedEnsemble(ctx, {k: fitted[k] for k in keys}, comb)
        profiles.append(profile_model(stack, data.X_test, name="gbmeta", n_reps=a.reps))

    recs = run["records"]
    rows = []
    for p in profiles:
        d = p.as_dict()
        rows.append({
            "model": pretty(d["model"]),
            "macro_f1": round(recs.get(d["model"], {}).get("metrics", {})
                              .get("macro_f1", float("nan")), 4),
            "p50_ms_batch1": round(d["latency"]["latency_batch1_p50_ms"], 4),
            "p99_ms_batch1": round(d["latency"]["latency_batch1_p99_ms"], 4),
            "peak_samples_per_s": int(d["latency"]["peak_throughput_sps"]),
            "peak_batch": d["latency"]["peak_throughput_batch"],
            "size_mb_gzip": d["footprint"].get("gzip_mb"),
            "n_trees_or_params": d["footprint"].get("n_trees") or d["footprint"].get("n_params"),
            "train_seconds": round(recs.get(d["model"], {}).get("fit_seconds", float("nan")), 1),
        })
    df = pd.DataFrame(rows).sort_values("p50_ms_batch1").reset_index(drop=True)

    save_table(df, "table7_deployment_cost",
               caption="Inference cost on the measurement host (CPU only; see text). "
                       "Latency is batch-1 p50 over 60 timed repetitions after 20 warm-up "
                       "calls; peak throughput is a different operating point reached at a "
                       "larger batch. GB-META is profiled end to end, including every base "
                       "model's forward pass.",
               label="tab:cost")
    write_json(ROOT / "paper" / "tables" / "cost_environment.json",
               {"dataset": a.dataset, "seed": a.seed, "n_test_rows": int(len(data.y_test)),
                "n_features": int(data.n_features), "reps": a.reps,
                "environment": environment_manifest()})
    print("\n=== TABLE 7: deployment cost ===")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
