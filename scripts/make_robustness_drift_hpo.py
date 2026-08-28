"""Robustness, concept drift, and the Bayesian-optimisation ablation.

Covers the three reviewer requests that the cached probability matrices cannot
answer, because each needs either a perturbed input or a refit model:

* adversarial/robustness sweeps  -- constrained Gaussian noise and feature masking
* concept drift                  -- chronological split on ToN-IoT, summarised by AUT
* Bayesian optimisation ablation -- Optuna, reporting the TEST gain alongside
                                    the validation gain a tuning run reports

    python scripts/make_robustness_drift_hpo.py --tag paper
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gbmeta.analysis import collect_runs, pretty, save_table  # noqa: E402
from gbmeta.config import BUDGET, RunConfig  # noqa: E402
from gbmeta.datasets import get_spec  # noqa: E402
from gbmeta.ensemble import MetaLearner, StackedEnsemble, compute_oof  # noqa: E402
from gbmeta.evaluate import compute_metrics  # noqa: E402
from gbmeta.hpo import build_tuned, tune_model  # noqa: E402
from gbmeta.models.base import ModelContext, available_models, build_model  # noqa: E402
from gbmeta.plots import degradation_curve, save  # noqa: E402
from gbmeta.robustness import (  # noqa: E402
    feature_drift_report, feature_masking_sweep, gaussian_noise_sweep,
    immutable_mask, robustness_summary, temporal_decay,
)
from gbmeta.runner import build_data  # noqa: E402
from gbmeta.utils import LOG, get_device, setup_logging, write_json  # noqa: E402

STACK_BASES = ("lightgbm", "xgboost", "catboost")


def _ctx(data, seed, device, budget):
    return ModelContext(n_classes=data.n_classes, n_features=data.n_features, seed=seed,
                        device=device, budget=budget, class_weights=data.class_weights,
                        feature_names=data.feature_names)


def _budget_from(run):
    stored = run["manifest"]["config"]["budget"]
    return replace(BUDGET, **{k: v for k, v in stored.items()
                              if k in BUDGET.__dataclass_fields__})


def robustness(tag, dataset, seed, runs):
    LOG.info("=" * 60); LOG.info("ROBUSTNESS on %s", dataset)
    budget = _budget_from(runs[dataset][seed])
    cfg = RunConfig(dataset=dataset, seed=seed, budget=budget,
                    device=get_device("auto"), tag=tag)
    ds, data = build_data(cfg)
    ctx = _ctx(data, seed, cfg.device, budget)
    sw = data.sample_weights(data.y_train)

    fitted = {}
    for m in (*STACK_BASES, "decision_tree"):
        if m in available_models():
            fitted[m] = build_model(m, ctx).fit(data.X_train, data.y_train,
                                                data.X_val, data.y_val, sample_weight=sw)
    keys = [k for k in STACK_BASES if k in fitted]
    oof = [compute_oof(k, ctx, data.X_train, data.y_train, budget.n_oof_folds, sw).oof_proba
           for k in keys]
    comb = MetaLearner(seed=seed).fit(oof, data.y_train)
    fitted["gbmeta"] = StackedEnsemble(ctx, {k: fitted[k] for k in keys}, comb)

    # Only features an attacker could plausibly control are perturbed; the mask
    # is a stated modelling assumption and is published with the result.
    mutable = ~immutable_mask(data.feature_names)
    LOG.info("%d/%d features treated as attacker-mutable", mutable.sum(), len(mutable))

    noise, mask, rows = {}, {}, []
    for m, mdl in fitted.items():
        g = gaussian_noise_sweep(mdl.predict_proba, data.X_test, data.y_test,
                                 data.n_classes, mutable=mutable)
        f = feature_masking_sweep(mdl.predict_proba, data.X_test, data.y_test,
                                  data.n_classes, mutable=mutable)
        noise[pretty(m)] = g.curve("macro_f1")
        mask[pretty(m)] = f.curve("macro_f1")
        sg, sf = robustness_summary(g), robustness_summary(f)
        rows.append({
            "model": pretty(m),
            "clean": round(sg["clean"], 4),
            "noise_0.10": round(dict(zip(*g.curve("macro_f1")))[0.1], 4),
            "noise_0.50": round(dict(zip(*g.curve("macro_f1")))[0.5], 4),
            "noise_1.00": round(dict(zip(*g.curve("macro_f1")))[1.0], 4),
            "rel_drop_at_1.0": round(sg["relative_drop_at_max"], 4),
            "mask_20pct": round(dict(zip(*f.curve("macro_f1")))[0.2], 4),
            "mask_60pct": round(dict(zip(*f.curve("macro_f1")))[0.6], 4),
        })
    df = pd.DataFrame(rows).sort_values("rel_drop_at_1.0")
    save_table(df, "table9_robustness",
               caption="Macro-F1 under constrained perturbation of attacker-mutable "
                       "features on Edge-IIoTset. Noise is Gaussian with the stated sigma "
                       "in robust-scaled units; masking zeroes the stated fraction of "
                       "mutable features per row.",
               label="tab:robust")
    save(degradation_curve(noise, "Gaussian noise $\\sigma$ (robust-scaled units)",
                           title=None), "fig_robust_noise")
    save(degradation_curve(mask, "Fraction of mutable features zeroed", title=None),
         "fig_robust_masking")
    print("\n=== TABLE 9: robustness ===\n", df.to_string(index=False))
    return df


def drift(tag, seed, runs):
    LOG.info("=" * 60); LOG.info("CONCEPT DRIFT on ton_iot (temporal split)")
    if "ton_iot" not in runs:
        LOG.warning("ton_iot not in study; skipping drift"); return None
    budget = _budget_from(runs["ton_iot"][seed])
    cfg = RunConfig(dataset="ton_iot", seed=seed, budget=budget,
                    device=get_device("auto"), tag=f"{tag}-temporal")
    ds, data = build_data(cfg, temporal=True)
    if data.splits.mode != "temporal" or ds.timestamps is None:
        LOG.warning("temporal split unavailable"); return None
    LOG.info("train ends %s, test starts %s",
             data.splits.meta.get("train_end"), data.splits.meta.get("test_start"))

    ctx = _ctx(data, seed, cfg.device, budget)
    sw = data.sample_weights(data.y_train)
    fitted = {}
    for m in (*STACK_BASES,):
        if m in available_models():
            fitted[m] = build_model(m, ctx).fit(data.X_train, data.y_train,
                                                data.X_val, data.y_val, sample_weight=sw)
    keys = list(fitted)
    oof = [compute_oof(k, ctx, data.X_train, data.y_train, budget.n_oof_folds, sw).oof_proba
           for k in keys]
    comb = MetaLearner(seed=seed).fit(oof, data.y_train)
    fitted["gbmeta"] = StackedEnsemble(ctx, {k: fitted[k] for k in keys}, comb)

    fd = feature_drift_report(data.X_train, data.X_test, data.feature_names)
    ts = ds.timestamps.iloc[data.splits.test].to_numpy()

    rows, curves = [], {}
    for m, mdl in fitted.items():
        dec = temporal_decay(mdl.predict_proba, data.X_test, data.y_test, ts,
                             data.n_classes, n_windows=6)
        rows.append({"model": pretty(m), "AUT": round(dec["aut"], 4),
                     "first_window": round(dec["first"], 4),
                     "last_window": round(dec["last"], 4),
                     "decay": round(dec["decay"], 4)})
        curves[pretty(m)] = ([w["window"] for w in dec["windows"]],
                             [w["macro_f1"] for w in dec["windows"]])
    df = pd.DataFrame(rows).sort_values("AUT", ascending=False)
    save_table(df, "table10_drift",
               caption="Temporal evaluation on ToN-IoT: models are trained on the earliest "
                       "traffic and evaluated on six consecutive later windows. AUT is the "
                       "time-decay-aware summary of macro-F1 over those windows.",
               label="tab:drift")
    save(degradation_curve(curves, "Consecutive test window (chronological)", title=None),
         "fig_drift_temporal")
    write_json(ROOT / "paper" / "tables" / "drift_feature_report.json", fd)
    print("\n=== TABLE 10: concept drift (ToN-IoT, temporal) ===\n", df.to_string(index=False))
    print(f"features with PSI > 0.25: {fd['n_psi_above_0.25']}/{fd['n_features']} "
          f"(mean PSI {fd['mean_psi']:.3f})")
    return df


def hpo(tag, dataset, seed, runs, n_trials):
    LOG.info("=" * 60); LOG.info("BAYESIAN OPTIMISATION ABLATION on %s", dataset)
    budget = _budget_from(runs[dataset][seed])
    budget = replace(budget, n_trials=n_trials)
    cfg = RunConfig(dataset=dataset, seed=seed, budget=budget,
                    device=get_device("auto"), tag=tag)
    ds, data = build_data(cfg)
    ctx = _ctx(data, seed, cfg.device, budget)
    sw = data.sample_weights(data.y_train)

    rows = []
    for m in ("lightgbm",):
        if m not in available_models():
            continue
        res = tune_model(m, ctx, data.X_train, data.y_train, data.X_val, data.y_val,
                         n_trials=n_trials, sample_weight=sw)
        tuned = build_tuned(m, ctx, res).fit(data.X_train, data.y_train,
                                             data.X_val, data.y_val, sample_weight=sw)
        test_tuned = compute_metrics(data.y_test, tuned.predict_proba(data.X_test),
                                     data.n_classes)["macro_f1"]
        test_default = runs[dataset][seed]["records"][m]["metrics"]["macro_f1"]
        rows.append({
            "model": pretty(m), "trials": res.n_trials, "pruned": res.n_pruned,
            "val_default": round(res.default_score, 4),
            "val_tuned": round(res.best_score, 4),
            "val_gain": round(res.improvement, 4),
            "test_default": round(test_default, 4),
            "test_tuned": round(test_tuned, 4),
            "test_gain": round(test_tuned - test_default, 4),
            "seconds": round(res.seconds),
        })
    df = pd.DataFrame(rows)
    save_table(df, "table11_hpo",
               caption="Contribution of Bayesian hyper-parameter optimisation. The "
                       "validation gain is what a tuning run reports; the test gain is what "
                       "it delivers. Reporting the former among test results overstates "
                       "what tuning contributes.",
               label="tab:hpo")
    print("\n=== TABLE 11: Bayesian optimisation ===\n", df.to_string(index=False))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--dataset", default="edge_iiotset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--skip", nargs="*", default=[], choices=["robustness", "drift", "hpo"])
    a = ap.parse_args(argv)
    setup_logging()

    runs = collect_runs(a.tag)
    if not runs:
        LOG.error("no completed runs under tag %r", a.tag); return 1

    if "robustness" not in a.skip:
        robustness(a.tag, a.dataset, a.seed, runs)
    if "drift" not in a.skip:
        drift(a.tag, a.seed, runs)
    if "hpo" not in a.skip:
        hpo(a.tag, a.dataset, a.seed, runs, a.trials)
    return 0


if __name__ == "__main__":
    sys.exit(main())
