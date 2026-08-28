"""Component ablation.

Quantifies the contribution of each base learner, of the meta-learner, and of
hyper-parameter search.

The design point: the expensive half of an ablation is free if the run already
cached out-of-fold and test probabilities. Dropping a base learner from the
stack, swapping the combiner, or changing the meta feature transform costs one
logistic-regression fit on a matrix that is already in memory -- no retraining,
no GPU. Only the ablations that genuinely change how a *base* model is trained
(hyper-parameter search, class weighting, deduplication) need a rerun, and those
are driven through :mod:`gbmeta.runner` with a distinct tag.

Every ablation row carries a paired bootstrap interval on the difference from
the full model, so "contribution" is a measured effect with uncertainty rather
than a difference of two point estimates.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .config import ALPHA, N_BOOTSTRAP
from .ensemble import MetaLearner, SoftVote, WeightedVote, stack_features
from .evaluate import (
    bootstrap_difference, bootstrap_indices, confusion_from_labels, metrics_from_confusion,
)
from .stats import mcnemar_test
from .utils import LOG, read_json


# --------------------------------------------------------------------------
# Loading cached artefacts
# --------------------------------------------------------------------------
def load_stack_arrays(run_dir) -> dict:
    """Read the cached OOF / test probability matrices for one run."""
    run_dir = Path(run_dir)
    probas = run_dir / "probas"
    keys = sorted(p.stem[:-4] for p in probas.glob("*_oof.npy"))
    if not keys:
        raise FileNotFoundError(f"no OOF matrices under {probas}")
    return {
        "base_keys": keys,
        "oof": {k: np.load(probas / f"{k}_oof.npy").astype(np.float64) for k in keys},
        "test": {k: np.load(probas / f"{k}_test.npy").astype(np.float64) for k in keys},
        "y_train": np.load(run_dir / "y_train.npy"),
        "y_test": np.load(run_dir / "y_test.npy"),
        "classes": read_json(run_dir / "classes.json"),
    }


def _fit_and_score(subset, arrays, combiner, n_classes):
    oof = [arrays["oof"][k] for k in subset]
    test = [arrays["test"][k] for k in subset]
    acc = [float((o.argmax(1) == arrays["y_train"]).mean()) for o in oof]
    combiner.fit(oof, arrays["y_train"], val_accuracies=acc)
    p = np.asarray(combiner.combine(test), dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(1)
    m = metrics_from_confusion(confusion_from_labels(arrays["y_test"], pred, n_classes))
    return pred, p, m


# --------------------------------------------------------------------------
# Free ablations
# --------------------------------------------------------------------------
def ablate_stack(
    run_dir, metric: str = "macro_f1", B: int = N_BOOTSTRAP, alpha: float = ALPHA,
    seed: int | None = None,
) -> dict:
    """Leave-one-out over base learners, plus every combiner and transform.

    The reference is the full stack with the logit-transformed meta-learner.
    Each row reports the metric, the paired difference from the reference, its
    bootstrap interval, and a McNemar p-value on the prediction disagreement.
    """
    arrays = load_stack_arrays(run_dir)
    # Reuse the run's own seed so the reference row reproduces the headline
    # GB-META number exactly rather than a re-seeded refit of it.
    if seed is None:
        manifest = Path(run_dir) / "manifest.json"
        seed = int(read_json(manifest)["config"]["seed"]) if manifest.exists() else 0
    keys, y_test = arrays["base_keys"], arrays["y_test"]
    C = int(max(y_test.max(), max(a.shape[1] for a in arrays["test"].values()) - 1) + 1)
    C = max(C, next(iter(arrays["test"].values())).shape[1])
    idx = bootstrap_indices(len(y_test), y_test, B, seed)

    ref_pred, _ref_p, ref_m = _fit_and_score(keys, arrays, MetaLearner(seed=seed), C)
    rows = [{
        "ablation": "full stack (reference)", "kind": "reference",
        "base_models": list(keys), "combiner": "gbmeta/logit",
        **{f"metric_{k}": v for k, v in ref_m.items()},
        "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "significant": False, "mcnemar_p": 1.0,
    }]

    def add(label, kind, subset, combiner, note=""):
        pred, _p, m = _fit_and_score(subset, arrays, combiner, C)
        d = bootstrap_difference(y_test, metric=metric, B=B, seed=seed, idx=idx,
                                 pred_a=pred, pred_b=ref_pred, n_classes=C)
        mc = mcnemar_test(y_test, pred, ref_pred)
        rows.append({
            "ablation": label, "kind": kind, "base_models": list(subset),
            "combiner": combiner.name, "note": note,
            **{f"metric_{k}": v for k, v in m.items()},
            "delta": d["difference"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "significant": d["excludes_zero"], "mcnemar_p": mc["p_value"],
            "n_discordant": mc["n_discordant"],
        })

    # 1. Remove one base learner at a time.
    for k in keys:
        subset = [x for x in keys if x != k]
        if len(subset) >= 1:
            add(f"-{k}", "leave-one-out", subset, MetaLearner(seed=seed),
                note=f"stack without {k}")

    # 2. Each base learner alone, through the same meta-learner (isolates the
    #    combiner's contribution from the base learner's raw strength).
    for k in keys:
        add(f"only {k}", "single-base", [k], MetaLearner(seed=seed))

    # 3. Replace the learned combiner with the heuristics it must beat.
    add("soft vote (no meta-learner)", "combiner", keys, SoftVote())
    add("accuracy-weighted vote", "combiner", keys, WeightedVote())
    add("meta-learner on raw probabilities", "combiner", keys,
        MetaLearner(seed=seed, transform="raw"),
        note="raw probabilities, the usual default")
    add("meta-learner on log probabilities", "combiner", keys,
        MetaLearner(seed=seed, transform="log"))

    # 4. Meta-learner regularisation sweep -- shows whether C matters at all.
    for c in (0.01, 0.1, 10.0):
        add(f"meta-learner C={c}", "meta-hparam", keys, MetaLearner(seed=seed, C=c))

    # 5. The baseline nobody reports: the single best base model, unstacked.
    best_key, best_pred, best_m = None, None, None
    for k in keys:
        pred = arrays["test"][k].argmax(1)
        m = metrics_from_confusion(confusion_from_labels(y_test, pred, C))
        if best_m is None or m[metric] > best_m[metric]:
            best_key, best_pred, best_m = k, pred, m
    d = bootstrap_difference(y_test, metric=metric, B=B, seed=seed, idx=idx,
                             pred_a=best_pred, pred_b=ref_pred, n_classes=C)
    mc = mcnemar_test(y_test, best_pred, ref_pred)
    rows.append({
        "ablation": f"best single base model ({best_key})", "kind": "no-ensemble",
        "base_models": [best_key], "combiner": "none",
        **{f"metric_{k}": v for k, v in best_m.items()},
        "delta": d["difference"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
        "significant": d["excludes_zero"], "mcnemar_p": mc["p_value"],
        "n_discordant": mc["n_discordant"],
        "note": "the number a reviewer will compare the whole framework against",
    })

    n_sig = sum(bool(r.get("significant")) for r in rows)
    LOG.info("stack ablation: %d rows, %d significantly different from the full stack",
             len(rows), n_sig)
    return {
        "metric": metric, "alpha": alpha, "n_bootstrap": B,
        "reference": {"base_models": list(keys), "combiner": "gbmeta/logit", **ref_m},
        "rows": rows,
        "verdict": _stack_verdict(rows, metric),
    }


def _stack_verdict(rows, metric: str) -> str:
    ref = rows[0]
    single = [r for r in rows if r["kind"] == "no-ensemble"]
    vote = [r for r in rows if r["ablation"] == "soft vote (no meta-learner)"]
    bits = []
    if single and not single[0]["significant"]:
        bits.append(
            f"the best single base model is within the confidence interval of the full stack "
            f"(delta {single[0]['delta']:+.4f} [{single[0]['ci_low']:+.4f}, {single[0]['ci_high']:+.4f}]) "
            f"-- stacking is not measurably better here"
        )
    elif single:
        bits.append(f"stacking beats the best single model by {-single[0]['delta']:+.4f} {metric}")
    if vote and not vote[0]["significant"]:
        bits.append("a plain soft vote is statistically indistinguishable from the learned meta-learner")
    loo = [r for r in rows if r["kind"] == "leave-one-out" and r["significant"]]
    if loo:
        bits.append("base learners whose removal significantly hurts: "
                    + ", ".join(sorted({set(rows[0]['base_models']).difference(r['base_models']).pop()
                                        for r in loo})))
    else:
        bits.append("no single base learner is individually necessary")
    return "; ".join(bits)


# --------------------------------------------------------------------------
# Rerun-based ablations
# --------------------------------------------------------------------------
#: Ablations that change how base models are trained. Each entry is a
#: ``RunConfig`` override applied on top of the main configuration; the runner
#: writes them under their own tag so nothing overwrites the headline results.
RERUN_ABLATIONS = {
    "no_class_weighting": {
        "description": "sample weights disabled for every learner",
        "params": {"class_weighting": False},
    },
    "no_dedup": {
        "description": "duplicate rows kept, so train/test share exact copies",
        "config": {"dedup": "none"},
    },
    "hpo": {
        "description": "Optuna-tuned base learners instead of hand-set defaults",
        "config": {"tune": True},
    },
}


def compare_runs(run_a, run_b, models=None, metric: str = "macro_f1",
                 B: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Paired comparison of the same models across two runs (e.g. HPO on/off).

    Valid only when both runs used the same seed and dataset, so the test rows
    are identical; this is checked rather than assumed.
    """
    a, b = Path(run_a), Path(run_b)
    ya, yb = np.load(a / "y_test.npy"), np.load(b / "y_test.npy")
    if not np.array_equal(ya, yb):
        raise ValueError(
            "runs have different test sets - a paired comparison would be meaningless. "
            "Use the same dataset and seed for both."
        )
    keys = models or sorted(
        {p.stem[:-5] for p in (a / "probas").glob("*_test.npy")}
        & {p.stem[:-5] for p in (b / "probas").glob("*_test.npy")}
    )
    C = int(ya.max()) + 1
    idx = bootstrap_indices(len(ya), ya, B, seed)
    rows = []
    for k in keys:
        pa = np.load(a / "probas" / f"{k}_test.npy").argmax(1)
        pb = np.load(b / "probas" / f"{k}_test.npy").argmax(1)
        d = bootstrap_difference(ya, metric=metric, B=B, seed=seed, idx=idx,
                                 pred_a=pa, pred_b=pb, n_classes=C)
        mc = mcnemar_test(ya, pa, pb)
        rows.append({
            "model": k,
            f"{metric}_a": metrics_from_confusion(confusion_from_labels(ya, pa, C))[metric],
            f"{metric}_b": metrics_from_confusion(confusion_from_labels(ya, pb, C))[metric],
            "delta": d["difference"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "significant": d["excludes_zero"], "mcnemar_p": mc["p_value"],
        })
    return {"run_a": str(a), "run_b": str(b), "metric": metric, "rows": rows}
