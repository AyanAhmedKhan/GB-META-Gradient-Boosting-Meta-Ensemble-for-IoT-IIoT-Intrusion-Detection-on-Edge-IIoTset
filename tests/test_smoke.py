"""Fast correctness checks that need no downloads and no GPU.

These are the invariants that, if broken, silently corrupt results rather than
raising: a preprocessing statistic leaking across the split, an OOF matrix
containing in-sample predictions, a stratified cap dropping a rare class, a
bootstrap interval that does not contain its own point estimate.

    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------------------
def test_stratified_cap_preserves_rare_classes():
    print("\nstratified capping")
    from gbmeta.datasets.base import cap_rows
    import pandas as pd

    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(50_000, int), np.ones(1_000, int),
                        np.full(200, 2), np.full(30, 3)])
    X = pd.DataFrame(rng.normal(size=(len(y), 4)))
    Xc, yc, _, rep = cap_rows(X, y, max_rows=5_000, min_rows_per_class=200, seed=0)

    counts = np.bincount(yc, minlength=4)
    check("every class survives", (counts > 0).all(), f"counts={counts}")
    check("rare class keeps all its rows", counts[3] == 30, f"got {counts[3]}")
    check("small class reaches the floor", counts[2] >= 200, f"got {counts[2]}")
    check("budget respected", len(yc) <= 5_000, f"got {len(yc)}")
    check("rows and labels stay aligned", len(Xc) == len(yc))


def test_dedup_removes_only_exact_duplicates():
    print("\ndeduplication")
    from gbmeta.datasets.base import deduplicate
    import pandas as pd

    X = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": [0, 0, 1, 1, 1]})
    y = np.array([0, 0, 1, 0, 1], dtype=np.int32)  # rows 3/4 differ in label only
    Xd, yd, _, rep = deduplicate(X, y)
    check("one exact duplicate removed", len(yd) == 4, f"got {len(yd)}")
    check("conflicting-label pair kept", rep["conflicting_label_duplicates"] == 1,
          f"got {rep['conflicting_label_duplicates']}")


def test_preprocessing_does_not_leak():
    print("\nleakage-free preprocessing")
    from gbmeta.datasets import load_dataset
    from gbmeta.preprocess import make_splits, prepare

    ds = load_dataset("synthetic", max_rows=3_000, seed=0)
    splits = make_splits(ds.y, 0.2, 0.15, seed=0)
    data = prepare(ds.X, ds.y, ds.class_names, splits, ds.spec.categorical_cols)

    # The imputer's medians must equal the medians of the TRAINING rows only.
    num_step = data.preprocessor.named_transformers_["num"]
    cols = data.preprocessor.transformers_[0][2]
    train_median = ds.X.iloc[splits.train][cols].median().to_numpy()
    all_median = ds.X[cols].median().to_numpy()
    fitted = num_step.named_steps["impute"].statistics_

    check("imputer fitted on training rows", np.allclose(fitted, train_median, equal_nan=True))
    differs = not np.allclose(train_median, all_median, equal_nan=True)
    check("train and full-data medians actually differ (test is meaningful)", differs,
          "medians identical -- this check would pass even with a leak")

    counts = np.bincount(data.y_train, minlength=data.n_classes)
    check("class weights use training counts only",
          len(data.class_weights) == data.n_classes and np.isfinite(data.class_weights).all(),
          f"weights={data.class_weights}")
    check("no split overlap",
          len(set(splits.train) & set(splits.test)) == 0
          and len(set(splits.val) & set(splits.test)) == 0)


def test_oof_is_out_of_sample():
    print("\nout-of-fold stacking")
    from gbmeta.datasets import load_dataset
    from gbmeta.config import SMOKE_BUDGET
    from gbmeta.ensemble import compute_oof
    from gbmeta.models.base import ModelContext
    from gbmeta.preprocess import make_splits, prepare
    import gbmeta.models.trees  # noqa: F401  (registration)

    ds = load_dataset("synthetic", max_rows=2_000, min_rows_per_class=30,
                      min_class_support=20, seed=0)
    splits = make_splits(ds.y, 0.2, 0.15, seed=0)
    data = prepare(ds.X, ds.y, ds.class_names, splits, ds.spec.categorical_cols)
    ctx = ModelContext(n_classes=data.n_classes, n_features=data.n_features, seed=0,
                       budget=SMOKE_BUDGET, class_weights=data.class_weights)

    res = compute_oof("decision_tree", ctx, data.X_train, data.y_train, n_folds=3)
    oof = res.oof_proba
    check("OOF covers every training row", not np.isnan(oof).any())
    check("OOF shape is (n_train, n_classes)", oof.shape == (len(data.y_train), data.n_classes))

    # A model refit on all of the training data scores far higher in-sample than
    # the OOF predictions do. If OOF accuracy matched in-sample accuracy, the
    # folds would be leaking.
    from gbmeta.models.base import build_model
    full = build_model("decision_tree", ctx).fit(data.X_train, data.y_train)
    in_sample = float((full.predict(data.X_train) == data.y_train).mean())
    oof_acc = float((oof.argmax(1) == data.y_train).mean())
    check("OOF accuracy is below in-sample accuracy", oof_acc < in_sample - 1e-6,
          f"oof={oof_acc:.4f} in_sample={in_sample:.4f}")


def test_metrics_and_bootstrap():
    print("\nmetrics and bootstrap")
    from gbmeta.evaluate import (bootstrap_difference, bootstrap_metric,
                                 compute_metrics, confusion_from_labels,
                                 metrics_from_confusion)
    from sklearn.metrics import f1_score, accuracy_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 3_000)
    pred = np.where(rng.random(3_000) < 0.85, y, rng.integers(0, 5, 3_000))

    fast = metrics_from_confusion(confusion_from_labels(y, pred, 5))
    check("fast accuracy matches sklearn", abs(fast["accuracy"] - accuracy_score(y, pred)) < 1e-9)
    check("fast macro-F1 matches sklearn",
          abs(fast["macro_f1"] - f1_score(y, pred, average="macro", zero_division=0)) < 1e-9)

    proba = np.eye(5)[pred] * 0.7 + 0.06
    m = compute_metrics(y, proba / proba.sum(1, keepdims=True), 5)
    check("compute_metrics agrees with the fast path",
          abs(m["macro_f1"] - fast["macro_f1"]) < 1e-9)
    check("ECE is in [0, 1]", 0.0 <= m["ece"] <= 1.0, f"ece={m['ece']}")

    ci = bootstrap_metric(y, y_pred=pred, metric="macro_f1", B=200, n_classes=5)
    check("CI contains its point estimate", ci.lo <= ci.point <= ci.hi,
          f"{ci.lo:.4f} <= {ci.point:.4f} <= {ci.hi:.4f}")
    check("CI has non-zero width", ci.hi > ci.lo)

    d = bootstrap_difference(y, pred_a=pred, pred_b=pred, metric="macro_f1", B=200, n_classes=5)
    check("identical models give a zero difference", abs(d["difference"]) < 1e-12)
    check("identical models are not significant", not d["excludes_zero"])


def test_statistics():
    print("\nstatistical tests")
    from gbmeta.stats import (RankMatrix, corrected_paired_t, friedman_test,
                              holm_correction, mcnemar_test,
                              nemenyi_critical_difference)

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2_000)
    good = np.where(rng.random(2_000) < 0.95, y, rng.integers(0, 4, 2_000))
    bad = np.where(rng.random(2_000) < 0.80, y, rng.integers(0, 4, 2_000))

    r = mcnemar_test(y, good, bad)
    check("McNemar detects a real gap", r["p_value"] < 0.01, f"p={r['p_value']}")
    check("McNemar favours the better model", r["favours"] == "a", r["favours"])
    same = mcnemar_test(y, good, good)
    check("identical predictions give p=1", same["p_value"] == 1.0)

    h = holm_correction([0.001, 0.02, 0.04, 0.9])
    check("Holm is monotone and >= raw",
          all(a >= b for a, b in zip(h["p_adjusted"], [0.001, 0.02, 0.04, 0.9])))

    # Demsar's published CD for k=4, N=5 at alpha=0.05 is 2.098.
    cd = nemenyi_critical_difference(4, 5, 0.05)
    check("Nemenyi CD matches the published value", abs(cd - 2.098) < 0.01, f"cd={cd:.4f}")

    S = np.array([[.95, .94, .90, .80], [.92, .93, .88, .79], [.97, .96, .93, .85],
                  [.91, .90, .87, .77], [.94, .95, .89, .81]])
    rm = RankMatrix(S, ["a", "b", "c", "d"], [f"d{i}" for i in range(5)])
    f = friedman_test(rm)
    check("Friedman detects a rank difference", f["significant"], str(f["iman_davenport_p_value"]))

    nb = corrected_paired_t([.91, .92, .90, .93, .91], [.89, .90, .88, .91, .90],
                            n_train=8_000, n_test=2_000)
    naive_would_be_smaller = nb["variance_correction"] > 1.0 / 5
    check("Nadeau-Bengio inflates the variance vs the naive test", naive_would_be_smaller,
          f"correction={nb['variance_correction']}")


def test_end_to_end_run():
    print("\nend-to-end run (synthetic)")
    from gbmeta.config import RunConfig, SMOKE_BUDGET
    from gbmeta.runner import run_dataset

    cfg = RunConfig(dataset="synthetic", seed=7, budget=SMOKE_BUDGET, device="cpu",
                    models=("logreg", "decision_tree", "random_forest", "lightgbm",
                            "soft_vote", "weighted_vote", "gbmeta"),
                    stack_bases=("lightgbm", "random_forest", "decision_tree"),
                    tag="pytest")
    out = run_dataset(cfg, force=True)
    recs = out["records"]
    check("every requested model produced a record", len(recs) >= 6, f"got {sorted(recs)}")
    check("gbmeta was built", "gbmeta" in recs)
    for name, rec in recs.items():
        m = rec["metrics"]
        check(f"{name}: metrics are finite and in range",
              0 <= m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1,
              f"acc={m['accuracy']} f1={m['macro_f1']}")

    for f in ("manifest.json", "leakage.json", "y_test.npy", "classes.json"):
        check(f"artefact written: {f}", (cfg.run_dir / f).exists())

    from gbmeta.ablation import ablate_stack
    ab = ablate_stack(cfg.run_dir, B=100)
    check("ablation produced rows", len(ab["rows"]) > 5, f"got {len(ab['rows'])}")
    check("reference row has zero delta", ab["rows"][0]["delta"] == 0.0)


def main() -> int:
    for fn in (test_stratified_cap_preserves_rare_classes,
               test_dedup_removes_only_exact_duplicates,
               test_preprocessing_does_not_leak,
               test_oof_is_out_of_sample,
               test_metrics_and_bootstrap,
               test_statistics,
               test_end_to_end_run):
        try:
            fn()
        except Exception as exc:
            import traceback
            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  [ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
