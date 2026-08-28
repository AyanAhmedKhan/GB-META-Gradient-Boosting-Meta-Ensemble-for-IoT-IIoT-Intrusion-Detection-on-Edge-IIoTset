"""Adversarial robustness and concept drift.

The two hardest properties to measure honestly on a benchmark: how a detector
degrades under perturbed input, and how it degrades over time.

What is measured, and why each choice is defensible on a free T4:

**Perturbation sweeps** (always run, ~seconds). Accuracy against increasing
Gaussian noise, feature masking, and per-feature corruption. Cheap, deterministic,
and they produce the robustness curve that the formal definition of a robustness
curve actually calls for. They are a *lower bound* on attacker effort, not an
attack.

**HopSkipJump** (optional, ~5 s/sample). The one black-box attack that genuinely
works on gradient-boosted trees. ZOO does not: it estimates gradients by finite
differences, and a tree ensemble is piecewise constant, so the estimated gradient
is exactly zero and the attack returns the input unchanged. HopSkipJump's
``mask`` argument is what makes the result meaningful for an IDS -- it pins the
features an attacker cannot control (packet counts they do not own, server-side
timers) and perturbs only the rest.

**Drift**. Feature-level PSI and KS between train and test, a temporal
evaluation with the AUT summary from the TESSERACT protocol, and an
unseen-attack holdout. Simulated covariate/prior shift covers the four datasets
that have no usable timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluate import confusion_from_labels, metrics_from_confusion
from .utils import LOG, optional_import

EPS = 1e-12


# ==========================================================================
# Perturbation sweeps
# ==========================================================================
#: Feature families an attacker plausibly cannot alter without breaking the
#: attack itself. Matched case-insensitively against feature names; used as the
#: default immutability mask so the reported robustness is not an artefact of
#: perturbing quantities the attacker never controls.
IMMUTABLE_PATTERNS = (
    "dst_", "dest", "dload", "dbytes", "dpkts", "dttl", "dwin", "dmean", "dinpkt",
    "djit", "dloss", "response", "resp_", "bwd_", "backward",
    "dst_host", "srv_", "same_srv", "diff_srv", "rerror", "serror",
)


def immutable_mask(feature_names, patterns=IMMUTABLE_PATTERNS) -> np.ndarray:
    """Boolean mask: ``True`` where the feature is treated as attacker-immutable.

    This is a stated modelling assumption, not a fact about the network. The
    NIDS threat-model literature is explicit that no universal mutable/immutable
    feature list exists, so the mask is published with the results and can be
    swapped by a reader who disagrees.
    """
    low = [str(f).lower() for f in feature_names]
    return np.array([any(p in f for p in patterns) for f in low], dtype=bool)


@dataclass
class SweepResult:
    kind: str
    levels: list
    metrics: list  # one metrics dict per level
    mutable_features: int
    total_features: int
    notes: str = ""

    def curve(self, metric="macro_f1"):
        return list(self.levels), [m[metric] for m in self.metrics]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "levels": self.levels,
            "metrics": self.metrics, "mutable_features": self.mutable_features,
            "total_features": self.total_features, "notes": self.notes,
        }


def _score(predict_fn, X, y, C):
    pred = np.asarray(predict_fn(X)).argmax(1)
    return metrics_from_confusion(confusion_from_labels(y, pred, C))


def gaussian_noise_sweep(
    predict_fn, X, y, n_classes, epsilons=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    mutable: np.ndarray | None = None, seed: int = 0, n_rows: int = 20_000,
) -> SweepResult:
    """Accuracy vs Gaussian perturbation, applied only to mutable features.

    Epsilon is in units of the *scaled* feature space (the pipeline's robust
    scaling), so eps=0.1 is a tenth of an inter-quartile range -- a magnitude an
    attacker can plausibly produce by padding or delaying traffic.
    """
    rng = np.random.default_rng(seed)
    if len(X) > n_rows:
        idx = rng.choice(len(X), n_rows, replace=False)
        X, y = X[idx], y[idx]
    mutable = np.ones(X.shape[1], bool) if mutable is None else mutable

    metrics = []
    for eps in epsilons:
        Xa = np.array(X, dtype=np.float32, copy=True)
        if eps > 0:
            noise = rng.normal(0.0, eps, size=Xa.shape).astype(np.float32)
            noise[:, ~mutable] = 0.0
            Xa += noise
        metrics.append(_score(predict_fn, Xa, y, n_classes))
    return SweepResult("gaussian_noise", list(epsilons), metrics,
                       int(mutable.sum()), int(len(mutable)),
                       notes="epsilon in robust-scaled units; immutable features untouched")


def feature_masking_sweep(
    predict_fn, X, y, n_classes, fractions=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6),
    mutable: np.ndarray | None = None, seed: int = 0, n_rows: int = 20_000,
) -> SweepResult:
    """Accuracy when a random fraction of mutable features is zeroed per row.

    Models sensor dropout and partial telemetry loss -- the realistic edge
    failure mode -- rather than an adversary.
    """
    rng = np.random.default_rng(seed)
    if len(X) > n_rows:
        idx = rng.choice(len(X), n_rows, replace=False)
        X, y = X[idx], y[idx]
    mutable = np.ones(X.shape[1], bool) if mutable is None else mutable
    cols = np.flatnonzero(mutable)

    metrics = []
    for frac in fractions:
        Xa = np.array(X, dtype=np.float32, copy=True)
        if frac > 0 and len(cols):
            drop = rng.random((len(Xa), len(cols))) < frac
            Xa[:, cols] = np.where(drop, 0.0, Xa[:, cols])
        metrics.append(_score(predict_fn, Xa, y, n_classes))
    return SweepResult("feature_masking", list(fractions), metrics,
                       int(mutable.sum()), int(len(mutable)),
                       notes="per-row random masking of mutable features")


def robustness_summary(sweep: SweepResult, metric="macro_f1", drop_threshold=0.10) -> dict:
    """Smallest perturbation that costs more than ``drop_threshold`` of the metric."""
    lv, vals = sweep.curve(metric)
    base = vals[0]
    breaking = next((l for l, v in zip(lv, vals) if base - v > drop_threshold), None)
    return {
        "kind": sweep.kind, "metric": metric, "clean": base,
        "worst": min(vals), "relative_drop_at_max": (base - vals[-1]) / max(base, EPS),
        f"breaking_level_at_{drop_threshold}": breaking,
        "auc_of_curve": float(np.trapz(vals, lv) / max(lv[-1] - lv[0], EPS)) if len(lv) > 1 else base,
    }


# ==========================================================================
# Black-box evasion (optional)
# ==========================================================================
def hopskipjump_attack(
    model, X, y, n_samples: int = 25, mutable: np.ndarray | None = None,
    max_iter: int = 10, max_eval: int = 1000, init_eval: int = 10, seed: int = 0,
) -> dict:
    """Constrained HopSkipJump evasion against a fitted model.

    Kept deliberately small (25 samples by default): the attack costs roughly
    5 s per sample at these reduced settings and ~24 s at library defaults, so a
    full test set is out of reach in a notebook. The result is reported as an
    evasion rate on a random subsample with its confidence interval, not as a
    dataset-level robustness claim.

    Returns ``{"ran": False, ...}`` when ART is absent, so the analysis
    continues without it.
    """
    art = optional_import("art")
    if art is None:
        return {"ran": False, "reason": "adversarial-robustness-toolbox not installed "
                                        "(pip install adversarial-robustness-toolbox packaging)"}
    try:
        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import SklearnClassifier
    except Exception as exc:  # pragma: no cover
        return {"ran": False, "reason": f"ART import failed: {exc}"}

    inner = getattr(model, "model", None)
    if inner is None:
        return {"ran": False, "reason": f"{getattr(model, 'name', '?')} has no sklearn estimator "
                                        "to wrap (stacked ensembles are attacked via their bases)"}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    Xs = np.ascontiguousarray(X[idx], dtype=np.float32)
    ys = y[idx]
    correct = model.predict(Xs) == ys
    if correct.sum() == 0:
        return {"ran": False, "reason": "model misclassifies every sampled row already"}
    Xs, ys = Xs[correct], ys[correct]

    try:
        clf = SklearnClassifier(model=inner)
        attack = HopSkipJump(classifier=clf, targeted=False, max_iter=max_iter,
                             max_eval=max_eval, init_eval=init_eval, verbose=False)
        kw = {}
        if mutable is not None:
            # ART's mask is 1 where perturbation is ALLOWED.
            kw["mask"] = np.tile(mutable.astype(np.float32), (len(Xs), 1))
        Xadv = attack.generate(x=Xs, **kw)
    except Exception as exc:
        return {"ran": False, "reason": f"attack failed: {exc}"}

    adv_pred = model.predict(Xadv)
    evaded = adv_pred != ys
    delta = Xadv - Xs
    n = len(ys)
    rate = float(evaded.mean())
    # Wilson interval: with n=25 a normal-approximation interval is meaningless.
    z = 1.96
    denom = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return {
        "ran": True, "attack": "HopSkipJump (untargeted, black-box)",
        "n_attacked": int(n), "evasion_rate": rate,
        "evasion_ci95": [float(max(0.0, centre - half)), float(min(1.0, centre + half))],
        "median_l2_perturbation": float(np.median(np.linalg.norm(delta, axis=1))),
        "median_linf_perturbation": float(np.median(np.abs(delta).max(axis=1))),
        "constrained": mutable is not None,
        "mutable_features": int(mutable.sum()) if mutable is not None else int(X.shape[1]),
        "caveat": "feature-space attack: it shows the decision surface is reachable, "
                  "not that a corresponding packet sequence can be built (problem-space "
                  "realisability is not established here)",
    }


def label_flip_sensitivity(
    fit_fn, X_train, y_train, X_test, y_test, n_classes,
    rates=(0.0, 0.01, 0.05, 0.1), seed: int = 0,
) -> SweepResult:
    """Training-time poisoning: flip a fraction of training labels and refit.

    The only robustness experiment here that requires retraining, so it is run
    for one cheap model rather than all of them.
    """
    rng = np.random.default_rng(seed)
    metrics = []
    for rate in rates:
        y = np.array(y_train, copy=True)
        if rate > 0:
            k = int(rate * len(y))
            pos = rng.choice(len(y), k, replace=False)
            y[pos] = rng.integers(0, n_classes, size=k)
        model = fit_fn(X_train, y)
        metrics.append(_score(model.predict_proba, X_test, y_test, n_classes))
        LOG.info("  label-flip %.0f%%: macro-F1 %.4f", 100 * rate, metrics[-1]["macro_f1"])
    return SweepResult("label_flip_poisoning", list(rates), metrics,
                       X_train.shape[1], X_train.shape[1],
                       notes="uniform random relabelling of a fraction of training rows")


# ==========================================================================
# Concept drift
# ==========================================================================
def population_stability_index(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two samples of one feature.

    ``PSI = sum_i (c_i - r_i) * ln(c_i / r_i)`` over quantile bins of the
    reference. The familiar 0.1 / 0.25 thresholds are industry folklore with no
    statistical derivation, so they are reported as bands, not as a test.
    """
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0].astype(float)
    c = np.histogram(cur, bins=edges)[0].astype(float)
    r = np.clip(r / max(r.sum(), 1), 1e-6, None)
    c = np.clip(c / max(c.sum(), 1), 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())


def feature_drift_report(X_ref, X_cur, feature_names, top_k: int = 15) -> dict:
    """Per-feature PSI and two-sample KS between a reference and current window."""
    from scipy import stats as sps

    rows = []
    for j, name in enumerate(feature_names):
        a, b = X_ref[:, j], X_cur[:, j]
        try:
            ks = sps.ks_2samp(a, b)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except Exception:  # pragma: no cover
            ks_stat, ks_p = float("nan"), float("nan")
        rows.append({"feature": str(name), "psi": population_stability_index(a, b),
                     "ks_statistic": ks_stat, "ks_p_value": ks_p})
    rows.sort(key=lambda r: -r["psi"])
    n_major = sum(r["psi"] > 0.25 for r in rows)
    return {
        "top_drifting_features": rows[:top_k],
        "n_features": len(rows),
        "n_psi_above_0.25": n_major,
        "n_psi_above_0.10": sum(r["psi"] > 0.10 for r in rows),
        "mean_psi": float(np.mean([r["psi"] for r in rows])),
        "threshold_note": "PSI bands 0.1/0.25 are conventional, not derived; KS p-values "
                          "are near-zero for any large sample and are reported for "
                          "completeness only",
    }


def area_under_time(scores, period_label: str = "window") -> dict:
    """AUT: the time-decay-aware summary from the TESSERACT protocol.

    ``AUT = (1/(N-1)) * sum_k (f(k+1) + f(k)) / 2`` -- the trapezoidal average of
    a metric over N consecutive test periods, normalised to [0, 1]. A single
    headline number computed on a random split cannot show decay; AUT can, which
    is why it is the metric a security-ML reviewer expects for temporal
    evaluation.
    """
    s = [float(v) for v in scores]
    n = len(s)
    if n < 2:
        return {"aut": s[0] if s else float("nan"), "n_periods": n,
                "note": "AUT needs at least two periods"}
    aut = sum((s[k + 1] + s[k]) / 2 for k in range(n - 1)) / (n - 1)
    return {
        "aut": float(aut), "n_periods": n, "period": period_label,
        "first": s[0], "last": s[-1], "decay": float(s[0] - s[-1]),
        "per_period": s,
    }


def temporal_decay(
    predict_fn, X_test, y_test, timestamps, n_classes, n_windows: int = 6,
    metric: str = "macro_f1",
) -> dict:
    """Split the test set into consecutive time windows and score each.

    Requires a dataset whose loader recovered a usable timestamp -- in this
    study, ToN-IoT. Everything else uses :func:`simulated_shift`.
    """
    order = np.argsort(np.asarray(timestamps), kind="stable")
    chunks = np.array_split(order, n_windows)
    scores, rows = [], []
    for i, ch in enumerate(chunks):
        m = _score(predict_fn, X_test[ch], y_test[ch], n_classes)
        scores.append(m[metric])
        rows.append({"window": i, "n_rows": int(len(ch)), **m})
    out = area_under_time(scores, "time window")
    out.update({"metric": metric, "windows": rows})
    LOG.info("temporal decay: %s %.4f -> %.4f (AUT %.4f)",
             metric, scores[0], scores[-1], out["aut"])
    return out


def simulated_shift(
    predict_fn, X_test, y_test, n_classes, feature_names=None,
    severities=(0.0, 0.1, 0.25, 0.5, 1.0), seed: int = 0, metric: str = "macro_f1",
) -> dict:
    """Covariate shift injected as a scale-and-offset on mutable features.

    Standard dataset-shift taxonomy: this is covariate shift, ``P(x)`` changes
    while ``P(y|x)`` is assumed fixed. It is a stand-in for the real thing and is
    labelled as such in the results, never as measured drift.
    """
    rng = np.random.default_rng(seed)
    mutable = (~immutable_mask(feature_names) if feature_names is not None
               else np.ones(X_test.shape[1], bool))
    scores, rows = [], []
    for s in severities:
        Xa = np.array(X_test, dtype=np.float32, copy=True)
        if s > 0:
            scale = 1.0 + rng.normal(0, s, size=X_test.shape[1]).astype(np.float32)
            offset = rng.normal(0, s, size=X_test.shape[1]).astype(np.float32)
            scale[~mutable], offset[~mutable] = 1.0, 0.0
            Xa = Xa * scale + offset
        m = _score(predict_fn, Xa, y_test, n_classes)
        scores.append(m[metric])
        rows.append({"severity": s, **m})
    return {"kind": "simulated_covariate_shift", "metric": metric,
            "severities": list(severities), "rows": rows,
            "aut": area_under_time(scores, "severity level")["aut"],
            "note": "synthetic covariate shift; P(y|x) held fixed by construction"}


def unseen_class_holdout(y, class_names, holdout_classes) -> dict:
    """Describe an open-set split where some attack classes never appear in training.

    The realistic drift for an IDS is not a shifted feature distribution -- it is
    an attack the detector has never seen. This helper produces the index masks;
    the runner uses them to build a train/test split where the held-out classes
    appear only at test time, and the interesting metric becomes how confidently
    the model mislabels them.
    """
    names = list(map(str, class_names))
    hold = [names.index(c) for c in holdout_classes if c in names]
    missing = [c for c in holdout_classes if c not in names]
    mask = np.isin(y, hold)
    return {
        "holdout_class_indices": hold,
        "holdout_class_names": [names[i] for i in hold],
        "not_found": missing,
        "n_holdout_rows": int(mask.sum()),
        "holdout_mask": mask,
        "protocol": "train on the remaining classes only; every holdout row is by "
                    "construction a forced error, so the reported quantity is the "
                    "model's confidence when it is certainly wrong",
    }


def river_drift_detectors(scores, detector: str = "adwin", **kw) -> dict:
    """Run a streaming drift detector over a sequence of per-row correctness flags.

    ``river`` is pinned to 0.22.0 in the requirements: 0.26.x pulls a dependency
    set that breaks a Colab session. Every detector shares the same interface --
    ``update(x)`` then read ``drift_detected`` -- so swapping one is a string
    change.
    """
    river = optional_import("river")
    if river is None:
        return {"ran": False, "reason": "river not installed (pip install river==0.22.0)"}
    from river import drift as rdrift

    factory = {
        "adwin": rdrift.ADWIN, "kswin": rdrift.KSWIN,
        "page_hinkley": rdrift.PageHinkley,
        "ddm": getattr(rdrift.binary, "DDM", None) if hasattr(rdrift, "binary") else None,
        "eddm": getattr(rdrift.binary, "EDDM", None) if hasattr(rdrift, "binary") else None,
    }
    cls = factory.get(detector)
    if cls is None:
        return {"ran": False, "reason": f"detector {detector!r} unavailable in river "
                                        f"{getattr(river, '__version__', '?')}"}
    det = cls(**kw)
    points = []
    for i, x in enumerate(scores):
        det.update(float(x))
        if getattr(det, "drift_detected", False):
            points.append(i)
    return {"ran": True, "detector": detector, "river_version": getattr(river, "__version__", "?"),
            "n_points": len(scores), "drift_indices": points, "n_drifts": len(points)}
