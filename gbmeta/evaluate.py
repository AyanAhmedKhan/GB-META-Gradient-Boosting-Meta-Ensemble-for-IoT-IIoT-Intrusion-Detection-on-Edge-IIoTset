"""Metrics, calibration, and bootstrap confidence intervals.

Every headline number carries a percentile bootstrap interval computed on the
*same* resample indices across models, so the paired difference between two
models has an interval too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .config import ALPHA, N_BOOTSTRAP
from .utils import LOG

#: Metrics that are computed from hard predictions alone.
LABEL_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                 "macro_precision", "macro_recall", "mcc", "kappa")
#: Metrics that need probabilities.
PROBA_METRICS = ("roc_auc_ovr_macro", "pr_auc_macro", "log_loss", "brier", "ece")


def _safe(fn, default=float("nan"), *a, **kw):
    try:
        return float(fn(*a, **kw))
    except Exception as exc:  # pragma: no cover - degenerate label sets
        LOG.debug("metric failed (%s): %s", getattr(fn, "__name__", fn), exc)
        return default


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """Confidence-binned ECE (Guo et al., ICML 2017).

    A 99%-accurate detector that is systematically over-confident on its 1% of
    errors is a different operational object from one that is not; ECE is what
    separates them, and it is missing from essentially every IDS paper.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, proba) -> float:
    """Mean squared error between the one-hot target and the predicted vector."""
    oh = np.zeros_like(proba)
    oh[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def compute_metrics(y_true, proba, n_classes: int | None = None) -> dict:
    """All headline metrics for one model on one test set."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    y_pred = proba.argmax(axis=1)
    C = n_classes or proba.shape[1]
    labels = np.arange(C)
    present = np.unique(y_true)

    out = {
        "accuracy": _safe(accuracy_score, float("nan"), y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, float("nan"), y_true, y_pred),
        "macro_f1": _safe(f1_score, float("nan"), y_true, y_pred,
                          average="macro", labels=labels, zero_division=0),
        "weighted_f1": _safe(f1_score, float("nan"), y_true, y_pred,
                             average="weighted", labels=labels, zero_division=0),
        "mcc": _safe(matthews_corrcoef, float("nan"), y_true, y_pred),
        "kappa": _safe(cohen_kappa_score, float("nan"), y_true, y_pred),
    }
    p, r, _f, _s = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    out["macro_precision"], out["macro_recall"] = float(p), float(r)

    # Probability-based metrics: restrict to classes actually present, otherwise
    # roc_auc_score raises and the whole row is lost.
    if len(present) > 1:
        sub = proba[:, present]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        remap = {c: i for i, c in enumerate(present)}
        y_sub = np.array([remap[v] for v in y_true])
        oh = np.eye(len(present))[y_sub]
        out["roc_auc_ovr_macro"] = _safe(roc_auc_score, float("nan"), y_sub, sub,
                                         multi_class="ovr", average="macro")
        out["pr_auc_macro"] = _safe(average_precision_score, float("nan"), oh, sub, average="macro")
        out["log_loss"] = _safe(log_loss, float("nan"), y_sub, sub, labels=list(range(len(present))))
    else:  # pragma: no cover
        out.update({"roc_auc_ovr_macro": float("nan"), "pr_auc_macro": float("nan"),
                    "log_loss": float("nan")})

    out["brier"] = multiclass_brier(y_true, proba)
    out["ece"] = expected_calibration_error(y_true, proba)
    return out


def per_class_report(y_true, y_pred, class_names) -> list:
    labels = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "class": str(class_names[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
            #: Flagged when the class is too small for its F1 to mean anything.
            "underpowered": bool(s[i] < 30),
        }
        for i in labels
    ]


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_indices(n: int, y: np.ndarray | None, B: int, seed: int,
                      stratified: bool = True) -> np.ndarray:
    """Resample indices once, shared by every model.

    Stratified resampling keeps each class's test support fixed across replicas.
    Without it, a 19-row class disappears from ~15% of replicas and the macro-F1
    interval becomes a statement about resampling noise in the class list rather
    than about the classifier.
    """
    rng = np.random.default_rng(seed)
    if not stratified or y is None:
        return rng.integers(0, n, size=(B, n))
    parts = [np.flatnonzero(y == c) for c in np.unique(y)]
    cols = [rng.choice(p, size=(B, len(p)), replace=True) for p in parts]
    return np.hstack(cols)


def confusion_from_labels(y_true, y_pred, C: int) -> np.ndarray:
    """O(n) confusion matrix via bincount -- the bootstrap inner loop."""
    return np.bincount(np.asarray(y_true) * C + np.asarray(y_pred),
                       minlength=C * C).reshape(C, C).astype(np.float64)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Label-based metrics derived from a confusion matrix alone.

    2000 bootstrap replicas x ``compute_metrics`` would spend minutes inside
    sklearn's validation code and recompute ROC curves nobody asked for. Every
    metric below is a closed-form function of the confusion matrix, so a replica
    costs one ``bincount``.
    """
    n = cm.sum()
    tp = np.diag(cm)
    pred_sum, true_sum = cm.sum(axis=0), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_sum > 0, tp / np.maximum(pred_sum, 1), 0.0)
        rec = np.where(true_sum > 0, tp / np.maximum(true_sum, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = true_sum > 0
    acc = tp.sum() / max(n, 1)

    po, pe = acc, float((true_sum * pred_sum).sum() / max(n * n, 1))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    cov = float((tp.sum() * n) - (true_sum * pred_sum).sum())
    denom = np.sqrt(max(n * n - (pred_sum ** 2).sum(), 0)) * np.sqrt(max(n * n - (true_sum ** 2).sum(), 0))
    mcc = cov / denom if denom > 0 else 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(rec[present].mean()) if present.any() else float("nan"),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * true_sum).sum() / max(n, 1)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "kappa": float(kappa),
        "mcc": float(mcc),
    }


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    alpha: float = ALPHA

    def as_tuple(self):
        return (self.point, self.lo, self.hi)

    def fmt(self, pct: bool = True, digits: int = 3) -> str:
        s = 100.0 if pct else 1.0
        u = "%" if pct else ""
        return f"{self.point*s:.{digits}f}{u} [{self.lo*s:.{digits}f}, {self.hi*s:.{digits}f}]"


def bootstrap_metric(
    y_true, proba=None, metric: str = "macro_f1", B: int = N_BOOTSTRAP,
    seed: int = 0, alpha: float = ALPHA, idx: np.ndarray | None = None,
    y_pred=None, n_classes: int | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI for one label-based metric.

    Accepts either probabilities or hard predictions; only the arg-max matters,
    so cached predictions are enough and the whole table can be rebuilt without
    retraining anything.
    """
    y_true = np.asarray(y_true)
    if y_pred is None:
        y_pred = np.asarray(proba).argmax(axis=1)
    y_pred = np.asarray(y_pred)
    C = n_classes or int(max(y_true.max(), y_pred.max()) + 1)
    if metric not in LABEL_METRICS:
        raise ValueError(
            f"{metric!r} needs probabilities; bootstrap is restricted to {LABEL_METRICS}"
        )
    if idx is None:
        idx = bootstrap_indices(len(y_true), y_true, B, seed)

    point = metrics_from_confusion(confusion_from_labels(y_true, y_pred, C))[metric]
    vals = np.empty(len(idx))
    for b, ii in enumerate(idx):
        vals[b] = metrics_from_confusion(confusion_from_labels(y_true[ii], y_pred[ii], C))[metric]
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(point), float(lo), float(hi), alpha)


def bootstrap_difference(
    y_true, proba_a=None, proba_b=None, metric: str = "macro_f1", B: int = N_BOOTSTRAP,
    seed: int = 0, alpha: float = ALPHA, idx: np.ndarray | None = None,
    pred_a=None, pred_b=None, n_classes: int | None = None,
) -> dict:
    """Paired bootstrap CI for ``metric(a) - metric(b)`` on shared resamples.

    Because both models are scored on the *same* resampled rows, the resampling
    noise common to them cancels. An interval excluding zero is evidence of a
    real difference; on a saturated dataset it usually does not, and saying so
    plainly is the honest result.
    """
    y_true = np.asarray(y_true)
    if pred_a is None:
        pred_a = np.asarray(proba_a).argmax(axis=1)
    if pred_b is None:
        pred_b = np.asarray(proba_b).argmax(axis=1)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    C = n_classes or int(max(y_true.max(), pred_a.max(), pred_b.max()) + 1)
    if idx is None:
        idx = bootstrap_indices(len(y_true), y_true, B, seed)

    d = np.empty(len(idx))
    for b, ii in enumerate(idx):
        yt = y_true[ii]
        ma = metrics_from_confusion(confusion_from_labels(yt, pred_a[ii], C))[metric]
        mb = metrics_from_confusion(confusion_from_labels(yt, pred_b[ii], C))[metric]
        d[b] = ma - mb
    point = (metrics_from_confusion(confusion_from_labels(y_true, pred_a, C))[metric]
             - metrics_from_confusion(confusion_from_labels(y_true, pred_b, C))[metric])
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "difference": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_bootstrap": int(len(idx)),
    }


# --------------------------------------------------------------------------
# Curve data (kept as arrays so plotting stays a separate, cheap step)
# --------------------------------------------------------------------------
@dataclass
class CurveData:
    roc: dict = field(default_factory=dict)
    pr: dict = field(default_factory=dict)
    reliability: dict = field(default_factory=dict)


def curve_data(y_true, proba, class_names, max_points: int = 512) -> CurveData:
    """One-vs-rest ROC and PR curves per class, plus a reliability diagram."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    cd = CurveData()

    def _thin(*arrs):
        n = len(arrs[0])
        if n <= max_points:
            return [a.tolist() for a in arrs]
        take = np.linspace(0, n - 1, max_points).astype(int)
        return [a[take].tolist() for a in arrs]

    for i, name in enumerate(class_names):
        pos = (y_true == i).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        fpr, tpr, _ = roc_curve(pos, proba[:, i])
        pr_p, pr_r, _ = precision_recall_curve(pos, proba[:, i])
        cd.roc[str(name)] = dict(zip(("fpr", "tpr"), _thin(fpr, tpr)))
        cd.roc[str(name)]["auc"] = float(_safe(roc_auc_score, float("nan"), pos, proba[:, i]))
        cd.pr[str(name)] = dict(zip(("precision", "recall"), _thin(pr_p, pr_r)))
        cd.pr[str(name)]["ap"] = float(_safe(average_precision_score, float("nan"), pos, proba[:, i]))

    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, 16)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin_lower": float(lo), "bin_upper": float(hi), "count": int(m.sum()),
            "mean_confidence": float(conf[m].mean()),
            "empirical_accuracy": float((pred[m] == y_true[m]).mean()),
        })
    cd.reliability = {"bins": bins, "ece": expected_calibration_error(y_true, proba)}
    return cd


def confusion(y_true, y_pred, n_classes: int, normalise: str | None = None) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    if normalise == "true":
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    return cm
