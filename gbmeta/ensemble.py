"""Stacking done properly, plus the two heuristic combiners it is compared to.

The v1 flaw: the meta-learner was fitted on base-model probabilities for the
*validation* split, and the same validation split had already been used for
early stopping of those base models. The meta-learner therefore learned to trust
models on data they had been tuned against.

Here the meta-learner is fitted on **out-of-fold** probabilities produced inside
the training split only (Wolpert's original prescription). The validation split
keeps its single job -- early stopping -- and the test split is touched exactly
once, at the end.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_OOF_FOLDS
from .models.base import BaseLearner, ModelContext, build_model
from .utils import LOG, purge

EPS = 1e-7


def stack_features(probas, transform: str = "logit") -> np.ndarray:
    """Concatenate per-model probability matrices into the meta feature matrix.

    ``transform="logit"`` maps each probability through ``log(p/(1-p))`` before
    stacking. A linear meta-learner on raw probabilities can only form convex-ish
    mixtures; on log-odds it can express genuine re-weighting of confident and
    unconfident models, which is the whole point of learning the combination.
    Both options are exposed because the ablation compares them.
    """
    mats = [np.asarray(p, dtype=np.float64) for p in probas]
    if transform == "logit":
        mats = [np.log(np.clip(m, EPS, 1 - EPS) / (1 - np.clip(m, EPS, 1 - EPS))) for m in mats]
    elif transform == "log":
        mats = [np.log(np.clip(m, EPS, 1.0)) for m in mats]
    elif transform != "raw":
        raise ValueError(f"unknown transform {transform!r}")
    return np.hstack(mats)


# --------------------------------------------------------------------------
# Out-of-fold generation
# --------------------------------------------------------------------------
@dataclass
class OOFResult:
    model_key: str
    oof_proba: np.ndarray  # (n_train, C) - never seen by the model that produced it
    fold_seconds: list = field(default_factory=list)
    fold_val_accuracy: list = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return float(sum(self.fold_seconds))


def compute_oof(
    model_key: str,
    ctx: ModelContext,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_OOF_FOLDS,
    sample_weight: np.ndarray | None = None,
) -> OOFResult:
    """K-fold out-of-fold probabilities over the TRAINING split.

    Each fold's held-out part also serves as that fold-model's early-stopping
    set. That is a deliberate, documented compromise: it keeps the fold count
    (and therefore the T4 budget) at K rather than K x inner-K, and it cannot
    leak into the reported test metric because the test split is not involved.
    The meta-learner is still fitted only on predictions the base model never
    trained on.
    """
    n, C = len(y), ctx.n_classes
    oof = np.full((n, C), np.nan, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=ctx.seed)
    res = OOFResult(model_key=model_key, oof_proba=oof)

    for k, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.perf_counter()
        m = build_model(model_key, ctx)
        sw = sample_weight[tr] if sample_weight is not None else None
        m.fit(X[tr], y[tr], X_val=X[va], y_val=y[va], sample_weight=sw)
        p = m.predict_proba(X[va])
        oof[va] = p.astype(np.float32)
        res.fold_seconds.append(time.perf_counter() - t0)
        res.fold_val_accuracy.append(float((p.argmax(1) == y[va]).mean()))
        LOG.info("  OOF %s fold %d/%d: acc=%.4f (%.1fs)",
                 model_key, k + 1, n_folds, res.fold_val_accuracy[-1], res.fold_seconds[-1])
        del m
        purge()

    if np.isnan(oof).any():  # pragma: no cover - StratifiedKFold covers every row
        raise RuntimeError(f"{model_key}: OOF matrix has gaps")
    return res


# --------------------------------------------------------------------------
# Combiners
# --------------------------------------------------------------------------
class Combiner:
    """Common interface for the three ways of merging base-model probabilities."""

    name = "combiner"

    def fit(self, oof_probas, y_train, val_accuracies=None) -> "Combiner":
        return self

    def combine(self, probas) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"combiner": self.name}


class SoftVote(Combiner):
    """Unweighted mean of probabilities -- the null hypothesis for stacking."""

    name = "soft_vote"

    def combine(self, probas):
        return np.mean([np.asarray(p, dtype=np.float64) for p in probas], axis=0)


class WeightedVote(Combiner):
    """Weights proportional to out-of-fold accuracy (the v1 'dynamic ensemble').

    On a saturated dataset every base model scores ~0.993, so the weights are
    all ~1/M and this degenerates to :class:`SoftVote`. Reporting that
    degeneracy is more informative than reporting the weights as if they meant
    something.
    """

    name = "weighted_vote"

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.weights_ = None

    def fit(self, oof_probas, y_train, val_accuracies=None):
        if val_accuracies is None:
            val_accuracies = [float((np.asarray(p).argmax(1) == y_train).mean()) for p in oof_probas]
        a = np.asarray(val_accuracies, dtype=np.float64) ** self.temperature
        self.weights_ = a / a.sum()
        self.accuracies_ = list(map(float, val_accuracies))
        return self

    def combine(self, probas):
        w = self.weights_.reshape(-1, 1, 1)
        return (w * np.stack([np.asarray(p, dtype=np.float64) for p in probas])).sum(axis=0)

    def describe(self):
        return {
            "combiner": self.name,
            "weights": None if self.weights_ is None else [round(float(w), 6) for w in self.weights_],
            "oof_accuracies": getattr(self, "accuracies_", None),
            "weight_spread": None if self.weights_ is None else float(self.weights_.max() - self.weights_.min()),
        }


class MetaLearner(Combiner):
    """Multinomial logistic regression on stacked (log-odds) probabilities."""

    name = "gbmeta"

    def __init__(self, C: float = 1.0, transform: str = "logit", max_iter: int = 2000, seed: int = 42):
        self.C, self.transform, self.max_iter, self.seed = C, transform, max_iter, seed
        self.clf = None

    def fit(self, oof_probas, y_train, val_accuracies=None):
        Z = stack_features(oof_probas, self.transform)
        # Standardise the meta features. Log-odds of a confident tree ensemble
        # reach +/-16, and lbfgs on unscaled columns of that magnitude stops at
        # the iteration cap without converging -- which would make the ablation
        # measure optimiser failure instead of model contribution.
        self.pipe_ = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter,
                                      n_jobs=-1, random_state=self.seed)),
        ])
        self.pipe_.fit(Z, y_train)
        self.clf = self.pipe_.named_steps["lr"]
        self.n_meta_features_ = Z.shape[1]
        return self

    def combine(self, probas):
        return self.pipe_.predict_proba(stack_features(probas, self.transform))

    def coefficient_mass(self, n_models: int) -> list:
        """Share of |coefficient| mass each base model receives.

        This is the honest version of the v1 'ensemble weight' table: it says how
        much the learned combination actually leans on each model, and it is what
        shows a non-converging base model contributing nothing.
        """
        if self.clf is None:
            return []
        W = np.abs(self.clf.coef_)  # (C, n_models * C)
        per_model = np.array_split(W, n_models, axis=1)
        mass = np.array([float(b.sum()) for b in per_model])
        return list(mass / max(mass.sum(), EPS))

    def describe(self):
        return {
            "combiner": self.name,
            "transform": self.transform,
            "C": self.C,
            "n_meta_features": getattr(self, "n_meta_features_", None),
            "n_iter": None if self.clf is None else int(np.max(self.clf.n_iter_)),
        }


COMBINERS = {"soft_vote": SoftVote, "weighted_vote": WeightedVote, "gbmeta": MetaLearner}


def make_combiner(key: str, seed: int = 42, **kw) -> Combiner:
    if key not in COMBINERS:
        raise KeyError(f"unknown combiner {key!r}; available: {sorted(COMBINERS)}")
    if key == "gbmeta":
        return MetaLearner(seed=seed, **kw)
    return COMBINERS[key](**kw)


# --------------------------------------------------------------------------
# Deployable wrapper
# --------------------------------------------------------------------------
class StackedEnsemble(BaseLearner):
    """A fitted stack presented as a single model.

    Built from already-trained base learners plus an already-fitted combiner, so
    constructing it costs nothing. The deployment profiler and the robustness
    harness need a single callable that reproduces the end-to-end inference
    path, including every base model's forward pass -- that is what this is for.
    """

    name = "gbmeta"

    def __init__(self, ctx: ModelContext, bases: dict, combiner: Combiner):
        super().__init__(ctx)
        self.base_keys = list(bases)
        self.bases = bases
        self.combiner = combiner
        self.name = combiner.name
        self.uses_gpu = any(getattr(b, "uses_gpu", False) for b in bases.values())
        self.fit_seconds = float(sum(getattr(b, "fit_seconds", 0.0) or 0.0 for b in bases.values()))

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        raise RuntimeError(
            "StackedEnsemble is assembled from fitted parts; use gbmeta.runner.run_dataset"
        )

    def predict_proba(self, X):
        probas = [self.bases[k].predict_proba(X) for k in self.base_keys]
        return self._check_proba(self.combiner.combine(probas), len(X))

    def complexity(self):
        out = {"base_models": self.base_keys, **self.combiner.describe()}
        for k, b in self.bases.items():
            for kk, vv in (b.complexity() or {}).items():
                out[f"{k}.{kk}"] = vv
        return out

    def _picklable_payload(self):
        return {
            "bases": {k: b._picklable_payload() for k, b in self.bases.items()},
            "combiner": self.combiner,
        }
