"""Uniform learner interface and registry.

Every model in the study -- boosted trees, neural nets, classical baselines and
the stacked ensemble -- exposes the same five methods, so the runner, the
ablation study, the deployment profiler and the robustness harness can all treat
them interchangeably. Anything model-specific lives behind this interface.
"""
from __future__ import annotations

import abc
import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Budget, BUDGET
from ..utils import LOG


@dataclass
class ModelContext:
    """Everything a learner factory needs, resolved once per run."""

    n_classes: int
    n_features: int
    seed: int = 42
    device: str = "cpu"
    budget: Budget = BUDGET
    #: Balanced class weights computed on the training split only.
    class_weights: np.ndarray | None = None
    feature_names: list = field(default_factory=list)
    #: Free-form overrides from HPO or ablation flags.
    params: dict = field(default_factory=dict)

    def with_params(self, **kw) -> "ModelContext":
        merged = dict(self.params)
        merged.update(kw)
        return ModelContext(
            n_classes=self.n_classes, n_features=self.n_features, seed=self.seed,
            device=self.device, budget=self.budget, class_weights=self.class_weights,
            feature_names=list(self.feature_names), params=merged,
        )


class BaseLearner(abc.ABC):
    """Minimal contract shared by every model."""

    name: str = "base"
    #: Set by concrete classes; the runner reports which models ran on GPU.
    uses_gpu: bool = False
    #: Populated by :meth:`fit`.
    fit_seconds: float = float("nan")

    def __init__(self, ctx: ModelContext) -> None:
        self.ctx = ctx
        self.classes_ = np.arange(ctx.n_classes)

    # -- required ---------------------------------------------------------
    @abc.abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> "BaseLearner":
        ...

    @abc.abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        ...

    # -- provided ---------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        return np.asarray(self.predict_proba(X)).argmax(axis=1)

    def complexity(self) -> dict:
        """Model-size descriptors for the deployment table."""
        return {}

    def save(self, path) -> Path:
        """Default: gzipped pickle. Tree backends override with native formats."""
        path = Path(path).with_suffix(".pkl.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def size_bytes(self, tmp_dir=None) -> dict:
        """Serialised size, raw and gzip-compressed."""
        import io
        buf = io.BytesIO()
        try:
            pickle.dump(self._picklable_payload(), buf, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # pragma: no cover
            LOG.warning("%s: not picklable (%s)", self.name, exc)
            return {"raw_bytes": None, "gzip_bytes": None}
        raw = buf.getvalue()
        return {"raw_bytes": len(raw), "gzip_bytes": len(gzip.compress(raw, 6))}

    def _picklable_payload(self):
        return self

    def _check_proba(self, p: np.ndarray, n_rows: int) -> np.ndarray:
        """Guarantee a dense (n, n_classes) row-stochastic float64 matrix.

        Boosted-tree libraries silently return an (n,) vector for binary
        problems and can drop columns for classes absent from a CV fold. Both
        would corrupt the stacked feature matrix in ways that are very hard to
        see later, so they are normalised here, once.
        """
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        C = self.ctx.n_classes
        if p.shape[1] != C:
            full = np.zeros((p.shape[0], C), dtype=np.float64)
            seen = getattr(self, "_fitted_classes", np.arange(p.shape[1]))
            full[:, np.asarray(seen, dtype=int)] = p
            p = full
        if p.shape[0] != n_rows:
            raise ValueError(f"{self.name}: expected {n_rows} rows of probabilities, got {p.shape[0]}")
        p = np.clip(p, 1e-12, 1.0)
        return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
Factory = Callable[[ModelContext], BaseLearner]
_MODELS: dict[str, Factory] = {}
_UNAVAILABLE: dict[str, str] = {}


def register_model(key: str) -> Callable[[Factory], Factory]:
    def deco(fn: Factory) -> Factory:
        _MODELS[key] = fn
        return fn
    return deco


def mark_unavailable(key: str, reason: str) -> None:
    """Record a model that cannot run here, so the manifest explains the gap."""
    _UNAVAILABLE[key] = reason


def available_models() -> list:
    return sorted(_MODELS)


def unavailable_models() -> dict:
    return dict(_UNAVAILABLE)


def build_model(key: str, ctx: ModelContext) -> BaseLearner:
    if key not in _MODELS:
        reason = _UNAVAILABLE.get(key, "not registered")
        raise KeyError(f"model {key!r} unavailable: {reason}. Available: {available_models()}")
    return _MODELS[key](ctx)
