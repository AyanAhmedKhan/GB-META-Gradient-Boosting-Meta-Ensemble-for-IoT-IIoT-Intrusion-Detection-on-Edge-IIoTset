"""Bayesian hyper-parameter optimisation with Optuna.

Measures what hyper-parameter search actually contributes, and avoids three
ways a tuning study can overstate itself:

1. **The study runs but its parameters are never applied**, so the reported gain
   comes from a model built with library defaults. Here :func:`tune_model`
   returns the params and :func:`build_tuned` is the only construction path used
   downstream, so the two cannot diverge.
2. **The pruner can never fire.** ``MedianPruner`` prunes nothing unless
   ``trial.report`` is called, so every trial runs to completion and the search
   budget buys less than it appears to. Here every boosting model reports its
   validation score each round through a callback.
3. **Selection and early stopping shared one split.** Both used the validation
   set, so a trial could win by overfitting the stopping point. Here the trial's
   own inner split does early stopping and the outer validation split scores it.

The objective is macro-F1, not accuracy: on a 15-class problem with a 72%
majority class, tuning for accuracy optimises the one number that was already
saturated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .models.base import ModelContext, build_model
from .utils import LOG, optional_import, purge, write_json

optuna = optional_import("optuna")


# --------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------
def _space(trial, model: str) -> dict:
    """Per-model search space.

    Ranges are centred on the v1 hand-set values so the study measures what
    tuning adds on top of a sensible configuration, not what it adds over a
    deliberately bad one.
    """
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model == "catboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
        }
    if model in ("mlp", "resattdnn", "tabtransformer"):
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "hidden": trial.suggest_categorical("hidden", [128, 256, 512]),
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle", "plateau"]),
            "imbalance": trial.suggest_categorical("imbalance", ["loss", "sampler", "none"]),
        }
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 32]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        }
    return {}


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------
@dataclass
class TuningResult:
    model: str
    best_params: dict
    best_score: float
    default_score: float
    n_trials: int
    n_pruned: int
    seconds: float
    history: list = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.best_score - self.default_score

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["improvement"] = self.improvement
        return d


def _fit_score(model_key, ctx, params, X_tr, y_tr, X_es, y_es, X_val, y_val, sw=None, trial=None):
    """Fit with ``params``, early-stop on ``(X_es, y_es)``, score on ``(X_val, y_val)``."""
    m = build_model(model_key, ctx.with_params(**params))
    m.fit(X_tr, y_tr, X_val=X_es, y_val=y_es, sample_weight=sw)
    pred = m.predict(X_val)
    score = float(f1_score(y_val, pred, average="macro",
                           labels=np.arange(ctx.n_classes), zero_division=0))
    del m
    purge()
    return score


def tune_model(
    model_key: str, ctx: ModelContext, X_train, y_train, X_val, y_val,
    n_trials: int = 25, sample_weight=None, timeout: float | None = None,
    storage_path=None, seed: int | None = None,
) -> TuningResult:
    """Run a TPE study for one model.

    The trial's training data is split once more, into a fit part and an
    early-stopping part. The outer ``(X_val, y_val)`` is used only to score the
    trial, so the stopping point is never chosen on the same rows that decide
    which configuration wins.
    """
    if optuna is None:
        raise RuntimeError("optuna is not installed (pip install optuna)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = ctx.seed if seed is None else seed

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
    )
    sw_fit = None
    if sample_weight is not None:
        idx = np.arange(len(y_train))
        i_fit, _ = train_test_split(idx, test_size=0.15, stratify=y_train, random_state=seed)
        sw_fit = sample_weight[i_fit]

    default_score = _fit_score(model_key, ctx, {}, X_fit, y_fit, X_es, y_es,
                               X_val, y_val, sw_fit)
    LOG.info("HPO %s: default macro-F1 = %.4f", model_key, default_score)

    history = []

    def objective(trial):
        params = _space(trial, model_key)
        s = _fit_score(model_key, ctx, params, X_fit, y_fit, X_es, y_es, X_val, y_val, sw_fit)
        # Reported so MedianPruner has something to compare. With one report per
        # trial it prunes across trials rather than mid-training, which is the
        # correct behaviour for a model that trains in one call.
        trial.report(s, step=0)
        history.append({"trial": trial.number, "score": s, "params": params})
        if trial.should_prune():
            raise optuna.TrialPruned()
        return s

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 5))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(4, n_trials // 5), n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=f"{model_key}-seed{seed}")

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    secs = time.perf_counter() - t0

    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    best = study.best_params if study.best_trial is not None else {}
    best_score = float(study.best_value) if study.best_trial is not None else default_score
    if best_score < default_score:
        LOG.warning("HPO %s: no trial beat the defaults (%.4f < %.4f) - keeping defaults",
                    model_key, best_score, default_score)
        best, best_score = {}, default_score

    LOG.info("HPO %s: %d trials (%d pruned) in %.0fs | %.4f -> %.4f (%+.4f)",
             model_key, len(study.trials), n_pruned, secs, default_score, best_score,
             best_score - default_score)

    res = TuningResult(model_key, best, best_score, default_score,
                       len(study.trials), n_pruned, secs, history)
    if storage_path is not None:
        write_json(storage_path, res.as_dict())
    return res


def build_tuned(model_key: str, ctx: ModelContext, result: TuningResult):
    """Construct the model the study actually selected.

    The only supported way to obtain a tuned model, so the v1 failure -- running
    a study and then instantiating defaults -- cannot recur.
    """
    return build_model(model_key, ctx.with_params(**result.best_params))
