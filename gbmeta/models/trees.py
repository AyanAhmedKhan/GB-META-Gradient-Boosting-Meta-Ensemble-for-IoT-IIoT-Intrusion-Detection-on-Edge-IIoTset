"""Gradient-boosted and classical baselines.

Fairness rules applied identically to every model here:

* class imbalance is handled by **per-sample weights** derived from the training
  split's balanced class weights -- never by a library-specific flag, so the
  "no class weighting" ablation is a single switch rather than three;
* early stopping uses the **validation** split, never the test split;
* every learner receives the same encoded matrix, so differences are the model,
  not the preprocessing.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ..utils import LOG, optional_import
from .base import BaseLearner, ModelContext, mark_unavailable, register_model

lgb = optional_import("lightgbm")
xgb = optional_import("xgboost")
cb = optional_import("catboost")


class _SklearnLike(BaseLearner):
    """Adapter for any estimator exposing ``fit`` / ``predict_proba``."""

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        # A chronological split can leave whole classes out of the training
        # partition -- on ToN-IoT four of ten attack types appear only later in
        # the capture. XGBoost's sklearn API rejects non-contiguous labels
        # outright, so labels are remapped to 0..K-1 here for every backend and
        # the original ids are recorded; ``_check_proba`` scatters the columns
        # back into the full class space, leaving zero probability on classes
        # the model never saw. That is the correct behaviour for an open-set
        # temporal evaluation: the model cannot predict what it was never shown.
        present = np.unique(y)
        self._fitted_classes = present.astype(int)
        self._needs_remap = len(present) != self.ctx.n_classes or present[-1] != len(present) - 1
        if self._needs_remap:
            lut = {int(c): i for i, c in enumerate(present)}
            y = np.array([lut[int(v)] for v in y], dtype=np.int32)
            if y_val is not None and X_val is not None:
                keep = np.isin(y_val, present)
                X_val = X_val[keep]
                y_val = np.array([lut[int(v)] for v in y_val[keep]], dtype=np.int32)
                if len(y_val) == 0:
                    X_val, y_val = None, None
            self._n_fit_classes = len(present)
        else:
            self._n_fit_classes = self.ctx.n_classes

        self.model = self._build()
        t0 = time.perf_counter()
        self._fit_impl(X, y, X_val, y_val, sample_weight)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _fit_impl(self, X, y, X_val, y_val, sample_weight):
        if sample_weight is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self._check_proba(self.model.predict_proba(X), len(X))

    def _picklable_payload(self):
        return self.model


# --------------------------------------------------------------------------
# Classical reference points
# --------------------------------------------------------------------------
class LogRegLearner(_SklearnLike):
    name = "logreg"

    def _build(self):
        p = self.ctx.params
        # lbfgs, not saga: on a dense 100-feature matrix saga needs several
        # thousand iterations to converge and silently returns an unconverged
        # model at the default cap, which would understate the baseline.
        return LogisticRegression(
            C=p.get("C", 1.0),
            solver="lbfgs",
            penalty="l2",
            max_iter=p.get("max_iter", 1000),
            n_jobs=-1,
            random_state=self.ctx.seed,
            tol=1e-4,
        )


class DecisionTreeLearner(_SklearnLike):
    name = "decision_tree"

    def _build(self):
        p = self.ctx.params
        return DecisionTreeClassifier(
            max_depth=p.get("max_depth", 12),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            random_state=self.ctx.seed,
        )

    def complexity(self):
        t = self.model.tree_
        return {"n_trees": 1, "n_nodes": int(t.node_count), "max_depth": int(t.max_depth)}


class RandomForestLearner(_SklearnLike):
    name = "random_forest"

    def _build(self):
        p = self.ctx.params
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=self.ctx.seed,
            bootstrap=True,
        )

    def complexity(self):
        nodes = int(sum(e.tree_.node_count for e in self.model.estimators_))
        return {"n_trees": len(self.model.estimators_), "n_nodes": nodes}


# --------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------
class LightGBMLearner(_SklearnLike):
    """CPU by default.

    LightGBM's CUDA/OpenCL builds are not shipped in the standard Colab wheel and
    the GPU histogram path has a long-standing empty-leaf-split defect on
    multiclass objectives, so the paper's CPU choice is kept -- and now stated
    as a deliberate decision rather than a workaround.
    """

    name = "lightgbm"

    def _build(self):
        p = self.ctx.params
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=getattr(self, "_n_fit_classes", self.ctx.n_classes),
            n_estimators=p.get("n_estimators", self.ctx.budget.n_estimators),
            learning_rate=p.get("learning_rate", 0.1),
            num_leaves=p.get("num_leaves", 63),
            max_bin=p.get("max_bin", 63),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.9),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0),
            n_jobs=-1,
            random_state=self.ctx.seed,
            verbose=-1,
            force_col_wise=True,
        )

    def _fit_impl(self, X, y, X_val, y_val, sample_weight):
        callbacks = []
        eval_set = None
        if X_val is not None and len(X_val):
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(self.ctx.budget.patience, verbose=False),
                lgb.log_evaluation(0),
            ]
        self.model.fit(
            X, y, sample_weight=sample_weight, eval_set=eval_set,
            eval_metric="multi_logloss", callbacks=callbacks,
        )

    def complexity(self):
        try:
            info = self.model.booster_.dump_model()["tree_info"]
            leaves = int(sum(t["num_leaves"] for t in info))
            return {
                "n_trees": len(info),
                "n_leaves": leaves,
                "n_nodes": 2 * leaves - len(info),
                "best_iteration": int(getattr(self.model, "best_iteration_", -1) or -1),
            }
        except Exception:  # pragma: no cover
            return {"n_trees": int(getattr(self.model, "n_estimators_", -1))}


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------
class XGBoostLearner(_SklearnLike):
    name = "xgboost"

    def __init__(self, ctx: ModelContext):
        super().__init__(ctx)
        self.uses_gpu = ctx.device == "cuda"

    def _build(self):
        p = self.ctx.params
        kw = dict(
            objective="multi:softprob",
            num_class=getattr(self, "_n_fit_classes", self.ctx.n_classes),
            n_estimators=p.get("n_estimators", self.ctx.budget.n_estimators),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 6),
            max_bin=p.get("max_bin", 64),
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0),
            min_child_weight=p.get("min_child_weight", 1.0),
            tree_method="hist",
            random_state=self.ctx.seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        # xgboost >= 2.0: device="cuda". Older: tree_method="gpu_hist".
        try:
            return xgb.XGBClassifier(device="cuda" if self.uses_gpu else "cpu",
                                     early_stopping_rounds=self.ctx.budget.patience, **kw)
        except TypeError:  # pragma: no cover - xgboost < 2.0
            if self.uses_gpu:
                kw["tree_method"] = "gpu_hist"
            return xgb.XGBClassifier(early_stopping_rounds=self.ctx.budget.patience, **kw)

    def _fit_impl(self, X, y, X_val, y_val, sample_weight):
        if X_val is not None and len(X_val):
            self.model.fit(X, y, sample_weight=sample_weight,
                           eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, sample_weight=sample_weight, verbose=False)

    def complexity(self):
        try:
            df = self.model.get_booster().trees_to_dataframe()
            return {
                "n_trees": int(df["Tree"].nunique()),
                "n_nodes": int(len(df)),
                "best_iteration": int(getattr(self.model, "best_iteration", -1) or -1),
            }
        except Exception:  # pragma: no cover
            return {"n_trees": int(self.model.n_estimators)}


# --------------------------------------------------------------------------
# CatBoost
# --------------------------------------------------------------------------
class CatBoostLearner(_SklearnLike):
    name = "catboost"

    def __init__(self, ctx: ModelContext):
        super().__init__(ctx)
        self.uses_gpu = ctx.device == "cuda"

    def _build(self):
        p = self.ctx.params
        return cb.CatBoostClassifier(
            loss_function="MultiClass",
            classes_count=getattr(self, "_n_fit_classes", self.ctx.n_classes),
            iterations=p.get("n_estimators", self.ctx.budget.n_estimators),
            learning_rate=p.get("learning_rate", 0.1),
            depth=p.get("depth", 6),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            border_count=p.get("border_count", 64),
            random_seed=self.ctx.seed,
            task_type="GPU" if self.uses_gpu else "CPU",
            devices="0" if self.uses_gpu else None,
            verbose=False,
            allow_writing_files=False,
        )

    def _fit_impl(self, X, y, X_val, y_val, sample_weight):
        kw = dict(sample_weight=sample_weight, verbose=False)
        if X_val is not None and len(X_val):
            kw["eval_set"] = (X_val, y_val)
            kw["early_stopping_rounds"] = self.ctx.budget.patience
            kw["use_best_model"] = True
        self.model.fit(X, y, **kw)

    def complexity(self):
        try:
            return {
                "n_trees": int(self.model.tree_count_),
                "best_iteration": int(self.model.get_best_iteration() or -1),
            }
        except Exception:  # pragma: no cover
            return {}


# --------------------------------------------------------------------------
# Registration (skipped cleanly when a backend is absent)
# --------------------------------------------------------------------------
register_model("logreg")(lambda ctx: LogRegLearner(ctx))
register_model("decision_tree")(lambda ctx: DecisionTreeLearner(ctx))
register_model("random_forest")(lambda ctx: RandomForestLearner(ctx))

if lgb is not None:
    register_model("lightgbm")(lambda ctx: LightGBMLearner(ctx))
else:  # pragma: no cover
    mark_unavailable("lightgbm", "pip install lightgbm")
    LOG.warning("lightgbm not installed - skipping")

if xgb is not None:
    register_model("xgboost")(lambda ctx: XGBoostLearner(ctx))
else:  # pragma: no cover
    mark_unavailable("xgboost", "pip install xgboost")
    LOG.warning("xgboost not installed - skipping")

if cb is not None:
    register_model("catboost")(lambda ctx: CatBoostLearner(ctx))
else:  # pragma: no cover
    mark_unavailable("catboost", "pip install catboost")
    LOG.warning("catboost not installed - skipping")
