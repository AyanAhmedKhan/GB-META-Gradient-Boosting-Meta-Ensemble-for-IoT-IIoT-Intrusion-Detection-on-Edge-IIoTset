"""Central configuration: seeds, budgets, paths, and the T4 resource envelope.

Every number that affects a reported result lives here so the paper's
"Experimental Setup" section can be generated from code rather than memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(os.environ.get("GBMETA_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("GBMETA_DATA", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("GBMETA_CACHE", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("GBMETA_RESULTS", ROOT / "results"))
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
#: Master seed. Single-run artifacts (figures, confusion matrices) use this.
SEED = 42

#: Seeds for repeated runs. Statistical tests that compare models across
#: independent repetitions consume this list. Five is the minimum that makes
#: a Wilcoxon signed-rank test meaningful; ten is better if the budget allows.
REPEAT_SEEDS: Sequence[int] = (42, 43, 44, 45, 46)

#: Folds used to generate out-of-fold predictions for the meta-learner.
#: The meta-learner is fitted on OOF predictions of the TRAINING set only --
#: never on validation or test predictions (that was the v1 flaw).
N_OOF_FOLDS = 5

#: Bootstrap resamples for confidence intervals on test-set metrics.
N_BOOTSTRAP = 2000

#: Two-sided significance level used throughout.
ALPHA = 0.05


# --------------------------------------------------------------------------
# Resource envelope
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Budget:
    """Hard caps chosen so the whole study fits a free Colab T4 session.

    ``max_rows`` is applied by *stratified* subsampling with a per-class floor,
    so rare attack classes survive the cut. See :func:`gbmeta.datasets.base.cap_rows`.
    """

    #: Rows kept per dataset after stratified subsampling.
    max_rows: int = 200_000
    #: Never drop a class below this many rows (unless it has fewer to begin with).
    min_rows_per_class: int = 200
    #: Classes rarer than this in the raw data are merged into ``__rare__``.
    #: Set to 0 to keep every class (Edge-IIoTset MITM has only 358 rows).
    min_class_support: int = 30
    #: Boosting rounds before early stopping.
    n_estimators: int = 600
    #: Early-stopping patience, shared by tree and neural models.
    patience: int = 30
    #: Folds used to build out-of-fold predictions for the meta-learner.
    n_oof_folds: int = N_OOF_FOLDS
    #: Optuna trials per tuned model.
    n_trials: int = 25
    #: Neural batch size (2048 fits T4 comfortably for a 100-feature MLP).
    batch_size: int = 2048
    #: Max epochs for neural baselines.
    max_epochs: int = 40
    #: Fraction of the (deduplicated) data held out as the untouched test set.
    test_size: float = 0.20
    #: Fraction of the remaining data used for early stopping / model selection.
    val_size: float = 0.15
    #: Exponent applied to balanced class weights: ``w_c = (N / (C * n_c)) ** p``.
    #: ``p=1`` is textbook "balanced"; on a 2000:1 imbalance that weights the
    #: rarest class 2000x and makes neural training diverge while leaving tree
    #: models unaffected. ``p=0.5`` (the default) is the standard tempered
    #: compromise, and the ablation reports both.
    class_weight_power: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


BUDGET = Budget()

#: A deliberately tiny budget for smoke tests and CI. ``python -m gbmeta.runner
#: --smoke`` uses this and completes on a CPU laptop in under a minute.
SMOKE_BUDGET = Budget(
    max_rows=4_000,
    min_rows_per_class=20,
    min_class_support=10,
    n_estimators=40,
    patience=5,
    n_oof_folds=3,
    n_trials=3,
    batch_size=256,
    max_epochs=4,
)


# --------------------------------------------------------------------------
# Model roster
# --------------------------------------------------------------------------
#: Base learners that feed the GB-META stack.
STACK_BASE_MODELS = ("lightgbm", "xgboost", "catboost")

#: Non-stacked reference points. ``logreg`` and ``random_forest`` anchor the
#: low end so a reviewer can see what the *easy* baseline already achieves --
#: on a saturated dataset that number is the real story.
CLASSICAL_BASELINES = ("logreg", "random_forest", "decision_tree")

#: Tabular deep-learning baselines. These are the models the v1 paper found
#: fail to converge; the failure is reproduced here under controlled conditions.
DEEP_BASELINES = ("mlp", "resattdnn", "tabtransformer")

#: Ensembles built from cached base-learner probabilities (no retraining).
ENSEMBLES = ("soft_vote", "weighted_vote", "gbmeta")

DEFAULT_MODELS = (
    CLASSICAL_BASELINES + STACK_BASE_MODELS + DEEP_BASELINES + ENSEMBLES
)


@dataclass
class RunConfig:
    """One fully-specified experiment: dataset x model-set x seed."""

    dataset: str
    seed: int = SEED
    models: Sequence[str] = field(default_factory=lambda: DEFAULT_MODELS)
    budget: Budget = BUDGET
    tune: bool = False
    #: Base learners feeding the stack. Falls back to whatever is installed, so
    #: a machine without CatBoost still produces a (smaller, clearly labelled)
    #: ensemble rather than no ensemble at all.
    stack_bases: Sequence[str] = STACK_BASE_MODELS
    dedup: str = "global"  # {"global", "none"} -- see preprocess.deduplicate
    device: str = "auto"  # {"auto", "cuda", "cpu"}
    tag: str = "main"

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.tag / self.dataset / f"seed{self.seed}"
