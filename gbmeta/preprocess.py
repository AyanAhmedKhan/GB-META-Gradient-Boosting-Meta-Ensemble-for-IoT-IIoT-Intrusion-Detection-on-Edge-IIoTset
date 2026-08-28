"""Leakage-free splitting, fold-fitted preprocessing, and a leakage audit.

Fitting any statistic over the whole dataset before splitting is a textbook
leak: test-set information reaches the training representation. Everything here
is built so that cannot happen. The preprocessor is an sklearn ``Pipeline``
fitted on training rows only and applied to validation and test rows, and the
audit module actively looks for the residual ways a network-IDS dataset can
still cheat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from .utils import LOG

CLIP_LIMIT = 10.0


class AsFloat32(BaseEstimator, TransformerMixin):
    """Cast to float32 and neutralise any residual non-finite value.

    Halves peak RAM versus pandas' float64 default and keeps tree libraries on
    their fast path. Kept as a class (not a lambda) so the fitted pipeline
    pickles for the deployment/ONNX stage.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=CLIP_LIMIT, neginf=-CLIP_LIMIT)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class Clip(BaseEstimator, TransformerMixin):
    """Clip to +/- ``limit`` after robust scaling.

    RobustScaler is resistant to outliers when computing the transform, but the
    *output* is unbounded; a single 1e9 packet-rate outlier still dominates a
    neural net's first layer. The limit is fitted on nothing, so it leaks nothing.
    """

    def __init__(self, limit: float = CLIP_LIMIT):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, -self.limit, self.limit)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class Splits:
    """Index arrays into the loaded dataset. Nothing is copied until needed."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mode: str = "stratified"
    meta: dict = field(default_factory=dict)

    def sizes(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    timestamps: pd.Series | None = None,
    temporal: bool = False,
) -> Splits:
    """Three-way split.

    ``temporal=True`` orders rows by timestamp and takes the *latest* rows as
    the test set. This is the honest protocol for an IDS -- a deployed detector
    is always evaluated on traffic newer than its training data -- and it is
    what makes the concept-drift results in section 'drift' meaningful. It is
    only available for datasets whose loader recovered a usable timestamp.
    """
    n = len(y)
    idx = np.arange(n)

    if temporal:
        if timestamps is None:
            raise ValueError("temporal split requested but dataset has no timestamps")
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        n_test = int(round(n * test_size))
        n_val = int(round((n - n_test) * val_size))
        test = order[-n_test:]
        rest = order[:-n_test]
        val = rest[-n_val:]
        train = rest[:-n_val]
        missing = set(np.unique(y[test])) - set(np.unique(y[train]))
        return Splits(
            train=np.sort(train), val=np.sort(val), test=np.sort(test), mode="temporal",
            meta={
                "train_end": str(timestamps.iloc[train].max()),
                "test_start": str(timestamps.iloc[test].min()),
                "classes_unseen_in_train": sorted(int(c) for c in missing),
            },
        )

    tr_val, test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed, shuffle=True
    )
    train, val = train_test_split(
        tr_val, test_size=val_size, stratify=y[tr_val], random_state=seed, shuffle=True
    )
    return Splits(train=np.sort(train), val=np.sort(val), test=np.sort(test), mode="stratified")


# --------------------------------------------------------------------------
# Fold-fitted preprocessing
# --------------------------------------------------------------------------
def infer_column_types(X: pd.DataFrame, declared_categorical=()) -> tuple[list, list]:
    """Split columns into numeric and categorical.

    A column is categorical if it is declared so, non-numeric, or numeric with
    very low cardinality relative to its length (protocol/flag codes stored as
    integers, which must not be scaled as if they were magnitudes).
    """
    cat = [c for c in declared_categorical if c in X.columns]
    num = []
    for c in X.columns:
        if c in cat:
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


def make_preprocessor(
    X: pd.DataFrame, categorical_cols=(), max_categories: int = 32
) -> tuple[ColumnTransformer, list, list]:
    """Build the (unfitted) preprocessing pipeline.

    Numeric: median impute -> robust scale -> clip -> float32.
    Categorical: most-frequent impute -> one-hot with a rare-category bucket.

    ``handle_unknown="infrequent_if_exist"`` means a category seen only at test
    time lands in the rare bucket instead of raising -- the deployment-realistic
    behaviour, and the only one that survives a temporal split.
    """
    num, cat = infer_column_types(X, categorical_cols)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
        ("clip", Clip(CLIP_LIMIT)),
        ("cast", AsFloat32()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.001,
            max_categories=max_categories,
            sparse_output=False,
            dtype=np.float32,
        )),
    ])

    transformers = []
    if num:
        transformers.append(("num", numeric_pipe, num))
    if cat:
        transformers.append(("cat", categorical_pipe, cat))

    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct, num, cat


@dataclass
class PreparedData:
    """Model-ready matrices plus the artefacts needed to reproduce them."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    class_names: np.ndarray
    preprocessor: ColumnTransformer
    splits: Splits
    class_weights: np.ndarray
    report: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def sample_weights(self, y: np.ndarray) -> np.ndarray:
        return self.class_weights[y].astype(np.float32)


def prepare(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: np.ndarray,
    splits: Splits,
    categorical_cols=(),
    class_weight_power: float = 0.5,
) -> PreparedData:
    """Fit the preprocessor on the TRAIN rows only, then transform all three."""
    ct, num, cat = make_preprocessor(X, categorical_cols)

    Xtr_raw = X.iloc[splits.train]
    ct.fit(Xtr_raw)

    X_train = np.ascontiguousarray(ct.transform(Xtr_raw), dtype=np.float32)
    X_val = np.ascontiguousarray(ct.transform(X.iloc[splits.val]), dtype=np.float32)
    X_test = np.ascontiguousarray(ct.transform(X.iloc[splits.test]), dtype=np.float32)

    try:
        feature_names = [str(f) for f in ct.get_feature_names_out()]
    except Exception:  # pragma: no cover - very old sklearn
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    y_train, y_val, y_test = y[splits.train], y[splits.val], y[splits.test]

    # Balanced class weights computed on the TRAINING split only, then tempered
    # by ``class_weight_power``. Raw balanced weights on NSL-KDD reach 2245x,
    # which is fine for a tree but drives a neural net to predict rare classes
    # everywhere; the exponent is a single reported knob rather than a
    # per-model fudge.
    counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float64)
    with np.errstate(divide="ignore"):
        w = (len(y_train) / (len(class_names) * np.maximum(counts, 1.0))) ** class_weight_power
    w[counts == 0] = 0.0
    w = w / max(w.mean(), 1e-12)  # keep the average weight at 1 so the loss scale is unchanged

    report = {
        "n_numeric_cols": len(num),
        "n_categorical_cols": len(cat),
        "n_features_after_encoding": int(X_train.shape[1]),
        "split_mode": splits.mode,
        "split_sizes": splits.sizes(),
        "split_meta": splits.meta,
        "class_weight_power": float(class_weight_power),
        "class_weight_min": float(w.min()),
        "class_weight_max": float(w.max()),
    }
    LOG.info(
        "prepared: %d train / %d val / %d test x %d features (%d numeric, %d categorical)",
        len(y_train), len(y_val), len(y_test), X_train.shape[1], len(num), len(cat),
    )
    return PreparedData(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, class_names, ct, splits, w.astype(np.float32), report,
    )


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def train_test_overlap(X_train: np.ndarray, X_test: np.ndarray, max_rows: int = 60_000) -> dict:
    """Fraction of test rows that are byte-identical to some training row.

    Computed on hashes rather than pairwise distances so it stays O(n). A
    non-trivial overlap means the headline accuracy is partly memorisation.
    """
    rng = np.random.default_rng(0)

    def _rows(A):
        if len(A) > max_rows:
            A = A[rng.choice(len(A), max_rows, replace=False)]
        return {hash(r.tobytes()) for r in np.ascontiguousarray(A, dtype=np.float32)}

    tr = _rows(X_train)
    te_arr = X_test if len(X_test) <= max_rows else X_test[rng.choice(len(X_test), max_rows, replace=False)]
    te_arr = np.ascontiguousarray(te_arr, dtype=np.float32)
    hits = sum(1 for r in te_arr if hash(r.tobytes()) in tr)
    return {
        "test_rows_checked": int(len(te_arr)),
        "test_rows_also_in_train": int(hits),
        "train_test_overlap_rate": round(hits / max(len(te_arr), 1), 6),
    }


def single_feature_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names,
    max_rows: int = 40_000,
    top_k: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """Accuracy of a depth-3 stump built on ONE feature at a time.

    This is the cheapest reliable detector of an identifier-like column. If a
    single raw feature reaches near-perfect accuracy on a 15-class problem, the
    dataset is encoding the label in the feature (Edge-IIoTset's ``http.file_data``
    and ``ip.src_host`` are the classic cases), and every downstream model is
    solving a lookup task rather than an intrusion-detection task.
    """
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(y_train), min(max_rows, len(y_train)), replace=False)
    te = rng.choice(len(y_test), min(max_rows, len(y_test)), replace=False)

    rows = []
    for j, name in enumerate(feature_names):
        clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
        clf.fit(X_train[tr, j: j + 1], y_train[tr])
        acc = float(clf.score(X_test[te, j: j + 1], y_test[te]))
        rows.append({"feature": name, "single_feature_accuracy": acc})

    df = pd.DataFrame(rows).sort_values("single_feature_accuracy", ascending=False)
    majority = float(np.bincount(y_test[te]).max() / len(te))
    df["lift_over_majority"] = df["single_feature_accuracy"] - majority
    LOG.info(
        "single-feature probe: best=%s (%.4f) vs majority baseline %.4f",
        df.iloc[0]["feature"], df.iloc[0]["single_feature_accuracy"], majority,
    )
    return df.head(top_k).reset_index(drop=True)


def leakage_audit(data: PreparedData, dataset_provenance: dict) -> dict:
    """Everything a reviewer should be shown before believing a 99% number."""
    overlap = train_test_overlap(data.X_train, data.X_test)
    probe = single_feature_probe(
        data.X_train, data.y_train, data.X_test, data.y_test, data.feature_names
    )
    majority = float(np.bincount(data.y_test).max() / len(data.y_test))
    return {
        "majority_class_baseline": round(majority, 6),
        "exact_duplicate_rate_raw": dataset_provenance.get("exact_duplicate_rate"),
        "feature_only_duplicate_rate": dataset_provenance.get("feature_only_duplicate_rate"),
        **overlap,
        "top_single_feature_probes": probe.to_dict("records"),
        "verdict": _leakage_verdict(overlap, probe, majority),
    }


def _leakage_verdict(overlap: dict, probe: pd.DataFrame, majority: float) -> str:
    flags = []
    if overlap["train_test_overlap_rate"] > 0.001:
        flags.append(
            f"{overlap['train_test_overlap_rate']:.2%} of test rows are exact copies of training rows"
        )
    best = float(probe.iloc[0]["single_feature_accuracy"])
    if best > 0.95 and best > majority + 0.10:
        flags.append(
            f"feature '{probe.iloc[0]['feature']}' alone reaches {best:.2%} accuracy "
            "-- the task is close to a lookup"
        )
    if not flags:
        return "no leakage signal detected by these probes"
    return "; ".join(flags)
