"""Dataset contract, registry, and the sampling/cleaning primitives.

Design rules that the whole study depends on:

* A loader returns **raw** features and string labels. It never scales, encodes
  or imputes -- those happen inside a fold-fitted pipeline (:mod:`gbmeta.preprocess`)
  so that no statistic ever crosses the train/test boundary.
* Row-count reduction happens **before** the split and is *stratified with a
  per-class floor*, so a 358-row MITM class survives a 10x subsample.
* Exact-duplicate removal happens **before** the split, deliberately. Duplicate
  flows that land on both sides of a split are the single largest source of
  inflated IDS accuracy; removing them globally is the conservative choice and
  it *lowers* reported scores. The duplicate rate is recorded, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..utils import LOG, sha256_file

RARE_LABEL = "__rare__"


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Everything a reviewer needs to obtain and verify the exact same data."""

    key: str
    display: str
    #: Multi-class label column in the raw file.
    label_col: str
    #: Binary (attack / normal) column, dropped for the multi-class study.
    binary_label_col: str | None = None
    #: Identifier-like columns removed before modelling. These are the columns
    #: that make an IDS look perfect for the wrong reason (IPs, ports, flow ids,
    #: raw payloads, timestamps).
    drop_cols: tuple = ()
    #: Column parsed into a monotonically ordered index for temporal drift
    #: experiments. ``None`` means drift must be simulated instead.
    timestamp_col: str | None = None
    #: Columns to force to ``category`` dtype (one-hot encoded downstream).
    categorical_cols: tuple = ()
    #: Human-readable acquisition instructions printed when the file is missing.
    source: str = ""
    kaggle: str | None = None
    files: tuple = ()
    license: str = "see source"
    notes: str = ""

    def local_paths(self, data_dir: Path | None = None) -> list[Path]:
        root = Path(data_dir or DATA_DIR) / self.key
        return [root / f for f in self.files]


@dataclass
class LoadedDataset:
    """Raw, unencoded features plus integer labels and full provenance."""

    key: str
    X: pd.DataFrame
    y: np.ndarray  # int32 class indices
    class_names: np.ndarray  # index -> original string label
    spec: DatasetSpec
    timestamps: pd.Series | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> dict:
        counts = pd.Series(self.y).value_counts().sort_index()
        return {
            "key": self.key,
            "display": self.spec.display,
            "n_rows": int(len(self.y)),
            "n_features_raw": int(self.X.shape[1]),
            "n_classes": self.n_classes,
            "has_timestamps": self.timestamps is not None,
            "class_counts": {
                str(self.class_names[i]): int(c) for i, c in counts.items()
            },
            "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
            **self.provenance,
        }


LoaderFn = Callable[..., LoadedDataset]
_REGISTRY: dict[str, tuple[DatasetSpec, LoaderFn]] = {}


def register(spec: DatasetSpec, loader: LoaderFn) -> None:
    _REGISTRY[spec.key] = (spec, loader)


def get_spec(key: str) -> DatasetSpec:
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key][0]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


def load_dataset(key: str, **kwargs) -> LoadedDataset:
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key][1](**kwargs)


# --------------------------------------------------------------------------
# Cleaning primitives
# --------------------------------------------------------------------------
def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and de-duplicate them.

    CICIDS2017 ships columns with leading spaces; Edge-IIoTset produces
    duplicate names after ``get_dummies``. Both break column-wise assignment
    in subtle, silent ways.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    seen: dict[str, int] = {}
    new = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            new.append(c)
    df.columns = new
    return df


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove zero-variance columns (they carry no signal and break scalers)."""
    nunique = df.nunique(dropna=False)
    dead = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=dead), dead


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS2017 flow rates contain literal ``inf``. Turn them into NaN so the
    fold-fitted imputer handles them -- never fill them with 0 globally."""
    num = df.select_dtypes(include=[np.number]).columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df


def deduplicate(
    X: pd.DataFrame, y: np.ndarray, timestamps: pd.Series | None = None
) -> tuple[pd.DataFrame, np.ndarray, pd.Series | None, dict]:
    """Drop rows identical in *both* features and label.

    Returns the kept data plus a report. Rows identical in features but with
    *conflicting* labels are kept (they are genuine label noise, and dropping
    them would silently make the task easier).
    """
    n0 = len(X)
    key = X.copy()
    key["__y__"] = y
    mask = ~key.duplicated(keep="first")
    del key

    feat_dupes = int(X.duplicated(keep="first").sum())
    kept = int(mask.sum())
    report = {
        "rows_before_dedup": n0,
        "rows_after_dedup": kept,
        "exact_duplicate_rate": round(1.0 - kept / max(n0, 1), 6),
        "feature_only_duplicate_rate": round(feat_dupes / max(n0, 1), 6),
        "conflicting_label_duplicates": int(feat_dupes - (n0 - kept)),
    }
    LOG.info(
        "dedup: %d -> %d rows (%.2f%% exact duplicates removed)",
        n0, kept, 100 * report["exact_duplicate_rate"],
    )
    ts = timestamps[mask.values].reset_index(drop=True) if timestamps is not None else None
    return X[mask.values].reset_index(drop=True), y[mask.values], ts, report


def merge_rare_classes(
    labels: pd.Series, min_support: int
) -> tuple[pd.Series, list[str]]:
    """Fold classes with fewer than ``min_support`` rows into ``__rare__``.

    Keeping a 5-row class produces per-class F1 numbers that are pure noise and
    makes stratified CV impossible. Merging is stated explicitly in the results
    rather than silently dropping the rows.
    """
    if min_support <= 0:
        return labels, []
    counts = labels.value_counts()
    rare = counts[counts < min_support].index.tolist()
    if not rare:
        return labels, []
    LOG.info("merging %d rare classes into %s: %s", len(rare), RARE_LABEL, rare)
    return labels.where(~labels.isin(rare), RARE_LABEL), rare


def cap_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    min_rows_per_class: int,
    seed: int,
    timestamps: pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series | None, dict]:
    """Stratified subsample to ``max_rows`` with a guaranteed per-class floor.

    Proportional stratification alone would reduce a 358-row class to 36 rows at
    a 10x cut, which destroys its per-class metrics. Instead every class first
    receives ``min(n_c, min_rows_per_class)`` rows, and the remaining budget is
    distributed proportionally to the leftover class sizes.

    The resulting sample is *not* i.i.d. with the original prior, so the
    per-class reweighting used downstream is reported alongside it.
    """
    n = len(y)
    if n <= max_rows:
        return X, y, timestamps, {"subsampled": False, "rows": n}

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    floor = np.minimum(counts, min_rows_per_class)
    budget = max_rows - int(floor.sum())
    if budget <= 0:
        # Even the floors exceed the budget: take a proportional cut of them.
        take = np.maximum(1, np.floor(floor * (max_rows / floor.sum())).astype(int))
    else:
        leftover = counts - floor
        share = leftover / max(leftover.sum(), 1)
        extra = np.floor(share * budget).astype(int)
        extra = np.minimum(extra, leftover)
        take = floor + extra
        # Hand out any rounding remainder to the largest classes.
        remainder = max_rows - int(take.sum())
        if remainder > 0:
            order = np.argsort(-(counts - take))
            for i in order:
                if remainder == 0:
                    break
                room = int(counts[i] - take[i])
                add = min(room, remainder)
                take[i] += add
                remainder -= add

    idx_parts = []
    for cls, k in zip(classes, take):
        pool = np.flatnonzero(y == cls)
        if k >= len(pool):
            idx_parts.append(pool)
        else:
            idx_parts.append(rng.choice(pool, size=int(k), replace=False))
    idx = np.sort(np.concatenate(idx_parts))

    report = {
        "subsampled": True,
        "rows_before_cap": n,
        "rows_after_cap": int(len(idx)),
        "per_class_kept": {int(c): int(t) for c, t in zip(classes, take)},
        "cap_seed": seed,
    }
    LOG.info("stratified cap: %d -> %d rows (floor=%d/class)", n, len(idx), min_rows_per_class)
    ts = timestamps.iloc[idx].reset_index(drop=True) if timestamps is not None else None
    return X.iloc[idx].reset_index(drop=True), y[idx], ts, report


# --------------------------------------------------------------------------
# Generic CSV assembly used by every concrete loader
# --------------------------------------------------------------------------
def assemble(
    spec: DatasetSpec,
    frames: Iterable[pd.DataFrame],
    max_rows: int,
    min_rows_per_class: int,
    min_class_support: int,
    seed: int,
    dedup: str = "global",
    label_map: Callable[[pd.Series], pd.Series] | None = None,
    extra_provenance: dict | None = None,
) -> LoadedDataset:
    """Turn raw CSV frames into a :class:`LoadedDataset`.

    Order of operations is deliberate and is the part reviewers should check:

    1. concatenate -> 2. sanitise names -> 3. extract + normalise labels ->
    4. drop identifier columns -> 5. inf -> NaN -> 6. drop constants ->
    7. merge rare classes -> 8. global exact dedup -> 9. stratified cap.

    Nothing here looks at a train/test split, because the split does not exist
    yet. Everything that *does* depend on the split (imputation, scaling,
    encoding, class weights) happens later, inside a fold-fitted pipeline.
    """
    df = pd.concat(list(frames), ignore_index=True, sort=False)
    df = sanitise_columns(df)
    prov: dict = {"raw_rows": int(len(df)), "raw_cols": int(df.shape[1])}

    if spec.label_col not in df.columns:
        raise KeyError(
            f"{spec.key}: label column {spec.label_col!r} not found. "
            f"Columns: {list(df.columns)[:40]}"
        )

    labels = df[spec.label_col].astype(str).str.strip()
    if label_map is not None:
        labels = label_map(labels)

    timestamps = None
    if spec.timestamp_col and spec.timestamp_col in df.columns:
        timestamps = pd.to_datetime(df[spec.timestamp_col], errors="coerce", format="mixed")
        if timestamps.isna().mean() > 0.5:
            LOG.warning("%s: timestamp column unparseable; drift will be simulated", spec.key)
            timestamps = None

    to_drop = [c for c in (*spec.drop_cols, spec.label_col, spec.binary_label_col) if c and c in df.columns]
    X = df.drop(columns=to_drop)
    prov["dropped_columns"] = to_drop
    del df

    X = replace_inf(X)
    X, dead = drop_constant_columns(X)
    prov["constant_columns_dropped"] = dead

    labels, merged = merge_rare_classes(labels, min_class_support)
    prov["merged_rare_classes"] = merged

    class_names = np.array(sorted(labels.unique()))
    lut = {c: i for i, c in enumerate(class_names)}
    y = labels.map(lut).to_numpy(dtype=np.int32)

    if dedup == "global":
        X, y, timestamps, dd = deduplicate(X, y, timestamps)
        prov.update(dd)
    else:
        prov["exact_duplicate_rate"] = float(X.duplicated().mean())
        prov["dedup"] = "disabled"

    X, y, timestamps, cap = cap_rows(
        X, y, max_rows, min_rows_per_class, seed, timestamps
    )
    prov.update(cap)

    # Classes can vanish after dedup/cap; re-index so labels stay contiguous.
    present = np.unique(y)
    if len(present) != len(class_names):
        remap = {old: new for new, old in enumerate(present)}
        y = np.array([remap[v] for v in y], dtype=np.int32)
        class_names = class_names[present]

    if extra_provenance:
        prov.update(extra_provenance)

    return LoadedDataset(
        key=spec.key, X=X, y=y, class_names=class_names, spec=spec,
        timestamps=timestamps, provenance=prov,
    )


def read_csvs(paths: Sequence[Path], **read_kwargs) -> tuple[list, dict]:
    """Read CSVs, hashing each one so results are traceable to exact bytes.

    Returns ``(frames, {filename: sha256_prefix})``. The hashes go into the run
    manifest, which is what lets someone else confirm they used the same file
    and not a differently-preprocessed Kaggle re-upload.
    """
    frames, hashes = [], {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python -m gbmeta.datasets.fetch <key>` "
                f"or place the file there manually."
            )
        hashes[p.name] = sha256_file(p)[:16]
        frames.append(pd.read_csv(p, low_memory=False, **read_kwargs))
        LOG.info("read %s: %s rows x %s cols", p.name, len(frames[-1]), frames[-1].shape[1])
    return frames, hashes
