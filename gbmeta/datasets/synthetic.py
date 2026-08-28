"""A synthetic IDS-shaped dataset used for smoke tests and CI.

It reproduces the properties that break real pipelines -- heavy class imbalance,
mixed numeric/categorical columns, missing values, infinities, a constant column,
duplicate rows, and one deliberately leaky identifier column -- so the leakage
audit and the whole runner can be exercised end to end in seconds, with no
downloads and no GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .base import DatasetSpec, LoadedDataset, assemble, register

SPEC = DatasetSpec(
    key="synthetic",
    display="Synthetic IDS (smoke test)",
    label_col="Attack_type",
    drop_cols=("flow_id", "leaky_host"),
    categorical_cols=("protocol", "service"),
    source="generated in-process; no download required",
    license="n/a",
    notes="Not a scientific result. Used to validate the pipeline itself.",
)


def load(
    max_rows: int = 8_000,
    min_rows_per_class: int = 20,
    min_class_support: int = 10,
    seed: int = 42,
    dedup: str = "global",
    n_classes: int = 8,
    n_features: int = 24,
    n_rows: int = 12_000,
    leaky: bool = True,
    **_,
) -> LoadedDataset:
    rng = np.random.default_rng(seed)

    # Exponentially decaying class priors: one dominant "Normal" class and a
    # long tail, mirroring Edge-IIoTset's 72% / 0.02% spread.
    w = np.exp(-np.arange(n_classes) * 0.9)
    w /= w.sum()
    X, y = make_classification(
        n_samples=n_rows, n_features=n_features, n_informative=max(8, n_features // 2),
        n_redundant=4, n_classes=n_classes, n_clusters_per_class=2,
        weights=w.tolist(), flip_y=0.01, class_sep=1.4, random_state=seed,
    )

    names = np.array(["Normal"] + [f"Attack_{i}" for i in range(1, n_classes)])
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_features)])
    df["Attack_type"] = names[y]

    # Categorical columns, correlated with the label but far from determining it.
    df["protocol"] = np.where(rng.random(n_rows) < 0.6 + 0.3 * (y > 0), "tcp", "udp")
    df["service"] = rng.choice(["http", "dns", "mqtt", "modbus", "-"], size=n_rows)

    # Identifier-shaped columns that must be dropped by the spec.
    df["flow_id"] = np.arange(n_rows)
    if leaky:
        # Perfectly predicts the label: the audit's single-feature probe must
        # flag this, and dropping it must visibly cost accuracy.
        df["leaky_host"] = [f"10.0.{v}.1" for v in y]

    # Real-world dirt.
    df.loc[rng.random(n_rows) < 0.02, "num_0"] = np.nan
    df.loc[rng.random(n_rows) < 0.005, "num_1"] = np.inf
    df["always_zero"] = 0.0
    dup = rng.choice(n_rows, size=n_rows // 20, replace=False)
    df = pd.concat([df, df.iloc[dup]], ignore_index=True)

    return assemble(
        SPEC, [df], max_rows=max_rows, min_rows_per_class=min_rows_per_class,
        min_class_support=min_class_support, seed=seed, dedup=dedup,
        extra_provenance={"synthetic": True, "leaky_column_present": leaky},
    )


register(SPEC, load)
