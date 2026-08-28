"""Decompose the train/test overlap into how much of the score it can explain.

The bare overlap rate is easy to over-read. What a reviewer actually needs is
the *ceiling* it implies: if every test row with an identical encoded twin in
training were answered by looking that twin up, and every other row were
answered by guessing the majority class, what accuracy would that reach?

That number is the honest upper bound on memorisation. Anything the models score
above it is generalisation, and reporting the two side by side prevents both the
"99% is all leakage" overclaim and the "duplicates do not matter" underclaim.

    python scripts/leakage_probe.py --max-rows 80000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gbmeta.analysis import save_table  # noqa: E402
from gbmeta.datasets import STUDY_DATASETS, load_dataset  # noqa: E402
from gbmeta.preprocess import make_splits, prepare  # noqa: E402
from gbmeta.utils import LOG, setup_logging  # noqa: E402


def probe(key: str, max_rows: int, seed: int) -> dict:
    ds = load_dataset(key, max_rows=max_rows, min_rows_per_class=200,
                      min_class_support=30, seed=seed, dedup="global")
    splits = make_splits(ds.y, 0.20, 0.15, seed=seed)
    data = prepare(ds.X, ds.y, ds.class_names, splits, ds.spec.categorical_cols)

    def as_bytes(A):
        return [r.tobytes() for r in np.ascontiguousarray(A, dtype=np.float32)]

    train_rows, test_rows = as_bytes(data.X_train), as_bytes(data.X_test)

    # Majority label per distinct encoded training vector -- the lookup table.
    table: dict = {}
    for r, y in zip(train_rows, data.y_train):
        table.setdefault(r, []).append(int(y))
    lookup = {r: max(set(v), key=v.count) for r, v in table.items()}

    hits = correct = 0
    for r, y in zip(test_rows, data.y_test):
        if r in lookup:
            hits += 1
            correct += int(lookup[r] == int(y))

    n = len(data.y_test)
    majority = int(np.bincount(data.y_train).argmax())
    # Lookup where possible, majority class everywhere else.
    memorisation_ceiling = (correct + sum(
        1 for r, y in zip(test_rows, data.y_test) if r not in lookup and int(y) == majority
    )) / n

    return {
        "dataset": ds.spec.display,
        "key": key,
        "rows_used": int(len(ds.y)),
        "raw_duplicate_rate_before_dedup": ds.provenance.get("exact_duplicate_rate"),
        "raw_duplicates_left_after_dedup": round(float(ds.X.duplicated().mean()), 6),
        "encoded_train_distinct": len(set(train_rows)),
        "encoded_train_rows": len(train_rows),
        "encoded_collapse_rate": round(1 - len(set(train_rows)) / len(train_rows), 4),
        "test_rows_with_train_twin": round(hits / n, 4),
        "majority_baseline": round(float(np.bincount(data.y_test).max() / n), 4),
        "memorisation_ceiling": round(float(memorisation_ceiling), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(STUDY_DATASETS))
    ap.add_argument("--max-rows", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    setup_logging()

    rows = []
    for k in a.datasets:
        try:
            rows.append(probe(k, a.max_rows, a.seed))
            LOG.info("%s: ceiling %.4f vs majority %.4f",
                     k, rows[-1]["memorisation_ceiling"], rows[-1]["majority_baseline"])
        except Exception as exc:
            LOG.exception("%s failed: %s", k, exc)

    df = pd.DataFrame(rows)
    save_table(df.drop(columns=["key"]), "table1b_memorisation_ceiling",
               caption="How much of the reported accuracy exact memorisation could explain. "
                       "The ceiling answers every test row with an identical encoded training "
                       "row by lookup and every other row with the majority class.",
               label="tab:memceiling")
    print("\n=== TABLE 1b: memorisation ceiling ===")
    print(df.drop(columns=["key"]).to_string(index=False))
    print("\nRead it as: any score above the ceiling column is generalisation, not recall "
          "of a seen row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
