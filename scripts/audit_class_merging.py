"""Quantify the rare-class merge, per benchmark.

The limitations section states that classes below the support floor are folded
into one rare category. A reader cannot judge whether that matters without
knowing how many classes it touches and how many rows they hold, so this
measures it directly from the loaded data rather than leaving it qualitative.

    python scripts/audit_class_merging.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from gbmeta.datasets import STUDY_DATASETS, load_dataset  # noqa: E402

MIN_SUPPORT = 30
OUT = ROOT / "paper" / "tables" / "table12_class_merging.csv"


def main() -> int:
    rows = []
    for key in STUDY_DATASETS:
        try:
            ds = load_dataset(key, min_class_support=MIN_SUPPORT)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {key}: could not load ({type(exc).__name__}: {exc})")
            continue

        # y is label-encoded, so the rare bucket has to be found by its position
        # in class_names rather than by the string itself.
        names = list(ds.class_names)
        y = pd.Series(ds.y).map(dict(enumerate(names)))
        counts = y.value_counts()
        n = len(y)
        merged_rows = int(counts.get("__rare__", 0))
        prov = getattr(ds, "provenance", {}) or {}
        merged_names = prov.get("merged_rare_classes") or []
        rows.append({
            "dataset": ds.spec.display,
            "rows_used": n,
            "classes_after_merge": int(y.nunique()),
            "classes_merged": len(merged_names),
            "merged_rows": merged_rows,
            "merged_row_fraction": round(merged_rows / n, 5) if n else 0.0,
            "smallest_kept_class": int(counts[counts.index != "__rare__"].min()),
        })
        print(f"  {ds.spec.display:<14} {n:>7,} rows  "
              f"{y.nunique():>3} classes after merge  "
              f"{len(merged_names):>2} merged  "
              f"{merged_rows:>5,} rows ({100 * merged_rows / n:.3f}%)")

    if rows:
        df = pd.DataFrame(rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
        print(f"\nwrote {OUT.relative_to(ROOT)}")
        tot = df["merged_rows"].sum()
        print(f"total rows folded into the rare bucket across all benchmarks: {tot:,}")
    return 0


if __name__ == "__main__":
    print(f"=== rare-class merge audit (support floor {MIN_SUPPORT}) ===\n")
    sys.exit(main())
