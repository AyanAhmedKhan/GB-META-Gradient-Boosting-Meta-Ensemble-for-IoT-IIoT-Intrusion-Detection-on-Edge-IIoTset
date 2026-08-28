"""Compare the 80,000-row study against a full-scale Edge-IIoTset replication.

The study subsamples each benchmark to 80,000 rows for compute reasons. This
reads the full-scale rerun (152,196 rows after deduplication, 103,492 train /
30,440 test) and reports whether the conclusions drawn at 80,000 hold at full
scale: whether GB-META still leads, and whether the ordering of the models is
preserved.

    python scripts/compare_fullscale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

FULL = ROOT / "results" / "fullscale" / "edge_iiotset"
TAB = ROOT / "paper" / "tables"
OUT = TAB / "table14_fullscale.csv"

NAME = {"gbmeta": "GB-META", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "catboost": "CatBoost", "soft_vote": "Soft Vote",
        "weighted_vote": "Weighted Vote", "decision_tree": "Decision Tree",
        "random_forest": "Random Forest", "mlp": "MLP", "resattdnn": "ResAttDNN",
        "logreg": "Logistic Regression"}


def read_seed(d: Path) -> dict:
    out = {}
    for f in (d / "models").glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        m = j.get("metrics", j)
        f1 = m.get("macro_f1", m.get("f1_macro"))
        if f1 is not None:
            out[NAME.get(f.stem, f.stem)] = float(f1)
    return out


def main() -> int:
    if not FULL.exists():
        print(f"no full-scale results at {FULL}")
        return 1

    per_seed = {}
    for d in sorted(FULL.glob("seed*")):
        r = read_seed(d)
        if r:
            per_seed[d.name] = r
    if not per_seed:
        print("no per-model metrics found; check the results layout")
        return 1

    full = pd.DataFrame(per_seed)
    full["full_mean"] = full.mean(axis=1)
    full["full_std"] = full.std(axis=1, ddof=1)

    sub = pd.read_csv(TAB / "table5_seed_variance.csv")
    sub = sub[sub.dataset == "edge_iiotset"].set_index("model")["mean"]

    df = full[["full_mean", "full_std"]].join(sub.rename("sub_mean"), how="left")
    df["delta"] = df["full_mean"] - df["sub_mean"]
    df = df.sort_values("full_mean", ascending=False)
    df.to_csv(OUT)

    print(f"=== Edge-IIoTset: full scale vs the 80,000-row study "
          f"({len(per_seed)} seeds) ===\n")
    print(f"  {'model':<20}{'full scale':>18}{'80k study':>12}{'delta':>10}")
    for m, r in df.iterrows():
        sm = f"{r['sub_mean']:.4f}" if pd.notna(r["sub_mean"]) else "--"
        dl = f"{r['delta']:+.4f}" if pd.notna(r["delta"]) else "--"
        print(f"  {m:<20}{f'{r.full_mean:.4f}+/-{r.full_std:.4f}':>18}{sm:>12}{dl:>10}")

    print(f"\nwrote {OUT.relative_to(ROOT)}")

    ranked = list(df.index)
    print(f"\n  full-scale leader : {ranked[0]}")
    if "GB-META" in ranked:
        print(f"  GB-META rank      : {ranked.index('GB-META') + 1} of {len(ranked)}")
    sub_rank = list(sub.sort_values(ascending=False).index)
    common = [m for m in ranked if m in sub_rank]
    print(f"  top-3 at full scale: {ranked[:3]}")
    print(f"  top-3 at 80k       : {[m for m in sub_rank if m in common][:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
