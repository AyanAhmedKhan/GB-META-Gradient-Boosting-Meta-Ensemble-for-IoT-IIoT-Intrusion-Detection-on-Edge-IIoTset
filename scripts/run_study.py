"""Run the full cross-dataset study, one (dataset, seed) at a time.

Sequential and resumable on purpose: each run is checkpointed the moment its
predictions exist, so an interrupted study loses at most one model. Progress and
per-run wall time are printed so the remaining cost is always visible.

    python scripts/run_study.py --tag paper --max-rows 80000 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gbmeta.config import BUDGET, RunConfig  # noqa: E402
from gbmeta.datasets import STUDY_DATASETS  # noqa: E402
from gbmeta.runner import run_dataset  # noqa: E402
from gbmeta.utils import LOG, get_device, setup_logging  # noqa: E402

#: TabTransformer is excluded by default: its O(features^2) attention makes it
#: 5-10x the cost of every other model, which is affordable on a T4 and not on
#: a CPU. Add it explicitly with --models when a GPU is available.
DEFAULT_MODELS = (
    "logreg", "decision_tree", "random_forest",
    "lightgbm", "xgboost", "catboost",
    "mlp", "resattdnn",
    "soft_vote", "weighted_vote", "gbmeta",
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--datasets", nargs="*", default=list(STUDY_DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--max-rows", type=int, default=80_000)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--order", default="seed-major", choices=["seed-major", "dataset-major"],
                    help="seed-major finishes one seed of EVERY dataset before starting the "
                         "next seed, so the cross-dataset tests (which use one seed per "
                         "dataset) become available as early as possible")
    args = ap.parse_args(argv)

    setup_logging()
    device = get_device(args.device)
    budget = replace(BUDGET, max_rows=args.max_rows, n_estimators=args.n_estimators,
                     n_oof_folds=args.oof_folds, max_epochs=args.max_epochs)

    if args.order == "seed-major":
        jobs = [(d, s) for s in args.seeds for d in args.datasets]
    else:
        jobs = [(d, s) for d in args.datasets for s in args.seeds]
    LOG.info("=" * 72)
    LOG.info("STUDY: %d datasets x %d seeds = %d runs | device=%s | %d rows | %d trees",
             len(args.datasets), len(args.seeds), len(jobs), device,
             budget.max_rows, budget.n_estimators)
    LOG.info("=" * 72)

    t0, times = time.perf_counter(), []
    for i, (dataset, seed) in enumerate(jobs, 1):
        t = time.perf_counter()
        cfg = RunConfig(dataset=dataset, seed=seed, models=tuple(args.models),
                        budget=budget, device=device, tag=args.tag)
        try:
            run_dataset(cfg)
        except Exception as exc:  # keep going: one broken dataset must not end the study
            LOG.exception("run failed: %s seed%d (%s)", dataset, seed, exc)
            continue
        times.append(time.perf_counter() - t)
        done, elapsed = i / len(jobs), (time.perf_counter() - t0) / 60
        eta = elapsed / max(done, 1e-9) - elapsed
        LOG.info(">>> [%d/%d] %s seed%d done in %.1f min | elapsed %.1f min | ETA %.1f min",
                 i, len(jobs), dataset, seed, times[-1] / 60, elapsed, eta)

    LOG.info("STUDY COMPLETE: %d/%d runs in %.1f min",
             len(times), len(jobs), (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
