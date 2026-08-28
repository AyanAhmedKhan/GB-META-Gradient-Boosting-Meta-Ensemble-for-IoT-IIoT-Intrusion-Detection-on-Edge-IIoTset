"""End-to-end experiment driver.

Two properties matter more than anything else here:

**Resumability.** A free Colab session dies. Every model's predictions are
cached to ``.npy`` the moment they exist, and a rerun skips any model whose
artefacts are already on disk. A five-dataset study can therefore be completed
across three sessions without losing work.

**Separation of training from analysis.** Training writes probability matrices;
significance testing, ablation, calibration, curves, and the deployment table
all read those matrices back. Nothing downstream ever retrains a model, so the
entire analysis can be re-run in seconds after a reviewer asks for a different
test.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import (
    BUDGET, CLASSICAL_BASELINES, DEEP_BASELINES, ENSEMBLES,
    RunConfig, SMOKE_BUDGET, STACK_BASE_MODELS, SEED,
)
from .datasets import load_dataset
from .ensemble import compute_oof, make_combiner
from .evaluate import compute_metrics, curve_data, per_class_report
from .models.base import ModelContext, available_models, build_model, unavailable_models
from .preprocess import PreparedData, leakage_audit, make_splits, prepare
from .utils import (
    LOG, environment_manifest, purge, read_json, resource_probe, set_seed,
    setup_logging, write_json,
)

# Import for side-effect: model registration.
from .models import neural as _neural  # noqa: F401
from .models import trees as _trees  # noqa: F401


def _paths(cfg: RunConfig) -> dict:
    root = cfg.run_dir
    return {
        "root": root,
        "probas": root / "probas",
        "models": root / "models",
        "curves": root / "curves",
    }


def _save_proba(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float32))


# --------------------------------------------------------------------------
# Data stage
# --------------------------------------------------------------------------
def build_data(cfg: RunConfig, temporal: bool = False) -> tuple:
    b = cfg.budget
    ds = load_dataset(
        cfg.dataset, max_rows=b.max_rows, min_rows_per_class=b.min_rows_per_class,
        min_class_support=b.min_class_support, seed=cfg.seed, dedup=cfg.dedup,
    )
    splits = make_splits(
        ds.y, test_size=b.test_size, val_size=b.val_size, seed=cfg.seed,
        timestamps=ds.timestamps, temporal=temporal,
    )
    data = prepare(ds.X, ds.y, ds.class_names, splits, ds.spec.categorical_cols,
                   class_weight_power=b.class_weight_power)
    return ds, data


# --------------------------------------------------------------------------
# Single-model stage
# --------------------------------------------------------------------------
def train_one(
    key: str, ctx: ModelContext, data: PreparedData, paths: dict,
    with_oof: bool, n_folds: int, force: bool = False,
) -> dict | None:
    """Train one model, cache its predictions, return its record.

    Returns ``None`` if the model's backend is missing -- the run continues and
    the manifest records the omission rather than the study dying halfway.
    """
    rec_path = paths["models"] / f"{key}.json"
    test_path = paths["probas"] / f"{key}_test.npy"
    if rec_path.exists() and test_path.exists() and not force:
        LOG.info("%-16s cached - skipping", key)
        return read_json(rec_path)

    if key not in available_models():
        LOG.warning("%-16s unavailable (%s) - skipping", key, unavailable_models().get(key))
        return None

    sw = data.sample_weights(data.y_train) if ctx.params.get("class_weighting", True) else None

    with resource_probe(f"train:{key}") as res:
        model = build_model(key, ctx)
        model.fit(data.X_train, data.y_train, data.X_val, data.y_val, sample_weight=sw)

    p_test = model.predict_proba(data.X_test)
    p_val = model.predict_proba(data.X_val)
    _save_proba(test_path, p_test)
    _save_proba(paths["probas"] / f"{key}_val.npy", p_val)

    oof_seconds = 0.0
    if with_oof:
        oof_path = paths["probas"] / f"{key}_oof.npy"
        if oof_path.exists() and not force:
            LOG.info("%-16s OOF cached", key)
        else:
            oof = compute_oof(key, ctx, data.X_train, data.y_train, n_folds, sw)
            _save_proba(oof_path, oof.oof_proba)
            oof_seconds = oof.seconds

    metrics = compute_metrics(data.y_test, p_test, data.n_classes)
    rec = {
        "model": key,
        "metrics": metrics,
        "per_class": per_class_report(data.y_test, p_test.argmax(1), data.class_names),
        "complexity": model.complexity(),
        "uses_gpu": bool(getattr(model, "uses_gpu", False)),
        "fit_seconds": float(getattr(model, "fit_seconds", float("nan"))),
        "oof_seconds": oof_seconds,
        "resources": {k: v for k, v in res.items() if k != "label"},
        "history": getattr(model, "history", None),
    }
    write_json(rec_path, rec)
    write_json(paths["curves"] / f"{key}.json", asdict(curve_data(data.y_test, p_test, data.class_names)))

    LOG.info("%-16s acc=%.4f macroF1=%.4f (fit %.1fs)",
             key, metrics["accuracy"], metrics["macro_f1"], rec["fit_seconds"])
    del model
    purge()
    return rec


# --------------------------------------------------------------------------
# Ensemble stage (no retraining -- reads cached OOF + test probabilities)
# --------------------------------------------------------------------------
def build_ensembles(cfg: RunConfig, data: PreparedData, paths: dict, base_keys) -> dict:
    have = [k for k in base_keys if (paths["probas"] / f"{k}_oof.npy").exists()]
    missing = sorted(set(base_keys) - set(have))
    if len(have) < 2:
        LOG.warning("only %d base models with OOF - skipping ensembles", len(have))
        return {}

    oof = [np.load(paths["probas"] / f"{k}_oof.npy") for k in have]
    test = [np.load(paths["probas"] / f"{k}_test.npy") for k in have]
    oof_acc = [float((o.argmax(1) == data.y_train).mean()) for o in oof]

    out = {}
    for name in ENSEMBLES:
        comb = make_combiner(name, seed=cfg.seed)
        comb.fit(oof, data.y_train, val_accuracies=oof_acc)
        p_test = np.asarray(comb.combine(test), dtype=np.float64)
        p_test = np.clip(p_test, 1e-12, None)
        p_test /= p_test.sum(axis=1, keepdims=True)
        _save_proba(paths["probas"] / f"{name}_test.npy", p_test)

        metrics = compute_metrics(data.y_test, p_test, data.n_classes)
        desc = comb.describe()
        if name == "gbmeta":
            desc["coefficient_mass"] = dict(zip(have, comb.coefficient_mass(len(have))))
        rec = {
            "model": name,
            "metrics": metrics,
            "per_class": per_class_report(data.y_test, p_test.argmax(1), data.class_names),
            "complexity": {"base_models": have, "base_models_missing": missing, **desc},
            "uses_gpu": False,
            "fit_seconds": 0.0,
            "oof_seconds": 0.0,
            "resources": {},
            "history": None,
        }
        write_json(paths["models"] / f"{name}.json", rec)
        write_json(paths["curves"] / f"{name}.json",
                   asdict(curve_data(data.y_test, p_test, data.class_names)))
        out[name] = rec
        LOG.info("%-16s acc=%.4f macroF1=%.4f", name, metrics["accuracy"], metrics["macro_f1"])
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_dataset(cfg: RunConfig, temporal: bool = False, force: bool = False) -> dict:
    setup_logging()
    # Library chatter that is irrelevant here: sklearn's "no feature names"
    # notice (matrices are intentionally numpy) and torch's nested-tensor note.
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    set_seed(cfg.seed)
    paths = _paths(cfg)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 72)
    LOG.info("dataset=%s seed=%d tag=%s device=%s", cfg.dataset, cfg.seed, cfg.tag, cfg.device)
    LOG.info("=" * 72)

    ds, data = build_data(cfg, temporal=temporal)
    write_json(paths["root"] / "dataset.json", ds.summary())

    audit_path = paths["root"] / "leakage.json"
    if force or not audit_path.exists():
        with resource_probe("leakage-audit"):
            write_json(audit_path, leakage_audit(data, ds.provenance))
    LOG.info("leakage verdict: %s", read_json(audit_path)["verdict"])

    for name, arr in (("y_train", data.y_train), ("y_val", data.y_val), ("y_test", data.y_test)):
        np.save(paths["root"] / f"{name}.npy", arr)
    write_json(paths["root"] / "classes.json", [str(c) for c in data.class_names])

    ctx = ModelContext(
        n_classes=data.n_classes, n_features=data.n_features, seed=cfg.seed,
        device=cfg.device, budget=cfg.budget, class_weights=data.class_weights,
        feature_names=data.feature_names,
    )

    stack_bases = [k for k in cfg.stack_bases if k in available_models()]
    if len(stack_bases) < len(cfg.stack_bases):
        LOG.warning("stack reduced to %s (missing backends: %s)",
                    stack_bases, sorted(set(cfg.stack_bases) - set(stack_bases)))

    records = {}
    # A stack base must be trained even if the caller left it out of --models,
    # otherwise the ensemble would silently be built from fewer learners.
    to_train = list(dict.fromkeys([k for k in cfg.models if k not in ENSEMBLES] + stack_bases))
    for key in to_train:
        rec = train_one(
            key, ctx, data, paths, with_oof=key in stack_bases,
            n_folds=cfg.budget.n_oof_folds,
            force=force,
        )
        if rec is not None:
            records[key] = rec

    if any(k in cfg.models for k in ENSEMBLES):
        records.update(build_ensembles(cfg, data, paths, stack_bases))

    manifest = {
        "config": {**asdict(cfg), "budget": cfg.budget.as_dict(), "models": list(cfg.models)},
        "environment": environment_manifest(),
        "dataset": ds.summary(),
        "preprocessing": data.report,
        "models_run": sorted(records),
        "models_unavailable": unavailable_models(),
        "temporal_split": temporal,
    }
    write_json(paths["root"] / "manifest.json", manifest)
    LOG.info("wrote %s", paths["root"])
    return {"manifest": manifest, "records": records, "paths": {k: str(v) for k, v in paths.items()}}


def load_run(cfg: RunConfig) -> dict:
    """Read back everything a completed run produced. No retraining."""
    paths = _paths(cfg)
    if not (paths["root"] / "manifest.json").exists():
        raise FileNotFoundError(f"no completed run at {paths['root']}")
    records = {p.stem: read_json(p) for p in sorted(paths["models"].glob("*.json"))}
    probas = {p.stem[:-5]: np.load(p) for p in sorted(paths["probas"].glob("*_test.npy"))}
    return {
        "manifest": read_json(paths["root"] / "manifest.json"),
        "leakage": read_json(paths["root"] / "leakage.json"),
        "classes": read_json(paths["root"] / "classes.json"),
        "records": records,
        "test_proba": probas,
        "y_test": np.load(paths["root"] / "y_test.npy"),
        "y_train": np.load(paths["root"] / "y_train.npy"),
        "paths": paths,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one dataset x seed of the GB-META v2 study")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--stack-bases", nargs="*", default=list(STACK_BASE_MODELS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dedup", default="global", choices=["global", "none"])
    ap.add_argument("--temporal", action="store_true", help="chronological split (needs timestamps)")
    ap.add_argument("--smoke", action="store_true", help="tiny budget, runs on a laptop CPU")
    ap.add_argument("--max-rows", type=int, default=None, help="override Budget.max_rows")
    ap.add_argument("--n-estimators", type=int, default=None, help="override boosting rounds")
    ap.add_argument("--max-epochs", type=int, default=None, help="override neural epochs")
    ap.add_argument("--oof-folds", type=int, default=None, help="override OOF fold count")
    ap.add_argument("--force", action="store_true", help="ignore cached artefacts")
    args = ap.parse_args(argv)

    from .utils import get_device
    models = tuple(args.models) if args.models else (
        CLASSICAL_BASELINES + STACK_BASE_MODELS + DEEP_BASELINES + ENSEMBLES
    )
    from dataclasses import replace as _replace
    budget = SMOKE_BUDGET if args.smoke else BUDGET
    overrides = {k: v for k, v in (
        ("max_rows", args.max_rows), ("n_estimators", args.n_estimators),
        ("max_epochs", args.max_epochs), ("n_oof_folds", args.oof_folds),
    ) if v is not None}
    if overrides:
        budget = _replace(budget, **overrides)
        LOG.info("budget overrides: %s", overrides)

    cfg = RunConfig(
        dataset=args.dataset, seed=args.seed, models=models,
        budget=budget,
        device=get_device(args.device), dedup=args.dedup,
        stack_bases=tuple(args.stack_bases),
        tag=args.tag + ("-smoke" if args.smoke else ""),
    )
    run_dataset(cfg, temporal=args.temporal, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
