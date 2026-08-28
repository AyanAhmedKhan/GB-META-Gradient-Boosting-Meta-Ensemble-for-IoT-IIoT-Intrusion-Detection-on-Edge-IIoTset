"""Build the self-contained Colab notebook.

The notebook is a *build artefact*, not a hand-edited file: it embeds the
``gbmeta`` package as a base64 tarball so a single .ipynb upload reproduces the
entire study on a fresh Colab runtime with no repository, no Drive mount and no
Kaggle credentials.

Regenerate after changing the package::

    python scripts/build_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "gbmeta"
OUT = ROOT / "notebooks" / "GB_META_v2_Colab.ipynb"


# --------------------------------------------------------------------------
def bundle_package() -> str:
    """tar.gz the package into a base64 string."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(PKG.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    raw = buf.getvalue()
    print(f"  package bundle: {len(raw) / 1024:.0f} KiB gzipped "
          f"({sum(1 for _ in PKG.rglob('*.py'))} modules)")
    return base64.b64encode(raw).decode("ascii")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").split("\n")}


def _wrap(b64: str, width: int = 96) -> str:
    return "\n".join(b64[i: i + width] for i in range(0, len(b64), width))


# --------------------------------------------------------------------------
def build_cells(b64: str) -> list:
    cells = []

    cells.append(md(r"""
# GB-META — a leakage-audited, T4-runnable IDS benchmark

This notebook rebuilds every number, table and figure for
**"GB-META: Gradient-Boosting Meta-Ensemble for IoT/IIoT Intrusion Detection on
Edge-IIoTset"**, from the raw datasets, on one free Colab T4.

| What it establishes | Where |
|---|---|
| Data leakage, quantified before any model is trained | §1 — duplicate rate, encoded train/test overlap with a memorisation ceiling, single-feature probe |
| Statistical significance | §3 — McNemar (exact), paired bootstrap intervals, Friedman + Iman-Davenport, Nemenyi CD diagram, Wilcoxon, Holm correction |
| External validation | §2 — five datasets: Edge-IIoTset, NSL-KDD, UNSW-NB15, ToN-IoT, CICIDS2017 |
| Deployment cost | §5 — batch-1 p50/p95/p99 latency, throughput sweep, model size, GPU energy via NVML, single-thread ONNX edge proxy |
| Component attribution | §4 — leave-one-base-out, combiner ablation, HPO on/off, class weighting, dedup, each with a bootstrap interval |
| Figure quality | every figure is written as vector PDF + 600 dpi PNG |
| Intervals throughout | §3 — every headline metric carries a percentile bootstrap interval |
| Confusion matrix, ROC, PR, calibration | §7 |
| Robustness and drift | §6 — constrained noise sweeps, HopSkipJump, PSI/KS drift, temporal AUT |

**Runtime.** Set `PROFILE` in §0.4. All three produce every table and figure — they
differ in sample size, seed count, and which optional stages run.

| | Wall clock on a free T4 | What you get |
|---|---|---|
| `"fast"` *(default)* | **~15–25 min** | 1 seed, 15k rows/dataset — the full results, significance tests, ablation, cost table and figures |
| `"quick"` | ~30–45 min | 3 seeds, 60k rows; adds Optuna, the dedup ablation, and black-box evasion |
| `"full"` | ~3–5 h | the configuration reported in the paper |

Runs are **resumable**: predictions are cached the moment they exist, so a
disconnect costs only the model that was training, and moving up a profile
re-uses nothing but wastes nothing.

**Everything is cached.** Training writes probability matrices to disk; every table,
test and figure is computed from those. Re-running the analysis costs seconds.
"""))

    # ---------------------------------------------------------------- setup
    cells.append(md("## 0 · Setup\n\n### 0.1 Runtime check"))
    cells.append(code(r"""
import os, sys, platform, subprocess

# OpenMP reads this at library load time, so it must be set before numpy/torch
# are imported. Colab has 2 vCPUs; leaving both to OpenMP is correct here.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 2))

print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())
print("CPUs    :", os.cpu_count())
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
          or "no GPU detected")
except FileNotFoundError:
    print("no GPU detected -- everything still runs on CPU, just slower")

try:
    import psutil
    print(f"RAM     : {psutil.virtual_memory().total/1e9:.1f} GB")
except ImportError:
    pass
"""))

    cells.append(md(r"""
### 0.2 Dependencies

Colab already ships lightgbm, xgboost, scikit-learn, torch and optuna. Only the
missing pieces are installed. Pins that matter:

* `river==0.22.0` — 0.26.x pulls a dependency set that breaks the session.
* `adversarial-robustness-toolbox` needs `packaging`, which it does not declare.
* `nvidia-ml-py`, **not** `pynvml` — the two collide on the same import name.
"""))
    cells.append(code(r"""
%%capture install_log
import sys
# Core (usually already present on Colab, pinned only if absent)
!pip install -q catboost statsmodels psutil
# Optional stages -- the notebook degrades gracefully if any of these fail
!pip install -q nvidia-ml-py onnxruntime skl2onnx onnxmltools onnx
!pip install -q "river==0.22.0"
!pip install -q packaging "adversarial-robustness-toolbox==1.20.1"
!pip install -q kagglehub tabulate
"""))
    cells.append(code(r"""
import importlib
report = {}
for m in ["numpy","pandas","sklearn","scipy","statsmodels","lightgbm","xgboost","catboost",
          "torch","optuna","matplotlib","onnxruntime","skl2onnx","onnxmltools",
          "pynvml","river","art","kagglehub"]:
    try:
        report[m] = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception:
        report[m] = None
have = {k: v for k, v in report.items() if v}
miss = [k for k, v in report.items() if not v]
print("installed:", have)
print("\nMISSING (those stages will be skipped and the omission recorded):", miss or "none")
"""))

    cells.append(md(r"""
### 0.3 Unpack the `gbmeta` package

The package is embedded in this notebook as a base64 tarball, so the notebook is
self-contained: no clone, no Drive, no upload. If you have the repository
checked out instead, the cell uses that and skips unpacking.
"""))
    cells.append(code(
        "import base64, io, tarfile, os, sys\n"
        "from pathlib import Path\n\n"
        "WORK = Path('/content') if Path('/content').exists() else Path.cwd()\n"
        "os.chdir(WORK)\n\n"
        "if (WORK / 'gbmeta' / 'runner.py').exists():\n"
        "    print('using the gbmeta package already present in', WORK)\n"
        "else:\n"
        "    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKAGE_B64)), mode='r:gz') as t:\n"
        "        t.extractall(WORK)\n"
        "    print('unpacked gbmeta to', WORK)\n\n"
        "if str(WORK) not in sys.path:\n"
        "    sys.path.insert(0, str(WORK))\n\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore', category=FutureWarning)\n"
        "warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')\n"
        "warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')\n\n"
        "from gbmeta.utils import setup_logging, environment_manifest, get_device\n"
        "LOG = setup_logging()\n"
        "DEVICE = get_device('auto')\n"
        "print('device:', DEVICE)\n"
    ))
    # The bundle lives in its own cell so the code above stays readable.
    cells.insert(len(cells) - 1, code(
        "# --- embedded gbmeta package (base64 tar.gz) -------------------------------\n"
        "PACKAGE_B64 = '''\n" + _wrap(b64) + "\n'''.replace('\\n', '')\n"
        "print(f'embedded package: {len(PACKAGE_B64)/1024:.0f} KiB base64')"
    ))

    # ---------------------------------------------------------------- config
    cells.append(md(r"""
### 0.4 Configuration

Everything that changes what gets reported lives in this one cell.
"""))
    cells.append(code(r"""
from dataclasses import replace
from gbmeta.config import BUDGET, RESULTS_DIR, FIG_DIR, TAB_DIR
from gbmeta.datasets import STUDY_DATASETS

# ---------------------------------------------------------------------------
# PROFILE controls how long the whole notebook takes. Every profile produces
# every table and figure -- they differ in sample size, seed count, and which
# of the expensive optional stages run.
#
#   "fast"   ~15-25 min   1 seed, 15k rows/dataset. Use this first.
#   "quick"  ~30-45 min   3 seeds, 60k rows, adds HPO + black-box attacks.
#   "full"   ~3-5 h       the configuration reported in the paper.
# ---------------------------------------------------------------------------
PROFILE = "fast"

DATASETS = list(STUDY_DATASETS)          # edge_iiotset, nslkdd, unsw_nb15, ton_iot, cicids2017
STACK_BASES = ("lightgbm", "xgboost", "catboost")

if PROFILE == "fast":
    SEEDS = [42]
    # TabTransformer is dropped here, not everywhere: its attention is O(F^2) in
    # the feature count, which makes it 5-10x the cost of every other model on a
    # 100-feature dataset. It returns in "quick".
    MODELS = ("logreg", "decision_tree", "random_forest",
              "lightgbm", "xgboost", "catboost", "mlp", "resattdnn",
              "soft_vote", "weighted_vote", "gbmeta")
    BUDGET_CFG = replace(BUDGET, max_rows=15_000, min_rows_per_class=120,
                         n_estimators=150, patience=20, n_oof_folds=3,
                         max_epochs=12, n_trials=6)
    N_BOOT, LATENCY_REPS = 400, 30
    ENERGY_SECONDS, ATTACK_SAMPLES = 0, 0
    COST_MODELS = ["decision_tree", "lightgbm", "xgboost"]
    FACTORIAL_GRID = [(i, s) for i in ("loss", "sampler") for s in ("onecycle", "cosine")]
    RUN_HPO, RUN_RERUN_ABLATION, RUN_DL_FACTORIAL = False, False, True

elif PROFILE == "quick":
    SEEDS = [42, 43, 44]
    MODELS = ("logreg", "decision_tree", "random_forest",
              "lightgbm", "xgboost", "catboost",
              "mlp", "resattdnn", "tabtransformer",
              "soft_vote", "weighted_vote", "gbmeta")
    BUDGET_CFG = replace(BUDGET, max_rows=60_000, n_estimators=300, patience=25,
                         n_oof_folds=3, max_epochs=20, n_trials=12)
    N_BOOT, LATENCY_REPS = 1000, 100
    ENERGY_SECONDS, ATTACK_SAMPLES = 0, 15
    COST_MODELS = ["decision_tree", "random_forest", "lightgbm", "xgboost", "catboost", "mlp"]
    FACTORIAL_GRID = [(i, s) for i in ("loss", "sampler", "none") for s in ("onecycle", "cosine")]
    RUN_HPO, RUN_RERUN_ABLATION, RUN_DL_FACTORIAL = True, True, True

else:  # "full"
    SEEDS = [42, 43, 44, 45, 46]
    MODELS = ("logreg", "decision_tree", "random_forest",
              "lightgbm", "xgboost", "catboost",
              "mlp", "resattdnn", "tabtransformer",
              "soft_vote", "weighted_vote", "gbmeta")
    BUDGET_CFG = replace(BUDGET, max_rows=200_000, n_estimators=600, patience=30,
                         n_oof_folds=5, max_epochs=40, n_trials=25)
    N_BOOT, LATENCY_REPS = 2000, 200
    ENERGY_SECONDS, ATTACK_SAMPLES = 35, 25
    COST_MODELS = ["decision_tree", "random_forest", "lightgbm", "xgboost", "catboost", "mlp"]
    FACTORIAL_GRID = [(i, s) for i in ("loss", "sampler", "none")
                             for s in ("onecycle", "cosine", "plateau")]
    RUN_HPO, RUN_RERUN_ABLATION, RUN_DL_FACTORIAL = True, True, True

# The seed used for every single-run artefact (confusion matrices, curves, cost).
MAIN_SEED = SEEDS[0]
TAG = f"paper-{PROFILE}"

# Rough cost model: (base fits + OOF fits) per dataset-seed, scaled by row count.
_fits = len(MODELS) - 3 + len(STACK_BASES) * BUDGET_CFG.n_oof_folds
_est = len(DATASETS) * len(SEEDS) * _fits * BUDGET_CFG.max_rows / 90_000
print(f"profile = {PROFILE}")
print(f"  {len(DATASETS)} datasets x {len(SEEDS)} seed(s) x {_fits} model fits")
print(f"  {BUDGET_CFG.max_rows:,} rows/dataset | {BUDGET_CFG.n_estimators} trees | "
      f"{BUDGET_CFG.n_oof_folds} OOF folds | bootstrap B={N_BOOT}")
print(f"  optional stages: HPO={RUN_HPO}  rerun-ablation={RUN_RERUN_ABLATION}  "
      f"DL-factorial={RUN_DL_FACTORIAL}  attack samples={ATTACK_SAMPLES}  "
      f"energy={ENERGY_SECONDS}s")
print(f"  rough training estimate: {_est:.0f}-{_est*1.8:.0f} min on a T4 "
      f"(plus ~3 min of downloads)")
print(f"  results -> {RESULTS_DIR/TAG}")
"""))

    cells.append(md(r"""
### 0.5 (Optional) persist results to Google Drive

A free Colab session ends without warning. Mounting Drive makes the run survive
a disconnect: re-running the training cell skips every model whose predictions
are already on disk. Skip this cell to keep everything in ephemeral storage.
"""))
    cells.append(code(r"""
USE_DRIVE = False   # set True to keep results across sessions

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    target = Path('/content/drive/MyDrive/gbmeta_v2')
    target.mkdir(parents=True, exist_ok=True)
    os.environ['GBMETA_RESULTS'] = str(target / 'results')
    os.environ['GBMETA_DATA']    = str(target / 'data')
    print('results and data will persist under', target)
    print('NOTE: restart the runtime and re-run from 0.3 so the new paths take effect.')
"""))

    # ---------------------------------------------------------------- data
    cells.append(md(r"""
## 1 · Data

### 1.1 Download

Five public benchmarks, smallest usable variant of each — about **680 MB** total.
No Kaggle credentials are needed. Exact files, row counts and the traps each one
carries are documented in `gbmeta/datasets/nids.py`.
"""))
    cells.append(code(r"""
from gbmeta.datasets.fetch import fetch, check, APPROX_MB

print("about to download ~%d MB" % sum(APPROX_MB.get(d, 0) for d in DATASETS))
for d in DATASETS:
    fetch(d)

for key, info in check().items():
    if key.replace('_full','') in DATASETS:
        print(('OK  ' if info['complete'] else 'MISS'), key,
              [f"{r['file'][:38]} {r['size_mb']}MB" for r in info['files']])
"""))

    cells.append(md(r"""
### 1.2 Leakage audit — run this *before* believing any accuracy number

Three probes per dataset:

1. **Exact-duplicate rate.** Duplicated flows that land on both sides of a split
   turn memorisation into apparent generalisation. They are removed globally
   (which *lowers* the scores) and the rate is reported.
2. **Train/test overlap** after splitting, on the encoded matrix.
3. **Single-feature probe.** A depth-3 stump on *one* feature at a time. If any
   single raw feature nearly solves a 15-class problem, the task is a lookup.

Two things this pipeline does that a naive one does not: the preprocessor is
fitted on training rows only, so no test statistic reaches it, and the binary
attack indicator that three of these benchmarks ship alongside the multi-class
label is dropped by the dataset spec before any model sees the matrix.
"""))
    cells.append(code(r"""
import numpy as np, pandas as pd
from gbmeta.config import RunConfig
from gbmeta.runner import build_data
from gbmeta.preprocess import leakage_audit

audit_rows = []
for d in DATASETS:
    cfg = RunConfig(dataset=d, seed=MAIN_SEED, budget=BUDGET_CFG, device=DEVICE,
                    models=MODELS, stack_bases=STACK_BASES, tag=TAG)
    ds, data = build_data(cfg)
    lk = leakage_audit(data, ds.provenance)
    probe = lk["top_single_feature_probes"][0]
    audit_rows.append({
        "dataset": ds.spec.display,
        "rows raw": ds.provenance.get("raw_rows"),
        "rows used": len(ds.y),
        "classes": ds.n_classes,
        "imbalance": round(max(np.bincount(ds.y)) / max(min(np.bincount(ds.y)), 1)),
        "dup rate": lk["exact_duplicate_rate_raw"],
        "train/test overlap": lk["train_test_overlap_rate"],
        "majority baseline": lk["majority_class_baseline"],
        "top single feature": probe["feature"][:26],
        "its accuracy": round(probe["single_feature_accuracy"], 4),
    })
    del ds, data

leak_df = pd.DataFrame(audit_rows)
display(leak_df)
"""))

    cells.append(md(r"""
> **Read the two right-hand columns first.** `majority baseline` is what a model
> that always predicts the largest class achieves. `its accuracy` is what a
> single feature achieves. Any headline accuracy should be compared against
> those, not against zero.
"""))

    # ---------------------------------------------------------------- training
    cells.append(md(r"""
## 2 · Train — five datasets × twelve models × N seeds

Resumable: every model's test/val/OOF probability matrix is written the moment
it exists, and a re-run skips anything already on disk. If the session dies,
re-run this cell.

**Stacking follows Wolpert's formulation.** The meta-learner is fitted on
*out-of-fold* predictions generated inside the training split. Fitting it on the
validation split instead would train it on the same rows every base model used
for early stopping, so it would learn to trust each model on data that model had
been tuned against.
"""))
    cells.append(code(r"""
import time
from gbmeta.runner import run_dataset

t_start = time.time()
for d in DATASETS:
    for s in SEEDS:
        cfg = RunConfig(dataset=d, seed=s, models=MODELS, stack_bases=STACK_BASES,
                        budget=BUDGET_CFG, device=DEVICE, tag=TAG)
        run_dataset(cfg)
        print(f"--- {d} seed{s} done | elapsed {(time.time()-t_start)/60:.1f} min ---\n")
print(f"TOTAL {(time.time()-t_start)/60:.1f} min")
"""))

    # ---------------------------------------------------------------- results
    cells.append(md(r"""
## 3 · Results and statistical significance

### 3.1 Per-dataset leaderboard with bootstrap intervals

Every metric carries a 95% percentile bootstrap interval, and all models share
one set of resample indices so the paired differences below are computed on the
same replicas.
"""))
    cells.append(code(r"""
from gbmeta.analysis import (collect_runs, results_table, significance_table,
                             cross_dataset_matrix, cross_dataset_significance,
                             seed_variance_table, seed_significance, leakage_table,
                             save_table, pretty)

RUNS = collect_runs(TAG)
print("datasets:", list(RUNS), "| seeds per dataset:",
      {k: sorted(v) for k, v in RUNS.items()})

TABLES = {}
for d in RUNS:
    run = RUNS[d][MAIN_SEED]
    tab = results_table(run, metric="macro_f1", B=N_BOOT)
    TABLES[d] = tab
    print(f"\n===== {d} =====")
    display(tab[["rank","model","accuracy","macro_f1","macro_f1_ci",
                 "balanced_accuracy","mcc","ece","train_seconds"]])
"""))

    cells.append(md(r"""
### 3.2 Is GB-META actually better? Paired bootstrap + McNemar

Two different questions:

* the **bootstrap interval** on the metric difference says how big the gap is
  and how uncertain — if it contains zero, the gap is not established;
* **McNemar** asks whether the two models disagree systematically on individual
  rows, which can be significant even when the metric gap is negligible.

Both are Holm-corrected across the family of comparisons.
"""))
    cells.append(code(r"""
SIG = {}
for d in RUNS:
    run = RUNS[d][MAIN_SEED]
    if "gbmeta" not in run["test_proba"]:
        continue
    sig = significance_table(run, reference="gbmeta", metric="macro_f1", B=N_BOOT)
    SIG[d] = sig
    print(f"\n===== {d}: everything vs GB-META =====")
    display(sig[["model","macro_f1","delta_vs_reference","ci_low","ci_high",
                 "ci_excludes_zero","mcnemar_p_holm","mcnemar_significant"]])
"""))

    cells.append(code(r"""
# One-line verdict per dataset: does GB-META beat the best single base learner
# by more than sampling noise?
verdicts = []
for d, sig in SIG.items():
    bases = sig[sig["key"].isin(STACK_BASES)]
    if bases.empty:
        continue
    best = bases.sort_values("macro_f1", ascending=False).iloc[0]
    verdicts.append({
        "dataset": d,
        "best single base": best["model"],
        "GB-META delta": round(best["delta_vs_reference"], 5),
        "95% CI": f"[{best['ci_low']:+.5f}, {best['ci_high']:+.5f}]",
        "established?": "yes" if best["ci_excludes_zero"] else "NO - within noise",
    })
verdict_df = pd.DataFrame(verdicts)
display(verdict_df)
"""))

    cells.append(md(r"""
### 3.3 Across datasets — Friedman, Nemenyi, Wilcoxon

A per-dataset win is weak evidence. The Friedman test on the rank matrix asks
whether the models differ at all across datasets; the Nemenyi post-hoc and its
critical-difference diagram say which pairs are separated.

**Power caveat, stated up front:** with N = 5 datasets the smallest attainable
two-sided Wilcoxon p-value is 0.0625, so that test *cannot* reject at α = 0.05
regardless of the data. A non-rejection there is a power limit, not evidence of
equivalence — the code emits this warning itself.
"""))
    cells.append(code(r"""
matrix = cross_dataset_matrix(RUNS, metric="macro_f1")
display(matrix.round(4))

xstat = cross_dataset_significance(RUNS, metric="macro_f1", reference="gbmeta")
print("\nFriedman:", {k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in xstat["friedman"].items() if k != "average_ranks"})
if "nemenyi" in xstat:
    print("critical difference:", round(xstat["nemenyi"]["critical_difference"], 3),
          "| significantly separated pairs:", xstat["nemenyi"]["n_significant"])
if "wilcoxon" in xstat and xstat["wilcoxon"].get("power_note"):
    print("\nWilcoxon power note:", xstat["wilcoxon"]["power_note"])
"""))
    cells.append(code(r"""
from gbmeta.plots import critical_difference_diagram, save, metric_bars_with_ci

if "nemenyi" in xstat:
    ranks = {pretty(k): v for k, v in xstat["nemenyi"]["average_ranks"].items()}
    fig = critical_difference_diagram(ranks, xstat["nemenyi"]["critical_difference"],
                                      len(xstat["datasets"]),
                                      "Macro-F1 ranks (Nemenyi, alpha=0.05)")
    save(fig, "fig_critical_difference")
    from IPython.display import Image, display as disp
    disp(Image(str(FIG_DIR / "fig_critical_difference.png"), width=760))
"""))

    cells.append(md(r"""
### 3.4 Seed-to-seed variance — the sanity check for any small improvement

If the standard deviation across seeds is larger than the improvement being
claimed, the improvement is not a result. The Nadeau–Bengio corrected paired
t-test is used rather than the naive paired t, whose variance term ignores the
overlap between training sets and rejects far too often.
"""))
    cells.append(code(r"""
var_df = seed_variance_table(RUNS, metric="macro_f1")
display(var_df[var_df["key"].isin(list(STACK_BASES) + ["gbmeta","soft_vote","random_forest"])]
        .round(5))

print("\nGB-META vs the strongest base learner, across seeds (Nadeau-Bengio corrected):")
for d in RUNS:
    base = (var_df[(var_df.dataset == d) & (var_df.key.isin(STACK_BASES))]
            .sort_values("mean", ascending=False))
    if base.empty:
        continue
    r = seed_significance(RUNS, d, "gbmeta", base.iloc[0]["key"])
    if "error" in r:
        print(f"  {d:14s} {r['error']}"); continue
    print(f"  {d:14s} vs {base.iloc[0]['key']:10s} "
          f"delta={r['mean_difference']:+.5f}  p={r['p_value']:.4f}  "
          f"{'SIGNIFICANT' if r['significant'] else 'not significant'}")
"""))

    # ---------------------------------------------------------------- ablation
    cells.append(md(r"""
## 4 · Ablation

### 4.1 Component ablation (free — no retraining)

Because out-of-fold and test probabilities are cached, removing a base learner
or swapping the combiner costs one logistic-regression fit. Each row carries a
paired bootstrap CI against the full stack, so "contribution" is a measured
effect with uncertainty.

The row that matters most to a reviewer is the last one: **the best single base
model, unstacked**. If the whole framework does not beat that by more than its
confidence interval, the framework's contribution is not accuracy.
"""))
    cells.append(code(r"""
from gbmeta.ablation import ablate_stack
from gbmeta.plots import ablation_forest_plot

ABL = {}
for d in RUNS:
    run_dir = RUNS[d][MAIN_SEED]["dir"]
    try:
        ab = ablate_stack(run_dir, metric="macro_f1", B=N_BOOT)
    except FileNotFoundError as e:
        print(f"{d}: {e}"); continue
    ABL[d] = ab
    rows = pd.DataFrame(ab["rows"])[["ablation","metric_macro_f1","delta",
                                     "ci_low","ci_high","significant","mcnemar_p"]]
    print(f"\n===== {d} =====\nVERDICT: {ab['verdict']}")
    display(rows.round(5))
"""))
    cells.append(code(r"""
from IPython.display import Image, display as disp
for d, ab in ABL.items():
    fig = ablation_forest_plot(ab["rows"], title=f"Component ablation — {d}")
    save(fig, f"fig_ablation_{d}")
    disp(Image(str(FIG_DIR / f"fig_ablation_{d}.png"), width=700))
"""))

    cells.append(md(r"""
### 4.2 Training-time ablations (require retraining)

Three switches that change how base models are trained, each written to its own
tag so nothing overwrites the headline results:

* **no class weighting** — sample weights disabled everywhere;
* **no deduplication** — duplicate rows kept, so train and test share exact
  copies. Expect the scores to go *up*; that rise is the size of the leak;
* **HPO** — Optuna-tuned base learners (§4.3).
"""))
    cells.append(code(r"""
from gbmeta.ablation import compare_runs

RUN_ABLATIONS = ["no_dedup"] if RUN_RERUN_ABLATION else []
ABL_DATASET = DATASETS[0]
if not RUN_ABLATIONS:
    print('skipped in this profile (retrains the whole roster). '
          'Set PROFILE="quick" to measure how much the duplicate rows are worth.')

for name in RUN_ABLATIONS:
    cfg = RunConfig(dataset=ABL_DATASET, seed=MAIN_SEED, models=MODELS,
                    stack_bases=STACK_BASES, budget=BUDGET_CFG, device=DEVICE,
                    tag=f"{TAG}-{name}",
                    dedup="none" if name == "no_dedup" else "global")
    run_dataset(cfg)
    cmp = compare_runs(RUNS[ABL_DATASET][MAIN_SEED]["dir"], cfg.run_dir, B=N_BOOT)
    print(f"\n===== {name} on {ABL_DATASET} "
          f"(a = headline / b = {name}) =====")
    display(pd.DataFrame(cmp["rows"]).round(5))
"""))

    cells.append(md(r"""
### 4.3 What Bayesian optimisation actually buys

A tuning study is easy to overstate: run the search, then build the final model
from library defaults, and the reported gain belongs to a model that was never
tuned. Here `tune_model` returns the parameters and `build_tuned` is the only
way to construct the tuned model, so the two cannot diverge. The objective is
validation macro-F1; the trial's own inner split does early stopping, so the
stopping point is not selected on the rows that pick the winner.
"""))
    cells.append(code(r"""
from gbmeta.hpo import tune_model, build_tuned
from gbmeta.models.base import ModelContext
from gbmeta.evaluate import compute_metrics

HPO_DATASET = DATASETS[0]
HPO_MODELS  = ["lightgbm"] if RUN_HPO else []
hpo_df = pd.DataFrame()
if not RUN_HPO:
    print(f'skipped in this profile ({BUDGET_CFG.n_trials} Optuna trials = '
          f'{BUDGET_CFG.n_trials} extra model fits). Set PROFILE="quick" to run it.')

cfg = RunConfig(dataset=HPO_DATASET, seed=MAIN_SEED, budget=BUDGET_CFG,
                device=DEVICE, models=MODELS, stack_bases=STACK_BASES, tag=TAG)
ds, data = build_data(cfg)
ctx = ModelContext(n_classes=data.n_classes, n_features=data.n_features,
                   seed=MAIN_SEED, device=DEVICE, budget=BUDGET_CFG,
                   class_weights=data.class_weights, feature_names=data.feature_names)
sw = data.sample_weights(data.y_train)

hpo_rows = []
for m in HPO_MODELS:
    res = tune_model(m, ctx, data.X_train, data.y_train, data.X_val, data.y_val,
                     n_trials=BUDGET_CFG.n_trials, sample_weight=sw)
    tuned = build_tuned(m, ctx, res).fit(data.X_train, data.y_train,
                                         data.X_val, data.y_val, sample_weight=sw)
    test_m = compute_metrics(data.y_test, tuned.predict_proba(data.X_test), data.n_classes)
    base_m = RUNS[HPO_DATASET][MAIN_SEED]["records"][m]["metrics"]
    hpo_rows.append({
        "model": m, "trials": res.n_trials, "pruned": res.n_pruned,
        "val macro-F1 default": round(res.default_score, 5),
        "val macro-F1 tuned":   round(res.best_score, 5),
        "val gain":             round(res.improvement, 5),
        "TEST macro-F1 default": round(base_m["macro_f1"], 5),
        "TEST macro-F1 tuned":   round(test_m["macro_f1"], 5),
        "TEST gain":             round(test_m["macro_f1"] - base_m["macro_f1"], 5),
        "seconds": round(res.seconds),
    })
hpo_df = pd.DataFrame(hpo_rows)
display(hpo_df)
print("\nReport the TEST gain, not the validation gain: a validation number "
      "quoted among test results overstates what tuning contributes.")
"""))

    # ---------------------------------------------------------------- deployment
    cells.append(md(r"""
## 5 · Deployment cost

Batch-1 p50/p95/p99 latency, a throughput sweep, serialised model size, and —
where a GPU is present — energy from NVML's hardware millijoule counter
(`nvmlDeviceGetTotalEnergyConsumption`, supported from Volta onward, so a T4
qualifies). Sampling `nvmlDeviceGetPowerUsage` instead would be measurably wrong.

Two things deliberately **not** reported:

* **CPU energy.** RAPL/`powercap` is not readable inside a Colab VM, so any CPU
  figure would be a TDP guess presented as a measurement.
* **Raspberry Pi / Jetson latency.** No calibrated Cortex-A72-vs-x86 ratio exists
  for tree ensembles. A single-threaded ONNX Runtime measurement is reported as
  a labelled *proxy* instead.
"""))
    cells.append(code(r"""
from gbmeta.deploy import profile_model, pin_single_thread
from gbmeta.models.base import build_model

COST_DATASET = DATASETS[0]      # COST_MODELS comes from the profile

cfg = RunConfig(dataset=COST_DATASET, seed=MAIN_SEED, budget=BUDGET_CFG, device=DEVICE,
                models=MODELS, stack_bases=STACK_BASES, tag=TAG)
ds, data = build_data(cfg)
ctx = ModelContext(n_classes=data.n_classes, n_features=data.n_features, seed=MAIN_SEED,
                   device=DEVICE, budget=BUDGET_CFG, class_weights=data.class_weights,
                   feature_names=data.feature_names)
sw = data.sample_weights(data.y_train)

from gbmeta.models.base import available_models
fitted, profiles = {}, []
for m in COST_MODELS:
    if m not in available_models():
        print("skip", m, "(backend missing)"); continue
    mdl = build_model(m, ctx).fit(data.X_train, data.y_train, data.X_val, data.y_val,
                                  sample_weight=sw)
    fitted[m] = mdl
    profiles.append(profile_model(mdl, data.X_test, name=m, n_reps=LATENCY_REPS,
                                  measure_energy_seconds=ENERGY_SECONDS,
                                  onnx_dir=RESULTS_DIR / TAG / "onnx"))
"""))
    cells.append(code(r"""
# Assemble the stacked ensemble from the models just fitted, so its cost includes
# every base model's forward pass -- the honest end-to-end inference path.
from gbmeta.ensemble import StackedEnsemble, compute_oof, MetaLearner

stack_keys = [m for m in STACK_BASES if m in fitted]
if len(stack_keys) >= 2:
    oof = [compute_oof(k, ctx, data.X_train, data.y_train,
                       BUDGET_CFG.n_oof_folds, sw).oof_proba for k in stack_keys]
    comb = MetaLearner(seed=MAIN_SEED).fit(oof, data.y_train)
    stack = StackedEnsemble(ctx, {k: fitted[k] for k in stack_keys}, comb)
    profiles.append(profile_model(stack, data.X_test, name="gbmeta", n_reps=LATENCY_REPS,
                                  measure_energy_seconds=ENERGY_SECONDS))
"""))
    cells.append(code(r"""
cost_rows = []
for p in profiles:
    d = p.as_dict()
    rec = RUNS[COST_DATASET][MAIN_SEED]["records"].get(d["model"], {})
    cost_rows.append({
        "model": pretty(d["model"]),
        "macro-F1": round(rec.get("metrics", {}).get("macro_f1", float("nan")), 4),
        "p50 ms (batch 1)": round(d["latency"]["latency_batch1_p50_ms"], 4),
        "p99 ms (batch 1)": round(d["latency"]["latency_batch1_p99_ms"], 4),
        "peak samples/s": int(d["latency"]["peak_throughput_sps"]),
        "at batch": d["latency"]["peak_throughput_batch"],
        "size MB (gz)": d["footprint"].get("gzip_mb"),
        "trees": d["footprint"].get("n_trees") or d["footprint"].get("n_params"),
        "GPU": d["environment"]["uses_gpu"],
        "J / 1k inf.": (round(d["energy"].get("joules_per_1000_inferences"), 4)
                        if d.get("energy", {}).get("joules_per_1000_inferences") else None),
        "ONNX 1-thread p50 ms": (round(d["onnx"]["cpu_single_thread"]["p50_ms"], 4)
                                 if d.get("onnx", {}).get("cpu_single_thread", {}).get("measured")
                                 else None),
    })
cost_df = pd.DataFrame(cost_rows).sort_values("p50 ms (batch 1)")
display(cost_df)
"""))
    cells.append(code(r"""
from gbmeta.plots import cost_quality_scatter
pts = [{"x": r["p50 ms (batch 1)"], "y": r["macro-F1"], "label": r["model"]}
       for r in cost_rows if r["macro-F1"] == r["macro-F1"]]
if pts:
    save(cost_quality_scatter(pts, title=f"Accuracy vs inference cost — {COST_DATASET}"),
         "fig_cost_quality")
    disp(Image(str(FIG_DIR / "fig_cost_quality.png"), width=620))
"""))

    # ---------------------------------------------------------------- robustness
    cells.append(md(r"""
## 6 · Robustness and concept drift

### 6.1 Perturbation sweeps

Accuracy against increasing Gaussian noise and against random feature masking,
applied **only to features an attacker could plausibly control**. The immutable
mask (destination-side counters, server timers, backward-flow statistics) is a
stated modelling assumption, published with the result — the NIDS threat-model
literature is explicit that no universal mutable/immutable list exists.
"""))
    cells.append(code(r"""
from gbmeta.robustness import (gaussian_noise_sweep, feature_masking_sweep,
                               immutable_mask, robustness_summary)
from gbmeta.plots import degradation_curve

imm = immutable_mask(data.feature_names)
mutable = ~imm
print(f"{mutable.sum()}/{len(mutable)} features treated as attacker-mutable")

noise_curves, mask_curves, rob_rows = {}, {}, []
for m, mdl in list(fitted.items())[:6]:
    g = gaussian_noise_sweep(mdl.predict_proba, data.X_test, data.y_test,
                             data.n_classes, mutable=mutable)
    f = feature_masking_sweep(mdl.predict_proba, data.X_test, data.y_test,
                              data.n_classes, mutable=mutable)
    noise_curves[pretty(m)] = g.curve("macro_f1")
    mask_curves[pretty(m)]  = f.curve("macro_f1")
    rob_rows.append({"model": pretty(m), **{f"noise_{k}": v for k, v in
                                            robustness_summary(g).items() if k != "kind"}})
display(pd.DataFrame(rob_rows).round(4))

save(degradation_curve(noise_curves, "Gaussian noise sigma (robust-scaled units)",
                       title="Robustness to constrained feature noise"), "fig_robust_noise")
save(degradation_curve(mask_curves, "Fraction of mutable features zeroed",
                       title="Robustness to partial telemetry loss"), "fig_robust_masking")
disp(Image(str(FIG_DIR / "fig_robust_noise.png"), width=560))
"""))

    cells.append(md(r"""
### 6.2 Black-box evasion (HopSkipJump)

ZOO is deliberately not used: it estimates gradients by finite differences, and a
tree ensemble is piecewise constant, so the estimate is exactly zero and the
attack returns the input unchanged. HopSkipJump is the decision-based attack that
does work on trees, and its `mask` argument enforces the immutability constraint
exactly.

Cost is ~5 s per sample, so this runs on a small random subsample and is reported
with a Wilson confidence interval rather than as a dataset-level claim. The result
is a *feature-space* evasion rate: it shows the decision boundary is reachable, not
that a corresponding packet sequence can be built.
"""))
    cells.append(code(r"""
from gbmeta.robustness import hopskipjump_attack

atk_rows = []
for m in (["lightgbm", "random_forest"] if ATTACK_SAMPLES else []):
    if m not in fitted:
        continue
    r = hopskipjump_attack(fitted[m], data.X_test, data.y_test,
                           n_samples=ATTACK_SAMPLES, mutable=mutable)
    r["model"] = pretty(m)
    atk_rows.append(r)
    print(m, "->", {k: v for k, v in r.items() if k not in ("caveat",)})
if atk_rows and atk_rows[0].get("ran"):
    print("\nCaveat carried into the paper:", atk_rows[0]["caveat"])
"""))

    cells.append(md(r"""
### 6.3 Concept drift

ToN-IoT is the only one of the five with a clean Unix-epoch timestamp
(Edge-IIoTset's `frame.time` was corrupted at publication — an unquoted Wireshark
string was comma-split, so the month and day are gone). For ToN-IoT the model is
trained on the earliest traffic and evaluated on consecutive later windows, and
summarised with **AUT**, the time-decay-aware metric from the TESSERACT protocol.
The remaining datasets get simulated covariate shift, clearly labelled as such.
"""))
    cells.append(code(r"""
from gbmeta.robustness import temporal_decay, simulated_shift, feature_drift_report

from gbmeta.datasets import get_spec

# Only ToN-IoT survived publication with a usable timestamp; if it is not in
# DATASETS, fall back to a random split and report simulated shift instead of
# pretending a temporal split happened.
timed = [d for d in DATASETS if get_spec(d).timestamp_col]
DRIFT_DATASET = timed[0] if timed else DATASETS[0]
cfg_t = RunConfig(dataset=DRIFT_DATASET, seed=MAIN_SEED, budget=BUDGET_CFG, device=DEVICE,
                  models=MODELS, stack_bases=STACK_BASES, tag=f"{TAG}-temporal")
ds_t, data_t = build_data(cfg_t, temporal=bool(timed))
HAS_TIME = data_t.splits.mode == "temporal" and ds_t.timestamps is not None
print(f"drift dataset: {DRIFT_DATASET} | split mode: {data_t.splits.mode} | {data_t.splits.meta}")
if not HAS_TIME:
    print("no usable timestamp in the selected datasets -- section 6.3 reports "
          "simulated covariate shift only, and says so.")

ctx_t = ModelContext(n_classes=data_t.n_classes, n_features=data_t.n_features,
                     seed=MAIN_SEED, device=DEVICE, budget=BUDGET_CFG,
                     class_weights=data_t.class_weights, feature_names=data_t.feature_names)
mdl_t = build_model("lightgbm", ctx_t).fit(
    data_t.X_train, data_t.y_train, data_t.X_val, data_t.y_val,
    sample_weight=data_t.sample_weights(data_t.y_train))

# How much did the input distribution actually move between train and test?
drift = feature_drift_report(data_t.X_train, data_t.X_test, data_t.feature_names)
print(f"\nfeatures with PSI > 0.25: {drift['n_psi_above_0.25']}/{drift['n_features']} "
      f"(mean PSI {drift['mean_psi']:.3f})")
display(pd.DataFrame(drift["top_drifting_features"]).round(4).head(8))
"""))
    cells.append(code(r"""
ts_test = ds_t.timestamps.iloc[data_t.splits.test].to_numpy() if HAS_TIME else None
if ts_test is not None:
    dec = temporal_decay(mdl_t.predict_proba, data_t.X_test, data_t.y_test,
                         ts_test, data_t.n_classes, n_windows=6)
    print(f"AUT = {dec['aut']:.4f} | first window {dec['first']:.4f} -> "
          f"last {dec['last']:.4f} (decay {dec['decay']:+.4f})")
    display(pd.DataFrame(dec["windows"])[["window","n_rows","accuracy","macro_f1"]].round(4))
    save(degradation_curve({"LightGBM (temporal split)":
                            ([w["window"] for w in dec["windows"]],
                             [w["macro_f1"] for w in dec["windows"]])},
                           "Consecutive test window (chronological)",
                           title=f"Concept drift — {DRIFT_DATASET} (AUT={dec['aut']:.4f})"),
         "fig_drift_temporal")
    disp(Image(str(FIG_DIR / "fig_drift_temporal.png"), width=560))
else:
    print("skipping temporal decay: no timestamps. Simulated shift follows.")

sim = simulated_shift(mdl_t.predict_proba, data_t.X_test, data_t.y_test,
                      data_t.n_classes, feature_names=data_t.feature_names)
display(pd.DataFrame(sim["rows"])[["severity","accuracy","macro_f1"]].round(4))
"""))

    cells.append(md(r"""
### 6.4 The deep-learning failure, as a controlled experiment

v1 observed ResAttDNN and TabTransformer collapsing to ~0.6% accuracy and
attributed it to a `WeightedRandomSampler` × `OneCycleLR` interaction. That was a
post-hoc explanation of an accident, and the notebook's actual scheduler
configuration contradicted it.

Here the two factors are orthogonal switches, so the claim becomes a factorial
experiment whose cells can be reported. If one cell collapses and the others do
not, the interaction is demonstrated; if none collapse, the v1 explanation was
wrong and the paper should say so.
"""))
    cells.append(code(r"""
from gbmeta.plots import training_dynamics

FACTORIAL_MODEL = "resattdnn"
hist, fac_rows = {}, []
if not RUN_DL_FACTORIAL:
    print('skipped in this profile.')
for imb, sch in (FACTORIAL_GRID if RUN_DL_FACTORIAL else []):
    c = ctx.with_params(imbalance=imb, scheduler=sch)
    m = build_model(FACTORIAL_MODEL, c).fit(data.X_train, data.y_train,
                                            data.X_val, data.y_val, sample_weight=sw)
    mt = compute_metrics(data.y_test, m.predict_proba(data.X_test), data.n_classes)
    key = f"{imb} + {sch}"
    hist[key] = m.history
    fac_rows.append({"imbalance": imb, "scheduler": sch,
                     "accuracy": round(mt["accuracy"], 4),
                     "macro_f1": round(mt["macro_f1"], 4),
                     "epochs": len(m.history),
                     "collapsed": mt["accuracy"] < 2.0 / data.n_classes})
    del m
fac_df = pd.DataFrame(fac_rows)
display(fac_df)
save(training_dynamics(hist, f"{FACTORIAL_MODEL}: sampler x scheduler"), "fig_dl_factorial")
disp(Image(str(FIG_DIR / "fig_dl_factorial.png"), width=640))
print("\nCollapsed cells:", fac_df[fac_df.collapsed][["imbalance","scheduler"]].to_dict("records")
      or "none - the v1 explanation is not reproduced under controlled conditions")
"""))

    # ---------------------------------------------------------------- figures
    cells.append(md(r"""
## 7 · Confusion matrix, ROC, PR, calibration, per-class

All at 600 dpi PNG plus vector PDF. The confusion matrix is row-normalised: with
a 72% majority class the raw counts are one bright cell and say nothing about the
minority classes, which are the only place the models differ.
"""))
    cells.append(code(r"""
import json
from gbmeta.plots import (confusion_matrix_figure, roc_figure, pr_figure,
                          reliability_figure, class_support_vs_f1)
from gbmeta.evaluate import confusion_from_labels

FIG_DATASET = DATASETS[0]
FIG_MODEL   = "gbmeta"
run = RUNS[FIG_DATASET][MAIN_SEED]
classes = run["classes"]
p = run["test_proba"][FIG_MODEL]
cm = confusion_from_labels(run["y_test"], p.argmax(1), len(classes))

save(confusion_matrix_figure(cm, classes, f"{pretty(FIG_MODEL)} — {FIG_DATASET}"),
     f"fig_confusion_{FIG_DATASET}")
curves = json.loads(Path(run["dir"] / "curves" / f"{FIG_MODEL}.json").read_text())
save(roc_figure(curves["roc"], f"ROC (one-vs-rest) — {FIG_DATASET}"), f"fig_roc_{FIG_DATASET}")
save(pr_figure(curves["pr"], f"Precision-recall — {FIG_DATASET}"), f"fig_pr_{FIG_DATASET}")
save(reliability_figure(curves["reliability"], pretty(FIG_MODEL)), f"fig_calibration_{FIG_DATASET}")
save(class_support_vs_f1(run["records"][FIG_MODEL]["per_class"],
                         f"Per-class F1 vs support — {FIG_DATASET}"), f"fig_support_{FIG_DATASET}")

for n in [f"fig_confusion_{FIG_DATASET}", f"fig_roc_{FIG_DATASET}",
          f"fig_calibration_{FIG_DATASET}", f"fig_support_{FIG_DATASET}"]:
    disp(Image(str(FIG_DIR / f"{n}.png"), width=600))
"""))
    cells.append(code(r"""
# Per-class table, with underpowered classes flagged rather than quietly reported.
pc = pd.DataFrame(run["records"][FIG_MODEL]["per_class"])
display(pc.round(4))
weak = pc[pc.underpowered]
if len(weak):
    print(f"\n{len(weak)} class(es) have fewer than 30 test rows: "
          f"{list(weak['class'])}. Their F1 values are reported but carry no "
          f"statistical weight, and the paper must say so.")
"""))

    # ---------------------------------------------------------------- export
    cells.append(md(r"""
## 8 · Export everything

Tables as CSV + Markdown + LaTeX (`booktabs`, ready to `\input`), figures as
PDF + PNG, plus the full run manifest so every number is traceable to an exact
environment and file hash.
"""))
    cells.append(code(r"""
from gbmeta.analysis import save_table
from gbmeta.utils import write_json, environment_manifest

save_table(leak_df, "table1_leakage_audit",
           caption="Dataset provenance and leakage audit.", label="tab:leakage")
for d, t in TABLES.items():
    save_table(t.drop(columns=["key"]), f"table2_results_{d}",
               caption=f"Model comparison on {d} with 95\\% bootstrap intervals.",
               label=f"tab:results_{d}")
for d, s in SIG.items():
    save_table(s.drop(columns=["key"]), f"table3_significance_{d}",
               caption=f"Paired bootstrap and McNemar tests against GB-META on {d}.",
               label=f"tab:sig_{d}")
save_table(matrix.reset_index(), "table4_cross_dataset",
           caption="Macro-F1 across five IDS datasets.", label="tab:cross")
save_table(var_df, "table5_seed_variance",
           caption="Seed-to-seed variation of macro-F1.", label="tab:seeds")
for d, ab in ABL.items():
    save_table(pd.DataFrame(ab["rows"]).drop(columns=["base_models"], errors="ignore"),
               f"table6_ablation_{d}",
               caption=f"Component ablation on {d}.", label=f"tab:abl_{d}")
save_table(cost_df, "table7_deployment_cost",
           caption="Inference cost and model size.", label="tab:cost")
if len(hpo_df):
    save_table(hpo_df, "table8_hpo", caption="Effect of Bayesian hyper-parameter optimisation.",
               label="tab:hpo")

write_json(TAB_DIR / "cross_dataset_significance.json", xstat)
write_json(TAB_DIR / "environment.json", environment_manifest())
print("tables ->", TAB_DIR)
print("figures ->", FIG_DIR)
"""))
    cells.append(code(r"""
import shutil
bundle = Path("/content/gbmeta_v2_paper_assets") if Path("/content").exists() else Path("paper_assets")
shutil.make_archive(str(bundle), "zip", root_dir=str(TAB_DIR.parent))
print("archive:", bundle.with_suffix(".zip"),
      f"({bundle.with_suffix('.zip').stat().st_size/1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(str(bundle.with_suffix(".zip")))
except Exception as e:
    print("(download only works in Colab)", e)
"""))

    # ---------------------------------------------------------------- summary
    cells.append(md(r"""
## 9 · What to write in the revision

Read the outputs above before writing, and let them decide the claim. Two
outcomes are possible, and both are publishable — but only one of them is
publishable *honestly* in each case.

**If §3.2 shows GB-META's advantage over the best single base learner has a
confidence interval that includes zero** (the likely outcome on saturated
benchmarks), then the accuracy claim cannot carry the paper. Say so, and move the
contribution to what the artefacts do establish:

- a leakage audit that quantifies duplicate rate, train/test overlap and
  single-feature predictability for five benchmarks (§1.2) — including the
  finding that a widely used UNSW-NB15 partition is ~39% exact duplicates;
- the first statistically-tested comparison on these datasets: McNemar, paired
  bootstrap CIs, Friedman + Nemenyi, Holm correction, with the power limits
  stated (§3);
- a deployment cost profile with honest measurement (§5), including what is *not*
  measurable in the environment;
- an ablation showing which components actually contribute (§4) — and, if the
  meta-learner does not beat a soft vote, saying that;
- a controlled factorial replacing v1's post-hoc deep-learning failure
  explanation (§6.4).

**If §3.2 shows the interval excludes zero on several datasets**, the accuracy
claim survives — but state the effect size in macro-F1 with its interval, not as
"+0.03 pp accuracy", and put the deployment cost of the stack next to it.

Either way, the audit numbers above are what the manuscript should quote,
and they should be quoted from `paper/tables/`, never from memory.

And two methodological fixes are now in the pipeline itself: the meta-learner is
fitted on out-of-fold predictions rather than the validation split, and the
binary ground-truth column `Attack_label` is dropped instead of being fed to the
models as a feature.
"""))
    return cells


# --------------------------------------------------------------------------
def main() -> int:
    print("building the Colab notebook ...")
    b64 = bundle_package()
    nb = {
        "cells": build_cells(b64),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"  wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KiB, "
          f"{len(nb['cells'])} cells: {n_code} code / {len(nb['cells'])-n_code} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
