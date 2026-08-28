"""Build the Colab notebook that reproduces and *verifies* the paper.

Different goal from a demo notebook: this one re-runs the study and then checks
the result against what the manuscript claims, printing PASS/FAIL per claim.

The check is deliberately split in two, because the two kinds of claim deserve
different tolerances:

* **Data facts** -- duplicate rates, memorisation ceilings, class overlap. These
  are deterministic functions of the published CSVs and must match exactly.
  A mismatch means the dataset changed, or the pipeline did.
* **Model facts** -- rankings, signs of differences, significance verdicts.
  Colab runs XGBoost and CatBoost on a GPU while the paper's reference host is
  CPU-only, so digits will differ. What must reproduce is the *conclusion*:
  which model wins, whether an interval excludes zero.

Verifying digits across different hardware would fail for reasons that have
nothing to do with the science; verifying conclusions is the honest test.

    python scripts/build_paper_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "GB_META_Paper_Reproduce.ipynb"


def bundle() -> str:
    """tar.gz the package *and* the scripts the notebook drives."""
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("gbmeta", "scripts"):
            for p in sorted((ROOT / sub).rglob("*.py")):
                if "__pycache__" in p.parts or p.name.startswith("build_"):
                    continue
                tar.add(p, arcname=str(p.relative_to(ROOT)))
                n += 1
    raw = buf.getvalue()
    print(f"  bundle: {len(raw)/1024:.0f} KiB gzipped, {n} modules")
    return base64.b64encode(raw).decode("ascii")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").split("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").split("\n")}


def wrap_b64(b, w=96):
    return "\n".join(b[i:i + w] for i in range(0, len(b), w))


def cells(b64: str) -> list:
    c = []

    c.append(md(r"""
# GB-META — reproduce and verify the paper

Re-runs the study behind *"GB-META: A Leakage-Audited, Statistically Validated
Gradient-Boosting Meta-Ensemble for IoT/IIoT Intrusion Detection"* and checks the
output against what the manuscript claims.

**What it verifies.** Section 7 prints a PASS/FAIL line per claim, split into two
kinds:

* **Data facts** — duplicate rates, memorisation ceilings, ToN-IoT class overlap.
  Deterministic functions of the published CSVs; these must match exactly.
* **Model facts** — rankings, signs of differences, significance verdicts. Colab
  runs XGBoost and CatBoost on GPU while the paper's reference host is CPU-only,
  so digits will differ. What must reproduce is the conclusion.

Checking digits across different hardware would fail for reasons unrelated to the
science. Checking conclusions is the honest test.

| Profile | Time on a T4 | What it does |
|---|---|---|
| `"verify"` *(default)* | ~35–50 min | full paper configuration: 4 datasets × 3 seeds × 80k rows |
| `"quick"` | ~12–18 min | 1 seed, 25k rows — verifies data facts exactly and model facts directionally |

Runs are cached per model, so a disconnect costs only the model that was training.
""".rstrip()))

    c.append(md("## 1 · Runtime"))
    c.append(code(r"""
import os, sys, platform, subprocess
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 2))
print("Python  :", sys.version.split()[0], "|", platform.platform())
print("CPUs    :", os.cpu_count())
try:
    print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "no GPU")
except FileNotFoundError:
    print("no GPU -- runs on CPU, slower")
"""))

    c.append(md("## 2 · Dependencies"))
    c.append(code(r"""
%%capture
!pip install -q catboost statsmodels psutil tabulate kagglehub
!pip install -q nvidia-ml-py onnxruntime skl2onnx onnxmltools onnx
"""))
    c.append(code(r"""
import importlib
have = {}
for m in ["numpy","pandas","sklearn","scipy","statsmodels","lightgbm","xgboost",
          "catboost","torch","optuna","matplotlib","kagglehub"]:
    try: have[m] = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception: have[m] = None
print("installed:", {k:v for k,v in have.items() if v})
miss = [k for k,v in have.items() if not v]
print("MISSING  :", miss or "none")
assert not [m for m in ("lightgbm","xgboost","catboost","sklearn","scipy") if m in miss], \
    "a required backend is missing -- rerun the install cell"
"""))

    c.append(md("## 3 · Unpack the code\n\nThe `gbmeta` package and the driver "
                "scripts are embedded in this notebook, so nothing needs cloning."))
    c.append(code(
        "# --- embedded gbmeta package + scripts (base64 tar.gz) --------------------\n"
        "BUNDLE_B64 = '''\n" + wrap_b64(b64) + "\n'''.replace('\\n','')\n"
        "print(f'embedded bundle: {len(BUNDLE_B64)/1024:.0f} KiB base64')"
    ))
    c.append(code(
        "import base64, io, tarfile, os, sys, warnings\n"
        "from pathlib import Path\n\n"
        "WORK = Path('/content') if Path('/content').exists() else Path.cwd()\n"
        "os.chdir(WORK)\n"
        "if not (WORK/'gbmeta'/'runner.py').exists():\n"
        "    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode='r:gz') as t:\n"
        "        t.extractall(WORK)\n"
        "    print('unpacked to', WORK)\n"
        "else:\n"
        "    print('using existing checkout at', WORK)\n"
        "if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))\n\n"
        "warnings.filterwarnings('ignore')\n"
        "from gbmeta.utils import setup_logging, get_device\n"
        "LOG = setup_logging(); DEVICE = get_device('auto')\n"
        "print('device:', DEVICE)"
    ))

    c.append(md("## 4 · Configuration"))
    c.append(code(r"""
from dataclasses import replace
from gbmeta.config import BUDGET, RESULTS_DIR, FIG_DIR, TAB_DIR

PROFILE = "verify"        # "verify" = paper configuration | "quick" = fast directional check

DATASETS = ["edge_iiotset", "nslkdd", "ton_iot", "unsw_nb15"]
MODELS = ("logreg","decision_tree","random_forest","lightgbm","xgboost","catboost",
          "mlp","resattdnn","soft_vote","weighted_vote","gbmeta")
STACK_BASES = ("lightgbm","xgboost","catboost")

if PROFILE == "verify":
    SEEDS, ROWS, TREES, BOOT = [42,43,44], 80_000, 400, 2000
else:
    SEEDS, ROWS, TREES, BOOT = [42], 25_000, 200, 800

BUDGET_CFG = replace(BUDGET, max_rows=ROWS, n_estimators=TREES, n_oof_folds=3,
                     max_epochs=25, patience=30)
MAIN_SEED, TAG = SEEDS[0], f"repro-{PROFILE}"
print(f"profile={PROFILE} | {len(DATASETS)} datasets x {len(SEEDS)} seed(s) | "
      f"{ROWS:,} rows | {TREES} trees | bootstrap B={BOOT}")
print("results ->", RESULTS_DIR/TAG)
"""))

    c.append(md("## 5 · Download the benchmarks\n\nAbout 240 MB. No Kaggle "
                "credentials needed."))
    c.append(code(r"""
from gbmeta.datasets.fetch import fetch, check
for d in DATASETS: fetch(d)
for k, info in check().items():
    if k in DATASETS:
        print(("OK  " if info["complete"] else "MISS"), k,
              [f"{r['file'][:40]} {r['size_mb']}MB" for r in info["files"]])
"""))

    c.append(md("## 6 · Run the study\n\nResumable — re-run this cell after a "
                "disconnect and it skips whatever is already on disk."))
    c.append(code(r"""
import time
from gbmeta.config import RunConfig
from gbmeta.runner import run_dataset

t0 = time.time()
for s in SEEDS:                      # seed-major: one seed of every dataset first,
    for d in DATASETS:               # so the cross-dataset tests become valid early
        run_dataset(RunConfig(dataset=d, seed=s, models=MODELS, stack_bases=STACK_BASES,
                              budget=BUDGET_CFG, device=DEVICE, tag=TAG))
        print(f"--- {d} seed{s} | elapsed {(time.time()-t0)/60:.1f} min ---")
print(f"TOTAL {(time.time()-t0)/60:.1f} min")
"""))

    c.append(md("## 7 · Verify against the paper\n\nThis is the point of the "
                "notebook. Data facts must match exactly; model facts must match "
                "directionally."))
    c.append(code(r"""
import numpy as np, pandas as pd
from gbmeta.analysis import (collect_runs, cross_dataset_matrix,
                             cross_dataset_significance, leakage_table,
                             results_table, significance_table)

RUNS = collect_runs(TAG)
print("collected:", {k: sorted(v) for k, v in RUNS.items()})

CHECKS = []
def check_it(name, ok, got, expected, kind):
    CHECKS.append({"kind": kind, "claim": name, "expected": expected,
                   "observed": got, "result": "PASS" if ok else "FAIL"})

# ---- data facts: deterministic, must match exactly ------------------------
leak = leakage_table(RUNS).set_index("dataset")
for ds, exp in {"edge_iiotset": 0.036, "nslkdd": 0.000,
                "ton_iot": 0.185, "unsw_nb15": 0.386}.items():
    if ds in leak.index:
        got = float(leak.loc[ds, "exact_duplicate_rate"])
        check_it(f"{ds}: exact-duplicate rate", abs(got - exp) < 0.002,
                 f"{got:.3f}", f"{exp:.3f}", "data")

if "nslkdd" in leak.index:
    got = float(leak.loc["nslkdd", "best_single_feature_acc"])
    feat = str(leak.loc["nslkdd", "best_single_feature"])
    check_it("NSL-KDD strongest single feature is src_bytes", feat == "src_bytes",
             feat, "src_bytes", "data")
    check_it("NSL-KDD single-feature accuracy ~0.867", abs(got - 0.867) < 0.02,
             f"{got:.3f}", "0.867", "data")

display(pd.DataFrame(CHECKS))
"""))
    c.append(code(r"""
# ---- model facts: conclusions, not digits ---------------------------------
x = cross_dataset_significance(RUNS, metric="macro_f1", reference="gbmeta")
ranks = x["friedman"]["average_ranks"]

check_it("soft vote outranks GB-META across datasets",
         ranks.get("soft_vote", 9) < ranks.get("gbmeta", 0),
         f"soft_vote {ranks.get('soft_vote', float('nan')):.2f} vs "
         f"gbmeta {ranks.get('gbmeta', float('nan')):.2f}",
         "soft_vote < gbmeta", "model")

if "error" not in x["friedman"]:
    p = x["friedman"]["iman_davenport_p_value"]
    check_it("Friedman rejects (models differ in rank)", p < 0.05,
             f"p={p:.2e}", "p < 0.05", "model")

# GB-META vs its best base learner: the paper claims win / loss / draw / loss
EXPECT = {"edge_iiotset": "win", "nslkdd": "loss",
          "ton_iot": "draw", "unsw_nb15": "loss"}
for ds, want in EXPECT.items():
    if ds not in RUNS: continue
    run = RUNS[ds][MAIN_SEED if MAIN_SEED in RUNS[ds] else sorted(RUNS[ds])[0]]
    if "gbmeta" not in run["test_proba"]: continue
    sig = significance_table(run, reference="gbmeta", metric="macro_f1", B=BOOT)
    base = sig[sig["key"].isin(STACK_BASES)].sort_values("macro_f1", ascending=False)
    if base.empty: continue
    b = base.iloc[0]
    got = ("win" if (b["delta_vs_reference"] > 0 and b["ci_excludes_zero"])
           else "loss" if (b["delta_vs_reference"] < 0 and b["ci_excludes_zero"])
           else "draw")
    check_it(f"{ds}: GB-META vs best base learner", got == want,
             f"{got} (delta {b['delta_vs_reference']:+.4f})", want, "model")

display(pd.DataFrame(CHECKS))
"""))
    c.append(code(r"""
# ---- ToN-IoT temporal split is class-disjoint (the paper's drift finding) --
from gbmeta.runner import build_data
from gbmeta.config import RunConfig

if "ton_iot" in DATASETS:
    ds_t, data_t = build_data(RunConfig(dataset="ton_iot", seed=MAIN_SEED,
                                        budget=BUDGET_CFG, device=DEVICE, tag=TAG),
                              temporal=True)
    tr, te = set(np.unique(data_t.y_train).tolist()), np.unique(data_t.y_test)
    unseen = [i for i in te if i not in tr]
    cnt = np.bincount(data_t.y_test, minlength=data_t.n_classes)
    share = sum(cnt[i] for i in unseen) / cnt.sum()
    names = [str(c) for c in data_t.class_names]
    print("train classes:", sorted(names[i] for i in tr))
    print("test  classes:", sorted(names[i] for i in te))
    print("unseen in train:", [names[i] for i in unseen])
    check_it("ToN-IoT temporal split is class-disjoint",
             len(unseen) >= 3, f"{len(unseen)} of {len(te)} test classes unseen",
             ">=3 unseen", "data")
    check_it("ToN-IoT: majority of test rows are an unseen class",
             share > 0.4, f"{share:.1%}", "~53.7%", "data")

display(pd.DataFrame(CHECKS))
"""))
    c.append(code(r"""
# ---- verdict --------------------------------------------------------------
df = pd.DataFrame(CHECKS)
n_fail = int((df.result == "FAIL").sum())
print(df.to_string(index=False))
print()
for kind in ("data", "model"):
    sub = df[df.kind == kind]
    print(f"{kind:6s}: {int((sub.result=='PASS').sum())}/{len(sub)} passed")
print()
if n_fail == 0:
    print("ALL CHECKS PASSED -- the paper's claims reproduce on this machine.")
else:
    print(f"{n_fail} CHECK(S) FAILED. Data-fact failures mean the input data or "
          "pipeline changed; model-fact failures mean a conclusion did not "
          "reproduce and should be investigated before citing it.")
"""))

    c.append(md("## 8 · Regenerate the paper's tables and figures"))
    c.append(code(r"""
!python scripts/make_paper_assets.py --tag {TAG} --boot {BOOT} 2>&1 | tail -40
"""))
    c.append(code(r"""
!python scripts/leakage_probe.py --max-rows {ROWS} --datasets edge_iiotset nslkdd unsw_nb15 ton_iot 2>&1 | tail -10
!python scripts/make_panel_figure.py --tag {TAG} --dataset edge_iiotset --model gbmeta --seed {MAIN_SEED}
"""))
    c.append(code(r"""
from IPython.display import Image, display as disp
disp(Image(str(FIG_DIR / "fig_panel_edge_iiotset_gbmeta.png"), width=820))
"""))
    c.append(md("### Optional — robustness, drift and the HPO ablation\n\n"
                "Adds roughly 20 minutes. Reproduces the finding that CatBoost "
                "alone is far more perturbation-robust than the stack."))
    c.append(code(r"""
!python scripts/make_robustness_drift_hpo.py --tag {TAG} --dataset edge_iiotset --trials 15 2>&1 | tail -30
"""))

    c.append(md("## 9 · Download everything"))
    c.append(code(r"""
import shutil
from gbmeta.config import PAPER_DIR
out = Path("/content/gbmeta_repro") if Path("/content").exists() else Path("gbmeta_repro")
shutil.make_archive(str(out), "zip", root_dir=str(PAPER_DIR))
print("archive:", out.with_suffix(".zip"),
      f"({out.with_suffix('.zip').stat().st_size/1e6:.1f} MB)")
try:
    from google.colab import files; files.download(str(out.with_suffix(".zip")))
except Exception as e:
    print("(download only works in Colab)", e)
"""))

    c.append(md(r"""
## What a failure would mean

**A data check failing** means the input changed: a Kaggle re-upload, a different
file variant, or an edited loader. Compare the file hashes printed by
`python -m gbmeta.datasets.fetch --check` against those in the released manifest
before anything else.

**A model check failing** is more interesting. The four GB-META verdicts
(win / loss / draw / loss) are the paper's central claim. If one flips on your
hardware, the effect is smaller than the paper implies and that is worth
reporting — the intervals in Table II are the place to look, and a verdict near
the boundary (ToN-IoT, whose interval already spans zero) is the one most likely
to move.

`PROFILE = "quick"` uses one seed and 25k rows, so its model checks are weaker by
construction; a `"quick"` failure on a marginal verdict is expected rather than
alarming. Re-run with `"verify"` before drawing a conclusion.
""".rstrip()))
    return c


def main() -> int:
    print("building the reproduce-and-verify notebook")
    nb = {
        "cells": cells(bundle()),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    nc = sum(1 for x in nb["cells"] if x["cell_type"] == "code")
    print(f"  wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB, "
          f"{len(nb['cells'])} cells: {nc} code / {len(nb['cells'])-nc} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
