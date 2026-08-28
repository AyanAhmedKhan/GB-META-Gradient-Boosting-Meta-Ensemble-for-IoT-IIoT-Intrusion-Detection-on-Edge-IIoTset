"""Execute the generated Colab notebook end to end, at toy scale.

A notebook that merely *parses* proves nothing -- the failures that matter are
integration ones: a renamed key, a moved attribute, a stage that assumes a
backend is present. This harness runs every code cell in order against real
(small) data, with three substitutions:

* ``pip``/``%%capture`` cells are skipped -- dependency installation is the one
  thing that cannot be tested off-Colab;
* the configuration cell is overridden with a toy profile;
* IPython's ``display``/``Image`` are stubbed.

Run it after every ``scripts/build_notebook.py``::

    python tests/test_notebook.py --datasets nslkdd
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "GB_META_v2_Colab.ipynb"


def _skip(src: str) -> bool:
    s = src.lstrip()
    return s.startswith("%%") or s.startswith("!pip") or "drive.mount" in s


# Appended *after* the notebook's own configuration cell, never instead of it.
# Replacing the cell would let the harness silently drift from the notebook --
# which is exactly the bug this file exists to catch. Every optional stage is
# forced on, so the toy run still exercises the code the fast profile skips.
CONFIG_OVERRIDE = """
# ---- test harness overrides ----
DATASETS = __TEST_DATASETS__
SEEDS = [42, 43]
MODELS = ("logreg", "decision_tree", "random_forest", "lightgbm", "xgboost",
          "catboost", "mlp", "resattdnn", "soft_vote", "weighted_vote", "gbmeta")
STACK_BASES = ("lightgbm", "random_forest", "decision_tree")
BUDGET_CFG = replace(BUDGET, max_rows=4000, min_rows_per_class=40, n_estimators=40,
                     patience=8, n_oof_folds=3, max_epochs=4, n_trials=3, batch_size=256)
N_BOOT, LATENCY_REPS = 150, 5
ENERGY_SECONDS, ATTACK_SAMPLES = 0, 0
COST_MODELS = ["decision_tree", "lightgbm", "random_forest"]
FACTORIAL_GRID = [("loss", "cosine"), ("sampler", "onecycle")]
RUN_HPO = RUN_RERUN_ABLATION = RUN_DL_FACTORIAL = True
MAIN_SEED = SEEDS[0]
TAG = "nbtest"
print("TEST OVERRIDES:", DATASETS, SEEDS, BUDGET_CFG.max_rows)
"""


def build_namespace(datasets):
    """Globals for the executed cells, with the IPython surface stubbed out."""
    import pandas as pd

    def display(*a, **k):
        for x in a:
            if isinstance(x, pd.DataFrame):
                print(x.head(8).to_string()[:1500])
            else:
                print(str(x)[:400])

    class _Image:
        def __init__(self, path, **k):
            self.path = path
            assert Path(path).exists(), f"figure not written: {path}"

        def __repr__(self):
            return f"<Image {Path(self.path).name}>"

    ns = {"__name__": "__main__", "display": display, "disp": display, "Image": _Image}
    ns["__TEST_DATASETS__"] = datasets
    return ns


def run(datasets, stop_on_error: bool, only_upto: int | None):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ns = build_namespace(datasets)
    failures, executed, skipped = [], 0, 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if only_upto is not None and i > only_upto:
            break
        src = "\n".join(cell["source"])
        if _skip(src):
            skipped += 1
            continue
        # Run the real configuration cell, then shrink it.
        if "PROFILE = " in src and "BUDGET_CFG" in src:
            src = src + "\n" + CONFIG_OVERRIDE.replace("__TEST_DATASETS__", repr(datasets))
        # The IPython import inside cells would shadow our stubs.
        src = src.replace("from IPython.display import Image, display as disp", "pass")
        src = src.replace("from IPython.display import Image, display as disp\n", "")

        print(f"\n{'='*70}\nCELL {i}\n{'-'*70}\n{src[:220]}...\n{'-'*70}")
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            executed += 1
        except Exception as exc:
            failures.append((i, f"{type(exc).__name__}: {exc}"))
            print(f"!!! CELL {i} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=6)
            if stop_on_error:
                break

    print(f"\n{'='*70}\nexecuted {executed} cells, skipped {skipped}, failed {len(failures)}")
    for i, msg in failures:
        print(f"  cell {i}: {msg}")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["nslkdd"])
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--upto", type=int, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)
    return run(a.datasets, a.stop_on_error, a.upto)


if __name__ == "__main__":
    sys.exit(main())
