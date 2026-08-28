# GB-META

A stacked meta-ensemble for multi-class IoT/IIoT intrusion detection, and the
leakage-audited evaluation protocol that establishes where its advantage holds.

GB-META combines LightGBM, XGBoost and CatBoost through a logistic-regression
meta-learner fitted on out-of-fold log-odds. On Edge-IIoTset it attains the best
macro-F1 of eleven models and the lowest calibration error of any model tested;
an ablation attributes that result to the learned combination step rather than to
the choice of base learners. Across four benchmarks the advantage is
benchmark-specific, and the repository reports the tests that establish its
limits as carefully as the result itself.

Everything here runs end to end on one free Google Colab T4.

---

## Quick start

Open **`notebooks/GB_META_v2_Colab.ipynb`** in Colab and run it top to bottom.
The notebook is self-contained: the `gbmeta` package is embedded as a base64
tarball, the datasets download without Kaggle credentials, and every table and
figure is written to `paper/`.

```
Runtime -> Change runtime type -> T4 GPU
```

| `PROFILE` | Wall clock, free T4 | What you get |
|---|---|---|
| `"fast"` *(default)* | ~15-25 min | every table and figure; 1 seed, 15k rows/dataset |
| `"quick"` | ~30-45 min | 3 seeds, 60k rows; adds HPO, the dedup ablation, black-box evasion |
| `"full"` | ~3-5 h | the published configuration; 5 seeds, 200k rows/dataset |

All three produce the complete results, significance tests, ablation, cost table
and figures. They differ in sample size, seed count, and which optional stages
run.

All are **resumable**: each model's predictions are cached the moment they exist
and a re-run skips whatever is already on disk, so moving up a profile later
wastes nothing. Mount Drive (cell 0.5) to survive a disconnected session.

Prefer the command line:

```bash
pip install -r requirements.txt
python -m gbmeta.datasets.fetch --all
python -m gbmeta.runner --dataset edge_iiotset --seed 42 --tag main
```

A laptop-sized end-to-end check that needs no GPU and no downloads:

```bash
python -m gbmeta.runner --dataset synthetic --smoke
```

---

## What the evaluation covers

| Concern | Method | Where |
|---|---|---|
| Data leakage | global exact-duplicate removal, encoded train/test overlap with a memorisation ceiling, single-feature predictability probe | `gbmeta/preprocess.py` |
| Significance within a benchmark | McNemar's exact test, paired percentile bootstrap over shared resample indices | `gbmeta/stats.py` |
| Significance across benchmarks | Friedman with Iman-Davenport, Nemenyi critical difference | `gbmeta/stats.py` |
| Significance across seeds | Nadeau-Bengio corrected paired *t*-test, Holm-corrected | `gbmeta/stats.py` |
| Component attribution | leave-one-base-out, combiner ablation, meta-feature transform, each with a bootstrap interval | `gbmeta/ablation.py` |
| Deployment cost | batch-1 p50/p95/p99 latency, throughput sweep, serialised size, GPU energy via NVML, ONNX export | `gbmeta/deploy.py` |
| Robustness and drift | constrained perturbation sweeps, HopSkipJump, PSI/KS, temporal AUT | `gbmeta/robustness.py` |
| Calibration | expected calibration error and reliability diagrams | `gbmeta/evaluate.py` |

Two design decisions carry most of the weight:

* **Training and analysis are separate.** Training writes probability matrices;
  significance tests, ablations, curves and cost tables all read them back.
  Re-running the entire analysis under a different test costs seconds, not hours.
* **Every optional backend degrades gracefully.** A machine without CatBoost,
  ART, ONNX or `river` still produces a complete, smaller results table, and the
  run manifest records exactly what was skipped and why.

---

## Datasets

The smallest usable variant of each, about 680 MB in total, no credentials
needed. `gbmeta/datasets/nids.py` records each file's licence text, required
citation, and the file-level traps it carries.

| Key | File | Rows | Classes | Notes |
|---|---|---|---|---|
| `edge_iiotset` | `ML-EdgeIIoT-dataset.csv` | 157,800 | 15 | `frame.time` is corrupt at source, so it cannot support a drift study |
| `nslkdd` | `KDDTrain+.txt` | 125,973 | 23 | headerless; field 43 (`difficulty`) is a function of the label and is dropped |
| `unsw_nb15` | `UNSW_NB15_testing-set.csv` | 175,341 | 10 | the Kaggle mirror **swaps** the train/test filenames; 38.6% exact duplicates |
| `ton_iot` | `Train_Test_Network.csv` | 461,043 | 10 | the only clean epoch timestamp, so it carries the temporal analysis |
| `cicids2017` | 5 day-files (MachineLearningCVE) | 692k+ | 11 | latin-1 labels, a duplicate column name, leading spaces |
| `botiot` | `..._10_best_Testing.csv` | 733,705 | 5 | optional; Normal = 107 rows, Theft = 14, flagged rather than hidden |

```bash
python -m gbmeta.datasets.fetch --all     # download
python -m gbmeta.datasets.fetch --check   # what is on disk, with hashes
```

Three of these ship a binary attack indicator alongside the multi-class label,
and NSL-KDD ships a column derived from the label. All are dropped by the dataset
specification before any model sees the feature matrix.

---

## Command line

```bash
# one dataset, one seed
python -m gbmeta.runner --dataset edge_iiotset --seed 42 --tag main

# the full study
python scripts/run_study.py --tag paper --seeds 42 43 44

# budget overrides
python -m gbmeta.runner --dataset ton_iot --max-rows 60000 --n-estimators 300 --oof-folds 3

# chronological split (ToN-IoT only)
python -m gbmeta.runner --dataset ton_iot --temporal --tag temporal
```

---

## Package layout

```
gbmeta/
  config.py        seeds, budgets, paths -- every number that affects a result
  utils.py         determinism, timing, RAM/VRAM probes, run manifests
  datasets/
    base.py        dataset contract, dedup, stratified capping, provenance
    nids.py        the real benchmarks and their file-level traps
    fetch.py       credential-free download of exactly the needed files
    synthetic.py   toy dataset for CI
  preprocess.py    fold-fitted pipeline and the leakage audit
  models/          uniform learner interface: trees, neural, registry
  ensemble.py      out-of-fold stacking and the heuristics it must beat
  hpo.py           Optuna, with a pruner that actually prunes
  evaluate.py      metrics, calibration, bootstrap intervals
  stats.py         McNemar, Nadeau-Bengio, Friedman, Nemenyi, Wilcoxon, Holm
  ablation.py      component ablation from cached predictions, no retraining
  deploy.py        latency, throughput, size, GPU energy, ONNX
  robustness.py    perturbation sweeps, HopSkipJump, PSI/KS, temporal AUT
  analysis.py      cached artefacts -> CSV / Markdown / LaTeX tables
  plots.py         publication figures (PDF and 600 dpi PNG)
  runner.py        resumable experiment driver
```

---

## Manuscript

`paper/` holds the LaTeX source, its bibliography, and the generated tables and
figures. Every table is produced from the result CSVs by `scripts/`, so the
manuscript cannot drift from the artefacts it reports:

```bash
python scripts/make_paper_tex.py        # tables from result CSVs
python scripts/verify_paper_numbers.py  # every prose number against its source
python scripts/check_ieee_conventions.py
cd paper && pdflatex GB_META_v2 && bibtex GB_META_v2 && pdflatex GB_META_v2
```

---

## Rebuilding the notebook

The notebook is a build artefact. Edit the package, then:

```bash
python scripts/build_notebook.py                  # re-embeds the package
python tests/test_notebook.py --datasets nslkdd   # executes every cell for real
```

`tests/test_notebook.py` runs all 31 executable cells against real data at toy
scale. That is the check that matters: a notebook that only *parses* proves
nothing.

---

## Citation

If you use this code or the protocol, please cite the paper:

```bibtex
@inproceedings{sharma2026gbmeta,
  author    = {Sharma, Jalaj and Khan, Ayan Ahmed and Sengar, Kaushal Pratap},
  title     = {{GB-META}: Gradient-Boosting Meta-Ensemble for {IoT/IIoT}
               Intrusion Detection on {Edge-IIoTset}},
  booktitle = {Proc. IEEE Conf.},
  year      = {2026}
}
```

Datasets carry their own terms; all require citation and permit academic use.
CICIDS2017 in particular mandates citing Sharafaldin et al. (ICISSP 2018)
regardless of the Kaggle mirror's licence tag.

## Licence

Code is released under the MIT Licence (`LICENSE`). Dataset licences are
separate and are recorded per dataset in `gbmeta/datasets/nids.py`.
