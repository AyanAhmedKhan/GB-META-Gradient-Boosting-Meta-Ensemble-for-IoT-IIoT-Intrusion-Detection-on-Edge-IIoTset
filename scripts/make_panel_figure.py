"""One two-column figure carrying four diagnostics at once.

A confusion matrix, ROC-AUC, precision-recall curves and evidence that the
reported probabilities mean something are all needed to characterise a
classifier beyond a single score. Four separate floats
would cost most of a page in a six-page paper; a 2x2 panel costs one third of one
and is easier to read as a single object.

Everything is drawn from cached artefacts -- no model is refitted.

    python scripts/make_panel_figure.py --dataset edge_iiotset --model gbmeta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gbmeta.analysis import collect_runs, pretty  # noqa: E402
from gbmeta.config import FIG_DIR  # noqa: E402
from gbmeta.evaluate import confusion_from_labels  # noqa: E402
from gbmeta.plots import PALETTE, save, use_paper_style  # noqa: E402
from gbmeta.utils import LOG, setup_logging  # noqa: E402


def build(dataset: str, model: str, tag: str, seed: int, max_curves: int = 6):
    import matplotlib.pyplot as plt

    runs = collect_runs(tag)
    run = runs[dataset][seed]
    classes = [str(c) for c in run["classes"]]
    proba = run["test_proba"][model]
    y = run["y_test"]
    curves = json.loads((run["dir"] / "curves" / f"{model}.json").read_text(encoding="utf-8"))

    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.9))
    (ax_cm, ax_roc), (ax_pr, ax_cal) = axes

    # (a) row-normalised confusion matrix ---------------------------------
    cm = confusion_from_labels(y, proba.argmax(1), len(classes)).astype(float)
    cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    n = len(classes)
    ax_cm.set_xticks(range(n)); ax_cm.set_yticks(range(n))
    short = [c if len(c) <= 12 else c[:11] + "." for c in classes]
    ax_cm.set_xticklabels(short, rotation=90, fontsize=5.2)
    ax_cm.set_yticklabels(short, fontsize=5.2)
    ax_cm.set_xlabel("Predicted", fontsize=7.5)
    ax_cm.set_ylabel("True", fontsize=7.5)
    ax_cm.set_title("(a) Confusion matrix (row-normalised)", fontsize=8)
    ax_cm.grid(False)
    fig.colorbar(im, ax=ax_cm, fraction=0.045, pad=0.02).ax.tick_params(labelsize=6)

    # (b) ROC, log FPR -- the axis an IDS actually operates on -------------
    roc = sorted(curves["roc"].items(), key=lambda kv: kv[1].get("auc", 0))[:max_curves]
    for i, (name, c) in enumerate(roc):
        ax_roc.plot(np.clip(np.asarray(c["fpr"], float), 1e-6, 1), c["tpr"],
                    color=PALETTE[i % len(PALETTE)], lw=1.1,
                    label=f"{name[:16]} ({c.get('auc', float('nan')):.3f})")
    ax_roc.plot([1e-6, 1], [1e-6, 1], ls=":", lw=0.8, color="#888888")
    ax_roc.set_xscale("log"); ax_roc.set_xlim(1e-6, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False positive rate (log)", fontsize=7.5)
    ax_roc.set_ylabel("True positive rate", fontsize=7.5)
    ax_roc.set_title("(b) ROC, one-vs-rest (worst 6 classes)", fontsize=8)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=5.4)

    # (c) precision-recall -- the right view under extreme imbalance -------
    pr = sorted(curves["pr"].items(), key=lambda kv: kv[1].get("ap", 0))[:max_curves]
    for i, (name, c) in enumerate(pr):
        ax_pr.plot(c["recall"], c["precision"], color=PALETTE[i % len(PALETTE)], lw=1.1,
                   label=f"{name[:16]} ({c.get('ap', float('nan')):.3f})")
    ax_pr.set_xlim(0, 1.02); ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall", fontsize=7.5); ax_pr.set_ylabel("Precision", fontsize=7.5)
    ax_pr.set_title("(c) Precision-recall (worst 6 classes)", fontsize=8)
    ax_pr.legend(loc="lower left", frameon=False, fontsize=5.4)

    # (d) reliability -----------------------------------------------------
    rel = curves["reliability"]
    conf = np.array([b["mean_confidence"] for b in rel["bins"]])
    acc = np.array([b["empirical_accuracy"] for b in rel["bins"]])
    ax_cal.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.9, label="perfect")
    ax_cal.plot(conf, acc, "o-", color=PALETTE[0], ms=3, lw=1.2, label="model")
    ax_cal.fill_between(conf, acc, conf, alpha=0.15, color=PALETTE[1])
    ax_cal.set_xlim(0, 1.02); ax_cal.set_ylim(0, 1.02)
    ax_cal.set_xlabel("Confidence", fontsize=7.5)
    ax_cal.set_ylabel("Empirical accuracy", fontsize=7.5)
    ax_cal.set_title(f"(d) Reliability (ECE = {rel['ece']:.4f})", fontsize=8)
    ax_cal.legend(loc="upper left", frameon=False, fontsize=6)

    for ax in (ax_roc, ax_pr, ax_cal):
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.6)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="edge_iiotset")
    ap.add_argument("--model", default="gbmeta")
    ap.add_argument("--tag", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    setup_logging()
    fig = build(a.dataset, a.model, a.tag, a.seed)
    save(fig, f"fig_panel_{a.dataset}_{a.model}")
    LOG.info("panel written to %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
