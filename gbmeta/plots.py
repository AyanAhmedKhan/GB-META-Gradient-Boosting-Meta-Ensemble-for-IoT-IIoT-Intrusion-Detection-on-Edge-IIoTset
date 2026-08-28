"""Publication-quality figures.

Reviewer minor comment: "improve Figure 1 and Figure 2 resolution". Every figure
here is written twice -- a vector PDF (what IEEE actually wants; it never
pixelates) and a 600 dpi PNG for drafts. Nothing is rasterised except image-like
content such as the confusion-matrix heatmap.

Matplotlib only, no seaborn: one less dependency to pin, and full control over
the colour map, which is chosen to be colour-blind safe and to survive greyscale
printing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FIG_DIR
from .utils import LOG, optional_import

mpl = optional_import("matplotlib")
if mpl is not None:
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
else:  # pragma: no cover
    plt = None

#: Okabe-Ito: distinguishable under all common colour-vision deficiencies and
#: still separable when printed in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000"]

RC = {
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,   # embed TrueType, not Type-3: required by IEEE eXpress
    "ps.fonttype": 42,
}


def use_paper_style() -> None:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting")
    plt.rcParams.update(RC)


def save(fig, name: str, out_dir=None, formats=("pdf", "png")) -> list:
    out_dir = Path(out_dir or FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    LOG.info("figure written: %s", ", ".join(str(p.name) for p in written))
    return written


# --------------------------------------------------------------------------
# Core result figures
# --------------------------------------------------------------------------
def confusion_matrix_figure(cm, class_names, title="", normalise=True, annotate_max=20):
    """Row-normalised confusion matrix.

    Row normalisation, not raw counts: with a 72% majority class the raw matrix
    is a single bright cell and communicates nothing about the minority classes,
    which are the only ones where the models differ.
    """
    use_paper_style()
    cm = np.asarray(cm, dtype=float)
    if normalise:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(min(0.42 * n + 2.2, 9), min(0.38 * n + 2.0, 8)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else None, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    if title:
        ax.set_title(title)
    ax.grid(False)
    if n <= annotate_max:
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if normalise else f"{int(v)}",
                            ha="center", va="center", fontsize=6,
                            color="white" if v > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                 label="Recall" if normalise else "Count")
    return fig


def roc_figure(curves: dict, title="", max_classes=8, macro_only=False):
    """One-vs-rest ROC curves on a log-scaled false-positive axis.

    A linear FPR axis is useless when every curve sits above 0.999 TPR; the log
    axis is where an IDS actually operates (false positives per million flows).
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    items = sorted(curves.items(), key=lambda kv: kv[1].get("auc", 0))[:max_classes]
    for i, (name, c) in enumerate(items):
        fpr = np.clip(np.asarray(c["fpr"], dtype=float), 1e-6, 1)
        ax.plot(fpr, c["tpr"], color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC {c.get('auc', float('nan')):.4f})")
    ax.plot([1e-6, 1], [1e-6, 1], ls=":", lw=0.8, color="#888888", label="chance")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    return fig


def pr_figure(curves: dict, title="", max_classes=8):
    """Precision-recall curves -- the right view under extreme class imbalance."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    items = sorted(curves.items(), key=lambda kv: kv[1].get("ap", 0))[:max_classes]
    for i, (name, c) in enumerate(items):
        ax.plot(c["recall"], c["precision"], color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AP {c.get('ap', float('nan')):.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    return fig


def reliability_figure(reliability: dict, title=""):
    """Reliability diagram with the confidence histogram underneath."""
    use_paper_style()
    bins = reliability["bins"]
    conf = np.array([b["mean_confidence"] for b in bins])
    acc = np.array([b["empirical_accuracy"] for b in bins])
    cnt = np.array([b["count"] for b in bins], dtype=float)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(3.6, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.9, label="perfect calibration")
    ax.plot(conf, acc, "o-", color=PALETTE[0], ms=3.5, label="model")
    ax.fill_between(conf, acc, conf, alpha=0.15, color=PALETTE[1], label="calibration gap")
    ax.set_ylabel("Empirical accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(f"{title}  (ECE = {reliability['ece']:.4f})" if title
                 else f"ECE = {reliability['ece']:.4f}")
    ax2.bar(conf, cnt / cnt.sum(), width=1 / max(len(bins), 1) * 0.8,
            color=PALETTE[0], alpha=0.6)
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Fraction")
    ax2.set_xlim(0, 1.02)
    return fig


# --------------------------------------------------------------------------
# Statistics figures
# --------------------------------------------------------------------------
def critical_difference_diagram(average_ranks: dict, cd: float, n_datasets: int, title=""):
    """Demsar-style critical-difference diagram.

    Rank 1 (best) sits on the left. Models joined by a horizontal bar are *not*
    significantly different at the stated alpha. On a saturated benchmark a
    single bar spanning every model is the honest headline result, and this
    figure is the clearest way to say so.
    """
    use_paper_style()
    names = sorted(average_ranks, key=lambda k: average_ranks[k])
    ranks = [float(average_ranks[n]) for n in names]
    k = len(names)
    lo, hi = float(np.floor(min(ranks))), float(np.ceil(max(ranks)))
    if hi - lo < 1:
        hi = lo + 1
    span = hi - lo

    half = (k + 1) // 2
    row_h = 0.26
    n_rows = half
    # Cliques computed first: they determine how much room is needed above.
    cliques, i = [], 0
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    # Clique bars sit between the axis and the first label row, so the rank
    # tick labels (above the axis) never collide with them.
    clique_top, clique_step = 0.10, 0.085
    first_row = clique_top + clique_step * max(len(cliques) - 1, 0) + 0.30
    y_bottom = first_row + row_h * (n_rows - 1) + 0.22
    fig_h = 1.05 + 0.26 * n_rows + 0.085 * len(cliques)

    fig, ax = plt.subplots(figsize=(5.6, fig_h))
    # Generous horizontal margins hold the model labels outside the rank axis.
    ax.set_xlim(lo - 0.45 * span, hi + 0.45 * span)
    ax.set_ylim(y_bottom, -0.62)
    ax.axis("off")

    # Rank axis.
    ax.plot([lo, hi], [0, 0], color="black", lw=1.1, zorder=2)
    for t in np.arange(lo, hi + 1e-9, 1.0):
        ax.plot([t, t], [0, -0.06], color="black", lw=0.9, zorder=2)
        ax.text(t, -0.09, f"{t:.0f}", ha="center", va="bottom", fontsize=8)

    # Critical-difference ruler, drawn above the tick labels.
    y_cd = -0.38
    ax.plot([lo, lo + cd], [y_cd, y_cd], color="black", lw=1.8, solid_capstyle="butt")
    for x in (lo, lo + cd):
        ax.plot([x, x], [y_cd - 0.035, y_cd + 0.035], color="black", lw=1.0)
    ax.text(lo + cd / 2, y_cd - 0.06, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    # Model stems: best half labelled on the left, worst half on the right.
    x_left, x_right = lo - 0.10 * span, hi + 0.10 * span
    for i, (nm, r) in enumerate(zip(names, ranks)):
        left = i < half
        row = i if left else (k - 1 - i)
        y = first_row + row_h * row
        x_end = x_left if left else x_right
        ax.plot([r, r], [0, y], color="black", lw=0.8, zorder=1)
        ax.plot([r, x_end], [y, y], color="black", lw=0.8, zorder=1)
        ax.text(x_end + (-0.02 * span if left else 0.02 * span), y,
                f"{nm} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=8)

    # Clique bars, stacked just below the axis.
    for depth, (i, j) in enumerate(cliques):
        y = clique_top + clique_step * depth
        ax.plot([ranks[i] - 0.015 * span, ranks[j] + 0.015 * span], [y, y],
                color=PALETTE[1], lw=3.0, solid_capstyle="round", zorder=3)

    if title:
        ax.set_title(f"{title}   (N = {n_datasets} datasets)", fontsize=9, pad=6)
    return fig


def ablation_forest_plot(rows, metric_label="Δ macro-F1", title=""):
    """Forest plot of ablation effects with bootstrap intervals.

    Reading it takes one second: any bar whose interval crosses the zero line is
    a component whose contribution could not be distinguished from noise.
    """
    use_paper_style()
    rows = [r for r in rows if r.get("kind") != "reference"]
    rows = sorted(rows, key=lambda r: r["delta"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 0.28 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        colour = PALETTE[1] if r.get("significant") else "#999999"
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=colour, lw=1.6,
                solid_capstyle="butt")
        ax.plot(r["delta"], i, "o", ms=4, color=colour)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["ablation"] for r in rows])
    ax.set_xlabel(f"{metric_label}  (vs full model, 95% bootstrap CI)")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if title:
        ax.set_title(title)
    handles = [
        plt.Line2D([], [], color=PALETTE[1], lw=1.6, marker="o", ms=4, label="CI excludes 0"),
        plt.Line2D([], [], color="#999999", lw=1.6, marker="o", ms=4, label="CI includes 0"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    return fig


def metric_bars_with_ci(rows, metric_label="Macro-F1", title="", baseline=None):
    """Per-model metric with its bootstrap interval, sorted best-first."""
    use_paper_style()
    rows = sorted(rows, key=lambda r: r["point"], reverse=True)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(4.8, 0.3 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        ax.barh(i, r["point"], color=PALETTE[0], alpha=0.75, height=0.6)
        ax.plot([r["lo"], r["hi"]], [i, i], color="black", lw=1.1)
    if baseline is not None:
        ax.axvline(baseline, color=PALETTE[1], ls="--", lw=1.0,
                   label=f"majority-class baseline ({baseline:.3f})")
        ax.legend(loc="lower right", frameon=False)
    ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    if title:
        ax.set_title(title)
    return fig


# --------------------------------------------------------------------------
# Robustness / drift / cost figures
# --------------------------------------------------------------------------
def degradation_curve(series: dict, xlabel, ylabel="Macro-F1", title="", logx=False):
    """Metric vs perturbation strength, one line per model."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, (name, (xs, ys)) in enumerate(series.items()):
        ax.plot(xs, ys, "o-", ms=3, color=PALETTE[i % len(PALETTE)], label=name)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.5)
    return fig


def cost_quality_scatter(points, xlabel="Median latency per sample (ms, log)",
                         ylabel="Macro-F1", title=""):
    """Accuracy against inference cost -- the figure a deployment reviewer wants.

    Models on the upper-left frontier are the only defensible choices; a stack
    that buys +0.001 macro-F1 for 3x the latency lands visibly off it.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, p in enumerate(points):
        ax.scatter(p["x"], p["y"], s=42, color=PALETTE[i % len(PALETTE)],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(p["label"], (p["x"], p["y"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=7)
    pts = sorted(points, key=lambda p: p["x"])
    frontier, best = [], -np.inf
    for p in pts:
        if p["y"] > best:
            best = p["y"]
            frontier.append(p)
    if len(frontier) > 1:
        ax.plot([p["x"] for p in frontier], [p["y"] for p in frontier],
                ls="--", lw=0.9, color="#666666", zorder=1, label="Pareto frontier")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig


def training_dynamics(histories: dict, title="", metric="val_macro_f1"):
    """Training curves for the deep baselines, with the learning-rate trace.

    This is the replacement for v1's Figure 2: instead of one collapsed run, it
    overlays the factorial cells so the sampler x scheduler interaction is
    visible rather than asserted.
    """
    use_paper_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
    )
    for i, (name, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h.get(metric, np.nan) for h in hist], "-o", ms=2.5,
                color=PALETTE[i % len(PALETTE)], label=name)
        ax2.plot(ep, [h["lr"] for h in hist], color=PALETTE[i % len(PALETTE)], lw=1.0)
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(frameon=False, fontsize=7)
    if title:
        ax.set_title(title)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("LR")
    ax2.set_yscale("log")
    return fig


def class_support_vs_f1(per_class_rows, title=""):
    """Per-class F1 against test support, log-x.

    Makes the underpowered classes obvious: a class with 19 test rows cannot
    support a claim about its F1 either way, and marking those points stops a
    reader from over-reading them.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    sup = np.array([r["support"] for r in per_class_rows], dtype=float)
    f1 = np.array([r["f1"] for r in per_class_rows], dtype=float)
    weak = np.array([r.get("underpowered", False) for r in per_class_rows])
    ax.scatter(sup[~weak], f1[~weak], s=34, color=PALETTE[0], label="adequate support", zorder=3)
    ax.scatter(sup[weak], f1[weak], s=42, facecolor="none", edgecolor=PALETTE[1],
               linewidth=1.2, label="support < 30 (underpowered)", zorder=3)
    for r in per_class_rows:
        ax.annotate(r["class"], (r["support"], r["f1"]), textcoords="offset points",
                    xytext=(4, 3), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Test-set support (rows, log scale)"); ax.set_ylabel("Per-class F1")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    if title:
        ax.set_title(title)
    return fig
