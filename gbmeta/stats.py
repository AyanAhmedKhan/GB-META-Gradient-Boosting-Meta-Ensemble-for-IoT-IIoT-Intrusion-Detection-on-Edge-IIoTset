"""Statistical significance testing.

Implements the tests the machine-learning methodology literature prescribes for
each comparison a benchmark study actually makes:

* **Within one dataset**, on one shared test set: McNemar's exact test on the
  disagreement counts (Dietterich 1998, the recommended test for a single test
  set), plus a paired bootstrap interval on the metric difference.
* **Across datasets**: Friedman on the rank matrix with the Iman-Davenport
  correction, Nemenyi post-hoc and a critical-difference diagram (Demsar 2006),
  and Wilcoxon signed-rank with Holm correction for pairwise claims.
* **Across repeated resamples of one dataset**: the Nadeau-Bengio corrected
  paired t-test, because the naive paired t-test over CV folds ignores the
  overlap between training sets and is anti-conservative by a wide margin.

Every function returns a plain dict so results serialise straight into the
paper's tables.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import ALPHA
from .utils import LOG, optional_import

statsmodels_ct = optional_import("statsmodels.stats.contingency_tables")
statsmodels_mt = optional_import("statsmodels.stats.multitest")

#: Nemenyi critical values q_alpha (Demsar 2006, Table 5) = studentized range
#: statistic divided by sqrt(2). Used only if scipy cannot supply them exactly.
_Q_ALPHA = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
           15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
           9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
           15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319},
}


# --------------------------------------------------------------------------
# Single test set: McNemar
# --------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b, exact: bool | None = None) -> dict:
    """McNemar's test on the two classifiers' disagreements.

    The contingency table counts only the cells where the models disagree:
    ``n01`` = A wrong / B right, ``n10`` = A right / B wrong. Cases both get
    right or both get wrong carry no information about which is better.

    ``exact=True`` uses the binomial test, which is required when
    ``n01 + n10 < 25``; above that the chi-square form with continuity
    correction agrees closely and is cheaper. Passing ``None`` selects
    automatically, which is what the reported tables use.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true

    n01 = int(np.sum(~a_ok & b_ok))   # only B correct
    n10 = int(np.sum(a_ok & ~b_ok))   # only A correct
    n_disc = n01 + n10
    if exact is None:
        exact = n_disc < 25

    if n_disc == 0:
        return {
            "n01_only_b_correct": 0, "n10_only_a_correct": 0, "n_discordant": 0,
            "statistic": 0.0, "p_value": 1.0, "exact": exact,
            "note": "models make identical predictions on every test row",
        }

    if statsmodels_ct is not None:
        table = np.array([[int(np.sum(a_ok & b_ok)), n10], [n01, int(np.sum(~a_ok & ~b_ok))]])
        r = statsmodels_ct.mcnemar(table, exact=exact, correction=not exact)
        stat, p = float(r.statistic), float(r.pvalue)
    elif exact:
        p = float(stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1.0) ** 2 / n_disc
        p = float(stats.chi2.sf(stat, df=1))

    return {
        "n01_only_b_correct": n01,
        "n10_only_a_correct": n10,
        "n_discordant": n_disc,
        "statistic": float(stat),
        "p_value": float(p),
        "exact": bool(exact),
        "favours": "a" if n10 > n01 else ("b" if n01 > n10 else "tie"),
        #: Odds ratio of A being the correct one among discordant pairs.
        "odds_ratio": float(n10 / n01) if n01 else float("inf"),
    }


def holm_correction(p_values, alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni step-down correction.

    Holm is uniformly more powerful than Bonferroni at the same family-wise
    error rate, so there is no reason to use plain Bonferroni for the pairwise
    model comparisons; it is used here for every family of tests reported.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return {"p_adjusted": [], "reject": [], "alpha": alpha}
    if statsmodels_mt is not None:
        reject, p_adj, _, _ = statsmodels_mt.multipletests(p, alpha=alpha, method="holm")
        return {"p_adjusted": p_adj.tolist(), "reject": reject.tolist(), "alpha": alpha}

    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return {"p_adjusted": adj.tolist(), "reject": (adj < alpha).tolist(), "alpha": alpha}


def pairwise_mcnemar(y_true, predictions: dict, alpha: float = ALPHA) -> dict:
    """All pairwise McNemar tests among models on one test set, Holm-corrected.

    ``predictions`` maps model name -> hard predictions on the same rows.
    """
    names = list(predictions)
    pairs, raw = [], []
    for a, b in itertools.combinations(names, 2):
        r = mcnemar_test(y_true, predictions[a], predictions[b])
        r["model_a"], r["model_b"] = a, b
        pairs.append(r)
        raw.append(r["p_value"])

    corr = holm_correction(raw, alpha)
    for r, padj, rej in zip(pairs, corr["p_adjusted"], corr["reject"]):
        r["p_adjusted_holm"] = float(padj)
        r["significant"] = bool(rej)
    n_sig = sum(r["significant"] for r in pairs)
    LOG.info("pairwise McNemar: %d/%d pairs significant after Holm (alpha=%.2f)",
             n_sig, len(pairs), alpha)
    return {"alpha": alpha, "n_pairs": len(pairs), "n_significant": n_sig, "pairs": pairs}


# --------------------------------------------------------------------------
# Repeated resampling: corrected paired t
# --------------------------------------------------------------------------
def corrected_paired_t(
    scores_a, scores_b, n_train: int, n_test: int, alpha: float = ALPHA
) -> dict:
    """Nadeau-Bengio corrected resampled paired t-test.

    The naive paired t over repeated splits treats the differences as
    independent, which they are not -- consecutive training sets overlap heavily.
    Nadeau and Bengio (2003) correct the variance by ``1/n + n_test/n_train``
    instead of ``1/n``, which is the difference between a test that rejects
    constantly and one that is honest.
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(d)
    if n < 2:
        return {"error": "need at least two repetitions"}
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d == 0:
        return {"mean_difference": mean_d, "t_statistic": 0.0, "p_value": 1.0,
                "df": n - 1, "significant": False, "note": "zero variance across repetitions"}

    correction = 1.0 / n + n_test / max(n_train, 1)
    t = mean_d / np.sqrt(correction * var_d)
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    half = stats.t.ppf(1 - alpha / 2, df=n - 1) * np.sqrt(correction * var_d)
    return {
        "mean_difference": mean_d,
        "std_difference": float(np.sqrt(var_d)),
        "t_statistic": float(t),
        "p_value": p,
        "df": n - 1,
        "n_repetitions": n,
        "variance_correction": float(correction),
        "ci_low": float(mean_d - half),
        "ci_high": float(mean_d + half),
        "significant": bool(p < alpha),
    }


def paired_t_naive(scores_a, scores_b, alpha: float = ALPHA) -> dict:
    """Uncorrected paired t-test -- reported only to show how much it overstates."""
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    t, p = stats.ttest_rel(scores_a, scores_b)
    return {"mean_difference": float(d.mean()), "t_statistic": float(t),
            "p_value": float(p), "significant": bool(p < alpha)}


# --------------------------------------------------------------------------
# Across datasets: Friedman / Nemenyi / Wilcoxon
# --------------------------------------------------------------------------
@dataclass
class RankMatrix:
    """Scores arranged as datasets (rows) x models (columns)."""

    scores: np.ndarray
    model_names: list
    dataset_names: list

    @property
    def ranks(self) -> np.ndarray:
        """Per-dataset ranks, 1 = best. Ties receive their average rank."""
        return np.vstack([stats.rankdata(-row, method="average") for row in self.scores])

    @property
    def average_ranks(self) -> np.ndarray:
        return self.ranks.mean(axis=0)


def friedman_test(rm: RankMatrix, alpha: float = ALPHA) -> dict:
    """Friedman test plus the Iman-Davenport F correction.

    Friedman's chi-square statistic is known to be conservative; Iman and
    Davenport's F-distributed variant is the one Demsar recommends and is what
    the significance claim is based on.
    """
    N, k = rm.scores.shape
    if N < 3 or k < 3:
        return {"error": f"Friedman needs >=3 datasets and >=3 models (got {N}x{k})"}

    chi2, p_chi2 = stats.friedmanchisquare(*[rm.scores[:, j] for j in range(k)])
    R = rm.average_ranks
    chi2_manual = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    denom = N * (k - 1) - chi2_manual
    if denom <= 0:
        f_stat, p_f = float("inf"), 0.0
    else:
        f_stat = (N - 1) * chi2_manual / denom
        p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (N - 1)))

    return {
        "n_datasets": int(N), "n_models": int(k),
        "chi2_statistic": float(chi2), "chi2_p_value": float(p_chi2),
        "iman_davenport_F": float(f_stat), "iman_davenport_p_value": float(p_f),
        "average_ranks": dict(zip(rm.model_names, map(float, R))),
        "significant": bool(p_f < alpha),
        "interpretation": (
            "at least one model differs in rank" if p_f < alpha
            else "no evidence that the models differ in rank across these datasets"
        ),
    }


def nemenyi_critical_difference(k: int, n_datasets: int, alpha: float = ALPHA) -> float:
    """CD = q_alpha * sqrt(k(k+1) / (6N)).

    ``q_alpha`` is the studentized range statistic at infinite degrees of
    freedom divided by sqrt(2); computed exactly from scipy when available and
    read from Demsar's table otherwise.
    """
    q = None
    try:
        q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    except Exception:  # pragma: no cover - scipy < 1.7
        q = None
    if q is None or not np.isfinite(q):
        table = _Q_ALPHA.get(round(alpha, 2))
        if table is None or k not in table:
            raise ValueError(f"no critical value for k={k}, alpha={alpha}")
        q = table[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_datasets)))


def nemenyi_posthoc(rm: RankMatrix, alpha: float = ALPHA) -> dict:
    """Pairwise Nemenyi comparison of average ranks.

    Two models differ significantly when their average ranks differ by more
    than the critical difference. Note the assumption: every model must have a
    score on every dataset, so a model that failed to run anywhere is excluded
    upstream rather than treated as a loss.
    """
    N, k = rm.scores.shape
    cd = nemenyi_critical_difference(k, N, alpha)
    R = rm.average_ranks
    pairs = []
    for i, j in itertools.combinations(range(k), 2):
        diff = float(abs(R[i] - R[j]))
        z = (R[i] - R[j]) / np.sqrt(k * (k + 1) / (6.0 * N))
        pairs.append({
            "model_a": rm.model_names[i], "model_b": rm.model_names[j],
            "rank_a": float(R[i]), "rank_b": float(R[j]),
            "rank_difference": diff,
            "p_value": float(2 * stats.norm.sf(abs(z))),
            "significant": bool(diff > cd),
        })
    return {
        "critical_difference": cd, "alpha": alpha, "n_datasets": int(N), "n_models": int(k),
        "average_ranks": dict(zip(rm.model_names, map(float, R))),
        "pairs": pairs,
        "n_significant": int(sum(p["significant"] for p in pairs)),
    }


def wilcoxon_across_datasets(rm: RankMatrix, reference: str, alpha: float = ALPHA) -> dict:
    """Wilcoxon signed-rank of ``reference`` against every other model.

    Preferred over a paired t-test across datasets because scores on different
    datasets are not commensurable and are certainly not normally distributed;
    the signed-rank test only assumes symmetry of the differences.
    """
    if reference not in rm.model_names:
        raise KeyError(reference)
    ri = rm.model_names.index(reference)
    rows, raw = [], []
    for j, name in enumerate(rm.model_names):
        if j == ri:
            continue
        a, b = rm.scores[:, ri], rm.scores[:, j]
        d = a - b
        if np.allclose(d, 0):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:  # pragma: no cover - all-zero differences
                stat, p = 0.0, 1.0
        rows.append({
            "reference": reference, "model": name,
            "median_difference": float(np.median(d)),
            "n_wins": int((d > 0).sum()), "n_losses": int((d < 0).sum()),
            "n_ties": int((d == 0).sum()),
            "statistic": float(stat), "p_value": float(p),
        })
        raw.append(float(p))

    corr = holm_correction(raw, alpha)
    for r, padj, rej in zip(rows, corr["p_adjusted"], corr["reject"]):
        r["p_adjusted_holm"] = float(padj)
        r["significant"] = bool(rej)

    # Power warning. With N datasets the smallest attainable two-sided p-value
    # is 2^-(N-1); at N=5 that is 0.0625, so the test *cannot* reject at
    # alpha=0.05 no matter how lopsided the results are. Reporting "not
    # significant" without this caveat would be misleading.
    N = rm.scores.shape[0]
    min_p = 2.0 ** (-(N - 1))
    note = None
    if min_p > alpha:
        note = (f"with N={N} datasets the minimum attainable two-sided p is {min_p:.4f} "
                f"> alpha={alpha}; a non-rejection here is a power limitation, not evidence "
                f"of equivalence")
        LOG.warning("Wilcoxon: %s", note)

    return {"alpha": alpha, "reference": reference, "comparisons": rows,
            "n_significant": int(sum(r["significant"] for r in rows)),
            "min_attainable_p": float(min_p), "power_note": note}


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------
def build_rank_matrix(results: dict, metric: str = "macro_f1") -> RankMatrix:
    """``{dataset: {model: metrics_dict}}`` -> a complete RankMatrix.

    Models missing from any dataset are dropped, with a warning: the Friedman /
    Nemenyi machinery requires a complete matrix, and silently imputing a score
    would manufacture a result.
    """
    datasets = sorted(results)
    common = set.intersection(*[set(results[d]) for d in datasets]) if datasets else set()
    dropped = sorted(set().union(*[set(results[d]) for d in datasets]) - common) if datasets else []
    if dropped:
        LOG.warning("excluded from cross-dataset tests (missing on some datasets): %s", dropped)
    models = sorted(common)
    scores = np.array([[float(results[d][m][metric]) for m in models] for d in datasets])
    return RankMatrix(scores=scores, model_names=models, dataset_names=datasets)
