"""Non-parametric tests for comparing classifiers.

Two levels of comparison are supported:

* node level, paired on the same test nodes: McNemar (pairwise) and Cochran's Q
  (all models at once). These use the per-node correct/incorrect outcome of each
  model's best-seed checkpoint and have real power on a 1000-node test split.
* seed level, one score per seed per model: Friedman (paired on seed),
  Kruskal-Wallis (unpaired) and pairwise Wilcoxon signed-rank. With five seeds
  these are weak: an exact two-sided Wilcoxon on five pairs cannot go below
  p = 0.0625, so treat them as descriptive.

All pairwise tables carry Holm-adjusted p-values.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["holm_correction", "mcnemar_test", "cochran_q_test", "pairwise_mcnemar",
           "friedman_test", "kruskal_test", "pairwise_wilcoxon"]


def holm_correction(p_values) -> np.ndarray:
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def mcnemar_test(correct_a, correct_b, exact: bool = True) -> dict:
    """McNemar test on paired correct/incorrect outcomes of two classifiers.

    b = A right, B wrong; c = A wrong, B right. Exact uses the binomial
    distribution on the discordant pairs (recommended when b + c < 25);
    otherwise the chi-square statistic with continuity correction.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("Inputs must have the same length")
    both = int((a & b_).sum())
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    neither = int((~a & ~b_).sum())
    n_disc = b + c
    if n_disc == 0:
        statistic, p_value = 0.0, 1.0
    elif exact:
        statistic = float(min(b, c))
        p_value = float(stats.binomtest(min(b, c), n_disc, 0.5).pvalue)
    else:
        statistic = (abs(b - c) - 1) ** 2 / n_disc
        p_value = float(stats.chi2.sf(statistic, df=1))
    return dict(both_correct=both, a_only=b, b_only=c, both_wrong=neither,
                statistic=statistic, p_value=p_value)


def cochran_q_test(correct_matrix) -> dict:
    """Cochran's Q for k classifiers on the same n items (bool matrix n x k)."""
    x = np.asarray(correct_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Need an n x k matrix with k >= 2")
    k = x.shape[1]
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    denominator = k * row.sum() - (row ** 2).sum()
    if denominator == 0:
        return dict(statistic=0.0, p_value=1.0, df=k - 1)
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denominator
    return dict(statistic=float(q), p_value=float(stats.chi2.sf(q, df=k - 1)), df=k - 1)


def pairwise_mcnemar(correct_df: pd.DataFrame, exact: bool = True) -> pd.DataFrame:
    """All pairwise McNemar tests over the columns of a nodes x models bool frame."""
    rows = []
    for a, b in combinations(correct_df.columns, 2):
        res = mcnemar_test(correct_df[a], correct_df[b], exact=exact)
        rows.append(dict(model_a=a, model_b=b,
                         acc_a=float(correct_df[a].mean()), acc_b=float(correct_df[b].mean()),
                         **res))
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm_correction(table["p_value"])
    return table


def friedman_test(scores: pd.DataFrame) -> dict:
    """Friedman test on a seeds x models score frame (paired on seed)."""
    columns = [scores[c].to_numpy() for c in scores.columns]
    statistic, p_value = stats.friedmanchisquare(*columns)
    return dict(statistic=float(statistic), p_value=float(p_value), df=len(columns) - 1)


def kruskal_test(scores: pd.DataFrame) -> dict:
    """Kruskal-Wallis on a seeds x models score frame (treats groups as unpaired)."""
    columns = [scores[c].to_numpy() for c in scores.columns]
    statistic, p_value = stats.kruskal(*columns)
    return dict(statistic=float(statistic), p_value=float(p_value), df=len(columns) - 1)


def pairwise_wilcoxon(scores: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests over the columns of a seeds x models frame."""
    rows = []
    for a, b in combinations(scores.columns, 2):
        diff = scores[a].to_numpy() - scores[b].to_numpy()
        if np.allclose(diff, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(scores[a], scores[b])
        rows.append(dict(model_a=a, model_b=b, mean_a=float(scores[a].mean()),
                         mean_b=float(scores[b].mean()), median_diff=float(np.median(diff)),
                         statistic=float(statistic), p_value=float(p_value)))
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm_correction(table["p_value"])
    return table
