"""
Statistics for cross-validated attribution comparisons.

Every comparison is paired within a fold (e.g. one dataset's importance against another's,
from the same fold's model), never against a pooled mean, so all tests here are paired with
n = number of folds.

Two caveats drive the choice of tests. First, n = 10 is small and fold accuracies are not
reliably normal, so the primary test is the non-parametric Wilcoxon signed-rank. Second,
cross-validation folds share training data and are therefore **not independent**: the naive
paired t-test underestimates variance and over-rejects. :func:`nadeau_bengio_corrected_t`
provides the conservative alternative, and both are reported.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy import stats


def _paired(a: Sequence[float], b: Sequence[float], min_n: int) -> np.ndarray:
    """
    Per-fold differences ``a - b`` of two paired sequences.

    :param a: First per-fold values.
    :param b: Second per-fold values, same fold order.
    :param int min_n: Minimum number of folds the test needs.
    :return: The differences.
    :rtype: numpy.ndarray
    :raises ValueError: If lengths differ or there are fewer than ``min_n`` folds.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Length mismatch: {x.shape} vs {y.shape}")
    if x.size < min_n:
        raise ValueError(f"Need at least {min_n} folds, got {x.size}")
    return np.asarray(x - y, dtype=float)


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """
    Wilcoxon signed-rank test on per-fold differences ``a - b``.

    :param a: First per-fold values.
    :param b: Second per-fold values, same fold order.
    :return: ``statistic``, ``p_value``, ``mean_difference`` and ``median_difference`` (both
        ``a - b``) and ``n``.
    :rtype: dict[str, float]
    :raises ValueError: If the two sequences differ in length or are empty.
    """
    differences = _paired(a, b, min_n=1)
    # Wilcoxon is undefined when every difference is zero; report p=1 rather than raising.
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(differences, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "n": float(differences.size),
    }


def nadeau_bengio_corrected_t(
    a: Sequence[float], b: Sequence[float], test_fraction: float
) -> dict[str, float]:
    """
    Paired t-test on ``a - b`` with the Nadeau-Bengio correction for k-fold cross-validation.

    Resampled folds overlap in their training sets, so the usual ``var/n`` underestimates the
    variance of the mean difference. The correction inflates it by ``1/n + test/train``
    (Nadeau & Bengio, *Machine Learning* 52(3), 2003).

    :param a: First per-fold values.
    :param b: Second per-fold values, same fold order.
    :param float test_fraction: Fraction of the data held out per fold, e.g. ``0.1`` for
        10-fold cross-validation.
    :return: ``t_statistic``, ``p_value``, ``mean_difference`` (``a - b``) and ``df``.
    :rtype: dict[str, float]
    :raises ValueError: If lengths differ, fewer than two folds are given, or
        ``test_fraction`` is not strictly between 0 and 1.
    """
    differences = _paired(a, b, min_n=2)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    n = differences.size
    mean_difference = float(differences.mean())
    variance = float(differences.var(ddof=1))
    if variance == 0.0:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": mean_difference,
            "df": float(n - 1),
        }

    corrected_variance = variance * (1.0 / n + test_fraction / (1.0 - test_fraction))
    t_statistic = mean_difference / math.sqrt(corrected_variance)
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
    return {
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "mean_difference": mean_difference,
        "df": float(n - 1),
    }


class FDRResult(NamedTuple):
    """Outcome of a Benjamini-Hochberg correction, in the input's order."""

    q_values: list[float]
    rejected: list[bool]
    n_rejected: int


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> FDRResult:
    """
    Benjamini-Hochberg false-discovery-rate correction.

    Applied separately within each family of comparisons (one family per game) rather than
    pooled across families.

    :param p_values: Raw p-values, in the order the comparisons are reported.
    :param float alpha: Target false-discovery rate.
    :return: Adjusted p-values and rejection decisions, in input order.
    :rtype: FDRResult
    :raises ValueError: If ``p_values`` is empty or ``alpha`` is outside (0, 1).
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("Cannot correct an empty family of p-values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)

    # Step-up: adjusted p in sorted order, then enforce monotonicity from the largest down.
    adjusted_sorted = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    q_values = np.empty(n, dtype=float)
    q_values[order] = adjusted_sorted
    rejected = q_values <= alpha
    return FDRResult(
        q_values=q_values.tolist(),
        rejected=[bool(r) for r in rejected],
        n_rejected=int(rejected.sum()),
    )


def spearman(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """
    Spearman rank correlation between two importance rankings.

    Used for robustness (sample versus mean baseline) and for the model-randomisation sanity
    check. With only six bands or eight channels the test is underpowered, so ``rho`` is the
    number to report and ``p_value`` is context.

    :param a: First set of per-channel scores.
    :param b: Second set of per-channel scores, same channel order.
    :return: ``rho``, ``p_value`` and ``n``.
    :rtype: dict[str, float]
    :raises ValueError: If lengths differ or fewer than three points are given.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Length mismatch: {x.shape} vs {y.shape}")
    if x.size < 3:
        raise ValueError("Need at least three points for a rank correlation")

    result = stats.spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": float(x.size),
    }
