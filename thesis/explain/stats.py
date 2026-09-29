"""
Statistics for cross-validated ablation comparisons.

Every ablation is compared against the *same fold's* unablated baseline, never against a
pooled mean, so all tests here are paired with n = number of folds.

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


def paired_wilcoxon(ablated: Sequence[float], baseline: Sequence[float]) -> dict[str, float]:
    """
    Wilcoxon signed-rank test on per-fold differences.

    :param ablated: Per-fold metric under ablation.
    :param baseline: Per-fold metric without ablation, same fold order.
    :return: ``statistic``, ``p_value``, ``mean_drop`` (baseline minus ablated, so a positive
        value means the ablation hurt), ``median_drop`` and ``n``.
    :rtype: dict[str, float]
    :raises ValueError: If the two sequences differ in length or are empty.
    """
    a = np.asarray(ablated, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Cannot test empty sequences")

    drops = b - a
    # Wilcoxon is undefined when every difference is zero; report p=1 rather than raising.
    if np.allclose(drops, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(a, b, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_drop": float(drops.mean()),
        "median_drop": float(np.median(drops)),
        "n": float(a.size),
    }


def nadeau_bengio_corrected_t(
    ablated: Sequence[float], baseline: Sequence[float], test_fraction: float
) -> dict[str, float]:
    """
    Paired t-test with the Nadeau-Bengio variance correction for k-fold cross-validation.

    Resampled folds overlap in their training sets, so the usual ``var/n`` underestimates the
    variance of the mean difference. The correction inflates it by ``1/n + test/train``
    (Nadeau & Bengio, *Machine Learning* 52(3), 2003).

    :param ablated: Per-fold metric under ablation.
    :param baseline: Per-fold metric without ablation, same fold order.
    :param float test_fraction: Fraction of the data held out per fold, e.g. ``0.1`` for
        10-fold cross-validation.
    :return: ``t_statistic``, ``p_value``, ``mean_drop`` and ``df``.
    :rtype: dict[str, float]
    :raises ValueError: If lengths differ, fewer than two folds are given, or
        ``test_fraction`` is not strictly between 0 and 1.
    """
    a = np.asarray(ablated, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError("Need at least two folds for a t-test")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    drops = b - a
    n = drops.size
    mean_drop = float(drops.mean())
    variance = float(drops.var(ddof=1))
    if variance == 0.0:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_drop": mean_drop,
            "df": float(n - 1),
        }

    corrected_variance = variance * (1.0 / n + test_fraction / (1.0 - test_fraction))
    t_statistic = mean_drop / math.sqrt(corrected_variance)
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
    return {
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "mean_drop": mean_drop,
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

    Applied separately within each family of comparisons — once across the eight channels,
    once across the five bands — rather than pooled across families.

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


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Percentile bootstrap confidence interval for the mean.

    Used for the error bars on per-channel accuracy drops, where n = 10 folds is too small to
    lean on a normal approximation.

    :param values: Per-fold values, typically accuracy drops.
    :param float confidence: Interval width, e.g. ``0.95``.
    :param int n_resamples: Bootstrap resample count.
    :param int seed: Seed, so published error bars are reproducible.
    :return: ``mean``, ``lower``, ``upper``.
    :rtype: dict[str, float]
    :raises ValueError: If ``values`` is empty or ``confidence`` is outside (0, 1).
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_resamples, v.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(v.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
    }


def spearman(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """
    Spearman rank correlation between two importance rankings.

    Agreement between independently derived rankings — ablation, integrated gradients,
    attention rollout — is the corroborating result of the explainability analysis. With only
    eight channels the test is underpowered, so ``rho`` is the number to report and
    ``p_value`` is context.

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
