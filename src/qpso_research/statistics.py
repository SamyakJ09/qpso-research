"""
Statistical analysis for PSO vs QPSO comparison.

Provides Wilcoxon signed-rank test (pairwise) and Friedman test
(3+ algorithms) with publication-ready result formatting.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two algorithms."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def winner(self) -> str:
        if not self.significant:
            return "No significant difference"
        return "Significant difference detected"


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Wilcoxon signed-rank test for pairwise comparison.

    Tests whether the median difference between paired observations is zero.
    Requires at least 6 paired observations for validity.

    Parameters
    ----------
    scores_a : final best scores from algorithm A across multiple runs
    scores_b : final best scores from algorithm B across multiple runs
    alpha    : significance level

    Returns
    -------
    ComparisonResult with test statistic and p-value
    """
    a = np.array(scores_a)
    b = np.array(scores_b)

    # Handle case where all differences are zero
    if np.allclose(a, b):
        return ComparisonResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            alpha=alpha,
        )

    stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
    return ComparisonResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
    )


def friedman_test(
    *score_lists: list[float],
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Friedman test for comparing 3+ algorithms.

    Non-parametric alternative to repeated-measures ANOVA.
    Each list should contain final scores from one algorithm
    across the same set of runs/benchmarks.

    Parameters
    ----------
    score_lists : variable number of score lists (one per algorithm)
    alpha       : significance level

    Returns
    -------
    ComparisonResult with test statistic and p-value
    """
    stat, p_value = stats.friedmanchisquare(*score_lists)
    return ComparisonResult(
        test_name="Friedman",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
    )


def format_results_table(
    results: dict[str, dict],
    benchmark_names: list[str],
) -> str:
    """
    Format multi-benchmark results as a publication-ready markdown table.

    Parameters
    ----------
    results : nested dict of {benchmark: {algorithm: {"mean": x, "std": y, "best": z}}}
    benchmark_names : list of benchmark names in display order

    Returns
    -------
    Markdown-formatted table string
    """
    # Collect algorithm names from first benchmark
    first = benchmark_names[0]
    algorithms = list(results[first].keys())

    # Header
    header = "| Benchmark | " + " | ".join(
        f"{alg} (mean +/- std)" for alg in algorithms
    ) + " |"
    separator = "|" + "|".join(["---"] * (len(algorithms) + 1)) + "|"

    rows = [header, separator]
    for bench in benchmark_names:
        cells = [bench]
        for alg in algorithms:
            r = results[bench][alg]
            cells.append(f"{r['mean']:.2e} +/- {r['std']:.2e}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)
