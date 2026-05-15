"""Statistical tests for comparing multi-objective algorithm performance.

Two functions are exposed:

    wilcoxon_test         — pairwise comparison of two algorithms (scipy)
    friedman_shaffer_test — N-way comparison with post-hoc correction (stac)

Both functions take lists of scalar metric values (one per run) and return
a dict with the test statistic, p-value, and a plain-English conclusion.
Input is raw metric values, not DataFrames — callers extract the relevant
column from the experiment loader output before calling these functions.
"""

from __future__ import annotations

from typing import Sequence

from scipy.stats import wilcoxon
from stac.nonparametric_tests import friedman_aligned_ranks_test, shaffer_multitest


def wilcoxon_test(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Wilcoxon signed-rank test for paired comparison of two algorithms.

    Tests the null hypothesis that the two distributions are identical.
    Use this when comparing exactly two algorithms on the same metric
    across the same set of runs (same seeds, same instances).

    Args:
        a: Metric values for algorithm A, one per run.
        b: Metric values for algorithm B, one per run.
        alpha: Significance level. Default 0.05.

    Returns:
        Dict with keys:
            statistic   — Wilcoxon W statistic
            p_value     — two-sided p-value
            significant — True if p_value < alpha
            conclusion  — plain-English string for the report
    """
    if len(a) != len(b):
        raise ValueError(
            f"Both sequences must have the same length. "
            f"Got {len(a)} and {len(b)}."
        )
    if len(a) < 2:
        raise ValueError("At least 2 observations are required.")

    stat, p_value = wilcoxon(a, b, alternative="two-sided")
    significant = bool(p_value < alpha)

    if significant:
        conclusion = (
            f"Significant difference detected (W={stat:.4f}, "
            f"p={p_value:.4f} < α={alpha})."
        )
    else:
        conclusion = (
            f"No significant difference detected (W={stat:.4f}, "
            f"p={p_value:.4f} ≥ α={alpha})."
        )

    return {
        "statistic": stat,
        "p_value": p_value,
        "significant": significant,
        "conclusion": conclusion,
    }


def friedman_shaffer_test(
    groups: dict[str, Sequence[float]],
    alpha: float = 0.05,
    maximize: bool = False,
) -> dict:
    """Friedman Aligned Ranks + Shaffer post-hoc for N-way comparison.

    Tests whether at least one algorithm performs significantly differently
    from the others. If the Friedman test is significant, the Shaffer
    post-hoc identifies which specific pairs differ.

    Use this when comparing three or more algorithms simultaneously on
    the same metric. Running multiple Wilcoxon tests instead would inflate
    the Type I error rate (multiple comparisons problem).

    Args:
        groups: Dict mapping algorithm name to list of metric values.
                All lists must have the same length (same number of runs).
                Example: {"NSGA-II": [...], "SPEA2": [...], "PACO": [...]}
        alpha: Significance level. Default 0.05.
        maximize: If True, larger values are treated as better. The function
                negates inputs before computing ranks so that the convention
                "lower rank = better" holds in the output regardless of metric
                direction. Use maximize=True for HV, PF, accuracy, etc.; leave
                False for SP, GD, error, time, etc.

    Returns:
        Dict with keys:
            friedman_statistic   — Friedman aligned ranks statistic
            friedman_p_value     — p-value from the Friedman test
            friedman_significant — True if Friedman p_value < alpha
            rankings             — dict mapping algorithm name to mean rank
            post_hoc             — list of dicts, one per pairwise comparison:
                                     comparison, z_value, p_value,
                                     adj_p_value, significant
                                   Empty list if Friedman is not significant.
            conclusion           — plain-English string for the report
    """
    names = list(groups.keys())
    values = [list(groups[name]) for name in names]

    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All groups must have the same number of observations. "
            f"Got: {dict(zip(names, lengths))}"
        )

    if maximize:
        values = [[-v for v in vlist] for vlist in values]

    friedman_stat, friedman_p, rankings, pivots = (
        friedman_aligned_ranks_test(*values)
    )
    friedman_significant = bool(friedman_p < alpha)
    rankings_dict = dict(zip(names, rankings))

    post_hoc = []
    if friedman_significant:
        d = dict(zip(names, pivots))
        comparisons, z_values, p_values, adj_p_values = shaffer_multitest(d)
        post_hoc = [
            {
                "comparison": comp,
                "z_value": float(z),
                "p_value": float(p),
                "adj_p_value": float(adj_p),
                "significant": bool(adj_p < alpha),
            }
            for comp, z, p, adj_p in zip(
                comparisons, z_values, p_values, adj_p_values
            )
        ]

    if friedman_significant:
        significant_pairs = [ph["comparison"] for ph in post_hoc if ph["significant"]]
        if significant_pairs:
            conclusion = (
                f"Friedman test significant (stat={friedman_stat:.4f}, "
                f"p={friedman_p:.4f}). Shaffer post-hoc significant pairs: "
                f"{', '.join(significant_pairs)}."
            )
        else:
            conclusion = (
                f"Friedman test significant (stat={friedman_stat:.4f}, "
                f"p={friedman_p:.4f}), but no individual pair differs "
                f"after Shaffer correction."
            )
    else:
        conclusion = (
            f"No significant difference among algorithms "
            f"(Friedman stat={friedman_stat:.4f}, p={friedman_p:.4f} "
            f"≥ α={alpha})."
        )

    return {
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "friedman_significant": friedman_significant,
        "rankings": rankings_dict,
        "post_hoc": post_hoc,
        "conclusion": conclusion,
    }