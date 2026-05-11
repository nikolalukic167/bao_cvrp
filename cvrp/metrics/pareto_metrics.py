"""Pareto front quality metrics for the Multi-Objective CVRP.

All functions accept a Pareto front as a list of (f1, f2) tuples — raw
floats, not inspyred Individual objects or SimpleNamespace. Callers are
responsible for extracting coordinates before calling these functions.
This keeps the module pure and testable in isolation.

Four metrics are implemented:
    |PF|  — cardinality (number of non-dominated solutions)
    HV    — hypervolume (2-D sweep line, O(n log n))
    SP    — spacing (Schott 1995), lower is better (0 = perfectly uniform)
    GD    — generational distance (van Veldhuizen & Lamont 1998)
"""

from __future__ import annotations

import math
from typing import Sequence


Front = Sequence[tuple[float, float]]


def compute_pf_size(front: Front) -> int:
    """Return the number of solutions in the Pareto front approximation."""
    return len(front)


def compute_hv(front: Front, reference_point: tuple[float, float]) -> float:
    """Compute the 2-D hypervolume dominated by the front.

    Uses a sweep-line algorithm (O(n log n)):
        1. Sort by f1 ascending (implies f2 descending for a valid front).
        2. For each consecutive pair, accumulate the rectangular area
           between the two solutions and the reference point.

    Both objectives are minimized. The reference point should strictly
    dominate all solutions (ref_f1 > max(f1), ref_f2 > max(f2)).
    Out-of-bounds contributions are clamped to 0.

    Args:
        front: List of (f1, f2) tuples.
        reference_point: (ref_f1, ref_f2) bounding the hypervolume region.

    Returns:
        Hypervolume as a non-negative float. Returns 0.0 for empty fronts.
    """
    if not front:
        return 0.0

    ref_f1, ref_f2 = reference_point
    sorted_front = sorted(front, key=lambda p: p[0])

    hv = 0.0
    for i, (f1, f2) in enumerate(sorted_front):
        if i + 1 < len(sorted_front):
            width = sorted_front[i + 1][0] - f1
        else:
            width = ref_f1 - f1
        height = ref_f2 - f2
        if width > 0 and height > 0:
            hv += width * height

    return hv


def compute_sp(front: Front) -> float:
    """Compute the Spacing metric (Schott 1995).

    Measures how uniformly solutions are distributed along the front.
    SP = 0 means perfectly equidistant. Lower is better.

    SP = std(d_i), where d_i is the minimum Manhattan distance from
    solution i to any other solution in the front.

    Args:
        front: List of (f1, f2) tuples.

    Returns:
        Spacing as a non-negative float. Returns 0.0 for fronts with
        fewer than 2 solutions (trivially uniform).
    """
    if len(front) < 2:
        return 0.0

    d: list[float] = []
    for i, (f1_i, f2_i) in enumerate(front):
        min_dist = min(
            abs(f1_i - f1_j) + abs(f2_i - f2_j)
            for j, (f1_j, f2_j) in enumerate(front)
            if j != i
        )
        d.append(min_dist)

    d_mean = sum(d) / len(d)
    variance = sum((di - d_mean) ** 2 for di in d) / len(d)
    return math.sqrt(variance)


def compute_gd(front: Front, reference_front: Front) -> float:
    """Compute the Generational Distance (van Veldhuizen & Lamont 1998).

    Measures how far the approximated front is from a reference front.
    Lower is better. GD = 0 means every solution lies exactly on the
    reference front.

    GD = mean(min Euclidean distance from each front solution to the
              closest solution in the reference front)

    The reference front is typically constructed as the non-dominated
    union of all algorithm archives across all 31 runs, built in
    experiment_loader.py.

    Args:
        front: The approximated Pareto front to evaluate.
        reference_front: The reference front to measure distance against.

    Returns:
        GD as a non-negative float. Returns float('inf') if either
        argument is empty (distance undefined).
    """
    if not front or not reference_front:
        return float("inf")

    total = 0.0
    for f1_i, f2_i in front:
        min_dist = min(
            math.hypot(f1_i - f1_r, f2_i - f2_r)
            for f1_r, f2_r in reference_front
        )
        total += min_dist

    return total / len(front)