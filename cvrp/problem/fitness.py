"""Fitness functions for the multi-objective CVRP.

Two objectives:
    f1 - total travel distance (with optional capacity-violation penalty)
    f2 - imbalance of demand across vehicles (standard deviation)
"""

from __future__ import annotations

import math
from typing import Sequence

from cvrp.problem.instance import CVRPInstance


def route_distance(route: Sequence[int], instance: CVRPInstance) -> float:
    """Euclidean distance of a single closed route depot -> customers -> depot."""
    if not route:
        return 0.0

    depot = instance.depot
    total = instance.distance(depot, route[0])
    for a, b in zip(route[:-1], route[1:]):
        total += instance.distance(a, b)
    total += instance.distance(route[-1], depot)
    return total


def route_load(route: Sequence[int], instance: CVRPInstance) -> int:
    """Total demand carried by a single route."""
    return sum(instance.demands[c] for c in route)


def compute_f1(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    penalty_weight: float = 1000.0,
) -> float:
    """Total distance plus a linear penalty for capacity violations.

    f1 = sum(route_distance) + penalty_weight * sum(max(0, load - Q))

    For feasible solutions (every route within capacity) the penalty term
    is zero, and f1 reduces to the pure total distance. The default
    penalty weight is large enough that a single unit of violation costs
    more than typical edge distances on the X-set instances.

    Args:
        routes: Decoded solution as a list of routes (customer ids).
        instance: The problem instance.
        penalty_weight: Multiplier for capacity violation. Set to 0
            to disable the penalty (useful for debugging).

    Returns:
        f1 as a non-negative float.
    """
    capacity = instance.capacity

    distance = 0.0
    violation = 0
    for route in routes:
        distance += route_distance(route, instance)
        load = route_load(route, instance)
        if load > capacity:
            violation += load - capacity

    return distance + penalty_weight * violation


def compute_f2(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
) -> float:
    """Standard deviation of per-vehicle demand (load imbalance).

    f2 = sqrt( mean((D_k - mean_D)^2) ),  D_k = total demand of route k

    A perfectly balanced solution has f2 = 0. Empty routes (vehicles not
    used) are excluded so that 'using fewer vehicles than allowed' does
    not artificially worsen the imbalance score.

    Args:
        routes: Decoded solution.
        instance: The problem instance.

    Returns:
        f2 as a non-negative float.
    """
    if not routes:
        return 0.0

    loads = [route_load(route, instance) for route in routes]
    mean_load = sum(loads) / len(loads)
    variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
    return math.sqrt(variance)


def evaluate(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    penalty_weight: float = 1000.0,
) -> tuple[float, float]:
    """Compute both objectives for one solution.

    Returns:
        (f1, f2) as a tuple.
    """
    return (
        compute_f1(routes, instance, penalty_weight),
        compute_f2(routes, instance),
    )