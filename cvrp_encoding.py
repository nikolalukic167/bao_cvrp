"""
Two representations for multi-objective CVRP:

1) Two-part (assignment + global visit order permutation)
2) Giant tour + greedy capacity split

Customer indices are 0..N-1. Vehicle labels in assignment are 0..V-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CVRPInstance:
    depot_xy: np.ndarray  # shape (2,)
    customer_xy: np.ndarray  # shape (N, 2)
    demands: np.ndarray  # shape (N,), non-negative
    vehicle_capacity: float
    n_vehicles: int

    def __post_init__(self) -> None:
        d = np.asarray(self.demands, dtype=float).reshape(-1)
        if d.size == 0 or np.any(d < 0):
            raise ValueError("demands must be non-empty and non-negative")
        if self.vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive")
        if self.n_vehicles < 1:
            raise ValueError("n_vehicles must be at least 1")


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def route_distance(depot: np.ndarray, customer_xy: np.ndarray, route: Sequence[int]) -> float:
    if not route:
        return 0.0
    dist = _euclid(depot, customer_xy[route[0]])
    for a, b in zip(route[:-1], route[1:]):
        dist += _euclid(customer_xy[a], customer_xy[b])
    dist += _euclid(customer_xy[route[-1]], depot)
    return dist


def total_distance(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float:
    return sum(route_distance(instance.depot_xy, instance.customer_xy, r) for r in routes)


def capacity_violation(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float:
    """Sum of (load - capacity)+ over routes (0 if feasible)."""
    viol = 0.0
    for r in routes:
        load = float(instance.demands[list(r)].sum()) if r else 0.0
        if load > instance.vehicle_capacity:
            viol += load - instance.vehicle_capacity
    return viol


def vehicle_loads_fixed_vehicles(
    instance: CVRPInstance, routes: Sequence[Sequence[int]]
) -> np.ndarray:
    """
    Loads for exactly n_vehicles entries: route k load, padded with 0 if fewer routes.
    If more routes than vehicles, raises ValueError (infeasible for fixed fleet).
    """
    loads = [float(instance.demands[list(r)].sum()) for r in routes]
    if len(loads) > instance.n_vehicles:
        raise ValueError("more routes than vehicles")
    if len(loads) < instance.n_vehicles:
        loads = loads + [0.0] * (instance.n_vehicles - len(loads))
    return np.asarray(loads, dtype=float)


def load_balance_metric(
    loads: np.ndarray, method: str = "std"
) -> float:
    if method == "std":
        return float(loads.std(ddof=0))
    if method == "range":
        return float(loads.max() - loads.min())
    raise ValueError("method must be 'std' or 'range'")


def decode_two_part(
    assignment: Sequence[int],
    visit_order: Sequence[int],
    n_vehicles: int,
) -> List[List[int]]:
    """
    assignment[i] in 0..V-1 is the vehicle for customer i.
    visit_order is a permutation of 0..N-1: global visit priority (earlier = sooner).

    Each vehicle's route is visit_order restricted to customers assigned to that vehicle.
    """
    assign = np.asarray(assignment, dtype=int).reshape(-1)
    order = np.asarray(visit_order, dtype=int).reshape(-1)
    n = assign.size
    if order.size != n:
        raise ValueError("assignment and visit_order must have length N")
    if set(order.tolist()) != set(range(n)):
        raise ValueError("visit_order must be a permutation of 0..N-1")
    if assign.min() < 0 or assign.max() >= n_vehicles:
        raise ValueError("assignment values must be in 0..V-1")

    buckets: List[List[int]] = [[] for _ in range(n_vehicles)]
    for cust in order.tolist():
        buckets[assign[cust]].append(cust)
    return buckets


def decode_giant_tour_greedy_split(
    giant_tour: Sequence[int], instance: CVRPInstance
) -> List[List[int]]:
    """
    Sequential greedy split: fill current route until next customer does not fit,
    then start a new route. Stops when all customers placed.

    Raises ValueError if more than n_vehicles routes are required.
    """
    tour = np.asarray(giant_tour, dtype=int).reshape(-1)
    n = instance.demands.size
    if tour.size != n or set(tour.tolist()) != set(range(n)):
        raise ValueError("giant_tour must be a permutation of 0..N-1")

    routes: List[List[int]] = []
    current: List[int] = []
    cap_left = instance.vehicle_capacity

    for cust in tour.tolist():
        need = float(instance.demands[cust])
        if need > instance.vehicle_capacity:
            raise ValueError("single customer demand exceeds vehicle capacity")
        if current and need > cap_left:
            routes.append(current)
            current = []
            cap_left = instance.vehicle_capacity
            if len(routes) >= instance.n_vehicles:
                raise ValueError("split requires more vehicles than available")
        current.append(cust)
        cap_left -= need

    if current:
        routes.append(current)
    return routes


def mo_objectives(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    *,
    balance_method: str = "std",
    capacity_penalty_weight: float = 0.0,
) -> Tuple[float, float]:
    """
    Returns (f1, f2) to minimize:
      f1: total distance + optional penalty * capacity violation
      f2: load balance (std or max-min) over n_vehicles padded loads
    """
    viol = capacity_violation(instance, routes)
    f1 = total_distance(instance, routes) + capacity_penalty_weight * viol
    loads = vehicle_loads_fixed_vehicles(instance, routes)
    f2 = load_balance_metric(loads, balance_method)
    return f1, f2
