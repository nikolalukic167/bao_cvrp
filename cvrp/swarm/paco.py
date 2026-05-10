"""Pareto Ant Colony Optimization for bi-objective CVRP (giant tour, Rep A).

Standalone class (not inspyred): constructs random permutations with a
dual-pheromone transition rule and maintains an external Pareto archive.
"""

from __future__ import annotations

from random import Random
from types import SimpleNamespace
from typing import Any

import numpy as np

from cvrp.problem.decoders import GiantTourDecoder
from cvrp.problem.fitness import evaluate
from cvrp.problem.instance import CVRPInstance


def _dominates(fa: tuple[float, float], fb: tuple[float, float]) -> bool:
    """Pareto dominance for minimization on (f1, f2)."""
    return (fa[0] <= fb[0] and fa[1] <= fb[1]) and (fa[0] < fb[0] or fa[1] < fb[1])


def _nondominated_filter(
    solutions: list[tuple[tuple[int, ...], tuple[float, float]]],
) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    """Return minimal non-dominated set (exact, quadratic)."""
    nd: list[tuple[tuple[int, ...], tuple[float, float]]] = []
    for perm, f in solutions:
        if any(_dominates(fj, f) for _, fj in nd):
            continue
        nd = [(p, fj) for p, fj in nd if not _dominates(f, fj)]
        nd.append((perm, f))
    return nd


def _trim_archive_even_spread(
    nd: list[tuple[tuple[int, ...], tuple[float, float]]],
    archive_size: int,
) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    if len(nd) <= archive_size:
        return nd
    nd_sorted = sorted(nd, key=lambda x: (x[1][0], x[1][1]))
    lin = np.linspace(0, len(nd_sorted) - 1, num=archive_size)
    picks = sorted({min(int(round(x)), len(nd_sorted) - 1) for x in lin})
    chosen = [nd_sorted[i] for i in picks]
    if len(chosen) < archive_size:
        extras = [
            nd_sorted[k]
            for k in range(len(nd_sorted))
            if k not in set(picks)
        ]
        for item in extras:
            if len(chosen) >= archive_size:
                break
            chosen.append(item)
    return chosen[:archive_size]


class ParetoACO:
    """Bi-objective ACO over giant-tour permutations (customer-id sequence).

    Two pheromone matrices tau1 / tau2 are updated from the external archive
    after each iteration. Transition probability combines both matrices with
    exponents alpha1 / alpha2 and distance heuristic eta_ij = 1 / d_ij.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        decoder: GiantTourDecoder | None = None,
        n_ants: int = 50,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        archive_size: int = 100,
        max_iterations: int = 100,
        tau0: float = 1.0,
        penalty_weight: float = 1000.0,
    ) -> None:
        self.instance = instance
        self.decoder = decoder if decoder is not None else GiantTourDecoder(instance)
        self.n_ants = n_ants
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.rho = rho
        self.archive_size = archive_size
        self.max_iterations = max_iterations
        self.tau0 = tau0
        self.penalty_weight = penalty_weight

        self.customers = list(instance.customers)
        self._n = len(self.customers)
        self._idx_to_cid = {i: self.customers[i] for i in range(self._n)}

        depot = instance.depot
        self._eta_depot = np.zeros(self._n)
        dist_m = np.zeros((self._n, self._n))
        for i in range(self._n):
            ci = self.customers[i]
            self._eta_depot[i] = 1.0 / max(instance.distance(depot, ci), 1e-9)
            for j in range(self._n):
                cj = self.customers[j]
                dist_m[i, j] = instance.distance(ci, cj)

        np.fill_diagonal(dist_m, 1e12)
        self._eta = 1.0 / np.maximum(dist_m, 1e-9)
        np.fill_diagonal(self._eta, 0.0)

        self.final_archive: list[Any] = []
        self.history: list[list[Any]] = []
        self.num_generations: int = 0
        self.num_evaluations: int = 0

        self.tau1: np.ndarray
        self.tau2: np.ndarray
        self._archive: list[tuple[tuple[int, ...], tuple[float, float]]]
        self._rng: Random

    def _initialize(self, seed: int | None) -> None:
        self._rng = Random(seed)
        n = self._n
        self.tau1 = np.full((n, n), self.tau0)
        self.tau2 = np.full((n, n), self.tau0)
        np.fill_diagonal(self.tau1, 0.0)
        np.fill_diagonal(self.tau2, 0.0)

        self._archive = []
        self.final_archive = []
        self.history = []
        self.num_generations = 0
        self.num_evaluations = 0

    def optimize(self, seed: int | None = None) -> list[Any]:
        """Run the colony loop; set ``final_archive`` and ``history``.

        ``final_archive`` entries expose ``candidate`` (permutation of
        customer ids) and ``fitness.values`` compatible with plotting.

        Treat ``max_iterations`` as generations for comparability with GAs.
        """
        self._initialize(seed)

        for _iteration in range(self.max_iterations):
            trails: list[tuple[tuple[int, ...], tuple[float, float]]] = []
            for _ in range(self.n_ants):
                trails.append(self._construct_solution())

            self._merge_archive(trails)
            self._update_pheromone()

            self.num_generations += 1

            pseudo_pop = []
            for perm, fv in trails:
                pseudo_pop.append(_make_holder(perm, fv, self._idx_to_cid))
            self.history.append(pseudo_pop)

        self.final_archive = [
            _make_holder(perm, fv, self._idx_to_cid)
            for perm, fv in self._archive
        ]
        return self.final_archive

    def _eval_perm_indices(self, perm_idx: tuple[int, ...]) -> tuple[float, float]:
        cand = tuple(self._idx_to_cid[i] for i in perm_idx)
        routes = self.decoder.decode(cand)
        self.num_evaluations += 1
        return evaluate(routes, self.instance, self.penalty_weight)

    def _construct_solution(self) -> tuple[tuple[int, ...], tuple[float, float]]:
        n = self._n
        remaining = set(range(n))
        tour: list[int] = []

        curr = self._weighted_choice_first(list(remaining))
        tour.append(curr)
        remaining.remove(curr)

        while remaining:
            nxt = self._weighted_transition(curr, list(remaining))
            tour.append(nxt)
            remaining.remove(nxt)
            curr = nxt

        perm_tuple = tuple(tour)
        fvals = self._eval_perm_indices(perm_tuple)
        return perm_tuple, fvals

    def _weighted_choice_first(self, cand: list[int]) -> int:
        weights = [self._eta_depot[c] ** self.beta for c in cand]
        return self._roulette_select(cand, weights)

    def _weighted_transition(self, curr: int, cand: list[int]) -> int:
        a1, a2, b = self.alpha1, self.alpha2, self.beta
        weights = []
        for j in cand:
            t1 = max(self.tau1[curr, j], 1e-300)
            t2 = max(self.tau2[curr, j], 1e-300)
            h = max(self._eta[curr, j], 1e-300)
            w = (t1**a1) * (t2**a2) * (h**b)
            weights.append(w)
        return self._roulette_select(cand, weights)

    def _roulette_select(self, cand: list[int], weights: list[float]) -> int:
        total = float(sum(weights))
        if total <= 0:
            return self._rng.choice(cand)
        r = self._rng.random() * total
        acc = 0.0
        for c, w in zip(cand, weights):
            acc += w
            if r <= acc:
                return c
        return cand[-1]

    def _merge_archive(
        self,
        trails: list[tuple[tuple[int, ...], tuple[float, float]]],
    ) -> None:
        combined = self._archive + trails
        nd = _nondominated_filter(combined)
        self._archive = _trim_archive_even_spread(nd, self.archive_size)

    def _update_pheromone(self) -> None:
        rho = self.rho
        self.tau1 *= 1.0 - rho
        self.tau2 *= 1.0 - rho
        self.tau1 = np.maximum(self.tau1, 1e-9)
        self.tau2 = np.maximum(self.tau2, 1e-9)
        np.fill_diagonal(self.tau1, 0.0)
        np.fill_diagonal(self.tau2, 0.0)

        for perm, (f1, f2) in self._archive:
            inv1 = 1.0 / (f1 + 1e-9)
            inv2 = 1.0 / (max(f2, 1e-12))
            d1 = rho * inv1
            d2 = rho * inv2
            for a, bn in zip(perm[:-1], perm[1:]):
                self.tau1[a, bn] += d1
                self.tau2[a, bn] += d2


def _make_holder(
    perm_idx: tuple[int, ...],
    fvals: tuple[float, float],
    idx_to_cid: dict[int, int],
) -> Any:
    """Build a minimal ``individual`` analog for archival / CSV export."""
    cand = tuple(idx_to_cid[i] for i in perm_idx)
    return SimpleNamespace(
        candidate=list(cand),
        fitness=SimpleNamespace(values=list(fvals)),
    )
