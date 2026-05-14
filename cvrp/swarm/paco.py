"""Pareto Ant Colony Optimization for bi-objective CVRP

Standalone class (not inspyred): constructs solutions probabilistically
with a dual-pheromone transition rule and maintains an external Pareto
archive. Supports two representations:

    "giant_tour" — pheromones on customer-pairs tau[i,j]; ants build a
                   permutation visiting customers sequentially.
    "cluster_first" — pheromones on customer-vehicle pairs tau[i,v];
                      ants build an assignment vector by choosing one
                      vehicle for each customer.

Two pheromone matrices tau1 / tau2 (one per objective) are updated from
the external archive after each iteration. Transition probability
combines both matrices with exponents alpha1 / alpha2. The distance
heuristic eta is used only in giant_tour mode; in cluster_first mode
there is no geometric distance between a customer and a vehicle id, so
construction relies on pheromones alone.
"""

from __future__ import annotations

from random import Random
from types import SimpleNamespace
from typing import Any

import numpy as np

from cvrp.problem.decoders import ClusterFirstDecoder, Decoder, GiantTourDecoder
from cvrp.problem.fitness import evaluate
from cvrp.problem.instance import CVRPInstance


def _dominates(fa: tuple[float, float], fb: tuple[float, float]) -> bool:
    """Pareto dominance for minimization on (f1, f2)."""
    return (fa[0] <= fb[0] and fa[1] <= fb[1]) and (fa[0] < fb[0] or fa[1] < fb[1])


def _nondominated_filter(
    solutions: list[tuple[tuple[int, ...], tuple[float, float]]],
) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    """Return the minimal non-dominated subset of the input."""
    nd: list[tuple[tuple[int, ...], tuple[float, float]]] = []
    for chrom, f in solutions:
        if any(_dominates(fj, f) for _, fj in nd):
            continue
        nd = [(c, fj) for c, fj in nd if not _dominates(f, fj)]
        nd.append((chrom, f))
    return nd


def _trim_archive_even_spread(
    nd: list[tuple[tuple[int, ...], tuple[float, float]]],
    archive_size: int,
) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    """Trim a non-dominated set to archive_size by picking evenly along f1."""
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
    """Bi-objective ACO with dual pheromones and external Pareto archive.

    Two pheromone matrices tau1, tau2 (one per objective) are reinforced
    by non-dominated solutions from the current iteration. The shape of
    these matrices depends on the representation:

        giant_tour:    (N, N) — pheromones on customer-pairs (i, j).
        cluster_first: (N, K) — pheromones on customer-vehicle pairs (i, v).

    Construction is dispatched to _construct_solution_giant_tour or
    _construct_solution_cluster_first based on self.representation.
    Update is dispatched similarly.

    After optimize() completes, self.history holds one snapshot per
    iteration. Each snapshot is a list of (chromosome, (f1, f2)) tuples,
    one per ant constructed in that iteration. The shape is parallel to
    NSGA2.history and SPEA2.history for analysis purposes.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        decoder: Decoder | None = None,
        representation: str = "giant_tour",
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
        if representation not in ("giant_tour", "cluster_first"):
            raise ValueError(
                f"Unknown representation {representation!r}. "
                "Must be 'giant_tour' or 'cluster_first'."
            )

        self.instance = instance
        self.representation = representation
        if decoder is not None:
            self.decoder = decoder
        elif representation == "giant_tour":
            self.decoder = GiantTourDecoder(instance)
        else:
            self.decoder = ClusterFirstDecoder(instance)

        self.n_ants = n_ants
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.rho = rho
        self.archive_size = archive_size
        self.max_iterations = max_iterations
        self.tau0 = tau0
        self.penalty_weight = penalty_weight

        # Customer list and indexing shared by both representations.
        self.customers = list(instance.customers)
        self._n = len(self.customers)
        self.num_vehicles = instance.num_vehicles

        # Precompute the distance heuristic eta[i, j] = 1 / d(c_i, c_j),
        # used only by giant_tour construction. The cluster_first mode
        # has no geometric distance between a customer and a vehicle id.
        depot = instance.depot
        dist_m = np.zeros((self._n, self._n))
        for i in range(self._n):
            ci = self.customers[i]
            for j in range(self._n):
                cj = self.customers[j]
                dist_m[i, j] = instance.distance(ci, cj)
        np.fill_diagonal(dist_m, 1e12)
        self._eta = 1.0 / np.maximum(dist_m, 1e-9)
        np.fill_diagonal(self._eta, 0.0)

        self._eta_depot = np.array(
            [
                1.0 / max(instance.distance(depot, self.customers[i]), 1e-9)
                for i in range(self._n)
            ]
        )

        # State produced by optimize().
        self.final_archive: list[Any] = []
        self.history: list[list[Any]] = []
        self.num_generations: int = 0
        self.num_evaluations: int = 0

        self.tau1: np.ndarray
        self.tau2: np.ndarray
        self._archive: list[tuple[tuple[int, ...], tuple[float, float]]]
        self._rng: Random

    def _initialize(self, seed: int | None) -> None:
        """Reset RNG, pheromones, archive, and run counters.

        Pheromone matrix shape depends on representation:
            giant_tour:    (N, N) with zero diagonal.
            cluster_first: (N, K).
        """
        self._rng = Random(seed)

        if self.representation == "giant_tour":
            n = self._n
            self.tau1 = np.full((n, n), self.tau0)
            self.tau2 = np.full((n, n), self.tau0)
            np.fill_diagonal(self.tau1, 0.0)
            np.fill_diagonal(self.tau2, 0.0)
        else:
            self.tau1 = np.full((self._n, self.num_vehicles), self.tau0)
            self.tau2 = np.full((self._n, self.num_vehicles), self.tau0)

        self._archive = []
        self.final_archive = []
        self.history = []
        self.num_generations = 0
        self.num_evaluations = 0

    def optimize(self, seed: int | None = None) -> list[Any]:
        """Run the colony loop and return the final non-dominated archive.

        Treats max_iterations as generations for symmetry with the GAs.
        After completion, final_archive holds SimpleNamespace items with
        .candidate (the chromosome) and .fitness.values = (f1, f2).
        """
        self._initialize(seed)

        for it in range(self.max_iterations):
            trails = []
            for _ in range(self.n_ants):
                chromosome = self._construct_solution()
                f1, f2 = self._evaluate(chromosome)
                trails.append((chromosome, (f1, f2)))

            # Merge current iteration trails into the external archive.
            self._archive = _nondominated_filter(self._archive + trails)
            self._archive = _trim_archive_even_spread(
                self._archive, self.archive_size
            )

            # Reinforce pheromones from the iteration trails (T in the pseudocode).
            self._update_pheromone(trails)

            self.history.append([
                (list(chromosome), tuple(fitness))
                for chromosome, fitness in trails
            ])
            self.num_generations = it + 1

        self.final_archive = [
            self._make_individual(c, f) for c, f in self._archive
        ]
        return self.final_archive

    def _construct_solution(self) -> tuple[int, ...]:
        """Dispatch construction to the representation-specific helper."""
        if self.representation == "giant_tour":
            return self._construct_solution_giant_tour()
        return self._construct_solution_cluster_first()

    def _construct_solution_giant_tour(self) -> tuple[int, ...]:
        """Build a customer-id permutation by sequential probabilistic choice.

        At each step the ant is at some customer (or the depot for the
        first pick). Next customer is chosen with probability
        proportional to tau1^alpha1 * tau2^alpha2 * eta^beta.
        """
        n = self._n
        unvisited = list(range(n))
        current = -1  # depot sentinel for the first pick
        perm: list[int] = []

        while unvisited:
            if current == -1:
                # First pick uses the depot heuristic and tau0 (uniform).
                eta_row = self._eta_depot[unvisited]
                tau1_row = np.full(len(unvisited), self.tau0)
                tau2_row = np.full(len(unvisited), self.tau0)
            else:
                idxs = np.array(unvisited)
                tau1_row = self.tau1[current, idxs]
                tau2_row = self.tau2[current, idxs]
                eta_row = self._eta[current, idxs]

            weights = (
                np.power(tau1_row, self.alpha1)
                * np.power(tau2_row, self.alpha2)
                * np.power(eta_row, self.beta)
            )
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                pick_local = self._rng.randrange(len(unvisited))
            else:
                probs = weights / total
                r = self._rng.random()
                cum = 0.0
                pick_local = len(unvisited) - 1
                for k, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        pick_local = k
                        break

            pick_global = unvisited.pop(pick_local)
            perm.append(self.customers[pick_global])
            current = pick_global

        return tuple(perm)

    def _construct_solution_cluster_first(self) -> tuple[int, ...]:
        """Build a vehicle-assignment vector by per-customer probabilistic choice.

        For each customer i (in customer-id order), one vehicle v in
        {1, ..., K} is chosen with probability proportional to
        tau1[i, v-1]^alpha1 * tau2[i, v-1]^alpha2. There is no
        geometric heuristic between customers and vehicle ids, so beta
        and eta are not used here.
        """
        chromosome: list[int] = []
        for i in range(self._n):
            tau1_row = self.tau1[i, :]
            tau2_row = self.tau2[i, :]
            weights = (
                np.power(tau1_row, self.alpha1)
                * np.power(tau2_row, self.alpha2)
            )
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                v = self._rng.randint(1, self.num_vehicles)
            else:
                probs = weights / total
                r = self._rng.random()
                cum = 0.0
                v = self.num_vehicles
                for k, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        v = k + 1
                        break
            chromosome.append(v)

        return tuple(chromosome)

    def _evaluate(self, chromosome: tuple[int, ...]) -> tuple[float, float]:
        """Decode the chromosome and compute (f1, f2). Counts one evaluation."""
        routes = self.decoder.decode(list(chromosome))
        f1, f2 = evaluate(routes, self.instance, self.penalty_weight)
        self.num_evaluations += 1
        return f1, f2

    def _update_pheromone(
        self,
        trails: list[tuple[tuple[int, ...], tuple[float, float]]],
    ) -> None:
        """Evaporate then deposit on pheromones using non-dominated trails.

        Only the non-dominated subset of the current iteration deposits.
        Deposit magnitude is 1 / (1 + f_k) per objective: lower fitness
        leaves stronger trails.
        """
        self.tau1 *= 1.0 - self.rho
        self.tau2 *= 1.0 - self.rho

        nd_trails = _nondominated_filter(trails)

        if self.representation == "giant_tour":
            for chromosome, (f1, f2) in nd_trails:
                delta1 = 1.0 / (1.0 + f1)
                delta2 = 1.0 / (1.0 + f2)
                # Map customer ids back to internal indices.
                cid_to_idx = {self.customers[i]: i for i in range(self._n)}
                for a, b in zip(chromosome[:-1], chromosome[1:]):
                    ia, ib = cid_to_idx[a], cid_to_idx[b]
                    self.tau1[ia, ib] += delta1
                    self.tau2[ia, ib] += delta2
        else:
            for chromosome, (f1, f2) in nd_trails:
                delta1 = 1.0 / (1.0 + f1)
                delta2 = 1.0 / (1.0 + f2)
                for i, v in enumerate(chromosome):
                    self.tau1[i, v - 1] += delta1
                    self.tau2[i, v - 1] += delta2

    def _make_individual(
        self,
        chromosome: tuple[int, ...],
        f: tuple[float, float],
    ) -> SimpleNamespace:
        """Wrap a (chromosome, fitness) pair to look like an inspyred Individual.

        Exposes .candidate (list) and .fitness.values (tuple) so that the
        executer and visualization helpers can treat PACO output the same
        way they treat NSGA2/SPEA2 output.
        """
        return SimpleNamespace(
            candidate=list(chromosome),
            fitness=SimpleNamespace(values=f),
        )