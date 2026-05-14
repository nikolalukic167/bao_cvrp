"""SPEA2 for Multi-Objective CVRP

Custom implementation of the Strength Pareto Evolutionary Algorithm 2
(Zitzler et al., 2001). Inspyred does not provide SPEA2, so we implement
it as a subclass of ec.EvolutionaryComputation following the pattern
from the course's reference notebook (multi_objective_implementation.ipynb):

  * 3 module-level helper functions implement the SPEA2 mechanics
    (strength-based fitness assignment, tournament selection,
    truncation replacement).
  * The SPEA2 class wires them in as selector and replacer, and uses
    inspyred's best_archiver to maintain the non-dominated archive A(t).

On top of that base, the class exposes a run(seed=None) method and
stores hyperparameters in __init__, so the executer can drive SPEA2 with
the same call-site as the NSGA2 wrapper.
"""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
from inspyred import ec

from cvrp.problem.cvrp_benchmark import CVRPBenchmark


def spea2_fitness_assignment(population: list, args: dict) -> np.ndarray:
    """SPEA2 fitness: F(i) = R(i) + D(i), to be minimized.

    R(i), the raw fitness, is the sum of strengths S(j) of all individuals
    j that dominate i. S(j) is the number of individuals j dominates.
    Non-dominated individuals have R(i) = 0.

    D(i), the density, is 1 / (sigma_i_k + 2), where sigma_i_k is the
    Euclidean distance in objective space to the k-th nearest neighbor,
    with k = sqrt(|population|) as suggested in the SPEA2 paper.
    """
    size = len(population)
    k = args.setdefault("k", int(np.sqrt(size)))

    # Strength S(i): number of individuals i dominates.
    strength = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i != j and population[i] > population[j]:
                strength[i] += 1

    # Raw fitness R(i): sum of strengths of i's dominators.
    raw_fitness = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i != j and population[j] > population[i]:
                raw_fitness[i] += strength[j]

    # k-NN density in objective space.
    distances = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                distances[i, j] = np.linalg.norm(
                    np.array(population[i].fitness.values)
                    - np.array(population[j].fitness.values)
                )

    fitness_values = np.zeros(size)
    for i in range(size):
        sorted_distances = np.sort(distances[i])
        density = 1.0 / (sorted_distances[k] + 2.0)
        fitness_values[i] = raw_fitness[i] + density

    return fitness_values


def spea2_tournament_selection(random: Random, population: list, args: dict) -> list:
    """Tournament selection driven by SPEA2 fitness (lower is better)."""
    fitness_assignment = spea2_fitness_assignment(population, args)
    num_selected = args.setdefault("num_selected", 1)
    tournament_size = args.setdefault("tournament_size", 2)
    if tournament_size > len(population):
        tournament_size = len(population)

    selected = []
    for _ in range(num_selected):
        tourn = random.sample(population, tournament_size)
        selected.append(
            min(tourn, key=lambda ind: fitness_assignment[population.index(ind)])
        )
    return selected


def spea2_replacement(
    random: Random, population: list, parents: list, offspring: list, args: dict
) -> list:
    """Combine population with offspring; keep |population| best by SPEA2 fitness."""
    combined = list(population)
    combined.extend(offspring)
    fitness_assignment = spea2_fitness_assignment(combined, args)
    sorted_inds = [ind for _, ind in sorted(zip(fitness_assignment, combined),
                                            key=lambda pair: pair[0])]
    return sorted_inds[: len(population)]


class SPEA2(ec.EvolutionaryComputation):
    """Run SPEA2 on a Multi-Objective CVRP benchmark.

    Subclasses inspyred's EvolutionaryComputation. Selector and replacer
    are the SPEA2 helpers above; archiver is best_archiver, which
    maintains the non-dominated archive A(t) (no fixed-size truncation,
    matching the reference notebook). Variators are injected by the
    caller, so the same class works with both representations: PMX +
    inversion for the giant-tour permutation (Rep A), and uniform +
    reset for the cluster-first integer encoding (Rep B).

    Attributes filled in by run():
        final_archive: list of non-dominated individuals (the Pareto
            front approximation found by the algorithm).
        history: list with one entry per generation. Each entry is a list
            of (chromosome, (f1, f2)) tuples — one per individual in that
            generation's population. Used for convergence and diversity
            plots.
        num_generations: number of generations actually executed
            (set by inspyred during evolve()).
        num_evaluations: total number of fitness evaluations performed
            (set by inspyred during evolve()).
    """

    def __init__(
        self,
        benchmark: CVRPBenchmark,
        variators: list,
        pop_size: int = 100,
        max_generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        tournament_size: int = 2,
    ) -> None:
        # Placeholder rng; replaced in run() with a seeded one.
        super().__init__(Random())

        # SPEA2-specific operators. The archiver maintains A(t).
        self.archiver = ec.archivers.best_archiver
        self.selector = spea2_tournament_selection
        self.replacer = spea2_replacement

        self.benchmark = benchmark
        self.variators = variators
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # Filled in by run()
        self.final_archive: list[Any] = []
        self.history: list[list[Any]] = []

    def _initialize(self) -> None:
        """Reset run-specific state so that calling run() multiple times is safe."""
        self.final_archive = []
        self.history = []
        self.num_generations = 0
        self.num_evaluations = 0

    def run(self, seed: int | None = None) -> list[Any]:
        """Execute the evolution loop and return the final non-dominated archive.

        Args:
            seed: Optional random seed. Pass a different value for each
                of the 31+ repeated runs in the experiment phase.
        """
        self._initialize()
        self._random = Random(seed)

        self.terminator = ec.terminators.generation_termination
        self.variator = self.variators
        self.observer = self._observer

        self.evolve(
            generator=self.benchmark.generator,
            evaluator=self.benchmark.evaluator,
            pop_size=self.pop_size,
            maximize=self.benchmark.maximize,
            bounder=self.benchmark.bounder,
            max_generations=self.max_generations,
            num_selected=self.pop_size,
            tournament_size=self.tournament_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )

        # best_archiver populates self.archive with non-dominated individuals.
        self.final_archive = list(self.archive)
        return self.final_archive

    def _observer(self, population, num_generations, num_evaluations, args) -> None:
        """Record a deep snapshot of the population every generation.

        Stores a list of (candidate, fitness.values) tuples so that the
        chromosome is preserved against later in-place mutation by inspyred.
        """
        from copy import deepcopy
        snapshot = [
            (deepcopy(ind.candidate), tuple(ind.fitness.values))
            for ind in population
        ]
        self.history.append(snapshot)