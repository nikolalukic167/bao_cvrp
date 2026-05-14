"""NSGA-II for Multi-Objective CVRP.

Thin wrapper around inspyred.ec.emo.NSGA2 that binds it to a CVRPBenchmark.
The class is responsible only for orchestrating the evolution loop and
exposing the final non-dominated archive plus a per-generation history.
"""

from __future__ import annotations

from random import Random
from typing import Any

from inspyred import ec

from cvrp.problem.cvrp_benchmark import CVRPBenchmark


class NSGA2:
    """Run NSGA-II on a Multi-Objective CVRP benchmark.

    Uses inspyred's built-in NSGA-II implementation. Variators are
    injected by the caller, which makes the same class usable with both
    representations: PMX + inversion for the giant-tour permutation
    (Rep A), and uniform crossover + reset mutation for the cluster-first
    integer encoding (Rep B).

    Attributes filled in by run():
        final_archive: list of non-dominated individuals (the Pareto front
            approximation found by the algorithm).
        history: list with one entry per generation. Each entry is a list
            of (chromosome, (f1, f2)) tuples — one per individual in that
            generation's population. Used for convergence and diversity
            plots.
        num_generations: number of generations actually executed.
        num_evaluations: total number of fitness evaluations performed.
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
        self.num_generations: int = 0
        self.num_evaluations: int = 0

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
        rng = Random(seed)

        algorithm = ec.emo.NSGA2(rng)
        algorithm.terminator = ec.terminators.generation_termination
        algorithm.variator = self.variators
        algorithm.observer = self._observer

        algorithm.evolve(
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

        # NSGA2 in inspyred maintains an archive of non-dominated solutions
        self.final_archive = list(algorithm.archive)
        self.num_generations = algorithm.num_generations
        self.num_evaluations = algorithm.num_evaluations
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