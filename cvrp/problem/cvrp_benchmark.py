"""CVRPBenchmark: glues the problem instance, decoder, generator, and
fitness function into the inspyred Benchmark interface.

This is the single entry point used by every algorithm. The benchmark
is decoder-agnostic: swapping the decoder (and its matching generator)
is enough to switch between Representation A and Representation B.
"""

from __future__ import annotations

from typing import Any, Callable

from inspyred import benchmarks, ec
from inspyred.ec.emo import Pareto

from cvrp.problem.decoders import Decoder
from cvrp.problem.fitness import compute_f1, compute_f2
from cvrp.problem.instance import CVRPInstance


# Type alias matching the inspyred generator signature.
Generator = Callable[[Any, dict[str, Any]], list[int]]


class CVRPBenchmark(benchmarks.Benchmark):
    """Multi-objective CVRP benchmark for inspyred.

    Both objectives are minimized:
        f1 = total travel distance (with capacity-violation penalty)
        f2 = standard deviation of per-vehicle demand

    The chosen decoder determines the representation used. The matching
    generator must be supplied alongside it.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        decoder: Decoder,
        generator: Generator,
        penalty_weight: float = 1000.0,
    ) -> None:
        self.instance = instance
        self.decoder = decoder
        self._generator = generator
        self.penalty_weight = penalty_weight

        # Inspyred wants the dimensionality (chromosome length).
        super().__init__(decoder.chromosome_size())

        # Discrete bounder: keeps gene values inside the valid range.
        low, high = decoder.gene_bounds()
        self.bounder = ec.DiscreteBounder(list(range(low, high + 1)))

        # Both objectives are minimized.
        self.maximize = False

    def generator(self, random: Any, args: dict[str, Any]) -> list[int]:
        """Inspyred-facing generator: delegates to the bound generator."""
        return self._generator(random, args)

    def evaluator(
        self,
        candidates: list[list[int]],
        args: dict[str, Any],
    ) -> list[Pareto]:
        """Evaluate each candidate, returning Pareto fitness objects."""
        fitness: list[Pareto] = []
        for chromosome in candidates:
            routes = self.decoder.decode(chromosome)
            f1 = compute_f1(routes, self.instance, self.penalty_weight)
            f2 = compute_f2(routes, self.instance)
            fitness.append(Pareto([f1, f2]))
        return fitness