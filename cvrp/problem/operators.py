"""Genetic variation operators for the 2 CVRP representations

Representation A (Permutation -- Giant Tour)
    Use the built-in operators from inspyred.ec.variators:
        - partially_matched_crossover (PMX)
        - inversion_mutation or ec.variators.swap_mutation
    These are well-tested permutation operators that preserve the
    "each customer appears exactly once" invariant.

Representation B (Integer Vector -- Cluster-First, Route-Second)
    Generic vector operators don't fit a discrete bounded domain
    {1, ..., K}, so we implement two operators tailored to this one:
        - uniform_crossover: per-gene random parent selection
        - reset_mutation: per-gene random reassignment to {1, ..., K}

All operators here follow the inspyred variator signature:
    (random, candidates, args) -> list of offspring
"""

from __future__ import annotations

from random import Random
from typing import Any

from cvrp.problem.instance import CVRPInstance


# ----------------------------------------
# Cluster-first, Route-second(integer vector) operators
# ----------------------------------------
def make_uniform_crossover(instance: CVRPInstance) -> "Variator":
    """Factory for uniform crossover on integer chromosomes.

    For each gene position, the child inherits the value from either
    parent with equal probability. Produces two children per pair of
    parents. With probability (1 - crossover_rate) the parents are
    copied unchanged. The crossover rate is read from
    args["crossover_rate"] (default 0.8) so the calling algorithm can
    set it per call.
    """

    def variator(random: Random, candidates: list[list[int]], args: dict[str, Any]) -> list[list[int]]:
        crossover_rate = args.get("crossover_rate", 0.8)
        offspring: list[list[int]] = []
        # Process candidates two at a time as parent pairs.
        for i in range(0, len(candidates) - 1, 2):
            parent_a = candidates[i]
            parent_b = candidates[i + 1]
            if random.random() < crossover_rate:
                child_a, child_b = [], []
                for gene_a, gene_b in zip(parent_a, parent_b):
                    if random.random() < 0.5:
                        child_a.append(gene_a)
                        child_b.append(gene_b)
                    else:
                        child_a.append(gene_b)
                        child_b.append(gene_a)
                offspring.extend([child_a, child_b])
            else:
                offspring.extend([list(parent_a), list(parent_b)])
        # If candidates count is odd, copy the last one unchanged.
        if len(candidates) % 2 == 1:
            offspring.append(list(candidates[-1]))
        return offspring

    return variator


def make_reset_mutation(instance: CVRPInstance) -> "Variator":
    """Factory for reset mutation on integer chromosomes.

    Each gene is independently replaced with probability mutation_rate
    by a uniformly random vehicle id in {1, ..., K}. The mutation rate
    is read from args["mutation_rate"] (default 0.05) so the calling
    algorithm can set it per call.
    """
    num_vehicles = instance.num_vehicles

    def variator(random: Random, candidates: list[list[int]], args: dict[str, Any]) -> list[list[int]]:
        mutation_rate = args.get("mutation_rate", 0.05)
        offspring: list[list[int]] = []
        for chromosome in candidates:
            mutant = list(chromosome)
            for i in range(len(mutant)):
                if random.random() < mutation_rate:
                    mutant[i] = random.randint(1, num_vehicles)
            offspring.append(mutant)
        return offspring

    return variator