"""Initial-population generators for the 2 CVRP representations

Each generator produces a single random chromose. The signature follows
the inspyred convention: generator(random, args) -> chromosome, where
'random' is the PRNG (Pseudo-Random Number Generator) instance managed
by the algorithm (used for reproducibility) and 'args' is an optional
parameter dict that we ignore.

These factories take an instance and return a generator function bound
to that instance, which can be passed directly as the 'generator'
argument of inspyred algorithms.
"""


from __future__ import annotations

from random import Random
from typing import Any, Callable

from cvrp.problem.instance import CVRPInstance


Generator = Callable[[Random, dict[str, Any]], list[int]]

def make_giant_tour_generator(instance: CVRPInstance) -> Generator:
    """Generator factory for the giant-tour representation

    Produces a uniformly random permutation of all customer ids.
    """
    customers = instance.customers

    def generator(random: Random, args: dict[str, Any]) -> list[int]:
        chromosome = list(customers)
        random.shuffle(chromosome)
        return chromosome

    return generator

def make_cluster_first_generator(instance: CVRPInstance) -> Generator:
    """Generator factory for the cluster first representation\

    Produces a vector of length N (number of customers) where each gene
    is a uniformly random vehicle id in {1, ..., K}.
    """
    num_customers = instance.num_customers
    num_vehicles = instance.num_vehicles

    def generator(random: Random, args: dict[str, Any]) -> list[int]:
        return [random.randint(1, num_vehicles) for _ in range(num_customers)]

    return generator




















