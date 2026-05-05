"""Solution decoders for the CVRP.

A decoder converts an abstract chromosome (the vector that the algorithm
manipulates) into a concrete set of routes (the real-world solution that
can be evaluated).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from cvrp.problem.instance import CVRPInstance


class Decoder(ABC):
    """Abstract base class for chromosome-to-routes decoders.

    A route is a list of customer node ids. The depot (always node 1) is
    implicit at the start and end of every route, so it never appears in
    the returned lists.
    """

    def __init__(self, instance: CVRPInstance) -> None:
        self.instance = instance

    @abstractmethod
    def decode(self, chromosome: Sequence[int]) -> list[list[int]]:
        """Convert a chromosome into a list of routes.

        Args:
            chromosome: The encoded solution, as a sequence of integers.

        Returns:
            A list of routes. Each route is a list of customer ids
            (depot excluded; it bookends every route implicitly).
        """

    def chromosome_size(self) -> int:
        """Length of the chromosome for this representation."""
        return self.instance.num_customers

    @abstractmethod
    def gene_bounds(self) -> tuple[int, int]:
        """Inclusive (min, max) bounds for each gene value."""


class GiantTourDecoder(Decoder):
    """Giant-tour representation with greedy capacity split.

    The chromosome is a permutation of all customer ids. Decoding scans
    it left-to-right, packing customers onto the current vehicle until
    the next one would exceed capacity Q; at that point a new vehicle
    is opened. Capacity feasibility is therefore guaranteed by construction.
    """

    def decode(self, chromosome: Sequence[int]) -> list[list[int]]:
        capacity = self.instance.capacity
        demands = self.instance.demands

        routes: list[list[int]] = []
        current_route: list[int] = []
        current_load = 0

        for customer in chromosome:
            demand = demands[customer]
            if current_load + demand <= capacity:
                # Customer fits on the current vehicle.
                current_route.append(customer)
                current_load += demand
            else:
                # Close the current route and start a new one.
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        # Append the last route if non-empty.
        if current_route:
            routes.append(current_route)

        return routes

    def gene_bounds(self) -> tuple[int, int]:
        # Genes are customer ids (excludes the depot).
        customers = self.instance.customers
        return (min(customers), max(customers))

class ClusterFirstDecoder(Decoder):
    """Cluster-first, route-second representation.

    The chromosome has one gene per customer; gene i is the id of the
    vehicle assigned to customer i (in {1, ..., K}).

    Decoding has two phases:
        1. Cluster: group customers by their assigned vehicle.
        2. Route: order each cluster using nearest-neighbor starting
           from the depot.

    Capacity feasibility is NOT guaranteed: a cluster may exceed Q.
    Such infeasibility is handled in the fitness function via a penalty,
    not here, so this decoder remains a pure chromosome-to-routes mapping.
    """

    def decode(self, chromosome: Sequence[int]) -> list[list[int]]:
        instance = self.instance
        customers = instance.customers
        num_vehicles = instance.num_vehicles

        if len(chromosome) != len(customers):
            raise ValueError(
                f"chromosome length {len(chromosome)} does not match "
                f"number of customers {len(customers)}"
            )

        # Phase 1 - Cluster: group customers by assigned vehicle.
        clusters: dict[int, list[int]] = {v: [] for v in range(1, num_vehicles + 1)}
        for customer, vehicle in zip(customers, chromosome):
            clusters[vehicle].append(customer)

        # Phase 2 - Route: order each non-empty cluster with nearest-neighbor.
        routes: list[list[int]] = []
        for vehicle in range(1, num_vehicles + 1):
            cluster = clusters[vehicle]
            if cluster:
                routes.append(self._nearest_neighbor_order(cluster))

        return routes

    def _nearest_neighbor_order(self, cluster: list[int]) -> list[int]:
        """Order a cluster of customers using nearest-neighbor from the depot."""
        instance = self.instance
        depot = instance.depot
        unvisited = list(cluster)
        ordered: list[int] = []
        current = depot

        while unvisited:
            nearest = min(unvisited, key=lambda c: instance.distance(current, c))
            ordered.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return ordered

    def gene_bounds(self) -> tuple[int, int]:
        return (1, self.instance.num_vehicles)