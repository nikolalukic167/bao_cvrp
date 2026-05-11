"""CVRP problem instance: in-memort representation"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

@dataclass
class CVRPInstance:
    """Multi-objective CVRP instance loaded from a VRPLIB-format .vrp file

    Node indices follow the VRPLIB convention: they start at 1, and node 1 is
    always the depot. Customers are nodes 2 through `dimension`.

    Attributes:
        name: Instance identifier (e.g. "X-n101-k25").
        dimension: Total number of nodes (depot + customers).
        capacity: Maximum load Q a single vehicle can carry.
        num_vehicles: Number of available vehicles K, parsed from the filename.
        depot: Index of the depot node (always 1 for standard CVRP).
        coords: Mapping node_id -> (x, y) coordinates.
        demands: Mapping node_id -> demand value.
    """

    name: str
    dimension: int
    capacity: int
    num_vehicles: int
    depot: int
    coords: dict[int, tuple[float, float]] = field(default_factory=dict)
    demands: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension <= 1:
            raise ValueError(f"dimension must be > 1, got  {self.dimension}")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got  {self.capacity}")
        if self.num_vehicles <= 0:
            raise ValueError(f"num_vehicles must be > 0, got  {self.num_vehicles}")
        if len(self.coords) != self.dimension:
            raise ValueError(
                f"expected {self.dimension} coordinates, got  {len(self.coords)}"
            )
        if len(self.demands) != self.dimension:
            raise ValueError(
                f"expected {self.dimension} demands, got  {len(self.demands)}"
            )
        if self.demands.get(self.depot, -1) != 0:
            raise ValueError(
                f"depot (node {self.depot} must have demand 0, "
                f"got  {self.demands.get(self.depot)}"
            )

    @property
    def num_customers(self) -> int:
        """Number of customer nodes (excludes the depot)"""
        return self.dimension - 1

    @property
    def customers(self) -> list[int]:
        """List of customer node ids, sorted ascending."""
        return [node for node in sorted(self.coords) if node != self.depot]

    @property
    def total_demand(self) -> int:
        """Sum of demands across all customers."""
        return sum(self.demands[customer] for customer in self.customers)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        return math.hypot(xi - xj, yi - yj)

    def __repr__(self) -> str:
        return (
            f"CVRPInstance(name='{self.name}', "
            f"customers={self.num_customers}, "
            f"vehicles={self.num_vehicles}, "
            f"capacity={self.capacity})"
        )



