"""VRPLIB-format .vrp file parser."""

from __future__ import annotations

import os
import re

from cvrp.problem.instance import CVRPInstance


def parse_vrp_file(filepath: str) -> CVRPInstance:
    """Parse a VRPLIB-format .vrp file into a CVRPInstance.

    The parser handles the standard CVRP sections:
        - Header key-value pairs (NAME, DIMENSION, CAPACITY, ...)
        - NODE_COORD_SECTION (node id, x, y)
        - DEMAND_SECTION (node id, demand)
        - DEPOT_SECTION (depot ids terminated by -1)

    The number of vehicles K is not stored inside the file; it is parsed
    from the filename, which follows the convention "X-n[T]-k[V].vrp".

    Args:
        filepath: Path to the .vrp file.

    Returns:
        A fully populated CVRPInstance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed or required fields are missing.
    """
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"file not found: {filepath}")

    with open(filepath, "r") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    name: str | None = None
    dimension: int | None = None
    capacity: int | None = None
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depots: list[int] = []

    section: str | None = None

    for line in lines:
        stripped = line.strip()

        # End of file marker: stop parsing.
        if stripped == "EOF":
            break

        # Section headers switch the parsing mode for subsequent lines.
        if stripped.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if stripped.startswith("DEMAND_SECTION"):
            section = "demands"
            continue
        if stripped.startswith("DEPOT_SECTION"):
            section = "depot"
            continue

        # Header key-value pairs: "KEY : VALUE".
        if ":" in stripped and section is None:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "NAME":
                name = value
            elif key == "DIMENSION":
                dimension = int(value)
            elif key == "CAPACITY":
                capacity = int(value)
            continue

        # Body of a section: parse based on current mode.
        tokens = stripped.split()
        if section == "coords" and len(tokens) >= 3:
            node_id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            coords[node_id] = (x, y)
        elif section == "demands" and len(tokens) >= 2:
            node_id = int(tokens[0])
            demand = int(tokens[1])
            demands[node_id] = demand
        elif section == "depot" and len(tokens) >= 1:
            depot_id = int(tokens[0])
            if depot_id == -1:
                section = None  # Depot section terminated.
            else:
                depots.append(depot_id)

    # Validate that all required header fields were found.
    if name is None:
        raise ValueError(f"NAME not found in {filepath}")
    if dimension is None:
        raise ValueError(f"DIMENSION not found in {filepath}")
    if capacity is None:
        raise ValueError(f"CAPACITY not found in {filepath}")
    if not depots:
        raise ValueError(f"DEPOT_SECTION not found or empty in {filepath}")

    num_vehicles = _parse_num_vehicles_from_filename(filepath)

    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        num_vehicles=num_vehicles,
        depot=depots[0],
        coords=coords,
        demands=demands,
    )


def _parse_num_vehicles_from_filename(filepath: str) -> int:
    """Extract the number of vehicles K from a filename like 'X-n101-k25.vrp'."""
    filename = os.path.basename(filepath)
    match = re.search(r"-k(\d+)", filename)
    if not match:
        raise ValueError(
            f"cannot extract num_vehicles from filename '{filename}'; "
            f"expected pattern '...-k<N>.vrp'"
        )
    return int(match.group(1))