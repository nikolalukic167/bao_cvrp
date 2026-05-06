"""Plotting helpers for CVRP debugging.

Two minimal functions used to sanity-check decoded solutions during
algorithm development:
    plot_instance - depot + customers as a 2D scatter
    plot_routes   - colored polylines per vehicle, overlaid on the scatter
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Sequence
import matplotlib.cm as cm

from cvrp.problem.instance import CVRPInstance


def plot_instance(instance: CVRPInstance, ax: Axes | None = None) -> Axes:
    """Draw the depot and customers of a CVRP instance on a 2D axis.

    The depot is rendered as a large black square; customers are small
    blue circles. Aspect ratio is locked to 1:1 so that distances on the
    plot are visually faithful to the underlying Euclidean geometry.

    Args:
        instance: The problem instance to plot.
        ax: Optional matplotlib Axes to draw on. If None, a new figure
            and axes are created.

    Returns:
        The Axes used for plotting (newly created or the one passed in),
        so the caller can add overlays or customize further.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Customer scatter
    customer_ids = instance.customers
    xs = [instance.coords[c][0] for c in customer_ids]
    ys = [instance.coords[c][1] for c in customer_ids]
    ax.scatter(xs, ys, s=20, c="steelblue", alpha=0.8, label="Customers", zorder=2)

    # Depot marker on top of the scatter
    depot_x, depot_y = instance.coords[instance.depot]
    ax.scatter(
        depot_x, depot_y,
        s=180, c="black", marker="s", label="Depot", zorder=3,
    )

    ax.set_aspect("equal")
    ax.set_title(f"{instance.name}  (N={instance.num_customers}, K={instance.num_vehicles}, Q={instance.capacity})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    return ax

def plot_routes(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    ax: Axes | None = None,
) -> Axes:
    """Overlay vehicle routes on a CVRP map.

    Each route is drawn as a closed polyline depot -> customers -> depot,
    in a distinct color. Designed to be called after plot_instance() on
    the same axis, but works standalone too (it creates a new axis if
    none is given).

    Color palette:
        - tab20  for solutions with at most 20 routes (clearly distinct)
        - hsv    for solutions with more routes (cyclic, less distinct
          but always available)

    Args:
        routes: Decoded solution as a list of routes. Each route is a
            list of customer node ids (depot excluded).
        instance: The problem instance the routes belong to.
        ax: Optional matplotlib Axes to draw on. If None, a new figure
            and axes are created.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    depot = instance.depot
    depot_xy = instance.coords[depot]

    # Pick palette based on route count
    n_routes = len(routes)
    if n_routes <= 20:
        cmap = cm.get_cmap("tab20", max(n_routes, 1))
    else:
        cmap = cm.get_cmap("hsv", n_routes)

    for idx, route in enumerate(routes):
        if not route:
            continue

        # Build closed polyline: depot -> customers -> depot
        xs = [depot_xy[0]] + [instance.coords[c][0] for c in route] + [depot_xy[0]]
        ys = [depot_xy[1]] + [instance.coords[c][1] for c in route] + [depot_xy[1]]

        ax.plot(
            xs, ys,
            color=cmap(idx),
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )

    return ax