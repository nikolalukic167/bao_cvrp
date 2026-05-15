"""Plotting helpers for CVRP instances, routes, Pareto fronts, and experiment results

Includes geographic plots (``plot_instance``, ``plot_routes``), single-front
visualizations (``plot_pareto_front``), iteration-wise convergence
(``plot_convergence`` with f1 proxy, ``plot_hv_convergence`` with HV),
cross-algorithm comparison (``plot_pareto_fronts_overlay``,
``plot_metric_boxplots``), and grid-search analysis (``plot_parameter_heatmap``).
"""

from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Iterable, Sequence

import matplotlib.cm as cm
import numpy as np

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


def _f1_values(population: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for ind in population:
        fv = getattr(getattr(ind, "fitness", None), "values", None)
        if fv is None:
            continue
        out.append(float(fv[0]))
    return out


def plot_pareto_front(archive: Sequence[Any], ax: Axes | None = None) -> Axes:
    """Scatter f2 vs f1 for non-dominated individuals (both minimized)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    f1 = []
    f2 = []
    for ind in archive:
        fv = getattr(getattr(ind, "fitness", None), "values", None)
        if fv is None or len(fv) < 2:
            continue
        f1.append(float(fv[0]))
        f2.append(float(fv[1]))

    ax.scatter(f1, f2, s=36, alpha=0.75, edgecolors="steelblue")
    ax.set_xlabel(r"$f_1$ (distance + penalty)")
    ax.set_ylabel(r"$f_2$ (load imbalance)")
    ax.set_title("Pareto-front approximation")
    ax.grid(True, alpha=0.3)
    return ax


def plot_convergence(history: Sequence[Sequence[Any]], ax: Axes | None = None) -> Axes:
    """Per-step f1 envelopes (proxy until Layer ~5 hypervolume)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not history:
        return ax

    best: list[float] = []
    med: list[float] = []
    worst: list[float] = []
    for population in history:
        xs = _f1_values(population)
        if not xs:
            continue
        arr = np.asarray(xs, dtype=float)
        best.append(float(np.min(arr)))
        med.append(float(np.median(arr)))
        worst.append(float(np.max(arr)))

    gens = np.arange(len(best))
    ax.fill_between(gens, worst, best, alpha=0.2, label="range (f1)")
    ax.plot(gens, best, label="best $f_1$")
    ax.plot(gens, med, label="median $f_1$")
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel(r"$f_1$")
    ax.set_title("Convergence proxy (distance objective)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax

def plot_pareto_fronts_overlay(
    fronts: dict[str, list[tuple[float, float]]],
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Overlay multiple Pareto fronts on the same axes for visual comparison.

    Each entry of `fronts` is plotted as a connected line (sorted by f1)
    with a distinct color. Useful for comparing the final archives of
    different algorithms on the same (representation, instance), or for
    comparing the same algorithm on different representations.

    Args:
        fronts: Mapping from label to list of (f1, f2) points.
        ax: Optional matplotlib Axes. If None, a new figure is created.
        title: Optional plot title.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    cmap = cm.get_cmap("tab10", max(len(fronts), 1))

    for i, (label, front) in enumerate(fronts.items()):
        if not front:
            continue
        sorted_front = sorted(front, key=lambda p: p[0])
        xs = [p[0] for p in sorted_front]
        ys = [p[1] for p in sorted_front]
        ax.plot(
            xs, ys,
            marker="o", linestyle="-", linewidth=1.2, markersize=5,
            alpha=0.8, color=cmap(i), label=label,
        )

    ax.set_xlabel(r"$f_1$ (distance + penalty)")
    ax.set_ylabel(r"$f_2$ (load imbalance)")
    ax.set_title(title if title else "Pareto fronts comparison")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax


def plot_metric_boxplots(
    series: dict[str, list[float]],
    metric_name: str = "HV",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Side-by-side boxplots of a metric across runs, one box per algorithm.

    Each entry of `series` is the list of per-run metric values for one
    algorithm. Boxes share an axis so they're directly comparable.

    Args:
        series: Mapping from algorithm label to list of metric values
            (one per run).
        metric_name: Label for the y-axis (HV, SP, GD, etc).
        ax: Optional matplotlib Axes. If None, a new figure is created.
        title: Optional plot title.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    labels = list(series.keys())
    data = [series[lab] for lab in labels]

    box = ax.boxplot(data, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    cmap = cm.get_cmap("tab10", max(len(labels), 1))
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.5)

    ax.set_ylabel(metric_name)
    ax.set_title(title if title else f"{metric_name} distribution across runs")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_hv_convergence(
    history: Sequence[Sequence[Any]],
    ref_point: tuple[float, float],
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Hypervolume of the archive per generation.

    Multi-objective equivalent of the classic best-fitness convergence
    curve. For each generation in `history`, computes HV of the
    individuals using the given reference point. Multiple algorithms
    can be overlaid by calling this repeatedly on the same axes with
    different labels.

    Args:
        history: List of populations (or archives), one per generation.
            Each population is a sequence of objects exposing
            `.fitness.values = (f1, f2)`.
        ref_point: (f1, f2) reference point dominating every observed
            solution. Use ExperimentLoader.compute_reference_point.
        ax: Optional matplotlib Axes.
        label: Optional series label (for legend when overlaying).

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    from cvrp.metrics.pareto_metrics import compute_hv

    hvs: list[float] = []
    for population in history:
        fronts = []
        for ind in population:
            fv = getattr(getattr(ind, "fitness", None), "values", None)
            if fv is None or len(fv) < 2:
                continue
            fronts.append((float(fv[0]), float(fv[1])))
        hvs.append(compute_hv(fronts, ref_point) if fronts else 0.0)

    gens = list(range(len(hvs)))
    ax.plot(gens, hvs, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel("Hypervolume")
    ax.set_title("HV convergence")
    if label:
        ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax


def plot_parameter_heatmap(
    combos: Sequence[dict[str, Any]],
    param_x: str,
    param_y: str,
    aggregate: str = "max",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Heatmap of HV across two hyperparameters.

    Each combo is a dict with keys `hyperparams` (a dict of parameter
    values) and `hv_mean` (a float). For each (param_x value, param_y
    value) pair, HV values across the remaining parameters are
    aggregated by `aggregate` ("max" or "mean") and shown as a colored
    cell. Each cell is annotated with its numeric HV.

    Args:
        combos: Output of grid_search_results.json["..."]["..."]
            ["all_combos"].
        param_x: Hyperparameter name to put on the x-axis.
        param_y: Hyperparameter name to put on the y-axis.
        aggregate: "max" (best HV achievable with that x,y pair) or
            "mean" (average HV over remaining parameters).
        ax: Optional matplotlib Axes.
        title: Optional plot title.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    x_values = sorted({c["hyperparams"][param_x] for c in combos})
    y_values = sorted({c["hyperparams"][param_y] for c in combos})

    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for c in combos:
        xi = x_values.index(c["hyperparams"][param_x])
        yi = y_values.index(c["hyperparams"][param_y])
        buckets[(xi, yi)].append(c["hv_mean"])

    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for (xi, yi), hvs in buckets.items():
        if aggregate == "max":
            matrix[yi, xi] = max(hvs)
        elif aggregate == "mean":
            matrix[yi, xi] = sum(hvs) / len(hvs)
        else:
            raise ValueError(f"Unknown aggregate {aggregate!r}. Use 'max' or 'mean'.")

    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(title if title else f"HV heatmap ({aggregate} over other params)")
    plt.colorbar(im, ax=ax, label="HV")

    # Annotate cells with numeric values
    finite = matrix[np.isfinite(matrix)]
    if finite.size > 0:
        midpoint = (finite.max() + finite.min()) / 2
        for yi in range(len(y_values)):
            for xi in range(len(x_values)):
                v = matrix[yi, xi]
                if np.isfinite(v):
                    color = "white" if v < midpoint else "black"
                    ax.text(xi, yi, f"{v:.2e}", ha="center", va="center",
                            color=color, fontsize=8)

    return ax

def plot_fitness_convergence(
    f_arr: np.ndarray,
    objective_name: str = "f1",
    ax: Axes | None = None,
    label_prefix: str | None = None,
) -> Axes:
    """Convergence of one fitness objective over generations.

    For each generation, computes max (worst), mean, and min (best)
    fitness within each population, then averages those across runs.
    Three curves are drawn so that both the spread (max vs min) and the
    central tendency (mean) of the population are visible across the
    evolution.

    Args:
        f_arr: numpy array of shape (n_runs, n_gens, n_pop) with fitness
            values for one objective. Use load_history()["f1"] or ["f2"].
        objective_name: Label for the y-axis ("f1" or "f2").
        ax: Optional matplotlib Axes.
        label_prefix: Optional prefix for legend entries (e.g. algorithm
            name for overlays).

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    f_max = f_arr.max(axis=2).mean(axis=0)   # (n_gens,) worst-in-pop avg
    f_mean = f_arr.mean(axis=2).mean(axis=0) # (n_gens,) mean-in-pop avg
    f_min = f_arr.min(axis=2).mean(axis=0)   # (n_gens,) best-in-pop avg

    gens = np.arange(len(f_max))
    pre = f"{label_prefix} " if label_prefix else ""
    ax.fill_between(gens, f_max, f_min, alpha=0.15)
    ax.plot(gens, f_max, linestyle="--", linewidth=1.2, label=f"{pre}max")
    ax.plot(gens, f_mean, linewidth=1.8, label=f"{pre}mean")
    ax.plot(gens, f_min, linestyle="--", linewidth=1.2, label=f"{pre}min")

    ax.set_xlabel("Generation")
    ax.set_ylabel(objective_name)
    ax.set_title(f"{objective_name} convergence (averaged over runs)")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_chromosome_convergence(
    chromosomes: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Chromosome similarity within the population over generations.

    For each generation, for each position in the chromosome, the
    fraction of the population that carries the most common value at
    that position is computed; the curve plots the mean of those
    fractions across positions, averaged across runs.

    Interpretation: starts near 1/pop_size (random) and rises toward 1
    as the population converges to similar chromosomes. Complementary
    to fitness convergence: shows whether the population is exploring
    diverse genotypes or collapsing onto one solution.

    Args:
        chromosomes: numpy array of shape (n_runs, n_gens, n_pop,
            chrom_len). Use load_history()["chromosomes"].
        ax: Optional matplotlib Axes.
        label: Optional series label.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_runs, n_gens, n_pop, chrom_len = chromosomes.shape
    convergence = np.zeros((n_runs, n_gens))

    for r in range(n_runs):
        for g in range(n_gens):
            pop = chromosomes[r, g]  # (n_pop, chrom_len)
            per_pos_max_count = np.zeros(chrom_len)
            for p in range(chrom_len):
                _vals, counts = np.unique(pop[:, p], return_counts=True)
                per_pos_max_count[p] = counts.max()
            convergence[r, g] = per_pos_max_count.mean() / n_pop

    avg = convergence.mean(axis=0)
    gens = np.arange(len(avg))
    ax.plot(gens, avg, marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean per-position homogeneity")
    ax.set_title("Chromosome convergence")
    ax.set_ylim(0, 1.05)
    if label:
        ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax


def plot_diversity(
    chromosomes: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Number of distinct chromosomes in the population per generation.

    Counts unique rows in each (run, generation) slice and averages
    across runs. A monotonically decreasing curve indicates the
    population is collapsing to fewer distinct solutions; a flat curve
    near pop_size indicates sustained genetic diversity.

    Args:
        chromosomes: numpy array of shape (n_runs, n_gens, n_pop,
            chrom_len). Use load_history()["chromosomes"].
        ax: Optional matplotlib Axes.
        label: Optional series label.

    Returns:
        The Axes used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_runs, n_gens, n_pop, _chrom_len = chromosomes.shape
    unique_counts = np.zeros((n_runs, n_gens))

    for r in range(n_runs):
        for g in range(n_gens):
            pop = chromosomes[r, g]
            unique_counts[r, g] = len(np.unique(pop, axis=0))

    avg = unique_counts.mean(axis=0)
    gens = np.arange(len(avg))
    ax.plot(gens, avg, marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Distinct chromosomes")
    ax.set_title(f"Population diversity (out of {n_pop})")
    ax.set_ylim(0, n_pop * 1.05)
    if label:
        ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax


def plot_critical_difference(
    rankings: dict[str, float],
    post_hoc: list[dict[str, Any]],
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """N-vs-N significance graph for Friedman + Shaffer comparison.

    Algorithms are shown as nodes, sized and colored by their mean rank
    (lower = better, smaller node). An edge connects two algorithms
    only when they do NOT differ significantly after Shaffer correction.
    Disconnected algorithms differ significantly. If Friedman is not
    significant overall, all algorithms are connected (complete graph)
    to indicate global equivalence.

    Follows the visualization pattern from the BAO course's statistics
    comparison notebook.

    Args:
        rankings: Mapping from algorithm name to mean rank. Use
            friedman_shaffer_test(...)["rankings"].
        post_hoc: List of pairwise comparison dicts with keys
            "comparison" and "significant". Empty list if Friedman is
            not significant. Use friedman_shaffer_test(...)["post_hoc"].
        ax: Optional matplotlib Axes.
        title: Optional plot title.

    Returns:
        The Axes used for plotting.
    """
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    names = list(rankings.keys())
    ranks = [rankings[n] for n in names]

    if post_hoc:
        g = nx.Graph()
        g.add_nodes_from([(n, {"rank": rankings[n]}) for n in names])
        for ph in post_hoc:
            if not ph.get("significant", True):
                parts = ph["comparison"].split("vs")
                if len(parts) == 2:
                    name_l = parts[0].strip()
                    name_r = parts[1].strip()
                    g.add_edge(name_l, name_r)
    else:
        g = nx.complete_graph(names)
        for n in names:
            g.nodes[n]["rank"] = rankings[n]

    pos = nx.kamada_kawai_layout(g, scale=0.5)
    node_sizes = [g.nodes[n]["rank"] * 80 for n in g.nodes()]
    node_colors = [g.nodes[n]["rank"] for n in g.nodes()]

    nx.draw_networkx(
        g, pos=pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="plasma",
        with_labels=True,
        font_color="w",
        font_size=9,
        font_weight="bold",
    )

    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=min(ranks), vmax=max(ranks)),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean rank (lower = better)")

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax