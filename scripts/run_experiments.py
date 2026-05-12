"""Run grid search, mini-grid, and full experiments for the PG7 CVRP project.

Three phases:
    grid       — full grid search on giant_tour for each algorithm; saves
                 best settings to experiments/best_hyperparams.json.
    mini-grid  — small variation around the giant_tour optimum on
                 cluster_first; updates the JSON with per-representation
                 hyperparameters.
    full       — independent runs per (algorithm, representation, instance),
                 writing one CSV per (algorithm, representation, instance).

The --smoke-test flag runs a tiny end-to-end version that exercises the
same code paths in a few minutes: one instance, reduced grids, fewer
seeds, and small algorithm budgets. Output goes to tmp/ to keep
experiments/ clean.

Usage:
    python -m scripts.run_experiments --phase all
    python -m scripts.run_experiments --phase grid
    python -m scripts.run_experiments --phase mini-grid
    python -m scripts.run_experiments --phase full --n-jobs 4
    python -m scripts.run_experiments --phase all --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from cvrp.experiment.ga.nsga2_executer import NSGA2Executer
from cvrp.experiment.ga.spea2_executer import SPEA2Executer
from cvrp.experiment.swarm.paco_executer import PACOExecuter
from cvrp.metrics.pareto_metrics import compute_hv


INSTANCES_FOLDER = "data/"
GRID_INSTANCE = "X-n101-k25.vrp"

FULL_EXPERIMENTS_FOLDER = "experiments/"
FULL_BEST_HYPERPARAMS_PATH = os.path.join(FULL_EXPERIMENTS_FOLDER, "best_hyperparams.json")
FULL_N_GRID_SEEDS = 5
FULL_N_FULL_SEEDS = 31

FULL_NSGA2_SPEA2_GRID = {
    "pop_size": [50, 100],
    "crossover_rate": [0.7, 0.9],
    "mutation_rate": [0.05, 0.1, 0.2],
}
FULL_PACO_GRID = {
    "n_ants": [30, 50],
    "alpha1": [0.5, 1.0, 2.0],
    "alpha2": [0.5, 1.0, 2.0],
    "beta": [2.0, 5.0],
    "rho": [0.05, 0.1],
}

FULL_NSGA2_SPEA2_MINI_GRID_PARAM = "mutation_rate"
FULL_NSGA2_SPEA2_MINI_GRID_VALUES = [0.05, 0.1, 0.2]
FULL_PACO_MINI_GRID = {
    "alpha1": [0.5, 1.0, 2.0],
    "rho": [0.05, 0.1],
}

FULL_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    "nsga2": {},
    "spea2": {},
    "paco": {},
}

SMOKE_EXPERIMENTS_FOLDER = "tmp/experiments_smoke/"
SMOKE_BEST_HYPERPARAMS_PATH = os.path.join(SMOKE_EXPERIMENTS_FOLDER, "best_hyperparams.json")
SMOKE_N_GRID_SEEDS = 2
SMOKE_N_FULL_SEEDS = 3

SMOKE_NSGA2_SPEA2_GRID = {
    "pop_size": [20],
    "crossover_rate": [0.9],
    "mutation_rate": [0.05, 0.1],
}
SMOKE_PACO_GRID = {
    "n_ants": [10],
    "alpha1": [0.5, 1.0],
    "alpha2": [1.0],
    "beta": [2.0],
    "rho": [0.1],
}

SMOKE_NSGA2_SPEA2_MINI_GRID_PARAM = "mutation_rate"
SMOKE_NSGA2_SPEA2_MINI_GRID_VALUES = [0.05, 0.1]
SMOKE_PACO_MINI_GRID = {
    "alpha1": [0.5, 1.0],
    "rho": [0.1],
}

SMOKE_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    "nsga2": {"max_generations": 15},
    "spea2": {"max_generations": 15},
    "paco": {"max_iterations": 15, "archive_size": 20},
}


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of a {param: [values]} dict into a list of dicts."""
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def grid_search_algorithm(
    executer_class: type,
    instance_name: str,
    grid: dict[str, list],
    extra_kwargs: dict[str, Any],
    representation: str,
    n_seeds: int,
) -> dict[str, Any]:
    """Run grid search on one representation; return the best combination.

    All combinations and seeds are run first, then a single reference point
    is computed from the union of all fronts. Per-combination HV is the
    mean across seeds using that shared reference point.
    """
    combinations = expand_grid(grid)
    print(f"  Running {len(combinations)} combinations x {n_seeds} seeds = "
          f"{len(combinations) * n_seeds} runs on {representation}")

    runs: list[dict[str, Any]] = []
    for combo_idx, combo in enumerate(combinations):
        executer = executer_class(
            instances_folder=INSTANCES_FOLDER, representation=representation
        )
        for seed in range(n_seeds):
            alg = executer.run_single_experiment(
                instance_name, seed=seed, **combo, **extra_kwargs
            )
            front = [
                (ind.fitness.values[0], ind.fitness.values[1])
                for ind in alg.final_archive
            ]
            runs.append({
                "combo_idx": combo_idx,
                "combo": combo,
                "seed": seed,
                "front": front,
            })
        print(f"    Combo {combo_idx + 1}/{len(combinations)} done: {combo}")

    all_points = [p for r in runs for p in r["front"]]
    if not all_points:
        raise RuntimeError("All runs returned empty fronts; cannot compute HV.")
    max_f1 = max(p[0] for p in all_points)
    max_f2 = max(p[1] for p in all_points)
    ref_point = (max_f1 * 1.1, max_f2 * 1.1)

    combo_hvs: dict[int, float] = {}
    for combo_idx in range(len(combinations)):
        seed_hvs = [
            compute_hv(r["front"], ref_point)
            for r in runs
            if r["combo_idx"] == combo_idx
        ]
        combo_hvs[combo_idx] = sum(seed_hvs) / len(seed_hvs)

    best_idx = max(combo_hvs, key=combo_hvs.get)
    return {
        "hyperparams": combinations[best_idx],
        "hv": combo_hvs[best_idx],
    }


def build_mini_grid_for_ga(
    giant_tour_hyperparams: dict[str, Any],
    mini_param: str,
    mini_values: list,
) -> dict[str, list]:
    """Build a mini-grid that varies one parameter and fixes the rest.

    Used for NSGA2 and SPEA2: fixes pop_size and crossover_rate from
    the giant_tour optimum, varies mutation_rate (or another single param).
    """
    grid = {}
    for key, value in giant_tour_hyperparams.items():
        if key == mini_param:
            grid[key] = mini_values
        else:
            grid[key] = [value]
    return grid


def build_mini_grid_for_paco(
    giant_tour_hyperparams: dict[str, Any],
    mini_grid: dict[str, list],
) -> dict[str, list]:
    """Build a mini-grid that varies the chosen PACO parameters.

    Fixes n_ants, alpha2, beta from the giant_tour optimum.
    Varies alpha1 and rho across the values in mini_grid.
    """
    grid = {}
    for key, value in giant_tour_hyperparams.items():
        if key in mini_grid:
            grid[key] = mini_grid[key]
        else:
            grid[key] = [value]
    return grid


def run_one_task(
    executer_class: type,
    algorithm: str,
    representation: str,
    instance_name: str,
    hyperparams: dict[str, Any],
    extra_kwargs: dict[str, Any],
    n_repeat: int,
    output_root: str,
) -> str:
    """Run repeated experiments for a single (algorithm, representation, instance).

    The output CSV path includes the representation as a subfolder. If the
    CSV already exists, the task is skipped, which allows resuming after
    an interruption without re-running completed work. CSV write is atomic:
    data is written to a .tmp file first, then renamed.
    """
    output_folder = os.path.join(output_root, algorithm, representation)
    stem = os.path.splitext(instance_name)[0]
    csv_path = os.path.join(output_folder, f"{stem}.csv")

    if os.path.exists(csv_path):
        return f"SKIP {csv_path} (already exists)"

    os.makedirs(output_folder, exist_ok=True)
    executer = executer_class(
        instances_folder=INSTANCES_FOLDER, representation=representation
    )
    df = executer.run_repeated_experiment(
        instance_name, n_repeat=n_repeat, **hyperparams, **extra_kwargs
    )
    tmp_path = csv_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)
    return f"OK   {csv_path}"


def run_full_experiments(
    best_hyperparams: dict[str, dict[str, dict[str, Any]]],
    extra_kwargs_by_algo: dict[str, dict[str, Any]],
    instances: list[str],
    n_repeat: int,
    output_root: str,
    n_jobs: int,
) -> None:
    """Run all (algorithm, representation, instance) tasks in parallel.

    Reads hyperparameters separately per representation from the nested
    JSON structure produced by grid + mini-grid phases.
    """
    algorithms = [
        ("nsga2", NSGA2Executer),
        ("spea2", SPEA2Executer),
        ("paco", PACOExecuter),
    ]
    representations = ["giant_tour", "cluster_first"]

    for algorithm, _ in algorithms:
        for representation in representations:
            if representation not in best_hyperparams.get(algorithm, {}):
                raise KeyError(
                    f"Missing hyperparams for {algorithm}/{representation}. "
                    f"Run --phase mini first, or use --phase all."
                )

    tasks = []
    for algorithm, executer_class in algorithms:
        extra = extra_kwargs_by_algo[algorithm]
        for representation in representations:
            hyperparams = best_hyperparams[algorithm][representation]["hyperparams"]
            for instance_name in instances:
                tasks.append((
                    executer_class,
                    algorithm,
                    representation,
                    instance_name,
                    hyperparams,
                    extra,
                    n_repeat,
                    output_root,
                ))

    print(f"Total tasks: {len(tasks)} "
          f"({len(algorithms)} algos x {len(representations)} reps "
          f"x {len(instances)} instances)")
    print(f"Each task runs {n_repeat} seeds sequentially")
    print(f"Joblib parallelism: n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_task)(*t) for t in tasks
    )
    for r in results:
        print(r)


def run_grid_phase(
    nsga2_spea2_grid: dict[str, list],
    paco_grid: dict[str, list],
    extra_kwargs: dict[str, dict[str, Any]],
    n_seeds: int,
    best_hyperparams_path: str,
) -> None:
    """Phase 1: grid search on giant_tour for all three algorithms."""
    print("=" * 60)
    print("PHASE 1 — GRID SEARCH ON GIANT_TOUR")
    print("=" * 60)

    best: dict[str, dict[str, dict[str, Any]]] = {}

    print("\nNSGA2...")
    nsga2_gt = grid_search_algorithm(
        NSGA2Executer, GRID_INSTANCE, nsga2_spea2_grid,
        extra_kwargs["nsga2"], "giant_tour", n_seeds,
    )
    best["nsga2"] = {"giant_tour": nsga2_gt}
    print(f"  Best NSGA2/giant_tour: {nsga2_gt}")

    print("\nSPEA2...")
    spea2_gt = grid_search_algorithm(
        SPEA2Executer, GRID_INSTANCE, nsga2_spea2_grid,
        extra_kwargs["spea2"], "giant_tour", n_seeds,
    )
    best["spea2"] = {"giant_tour": spea2_gt}
    print(f"  Best SPEA2/giant_tour: {spea2_gt}")

    print("\nPACO...")
    paco_gt = grid_search_algorithm(
        PACOExecuter, GRID_INSTANCE, paco_grid,
        extra_kwargs["paco"], "giant_tour", n_seeds,
    )
    best["paco"] = {"giant_tour": paco_gt}
    print(f"  Best PACO/giant_tour: {paco_gt}")

    os.makedirs(os.path.dirname(best_hyperparams_path), exist_ok=True)
    with open(best_hyperparams_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved best hyperparams to {best_hyperparams_path}")


def run_mini_grid_phase(
    nsga2_spea2_mini_param: str,
    nsga2_spea2_mini_values: list,
    paco_mini_grid: dict[str, list],
    extra_kwargs: dict[str, dict[str, Any]],
    n_seeds: int,
    best_hyperparams_path: str,
) -> None:
    """Phase 2: mini-grid on cluster_first, anchored on the giant_tour optimum."""
    print("=" * 60)
    print("PHASE 2 — MINI-GRID ON CLUSTER_FIRST")
    print("=" * 60)

    with open(best_hyperparams_path) as f:
        best = json.load(f)

    print("\nNSGA2...")
    nsga2_gt_hp = best["nsga2"]["giant_tour"]["hyperparams"]
    nsga2_mini = build_mini_grid_for_ga(
        nsga2_gt_hp, nsga2_spea2_mini_param, nsga2_spea2_mini_values
    )
    nsga2_cf = grid_search_algorithm(
        NSGA2Executer, GRID_INSTANCE, nsga2_mini,
        extra_kwargs["nsga2"], "cluster_first", n_seeds,
    )
    best["nsga2"]["cluster_first"] = nsga2_cf
    print(f"  Best NSGA2/cluster_first: {nsga2_cf}")

    print("\nSPEA2...")
    spea2_gt_hp = best["spea2"]["giant_tour"]["hyperparams"]
    spea2_mini = build_mini_grid_for_ga(
        spea2_gt_hp, nsga2_spea2_mini_param, nsga2_spea2_mini_values
    )
    spea2_cf = grid_search_algorithm(
        SPEA2Executer, GRID_INSTANCE, spea2_mini,
        extra_kwargs["spea2"], "cluster_first", n_seeds,
    )
    best["spea2"]["cluster_first"] = spea2_cf
    print(f"  Best SPEA2/cluster_first: {spea2_cf}")

    print("\nPACO...")
    paco_gt_hp = best["paco"]["giant_tour"]["hyperparams"]
    paco_mini = build_mini_grid_for_paco(paco_gt_hp, paco_mini_grid)
    paco_cf = grid_search_algorithm(
        PACOExecuter, GRID_INSTANCE, paco_mini,
        extra_kwargs["paco"], "cluster_first", n_seeds,
    )
    best["paco"]["cluster_first"] = paco_cf
    print(f"  Best PACO/cluster_first: {paco_cf}")

    with open(best_hyperparams_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nUpdated best hyperparams in {best_hyperparams_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["grid", "mini-grid", "full", "all"],
        default="all",
        help="Which phase to run.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Joblib n_jobs for the full phase; -1 = all cores.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end version that finishes in a few minutes.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        experiments_folder = SMOKE_EXPERIMENTS_FOLDER
        best_hyperparams_path = SMOKE_BEST_HYPERPARAMS_PATH
        n_grid_seeds = SMOKE_N_GRID_SEEDS
        n_full_seeds = SMOKE_N_FULL_SEEDS
        nsga2_spea2_grid = SMOKE_NSGA2_SPEA2_GRID
        paco_grid = SMOKE_PACO_GRID
        nsga2_spea2_mini_param = SMOKE_NSGA2_SPEA2_MINI_GRID_PARAM
        nsga2_spea2_mini_values = SMOKE_NSGA2_SPEA2_MINI_GRID_VALUES
        paco_mini_grid = SMOKE_PACO_MINI_GRID
        extra_kwargs = SMOKE_EXTRA_KWARGS
        instances_override = [GRID_INSTANCE]
        print("*** SMOKE TEST MODE ***")
    else:
        experiments_folder = FULL_EXPERIMENTS_FOLDER
        best_hyperparams_path = FULL_BEST_HYPERPARAMS_PATH
        n_grid_seeds = FULL_N_GRID_SEEDS
        n_full_seeds = FULL_N_FULL_SEEDS
        nsga2_spea2_grid = FULL_NSGA2_SPEA2_GRID
        paco_grid = FULL_PACO_GRID
        nsga2_spea2_mini_param = FULL_NSGA2_SPEA2_MINI_GRID_PARAM
        nsga2_spea2_mini_values = FULL_NSGA2_SPEA2_MINI_GRID_VALUES
        paco_mini_grid = FULL_PACO_MINI_GRID
        extra_kwargs = FULL_EXTRA_KWARGS
        instances_override = None

    os.makedirs(experiments_folder, exist_ok=True)

    if args.phase in ("grid", "all"):
        run_grid_phase(
            nsga2_spea2_grid, paco_grid, extra_kwargs,
            n_grid_seeds, best_hyperparams_path,
        )

    if args.phase in ("mini-grid", "all"):
        run_mini_grid_phase(
            nsga2_spea2_mini_param, nsga2_spea2_mini_values,
            paco_mini_grid, extra_kwargs,
            n_grid_seeds, best_hyperparams_path,
        )

    if args.phase in ("full", "all"):
        print("\n" + "=" * 60)
        print("PHASE 3 — FULL EXPERIMENTS")
        print("=" * 60)

        with open(best_hyperparams_path) as f:
            best = json.load(f)

        if instances_override is not None:
            instances = instances_override
        else:
            dummy = NSGA2Executer(
                instances_folder=INSTANCES_FOLDER, representation="giant_tour"
            )
            instances = dummy.instances

        run_full_experiments(
            best, extra_kwargs, instances,
            n_full_seeds, experiments_folder, args.n_jobs,
        )
        print("\nDone.")


if __name__ == "__main__":
    main()