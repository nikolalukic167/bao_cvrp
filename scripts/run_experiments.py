"""Run grid search and full experiments for the PG7 CVRP project.

Two phases:
    grid — full grid search on both representations for each algorithm;
           saves best settings to experiments/best_hyperparams.json.
    full — independent runs per (algorithm, representation, instance),
           writing one CSV per (algorithm, representation, instance).

The --smoke-test flag runs a tiny end-to-end version that exercises the
same code paths in a few minutes: one instance, reduced grids, fewer
seeds, and small algorithm budgets. Output goes to tmp/ to keep
experiments/ clean.

Usage:
    python -m scripts.run_experiments --phase all
    python -m scripts.run_experiments --phase grid
    python -m scripts.run_experiments --phase full --n-jobs 4
    python -m scripts.run_experiments --phase all --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from typing import Any

from joblib import Parallel, delayed

from cvrp.experiment.ga.nsga2_executer import NSGA2Executer
from cvrp.experiment.ga.spea2_executer import SPEA2Executer
from cvrp.experiment.swarm.paco_executer import PACOExecuter
from cvrp.metrics.pareto_metrics import compute_hv


INSTANCES_FOLDER = "data/"
GRID_INSTANCE = "X-n101-k25.vrp"

FULL_EXPERIMENTS_FOLDER = "experiments/"
FULL_BEST_HYPERPARAMS_PATH = os.path.join(FULL_EXPERIMENTS_FOLDER, "best_hyperparams.json")
FULL_GRID_RESULTS_PATH = os.path.join(FULL_EXPERIMENTS_FOLDER, "grid_search_results.json")
FULL_N_GRID_SEEDS = 5
FULL_N_FULL_SEEDS = 31

FULL_NSGA2_SPEA2_GRID = {
    "pop_size": [50, 100],
    "crossover_rate": [0.7, 0.9],
    "mutation_rate": [0.05, 0.1, 0.2],
}
FULL_PACO_GRID_GIANT_TOUR = {
    "n_ants": [30, 50],
    "alpha1": [0.5, 1.0, 2.0],
    "alpha2": [0.5, 1.0, 2.0],
    "beta": [2.0, 5.0],
    "rho": [0.05, 0.1],
}
# beta is excluded: on cluster_first construction uses only
# tau1^alpha1 * tau2^alpha2 — no geometric heuristic eta exists
# between a customer and a vehicle id, so beta has no effect.
FULL_PACO_GRID_CLUSTER_FIRST = {
    "n_ants": [30, 50],
    "alpha1": [0.5, 1.0, 2.0],
    "alpha2": [0.5, 1.0, 2.0],
    "rho": [0.05, 0.1],
}

FULL_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    "nsga2": {},
    "spea2": {},
    "paco": {},
}

SMOKE_EXPERIMENTS_FOLDER = "tmp/experiments_smoke/"
SMOKE_BEST_HYPERPARAMS_PATH = os.path.join(SMOKE_EXPERIMENTS_FOLDER, "best_hyperparams.json")
SMOKE_GRID_RESULTS_PATH = os.path.join(SMOKE_EXPERIMENTS_FOLDER, "grid_search_results.json")
SMOKE_N_GRID_SEEDS = 2
SMOKE_N_FULL_SEEDS = 3

SMOKE_NSGA2_SPEA2_GRID = {
    "pop_size": [20],
    "crossover_rate": [0.9],
    "mutation_rate": [0.05, 0.1],
}
SMOKE_PACO_GRID_GIANT_TOUR = {
    "n_ants": [10],
    "alpha1": [0.5, 1.0],
    "alpha2": [1.0],
    "beta": [2.0],
    "rho": [0.1],
}
SMOKE_PACO_GRID_CLUSTER_FIRST = {
    "n_ants": [10],
    "alpha1": [0.5, 1.0],
    "alpha2": [1.0],
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
    """Run grid search on one representation; return the best combination
    plus per-combination metadata for downstream analysis.

    All combinations and seeds are run first, then a single reference point
    is computed from the union of all fronts. Per-combination HV is the
    mean across seeds using that shared reference point.

    Returns a dict with:
        hyperparams - the winning hyperparameter combination
        hv         - mean HV of the winning combination
        ref_point  - reference point used (max_f1*1.1, max_f2*1.1)
        all_combos - list of {hyperparams, hv_mean, hv_per_seed} for every
                     combination, in the order produced by expand_grid.
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

    combo_seed_hvs: dict[int, list[float]] = {}
    combo_mean_hvs: dict[int, float] = {}
    for combo_idx in range(len(combinations)):
        seed_hvs = [
            compute_hv(r["front"], ref_point)
            for r in runs
            if r["combo_idx"] == combo_idx
        ]
        combo_seed_hvs[combo_idx] = seed_hvs
        combo_mean_hvs[combo_idx] = sum(seed_hvs) / len(seed_hvs)

    best_idx = max(combo_mean_hvs, key=combo_mean_hvs.get)

    all_combos = [
        {
            "hyperparams": combinations[i],
            "hv_mean": combo_mean_hvs[i],
            "hv_per_seed": combo_seed_hvs[i],
        }
        for i in range(len(combinations))
    ]

    return {
        "hyperparams": combinations[best_idx],
        "hv": combo_mean_hvs[best_idx],
        "ref_point": list(ref_point),
        "all_combos": all_combos,
    }


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
    JSON structure produced by the grid phase.
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
                    f"Run --phase grid first, or use --phase all."
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
    paco_grid_giant_tour: dict[str, list],
    paco_grid_cluster_first: dict[str, list],
    extra_kwargs: dict[str, dict[str, Any]],
    n_seeds: int,
    best_hyperparams_path: str,
    grid_results_path: str,
) -> None:
    """Grid search on both representations for all three algorithms.

    Writes two output files:
        best_hyperparams_path - only the winning combo per (algorithm,
            representation), consumed by the full phase. Structure is
            unchanged from earlier runs to keep that phase backward
            compatible.
        grid_results_path - every combination with per-seed HVs and the
            reference point used, for downstream heatmap and analysis.

    PACO uses separate grids per representation: beta is included for
    giant_tour (where eta^beta guides construction geometrically) but
    excluded for cluster_first (where no geometric heuristic exists
    between a customer and a vehicle id, so beta has no effect).
    """
    print("=" * 60)
    print("PHASE 1 — GRID SEARCH")
    print("=" * 60)

    ga_algorithms = [
        ("nsga2", NSGA2Executer, nsga2_spea2_grid),
        ("spea2", SPEA2Executer, nsga2_spea2_grid),
    ]
    paco_grids = {
        "giant_tour": paco_grid_giant_tour,
        "cluster_first": paco_grid_cluster_first,
    }
    representations = ["giant_tour", "cluster_first"]

    best: dict[str, dict[str, dict[str, Any]]] = {}
    grid_results: dict[str, dict[str, dict[str, Any]]] = {}

    def store(algorithm: str, representation: str, result: dict[str, Any]) -> None:
        best[algorithm][representation] = {
            "hyperparams": result["hyperparams"],
            "hv": result["hv"],
        }
        grid_results[algorithm][representation] = {
            "ref_point": result["ref_point"],
            "all_combos": result["all_combos"],
        }

    for algorithm, executer_class, grid in ga_algorithms:
        print(f"\n{algorithm.upper()}...")
        best[algorithm] = {}
        grid_results[algorithm] = {}
        for representation in representations:
            result = grid_search_algorithm(
                executer_class, GRID_INSTANCE, grid,
                extra_kwargs[algorithm], representation, n_seeds,
            )
            store(algorithm, representation, result)
            print(f"  Best {algorithm}/{representation}: "
                  f"hyperparams={result['hyperparams']}, hv={result['hv']:.2f}")

    print("\nPACO...")
    best["paco"] = {}
    grid_results["paco"] = {}
    for representation in representations:
        result = grid_search_algorithm(
            PACOExecuter, GRID_INSTANCE, paco_grids[representation],
            extra_kwargs["paco"], representation, n_seeds,
        )
        store("paco", representation, result)
        print(f"  Best paco/{representation}: "
              f"hyperparams={result['hyperparams']}, hv={result['hv']:.2f}")

    os.makedirs(os.path.dirname(best_hyperparams_path) or ".", exist_ok=True)
    with open(best_hyperparams_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved best hyperparams to {best_hyperparams_path}")

    os.makedirs(os.path.dirname(grid_results_path) or ".", exist_ok=True)
    with open(grid_results_path, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"Saved full grid results to {grid_results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["grid", "full", "all"],
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
        grid_results_path = SMOKE_GRID_RESULTS_PATH
        n_grid_seeds = SMOKE_N_GRID_SEEDS
        n_full_seeds = SMOKE_N_FULL_SEEDS
        nsga2_spea2_grid = SMOKE_NSGA2_SPEA2_GRID
        paco_grid_giant_tour = SMOKE_PACO_GRID_GIANT_TOUR
        paco_grid_cluster_first = SMOKE_PACO_GRID_CLUSTER_FIRST
        extra_kwargs = SMOKE_EXTRA_KWARGS
        instances_override = [GRID_INSTANCE]
        print("*** SMOKE TEST MODE ***")
    else:
        experiments_folder = FULL_EXPERIMENTS_FOLDER
        best_hyperparams_path = FULL_BEST_HYPERPARAMS_PATH
        grid_results_path = FULL_GRID_RESULTS_PATH
        n_grid_seeds = FULL_N_GRID_SEEDS
        n_full_seeds = FULL_N_FULL_SEEDS
        nsga2_spea2_grid = FULL_NSGA2_SPEA2_GRID
        paco_grid_giant_tour = FULL_PACO_GRID_GIANT_TOUR
        paco_grid_cluster_first = FULL_PACO_GRID_CLUSTER_FIRST
        extra_kwargs = FULL_EXTRA_KWARGS
        instances_override = None

    os.makedirs(experiments_folder, exist_ok=True)

    if args.phase in ("grid", "all"):
        run_grid_phase(
            nsga2_spea2_grid, paco_grid_giant_tour, paco_grid_cluster_first,
            extra_kwargs, n_grid_seeds, best_hyperparams_path,
            grid_results_path,
        )

    if args.phase in ("full", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2 — FULL EXPERIMENTS")
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