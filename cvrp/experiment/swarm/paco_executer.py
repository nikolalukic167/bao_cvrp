"""Pareto ACO experiment orchestrator

Provides three public methods (run_single_experiment,
run_repeated_experiment, run_all_experiments) that build the problem
artifacts from an instance file and run the ParetoACO algorithm,
persisting results to CSV.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pandas as pd

from cvrp.problem.cvrp_benchmark import CVRPBenchmark
from cvrp.problem.decoders import ClusterFirstDecoder, GiantTourDecoder
from cvrp.problem.generators import (
    make_cluster_first_generator,
    make_giant_tour_generator,
)
from cvrp.problem.parser import parse_vrp_file
from cvrp.swarm.paco import ParetoACO


class PACOExecuter:
    """Run Pareto ACO experiments on benchmark instances.

    Two representations are supported:
        "giant_tour" — pheromones on customer-pairs; ants build a
                       permutation of customer ids.
        "cluster_first" — pheromones on customer-vehicle pairs; ants
                          build an assignment vector mapping each
                          customer to a vehicle.
    """

    def __init__(
        self,
        instances_folder: str = "data/",
        representation: str = "giant_tour",
    ) -> None:
        if representation not in ("giant_tour", "cluster_first"):
            raise ValueError(
                f"Unknown representation {representation!r}. "
                "Must be 'giant_tour' or 'cluster_first'."
            )

        self.instances_folder = instances_folder
        self.representation = representation
        self.instances = sorted(
            f for f in os.listdir(instances_folder) if f.endswith(".vrp")
        )

    def run_single_experiment(
        self,
        instance_name: str,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ParetoACO:
        """Run PACO once on a single instance.

        Args:
            instance_name: Filename inside instances_folder, e.g. "X-n101-k25.vrp".
            seed: Random seed forwarded to ParetoACO.optimize().
            **kwargs: Hyperparameters forwarded to ParetoACO.__init__()
                (n_ants, max_iterations, alpha1, alpha2, beta, rho, etc.).

        Returns:
            The ParetoACO instance after optimize() completes. Access
            .final_archive, .history, .num_evaluations, .num_generations.
        """
        benchmark = self._build_benchmark(instance_name)
        paco = ParetoACO(
            instance=benchmark.instance,
            decoder=benchmark.decoder,
            representation=self.representation,
            **kwargs,
        )
        paco.optimize(seed=seed)
        return paco

    def run_repeated_experiment(
            self,
            instance_name: str,
            n_repeat: int = 31,
            **kwargs: Any,
    ) -> tuple[pd.DataFrame, list]:
        """Run PACO n_repeat times on the same instance, varying the seed.

        Returns:
            df: Long-format DataFrame with one row per Pareto solution per
                run. Columns: run, solution_idx, f1, f2, chromosome,
                n_evaluations, n_generations. The chromosome column is a
                comma-separated string of integers, parseable with
                list(map(int, s.split(','))).
            histories: List of length n_repeat. histories[i] is the
                per-iteration history of run i (list of iterations, each a
                list of (chromosome, (f1, f2)) tuples for the ants in that
                iteration). For persistence to .npz, pass this to
                cvrp.experiment.history_io.save_history.
        """
        rows: list[dict[str, Any]] = []
        histories: list = []
        for run_idx in range(n_repeat):
            print(f"{instance_name} - run {run_idx + 1}/{n_repeat}")
            paco = self.run_single_experiment(
                instance_name, seed=run_idx, **kwargs
            )
            histories.append(paco.history)
            for sol_idx, ind in enumerate(paco.final_archive):
                f1, f2 = ind.fitness.values
                rows.append({
                    "run": run_idx,
                    "solution_idx": sol_idx,
                    "f1": f1,
                    "f2": f2,
                    "chromosome": ",".join(str(g) for g in ind.candidate),
                    "n_evaluations": paco.num_evaluations,
                    "n_generations": paco.num_generations,
                })
        return pd.DataFrame(rows), histories

    def run_all_experiments(
        self,
        experiment_folder: str,
        n_repeat: int = 31,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run repeated experiments across all instances, one CSV per instance.

        Args:
            experiment_folder: Output folder for CSVs.
            n_repeat: Number of independent runs per instance.
            overwrite: If True, wipe the output folder before writing.
            **kwargs: Hyperparameters forwarded to ParetoACO.
        """
        if os.path.isdir(experiment_folder):
            if overwrite:
                shutil.rmtree(experiment_folder)
            else:
                raise ValueError(
                    f"Folder {experiment_folder} already exists. "
                    "Set overwrite=True to wipe and rewrite it."
                )
        os.makedirs(experiment_folder)

        for instance_name in self.instances:
            print(f"Running experiments for {instance_name}")
            df, _histories = self.run_repeated_experiment(
                instance_name, n_repeat=n_repeat, **kwargs
            )
            stem = os.path.splitext(instance_name)[0]
            csv_path = os.path.join(experiment_folder, f"{stem}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")

    def _build_benchmark(self, instance_name: str) -> CVRPBenchmark:
        """Parse the instance file and build a CVRPBenchmark.

        The decoder and generator are chosen by self.representation.
        """
        instance_path = os.path.join(self.instances_folder, instance_name)
        instance = parse_vrp_file(instance_path)

        if self.representation == "giant_tour":
            decoder = GiantTourDecoder(instance)
            generator = make_giant_tour_generator(instance)
        else:
            decoder = ClusterFirstDecoder(instance)
            generator = make_cluster_first_generator(instance)

        return CVRPBenchmark(instance, decoder, generator)