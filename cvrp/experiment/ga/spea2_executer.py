"""SPEA2 experiment orchestrator for the Multi-Objective CVRP.

Follows the Zubora Gabora ExperimentExecuter pattern: three public
methods (run_single_experiment, run_repeated_experiment,
run_all_experiments) that build a CVRPBenchmark from an instance file
and run the SPEA2 algorithm, persisting results to CSV.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pandas as pd
from inspyred import ec

from cvrp.ga.spea2 import SPEA2
from cvrp.problem.cvrp_benchmark import CVRPBenchmark
from cvrp.problem.decoders import GiantTourDecoder
from cvrp.problem.generators import make_giant_tour_generator
from cvrp.problem.parser import parse_vrp_file


class SPEA2Executer:
    """Runs SPEA2 experiments on the CVRP benchmark instances.

    Builds a CVRPBenchmark from each .vrp file, instantiates the SPEA2
    algorithm with the variators matching the chosen representation, and
    persists per-run results to CSV in long format (one row per Pareto
    solution per run).

    Only the giant-tour representation is supported in Layer 4. The
    cluster-first representation will be added in Layer 5 alongside
    its custom operators.
    """

    def __init__(
        self,
        instances_folder: str = "data/",
        representation: str = "giant_tour",
    ) -> None:
        if representation != "giant_tour":
            raise NotImplementedError(
                f"Representation {representation!r} not supported yet. "
                "Only 'giant_tour' is available in Layer 4; "
                "'cluster_first' will be added in Layer 5."
            )

        self.instances_folder = instances_folder
        self.representation = representation
        self.instances = sorted(
            f for f in os.listdir(instances_folder) if f.endswith(".vrp")
        )

        # Variators for Rep A (giant-tour permutation): PMX crossover
        # and inversion mutation, both built into inspyred.
        self.variators = [
            ec.variators.partially_matched_crossover,
            ec.variators.inversion_mutation,
        ]

    def run_single_experiment(
        self,
        instance_name: str,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SPEA2:
        """Run SPEA2 once on a single instance.

        Args:
            instance_name: Filename inside instances_folder, e.g. "X-n101-k25.vrp".
            seed: Random seed forwarded to SPEA2.run().
            **kwargs: Hyperparameters forwarded to SPEA2.__init__()
                (pop_size, max_generations, crossover_rate, etc.).

        Returns:
            The SPEA2 instance after run() completes. Access
            .final_archive, .history, .num_evaluations, .num_generations.
        """
        benchmark = self._build_benchmark(instance_name)
        spea2 = SPEA2(benchmark=benchmark, variators=self.variators, **kwargs)
        spea2.run(seed=seed)
        return spea2

    def run_repeated_experiment(
        self,
        instance_name: str,
        n_repeat: int = 31,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run SPEA2 n_repeat times on the same instance, varying the seed.

        Returns a long-format DataFrame: one row per Pareto solution
        per run. Columns: run, solution_idx, f1, f2, n_evaluations,
        n_generations.
        """
        rows: list[dict[str, Any]] = []
        for run_idx in range(n_repeat):
            print(f"{instance_name} - run {run_idx + 1}/{n_repeat}")
            spea2 = self.run_single_experiment(
                instance_name, seed=run_idx, **kwargs
            )
            for sol_idx, ind in enumerate(spea2.final_archive):
                f1, f2 = ind.fitness.values
                rows.append({
                    "run": run_idx,
                    "solution_idx": sol_idx,
                    "f1": f1,
                    "f2": f2,
                    "n_evaluations": spea2.num_evaluations,
                    "n_generations": spea2.num_generations,
                })
        return pd.DataFrame(rows)

    def run_all_experiments(
        self,
        experiment_folder: str,
        n_repeat: int = 31,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run repeated experiments across all instances, one CSV per instance.

        Args:
            experiment_folder: Output folder for CSVs (e.g. "experiments/spea2/").
            n_repeat: Number of independent runs per instance.
            overwrite: If True, wipe the output folder before writing.
            **kwargs: Hyperparameters forwarded to SPEA2 (applied to all
                instances; per-instance scaling is deferred to Layer 5).
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
            df = self.run_repeated_experiment(
                instance_name, n_repeat=n_repeat, **kwargs
            )
            stem = os.path.splitext(instance_name)[0]
            csv_path = os.path.join(experiment_folder, f"{stem}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")

    def _build_benchmark(self, instance_name: str) -> CVRPBenchmark:
        """Parse the instance file and build a CVRPBenchmark for it."""
        instance_path = os.path.join(self.instances_folder, instance_name)
        instance = parse_vrp_file(instance_path)
        decoder = GiantTourDecoder(instance)
        generator = make_giant_tour_generator(instance)
        return CVRPBenchmark(instance, decoder, generator)