"""Pareto ACO experiment orchestrator — mirrors NSGA2Executer."""

from __future__ import annotations

import os
import shutil
from typing import Any

import pandas as pd

from cvrp.problem.cvrp_benchmark import CVRPBenchmark
from cvrp.problem.decoders import GiantTourDecoder
from cvrp.problem.generators import make_giant_tour_generator
from cvrp.problem.parser import parse_vrp_file
from cvrp.swarm.paco import ParetoACO


class PACOExecuter:
    """Run Pareto ACO experiments on benchmark instances.

    Layer 4: giant-tour representation only. Layer 5 can inject another
    decoder/generator pairing without modifying ``ParetoACO``.
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

    def run_single_experiment(
        self,
        instance_name: str,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ParetoACO:
        instance_path = os.path.join(self.instances_folder, instance_name)
        instance = parse_vrp_file(instance_path)
        decoder = GiantTourDecoder(instance)

        paco = ParetoACO(instance=instance, decoder=decoder, **kwargs)
        paco.optimize(seed=seed)
        return paco

    def run_repeated_experiment(
        self,
        instance_name: str,
        n_repeat: int = 31,
        **kwargs: Any,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run_idx in range(n_repeat):
            print(f"{instance_name} - run {run_idx + 1}/{n_repeat}")
            paco = self.run_single_experiment(
                instance_name, seed=run_idx, **kwargs
            )
            for sol_idx, ind in enumerate(paco.final_archive):
                f1, f2 = ind.fitness.values
                rows.append({
                    "run": run_idx,
                    "solution_idx": sol_idx,
                    "f1": f1,
                    "f2": f2,
                    "n_evaluations": paco.num_evaluations,
                    "n_generations": paco.num_generations,
                })
        return pd.DataFrame(rows)

    def run_all_experiments(
        self,
        experiment_folder: str,
        n_repeat: int = 31,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
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
        instance_path = os.path.join(self.instances_folder, instance_name)
        instance = parse_vrp_file(instance_path)
        decoder = GiantTourDecoder(instance)
        generator = make_giant_tour_generator(instance)
        return CVRPBenchmark(instance, decoder, generator)
