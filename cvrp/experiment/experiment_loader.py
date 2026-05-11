"""Loader for experiment CSV results

Centralizes access to the per-algorithm CSV outputs produced by the
executers. Computes reference values (HV reference point, GD reference
front) from the full dataset so that metrics are comparable across
algorithms. Extracts scalar metric series ready to feed into the
statistical tests.
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd

from cvrp.metrics.pareto_metrics import (
    compute_gd,
    compute_hv,
    compute_pf_size,
    compute_sp,
)


Metric = Literal["hv", "sp", "gd", "pf"]


class ExperimentLoader:
    """Load and query experiment results across algorithms and instances.

    The constructor scans `experiments_folder` and loads every CSV into a
    single combined DataFrame, tagged with algorithm and instance columns.
    Subsequent queries operate on this in-memory frame without re-reading
    disk.

    Expected folder layout:
        experiments_folder/
            nsga2/X-n101-k25.csv
                  X-n129-k18.csv
            spea2/X-n101-k25.csv
                  ...
            paco/X-n101-k25.csv
                  ...

    Each CSV has the columns produced by the executers:
        run, solution_idx, f1, f2, n_evaluations, n_generations
    """

    def __init__(self, experiments_folder: str) -> None:
        self.experiments_folder = experiments_folder
        self.data = self._load_all()

    def _load_all(self) -> pd.DataFrame:
        """Read every CSV under experiments_folder into one DataFrame."""
        frames = []
        for algorithm in sorted(os.listdir(self.experiments_folder)):
            algorithm_dir = os.path.join(self.experiments_folder, algorithm)
            if not os.path.isdir(algorithm_dir):
                continue
            for csv_name in sorted(os.listdir(algorithm_dir)):
                if not csv_name.endswith(".csv"):
                    continue
                instance = os.path.splitext(csv_name)[0]
                df = pd.read_csv(os.path.join(algorithm_dir, csv_name))
                df["algorithm"] = algorithm
                df["instance"] = instance
                frames.append(df)

        if not frames:
            raise ValueError(
                f"No CSV files found under {self.experiments_folder!r}. "
                f"Run experiments first (Layer 5.4)."
            )

        return pd.concat(frames, ignore_index=True)

    def algorithms(self) -> list[str]:
        """Return the list of algorithms found in the data."""
        return sorted(self.data["algorithm"].unique())

    def instances(self) -> list[str]:
        """Return the list of instances found in the data."""
        return sorted(self.data["instance"].unique())

    def get_runs(
        self, algorithm: str, instance: str
    ) -> list[list[tuple[float, float]]]:
        """Return the per-run Pareto fronts for one (algorithm, instance).

        Each front is a list of (f1, f2) tuples. The outer list has one
        entry per run.
        """
        subset = self.data[
            (self.data["algorithm"] == algorithm)
            & (self.data["instance"] == instance)
        ]
        runs = []
        for run_idx in sorted(subset["run"].unique()):
            run_df = subset[subset["run"] == run_idx]
            front = list(zip(run_df["f1"].tolist(), run_df["f2"].tolist()))
            runs.append(front)
        return runs

    def compute_reference_point(
        self, instance: str
    ) -> tuple[float, float]:
        """Compute a reference point for HV on the given instance.

        Uses (max(f1) * 1.1, max(f2) * 1.1) across all algorithms and
        runs so the point strictly dominates every observed solution.
        """
        subset = self.data[self.data["instance"] == instance]
        return (subset["f1"].max() * 1.1, subset["f2"].max() * 1.1)

    def compute_reference_front(
        self, instance: str
    ) -> list[tuple[float, float]]:
        """Compute a pseudo-reference front for GD on the given instance.

        Combines all solutions from all algorithms and runs, then keeps
        only the non-dominated ones (both objectives minimized).
        """
        subset = self.data[self.data["instance"] == instance]
        points = list(zip(subset["f1"].tolist(), subset["f2"].tolist()))
        points = list(set(points))

        non_dominated = []
        for p in points:
            dominated = False
            for q in points:
                if q == p:
                    continue
                if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(p)
        return non_dominated

    def compute_metric_series(
        self, algorithm: str, instance: str, metric: Metric
    ) -> list[float]:
        """Compute a metric across runs for one (algorithm, instance).

        Returns a list with one scalar per run, in run order. Ready to
        feed into wilcoxon_test or friedman_shaffer_test.

        For HV, the reference point is computed from the full dataset.
        For GD, the reference front is computed from the full dataset.
        """
        runs = self.get_runs(algorithm, instance)

        if metric == "hv":
            ref_point = self.compute_reference_point(instance)
            return [compute_hv(front, ref_point) for front in runs]
        if metric == "sp":
            return [compute_sp(front) for front in runs]
        if metric == "gd":
            ref_front = self.compute_reference_front(instance)
            return [compute_gd(front, ref_front) for front in runs]
        if metric == "pf":
            return [float(compute_pf_size(front)) for front in runs]

        raise ValueError(
            f"Unknown metric {metric!r}. Must be one of: hv, sp, gd, pf."
        )