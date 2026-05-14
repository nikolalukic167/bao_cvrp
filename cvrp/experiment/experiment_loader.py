"""Loader for experiment CSV results

Centralizes access to the per-(algorithm, representation) CSV outputs
produced by the executers. Computes reference values (HV reference
point, GD reference front) per (representation, instance) so that
metrics are comparable across algorithms on the same representation
without mixing the very different scales between representations.

Each algorithm's CSV outputs live under:
    experiments_folder/{algorithm}/{representation}/{instance}.csv

For example:
    experiments/nsga2/giant_tour/X-n101-k25.csv
    experiments/nsga2/cluster_first/X-n101-k25.csv
"""

from __future__ import annotations

import os
import numpy as np
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
    """Load and query experiment results across algorithms, representations
    and instances.

    The constructor scans `experiments_folder` and loads every CSV into a
    single combined DataFrame, tagged with algorithm, representation, and
    instance columns. Subsequent queries operate on this in-memory frame
    without re-reading disk.

    Expected folder layout:
        experiments_folder/
            nsga2/
                giant_tour/X-n101-k25.csv
                cluster_first/X-n101-k25.csv
                ...
            spea2/
                giant_tour/...
                cluster_first/...
            paco/
                giant_tour/...
                cluster_first/...
            The optional history_folder (default: experiments_folder/history)
            contains .npz files with full per-generation history:
            history/
                nsga2/giant_tour/X-n101-k25.npz
                ...

    Each CSV has the columns produced by the executers:
        run, solution_idx, f1, f2, n_evaluations, n_generations
    """

    def __init__(
            self,
            experiments_folder: str,
            history_folder: str | None = None,
    ) -> None:
        self.experiments_folder = experiments_folder
        self.history_folder = history_folder or os.path.join(
            experiments_folder, "history"
        )
        self.data = self._load_all()

    def _load_all(self) -> pd.DataFrame:
        """Read every CSV under experiments_folder into one DataFrame.

        Traverses the three-level folder hierarchy (algorithm,
        representation, instance) and tags every row with the
        corresponding labels.
        """
        frames = []
        for algorithm in sorted(os.listdir(self.experiments_folder)):
            algorithm_dir = os.path.join(self.experiments_folder, algorithm)
            if not os.path.isdir(algorithm_dir):
                continue
            for representation in sorted(os.listdir(algorithm_dir)):
                representation_dir = os.path.join(algorithm_dir, representation)
                if not os.path.isdir(representation_dir):
                    continue
                for csv_name in sorted(os.listdir(representation_dir)):
                    if not csv_name.endswith(".csv"):
                        continue
                    instance = os.path.splitext(csv_name)[0]
                    df = pd.read_csv(os.path.join(representation_dir, csv_name))
                    df["algorithm"] = algorithm
                    df["representation"] = representation
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

    def representations(self) -> list[str]:
        """Return the list of representations found in the data."""
        return sorted(self.data["representation"].unique())

    def instances(self) -> list[str]:
        """Return the list of instances found in the data."""
        return sorted(self.data["instance"].unique())

    def get_runs(
        self, algorithm: str, representation: str, instance: str
    ) -> list[list[tuple[float, float]]]:
        """Return the per-run Pareto fronts for one
        (algorithm, representation, instance).

        Each front is a list of (f1, f2) tuples. The outer list has one
        entry per run.
        """
        subset = self.data[
            (self.data["algorithm"] == algorithm)
            & (self.data["representation"] == representation)
            & (self.data["instance"] == instance)
        ]
        runs = []
        for run_idx in sorted(subset["run"].unique()):
            run_df = subset[subset["run"] == run_idx]
            front = list(zip(run_df["f1"].tolist(), run_df["f2"].tolist()))
            runs.append(front)
        return runs

    def compute_reference_point(
        self, representation: str, instance: str
    ) -> tuple[float, float]:
        """Compute a reference point for HV on (representation, instance).

        Uses (max(f1) * 1.1, max(f2) * 1.1) across all algorithms and
        runs on this (representation, instance) so the point strictly
        dominates every observed solution. Reference points are not
        shared between representations because their objective ranges
        differ by orders of magnitude.
        """
        subset = self.data[
            (self.data["representation"] == representation)
            & (self.data["instance"] == instance)
        ]
        return (subset["f1"].max() * 1.1, subset["f2"].max() * 1.1)

    def compute_reference_front(
        self, representation: str, instance: str
    ) -> list[tuple[float, float]]:
        """Compute a pseudo-reference front for GD on (representation,
        instance).

        Combines all solutions from all algorithms and runs on this
        (representation, instance), then keeps only the non-dominated
        ones (both objectives minimized).
        """
        subset = self.data[
            (self.data["representation"] == representation)
            & (self.data["instance"] == instance)
        ]
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
        self,
        algorithm: str,
        representation: str,
        instance: str,
        metric: Metric,
    ) -> list[float]:
        """Compute a metric across runs for one (algorithm, representation,
        instance).

        Returns a list with one scalar per run, in run order. Ready to
        feed into wilcoxon_test or friedman_shaffer_test.

        For HV, the reference point is computed from the full subset on
        this (representation, instance). For GD, the reference front is
        computed similarly.
        """
        runs = self.get_runs(algorithm, representation, instance)

        if metric == "hv":
            ref_point = self.compute_reference_point(representation, instance)
            return [compute_hv(front, ref_point) for front in runs]
        if metric == "sp":
            return [compute_sp(front) for front in runs]
        if metric == "gd":
            ref_front = self.compute_reference_front(representation, instance)
            return [compute_gd(front, ref_front) for front in runs]
        if metric == "pf":
            return [float(compute_pf_size(front)) for front in runs]

        raise ValueError(
            f"Unknown metric {metric!r}. Must be one of: hv, sp, gd, pf."
        )

    def load_history(
            self,
            algorithm: str,
            representation: str,
            instance: str,
    ) -> dict[str, np.ndarray]:
        """Load per-generation history for one (algorithm, representation, instance).

        Reads the .npz file persisted by save_history during full experiments.
        Returns a dict with three numpy arrays of uniform shape:

            chromosomes: int32   (n_runs, n_gens, n_pop, chrom_len)
            f1:          float64 (n_runs, n_gens, n_pop)
            f2:          float64 (n_runs, n_gens, n_pop)

        Used by convergence plots (fitness, chromosome similarity, diversity)
        which need the full evolution trajectory across all 31 runs.
        """
        npz_path = os.path.join(
            self.history_folder, algorithm, representation, f"{instance}.npz"
        )
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"History file not found: {npz_path}. "
                f"Run full experiments to generate per-generation history."
            )
        data = np.load(npz_path)
        return {
            "chromosomes": data["chromosomes"],
            "f1": data["f1"],
            "f2": data["f2"],
        }