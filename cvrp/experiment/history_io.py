"""I/O helpers for per-generation experiment history.

The full evolution history (all individuals at all generations across
all 31 runs) is too large for CSV. Numpy's compressed .npz format gives
roughly 5-10x compression over JSON for uniform integer/float arrays.

Each .npz file corresponds to one (algorithm, representation, instance)
combination and contains three arrays of uniform shape:

    chromosomes: int32   shape (n_runs, n_gens, n_pop, chrom_len)
    f1:          float64 shape (n_runs, n_gens, n_pop)
    f2:          float64 shape (n_runs, n_gens, n_pop)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def save_history(
    histories: list[list[list[tuple[list[int], tuple[float, float]]]]],
    path: str,
) -> None:
    """Persist a list-of-runs history to a compressed .npz file.

    Args:
        histories: Length n_runs list. Each entry is a per-generation
            history: list of generations, each generation a list of
            (chromosome, (f1, f2)) tuples. All runs must have the same
            number of generations, the same population size at each
            generation, and the same chromosome length.
        path: Output path. Parent directory is created if missing.

    Raises:
        ValueError: If histories is empty or has ragged shapes.
    """
    if not histories:
        raise ValueError("Empty histories list — nothing to save.")

    n_runs = len(histories)
    n_gens = len(histories[0])
    if n_gens == 0:
        raise ValueError("First run has zero generations.")
    n_pop = len(histories[0][0])
    if n_pop == 0:
        raise ValueError("First generation of first run has zero individuals.")
    chrom_len = len(histories[0][0][0][0])

    for i, run_hist in enumerate(histories):
        if len(run_hist) != n_gens:
            raise ValueError(
                f"Run {i} has {len(run_hist)} generations, expected {n_gens}."
            )
        for g, gen in enumerate(run_hist):
            if len(gen) != n_pop:
                raise ValueError(
                    f"Run {i}, generation {g} has {len(gen)} individuals, "
                    f"expected {n_pop}."
                )
            for k, (chrom, _fit) in enumerate(gen):
                if len(chrom) != chrom_len:
                    raise ValueError(
                        f"Run {i}, generation {g}, individual {k} has "
                        f"chromosome length {len(chrom)}, expected {chrom_len}."
                    )

    chromosomes = np.zeros((n_runs, n_gens, n_pop, chrom_len), dtype=np.int32)
    f1_arr = np.zeros((n_runs, n_gens, n_pop), dtype=np.float64)
    f2_arr = np.zeros((n_runs, n_gens, n_pop), dtype=np.float64)

    for i, run_hist in enumerate(histories):
        for g, gen in enumerate(run_hist):
            for k, (chrom, fit) in enumerate(gen):
                chromosomes[i, g, k] = chrom
                f1_arr[i, g, k] = fit[0]
                f2_arr[i, g, k] = fit[1]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        chromosomes=chromosomes,
        f1=f1_arr,
        f2=f2_arr,
    )


def load_history(path: str) -> Any:
    """Load a history archive saved by save_history.

    Returns a numpy NpzFile object with three keys:
        chromosomes: int32   (n_runs, n_gens, n_pop, chrom_len)
        f1:          float64 (n_runs, n_gens, n_pop)
        f2:          float64 (n_runs, n_gens, n_pop)

    Access them as data["chromosomes"], data["f1"], data["f2"].
    """
    return np.load(path)