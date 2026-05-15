# PG7 — Multi-Objective CVRP

Multi-objective metaheuristic comparison for the Capacitated Vehicle Routing Problem (CVRP). Bi-objective formulation:
- **f1**: total distance + capacity-overflow penalty
- **f2**: load imbalance across vehicles

Three algorithms (**NSGA-II**, **SPEA2**, **Pareto ACO**) on two representations (**giant tour**, **cluster first**), evaluated on **12 PyVRP X-n\* benchmark instances** with **31 independent runs** each.

## Project structure

```
bao_cvrp/
├── cvrp/
│   ├── ga/                          # GA algorithms
│   │   ├── nsga2.py                 # inspyred NSGA-II wrapper
│   │   └── spea2.py                 # custom SPEA2 implementation
│   ├── swarm/
│   │   └── paco.py                  # standalone Pareto ACO
│   ├── problem/
│   │   ├── cvrp_benchmark.py        # inspyred Benchmark subclass
│   │   ├── parser.py                # .vrp file parser
│   │   ├── decoders.py              # GiantTourDecoder, ClusterFirstDecoder
│   │   ├── fitness.py               # evaluate(routes, instance)
│   │   ├── operators.py             # uniform crossover + reset mutation
│   │   └── generators.py            # per-representation generators
│   ├── metrics/
│   │   ├── pareto_metrics.py        # HV, SP, GD, PF
│   │   └── statistical_tests.py     # Wilcoxon, Friedman+Shaffer
│   └── experiment/
│       ├── ga/                      # NSGA2Executer, SPEA2Executer
│       ├── swarm/                   # PACOExecuter
│       ├── experiment_loader.py     # CSV + .npz loader
│       ├── history_io.py            # .npz save/load
│       └── visualization.py         # plotting helpers
├── data/                            # 12 X-n*.vrp files
├── experiments/
│   ├── {algo}/{rep}/*.csv           # final archives per run
│   ├── history/{algo}/{rep}/*.npz   # gitignored — per-gen history
│   ├── best_hyperparams.json
│   └── grid_search_results.json
├── notebooks/
│   ├── run_nsga2.ipynb              # single-instance demo
│   ├── run_spea2.ipynb
│   ├── run_paco.ipynb
│   └── compare_results.ipynb        # final cross-algorithm analysis
├── scripts/
│   └── run_experiments.py           # grid + full pipeline
├── stac/                            # local patched stac (not pip)
├── requirements.txt
└── README.md
```

## Installation

Python 3.12+ required.

```bash
python -m venv .venv
source .venv/bin/activate          # on Linux/Mac
.venv\Scripts\Activate.ps1         # on Windows PowerShell

pip install -r requirements.txt
```

The `stac` library is included as a local patched folder under `stac/` (not installed via pip due to a numpy 2.x compatibility patch).

## Running experiments

**Phase 1 — grid search** on a single instance (X-n101-k25) with `inspyred`-driven hyperparameter sweep:

```bash
python -m scripts.run_experiments --phase grid
```

Produces `experiments/best_hyperparams.json` and `experiments/grid_search_results.json`. Cost: ~1.5h.

**Phase 2 — full experiments** across all 12 instances with the winning hyperparameters from Phase 1:

```bash
python -m scripts.run_experiments --phase full --n-jobs 8
```

Adjust `--n-jobs` to your machine (8 is safe for 16-core CPUs; reduce if memory-limited). Cost: ~3-4h on 8 cores.

**Idempotency:** each `(algorithm, representation, instance)` task is skipped if both its CSV and .npz already exist. Re-running resumes from where it left off.

**Smoke test** (1 instance, tiny budget, ~1 min):

```bash
python -m scripts.run_experiments --phase full --smoke-test
```

## Running notebooks

```bash
jupyter notebook
```

- `notebooks/run_{nsga2,spea2,paco}.ipynb` — per-algorithm single-instance demos
- `notebooks/compare_results.ipynb` — final cross-algorithm analysis (depends on full experiments)

`compare_results.ipynb` expects:
- CSV files in `experiments/{algo}/{rep}/*.csv`
- .npz files in `experiments/history/{algo}/{rep}/*.npz`
- `experiments/best_hyperparams.json`, `experiments/grid_search_results.json`

## Regenerating gitignored history files

The `experiments/history/` directory (~1.7 GB across 36 .npz files; 4 files exceed GitHub's 100 MB limit) is gitignored. It contains per-generation chromosomes and fitnesses needed by the convergence and diversity plots in `compare_results.ipynb`.

Regenerate:
```bash
python -m scripts.run_experiments --phase full --n-jobs 8
```
(~3-4 hours)

## Output formats

### CSV (`experiments/{algo}/{rep}/{instance}.csv`)

One row per Pareto solution per run:

| column | type | description |
|---|---|---|
| `run` | int | 0..30 |
| `solution_idx` | int | index within final archive |
| `f1` | float | distance + penalty |
| `f2` | float | load imbalance |
| `chromosome` | str | comma-separated genes (`"1,5,2,4,..."`) |
| `n_evaluations` | int | total evals in the run |
| `n_generations` | int | generations completed |

### .npz (`experiments/history/{algo}/{rep}/{instance}.npz`)

Compressed numpy archive with three arrays of uniform shape:

```python
data["chromosomes"]  # int32   shape (n_runs, n_gens, n_pop, chrom_len)
data["f1"]           # float64 shape (n_runs, n_gens, n_pop)
data["f2"]           # float64 shape (n_runs, n_gens, n_pop)
```

Use `ExperimentLoader.load_history(algo, rep, instance)` to access.

### `best_hyperparams.json`

```json
{
  "nsga2": {
    "giant_tour":    {"hyperparams": {...}, "hv": ...},
    "cluster_first": {"hyperparams": {...}, "hv": ...}
  },
  "spea2": {...},
  "paco":  {...}
}
```

### `grid_search_results.json`

Same as above plus, per (algo, rep), an `all_combos` list with the full HV landscape:

```json
{
  "ref_point": [f1_max, f2_max],
  "all_combos": [
    {"hyperparams": {...}, "hv_mean": ..., "hv_per_seed": [...]},
    ...
  ]
}
```

## Algorithms

### NSGA-II (Deb et al., 2002)
inspyred's built-in `ec.emo.NSGA2` with custom variators per representation. Non-dominated sorting + crowding distance for replacement, tournament selection driven by Pareto dominance + crowding.

### SPEA2 (Zitzler et al., 2001)
Custom implementation subclassing `inspyred.ec.EvolutionaryComputation`. Strength-based fitness `F(i) = R(i) + D(i)`, tournament selection, truncation replacement, `best_archiver` for the external archive.

### Pareto ACO (Doerner et al., 2004)
Standalone class (not inspyred). Two pheromone matrices `τ1`, `τ2` (one per objective), construction probability `∝ τ1^α1 · τ2^α2 · η^β`. External Pareto archive trimmed by even spread along f1.

### Representations

- **`giant_tour`** — permutation of customer ids. Decoder splits the sequence at capacity, so feasibility is structural.
- **`cluster_first`** — integer vector assigning each customer to a vehicle. Capacity is not enforced; infeasibility is penalized in f1.

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* 7, 1-30.
- Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEC* 6(2).
- Zitzler, E., Laumanns, M., Thiele, L. (2001). SPEA2: Improving the Strength Pareto Evolutionary Algorithm. TIK Tech. Report 103.
- Doerner, K., Gutjahr, W.J., Hartl, R.F., Strauss, C., Stummer, C. (2004). Pareto ant colony optimization. *Annals of OR* 131.