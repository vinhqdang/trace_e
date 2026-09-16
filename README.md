# trace_e

Evaluation pipeline for two related problems on graphs:

* **Problem A - source detection ("patient zero")**: given a snapshot of an
  SIR epidemic on a known graph, rank the nodes by how likely they are to be
  the origin.
* **Problem B - misinformation blocking / influence limitation**: given a bad
  cascade spreading from a known seed set, choose a budget-limited set of
  nodes to block or counter-seed so that the cascade reaches as few nodes as
  possible.

The repository is organised so that established baselines run under one
protocol first, and new methods are plugged into the same registry and
compared on the same logged runs. See `docs/SCOPING.md` for the literature
scoping (baselines, datasets, metrics, open gaps) that drives the design.

## Layout

```
data/networks/          edge lists (0-indexed) of the empirical contact networks
trace_e/graphs.py       loaders (empirical + ER/BA/WS/tree synthetic)
trace_e/simulate.py     continuous-time SIR simulator (benchmark semantics)
trace_e/metrics.py      top-k, error distance, reciprocal rank, credible set, Brier, resistance score
trace_e/logging_utils.py run directories, JSONL per-instance logs, summary CSV
trace_e/source/         Problem A detectors, registry, `proposed.py` slot
trace_e/blocking/       Problem B cascade models, blockers, registry
trace_e/eval/           command-line runners
results/                summary_*.csv + results/runs/<run_id>/ (config.json, run.log, instances_*.jsonl.gz)
scripts/                batch scripts
tests/                  pytest smoke tests
```

## Setup

```
pip install -r requirements.txt
python -m pytest -q
```

CPU is enough for everything in the default protocol; the GNN baselines are
small (16 hidden units, 5 layers) and train in seconds to minutes.

## Problem A: run the baselines

```
python -m trace_e.eval.run_source --network karate
python -m trace_e.eval.run_source --network powergrid --methods jordan,rumor,netsleuth,dmp,mcs,gcn --test-sims 10
```

Protocol (matches the benchmark of Sterchi, Brack & Hilfiker 2026):
continuous-time SIR, recovery rate `nu = 1`, infection rate `beta = 1.3` per
edge, snapshot at `T = 0.85`. Training simulations perturb `beta` per outbreak
by a log-normal with `sigma = 0.5`; test simulations use `sigma = 0` and a
fixed seed. Training: 10 outbreaks per source node; test: 100 per node. Node
states are `0 = S`, `1 = I`, `2 = R`. Every method outputs a score vector over
nodes; susceptible nodes are excluded.

Available detectors (`--methods`):

| name | reference | needs simulations |
|---|---|---|
| `random` | uniform over infected nodes | no |
| `degree` | degree in the infected subgraph | no |
| `jordan` | Jordan center (Zhu & Ying 2016) | no |
| `distance` | distance centrality (Comin & da Fontoura Costa 2011) | no |
| `closeness`, `betweenness` | centrality of the infected subgraph | no |
| `rumor` | rumor centrality on BFS tree (Shah & Zaman 2011) | no |
| `netsleuth` | NetSleuth Laplacian eigenvector (Prakash et al. 2012) | no |
| `dmp` | dynamic message passing likelihood (Lokhov et al. 2014) | no |
| `sme` | soft margin estimator (Antulov-Fantulin et al. 2015) | yes |
| `mcs` | Monte-Carlo mean-field likelihood (Sterchi et al. 2026) | yes |
| `mlp` | per-node MLP, no message passing | yes |
| `gcn` | GCN source detector (Dong et al. 2019 style) | yes |
| `gcn_skip` | GCN with batch-norm and residuals (Shah et al. 2020 style) | yes |
| `igcn` | state-pair conditioned aggregation (Guo et al. 2021) | yes |
| `proposed` | slot for the method under development | yes |

Metrics per instance: top-1/3/5 accuracy (ties broken at random), error
distance in hops between MAP estimate and true source, reciprocal rank,
and for probabilistic methods the 90% credible-set size, Brier score and
resistance score. Fit time and inference time per instance are logged.

Outputs:

* `results/summary_source.csv` - one row per (run, network, method).
* `results/runs/<run_id>/run.log` - progress log (also printed to stdout).
* `results/runs/<run_id>/instances_<method>.jsonl.gz` - per-instance metrics.
* `results/runs/<run_id>/config.json` - full configuration and git commit.

`scripts/run_source_all.sh` runs the whole network suite and commits the
results after each network.

## Problem B: run the blocking baselines

```
python -m trace_e.eval.run_blocking --network dolphin --mode block --budgets 1,2,5,10
python -m trace_e.eval.run_blocking --network cahepth --mode counter --n-seeds 5 --budgets 5,10,20
```

Protocol: the graph becomes an independent-cascade (IC) model with edge
activation probabilities from `--prob-model` (`wc` weighted cascade
`1/deg(v)`, `const` p, `tri` trivalency). For each of `--n-instances` random
bad seed sets and each budget, every method picks an intervention set. The
expected number of bad adopters with and without intervention is estimated
with `--n-mc` Monte-Carlo cascades using common random numbers across methods.

* `--mode block`: chosen nodes are removed (influence minimisation by vertex
  blocking; Wang et al. 2013, Xie et al. 2023).
* `--mode counter`: chosen nodes seed a competing good campaign that starts
  `--good-delay` rounds after the bad one; first arrival wins, ties go to the
  good campaign (influence limitation; Budak et al. 2011).

Available blockers (`--methods`): `random`, `degree`, `pagerank`, `proximity`
(closest to the seeds), `reach` (Monte-Carlo reach probability times degree),
`greedy` (CELF lazy greedy on a shared pool of live-edge samples; the
(1-1/e) reference in counter mode), `proposed` (slot).

Metrics: saved fraction `1 - after/before`, bad fraction of the graph, bad
count, selection time; one summary row per (network, mode, method, budget) in
`results/summary_blocking.csv`, per-instance JSONL with the chosen sets.

`scripts/run_blocking_all.sh` runs both modes on the network suite.

## Reports

`python scripts/aggregate.py` rebuilds `results/REPORT.md` from the summary
CSVs (latest run per configuration) and, for Problem A, recomputes the
metrics restricted to outbreaks with at least two infected nodes from the
per-instance logs (singleton outbreaks are trivially solved and inflate every
method equally).

## Adding a method

1. Create a class in `trace_e/source/` (or `trace_e/blocking/`) that subclasses
   `SourceDetector` (or `Blocker`), set `name`, implement `fit`/`score`
   (or `select`), and decorate it with `@register`.
2. Import the module in the package `__init__.py` (the `proposed.py` files are
   already wired).
3. Run the runner with `--methods <name>`; the row lands in the same summary
   CSV as the baselines.

## Larger graphs

`python scripts/fetch_snap.py ca-GrQc ca-HepTh soc-Epinions1` downloads the
SNAP graphs commonly used in influence-maximisation papers and writes them to
`data/networks/` as `cagrqc` (5,241 nodes), `cahepth` (9,875 nodes; the
NetHEPT family) and `socepinions1` (75,879 nodes; not committed because of
its size). Synthetic graphs are available through specs such as `ba:2000:m=2`,
`er:2000`, `ws:1000:k=6:p=0.05`, `tree:500`.

## Data sources

Edge lists in `data/networks/` are the static networks released with the
Sterchi et al. benchmark (Zachary karate club; Iceland sexual-contact network;
dolphin social network; fraternity; SocioPatterns workplace, high school and
conference face-to-face projections; Western US power grid; country-level
air-traffic network). Original sources are listed in `docs/SCOPING.md`.
