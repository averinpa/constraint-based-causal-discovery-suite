# Tutorial: a first simulation

This tutorial walks through one full invocation of the simulator,
from a config dictionary to a generated dataset and an attached
d-separation CI oracle. The reader is assumed to be familiar with
directed acyclic graphs, conditional independence, and the standard
parent-child notation; for the underlying mathematics see
[Explanation: SCM formulation](explanation/scm_formulation.md).

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install "dagsampler @ git+https://github.com/averinpa/dagsampler.git"
```

The hard dependencies are NumPy, pandas, NetworkX, and SciPy. The
optional `[examples]` extra adds Jupyter and Matplotlib for
notebook-driven experimentation; `[docs]` adds the documentation
toolchain.

## A minimal custom DAG

```python
from dagsampler import CausalDataGenerator

config = {
    "simulation_params": {"n_samples": 200, "seed": 42},
    "graph_params": {
        "type": "custom",
        "nodes": ["X", "Y", "Z"],
        "edges": [["X", "Z"], ["Y", "Z"]],
    },
}
result = CausalDataGenerator(config).simulate()
```

The DAG `X → Z, Y → Z` is a v-structure with `Z` as the collider.
Without further configuration, `dagsampler` chooses node types,
structural mechanisms, and noise families for each node from
sensible defaults. The returned dictionary contains:

- `result["data"]` — a `pandas.DataFrame` of shape `(n_samples,
  n_vars)` with one column per node.
- `result["dag"]` — the underlying `networkx.DiGraph`.
- `result["parametrization"]` — the full resolved configuration,
  including every default that the simulator filled in. Saving
  this out and loading it back regenerates the same data exactly.

For the detailed parameter dictionary that controls every choice
the simulator makes, see
[How-to: configuration cookbook](howto/config_cookbook.md).

## Reproducibility

The simulator separates two random streams:

- `rng_structure` controls the data-generating process — random DAG
  topology, sampled structural weights, intercepts, thresholds,
  stratum means.
- `rng_data` controls the per-sample draws — exogenous variable
  values, noise draws, Bernoulli and categorical sampling.

Use the single-seed convenience form (`seed`) for one-off
examples; for benchmarks, pin the streams independently:

```python
config["simulation_params"] = {
    "n_samples": 200,
    "seed_structure": 11,  # data-generating process
    "seed_data": 12,       # finite-sample draws
}
```

Holding `seed_structure` fixed while varying `seed_data` measures
how a downstream method (e.g. a CI test) behaves on different
finite samples from the *same* DGP.

## The CI oracle (v0.2.0)

`dagsampler` can attach a conditional-independence oracle to the
generator that answers `X ⫫ Y | S` queries by d-separation on the
generated DAG. The oracle is a Python object satisfying the
structural `cbcd.CITest` Protocol:

```python
gen = CausalDataGenerator(config)
result = gen.simulate()
oracle = gen.as_ci_oracle()

oracle.n_vars              # 3
oracle(0, 1, [2])          # 1.0  (d-separated → "p-value" 1.0)
oracle(0, 1, [])           # 0.0  (d-connected   → "p-value" 0.0)
```

Because the oracle satisfies `cbcd.CITest` directly, it can be
passed without a wrapper to any constraint-based algorithm in
the [`cbcd`](https://github.com/averinpa/cbcd) sister package:

```python
from cbcd import pc
true_cpdag = pc(result["data"], ci_test=oracle, alpha=0.05)
```

PC under a perfect d-separation oracle is sound and complete, so
`true_cpdag` is the *true* CPDAG of the generating DAG — useful as
a gold-standard benchmark against which a finite-sample recovery
can be scored. See the suite tutorial at
`suite/docs/tutorial.md` for the full data-generation through
metric-scoring workflow.

A precomputed d-separation table covering all conditioning sets up
to a fixed cardinality is also available (`store_ci_oracle = true`,
`ci_oracle_max_cond_set = k`); see
[How-to: working with the CI oracle](howto/ci_oracle.md).

## Templates for common DAG shapes

For the structural patterns that recur in CI-test benchmarks —
chains, forks, colliders, independent variable sets — `dagsampler`
ships small helper functions that build the configuration
dictionary for you:

```python
from dagsampler import CausalDataGenerator, chain_config

cfg = chain_config(
    var_specs=[
        {"name": "X", "type": "continuous"},
        {"name": "M", "type": "continuous"},
        {"name": "Y", "type": "continuous"},
    ],
    mechanism="linear",
    n_samples=200,
    seed=0,
)
result = CausalDataGenerator(cfg).simulate()
```

The full template reference is at
[How-to: template configurations](howto/templates.md).

## CLI

The package installs a `dagsampler-generate` console script for
running a configuration without writing Python:

```bash
dagsampler-generate \
  --config config.json \
  --output dataset.csv \
  --params-out params.json \
  --edges-out edges.json
```

`--params-out` writes the fully-resolved parametrization;
`--edges-out` writes the realised DAG edge list.

## What is next

- **All configuration options** —
  [How-to: configuration cookbook](howto/config_cookbook.md).
- **The mathematical model** —
  [Explanation: SCM formulation](explanation/scm_formulation.md).
- **Cross-package use with `cbcd`** —
  [How-to: working with the CI oracle](howto/ci_oracle.md).
