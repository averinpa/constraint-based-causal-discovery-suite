# dagsampler

> [!IMPORTANT]
> **This repository is archived.** `dagsampler` has moved to the
> [constraint-based-causal-discovery-suite](https://github.com/averinpa/constraint-based-causal-discovery-suite)
> umbrella, where it lives at
> [`dagsampler/`](https://github.com/averinpa/constraint-based-causal-discovery-suite/tree/main/dagsampler).
>
> The PyPI package name **`dagsampler`** is unchanged — `pip install
> dagsampler` continues to work, with future releases (0.2.0+)
> published from the suite repo. This archive is kept read-only for
> historical reference; the v0.1.0 source remains here at the last
> commit before the move.

[![PyPI version](https://img.shields.io/pypi/v/dagsampler.svg)](https://pypi.org/project/dagsampler/)
[![Python versions](https://img.shields.io/pypi/pyversions/dagsampler.svg)](https://pypi.org/project/dagsampler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-averinpa.github.io-blue.svg)](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/)

Configurable causal DAG simulator for synthetic mixed-type data and CI test benchmarks.

[Documentation](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/) · [Changelog](CHANGELOG.md)

## What it provides

- `CausalDataGenerator` class for configurable simulation
- Support for `custom` and `random` DAGs
- Mixed continuous/binary/categorical nodes (configurable categorical cardinality)
- Structural forms: `linear`, `polynomial`, `interaction`, `sigmoid`, `cos`, `sin`, `stratum_means`
- Optional element-wise `post_transform` (`tanh`, `sin`, `cos`, `exp_neg_abs`, `sqrt_abs`, `relu`, `sign`)
- Cross-type mechanisms:
  - continuous -> categorical (`categorical_model.name = "threshold"`)
  - categorical -> continuous (`functional_form.name = "stratum_means"`, including mixed-parent cases with `metric_weights`)
- Noise models:
  - additive (`gaussian`, `student_t`, `gamma`, `exponential`, `laplace`, `cauchy`, `uniform`)
  - multiplicative (`gaussian`, `student_t`, `gamma`, `exponential`)
  - heteroskedastic (`abs_first_parent`, `abs_parent_plus_const`, `mean_abs_plus_const`)
- Random weight sampling controls (including exclusion band around zero)
- `force_uniform_marginals` for balanced exogenous binary / categorical draws
- Template helpers (`chain_config`, `fork_config`, `collider_config`, `independence_config`)
- Reproducibility via `seed_structure` and `seed_data` (or single `seed`)
- Optional d-separation CI oracle output (`store_ci_oracle=true`)

## Installation

From PyPI:

```bash
pip install dagsampler
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install dagsampler
```

From GitHub (latest `main`):

```bash
uv pip install "dagsampler @ git+https://github.com/averinpa/dagsampler.git"
```

## Quick start (Python API)

```python
from dagsampler import CausalDataGenerator

config = {
    "simulation_params": {"n_samples": 200, "seed": 42},
    "graph_params": {
        "type": "custom",
        "nodes": ["X", "Y", "Z1"],
        "edges": [["X", "Z1"], ["Y", "Z1"]],
    },
}

result = CausalDataGenerator(config).simulate()
data = result["data"]
dag = result["dag"]
params = result["parametrization"]
```

## CLI

The package exposes `dagsampler-generate`.

```bash
dagsampler-generate \
  --config config.json \
  --output dataset.csv \
  --params-out params.json \
  --edges-out edges.json
```

`config.json` must contain the same structure used by `CausalDataGenerator`.

## Learn more

- [Documentation](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/) — full reference for every config option, mechanism, and noise model.
- [Tutorial](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/tutorial.html) — narrative walkthrough.
- [How-to guides](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/howto/) — task-focused recipes.
- [Explanation](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/explanation/) — model formulations and design rationale.
- [API reference](https://averinpa.github.io/constraint-based-causal-discovery-suite/dagsampler/reference/) — every public function and class.
- [`examples/`](examples/) — runnable notebooks.

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
```
