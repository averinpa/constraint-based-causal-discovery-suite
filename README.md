# dagsampler

[![PyPI version](https://img.shields.io/pypi/v/dagsampler.svg)](https://pypi.org/project/dagsampler/)
[![Python versions](https://img.shields.io/pypi/pyversions/dagsampler.svg)](https://pypi.org/project/dagsampler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-averinpa.github.io-blue.svg)](https://averinpa.github.io/dagsampler/)

Configurable causal DAG simulator for synthetic mixed-type data and CI test benchmarks.

[Documentation](https://averinpa.github.io/dagsampler/) · [Changelog](CHANGELOG.md)

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

## Random weights away from zero

To guarantee a minimum signal strength on every edge — so randomly sampled
weights don't end up effectively muting a parent — configure:

```json
{
  "simulation_params": {
    "random_weight_low": -1.5,
    "random_weight_high": 1.5,
    "random_weight_min_abs": 0.1
  }
}
```

This samples random structural weights from:
- `[-1.5, -0.1] U [0.1, 1.5]`

By default, categorical parents are not allowed with metric functional forms
(`linear`, `polynomial`, `interaction`). Set:
- `"categorical_parent_metric_form_policy": "stratum_means"`
to auto-redirect those cases to `stratum_means`.

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

For heteroskedastic noise, use `noise_model.func` from:
- `abs_first_parent`
- `abs_parent_plus_const`
- `mean_abs_plus_const`

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
```
