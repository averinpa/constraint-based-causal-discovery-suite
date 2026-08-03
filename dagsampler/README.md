# dagsampler

> [!NOTE]
> `dagsampler` is developed in the
> [constraint-based-causal-discovery-suite](https://github.com/averinpa/constraint-based-causal-discovery-suite)
> monorepo. `pip install dagsampler` installs the released package from PyPI.

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
  - continuous -> categorical (`categorical_model.name = "threshold"`), with an opt-in standardized (design-A) threshold mode (`threshold_standardized`) that discretizes a unit-variance linear-Gaussian latent at equal-probability cutpoints
  - categorical -> continuous (`functional_form.name = "stratum_means"`, including mixed-parent cases with `metric_weights`)
- Opt-in spread-controlled softmax/logistic weights (`softmax_weight_mode = "spread"`) for a detectable, balance-preserving logit contrast (default `"gaussian"` preserves legacy behaviour)
- Noise models:
  - additive (`gaussian`, `student_t`, `gamma`, `exponential`, `laplace`, `cauchy`, `uniform`)
  - multiplicative (`gaussian`, `student_t`, `gamma`, `exponential`)
  - heteroskedastic (`abs_first_parent`, `abs_parent_plus_const`, `mean_abs_plus_const`); base distribution selectable via `dist` (`gaussian` default, `student_t`, `laplace`, `uniform`, `gamma`, `exponential`)
  - shape / tail-shape (`skew_first_parent`, `skew_tanh_first_parent`, `skew_mean_parents`) — a parent drives the noise skewness with mean and variance held fixed (a higher-moment edge)
- Random weight sampling controls (including exclusion band around zero)
- `force_uniform_marginals` for balanced exogenous binary / categorical draws
- Template helpers (`chain_config`, `fork_config`, `collider_config`, `independence_config`)
- Reproducibility via `seed_structure` and `seed_data` (or single `seed`)
- Optional d-separation CI oracle output (`store_ci_oracle=true`)
- Time-series (stationary SVAR) support with latent confounders: `TimeSeriesSpec` / `random_ts_spec`, a lagged d-separation oracle (`LaggedDSeparationOracle`), and mixed-type `simulate_svar` (see [How-to: time series](docs/howto/timeseries.md))

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
