# `citk`: A Conditional Independence Test Toolkit

`citk` is a Python library that provides a comprehensive and modern toolkit for conditional independence (CI) testing. It is designed to be seamlessly integrated with the [`causal-learn`](https://github.com/cmu-phil/causal-learn) package and offers a collection of classical, statistical, and advanced machine learning-based CI tests.

The library is structured to be a powerful benchmark for causal discovery and a practical toolkit for researchers and practitioners.

## Features

- **Survey-aligned coverage**: 19 conditional independence tests organised by the six families of the Paper 0 survey (partial correlation, contingency table, regression, nearest neighbor, kernel, machine-learning-based) plus four robustness-wrapper adapters.
- **`causal-learn` compatible**: All tests are designed as drop-in replacements for the standard tests in the `causal-learn` ecosystem, allowing you to easily use them with algorithms like PC.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/averinpa/citk.git
```

For local development with extras:

```bash
uv sync --all-extras
```

Optional dependency groups in `pyproject.toml`:

- `pycomets` — required for `gcm`, `wgcm`, `pcm` (installs `pycomets` and `xgboost`)
- `tigramite` — required for `cmiknn`, `cmiknn_mixed`, `regci`
- `r` — required for `kci`-via-RCIT, `rcit`, `rcot`, `ci_mm`, `hartemink_chisq` (installs `rpy2`); the corresponding R packages must also be installed:
  - `RCIT` from GitHub `ericstrobl/RCIT` (for `rcit`, `rcot`)
  - `MXM` from CRAN (for `ci_mm`)
  - `bnlearn` from CRAN (for `hartemink_chisq`)

The `mcmiknn` wrapper additionally requires a local checkout of the upstream `mCMIkNN` repository on disk (see :doc:`docs/source/guides/installation`).

## Quickstart Example

Here is a simple example of how to use a `citk` test within the `causal-learn` PC algorithm.

```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import citk.tests 

# 1. Generate some data
np.random.seed(42)
data = np.random.randn(200, 3)
data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 1] + 0.1 * np.random.randn(200)

# 2. Run the PC algorithm using a citk test
# Example test ids: "fisherz_citk", "spearman", "gsq", "chisq", "kci", "gcm"
cg = pc(data, alpha=0.05, indep_test='spearman')

# 3. View the learned graph
print("Learned Graph Edges:")
print(cg.G.get_edges())
```

## Available Tests

| Test Name | Family | Wrapped From |
|---|---|---|
| `fisherz_citk` | Partial Correlation | `causal-learn` (`CIT(..., method_name="fisherz")`) |
| `spearman` | Partial Correlation | `causal-learn` Fisher-Z on ranked data |
| `chisq` | Contingency Table | `causal-learn` (`Chisq_or_Gsq(..., method_name="chisq")`) |
| `gsq` | Contingency Table | `causal-learn` (`Chisq_or_Gsq(..., method_name="gsq")`) |
| `regci` | Regression | `tigramite.independence_tests.regressionCI.RegressionCI` (optional) |
| `ci_mm` | Regression | R `MXM::ci.mm` via `rpy2` (optional) |
| `cmiknn` | Nearest Neighbor | `tigramite.independence_tests.cmiknn.CMIknn` (optional) |
| `cmiknn_mixed` | Nearest Neighbor | `tigramite` CMIknnMixed wrapper (optional) |
| `mcmiknn` | Nearest Neighbor | Local wrapper from `/Users/pavelaverin/Projects/vendor/mCMIkNN/src` (optional) |
| `kci` | Kernel | `causal-learn` Python KCI |
| `rcit` | Kernel | R `RCIT::RCIT` via `rpy2` (optional) |
| `rcot` | Kernel | R `RCIT::RCoT` via `rpy2` (optional) |
| `gcm` | Machine-Learning-Based | `pycomets` GCM with random forest regression (optional) |
| `wgcm` | Machine-Learning-Based | `pycomets` WGCM with random forest regression (optional) |
| `pcm` | Machine-Learning-Based | `pycomets` PCM with random forest regression (optional) |
| `disc_chisq` | Robustness Wrappers | Native `citk` equal-frequency discretization + `causal-learn` Chi-Square |
| `disc_gsq` | Robustness Wrappers | Native `citk` equal-frequency discretization + `causal-learn` G-Square |
| `dummy_fisherz` | Robustness Wrappers | Native `citk` one-hot encoding + `causal-learn` Fisher-Z aggregation |
| `hartemink_chisq` | Robustness Wrappers | R `bnlearn` Hartemink discretization + `causal-learn` Chi-Square (optional) |

### Module Layout (Survey Taxonomy)

- `citk/tests/partial_correlation_tests.py`
- `citk/tests/contingency_table_tests.py`
- `citk/tests/regression_tests.py`
- `citk/tests/nearest_neighbor_tests.py`
- `citk/tests/kernel_tests.py`
- `citk/tests/ml_based_tests.py`
- `citk/tests/adapter_tests.py`

For detailed documentation on each test and its parameters, please see our full documentation page [HERE](https://averinpa.github.io/citk/).
