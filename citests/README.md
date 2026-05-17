# `citests`: A Conditional Independence Test Toolkit

[![Documentation](https://img.shields.io/badge/docs-averinpa.github.io-blue.svg)](https://averinpa.github.io/constraint-based-causal-discovery-suite/citests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Full [documentation](https://averinpa.github.io/constraint-based-causal-discovery-suite/citests/) is hosted on GitHub Pages.

`citests` is a Python library that provides a comprehensive toolkit for conditional independence (CI) testing. It is designed to be seamlessly integrated with the [`causal-learn`](https://github.com/cmu-phil/causal-learn) package and offers a collection of partial correlation, contingency table, regression, nearest neighbor, kernel, and machine-learning-based CI tests, plus adapter strategies.

The library is structured to be a powerful benchmark for causal discovery and a practical toolkit for researchers and practitioners.

## Features

- **`causal-learn` compatible**: All tests are designed as drop-in replacements for the standard tests in the `causal-learn` ecosystem, allowing you to easily use them with algorithms like PC.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/averinpa/citests.git
```

For local development with extras:

```bash
uv sync --all-extras
```

Optional dependency groups in `pyproject.toml`:

- `pycomets` — required for `gcm`, `wgcm`, `pcm` (installs `xgboost` only; `pycomets` itself is GitHub-only and must be installed separately: `pip install git+https://github.com/shimenghuang/pycomets.git`)
- `tigramite` — required for `cmiknn`, `cmiknn_mixed`, `regci`
- `r` — required for `rcit`, `rcot`, `ci_mm`, `hartemink_chisq` (installs `rpy2`); the corresponding R packages must also be installed:
  - `RCIT` from GitHub `ericstrobl/RCIT` (for `rcit`, `rcot`)
  - `MXM` from CRAN (for `ci_mm`)
  - `bnlearn` from CRAN (for `hartemink_chisq`)

The `mcmiknn` test uses a vendored copy of the upstream [hpi-epic/mCMIkNN](https://github.com/hpi-epic/mCMIkNN) source under `citests/_vendor/indeptests/`; no additional installation is required.

## Quickstart Example

Here is a simple example of how to use a `citests` test within the `causal-learn` PC algorithm.

```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import citests.tests 

# 1. Generate some data
np.random.seed(42)
data = np.random.randn(200, 3)
data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 1] + 0.1 * np.random.randn(200)

# 2. Run the PC algorithm using a citests test
# Example test ids: "fisherz_citests", "spearman", "gsq", "chisq", "kci", "gcm"
cg = pc(data, alpha=0.05, indep_test='spearman')

# 3. View the learned graph
print("Learned Graph Edges:")
print(cg.G.get_edges())
```

## Available Tests

| Test Name | Family | Wrapped From |
|---|---|---|
| `fisherz_citests` | Partial Correlation | `causal-learn` (`CIT(..., method_name="fisherz")`) |
| `spearman` | Partial Correlation | `causal-learn` Fisher-Z on ranked data |
| `chisq` | Contingency Table | `causal-learn` (`Chisq_or_Gsq(..., method_name="chisq")`) |
| `gsq` | Contingency Table | `causal-learn` (`Chisq_or_Gsq(..., method_name="gsq")`) |
| `regci` | Regression | `tigramite.independence_tests.regressionCI.RegressionCI` (optional) |
| `ci_mm` | Regression | R `MXM::ci.mm` via `rpy2` (optional) |
| `cmiknn` | Nearest Neighbor | `tigramite.independence_tests.cmiknn.CMIknn` (optional) |
| `cmiknn_mixed` | Nearest Neighbor | `tigramite` CMIknnMixed wrapper (optional) |
| `mcmiknn` | Nearest Neighbor | Vendored `indeptests.mCMIkNN` from [hpi-epic/mCMIkNN](https://github.com/hpi-epic/mCMIkNN) (no install required) |
| `kci` | Kernel | `causal-learn` Python KCI |
| `rcit` | Kernel | R `RCIT::RCIT` via `rpy2` (optional) |
| `rcot` | Kernel | R `RCIT::RCoT` via `rpy2` (optional) |
| `gcm` | Machine-Learning-Based | `pycomets` GCM with random forest regression (optional) |
| `wgcm` | Machine-Learning-Based | `pycomets` WGCM with random forest regression (optional) |
| `pcm` | Machine-Learning-Based | `pycomets` PCM with random forest regression (optional) |
| `disc_chisq` | Adapter Strategies | Native `citests` equal-frequency discretization + `causal-learn` Chi-Square |
| `disc_gsq` | Adapter Strategies | Native `citests` equal-frequency discretization + `causal-learn` G-Square |
| `dummy_fisherz` | Adapter Strategies | Native `citests` one-hot encoding + `causal-learn` Fisher-Z aggregation |
| `hartemink_chisq` | Adapter Strategies | R `bnlearn` Hartemink discretization + `causal-learn` Chi-Square (optional) |

### Module Layout (Survey Taxonomy)

- `citests/tests/partial_correlation_tests.py`
- `citests/tests/contingency_table_tests.py`
- `citests/tests/regression_tests.py`
- `citests/tests/nearest_neighbor_tests.py`
- `citests/tests/kernel_tests.py`
- `citests/tests/ml_based_tests.py`
- `citests/tests/adapter_tests.py`

For detailed documentation on each test and its parameters, please see our full documentation page [HERE](https://averinpa.github.io/constraint-based-causal-discovery-suite/citests/).

## Acknowledgements

`citests` is a toolkit-style assembly. Several tests are adapters over
upstream implementations:

- **KCI** wraps `causallearn.utils.cit.KCI` from
  [causal-learn](https://github.com/py-why/causal-learn) (MIT) under
  the optional `[causallearn]` extra.
- **disc_chisq / disc_gsq / dummy_fisherz** route discretised /
  one-hot data through causal-learn's Chi-Square, G-Square, and
  Fisher-Z back-ends respectively.
- **mCMIkNN** is vendored verbatim from
  [hpi-epic/mCMIkNN](https://github.com/hpi-epic/mCMIkNN) (MIT). See
  [`citests/_vendor/NOTICE.md`](citests/_vendor/NOTICE.md) for the full
  attribution including authors, paper citation, and vendored
  revision SHA.
- **hartemink_chisq** uses `bnlearn` (R) for Hartemink discretisation
  via `rpy2`, paired with causal-learn's Chi-Square test.
- **RCIT / RCoT** wrap the R [RCIT
  package](https://CRAN.R-project.org/package=RCIT) via `rpy2` under
  the optional `[r]` extra.
- **CiMM** wraps the `ci.mm` test from the R [MXM
  package](https://CRAN.R-project.org/package=MXM) (GPL-2+) via
  `rpy2` under the optional `[r]` extra; MXM is invoked rather than
  vendored, so the GPL boundary remains in the user's R installation.
- **CMIknn / RegressionCI** wrap
  [tigramite](https://github.com/jakobrunge/tigramite) (GPL-3) under
  the optional `[tigramite]` extra; tigramite is invoked at the
  user's installation rather than vendored, so the GPL-3 boundary
  remains in the user's environment.

Native `citests` tests (FisherZ, Spearman, χ²/G², regression-based) are
independent implementations. Cross-package interop with `cbcd` is via
the structural `cbcd.CITest` Protocol — neither package imports the
other.

## License

[MIT License](LICENSE).
