# citk TODO

## Bugs

- [x] `_glm_conditional_independence_test` mutates `df.columns` in place — fixed with `.copy()`.
- [x] `pingouin` imported but never used — removed.
- [x] `Ridge` imported but never used — removed.
- [x] `pd.api.types.is_categorical_dtype` deprecated — replaced with `dtype.name == 'category'`.
- [x] `pyproject.toml` missing dependencies — added scikit-learn and statsmodels; rpy2 as optional (`lightgbm` moved to optional extra).
- [x] `pyproject.toml` author placeholder — updated to Pavel Averin.

## Architecture improvements

- [x] **Extract cache boilerplate into base class.** `CITKTest.__call__` handles caching, subclasses implement `_compute`.
- [x] **Declare supported data types per test.** `supported_dtypes` class attribute on all tests.

## Remove lightgbm as a core dependency

- [x] **DML/EDML: replace default LightGBM regressor with `sklearn.ensemble.HistGradientBoostingRegressor`** — users can still pass a custom LightGBM model via `model=` kwarg.
- [x] **CRIT: replace hardcoded LightGBM quantile regressors in `_conformalized_ci_test`** — switched to `sklearn.ensemble.GradientBoostingRegressor(loss='quantile', alpha=...)` with optional `quantile_model_factory`.
- [x] **Move lightgbm from `dependencies` to `[project.optional-dependencies]`** in `pyproject.toml` — now in optional `ml` extra.
- [x] **Update smoke tests** — ML smoke checks no longer require lightgbm or dcor.

## New tests for Paper 1 benchmark

### Kernel family (via rpy2 — R RCIT package)

- [x] **RCoT** — Implemented via R `RCIT::RCoT` wrapper and registered as `'rcot'`.
- [x] **RCIT** — Implemented via R `RCIT::RCIT` wrapper and registered as `'rcit'`.
- [x] **KCI via R** — Rewired to R `RCIT::KCIT` via rpy2 with n<=2000 guard. Registration kept as `'kci'`.

### kNN CMI family

- [x] **CMIknn** — Wrapped `tigramite.independence_tests.cmiknn.CMIknn`. Registered as `'cmiknn'`.
- [x] **CMIknnMixed** — Wrapped tigramite `CMIknnMixed` (with import-path fallback). Registered as `'cmiknn_mixed'`.
- [x] **mCMIkNN** — Added wrapper from `/Users/pavelaverin/Projects/vendor/mCMIkNN/src` with lazy local import. Registered as `'mcmiknn'`.

### Regression family

- [x] **RegressionCI** — Wrapped `tigramite.independence_tests.regressionCI.RegressionCI`. Registered as `'regci'`.

### GCM family

- [x] **GCM-linear** — Implemented with OLS residualization and asymptotic normal test statistic. Registered as `'gcm_linear'`.
- [x] **GCM-RF** — Implemented with Random Forest residualization. Registered as `'gcm_rf'`.
- [x] **WGCM-RF** — Implemented with RF sample splitting and weighted residual product statistic. Registered as `'wgcm_rf'`.

### Discretization-aware

- [x] **DCT** — Added wrapper from `/Users/pavelaverin/Projects/vendor/DCT` with lazy local import. Registered as `'dct'`.

### Adapter strategies

These wrap existing tests with a data transformation step. Each is a thin wrapper, not a new test family.

- [x] **Discretize + Chi-squared** — Equal-frequency discretization adapter implemented. Registered as `'disc_chisq'`.
- [x] **Discretize + G-squared** — Equal-frequency discretization adapter implemented. Registered as `'disc_gsq'`.
- [x] **Dummy-code + Fisher Z** — One-hot encoding adapter with Fisher combined p-value implemented. Registered as `'dummy_fisherz'`.
- [x] **Hartemink + Chi-squared** — Implemented via R `bnlearn` Hartemink discretization + chi-squared. Registered as `'hartemink_chisq'`.

## Reorganize test modules by statistical family

Current modules are grouped by **implementation source** (`simple_tests`, `statistical_model_tests`, `ml_based_tests`, `r_based_tests`, `extended_tests`, `tigramite_based_tests`). Reorganize to match the **survey taxonomy** (Section 5):

- [x] **`partial_correlation_tests.py`** — Added taxonomy module for FisherZ, Spearman.
- [x] **`contingency_table_tests.py`** — Added taxonomy module for ChiSq, GSq.
- [x] **`regression_tests.py`** — Added taxonomy module for RegressionCI.
- [x] **`nearest_neighbor_tests.py`** — Added taxonomy module for CMIknn, CMIknnMixed, mCMIkNN.
- [x] **`kernel_tests.py`** — Added taxonomy module for KCI, RCIT, RCoT.
- [x] **`ml_based_tests.py`** — Re-exported GCM family alongside DML/CRIT/EDML in taxonomy context.
- [x] **`adapter_tests.py`** — Added taxonomy module for DiscChiSq, DiscGSq, DummyFisherZ, HarteminkChiSq.
- [x] **Update `__init__.py`** — switched package exports to taxonomy modules.
- [x] **Update tests/README and docs** — reflected new module names and taxonomy layout.

## Remove unused regression-based tests

These tests are not needed for Paper 1 benchmark and will be replaced by RegressionCI from tigramite.

- [x] **Delete Regression** (`'reg'`) — Removed legacy `Regression` implementation.
- [x] **Delete Logit** (`'logit'`) — Removed legacy `Logit` implementation.
- [x] **Delete Poisson** (`'pois'`) — Removed legacy `Poisson` implementation.
- [x] **Clean up registrations and imports** — Removed deleted tests from exports and smoke tests; docs updated.

## rpy2 integration

- [x] **Make rpy2 optional.** Current R-backed tests use a lazy import and raise clear install guidance when `rpy2`/`RCIT` is missing.
- [x] **Create `citk/tests/r_based_tests.py`** — rpy2-dependent tests isolated and imported conditionally in `__init__.py`.
- [x] **Document R package requirements** — RCIT package from GitHub (`ericstrobl/RCIT`), bnlearn from CRAN.

## Testing

- [x] **Add pytest suite.** Smoke tests for all 12 existing tests (null → p > 0.05, dependent → p < 0.05).
- [x] **CI via GitHub Actions** — run pytest on push. rpy2 tests can be skipped in CI if R is not available.

## Switch to uv

- [x] **Replace setuptools with hatchling** in `pyproject.toml` build-system.
- [x] **Fix `pyproject.toml`** — author, requires-python, dependencies.
- [x] **Add optional dependency groups** — r, docs, dev.
- [x] **Add `.python-version`** file with `3.11`.
- [x] **Add `uv.lock` to `.gitignore`.**
- [x] **Delete `environment.yml`.**
- [x] **Run `uv sync --all-extras`** to create venv.

## Documentation

- [x] Update README with new test inventory.
- [x] Update `docs/source/guides/choosing_a_test.md` with new tests.
