# Changelog

All notable changes to `citk` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased] (pre-v0.1.0)

The pre-v0.1.0 cycle scopes `citk` to exactly the 19 Paper 1 benchmark tests
and aligns the module layout, public API, and documentation with the Paper 0
survey taxonomy.

### Added

- Per-test documentation pages for all 19 Paper 1 benchmark tests, grouped
  under the seven survey-taxonomy headers in `docs/source/tests/index.rst`.
- New tests: `regci`, `ci_mm`, `cmiknn`, `cmiknn_mixed`, `mcmiknn`, `rcit`,
  `rcot`, `gcm`, `wgcm`, `pcm`, `disc_chisq`, `disc_gsq`, `dummy_fisherz`,
  `hartemink_chisq`.
- Optional dependency group `pycomets` providing `pycomets` and `xgboost` for
  the `gcm`, `wgcm`, `pcm` family.
- Optional dependency group `tigramite` for `cmiknn`, `cmiknn_mixed`, `regci`.
- `sphinx.ext.coverage` extension wired into the docs build for module-level
  coverage checks.

### Changed

- Module layout reorganised to match the Paper 0 survey taxonomy. The seven
  family modules `partial_correlation_tests`, `contingency_table_tests`,
  `regression_tests`, `nearest_neighbor_tests`, `kernel_tests`,
  `ml_based_tests`, and `adapter_tests` are now the canonical public surface
  for `citk.tests`; source-grouped modules sit behind them as implementation
  files.
- `KCI` moved from `ml_based_tests.py` into `kernel_tests.py` to match the
  survey's kernel-family classification.
- `ml_based_tests.py` repurposed as the GCM-family taxonomy stub re-exporting
  `GCM`, `WGCM`, `PCM` from `pycomets_tests.py`.
- `kci` resolves to the Python `causal-learn` KCI implementation; the previous
  import-order shadowing with the R wrapper has been removed.
- API reference consolidated into a single
  `docs/source/api/citk.rst` page with sections matching the seven taxonomy
  families plus a Base Class section.
- `:undoc-members:` removed from `autodoc_default_options` so missing public
  docstrings surface during build instead of being papered over.
- Local-wrapper paths moved from `/Users/pavelaverin/Projects/<repo>` to
  `/Users/pavelaverin/Projects/vendor/<repo>` for `mcmiknn`.

### Removed

- Tests outside the 19 Paper 1 benchmark scope: `rf`, `dml`, `crit`, `edml`,
  `dct`, and their per-test doc pages.
- Unregistered native GCM helpers: `GCMLinear`, `GCMRF`, `WGCMRF`.
- R-based wrappers superseded by their Python equivalents: `RGCM` (replaced
  by `pycomets`-backed `GCM`) and `RKCIT` (Python `causal-learn` `KCI` is
  now canonical for `kci`).
- Optional dependency group `ml = ["lightgbm"]`; no shipped test uses
  `lightgbm` after the cleanup.
- Stale orphan doc pages: `logit_test.md`, `poisson_test.md`,
  `regression_test.md`.
- Orphan API reference files merged into the main API page:
  `api/citk.tests.rst`, `api/citk.tests.base.rst`.
