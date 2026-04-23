# Changelog

All notable changes to `dagsampler` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-23

First public release.

### Added

- `CausalDataGenerator` class for generating synthetic data from configurable
  causal DAGs.
- `custom` and `random` DAG generation with separate `seed_structure` /
  `seed_data` random streams (or a single convenience `seed`).
- Mixed continuous, binary, and categorical nodes with configurable
  cardinality.
- Structural forms: `linear`, `polynomial`, `interaction`, `sigmoid`, `cos`,
  `sin`, `stratum_means`.
- Optional element-wise `post_transform` (`tanh`, `sin`, `cos`, `exp_neg_abs`,
  `sqrt_abs`, `relu`, `sign`).
- Cross-type mechanisms:
  - continuous → categorical via `categorical_model.name = "threshold"`
  - categorical → continuous via `functional_form.name = "stratum_means"`,
    including mixed-parent cases with a `metric_weights` linear contribution.
- Noise models:
  - additive (`gaussian`, `student_t`, `gamma`, `exponential`, `laplace`,
    `cauchy`, `uniform`)
  - multiplicative (`gaussian`, `student_t`, `gamma`, `exponential`)
  - heteroskedastic (`abs_first_parent`, `abs_parent_plus_const`,
    `mean_abs_plus_const`)
- Random structural weight sampling controls: `random_weight_low`,
  `random_weight_high`, and `random_weight_min_abs` (excludes near-zero
  coefficients to guarantee minimum signal strength).
- `force_uniform_marginals` flag for balanced exogenous binary / categorical
  draws.
- `binary_proportion` / `categorical_proportion` controls for random node-type
  assignment in random DAGs.
- Template helpers in `dagsampler.templates`: `chain_config`,
  `fork_config`, `collider_config`, `indep_config`, `independence_config`.
- Optional d-separation CI oracle output via `store_ci_oracle` and
  `ci_oracle_max_cond_set`.
- `dagsampler-generate` CLI entry point with `--config`, `--output`,
  `--params-out`, and `--edges-out` flags.
- Sphinx documentation covering overview, model formulations, configuration
  examples, templates, usage, and API reference.
- Test suite of 50 tests covering noise models, graph generation, mixed-type
  edge cases, post-transform behavior, sigmoid/cos/sin functional forms, and
  template smoke tests.

### Changed

- License changed from `Proprietary` to `MIT` for public release.
