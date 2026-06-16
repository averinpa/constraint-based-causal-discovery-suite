# Changelog

All notable changes to `dagsampler` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Unreleased

### Added

- **Tail-shape ("shape") noise model** via `noise_model = {"name": "shape", ...}`. A
  parent drives the *skewness* of a continuous child's noise while its conditional mean
  and variance are held fixed: a higher-moment edge that is invisible to mean- and
  covariance-based CI tests (Fisher's Z, GCM, WGCM, PCM) and to location-scale tests, and
  detectable only by a test that looks at the conditional distribution (e.g. a
  quantile-based CI test). Noise is drawn from a per-row standardized skew-normal
  (Azzalini's representation), affinely rescaled so mean `0` and variance `std**2` hold for
  every row regardless of the shape parameter `alpha`; only the skewness (monotone in
  `alpha`) depends on the parents. The parent→`alpha` map is selected by `func`, one of
  `skew_first_parent` (`alpha = 4 * first_parent`), `skew_tanh_first_parent`
  (`alpha = 8 * tanh(first_parent)`, bounded), `skew_mean_parents`
  (`alpha = 4 * mean(parents)`), or any callable `parent_frame -> alpha` array. `std`
  defaults to `Uniform(0.5, 1.5)`. The model is **opt-in only**: it is never selected by
  the random noise default, so 0.1.0–0.3.0 behaviour is preserved.

- **Configurable base distribution for heteroskedastic noise** via
  `noise_model = {"name": "heteroskedastic", "func": ..., "dist": ...}`. The base draw is
  standardized to unit variance, so the heteroskedastic function still sets the
  *conditional standard deviation* whatever the base distribution. `dist` is one of
  `gaussian` (default), `student_t` (`df > 2`, heavy-tailed), `laplace`, `uniform`,
  `gamma` (`shape`, right-skewed), `exponential`; `cauchy` is rejected (no finite
  variance). This makes heavy-tailed *and* heteroscedastic nulls expressible directly.
  The default `gaussian` is byte-identical to the 0.1.0–0.3.0 heteroskedastic path.

## [0.3.0] - 2026-06-06

### Added

- **Spread-controlled softmax weights** via `simulation_params['softmax_weight_mode']`.
  Default `'gaussian'` preserves the 0.1.0/0.2.0 behavior (N(0, std^2) floored at
  `random_weight_min_abs`, std 0.25 for categorical parents / 0.5 for continuous, now also
  overridable via `softmax_gaussian_std_kk` / `softmax_gaussian_std_ck`). Setting
  `'spread'` draws each weight set, removes the components the softmax is invariant to, and
  rescales the residual contrast to a spread drawn from `softmax_spread_kk` /
  `softmax_spread_ck` = `[lo, hi]` (required in this mode; no scale is hardcoded). The lower
  bound guarantees a detectable logit contrast even for binary children; class balance is
  unaffected.
- **Standardized (design-A) `threshold` model** via `simulation_params['threshold_standardized']`.
  When set, the latent score gets absolute Gaussian noise (`threshold_noise_abs`, default
  1.0), is standardized to unit variance, and is binned at equal-probability cutpoints
  `Phi^{-1}(j/c)`. This makes the threshold a discretized linear-Gaussian latent: the
  coefficient sets only the latent signal-to-noise ratio and the marginal category
  distribution is uniform regardless of the coefficient. **Default `False` preserves the
  0.2.0 raw-score / random-cutpoint behavior.**

- **Stratum-mean spread** via `simulation_params['strata_means_spread'] = [lo, hi]`. When
  set, a per-node `sigma_mu ~ Uniform(lo, hi)` is drawn and the auto-sampled stratum means
  are scattered as `N(0, sigma_mu^2)`, controlling the between-stratum variance (how strongly
  a categorical parent shifts a continuous child's mean). Default (unset) keeps the legacy
  `N(0, 1)`.

### Notes

- All new behavior is opt-in via `simulation_params`; existing configs and the committed
  0.1.0/0.2.0 benchmark runs are unaffected. Calibrated band values are supplied by the
  consuming experiment, not hardcoded in the package.

## [0.2.0] - 2026-05-21

### Added

- Optional `categorical_model.noise_scale` for the `threshold` categorical
  model (ordered-probit form). When `> 0`, idiosyncratic latent noise is
  added to the linear index `w·parents` before discretization, so the
  categorical retains residual variation given its parents. The noise SD is
  scaled by the SD of `w·parents`, so `noise_scale` is a noise-to-signal SD
  ratio that holds the conditional dependence strength constant across
  cardinalities and parent mechanisms. **Default `0.0` preserves the
  deterministic (pure-discretization) behavior of 0.1.0** — existing
  configs and the committed 0.1.0 benchmark runs are unaffected. Fixes
  unfaithful threshold alternatives in mixed-type benchmarks (a thresholded
  child was previously a pure deterministic function of its parents).
- `CausalDataGenerator.as_ci_oracle()` returning a `DSeparationOracle`
  that conforms structurally to the `cbcd.CITest` Protocol
  (`n_vars`, `__call__(x, y, S) -> float`, `details(x, y, S)` with
  `.p_value`). Variable indices map to the alphabetically-sorted column
  order of the generated dataframe; p-values are 1.0 for d-separated
  pairs and 0.0 otherwise. No dependency on `cbcd` — conformance is
  purely structural.
- `DSeparationOracle` exported from the top-level `dagsampler` package.

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
