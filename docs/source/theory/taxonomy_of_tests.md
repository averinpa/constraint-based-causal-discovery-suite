# A Taxonomy of Conditional Independence Tests

The 19 conditional independence tests in `citk` are organised under the six families of the Paper 0 survey, plus a group of adapter strategies that wrap base tests rather than constituting a distinct family. Understanding these groups helps you reason about which test fits a given research question and data type.

## 1. Partial Correlation

Tests based on (Pearson- or Spearman-)partial correlation between two variables after controlling for the conditioning set.

- **Core idea**: Fit out the conditioning set linearly (or after ranking) and test whether the residualised correlation is zero.
- **Examples in `citk`**: `fisherz_citk`, `spearman`.
- **Strengths**: Computationally very fast; statistically efficient under the relevant assumptions (linearity, Gaussianity, or monotonicity).
- **Weaknesses**: Low power against non-linear or non-monotonic dependence.

## 2. Contingency Table

Classical statistical tests designed for discrete (categorical) variables.

- **Core idea**: Compare observed and expected cell counts in a stratified contingency table.
- **Examples in `citk`**: `chisq`, `gsq`.
- **Strengths**: Well-understood asymptotic theory; robust on truly categorical data.
- **Weaknesses**: Requires discrete data; loses power as the contingency table grows sparse relative to the sample size.

## 3. Regression

Parametric likelihood-ratio tests built on regression models, with link functions chosen per variable type.

- **Core idea**: Compare nested regression fits with and without the variable of interest in the predictor set; the likelihood-ratio statistic is asymptotically chi-squared under the null.
- **Examples in `citk`**: `regci` (tigramite RegressionCI), `ci_mm` (R MXM ci.mm — symmetric, both directions combined).
- **Strengths**: Native support for mixed continuous and discrete data; small-sample behaviour better than non-parametric tests when the model class is appropriate.
- **Weaknesses**: Power degrades when the linear / logistic link misrepresents the true dependence.

## 4. Nearest Neighbor

Non-parametric tests based on $k$-nearest-neighbour estimators of conditional mutual information, paired with permutation-based p-values.

- **Core idea**: Estimate $I(X; Y \mid Z)$ from local neighbourhood statistics and assess significance with a local-permutation null.
- **Examples in `citk`**: `cmiknn`, `cmiknn_mixed`, `mcmiknn`.
- **Strengths**: Detects arbitrary non-linear dependence; mixed-data variants handle ties on discrete coordinates.
- **Weaknesses**: Requires adequate sample size for stable density estimation; permutation p-values are computationally non-trivial.

## 5. Kernel

Non-parametric tests that operate in a Reproducing Kernel Hilbert Space (RKHS), with Hilbert-Schmidt independence criteria as the underlying dependence measure.

- **Core idea**: Map data into an RKHS and test for independence in the residualised kernel features; under a universal kernel, the criterion is zero exactly when the variables are independent.
- **Examples in `citk`**: `kci` (exact, Python causal-learn implementation), `rcit` and `rcot` (random Fourier feature approximations, R RCIT package).
- **Strengths**: Detects arbitrary smooth dependence; few distributional assumptions.
- **Weaknesses**: Exact `kci` is at least quadratic in sample size; sensitivity to kernel and bandwidth choice.

## 6. Machine-Learning-Based

Tests built around nuisance regressions estimated by flexible ML predictors, with calibrated test statistics derived from the residual structure.

- **Core idea**: Regress $X$ and $Y$ on the conditioning set $Z$ using an ML method, then test the residuals for non-zero covariance (GCM), weighted covariance (WGCM), or projected covariance (PCM).
- **Examples in `citk`**: `gcm`, `wgcm`, `pcm` (all via `pycomets` with random forest regression by default).
- **Strengths**: Asymptotic-normal calibration with flexible nuisance models; `wgcm` adds power on localised dependence; `pcm` is assumption-lean and robust to weakly identified predictors.
- **Weaknesses**: Requires sufficient sample size for nuisance estimation rates to hold; test calibration depends on the rate condition.

## 7. Adapter Strategies

Adapters that modify or wrap a base test rather than constituting a distinct family. The survey describes these as robustness layers — transformations applied on top of an existing CI test rather than a seventh family.

- **Core idea**: Transform the data — discretise, dummy-encode, or apply an information-preserving binning — and then call a base CI test on the transformed data.
- **Examples in `citk`**: `disc_chisq`, `disc_gsq` (equal-frequency discretisation + `chisq`/`gsq`); `dummy_fisherz` (one-hot encoding + Fisher's combined `fisherz`); `hartemink_chisq` (Hartemink information-preserving discretisation via R `bnlearn` + `chisq`).
- **Strengths**: Lets classical tests apply to data types they were not designed for; useful baselines for mixed-data settings.
- **Weaknesses**: Inherits the assumptions of both the transformation and the base test; performance depends on whether the transformation preserves the dependence structure.
