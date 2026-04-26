# How to Choose a Conditional Independence Test

Choosing the right conditional independence (CI) test depends on the type of your data and the assumptions you are willing to make about the underlying dependence. `citk` ships 19 tests organised under the six survey families plus four adapter strategies; this guide gives a practical mapping from data + assumptions to tests.

## Key Considerations

### 1. Data Type

- **All continuous**:
    - `fisherz_citk`: linear, Gaussian — fastest baseline.
    - `spearman`: monotonic but not necessarily linear — robust non-parametric alternative.
    - `kci`: kernel-based, captures arbitrary non-linear dependence.
    - `rcit`, `rcot`: random-Fourier-feature approximations to KCI; faster on larger samples.
    - `cmiknn`: kNN-based conditional mutual information with local-permutation p-values.

- **All discrete (categorical)**:
    - `gsq` (G-test) or `chisq` (Chi-Square): classical contingency-table tests.
    - `dummy_fisherz`: one-hot encoding adapter that aggregates Fisher-Z calls; competitive when categorical cardinalities are moderate.

- **Mixed continuous + discrete**:
    - `cmiknn_mixed`: mixed-type kNN CMI estimator (tigramite).
    - `mcmiknn`: another mixed-type kNN CMI implementation (vendored from upstream `hpi-epic/mCMIkNN`).
    - `regci`: parametric likelihood-ratio test using GLM regression chosen per response type (continuous → linear, discrete → logistic).
    - `ci_mm`: symmetric likelihood-ratio test from R `MXM` that runs both regression directions and combines them.
    - `gcm`, `wgcm`, `pcm`: ML-residualisation tests using random forest regression (via `pycomets`); flexible, asymptotically calibrated, and the RF nuisance regressions handle continuous, discrete, or mixed inputs natively.
    - `disc_chisq`, `disc_gsq`: equal-frequency discretisation adapters around classical discrete tests.
    - `hartemink_chisq`: information-preserving Hartemink discretisation (via R `bnlearn`) + Chi-Square; better dependence preservation than equal-frequency binning.

### 2. Relationship Type

- **Linear**: `fisherz_citk` is the computationally efficient choice when both Gaussianity and linearity hold.
- **Monotonic**: `spearman` works on ranks; robust to non-linearities as long as the relationship is monotonic.
- **Non-linear / complex**: kernel tests (`kci`, `rcit`, `rcot`), kNN-based tests (`cmiknn`, `cmiknn_mixed`, `mcmiknn`), and ML-residualisation tests (`gcm`, `wgcm`, `pcm`) are all designed to detect arbitrary dependence at higher computational cost. `wgcm` and `pcm` add power on alternatives where the dependence is localised in the conditioning space or where the predictor is weakly identified.

### 3. Sample Size

- **Small samples**: classical tests (`fisherz_citk`, `spearman`, `chisq`, `gsq`) are most reliable; non-parametric and ML-based tests need more data for stable estimation.
- **Large samples**: kernel tests (especially exact `kci`) become expensive — prefer `rcit`/`rcot` for random-feature approximations, or `gcm`/`wgcm`/`pcm` for ML-residualisation with linear cost.
- **Very large samples**: `kci` is roughly quadratic in $n$; consider capping or switching to a faster family.

## Summary Table

| Test Name | Family | Data Type | Relationship Type | Key Assumption(s) |
|-----------|--------|-----------|-------------------|-------------------|
| `fisherz_citk` | Partial Correlation | Continuous | Linear | Approximate Gaussianity |
| `spearman` | Partial Correlation | Continuous | Monotonic | Monotonicity |
| `chisq` | Contingency Table | Discrete | Any | Adequate cell counts |
| `gsq` | Contingency Table | Discrete | Any | Adequate cell counts |
| `regci` | Regression | Mixed or continuous | Any (within model class) | Correct GLM specification per variable type; requires `tigramite` |
| `ci_mm` | Regression | Mixed | Any (within model class) | Correct linear/logistic per variable; requires `rpy2` + R `MXM` |
| `cmiknn` | Nearest Neighbor | Continuous | Any | Sample size adequate for kNN density estimation; requires `tigramite` |
| `cmiknn_mixed` | Nearest Neighbor | Mixed | Any | Variable types declared via `data_type`; requires `tigramite` |
| `mcmiknn` | Nearest Neighbor | Mixed | Any | Vendored upstream `mCMIkNN`; no extra install required |
| `kci` | Kernel | Continuous | Any | Suitable kernel choice; cost is at least quadratic in $n$ |
| `rcit` | Kernel | Continuous | Any | Random-feature approximation; requires `rpy2` + R `RCIT` |
| `rcot` | Kernel | Continuous | Any | Random-feature approximation with reduced-dim conditioning; requires `rpy2` + R `RCIT` |
| `gcm` | Machine-Learning-Based | Mixed or continuous | Any | Consistent nuisance regression; requires `pycomets` |
| `wgcm` | Machine-Learning-Based | Mixed or continuous | Any (esp. localised) | Consistent nuisance regression + sample splitting; requires `pycomets` |
| `pcm` | Machine-Learning-Based | Mixed or continuous | Any (assumption-lean) | Consistent residualisation; requires `pycomets` |
| `disc_chisq` | Adapter Strategies | Mixed or continuous | Any | Discretisation preserves dependence; ChiSq cell-count rule |
| `disc_gsq` | Adapter Strategies | Mixed or continuous | Any | Discretisation preserves dependence; GSq cell-count rule |
| `dummy_fisherz` | Adapter Strategies | Mixed or discrete | Any (encoded space) | One-hot encoding fidelity; combined p-values approximation |
| `hartemink_chisq` | Adapter Strategies | Mixed or continuous | Any | Information-preserving discretisation; requires `rpy2` + R `bnlearn` |
