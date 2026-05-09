# `pc`

```text
cbcd.pc(data, ci_test="fisherz", alpha=0.05, *,
        var_names=None, background=None,
        skeleton=None, collider=None, rules=None)
```

Recover a CPDAG from i.i.d. observational data using the PC algorithm.

## Parameters

- **data** : `ndarray` of shape `(n_samples, n_vars)` or `pandas.DataFrame`
  Observational sample. `pandas.DataFrame` columns become `var_names`
  unless `var_names` is provided explicitly.
- **ci_test** : `str` or `CITest`, default `"fisherz"`
  Conditional-independence test. Strings dispatch through the
  `cbcd.citest.make_ci_test` registry. Any object satisfying the
  structural [`cbcd.CITest`](citest.md) Protocol is accepted —
  including `citk` test classes and `dagsampler.DSeparationOracle`.
- **alpha** : `float`, default `0.05`
  Significance level for individual CI tests. Must be in `(0, 1)`.
- **var_names** : `Sequence[str]`, optional
  Variable names. Falls back to the DataFrame's columns or to
  positional indices.
- **background** : [`BackgroundKnowledge`](background.md), optional
  Required and forbidden adjacencies / orientations.
- **skeleton, collider, rules** : optional Protocol-conforming objects
  Override individual phases of the algorithm. See
  [`SkeletonAlgorithm`](skeleton.md), [`ColliderOrienter`](collider.md),
  [`CPDAGRules`](rules.md).

## Returns

- **cpdag** : [`cbcd.CPDAG`](cpdag.md)
  Recovered completed partially directed acyclic graph.

## Raises

- `CBCDInputError` — `alpha` out of `(0, 1)`, or `data` not 2-D, or
  `var_names` length mismatched.
- `CBCDDataError` — data is not numerically suitable for the chosen
  `ci_test` (e.g., non-numeric column with `"fisherz"`).

## Examples

```pycon
>>> import numpy as np
>>> from cbcd import pc
>>> rng = np.random.default_rng(0)
>>> data = rng.standard_normal((1000, 4))
>>> cpdag = pc(data, alpha=0.05)
>>> cpdag.endpoints.shape
(4, 4)
```

## See also

- [`fci`](fci.md) — analogue tolerating latent confounders.
- [`pcmci`](pcmci.md) — time-series adaptation.
- [`make_ci_test`](make_ci_test.md) — registry-based CI test factory.

## References

Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation,
Prediction, and Search* (2nd ed.). MIT Press.
