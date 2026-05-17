---
sd_hide_title: true
---

# citests

`citests` is a Python toolkit of conditional-independence tests covering
the major families surveyed in the modern literature: partial-
correlation, contingency-table, regression-based, nearest-neighbour,
kernel, and machine-learning–based tests. It targets the
constraint-based causal-discovery setting, where the choice of CI
test is a first-class modelling decision.

## Scope

The package implements **19 tests** across **six survey families**
plus an **adapter** family for discretization / one-hot strategies.
All tests share a common base class (`CITKTest`) and satisfy the
structural `cbcd.CITest` Protocol so that any test can be passed to
any constraint-based algorithm in `cbcd` without an adapter.

| family | tests |
|---|---|
| Partial correlation | `FisherZ`, `Spearman` |
| Contingency table | `ChiSq`, `GSq` |
| Regression | `RegressionCI`, `CiMM` |
| Nearest neighbour | `CMIknn`, `CMIknnMixed`, `MCMIknn` |
| Kernel | `KCI`, `RCIT`, `RCoT` |
| ML-based | `GCM`, `WGCM`, `PCM` |
| Adapter | `DiscChiSq`, `DiscGSq`, `DummyFisherZ`, `HarteminkChiSq` |

A taxonomy with definitions and references appears in
[Explanation: taxonomy of CI tests](explanation/taxonomy_of_tests.md).

## Architectural commitments

`citests` does not import `cbcd`, `bnmetrics`, or `dagsampler`. Cross-package
interoperability flows through the structural `cbcd.CITest` Protocol:
every `CITKTest` subclass exposes `n_vars`, `__call__(X, Y, S)`, and
`details(X, Y, S)`, making it directly consumable by
`cbcd.pc(data, ci_test=...)` and the rest of the cbcd algorithm
suite. The Protocol contract is stable across citests's v0.x line under
decision **D14** of the package's API stability document.

Optional backends — `tigramite`, `pycomets`, R via `rpy2` — are
gated behind extras (`[tigramite]`, `[pycomets]`, `[r]`); the core
package installs only the partial-correlation and (when
`[causallearn]` is also installed) contingency-table families.

## Reading this documentation

New users should start with the [Tutorial](tutorial.md).
Practitioners with a specific
goal should consult the [How-to](howto/index.md) section, especially
[How to choose a CI test](howto/choosing_a_test.md). The
[Reference](reference/index.md) is regenerated from docstrings on
every build. The [Explanation](explanation/index.md) section
discusses the underlying statistical theory, the survey taxonomy,
and the API stability contract.

Detailed per-test pages — assumptions, mathematical formulation,
code examples, references — live under [Tests](tests/index.rst);
they are organised by survey family.

```{toctree}
:maxdepth: 1
:caption: Tutorial
:hidden:

tutorial
```

```{toctree}
:maxdepth: 2
:caption: How-to
:hidden:

howto/index
```

```{toctree}
:maxdepth: 2
:caption: Tests
:hidden:

tests/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Explanation
:hidden:

explanation/index
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

contributing
changelog
```
