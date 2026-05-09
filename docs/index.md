---
sd_hide_title: true
---

# cbcd

`cbcd` (constraint-based causal discovery) is a Python library
implementing the PC and FCI algorithm families for i.i.d. data and
the PCMCI family for time-series data. It targets the research and
applied-causal-inference setting: every published algorithm in the
package carries a structural-regression test against a d-separation
oracle and, where a reference implementation exists, a parity test
against it.

## Scope

The package implements:

- **PC family** — `cbcd.pc` (Spirtes & Glymour, 1991; Spirtes et al.,
  2000), returning a CPDAG over an observed variable set under
  causal sufficiency.
- **FCI family** — `cbcd.fci`, `cbcd.rfci`, `cbcd.anytime_fci`
  (Spirtes et al., 2000; Colombo et al., 2012; Spirtes, 2001),
  returning a partial ancestral graph (PAG) without assuming causal
  sufficiency.
- **PCMCI family** — `cbcd.pcmci` (Runge et al., 2019), recovering
  a time-series CPDAG over lagged variables under stationarity.

Algorithms are wired from a small set of structural Protocols
(`CITest`, `SkeletonAlgorithm`, `ColliderOrienter`, `CPDAGRules`,
`PAGRules`, `PAGSkeletonRefinement`); composing a new algorithm is
mostly plumbing existing pieces. The full Protocol contract is
frozen under decision **D15** for v0.x and is documented in the
[Reference](reference/index.md).

## Architectural commitments

`cbcd` does **not** depend on `causal-learn`, directly or
transitively. CI tests plug in through the structural `cbcd.CITest`
Protocol and may originate from `cbcd.citest` (the bundled
linear-Gaussian and χ² tests), from `citk` (a sister toolkit covering
kernel, regression-based, and ML-based tests), or from any
user-defined object satisfying the Protocol's three members
(`n_vars`, `__call__`, `details`). The contract is enforced at
construction time by Python's structural duck-typing — no inheritance
is required.

A d-separation oracle suitable for structural-regression testing is
provided by the sister package `dagsampler` via its
`as_ci_oracle()` method.

## Reading this documentation

The site follows the **Diátaxis** layout. New users should start
with the [Tutorial](tutorial.md). Practitioners with a specific task
should consult the [How-to](howto/index.md) section. The
[Reference](reference/index.md) is automatically generated from
docstrings on every build. The [Explanation](explanation/index.md)
section discusses theory, design rationale, and the rationale
behind specific implementation choices.

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

## References

- Colombo, D., Maathuis, M. H., Kalisch, M., & Richardson, T. S.
  (2012). Learning high-dimensional directed acyclic graphs with
  latent and selection variables. *The Annals of Statistics*, 40(1),
  294–321.
- Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic,
  D. (2019). Detecting and quantifying causal associations in large
  nonlinear time series datasets. *Science Advances*, 5(11),
  eaau4996.
- Spirtes, P. (2001). An anytime algorithm for causal inference.
  *Proceedings of AISTATS 2001*, 213–221.
- Spirtes, P., & Glymour, C. (1991). An algorithm for fast recovery
  of sparse causal graphs. *Social Science Computer Review*, 9(1),
  62–72.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
  Prediction, and Search* (2nd ed.). MIT Press.
