---
sd_hide_title: true
---

# dagsampler

`dagsampler` is a Python library for generating synthetic data from
configurable directed acyclic graphs. It targets the research and
benchmarking setting: every choice (graph topology, node type,
mechanism family, noise model, post-nonlinear transform) is
explicit and reproducible from a single configuration object, and
the simulator can additionally emit a d-separation oracle for
constraint-based discovery benchmarks.

## Scope

The package implements:

- **DAG generation** — `custom` (user-defined node and edge sets)
  and `random` (random acyclic edges over an ordered node set).
- **Mixed node types** — continuous, binary, and categorical
  variables with configurable cardinality.
- **Structural mechanisms** — linear, polynomial, interaction,
  sigmoid (`tanh`), cosine, sine, and stratum-means functional
  forms; logistic and threshold categorical models.
- **Noise models** — additive (Gaussian, Student-*t*, gamma,
  exponential, Laplace, Cauchy, uniform), multiplicative, and
  heteroskedastic Gaussian.
- **Post-nonlinear transforms** — element-wise `tanh`, `sin`,
  `cos`, `exp_neg_abs`, `sqrt_abs`, `relu`, `sign` applied after
  the structural function and noise.
- **Cross-type mechanisms** — continuous-to-categorical via
  threshold model; categorical-to-continuous via stratum-means,
  with optional metric-parent contributions.
- **Reproducibility controls** — separate `seed_structure` and
  `seed_data` streams so the data-generating process and the
  finite-sample draws can be pinned independently.
- **CI oracle** — optional d-separation truth table over the
  generated DAG, plus the `as_ci_oracle()` method (since v0.2.0)
  returning a `cbcd.CITest`-conforming object usable directly in
  constraint-based algorithms.

The full mathematical specification of the simulator — node-type
combinations, structural equations, noise families, link functions,
compatibility matrix — is documented in
[Explanation: SCM formulation](explanation/scm_formulation.md).

## Architectural commitments

`dagsampler` does not import `cbcd`, `bnmetrics`, or `citests`. Cross-package
interoperability flows through structural Protocols defined in the
sister packages:

- **CI oracle.** `CausalDataGenerator.as_ci_oracle()` returns a
  `DSeparationOracle` instance that satisfies the `cbcd.CITest`
  Protocol — `n_vars: int`, `__call__(x, y, S) -> float`,
  `details(x, y, S)` returning an object with `.p_value` — so the
  oracle plugs directly into `cbcd.pc(data, ci_test=...)` without
  a wrapper.
- **True graph.** The `result["dag"]` value (a `networkx.DiGraph`)
  is accepted by `bnmetrics.to_graphlike` for structural comparison
  against any cbcd algorithm output.

These contracts let `dagsampler` ship as an independently-versioned
package while still composing cleanly with the rest of the suite.

## Reading this documentation

The site follows the **Diátaxis** layout. New users should start
with the [Tutorial](tutorial.md). Practitioners with a specific
configuration goal should consult the [How-to](howto/index.md)
section, especially the
[configuration cookbook](howto/config_cookbook.md). The
[Reference](reference/index.md) is regenerated from docstrings on
every build. The [Explanation](explanation/index.md) section
documents the underlying SCM formulation, the CI oracle, and the
seeding model.

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
