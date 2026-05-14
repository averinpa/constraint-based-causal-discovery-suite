---
sd_hide_title: true
---

# bnm

`bnm` (Bayesian network metrics) is a Python library for comparing
and visualising directed acyclic graphs (DAGs), completed partially
directed acyclic graphs (CPDAGs), and partial ancestral graphs
(PAGs). It implements a wide range of comparative and descriptive
metrics, Markov-blanket comparison, and graph visualisations.

## Scope

The package implements:

- **Descriptive metrics** — `bnm.count_edges`, `bnm.count_colliders`,
  `bnm.count_directed_arcs`, etc. — over a single graph.
- **Comparative metrics** — `bnm.shd` (Tsamardinos et al., 2006),
  `bnm.hd`, `bnm.f1`, `bnm.precision`, `bnm.recall`,
  `bnm.count_additions`, `bnm.count_deletions`, `bnm.count_reversals`.
- **Structural Intervention Distance** — `bnm.sid` (Peters &
  Bühlmann, 2015), with both lower and upper bounds when comparing
  a DAG to a CPDAG.
- **Markov-blanket comparison** — `bnm.markov_blanket`, returning
  the parents, children, and spouses of a node.
- **Visualisation** — `bnm.plot_graph`, `bnm.plot_side_by_side`,
  `bnm.plot_sid_matrix` (gated behind the optional `[viz]` extra).

Inputs are accepted through the structural `bnm.GraphLike` Protocol,
which is satisfied by every graph type in the sister package
[`cbcd`](https://github.com/averinpa/cbcd) (DAG, CPDAG, PAG, MAG)
without conversion. Plain `networkx.DiGraph` instances and raw
`int8` endpoint matrices are also accepted via `bnm.to_graphlike`.

## Architectural commitments

`bnm` does not depend on `cbcd`, `dagsampler`, or `causal-learn`.
Cross-package interoperability is mediated by the
`bnm.GraphLike` Protocol — any object exposing `n_vars: int`,
`endpoints: ndarray[int8]`, and `var_names: tuple[str, ...]`
satisfies it. Validation is performed by `bnm.to_graphlike` at the
input boundary; downstream metric functions trust the normalised
representation.

The endpoint-mark convention follows cbcd's: `0` for no edge, `1`
for TAIL, `2` for ARROW, `3` for CIRCLE.

## Reading this documentation

New users should start
with the [Tutorial](tutorial.md). Practitioners with a specific
goal should consult the [How-to](howto/index.md) section. The
[Reference](reference/index.md) is regenerated from docstrings on
every build. The [Explanation](explanation/index.md) section
discusses the mathematical foundations of the metrics, the SID
algorithm, the `GraphLike` Protocol, and Markov-blanket scoping.

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

- Peters, J., & Bühlmann, P. (2015). Structural intervention
  distance for evaluating causal graphs. *Neural Computation*,
  27(3), 771–799.
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The
  max-min hill-climbing Bayesian network structure learning
  algorithm. *Machine Learning*, 65(1), 31–78.
