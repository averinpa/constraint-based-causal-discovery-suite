# `bnmetrics`: Bayesian Network Metrics

`bnmetrics` is a Python library for evaluating, comparing, and visualising
DAGs, CPDAGs, and PAGs — descriptive structural metrics, comparative
metrics including SHD, HD, F1, and SID, and Markov-blanket-scoped
analysis, with optional graphviz and plotly visualisations.

`bnmetrics` is part of the [constraint-based causal discovery
suite](https://github.com/averinpa/constraint-based-causal-discovery-suite)
alongside `cbcd` (algorithms), `citests` (CI tests), and `dagsampler`
(simulator). Any object satisfying the `bnmetrics.GraphLike` Protocol —
including `cbcd`'s `DAG`, `CPDAG`, and `PAG` instances — drives every
metric without imports between the packages.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-averinpa.github.io-blue.svg)](https://averinpa.github.io/constraint-based-causal-discovery-suite/bnmetrics/)

## Features

- **Descriptive metrics** — `count_edges`, `count_nodes`,
  `count_colliders`, `count_root_nodes`, `count_leaf_nodes`,
  `count_isolated_nodes`, `count_directed_arcs`,
  `count_undirected_arcs`, `count_bidirected_arcs`,
  `count_circle_edges`, `count_reversible_arcs`, `in_degree`,
  `out_degree`.
- **Comparative metrics** — `shd`, `hd`, `f1`, `precision`, `recall`,
  `true_positives`, `false_positives`, `false_negatives`,
  `count_additions`, `count_deletions`, `count_reversals`.
- **Structural Intervention Distance** — `sid()` after Peters &
  Bühlmann (2015), returning the SID together with bounds for CPDAG
  comparison.
- **Markov-blanket scoping** — `markov_blanket(g, var)` returns a
  sub-`GraphLike` that can be passed back to any metric.
- **Multi-metric comparison** — `compare(true, estimate)` produces a
  `Comparison` exposing every descriptive and comparative metric in
  a single call, with optional pandas export.
- **Visualisation** (optional `viz` extra) — side-by-side graphviz
  comparison with true-positive highlighting and a plotly heatmap
  for SID's incorrect-edge matrix.

## Installation

```bash
pip install bnmetrics                # core (numpy only)
pip install bnmetrics[viz]           # + graphviz, plotly, ipython
pip install bnmetrics[networkx]      # + networkx (DiGraph adapter input)
pip install bnmetrics[pandas]        # + pandas (compare().to_dataframe())
```

## Documentation

- Full [documentation](https://averinpa.github.io/constraint-based-causal-discovery-suite/bnmetrics/) is hosted on GitHub Pages.
- [Examples](examples/) — runnable Jupyter notebooks.
- [CHANGELOG](CHANGELOG.md).

## Acknowledgements

`bnmetrics` is the Python successor to
[DAGMetrics](https://github.com/averinpa/DAGMetrics), an R package by
the same author for analysing Bayesian networks in microbial
abundance data ([Averin et al.,
2025](https://doi.org/10.3390/agronomy15040987)). The metric
definitions — Hamming distance, structural Hamming distance, F1,
additions / deletions / reversals, reversible-arc counts, the
Markov-blanket subgraph construction — are derivative of the R
original.

The Structural Intervention Distance follows [Peters & Bühlmann
(2015)](https://doi.org/10.1162/NECO_a_00708); the implementation
operates directly on the int8 endpoint-mark matrix used throughout
the suite.

`bnmetrics` v0.2.x is a full Python rewrite around a canonical `(n, n)`
int8 endpoint-mark matrix matching `cbcd`'s representation.
Cross-package interop with `cbcd` and `dagsampler` is via the
structural `bnmetrics.GraphLike` Protocol — no imports between the
packages.

## License

[MIT](LICENSE)
