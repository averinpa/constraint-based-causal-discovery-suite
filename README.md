# BNM — Bayesian Network Metrics

> [!IMPORTANT]
> **This repository is archived.** `bnm` has moved to the
> [constraint-based-causal-discovery-suite](https://github.com/averinpa/constraint-based-causal-discovery-suite)
> umbrella, where it lives at
> [`bnm/`](https://github.com/averinpa/constraint-based-causal-discovery-suite/tree/main/bnm).
> All future development happens there. This archive is kept
> read-only for historical reference; v0.2 onwards is in the suite
> repo, with a Python-native int8 endpoint-mark matrix
> representation and a 535-test suite.

[![CI](https://github.com/averinpa/bnm/actions/workflows/ci.yml/badge.svg)](https://github.com/averinpa/bnm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

**BNM** is a Python package for evaluating, comparing, and visualizing
DAGs, CPDAGs, and PAGs. It is part of the
[constraint-based causal discovery suite](../) alongside
[`cbcd`](../cbcd/) (algorithms), [`citk`](../citk/) (CI tests), and
[`dagsampler`](../dagsampler/) (simulator).

Originally developed as
[DAGMetrics](https://github.com/averinpa/DAGMetrics) in R for analyzing
Bayesian networks in microbial abundance data
[(Averin et al., 2025)](https://doi.org/10.3390/agronomy15040987).

## Status

**v0.2 (in development).** Internal storage switched from
`networkx.DiGraph + edge type attrs` to a canonical `(n, n) int8`
endpoint-mark matrix matching cbcd's convention. The `BNMetrics`
god-class from 0.1.x is replaced by composable functions plus a
`compare()` façade. See [`docs/audit.md`](docs/audit.md) for the
0.1.x audit and [`docs/design/api_v0.py`](docs/design/api_v0.py) for
the design contracts.

## Key features

- **Cross-package interop via a structural Protocol.** `bnm.GraphLike`
  is a duck-typed Protocol over `(n_vars, endpoints, var_names)`.
  cbcd's `DAG`, `CPDAG`, `PAG` instances conform with zero adaptation —
  no imports between cbcd and bnm.
- **Descriptive metrics.** `count_edges`, `count_colliders`,
  `count_root_nodes`, `count_leaf_nodes`, `count_isolated_nodes`,
  `count_directed_arcs`, `count_undirected_arcs`,
  `count_bidirected_arcs`, `count_circle_edges`, `count_reversible_arcs`,
  `in_degree`, `out_degree`.
- **Comparative metrics.** `shd`, `hd`, `f1`, `precision`, `recall`,
  `true_positives`, `false_positives`, `false_negatives`,
  `count_additions`, `count_deletions`, `count_reversals`.
- **Structural Intervention Distance (SID).** Faithful port of
  Peters & Bühlmann (2015) on the int8 endpoint matrix; fixes two
  0.1.x bugs (empty-possible-parents crash, hash-seed
  non-determinism) and one upper-bound under-counting bug — see
  audit §1, §6, §8.
- **Markov-blanket subgraphs.** `markov_blanket(g, var)` returns a
  sub-`GraphLike` you can pass back to any metric.
- **Visualization** (optional `viz` extra). Side-by-side graphviz
  comparison with true-positive highlighting; plotly heatmap for SID's
  `incorrect_mat`.

## Installation

```bash
pip install bnm                # core (numpy only)
pip install bnm[viz]           # + graphviz, plotly, ipython
pip install bnm[networkx]      # + networkx (for nx.DiGraph adapter input)
pip install bnm[pandas]        # + pandas (for compare().to_dataframe())
pip install bnm[dev]           # + pytest, ruff, mypy, networkx, pandas
```

Development (suite-internal):

```bash
cd ~/Projects/suite/bnm
uv sync --all-extras
uv run pytest
```

## Quick start

```python
import numpy as np
import bnm

# Construct a DAG via the int8 endpoint matrix...
arr = np.array([
    [0, 2, 0],   # A → B (mark at B is ARROW=2)
    [1, 0, 2],   # mark at A is TAIL=1; B → C
    [0, 1, 0],   # mark at B is TAIL=1
], dtype=np.int8)
truth = bnm.to_graphlike(arr, var_names=("A", "B", "C"))

# ...or pass a cbcd graph directly — bnm.GraphLike conformance is duck-typed:
# from cbcd.graph.dag import DAG
# truth = DAG.from_directed_edges(3, [(0, 1), (1, 2)], var_names=("A","B","C"))

# Descriptive
print(bnm.count_edges(truth), bnm.count_colliders(truth))

# Comparative against an estimate
estimate = ...  # e.g. cbcd.pc(data) output
print("SHD:", bnm.shd(truth, estimate))
print("F1: ", bnm.f1(truth, estimate))

# SID
sid = bnm.sid(truth, estimate)
print(f"SID={sid.sid}, bounds=[{sid.sid_lower_bound}, {sid.sid_upper_bound}]")

# Markov-blanket-scoped comparison
mb_true = bnm.markov_blanket(truth, "B")
mb_est  = bnm.markov_blanket(estimate, "B")
print("MB(B) F1:", bnm.f1(mb_true, mb_est))
```

## Documentation

- [v0.1.x → v0.2 audit](docs/audit.md)
- [Design doc (Protocols, signatures, decisions)](docs/design/api_v0.py)
- [Per-package journal](docs/journal.md)
- [CHANGELOG](CHANGELOG.md)

## Acknowledgements

`bnm` v0.1.0 was a port of the [DAGMetrics R
package](https://github.com/averinpa/DAGMetrics) by the same author.
v0.2.x is a full Python rewrite around an int8 endpoint-mark matrix
matching `cbcd`'s convention; the metric definitions remain
derivative of the R original. The audit at
[`docs/audit.md`](docs/audit.md) catalogues the eight bugs found in
the 0.1.x implementation; v0.2 fixes all of them.

## License

[MIT License](LICENSE).

## Author

Pavel Averin. GitHub: [@averinpa](https://github.com/averinpa).
