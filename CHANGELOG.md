# Changelog

## 0.2.0 (in development)

**Breaking rewrite.** v0.2 is a full rewrite around a canonical int8
endpoint-mark matrix matching cbcd's convention; the 0.1.x
`networkx.DiGraph + edge type attr` representation is gone. Migration
notes below.

### Migration

| 0.1.x | 0.2 |
|---|---|
| `BNMetrics(g1, g2).shd` | `bnm.shd(g1, g2)` |
| `BNMetrics(g1, g2).f1_score` | `bnm.f1(g1, g2)` |
| `BNMetrics(g1, g2).sid()` | `bnm.sid(g1, g2)` (returns `SIDResult` dataclass) |
| `BNMetrics(g1, g2).compare_df(...)` | `bnm.compare(g1, g2, ...)` (planned) |
| `bnm.generate_random_dag(...)` | use `dagsampler.CausalDataGenerator` |
| `bnm.generate_synthetic_data_from_dag(...)` | use `dagsampler.CausalDataGenerator` |
| `bnm.utils.dag_to_cpdag(...)` | use cbcd's `DAG.to_cpdag()` |
| `bnm.utils.mark_and_collapse_bidirected_edges` | replaced by adapter; bidirected `nx.DiGraph` input now raises (no silent collapse) |

### Added

- **`bnm.GraphLike` Protocol** — structural duck-typed contract over
  `(n_vars, endpoints, var_names)`. cbcd's `DAG`/`CPDAG`/`PAG` conform
  with zero adaptation; no imports between bnm and cbcd.
- **`bnm.EndpointMark`** — `NO_EDGE`, `TAIL`, `ARROW`, `CIRCLE`. Numeric
  values match cbcd's so the int8 matrix is the only interop currency.
- **`bnm.to_graphlike(obj)`** — adapter from cbcd graphs / nx.DiGraph /
  ndarray / list-of-lists to a normalised `GraphLike` instance.
- **`bnm.SIDResult`** — frozen dataclass with `sid`,
  `sid_lower_bound`, `sid_upper_bound`, `incorrect_mat`, plus an
  `is_tight` property.
- **`bnm.markov_blanket(g, var)`** — returns a sub-`GraphLike` over
  the blanket of `var`; usable directly with any other bnm function.
- **`bnm.compare(g1, g2=None, ...)` + `Comparison` dataclass** —
  multi-metric façade. Computes any subset of descriptive /
  comparative / SID / per-Markov-blanket metrics in one call and
  returns a frozen dataclass. `bnm.to_dataframe(c)` renders it as a
  wide-format pandas DataFrame (lazy pandas import).
- **`bnm.DESCRIPTIVE_METRIC_NAMES` / `COMPARATIVE_METRIC_NAMES`** —
  the canonical metric-name registries, also accepted as the iterable
  argument to `compare()`.
- **`bnm.compare_models_descriptive(graphs, model_names, ...)`** —
  Plotly subplot grid, one panel per descriptive metric, x-axis =
  model labels. With ``per_node=True`` exposes a node-selection
  dropdown.
- **`bnm.compare_models_comparative(graphs, model_names, *, metric,
  ...)`** — Plotly heatmap of one comparative metric across all pairs
  of models. With ``per_node=True`` exposes a node-selection dropdown.
- **`bnm.analyse_mb(g, ...)`** — single-graph distribution view. For
  each descriptive metric, plots a value-count bar chart over the
  graph's `n` Markov blankets. Useful for characterising local-
  structure heterogeneity within one DAG/CPDAG.
- **`use cases/` notebooks migrated to the v0.2 API.** All four
  notebooks (`evaluate single DAG`, `compare two DAGs`,
  `compare algorithms`, `sid`) rewritten end-to-end against the new
  surface. Self-contained: no hard dependency on `cbcd` or other PC
  implementations (the "learned graph" examples use a built-in
  `perturb()` helper as a stand-in; users substitute `cbcd.pc(...)`
  for real workflows).
- All viz functions accept ``save=path`` (format inferred from
  extension; HTML always works, static formats need ``kaleido``).
- **PAG support (partial)** — bidirected edges are first-class via
  `EndpointMark.ARROW/ARROW`; `count_bidirected_arcs` and
  `count_circle_edges` are new. Several metrics generalise to PAG
  inputs; SID still requires DAG/CPDAG.
- **Build system** — `pyproject.toml` + `hatchling` + `uv` (Python ≥
  3.11). Soft extras: `[networkx]`, `[pandas]`, `[viz]`, `[docs]`,
  `[dev]`. The metric layer hard-depends only on numpy.
- **Tests** — 427 pytest tests, mypy strict, ruff. Hand-computed
  canonical fixtures for every metric, plus snapshot parity against a
  frozen `tests/fixtures_legacy.json` (82 fixtures, 87 pairs)
  generated from 0.1.x.
- **Documentation** — `docs/audit.md` (8 bugs catalogued in 0.1.x),
  `docs/design/api_v0.py` (sectioned design doc), `docs/journal.md`
  (per-package implementation history).

### Fixed

- **Audit §1**: SID no longer crashes on
  `np.meshgrid(*[[0,1]] * 0)` when the estimated CPDAG has a node with
  zero possible-parent candidates.
- **Audit §6**: SID is reproducible across `PYTHONHASHSEED` values —
  no more `set(G.nodes())` in any path that affects output.
- **Audit §7**: `count_reversals` no longer under-counts when an
  undirected `g2` edge is stored in nx.DiGraph in the opposite
  direction from `g1`'s directed edge.
- **Audit §8**: `sid_upper_bound` now reflects the true maximum SID
  over the equivalence class rather than the maximum
  attributable-to-smallest-mmm-row count.

### Deferred to v0.2.1+

*(none for v0.2.0; all in-scope work is complete.)*

### Removed

- The `BNMetrics` god-class.
- `bnm.utils.generate_random_dag`, `generate_synthetic_data_from_dag`
  (overlap with `dagsampler`).
- `bnm.utils.dag_to_cpdag` (overlap with `cbcd.DAG.to_cpdag()`).
- `bnm.utils.mark_and_collapse_bidirected_edges` (replaced by adapter
  with stricter semantics).
- `setup.py` and `requirements.txt` (replaced by `pyproject.toml`).

## 0.1.0 (2024)

Initial release of bnm as a port of the R package
[DAGMetrics](https://github.com/averinpa/DAGMetrics).
