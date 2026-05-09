# Changelog

## 0.2.2 (in development)

Viz polish. No metric-layer changes; suite parity numerically
unchanged (5/5 fixtures still within bounds).

### Changed (breaking — pre-release only)

- **`bnm.plot_side_by_side`**: the `highlight_true_positives: bool`
  kwarg is replaced by `mode: Literal["matches", "diff", "none"] =
  "matches"`. Migration:

  | v0.2.0/0.2.1 | v0.2.2 |
  |---|---|
  | `highlight_true_positives=True` (default) | `mode="matches"` (default) |
  | `highlight_true_positives=False` | `mode="none"` |
  | *(no equivalent)* | `mode="diff"` |

  Since v0.2.x is still pre-release (no PyPI publication yet), this
  is a clean rename rather than a deprecation alias.

### Added

- **`bnm.plot_side_by_side(..., mode="diff")`** — highlights edges
  that *differ* between g1 and g2 (additions, deletions, reversals,
  kind changes). Each side's edge is highlighted in whichever panel
  contains it. Useful for "show me what changed" diffs.
- **`highlight_node_color` / `highlight_edge_color` kwargs** on both
  `plot_graph` and `plot_side_by_side`. Defaults preserve the v0.2
  pastel palette (`#c8e6c9` node fill, `#f08080` edge stroke); pass
  any graphviz-accepted colour string to override per-call.

### Fixed

- **CIRCLE-edge matching granularity** in `plot_side_by_side(mode=
  "matches")`. Pre-fix, `_matching_edges` collapsed every CIRCLE-
  bearing edge into a single `"circle"` bucket, so a `(CIRCLE,
  ARROW)` edge in g1 and a `(CIRCLE, CIRCLE)` edge in g2 — different
  PAG topologies — were incorrectly reported as matching. Matching
  now compares the full `(mij, mji)` mark pair for CIRCLE edges.
- Resolves design-doc open questions **O1** and **O2**.

### Tests

- 12 new tests in `tests/viz/test_viz_v0_2_2.py` covering the colour
  kwargs, CIRCLE matching granularity, and `mode="matches" / "diff"
  / "none" / invalid`.

## 0.2.1 (in development)

Pure performance follow-up to 0.2.0. No public API changes.

### Changed

- **`bnm.compare`** now normalises `g1` / `g2` to internal `_Graph`
  instances exactly once. Downstream metric calls reuse the
  normalised inputs and skip the per-call O(n²) endpoint validation.
  In v0.2.0 the per-node loop revalidated each external graph on
  every metric and every variable, so `compare(per_node=True)` on a
  1000-node external GraphLike took minutes; the demo notebook
  `evaluate single DAG.ipynb` was downsized from `n_nodes=1000` to
  `n_nodes=200` to compensate. This release restores the original
  scale: `compare(per_node=True)` at n=200 now completes in
  hundreds of milliseconds; n=1000 is back in the seconds range.
- **`bnm.adapter._validate_endpoints`** is now fully vectorised over
  the `(n, n)` matrix (no Python `for i, j` loop). Single-shot
  metric calls — `bnm.shd(g1, g2)`, `bnm.sid(g1, g2)`, etc. — pick
  up a free speedup on every external GraphLike input.

### Tests

- 4 new regression tests in `tests/compare/test_compare_perf.py`:
  - `_validate_endpoints` runs at most once per distinct external
    GraphLike across a full `compare(per_node=True)` call (= 1 in
    single-graph mode, 2 in two-graph mode).
  - Wall-clock bound: `compare(per_node=True)` on a 200-node
    external GraphLike completes in under one second.
  - Wall-clock bound: `_validate_endpoints` on a 1000×1000 valid
    matrix completes in under half a second.

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
| `BNMetrics(g1, g2).compare_df(...)` | `bnm.compare(g1, g2, ...)` + `bnm.to_dataframe(c)` |
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
