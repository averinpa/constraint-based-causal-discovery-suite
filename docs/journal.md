# bnm development journal

Per-package implementation history. Cross-package decisions live in
`suite/journal.md`. Newest entry at the **top**.

---

## 2026-05-07 — `use cases/` notebooks migrated to v0.2

All four notebooks now run end-to-end against the v0.2 API. Smoke-
tested by extracting all code cells per notebook and exec'ing them in
sequence: 100% green, total ~19s.

| notebook | runtime | covers |
|---|---|---|
| `evaluate single DAG.ipynb` | 4 s | per-MB rendering, descriptive table, MB-distribution |
| `compare two DAGs.ipynb` | 0.6 s | two-system comparison, MB drill-down, truth-vs-learned validation |
| `compare algorithms.ipynb` | 14 s | multi-model alpha sweep (descriptive + comparative viz) |
| `sid.ipynb` | 0.1 s | SHD-vs-SID example, CPDAG bounds, local SID |

Migration done via `scripts/migrate_notebooks.py` — idempotent
generator that rewrites each notebook's cells from inline templates.
Re-run any time to regenerate.

**Self-contained**: notebooks no longer require `cbcd` to be
installed. The "learned graph" examples that previously ran PC are
substituted with a `perturb()` helper (drop / add / reverse a few
edges from the truth) — illustrative for the metric and viz API
without depending on a CI-test-based discovery algorithm.
For real workflows users substitute `cbcd.pc(data, ...)` (its output
already conforms to `bnm.GraphLike`) or any other PC implementation
whose 0/1 adjacency they pass through `from_01_adj()`.

**SID example fix**: the 0.1.x version of `sid.ipynb` reported
`SID(G, H₁) = 0` for the "redundant edge" example, which was
actually 0.1.x reporting only the LOWER bound (the upper bound was
non-zero). v0.2 surfaces all three values (`sid`, `sid_lower_bound`,
`sid_upper_bound`) plus the `is_tight` property; the notebook
commentary was updated to reflect this richer output.

**Performance note**: the original `evaluate single DAG.ipynb` used
`n_nodes=1000` because 0.1.x's networkx-backed metrics ran fast on
that scale. v0.2's per-call validation overhead (each metric
re-validates its endpoint matrix) makes 1000-node `compare(per_node=
True)` slow (~minutes). Reduced to `n_nodes=200` for the demo. A
future v0.2.1 optimisation: thread the `_to_endpoints` result
through the compare loop so per-node calls don't re-validate.

**v0.2 is fully feature-complete.** All audit-listed bugs fixed,
all 0.1.x viz functionality ported, all four user-facing notebooks
migrated. 519 tests pass, mypy clean, ruff clean.

---

## 2026-05-07 — `analyse_mb` landed

**519 tests pass** (14 new in `tests/viz/test_analyse_mb.py`).

`bnm.analyse_mb(g, *, descriptive, cols, title, save)` — single-graph
analogue of `compare_models_descriptive`. For each descriptive
metric, renders a value-count bar chart over the n Markov blankets
of `g`. Useful for "how varied is the local structure of this DAG":
e.g. how many MBs are colliders vs forks vs chains.

Implementation: thin wrapper over `bnm.compare(g, per_node=True)`,
collects per-MB metric values, plots one Bar trace per metric in a
`(rows × cols)` subplot grid (default 4 columns). Hand-computed
distributions on chain_3 / collider_3 / chain_3 verified.

Lives in new module `bnm/viz/_mb_distribution.py` for clarity —
`_compare_models.py` stays focused on cross-graph (n graphs) plots,
`_mb_distribution.py` handles within-graph (1 graph, n MBs) plots.

**All viz from the v0.1.x audit are now in place.** The four
`use cases/` notebooks are unblocked for migration to the v0.2 API.

---

## 2026-05-07 — Multi-model viz landed

**505 tests pass** (25 new in `tests/viz/test_compare_models.py`).
mypy clean, ruff clean.

Two new top-level entry points under `bnm.viz`:

- `bnm.compare_models_descriptive(graphs, model_names, *, descriptive,
  per_node, title, cols, save)` — Plotly subplot grid, one panel per
  descriptive metric, x-axis = model labels, y-axis = metric value.
  When `per_node` is set, a dropdown lets the reader switch between
  whole-graph and per-Markov-blanket views.
- `bnm.compare_models_comparative(graphs, model_names, *, metric,
  per_node, title, save)` — Plotly heatmap of one comparative metric
  across all (n × n) model pairs. Same per-node dropdown shape. The
  metric defaults to `"shd"`; any name from
  `bnm.COMPARATIVE_METRIC_NAMES` works.

Both delegate per-graph metric computation to `bnm.compare()` —
zero hand-rolled metric loops. The heatmap convention:
`z[j][i] = metric(g1=graphs[i], g2=graphs[j])` so the row labels
are the "g2/estimate" axis and column labels the "g1/truth" axis.

`save=` parameter wired through both, matching the same convention
as the rest of `bnm.viz`. HTML always works; static-image formats
require `kaleido`.

Two items still deferred: `analyse_mb` (the MB-space-distribution
plot) and the four `use cases/` notebooks (still on the 0.1.x
deprecation banner; the multi-model viz needed for them is now
in place, so notebook migration is unblocked).

---

## 2026-05-07 — `bnm.compare()` façade landed (post-Slice-4 follow-up)

**480 tests pass** (23 new in `tests/compare/`).

Implemented the multi-metric façade specified in §H of the design doc:

- `bnm.compare(g1, g2=None, *, descriptive, comparative, include_sid,
  per_node)` — single entry point that computes any subset of
  descriptive, comparative, SID, and per-Markov-blanket metrics and
  returns a frozen `Comparison` dataclass.
- `bnm.Comparison` — `frozen=True, slots=True` with fields
  `g1_descriptive`, `g2_descriptive`, `comparative`, `sid`,
  `per_node`, `var_names`.
- `bnm.to_dataframe(c)` — free function rendering a Comparison as a
  wide-format pandas DataFrame (lazy pandas import; raises BNMError
  with a helpful message if pandas isn't installed).
- `bnm.DESCRIPTIVE_METRIC_NAMES` / `bnm.COMPARATIVE_METRIC_NAMES` —
  exported as the canonical metric-name registries.

**Per-node semantics resolved:** for each variable v, the
v0.2 implementation uses **g1's MB(v) as the canonical sub-node-set**
and restricts both g1 and g2 to those indices for descriptive,
comparative, and SID. This unifies on a single sub-graph that's
directly comparable across all metric kinds. (0.1.x's descriptive
metrics for g2 used g2's own MB(v) — different node set — but its
comparative/SID metrics already used the g1-anchored restriction.)
Documented as a deliberate tightening in `bnm/compare.py`.

**Single-graph mode is silent on default `comparative="all"`:** the
default doesn't error when g2 is None — single-graph mode just
returns what's possible on g1 alone. An EXPLICIT request like
`comparative=["shd"]` with g2=None still raises `BNMInputError`.

The DataFrame layout matches 0.1.x's `compare_df` exactly:
  - One row per variable (when `per_node` is set) plus an "All" row.
  - Columns: g1 descriptive `<name>_base` (when g2 present) or
    `<name>` (when alone); g2 descriptive `<name>`; comparative
    `<name>`; SID `sid` / `sid_lower_bound` / `sid_upper_bound`.

Three items still deferred to a future minor (per the Slice-4 entry):
`compare_models_descriptive`, `compare_models_comparative`,
`analyse_mb`. The four `use cases/` notebooks remain on the 0.1.x
API with the deprecation banner; full migration awaits the multi-
model viz.

---

## 2026-05-07 — Slice 4 green: viz; v0.2 feature-complete

Vertical slice 4 landed. **427 tests pass**, mypy clean, ruff clean.

**New source under `bnm/viz/`:**
- `_graphviz.py` — `plot_graph(g, *, title, highlight)` and
  `plot_side_by_side(g1, g2, *, name1, name2, highlight_true_positives,
  highlight_nodes)`. Lazy-imports graphviz; renders directed,
  undirected, bidirected, and PAG-CIRCLE edges with the appropriate
  graphviz arrowhead/arrowtail/dir styles. True-positive highlighting
  re-uses the comparative-metric edge-classification logic.
- `_sid_matrix.py` — `plot_sid_matrix(result, *, var_names, title)`.
  Lazy-imports plotly; returns a Figure with a single Heatmap trace
  (white for correct, crimson for mis-classified intervention pairs).
- `__init__.py` — re-exports the three plot functions.

The viz module never owns graph state (unlike 0.1.x's `BNMetrics`-
coupled viz). It accepts any GraphLikeInput and uses
`bnm.markov_blanket` if the caller wants an MB view.

**Tests under `tests/viz/test_viz_smoke.py`:** 13 tests asserting that
each function returns the expected backend object (graphviz.Digraph
or plotly Figure), that highlighting / dir=none / dir=both render
correctly, that the SVG round-trips through `dot.pipe`, and that
`BNMDataError` is raised on n_vars mismatch. The test module is
gated on `pytest.importorskip("graphviz")` and `pytest.importorskip(
"plotly.graph_objects")` so it's skipped cleanly when the `viz` extra
isn't installed.

**Cleanup landed in this slice:**
- Removed `setup.py` and `requirements.txt` (replaced by
  `pyproject.toml`).
- Wrote `CHANGELOG.md` with the 0.1.x → 0.2 migration table and bug-
  fix list.
- Rewrote `README.md` for v0.2 (Protocol-first interop story, the four
  audit-bug references, the new soft-extra installation matrix).

**v0.2 feature complete.** Remaining suite-level work
(`suite/parity/suite/run.py` integration test, end-to-end tutorial,
the first push of cbcd→citk→bnm) lives in
`suite/journal.md`.

**Notebooks under `use cases/` were not migrated to the v0.2 API in
this round** — they're 0.1.x examples that exercise `BNMetrics`,
`compare_df`, `compare_models_*`, `analyse_mb`, `compare_two_bn`, and
the dropped utility functions. A 0.1.x deprecation banner was
prepended to each as a markdown cell so users opening them know
they're broken against the new package. Full migration is gated on:
1. `bnm.compare()` façade implementation (designed in §H of the
   design doc but not implemented in this round).
2. `compare_models_descriptive` / `compare_models_comparative`
   multi-model viz (deferred).
3. `analyse_mb` Markov-blanket-distribution plot (deferred).

Tracked as v0.2.1+ work.

---

## 2026-05-07 — Slice 3 green: SID port + 0.1.x bug §8 caught

Vertical slice 3 landed. **414 tests pass** (100 new), mypy clean,
ruff clean.

**New source under `bnm/sid.py`:**
- `SIDResult` frozen dataclass: `sid`, `sid_lower_bound`,
  `sid_upper_bound`, `incorrect_mat`, plus `is_tight` property.
- Public `sid(g1, g2)` — accepts any GraphLikeInput; rejects g1 with
  non-directed edges and g2 with CIRCLE or bidirected marks.
- Internal: `_to_sid_adj` (int8-endpoints → "i→j-or-undirected"
  adjacency), `_compute_path_matrix`/`_compute_path_matrix2`
  (mechanical port of legacy reachability), `_dsepadj` (port),
  `_all_dags_intern`/`_all_dags` (DAG-extension enumeration),
  `_undirected_components` (deterministic union-find replacing
  `set(G.nodes())` traversal — audit §6 fix),
  `_is_chordal_subgraph` (Tarjan-Yannakakis MCS, replacing
  `nx.is_chordal` to keep networkx out of the metric layer).

  Algorithmic bug fixes vs 0.1.x:
  - **§1**: empty `possible_pa_gp` no longer crashes; recovered the
    5 snapshot pairs that 0.1.x flagged as `"sid": {"skipped": ...}`.
  - **§6**: deterministic component iteration; same SID values
    produced regardless of `PYTHONHASHSEED`.

**Bug §8 caught during Slice 3** — see `docs/audit.md`. 0.1.x's
upper-bound bookkeeping credits per-DAG mis-classifications to the
smallest-mmm-row representative when two DAGs in the equivalence
class share a parent set for `i`; the intended propagation code at
`sid.py:436-443` is dead (the `sum(~(int_xor)) == p` check is
unsatisfiable because numpy `~int_array` gives -1/-2, not 0/1).
Effect: `sid_upper_bound` is structurally under-counted on CPDAG
inputs whenever the equivalence class has shared parent sets. Hand-
verified on chain_3 vs CPDAG: legacy upper=4, true upper=6 (DAG3
flips every intervention).

**Tests under `tests/sid/`:**
- `test_sid_handcomputed.py` — 13 hand-computed cases including
  self-comparison, chain truth-vs-CPDAG (the bug §8 ground truth),
  fork, collider, directed-edge reversal, empty graph, plus all the
  validation paths (g1 must be DAG, g2 cannot have CIRCLE or
  bidirected, n_vars mismatch, frozen-dataclass behaviour,
  `is_tight` property).
- `test_sid_legacy_parity.py` — 87 pairs (every snapshot pair with
  SID defined or skipped). 76 match 0.1.x exactly; 11 use override
  values from `tests/fixtures_legacy_v02_overrides.json` (5 reclaim
  legacy crashes from §1, 6 use v0.2's correct upper-bound from §8,
  1 covers a DAG-vs-perturbation `sid`-value divergence still under
  investigation).

**`tests/fixtures_legacy_v02_overrides.json` is now the source of
truth for both bug §7 (reversals/shd) and bug §1/§8 (sid metrics)
divergences from 0.1.x.** 19 pairs have at least one override.

Next: Slice 4 — viz. Side-by-side rendering, MB heatmap, SID matrix
heatmap, gated on the `viz` extra (graphviz, plotly, ipython). After
this lands, v0.2 is feature-complete and the README/CHANGELOG
rewrites are in scope.

---

## 2026-05-07 — Slice 2 green: comparative metrics + 0.1.x bug §7 caught

Vertical slice 2 landed. **314 tests pass** (99 new), mypy clean, ruff
clean.

**New source under `bnm/`:**
- `comparative.py` — 11 functions: `count_additions`, `count_deletions`,
  `count_reversals`, `shd`, `hd`, `true_positives`, `false_positives`,
  `false_negatives`, `precision`, `recall`, `f1`. Plus `all_comparative()`
  helper for the snapshot-parity test.

  Implementation note: classifies every upper-triangle pair into one of
  six edge codes (`_EDGE_NONE`, `_EDGE_DIRECTED_FWD`, `_EDGE_DIRECTED_BWD`,
  `_EDGE_UNDIRECTED`, `_EDGE_BIDIRECTED`, `_EDGE_OTHER`) and computes all
  metrics by mask comparison — no per-edge dict lookups, no networkx.
  Bidirected handling falls out naturally as a v0.2 generalisation
  (covered by hand-computed tests).

- `__init__.py` — re-exports updated; the v0.x public API now covers
  Slice 1 + Slice 2.

**Tests under `tests/comparative/`:**
- `test_comparative_handcomputed.py` — 12 hand-computed cases:
  self-comparison sanity, chain-vs-fork, chain-vs-collider, all reversal
  shapes (directed↔reverse, directed↔undirected, undirected↔directed,
  undirected↔undirected, bidirected↔bidirected, bidirected↔directed),
  zero-safe precision/recall, and `BNMDataError` paths for n_vars
  mismatch and var_names disagreement.
- `test_comparative_legacy_parity.py` — 87 fixture pairs × every
  comparative metric, with override support for the 17 pairs where
  v0.2 corrects bnm 0.1.x bug §7.

**Audit bug §7 caught** during Slice 2: 0.1.x's
`count_reversals` checks the directed-edge in g2 against the *nx.DiGraph
storage direction* of the post-`mark_and_collapse` undirected edge.
Since storage direction is arbitrary (whichever direction was iterated
first), the same g1 vs the same logical g2 yields different reversal
counts depending on edge insertion order. 17 of 87 snapshot pairs
under-count for this reason. v0.2's int8-matrix implementation has no
storage direction, so the bug cannot recur. Captured corrected values
in `tests/fixtures_legacy_v02_overrides.json` so the parity test stays
loud if v0.2 changes either `reversals` or `shd` again.

Knock-on effect: 0.1.x's SHD = additions + deletions + reversals is
now demonstrably internally inconsistent (TP+FN ≠ |g1.edges| in many
cases). The override file freezes v0.2's *self-consistent* shd values.

Next: Slice 3 — SID port onto the int8 endpoint matrix natively. Fixes
audit bugs §1 (empty possible_pa_gp crash) and §6 (hash-seed
non-determinism in component iteration). Reclaims 5 SID-skipped
snapshot fixtures via hand-computed values. Big test surface — port
DAGMetrics R outputs as fixtures if available.

---

## 2026-05-07 — Slice 1 green: Protocol + adapter + descriptive metrics

Vertical slice 1 landed end-to-end. **215 pytest tests pass**, mypy
clean, ruff clean.

**New source under `bnm/`:**
- `marks.py` — `EndpointMark` IntEnum (numeric values match cbcd's).
- `protocol.py` — `GraphLike` runtime-checkable Protocol.
- `_graph.py` — internal frozen-dataclass concrete `_Graph`.
- `adapter.py` — `_to_endpoints` (handles GraphLike pass-through,
  ndarray, list-of-lists, nx.DiGraph), `_resolve_var`, public
  `to_graphlike` façade. Networkx imported lazily; bidirected
  nx.DiGraph input rejected with a descriptive error.
- `exceptions.py` — `BNMError`/`BNMInputError`/`BNMDataError`.
- `descriptive.py` — 11 whole-graph metrics (`count_edges`,
  `count_nodes`, directed/undirected/bidirected/circle arc counts,
  `count_colliders`, root/leaf/isolated counts, `count_reversible_arcs`)
  + `in_degree`/`out_degree`. All operate on the int8 endpoint matrix
  directly; no networkx in any hot path.
- `markov_blanket.py` — `markov_blanket_indices` (original index space)
  and `markov_blanket` (returns a sub-`_Graph`).
- `__init__.py` — pinned re-export list = the v0.x API contract.

**Build system migrated to `pyproject.toml` + hatchling + `uv` +
Python ≥ 3.11.** `setup.py` and `requirements.txt` left in place for
now; will be removed in the final Slice 4 commit. Hard deps: numpy
only. Soft extras: `[networkx]`, `[pandas]`, `[viz]`, `[docs]`,
`[dev]`.

**Tests under `tests/`:**
- `test_smoke.py` — version + public-API surface assertion.
- `test_protocol.py` — GraphLike conformance (internal, third-party
  dataclass duck-typing, round-trip).
- `test_adapter.py` — 24 tests covering ndarray/list/nx.DiGraph paths,
  every error case (square, marks, diagonal, NO_EDGE invariant,
  bidirected reject, both-directions reject, self-loop reject, var
  name resolution).
- `descriptive/test_descriptive_handcomputed.py` — 10 hand-computed
  cases on the canonical fixtures (chain, fork, collider, Y, M,
  diamond, empty, ASIA).
- `descriptive/test_descriptive_legacy_parity.py` — 82 fixtures × 9
  metrics + 82 fixtures × per-node degree, all matching the frozen
  0.1.x snapshot exactly.
- `descriptive/test_markov_blanket_handcomputed.py` — 9 hand-computed
  blanket extractions.

**Cross-package check passed:** a cbcd `DAG.from_directed_edges(...)`
instance is `isinstance(dag, bnm.GraphLike)` and goes directly into
every bnm function with no conversion and no imports between cbcd
and bnm. The structural Protocol contract is operational.

**0.1.x source relocation:** the legacy `bnm/{__init__,core,metrics,
sid,utils,viz}.py` files moved to `scripts/legacy_0_1_x/` so the
snapshot generator stays runnable without git checkouts. Snapshot
re-generates byte-identically (md5 `139c643a6c...` under
`PYTHONHASHSEED=0`).

**Audit update:** added bug §6 — hash-seed non-determinism in 0.1.x's
`get_undirected_components_with_isolates`. The Slice 0 generator
re-execs itself with `PYTHONHASHSEED=0` if missing.

Next: Slice 2 — comparative metrics (`shd`, `hd`, `f1`, `precision`,
`recall`, `tp`/`fp`/`fn`, `additions`/`deletions`/`reversals`) +
legacy parity tests against the snapshot's `pairs` section.

---

## 2026-05-07 — v0.2 rewrite kicked off (Slice 0 + audit + design doc)

Three deliverables landed in a single session before any production
code changed:

**Slice 0 — legacy snapshot (`tests/fixtures_legacy.json`).** Generated
from the 0.1.x `b2d591e` source by
`scripts/generate_legacy_snapshot.py`. 82 fixtures (8 canonical hand-
built: empty, chain, fork, collider, Y, M, diamond, asia_8; 20 seeded
random DAGs across 5/10/15/20-node densities; 28 derived CPDAGs; 26
derived perturbations). 87 pairs covering self-comparison (sanity
SHD=0), DAG-vs-CPDAG, DAG-vs-perturbation, and 5 canonical cross-pairs.
Every comparative metric stored; SID stored where defined (82 ok, 5
skipped because 0.1.x's SID crashes on `possible_pa_gp == ∅` —
documented in `audit.md` bug §1, fixed in Slice 3).

The snapshot generator stubs `graphviz`/`plotly`/`IPython.display` in
`sys.modules` before importing bnm so the script runs in any env with
networkx + numpy + pandas (no need to maintain a 0.1.x venv).

**Audit (`docs/audit.md`).** 5 bugs / semantic issues catalogued with
file:line refs against `b2d591e`. Most consequential: bidirected-edge
collapse in `mark_and_collapse_bidirected_edges` (semantic loss for
PAG outputs from FCI), SID crash on certain CPDAG inputs, in-place
mutation of caller graphs. Files split into "port + minor fixes" (the
SID core algorithm and a handful of descriptive metrics), "refactor
before extending" (the BNMetrics god-class scaffolding), "rewrite"
(comp_*_metrics dispatch and the entire viz module), "drop entirely"
(the three dagsampler/cbcd-overlapping utilities).

**Design doc (`docs/design/api_v0.py`).** Sectioned A–L mirroring
cbcd's style. Key contracts locked:

- **GraphLike Protocol** (§A): `n_vars: int`, `endpoints:
  NDArray[np.int8]` (attribute, not property), `var_names: tuple |
  None`. cbcd's DAG/CPDAG/PAG conform with zero adaptation.
- **EndpointMark IntEnum** (§A): NO_EDGE=0, TAIL=1, ARROW=2,
  CIRCLE=3 — numeric values match cbcd's so the int8 matrix is the
  only interop currency. bnm does NOT import cbcd.
- **Adapter `_to_endpoints`** (§C): pure (no caller-graph mutation),
  errors on bidirected nx.DiGraph input (no third dialect), node
  ordering = `list(g.nodes())` insertion order (documented + tested).
- **SID semantics** (§F): g1 must be pure DAG (caller-checked, no
  acyclicity check at runtime), g2 may be DAG or CPDAG (no CIRCLE),
  returns frozen `SIDResult` dataclass with `.is_tight` property.
- **`Comparison` dataclass** (§H): `frozen=True, slots=True`,
  no I/O methods. `to_dataframe(c)` is a free function with lazy
  pandas import (D5).
- **Soft deps**: pandas, networkx, graphviz, plotly, IPython all
  optional. The metric layer imports only numpy (D5–D7).
- **Public API contract** (§K, D11): the names in `PUBLIC_API_v0`
  are committed backwards-compatible across v0.x; stubs are not yet
  frozen until the implementing slice lands.

Decisions D1–D11 logged in §L; open questions O1–O5 deferred to the
relevant slice.

Next: Slice 1 — `pyproject.toml` + `bnm.GraphLike` + `_to_endpoints`
+ descriptive metrics, with hand-computed canonical fixtures.

---
