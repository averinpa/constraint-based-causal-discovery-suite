# Audit of `bnm` 0.1.x

**Date:** 2026-05-07
**Source tree audited:** `bnm/` at git
`b2d591e1fb22550d94b82391c7304077a7a3ec07` (branch `main`).
**Scope:** all of `bnm/bnm/` (`__init__.py`, `core.py`, `metrics.py`,
`sid.py`, `utils.py`, `viz.py`) plus packaging/setup.py.

This audit informs `bnm`'s clean-room rewrite to `v0.2`. References are
`file:line` against the audited tree.

## Verdict

**Functional research code, not yet a production base.** The metric math
is mostly correct on directed/undirected DAGs and CPDAGs, the SID port
of Jonas Peters' R package is the load-bearing original contribution,
and the per-MB visualization is a real differentiator. But the package
has no tests, no Protocol contract, no design doc, three duplicated
hand-rolled graph utilities that overlap with sister packages
(`dagsampler`, `cbcd`), one input format that quietly conflates
bidirected and undirected edges, and at least one SID crash on
otherwise-valid CPDAG inputs. The structure does not scale to the
suite's correctness bar (SHD = 0 vs oracle on every algorithm) without
a rewrite.

## Files to keep mostly as-is (port + minor fixes)

- **`sid.py:8-17` (`compute_path_matrix`)** — repeated-squaring
  reachability matrix, `O(n² · log n)`. Correct, clean. Port to
  v0.2 unchanged but rebound to int8 endpoint matrix input.
- **`sid.py:19-47` (`compute_path_matrix2`)** — same shape with cond-set
  edge removal. Port unchanged.
- **`sid.py:124-274` (`dsepadj`)** — d-separation reachability oracle
  used inside SID. Algorithmically faithful to the R reference. Port
  unchanged but with integer-index inputs (no name lookups).
- **`metrics.py:26-67` (`count_colliders`)** — correct, treats
  `type='undirected'` parents as non-collider parents. Port; rebind to
  `endpoints[i, j] == ARROW` rather than dict-attr lookups.
- **`metrics.py:258-301` (`count_reversible_arcs`)** — the only
  non-trivial descriptive metric, computes "directed arcs not in any
  collider". Correct. Port to v0.2 with the same semantics.

## Refactor before extending

- **`core.py:14-200` (`BNMetrics` class)** — god-class; init carries 6
  branches over `(G1, G2)` input combinations, builds an internal
  `graph_dict` mapping every node to a triple of MB subgraphs (`d1`,
  `d2`, `d3`), then dispatches `compile_descriptive_metrics`,
  `compile_comparison_metrics`, and viz from the same surface. **v0.2
  drops this entirely** in favour of pure functions + a thin
  `compare()` function returning a `Comparison` dataclass (per-plan
  decision).
- **`core.py:202-233` (`_merge_graphs_no_duplicates_clean`)** — manual
  dedup over edges using `(u, v, edge_type)` triples. Reimplemented in
  v0.2 as endpoint-matrix union with no per-edge dict lookup.
- **`utils.py:6-34` (`mark_and_collapse_bidirected_edges`)** —
  destructively normalises a `nx.DiGraph` so every edge has a `type`
  attr ∈ {`directed`, `undirected`}. **v0.2 replaces this with
  `_to_endpoints` (pure, returns int8 matrix) that errors on
  bidirected input from `nx.DiGraph`**. Bidirected callers route
  through cbcd's `PAG` (which conforms to `GraphLike`) or pass a raw
  int8 matrix.
- **`utils.py:36-58` (`graph_to_matrix`)** — same job as
  `_to_endpoints` but produces a {0, 1} adjacency rather than the
  `EndpointMark`-coded int8 matrix v0.2 needs. Drop; the v0.2 adapter
  subsumes it.
- **`utils.py:60-69` (`get_undirected_components_with_isolates`)** —
  small but correct. Reimplement on the int8 matrix using
  `np.unique`/union-find rather than constructing a temporary
  `nx.Graph`.
- **`utils.py:71-126` (`get_markov_blanket_subgraph`)** — correct
  semantics (parents + children + co-parents-of-children + undirected
  neighbours + self). v0.2's `bnm.markov_blanket(g, var)` returns a
  sub-`GraphLike` over the int8 matrix using the same set definition.

## Rewrite (do not carry forward)

- **`core.py:235-300` (`compile_descriptive_metrics`)** and
  **`core.py:302-360` (`compile_comparison_metrics`)** — function
  dispatch via two dicts of names → callables, then `df.loc[df["..."]
  == node, name] = func(...)` row-by-row pandas updates inside double
  loops. Replace with a single `compare()` function that builds a
  `Comparison` dataclass; the dataframe view is a free function
  `bnm.to_dataframe(c)` (lazy pandas import).
- **`core.py:481-559` (`sid` method)** and
  **`core.py:631-676` (`_mark_true_positives_color_both`)** — graph
  merging + name-based traversal logic that breaks down once we move
  to integer-indexed int8 matrices. Drop the methods; the underlying
  `sid_metric` algorithm in `sid.py` is what survives.
- **`core.py:703-906` (Graphviz/IPython rendering)** — reimplement in
  Slice 4 against the new `GraphLike` shape. Drop the `BNMetrics`
  ownership of viz; Slice 4's `bnm/viz/` is gated on the `viz` extra.
- **`viz.py` entire file (358 lines)** — `compare_models_descriptive`,
  `compare_models_comparative`, `analyse_mb`, plus their plot
  helpers, all driven through the `BNMetrics` god-class. Slice 4
  rewrites the surviving features (side-by-side viz, MB heatmap,
  MB-space distribution plot) against the new function-first API.

## Drop entirely (overlap with sister packages)

- **`utils.py:128-181` (`generate_random_dag`)** — dagsampler does
  this with structure templates, parametrized DAG families, and
  reproducibility guarantees. v0.2 callers use
  `dagsampler.CausalDataGenerator` (or a ready-made template) instead.
- **`utils.py:218-247` (`generate_synthetic_data_from_dag`)** —
  dagsampler does this better (multiple SCM mechanisms, mixed-type
  variables, stationary VAR mode). Drop unconditionally.
- **`utils.py:183-216` (`dag_to_cpdag`)** — cbcd has `DAG.to_cpdag()`
  with the proper Meek-rules-based v-structure detection. The 0.1.x
  implementation is correct on simple cases but doesn't run Meek's
  edge-orientation closure. Drop; users go through cbcd.

## Bugs and semantic issues found

1. **SID crash on certain CPDAG inputs (`sid.py:371`).** When
   `gp_is_essential_graph` flips to `False` mid-iteration and the
   current node has zero possible parents, `np.meshgrid(*[[0,1]] * 0)`
   produces an empty array and the subsequent `.reshape(-1, 0)` raises
   `ValueError: cannot reshape array of size 0 into shape (0)`.
   Reproduced on 5 of the 30 random fixtures during snapshot
   generation. **Fix in v0.2:** Slice 3 special-cases empty
   `possible_pa_gp` to a single trivial parent-set assignment; those
   five fixtures are reclaimed in `tests/sid/test_sid_handcomputed.py`
   with hand-computed expected values.

2. **Bidirected-edge semantic loss (`utils.py:18-32`).** The
   `mark_and_collapse_bidirected_edges` routine sees `A↔B` (i.e.
   both `A→B` and `B→A` present) and rewrites it to a single edge with
   `type='undirected'`. This loses the
   bidirected-vs-undirected-vs-confounded distinction that PAG outputs
   from FCI carry. SHD/HD/F1 against an FCI output therefore measure a
   strictly weaker quantity than the published metric. **Fix in v0.2:**
   bidirected edges are first-class via `EndpointMark.ARROW/ARROW`;
   the `_to_endpoints` adapter errors on `nx.DiGraph` input that
   contains bidirected edges (no auto-collapse), forcing PAG callers
   onto the proper typed channel.

3. **`compare_df` quietly returns `None` (`core.py:451-453`,
   `core.py:467-469`, `core.py:475-479`).** When the caller's metric
   selection is invalid, the method `print('please specify ...')` and
   returns `None`. No exception, no clear failure mode. **Fix in
   v0.2:** the `compare()` function raises `BNMInputError` on
   inconsistent inputs.

4. **In-place mutation of caller graph (`utils.py:19`).** The
   `mark_and_collapse_bidirected_edges` function does `G = graph.copy()`
   *but* `BNMetrics.__init__` (`core.py:79, 86, 104, 108`) stores the
   result on `self.G1`/`self.G2` and the caller's input is later
   unmodifiable in the dict-attr sense (it now has `type` attrs added
   silently). **Fix in v0.2:** `_to_endpoints` is pure (no input
   mutation, returns a fresh int8 array), and the public functions
   never reassign caller-owned objects.

5. **`count_in_degree(g, 'All')` sentinel (`metrics.py:303-327`).** The
   per-node degree functions have an overloaded "All" sentinel that
   returns `np.nan` to support the `compile_descriptive_metrics`
   table-building path. **Fix in v0.2:** per-node functions take a
   genuine `var` and aggregates are separate functions or computed by
   `compare()`.

6. **Hash-seed non-determinism in SID (`utils.py:60-62` →
   `sid.py:325`).** `get_undirected_components_with_isolates` calls
   `set(G.nodes())`, then `nx.connected_components` on the result.
   Component iteration order is sensitive to PYTHONHASHSEED, and
   downstream the SID upper-bound depends on the order in which
   non-essential parent-set combinations are enumerated. Concrete
   symptom: regenerating the legacy snapshot under different hash
   seeds yields different `sid_upper_bound` values on roughly 5/30
   random fixtures. **Fix in v0.2:** Slice 3 sorts component nodes
   and parent-set candidate orderings deterministically (no
   `set(...)` in any path that affects output). The Slice 0 generator
   pins `PYTHONHASHSEED=0` (re-execs itself if unset) so the frozen
   snapshot is reproducible.

7. **Reversal under-counting on directed→undirected
   (`metrics.py:445-453`).** When `g1` has a directed edge `u→v` and
   `g2` has it as undirected, 0.1.x checks
   ``G2.has_edge(u, v) and G2[u][v].get("type") == "undirected"`` —
   i.e. the undirected edge in `g2` is required to be stored in the
   nx.DiGraph in the SAME direction as `g1`'s directed edge. But
   nx.DiGraph stores undirected edges (post-`mark_and_collapse`) in
   whichever direction was iterated first, so the storage direction
   is arbitrary. Result: `count_reversals` mis-classifies many real
   reversals as "neither TP nor reversal" — silently breaking SHD's
   internal consistency (SHD is supposed to equal
   `additions + deletions + reversals`, and `tp + fn` should equal
   the total of g1's edges, but in 0.1.x the
   directed→undirected-stored-backwards case lands in *neither*
   bucket of the SHD partition). Concrete symptom: 17 of 87 pairs
   in the legacy snapshot have `reversals` and `shd` values that
   under-count vs. the true orientation difference. **Fix in v0.2:**
   `count_reversals` is computed on the int8 endpoint matrix where
   undirected has no implied direction; the bug cannot recur.
   `tests/fixtures_legacy_v02_overrides.json` captures v0.2's
   corrected `reversals`/`shd` values for these 17 pairs so the
   parity test passes loudly when v0.2 changes either metric again.

8. **SID upper-bound under-counting on CPDAG inputs
   (`sid.py:435-481`).** When the estimated graph is a CPDAG with
   a non-trivial equivalence class, 0.1.x's per-DAG mis-classification
   accumulator (`incorrect_sum`) credits each (i, j) mis-classification
   to the *smallest-mmm-row representative* of i's parent set under
   `np.unique(...)` + `np.sort(unique_indices)`. Two DAGs in the
   equivalence class that share a parent set for some `i` therefore
   have the increment recorded only against one of them — the
   suspicious "propagation" code at `sid.py:436-443` is meant to
   distribute it but is dead (`sum(~(int_array_xor)) == p` never
   holds because `~int(0)` is `-1`, not `1`). Net effect:
   `sid_upper_bound` is reported as the maximum *attributable* count,
   not the true maximum SID over the equivalence class. Hand-verified
   on `chain_3 vs cpdag(chain_3)`: legacy upper = 4, but DAG3
   (A←B←C in the equivalence class) actually has SID = 6 (every
   intervention pair is mis-classified relative to truth A→B→C).
   Concrete symptom: 6 of 82 SID-defined pairs in the legacy snapshot
   have under-counted `sid_upper_bound`; one DAG-vs-perturbation pair
   (`random_n20_p10_2__vs__perturbed`) also has `sid` itself diverge
   for related reasons. **Fix in v0.2:** SID is computed on the int8
   endpoint matrix with deterministic component ordering (audit §6
   fix) and the bound bookkeeping uses each DAG-row's actual
   parent-set behaviour, not a representative. Verified against
   hand-computed Peters & Bühlmann (2015) values on the canonical
   chain/fork/collider fixtures. The override file records v0.2's
   corrected SID and bound values for the 11 affected pairs.

## Top issues by impact (ranked)

1. **No tests.** Zero `tests/` directory, zero pytest fixtures, zero
   regressions. Bug §1 (SID crash) was discovered only because
   snapshot-generation actually runs the metric across a fixture set;
   nothing in the package itself exercises SID at all. **Resolved in
   v0.2:** Slices 1–3 each ship green pytest suites with hand-computed
   canonical fixtures plus the legacy snapshot at
   `tests/fixtures_legacy.json`.

2. **No Protocol contract for graph input.** The package is married to
   `nx.DiGraph + type-attr`. cbcd output (int8 matrix), dagsampler
   output (cbcd graph types), and any caller who already has a sparse
   matrix all need conversion through ad-hoc `if isinstance(...)`
   ladders (`core.py:77-114`). **Resolved in v0.2:** `bnm.GraphLike`
   Protocol is the canonical contract; cbcd's `DAG`/`CPDAG`/`PAG`
   conform with zero adaptation.

3. **Bidirected collapse (Bug §2)** silently downgrades PAG semantics.

4. **SID crash (Bug §1)** — small surface (5 of 30 random fixtures),
   but it's the metric the package is most known for.

5. **`BNMetrics` god-class** couples graph storage, metric dispatch,
   per-MB scaffolding, and viz into one class. Hard to reuse pieces
   without instantiating the whole thing. **Resolved in v0.2:** pure
   functions + a `compare()` entry point.

## Test coverage assessment

There are **no tests**. There are four Jupyter notebooks under
`bnm/use cases/` that demonstrate the API on small inputs but do not
assert anything; they install bnm 0.1.x from GitHub at the top of each
notebook so they are not part of the package distribution. v0.2's
notebooks are updated alongside Slice 4.

## What this audit changes about the rewrite plan

The pre-audit assumption was "rewrite mirrors cbcd's slices, drop the
3 dagsampler-overlapping utils, port SID, ship tests." The audit
confirms most of that and adds five concrete adjustments:

1. **SID port (Slice 3) gets a special-case fix** for the
   `possible_pa_gp == ∅` crash. The 5 affected fixtures in
   `tests/fixtures_legacy.json` carry `"sid": {"skipped": ...}` and
   are reclaimed via hand-computed values.
2. **`_to_endpoints` adapter errors on bidirected `nx.DiGraph`
   input** rather than collapsing. Bidirected callers must route
   through cbcd's `PAG` or a raw int8 matrix.
3. **`compare()` is the *only* multi-metric entry point**
   — the `BNMetrics` god-class is dropped in full, no shim.
4. **Per-node metrics drop the `'All'` sentinel** — they take a real
   `var`; aggregates are separate.
5. **`compare()` raises `BNMInputError` on bad input** rather than
   returning `None` after a `print()`.

The four-slice sequencing from the journal stands; this audit just
puts firm contracts on the questions that were left open.
