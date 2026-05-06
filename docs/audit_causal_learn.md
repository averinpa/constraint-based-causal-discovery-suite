# Audit of `causal-learn` constraint-based code

**Date:** 2026-05-05
**Source tree audited:** `/Users/pavelaverin/Projects/vendor/causal-learn/` (commit `070d5ea`)
**Scope:** files under `causallearn/search/ConstraintBased/`, `causallearn/utils/PCUtils/`, `causallearn/utils/cit.py`, and `causallearn/utils/FAS.py`. Test files in `tests/Test{PC,FCI,CDNOD,CIT}.py` reviewed for coverage.

This audit informs `cbcd`'s clean-room rewrite. References are `file:line` against the audited tree.

## Verdict

**Not a good production base — usable research code.** The codebase is a thin Python port of Tetrad with heavy duplication, no parallelism, weak abstractions, two real bugs, and several O(2^n) hot paths in FCI. Core math (PC-stable skeleton + most CI tests) is correct and validated, but the surrounding scaffolding should not be carried forward.

## Files to keep mostly as-is (port + minor fixes)

- **`cit.py:86-191` (`CIT_Base`) + `FisherZ` + `Chisq_or_Gsq`.** Cleanest module in the package. Per-instance `pvalue_cache` keyed by stringized `_stringize(Xs;Ys|S)` (`cit.py:150-176`); correlation matrix precomputed once in `FisherZ.__init__` (`cit.py:197`); discrete CIT vectorised via two-strategy bincount/unique tables with a memory threshold (`cit.py:339-390`). Port these classes, fix the cache-key bug (see Bugs §1), and they become `cbcd.citest`.
- **`SkeletonDiscovery.skeleton_discovery` (`SkeletonDiscovery.py:16-138`).** Correct PC-stable implementation with proper batched edge removal at depth-end. Validated against d-separation oracle on 13 bnlearn graphs in `TestPC.py:357-382`. Port semantics into the new `SkeletonAlgorithm` Protocol.
- **`TestPC.py`, `TestFCI.py`.** Actually assert structural equality / SHD against ground truth. Port as regression fixtures.

## Refactor before extending

- **`PC.py:111-133` and `PC.py:222-244`** — 23-line if/elif ladder over `uc_rule` × `uc_priority` magic ints, duplicated verbatim in `mvpc_alg`, `cdnod_alg` (`CDNOD.py:99-121`), and again in `mvcdnod_alg` (`CDNOD.py:181-203`). **Fix in cbcd:** strategy registry of `ColliderOrienter` classes (see `design/api_v0.py` §E).
- **`UCSepset.py:51-103, 189-241`** — priority-0/1/2 blocks copy-pasted across `uc_sepset` and `maxp` (~50 lines each). **Fix in cbcd:** extract `_orient_collider(cg, x, y, z, priority)`; concrete classes per orienter strategy.
- **`Helper.py`** (726 lines) — 100+ lines of commented-out reference code at top (`Helper.py:13-97`); `tetrad2adjmat`/`adjmat2tetrad` (`Helper.py:427-516`) belong in I/O; `find_unshielded_triples` (`Helper.py:268-271`) materialises O(E²) tuples. **Fix in cbcd:** split into `cbcd.io`, `cbcd.graph.queries`, `cbcd.mvpc`.
- **`FAS.py:113`** — `np.delete(Neigh_x, np.where(Neigh_x == y))` inside the hot inner loop. **Fix:** precompute neighbour sets.
- **`PC.py:14, FCI.py:15, CDNOD.py:12, FAS.py:13`** — all do `from causallearn.utils.cit import *`. **Fix:** explicit imports.

## Rewrite (do not carry forward)

- **`FCI.py` orientation rules (lines 308-909).** Pseudo-Java with `__contains__`/`get_nowait`, manual `ChoiceGenerator` (`FCI.py:318-323`) instead of `itertools.combinations`, `Queue` from `queue` (locking overhead) instead of `collections.deque`. `rule8` (`FCI.py:778-806`) is a 14-line single boolean expression. `removeByPossibleDsep` (`FCI.py:1000-1058`) duplicates the entire test loop for `(a,b)` and `(b,a)`. **A clean Python rewrite using a `PAG` class with edge-mark tables would be ~⅓ the code.**
- **`getPossibleDsep` (`FCI.py:188-266`).** Hand-rolled BFS with `Queue` plus a hand-managed `previous` dict; `e is None` distance hack at line 222-224; default `maxPathLength=-1` silently turned into 1000 (line 225). Use networkx or a clean BFS.
- **MVPC plumbing in `PC.py:254-476`.** `detect_parent` (`PC.py:302-391`) reimplements skeleton search by hand instead of calling `skeleton_discovery`, with three "Adaptation" comment blocks. `skeleton_correction` is another 67-line copy of the FAS loop (`PC.py:442-474`). **Fix in cbcd:** `mvpc()` is its own function but composes the same `SkeletonAlgorithm` Protocol; its corrections are a separate composable phase.
- **`Helper.py:521-727` MVPC virtual-data generation** — uses `np.random.shuffle` on global RNG (`Helper.py:547`); non-reproducible without external `np.random.seed`. **Fix in cbcd:** every randomized algorithm accepts `random_state: int | np.random.Generator | None`; no global RNG fallback (settled as decision D4).
- **`Meek.py:48-91`** — calls `is_ancestor_of` per Meek-rule application — graph traversal inside a fixpoint loop, making Meek O(V³ · ancestor-cost). Standard Meek doesn't need ancestor checks (the rule already guarantees acyclicity); looks like a defensive paste. Verify and remove.

## Bugs found

1. **Cache key collision in `cit.py:98`.** Cache key uses `hashlib.md5(str(data).encode())`. `str(np.ndarray)` truncates with `...` for large arrays — two different datasets that differ only in the middle hash equal, silently corrupting the cache. **Fix in cbcd:** `CachedCITest` keys by *test instance identity* + `(x, y, frozenset(S))`. No fragile data-content hashing.
2. **CDNOD direction inconsistency.** `cdnod_alg` (`CDNOD.py:94`) orients `c_indx → X`; `mvcdnod_alg` (`CDNOD.py:179`) orients `X → c_indx`. One is wrong. `TestCDNOD.py` only calls `draw_pydot_graph()` and never asserts — uncaught. **Fix in cbcd:** canonical direction `c_indx → X` (decision D8) per Huang et al. 2019; enforced via `BackgroundKnowledge` that forbids edges into the context node.

## Top issues by impact (ranked)

1. **No PC orientation strategy abstraction.** `PC.py:111-133` and 3 verbatim copies hardcode magic ints `uc_rule∈{0,1,2}` × `uc_priority∈{-1..4}`. **Resolved in cbcd:** §E `ColliderOrienter` Protocol + concrete classes.
2. **Conservative-PC / majority-PC missing entirely.** `UCSepset.py` has only `uc_sepset`, `maxp`, `definite_maxp`. No conservative orientation (Ramsey 2006), no majority rule, no anytime-FCI, no RFCI. **Resolved in cbcd:** §E `ConservativeOrienter`, `MajorityOrienter` stubs; §G `rfci()`, `anytime_fci()`.
3. **No parallelism for independent CI tests.** `SkeletonDiscovery.py:69-119` and `FAS.py:89-137` iterate serially even though all CI tests at a given depth are embarrassingly parallel under PC-stable. For KCI on n=20, depth-2 this is 1000s of independent kernel computations. **Resolved in cbcd:** decision D7 — joblib via `n_jobs` plumbed to skeleton + refinement; algorithms parallelise per-depth CI test batches.
4. **`removeByPossibleDsep` is exponential.** `FCI.py:1014-1058` enumerates all 2^|possibleDsep| subsets twice (once for `(a,b)`, once for `(b,a)`) with no early termination. The `depth` parameter is ignored. **Resolved in cbcd:** `PossibleDSepRefinement` (§F) iterates increasing-size with early break, single (X, Y) test pass, parallel-aware.
5. **`cit.py:98` data-hash bug** (see Bugs §1).

## Test coverage assessment

- **`TestPC.py`** — genuinely good. Compares full graph matrices to fixtures and asserts SHD=0 vs d-separation oracle on 13 graphs. **Port verbatim as regression fixtures.**
- **`TestFCI.py`** — checks adj/arrow precision/recall against ground-truth PAGs. Solid. Port.
- **`TestCDNOD.py:10-46`** — **broken**: only calls `draw_pydot_graph()` and `print('finish')`, no assertions. Bug §2 went uncaught here. Do not port; rewrite from scratch with structural assertions.
- **`TestCIT.py`** — 17 lines, calls `cit.fisherz` which doesn't exist on the class. Dead. Do not port.

## What this audit changes about the cbcd plan

The original recommendation was "fork causal-learn and extend." This audit reversed that: **start cbcd from scratch**, vendor only the verified pieces (`cit.py` algorithms after fixing Bug §1, `SkeletonDiscovery` semantics, `TestPC.py` / `TestFCI.py` fixtures), and use the Section A–J Protocols as the new foundation. Refactoring 4× duplicated PC orchestration + rewriting 600 lines of FCI orientation + fixing Helper.py is more effort than building it cleanly with a strategy-based API, and forking would inherit the latent bugs (md5 collision, CDNOD direction) into the first cbcd release.
