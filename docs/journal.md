# cbcd development journal

Append-only log of what's been done, when, and why. New entries at the **top**.

---

## 2026-05-06 — O5 resolved: v0.x API stability committed

Open question O5 of `docs/design/api_v0.py` — when to commit to API
stability — is now closed. The original proposal: freeze v0.x after PC,
FCI, and PCMCI are implemented end-to-end. All three shipped today; the
gate is met.

Recorded as **D15** in the design's decisions block.

The committed surface is what `cbcd/__init__.py` re-exports today.
Across all v0.x minor and patch bumps:

* **Backwards-compatible:** breaking changes (signature changes, removed
  exports, behaviour changes invalidating caller assumptions) require a
  major version bump to v1.0.
* **Additive changes** (new top-level algorithms, new kwargs with
  backwards-compatible defaults, new CI tests via the registries, new
  result-type fields with safe defaults) ship without notice.

What's NOT frozen yet — additions allowed, but each must conform to its
current §G / §H signature when implemented:

* Unimplemented variants (`MaxPOrienter`, `ConservativeOrienter`,
  `MajorityOrienter`, `DefiniteMaxPOrienter`, `conservative_pc`,
  `majority_pc`, `mvpc`, `cdnod`, `jci`, `iod`, `pcmci_plus`, `lpcmci`,
  `tsfci`, `svar_fci`, `j_pcmci`).
* `MAG` methods currently raising `NotImplementedError`.
* `pc_alpha=None` auto-tune (open question O4).
* `RunRecorder` semantics beyond the `NullRecorder` no-op (gated on
  `InMemoryRecorder` / `FileRecorder` / `.cbcd` archive landing).
* Internal modules not exported from `cbcd/__init__.py`.
* Open questions O1, O2, O3.

The full breakdown of frozen vs. not-frozen lives in D15 alongside the
other settled decisions. `README.md` and `CLAUDE.md` updated to reflect
the commitment.

No code changed; this is a documentation-only commit.

---

## 2026-05-06 — Third implementation slice: end-to-end `pcmci()`

Third vertical slice from `docs/design/api_v0.py`: vanilla PCMCI (Runge et
al. 2019) implemented end-to-end. This is the first §H (time-series) slice
and the third pressure test on the design overall — after PC (CPDAG output,
i.i.d.) and FCI (PAG output, i.i.d.), PCMCI exercises the parallel
time-series API with a per-target search shape and a 3D endpoint matrix.

225 tests pass; ruff, mypy, and ruff format all clean. The slice was
plan-approved as `~/.claude/plans/prancing-temporal-tigramite.md`.

### M1 — Lagged primitives + time-series graph types

- `cbcd/timeseries/lagged.py` — `LaggedVar` (frozen, hashable, `lag ≤ 0`
  enforced at construction), `LaggedDataset` (`max_lag < T - 1` validated),
  `LaggedBackgroundKnowledge` (D5 fail-fast: time-direction check
  `src.lag ≤ dst.lag`, no overlap with `forbidden_lagged`, no required
  contemporaneous edges that contradict `no_contemporaneous` or
  `contemporaneous_tiers`, no required autoregressive edges when
  `no_autoregressive` set).
- `cbcd/timeseries/graph.py` — `LaggedEdge`, `_LaggedGraphBase` ABC,
  `TimeSeriesDAG`, `TimeSeriesCPDAG`, `PartialTimeSeriesCPDAG` (stub for
  PCMCI+). Storage convention codified: `endpoints[0]` is symmetric in
  NO_EDGE (lag-0 edges follow the i.i.d. convention); `endpoints[τ ≥ 1]`
  is *not* mirror-symmetric (the cell `endpoints[τ, j, i]` is a different
  edge `j_{t-τ} → i_t`). Past-time mark on a lagged edge is implicitly
  TAIL — sufficient for vanilla PCMCI; LPCMCI / SVAR-FCI will extend.
- `to_summary_graph()` and `to_contemporaneous_graph()` projections on
  both DAG and CPDAG variants — typed with subclass-pinned returns
  (no union antipattern).

### M2 — Lagged CI test layer + PC₁ skeleton

- `cbcd/timeseries/citest.py` — `LaggedCITest` Protocol, `LaggedCITestResult`,
  `CachedLaggedCITest` (per-instance cache keyed on
  `(min, max, frozenset(S))` over `LaggedVar` tuples; mirrors the i.i.d.
  cache fix from the PC slice), `ParCorr` (linear partial correlation on
  the lagged design matrix: stack `data[ml-τ : T-τ, :]` blocks as columns,
  Schur-complement of the correlation submatrix → r-to-z transform with
  `df = T - max_lag - |S| - 3`), `make_lagged_ci_test` factory +
  `register_lagged_ci_test` extension hook. Built-in registry: only
  `"parcorr"` this slice.
- `cbcd/timeseries/skeleton.py` — `LaggedSkeleton` dataclass (per-target
  parent sets + sepsets), `LaggedSkeletonAlgorithm` Protocol,
  `PC1Skeleton`. PC₁ is a per-target search rather than a global skeleton
  scan: candidate parents `{(X, -τ) : τ ∈ [1, max_lag]}` (autocorrelation
  included) are pruned by depth, ordered by `pval_max` (smallest p =
  strongest evidence first → highest priority as a conditioning member).
  Sepsets are recorded for every removed (Z, target) pair.

### M3 — `pcmci()` composition + structural regression

- `cbcd/timeseries/algorithms.py` — `pcmci()` runs PC₁ → MCI step (for
  each candidate `(X, -τ)`-to-`Y` edge: condition on
  `̂P(Y) ∪ shifted ̂P(X)` minus the candidate itself; shifted parents of
  X at lag `-τ` are `{(Z, -(τ+σ)) : (Z, -σ) ∈ ̂P(X), τ+σ ≤ max_lag}`)
  and formats the result as a `TimeSeriesCPDAG` with all lagged edges
  directed past→present and no contemporaneous edges (vanilla PCMCI).
  `pc_alpha=None` defaults to `alpha` per decision below.
- `tests/timeseries/oracle.py` — `DSeparationOracleLagged` unrolls a
  true `TimeSeriesDAG` into a static `nx.DiGraph` over `(var, t)` for
  `t ∈ [0, T_horizon]` and answers `is_d_separator` queries via
  `networkx.is_d_separator`. Stationarity is exploited: the oracle picks
  a sufficiently-interior reference time `T_ref` and translates
  `LaggedVar(v, -τ) → (v, T_ref - τ)`.
- `tests/timeseries/fixtures.py` — three VAR fixtures: AR(1) on a
  single var, 2-var VAR(1) (full autocorrelation + cross-effects), and
  3-var sparse VAR(2) (mixed lags, no autocorrelation).

### Verification

- **Structural regression** (`tests/timeseries/test_pcmci_oracle.py`):
  all three VAR fixtures recover the expected `TimeSeriesCPDAG` with
  SHD = 0 endpoint-by-endpoint. Recovered marks are subsets of
  `{NO_EDGE, ARROW}` (vanilla PCMCI never produces TAIL marks).
- **PC₁** (`tests/timeseries/test_pc1.py`): per-target parent
  recovery matches the truth on every fixture; sepsets recorded for
  pairs PC₁ ruled out (the `X_{t-2} → Y_t` non-edge in `sparse_var2`).
- **ParCorr math** (`tests/timeseries/test_citest_parcorr.py`):
  unconditional, strong-lagged-dependence, and chain-blocked-by-mediator
  cases all behave correctly on simulated VAR data; `n_effective =
  T - max_lag` reported; rejects same-column queries, S overlap,
  out-of-horizon lags, and zero-variance columns.
- **CachedLaggedCITest** (`tests/timeseries/test_cached_lagged_ci.py`):
  cache key is unordered in (x, y) and in S; pass-through when
  `cache=False`; `is_cached(...)` works.
- **Inputs** (`tests/timeseries/test_pcmci_inputs.py`): alpha bounds,
  `pc_alpha=None` ≡ `pc_alpha=alpha`, `n_jobs != 1` rejection,
  CI-test/dataset n_vars and max_lag mismatch, string CI-test
  resolution to `ParCorr`.

### Decisions taken in this slice

- **D9 reaffirmed** (stationary-only v0): vanilla PCMCI assumes
  stationarity; the `LaggedDataset` is the boundary.
- **`pc_alpha=None` policy**: default to `alpha`. Open question O4
  (tigramite's `{0.05, 0.1, 0.2, 0.3, 0.4}` grid + AIC) is left open
  and will resurface when LPCMCI / PCMCI+ ship.
- **CI test surface**: only `"parcorr"` registered. `gpdc`, `cmi_knn`,
  `regci` deferred.
- **Subpackage layout**: `cbcd/timeseries/` mirrors `cbcd/graph/` and
  `cbcd/algorithms/`; lagged code is visibly parallel to the i.i.d.
  side rather than scattered.

### Out of scope (deferred)

- **PCMCI+** (contemporaneous edges via a second PC step). Requires
  `LaggedColliderOrienter`, `LaggedCPDAGRules`, `PartialTimeSeriesCPDAG`
  flow.
- **LPCMCI / tsFCI / SVAR-FCI** (latent-aware time-series; PAG output).
  Requires `PartialTimeSeriesPAG`, `TimeSeriesPAG`, `LaggedPAGRules`.
- **J-PCMCI** (multi-dataset).
- **`pc_alpha=None` auto-tune** (open question O4).
- **Additional lagged CI tests**: `gpdc`, `cmi_knn`, `regci`.
- **`recorder` integration** — accepted and validated, but no-op
  (`NullRecorder`), matching PC/FCI.
- **`n_jobs > 1`**: rejected.
- **Regime-switching variants** (RPCMCI, regime-FCI): out per D9.
- **`LaggedEdge.__str__`** pretty-printing.

After this slice the design has been pressure-tested on three
diverse algorithm shapes (CPDAG/i.i.d., PAG/i.i.d., CPDAG/time-series).
Open question **O5** (committing to v0.x API stability) becomes
addressable; resolution is its own decision and is not part of this
slice.

---

## 2026-05-06 — Second implementation slice: end-to-end `fci()`

Implemented the FCI family end-to-end against `docs/design/api_v0.py`: PAG /
PartialPAG / MAG graph types (§D), `apply_to_pag` (§E), `FCIRules` with all
ten of Zhang's R1–R10 (§F), `PossibleDSepRefinement` (§F), `FAS` skeleton
wrapper (§C), and `fci()` / `rfci()` / `anytime_fci()` composition (§G). 162
tests pass; ruff, mypy, and ruff format all clean.

This is the second pressure test on the §A–§G abstractions. Plan was written
to `~/.claude/plans/glistening-jumping-catmull.md`.

### M1 — PAG / PartialPAG / MAG + `apply_to_pag` + FAS

- `cbcd/graph/pag.py`: `PartialPAG` (working canvas; carries optional
  `sepsets` field per the plan), `PAG` (closed result; `definite_edges()`
  surfaces edges with no CIRCLE on either end, `possibly_directed(i, j)`
  reports whether a directed orientation `i → j` is consistent with the
  marks). `MAG` is a minimal stub: TAIL/ARROW marks only, every edge must
  be directed or bidirected; `is_ancestor_of` / `m_separated` / `to_pag`
  raise `NotImplementedError` and are deferred to the latent-projection
  slice.
- `cbcd/collider.py`: `ColliderDecisions.apply_to_pag` mirrors
  `apply_to_cpdag`. Initializes every skeleton edge as `CIRCLE—CIRCLE`,
  writes `ARROW` at Z on both arms of each collider triple, and propagates
  the skeleton's sepsets onto the resulting `PartialPAG`. Last-write
  semantics on the Z-mark when collider triples overlap (D14, mirroring
  `apply_to_cpdag`).
- `cbcd/skeleton.py`: `FAS` class — composition over `PCStable`, forwards
  `__call__`. Avoids leaking `track_max_pvalue` into FCI's default surface;
  cheap extension point for future FCI-specific tweaks.
- Re-exports updated on `cbcd/graph/__init__.py` and `cbcd/__init__.py`.

### M2 — Graph queries + refinement + FCIRules R1–R10

- `cbcd/graph/queries.py`: `possible_dsep(endpoints, x, y)` (BFS over
  `(vertex, predecessor)` states; admit only collider-or-triangle
  triples), `find_uncovered_circle_path`, `find_uncovered_pd_path`,
  `find_discriminating_path`. "Uncovered" defined per the standard:
  every consecutive triple unshielded (path *endpoints* may themselves
  be adjacent — needed for R5 and R9). The discriminating-path search
  grows backwards from `a` toward θ, requiring intermediate vertices to
  be parents of `c` AND colliders on the path; the θ candidate is
  required only to be non-adjacent to `c` (corrected during M2 — initial
  implementation incorrectly required θ to be a parent of `c` too).
- `cbcd/refinement.py`: `PAGSkeletonRefinement` Protocol +
  `PossibleDSepRefinement`. Increasing-size enumeration over
  Possible-D-Sep with early break — replaces causal-learn's
  `removeByPossibleDsep` (audit FCI.py:1000–1058) which enumerated the
  full powerset twice per edge. Removed edges have their orientations
  wiped back to `CIRCLE—CIRCLE` so the caller can re-run collider
  classification on the refined skeleton (D13 — see below).
- `cbcd/rules.py`: `PAGRules` Protocol + `FCIRules` class with R1–R10.
  Mirrors `MeekRules` shape: `__call__(graph, *, background, max_iterations)
  -> PAG`, copy endpoints once at entry, one `_apply_zhang_rN` helper per
  rule, fixpoint loop filters by `self.rules`. Rule names typed as
  `frozenset[str]` (loose, matching §F's design). New helper `_set_mark`
  is the PAG analogue of `_try_orient`: writes a *single* endpoint mark
  with a background-knowledge guard that refuses writes which would
  result in a forbidden directed orientation.

### M3 — `fci()` / `rfci()` / `anytime_fci()` + structural regression

- `cbcd/algorithms/fci.py`: `fci()` runs the two-pass pipeline (D13):
  `_normalize_data → make_ci_test → CachedCITest → FAS → SepsetOrienter
  → apply_to_pag → PossibleDSepRefinement → re-run SepsetOrienter on the
  refined skeleton → apply_to_pag → FCIRules → PAG`. After refinement,
  the second collider pass uses a freshly-constructed `Skeleton` from
  the refined `PartialPAG`'s adjacency, carrying through the witness
  sepsets recorded by refinement. `rfci()` is `fci(refinement=None,
  rules=FCIRules({R1..R4}))`. `anytime_fci(data, max_cond_set, ...)` is
  `fci(max_cond_set=...)` with `max_cond_set` positional + required.
- `tests/fixtures_pag.py`: 4 hand-written DAG-with-latent fixtures with
  expected PAG matrices: Y-structure, chain, fork, and a confounded
  4-node case (`0 → 2, 1 → 2, 2 → 3, L → 1, L → 3`) that exercises R1,
  R2, and R4. The R4 case is particularly load-bearing — the R4
  discriminating-path orientation `1 → 3` is correct (not `1 ↔ 3`)
  because `1 → 2 → 3` makes 1 an ancestor of 3 in the original DAG, so
  ancestrality forces the MAG edge to be directed, not bidirected.
- `tests/oracle_pag.py`: `DSeparationOracleProjected` answers d-sep on a
  full DAG (latents + observed) but exposes only `n_observed` to the
  algorithm — gives FCI the right CI surface without revealing latents.

### Verification

- **Structural regression** (`tests/algorithms/test_fci_oracle.py`): all
  four PAG fixtures recover with SHD = 0 endpoint-by-endpoint. Recovered
  marks are subsets of `{NO_EDGE, TAIL, ARROW, CIRCLE}` on every fixture.
- **Inputs** (`tests/algorithms/test_fci_inputs.py`): alpha bounds,
  `n_jobs != 1` rejection, ci_test dim mismatch, DataFrame ≡ ndarray.
- **rfci** (`tests/algorithms/test_rfci.py`): on no-latent fixtures
  (where R5+ would not fire and refinement is a no-op) `rfci()` and
  `fci()` produce the same PAG.
- **anytime_fci** (`tests/algorithms/test_anytime_fci.py`): with
  `max_cond_set=0` on the confounded-chain fixture, the recovered PAG
  has strictly more adjacencies than the unbounded `fci()` (since the
  `(0, 3)` sepset has size 2 and is unreachable under the cap).
- **Per-rule unit tests** (`tests/rules/test_fci.py`): R1, R2, R3, R4
  (both branches), R5, R6, R7, R8 (both patterns), R9 each have a
  minimal-firing test plus a non-firing pattern; R1 has a
  background-knowledge block. Plus subsetting via `rules=frozenset(...)`
  and convergence on a closed PAG.
- **Graph queries** (`tests/graph/test_queries.py`): uncovered circle
  paths, PD paths (including a "blocked by arrow at start" negative
  case), Possible-D-Sep over collider chains, discriminating paths
  (minimal `p=0` case + two negatives).
- **Refinement** (`tests/refinement/test_possible_dsep.py`): with a
  stub CI test that returns `p > alpha` only for the size-2 sepset
  `{1, 2}`, refinement removes the spurious `(0, 3)` edge and records
  the witness; with `max_cond_set=1` the same edge is *not* removed.

162 tests, ruff clean, mypy clean, ruff format clean.

### Decisions taken during this slice (added to §I)

- **D13. Two-pass FCI shape.** After `PossibleDSepRefinement` removes
  edges, `fci()` re-runs the collider step on the refined skeleton
  before invoking `FCIRules`. This matches Zhang/Spirtes pseudocode
  and the `causal-learn` reference: PossibleDSep can drop edges that
  change which triples are unshielded, so the prior collider
  classification is stale.
- **D14. PAG collider conflict semantics.** `apply_to_pag` uses
  last-write semantics on the Z-endpoint mark when collider triples
  overlap, mirroring `apply_to_cpdag`. Any conflict implies the
  skeleton + sepsets are mutually inconsistent — surfacing it via the
  recorder (when `RunRecorder` is fleshed out) is the right place to
  detect it, not by silently preserving a CIRCLE mark.

### Out of scope (deferred to next slice)

- `MAG.is_ancestor_of` / `m_separated` / `to_pag` — minimal class only.
- Programmatic `DAG → MAG → PAG` projection oracle (hand-written
  fixtures used here).
- `n_jobs > 1` in `PossibleDSepRefinement` and `FAS` (rejected, matching
  `PCStable`'s policy).
- `MaxPOrienter` / `ConservativeOrienter` / `MajorityOrienter` integration
  with `apply_to_pag`.
- GFCI (score-based skeleton; design line 641).
- `recorder` integration in `FCIRules` / `PossibleDSepRefinement` —
  accepted and validated, but no-op (`NullRecorder`), matching the PC
  slice.
- `Edge.__str__` PAG-aware glyphs (`o-o`, `o->`).
- Selection-bias / m-separation handling beyond MAG's mark validation.

---

## 2026-05-06 — First implementation slice: end-to-end `pc()`

Implemented the first vertical slice of the design from `docs/design/api_v0.py`: a working `pc()` that exercises §A–§G and §J at minimal scope. Plan was written to `~/.claude/plans/goofy-finding-lemon.md`. 84 tests pass, ruff + mypy clean.

### M1 — Graph types + CI test layer (foundation)

- `cbcd/graph/`: `EndpointMark` (NO_EDGE/TAIL/ARROW/CIRCLE int8), `Edge` value class, `_GraphBase` ABC, `DAG` (TAIL/ARROW only, acyclicity validated on construction via Kahn topo-sort), `CPDAG` + `PartialCPDAG` with `directed_edges`/`undirected_edges`/`parents`/`neighbors`/`adjacent`/`to_dag_extension` (Dor–Tarsi 1992).
- `cbcd/citest/`: `CITest` runtime-checkable Protocol, `CITestResult` frozen dataclass, `CachedCITest` keyed on `(min(x,y), max(x,y), frozenset(S))` per-instance — no `str(ndarray)` md5 hashing (audit pitfall #1), `FisherZ` via Schur-complement of correlation submatrix → r-to-z transform.

### M2 — Skeleton + collider + rules (causal logic)

- `cbcd/background.py`: `BackgroundKnowledge` frozen dataclass with D5 validation in `__post_init__` (forbidden ∩ required, required-edge cycles, tier contradictions, multi-tier membership). Tier ordering implies forbidden_directed via `is_forbidden_directed`.
- `cbcd/skeleton.py`: `Skeleton` dataclass, `SkeletonAlgorithm` Protocol, `PCStable` — snapshot adjacency at start of each depth, iterate to fixpoint within depth. **Bug found and fixed during M2**: initial implementation only tried conditioning sets from `adj(x) \ {y}` (the lower-indexed endpoint). PC requires trying both `adj(x) \ {y}` and `adj(y) \ {x}` since the separating set may live in either neighbour set; the bnlearn ASIA fixture caught this (edge 1–7 needs sepset {3, 5} or {2, 5} or {4, 5}, all reachable only from one side). Fix: iterate over all (x, y) ordered pairs with a `marked` set to dedupe within a depth.
- `cbcd/collider.py`: `ColliderDecisions` (immutable triple-classification record) + `apply_to_cpdag(skeleton) → PartialCPDAG`, `ColliderOrienter` Protocol with `requires_max_pvalues` flag, `SepsetOrienter` (vanilla R0).
- `cbcd/rules.py`: `CPDAGRules` Protocol, `MeekRules` R1–R4 — pure structural pattern matches over triples/quadruples; **no `is_ancestor_of` calls** (audit pitfall #6). Background-knowledge constraints respected: `forbidden_directed` blocks rule firings, `required_directed` pre-orients before iteration begins.

### M3 — Composition + factory + structural regression

- `cbcd/recording.py`: `RunRecorder` Protocol + `NullRecorder` (every method a no-op). Plumbed through phase APIs so future recorders slot in without signature churn. `InMemoryRecorder` and `FileRecorder` deferred.
- `cbcd/citest/factory.py`: `make_ci_test(name, data, **kwargs)` + `register_ci_test(name, factory)`; only `"fisherz"` registered for this slice.
- `cbcd/algorithms/_data.py`: `_normalize_data` per D1 — accepts ndarray or DataFrame, propagates `var_names`.
- `cbcd/algorithms/pc.py`: `pc()` wires `_normalize_data → make_ci_test → CachedCITest → PCStable → SepsetOrienter → apply_to_cpdag → MeekRules → CPDAG`.
- `cbcd/__init__.py`: re-exported public surface (`pc`, `CPDAG`, `BackgroundKnowledge`, `make_ci_test`, etc.).

### Verification

- **Structural regression** (`tests/algorithms/test_pc_oracle.py`): with d-separation oracle as CI test, `pc()` recovers the exact correct CPDAG (SHD = 0 endpoint-by-endpoint) on all 6 fixtures: Y-structure, fork, chain, M-structure, diamond, bnlearn ASIA (8 nodes). Used `networkx.is_d_separator` as the oracle ground truth — initial hand-rolled Bayes-ball had a back-door bug on the ASIA case that masked the PC-stable bug above.
- **Linear-Gaussian end-to-end** (`tests/algorithms/test_pc_fisherz.py`): sampled 20k rows from SCMs over Y/chain/diamond, recovered CPDAG matches truth at α=0.01.
- **I/O** (`tests/algorithms/test_pc_inputs.py`): DataFrame + ndarray equivalence; explicit vs string-named CI test equivalence; alpha bounds; `n_jobs != 1` raises; `BackgroundKnowledge` constraints respected (required directed, forbidden adjacent).
- **Caching** (`tests/algorithms/test_pc_caching.py`): asserts `CachedCITest` lets each `(x, y, frozenset(S))` reach the inner test at most once across an entire `pc()` run.

84 tests, ruff clean, mypy clean (pandas/scipy/networkx untyped-imports allowed via `[tool.mypy.overrides]` in `pyproject.toml`).

### Tooling adjustments

- `pyproject.toml`: ruff `extend-exclude = ["docs"]` so the design stub at `docs/design/api_v0.py` isn't linted as runtime code.
- `pyproject.toml`: added `[tool.mypy]` block with `ignore_missing_imports` for pandas/scipy/joblib/networkx/sklearn.

### Out of scope (per plan, deferred to next slice)

- `MaxPOrienter`, `ConservativeOrienter`, `MajorityOrienter`, `DefiniteMaxPOrienter`, `conservative_pc`, `majority_pc`, `mvpc`.
- FCI / PAG / `PAGSkeletonRefinement` / Zhang R1–R10.
- joblib parallelism (n_jobs is plumbed but rejects > 1).
- `InMemoryRecorder`, `FileRecorder`, `.cbcd` archive.
- Additional CI tests: chisq, gsq, KCI, partialcorr.

### Design changes during implementation

None. The §A–§G + §J signatures from `docs/design/api_v0.py` matched the implementation; no edits to the design doc or §I decisions log were needed.

---

## 2026-05-05 — Project bootstrap

First working session. Took the project from "idea" to "designed and scaffolded, ready to implement."

### Decisions taken

1. **Don't fork `causal-learn` — start `cbcd` from scratch.** Original plan was to fork and refactor; the audit reversed that. See `audit_causal_learn.md` for the reasoning. Vendor only verified semantics (`SkeletonDiscovery`, the math inside `FisherZ` / `Chisq_or_Gsq`, `TestPC.py` / `TestFCI.py` fixtures), reimplement everything else against the new design.
2. **Package name**: `cbcd`. Python ≥3.11. MIT license. Hatchling build backend. uv for dev.
3. **Dependency stance**: `cbcd` does not depend on `causal-learn` directly or transitively. `citk` is a separate user-owned project on a different timeline (~6 months from now); `cbcd` does not depend on `citk` either while `citk` still pulls in `causal-learn`. `cbcd` ships its own minimal CI test layer for now. Saved to long-term memory.
4. **Pressure-tested the design walkthrough before any implementation.** Sections A–J of `design/api_v0.py` were each proposed, critiqued, refined, and applied incrementally. The decisions log at §I has 12 settled decisions and 5 remaining open questions.

### Repository scaffolded

- `pyproject.toml` (hatchling, py≥3.11, MIT, ruff, pytest)
- `cbcd/__init__.py`, `cbcd/exceptions.py` (`CBCDError`, `CBCDInputError`, `CBCDDataError`)
- `tests/test_smoke.py`
- `LICENSE`, `README.md`, `.gitignore`
- `git init` (no commits yet)

### Audit of `causal-learn` completed

Full report at `audit_causal_learn.md`. Highlights:

- **2 real bugs found**: cache-key collision (`cit.py:98` md5 on str of ndarray truncates), and CDNOD direction inconsistency (`CDNOD.py:94` vs `CDNOD.py:179`).
- **Major gaps**: no conservative-PC, majority-PC, RFCI, anytime-FCI, or parallel CI testing.
- **Performance**: `removeByPossibleDsep` (`FCI.py:1014-1058`) is O(2^|possible-D-sep|) and ignores its own `depth` parameter.
- **Code health**: 4× duplicated PC orchestration, 600 lines of pseudo-Java FCI orientation, 100+ lines of commented-out code in `Helper.py`.
- **Test coverage**: PC and FCI tests are solid (assert structural equality); CDNOD tests are broken (no assertions); CIT tests are dead.

Verdict: refactoring is more work than rewriting. `cbcd` starts clean.

### Phase-1 API design completed (`docs/design/api_v0.py`, ~1880 lines)

Pressure-tested through 10 design sessions, one per concern. Final shape:

| Section | Surface area |
|---------|--------------|
| **A** CI tests | `CITest` Protocol, `CITestResult`, `CachedCITest` (caching + optional recording, fixes the md5 bug) |
| **B** Background knowledge | `BackgroundKnowledge` dataclass — first-class, with tiers, forbidden/required edges/adjacencies |
| **C** Skeleton phase | `Skeleton` dataclass, `SkeletonAlgorithm` Protocol, `PCStable`, `FAS` |
| **D** Graph types | `EndpointMark` enum, `Edge` accessor, `_GraphBase` ABC, distinct classes `DAG` / `CPDAG` / `MAG` / `PAG` plus `Partial*` intermediates. CPDAG carries `ambiguous_triples` and `definite_non_colliders` for conservative/majority orientation. |
| **E** Collider orientation | `ColliderDecisions` value object (`apply_to_cpdag`, `apply_to_pag`), `ColliderOrienter` Protocol with `requires_max_pvalues` flag, concrete `SepsetOrienter` / `MaxPOrienter` / `ConservativeOrienter` / `MajorityOrienter` / `DefiniteMaxPOrienter` |
| **F** Edge orientation rules | `CPDAGRules` (Meek R1–R4), `PAGRules` (Zhang R1–R10, subsettable), `PAGSkeletonRefinement` (Possible-D-Sep step replacing causal-learn's exponential bug) |
| **G** Algorithms | `pc`, `conservative_pc`, `majority_pc`, `mvpc`, `fci`, `rfci`, `anytime_fci`, `cdnod`, `jci`, `iod`. Plus `make_ci_test` factory and `register_ci_test` extension. `EnsembleCITest` Protocol + `PAGEquivalenceClass` for multi-dataset methods. |
| **H** Time-series | Parallel API: `LaggedVar`, `LaggedDataset`, `LaggedBackgroundKnowledge`, `LaggedCITest`, `TimeSeriesDAG` / `TimeSeriesCPDAG` / `TimeSeriesPAG`, `pcmci`, `pcmci_plus`, `lpcmci`, `tsfci`, `svar_fci`, `j_pcmci`. 3D `endpoints` matrix `(max_lag+1, n_vars, n_vars)`. |
| **J** Recording / audit trail | `RunRecorder` Protocol, `NullRecorder` (default, zero overhead), `InMemoryRecorder`, `FileRecorder`. 5 record types. `RunRecord` snapshot persists to a single `.cbcd` archive (zip of per-event-type Parquet files + JSON manifest). Resumable from `last_checkpoint()`; recoverable from working dir if a run crashes. |
| **I** Decisions log + open questions | 12 settled (D1–D12), 5 open (O1–O5). |

### Decisions settled (§I of `design/api_v0.py`)

D1 data normalization · D2 CI test factory + registry · D3 mvpc as separate function · D4 explicit `random_state` (no global RNG) · D5 BackgroundKnowledge raises on inconsistency · D6 EnsembleCITest + PAGEquivalenceClass shape · D7 joblib + n_jobs propagation policy · D8 CDNOD canonical `c_indx → X` · D9 stationary-only time-series scope for v0 · D10 `max_cond_set` parameter naming · D11 audit-trail / `RunRecorder` system · D12 fused cache+recording in `CachedCITest`.

### Open questions remaining (§I of `design/api_v0.py`)

O1 mixed-data CI test default · O2 EnsembleCITest p-combination policy · O3 mvpc default missingness mechanism · O4 PCMCI auto-tune grid · O5 API stability commitment trigger.

### What's NOT done yet

- Zero implementation code. Every function in `design/api_v0.py` is a stub.
- No tests beyond the smoke test.
- No CI tests yet — even Fisher-Z hasn't been written.
- No `git commit` yet.

### Next session — recommended starting point

1. Implement `cbcd/citest/` layer: `CITest` Protocol concrete + `FisherZ`, `ChisqGsq`, `PartialCorr`, `KCI` (in that order). `CachedCITest` with cache-key fix.
2. Implement `cbcd/graph/` types (endpoint-mark int8 matrix storage, validation per subclass).
3. Implement `cbcd/skeleton/PCStable` (port `SkeletonDiscovery` semantics).
4. Implement `cbcd/orient/SepsetOrienter` + `cbcd/orient/MeekRules`.
5. Wire `cbcd.pc()`. Port `TestPC.py` fixtures, run, expect SHD=0 vs d-separation oracle on 13 graphs.

After PC is working end-to-end, the design will have faced its first real diversity check. Then implement FCI (Section F's PAG rules + Possible-D-Sep) for the second pressure test. Only then commit to v0.x API stability (open question O5).

---
