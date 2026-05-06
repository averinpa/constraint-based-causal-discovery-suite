# cbcd development journal

Append-only log of what's been done, when, and why. New entries at the **top**.

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
