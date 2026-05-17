# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

`cbcd` (constraint-based causal discovery) has **three vertical slices working end-to-end**: `pc()` (PC family — i.i.d., CPDAG output), `fci()` (FCI family — i.i.d., PAG output), and `pcmci()` (PCMCI family — time-series, TimeSeriesCPDAG output). PCMCI was the third pressure test on the design; the §A–§G abstractions are validated against both CPDAG and PAG outputs in the i.i.d. layer, and §H is validated against the per-target search shape with a 3D endpoint matrix in the time-series layer.

**v0.x API stability commitment (D15, 2026-05-06)**: the public surface re-exported from `cbcd/__init__.py` is committed backwards-compatible across all v0.x minor and patch bumps. Breaking changes wait for v1.0. Additive changes (new algorithms, new kwargs with safe defaults, new CI tests via the registries) ship in minor bumps without notice. Stub-only items in the design (e.g., `MaxPOrienter`, `lpcmci`, `MAG` methods) are NOT yet frozen — when implemented, they must conform to their current §G / §H signatures. See D15 in `docs/design/api_v0.py` for the full frozen-vs-not breakdown.

As of the latest journal entry, runtime code covers:

**i.i.d. layer (§A–§G):**
- `cbcd/graph/` — `EndpointMark`, `Edge`, `_GraphBase`, `DAG`, `CPDAG` + `PartialCPDAG` (Dor–Tarsi extension), `PAG` + `PartialPAG`, `MAG` (minimal stub — methods deferred), graph queries (`possible_dsep`, `find_uncovered_circle_path`, `find_uncovered_pd_path`, `find_discriminating_path`)
- `cbcd/citest/` — `CITest` Protocol, `CITestResult`, `CachedCITest` (frozenset-keyed), `FisherZ`, `make_ci_test` factory + `register_ci_test`
- `cbcd/background.py` — `BackgroundKnowledge` with D5 inconsistency validation
- `cbcd/skeleton.py` — `Skeleton`, `PCStable`, `FAS`
- `cbcd/collider.py` — `ColliderDecisions` (`apply_to_cpdag` and `apply_to_pag`), `SepsetOrienter`
- `cbcd/rules.py` — `MeekRules` R1–R4 (CPDAG), `FCIRules` R1–R10 Zhang 2008 (PAG)
- `cbcd/refinement.py` — `PAGSkeletonRefinement` + `PossibleDSepRefinement`
- `cbcd/algorithms/pc.py` — `pc()`
- `cbcd/algorithms/fci.py` — `fci()` (two-pass), `rfci()`, `anytime_fci()`

**time-series layer (§H):**
- `cbcd/timeseries/lagged.py` — `LaggedVar`, `LaggedDataset`, `LaggedBackgroundKnowledge`
- `cbcd/timeseries/graph.py` — `LaggedEdge`, `_LaggedGraphBase`, `TimeSeriesDAG`, `TimeSeriesCPDAG`, `PartialTimeSeriesCPDAG` (stub for PCMCI+)
- `cbcd/timeseries/citest.py` — `LaggedCITest` Protocol, `LaggedCITestResult`, `CachedLaggedCITest`, `ParCorr`, `make_lagged_ci_test` + `register_lagged_ci_test`
- `cbcd/timeseries/skeleton.py` — `LaggedSkeleton`, `PC1Skeleton` (per-target lagged-parent pruning by association strength)
- `cbcd/timeseries/algorithms.py` — `pcmci()` (PC₁ + MCI two-stage)

**shared:**
- `cbcd/exceptions.py`, `cbcd/recording.py`

Still stub-only in `docs/design/api_v0.py` (not yet implemented): `MaxPOrienter` / `ConservativeOrienter` / `MajorityOrienter` / `DefiniteMaxPOrienter`, `conservative_pc` / `majority_pc` / `mvpc`, `MAG` methods, CDNOD / JCI / IOD, **PCMCI+ / LPCMCI / tsFCI / SVAR-FCI / J-PCMCI**, `PartialTimeSeriesPAG` / `TimeSeriesPAG` and time-series PAG rules, `InMemoryRecorder` / `FileRecorder` / `.cbcd` archive, joblib parallelism, additional CI tests (chisq, gsq, partialcorr, KCI for i.i.d.; gpdc, cmi_knn, regci for time-series), `pc_alpha=None` auto-tune (open question O4). When asked to implement one of these, expect to be filling in stubs from `docs/design/api_v0.py` against the patterns established in the PC, FCI, and PCMCI slices.

## Commands

Dev environment is managed with `uv` (Python ≥3.11, hatchling build backend).

```bash
uv sync --all-extras          # install with dev/parallel/viz/docs extras
uv run pytest                 # run tests (testpaths = ["tests"])
uv run pytest tests/test_smoke.py::test_version_is_set   # single test
uv run pytest -n auto         # parallel (pytest-xdist, from dev extras)
uv run ruff check .           # lint (rules: E, F, I, B, UP, SIM; line-length 100; E501 ignored)
uv run ruff format .          # format
uv run mypy cbcd              # type-check
```

## Architecture

### Composition-first design

Every algorithm is wired from a small, fixed set of Protocols defined in `docs/design/api_v0.py`. A new algorithm should mostly be plumbing existing pieces, not a new monolithic function. The Protocols are:

- `CITest` — conditional independence test (§A)
- `SkeletonAlgorithm` — skeleton-discovery phase (§C)
- `ColliderOrienter` — collider-orientation strategy (§E)
- `CPDAGRules` / `PAGRules` / `PAGSkeletonRefinement` — edge-orientation rules (§F)

Top-level functions (`pc`, `fci`, `cdnod`, `rfci`, `pcmci`, etc.) compose these pieces in §G (i.i.d.) and §H (time-series). Graph types (`DAG`, `CPDAG`, `MAG`, `PAG` + partial intermediates) are distinct classes over an int8 endpoint-mark matrix (§D). Audit/recording is built in via `RunRecorder` and the `.cbcd` archive format (§J).

`docs/design/api_v0.py` is sectioned A–J. **§I is the decisions log + open questions** — read it first when an implementation choice feels ambiguous.

### Dependency stance (load-bearing)

- **`cbcd` does NOT depend on `causal-learn`**, directly or transitively. The `causal-learn` repository sometimes present at `../../vendor/causal-learn/` is **read-only reference material** — used to port test fixtures and verify behavioural parity. Never import it from `cbcd` code or tests.
- **`cbcd` does NOT depend on `citests`** (a separate user-owned library at `../../citests/`). `citests` currently pulls in `causal-learn` and will not change for ~6 months. After late 2026, `citests` may interoperate via the `CITest` Protocol — until then, neither package depends on the other.
- `cbcd` therefore ships its own minimal CI test layer (Fisher-Z, χ², G², partial correlation, KCI planned).

### Exception hierarchy

All errors derive from `CBCDError`. `CBCDInputError` (also `ValueError`) is for caller mistakes; `CBCDDataError` (also `ValueError`) is for structurally unsuitable input data. Tests use these subclass relationships — preserve them.

## Workflow when the design changes

The design is the source of truth before implementation. When a design decision changes:

1. Update the affected section of `docs/design/api_v0.py` **and** the §I decisions log in the same file.
2. Append a dated entry to `docs/journal.md` (newest at the top) describing what changed and why.
3. If the change invalidates a finding in `docs/audit_causal_learn.md`, note it there too.

`docs/audit_causal_learn.md` documents specific bugs and design issues in `causal-learn` (with file:line references) that motivated rewriting rather than forking — consult it before reproducing patterns from the reference repo.

## Correctness bar

Every algorithm ships with structure-level regression tests (SHD against a d-separation oracle on standard graphs), not just smoke tests. Three reference patterns:

* **PC** (`tests/algorithms/test_pc_oracle.py`): runs `pc()` against a `networkx.is_d_separator` oracle on six fixtures (Y, fork, chain, M, diamond, bnlearn ASIA) and asserts SHD = 0 endpoint-by-endpoint.
* **FCI** (`tests/algorithms/test_fci_oracle.py`): runs `fci()` against `DSeparationOracleProjected` (operates on a full DAG with latents, exposes only observed indices) on hand-written fixtures from `tests/fixtures_pag.py`, including a confounded chain that exercises Zhang R1, R2, and R4. Same SHD = 0 bar.
* **PCMCI** (`tests/timeseries/test_pcmci_oracle.py`): runs `pcmci()` against `DSeparationOracleLagged` (unrolls a `TimeSeriesDAG` to a static `nx.DiGraph` over `(var, t)` and answers via `networkx.is_d_separator` at a stationary reference time) on three VAR fixtures (AR(1), 2-var VAR(1), sparse VAR(2)). Same SHD = 0 bar.

New algorithm tests should follow one of these patterns.
