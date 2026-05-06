# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

`cbcd` (constraint-based causal discovery) has **two vertical slices working end-to-end**: `pc()` (PC family) and `fci()` (FCI family — `fci()`, `rfci()`, `anytime_fci()`). FCI was the second pressure test on the §A–§G abstractions; the design has now been validated against both a CPDAG output (PC) and a PAG output (FCI with CIRCLE marks, Zhang R1–R10, Possible-D-Sep refinement). As of the latest journal entry, runtime code covers:

- `cbcd/graph/` — `EndpointMark`, `Edge`, `_GraphBase`, `DAG`, `CPDAG` + `PartialCPDAG` (Dor–Tarsi extension), `PAG` + `PartialPAG`, `MAG` (minimal stub — methods deferred), graph queries (`possible_dsep`, `find_uncovered_circle_path`, `find_uncovered_pd_path`, `find_discriminating_path`)
- `cbcd/citest/` — `CITest` Protocol, `CITestResult`, `CachedCITest` (frozenset-keyed; fixes the `causal-learn` md5 cache bug), `FisherZ`, `make_ci_test` factory + `register_ci_test`
- `cbcd/background.py` — `BackgroundKnowledge` with D5 inconsistency validation
- `cbcd/skeleton.py` — `Skeleton`, `PCStable`, `FAS` (composition over `PCStable`, FCI's default skeleton)
- `cbcd/collider.py` — `ColliderDecisions` (with both `apply_to_cpdag` and `apply_to_pag`), `SepsetOrienter`
- `cbcd/rules.py` — `MeekRules` R1–R4 for CPDAG, `FCIRules` R1–R10 (Zhang 2008) for PAG. Pure structural, single-mark writes via `_set_mark` for FCI (and `_try_orient` for Meek).
- `cbcd/refinement.py` — `PAGSkeletonRefinement` Protocol + `PossibleDSepRefinement` (increasing-size enumeration with early break; replaces the exponential `removeByPossibleDsep` in `causal-learn`)
- `cbcd/recording.py` — `RunRecorder` Protocol + `NullRecorder`
- `cbcd/algorithms/pc.py` — `pc()` end-to-end
- `cbcd/algorithms/fci.py` — `fci()` (two-pass: skeleton → collider → refinement → re-run collider → rules), `rfci()` (no refinement, R1–R4 only), `anytime_fci()` (depth-capped)

Still stub-only in `docs/design/api_v0.py` (not yet implemented): `MaxPOrienter` / `ConservativeOrienter` / `MajorityOrienter` / `DefiniteMaxPOrienter`, `conservative_pc` / `majority_pc` / `mvpc`, `MAG` methods (`is_ancestor_of` / `m_separated` / `to_pag` — gated on the latent-projection slice), CDNOD / JCI / IOD / time-series, `InMemoryRecorder` / `FileRecorder` / `.cbcd` archive, joblib parallelism, additional CI tests (chisq, gsq, partialcorr, KCI). When asked to implement one of these, expect to be filling in stubs from `docs/design/api_v0.py` against the patterns established in the PC and FCI slices.

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
- **`cbcd` does NOT depend on `citk`** (a separate user-owned library at `../../citk/`). `citk` currently pulls in `causal-learn` and will not change for ~6 months. After late 2026, `citk` may interoperate via the `CITest` Protocol — until then, neither package depends on the other.
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

Every algorithm ships with structure-level regression tests (SHD against a d-separation oracle on standard graphs), not just smoke tests. Two reference patterns:

* **PC** (`tests/algorithms/test_pc_oracle.py`): runs `pc()` against a `networkx.is_d_separator` oracle on six fixtures (Y, fork, chain, M, diamond, bnlearn ASIA) and asserts SHD = 0 endpoint-by-endpoint.
* **FCI** (`tests/algorithms/test_fci_oracle.py`): runs `fci()` against `DSeparationOracleProjected` (operates on a full DAG with latents, exposes only observed indices) on hand-written fixtures from `tests/fixtures_pag.py`, including a confounded chain that exercises Zhang R1, R2, and R4. Same SHD = 0 bar.

New algorithm tests should follow one of these patterns.
