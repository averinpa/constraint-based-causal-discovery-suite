# cbcd documentation

This folder is the canonical home for design discussion, decision history, and reference material that supports the `cbcd` package. Code lives in `../cbcd/`; tests in `../tests/`. Everything here is markdown or Python design stubs — nothing in `docs/` is imported at runtime.

## What `cbcd` is

A Python library aiming to be the comprehensive home for **constraint-based causal discovery** algorithms — both i.i.d. and time-series. Coverage target: PC and variants (stable, conservative, majority, parallel), FCI and variants (RFCI, FCI+, anytime-FCI, GFCI), CDNOD, IOD, JCI, plus time-series methods (PCMCI, PCMCI+, LPCMCI, J-PCMCI, tsFCI, SVAR-FCI).

Design priorities, in order:

1. **Composition over hardcoding.** Algorithms are wired from a small set of Protocols: `CITest`, `SkeletonAlgorithm`, `ColliderOrienter`, `EdgeRules`, `PAGSkeletonRefinement`. Adding a new algorithm should be mostly plumbing existing pieces, not writing a new monolithic function.
2. **Auditability.** Every algorithm captures a full audit trail (`RunRecorder`) — every CI test, collider decision, and rule firing is logged with timing, persistable to a single `.cbcd` archive, queryable as DataFrames, and resumable from checkpoints.
3. **Correctness over cleverness.** Each algorithm ships with structure-level tests (SHD against d-separation oracle on standard graphs), not just smoke tests.

## Dependency stance

`cbcd` does **not** depend on `causal-learn`, directly or transitively. The `causal-learn` repository at `../../vendor/causal-learn/` is read-only reference material — used to port test fixtures and verify behavioural parity, never imported.

`citests` (`../../citests/`) is a separate user-owned library focused on conditional-independence test research. It currently depends on `causal-learn` for PhD work and will not change for ~6 months. **`cbcd` ships its own minimal CI test layer** (Fisher-Z, χ², G², partial correlation, KCI) for now. After late 2026, `citests` may be extended to interoperate with `cbcd` via the `CITest` Protocol; until then, neither package depends on the other.

## How to navigate this folder

| File | Purpose |
|------|---------|
| `README.md` (this) | Orientation and pointers |
| `audit_causal_learn.md` | Full audit of `causal-learn`'s constraint-based code: what to keep, refactor, rewrite. Includes file:line references to specific bugs and design issues. |
| `design/api_v0.py` | The Phase-1 API design as Python stubs (Protocols, ABCs, dataclasses, function signatures). Nothing executes — it's a contract document. Section §I inside the file is the **decisions log + open questions**. |

## How to read `design/api_v0.py`

Sections are lettered A–J. Read top to bottom on first pass; the design builds incrementally:

- **A.** `CITest` Protocol + caching/recording wrapper.
- **B.** `BackgroundKnowledge` dataclass.
- **C.** `Skeleton` + `SkeletonAlgorithm` Protocol.
- **D.** Graph types: `DAG`, `CPDAG`, `MAG`, `PAG` (plus partial intermediates) — endpoint-mark int8 matrix storage.
- **E.** Collider orientation: `ColliderDecisions` value object + `ColliderOrienter` Protocol + concrete classes (Sepset, MaxP, Conservative, Majority, DefiniteMaxP).
- **F.** Edge orientation rules: `CPDAGRules` (Meek), `PAGRules` (Zhang R1–R10), `PAGSkeletonRefinement` (Possible-D-Sep step).
- **G.** Top-level algorithm composition: `pc`, `fci`, `cdnod`, `jci`, `iod` and their variants. CI test factory + registry.
- **H.** Time-series API: `LaggedVar`, `LaggedDataset`, time-series graph types, `pcmci`, `pcmci_plus`, `lpcmci`, `tsfci`, `svar_fci`, `j_pcmci`.
- **J.** Recording / audit trail: `RunRecorder`, `NullRecorder`, `InMemoryRecorder`, `FileRecorder`, `RunRecord`, `.cbcd` archive format.
- **I.** Decisions log (D1–D12) + remaining open questions (O1–O5). **This is the place to look for "what's been settled and why."**

## When the design changes

1. Update `design/api_v0.py` — both the affected section and the §I decisions log.
2. If the change invalidates a finding from `audit_causal_learn.md`, note it there too.
