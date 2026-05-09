# Suite-level development journal

Cross-package decisions, suite design changes, push/release coordination,
and audit findings that affect more than one package. Newest entry at
the **top**.

For per-package implementation history, see each package's
`docs/journal.md`. cbcd's is the reference style.

---

## 2026-05-09 — bnm v0.2.2 viz polish (later same day)

**bnm 0.2.2.dev0.** Closed all remaining v0.2.x viz items from the
suite-level scoping conversation:

- `plot_side_by_side` gains `mode: Literal["matches", "diff", "none"]`
  (replaces the `highlight_true_positives: bool` kwarg). `"diff"`
  highlights additions/deletions/reversals/kind-changes; useful for
  "show me what changed".
- `highlight_node_color` / `highlight_edge_color` kwargs on
  `plot_graph` and `plot_side_by_side`. Defaults preserve pastel.
- CIRCLE-edge matching now compares the full `(mij, mji)` pair so
  different PAG topologies don't false-match (was a latent bug since
  Slice 4).

535 tests pass (+12 v0.2.2 viz). Suite parity 5/5 within bounds —
viz layer doesn't touch metric numerics. Design doc §L marks O1 and
O2 resolved. See `bnm/docs/journal.md` for the full entry and the
migration table for the kwarg rename. Pre-release breaking change
(no PyPI users affected).

**Push policy unchanged.** v0.2.2.dev0 is local-only.

---

## 2026-05-09 — bnm v0.2.1 perf fix; suite docs corrected

**bnm 0.2.1.dev0.** Landed the deferred performance optimisation flagged
in `bnm/docs/journal.md` (2026-05-07): `bnm.compare()` now normalises
external GraphLike inputs to `_Graph` instances exactly once, and
`_validate_endpoints` is fully vectorised. Per-node compare on a
1000-node external graph went from minutes to seconds; n=200 from
seconds to <0.1s. No public API change. 523 tests pass (+4 perf
regression guards). See `bnm/docs/journal.md` for the full entry.

**Suite parity harness re-run** still 5/5 fixtures within bounds with
the optimised compare path — the optimisation produces identical
numeric output, only the time budget changed.

**Suite-doc drift corrected.** This file's earlier 0.2.0 entries
(2026-05-07 and 2026-05-08) said `bnm.compare()` and `to_dataframe()`
were "deferred to v0.2.1 or later". They actually shipped on
2026-05-07 along with `compare_models_descriptive`,
`compare_models_comparative`, `analyse_mb`, and the four `use cases/`
notebook migrations — see `bnm/docs/journal.md` 2026-05-07 entries.
The actual remaining 0.2.x viz items (O1: CIRCLE-mark rendering
convention; O2: `plot_side_by_side` difference mode; highlight-color
kwarg) are now tracked for **bnm v0.2.2**, not v0.2.1. The
`suite/CLAUDE.md` "currently in flight" section will be touched on
the next refresh — leaving it for a deliberate sweep rather than
inline-patching.

**Push policy unchanged.** No bnm push (or any other package's first
push) without explicit user instruction. v0.2.1.dev0 is local-only.

---

## 2026-05-08 — Suite integration test at `parity/suite/run.py`

Closed the last suite-level gate before the first public push. The
integration script chains all four packages — dagsampler → citk → cbcd
→ bnm — on a 5-fixture set (collider_3, fork_3, chain_3, diamond_4,
asia_like_5) and asserts per-fixture SHD/F1 bounds. Layout follows
`cbcd/parity/<area>/run.py` convention.

**Live run on calibration day:** 5/5 fixtures within bounds, exit 0.

```
collider_3      n= 2000  SHD= 0/ 1 ✓   F1=1.00/0.90 ✓
fork_3          n= 3000  SHD= 2/ 3 ✓   F1=0.00/0.00 ✓
chain_3         n= 6000  SHD= 0/ 2 ✓   F1=1.00/0.50 ✓
diamond_4       n= 3000  SHD= 2/ 4 ✓   F1=0.50/0.30 ✓
asia_like_5     n= 3000  SHD= 5/ 8 ✓   F1=0.40/0.25 ✓
```

**Bound-setting policy.** This is a regression detector, not an
algorithmic-precision benchmark. Bounds were calibrated 2026-05-08
with seed_structure=1, seed_data=2, alpha=0.05, and given ~50–100%
SHD headroom over the observed FisherZ recoveries. Each fixture's
"observed" line is preserved in a comment in `run.py` as the
regression baseline. Some fixtures (fork_3) accept high SHD because
PC's orientation of CPDAG-undirected edges drifts under FisherZ
noise — that's a known statistical-recovery quirk, not a wiring bug.

**Structural-Protocol assertion.** The script also runtime-checks
`isinstance(gen.as_ci_oracle(), cbcd.CITest)` and
`isinstance(citk.FisherZ(data), cbcd.CITest)` before each fixture
runs. If either ever returns False, the suite's no-cross-package-
imports architecture has broken. This pins the contract.

**Layout.** `suite/parity/suite/` is a uv-managed runner project
(`package = false`) whose `pyproject.toml` declares the four sister
packages as editable path deps. `uv sync && uv run python run.py`
provisions the venv and runs the harness. README at
`suite/parity/suite/README.md` covers the run procedure and how to
adjust bounds.

**All three suite-level gates from the 2026-05-07 entry are now
closed.** First public push of cbcd / citk / bnm / dagsampler 0.2.0
remains contingent on explicit user instruction per package, per
the no-push contract.

---

## 2026-05-08 — End-to-end tutorial at `docs/tutorial.md`

Wrote the 10-line cross-package story now that `as_ci_oracle()` is in
place. Sits at `suite/docs/tutorial.md`. **All four packages**
(dagsampler, citk, cbcd, bnm) participate — citk supplies the
empirical FisherZ test, demonstrating the second `CITest`-Protocol
arrow in addition to dagsampler's d-sep oracle.

The canonical example: a 3-node collider `A → C ← B`, n=3000, recovers
exactly under both PC paths (oracle and citk's FisherZ) and yields
`SHD: 0, F1: 1.0`. The tutorial code is run verbatim before each
journal update — the printed numbers in the doc are real, not
illustrative. Verified `isinstance(citk.FisherZ(data), cbcd.CITest)`
passes; no adapter shim required.

**Visualisation pass (later same day).** Added a "Visualize the
comparison" section to the tutorial with two paired figures via
`bnm.plot_side_by_side`:

- collider (clean recovery): both panels render identically — the
  visual counterpart of `SHD: 0`.
- diamond (noisier): two TP edges into `D` highlighted in pastel
  red, two upper edges flipped — the visual counterpart of `SHD=2,
  F1=0.50`.

Figures live at `suite/docs/figures/tutorial_{collider,diamond}_{true_cpdag,recovered}.{svg,png}`.
Tutorial embeds the PNGs in a 2-column markdown table for clean
side-by-side display in any markdown renderer.

bnm's highlight palette was switched to pastel (`#c8e6c9` for node
fill, `#f08080` for matching-edge stroke) at the same time so the
embedded figures read well in light-themed docs. See bnm's journal
for details; all 519 bnm tests still pass, suite parity harness
unaffected.

**The pedagogical move worth flagging:** the tutorial uses
`pc(data, ci_test=gen.as_ci_oracle())` as the **gold-standard**
recovery rather than constructing a true CPDAG by hand. That works
because PC under a perfect d-separation oracle returns the true
CPDAG by construction (cbcd's structural-regression bar). This
sidesteps the missing `DAG.to_cpdag()` converter in cbcd and gives
a fully apples-to-apples comparison against the empirical FisherZ
recovery, in two `pc()` calls.

**Remaining suite-level work before the first public push:**

1. ~~dagsampler `as_ci_oracle()`~~ — done.
2. ~~End-to-end tutorial~~ — done.
3. **`suite/parity/suite/run.py` integration test** — chain
   dagsampler → cbcd → bnm on a fixture set, assert SHD/F1 bounds.
4. **First public push** of cbcd then citk then bnm (then dagsampler
   0.2.0). Push policy unchanged: each package waits on explicit
   user instruction naming it.

**Side note for the future:** if cbcd later adds a public
`DAG.to_cpdag()` (or a `cpdag_of(dag)` helper), the tutorial can
collapse from two `pc()` calls to one and a direct conversion.
Filed in cbcd's territory; not actioned here.

---

## 2026-05-08 — dagsampler `as_ci_oracle()` lands; second cross-package Protocol wired

Closed one of the three remaining suite-level gates from the
2026-05-07 entry: `CausalDataGenerator.as_ci_oracle()` now returns a
`DSeparationOracle` that conforms structurally to the `cbcd.CITest`
Protocol. End-to-end check passes:
`cbcd.pc(data, ci_test=gen.as_ci_oracle())` recovers the correct
CPDAG with `isinstance(oracle, cbcd.CITest) is True` — and dagsampler
still has zero `import cbcd` lines.

**Bumped dagsampler 0.1.0 → 0.2.0.** Per-package details and rationale
in `dagsampler/docs/journal.md` (first entry of that file).

**Cross-package boundaries now wired:**

| boundary | status |
|---|---|
| `citk → cbcd` (`cbcd.CITest`) | wired (cbcd D15) |
| `dagsampler → cbcd` (CI oracle via `CITest`) | **wired (this entry)** |
| `cbcd → bnm` / `dagsampler → bnm` (`bnm.GraphLike`) | wired (bnm v0.2, 2026-05-07) |

All three structural-Protocol arrows in the suite diagram are now
load-bearing rather than aspirational.

**Remaining suite-level work before the first public push:**

1. ~~dagsampler `as_ci_oracle()`~~ — done.
2. **`suite/parity/suite/run.py` integration test** — chain
   dagsampler → cbcd → bnm on a fixture set, assert SHD/F1 bounds.
3. **End-to-end tutorial** at `suite/docs/tutorial.md` — the 10-line
   cross-package story, now writable using `as_ci_oracle()` directly.
4. **First public push** of cbcd then citk then bnm. dagsampler 0.2.0
   also waits for explicit user instruction.

---

## 2026-05-07 — bnm v0.2 feature complete; gating-package status cleared

bnm v0.2 rewrite finished end-to-end in a single session. **427 tests
pass**, mypy clean, ruff clean. See `bnm/docs/journal.md` for the per-
slice details.

**Deliverables (chronological):**

1. Slice 0 — `tests/fixtures_legacy.json` frozen (82 fixtures, 87
   pairs) under `PYTHONHASHSEED=0`. Generator script preserved at
   `bnm/scripts/generate_legacy_snapshot.py`; legacy 0.1.x source
   relocated to `bnm/scripts/legacy_0_1_x/` so the generator stays
   runnable without git checkouts.
2. Audit at `bnm/docs/audit.md`. **8 bugs** catalogued in 0.1.x with
   file:line refs. §1 (SID empty-parents crash), §6 (hash-seed
   non-determinism in SID), §7 (reversal under-counting from
   nx-storage-direction asymmetry), §8 (SID upper-bound under-counting
   on CPDAG inputs) all fixed and verified in v0.2.
3. Design doc at `bnm/docs/design/api_v0.py`. Sectioned A–L; 11
   decisions, 5 open questions deferred.
4. Slice 1 — `bnm.GraphLike` Protocol + adapter + 11 descriptive
   metrics + Markov blanket + 215 tests. Cross-package interop
   verified (cbcd `DAG` → bnm functions with zero conversion, zero
   imports).
5. Slice 2 — comparative metrics (SHD, HD, F1, P/R, TP/FP/FN,
   additions/deletions/reversals) + 99 new tests. Bug §7 caught.
6. Slice 3 — SID port onto int8 endpoint matrix natively + 100 new
   tests. Bug §1 fixed (5 fixtures reclaimed), §6 fixed (deterministic
   component iteration), §8 caught (legacy upper-bound under-counting
   verified by hand-computing chain DAGs in equivalence class).
7. Slice 4 — viz: `plot_graph`, `plot_side_by_side`,
   `plot_sid_matrix`. Lazy-imports `viz` extra deps. 13 smoke tests
   that skip cleanly without the extra installed.

**Architectural pattern verified:** the structural-Protocol contract
between cbcd and bnm works. `cbcd.graph.dag.DAG(...)` instances
satisfy `bnm.GraphLike` and pass directly into every bnm function
with no conversion, no shim, and no imports between the two packages.
This is the load-bearing claim from the suite design.

**Build system:** bnm migrated to `pyproject.toml` + `hatchling` +
`uv` + Python ≥ 3.11, matching cbcd's conventions. Hard deps: numpy.
Soft extras: `[networkx]`, `[pandas]`, `[viz]`, `[docs]`, `[dev]`.

**Tests/fixtures_legacy_v02_overrides.json** is now the canonical
record of intentional v0.2-vs-0.1.x semantic divergences: 19 pairs
with at least one override (17 for §7's reversals/shd, 11 for §1+§6+§8
SID values, with overlap between groups).

**Remaining suite-level work** (gating the first public push):

1. **`suite/parity/suite/run.py` integration test** — chain
   dagsampler → cbcd → bnm on a fixture set, assert SHD/F1 bounds.
2. **End-to-end tutorial** at `suite/docs/tutorial.md` — the 10-line
   cross-package story.
3. **First public push** of cbcd then citk then bnm. Push policy
   unchanged — no package pushes without explicit user instruction
   naming the package.

**Open questions deferred from the bnm rewrite** (see
`bnm/docs/design/api_v0.py` §L for the full list):
- O1, O2 (Slice 4 viz refinements): CIRCLE-mark rendering convention,
  difference-mode side-by-side viz.
- O3 (deferred): SID generalisation to PAGs.
- O4 (deferred): time-series GraphLike for cbcd's `TimeSeriesCPDAG`.

The `compare()` façade and `to_dataframe()` were specified in the
design doc but not implemented in this round; defer to v0.2.1 or
later.

---

## 2026-05-06 — Realistic positioning of the suite

Talked through what would make this "the best constraint-based
causal discovery suite ever," and pushed back on the framing.

**Position taken:** "comprehensive coverage" is the wrong target.
The literature has a long tail of methods that almost nobody uses;
faithfully implementing all of them buys a thicker README and almost
no real users while diluting the correctness bar that's currently
the suite's main differentiator. `causal-learn` is comprehensive-ish
and had 2 real bugs + multiple O(2^n) hot paths + broken tests; users
who hit those switch libraries. The suite's credibility comes from
*every* shipped algorithm matching SHD = 0 against an oracle and
parity against a reference — that bar can't survive "everything
ever."

**Better goal:** "the most rigorous and most-used constraint-based
suite," with three sub-goals in priority order:

1. **Coverage that matters** — every method anyone actually runs,
   roughly 50–80 things across all four packages, not hundreds.
   Constraint-based: PC family, FCI family, PCMCI family, CDNOD,
   GFCI, JCI, IOD. CI tests: linear-Gaussian, χ²/G², KCI, GPDC,
   regression, CMI-knn, mixed-data. Metrics: SHD, F1, SID, MEC
   equivalence. Simulation: structures + mechanisms + noise covering
   the empirical-literature benchmarks.
2. **Correctness as the differentiator** — every published algorithm
   carries a parity result against a reference if one exists, a
   structural-regression bar against an oracle, and a reproduction
   of at least one published benchmark.
3. **Compounding adoption** — publish all four this quarter, write
   the suite tutorial, get one outside user. Each compounds.

**Recommended cadence:** scaffold finished first (bnm rewrite,
dagsampler `as_ci_oracle`, suite tutorial — 2-3 weeks), then publish,
then pick the next 3 algorithms by usage data, not by completeness.
Re-evaluate every six months.

This reframing is what the suite-level CLAUDE.md and roadmap should
target going forward. "Best ever" is achievable but not yet earned;
the architecture is good enough that the work compounds.

---

## 2026-05-06 — bnm rewrite scheduled (`0.1.x` → `0.2`)

Audit of the existing bnm:

- Public API: `BNMetrics` class + functions (`shd`, `hd`, `precision`,
  `recall`, `f1_score`, `sid`, `count_colliders`, etc.) plus
  `compare_models_*` for visualization.
- **Input format: `networkx.DiGraph`** with edge attribute
  `type ∈ {'directed', 'undirected'}`. Not cbcd's int8 endpoint
  matrix; not causal-learn's signed-int matrix.
- **Distinctive features cbcd doesn't have**: SID (Structural
  Intervention Distance, ~270 lines of careful path-matrix
  algorithm), local Markov-blanket comparisons, side-by-side viz.
  These are the load-bearing parts to preserve.
- **Overlapping utilities**: `generate_random_dag`,
  `generate_synthetic_data_from_dag` (dagsampler does both better),
  `dag_to_cpdag` (cbcd has `DAG.to_cpdag()`). To drop in v0.2.
- **No design doc, no Protocol adapter, no `tests/` parity** with
  cbcd / citk.

**Decision: full rewrite to v0.2**, mirroring cbcd's slice process:

1. Audit pass (catalogue every metric, separate "port-as-is" from
   "rewrite cleanly" from "drop").
2. Design doc at `bnm/docs/design/api_v0.py` — Protocols,
   dataclasses, function signatures, no executable bodies.
3. Implement in slices:
   - **Slice 1**: `bnm.GraphLike` Protocol + `_to_endpoints` adapter
     (accepts cbcd graphs, networkx DiGraph, or raw int8 matrices) +
     descriptive metrics. Ship green tests.
   - **Slice 2**: comparative metrics (SHD, HD, F1, precision,
     recall, TP/FP/FN). Tests against hand-computed values + cross-
     check vs old bnm 0.1.x on a fixture set.
   - **Slice 3**: SID port. Big test surface (port DAGMetrics R
     outputs as fixtures if available).
   - **Slice 4**: visualization.

The rewrite specifically targets the **internal storage convention**:
move from networkx-DiGraph-with-edge-attribute to cbcd's int8
endpoint matrix natively, so bnm and cbcd share the same canonical
shape and metrics work directly on cbcd output with zero conversion
overhead.

Versioning: bump to **0.2.0** with the rewrite, document breaking
changes prominently in `CHANGELOG` and README. bnm hasn't yet made
a D15-style API stability commitment; the commit happens *after* the
rewrite, on the new surface.

Estimated effort: 2–3 focused sessions, similar to a cbcd vertical
slice.

---

## 2026-05-06 — Suite design: structural Protocols at every cross-package boundary

Settled on the architectural pattern that the suite will follow as it
publishes:

```
dagsampler ─── true_dag, data, ci_oracle ──▶ cbcd ─── recovered ──▶ bnm
                                              ▲                     ▲
                                              │ Protocol            │ Protocol
                                            citk                  (graph)
```

Every arrow is a structural Protocol. **No package imports another.**
This is already in place between cbcd and citk via `cbcd.CITest`
(committed under D15); it should extend to:

| boundary | contract | who defines it |
|---|---|---|
| `citk → cbcd` | `cbcd.CITest` Protocol (`n_vars`, `__call__`, `details`) | cbcd (frozen under D15) |
| `dagsampler → cbcd` (CI oracle) | `cbcd.CITest` Protocol — dagsampler exposes `as_ci_oracle()` returning a callable conforming object | cbcd |
| `dagsampler → bnm` / `cbcd → bnm` (graphs) | `bnm.GraphLike` Protocol (`n_vars`, `endpoints` int8 matrix) | bnm (TBD; defined as part of the rewrite) |

Concrete asks falling out of this:

- **dagsampler `as_ci_oracle()`**: small change wrapping the
  precomputed d-sep table as a `CITest`-conforming object. Bumps
  dagsampler to 0.2.
- **bnm `GraphLike` Protocol + `_to_endpoints` converter** in v0.2
  (during the rewrite). Accepts cbcd graphs directly; backwards-
  compatible with networkx-DiGraph users via the converter.

If a `from <other_package> import …` line shows up inside any of
the four packages, treat it as a smell and revisit the Protocol
contract.

---

## 2026-05-06 — Suite scaffolded

Moved from per-package directories under `~/Projects/` into a single
umbrella at `~/Projects/suite/`:

- `~/Projects/cbcd/`        → `~/Projects/suite/cbcd/`
- `~/Projects/citk/`        → `~/Projects/suite/citk/`
- `~/Projects/simulation/`  → `~/Projects/suite/dagsampler/` (renamed
  to match the package name)
- `~/Projects/vendor/`      → `~/Projects/suite/vendor/`
- `~/Projects/suite/bnm/`   ← cloned fresh from
  `github.com/averinpa/bnm` (legacy `0.1.x`; rewrite pending)

Suite-level files added:

- `suite/CLAUDE.md` — instructions for any Claude session opened from
  the umbrella. Lists the four packages, their roles, the Protocol
  contracts at every cross-package boundary, the working pattern.
- `suite/journal.md` — this file. Cross-package decisions go here;
  per-package commits stay in each package's own `docs/journal.md`.

Auto-memory replicated to the new cwd hashes:
`~/.claude/projects/-Users-pavelaverin-Projects-suite/memory/` and
`~/.claude/projects/-Users-pavelaverin-Projects-suite-{cbcd,citk,bnm}/memory/`
all hold the no-push feedback memory for backwards compatibility.

No code changed; no git history affected. Each package's git remote,
local commits, and venv are unchanged.

### Suite picture (carried forward from the design discussion)

```
dagsampler ─── true_dag, data, ci_oracle ──▶ cbcd ─── recovered ──▶ bnm
                                              ▲                     ▲
                                              │ Protocol            │ Protocol
                                            citk                  (graph)
```

Every arrow is a structural Protocol. No package imports another.

### Currently in flight

1. **bnm v0.1.x → v0.2 rewrite.** Audit-first, then design doc, then
   implement in slices. Mirror cbcd's M1/M2/M3 process (audit pass →
   `docs/design/api_v0.py` → vertical slices with structural
   regression tests). Distinguishing parts to preserve: SID metric,
   local Markov-blanket comparisons, viz. Drop dagsampler-overlapping
   utilities (`generate_random_dag`, `generate_synthetic_data_from_dag`).
   Switch internal representation from `networkx.DiGraph` to cbcd's
   int8 endpoint matrix convention.

2. **dagsampler `as_ci_oracle()` method.** Small change, big leverage:
   exposes the precomputed d-sep table as an object satisfying
   `cbcd.CITest`, so `cbcd.pc(data, ci_test=gen.as_ci_oracle())` works
   directly without writing a custom oracle. Bumps dagsampler to 0.2.

3. **Suite-level integration test.** Once bnm v0.2 lands, add
   `suite/parity/suite/run.py` that chains dagsampler → cbcd → bnm on a
   fixture set and asserts SHD/F1 bounds. Catches integration regressions
   without coupling versions.

4. **End-to-end tutorial.** `suite/docs/tutorial.md` with the 10-line
   cross-package story. Load-bearing user-facing doc; without it, the
   suite reads as four unrelated packages.

5. **First public push** of cbcd and citk (in that order). Gate is
   currently the bnm rewrite — once bnm v0.2 is at parity with the
   others, the suite goes public as one ecosystem rather than three
   ready packages and one legacy.

### Push policy

No package is pushed to its remote without an explicit user instruction
naming that specific package. See
`~/.claude/projects/-Users-pavelaverin-Projects-suite/memory/feedback_no_push.md`
for the full contract.
