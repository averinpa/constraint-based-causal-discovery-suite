# Implementation spec: Cell 5 hardening (local · iid · sufficient)

Cell 5 = **local · i.i.d. · causally-sufficient · (coordinated) multi-target** causal discovery →
local **CPDAG** around a query target set. The champion (**CML**, `cbcd/algorithms/cml.py`, corpus
algo109) is already built and, per the 2026-07-17 falsification sweep, is the correct best-in-class
pick for this exact cell (only proven sound+complete *coordinated multi-target* local-CPDAG method).
`local_discovery` (region-grow, `cbcd/algorithms/local.py`) is the secondary. **This task HARDENS the
cell, it does not replace the algorithm.**

## HARD CONSTRAINT: everything is built on cbcd's own protocols
No new abstractions, no bespoke graph/test/recorder types. Use these cbcd contracts BY NAME:
- **CI tests:** the `CITest` protocol (`cbcd/citest/protocol.py`): `__call__(x, y, S) -> float`
  (p-value), `details(...) -> CITestResult`, optional `is_cached(...)`. Every test in the cell goes
  through this — never call an estimator directly.
- **The soundness oracle:** `tests/oracle.py::DSeparationOracle` — an existing `CITest` implementation
  (p=1.0 iff d-separated on a known DAG, via networkx). This IS the sound reference machinery; do not
  hand-roll a new oracle.
- **The sound reference graph:** cbcd's own `pc()` (`cbcd/algorithms/pc.py`) run on the FULL graph
  under the same `DSeparationOracle`, then restricted to the region — this is the ground-truth local
  CPDAG. (Mirror the existing `test_local_full_region_matches_global`, `tests/test_local_discovery.py`.)
- **Graphs:** `CPDAG` (`cbcd/graph/cpdag.py`), endpoint marks via `EndpointMark`. Compare via endpoint
  matrices, as the rest of the suite does.
- **Instrumentation:** the `RunRecorder` protocol (`cbcd/recording.py`: `begin_run`/`record_ci`/
  `finish_run`) — already threaded through `cml`/`local_discovery` via `iid_run` (`cbcd/_run.py`).
- **Background knowledge:** `BackgroundKnowledge` (`cbcd/background.py`: `forbidden_directed`,
  `required_directed`, `forbidden_adjacent`, `tiers`).
- MB primitives (if Task 4): `cbcd/mb.py` (`iamb`, `inter_iamb`, `grow_shrink`, `mmpc`).

If any check needs behaviour cbcd doesn't expose, EXTEND cbcd's own code (like `MajorityColliderOrienter`
was added for PCMCI+), don't fork a private copy. cbcd is MIT; keep it clean-room.

---

## Task 1 (MUST) — Soundness + COORDINATION stress, committed
Mirror the cell-4 treatment (`tests/timeseries/test_lpcmci_soundness.py`) for the i.i.d. local cell.

- Random DAGs (vary n, density, target-set size), a `DSeparationOracle` over each, `alpha` irrelevant
  at oracle. For each: run `cml(...)` and `local_discovery(...)` on a target SET, and compare to the
  **region-restriction of `pc()`** run on the full graph under the same oracle.
- **Assert 0 wrong-orientation violations** across the battery: every committed endpoint in the local
  CPDAG must match the global-restricted CPDAG (a `TAIL`/`ARROW` the reference doesn't have = a
  violation). Report the true count; **0 is the gate**.
- **THE cell-5-specific subtlety — exercise COORDINATED MULTI-TARGET queries.** The stress MUST include
  target sets whose neighborhoods **overlap**, because CML's coordination (consistent orientation
  across shared boundaries / between-neighborhood inducing-path edges) is exactly where a bug would
  hide — and a single-target-only stress would pass while missing it. Include: (a) single target,
  (b) disjoint-neighborhood targets, (c) **overlapping-neighborhood targets**, and assert boundary
  orientations are consistent (no node oriented one way from neighborhood A and the opposite from B).
- Commit as `tests/test_local_discovery_soundness.py` (fixed seed, bounded N for CI time), with a
  completeness characterization (exact-match % vs the global-restricted reference) as a loose
  regression guard.
- Route every CI query through the `CITest`/`RunRecorder` path already in the algorithms — do not
  bypass instrumentation.

## Task 2 (MUST) — CML docstring precision + defensive citations
Edit `cbcd/algorithms/cml.py`'s docstring: replace unqualified "best-in-class" with the exact claim —
**"sound + complete under faithfulness, causal sufficiency, and Assumption 1 (no inducing path between
two same-neighbourhood nodes routed through a different neighbourhood); completeness relative to the
constructed neighbourhood graph G*_N."** Add: (a) it **sidesteps the LocalPC unsoundness** because its
phase-2 separating-set search is over N¹ = the full Markov blanket (spouses included), so LOAD's
critique (algo113 App. A) does not apply; (b) pre-cite **LDECC** (unsound+incomplete, LOAD App. A) and
**LOAD/SNAP** (algo113/algo151 — different output: adjustment set, not local CPDAG) so reviewers'
"why not X?" is answered in-code.

## Task 3 (MUST) — Background-knowledge tests (4-test MARVEL pattern)
`background` is threaded into `cml`/`local_discovery(_latent)` but untested. Mirror MARVEL's four BK
tests (`tests/test_marvel_background.py`) using cbcd `BackgroundKnowledge`, adapted to the local cell:
(1) local result honours `required_directed` / `forbidden_directed` inside the region;
(2) `forbidden_adjacent` effect; (3) tier ordering respected; (4) `background=None` is a regression
(unchanged output). Note honestly whether `background` is consulted during region-grow or only at
orientation (the local analog of MARVEL's co-parent nuance) — if region-grow ignores it, document that.

## Task 4 (OPTIONAL — judgment call, likely defer) — CMB anchor
Only if a sound+complete *minimal-assumption* anchor is wanted (CMB needs no Assumption 1). Build **CMB**
(Causal Markov Blanket, Gao & Ji 2015, algo110) **natively on cbcd protocols**: MB via `cbcd/mb.py`
(`grow_shrink`/`iamb`), orientation via cbcd's shared machinery, output a `CPDAG`, `iid_run` +
`RunRecorder` + `CITest` + `background` threaded exactly like `cml`. It is **single-target** (a
sanity/cross-check anchor, NOT a co-champion — it does not satisfy the multi-target-coordination
requirement). Skip unless explicitly requested; Task 1's global-restricted oracle already provides an
independent sound+complete reference.

---

## Validation gates
1. Task-1 soundness = **0 violations** across the battery incl. overlapping-neighbourhood targets;
   committed test green.
2. Coordination check: consistent boundary orientation on overlapping-neighbourhood target sets.
3. Full suite green (currently 374); ruff + mypy clean; `grep -rn tigramite cbcd/` empty.
4. No new abstractions — every deliverable uses the cbcd contracts named above.

## Report back
(a) Task-1 soundness violation count (must be 0) + completeness %; (b) committed test name + what it
asserts, esp. the overlapping-neighbourhood coordination case; (c) docstring change; (d) BK test
results + the honest region-grow-vs-orientation note; (e) whether Task 4 was done or deferred;
(f) suite count, grep, ruff/mypy.

## Framing
Cell 5 is a **control-group cell, not the thesis** — goal is "proven sound + defensible, cheaply."
Don't over-invest; the contribution is cells 7–8 + GFCM. Related: [[cbcd-8cell-buildout-status]],
[[algorithm-cell-map]].
