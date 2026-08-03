# Implementation spec: FAITHFUL LPCMCI (cell 4) — cbcd-native, clean-room

Replace cbcd's current windowed-FCI `lpcmci` stand-in with **faithful LPCMCI** (Gerhardus & Runge,
NeurIPS 2020; corpus `algo149`). Cell 4 = full · time series · non-sufficient -> **TimeSeriesPAG**
(lagged + contemporaneous + latent-confounder `<->` edges). The high-recall latent-TS champion.

## SOURCE OF TRUTH (read this first, and ONLY this for logic)
`scratchpad/lpcmci_sm_reference.md` — the authors' exact published math: middle-mark semantics (§3.2),
orientation rules R0′–R10′ + APR + MMR (§S4), Algorithms 1/S2/S3/S4 pseudocode (§S5), apds/napds set
definitions (§S7), Lemma S8 middle-mark algebra. Everything you implement must trace to a line there.

## LICENSE FIREWALL (non-negotiable)
cbcd is MIT; tigramite is GPL-3.0. **Never open `tigramite/*.py` to read logic.** Build ONLY from the
SM reference math. tigramite is used solely as a BLACK-BOX output comparator in `parity/lpcmci/`
(construct data, run tigramite `LPCMCI`, compare endpoint matrices — never inspect its source).
`grep -rn tigramite cbcd/` MUST return empty at the end (tigramite only in `parity/`). No tigramite
string-marks (`"-->"`, `"o-o"`, `"x-x"`, `"<--"`, `"o->"` …) anywhere in `cbcd/`.

## What is wrong today
`cbcd/timeseries/lpcmci.py::lpcmci` is a clean-room **windowed FCI / SVAR-FCI** (FAS → colliders →
Possible-D-Sep → time-order arrowheads → Zhang R1–R10). It is oracle-correct (recovers the bidirected
latent edge) but it is NOT LPCMCI: it lacks middle marks, the iterative preliminary phase, the MCI
`S ∪ S_def` conditioning, and the S2/S3 removal discipline — so it has LOWER finite-sample recall,
which is LPCMCI's entire reason to exist. Keep the useful scaffolding (`_GridCITest`, grid encoding,
`mci_skeleton`, `_mci_cond_set`) but build the real algorithm.

## The one hard design decision: the LPCMCI-PAG graph type + homology
LPCMCI operates on a STATIONARY graph where every edge is an equivalence class under time shift, and
each edge carries endpoint marks AND a middle mark. Represent it natively:

- **Grid encoding (reuse existing):** node `v*(τ_max+1) + (-lag)`, `lag ∈ [-τ_max, 0]`, decoded by
  `_decode_grid`. Present slice = lag 0. `LaggedVar(var, lag)` is the human form.
- **Canonical edge:** because of stationarity, every edge has a homologous copy whose LATER endpoint
  is at lag 0. Store the graph keyed by canonical edges `(i, j, τ)` meaning `X^i_{t-τ} — X^j_t`
  (`τ ∈ 0..τ_max`; for `τ=0` keep `i<j`). Each canonical edge holds: mark at the `X^i_{t-τ}` end,
  mark at the `X^j_t` end, and a middle mark. Use cbcd `EndpointMark` for the two endpoint marks
  (ARROW=2, TAIL=1 used for tail, CIRCLE=1 domain — **note**: cbcd currently overloads value 1 for
  both circle and tail context; add an explicit representation if needed to disambiguate tail vs
  circle, since LPCMCI needs all three of {circle, head, tail} plus a conflict mark `x`). Middle marks
  are a NEW small enum/array `{empty, '?', 'L', 'R', '!'}`.
- **Homology helper:** given any two grid nodes `X^a_{la}, X^b_{lb}`, map to the canonical `(i,j,τ)`
  by shifting so the later node sits at lag 0; read/write marks through that canonical key so every
  operation automatically applies to all time-shifted copies. This single helper is what enforces
  stationarity + order-independence (Thm 3). Build and unit-test it FIRST.
- **Path/adjacency expansion:** apds/napds sets and R9′/R10′ need paths over the grid. Expand a
  present node's neighbours from canonical edges; expand a past node `X^k_{t-τ'}`'s neighbours by
  shifting to canonical form. Keep a `neighbors(grid_node)` accessor that respects homology.

Suggested module split (all under `cbcd/timeseries/`, all MIT-native):
- `lpcmci_pag.py` — the LPCMCI-PAG class: endpoint-mark + middle-mark arrays over canonical edges,
  homology map, `neighbors`, `parents`, `adjacencies`, `is_ancestor_mark`, APR/MMR appliers, and
  `to_timeseries_pag()` (strip middle marks → ordinary `TimeSeriesPAG` for return).
- `lpcmci_rules.py` — R0′a–d, R1′, R2′, R3′, R4 (standard for S2/S3 line 22) & R4′, R8′, R9′, R10′,
  each as a pure function proposing orientations/removals; plus the Alg-S4 driver (ordered rule list,
  restart-on-change, conflict→`x`, weak-minimality of removed sepsets). The R0′/R4′ variants that do
  CI tests take the CI callable + `S_def`.
- `lpcmci_sets.py` — `apds`, `napds1`, `napds2` per Definitions S4/S5, plus the modified majority-rule
  sepset-membership queries (search in apds sets, union-in S2/S3 sepsets, order by `I_min`).
- `lpcmci.py` — Algorithm 1 orchestration (init, k preliminary rounds of S2 with parent carry-over,
  final S2 then S3), `I_min`/`SepSet` memories, `RunRecorder` instrumentation, `background` (time
  order is a tier; also honour `no_contemporaneous`/forbidden lag-0), returns `TimeSeriesPAG`.

Reuse cbcd's CI plumbing: the cached `LaggedCITest` + `_mci_cond_set` idea for `S ∪ S_def`; the
`I_min` memory = min |test statistic| across tests (ParCorr → |partial correlation|; keep it in the
CI adapter so it also feeds the recorder). Record every CI test through `RunRecorder` (grid-node ids).

## API (keep the signature; make `k` real)
```
def lpcmci(data, *, ci_test="parcorr", tau_max=None, alpha=0.05, k=4,
           max_cond_set=None, background=None, var_names=None,
           recorder=None, run_id=None) -> TimeSeriesPAG
```
`k` = preliminary iterations (now functional, not a no-op). `tau_max` must equal `data.max_lag`.
Keep `lpcmci_skeleton` and `mci_skeleton` exported (useful primitives). Export `lpcmci` as before.

## Validation / parity gate (build stage-by-stage, validate each)
1. **Homology unit tests** — writing an edge writes all shifted copies; neighbours respect shift.
2. **Rules unit tests** — each R-rule on a hand-built LPCMCI-PAG fixture gives the SM-specified
   orientation; APR/MMR/Lemma-S8 algebra correct; conflict → `x`; weak-minimality trimming correct.
3. **Oracle test (STRUCTURAL GATE):** under a TS m-separation oracle (marginalize latents), `lpcmci`
   returns the true ts-PAG including a bidirected `<->` latent-confounded edge. Sound + complete.
4. **Recall demo:** a concrete autocorrelated latent DGP where faithful `lpcmci` recovers MORE true
   adjacencies (higher recall) than the old windowed-FCI path and than `pcmci_plus` (which cannot even
   represent the `<->` edge). This is the reason the algorithm exists — make it a test.
5. **PARITY vs tigramite LPCMCI** (`parity/lpcmci/run.py`, mirror `parity/pcmciplus`): random
   SVAR-with-latents (marginalize some variables), same ParCorr/τ_max/α/k, compare full ts-PAG
   endpoint matrices; report agreement rate; pinpoint disagreements. Aim high; characterize residual
   as borderline finite-sample flips. **Do not fabricate parity; report the true number.**
6. **Regression:** full suite green (currently 349); `pcmci`/`pcmci_plus` unchanged; ruff + mypy clean.

## Honesty / expectations
LPCMCI is the most intricate algorithm in the suite (middle marks + homology + iterative phases +
majority-rule sepset queries). If exact tigramite parity proves infeasible in one pass, deliver a
version that PASSES gates 1–4 (oracle-sound + higher recall than the stand-in) + a precise report of
where/why finite-sample output diverges from tigramite — do NOT force or fake it. The oracle test is
the real correctness gate; parity is corroboration.

## Scope
Latents, **no selection bias** (Thm 2 assumes none — rules R5/R6/R7 omitted, correct). Stationarity
assumed. `tau_max ≥` true max lag.

Related: [[algorithm-cell-map]] (cell 4). Supersedes [[lpcmci-spec]] (which said "port from tigramite"
— that was a license error; this spec is clean-room from the SM). Finance-critical cell (TS + latent
factors / common shocks).
