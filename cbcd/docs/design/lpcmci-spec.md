# Implementation spec: faithful LPCMCI (cell 4)

Complete cbcd's half-built `lpcmci` into **faithful LPCMCI** (Gerhardus & Runge, NeurIPS 2020; corpus
`algo149`). Cell 4 = full · time series · non-sufficient -> **TimeSeriesPAG** (lagged + contemporaneous
+ latent-confounder bidirected edges). The high-recall latent-TS champion; the FCI analog for time
series. Pure constraint-based.

## Current state (what's wrong today)
- `cbcd.lpcmci` is **not exported** and the existing `cbcd/timeseries/lpcmci.py` is a **sound
  tsFCI/SVAR-FCI stand-in**, NOT faithful LPCMCI (its own docstring: "wiring MCI recall in + full
  tigramite parity is 5d"). No `parity/lpcmci` harness. The 8 passing tests validate the *stand-in*.
- `mci_skeleton` (the MCI-recall skeleton) exists and is the reusable starting point.

## Build approach: PORT FROM TIGRAMITE, validate by parity
LPCMCI's exact orientation rules and the S2/S3 removal sub-algorithms are in the paper's Supplementary
Material (not reliably in the corpus md). **Do not re-derive from the paper** — port tigramite's
authoritative `LPCMCI` implementation faithfully to cbcd, swapping only the plumbing, and gate on
exact parity with tigramite. Use `algo149` §3.2-3.4 to *understand* the design (below), tigramite for
the exact logic.

## Algorithm design (algo149 §3.2-3.4, for understanding)
- **Middle marks + LPCMCI-PAGs:** every edge carries a middle mark (`?`, `L`, `R`, `!`, empty)
  encoding intermediate (non-)ancestorship, so orientations can be applied *early and iteratively*
  and reused. Total order = time order (`X_{t-τ} < X_t` iff τ>0, ties by index). Lagged links init as
  `→` (L-mark), contemporaneous as `o-o`. Converged => all middle marks empty => a standard PAG.
- **Two effect-size principles:** discovered (non-)ancestorships (a) CONSTRAIN conditioning sets
  (drop non-ancestors) and (b) EXTEND them with the tested pair's known PARENTS (`S ∪ S_def`,
  `S_def = pa(pair)`), which removes autocorrelation and boosts recall (the MCI idea).
- **Algorithm 1:** init complete graph; **preliminary phase** = k iterations of {Alg S2 (remove
  links + apply orientation rules) ; re-init C(G) but carry over discovered parentships}; **final
  phase** = one more Alg S2 then Alg S3 (second removal, like SVAR-FCI's Possible-D-Sep for
  non-ancestor pairs). Returns the PAG. `k` = # preliminary iterations (hyperparameter).
- **Stationarity enforced:** every removal/orientation is applied to all homologous (time-shifted)
  edges. **Order-independent** (Thm 3). Sound + complete under faithfulness, no selection bias (Thm 2).

## cbcd integration
- Reuse `mci_skeleton` (MCI-recall skeleton) as the skeleton engine; `TimeSeriesPAG`
  (`cbcd/timeseries/graph.py`) as output; the cached lagged CI test + `RunRecorder`; `background`
  (time order as tier). Replace/supersede the current tsFCI stand-in in `lpcmci.py`.
- The middle-mark machinery is new state to add (per-edge middle marks + the LPCMCI orientation rules
  ported from tigramite). Keep it internal; the returned `TimeSeriesPAG` has ordinary PAG marks.

## API
```
def lpcmci(data, *, ci_test="parcorr", tau_max, alpha=0.05, k=4,
           max_cond_set=None, background=None, var_names=None,
           recorder=None, run_id=None) -> TimeSeriesPAG
```
`k` = preliminary iterations (tigramite default). **Export `cbcd.lpcmci`** from `cbcd/timeseries/__init__.py`
and `cbcd/__init__.py` (it currently isn't).

## Validation / parity gate
1. **PRIMARY — parity vs tigramite `LPCMCI`.** Add `parity/lpcmci/run.py` (mirror `parity/pcmciplus`):
   random SVAR-with-latents (marginalize some variables), same ParCorr / `tau_max` / `alpha` / `k`,
   compare the full time-series PAG endpoint matrices. Report agreement rate; pinpoint disagreements.
2. **Oracle test:** under a time-series m-separation oracle (latents marginalized), recover the true
   ts-PAG including a bidirected (latent-confounded) edge. The structural gate.
3. **Regression:** the existing timeseries suite stays green; replacing the stand-in must not break
   `pcmci`/`pcmci_plus`.
4. **Latent-coverage demo:** a DGP with a latent confounder that surfaces as a bidirected edge which
   the sufficient methods (`pcmci`/`pcmci_plus`) cannot represent but `lpcmci` recovers.

## Honesty / expectations
- Like PCMCI+, **tigramite is a peer implementation, not ground truth on finite samples** — expect
  borderline single-edge finite-sample differences on weak edges; the **oracle test is the structural
  gate**. Aim for high finite-sample agreement + exact oracle recovery; characterize any residual.
- LPCMCI is intricate and iterative; if faithful parity proves infeasible in one pass, deliver a
  correct-at-oracle version + a precise report of where/why it diverges from tigramite, rather than
  forcing it.

## Scope
- Latents, **no selection bias** (Thm 2). Stationarity assumed. `tau_max` >= true max lag.

Related: [[algorithm-cell-map]] (cell 4 = LPCMCI). Paper `algo149`. Finance-critical cell (TS + latent factors).
