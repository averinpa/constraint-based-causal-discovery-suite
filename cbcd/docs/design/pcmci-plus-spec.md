# Implementation spec: PCMCI+ (cell 3 champion)

Native cbcd implementation of **PCMCI+** (Runge, UAI 2020; corpus `algo148`). Cell 3 = full · time
series · sufficient -> **TimeSeriesCPDAG**. It upgrades cbcd's lagged-only `pcmci` (`algo61`, the
anchor) by adding **contemporaneous** (same-timestep) link discovery + orientation. Pure
constraint-based. Reference implementation to parity against: **tigramite `PCMCIplus`** (installed).

## What it adds over `pcmci`
cbcd's `pcmci` recovers only lagged edges `X^i_{t-τ} -> X^j_t` (τ>0), which are auto-oriented by time
order. PCMCI+ additionally recovers **contemporaneous** edges `X^i_t -o X^j_t` and orients them with
collider + Meek rules (time order gives no direction at τ=0). At daily/coarse sampling, much real
structure is contemporaneous, so this is the completeness upgrade for the sufficient-TS cell.

## Algorithm (algo148 §3.1: Alg 1 + Alg 2 + orientation)
Two skeleton phases, then PC-style orientation. Fully order-independent (PC-stable variant).

**Phase 1 - lagged conditioning (Alg 1):** for each variable `X^j_t`, estimate its lagged parents
`B̂⁻_t(X^j)` by testing lagged pairs `(X^i_{t-τ}, X^j_t)`, τ>0, conditioning greedily on only the
**strongest p** current adjacencies (not all p-subsets). Polynomial cost. **This is essentially what
cbcd's `pcmci` PC1 phase already computes — reuse it to produce `B̂⁻` per node.**

**Phase 2 - contemporaneous conditioning with MCI (Alg 2):** initialize the graph with **all
contemporaneous adjacencies** + the lagged adjacencies from `B̂⁻`. Test each adjacent pair (lagged
unordered τ>0; contemporaneous **ordered** τ=0) iterating only through **contemporaneous** subsets
`S ⊆ A_t(X^j)`, using the **MCI test**:

```
X^i_{t-τ} ⊥ X^j_t  |  S,  B̂⁻_t(X^j)\{X^i_{t-τ}},  B̂⁻_{t-τ}(X^i)
```

i.e., condition on the contemporaneous subset `S` **plus the lagged parents of BOTH endpoints**.
Conditioning on both endpoints' lagged parents blocks lagged paths and removes autocorrelation ->
larger effect size + better-calibrated tests (Thm 4). (Dropping the source's `B̂⁻_{t-τ}` is the weaker
`PCMCI+_0` variant — implement the full PCMCI+.) Skeleton p-value of a pair = max p over its tests.

**Phase 3 - orientation:** collider phase + Meek-rule phase, **equivalent to PC's** (majority-rule for
ambiguous triples), except the extra collider-phase CI tests also use the MCI test above. Lagged
edges are auto-oriented (τ>0). Contemporaneous edges get collider + Meek orientation. Output =
TimeSeriesCPDAG (lagged directed + contemporaneous CPDAG marks).

**Complexity:** Alg 1 polynomial; Alg 2's exponential worst case is only in `N` (num variables), not
`N·τ_max` -> far faster than PC on the full time-lagged graph.

## cbcd integration (reuse, don't rebuild)
- **Lagged phase + MCI machinery:** `cbcd/timeseries/algorithms.py` (`pcmci`), `timeseries/skeleton.py`
  (`PC1Skeleton`), `timeseries/citest.py` (`LaggedCITest`, which already does MCI-style conditioning).
  Reuse the PC1 lagged phase to get `B̂⁻`; reuse the MCI conditioning for test (2).
- **Contemporaneous orientation:** adapt cbcd's iid collider (`collider.py`) + Meek rules (`rules.py`)
  to the contemporaneous sub-graph of the time-series graph (lagged edges pre-oriented, fixed).
- **Output:** `cbcd/timeseries/graph.py` `TimeSeriesCPDAG`.
- **Instrumentation / caching / background:** `RunRecorder`, cached lagged CI test, and thread
  `background` (tiers are natural here — time order is itself a tier) as the family does.

## API
Add to `cbcd/timeseries/algorithms.py`:
```
def pcmci_plus(data, *, ci_test="parcorr", tau_max, alpha=0.05,
               max_cond_set=None, background=None, var_names=None,
               recorder=None, run_id=None) -> TimeSeriesCPDAG
```
Match `pcmci`'s signature conventions. Export from `cbcd/timeseries/__init__.py` and `cbcd/__init__.py`.

## Validation / parity gate
1. **External parity = tigramite `PCMCIplus`** (installed; same pattern as `parity/pcmci` vs tigramite
   PCMCI). Add `parity/pcmciplus/run.py`: on random SVAR/time-series models, `cbcd.pcmci_plus()` and
   tigramite `PCMCIplus` must produce the **same time-series graph** (lagged + contemporaneous
   endpoints), same `tau_max`/`alpha`/CI test. This is the primary correctness gate.
2. **Oracle test:** under a time-series d-separation oracle, `pcmci_plus` recovers the true
   TimeSeriesCPDAG including contemporaneous edges (see how `pcmci` is oracle-tested in
   `tests/timeseries/`).
3. **Regression:** `pcmci` (lagged-only) behavior unchanged; full suite green (currently 338).
4. **Contemporaneous coverage:** a test on a DGP with a known contemporaneous edge that `pcmci` misses
   and `pcmci_plus` recovers — the concrete reason to add it.

## Scope / caveats
- Causal sufficiency (no latents) = cell 3. The latent-TS version is LPCMCI (cell 4, separate).
- Stationarity assumed (same as `pcmci`).
- `tau_max` must be >= the true max lag; too-large `tau_max` degrades little (unlike Granger).

Related: [[algorithm-cell-map]] (cell 3 = PCMCI anchor + PCMCI+ champion). Anchor `pcmci` = `algo61`.
