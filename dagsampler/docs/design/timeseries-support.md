# Design note: time-series support for dagsampler

**Status:** Phase 1 DONE (2026-07-18), Phase 2 DONE (2026-07-18). All in `src/dagsampler/timeseries.py`,
tests in `tests/test_timeseries.py` (22 ts tests; full suite 106 green).

**Phase 2 as built (deviates from the sketch below, deliberately):** rather than unroll to T slices and
vectorize over independent replicas, `simulate_svar` runs a genuine `O(T)` recurrence and returns ONE
long stationary series, so `n` effective observations = series length -- what a finite-sample CI
benchmark needs. Coefficients are tied per structural edge; a reduced-form companion spectral-radius
guard (`(I-Cc)^{-1} M_tau` blocks) stabilizes the linear core; mixed types use identity/threshold/
discretize links; heavy-tailed innovations (student_t/laplace/cauchy/gamma/exponential, centred) drive
the GFCM tail stressors. Entry points: `simulate_svar(spec, SVARParams)` and the config-dict
`TimeSeriesDataGenerator`. It does NOT reuse `CausalDataGenerator`'s per-node engine (the replica data
model is wrong for one-long-series); it reuses the distribution-name conventions only. Known
simplification: a discrete parent feeds forward its integer code (documented in the module).

---
Original design (2026-07-18):
**Motivation:** cbcd cells 7–8 (local·ts·latent) and the Paper 3 benchmark need a principled,
reusable, mixed-type temporal DGP + a lagged CI oracle. Today the time-series DGP is hand-rolled
inside cbcd test files (`tests/timeseries/{fixtures,oracle}.py`, `_rand_structure`/`_Oracle` in
`test_lpcmci_soundness.py`). This note specifies folding that into dagsampler as a real feature.

## Key insight — a stationary time-series graph is a structured *static* DAG

Unroll the process over `t = 0 .. T`: nodes `(var, t)`, contemporaneous edges within a slice,
lagged edges `(i, t-τ) → (j, t)`. The result is a plain acyclic `networkx.DiGraph`. dagsampler's
entire engine already operates on static DiGraphs:

- `causal_sim.CausalDataGenerator._create_graph` builds the DiGraph from `graph_params.edges`.
- `simulate()` does `nx.topological_sort` then per-node mechanism dispatch (`_generate_node_data`) —
  the full mixed-type mechanism library (linear/poly/interaction/heteroskedastic/skew/softmax, all
  noise models) applies **unchanged** to the unrolled nodes.
- `oracle.DSeparationOracle` answers d-separation on a static DiGraph.

So this is **reuse via unrolling, not a rewrite**. The new code is a builder (ts-spec → unrolled
config), a reshaper (panel → `T × n_vars` frame), an oracle wrapper (LaggedVar contract), and — for
realistic data — stationarity/stability handling.

## Phase 1 (MVP): structure + lagged d-sep oracle  — ~half a day

This is all the current cbcd soundness/benchmark work needs (its tests use dummy data + the oracle,
not realistic dynamics). It retires the hand-rolled `_rand_structure`/`_Oracle`.

Components:
1. **TS spec** — `n_vars`, `max_lag`, `lagged_edges: list[(i, j, τ)]` (τ≥1), `contemp_edges:
   list[(i, j)]` (τ=0, must be acyclic within a slice). Optional random generator with a
   bounded-degree / density knob (density = C/p for the fixed-degree regime — see the cbcd crossover
   benchmark).
2. **Unroller** — expand to a static DiGraph over `(var, t)` for `t = 0 .. t_horizon` (choose
   `t_horizon = max(k·max_lag, const)` deep enough that any queried lag maps to a valid interior
   time). Reuse `_create_graph`'s DAG validation.
3. **Lagged d-sep oracle** — wrap `DSeparationOracle` on the unrolled DiGraph, exposing the
   `cbcd.timeseries.LaggedCITest` contract: `n_vars`, `max_lag`, `__call__(x: LaggedVar, y:
   LaggedVar, S) -> float`, `details(...)`, plus `is_ancestor(a_grid, b_grid)` for soundness tests.
   Map `LaggedVar(v, -τ) → (v, t_ref - τ)` at a fixed interior `t_ref`. This is a direct lift of the
   existing cbcd test `_Oracle`.

API sketch:
```python
from dagsampler import TimeSeriesSpec, unroll, LaggedDSeparationOracle
spec = TimeSeriesSpec(n_vars=5, max_lag=2,
                      lagged_edges=[(0,1,1),(1,2,2)], contemp_edges=[(0,2)])
oracle = LaggedDSeparationOracle(spec)          # conforms to cbcd.timeseries.LaggedCITest
```

## Phase 2: stationary mixed-type SVAR *data* generation  — ~1–3 days

Needed for finite-sample benchmarks and the GFCM-in-cell-8 tail experiments (real dynamics). Reuses
the mechanism library on the unrolled graph, but three things need care:

1. **Stationarity (tied weights).** dagsampler samples edge weights per-edge; the unrolled copies of
   one lagged edge `(i→j, τ)` at different `t` must **share** the same weight/mechanism. The builder
   assigns weights per ts-edge, then stamps the shared value onto every unrolled instance (rather
   than letting `_sample_random_weight` fire independently per `t`).
2. **Stability.** A random SVAR can be explosive. Either (a) check the companion-matrix spectral
   radius `< 1` and resample/rescale weights until stable, or (b) accept transients and rely on
   burn-in. For linear/continuous mechanisms the spectral check is clean; for mixed-type/nonlinear,
   fall back to burn-in + a variance-blowup guard.
3. **Burn-in + mixed-type recurrence.** Discard the first `~b·max_lag` rows. Verify categorical /
   threshold / stratum-means mechanisms behave sanely when fed their own lagged outputs (a
   continuous→categorical→continuous loop across time is the risky case — test it).

API sketch (extends the existing config, so `CausalDataGenerator` stays the entry point):
```python
config = {
  "graph_params": {"n_vars": 5, "max_lag": 2,
                   "lagged_edges": [...], "contemp_edges": [...]},
  "simulation_params": {"n_timesteps": 2000, "burn_in": 200, "seed": 7,
                        "stationarity": "spectral"|"burn_in", ...mechanisms as today...},
}
out = CausalDataGenerator(config).simulate()   # out["data"] is T×n_vars; out["ts_dag"], out["ci_oracle"]
```

Prefer extending `graph_params`/`simulation_params` over a separate class, so the whole mixed-type
mechanism/noise surface is inherited for free. A thin `TimeSeriesDataGenerator` wrapper can provide
the ergonomic ts-first API on top.

## Decisions / open questions
- **Config shape:** extend `graph_params` (recommended — inherits everything) vs a new class. Lean extend.
- **Contemporaneous edges in Phase 1 oracle:** support from day one (LPCMCI/PCMCI+ need them) —
  acyclicity enforced within a slice by the unroller's DAG check.
- **Latents:** mark a subset of series unobserved (drop their columns from the frame; keep them in
  the unrolled DAG so the oracle answers *marginal* d-sep). Needed for cell 8. Cheap in Phase 1.
- **Selection bias / non-stationarity / regime switches:** out of scope; future.
- **Stability policy default:** `spectral` for all-continuous-linear, else `burn_in`.

## Sequencing
Phase 1 first — it's ~half a day, unblocks and de-duplicates the cbcd ts tests immediately, and is
oracle-only (no dynamics risk). Phase 2 lands before the finite-sample / GFCM-tail benchmark phase.
Both ship from the monorepo under the `dagsampler` PyPI name (0.3.0+ line).
