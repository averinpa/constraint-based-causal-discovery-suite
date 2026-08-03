# Time series (stationary SVAR)

The `dagsampler.timeseries` module generates discrete-time stationary causal
processes with latent confounders, plus a matching lagged d-separation oracle.
It is the time-series analogue of `CausalDataGenerator` + the CI oracle.

## Define a structure

A `TimeSeriesSpec` has contemporaneous edges (same time slice, must be acyclic)
and lagged edges `(i, j, tau)` meaning series `i` at `t - tau` causes `j` at `t`.
The last `n_latent` series are unobserved.

```python
from dagsampler import TimeSeriesSpec

spec = TimeSeriesSpec(
    n_vars=5,
    max_lag=2,
    lagged_edges=[(0, 1, 1), (1, 2, 2)],   # 0[t-1]->1[t], 1[t-2]->2[t]
    contemp_edges=[(0, 2)],                # 0[t]->2[t]
    n_latent=1,                            # series 4 is latent
)
spec.n_observed  # 4
```

Or draw a random bounded-degree spec (edge density `degree / n_vars`, so expected
degree stays roughly constant as the graph grows):

```python
from dagsampler import random_ts_spec

spec = random_ts_spec(n_observed=4, max_lag=1, degree=1.5, n_latent=1, seed=0)
```

## Query the lagged d-separation oracle

`LaggedDSeparationOracle` answers m-separation on the *observed* marginal: latents
are present in the unrolled graph (so they open and close paths correctly) but are
never queried or conditioned on. It returns `1.0` when the pair is d-separated
given `S`, else `0.0`, and conforms to the `cbcd.timeseries.LaggedCITest` protocol
without importing cbcd. Query with `LaggedVar(var, lag)`, where `lag <= 0` (`0` is
the present, `-tau` is `tau` steps in the past).

```python
from dagsampler import LaggedDSeparationOracle, LaggedVar

oracle = LaggedDSeparationOracle(spec)
p = oracle(LaggedVar(0, -1), LaggedVar(1, 0), [])          # 0[t-1] vs 1[t], no conditioning
p = oracle(LaggedVar(0, -1), LaggedVar(2, 0), [LaggedVar(1, -1)])
```

`unroll(spec, t_horizon=None)` returns the static `(var, t)` DAG the oracle reasons
over, if you need it directly.

## Simulate data

`simulate_svar` draws one long stationary mixed-type series. Stationarity is
enforced by rescaling the lagged coefficients until the companion spectral radius
is `<= target_spectral_radius` (default `0.9`); series are continuous by default,
with `binary_frac` / `categorical_frac` (or explicit `var_types`) assigning
discrete types.

```python
from dagsampler import simulate_svar, SVARParams

out = simulate_svar(spec, SVARParams(n_timesteps=2000, seed=0))
out["data"]             # T x n_observed pandas DataFrame (latent series dropped)
out["oracle"]           # the matching LaggedDSeparationOracle ground truth
out["spectral_radius"]  # of the stabilized linear core
```

`TimeSeriesDataGenerator(config)` offers a config-dict entry point mirroring
`CausalDataGenerator`, if you prefer configuration over the dataclasses.
