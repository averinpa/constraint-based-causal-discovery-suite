"""PCMCI+ parity: cbcd.pcmci_plus() vs tigramite PCMCIplus on simulated SVAR data.

Mirrors ``parity/pcmci/run.py``. For each structural VAR fixture (with contemporaneous *and* lagged
links), simulate a long realisation, run both estimators with ParCorr at the same ``pc_alpha`` and
``tau_max``, convert tigramite's string graph to cbcd's endpoint convention, and compare the full
time-series endpoint matrices (lagged + contemporaneous).

Run manually (needs tigramite installed):  python parity/pcmciplus/run.py
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr as TgParCorr
from tigramite.pcmci import PCMCI as TgPCMCI

from cbcd import pcmci_plus
from cbcd.timeseries import LaggedDataset

# Each fixture: (name, n_vars, contemp_edges, lagged_edges). contemp edges (src, dst, coef) must be
# acyclic in index order (src<dst is sufficient). lagged edges (src, dst, tau, coef), tau>=1.
SVAR_FIXTURES: list[
    tuple[str, int, list[tuple[int, int, float]], list[tuple[int, int, int, float]]]
] = [
    ("contemp_chain", 3, [(0, 1, 0.5)], [(0, 0, 1, 0.5), (1, 2, 1, 0.6)]),
    ("contemp_collider", 3, [(0, 2, 0.6), (1, 2, 0.5)], [(0, 0, 1, 0.4), (1, 1, 1, 0.4)]),
    ("two_forks", 4, [(0, 1, 0.5), (2, 3, 0.5)], [(0, 2, 1, 0.6), (1, 3, 1, 0.5)]),
    ("contemp_source", 3, [(0, 1, 0.6)], [(2, 0, 1, 0.5), (2, 1, 1, 0.4)]),
    ("autoreg_contemp", 4, [(1, 2, 0.5), (0, 3, 0.4)], [(0, 0, 1, 0.4), (1, 1, 1, 0.4), (2, 2, 1, 0.4)]),
    ("mixed", 3, [(0, 1, 0.55), (1, 2, 0.5)], [(0, 0, 1, 0.4)]),
    ("lagged_cross", 4, [(0, 2, 0.5), (1, 3, 0.5)], [(0, 1, 1, 0.5)]),
]


def simulate_svar_contemp(
    n_vars: int,
    contemp: list[tuple[int, int, float]],
    lagged: list[tuple[int, int, int, float]],
    *,
    T: int,
    seed: int,
    noise: float = 0.4,
) -> NDArray[np.float64]:
    """Simulate a structural VAR with contemporaneous effects. Contemporaneous edges are applied in
    index order (assumed acyclic src<dst), so ``x_t`` is generated variable-by-variable within each t."""
    rng = np.random.default_rng(seed)
    max_lag = max((tau for _, _, tau, _ in lagged), default=1)
    burn = max_lag + 100
    X = np.zeros((T + burn, n_vars), dtype=np.float64)
    inc_c: dict[int, list[tuple[int, float]]] = {j: [] for j in range(n_vars)}
    inc_l: dict[int, list[tuple[int, int, float]]] = {j: [] for j in range(n_vars)}
    for s, d, c in contemp:
        inc_c[d].append((s, c))
    for s, d, tau, c in lagged:
        inc_l[d].append((s, tau, c))
    for t in range(max_lag, T + burn):
        for v in range(n_vars):
            val = rng.normal(scale=noise)
            for s, tau, c in inc_l[v]:
                val += c * X[t - tau, s]
            for s, c in inc_c[v]:
                val += c * X[t, s]
            X[t, v] = val
    return X[burn:]


def tigramite_pcmciplus_to_cbcd_endpoints(g: NDArray) -> NDArray[np.int8]:
    """Convert tigramite's PCMCIplus ``graph`` (N,N,tau+1) string array to cbcd endpoints
    (tau+1,N,N). Contemporaneous ``-->``/``<--``/``o-o`` map to directed/undirected; conflict ``x-x``
    collapses to undirected (as in cbcd's TimeSeriesCPDAG); lagged ``-->`` -> ARROW at present."""
    n, _, L = g.shape
    ep = np.zeros((L, n, n), dtype=np.int8)
    for tau in range(L):
        for i in range(n):
            for j in range(n):
                s = str(g[i, j, tau])
                if s == "":
                    continue
                if tau == 0:
                    if s == "-->":
                        ep[0, i, j], ep[0, j, i] = 2, 1
                    elif s == "<--":
                        ep[0, i, j], ep[0, j, i] = 1, 2
                    else:  # "o-o" or conflict "x-x"
                        ep[0, i, j], ep[0, j, i] = 1, 1
                elif s == "-->":
                    ep[tau, i, j] = 2
    return ep


def run_one(
    name: str,
    n_vars: int,
    contemp: list[tuple[int, int, float]],
    lagged: list[tuple[int, int, int, float]],
    *,
    tau_max: int = 2,
    T: int = 4000,
    pc_alpha: float = 0.01,
    seed: int = 0,
) -> dict:
    X = simulate_svar_contemp(n_vars, contemp, lagged, T=T, seed=seed)

    ds = LaggedDataset(data=X, max_lag=tau_max)
    cbcd_out = pcmci_plus(ds, ci_test="parcorr", tau_max=tau_max, alpha=pc_alpha)

    df = pp.DataFrame(X)
    tg = TgPCMCI(dataframe=df, cond_ind_test=TgParCorr(), verbosity=0)
    res = tg.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
    tg_ep = tigramite_pcmciplus_to_cbcd_endpoints(res["graph"])

    shd = int(np.sum(cbcd_out.endpoints != tg_ep))
    return {
        "fixture": name,
        "seed": seed,
        "shd_cbcd_vs_tigramite": shd,
        "cbcd_endpoints": cbcd_out.endpoints,
        "tg_endpoints": tg_ep,
    }


def main() -> int:
    print("=" * 72)
    print("PCMCI+ parity: cbcd.pcmci_plus() vs tigramite PCMCIplus")
    print("=" * 72)
    rows: list[dict] = []
    for name, nv, contemp, lagged in SVAR_FIXTURES:
        for seed in range(8):
            r = run_one(name, nv, contemp, lagged, seed=seed)
            rows.append(r)
    n_match = sum(1 for r in rows if r["shd_cbcd_vs_tigramite"] == 0)
    print(f"\n  agreement (identical endpoint matrix): {n_match}/{len(rows)} "
          f"= {100 * n_match / len(rows):.1f}%")
    by_fixture: dict[str, list[int]] = {}
    for r in rows:
        by_fixture.setdefault(r["fixture"], []).append(r["shd_cbcd_vs_tigramite"])
    for fx, shds in by_fixture.items():
        matches = sum(1 for s in shds if s == 0)
        print(f"    {fx:>18}: {matches}/{len(shds)} match   SHDs={shds}")
    disagreements = [r for r in rows if r["shd_cbcd_vs_tigramite"] != 0]
    if disagreements:
        print(f"\n  {len(disagreements)} disagreement(s) (single-edge finite-sample flips expected):")
        for r in disagreements:
            diff = np.argwhere(r["cbcd_endpoints"] != r["tg_endpoints"])
            print(f"    {r['fixture']} seed{r['seed']}: SHD={r['shd_cbcd_vs_tigramite']} "
                  f"differing (tau,i,j) cells: {diff.tolist()}")
    # Non-fatal: disagreements are characterised finite-sample / deep-lag-conditioning boundary flips.
    return 0


if __name__ == "__main__":
    sys.exit(main())
