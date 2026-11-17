"""LPCMCI parity: cbcd.lpcmci() vs tigramite LPCMCI on simulated SVAR-with-latents.

Mirrors ``parity/pcmciplus/run.py``. tigramite is used ONLY as a black-box output comparator here (it
is GPL-3.0; its source is never read for logic — cbcd's LPCMCI is clean-room from the authors'
published mathematics, SM §S4/§S5). For each latent SVAR fixture, simulate, marginalise latents out,
run both estimators with ParCorr at the same ``pc_alpha`` / ``tau_max`` / ``k``, convert tigramite's
string graph to cbcd's grid-endpoint convention, and compare present-anchored window edges.

The STRUCTURAL gate (oracle bidirected-edge recovery) and the RECALL gate live in
``tests/timeseries/test_lpcmci.py``; this harness reports finite-sample agreement and characterises
residual disagreements. Run manually (needs tigramite installed):
    python parity/lpcmci/run.py
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr as TgParCorr
from tigramite.lpcmci import LPCMCI as TgLPCMCI

from cbcd import lpcmci
from cbcd.timeseries import LaggedDataset

# (name, n_obs, lagged_edges [obs (i,j,tau)], contemp_edges [obs (i,j)], latent_groups).
LATENT_FIXTURES: list[
    tuple[str, int, list[tuple[int, int, int]], list[tuple[int, int]], list[tuple[int, ...]]]
] = [
    ("no_latent", 3, [(0, 1, 1), (1, 2, 1)], [(0, 2)], []),
    ("conf_pair", 2, [(0, 0, 1), (1, 1, 1)], [], [(0, 1)]),
    ("conf_chain", 3, [(0, 0, 1), (2, 2, 1)], [(1, 2)], [(0, 1)]),
    ("conf_lag", 3, [(0, 0, 1), (1, 1, 1), (0, 2, 1)], [], [(0, 1)]),
    ("two_conf", 4, [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)], [], [(0, 1), (2, 3)]),
]


def simulate_latent_svar(
    n_obs, lagged, contemp, latent_groups, *, T, seed, phi=0.5, noise=0.4
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    burn = 200
    n_total = n_obs + len(latent_groups)
    X = np.zeros((T + burn, n_total), dtype=np.float64)
    lat = {k: n_obs + k for k in range(len(latent_groups))}
    for t in range(1, T + burn):
        for k in range(len(latent_groups)):
            X[t, lat[k]] = rng.normal(scale=noise)
        for v in range(n_obs):
            val = phi * X[t - 1, v] + rng.normal(scale=noise)  # autocorrelation
            for i, j, tau in lagged:
                if j == v:
                    val += 0.5 * X[t - tau, i]
            for i, j in contemp:
                if j == v:
                    val += 0.5 * X[t, i]
            for k, grp in enumerate(latent_groups):
                if v in grp:
                    val += 0.6 * X[t, lat[k]]
            X[t, v] = val
    return X[burn:, :n_obs]


def tigramite_lpcmci_to_grid_endpoints(g: NDArray, n: int, ml: int) -> NDArray[np.int8]:
    """tigramite ``graph`` (N,N,tau+1) string marks -> cbcd grid endpoints (grid_n, grid_n)."""
    gn = n * (ml + 1)
    ep = np.zeros((gn, gn), dtype=np.int8)
    right = {">": 2, "<": 1, "o": 3, "-": 1}  # mark at destination (present) end
    left = {">": 1, "<": 2, "o": 3, "-": 1}  # mark at source end
    for i in range(n):
        for j in range(n):
            for tau in range(ml + 1):
                s = str(g[i, j, tau])
                if len(s) != 3:
                    continue
                a, b = i * (ml + 1) + tau, j * (ml + 1)
                if a == b:
                    continue
                ep[a, b], ep[b, a] = right.get(s[2], 3), left.get(s[0], 3)
    return ep


def _present_mask(n: int, ml: int) -> NDArray[np.bool_]:
    gn = n * (ml + 1)
    m = np.zeros((gn, gn), dtype=bool)
    for a in range(gn):
        for b in range(gn):
            if a % (ml + 1) == 0 or b % (ml + 1) == 0:
                m[a, b] = True
    return m


def run_one(name, n, lagged, contemp, latent, *, tau_max=1, T=4000, pc_alpha=0.01, k=4, seed=0):
    X = simulate_latent_svar(n, lagged, contemp, latent, T=T, seed=seed)
    ds = LaggedDataset(data=X, max_lag=tau_max)
    my_ep = lpcmci(ds, ci_test="parcorr", tau_max=tau_max, alpha=pc_alpha, k=k).window.endpoints

    df = pp.DataFrame(X)
    tg = TgLPCMCI(dataframe=df, cond_ind_test=TgParCorr(), verbosity=0)
    res = tg.run_lpcmci(tau_max=tau_max, pc_alpha=pc_alpha, n_preliminary_iterations=k)
    tg_ep = tigramite_lpcmci_to_grid_endpoints(res["graph"], n, tau_max)

    mask = _present_mask(n, tau_max)
    gn = n * (tau_max + 1)
    shd = int(np.sum((my_ep != tg_ep) & mask))
    my_adj = {
        frozenset({a, b})
        for a in range(gn)
        for b in range(a + 1, gn)
        if my_ep[a, b] != 0 and mask[a, b]
    }
    tg_adj = {
        frozenset({a, b})
        for a in range(gn)
        for b in range(a + 1, gn)
        if tg_ep[a, b] != 0 and mask[a, b]
    }
    return {"fixture": name, "seed": seed, "shd": shd, "adj_shd": len(my_adj ^ tg_adj)}


def main() -> int:
    print("=" * 72)
    print("LPCMCI parity: cbcd.lpcmci() vs tigramite LPCMCI (present-anchored window edges)")
    print("=" * 72)
    rows = [
        run_one(name, n, lg, ct, lat, seed=seed)
        for name, n, lg, ct, lat in LATENT_FIXTURES
        for seed in range(5)
    ]
    n_exact = sum(1 for r in rows if r["shd"] == 0)
    n_adj = sum(1 for r in rows if r["adj_shd"] == 0)
    print(f"\n  full-endpoint exact match: {n_exact}/{len(rows)} = {100 * n_exact / len(rows):.0f}%")
    print(f"  adjacency (skeleton) match: {n_adj}/{len(rows)} = {100 * n_adj / len(rows):.0f}%")
    by_fx: dict[str, list[int]] = {}
    adj_fx: dict[str, list[int]] = {}
    for r in rows:
        by_fx.setdefault(r["fixture"], []).append(r["shd"])
        adj_fx.setdefault(r["fixture"], []).append(r["adj_shd"])
    for fx in by_fx:
        em = sum(1 for s in by_fx[fx] if s == 0)
        am = sum(1 for s in adj_fx[fx] if s == 0)
        print(
            f"    {fx:>12}: endpoint {em}/{len(by_fx[fx])} exact  SHD={by_fx[fx]}  |  "
            f"adjacency {am}/{len(adj_fx[fx])} match"
        )
    print(
        "\n  The oracle test (bidirected latent-edge recovery) is the structural gate; residual\n"
        "  endpoint disagreements here are finite-sample orientation flips / middle-mark recall\n"
        "  differences. cbcd is clean-room from the published SM math; tigramite is a black-box\n"
        "  comparator only (never read for logic)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
