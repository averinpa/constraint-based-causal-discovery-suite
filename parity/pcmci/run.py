"""PCMCI parity: cbcd.pcmci() vs tigramite PCMCI on simulated VAR data.

For each VAR fixture, simulate a long realisation, run cbcd.pcmci() and
tigramite's PCMCI with ParCorr at the same alpha and pc_alpha, then convert
tigramite's link matrix to cbcd's endpoint convention and compare.
"""

from __future__ import annotations

import sys

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr as TgParCorr
from tigramite.pcmci import PCMCI as TgPCMCI

from cbcd import pcmci as cbcd_pcmci
from cbcd.timeseries import LaggedDataset
from parity._compare import shd_endpoints, simulate_var, tigramite_graph_to_cbcd_endpoints

# Each fixture is a (name, n_vars, max_lag, var_edges) tuple where var_edges
# is a list of (src_var, dst_var, tau, coef).
VAR_FIXTURES: list[tuple[str, int, int, list[tuple[int, int, int, float]]]] = [
    ("ar1_single", 1, 1, [(0, 0, 1, 0.6)]),
    (
        "two_var_var1",
        2,
        1,
        [(0, 0, 1, 0.5), (0, 1, 1, 0.6), (1, 0, 1, 0.3), (1, 1, 1, 0.4)],
    ),
    (
        "sparse_var2",
        3,
        2,
        [(0, 1, 1, 0.7), (1, 2, 2, 0.6), (0, 2, 2, 0.5)],
    ),
]


def _expected_endpoints(
    n_vars: int, max_lag: int, edges: list[tuple[int, int, int, float]]
) -> np.ndarray:
    ep = np.zeros((max_lag + 1, n_vars, n_vars), dtype=np.int8)
    for src, dst, tau, _ in edges:
        ep[tau, src, dst] = 2  # ARROW at dst (mark at present-time end)
    return ep


def run_one(
    name: str,
    n_vars: int,
    max_lag: int,
    edges: list[tuple[int, int, int, float]],
    *,
    T: int = 2000,
    alpha: float = 0.01,
    pc_alpha: float = 0.05,
) -> dict:
    data = simulate_var(edges, n_vars=n_vars, T=T, seed=hash(name) & 0xFFFF)
    expected = _expected_endpoints(n_vars, max_lag, edges)

    # cbcd
    ds = LaggedDataset(data=data, max_lag=max_lag)
    cbcd_out = cbcd_pcmci(ds, ci_test="parcorr", alpha=alpha, pc_alpha=pc_alpha)

    # tigramite
    df = pp.DataFrame(data)
    tg = TgPCMCI(dataframe=df, cond_ind_test=TgParCorr(), verbosity=0)
    res = tg.run_pcmci(tau_max=max_lag, pc_alpha=pc_alpha, alpha_level=alpha)
    tg_endpoints = tigramite_graph_to_cbcd_endpoints(res["graph"])

    shd_cbcd_truth = shd_endpoints(cbcd_out.endpoints, expected)
    shd_tg_truth = shd_endpoints(tg_endpoints, expected)
    shd_cbcd_tg = shd_endpoints(cbcd_out.endpoints, tg_endpoints)

    return {
        "fixture": name,
        "T": T,
        "alpha": alpha,
        "pc_alpha": pc_alpha,
        "shd_cbcd_vs_truth": shd_cbcd_truth,
        "shd_tigramite_vs_truth": shd_tg_truth,
        "shd_cbcd_vs_tigramite": shd_cbcd_tg,
        "cbcd_endpoints": cbcd_out.endpoints.tolist(),
        "tg_endpoints": tg_endpoints.tolist(),
        "expected_endpoints": expected.tolist(),
    }


def main() -> int:
    print("=" * 72)
    print("PCMCI parity: cbcd.pcmci() vs tigramite PCMCI")
    print("=" * 72)
    rows: list[dict] = []
    for name, nv, ml, edges in VAR_FIXTURES:
        r = run_one(name, nv, ml, edges)
        rows.append(r)
        match_cbcd = "✓" if r["shd_cbcd_vs_truth"] == 0 else "✗"
        match_tg = "✓" if r["shd_tigramite_vs_truth"] == 0 else "✗"
        match_pair = "✓" if r["shd_cbcd_vs_tigramite"] == 0 else "✗"
        print(
            f"  {r['fixture']:>14}  "
            f"cbcd↔truth={r['shd_cbcd_vs_truth']:>2} {match_cbcd}   "
            f"tg↔truth={r['shd_tigramite_vs_truth']:>2} {match_tg}   "
            f"cbcd↔tg={r['shd_cbcd_vs_tigramite']:>2} {match_pair}"
        )
    n_match_cbcd = sum(1 for r in rows if r["shd_cbcd_vs_truth"] == 0)
    n_match_tg = sum(1 for r in rows if r["shd_tigramite_vs_truth"] == 0)
    n_pair = sum(1 for r in rows if r["shd_cbcd_vs_tigramite"] == 0)
    print(
        f"  ── totals: cbcd matches truth on {n_match_cbcd}/{len(rows)};  "
        f"tigramite matches truth on {n_match_tg}/{len(rows)};  "
        f"cbcd matches tigramite on {n_pair}/{len(rows)}"
    )
    real_disagreements = sum(
        1
        for r in rows
        if r["shd_cbcd_vs_truth"] == 0
        and r["shd_tigramite_vs_truth"] == 0
        and r["shd_cbcd_vs_tigramite"] != 0
    )
    print(f"  ── real disagreements (both correct, but differ): {real_disagreements}")
    if real_disagreements:
        print()
        for r in rows:
            if (
                r["shd_cbcd_vs_truth"] == 0
                and r["shd_tigramite_vs_truth"] == 0
                and r["shd_cbcd_vs_tigramite"] != 0
            ):
                print(f"DETAIL — {r['fixture']}: cbcd and tigramite both match truth, but differ")
                print("  cbcd:")
                print("    " + str(np.array(r["cbcd_endpoints"])).replace("\n", "\n    "))
                print("  tigramite:")
                print("    " + str(np.array(r["tg_endpoints"])).replace("\n", "\n    "))
    return 0 if real_disagreements == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
