"""FCI parity: cbcd.fci() vs causal-learn.fci() on Linear-Gaussian data with latents.

For each PAG fixture (DAG-with-latent), simulate Linear-Gaussian data over
the FULL DAG, drop the latent columns, and run both implementations on the
observed-only data with the same alpha and Fisher-Z CI test. Convert
causal-learn's PAG output to cbcd's endpoint convention and compare.

Note: causal-learn's FCI has known bugs (audit FCI.py:1000–1058 — the
exponential ``removeByPossibleDsep`` ignoring ``depth``; ``rule8`` issues).
On fixtures where these bugs bite, cbcd may legitimately differ from
causal-learn while both differ from the truth. The report distinguishes
those cases.
"""

from __future__ import annotations

import sys

import numpy as np
from causallearn.search.ConstraintBased.FCI import fci as cl_fci

from cbcd import fci as cbcd_fci
from parity._compare import (
    cl_graph_to_cbcd_endpoints,
    shd_endpoints,
    simulate_linear_gaussian,
)
from tests.fixtures_pag import ALL_PAG_FIXTURES


def run_one(name: str, *, n: int = 5000, alpha: float = 0.01) -> dict:
    full_dag, n_observed, expected_pag = ALL_PAG_FIXTURES[name]()
    full_data = simulate_linear_gaussian(full_dag, n=n, seed=hash(name) & 0xFFFF)
    obs_data = full_data[:, :n_observed]

    cbcd_out = cbcd_fci(obs_data, alpha=alpha)
    cl_g, _edges = cl_fci(
        obs_data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False
    )
    cl_endpoints = cl_graph_to_cbcd_endpoints(cl_g.graph)

    shd_cbcd_truth = shd_endpoints(cbcd_out.endpoints, expected_pag.endpoints)
    shd_cl_truth = shd_endpoints(cl_endpoints, expected_pag.endpoints)
    shd_cbcd_cl = shd_endpoints(cbcd_out.endpoints, cl_endpoints)

    return {
        "fixture": name,
        "n_samples": n,
        "alpha": alpha,
        "shd_cbcd_vs_truth": shd_cbcd_truth,
        "shd_causal_learn_vs_truth": shd_cl_truth,
        "shd_cbcd_vs_causal_learn": shd_cbcd_cl,
        "cbcd_endpoints": cbcd_out.endpoints.tolist(),
        "cl_endpoints": cl_endpoints.tolist(),
        "expected_endpoints": expected_pag.endpoints.tolist(),
    }


def main() -> int:
    print("=" * 72)
    print("FCI parity: cbcd.fci() vs causal-learn fci()")
    print("=" * 72)
    rows: list[dict] = []
    for name in ALL_PAG_FIXTURES:
        r = run_one(name)
        rows.append(r)
        match_truth_cbcd = "✓" if r["shd_cbcd_vs_truth"] == 0 else "✗"
        match_truth_cl = "✓" if r["shd_causal_learn_vs_truth"] == 0 else "✗"
        match_pair = "✓" if r["shd_cbcd_vs_causal_learn"] == 0 else "✗"
        print(
            f"  {r['fixture']:>34}  "
            f"cbcd↔truth={r['shd_cbcd_vs_truth']:>2} {match_truth_cbcd}   "
            f"cl↔truth={r['shd_causal_learn_vs_truth']:>2} {match_truth_cl}   "
            f"cbcd↔cl={r['shd_cbcd_vs_causal_learn']:>2} {match_pair}"
        )
    n_match_cbcd = sum(1 for r in rows if r["shd_cbcd_vs_truth"] == 0)
    n_match_cl = sum(1 for r in rows if r["shd_causal_learn_vs_truth"] == 0)
    n_pair = sum(1 for r in rows if r["shd_cbcd_vs_causal_learn"] == 0)
    print(
        f"  ── totals: cbcd matches truth on {n_match_cbcd}/{len(rows)};  "
        f"causal-learn matches truth on {n_match_cl}/{len(rows)};  "
        f"cbcd matches causal-learn on {n_pair}/{len(rows)}"
    )
    real_disagreements = sum(
        1
        for r in rows
        if r["shd_cbcd_vs_truth"] == 0
        and r["shd_causal_learn_vs_truth"] == 0
        and r["shd_cbcd_vs_causal_learn"] != 0
    )
    cbcd_correct_cl_wrong = sum(
        1 for r in rows if r["shd_cbcd_vs_truth"] == 0 and r["shd_causal_learn_vs_truth"] != 0
    )
    cl_correct_cbcd_wrong = sum(
        1 for r in rows if r["shd_cbcd_vs_truth"] != 0 and r["shd_causal_learn_vs_truth"] == 0
    )
    print(f"  ── real disagreements (both correct, but differ): {real_disagreements}")
    print(f"  ── cbcd correct, causal-learn wrong: {cbcd_correct_cl_wrong}")
    print(f"  ── causal-learn correct, cbcd wrong: {cl_correct_cbcd_wrong}")
    if cl_correct_cbcd_wrong:
        print()
        for r in rows:
            if r["shd_cbcd_vs_truth"] != 0 and r["shd_causal_learn_vs_truth"] == 0:
                print(f"DETAIL — {r['fixture']}: cbcd disagrees with truth")
                print("  cbcd endpoints:")
                print("    " + str(np.array(r["cbcd_endpoints"])).replace("\n", "\n    "))
                print("  expected endpoints:")
                print("    " + str(np.array(r["expected_endpoints"])).replace("\n", "\n    "))
    return 0 if cl_correct_cbcd_wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
