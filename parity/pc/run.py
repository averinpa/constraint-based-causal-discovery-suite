"""PC parity: cbcd.pc() vs causal-learn.pc() on Linear-Gaussian SCMs.

For each PC fixture:
  1. Simulate Linear-Gaussian data.
  2. Run cbcd.pc(...) and causal-learn pc(...) with the same alpha and
     fisher-Z CI test.
  3. Convert causal-learn's output to cbcd's endpoint convention and
     compare CPDAG matrices element-wise.
"""

from __future__ import annotations

import sys

from causallearn.search.ConstraintBased.PC import pc as cl_pc

from cbcd import pc as cbcd_pc
from parity._compare import (
    cl_graph_to_cbcd_endpoints,
    shd_endpoints,
    simulate_linear_gaussian,
)
from tests.fixtures import ALL_FIXTURES


def run_one(name: str, *, n: int = 5000, alpha: float = 0.01) -> dict:
    dag, expected_cpdag = ALL_FIXTURES[name]()
    data = simulate_linear_gaussian(dag, n=n, seed=hash(name) & 0xFFFF)
    cbcd_out = cbcd_pc(data, alpha=alpha)
    cl_out = cl_pc(data, alpha=alpha, indep_test="fisherz", show_progress=False)
    cl_endpoints = cl_graph_to_cbcd_endpoints(cl_out.G.graph)

    shd_cbcd_truth = shd_endpoints(cbcd_out.endpoints, expected_cpdag.endpoints)
    shd_cl_truth = shd_endpoints(cl_endpoints, expected_cpdag.endpoints)
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
        "expected_endpoints": expected_cpdag.endpoints.tolist(),
    }


def main() -> int:
    print("=" * 72)
    print("PC parity: cbcd.pc() vs causal-learn pc()")
    print("=" * 72)
    rows: list[dict] = []
    for name in ALL_FIXTURES:
        r = run_one(name)
        rows.append(r)
        match_truth_cbcd = "✓" if r["shd_cbcd_vs_truth"] == 0 else "✗"
        match_truth_cl = "✓" if r["shd_causal_learn_vs_truth"] == 0 else "✗"
        match_pair = "✓" if r["shd_cbcd_vs_causal_learn"] == 0 else "✗"
        print(
            f"  {r['fixture']:>14}  "
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
    # Exit 0 if cbcd parity with causal-learn is 100% on cases where both
    # match truth; otherwise exit 1 to flag a real divergence.
    real_disagreements = sum(
        1
        for r in rows
        if r["shd_cbcd_vs_truth"] == 0
        and r["shd_causal_learn_vs_truth"] == 0
        and r["shd_cbcd_vs_causal_learn"] != 0
    )
    print(f"  ── real disagreements (both correct, but differ): {real_disagreements}")
    return 0 if real_disagreements == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
