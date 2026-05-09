"""Suite integration test: dagsampler → citk → cbcd → bnm on a fixture set.

For each fixture DAG:

1. Generate data with `dagsampler.CausalDataGenerator`.
2. Run `cbcd.pc()` twice on that data:
   - With dagsampler's d-separation CI oracle (`gen.as_ci_oracle()`).
     PC under a perfect oracle returns the true CPDAG by construction.
   - With `citk.FisherZ` on the simulated data — the empirical recovery.
3. Score the empirical recovery against the gold-standard recovery
   using `bnm.shd` and `bnm.f1`.
4. Compare against per-fixture bounds with headroom over observed
   2026-05-08 calibration values.

Exits 0 iff every fixture's SHD and F1 land within bounds. The script
does not assert algorithmic precision — that's the job of each
package's own parity harness. This script catches:

* Cross-package wiring regressions (e.g., dagsampler's oracle stops
  satisfying `cbcd.CITest`; citk's FisherZ stops being callable from
  cbcd; bnm refuses cbcd's CPDAG output).
* Gross structural regressions in any single package.

Run from any venv that has the four packages installed by path:

    cd ~/Projects/suite/parity/suite
    uv sync         # one-time
    uv run python run.py
"""

from __future__ import annotations

import sys
from typing import Any

import bnm
from cbcd import pc
from cbcd.citest.protocol import CITest
from citk.tests.partial_correlation_tests import FisherZ
from dagsampler import CausalDataGenerator


# Bounds calibrated 2026-05-08 with seed_structure=1, seed_data=2,
# binary_proportion=0.0, alpha=0.05. The "observed" column records
# what FisherZ recovered against the oracle recovery; bounds add
# headroom so the test is a regression detector, not a precision
# benchmark. Tighten bounds if a fixture stays comfortably below them
# across many runs; loosen if a fixture genuinely flakes.
FIXTURES: dict[str, dict[str, Any]] = {
    "collider_3": {
        "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "C"], ["B", "C"]]},
        "n_samples": 2000,
        # observed: SHD=0, F1=1.00
        "max_shd": 1,
        "min_f1": 0.90,
    },
    "fork_3": {
        "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["A", "C"]]},
        "n_samples": 3000,
        # observed: SHD=2, F1=0.00 (FisherZ orients an undirected CPDAG;
        # known orientation drift, not a regression)
        "max_shd": 3,
        "min_f1": 0.0,
    },
    "chain_3": {
        "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]},
        "n_samples": 6000,
        # observed: SHD=0, F1=1.00
        "max_shd": 2,
        "min_f1": 0.50,
    },
    "diamond_4": {
        "graph": {
            "nodes": ["A", "B", "C", "D"],
            "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]],
        },
        "n_samples": 3000,
        # observed: SHD=2, F1=0.50
        "max_shd": 4,
        "min_f1": 0.30,
    },
    "asia_like_5": {
        "graph": {
            "nodes": ["A", "B", "C", "D", "E"],
            "edges": [["A", "C"], ["B", "C"], ["C", "D"], ["C", "E"]],
        },
        "n_samples": 3000,
        # observed: SHD=5, F1=0.40
        "max_shd": 8,
        "min_f1": 0.25,
    },
}


def run_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Execute the dagsampler→citk→cbcd→bnm chain on one fixture."""
    cfg = {
        "simulation_params": {
            "n_samples": spec["n_samples"],
            "seed_structure": 1,
            "seed_data": 2,
            "binary_proportion": 0.0,
        },
        "graph_params": {"type": "custom", **spec["graph"]},
    }
    gen = CausalDataGenerator(cfg)
    result = gen.simulate()

    oracle = gen.as_ci_oracle()
    if not isinstance(oracle, CITest):
        raise RuntimeError(
            f"{name}: dagsampler oracle does not satisfy cbcd.CITest"
        )

    fisherz = FisherZ(result["data"].to_numpy())
    if not isinstance(fisherz, CITest):
        raise RuntimeError(
            f"{name}: citk.FisherZ does not satisfy cbcd.CITest"
        )

    true_cpdag = pc(result["data"], ci_test=oracle, alpha=0.05)
    recovered = pc(result["data"], ci_test=fisherz, alpha=0.05)

    shd = int(bnm.shd(true_cpdag, recovered))
    f1 = float(bnm.f1(true_cpdag, recovered))
    hd = int(bnm.hd(true_cpdag, recovered))

    return {
        "fixture": name,
        "n_samples": spec["n_samples"],
        "shd": shd,
        "f1": f1,
        "hd": hd,
        "max_shd": spec["max_shd"],
        "min_f1": spec["min_f1"],
        "shd_ok": shd <= spec["max_shd"],
        "f1_ok": f1 >= spec["min_f1"],
    }


def main() -> int:
    print("=" * 78)
    print("Suite integration: dagsampler → citk → cbcd → bnm")
    print("=" * 78)
    rows: list[dict[str, Any]] = []
    for name, spec in FIXTURES.items():
        try:
            row = run_one(name, spec)
        except Exception as exc:
            print(f"  {name:<14}  ✗ exception: {type(exc).__name__}: {exc}")
            rows.append({"fixture": name, "shd_ok": False, "f1_ok": False, "exception": exc})
            continue
        rows.append(row)
        shd_mark = "✓" if row["shd_ok"] else "✗"
        f1_mark = "✓" if row["f1_ok"] else "✗"
        print(
            f"  {row['fixture']:<14}  n={row['n_samples']:>5}  "
            f"SHD={row['shd']:>2}/{row['max_shd']:>2} {shd_mark}   "
            f"F1={row['f1']:>4.2f}/{row['min_f1']:>4.2f} {f1_mark}   "
            f"HD={row['hd']:>2}"
        )

    n_ok = sum(1 for r in rows if r.get("shd_ok") and r.get("f1_ok"))
    print(f"  ── {n_ok}/{len(rows)} fixtures within bounds")

    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
