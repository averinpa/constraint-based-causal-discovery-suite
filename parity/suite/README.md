# Suite integration harness

Chains all four packages — **dagsampler → citk → cbcd → bnm** — on a
fixture set and asserts SHD/F1 bounds. The harness is a regression
detector for cross-package wiring and gross structural breakage; it is
not a precision benchmark for any individual algorithm. Each package
has its own internal correctness tests and parity harness.

## What it tests

For each fixture DAG:

1. `dagsampler.CausalDataGenerator` simulates data and exposes a
   d-separation CI oracle via `gen.as_ci_oracle()`.
2. `cbcd.pc()` runs twice — with the dagsampler oracle (gold-standard
   recovery: PC under a perfect oracle returns the true CPDAG by
   construction) and with `citk.tests.partial_correlation_tests.FisherZ`
   on the simulated data (empirical recovery).
3. `bnm.shd` and `bnm.f1` score the empirical recovery against the
   gold-standard recovery.
4. Per-fixture bounds (calibrated 2026-05-08, with headroom) are
   asserted.

The harness also asserts that both the dagsampler oracle and citk's
FisherZ satisfy `cbcd.CITest` structurally — `isinstance(_, CITest)`
must be True. This pins the structural-Protocol contract that the
suite's no-cross-package-imports architecture depends on.

## Running it

```bash
cd ~/Projects/suite/parity/suite
uv sync          # one-time: builds a venv with all four sister packages
uv run python run.py
```

Exit code `0` iff every fixture lands within bounds.

## Adjusting bounds

Bounds live in `FIXTURES` in `run.py`. Each entry has a `max_shd` and
`min_f1` set with headroom over what FisherZ delivered on
2026-05-08 (annotated in comments). Tighten if a fixture stays
comfortably below them across many runs; loosen if it genuinely
flakes. The "observed" line in each comment is the calibration value
— don't lose it; that's the regression baseline.
