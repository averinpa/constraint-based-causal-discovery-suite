# `cbcd`: Constraint-Based Causal Discovery

[![Documentation](https://img.shields.io/badge/docs-averinpa.github.io-blue.svg)](https://averinpa.github.io/constraint-based-causal-discovery-suite/cbcd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`cbcd` is a Python library aiming to be a comprehensive home for **constraint-based causal discovery** algorithms — both i.i.d. and time-series.

**Documentation:** <https://averinpa.github.io/constraint-based-causal-discovery-suite/cbcd/>

The package is being built from scratch (informed by, but not depending on, [`causal-learn`](https://github.com/cmu-phil/causal-learn)) with three goals:

1. **Coverage.** A single library housing the standard family — PC and its variants (stable, conservative, majority, parallel), FCI and its variants (RFCI, FCI+, anytime-FCI, GFCI), CDNOD, IOD, JCI — plus time-series methods (PCMCI, PCMCI+, LPCMCI, J-PCMCI, tsFCI, SVAR-FCI).
2. **Composable design.** Algorithms are built from small, reusable pieces: skeleton phase, collider-orientation strategy, edge-orientation rules, graph type. Adding a new algorithm should be mostly composition.
3. **Correctness over cleverness.** Every algorithm ships with structure-level regression tests (SHD against d-separation oracle on standard graphs), not just smoke tests.

`cbcd` ships a minimal built-in CI test layer (Fisher-Z, χ², G², partial correlation) and defines a `CITest` Protocol. Any object satisfying the Protocol can drive any `cbcd` algorithm — including, eventually, [`citk`](https://github.com/averinpa/citk) tests once `citk` is adapted for `cbcd` interop (planned, not before late 2026).

## Status

The v0.x public API (everything re-exported from `cbcd`) is committed to
backwards compatibility across all v0.x minor and patch bumps. Breaking
changes will only appear at v1.0. Additive changes (new algorithms, new
keyword arguments with safe defaults, new CI tests via the registries)
may ship in any minor bump without notice.

Implemented end-to-end so far: `pc()` (PC family), `fci()` / `rfci()` /
`anytime_fci()` (FCI family), `pcmci()` (vanilla PCMCI). Other entries
in the design (PCMCI+, LPCMCI, conservative-/majority-PC, CDNOD, J-PCMCI,
…) are still stub-level — when they land, their signatures will conform
to the v0 design they're already specced against.

## Installation

For local development with extras:

```bash
uv sync --all-extras
```

## Acknowledgements

`cbcd` is based on
[causal-learn](https://github.com/py-why/causal-learn). The starting
point for the design was a careful study of causal-learn's source,
captured in
[`docs/audit_causal_learn.md`](docs/audit_causal_learn.md). The core
algorithmic content — PC-stable skeleton, Fisher-Z and χ²/G² CI
tests, FCI orientation rules, and the test-fixture set (Y-structure,
fork, chain, M-structure, diamond, plus the ASIA Bayesian network) —
comes from causal-learn (and from the constraint-based discovery
literature it implements). What `cbcd` adds is structural: a
composition-first design with explicit Protocol abstractions,
joblib-based parallelism, PAG-aware graph types, and a time-series
PCMCI slice.

`cbcd` is a clean-room re-implementation in code; no causal-learn
source is vendored or imported at runtime. Behavioural parity
against causal-learn (and against
[tigramite](https://github.com/jakobrunge/tigramite) for PCMCI) is
verified by the parity harness — see
[`docs/parity_report.md`](docs/parity_report.md) for the comparison
summary (13/13 fixtures match). The intellectual debt to
causal-learn is substantial; this work would not exist without it.

## License

MIT
