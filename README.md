# `cbcd`: Constraint-Based Causal Discovery

`cbcd` is a Python library aiming to be a comprehensive home for **constraint-based causal discovery** algorithms — both i.i.d. and time-series.

The package is being built from scratch (informed by, but not depending on, [`causal-learn`](https://github.com/cmu-phil/causal-learn)) with three goals:

1. **Coverage.** A single library housing the standard family — PC and its variants (stable, conservative, majority, parallel), FCI and its variants (RFCI, FCI+, anytime-FCI, GFCI), CDNOD, IOD, JCI — plus time-series methods (PCMCI, PCMCI+, LPCMCI, J-PCMCI, tsFCI, SVAR-FCI).
2. **Composable design.** Algorithms are built from small, reusable pieces: skeleton phase, collider-orientation strategy, edge-orientation rules, graph type. Adding a new algorithm should be mostly composition.
3. **Correctness over cleverness.** Every algorithm ships with structure-level regression tests (SHD against d-separation oracle on standard graphs), not just smoke tests.

`cbcd` ships a minimal built-in CI test layer (Fisher-Z, χ², G², partial correlation) and defines a `CITest` Protocol. Any object satisfying the Protocol can drive any `cbcd` algorithm — including, eventually, [`citk`](https://github.com/averinpa/citk) tests once `citk` is adapted for `cbcd` interop (planned, not before late 2026).

## Status

Early development. API not yet stable.

## Installation

For local development with extras:

```bash
uv sync --all-extras
```

## License

MIT
