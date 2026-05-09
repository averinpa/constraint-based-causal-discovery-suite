# Notice and prior art

This suite is built on, validates against, and in one case vendors
from existing implementations. The relationships are summarised below.

## causal-learn

[github.com/py-why/causal-learn](https://github.com/py-why/causal-learn) — MIT.

`cbcd` is based on causal-learn. The starting point of `cbcd`'s
design was a careful study of causal-learn's source — its algorithms,
its abstractions, its CI-test patterns, its test fixtures — captured
in the audit at `cbcd/docs/audit_causal_learn.md` (commit `070d5ea`).
The core algorithmic content — PC-stable skeleton, Fisher-Z and
χ²/G² CI tests, FCI orientation rules, and the test-fixture set
(Y-structure, fork, chain, M-structure, diamond, plus the ASIA
Bayesian network) — comes from causal-learn (and from the constraint-
based discovery literature it implements). What `cbcd` adds is
structural: a composition-first design with explicit Protocol
abstractions, joblib-based parallelism, PAG-aware graph types with
Zhang's R1–R10 ported in full, and a time-series PCMCI slice.

`cbcd` is a clean-room re-implementation in code: no causal-learn
source is vendored or imported at runtime. Behavioural parity is
verified by the harness at `parity/` — see
`cbcd/docs/parity_report.md` (13/13 fixtures match). The intellectual
debt to causal-learn is substantial; this work would not exist
without it.

`citk.tests.kernel_tests.KCI` is a thin wrapper around
`causallearn.utils.cit.KCI`, available under `citk`'s optional
`[causallearn]` extra.

## tigramite

[github.com/jakobrunge/tigramite](https://github.com/jakobrunge/tigramite) — GPL-3.

Used as a parity-validation reference for `cbcd.timeseries.pcmci()`,
which is an independent implementation. Tigramite is **not** imported,
vendored, or otherwise carried in any package's runtime. `citk`
provides optional adapters for tigramite-based tests (CMIknn,
RegressionCI) under its `[tigramite]` extra; those adapters invoke
tigramite at the user's installation rather than vendoring its code,
so the GPL-3 boundary is the user's own environment.

## DAGMetrics (R)

[github.com/averinpa/DAGMetrics](https://github.com/averinpa/DAGMetrics).

`bnm` v0.1.0 was a port of DAGMetrics by the same author. v0.2.x is
a full Python rewrite around an int8 endpoint-mark matrix; the metric
definitions remain derivative of the R original.

## mCMIkNN (vendored)

[github.com/hpi-epic/mCMIkNN](https://github.com/hpi-epic/mCMIkNN) — MIT.

Vendored verbatim into `citk/citk/_vendor/indeptests/`. The MIT
license requires preservation of copyright; full attribution
including authors (Hügle, Hagedorn), the original publication
(ECML PKDD 2023), and the vendored revision SHA is in
`citk/citk/_vendor/NOTICE.md`.

## R-package adapters (citk)

`citk` provides optional adapters via `rpy2` for several R packages.
None of the R code is vendored; users install the R packages locally
under the optional `[r]` extra.

- **RCIT / RCoT** — Strobl, Zhang, Visweswaran 2019.
- **bnlearn** (R package) — Scutari 2010, used by `HarteminkChiSq`
  for discretisation.
- **MXM** — Lagani, Athineou, Farcomeni, Tsagris, Tsamardinos 2017.
