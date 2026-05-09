# Suite-level CLAUDE.md

This directory is the **monorepo for a four-package constraint-based
causal discovery suite**. Talk to Claude from here for any work that
touches more than one package; talk to Claude from inside an individual
package (`cd <pkg> && claude`) for deep package-specific work.

## The four packages

| package | purpose | location | state |
|---|---|---|---|
| **dagsampler** | configurable DAG / SCM simulator (synthetic mixed-type data, optional CI oracle) | `dagsampler/` | v0.1.0 on PyPI; v0.2.0+ ships from this monorepo |
| **cbcd** | constraint-based causal discovery algorithms (PC, FCI, RFCI, anytime-FCI, PCMCI) | `cbcd/` | v0.x API stable per D15 |
| **citk** | conditional independence test toolkit (FisherZ, Spearman native; KCI / CMIknn / RegressionCI / GCM / etc. via optional extras) | `citk/` | decoupled from causal-learn (optional `[causallearn]` extra) |
| **bnm** | DAG / CPDAG / PAG comparison metrics + visualization (SHD, HD, F1, SID, Markov-blanket comparisons) | `bnm/` | v0.2.2.dev0; legacy v0.1.0 GitHub repo archived |

`vendor/` contains read-only reference repos (`causal-learn`,
`tigramite`, `mCMIkNN`, `pgmpy`, `DCT`) used for parity testing.
**Gitignored — never distributed with the suite, never imported from
suite code.** Anyone wanting to run the parity harness clones them
locally.

## Architectural pattern

Every cross-package interaction goes through a **structural Protocol**
— no package imports another. All three Protocol arrows are wired and
verified at runtime by the suite parity harness at
`parity/suite/run.py`.

```
dagsampler ─── true_dag, data, ci_oracle ──▶ cbcd ─── recovered ──▶ bnm
                                              ▲                     ▲
                                              │ Protocol            │ Protocol
                                            citk                  (graph)
                                              │
                                          (CI tests via cbcd.CITest)
```

| boundary | contract | defined by |
|---|---|---|
| `citk → cbcd` | `cbcd.CITest` Protocol (`n_vars`, `__call__`, `details`) | cbcd (frozen under D15) |
| `dagsampler → cbcd` (CI oracle) | `cbcd.CITest` Protocol — `dagsampler.CausalDataGenerator.as_ci_oracle()` returns a conforming object | cbcd |
| `dagsampler → bnm` / `cbcd → bnm` | `bnm.GraphLike` Protocol (`n_vars`, `endpoints` int8 matrix, `var_names`) | bnm |

If you find yourself adding `from cbcd import ...` inside citk, bnm,
or dagsampler — stop and revisit the Protocol contract. Same for any
reverse direction. The parity harness runtime-checks `isinstance`
conformance against `cbcd.CITest` and `bnm.GraphLike` before each
fixture; a contract break trips the harness.

## Where to write logs

- **`suite/journal.md`** *(local-only — gitignored)*: cross-package
  decisions, push/release coordination, audit findings that affect
  more than one package. Newest entry on top, dated.
- **`<pkg>/docs/journal.md`** *(local-only — gitignored)*: per-package
  implementation history. Maintainer working file; retained on disk
  per developer but not committed.
- **`<pkg>/docs/design/api_v0.py`** *(tracked, where present)*:
  per-package design source-of-truth, pressure-tested before
  implementation. cbcd's is the canonical reference
  (`cbcd/docs/design/api_v0.py`).

## Commands by package

Each package has its own dev environment via `uv` (Python ≥ 3.11,
hatchling backend). No shared venv across packages.

```bash
# cbcd       (algorithms)
cd cbcd       && uv sync --all-extras && uv run pytest

# citk       (CI tests)
cd citk       && uv sync --all-extras && uv run pytest

# dagsampler (simulator)
cd dagsampler && uv sync --all-extras && uv run pytest

# bnm        (metrics + viz)
cd bnm        && uv sync --all-extras && uv run pytest

# Suite-level integration (chains all four):
cd parity/suite && uv sync && uv run python run.py
```

## Release coordination

This monorepo is the canonical source. Per-package PyPI releases are
triggered by `<pkg>-v<version>` tags (e.g. `dagsampler-v0.2.0`)
through a release workflow keyed on the changed subdirectory.

- **dagsampler**: `dagsampler` on PyPI; v0.1.0 published from the
  legacy archived repo. Future releases (0.2.0+) ship from this
  monorepo under the same PyPI name.
- **cbcd / citk / bnm**: not yet on PyPI. First publish requires
  explicit user instruction per package; CI scaffolding for the
  release workflow lives at the umbrella level.

The legacy per-package GitHub repos for **bnm** and **dagsampler**
are archived (read-only, banner pointing here). The legacy **citk**
and **cbcd** repos were deleted; their content is in this monorepo.

## Attribution and prior art

`NOTICE.md` at the umbrella root documents the relationship to
upstream sources: `causal-learn` (cbcd's intellectual basis,
clean-room re-implementation in code), `tigramite` (parity-validation
reference for PCMCI; GPL-3 boundary remains in user's environment for
optional citk adapters), `DAGMetrics-R` (bnm's R original), `mCMIkNN`
(vendored verbatim into citk under MIT). Per-package READMEs carry
the package-specific subset of this attribution.
