# Suite-level CLAUDE.md

This directory is the **umbrella for a four-package constraint-based
causal discovery suite**. Talk to Claude from here for any work that
touches more than one package; talk to Claude from inside an individual
package for deep package-specific work.

## The four packages

| package | purpose | location | public state |
|---|---|---|---|
| **dagsampler** | configurable DAG / SCM simulator (synthetic mixed-type data, optional CI oracle table) | `dagsampler/` | published on PyPI as `dagsampler==0.1.0` |
| **cbcd** | constraint-based causal discovery algorithms (PC, FCI, RFCI, anytime-FCI, PCMCI) | `cbcd/` | local-only; v0.x API stable per D15 |
| **citk** | conditional independence test toolkit (FisherZ, Spearman native; KCI/CMIknn/RegressionCI/GCM/etc. via optional extras) | `citk/` | local-only; Phase 1+2 committed (decoupled from causal-learn) |
| **bnm** | DAG/CPDAG comparison metrics + visualization (SHD, HD, F1, SID, Markov-blanket comparisons) | `bnm/` | `0.1.0` legacy on GitHub; rewrite pending |

Plus `vendor/` with read-only reference repos (`causal-learn`, `tigramite`, etc.)
used for parity testing — never imported from suite code.

## Architectural pattern

Every cross-package interaction goes through a **structural Protocol** —
no package imports another. The pattern is already in place between
cbcd and citk; bnm and dagsampler should follow the same template.

```
dagsampler ─── true_dag, data, ci_oracle ──▶ cbcd ─── recovered ──▶ bnm
                                              ▲                     ▲
                                              │ Protocol            │ Protocol
                                            citk                  (graph)
                                              │
                                          (CI tests via cbcd.CITest)
```

| boundary | contract | who defines it |
|---|---|---|
| `citk → cbcd` | `cbcd.CITest` Protocol (`n_vars`, `__call__`, `details`) | cbcd (frozen under D15) |
| `dagsampler → cbcd` (CI oracle) | `cbcd.CITest` Protocol — dagsampler exposes `as_ci_oracle()` returning a callable conforming object | cbcd |
| `dagsampler → bnm` / `cbcd → bnm` (graphs) | `bnm.GraphLike` Protocol (`n_vars`, `endpoints` int8 matrix) | bnm (TBD; part of the rewrite) |

If you find yourself adding a `from cbcd import ...` line inside citk,
bnm, or dagsampler — stop and revisit the Protocol contract. Same for
any reverse direction.

## Where to write logs

- **`suite/journal.md`**: cross-package decisions, suite design changes,
  push/release coordination, audit findings that affect more than one
  package. Newest entry on top, dated.
- **`<pkg>/docs/journal.md`**: per-package implementation history.
  cbcd already has one. citk, bnm, dagsampler get theirs as work
  starts.
- **`<pkg>/docs/design/api_v0.md` (or `.py`)**: per-package design
  source-of-truth, pressure-tested before implementation. cbcd's is
  the reference (`cbcd/docs/design/api_v0.py`).

## Working pattern

```bash
# Cross-package work — full read/write to all four packages, no
# permission friction:
cd ~/Projects/suite
claude

# Deep package-specific work — tighter context, faster:
cd ~/Projects/suite/<pkg>
claude
```

## Commands by package

Each package has its own dev environment and tooling. No shared venv.

```bash
# cbcd
cd cbcd && uv sync --all-extras && uv run pytest

# citk
cd citk && uv sync --all-extras && uv run pytest

# dagsampler
cd dagsampler && uv sync --all-extras && uv run pytest

# bnm (current 0.1.x; setuptools-based)
cd bnm && pip install -e . && pytest
```

## Currently in flight (top of mind)

See `suite/journal.md` for the latest. As of suite scaffold:

- bnm v0.1.x → v0.2 rewrite **shipped 2026-05-07** (see journal).
- dagsampler `as_ci_oracle()` **shipped 2026-05-08**, dagsampler 0.2.0.
- End-to-end tutorial **shipped 2026-05-08** at `suite/docs/tutorial.md`
  — 10-line copy-pasteable dagsampler → citk → cbcd → bnm story;
  verified to produce `SHD: 0, F1: 1.0` on the canonical collider
  example.
- citk taxonomic restructure **shipped 2026-05-08** — `citk/tests/`
  is now organized strictly by survey family.
- Suite-level integration test **shipped 2026-05-08** at
  `suite/parity/suite/run.py` — 5 fixtures passing, `uv sync && uv
  run python run.py` from the parity dir.
- First public push of cbcd then citk then bnm (then dagsampler 0.2.0)
  — pending explicit user instruction per package.

## Push policy

**No package is pushed without explicit user instruction.** dagsampler
is already public on PyPI; further releases there also require
explicit go. cbcd, citk, bnm have local commits ahead of remotes;
none push until the user names the package. See the saved memory
for the full no-push contract.
