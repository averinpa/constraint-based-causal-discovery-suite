# Parity report — cbcd vs causal-learn / tigramite

**Date:** 2026-05-06
**Versions:** `causal-learn 0.1.4.5`, `tigramite 5.2.10.1`
**Harness:** `parity/run_all.py` (commit `e4540c9`).

## Summary

| Family | Fixtures | cbcd matches truth | reference matches truth | cbcd ≡ reference | real disagreements |
|--------|---------:|-------------------:|------------------------:|-----------------:|-------------------:|
| PC     | 6/6      | 6                  | 6                       | 6                | 0                  |
| FCI    | 4/4      | 4                  | 4                       | 4                | 0                  |
| PCMCI  | 3/3      | 3                  | 3                       | 3                | 0                  |
| **all**| **13/13**| **13**             | **13**                  | **13**           | **0**              |

cbcd's outputs are byte-equal to the references on every fixture in the
harness. No "real disagreements" — cases where both implementations match
the d-sep / m-sep truth but produce different endpoint matrices.

## What the harness covers

### PC family (`parity/pc/run.py`)

For each fixture from `tests/fixtures.py` (Y-structure, fork, chain,
M-structure, diamond, bnlearn ASIA), simulate Linear-Gaussian data
(`n=5000`, `edge_coef=0.7`, `noise_scale=0.5`, fixed seed), run
`cbcd.pc()` and `causal-learn pc()` with `alpha=0.01` and Fisher-Z, then
compare CPDAG matrices after converting causal-learn's mark convention
to cbcd's.

All six fixtures yield identical CPDAGs.

### FCI family (`parity/fci/run.py`)

For each fixture from `tests/fixtures_pag.py` (Y-structure, chain,
fork, confounded_chain_through_collider — the one that exercises
Zhang R1, R2, and R4), simulate Linear-Gaussian data over the FULL
DAG (with latents), drop the latent columns, and run both
implementations on the observed-only data with `alpha=0.01` and
Fisher-Z.

All four fixtures yield identical PAGs, including the confounded
chain that requires the discriminating-path R4 to fire correctly.

The audit (`docs/audit_causal_learn.md`) flagged
`removeByPossibleDsep` as exponential and the CDNOD direction
inconsistency in causal-learn. None of those bugs is exercised by
this fixture set — the four cases here have small Possible-D-Sep
sets that fit comfortably inside causal-learn's depth limit, and
neither uses CDNOD. To pressure-test the audited bugs we'd need
larger fixtures with deep PossibleDSep paths; not in the harness yet.

### PCMCI family (`parity/pcmci/run.py`)

Three VAR fixtures (`ar1_single`, `two_var_var1`, `sparse_var2` with
mixed `tau=1` and `tau=2` edges), each simulated for `T=2000` rows with
fixed coefficients and noise. Both `cbcd.pcmci()` and tigramite's PCMCI
run with ParCorr, `alpha=0.01`, `pc_alpha=0.05`, and `tau_max =
fixture.max_lag`. cbcd's `TimeSeriesCPDAG.endpoints` is compared to
tigramite's `results['graph']` after string→mark conversion.

All three fixtures yield identical lagged endpoint arrays.

## What this does NOT prove

- **Edge cases near the alpha boundary.** Where p-values land within
  ~10% of `alpha`, finite-sample noise can flip individual edges. Both
  implementations are deterministic given a fixed seed, but a different
  random draw may swing borderline edges. The harness uses fixed seeds.
- **Selection-bias scenarios.** Zhang R5–R7 (selection bias) do not
  fire on any current FCI fixture. cbcd has rule-level unit tests for
  R5–R7 in `tests/rules/test_fci.py`, but not parity tests against
  causal-learn for those rules.
- **PCMCI+ / LPCMCI / tsFCI / SVAR-FCI / J-PCMCI.** Not implemented in
  cbcd yet, so no parity check possible.
- **Conservative-PC / Majority-PC / mvpc / CDNOD / JCI / IOD.** Same —
  not implemented in cbcd yet.
- **Larger graphs that stress causal-learn's audited bugs.** The
  fixture set is intentionally small to make hand-verification easy.
  Expanding to graphs with deep Possible-D-Sep paths, autocorrelation
  in the latent, or more variables would be a useful follow-up.

## Reproducing

```bash
uv pip install causal-learn tigramite   # not in pyproject.toml
uv run python -m parity.run_all
```

The reference packages are installed into the cbcd venv on demand but
are not part of cbcd's declared dependencies. A fresh `uv sync` will
remove them; re-run the install above to repeat the harness.

## Audit context

The original audit at `docs/audit_causal_learn.md` flagged five top
issues in causal-learn (md5 cache collision, CDNOD direction
inconsistency, missing parallelism, exponential `removeByPossibleDsep`,
and 4× duplicated PC orchestration). None of these bite on the
fixtures in this harness, so causal-learn's outputs are correct here
and cbcd matches them exactly. A larger fixture set could begin to
expose the exponential refinement and the cache collision; that's
deferred work.
