# citk development journal

Append-only log of what's been done, when, and why. New entries at the **top**.

For cross-package decisions affecting citk's relationship to cbcd / bnm /
dagsampler, see `~/Projects/suite/journal.md`.

---

## 2026-05-08 — Taxonomic restructure of `citk/tests/`

`citk/tests/` is now organized strictly by **survey family**, not by
backend. Family modules — `partial_correlation_tests`,
`contingency_table_tests`, `regression_tests`, `nearest_neighbor_tests`,
`kernel_tests`, `ml_based_tests`, `adapter_tests` — are the canonical
homes; the backend-named modules that previously held the
implementations have been deleted.

**Deleted:**

- `simple_tests.py`
- `extended_tests.py`
- `r_based_tests.py`
- `tigramite_based_tests.py`
- `pycomets_tests.py`
- `external_repo_tests.py`

**Class moves:**

| class | old file | new file |
|---|---|---|
| `FisherZ`, `Spearman` | `simple_tests.py` | `partial_correlation_tests.py` |
| `ChiSq`, `GSq` | `simple_tests.py` | `contingency_table_tests.py` |
| `RegressionCI` | `tigramite_based_tests.py` | `regression_tests.py` |
| `CiMM` | `r_based_tests.py` | `regression_tests.py` |
| `CMIknn`, `CMIknnMixed` | `tigramite_based_tests.py` | `nearest_neighbor_tests.py` |
| `MCMIknn` | `external_repo_tests.py` | `nearest_neighbor_tests.py` |
| `RCoT`, `RCIT` | `r_based_tests.py` | `kernel_tests.py` |
| `KCI` | (already in `kernel_tests.py`) | unchanged |
| `GCM`, `WGCM`, `PCM` | `pycomets_tests.py` | `ml_based_tests.py` |
| `DiscChiSq`, `DiscGSq`, `DummyFisherZ` | `extended_tests.py` | `adapter_tests.py` |
| `HarteminkChiSq` | `r_based_tests.py` | `adapter_tests.py` |

**New private module: `_backends.py`** — holds the cross-cutting
loaders that used to live duplicated in backend files:
`_load_rcit_package`, `_load_bnlearn_package`, `_load_mxm_package`
(rpy2-side); `_load_tigramite`, `_load_tigramite_test_class`,
`_extract_tigramite_pvalue`, and the shared `_TigramiteBase`
(tigramite-side); plus the small R conversion helpers `_to_r_vector`,
`_to_r_matrix`, `_extract_rcit_p_value`. Family modules import only
what they need.

**Public API: unchanged.** `citk.tests.__init__` already re-exported
every test class by family, and the family-named modules already
existed as thin re-export shims, so all canonical user-facing imports
(`from citk.tests import FisherZ`, `from citk.tests.kernel_tests import
KCI`, etc.) continue to work without change.

**Internal stragglers updated:**

- `tests/test_ci_smoke.py`: `external_repo_tests.MCMIknn` →
  `nearest_neighbor_tests.MCMIknn`; `r_based_tests.CiMM` →
  `regression_tests.CiMM` (two call sites).
- `examples/run_fisherz_spearman_synthetic.py`: `simple_tests.FisherZ` →
  `partial_correlation_tests.FisherZ`.
- `citk/tests/base.py`: docstring reference to `extended_tests` →
  `adapter_tests`.
- `docs/source/tests/{fisher_z_test,spearman,chi_sq_test,g_sq_test}.md`:
  `:class:` references updated to point at the family modules.

**Tests:** 38 passing, 11 skipped, 3 failing — same numbers as before
the refactor. The 3 failures are a pre-existing tigramite/numpy
version drift (`corrcoef() got an unexpected keyword argument 'ddof'`),
verified by running the same tests against the pre-refactor tree via
`git stash`. Not introduced here.

**Suite-tutorial verbatim re-run:** `SHD: 0, F1: 1.0` unchanged on the
canonical 3-node collider example. The user-facing tutorial path
(`from citk.tests.partial_correlation_tests import FisherZ`) is now
the canonical taxonomic home, not a re-export.

**Not yet pushed.** Per the no-push contract, citk waits on explicit
user instruction before any first push to its remote.

---

## 2026-05-06 — Suite migration

citk moved from `~/Projects/citk/` to `~/Projects/suite/citk/` as part
of scaffolding the four-package suite umbrella. No code, no commits, no
remotes affected. The cbcd install in citk's venv was refreshed to point
to the new path (`~/Projects/suite/cbcd/`); cbcd-compat tests (6/6) still
pass.

---

## 2026-05-06 — Phase 2: decouple from causal-learn (now an optional `[causallearn]` extra)

citk's core (FisherZ, Spearman, base class, registration helper) no
longer requires causal-learn. The package can be installed and used
standalone — most usefully alongside cbcd, where citk classes plug in
via the structural `cbcd.CITest` Protocol with no cross-package imports.

### Changes

- `citk/tests/base.py` — `CITKTest` is now a standalone class. Methods
  previously inherited from `causallearn.utils.cit.CIT_Base`
  (`assert_input_data_is_valid`, `check_cache_method_consistent`,
  `save_to_local_cache`, `get_formatted_XYZ_and_cachekey`) are
  reimplemented locally. The sha256 cache-key behaviour is preserved.
  The `NO_SPECIFIED_PARAMETERS_MSG` sentinel is re-defined with the
  same string value so existing JSON caches stay readable.

- `citk/tests/_register.py` — new module. `maybe_register(name, cls)`
  silently no-ops when causal-learn isn't installed; otherwise
  dynamically creates a `CIT_Base`-inheriting adapter (multiple
  inheritance with `citk_cls` first in MRO) and registers it. Every
  `register_ci_test` call across the test modules has been replaced
  with `maybe_register`.

- `citk/tests/simple_tests.py` — `FisherZ` and `Spearman` re-implemented
  natively (Schur-complement partial correlation + Fisher-Z transform).
  No more wrapping causal-learn's `CIT` class. `ChiSq` / `GSq` still
  wrap causal-learn's `Chisq_or_Gsq` but their definitions are now gated
  behind `try/except` — when causal-learn is missing, they bind to
  `None` placeholders.

- `citk/tests/extended_tests.py` — `DiscChiSq`, `DiscGSq`, `DummyFisherZ`
  similarly gated; bound to `None` without causal-learn.

- `citk/tests/kernel_tests.py` — `KCI` similarly gated. RCIT/RCoT
  re-exports unchanged (R-based, independent of causal-learn).

- `citk/tests/r_based_tests.py` — `HarteminkChiSq` raises
  `CITKDependencyError` on construction if causal-learn is missing
  (uses `Chisq_or_Gsq` internally). Other R-based tests (`RCIT`,
  `RCoT`, `CiMM`) stay functional regardless.

- `citk/tests/tigramite_based_tests.py` / `pycomets_tests.py` /
  `external_repo_tests.py` — only registration imports changed
  (`register_ci_test` → `maybe_register`); the test classes themselves
  wrap their respective backends and never touched causal-learn beyond
  the registration glue.

- `pyproject.toml` — causal-learn moved out of `dependencies` and into
  `[project.optional-dependencies]` as the new `causallearn` extra.

### Verification

- **`tests/test_cbcd_compat.py`**: 6/6 pass. Includes
  `isinstance(cit, cbcd.CITest)` Protocol check, cbcd.pc end-to-end
  with citk's FisherZ, CachedCITest caching invariants.

- **With causal-learn UNINSTALLED**: every `from citk.tests... import
  X` still imports cleanly; FisherZ + Spearman work; ChiSq/GSq/KCI/
  DiscChiSq are `None` placeholders; `cbcd.pc(data,
  ci_test=citk.FisherZ(data))` runs end-to-end.

- **With causal-learn INSTALLED**: `maybe_register` registers all citk
  classes with causal-learn's PC dispatch via dynamic adapter
  subclasses. `tests/test_pc_dispatch.py` confirms 12/19 dispatch
  paths work; the remaining 7 are pre-existing tigramite/numpy compat
  issues (`numba` missing, `corrcoef(ddof=...)` removed from numpy)
  unrelated to this refactor.

Commit: `8db7f2a`.

---

## 2026-05-06 — Phase 1: cbcd Protocol conformance (`CITKResult` + `n_vars` + `details()`)

Make every citk CI test plug into cbcd algorithms via the structural
`cbcd.CITest` Protocol — no inheritance from a cbcd class, no import of
cbcd from citk code. Three additions to `CITKTest`:

- **`CITKResult`** dataclass (`citk/tests/base.py`): field-compatible
  with `cbcd.CITestResult` (`p_value`, `statistic`, `df`, `n_effective`,
  `extra`) so cbcd's `CachedCITest` can read `.p_value` from cached
  results without any cross-package import.

- **`n_vars`** property: alias for `self.num_features`. Required by the
  `cbcd.CITest` Protocol.

- **`details(X, Y, condition_set)`** method: default wraps `__call__`
  in a `CITKResult`; subclasses with richer diagnostics can override.

Plus `tests/test_cbcd_compat.py` — six smoke tests including:

- `isinstance(cit, cbcd.CITest)` Protocol check.
- `cbcd.pc(data, ci_test=citk.FisherZ(data))` end-to-end recovers the
  chain CPDAG.
- `CachedCITest` wrapping works correctly (unordered `(x, y)` cache key).

No breaking changes: every existing causal-learn integration path
continues to work as before. Phase 1 was the small, low-risk first
step toward the cbcd↔citk integration; Phase 2 (above) is the
follow-up that decouples the package from causal-learn entirely.

Commit: `4a0ce49`.
