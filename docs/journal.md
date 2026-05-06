# citk development journal

Append-only log of what's been done, when, and why. New entries at the **top**.

For cross-package decisions affecting citk's relationship to cbcd / bnm /
dagsampler, see `~/Projects/suite/journal.md`.

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
