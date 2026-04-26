# API Stability — v0.1.0 Contract

`citk` follows a strict additive-only policy starting from v0.1.0. This page documents the public surface, the guarantees attached to it, and the deliberate edges that v0.1.0 will *not* hide. Read this before depending on citk in another package or pinning a version in CI.

## Versioning policy

`citk` uses semantic versioning.

- **Patch (`0.1.x`)**: bug fixes only. No public-surface changes.
- **Minor (`0.y.0`, `y > 1`)**: additive changes only. New tests, new kwargs (with backwards-compatible defaults), new helpers, new exception subclasses. Existing user code continues to work without modification.
- **Major (`y.0.0`, `y > 0`)**: may remove or rename public symbols. Pre-v1 minor releases follow the additive rule above; the stricter v1 contract applies once tagged.

Any breaking change before v1 will be flagged in the release notes and given a deprecation warning for at least one minor cycle.

## Stable public surface

The following symbols are part of the v0.1.0 contract. Anything not listed is internal and may change without notice.

### Test classes (19)

Importable from `citk.tests`:

| Family | Classes |
|---|---|
| Partial Correlation | `FisherZ`, `Spearman` |
| Contingency Table | `ChiSq`, `GSq` |
| Regression | `RegressionCI`, `CiMM` |
| Nearest Neighbor | `CMIknn`, `CMIknnMixed`, `MCMIknn` |
| Kernel | `KCI`, `RCIT`, `RCoT` |
| ML-Based | `GCM`, `WGCM`, `PCM` |
| Adapter Strategies | `DiscChiSq`, `DiscGSq`, `DummyFisherZ`, `HarteminkChiSq` |

Each class follows the same protocol:

```python
test = TestClass(data, cache_path=None, **per_test_kwargs)
p_value = test(X, Y, condition_set)        # int, int, list[int] | None → float
test.save_cache()                           # explicit cache flush
```

Per-test constructor kwargs are documented on the test's reference page under :doc:`/tests/index`.

### Base class

`citk.tests.base.CITKTest` is the abstract base. It is technically importable but **not** part of the v0.1.0 contract — its private implementation may change. Subclassing `CITKTest` to register a custom CI test is supported via `causallearn.utils.cit.register_ci_test`.

### Exception hierarchy

Importable from `citk` (top-level):

| Class | Inherits from | Raised when |
|---|---|---|
| `CITKError` | `Exception` | Base; catch this for any citk failure. |
| `CITKDependencyError` | `CITKError`, `ImportError` | An optional dependency (e.g. `rpy2`, an R package, `tigramite`) is missing or fails to load. |
| `CITKComputationError` | `CITKError`, `RuntimeError` | A test failed during computation: numerical issue, unexpected upstream result shape, or an exception escaping from a wrapped library. |
| `CITKDataError` | `CITKError`, `ValueError` | The input data is invalid for the requested test (declared but currently unused; reserved for future v0.x additions). |

Each leaf multiple-inherits from a relevant stdlib type, so existing user code that catches `ImportError` / `RuntimeError` / `ValueError` continues to work unchanged.

#### Exception policy: dep wrapping

The `CITKTest.__call__` boundary wraps any non-`CITKError` exception escaping from `_compute()` in a `CITKComputationError`, with the original exception preserved on `__cause__`. **This is a deliberate v0.1.0 contract change** from earlier ad-hoc behavior: users who previously caught e.g. `rpy2.rinterface.RRuntimeError` directly will now receive `CITKComputationError`. To inspect the original cause:

```python
try:
    test(X, Y, S)
except CITKComputationError as exc:
    underlying = exc.__cause__   # the rpy2 / numpy / scipy / tigramite original
    ...
```

`CITKDependencyError` is *not* re-wrapped at the boundary (it inherits from `CITKError` and propagates as-is), so dependency-missing failures retain their semantic distinction from computation failures.

### Helpers

Importable from `citk.tests.base`:

- `hash_parameters(params: Mapping | None) -> str` — stable sha256 hex of canonicalised constructor kwargs, used as the cache `parameters_hash`. Returns the literal `"NO SPECIFIED PARAMETERS"` for empty / None input. Order-independent over dict keys; handles numpy arrays by typed canonicalisation.
- `inner_test_kwargs(kwargs: Mapping) -> dict` — strips `cache_path` from a kwargs dict before forwarding to a wrapped upstream test instance. citk's outer wrapper owns the cache.
- `CACHE_FORMAT_VERSION = "1.0"` — module constant.

### Cache file format

JSON, with three required top-level fields and zero or more p-value entries:

```json
{
  "format_version": "1.0",
  "data_hash": "<sha256 hex of np.ascontiguousarray(data).tobytes()>",
  "method_name": "<test method name, e.g. 'fisherz_citk'>",
  "parameters_hash": "<sha256 hex of canonicalised kwargs, or NO SPECIFIED PARAMETERS>",
  "<X>;<Y>": "<float as string>",
  "<X>;<Y>|<S0>,<S1>,...": "<float as string>"
}
```

Cache load policy:

- A cache whose `format_version` does not match the running citk version is **silently regenerated** with a `RuntimeWarning`.
- A cache whose `data_hash` mismatches the current data is regenerated.
- Empty or unreadable cache files start fresh.
- A cache whose `method_name` or `parameters_hash` does not match the test instance raises `AssertionError` (an actual programming error: same file, different test or different parameters).

This means caches generated under v0.1.0 are *not* portable to a v0.2.0 that bumps `format_version`; users should expect to regenerate. The format_version field exists precisely so future bumps degrade gracefully rather than silently corrupting.

## Out-of-contract: causal-learn-inherited methods

`CITKTest` inherits four public methods from `causallearn.utils.cit.CIT_Base` that are visible on every test instance:

- `assert_input_data_is_valid(allow_nan=False, allow_inf=False)`
- `check_cache_method_consistent(method_name, parameters_hash)`
- `get_formatted_XYZ_and_cachekey(X, Y, condition_set)`
- `save_to_local_cache()`

These are **inherited from causal-learn upstream** and are not part of citk's v0.1.0 contract. Their signatures and behaviour follow whatever version of `causal-learn` is installed; citk does not guarantee stability across causal-learn upgrades for these methods. If you need stable behaviour, vendor or pin causal-learn explicitly.

## Per-test edges

Two known asymmetries across the 19 tests are documented on per-test pages and are *not* considered bugs in v0.1.0:

1. **NaN as a p-value (not as an exception).** GCM / WGCM / PCM may return `NaN` for degenerate data (e.g., the pycomets `RuntimeWarning: invalid value in scalar divide` path), rather than raising. Downstream consumers must decide whether to treat `NaN` as missing or to filter it. This is intentional and matches the harness expectation; raising would be the breaking change.
2. **Empty conditioning set semantics.** GCM / WGCM / PCM substitute a constant column $Z = 0$ when the conditioning set is empty, instead of taking the no-conditioning path. The other 16 tests pass empty `Z` through unchanged. See the per-test pages for :doc:`/tests/gcm_test`, :doc:`/tests/wgcm_test`, :doc:`/tests/pcm_test`.

## What the harness relies on (v0.1.0 minimum protocol)

The reference consumer (the Paper 1 benchmark harness) exercises a strict subset of the public surface. v0.1.0 guarantees this subset will not change:

- Constructor: `cls(data: np.ndarray, data_type=data_type_array, **test_kwargs)` works for every class.
- Callable: `test(x_idx: int, y_idx: int, s_idx: list[int]) -> float`.
- No instance attribute access (no `.pvalue`, `.statistic`, `.last_stat`, etc.).
- Exceptions are caught generically; no specific exception type is required.

If your downstream tooling stays within this subset, v0.1.0 → v0.x.0 upgrades will not require code changes.
