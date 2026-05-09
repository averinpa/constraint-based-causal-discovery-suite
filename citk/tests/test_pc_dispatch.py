"""End-to-end tests of the PC-algorithm dispatch path that the harness uses.

For each of the 19 CI tests, this module verifies that
``pc(data, indep_test=registered_name, cache_path=..., data_type=...)`` runs
to completion and writes a well-formed v0.1.0 cache JSON. Each test runs in
its own ``tmp_path`` directory so JSON outputs are isolated per test and
auto-cleaned by pytest.

Tests with optional dependencies (rpy2, tigramite, pycomets) are gated via
``pytest.importorskip`` and skip cleanly when the dep is unavailable.

This complements ``test_ci_smoke.py`` (standalone calls only) so every test
is exercised in both forms.
"""
import gc
import json
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pytest

# Trigger register_ci_test() side effects so pc() can dispatch by name.
import citk.tests  # noqa: F401
from causallearn.search.ConstraintBased.PC import pc


# ---------------------------------------------------------------------------
# Per-test dispatch table
# ---------------------------------------------------------------------------
# (registered_name, dtype, per_test_kwargs, optional_dep_module_name)
DISPATCH_TABLE: list[Tuple[str, str, dict, Optional[str]]] = [
    # Always-available
    ("fisherz_citk",   "continuous", {},                                    None),
    ("spearman",       "continuous", {},                                    None),
    ("chisq",          "discrete",   {},                                    None),
    ("gsq",            "discrete",   {},                                    None),
    ("kci",            "continuous", {},                                    None),
    ("disc_chisq",     "continuous", {"n_bins": 4},                         None),
    ("disc_gsq",       "continuous", {"n_bins": 4},                         None),
    ("dummy_fisherz",  "discrete",   {},                                    None),
    ("mcmiknn",        "continuous", {"test_kwargs": {"Mperm": 49}},        None),
    # Optional-dep gated
    ("rcit",           "continuous", {},                                    "rpy2"),
    ("rcot",           "continuous", {},                                    "rpy2"),
    ("hartemink_chisq","continuous", {"breaks": 3, "ibreaks": 6},           "rpy2"),
    ("ci_mm",          "continuous", {},                                    "rpy2"),
    ("cmiknn",         "continuous", {"test_kwargs": {"sig_samples": 49}},  "tigramite"),
    ("cmiknn_mixed",   "continuous", {"test_kwargs": {"sig_samples": 49}},  "tigramite"),
    ("regci",          "continuous", {},                                    "tigramite"),
    ("gcm",            "continuous", {},                                    "pycomets"),
    ("wgcm",           "continuous", {},                                    "pycomets"),
    ("pcm",            "continuous", {},                                    "pycomets"),
]


def _make_chain_data(dtype: str, seed: int, n: int = 80) -> np.ndarray:
    """Generate a 3-variable chain X → Z → Y so PC has something to learn."""
    rng = np.random.default_rng(seed)
    if dtype == "continuous":
        x = rng.normal(size=n)
        z = 0.7 * x + 0.4 * rng.normal(size=n)
        y = 0.7 * z + 0.4 * rng.normal(size=n)
        return np.column_stack([x, y, z])
    x = rng.integers(0, 3, size=n)
    z = (x + rng.integers(0, 2, size=n)) % 3
    y = (z + rng.integers(0, 2, size=n)) % 3
    return np.column_stack([x, y, z]).astype(float)


def _data_type_for(dtype: str, shape: Tuple[int, int]) -> np.ndarray:
    """Build the (n, p) data_type array the harness passes through pc().

    Mirrors the tier2 convention from
    ``papers/paper1-benchmark/experiments/tier2/run_tier2.py:289-291``: a per-
    column dtype tag tiled to the full data shape so tigramite's DataFrame
    accepts it without a shape-mismatch error.
    """
    val = 0 if dtype == "continuous" else 1
    return np.full(shape, val, dtype=int)


# ---------------------------------------------------------------------------
# PC + cache JSON dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,dtype,kwargs,dep",
    DISPATCH_TABLE,
    ids=[t[0] for t in DISPATCH_TABLE],
)
def test_pc_dispatch_writes_v0_1_0_cache(
    name: str,
    dtype: str,
    kwargs: dict,
    dep: Optional[str],
    tmp_path,
) -> None:
    """Run pc(...) end-to-end and verify the v0.1.0 cache contract.

    Asserts:
    - pc() returns a CausalGraph without raising.
    - cache JSON exists at the requested path.
    - format_version == "1.0".
    - data_hash is a 64-char sha256 hex digest.
    - method_name and parameters_hash are populated.
    - at least one p-value entry was written.
    """
    if dep is not None:
        pytest.importorskip(dep)

    data = _make_chain_data(dtype, seed=hash(name) & 0xFFFF)
    data_type = _data_type_for(dtype, data.shape)
    cache_path = tmp_path / f"{name}_cache.json"

    pc_kwargs: dict[str, Any] = {
        "alpha": 0.05,
        "indep_test": name,
        "show_progress": False,
        "cache_path": str(cache_path),
        "data_type": data_type,
    }
    pc_kwargs.update(kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pc(data, **pc_kwargs)

    assert result is not None
    assert result.G is not None

    # The cache-bearing test instance is local to causal-learn's pc_alg() and
    # becomes GC-eligible once pc() returns. CPython refcount-GC normally fires
    # __del__ immediately, but some backends (e.g. tigramite) hold reference
    # cycles that defer collection — force a cycle pass so save_cache() runs
    # before pytest tears down tmp_path.
    del result
    gc.collect()

    assert cache_path.exists(), f"{name} did not write a cache file"
    with cache_path.open() as fh:
        cache = json.load(fh)

    assert cache["format_version"] == "1.0", \
        f"{name} cache missing v0.1.0 format_version"
    assert len(cache["data_hash"]) == 64, \
        f"{name} data_hash is not a 64-char sha256 hex (got {len(cache['data_hash'])})"
    assert "method_name" in cache
    assert "parameters_hash" in cache

    pvalue_keys = [k for k, v in cache.items()
                   if ";" in k and isinstance(v, str)]
    assert pvalue_keys, f"{name} cache contains no p-value entries"


# ---------------------------------------------------------------------------
# Cache reload semantics: second pc() with the same path hits the existing
# entries without raising.
# ---------------------------------------------------------------------------


def test_cache_reload_hits_existing_entries(tmp_path):
    """A second pc() against the same cache_path must reuse cached p-values."""
    data = _make_chain_data("continuous", seed=0)
    data_type = _data_type_for("continuous", data.shape[1])
    cache_path = tmp_path / "fisherz_reload.json"

    common = {
        "alpha": 0.05,
        "indep_test": "fisherz_citk",
        "show_progress": False,
        "cache_path": str(cache_path),
        "data_type": data_type,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pc(data, **common)
    del result
    gc.collect()
    first = json.loads(cache_path.read_text())
    n_first = sum(1 for k in first if ";" in k)
    assert n_first > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pc(data, **common)
    del result
    gc.collect()
    second = json.loads(cache_path.read_text())

    assert second["data_hash"] == first["data_hash"]
    assert second["format_version"] == first["format_version"]
    # Re-running on the same data may add tests we hadn't done before, but
    # never less than what was already there.
    assert sum(1 for k in second if ";" in k) >= n_first


# ---------------------------------------------------------------------------
# Stale (pre-v0.1.0) cache regenerates with a RuntimeWarning
# ---------------------------------------------------------------------------


def test_pc_dispatch_invalidates_stale_cache(tmp_path):
    """A pre-v0.1.0 cache (md5 hash, no format_version) must regenerate
    with a RuntimeWarning rather than asserting on data_hash mismatch."""
    cache_path = tmp_path / "stale_cache.json"
    cache_path.write_text(json.dumps({
        "data_hash": "deadbeef" * 4,  # md5-style, not sha256
        "method_name": "fisherz_citk",
        "parameters_hash": "NO SPECIFIED PARAMETERS",
        "0;1|2": "0.5",  # bogus pre-existing entry
    }))

    data = _make_chain_data("continuous", seed=1)
    data_type = _data_type_for("continuous", data.shape[1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pc(
            data,
            alpha=0.05,
            indep_test="fisherz_citk",
            show_progress=False,
            cache_path=str(cache_path),
            data_type=data_type,
        )
        mismatch = [w for w in caught
                    if issubclass(w.category, RuntimeWarning)
                    and "mismatch" in str(w.message)]
        assert mismatch, "stale cache must emit a format/hash-mismatch RuntimeWarning"
    del result
    gc.collect()

    rebuilt = json.loads(cache_path.read_text())
    assert rebuilt["format_version"] == "1.0"
    assert len(rebuilt["data_hash"]) == 64
    assert "0;1|2" not in rebuilt or rebuilt["0;1|2"] != "0.5"
