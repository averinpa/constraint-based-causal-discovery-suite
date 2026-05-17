"""cbcd interop smoke tests.

Verifies that citests's CI test classes satisfy the structural ``cbcd.CITest``
Protocol — no inheritance from a cbcd class, no import of cbcd in citests's
own code, just shape-conformant attributes/methods.

Skipped when cbcd is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

cbcd = pytest.importorskip("cbcd")

from citests import CITKResult
from citests.tests.partial_correlation_tests import FisherZ, Spearman


def _chain_data(seed: int = 0, n: int = 1000) -> np.ndarray:
    """Generate Linear-Gaussian data from a chain X → Z → Y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = 0.7 * x + rng.normal(size=n) * 0.5
    y = 0.7 * z + rng.normal(size=n) * 0.5
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_citk_fisherz_satisfies_cbcd_citest_protocol():
    data = _chain_data()
    cit = FisherZ(data)
    assert isinstance(cit, cbcd.CITest)


def test_citk_test_exposes_n_vars_attribute():
    data = _chain_data()
    cit = FisherZ(data)
    assert cit.n_vars == 3
    assert cit.n_vars == cit.num_features


def test_citk_test_details_returns_object_with_p_value():
    data = _chain_data()
    cit = FisherZ(data)
    res = cit.details(0, 1, [2])
    assert isinstance(res, CITKResult)
    assert hasattr(res, "p_value")
    assert isinstance(res.p_value, float)
    assert 0.0 <= res.p_value <= 1.0


# ---------------------------------------------------------------------------
# End-to-end use through cbcd algorithms
# ---------------------------------------------------------------------------


def test_citk_fisherz_drives_cbcd_pc():
    """``cbcd.pc()`` accepts a citests FisherZ instance and recovers the
    chain CPDAG (X — Z — Y, no v-structure)."""
    data = _chain_data(seed=1)
    cit = FisherZ(data)
    cpdag = cbcd.pc(data, ci_test=cit, alpha=0.01)
    assert cpdag.n_vars == 3
    # Chain X → Z → Y has no unshielded collider; PC's CPDAG is fully
    # undirected (every edge is TAIL-TAIL).
    directed = set(cpdag.directed_edges())
    undirected = {frozenset(s) for s in cpdag.undirected_edges()}
    assert directed == set()
    assert undirected == {frozenset({0, 2}), frozenset({1, 2})}


def test_citk_spearman_drives_cbcd_pc():
    data = _chain_data(seed=2)
    cit = Spearman(data)
    cpdag = cbcd.pc(data, ci_test=cit, alpha=0.01)
    assert cpdag.n_vars == 3


def test_citk_fisherz_caches_through_cbcd():
    """When wrapped by cbcd.CachedCITest (which the algorithms do
    automatically), citests's details() return value is what gets cached.
    Subsequent identical calls must return the same float."""
    data = _chain_data(seed=3)
    cit = FisherZ(data)
    cached = cbcd.CachedCITest(cit)
    p1 = cached(0, 1, [2])
    p2 = cached(0, 1, [2])
    p3 = cached(1, 0, [2])  # unordered key: same cache hit
    assert p1 == p2 == p3
    assert cached.is_cached(0, 1, [2])
