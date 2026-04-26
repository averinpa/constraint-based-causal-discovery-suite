"""Tests for the v0.1.0 validation surface: validate_data + kwargs allowlist."""
import numpy as np
import pytest

from citk.tests import (
    ChiSq,
    CiMM,
    CMIknn,
    CMIknnMixed,
    DiscChiSq,
    DiscGSq,
    DummyFisherZ,
    FisherZ,
    GCM,
    GSq,
    HarteminkChiSq,
    KCI,
    MCMIknn,
    PCM,
    RCIT,
    RCoT,
    RegressionCI,
    Spearman,
    WGCM,
)


ALL_19 = [
    FisherZ, Spearman, ChiSq, GSq, RegressionCI, CiMM,
    CMIknn, CMIknnMixed, MCMIknn, KCI, RCIT, RCoT,
    GCM, WGCM, PCM,
    DiscChiSq, DiscGSq, DummyFisherZ, HarteminkChiSq,
]


# ---------------------------------------------------------------------------
# Finding 1: validate_data classmethod
# ---------------------------------------------------------------------------


def test_validate_data_continuous_passes_continuous_test():
    np.random.seed(0)
    ok, reason = FisherZ.validate_data(np.random.randn(100, 3))
    assert ok is True
    assert reason == ""


def test_validate_data_continuous_fails_discrete_test():
    np.random.seed(0)
    ok, reason = ChiSq.validate_data(np.random.randn(100, 3))
    assert ok is False
    assert "continuous" in reason
    assert "ChiSq" in reason


def test_validate_data_discrete_passes_discrete_test():
    np.random.seed(0)
    ok, reason = ChiSq.validate_data(np.random.randint(0, 3, (100, 3)))
    assert ok is True
    assert reason == ""


def test_validate_data_mixed_test_accepts_anything():
    np.random.seed(0)
    cont = np.random.randn(100, 3)
    disc = np.random.randint(0, 3, (100, 3)).astype(float)
    for cls in (DiscChiSq, RegressionCI, CMIknn, MCMIknn):
        ok_c, _ = cls.validate_data(cont)
        ok_d, _ = cls.validate_data(disc)
        assert ok_c and ok_d, f"{cls.__name__} should accept any dtype"


def test_validate_data_does_not_raise_or_construct():
    """validate_data is pure metadata; it must not construct the test or
    touch upstream dependencies."""
    np.random.seed(0)
    # Tests with optional deps we don't have installed: still safe to call
    ok, reason = HarteminkChiSq.validate_data(np.random.randn(50, 3))
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# Finding 2: strict kwargs allowlist
# ---------------------------------------------------------------------------


def test_unknown_kwarg_raises():
    np.random.seed(0)
    data = np.random.randint(0, 3, (50, 3)).astype(float)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ChiSq(data, methodname="chisq")  # typo: should be method_name (or none)


def test_typo_on_per_test_kwarg_raises():
    np.random.seed(0)
    data = np.random.randn(100, 3)
    with pytest.raises(TypeError, match="n_bin"):
        DiscChiSq(data, n_bin=4)  # typo on the per-test API kwarg


def test_accepted_kwarg_passes():
    np.random.seed(0)
    data = np.random.randn(100, 3)
    t = DiscChiSq(data, n_bins=3)
    p = t(0, 1, [2])
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_protocol_kwarg_accepted_on_all_19():
    """data_type is universally tolerated on every test for harness
    compatibility. Construction must not raise TypeError on any of the 19."""
    np.random.seed(0)
    data = np.random.randn(60, 3)
    data_type = np.array([[0, 0, 0]])
    for cls in ALL_19:
        try:
            cls(data, data_type=data_type)
        except TypeError as exc:  # pragma: no cover — failure path
            pytest.fail(f"{cls.__name__} rejected data_type: {exc}")
        except Exception:
            # Any non-TypeError (CITKDependencyError, computation error, etc.)
            # is acceptable — we only care that the kwargs allowlist accepts
            # the protocol kwarg.
            pass


def test_error_message_lists_consumed_and_protocol_kwargs():
    np.random.seed(0)
    data = np.random.randn(50, 3)
    with pytest.raises(TypeError) as excinfo:
        DiscChiSq(data, totally_bogus=1)
    msg = str(excinfo.value)
    assert "Accepted (consumed)" in msg
    assert "n_bins" in msg
    assert "Accepted (protocol-tolerated)" in msg
    assert "data_type" in msg
