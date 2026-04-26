import importlib

import numpy as np
import pytest

from citk.tests.adapter_tests import DiscChiSq, DiscGSq, DummyFisherZ
from citk.tests.external_repo_tests import MCMIknn
from citk.tests.contingency_table_tests import GSq, ChiSq
from citk.tests.partial_correlation_tests import FisherZ, Spearman


def _continuous_data(seed: int = 0, n: int = 250):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y_ind = rng.normal(size=n)
    y_dep = x + 0.2 * rng.normal(size=n)
    data_ind = np.column_stack([x, y_ind])
    data_dep = np.column_stack([x, y_dep])
    return data_ind, data_dep


def _discrete_data(seed: int = 1, n: int = 400):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, size=n)
    y_ind = rng.integers(0, 3, size=n)
    y_dep = x.copy()
    flip = rng.random(n) < 0.1
    y_dep[flip] = rng.integers(0, 3, size=flip.sum())
    data_ind = np.column_stack([x, y_ind])
    data_dep = np.column_stack([x, y_dep])
    return data_ind, data_dep


@pytest.mark.parametrize("test_cls", [FisherZ, Spearman])
def test_continuous_smoke(test_cls):
    data_ind, data_dep = _continuous_data()
    p_ind = test_cls(data_ind)(0, 1)
    p_dep = test_cls(data_dep)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


@pytest.mark.parametrize("test_cls", [GSq, ChiSq])
def test_discrete_smoke(test_cls):
    data_ind, data_dep = _discrete_data()
    p_ind = test_cls(data_ind)(0, 1)
    p_dep = test_cls(data_dep)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05

def test_kci_smoke():
    kernel_module = importlib.import_module("citk.tests.kernel_tests")
    data_ind, data_dep = _continuous_data(seed=4, n=150)
    p_ind = kernel_module.KCI(data_ind)(0, 1)
    p_dep = kernel_module.KCI(data_dep)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


@pytest.mark.parametrize("test_cls", [DiscChiSq, DiscGSq])
def test_discretize_adapters_smoke(test_cls):
    data_ind, data_dep = _continuous_data(seed=10, n=300)
    p_ind = test_cls(data_ind, n_bins=4)(0, 1)
    p_dep = test_cls(data_dep, n_bins=4)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_dummy_fisherz_smoke():
    data_ind, data_dep = _discrete_data(seed=11, n=400)
    p_ind = DummyFisherZ(data_ind)(0, 1)
    p_dep = DummyFisherZ(data_dep)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_rcot_smoke():
    pytest.importorskip("rpy2")
    from citk.tests.kernel_tests import RCoT

    data_ind, data_dep = _continuous_data(seed=9, n=300)
    p_ind = RCoT(data_ind)(0, 1)
    p_dep = RCoT(data_dep)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_cmiknn_smoke():
    pytest.importorskip("tigramite")
    from citk.tests.nearest_neighbor_tests import CMIknn

    data_ind, data_dep = _continuous_data(seed=14, n=120)
    cmiknn_kwargs = {"test_kwargs": {"sig_samples": 49}}
    p_ind = CMIknn(data_ind, **cmiknn_kwargs)(0, 1)
    p_dep = CMIknn(data_dep, **cmiknn_kwargs)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_regci_smoke():
    pytest.importorskip("tigramite")
    from citk.tests.regression_tests import RegressionCI

    data_ind, data_dep = _continuous_data(seed=15, n=400)
    data_type = np.zeros(data_ind.shape, dtype="int32")
    p_ind = RegressionCI(data_ind, data_type=data_type)(0, 1)
    p_dep = RegressionCI(data_dep, data_type=data_type)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_hartemink_chisq_smoke():
    pytest.importorskip("rpy2")
    from citk.tests.adapter_tests import HarteminkChiSq

    data_ind, data_dep = _continuous_data(seed=16, n=300)
    p_ind = HarteminkChiSq(data_ind, breaks=3, ibreaks=6)(0, 1)
    p_dep = HarteminkChiSq(data_dep, breaks=3, ibreaks=6)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


def test_mcmiknn_smoke():
    data_ind, data_dep = _continuous_data(seed=17, n=120)
    mcmiknn_kwargs = {"test_kwargs": {"Mperm": 49}}
    p_ind = MCMIknn(data_ind, **mcmiknn_kwargs)(0, 1)
    p_dep = MCMIknn(data_dep, **mcmiknn_kwargs)(0, 1)
    assert p_ind > 0.05
    assert p_dep < 0.05


