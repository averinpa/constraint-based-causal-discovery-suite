import importlib
from pathlib import Path

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


def test_rcot_missing_rpy2_has_clear_error():
    if importlib.util.find_spec("rpy2") is not None:
        pytest.skip("rpy2 is installed; missing-dependency path is not applicable")

    from citk.tests.r_based_tests import RCoT

    data_ind, _ = _continuous_data(seed=9, n=80)
    with pytest.raises(ImportError, match="rpy2"):
        RCoT(data_ind)(0, 1)


def test_cmiknn_missing_tigramite_has_clear_error():
    if importlib.util.find_spec("tigramite") is not None:
        pytest.skip("tigramite is installed; missing-dependency path is not applicable")

    from citk.tests.tigramite_based_tests import CMIknn

    data_ind, _ = _continuous_data(seed=14, n=80)
    with pytest.raises(ImportError, match="tigramite"):
        CMIknn(data_ind)(0, 1)


def test_regci_missing_tigramite_has_clear_error():
    if importlib.util.find_spec("tigramite") is not None:
        pytest.skip("tigramite is installed; missing-dependency path is not applicable")

    from citk.tests.tigramite_based_tests import RegressionCI

    data_ind, _ = _continuous_data(seed=15, n=80)
    with pytest.raises(ImportError, match="tigramite"):
        RegressionCI(data_ind)(0, 1)


def test_hartemink_missing_rpy2_has_clear_error():
    if importlib.util.find_spec("rpy2") is not None:
        pytest.skip("rpy2 is installed; missing-dependency path is not applicable")

    from citk.tests.r_based_tests import HarteminkChiSq

    data_ind, _ = _continuous_data(seed=16, n=80)
    with pytest.raises(ImportError, match="rpy2"):
        HarteminkChiSq(data_ind)(0, 1)


def test_mcmiknn_missing_local_repo_has_clear_error():
    repo_path = Path("/Users/pavelaverin/Projects/vendor/mCMIkNN/src")
    if repo_path.exists():
        pytest.skip("local mCMIkNN repo is present; missing-repo path is not applicable")

    data_ind, _ = _continuous_data(seed=17, n=80)
    with pytest.raises(ImportError, match="mCMIkNN wrapper requires local source"):
        MCMIknn(data_ind)(0, 1)


