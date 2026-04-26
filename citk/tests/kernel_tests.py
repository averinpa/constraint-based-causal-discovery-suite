"""Kernel CI tests (survey family)."""

from causallearn.utils.cit import KCI as KCI_test, register_ci_test

from .base import CITKTest, inner_test_kwargs
from .r_based_tests import RCIT, RCoT  # noqa: F401  (re-exported)


class KCI(CITKTest):
    """
    Wrapper for the Kernel Conditional Independence (KCI) test from the causal-learn library.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    **kwargs : dict
        Additional keywords for the KCI test. See causal-learn documentation.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('kci', "NO SPECIFIED PARAMETERS")
        self.kci_instance = KCI_test(data, **inner_test_kwargs(kwargs))

    def _compute(self, X, Y, condition_set=None, **kwargs):
        return float(self.kci_instance(X, Y, condition_set))


register_ci_test("kci", KCI)


__all__ = ["KCI", "RCIT", "RCoT"]
