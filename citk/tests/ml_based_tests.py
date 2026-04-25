from .base import CITKTest
from causallearn.utils.cit import register_ci_test, KCI as KCI_test


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
        self.kci_instance = KCI_test(data, **kwargs)

    def _compute(self, X, Y, condition_set=None, **kwargs):
        return float(self.kci_instance(X, Y, condition_set))


register_ci_test("kci", KCI)
