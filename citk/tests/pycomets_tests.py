"""CI tests based on pycomets: GCM, wGCM, PCM (all using RF regression by default)."""

import contextlib
import io
from typing import Any, List, Optional

import numpy as np
from causallearn.utils.cit import register_ci_test

from citk.exceptions import CITKDependencyError
from .base import CITKTest


def _load_pycomets_gcm():
    try:
        from pycomets.gcm import GCM as PyGCMImpl, WGCM as PyWGCMImpl
        from pycomets.regression import RF
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "pycomets-based CI tests (GCM, WGCM, PCM) require optional "
            "dependency 'pycomets'. Install with: pip install 'citk[ml]' "
            "(or uv sync --extra ml)."
        ) from exc
    return PyGCMImpl, PyWGCMImpl, RF


def _load_pycomets_pcm():
    try:
        from pycomets.pcm import PCM as PyPCMImpl
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "pycomets-based CI tests (GCM, WGCM, PCM) require optional "
            "dependency 'pycomets'. Install with: pip install 'citk[ml]' "
            "(or uv sync --extra ml)."
        ) from exc
    return PyPCMImpl


class GCM(CITKTest):
    """GCM test via pycomets (Shah & Peters, 2020).

    Uses RF regression by default (pycomets default).
    In-sample residuals, no cross-fitting.
    """
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("gcm", "pycomets_GCM_RF")

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any) -> float:
        PyGCMImpl, _, RF = _load_pycomets_gcm()

        x = self.data[:, X].astype(float)
        y = self.data[:, Y].astype(float)

        gcm = PyGCMImpl()
        if condition_set:
            z = self.data[:, condition_set].astype(float)
        else:
            z = np.zeros((len(self.data), 1))

        with contextlib.redirect_stdout(io.StringIO()):
            gcm.test(y, x, z, reg_yz=RF(), reg_xz=RF(), show_summary=False)

        return float(gcm.pval)


class WGCM(CITKTest):
    """Weighted GCM test via pycomets (Scheidegger et al., 2022).

    Implements WGCM.est: sample-split (50/50) weight estimation. Nuisance
    regressions ``E[Y|Z]`` and ``E[X|Z]`` use RF; the weight regression
    ``E[(rX·rY)|Z]`` uses pycomets's KRR default. The cache fingerprint
    ``"pycomets_WGCM_RF"`` reflects the two RF nuisance choices but not
    the KRR weight choice — the v0.1.0 contract freezes both.

    Empty conditioning set falls back to ``GCM`` (the unweighted reduction
    when there is no Z to localise weights on), so the test is safe to use
    inside PC at depth 0.

    Limitations (v0.1.0)
    --------------------
    - Results are not bit-reproducible across processes. pycomets's
      ``WGCM.test`` uses ``rng=np.random.default_rng()`` as a mutable
      default arg, and its RF wrapper does not surface ``random_state``.
      Type-I and power are statistically valid; only the per-call p-value
      varies between fresh Python processes.
    """
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("wgcm", "pycomets_WGCM_RF")

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any) -> float:
        PyGCMImpl, PyWGCMImpl, RF = _load_pycomets_gcm()

        x = self.data[:, X].astype(float)
        y = self.data[:, Y].astype(float)

        if condition_set:
            z = self.data[:, condition_set].astype(float)
            wgcm = PyWGCMImpl()
            with contextlib.redirect_stdout(io.StringIO()):
                wgcm.test(y, x, z, reg_yz=RF(), reg_xz=RF(), show_summary=False)
            return float(wgcm.pval)

        # |S|=0: WGCM's sign-of-weight collapses on constant Z → fall back to GCM.
        z = np.zeros((len(self.data), 1))
        gcm = PyGCMImpl()
        with contextlib.redirect_stdout(io.StringIO()):
            gcm.test(y, x, z, reg_yz=RF(), reg_xz=RF(), show_summary=False)
        return float(gcm.pval)


class PCM(CITKTest):
    """Projected Covariance Measure test via pycomets (Lundborg et al., 2022).

    Uses RF regression with sample splitting (pycomets default).
    """
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("pcm", "pycomets_PCM_RF")

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any) -> float:
        PyPCMImpl = _load_pycomets_pcm()

        x = self.data[:, X].astype(float)
        y = self.data[:, Y].astype(float)

        pcm = PyPCMImpl()
        if condition_set:
            z = self.data[:, condition_set].astype(float)
        else:
            z = np.zeros((len(self.data), 1))

        with contextlib.redirect_stdout(io.StringIO()):
            pcm.test(y, x, z, show_summary=False)

        return float(pcm.pval)


register_ci_test("gcm", GCM)
register_ci_test("wgcm", WGCM)
register_ci_test("pcm", PCM)
