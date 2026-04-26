import importlib
from typing import List, Optional

import numpy as np
from causallearn.utils.cit import register_ci_test

from .base import CITKTest, hash_parameters


def _load_tigramite():
    try:
        tp = importlib.import_module("tigramite.data_processing")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Tigramite-based CI tests require optional dependency 'tigramite'. "
            "Install with: pip install tigramite."
        ) from exc
    return tp


def _load_tigramite_test_class(candidates: List[str]):
    last_exc = None
    for path in candidates:
        module_name, class_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            last_exc = exc
    raise ImportError(f"Could not import tigramite test class from {candidates}") from last_exc


def _extract_tigramite_pvalue(result) -> float:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return float(result[1])
        raise RuntimeError("Unexpected tuple result from tigramite run_test.")
    if isinstance(result, dict):
        for key in ("pval", "p_value", "p"):
            if key in result:
                return float(result[key])
    if isinstance(result, (float, np.floating)):
        return float(result)
    raise RuntimeError(f"Unexpected tigramite result type: {type(result)}")


class _TigramiteBase(CITKTest):
    supported_dtypes = {"continuous", "discrete"}
    method_name = ""
    class_candidates: List[str] = []

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.test_kwargs = kwargs.get("test_kwargs", {})
        self.data_type = kwargs.get("data_type", None)
        self.check_cache_method_consistent(
            self.method_name,
            hash_parameters({"test_kwargs": self.test_kwargs, "data_type": self.data_type}),
        )

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        tp = _load_tigramite()
        test_cls = _load_tigramite_test_class(self.class_candidates)

        dataframe = tp.DataFrame(self.data, data_type=self.data_type)
        test = test_cls(**self.test_kwargs)
        test.set_dataframe(dataframe)
        z = [(int(c), 0) for c in condition_set]
        result = test.run_test(X=[(int(X), 0)], Y=[(int(Y), 0)], Z=z, tau_max=0)
        return _extract_tigramite_pvalue(result)


class CMIknn(_TigramiteBase):
    method_name = "cmiknn"
    class_candidates = ["tigramite.independence_tests.cmiknn.CMIknn"]


class CMIknnMixed(_TigramiteBase):
    method_name = "cmiknn_mixed"
    class_candidates = [
        "tigramite.independence_tests.cmiknn.CMIknnMixed",
        "tigramite.independence_tests.cmiknn_mixed.CMIknnMixed",
    ]


class RegressionCI(_TigramiteBase):
    method_name = "regci"
    class_candidates = ["tigramite.independence_tests.regressionCI.RegressionCI"]


register_ci_test("cmiknn", CMIknn)
register_ci_test("cmiknn_mixed", CMIknnMixed)
register_ci_test("regci", RegressionCI)
