import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from causallearn.utils.cit import NO_SPECIFIED_PARAMETERS_MSG, register_ci_test

from .base import CITKTest


MCMIKNN_SRC_PATH = Path("/Users/pavelaverin/Projects/vendor/mCMIkNN/src")


def _import_from_repo(repo_path: Path, module_candidates: List[str], install_hint: str):
    if not repo_path.exists():
        raise ImportError(install_hint)

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    last_exc = None
    for module_name in module_candidates:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            last_exc = exc
    raise ImportError(f"Could not import any of {module_candidates} from {repo_path}") from last_exc


def _extract_pvalue(result) -> float:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return float(result[1])
    if isinstance(result, dict):
        for key in ("pval", "p_value", "p", "pvalue"):
            if key in result:
                return float(result[key])
    if isinstance(result, (float, np.floating)):
        return float(result)
    raise RuntimeError(f"Unexpected result type for p-value extraction: {type(result)}")


class MCMIknn(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.test_kwargs = kwargs.get("test_kwargs", {})
        self.check_cache_method_consistent("mcmiknn", NO_SPECIFIED_PARAMETERS_MSG)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        module = _import_from_repo(
            MCMIKNN_SRC_PATH,
            module_candidates=["mCMIkNN", "mcmiknn"],
            install_hint=(
                "mCMIkNN wrapper requires local source at "
                f"{MCMIKNN_SRC_PATH}. Clone/build that repository first."
            ),
        )

        # Try known class/function names from likely package variants.
        test_obj = None
        for name in ("MCMIknn", "mCMIkNN", "CMIknnMixed", "CMIknn"):
            if hasattr(module, name):
                candidate = getattr(module, name)
                if callable(candidate):
                    try:
                        test_obj = candidate(**self.test_kwargs)
                    except TypeError:
                        # some implementations may be plain functions
                        test_obj = candidate
                break

        if test_obj is None:
            raise RuntimeError(
                "mCMIkNN module imported, but no supported entry point found "
                "(expected one of: MCMIknn, mCMIkNN, CMIknnMixed, CMIknn)."
            )

        z = condition_set or []
        if hasattr(test_obj, "set_dataframe") and hasattr(test_obj, "run_test"):
            # Tigramite-like interface
            tp = importlib.import_module("tigramite.data_processing")
            df = tp.DataFrame(self.data)
            test_obj.set_dataframe(df)
            result = test_obj.run_test(X=[(int(X), 0)], Y=[(int(Y), 0)], Z=[(int(c), 0) for c in z], tau_max=0)
            return _extract_pvalue(result)

        if callable(test_obj):
            try:
                result = test_obj(self.data[:, X], self.data[:, Y], self.data[:, z] if z else None)
            except TypeError as exc:
                raise RuntimeError(
                    "mCMIkNN callable exists but signature is unsupported by this wrapper. "
                    "Pass a compatible adapter in test_kwargs or adjust wrapper."
                ) from exc
            return _extract_pvalue(result)

        raise RuntimeError("Unsupported mCMIkNN object type after import.")


register_ci_test("mcmiknn", MCMIknn)
