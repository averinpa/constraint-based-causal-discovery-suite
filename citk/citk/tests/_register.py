"""Optional registration helper for ``causallearn.search.ConstraintBased.PC``.

citk's CI test classes are written against the standalone :class:`CITKTest`
base — they do not inherit from ``causallearn.utils.cit.CIT_Base``. But
causal-learn's :func:`register_ci_test` rejects any class that isn't a
``CIT_Base`` subclass. To bridge the gap without making causal-learn a
hard dependency, this helper creates a thin multiple-inheritance adapter
class on the fly when causal-learn is available, and silently no-ops
otherwise.
"""
from __future__ import annotations

from typing import Type

from citk.tests.base import CITKTest


def maybe_register(name: str, citk_cls: Type[CITKTest]) -> None:
    """Register ``citk_cls`` with causal-learn's PC dispatch under ``name``.

    No-op if causal-learn is not installed. The registered class is a
    dynamically-created adapter that satisfies
    ``issubclass(adapter, CIT_Base)`` — it simply inherits from both
    ``citk_cls`` and ``CIT_Base``, with ``citk_cls`` first in the MRO so
    citk's ``__init__`` and ``__call__`` win.
    """
    try:
        from causallearn.utils.cit import CIT_Base, register_ci_test
    except ImportError:
        return
    adapter_name = f"_CausalLearn_{citk_cls.__name__}"
    adapter = type(adapter_name, (citk_cls, CIT_Base), {})
    register_ci_test(name, adapter)
