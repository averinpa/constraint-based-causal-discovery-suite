"""Base class and cache helpers for citk conditional independence tests."""
import codecs
import hashlib
import json
import os
import time
import warnings
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np
from causallearn.utils.cit import CIT_Base, NO_SPECIFIED_PARAMETERS_MSG

from citk.exceptions import CITKComputationError, CITKError

CACHE_FORMAT_VERSION = "1.0"


def _canonicalize_for_hash(value: Any) -> Any:
    """Convert a parameter value into a JSON-serialisable form that hashes
    identically across runs (numpy arrays → typed dict, mappings → sorted dict)."""
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "data": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(k): _canonicalize_for_hash(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def inner_test_kwargs(kwargs: Mapping[str, Any]) -> dict:
    """Return ``kwargs`` filtered for forwarding to a wrapped upstream test
    instance. The citk outer wrapper owns the cache, so ``cache_path`` must
    not leak through to an inner instance whose hashing scheme may differ.
    """
    return {k: v for k, v in kwargs.items() if k != "cache_path"}


def hash_parameters(params: Optional[Mapping[str, Any]]) -> str:
    """Return a stable sha256 hex digest of `params`, or the
    ``NO_SPECIFIED_PARAMETERS_MSG`` sentinel if `params` is empty / None.

    Used by every test class to fingerprint its constructor kwargs into the
    cache's ``parameters_hash`` field. The hash is key-order independent and
    handles numpy arrays.
    """
    if not params:
        return NO_SPECIFIED_PARAMETERS_MSG
    canonical = _canonicalize_for_hash(dict(params))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CITKTest(CIT_Base):
    """Abstract base class for all conditional independence tests in citk.

    Standardises the interface to be compatible with causal-learn and
    implements a versioned, content-addressed file-based cache.
    """
    supported_dtypes: set = set()

    def __init__(
        self,
        data: np.ndarray,
        cache_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the test and (optionally) load a JSON p-value cache.

        Parameters
        ----------
        data
            Sample matrix in shape ``(n, p)``.
        cache_path
            Optional path to a JSON cache file used to memoise p-values across
            calls. The cache is keyed by ``(data_hash, method_name,
            parameters_hash)`` and stamped with ``format_version`` so v0.1.0
            caches can be detected and invalidated by future releases.
        """
        # Call parent with cache_path=None so its MD5-based load path is bypassed;
        # we handle hashing and load below with sha256 + format_version validation.
        super().__init__(data, cache_path=None, **kwargs)
        self.data_hash = hashlib.sha256(
            np.ascontiguousarray(data).tobytes()
        ).hexdigest()
        self.cache_path = cache_path
        self.pvalue_cache = {
            "format_version": CACHE_FORMAT_VERSION,
            "data_hash": self.data_hash,
        }
        if cache_path is not None:
            assert cache_path.endswith(".json"), "Cache must be stored as .json file."
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                try:
                    with codecs.open(cache_path, "r") as fin:
                        loaded = json.load(fin)
                except json.JSONDecodeError:
                    loaded = {}
                if (
                    loaded.get("format_version") == CACHE_FORMAT_VERSION
                    and loaded.get("data_hash") == self.data_hash
                ):
                    self.pvalue_cache = loaded
                else:
                    # Stale, pre-versioned, or unreadable cache. Regenerate
                    # rather than raise — but warn so debugging is possible.
                    warnings.warn(
                        f"Cache {cache_path} format/hash mismatch; regenerating.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                parent_dir = os.path.dirname(cache_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

    def _normalize_condition_set(
        self, condition_set: Optional[Iterable[int]]
    ) -> List[int]:
        if condition_set is None:
            return []
        return list(condition_set)

    def save_cache(self) -> None:
        """Explicitly save the p-value cache to ``cache_path``.

        More reliable than relying on the garbage collector via ``__del__`` or
        on the parent's 30-second auto-save interval.
        """
        if hasattr(self, "cache_path") and self.cache_path is not None:
            try:
                if hasattr(self, "pvalue_cache"):
                    with codecs.open(self.cache_path, "w") as fout:
                        fout.write(json.dumps(self.pvalue_cache, indent=2))
            except Exception as e:
                print(f"Error saving cache for {self.__class__.__name__}: {e}")

    def __del__(self):
        """Save the cache on garbage collection as a last-chance flush."""
        self.save_cache()

    def __call__(
        self,
        X: int,
        Y: int,
        condition_set: Optional[Iterable[int]] = None,
        **kwargs: Any,
    ) -> float:
        condition_set = self._normalize_condition_set(condition_set)
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        try:
            p_value = float(self._compute(X, Y, condition_set, **kwargs))
        except CITKError:
            # Already a typed citk exception (e.g. CITKDependencyError); re-raise.
            raise
        except Exception as exc:
            raise CITKComputationError(
                f"{type(self).__name__} failed for X={X}, Y={Y}, "
                f"S={condition_set}: {exc}"
            ) from exc
        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError("Subclasses must implement _compute.")
