"""Exception hierarchy for citk.

All citk-defined exceptions inherit from :class:`CITKError`. The leaf classes
also inherit from a relevant stdlib exception so existing user code that
catches :class:`ImportError`, :class:`RuntimeError`, or :class:`ValueError`
continues to work unchanged.

Catch :class:`CITKError` to handle any citk failure uniformly; catch a
subclass to discriminate between the failure modes.
"""


class CITKError(Exception):
    """Base class for all citk-specific exceptions."""


class CITKDependencyError(CITKError, ImportError):
    """Raised when an optional dependency required by a test is missing
    or cannot be loaded (e.g. ``rpy2``, an R package, ``tigramite``).
    """


class CITKComputationError(CITKError, RuntimeError):
    """Raised when a test fails during computation due to a numerical
    issue or an unexpected error escaping from a wrapped upstream library.
    """


class CITKDataError(CITKError, ValueError):
    """Raised when the input data is invalid for the requested test
    (wrong dtype, degenerate, NaN where not allowed, etc.).
    """
