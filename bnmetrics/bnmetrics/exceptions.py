"""bnmetrics exception hierarchy."""

from __future__ import annotations


class BNMError(Exception):
    """Base class for all bnmetrics errors."""


class BNMInputError(BNMError, ValueError):
    """Caller-side bug: invalid argument shape, type, or combination."""


class BNMDataError(BNMError, ValueError):
    """Structurally valid input that violates a metric's preconditions."""
