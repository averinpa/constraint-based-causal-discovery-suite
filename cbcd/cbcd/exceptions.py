"""Exception types raised by cbcd."""


class CBCDError(Exception):
    """Base class for all cbcd errors."""


class CBCDInputError(CBCDError, ValueError):
    """Caller passed an invalid argument (wrong shape, missing key, bad option)."""


class CBCDDataError(CBCDError, ValueError):
    """Input data is structurally unsuitable (e.g. wrong dtype, NaNs where not supported)."""
