"""Smoke tests — package imports and exposes its public surface."""

import cbcd


def test_version_is_set() -> None:
    assert isinstance(cbcd.__version__, str)
    assert cbcd.__version__


def test_exceptions_exposed() -> None:
    assert issubclass(cbcd.CBCDInputError, cbcd.CBCDError)
    assert issubclass(cbcd.CBCDDataError, cbcd.CBCDError)
