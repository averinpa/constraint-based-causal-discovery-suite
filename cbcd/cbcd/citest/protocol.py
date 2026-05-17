"""CITest Protocol and result type."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CITestResult:
    """Outcome of a single conditional-independence call."""

    p_value: float
    statistic: float | None = None
    df: int | None = None
    n_effective: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class CITest(Protocol):
    """Conditional-independence test interface.

    A ``CITest`` answers ``X ⫫ Y | S`` queries on integer variable indices in
    ``[0, n_vars)``. ``__call__`` returns the p-value; ``details`` returns the
    full ``CITestResult``.

    Conformance is **structural, not nominal**. Any object with the three
    members below satisfies the Protocol — no inheritance, no import of
    ``cbcd`` required. Third-party CI test libraries (e.g. ``citests``) plug in
    by exposing classes with this shape:

    * ``n_vars: int`` attribute.
    * ``__call__(x: int, y: int, S: Sequence[int]) -> float``.
    * ``details(x: int, y: int, S: Sequence[int])`` returning any object
      with a ``.p_value: float`` attribute. (cbcd's algorithms only read
      ``.p_value`` from cached results; the richer ``CITestResult`` fields
      ``statistic`` / ``df`` / ``n_effective`` / ``extra`` are optional but
      recommended for diagnostics.)
    """

    n_vars: int

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float: ...

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult: ...
