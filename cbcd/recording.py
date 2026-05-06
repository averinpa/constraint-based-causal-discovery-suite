"""RunRecorder Protocol + NullRecorder.

Audit-trail interface from §J of the design. Only the no-op recorder ships in
this slice; ``InMemoryRecorder`` and ``FileRecorder`` are deferred. The
Protocol is defined now so phase APIs accept ``recorder=...`` from day one
without later signature churn.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunRecorder(Protocol):
    """Audit-trail sink. All hooks may be no-ops."""

    def record_ci(
        self,
        *,
        x: int,
        y: int,
        S: tuple[int, ...],
        p_value: float,
        depth: int,
        was_cache_hit: bool,
    ) -> None: ...

    def record_collider(
        self,
        *,
        triple: tuple[int, int, int],
        classification: str,
        orienter: str,
    ) -> None: ...

    def record_rule(
        self,
        *,
        rule_set: str,
        rule_name: str,
        affected_edge: tuple[int, int],
    ) -> None: ...


class NullRecorder:
    """Default recorder. Every method is a no-op; near-zero overhead."""

    def record_ci(
        self,
        *,
        x: int,
        y: int,
        S: tuple[int, ...],
        p_value: float,
        depth: int,
        was_cache_hit: bool,
    ) -> None:
        return

    def record_collider(
        self,
        *,
        triple: tuple[int, int, int],
        classification: str,
        orienter: str,
    ) -> None:
        return

    def record_rule(
        self,
        *,
        rule_set: str,
        rule_name: str,
        affected_edge: tuple[int, int],
    ) -> None:
        return


def _resolve_recorder(recorder: RunRecorder | None) -> RunRecorder:
    return recorder if recorder is not None else NullRecorder()
