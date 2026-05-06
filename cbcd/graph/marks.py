"""Endpoint marks and Edge accessor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class EndpointMark(IntEnum):
    """Mark on one end of an edge.

    Stored in the (n, n) endpoint matrix of a graph: ``endpoints[i, j]`` is the
    mark at node ``j`` of the edge between ``i`` and ``j``. The pair
    ``(endpoints[j, i], endpoints[i, j])`` together specify the edge type.

    NO_EDGE on either end means no edge exists. Both ends must be ``NO_EDGE``
    or both non-``NO_EDGE``.
    """

    NO_EDGE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3


@dataclass(frozen=True, slots=True)
class Edge:
    """An edge between nodes ``i`` and ``j`` with a mark on each end."""

    i: int
    j: int
    mark_at_i: EndpointMark
    mark_at_j: EndpointMark

    @property
    def is_directed(self) -> bool:
        return (self.mark_at_i is EndpointMark.TAIL and self.mark_at_j is EndpointMark.ARROW) or (
            self.mark_at_i is EndpointMark.ARROW and self.mark_at_j is EndpointMark.TAIL
        )

    @property
    def is_undirected(self) -> bool:
        return self.mark_at_i is EndpointMark.TAIL and self.mark_at_j is EndpointMark.TAIL

    @property
    def is_bidirected(self) -> bool:
        return self.mark_at_i is EndpointMark.ARROW and self.mark_at_j is EndpointMark.ARROW

    @property
    def is_circle_circle(self) -> bool:
        return self.mark_at_i is EndpointMark.CIRCLE and self.mark_at_j is EndpointMark.CIRCLE
