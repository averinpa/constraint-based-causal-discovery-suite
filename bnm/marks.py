"""Endpoint marks for the int8 endpoint matrix.

`EndpointMark` is the canonical encoding for one end of an edge in a
graph stored as an `(n, n)` int8 matrix. Numeric values match cbcd's
`cbcd.graph.EndpointMark` so the int8 matrix is the only interop
currency between bnm and cbcd — neither package imports the other.
"""

from __future__ import annotations

from enum import IntEnum


class EndpointMark(IntEnum):
    """Mark on one end of an edge.

    `endpoints[i, j]` is the mark at node `j` of the edge between `i`
    and `j`. The pair `(endpoints[j, i], endpoints[i, j])` together
    specifies the edge type:

    | (mark at i, mark at j) | edge type        |
    |------------------------|------------------|
    | (TAIL, ARROW)          | i → j (directed) |
    | (ARROW, TAIL)          | i ← j (directed) |
    | (TAIL, TAIL)           | i — j (undirected) |
    | (ARROW, ARROW)         | i ↔ j (bidirected) |
    | (CIRCLE, *)            | PAG circle endpoint |
    | (NO_EDGE, NO_EDGE)     | no edge          |

    NO_EDGE on either end means no edge exists; both ends must be
    NO_EDGE or both non-NO_EDGE.
    """

    NO_EDGE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3
