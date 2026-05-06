"""Partial Ancestral Graph (PAG) and Maximal Ancestral Graph (MAG) types."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError
from cbcd.graph.base import _GraphBase
from cbcd.graph.marks import EndpointMark


def _validate_pag_marks(endpoints: NDArray[np.int8]) -> None:
    permitted = {
        int(EndpointMark.NO_EDGE),
        int(EndpointMark.TAIL),
        int(EndpointMark.ARROW),
        int(EndpointMark.CIRCLE),
    }
    marks_seen = {int(m) for m in np.unique(endpoints)}
    unsupported = marks_seen - permitted
    if unsupported:
        raise CBCDInputError(
            f"PAG only supports NO_EDGE/TAIL/ARROW/CIRCLE marks; got {sorted(unsupported)}"
        )


class PartialPAG(_GraphBase):
    """Working canvas for the FCI family between collider orientation and rule closure.

    Marks ∈ {NO_EDGE, TAIL, ARROW, CIRCLE}. Carries optional ``sepsets`` —
    witnesses recorded by a ``PAGSkeletonRefinement`` step so downstream rules
    can look them up.
    """

    sepsets: dict[frozenset[int], tuple[int, ...]] | None

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
        sepsets: dict[frozenset[int], tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)
        _validate_pag_marks(self.endpoints)
        self.sepsets = sepsets


class PAG(_GraphBase):
    """Partial Ancestral Graph — the closed result of the FCI family.

    Marks ∈ {NO_EDGE, TAIL, ARROW, CIRCLE}. CIRCLE indicates an endpoint that
    is not identified by the constraints (could be either TAIL or ARROW in a
    DAG of the equivalence class).
    """

    sepsets: dict[frozenset[int], tuple[int, ...]] | None

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
        sepsets: dict[frozenset[int], tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)
        _validate_pag_marks(self.endpoints)
        self.sepsets = sepsets

    def definite_edges(self) -> tuple[tuple[int, int, EndpointMark, EndpointMark], ...]:
        """Edges with no CIRCLE on either endpoint, as ``(i, j, mark_at_i, mark_at_j)``.

        Each edge is reported once with ``i < j``.
        """
        out: list[tuple[int, int, EndpointMark, EndpointMark]] = []
        n = self.n_vars
        for i in range(n):
            for j in range(i + 1, n):
                mij = int(self.endpoints[i, j])
                mji = int(self.endpoints[j, i])
                if mij == EndpointMark.NO_EDGE:
                    continue
                if mij == EndpointMark.CIRCLE or mji == EndpointMark.CIRCLE:
                    continue
                out.append((i, j, EndpointMark(mji), EndpointMark(mij)))
        return tuple(out)

    def possibly_directed(self, i: int, j: int) -> bool:
        """Whether ``i → j`` is consistent with the recorded marks.

        True iff there is an edge between ``i`` and ``j``, the mark at ``j``
        is ARROW or CIRCLE, and the mark at ``i`` is TAIL or CIRCLE.
        """
        mij = int(self.endpoints[i, j])
        mji = int(self.endpoints[j, i])
        if mij == EndpointMark.NO_EDGE:
            return False
        return mij in (int(EndpointMark.ARROW), int(EndpointMark.CIRCLE)) and mji in (
            int(EndpointMark.TAIL),
            int(EndpointMark.CIRCLE),
        )


class MAG(_GraphBase):
    """Maximal Ancestral Graph (minimal stub for §D completeness).

    Permits TAIL/ARROW marks only; every edge must be directed ``→``, ``←``,
    or bidirected ``↔``. Ancestrality and maximality (no inducing paths) are
    *not* validated here — those checks plus ``is_ancestor_of`` /
    ``m_separated`` / ``to_pag`` arrive in the slice that adds latent
    projection. For now this class exists so PAG-aware code can refer to it
    by type without falling through to a runtime ``hasattr``.
    """

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)
        permitted = {
            int(EndpointMark.NO_EDGE),
            int(EndpointMark.TAIL),
            int(EndpointMark.ARROW),
        }
        marks_seen = {int(m) for m in np.unique(self.endpoints)}
        unsupported = marks_seen - permitted
        if unsupported:
            raise CBCDInputError(
                f"MAG only supports NO_EDGE/TAIL/ARROW marks; got {sorted(unsupported)}"
            )
        n = n_vars
        for i in range(n):
            for j in range(i + 1, n):
                mij = int(self.endpoints[i, j])
                mji = int(self.endpoints[j, i])
                if mij == EndpointMark.NO_EDGE:
                    continue
                if mij == EndpointMark.ARROW and mji == EndpointMark.TAIL:
                    continue
                if mij == EndpointMark.TAIL and mji == EndpointMark.ARROW:
                    continue
                if mij == EndpointMark.ARROW and mji == EndpointMark.ARROW:
                    continue
                raise CBCDInputError(
                    f"MAG edge ({i}, {j}) must be directed or bidirected; got marks=({mji}, {mij})"
                )

    def is_ancestor_of(self, i: int, j: int) -> bool:
        raise NotImplementedError("MAG.is_ancestor_of is deferred to the latent-projection slice")

    def m_separated(self, x: int, y: int, S: tuple[int, ...]) -> bool:
        raise NotImplementedError("MAG.m_separated is deferred to the latent-projection slice")

    def to_pag(self) -> PAG:
        raise NotImplementedError("MAG.to_pag is deferred to the latent-projection slice")
