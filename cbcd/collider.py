"""Collider orientation: ColliderDecisions, ColliderOrienter Protocol, SepsetOrienter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.graph.cpdag import PartialCPDAG
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PartialPAG
from cbcd.skeleton import Skeleton


def _canonicalize_triple(x: int, z: int, y: int) -> tuple[int, int, int]:
    a, b = (x, y) if x < y else (y, x)
    return (a, z, b)


def _unshielded_triples(adj: np.ndarray) -> list[tuple[int, int, int]]:
    n = adj.shape[0]
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for z in range(n):
        nbrs = [int(i) for i in np.where(adj[z])[0]]
        for i, x in enumerate(nbrs):
            for y in nbrs[i + 1 :]:
                if not adj[x, y]:
                    triple = _canonicalize_triple(x, z, y)
                    if triple not in seen:
                        seen.add(triple)
                        out.append(triple)
    return out


@dataclass(frozen=True)
class ColliderDecisions:
    """Pure-data classification of unshielded triples; no graph mutation here.

    Triples are canonicalized as (X, Z, Y) with X < Y; Z is the candidate
    collider node (middle of the unshielded path X — Z — Y).
    """

    colliders: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
    non_colliders: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
    ambiguous: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)

    def apply_to_cpdag(
        self,
        skeleton: Skeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialCPDAG:
        n = skeleton.n_vars
        endpoints = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if skeleton.adj[i, j]:
                    endpoints[i, j] = EndpointMark.TAIL
                    endpoints[j, i] = EndpointMark.TAIL

        for x, z, y in self.colliders:
            # Orient X → Z and Y → Z. If a previous decision wrote the opposite
            # endpoint, we keep the arrowhead at Z (matching causal-learn's
            # last-write semantics for vanilla PC; conflicts are rare and Meek
            # may reconcile downstream).
            if endpoints[x, z] != EndpointMark.NO_EDGE:
                endpoints[x, z] = EndpointMark.ARROW
                endpoints[z, x] = EndpointMark.TAIL
            if endpoints[y, z] != EndpointMark.NO_EDGE:
                endpoints[y, z] = EndpointMark.ARROW
                endpoints[z, y] = EndpointMark.TAIL

        return PartialCPDAG(
            n_vars=n,
            endpoints=endpoints,
            var_names=var_names,
            ambiguous_triples=self.ambiguous,
        )

    def apply_to_pag(
        self,
        skeleton: Skeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialPAG:
        """Lay collider arrows onto a PAG canvas.

        Initializes every skeleton edge as ``CIRCLE—CIRCLE``; for each collider
        triple ``(X, Z, Y)`` writes ``ARROW`` at Z on both arms (mark at X and
        Y stays CIRCLE). Last-write semantics on the Z-endpoint mark when
        triples overlap, mirroring ``apply_to_cpdag``.
        """
        n = skeleton.n_vars
        endpoints = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if skeleton.adj[i, j]:
                    endpoints[i, j] = EndpointMark.CIRCLE
                    endpoints[j, i] = EndpointMark.CIRCLE

        for x, z, y in self.colliders:
            if endpoints[x, z] != EndpointMark.NO_EDGE:
                endpoints[x, z] = EndpointMark.ARROW
            if endpoints[y, z] != EndpointMark.NO_EDGE:
                endpoints[y, z] = EndpointMark.ARROW

        sepsets = dict(skeleton.sepsets) if skeleton.sepsets else None
        return PartialPAG(
            n_vars=n,
            endpoints=endpoints,
            var_names=var_names,
            sepsets=sepsets,
        )


class ColliderOrienter(Protocol):
    requires_max_pvalues: bool

    def __call__(
        self,
        skeleton: Skeleton,
        ci: CITest,
        *,
        alpha: float,
        background: BackgroundKnowledge | None = None,
    ) -> ColliderDecisions: ...


class SepsetOrienter:
    """Vanilla PC R0: triple X–Z–Y is a collider iff Z is not in sepset({X, Y})."""

    requires_max_pvalues = False

    def __call__(
        self,
        skeleton: Skeleton,
        ci: CITest,
        *,
        alpha: float,
        background: BackgroundKnowledge | None = None,
    ) -> ColliderDecisions:
        del ci, alpha  # SepsetOrienter doesn't re-test.
        colliders: set[tuple[int, int, int]] = set()
        non_colliders: set[tuple[int, int, int]] = set()
        for x, z, y in _unshielded_triples(skeleton.adj):
            sep = skeleton.sepsets.get(frozenset({x, y}))
            if sep is None:
                # No recorded sepset → cannot classify.
                continue
            if z in sep:
                non_colliders.add((x, z, y))
            else:
                colliders.add((x, z, y))

        if background is not None:
            colliders = {
                (x, z, y)
                for (x, z, y) in colliders
                if not (
                    background.is_required_directed(z, x) or background.is_required_directed(z, y)
                )
            }

        return ColliderDecisions(
            colliders=frozenset(colliders),
            non_colliders=frozenset(non_colliders),
            ambiguous=frozenset(),
        )
