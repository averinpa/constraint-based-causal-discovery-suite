"""Edge orientation rules. CPDAGRules Protocol + MeekRules R1–R4."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from cbcd.background import BackgroundKnowledge
from cbcd.exceptions import CBCDInputError
from cbcd.graph.cpdag import CPDAG, PartialCPDAG
from cbcd.graph.marks import EndpointMark


class CPDAGRules(Protocol):
    def __call__(
        self,
        graph: PartialCPDAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
    ) -> CPDAG: ...


class MeekRules:
    """Meek (1995) R1–R4 applied to fixpoint on a PartialCPDAG.

    No ancestor queries are issued — the rules are local pattern matches over
    triples / quadruples (audit pitfall #6).

    Background knowledge constraints:

    * ``forbidden_directed`` (u, v): if a rule would orient u → v, suppress it.
    * ``required_directed`` (u, v): pre-orient u → v before running rules.
    """

    _ALL_RULES: frozenset[Literal["R1", "R2", "R3", "R4"]] = frozenset({"R1", "R2", "R3", "R4"})

    def __init__(
        self,
        *,
        rules: frozenset[Literal["R1", "R2", "R3", "R4"]] | None = None,
    ) -> None:
        self.rules = MeekRules._ALL_RULES if rules is None else frozenset(rules)
        unsupported = self.rules - MeekRules._ALL_RULES
        if unsupported:
            raise CBCDInputError(f"unsupported Meek rules: {sorted(unsupported)}")

    def __call__(
        self,
        graph: PartialCPDAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
    ) -> CPDAG:
        endpoints = graph.endpoints.copy()
        n = graph.n_vars

        if background is not None:
            for u, v in background.required_directed:
                if u >= n or v >= n:
                    continue
                if endpoints[u, v] != EndpointMark.NO_EDGE:
                    _try_orient(endpoints, u, v, background)

        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            changed = False
            if "R1" in self.rules and _apply_r1(endpoints, n, background):
                changed = True
            if "R2" in self.rules and _apply_r2(endpoints, n, background):
                changed = True
            if "R3" in self.rules and _apply_r3(endpoints, n, background):
                changed = True
            if "R4" in self.rules and _apply_r4(endpoints, n, background):
                changed = True
            if not changed:
                break
            iteration += 1

        return CPDAG(
            n_vars=n,
            endpoints=endpoints,
            var_names=graph.var_names,
            ambiguous_triples=graph.ambiguous_triples,
        )


def _is_directed(endpoints: np.ndarray, u: int, v: int) -> bool:
    return bool(
        endpoints[u, v] == EndpointMark.ARROW and endpoints[v, u] == EndpointMark.TAIL
    )


def _is_undirected(endpoints: np.ndarray, u: int, v: int) -> bool:
    return bool(
        endpoints[u, v] == EndpointMark.TAIL and endpoints[v, u] == EndpointMark.TAIL
    )


def _adjacent(endpoints: np.ndarray, u: int, v: int) -> bool:
    return bool(endpoints[u, v] != EndpointMark.NO_EDGE)


def _try_orient(
    endpoints: np.ndarray,
    u: int,
    v: int,
    background: BackgroundKnowledge | None,
) -> bool:
    """Orient u → v if currently undirected and not blocked by background. Return True if changed."""
    if not _is_undirected(endpoints, u, v):
        return False
    if background is not None and background.is_forbidden_directed(u, v):
        return False
    endpoints[u, v] = EndpointMark.ARROW
    endpoints[v, u] = EndpointMark.TAIL
    return True


def _apply_r1(endpoints: np.ndarray, n: int, bk: BackgroundKnowledge | None) -> bool:
    """R1: a → b — c, a not adjacent c ⟹ b → c."""
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b or not _is_directed(endpoints, a, b):
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                if not _is_undirected(endpoints, b, c):
                    continue
                if _adjacent(endpoints, a, c):
                    continue
                if _try_orient(endpoints, b, c, bk):
                    changed = True
    return changed


def _apply_r2(endpoints: np.ndarray, n: int, bk: BackgroundKnowledge | None) -> bool:
    """R2: a → b → c, a — c ⟹ a → c."""
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b or not _is_directed(endpoints, a, b):
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                if not _is_directed(endpoints, b, c):
                    continue
                if not _is_undirected(endpoints, a, c):
                    continue
                if _try_orient(endpoints, a, c, bk):
                    changed = True
    return changed


def _apply_r3(endpoints: np.ndarray, n: int, bk: BackgroundKnowledge | None) -> bool:
    """R3: a — b, a — c, a — d, c → b, d → b, c not adjacent d ⟹ a → b."""
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b or not _is_undirected(endpoints, a, b):
                continue
            # Find pairs (c, d) of undirected neighbours of a, both pointing to b.
            nbrs_a = [
                k
                for k in range(n)
                if k != a and k != b and _is_undirected(endpoints, a, k)
            ]
            for i, c in enumerate(nbrs_a):
                if not _is_directed(endpoints, c, b):
                    continue
                for d in nbrs_a[i + 1 :]:
                    if not _is_directed(endpoints, d, b):
                        continue
                    if _adjacent(endpoints, c, d):
                        continue
                    if _try_orient(endpoints, a, b, bk):
                        changed = True
                        break
                else:
                    continue
                break
    return changed


def _apply_r4(endpoints: np.ndarray, n: int, bk: BackgroundKnowledge | None) -> bool:
    """R4: a — b, a — c, a — d, c → d, d → b, b not adjacent c ⟹ a → b.

    Standard formulation; only fires with background knowledge typically.
    """
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b or not _is_undirected(endpoints, a, b):
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                if not _is_undirected(endpoints, a, c):
                    continue
                if _adjacent(endpoints, b, c):
                    continue
                for d in range(n):
                    if d in (a, b, c):
                        continue
                    if not _is_undirected(endpoints, a, d):
                        continue
                    if not _is_directed(endpoints, c, d):
                        continue
                    if not _is_directed(endpoints, d, b):
                        continue
                    if _try_orient(endpoints, a, b, bk):
                        changed = True
                        break
                else:
                    continue
                break
    return changed
