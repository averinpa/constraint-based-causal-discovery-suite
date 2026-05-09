"""Edge orientation rules.

CPDAGRules Protocol + MeekRules R1–R4 for the PC family.
PAGRules Protocol + FCIRules R1–R10 for the FCI family.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from cbcd.background import BackgroundKnowledge
from cbcd.exceptions import CBCDInputError
from cbcd.graph.cpdag import CPDAG, PartialCPDAG
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG, PartialPAG
from cbcd.graph.queries import (
    find_discriminating_path,
    find_uncovered_circle_path,
    find_uncovered_pd_path,
)


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
    return bool(endpoints[u, v] == EndpointMark.ARROW and endpoints[v, u] == EndpointMark.TAIL)


def _is_undirected(endpoints: np.ndarray, u: int, v: int) -> bool:
    return bool(endpoints[u, v] == EndpointMark.TAIL and endpoints[v, u] == EndpointMark.TAIL)


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
            nbrs_a = [k for k in range(n) if k != a and k != b and _is_undirected(endpoints, a, k)]
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


# =============================================================================
# PAG rules — FCIRules with Zhang R1–R10
# =============================================================================


class PAGRules(Protocol):
    def __call__(
        self,
        graph: PartialPAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
    ) -> PAG: ...


# Zhang's R1–R10. Names live in this set; FCIRules.__init__ rejects unknowns.
_FCI_ALL_RULES: frozenset[str] = frozenset({f"R{i}" for i in range(1, 11)})


class FCIRules:
    """Zhang (2008) R1–R10 applied to fixpoint on a ``PartialPAG``.

    R1–R3: simple endpoint propagations.
    R4: discriminating-path rule (uses ``PartialPAG.sepsets``).
    R5–R7: undirected-edge / circle-path rules (selection-bias-aware).
    R8: composition of mediator chains.
    R9–R10: uncovered potentially-directed paths (completeness rules).

    Subset via ``rules=``; all rules listed in the set are applied,
    everything else skipped.
    """

    def __init__(self, *, rules: frozenset[str] | None = None) -> None:
        chosen = _FCI_ALL_RULES if rules is None else frozenset(rules)
        unsupported = chosen - _FCI_ALL_RULES
        if unsupported:
            raise CBCDInputError(f"unsupported FCI rules: {sorted(unsupported)}")
        self.rules = chosen

    def __call__(
        self,
        graph: PartialPAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
    ) -> PAG:
        endpoints = graph.endpoints.copy()
        n = graph.n_vars
        sepsets = graph.sepsets if graph.sepsets is not None else {}

        if background is not None:
            for u, v in background.required_directed:
                if u >= n or v >= n:
                    continue
                if endpoints[u, v] == EndpointMark.NO_EDGE:
                    continue
                _set_mark(endpoints, u, v, EndpointMark.ARROW, background)
                _set_mark(endpoints, v, u, EndpointMark.TAIL, background)

        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            changed = False
            if "R1" in self.rules and _apply_zhang_r1(endpoints, n, background):
                changed = True
            if "R2" in self.rules and _apply_zhang_r2(endpoints, n, background):
                changed = True
            if "R3" in self.rules and _apply_zhang_r3(endpoints, n, background):
                changed = True
            if "R4" in self.rules and _apply_zhang_r4(endpoints, n, sepsets, background):
                changed = True
            if "R5" in self.rules and _apply_zhang_r5(endpoints, n, background):
                changed = True
            if "R6" in self.rules and _apply_zhang_r6(endpoints, n, background):
                changed = True
            if "R7" in self.rules and _apply_zhang_r7(endpoints, n, background):
                changed = True
            if "R8" in self.rules and _apply_zhang_r8(endpoints, n, background):
                changed = True
            if "R9" in self.rules and _apply_zhang_r9(endpoints, n, background):
                changed = True
            if "R10" in self.rules and _apply_zhang_r10(endpoints, n, background):
                changed = True
            if not changed:
                break
            iteration += 1

        return PAG(
            n_vars=n,
            endpoints=endpoints,
            var_names=graph.var_names,
            sepsets=dict(sepsets) if sepsets else None,
        )


# --- Single-mark write with background-knowledge guard ----------------------


def _set_mark(
    endpoints: NDArray[np.int8],
    source: int,
    target: int,
    new_mark: EndpointMark,
    bk: BackgroundKnowledge | None,
) -> bool:
    """Write ``endpoints[source, target] = new_mark`` (mark at ``target`` on edge
    {source, target}) iff different and not blocked by background. Return True
    on change.
    """
    if endpoints[source, target] == EndpointMark.NO_EDGE:
        return False
    if endpoints[source, target] == new_mark:
        return False
    if bk is not None:
        other_mark = int(endpoints[target, source])
        if (
            new_mark == EndpointMark.ARROW
            and other_mark == int(EndpointMark.TAIL)
            and bk.is_forbidden_directed(source, target)
        ):
            return False
        if (
            new_mark == EndpointMark.TAIL
            and other_mark == int(EndpointMark.ARROW)
            and bk.is_forbidden_directed(target, source)
        ):
            return False
    endpoints[source, target] = new_mark
    return True


def _has_pag_edge(endpoints: NDArray[np.int8], i: int, j: int) -> bool:
    return bool(endpoints[i, j] != EndpointMark.NO_EDGE)


# --- R1: α *→ β o─* γ, α not adjacent γ  ⟹  α *→ β → γ -----------------------


def _apply_zhang_r1(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for beta in range(n):
            if alpha == beta or endpoints[alpha, beta] != EndpointMark.ARROW:
                continue
            for gamma in range(n):
                if gamma in (alpha, beta):
                    continue
                if endpoints[gamma, beta] != EndpointMark.CIRCLE:
                    continue
                if _has_pag_edge(endpoints, alpha, gamma):
                    continue
                if _set_mark(endpoints, gamma, beta, EndpointMark.TAIL, bk):
                    changed = True
                if _set_mark(endpoints, beta, gamma, EndpointMark.ARROW, bk):
                    changed = True
    return changed


# --- R2: (α → β *→ γ) or (α *→ β → γ), with α *─o γ  ⟹  α *→ γ -------------


def _apply_zhang_r2(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for gamma in range(n):
            if alpha == gamma:
                continue
            if endpoints[alpha, gamma] != EndpointMark.CIRCLE:
                continue
            for beta in range(n):
                if beta in (alpha, gamma):
                    continue
                if endpoints[beta, gamma] != EndpointMark.ARROW:
                    continue
                if endpoints[alpha, beta] != EndpointMark.ARROW:
                    continue
                # Pattern A: α → β  (TAIL at α on edge α-β) AND β *→ γ
                pattern_a = endpoints[beta, alpha] == EndpointMark.TAIL
                # Pattern B: α *→ β AND β → γ  (TAIL at β on edge β-γ)
                pattern_b = endpoints[gamma, beta] == EndpointMark.TAIL
                if not (pattern_a or pattern_b):
                    continue
                if _set_mark(endpoints, alpha, gamma, EndpointMark.ARROW, bk):
                    changed = True
    return changed


# --- R3: α *→ β ←* γ, α *─o θ o─* γ, α not adj γ, θ *─o β  ⟹  θ *→ β --------


def _apply_zhang_r3(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for beta in range(n):
        for theta in range(n):
            if theta == beta:
                continue
            if endpoints[theta, beta] != EndpointMark.CIRCLE:
                continue
            # Look for α and γ with the required pattern.
            for alpha in range(n):
                if alpha in (beta, theta):
                    continue
                if endpoints[alpha, beta] != EndpointMark.ARROW:
                    continue
                if endpoints[alpha, theta] != EndpointMark.CIRCLE:
                    continue
                for gamma in range(n):
                    if gamma in (alpha, beta, theta):
                        continue
                    if endpoints[gamma, beta] != EndpointMark.ARROW:
                        continue
                    if endpoints[gamma, theta] != EndpointMark.CIRCLE:
                        continue
                    if _has_pag_edge(endpoints, alpha, gamma):
                        continue
                    if _set_mark(endpoints, theta, beta, EndpointMark.ARROW, bk):
                        changed = True
    return changed


# --- R4: discriminating-path rule -------------------------------------------


def _apply_zhang_r4(
    endpoints: NDArray[np.int8],
    n: int,
    sepsets: dict[frozenset[int], tuple[int, ...]],
    bk: BackgroundKnowledge | None,
) -> bool:
    """For each (a, b, c) with b o─* c, search for a discriminating path
    ⟨θ, ..., a, b, c⟩ for b. If b ∈ Sepset(θ, c): orient b-c as b → c.
    Otherwise: orient ⟨a, b, c⟩ as a ↔ b ↔ c."""
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                if endpoints[c, b] != EndpointMark.CIRCLE:
                    continue
                path = find_discriminating_path(endpoints, a, b, c)
                if path is None:
                    continue
                theta = path[0]
                witness = sepsets.get(frozenset({theta, c}))
                if witness is not None and b in witness:
                    if _set_mark(endpoints, c, b, EndpointMark.TAIL, bk):
                        changed = True
                    if _set_mark(endpoints, b, c, EndpointMark.ARROW, bk):
                        changed = True
                else:
                    # Orient a ↔ b and b ↔ c (set arrows at b on both arms,
                    # arrow at c on edge b-c).
                    if _set_mark(endpoints, a, b, EndpointMark.ARROW, bk):
                        changed = True
                    if _set_mark(endpoints, c, b, EndpointMark.ARROW, bk):
                        changed = True
                    if _set_mark(endpoints, b, c, EndpointMark.ARROW, bk):
                        changed = True
    return changed


# --- R5: uncovered circle path → orient α o─o β + path edges as undirected --


def _apply_zhang_r5(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for beta in range(alpha + 1, n):
            if not (
                endpoints[alpha, beta] == EndpointMark.CIRCLE
                and endpoints[beta, alpha] == EndpointMark.CIRCLE
            ):
                continue
            path = find_uncovered_circle_path(endpoints, alpha, beta)
            if path is None:
                continue
            # R5's endpoint conditions: α non-adjacent to last intermediate;
            # β non-adjacent to first intermediate. (path[1] = first intermediate;
            # path[-2] = last intermediate.)
            if len(path) < 3:
                continue
            first_int = path[1]
            last_int = path[-2]
            if _has_pag_edge(endpoints, alpha, last_int):
                continue
            if _has_pag_edge(endpoints, beta, first_int):
                continue
            # Orient α o─o β as α — β.
            if _set_mark(endpoints, alpha, beta, EndpointMark.TAIL, bk):
                changed = True
            if _set_mark(endpoints, beta, alpha, EndpointMark.TAIL, bk):
                changed = True
            # Orient every edge on the path as undirected.
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if _set_mark(endpoints, u, v, EndpointMark.TAIL, bk):
                    changed = True
                if _set_mark(endpoints, v, u, EndpointMark.TAIL, bk):
                    changed = True
    return changed


# --- R6: α — β o─* γ  ⟹  β ─* γ (set TAIL at β) -----------------------------


def _apply_zhang_r6(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for beta in range(n):
            if alpha == beta:
                continue
            # α — β: TAIL at α (endpoints[β, α] == TAIL) and TAIL at β.
            if not (
                endpoints[beta, alpha] == EndpointMark.TAIL
                and endpoints[alpha, beta] == EndpointMark.TAIL
            ):
                continue
            for gamma in range(n):
                if gamma in (alpha, beta):
                    continue
                if endpoints[gamma, beta] != EndpointMark.CIRCLE:
                    continue
                if _set_mark(endpoints, gamma, beta, EndpointMark.TAIL, bk):
                    changed = True
    return changed


# --- R7: α ─o β o─* γ, α not adj γ  ⟹  β ─* γ (set TAIL at β) ---------------


def _apply_zhang_r7(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for beta in range(n):
            if alpha == beta:
                continue
            # α ─o β: TAIL at α (endpoints[β, α] == TAIL), CIRCLE at β.
            if endpoints[beta, alpha] != EndpointMark.TAIL:
                continue
            if endpoints[alpha, beta] != EndpointMark.CIRCLE:
                continue
            for gamma in range(n):
                if gamma in (alpha, beta):
                    continue
                if endpoints[gamma, beta] != EndpointMark.CIRCLE:
                    continue
                if _has_pag_edge(endpoints, alpha, gamma):
                    continue
                if _set_mark(endpoints, gamma, beta, EndpointMark.TAIL, bk):
                    changed = True
    return changed


# --- R8: (α → β → γ or α ─o β → γ) and α o→ γ  ⟹  α → γ --------------------


def _apply_zhang_r8(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for gamma in range(n):
            if alpha == gamma:
                continue
            # α o→ γ: CIRCLE at α (endpoints[γ, α] == CIRCLE), ARROW at γ.
            if endpoints[gamma, alpha] != EndpointMark.CIRCLE:
                continue
            if endpoints[alpha, gamma] != EndpointMark.ARROW:
                continue
            for beta in range(n):
                if beta in (alpha, gamma):
                    continue
                # β → γ: TAIL at β (endpoints[γ, β] == TAIL), ARROW at γ.
                if endpoints[gamma, beta] != EndpointMark.TAIL:
                    continue
                if endpoints[beta, gamma] != EndpointMark.ARROW:
                    continue
                # α → β  (TAIL at α, ARROW at β)  OR  α ─o β  (TAIL at α, CIRCLE at β).
                if endpoints[beta, alpha] != EndpointMark.TAIL:
                    continue
                mark_at_beta = endpoints[alpha, beta]
                if mark_at_beta not in (
                    EndpointMark.ARROW,
                    EndpointMark.CIRCLE,
                ):
                    continue
                if _set_mark(endpoints, gamma, alpha, EndpointMark.TAIL, bk):
                    changed = True
    return changed


# --- R9: α o→ γ + uncovered PD path α=v_0 → v_1 → ... → v_k=γ
#         with γ and v_1 non-adjacent  ⟹  α → γ ------------------------------


def _apply_zhang_r9(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for gamma in range(n):
            if alpha == gamma:
                continue
            if endpoints[gamma, alpha] != EndpointMark.CIRCLE:
                continue
            if endpoints[alpha, gamma] != EndpointMark.ARROW:
                continue
            path = find_uncovered_pd_path(endpoints, alpha, gamma)
            if path is None or len(path) < 3:
                continue
            v1 = path[1]
            if _has_pag_edge(endpoints, gamma, v1):
                continue
            if _set_mark(endpoints, gamma, alpha, EndpointMark.TAIL, bk):
                changed = True
    return changed


# --- R10: α o→ γ, β → γ ← δ, two uncovered PD paths from α with distinct
#          first vertices μ, ω, μ ≠ ω, μ not adj ω  ⟹  α → γ -----------------


def _apply_zhang_r10(endpoints: NDArray[np.int8], n: int, bk: BackgroundKnowledge | None) -> bool:
    changed = False
    for alpha in range(n):
        for gamma in range(n):
            if alpha == gamma:
                continue
            if endpoints[gamma, alpha] != EndpointMark.CIRCLE:
                continue
            if endpoints[alpha, gamma] != EndpointMark.ARROW:
                continue
            parents_of_gamma = [
                v
                for v in range(n)
                if v != gamma
                and endpoints[gamma, v] == EndpointMark.TAIL
                and endpoints[v, gamma] == EndpointMark.ARROW
            ]
            if len(parents_of_gamma) < 2:
                continue
            for i, beta in enumerate(parents_of_gamma):
                for delta in parents_of_gamma[i + 1 :]:
                    p1 = find_uncovered_pd_path(endpoints, alpha, beta)
                    p2 = find_uncovered_pd_path(endpoints, alpha, delta)
                    if p1 is None or p2 is None:
                        continue
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    mu = p1[1]
                    omega = p2[1]
                    if mu == omega:
                        continue
                    if _has_pag_edge(endpoints, mu, omega):
                        continue
                    if _set_mark(endpoints, gamma, alpha, EndpointMark.TAIL, bk):
                        changed = True
                        break
                else:
                    continue
                break
    return changed
