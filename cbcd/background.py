"""Background knowledge: tiers, forbidden / required edges, forbidden adjacencies."""

from __future__ import annotations

from dataclasses import dataclass, field

from cbcd.exceptions import CBCDInputError


def _has_directed_cycle(n_vars: int, edges: frozenset[tuple[int, int]]) -> bool:
    children: dict[int, list[int]] = {i: [] for i in range(n_vars)}
    indeg = [0] * n_vars
    for u, v in edges:
        children[u].append(v)
        indeg[v] += 1
    stack = [i for i in range(n_vars) if indeg[i] == 0]
    visited = 0
    while stack:
        u = stack.pop()
        visited += 1
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    # Edges may reference vertices >= n_vars; treat those as cycles too.
    referenced = {u for u, _ in edges} | {v for _, v in edges}
    return visited != len(referenced | set(range(n_vars)))


@dataclass(frozen=True)
class BackgroundKnowledge:
    """Domain constraints applied throughout an algorithm run.

    All sets are normalized to ``frozenset`` on construction. Validation
    follows decision D5: fail fast at construction on inconsistencies.
    """

    forbidden_directed: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    required_directed: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    forbidden_adjacent: frozenset[frozenset[int]] = field(default_factory=frozenset)
    tiers: tuple[frozenset[int], ...] = ()

    def __post_init__(self) -> None:
        forbidden = frozenset(self.forbidden_directed)
        required = frozenset(self.required_directed)
        forbidden_adj = frozenset(frozenset(p) for p in self.forbidden_adjacent)
        tiers = tuple(frozenset(t) for t in self.tiers)
        object.__setattr__(self, "forbidden_directed", forbidden)
        object.__setattr__(self, "required_directed", required)
        object.__setattr__(self, "forbidden_adjacent", forbidden_adj)
        object.__setattr__(self, "tiers", tiers)

        for u, v in required:
            if u == v:
                raise CBCDInputError(f"required_directed self-loop: ({u}, {v})")

        for edge in required:
            if edge in forbidden:
                raise CBCDInputError(f"required_directed edge {edge} is also in forbidden_directed")

        for u, v in required:
            pair = frozenset({u, v})
            if pair in forbidden_adj:
                raise CBCDInputError(
                    f"required_directed edge {(u, v)} contradicts forbidden_adjacent {set(pair)}"
                )

        if required:
            n_implied = max(max(u, v) for u, v in required) + 1
            if _has_directed_cycle(n_implied, required):
                raise CBCDInputError("required_directed contains a directed cycle")

        if tiers:
            tier_of: dict[int, int] = {}
            for level, members in enumerate(tiers):
                for v in members:
                    if v in tier_of:
                        raise CBCDInputError(
                            f"node {v} appears in multiple tiers ({tier_of[v]}, {level})"
                        )
                    tier_of[v] = level
            for u, v in required:
                if u in tier_of and v in tier_of and tier_of[u] > tier_of[v]:
                    raise CBCDInputError(
                        f"required edge ({u}, {v}) violates tiers: tier({u})={tier_of[u]} "
                        f"> tier({v})={tier_of[v]}"
                    )

    def is_forbidden_directed(self, u: int, v: int) -> bool:
        if (u, v) in self.forbidden_directed:
            return True
        # Tier ordering: edge from later tier to earlier tier is forbidden.
        for level, members in enumerate(self.tiers):
            if u in members:
                u_tier = level
                break
        else:
            u_tier = -1
        for level, members in enumerate(self.tiers):
            if v in members:
                v_tier = level
                break
        else:
            v_tier = -1
        return u_tier >= 0 and v_tier >= 0 and u_tier > v_tier

    def is_forbidden_adjacent(self, u: int, v: int) -> bool:
        return frozenset({u, v}) in self.forbidden_adjacent

    def is_required_directed(self, u: int, v: int) -> bool:
        return (u, v) in self.required_directed
