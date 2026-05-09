"""FCIRules R1–R10 unit tests.

Each rule has a minimal-firing test, a pattern that should NOT fire, and (where
relevant) a background-knowledge block test. Plus subsetting and convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.background import BackgroundKnowledge
from cbcd.exceptions import CBCDInputError
from cbcd.graph import EndpointMark, PartialPAG
from cbcd.rules import FCIRules

ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
CIRC = EndpointMark.CIRCLE


def _ep(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


def _run(graph: PartialPAG, *, rules: frozenset[str] | None = None, bk=None):
    return FCIRules(rules=rules)(graph, background=bk)


# --- R1 --------------------------------------------------------------------


def test_r1_fires() -> None:
    # 0 *→ 1 o─o 2, 0 not adjacent 2 → orient 1 → 2.
    g = PartialPAG(3, _ep(3, [(0, 1, CIRC, ARR), (1, 2, CIRC, CIRC)]))
    out = _run(g, rules=frozenset({"R1"}))
    assert out.endpoints[2, 1] == TAIL
    assert out.endpoints[1, 2] == ARR


def test_r1_does_not_fire_when_alpha_gamma_adjacent() -> None:
    # 0 *→ 1 o─o 2 with 0 — 2 adjacent → R1 should NOT fire.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, CIRC, ARR), (1, 2, CIRC, CIRC), (0, 2, CIRC, CIRC)]),
    )
    out = _run(g, rules=frozenset({"R1"}))
    assert out.endpoints[2, 1] == CIRC
    assert out.endpoints[1, 2] == CIRC


def test_r1_blocked_by_forbidden_directed() -> None:
    # 0 *→ 1 o─o 2, 0 not adjacent 2; forbid 1 → 2.
    g = PartialPAG(3, _ep(3, [(0, 1, CIRC, ARR), (1, 2, CIRC, CIRC)]))
    bk = BackgroundKnowledge(forbidden_directed=frozenset({(1, 2)}))
    out = _run(g, rules=frozenset({"R1"}), bk=bk)
    # The arrow-write at 2 (which would create 1→2 with TAIL at 1) must be blocked.
    # Setting TAIL at 1 first leaves CIRCLE at 2; then setting ARROW at 2
    # would close the directed edge — blocked. So mark at 2 stays CIRCLE.
    assert out.endpoints[1, 2] == CIRC


# --- R2 --------------------------------------------------------------------


def test_r2_pattern_a_fires() -> None:
    # α → β *→ γ with α o─o γ → set arrow at γ on edge α-γ.
    # 0 → 1: TAIL at 0 (endpoints[1, 0] = TAIL), ARROW at 1 (endpoints[0, 1] = ARROW).
    # 1 *→ 2: ARROW at 2 (endpoints[1, 2] = ARROW); mark at 1 = anything.
    # 0 o─o 2.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, ARR), (1, 2, CIRC, ARR), (0, 2, CIRC, CIRC)]),
    )
    out = _run(g, rules=frozenset({"R2"}))
    # CIRCLE at 2 on edge 0-2 should become ARROW.
    assert out.endpoints[0, 2] == ARR


def test_r2_does_not_fire_without_circle_at_gamma() -> None:
    # α → β *→ γ but α-γ is α → γ already (TAIL at α, ARROW at γ).
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, ARR), (1, 2, CIRC, ARR), (0, 2, TAIL, ARR)]),
    )
    out = _run(g, rules=frozenset({"R2"}))
    # No change: endpoints[0, 2] already ARR.
    assert out.endpoints[0, 2] == ARR


# --- R3 --------------------------------------------------------------------


def test_r3_fires() -> None:
    # α=0, β=1, γ=2, θ=3.
    # α *→ β ←* γ: arrow at β from α and γ.
    # α *─o θ o─* γ: circle at θ on both edges α-θ and γ-θ.
    # α not adj γ.
    # θ *─o β: circle at β on edge θ-β.
    g = PartialPAG(
        4,
        _ep(
            4,
            [
                (0, 1, CIRC, ARR),  # α=0 *→ β=1
                (2, 1, CIRC, ARR),  # γ=2 *→ β=1
                (0, 3, CIRC, CIRC),  # α-θ: circle at θ; mark at α = CIRCLE
                (2, 3, CIRC, CIRC),  # γ-θ: circle at θ
                (3, 1, CIRC, CIRC),  # θ-β: circle at β
            ],
        ),
    )
    out = _run(g, rules=frozenset({"R3"}))
    assert out.endpoints[3, 1] == ARR  # arrow at β=1 from θ=3


# --- R4 --------------------------------------------------------------------


def test_r4_fires_with_sepset_in_witness() -> None:
    # Discriminating path ⟨θ=0, a=1, b=2, c=3⟩ with sepset(θ, c) = {b}.
    # Since b ∈ Sepset(θ, c), orient b o─* c as b → c.
    # b o─o c initially.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, TAIL),
            (1, 3, TAIL, ARR),
            (2, 3, CIRC, CIRC),
        ],
    )
    g = PartialPAG(4, ep, sepsets={frozenset({0, 3}): (2,)})
    out = _run(g, rules=frozenset({"R4"}))
    # b → c: TAIL at b (endpoints[3, 2] = TAIL) and ARROW at c (endpoints[2, 3] = ARROW).
    assert out.endpoints[3, 2] == TAIL
    assert out.endpoints[2, 3] == ARR


def test_r4_fires_without_sepset_in_witness() -> None:
    # Same path but sepset(θ, c) = {} (b NOT in sepset).
    # → orient ⟨a, b, c⟩ as a ↔ b ↔ c.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, TAIL),
            (1, 3, TAIL, ARR),
            (2, 3, CIRC, CIRC),
        ],
    )
    g = PartialPAG(4, ep, sepsets={frozenset({0, 3}): ()})
    out = _run(g, rules=frozenset({"R4"}))
    # a ↔ b: ARROW at b on edge a-b. endpoints[1, 2] should be ARROW.
    assert out.endpoints[1, 2] == ARR
    # b ↔ c: ARROW at b and at c on edge b-c.
    assert out.endpoints[3, 2] == ARR
    assert out.endpoints[2, 3] == ARR


# --- R5 --------------------------------------------------------------------


def test_r5_fires_on_uncovered_circle_path() -> None:
    # 0 o─o 1 o─o 2 o─o 3 o─o 0  (a 4-cycle of o-o edges).
    # Uncovered circle path 0-1-2-3 between 0 and 3; (0, 1, 2) interior triple
    # has 0 not adj 2; (1, 2, 3) has 1 not adj 3. So uncovered.
    # Endpoint conditions: 0 not adj 2 (last_int) — already; 3 not adj 1
    # (first_int) — already.
    g = PartialPAG(
        4,
        _ep(
            4,
            [
                (0, 1, CIRC, CIRC),
                (1, 2, CIRC, CIRC),
                (2, 3, CIRC, CIRC),
                (0, 3, CIRC, CIRC),
            ],
        ),
    )
    out = _run(g, rules=frozenset({"R5"}))
    # 0 — 3: TAIL at both ends.
    assert out.endpoints[0, 3] == TAIL
    assert out.endpoints[3, 0] == TAIL
    # 0 — 1, 1 — 2, 2 — 3 also undirected.
    assert out.endpoints[0, 1] == TAIL
    assert out.endpoints[1, 0] == TAIL


# --- R6 --------------------------------------------------------------------


def test_r6_fires() -> None:
    # α — β o─* γ → orient β-γ with TAIL at β.
    # 0 — 1: TAIL at both. 1 o─o 2.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, TAIL), (1, 2, CIRC, CIRC)]),
    )
    out = _run(g, rules=frozenset({"R6"}))
    assert out.endpoints[2, 1] == TAIL  # tail at β=1 from γ=2


# --- R7 --------------------------------------------------------------------


def test_r7_fires() -> None:
    # α ─o β o─* γ, α not adj γ → tail at β on edge β-γ.
    # 0 ─o 1 (TAIL at 0, CIRCLE at 1): endpoints[1, 0] = TAIL, endpoints[0, 1] = CIRC.
    # 1 o─o 2: endpoints[1, 2] = CIRC, endpoints[2, 1] = CIRC.
    # 0 not adj 2.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, CIRC), (1, 2, CIRC, CIRC)]),
    )
    out = _run(g, rules=frozenset({"R7"}))
    assert out.endpoints[2, 1] == TAIL


# --- R8 --------------------------------------------------------------------


def test_r8_fires_with_arrow_chain() -> None:
    # α=0 → β=1 → γ=2, plus α o→ γ. → α → γ.
    # 0 → 1: endpoints[1, 0] = TAIL, endpoints[0, 1] = ARR.
    # 1 → 2: endpoints[2, 1] = TAIL, endpoints[1, 2] = ARR.
    # 0 o→ 2: endpoints[2, 0] = CIRC, endpoints[0, 2] = ARR.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, ARR), (1, 2, TAIL, ARR), (0, 2, CIRC, ARR)]),
    )
    out = _run(g, rules=frozenset({"R8"}))
    assert out.endpoints[2, 0] == TAIL  # mark at α=0 from γ=2 → TAIL


def test_r8_fires_with_circle_to_arrow_chain() -> None:
    # α=0 ─o β=1 → γ=2, plus α o→ γ → α → γ.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, CIRC), (1, 2, TAIL, ARR), (0, 2, CIRC, ARR)]),
    )
    out = _run(g, rules=frozenset({"R8"}))
    assert out.endpoints[2, 0] == TAIL


# --- R9 --------------------------------------------------------------------


def test_r9_fires_on_uncovered_pd_path() -> None:
    # α=0 o→ γ=3, plus uncovered PD path 0 → 1 → 2 → 3 with γ=3 non-adj v_1=1.
    # 0 o→ 3: endpoints[3, 0] = CIRC, endpoints[0, 3] = ARR.
    # 0 → 1: endpoints[1, 0] = TAIL, endpoints[0, 1] = ARR.
    # 1 → 2: endpoints[2, 1] = TAIL, endpoints[1, 2] = ARR.
    # 2 → 3: endpoints[3, 2] = TAIL, endpoints[2, 3] = ARR.
    # 1 and 3 non-adjacent (no edge between them — yes).
    g = PartialPAG(
        4,
        _ep(
            4,
            [
                (0, 3, CIRC, ARR),
                (0, 1, TAIL, ARR),
                (1, 2, TAIL, ARR),
                (2, 3, TAIL, ARR),
            ],
        ),
    )
    out = _run(g, rules=frozenset({"R9"}))
    assert out.endpoints[3, 0] == TAIL


# --- subsetting + convergence ----------------------------------------------


def test_unsupported_rule_name_raises() -> None:
    with pytest.raises(CBCDInputError):
        FCIRules(rules=frozenset({"R99"}))


def test_subset_only_fires_listed_rules() -> None:
    # R2-firing pattern; running with rules={"R1"} should not change anything.
    g = PartialPAG(
        3,
        _ep(3, [(0, 1, TAIL, ARR), (1, 2, CIRC, ARR), (0, 2, CIRC, CIRC)]),
    )
    out = _run(g, rules=frozenset({"R1"}))
    # CIRCLE at 2 should remain CIRCLE.
    assert out.endpoints[0, 2] == CIRC


def test_idempotent_at_fixpoint() -> None:
    # A closed PAG run through full FCIRules: no change.
    ep = _ep(3, [(0, 1, TAIL, ARR), (1, 2, TAIL, ARR), (0, 2, TAIL, ARR)])
    g = PartialPAG(3, ep)
    out = FCIRules()(g)
    assert np.array_equal(out.endpoints, ep)
