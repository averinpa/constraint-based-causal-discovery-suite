"""BackgroundKnowledge validation per decision D5."""

from __future__ import annotations

import pytest

from cbcd.background import BackgroundKnowledge
from cbcd.exceptions import CBCDInputError


def test_default_construction() -> None:
    bk = BackgroundKnowledge()
    assert bk.forbidden_directed == frozenset()
    assert bk.required_directed == frozenset()


def test_required_directed_query() -> None:
    bk = BackgroundKnowledge(required_directed=frozenset({(0, 1)}))
    assert bk.is_required_directed(0, 1)
    assert not bk.is_required_directed(1, 0)


def test_forbidden_directed_query() -> None:
    bk = BackgroundKnowledge(forbidden_directed=frozenset({(0, 1)}))
    assert bk.is_forbidden_directed(0, 1)
    assert not bk.is_forbidden_directed(1, 0)


def test_required_and_forbidden_overlap_rejected() -> None:
    with pytest.raises(CBCDInputError, match="forbidden_directed"):
        BackgroundKnowledge(
            forbidden_directed=frozenset({(0, 1)}),
            required_directed=frozenset({(0, 1)}),
        )


def test_required_and_forbidden_adjacent_overlap_rejected() -> None:
    with pytest.raises(CBCDInputError, match="forbidden_adjacent"):
        BackgroundKnowledge(
            required_directed=frozenset({(0, 1)}),
            forbidden_adjacent=frozenset({frozenset({0, 1})}),
        )


def test_required_self_loop_rejected() -> None:
    with pytest.raises(CBCDInputError, match="self-loop"):
        BackgroundKnowledge(required_directed=frozenset({(0, 0)}))


def test_required_cycle_rejected() -> None:
    with pytest.raises(CBCDInputError, match="cycle"):
        BackgroundKnowledge(required_directed=frozenset({(0, 1), (1, 2), (2, 0)}))


def test_tier_violation_rejected() -> None:
    with pytest.raises(CBCDInputError, match="tier"):
        BackgroundKnowledge(
            tiers=(frozenset({0}), frozenset({1})),
            required_directed=frozenset({(1, 0)}),
        )


def test_tier_node_in_multiple_tiers_rejected() -> None:
    with pytest.raises(CBCDInputError, match="multiple tiers"):
        BackgroundKnowledge(tiers=(frozenset({0, 1}), frozenset({1, 2})))


def test_tier_implies_forbidden_directed() -> None:
    # Edges from tier 1 to tier 0 are forbidden by tier ordering.
    bk = BackgroundKnowledge(tiers=(frozenset({0}), frozenset({1})))
    assert bk.is_forbidden_directed(1, 0)
    assert not bk.is_forbidden_directed(0, 1)
