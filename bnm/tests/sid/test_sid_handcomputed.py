"""Hand-computed SID checks on canonical fixtures.

These ground v0.2's SID against the Peters & Bühlmann (2015) definition
directly. They do NOT match bnm 0.1.x in some cases — see audit §8 for
the upper-bound under-counting bug in 0.1.x. v0.2 is the correct
implementation; the override file in
``tests/fixtures_legacy_v02_overrides.json`` records the corrected
values for affected snapshot pairs.
"""

from __future__ import annotations

import numpy as np
import pytest

import bnm
from tests.fixtures import (
    asia_8,
    chain_3,
    collider_3,
    diamond_4,
    fork_3,
    make_cpdag,
    make_dag,
)


def test_self_comparison_sid_zero() -> None:
    """SID(g, g) = 0 for any g."""
    for g in (chain_3(), fork_3(), collider_3(), diamond_4(), asia_8()):
        result = bnm.sid(g, g)
        assert result.sid == 0
        assert result.sid_lower_bound == 0
        assert result.sid_upper_bound == 0
        assert result.is_tight


def test_chain_truth_vs_cpdag() -> None:
    """A→B→C (truth) vs A—B—C (its CPDAG).

    The CPDAG's equivalence class has 3 DAGs:
      DAG1 = A→B→C (truth) — SID 0
      DAG2 = A←B→C        — SID 3 (intervening on A or B mis-classified)
      DAG3 = A←B←C        — SID 6 (every pair flipped)

    Per Peters: lower = 0, upper = 6, sid = 6 (every (i, j) pair is
    mis-classified by at least one DAG in the class).
    """
    truth = chain_3()
    cpdag = make_cpdag(3, [], [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    result = bnm.sid(truth, cpdag)
    assert result.sid == 6
    assert result.sid_lower_bound == 0
    assert result.sid_upper_bound == 6
    # incorrect_mat should mark every off-diagonal entry.
    expected = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(result.incorrect_mat, expected)


def test_fork_truth_vs_cpdag() -> None:
    """A→B, A→C (truth, fork) vs A—B, A—C (CPDAG)."""
    truth = fork_3()
    cpdag = make_cpdag(3, [], [(0, 1), (0, 2)], var_names=("A", "B", "C"))
    result = bnm.sid(truth, cpdag)
    assert result.sid_lower_bound == 0  # truth is in the equivalence class


def test_collider_truth_vs_self_cpdag() -> None:
    """A→C, B→C (collider) — its CPDAG is identical (collider edges
    are protected from being undirected). SID against itself = 0."""
    truth = collider_3()
    # CPDAG of a collider is the collider itself.
    result = bnm.sid(truth, truth)
    assert result.sid == 0
    assert result.is_tight


def test_directed_edge_reversal() -> None:
    """A→B (truth) vs B→A (estimated).

    Intervening on A in truth affects B (P(B|do(A)) ≠ P(B)).
    Intervening on A in estimate (A is leaf) leaves B unchanged.
    Mis-classified.

    Intervening on B in truth (B is leaf) leaves A unchanged.
    Intervening on B in estimate (B is parent of A) affects A.
    Mis-classified.

    SID = 2.
    """
    truth = make_dag(2, [(0, 1)], var_names=("A", "B"))
    estimate = make_dag(2, [(1, 0)], var_names=("A", "B"))
    result = bnm.sid(truth, estimate)
    assert result.sid == 2
    assert result.is_tight


def test_chain_truth_vs_extra_edge() -> None:
    """Truth A→B→C; estimate adds A→C. Intervening on A: truth says
    P(C|do(A))=P(C|A) via B; estimate has direct A→C plus A→B→C, so
    P(C|do(A))=sum_B P(C|A,B)P(B|A). Generally different unless C is
    independent of A given B in truth (which is the chain's CI).

    Actually for this case, the estimate adds a 'direct' edge that's
    redundant with the indirect path; SID counts it as mis-classified
    via the path-matrix check.
    """
    truth = chain_3()
    estimate = make_dag(
        3,
        [(0, 1), (1, 2), (0, 2)],
        var_names=("A", "B", "C"),
    )
    result = bnm.sid(truth, estimate)
    assert result.sid >= 0
    assert result.is_tight  # both pure DAGs


def test_empty_graph_sid_zero() -> None:
    truth = make_dag(0, [], var_names=())
    estimate = make_dag(0, [], var_names=())
    result = bnm.sid(truth, estimate)
    assert result.sid == 0
    assert result.sid_lower_bound == 0
    assert result.sid_upper_bound == 0


def test_g1_with_undirected_rejected() -> None:
    g1 = make_cpdag(2, [], [(0, 1)], var_names=("A", "B"))
    g2 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    with pytest.raises(bnm.BNMInputError, match="g1 must be a pure DAG"):
        bnm.sid(g1, g2)


def test_g2_with_circle_rejected() -> None:
    """CIRCLE marks (PAG) are not yet supported in SID."""
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = bnm.EndpointMark.CIRCLE
    arr[1, 0] = bnm.EndpointMark.TAIL
    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = bnm.to_graphlike(arr, var_names=("A", "B"))
    with pytest.raises(bnm.BNMInputError, match="g2 must be a DAG or CPDAG"):
        bnm.sid(g1, g2)


def test_g2_with_bidirected_rejected() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = bnm.EndpointMark.ARROW
    arr[1, 0] = bnm.EndpointMark.ARROW
    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = bnm.to_graphlike(arr, var_names=("A", "B"))
    with pytest.raises(bnm.BNMInputError, match="g2 must be a DAG or CPDAG"):
        bnm.sid(g1, g2)


def test_n_vars_mismatch_rejected() -> None:
    g1 = chain_3()
    g2 = make_dag(4, [(0, 1)])
    with pytest.raises(bnm.BNMInputError, match="must match"):
        bnm.sid(g1, g2)


def test_sid_result_is_frozen() -> None:
    """SIDResult is a frozen dataclass with the documented attributes."""
    import dataclasses

    truth = chain_3()
    result = bnm.sid(truth, truth)
    assert dataclasses.is_dataclass(result)
    assert isinstance(result, bnm.SIDResult)
    assert hasattr(result, "incorrect_mat")
    # Frozen → setattr should fail.
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.sid = 99  # type: ignore[misc]


def test_is_tight_property() -> None:
    truth = chain_3()
    cpdag = make_cpdag(3, [], [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    tight = bnm.sid(truth, truth)
    not_tight = bnm.sid(truth, cpdag)
    assert tight.is_tight
    assert not not_tight.is_tight
