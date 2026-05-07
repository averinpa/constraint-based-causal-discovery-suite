"""Hand-computed comparative metric checks on canonical fixtures."""

from __future__ import annotations

import bnm
from tests.fixtures import chain_3, collider_3, fork_3, make_cpdag, make_dag


def test_self_comparison_is_perfect() -> None:
    g = chain_3()
    assert bnm.shd(g, g) == 0
    assert bnm.hd(g, g) == 0
    assert bnm.true_positives(g, g) == 2
    assert bnm.false_positives(g, g) == 0
    assert bnm.false_negatives(g, g) == 0
    assert bnm.precision(g, g) == 1.0
    assert bnm.recall(g, g) == 1.0
    assert bnm.f1(g, g) == 1.0


def test_chain_vs_fork() -> None:
    """A→B→C vs A→B, A→C. Shared edge A→B; chain has B→C, fork has A→C.
    Adjacencies: chain={(A,B),(B,C)}; fork={(A,B),(A,C)}.
    additions = 1 (A,C in fork only); deletions = 1 (B,C in chain only).
    The shared adjacency is A↔B, both directed A→B → no reversal there.
    """
    chain = chain_3()
    fork = fork_3()
    assert bnm.count_additions(chain, fork) == 1
    assert bnm.count_deletions(chain, fork) == 1
    assert bnm.count_reversals(chain, fork) == 0
    assert bnm.shd(chain, fork) == 2
    assert bnm.hd(chain, fork) == 2
    assert bnm.true_positives(chain, fork) == 1
    assert bnm.false_positives(chain, fork) == 1
    assert bnm.false_negatives(chain, fork) == 1
    assert bnm.precision(chain, fork) == 0.5
    assert bnm.recall(chain, fork) == 0.5
    assert bnm.f1(chain, fork) == 0.5


def test_chain_vs_collider() -> None:
    """A→B→C vs A→C, B→C. Shared: nothing.
    chain: {(A→B), (B→C)}; collider: {(A→C), (B→C)}.
    Adjacency overlap: (B,C) is shared. (A,B) only in chain, (A,C) only in collider.
    On (B,C): both directed B→C → match (TP=1).
    additions = 1, deletions = 1.
    """
    chain = chain_3()
    coll = collider_3()
    assert bnm.count_additions(chain, coll) == 1  # (A,C) in coll only
    assert bnm.count_deletions(chain, coll) == 1  # (A,B) in chain only
    assert bnm.count_reversals(chain, coll) == 0  # B→C matches B→C
    assert bnm.true_positives(chain, coll) == 1


def test_directed_reversal_counted_as_reversal() -> None:
    """A→B vs B→A. SHD=1 (single reversal, no add/delete)."""
    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = make_dag(2, [(1, 0)], var_names=("A", "B"))
    assert bnm.count_additions(g1, g2) == 0
    assert bnm.count_deletions(g1, g2) == 0
    assert bnm.count_reversals(g1, g2) == 1
    assert bnm.shd(g1, g2) == 1
    assert bnm.hd(g1, g2) == 0
    assert bnm.true_positives(g1, g2) == 0


def test_directed_to_undirected_is_reversal() -> None:
    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = make_cpdag(2, [], [(0, 1)], var_names=("A", "B"))
    assert bnm.count_reversals(g1, g2) == 1
    assert bnm.shd(g1, g2) == 1
    assert bnm.hd(g1, g2) == 0
    assert bnm.true_positives(g1, g2) == 0


def test_undirected_to_directed_is_reversal() -> None:
    g1 = make_cpdag(2, [], [(0, 1)], var_names=("A", "B"))
    g2 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    assert bnm.count_reversals(g1, g2) == 1


def test_undirected_match_is_tp() -> None:
    """Both have the same undirected edge → TP, no reversal."""
    g1 = make_cpdag(2, [], [(0, 1)], var_names=("A", "B"))
    g2 = make_cpdag(2, [], [(0, 1)], var_names=("A", "B"))
    assert bnm.true_positives(g1, g2) == 1
    assert bnm.count_reversals(g1, g2) == 0
    assert bnm.shd(g1, g2) == 0


def test_n_vars_mismatch_raises_data_error() -> None:
    import pytest

    g1 = chain_3()
    g2 = make_dag(4, [(0, 1)])
    with pytest.raises(bnm.BNMDataError, match="must match"):
        bnm.shd(g1, g2)


def test_var_name_mismatch_raises_data_error() -> None:
    import pytest

    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = make_dag(2, [(0, 1)], var_names=("X", "Y"))
    with pytest.raises(bnm.BNMDataError, match="differ"):
        bnm.shd(g1, g2)


def test_precision_recall_zero_safe() -> None:
    """Empty G2 → precision and recall both well-defined as 0.0."""
    g1 = chain_3()
    g2 = make_dag(3, [], var_names=("A", "B", "C"))
    assert bnm.precision(g1, g2) == 0.0
    assert bnm.recall(g1, g2) == 0.0
    assert bnm.f1(g1, g2) == 0.0


def test_bidirected_match_is_tp() -> None:
    """v0.2 generalisation: two graphs with bidirected A↔B.

    Built directly via int8 matrix because the nx.DiGraph adapter
    rejects bidirected.
    """
    import numpy as np

    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = bnm.EndpointMark.ARROW
    arr[1, 0] = bnm.EndpointMark.ARROW
    g1 = bnm.to_graphlike(arr.copy(), var_names=("A", "B"))
    g2 = bnm.to_graphlike(arr.copy(), var_names=("A", "B"))
    assert bnm.true_positives(g1, g2) == 1
    assert bnm.shd(g1, g2) == 0


def test_bidirected_vs_directed_is_reversal() -> None:
    import numpy as np

    bidir = np.zeros((2, 2), dtype=np.int8)
    bidir[0, 1] = bnm.EndpointMark.ARROW
    bidir[1, 0] = bnm.EndpointMark.ARROW

    g1 = bnm.to_graphlike(bidir, var_names=("A", "B"))
    g2 = bnm.to_graphlike([[0, 2], [1, 0]], var_names=("A", "B"))  # A→B
    assert bnm.count_reversals(g1, g2) == 1
    assert bnm.shd(g1, g2) == 1
