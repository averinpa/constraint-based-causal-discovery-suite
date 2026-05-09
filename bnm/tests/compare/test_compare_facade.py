"""Hand-computed checks for `bnm.compare` and `bnm.to_dataframe`."""

from __future__ import annotations

import pytest

import bnm
from tests.fixtures import asia_8, chain_3, collider_3, fork_3, make_dag

# ---- single-graph mode (g2=None) ------------------------------------


def test_compare_single_graph_descriptive_only() -> None:
    g = chain_3()
    c = bnm.compare(g)
    assert c.g1_descriptive["n_edges"] == 2
    assert c.g1_descriptive["n_colliders"] == 0
    assert c.g1_descriptive["n_root_nodes"] == 1
    assert c.g2_descriptive is None
    assert c.comparative is None
    assert c.sid is None
    assert c.per_node is None
    assert c.var_names == ("A", "B", "C")


def test_compare_single_graph_specific_metrics() -> None:
    g = chain_3()
    c = bnm.compare(g, descriptive=["n_edges", "n_colliders"])
    assert set(c.g1_descriptive.keys()) == {"n_edges", "n_colliders"}


def test_compare_single_graph_no_descriptive() -> None:
    g = chain_3()
    c = bnm.compare(g, descriptive=None)
    assert c.g1_descriptive == {}


def test_compare_single_graph_default_comparative_silently_skipped() -> None:
    """The implicit default ``comparative="all"`` is silently skipped
    when g2 is None — single-graph mode just gives what's possible
    on g1 alone."""
    g = chain_3()
    c = bnm.compare(g)
    assert c.comparative is None  # not an error


def test_compare_single_graph_explicit_comparative_list_raises() -> None:
    """An EXPLICIT comparative list with no g2 is a caller error."""
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="g2 is None"):
        bnm.compare(g, comparative=["shd", "f1"])


def test_compare_single_graph_sid_requires_g2() -> None:
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="g2 is None"):
        bnm.compare(g, comparative=None, include_sid=True)


# ---- two-graph mode --------------------------------------------------


def test_compare_two_graphs_self() -> None:
    g = chain_3()
    c = bnm.compare(g, g)
    assert c.g1_descriptive["n_edges"] == 2
    assert c.g2_descriptive == c.g1_descriptive
    assert c.comparative["shd"] == 0
    assert c.comparative["f1"] == 1.0
    assert c.sid is None  # not requested


def test_compare_two_graphs_with_sid() -> None:
    g = chain_3()
    c = bnm.compare(g, g, include_sid=True)
    assert isinstance(c.sid, bnm.SIDResult)
    assert c.sid.sid == 0
    assert c.sid.is_tight


def test_compare_chain_vs_fork() -> None:
    chain = chain_3()
    fork = fork_3()
    c = bnm.compare(chain, fork)
    # SHD = 2 (1 deletion, 1 addition), F1 = 0.5 (TP=1)
    assert c.comparative["shd"] == 2
    assert c.comparative["additions"] == 1
    assert c.comparative["deletions"] == 1
    assert c.comparative["tp"] == 1
    assert c.comparative["f1"] == 0.5


def test_compare_specific_comparative_metrics() -> None:
    g1 = chain_3()
    g2 = fork_3()
    c = bnm.compare(g1, g2, descriptive=None, comparative=["shd", "f1"])
    assert set(c.comparative.keys()) == {"shd", "f1"}


def test_compare_unknown_metric_raises() -> None:
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="unknown descriptive metric"):
        bnm.compare(g, descriptive=["not-a-metric"])


def test_compare_returns_frozen_dataclass() -> None:
    import dataclasses

    g = chain_3()
    c = bnm.compare(g)
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.g1_descriptive = {}  # type: ignore[misc]


# ---- per-node --------------------------------------------------------


def test_per_node_true_iterates_all_vars() -> None:
    g = chain_3()
    c = bnm.compare(g, descriptive=["n_edges"], per_node=True)
    assert c.per_node is not None
    assert set(c.per_node.keys()) == {"A", "B", "C"}


def test_per_node_iterable_subset() -> None:
    g = chain_3()
    c = bnm.compare(g, descriptive=["n_edges"], per_node=["A", "C"])
    assert set(c.per_node.keys()) == {"A", "C"}


def test_per_node_with_two_graphs_uses_g1_mb_anchor() -> None:
    """With g2 provided, per_node values use g1's MB indices for both
    g1 and g2 — yielding the `_base` suffix on g1's descriptive
    columns and unsuffixed on g2's + comparative."""
    g1 = collider_3()
    g2 = collider_3()  # same → all comparative should be exact match
    c = bnm.compare(g1, g2, per_node=True)
    assert c.per_node is not None
    a_row = c.per_node["A"]
    # g1 descriptive uses _base suffix when g2 is present.
    assert "n_edges_base" in a_row
    assert "n_edges" in a_row
    # Comparative on the MB sub-graph: same → SHD = 0.
    assert a_row["shd"] == 0
    assert a_row["f1"] == 1.0


def test_per_node_with_sid() -> None:
    g = chain_3()
    c = bnm.compare(g, g, include_sid=True, per_node=True)
    assert c.per_node is not None
    for var in ("A", "B", "C"):
        row = c.per_node[var]
        # Self-comparison → SID = 0 on every MB.
        assert row.get("sid") == 0


def test_per_node_index_keyed_when_no_var_names() -> None:
    g = make_dag(3, [(0, 1), (1, 2)])
    c = bnm.compare(g, descriptive=["n_edges"], per_node=True)
    assert c.per_node is not None
    assert set(c.per_node.keys()) == {0, 1, 2}


# ---- to_dataframe ----------------------------------------------------


def test_to_dataframe_single_graph_one_row_all() -> None:
    pytest.importorskip("pandas")
    g = chain_3()
    c = bnm.compare(g)
    df = bnm.to_dataframe(c)
    assert list(df["node_name"]) == ["All"]
    assert "n_edges" in df.columns
    assert df.loc[df["node_name"] == "All", "n_edges"].iloc[0] == 2
    # No `_base` suffix when g2 is absent.
    assert "n_edges_base" not in df.columns


def test_to_dataframe_two_graphs_uses_base_suffix() -> None:
    pytest.importorskip("pandas")
    g1 = chain_3()
    g2 = fork_3()
    c = bnm.compare(g1, g2)
    df = bnm.to_dataframe(c)
    assert "n_edges_base" in df.columns  # g1
    assert "n_edges" in df.columns  # g2
    assert "shd" in df.columns
    row = df[df["node_name"] == "All"].iloc[0]
    assert row["n_edges_base"] == 2
    assert row["n_edges"] == 2
    assert row["shd"] == 2


def test_to_dataframe_with_per_node() -> None:
    pytest.importorskip("pandas")
    g = chain_3()
    c = bnm.compare(g, g, per_node=True)
    df = bnm.to_dataframe(c)
    assert set(df["node_name"]) == {"All", "A", "B", "C"}
    # Self-comparison → all SHDs are 0
    assert (df["shd"] == 0).all()


def test_to_dataframe_with_sid() -> None:
    pytest.importorskip("pandas")
    g = chain_3()
    c = bnm.compare(g, g, include_sid=True)
    df = bnm.to_dataframe(c)
    assert "sid" in df.columns
    assert "sid_lower_bound" in df.columns
    assert "sid_upper_bound" in df.columns
    assert df["sid"].iloc[0] == 0


def test_to_dataframe_pandas_missing_raises_bnm_error(monkeypatch) -> None:
    """If pandas is somehow not importable, surface a clear error."""
    import sys

    g = chain_3()
    c = bnm.compare(g)
    # Simulate missing pandas.
    saved = sys.modules.pop("pandas", None)
    monkeypatch.setitem(sys.modules, "pandas", None)
    try:
        with pytest.raises(bnm.BNMError, match="pandas"):
            bnm.to_dataframe(c)
    finally:
        if saved is not None:
            sys.modules["pandas"] = saved
        else:
            sys.modules.pop("pandas", None)


# ---- end-to-end on ASIA ---------------------------------------------


def test_asia_full_compare() -> None:
    """Realistic case: ASIA truth vs itself with all metrics + SID +
    per-node."""
    g = asia_8()
    c = bnm.compare(g, g, include_sid=True, per_node=True)
    assert c.g1_descriptive["n_edges"] == 8
    assert c.g2_descriptive["n_edges"] == 8
    assert c.comparative["shd"] == 0
    assert c.comparative["f1"] == 1.0
    assert c.sid.sid == 0
    assert c.per_node is not None
    assert set(c.per_node.keys()) == {
        "asia",
        "tub",
        "smoke",
        "lung",
        "bronc",
        "either",
        "xray",
        "dysp",
    }
    for row in c.per_node.values():
        assert row["shd"] == 0
        assert row["f1"] == 1.0
