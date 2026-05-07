"""Smoke tests for `bnm.compare_models_descriptive` and
`bnm.compare_models_comparative`."""

from __future__ import annotations

import numpy as np
import pytest

import bnm
from tests.fixtures import asia_8, chain_3, collider_3, fork_3, make_dag

pytest.importorskip("plotly.graph_objects")
pytest.importorskip("plotly.subplots")


# ---- compare_models_descriptive --------------------------------------


def _three_models() -> tuple[list[object], list[str]]:
    """A trio of small DAGs with shared var_names so per_node works."""
    return (
        [chain_3(), fork_3(), collider_3()],
        ["chain", "fork", "collider"],
    )


def test_descriptive_returns_plotly_figure() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_descriptive(graphs, names)
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")
    # No per_node → only the 'All' node, so dropdown should be empty.
    assert len(fig.layout.updatemenus) == 0


def test_descriptive_traces_one_per_metric() -> None:
    """With per_node=False there's only the 'All' node, so traces ==
    n_metrics."""
    graphs, names = _three_models()
    fig = bnm.compare_models_descriptive(graphs, names)
    n_metrics = len(bnm.DESCRIPTIVE_METRIC_NAMES)
    assert len(fig.data) == n_metrics


def test_descriptive_with_per_node_adds_dropdown() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_descriptive(graphs, names, per_node=True)
    # Dropdown should be populated with one button per node ('All' +
    # each variable).
    assert len(fig.layout.updatemenus) == 1
    button_labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "All" in button_labels
    assert "A" in button_labels
    assert "B" in button_labels
    assert "C" in button_labels


def test_descriptive_specific_metrics() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_descriptive(graphs, names, descriptive=["n_edges", "n_colliders"])
    # 2 metrics × 1 node ('All') = 2 traces
    assert len(fig.data) == 2


def test_descriptive_unknown_metric_raises() -> None:
    graphs, names = _three_models()
    with pytest.raises(bnm.BNMInputError, match="unknown descriptive metric"):
        bnm.compare_models_descriptive(graphs, names, descriptive=["not-a-metric"])


def test_descriptive_length_mismatch_raises() -> None:
    graphs, _ = _three_models()
    with pytest.raises(bnm.BNMInputError, match="model_names"):
        bnm.compare_models_descriptive(graphs, ["only_two", "labels"])


def test_descriptive_empty_input_raises() -> None:
    with pytest.raises(bnm.BNMInputError, match="at least one"):
        bnm.compare_models_descriptive([], [])


def test_descriptive_x_axis_uses_model_names() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_descriptive(graphs, names, descriptive=["n_edges"])
    # x-values of the single 'All' trace are the model names in order.
    assert list(fig.data[0].x) == names


def test_descriptive_save_html(tmp_path) -> None:
    graphs, names = _three_models()
    out = tmp_path / "models.html"
    bnm.compare_models_descriptive(graphs, names, save=out)
    assert out.exists()
    assert "plotly" in out.read_text().lower()


def test_descriptive_works_on_index_only_graphs() -> None:
    """Graphs without var_names: per_node still works, keys are
    stringified ints."""
    g1 = make_dag(3, [(0, 1), (1, 2)])
    g2 = make_dag(3, [(0, 1)])
    fig = bnm.compare_models_descriptive(
        [g1, g2], ["chain", "shorter"], per_node=True, descriptive=["n_edges"]
    )
    button_labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "All" in button_labels
    # node keys appear as "0", "1", "2"
    assert "0" in button_labels


# ---- compare_models_comparative -------------------------------------


def test_comparative_returns_plotly_heatmap() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd")
    assert hasattr(fig, "data")
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_comparative_z_matrix_is_n_by_n() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd")
    z = fig.data[0].z
    assert len(z) == 3
    assert all(len(row) == 3 for row in z)


def test_comparative_diagonal_is_zero_for_self_pairs() -> None:
    """Pairing g_i vs g_i has SHD = 0 on the diagonal."""
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd")
    z = fig.data[0].z
    for i in range(3):
        assert z[i][i] == 0


def test_comparative_chain_vs_fork_matches_pairwise_shd() -> None:
    """The off-diagonal entries should equal direct bnm.shd() calls."""
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd")
    z = fig.data[0].z
    # z[j][i] = shd(g1=graphs[i], g2=graphs[j])
    expected_chain_vs_fork = bnm.shd(graphs[0], graphs[1])  # i=0 chain, j=1 fork
    assert z[1][0] == expected_chain_vs_fork


def test_comparative_with_per_node_dropdown() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd", per_node=True)
    assert len(fig.layout.updatemenus) == 1
    button_labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "All" in button_labels
    assert "A" in button_labels


def test_comparative_with_per_node_traces_one_per_node() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd", per_node=True)
    # 4 nodes ('All' + A + B + C) → 4 heatmap traces (only first visible)
    assert len(fig.data) == 4
    visibilities = [t.visible for t in fig.data]
    assert visibilities[0] is True
    assert visibilities[1] is False


def test_comparative_unknown_metric_raises() -> None:
    graphs, names = _three_models()
    with pytest.raises(bnm.BNMInputError, match="unknown comparative metric"):
        bnm.compare_models_comparative(graphs, names, metric="not-a-metric")


def test_comparative_length_mismatch_raises() -> None:
    graphs, _ = _three_models()
    with pytest.raises(bnm.BNMInputError, match="model_names"):
        bnm.compare_models_comparative(graphs, ["only_two", "labels"])


def test_comparative_too_few_models_raises() -> None:
    with pytest.raises(bnm.BNMInputError, match="at least two"):
        bnm.compare_models_comparative([chain_3()], ["only"])


def test_comparative_save_html(tmp_path) -> None:
    graphs, names = _three_models()
    out = tmp_path / "comp.html"
    bnm.compare_models_comparative(graphs, names, metric="shd", save=out)
    assert out.exists()


def test_comparative_default_metric_is_shd() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names)
    # Single trace with default metric.
    assert len(fig.data) == 1
    assert "shd" in fig.layout.title.text.lower()


def test_comparative_works_with_f1() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="f1")
    z = fig.data[0].z
    # Self-pair → F1 = 1.0
    for i in range(3):
        assert z[i][i] == 1.0


def test_comparative_heatmap_axes_use_model_names() -> None:
    graphs, names = _three_models()
    fig = bnm.compare_models_comparative(graphs, names, metric="shd")
    assert list(fig.data[0].x) == names
    assert list(fig.data[0].y) == names


# ---- end-to-end: realistic use case -------------------------------


def test_descriptive_realistic_asia_alpha_sweep() -> None:
    """Simulate 'compare PC alpha values' workflow: 3 estimated DAGs
    (variations of asia) compared on descriptive metrics."""
    g_truth = asia_8()
    # Simulate 3 estimates with progressively dropped edges.
    arr_truth = np.array(g_truth.endpoints)
    g_est1 = bnm.to_graphlike(arr_truth, var_names=g_truth.var_names)
    arr2 = arr_truth.copy()
    arr2[0, 1] = 0  # drop asia → tub
    arr2[1, 0] = 0
    g_est2 = bnm.to_graphlike(arr2, var_names=g_truth.var_names)
    arr3 = arr2.copy()
    arr3[2, 3] = 0  # drop smoke → lung
    arr3[3, 2] = 0
    g_est3 = bnm.to_graphlike(arr3, var_names=g_truth.var_names)

    fig = bnm.compare_models_descriptive(
        [g_est1, g_est2, g_est3],
        ["full", "drop_asia_tub", "drop_smoke_lung_too"],
        descriptive=["n_edges", "n_colliders"],
    )
    # 2 metrics, single 'All' node → 2 traces.
    assert len(fig.data) == 2
    # n_edges trace: should be [8, 7, 6] across the three models.
    n_edges_trace = fig.data[0]
    assert list(n_edges_trace.y) == [8, 7, 6]


def test_comparative_realistic_asia_alpha_sweep() -> None:
    g_truth = asia_8()
    arr_truth = np.array(g_truth.endpoints)
    g_est1 = bnm.to_graphlike(arr_truth, var_names=g_truth.var_names)
    arr2 = arr_truth.copy()
    arr2[0, 1] = 0
    arr2[1, 0] = 0
    g_est2 = bnm.to_graphlike(arr2, var_names=g_truth.var_names)

    fig = bnm.compare_models_comparative(
        [g_truth, g_est1, g_est2],
        ["truth", "full", "drop_one"],
        metric="shd",
    )
    z = fig.data[0].z
    # truth vs full → 0; truth vs drop_one → 1; full vs drop_one → 1
    assert z[0][0] == 0  # truth vs truth
    assert z[1][0] == 0  # truth vs full
    assert z[2][0] == 1  # truth vs drop_one
