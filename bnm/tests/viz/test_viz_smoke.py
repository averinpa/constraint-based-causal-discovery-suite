"""Smoke tests for bnm.viz — verify rendering doesn't crash and the
returned objects have the expected shape. Skipped if the `viz` extra
isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

import bnm
from tests.fixtures import asia_8, chain_3, collider_3, fork_3, make_cpdag

graphviz = pytest.importorskip("graphviz")
go = pytest.importorskip("plotly.graph_objects")


def test_plot_graph_returns_dot_with_chain_edges() -> None:
    g = chain_3()
    dot = bnm.plot_graph(g, title="chain")
    src = dot.source
    # Two directed edges A→B and B→C
    assert "A -> B" in src
    assert "B -> C" in src
    assert "A -> C" not in src


def test_plot_graph_undirected_edge_renders_with_dir_none() -> None:
    cpdag = make_cpdag(3, [], [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    dot = bnm.plot_graph(cpdag)
    src = dot.source
    assert "dir=none" in src.replace('"', "")
    # No arrows on undirected edges (single-edge entries with no
    # arrowheadgo through dir=none).
    assert "A -> B" in src
    assert "B -> C" in src


def test_plot_graph_highlight_node_filled_pastel_green() -> None:
    g = collider_3()
    dot = bnm.plot_graph(g, highlight=["C"])
    src = dot.source
    assert "#c8e6c9" in src
    assert "fillcolor=#c8e6c9" in src.replace('"', "")


def test_plot_graph_with_int_indices() -> None:
    g = collider_3()
    dot = bnm.plot_graph(g, highlight=[2])
    src = dot.source
    assert "#c8e6c9" in src


def test_plot_graph_renders_svg() -> None:
    g = chain_3()
    dot = bnm.plot_graph(g)
    svg = dot.pipe(format="svg").decode("utf-8")
    assert "<svg" in svg


def test_plot_side_by_side_returns_two_dots() -> None:
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g, name1="A", name2="B")
    assert "A -> B" in dot1.source
    assert "A -> B" in dot2.source
    # Self-comparison: every edge is a TP → all painted pastel red.
    assert "#f08080" in dot1.source
    assert "#f08080" in dot2.source


def test_plot_side_by_side_mode_none_skips_edge_highlighting() -> None:
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g, mode="none")
    assert "#f08080" not in dot1.source
    assert "#f08080" not in dot2.source


def test_plot_side_by_side_highlight_node_in_both() -> None:
    g1 = chain_3()
    g2 = fork_3()
    dot1, dot2 = bnm.plot_side_by_side(g1, g2, highlight_nodes=["A"])
    assert "#c8e6c9" in dot1.source
    assert "#c8e6c9" in dot2.source


def test_plot_side_by_side_n_vars_mismatch_raises() -> None:
    g1 = chain_3()
    g2 = asia_8()
    with pytest.raises(bnm.BNMDataError, match="variables"):
        bnm.plot_side_by_side(g1, g2)


def test_plot_sid_matrix_returns_plotly_figure() -> None:
    result = bnm.sid(chain_3(), chain_3())
    fig = bnm.plot_sid_matrix(result)
    assert hasattr(fig, "data")
    # Single Heatmap trace.
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_plot_sid_matrix_uses_var_names() -> None:
    result = bnm.sid(chain_3(), chain_3())
    fig = bnm.plot_sid_matrix(result, var_names=("X", "Y", "Z"))
    assert list(fig.data[0].x) == ["X", "Y", "Z"]


def test_plot_sid_matrix_title_default() -> None:
    truth = chain_3()
    cpdag = make_cpdag(3, [], [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    result = bnm.sid(truth, cpdag)
    fig = bnm.plot_sid_matrix(result)
    assert "SID:" in fig.layout.title.text


def test_bidirected_edge_renders_with_dir_both() -> None:
    """v0.2 supports bidirected via int8 input."""
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = bnm.EndpointMark.ARROW
    arr[1, 0] = bnm.EndpointMark.ARROW
    g = bnm.to_graphlike(arr, var_names=("A", "B"))
    dot = bnm.plot_graph(g)
    src = dot.source
    assert "dir=both" in src.replace('"', "")


# ---- subtle style + direction parameter + PAG endpoint marks ---------


def test_subtle_style_applies_graph_attrs() -> None:
    """Default ``style='subtle'`` sets ellipse + light grey fill +
    Helvetica + thin border."""
    g = chain_3()
    dot = bnm.plot_graph(g)
    src = dot.source.replace('"', "")
    assert "rankdir=TB" in src
    assert "bgcolor=white" in src
    assert "shape=ellipse" in src
    assert "fontname=Helvetica" in src
    # Subtle's distinctive node fill:
    assert "style=filled" in src
    assert "fillcolor=#f8f8f8" in src
    assert "color=#888888" in src


def test_subtle_style_explicit() -> None:
    g = chain_3()
    dot = bnm.plot_graph(g, style="subtle")
    assert "shape=ellipse" in dot.source


def test_unknown_style_raises() -> None:
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="unknown viz style"):
        bnm.plot_graph(g, style="not-a-style")  # type: ignore[arg-type]


def test_subtle_style_in_side_by_side() -> None:
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g, style="subtle")
    assert "shape=ellipse" in dot1.source
    assert "shape=ellipse" in dot2.source


def test_highlight_overrides_style_node_fill() -> None:
    """Highlighting paints the node pastel green, overriding the
    style's default fill."""
    g = collider_3()
    dot = bnm.plot_graph(g, highlight=["C"])
    src = dot.source
    assert "#c8e6c9" in src
    assert "shape=ellipse" in src  # style is still applied to other nodes


# ---- direction parameter --------------------------------------------


def test_direction_TB_default() -> None:
    g = chain_3()
    dot = bnm.plot_graph(g)
    assert "rankdir=TB" in dot.source


def test_direction_LR_explicit() -> None:
    g = chain_3()
    dot = bnm.plot_graph(g, direction="LR")
    assert "rankdir=LR" in dot.source


def test_direction_auto_picks_LR_for_chain() -> None:
    """A 3-node chain has avg breadth = 1.0, so auto → LR."""
    g = chain_3()
    dot = bnm.plot_graph(g, direction="auto")
    assert "rankdir=LR" in dot.source


def test_direction_auto_picks_TB_for_asia() -> None:
    """ASIA has 8 nodes, layers = 4 (asia→tub→either→dysp), so
    avg breadth = 2.0 → auto → TB."""
    g = asia_8()
    dot = bnm.plot_graph(g, direction="auto")
    assert "rankdir=TB" in dot.source


def test_direction_invalid_raises() -> None:
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="unknown viz direction"):
        bnm.plot_graph(g, direction="diagonal")  # type: ignore[arg-type]


def test_direction_propagates_to_side_by_side() -> None:
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g, direction="LR")
    assert "rankdir=LR" in dot1.source
    assert "rankdir=LR" in dot2.source


def test_direction_auto_on_empty_graph() -> None:
    arr = np.zeros((0, 0), dtype=np.int8)
    g = bnm.to_graphlike(arr)
    dot = bnm.plot_graph(g, direction="auto")
    # No-op edge-case: falls back to TB.
    assert "rankdir=TB" in dot.source


# ---- save= parameter -------------------------------------------------


def test_plot_graph_save_svg(tmp_path) -> None:
    out = tmp_path / "g.svg"
    g = chain_3()
    dot = bnm.plot_graph(g, save=out)
    assert out.exists()
    content = out.read_text()
    assert "<svg" in content
    # Function still returns the dot for inline display.
    assert "rankdir=TB" in dot.source


def test_plot_graph_save_png(tmp_path) -> None:
    out = tmp_path / "g.png"
    bnm.plot_graph(chain_3(), save=out)
    assert out.exists()
    # PNG magic bytes.
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_graph_save_pdf(tmp_path) -> None:
    out = tmp_path / "g.pdf"
    bnm.plot_graph(chain_3(), save=out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_plot_graph_save_dot_source(tmp_path) -> None:
    out = tmp_path / "g.dot"
    bnm.plot_graph(chain_3(), save=out)
    src = out.read_text()
    # Raw DOT source should mention rankdir + node names.
    assert "rankdir=TB" in src
    assert '"A"' in src or "A " in src


def test_plot_graph_save_creates_parent_dirs(tmp_path) -> None:
    out = tmp_path / "nested" / "subdir" / "g.svg"
    bnm.plot_graph(chain_3(), save=out)
    assert out.exists()


def test_plot_graph_save_unknown_extension_raises(tmp_path) -> None:
    out = tmp_path / "g.tiff"
    with pytest.raises(bnm.BNMInputError, match="unsupported save format"):
        bnm.plot_graph(chain_3(), save=out)


def test_plot_graph_save_no_extension_raises(tmp_path) -> None:
    out = tmp_path / "g_no_ext"
    with pytest.raises(bnm.BNMInputError, match="no extension"):
        bnm.plot_graph(chain_3(), save=out)


def test_plot_graph_save_accepts_string_path(tmp_path) -> None:
    out = tmp_path / "g.svg"
    bnm.plot_graph(chain_3(), save=str(out))
    assert out.exists()


# ---- save= for plot_side_by_side ------------------------------------


def test_side_by_side_save_single_path_derives_two_files(tmp_path) -> None:
    out = tmp_path / "compare.svg"
    g = chain_3()
    bnm.plot_side_by_side(g, g, name1="truth", name2="estimate", save=out)
    assert (tmp_path / "compare_truth.svg").exists()
    assert (tmp_path / "compare_estimate.svg").exists()


def test_side_by_side_save_tuple_uses_explicit_paths(tmp_path) -> None:
    p1 = tmp_path / "left.svg"
    p2 = tmp_path / "right.png"
    bnm.plot_side_by_side(chain_3(), chain_3(), save=(p1, p2))
    assert p1.exists()
    assert p2.exists()
    assert p2.read_bytes().startswith(b"\x89PNG")


def test_side_by_side_save_tuple_wrong_length_raises(tmp_path) -> None:
    p1 = tmp_path / "x.svg"
    with pytest.raises(bnm.BNMInputError, match="exactly two entries"):
        bnm.plot_side_by_side(chain_3(), chain_3(), save=(p1,))


# ---- save= for plot_sid_matrix --------------------------------------


def test_sid_matrix_save_html(tmp_path) -> None:
    out = tmp_path / "sid.html"
    result = bnm.sid(chain_3(), chain_3())
    bnm.plot_sid_matrix(result, save=out)
    assert out.exists()
    content = out.read_text()
    assert "plotly" in content.lower()


def test_sid_matrix_save_no_extension_raises(tmp_path) -> None:
    out = tmp_path / "sid"
    result = bnm.sid(chain_3(), chain_3())
    with pytest.raises(bnm.BNMInputError, match="no extension"):
        bnm.plot_sid_matrix(result, save=out)


def test_sid_matrix_save_unsupported_format_raises(tmp_path) -> None:
    out = tmp_path / "sid.tiff"
    result = bnm.sid(chain_3(), chain_3())
    with pytest.raises(bnm.BNMInputError, match="unsupported save format"):
        bnm.plot_sid_matrix(result, save=out)


def test_sid_matrix_save_static_image_requires_kaleido(tmp_path) -> None:
    """Static image formats route through fig.write_image which needs
    kaleido. If kaleido isn't installed (the common case for `viz`-
    extra-only installs), we surface a helpful error mentioning it.
    """
    pytest.importorskip("plotly.graph_objects")
    try:
        import kaleido  # noqa: F401, PLC0415

        kaleido_installed = True
    except ImportError:
        kaleido_installed = False

    out = tmp_path / "sid.png"
    result = bnm.sid(chain_3(), chain_3())
    if kaleido_installed:
        bnm.plot_sid_matrix(result, save=out)
        assert out.exists()
    else:
        with pytest.raises(bnm.BNMError, match="kaleido"):
            bnm.plot_sid_matrix(result, save=out)


# ---- PAG endpoint marks (CIRCLE) -------------------------------------


def _pag_three_node_fixture():
    """A small 3-node PAG: A o-> B (CIRCLE at A, ARROW at B), B o-o C
    (CIRCLE at both ends). Used to verify graphviz arrowhead/arrowtail
    rendering for the PAG convention."""
    arr = np.zeros((3, 3), dtype=np.int8)
    # A o-> B  ⇒ at B (mark on j=1) is ARROW; at A (mark on i=0) is CIRCLE
    arr[0, 1] = bnm.EndpointMark.ARROW
    arr[1, 0] = bnm.EndpointMark.CIRCLE
    # B o-o C  ⇒ both ends CIRCLE
    arr[1, 2] = bnm.EndpointMark.CIRCLE
    arr[2, 1] = bnm.EndpointMark.CIRCLE
    return bnm.to_graphlike(arr, var_names=("A", "B", "C"))


def test_pag_arrow_circle_mark_renders_with_normal_and_odot() -> None:
    """A o-> B should render as edge A→B with arrowhead=normal,
    arrowtail=odot, dir=both."""
    g = _pag_three_node_fixture()
    dot = bnm.plot_graph(g)
    src = dot.source.replace('"', "")
    # The CIRCLE-bearing edge takes the explicit-arrowhead branch.
    assert "arrowhead=normal" in src
    assert "arrowtail=odot" in src
    assert "dir=both" in src


def test_pag_circle_circle_mark_renders_with_two_odots() -> None:
    """B o-o C should render with arrowhead=odot, arrowtail=odot, dir=both."""
    g = _pag_three_node_fixture()
    dot = bnm.plot_graph(g)
    src = dot.source.replace('"', "")
    # Both ends circles → both arrowhead and arrowtail are odot.
    # Count the "odot" occurrences across both edges in the fixture:
    # edge A o-> B contributes 1 odot (arrowtail), edge B o-o C
    # contributes 2 odots. Total = 3.
    assert src.count("odot") == 3


def test_pag_render_round_trips_to_svg() -> None:
    g = _pag_three_node_fixture()
    dot = bnm.plot_graph(g)
    svg = dot.pipe(format="svg").decode("utf-8")
    assert "<svg" in svg
