"""Multi-model comparison plots.

Two top-level entry points:

- :func:`compare_models_descriptive` — for a list of model DAGs/CPDAGs
  with a name each, compute every requested descriptive metric on
  every model and render a subplot grid (one panel per metric, x-axis
  = model name, y-axis = metric value). When ``per_node`` is set, a
  Plotly dropdown lets the reader switch between whole-graph and
  per-Markov-blanket views.

- :func:`compare_models_comparative` — for the same input, compute
  one comparative metric across **all pairs** of models and render
  the resulting (n_models × n_models) heatmap. Same per-node dropdown
  shape.

Both delegate the metric computation to :func:`bnmetrics.compare`. Plotly
is lazy-imported (the ``viz`` extra). Static-image ``save=`` formats
require ``kaleido``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from bnmetrics.compare import (
    COMPARATIVE_METRIC_NAMES,
    DESCRIPTIVE_METRIC_NAMES,
    compare,
)
from bnmetrics.exceptions import BNMError, BNMInputError
from bnmetrics.viz._sid_matrix import _save_figure

PathLike = str | os.PathLike[str]


def _require_plotly_subplots() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415
    except ImportError as exc:
        raise BNMError(
            "bnmetrics.viz multi-model plots require the `viz` extra. Install with "
            "`pip install bnmetrics[viz]` (adds graphviz, plotly, ipython)."
        ) from exc
    return go, make_subplots


# ---- compare_models_descriptive --------------------------------------


def _resolve_descriptive(
    spec: Iterable[str] | Literal["all"],
) -> tuple[str, ...]:
    if spec == "all":
        return DESCRIPTIVE_METRIC_NAMES
    out = tuple(spec)
    bad = set(out) - set(DESCRIPTIVE_METRIC_NAMES)
    if bad:
        raise BNMInputError(
            f"compare_models_descriptive: unknown descriptive metric(s) "
            f"{sorted(bad)}; available: {list(DESCRIPTIVE_METRIC_NAMES)}"
        )
    return out


def _stringify_node_key(key: str | int) -> str:
    return key if isinstance(key, str) else str(key)


def compare_models_descriptive(
    graphs: Sequence[object],
    model_names: Sequence[str],
    *,
    descriptive: Iterable[str] | Literal["all"] = "all",
    per_node: bool | Iterable[int | str] = False,
    title: str | None = None,
    cols: int = 3,
    save: PathLike | None = None,
) -> Any:
    """Plot descriptive metrics across multiple models.

    Args:
        graphs: ordered list of GraphLikeInputs (cbcd graphs,
            ndarrays, lists, nx.DiGraph instances).
        model_names: ordered list of model labels matching ``graphs``;
            used as the x-axis tick labels in each subplot.
        descriptive: which descriptive metrics to plot — an iterable
            of names from :data:`bnmetrics.DESCRIPTIVE_METRIC_NAMES`, or
            the literal ``"all"`` (default).
        per_node: ``True`` to also compute per-Markov-blanket
            sub-results and expose a dropdown to switch between nodes;
            an iterable of variable handles for a subset; ``False``
            (default) for whole-graph only.
        title: figure title; defaults to ``"Descriptive metrics"``.
        cols: number of subplot columns; rows derived from metric count.
        save: optional path. ``.html`` always works; static formats
            (``.png``, ``.svg``, ``.pdf``, etc.) require ``kaleido``.

    Returns:
        plotly.graph_objects.Figure
    """
    if len(graphs) != len(model_names):
        raise BNMInputError(
            f"compare_models_descriptive: graphs has {len(graphs)} entries "
            f"but model_names has {len(model_names)}"
        )
    if not graphs:
        raise BNMInputError("compare_models_descriptive: at least one graph required")

    metric_names = _resolve_descriptive(descriptive)

    # Compute every metric on every model.
    # rows: {"model_name": str, "node_name": str, <metric>: float, ...}
    rows: list[dict[str, Any]] = []
    for g, mname in zip(graphs, model_names, strict=True):
        c = compare(g, descriptive=metric_names, per_node=per_node)
        rows.append({"model_name": mname, "node_name": "All", **c.g1_descriptive})
        if c.per_node is not None:
            for node_key, metrics in c.per_node.items():
                rows.append(
                    {
                        "model_name": mname,
                        "node_name": _stringify_node_key(node_key),
                        **metrics,
                    }
                )

    # Distinct nodes with 'All' first.
    nodes_seen: list[str] = []
    seen: set[str] = set()
    for r in rows:
        n = r["node_name"]
        if n not in seen:
            seen.add(n)
            nodes_seen.append(n)
    nodes_seen.sort(key=lambda n: (0 if n == "All" else 1, n))

    go, make_subplots = _require_plotly_subplots()
    n_metrics = len(metric_names)
    n_rows = (n_metrics + cols - 1) // cols

    fig = make_subplots(
        rows=n_rows,
        cols=cols,
        subplot_titles=list(metric_names),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # For each metric panel, add one trace per node; visibility starts
    # at the first node ('All' sorts first). Order: panel-major, then
    # node — matches 0.1.x's traces-per-panel layout for stable
    # dropdown bookkeeping.
    default_node = nodes_seen[0]
    for i, metric in enumerate(metric_names):
        panel_row = i // cols + 1
        panel_col = i % cols + 1
        for node in nodes_seen:
            xs: list[str] = []
            ys: list[float] = []
            for r in rows:
                if r["node_name"] == node and metric in r:
                    xs.append(r["model_name"])
                    ys.append(r[metric])
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=node,
                    marker={"color": "#1E3A8A", "symbol": "diamond"},
                    line={"color": "#1E3A8A"},
                    visible=(node == default_node),
                    showlegend=False,
                ),
                row=panel_row,
                col=panel_col,
            )

    # Dropdown: one button per node. Each button toggles the visibility
    # of the (n_metrics) traces assigned to that node (one per panel).
    n_total = n_metrics * len(nodes_seen)
    buttons: list[dict[str, Any]] = []
    for node in nodes_seen:
        visibility = [False] * n_total
        for i in range(n_metrics):
            trace_idx = i * len(nodes_seen) + nodes_seen.index(node)
            visibility[trace_idx] = True
        buttons.append(
            {
                "label": node,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"{title or 'Descriptive metrics'} — Node: {node}"},
                ],
            }
        )

    fig.update_layout(
        height=max(300, 250 * n_rows),
        width=max(800, 350 * cols),
        title=f"{title or 'Descriptive metrics'} — Node: {default_node}",
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
        updatemenus=[
            {
                "active": 0,
                "buttons": buttons,
                "direction": "down",
                "x": 1.01,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ]
        if len(nodes_seen) > 1
        else [],
    )
    fig.update_xaxes(tickangle=45)

    if save is not None:
        _save_figure(fig, save)
    return fig


# ---- compare_models_comparative --------------------------------------


def _resolve_comparative_metric(metric: str) -> str:
    if metric not in COMPARATIVE_METRIC_NAMES:
        raise BNMInputError(
            f"compare_models_comparative: unknown comparative metric "
            f"{metric!r}; available: {list(COMPARATIVE_METRIC_NAMES)}"
        )
    return metric


def compare_models_comparative(
    graphs: Sequence[object],
    model_names: Sequence[str],
    *,
    metric: str = "shd",
    per_node: bool | Iterable[int | str] = False,
    title: str | None = None,
    save: PathLike | None = None,
) -> Any:
    """Plot a single comparative metric across all pairs of models.

    Args:
        graphs: ordered list of GraphLikeInputs.
        model_names: matching list of labels.
        metric: which comparative metric to display. Default ``"shd"``.
            Must be one of :data:`bnmetrics.COMPARATIVE_METRIC_NAMES`.
        per_node: when set, expose a Plotly dropdown to switch
            between whole-graph and per-Markov-blanket heatmaps.
        title: figure title; defaults to ``"<metric> heatmap"``.
        save: optional path; same conventions as
            :func:`compare_models_descriptive`.

    Returns:
        plotly.graph_objects.Figure with one heatmap (per-node dropdown
        when applicable). Each heatmap is ``(n_models × n_models)``;
        ``z[j, i]`` is the metric of ``g1=graphs[i]`` vs
        ``g2=graphs[j]``.
    """
    metric = _resolve_comparative_metric(metric)
    n = len(graphs)
    if n != len(model_names):
        raise BNMInputError(
            f"compare_models_comparative: graphs has {n} entries "
            f"but model_names has {len(model_names)}"
        )
    if n < 2:
        raise BNMInputError("compare_models_comparative: at least two graphs required")

    # First pass: figure out which node names are involved (when
    # per_node is requested). We need to compute one comparison to
    # discover the per-node keys; pick (g0, g0).
    if per_node:
        sentinel = compare(graphs[0], graphs[0], comparative=[metric], per_node=per_node)
        per_node_keys: list[str] = ["All"] + [
            _stringify_node_key(k) for k in (sentinel.per_node or {})
        ]
    else:
        per_node_keys = ["All"]

    # Second pass: full pairwise sweep, collect z-matrices per node.
    # mats[node]: (n, n) list of lists, mats[node][j][i] = metric(g_i, g_j).
    mats: dict[str, list[list[float | None]]] = {
        node: [[None] * n for _ in range(n)] for node in per_node_keys
    }
    for i in range(n):
        for j in range(n):
            c = compare(
                graphs[i],
                graphs[j],
                descriptive=None,
                comparative=[metric],
                per_node=per_node,
            )
            assert c.comparative is not None
            mats["All"][j][i] = float(c.comparative[metric])
            if c.per_node is not None:
                for node_key, metrics in c.per_node.items():
                    nk = _stringify_node_key(node_key)
                    if metric in metrics:
                        mats[nk][j][i] = float(metrics[metric])

    go, _ = _require_plotly_subplots()

    # One Heatmap trace per node; first node visible, others hidden.
    traces: list[Any] = []
    default_node = per_node_keys[0]
    fig_title = title or f"{metric} heatmap"
    for node in per_node_keys:
        z = mats[node]
        # Compute z-range only over non-None entries.
        flat = [v for row in z for v in row if v is not None]
        zmin = min(flat) if flat else 0
        zmax = max(flat) if flat else 1
        text = [[f"{v:.2f}" if v is not None else "" for v in row] for row in z]
        traces.append(
            go.Heatmap(
                z=z,
                x=list(model_names),
                y=list(model_names),
                colorscale="Blues",
                colorbar={"title": metric},
                visible=(node == default_node),
                text=text,
                texttemplate="%{text}",
                textfont={"size": 9},
                zmin=zmin,
                zmax=zmax,
            )
        )

    fig = go.Figure(data=traces)

    buttons: list[dict[str, Any]] = []
    for i, node in enumerate(per_node_keys):
        visibility = [k == i for k in range(len(per_node_keys))]
        buttons.append(
            {
                "label": node,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"{fig_title} — Node: {node}"},
                ],
            }
        )

    fig.update_layout(
        title=f"{fig_title} — Node: {default_node}",
        xaxis_title="g1 (column)",
        yaxis_title="g2 (row)",
        updatemenus=[
            {
                "active": 0,
                "buttons": buttons,
                "direction": "down",
                "x": 1.02,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ]
        if len(per_node_keys) > 1
        else [],
        width=max(700, 80 * n + 200),
        height=max(600, 80 * n + 100),
    )

    if save is not None:
        _save_figure(fig, save)
    return fig


# Re-export under bnmetrics.viz.
__all__ = [
    "compare_models_descriptive",
    "compare_models_comparative",
]
