"""Distribution of descriptive metrics across the Markov blankets of
a single graph.

`analyse_mb(g)` is the single-graph analogue of
``compare_models_descriptive``: instead of plotting one metric per
panel across multiple models, it plots the *distribution* of each
metric across the ``n`` Markov blankets within one graph.

Useful for characterising local-structure heterogeneity — e.g. "how
many MBs in this DAG are colliders vs forks vs chains."
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

from bnmetrics.compare import DESCRIPTIVE_METRIC_NAMES, compare
from bnmetrics.exceptions import BNMInputError
from bnmetrics.viz._compare_models import _require_plotly_subplots
from bnmetrics.viz._sid_matrix import _save_figure

PathLike = str | os.PathLike[str]


def _resolve_descriptive(
    spec: Iterable[str] | Literal["all"],
) -> tuple[str, ...]:
    if spec == "all":
        return DESCRIPTIVE_METRIC_NAMES
    out = tuple(spec)
    bad = set(out) - set(DESCRIPTIVE_METRIC_NAMES)
    if bad:
        raise BNMInputError(
            f"analyse_mb: unknown descriptive metric(s) {sorted(bad)}; "
            f"available: {list(DESCRIPTIVE_METRIC_NAMES)}"
        )
    return out


def analyse_mb(
    g: object,
    *,
    descriptive: Iterable[str] | Literal["all"] = "all",
    cols: int = 4,
    title: str | None = None,
    save: PathLike | None = None,
) -> Any:
    """Plot value-count bar charts for each descriptive metric across
    every Markov blanket of ``g``.

    Args:
        g: any GraphLikeInput.
        descriptive: which descriptive metrics to plot. Default
            ``"all"`` of :data:`bnmetrics.DESCRIPTIVE_METRIC_NAMES`.
        cols: number of subplot columns; rows derived from metric count.
        title: figure title; defaults to ``"Markov-blanket-space
            distribution"``.
        save: optional path. ``.html`` always works; static formats
            need ``kaleido``.

    Returns:
        plotly.graph_objects.Figure with one bar-chart panel per
        metric. Bar height = number of variables whose MB has the
        given metric value.
    """
    metric_names = _resolve_descriptive(descriptive)

    # Compute per-MB descriptive metrics for every variable in g.
    c = compare(g, descriptive=metric_names, per_node=True)
    if not c.per_node:
        raise BNMInputError("analyse_mb: graph has no variables; nothing to plot")

    # values[metric] = list of metric values across the n MBs.
    values: dict[str, list[float]] = {m: [] for m in metric_names}
    for metrics in c.per_node.values():
        for m in metric_names:
            if m in metrics:
                values[m].append(metrics[m])

    go, make_subplots = _require_plotly_subplots()
    n_metrics = len(metric_names)
    n_rows = (n_metrics + cols - 1) // cols

    fig = make_subplots(
        rows=n_rows,
        cols=cols,
        subplot_titles=list(metric_names),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    for i, metric in enumerate(metric_names):
        row = i // cols + 1
        col = i % cols + 1
        # Value-count bar chart, sorted by value.
        counts = Counter(values[metric])
        sorted_pairs = sorted(counts.items(), key=lambda kv: kv[0])
        xs = [_format_x(v) for v, _ in sorted_pairs]
        ys = [count for _, count in sorted_pairs]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                marker={"color": "#1E3A8A"},
                name=metric,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Y-axis label on first column only; X-axis label on bottom row only.
    fig.update_yaxes(title_text="Frequency", col=1)
    fig.update_xaxes(title_text="Metric value", row=n_rows)

    fig.update_layout(
        title=title or "Markov-blanket-space distribution",
        height=max(300, 250 * n_rows),
        width=max(800, 280 * cols),
        font={"size": 11},
        margin={"t": 100},
    )

    if save is not None:
        _save_figure(fig, save)
    return fig


def _format_x(value: float) -> str:
    """Render a metric value for the bar chart's x-axis label.

    Most descriptive metrics are integers (counts), so we strip the
    trailing ``.0`` for cleaner ticks. Non-integer floats are kept
    as-is.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


__all__ = ["analyse_mb"]
