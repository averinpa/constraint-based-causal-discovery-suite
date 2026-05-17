"""Plotly heatmap of an SIDResult's incorrect_mat."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bnmetrics.exceptions import BNMError, BNMInputError

PathLike = str | os.PathLike[str]

# Format buckets for plotly. ``html`` always works (writes a self-
# contained HTML doc). The image formats route through
# `Figure.write_image()`, which requires ``kaleido`` to be installed.
_PLOTLY_HTML = frozenset({"html"})
_PLOTLY_IMAGE = frozenset({"png", "jpg", "jpeg", "svg", "pdf", "webp"})


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise BNMError(
            "bnmetrics.viz.plot_sid_matrix requires the `viz` extra. Install with "
            "`pip install bnmetrics[viz]` (adds graphviz, plotly, ipython)."
        ) from exc
    return go


def _save_figure(fig: Any, path: PathLike) -> None:
    """Persist a plotly figure to disk; format inferred from extension.

    HTML output works out of the box; static-image formats (png, svg,
    pdf, jpg, jpeg, webp) require ``kaleido`` (`pip install kaleido`).
    """
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower()
    if not fmt:
        raise BNMInputError(
            f"save path {p!s} has no extension; cannot infer format. "
            f"Use 'html' or one of {sorted(_PLOTLY_IMAGE)}."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt in _PLOTLY_HTML:
        fig.write_html(str(p))
        return
    if fmt in _PLOTLY_IMAGE:
        try:
            fig.write_image(str(p), format=fmt)
        except (ValueError, ImportError) as exc:
            raise BNMError(
                f"saving plotly figure to {fmt!r} requires kaleido. "
                f"Install with `pip install kaleido`, or save to .html instead."
            ) from exc
        return
    raise BNMInputError(
        f"unsupported save format {fmt!r} for plotly output; "
        f"supported: html, {sorted(_PLOTLY_IMAGE)}"
    )


def plot_sid_matrix(
    result: Any,
    *,
    var_names: Sequence[str] | None = None,
    title: str | None = None,
    save: PathLike | None = None,
) -> Any:
    """Render `result.incorrect_mat` as a plotly heatmap.

    Args:
        result: a :class:`bnmetrics.SIDResult`.
        var_names: optional axis labels; defaults to integer indices.
        title: optional figure title; defaults to "SID: <value>".
        save: optional path. ``.html`` always works; static formats
            (``.png``, ``.svg``, ``.pdf``, ``.jpg``, ``.jpeg``,
            ``.webp``) require ``kaleido`` to be installed.

    Returns:
        plotly.graph_objects.Figure
    """
    go = _require_plotly()
    mat = result.incorrect_mat
    n = mat.shape[0]
    labels = list(var_names) if var_names is not None else [str(i) for i in range(n)]
    fig_title = title if title is not None else f"SID: {int(result.sid)}"

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            showscale=False,
            xgap=1,
            ygap=1,
            zmin=0,
            zmax=1,
            colorscale=[[0.0, "white"], [0.9999, "white"], [1.0, "crimson"]],
        )
    )
    fig.update_layout(
        title=fig_title,
        xaxis={"showgrid": False, "tickangle": 90, "side": "top"},
        yaxis={"showgrid": False, "autorange": "reversed"},
        width=500,
        height=500,
    )
    if save is not None:
        _save_figure(fig, save)
    return fig
