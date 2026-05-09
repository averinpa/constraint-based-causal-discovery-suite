"""Graph + SID visualization (Slice 4).

All viz functions are gated on the ``viz`` extra (`graphviz`, `plotly`,
`ipython`). They lazy-import their backend so the metric layer stays
free of UI deps. Importing :mod:`bnm.viz` itself does NOT trigger any
optional imports — only calling a function does.

Design notes:

- Per the audit, 0.1.x's viz was tightly coupled to the ``BNMetrics``
  god-class and did its own MB-subgraph extraction. v0.2's viz takes
  any GraphLikeInput and uses :func:`bnm.markov_blanket` when the
  caller asks for an MB view; viz never owns graph state.
- Graphviz output is returned as an SVG string (for non-IPython
  callers) and additionally displayed in a Jupyter context if IPython
  is available.
- Plotly figures are returned to the caller; they choose to display
  via ``fig.show()`` or save.
"""

from __future__ import annotations

from bnm.viz._compare_models import (
    compare_models_comparative,
    compare_models_descriptive,
)
from bnm.viz._graphviz import plot_graph, plot_side_by_side
from bnm.viz._mb_distribution import analyse_mb
from bnm.viz._sid_matrix import plot_sid_matrix

__all__ = [
    "analyse_mb",
    "compare_models_comparative",
    "compare_models_descriptive",
    "plot_graph",
    "plot_side_by_side",
    "plot_sid_matrix",
]
