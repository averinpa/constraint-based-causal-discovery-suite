"""bnm — Bayesian network metrics.

Public API for v0.2.x. The names listed in ``__all__`` are committed
backwards-compatible across all v0.2.x and v0.3.x releases (the v0.x
API contract). Stub items (PAG viz, time-series GraphLike) are not yet
frozen until the implementing slice lands — see
``docs/design/api_v0.py`` §K for the frozen-vs-not breakdown.
"""

from __future__ import annotations

from bnm.adapter import to_graphlike
from bnm.comparative import (
    count_additions,
    count_deletions,
    count_reversals,
    f1,
    false_negatives,
    false_positives,
    hd,
    precision,
    recall,
    shd,
    true_positives,
)
from bnm.compare import (
    COMPARATIVE_METRIC_NAMES,
    DESCRIPTIVE_METRIC_NAMES,
    Comparison,
    compare,
    to_dataframe,
)
from bnm.descriptive import (
    count_bidirected_arcs,
    count_circle_edges,
    count_colliders,
    count_directed_arcs,
    count_edges,
    count_isolated_nodes,
    count_leaf_nodes,
    count_nodes,
    count_reversible_arcs,
    count_root_nodes,
    count_undirected_arcs,
    in_degree,
    out_degree,
)
from bnm.exceptions import BNMDataError, BNMError, BNMInputError
from bnm.markov_blanket import markov_blanket, markov_blanket_indices
from bnm.marks import EndpointMark
from bnm.protocol import GraphLike
from bnm.sid import SIDResult, sid
from bnm.viz import (
    analyse_mb,
    compare_models_comparative,
    compare_models_descriptive,
    plot_graph,
    plot_sid_matrix,
    plot_side_by_side,
)

__version__ = "0.2.2.dev0"

__all__ = [
    # graph contract + marks
    "EndpointMark",
    "GraphLike",
    "to_graphlike",
    # exceptions
    "BNMDataError",
    "BNMError",
    "BNMInputError",
    # descriptive
    "count_bidirected_arcs",
    "count_circle_edges",
    "count_colliders",
    "count_directed_arcs",
    "count_edges",
    "count_isolated_nodes",
    "count_leaf_nodes",
    "count_nodes",
    "count_reversible_arcs",
    "count_root_nodes",
    "count_undirected_arcs",
    "in_degree",
    "out_degree",
    # comparative
    "count_additions",
    "count_deletions",
    "count_reversals",
    "f1",
    "false_negatives",
    "false_positives",
    "hd",
    "precision",
    "recall",
    "shd",
    "true_positives",
    # markov blanket
    "markov_blanket",
    "markov_blanket_indices",
    # multi-metric comparison
    "COMPARATIVE_METRIC_NAMES",
    "Comparison",
    "DESCRIPTIVE_METRIC_NAMES",
    "compare",
    "to_dataframe",
    # sid
    "SIDResult",
    "sid",
    # viz (gated on `viz` extra; ImportError surfaces only when called)
    "analyse_mb",
    "compare_models_comparative",
    "compare_models_descriptive",
    "plot_graph",
    "plot_side_by_side",
    "plot_sid_matrix",
]
