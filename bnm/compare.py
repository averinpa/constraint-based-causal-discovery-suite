"""Multi-metric comparison façade.

See ``docs/design/api_v0.py`` §H for the contract.

Public surface:
  - :class:`Comparison` — frozen dataclass holding every requested
    metric value at the whole-graph level (and optionally per-node).
  - :func:`compare` — the entry point. Computes any subset of
    descriptive / comparative / SID / per-Markov-blanket metrics in
    one call.
  - :func:`to_dataframe` — free function rendering a `Comparison` as
    a wide-format pandas DataFrame. Lazy-imports pandas.
  - :data:`DESCRIPTIVE_METRIC_NAMES`, :data:`COMPARATIVE_METRIC_NAMES`
    — the canonical metric-name registries.

Per-node semantics:
  - The Markov blanket of variable ``v`` in ``g1`` defines the
    sub-node-set. Both ``g1`` and (if provided) ``g2`` are restricted
    to those indices for descriptive AND comparative AND SID metrics
    on the per-node row.
  - This differs from 0.1.x slightly: 0.1.x's descriptive metrics for
    g2 were computed on g2's OWN MB(v) (different node set), but
    0.1.x's comparative and SID metrics already used the
    g1-MB-anchored restriction. v0.2 unifies on the
    g1-MB-anchored restriction so all three metric kinds are
    directly comparable on the same sub-graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bnm._graph import _Graph
from bnm.adapter import _resolve_var, _to_endpoints, to_graphlike
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
)
from bnm.exceptions import BNMDataError, BNMInputError
from bnm.markov_blanket import markov_blanket_indices
from bnm.sid import SIDResult, sid

DESCRIPTIVE_METRIC_NAMES: tuple[str, ...] = (
    "n_edges",
    "n_nodes",
    "n_directed_arcs",
    "n_undirected_arcs",
    "n_bidirected_arcs",
    "n_circle_edges",
    "n_colliders",
    "n_root_nodes",
    "n_leaf_nodes",
    "n_isolated_nodes",
    "n_reversible_arcs",
)

COMPARATIVE_METRIC_NAMES: tuple[str, ...] = (
    "additions",
    "deletions",
    "reversals",
    "shd",
    "hd",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
)

_DESC_FNS = {
    "n_edges": count_edges,
    "n_nodes": count_nodes,
    "n_directed_arcs": count_directed_arcs,
    "n_undirected_arcs": count_undirected_arcs,
    "n_bidirected_arcs": count_bidirected_arcs,
    "n_circle_edges": count_circle_edges,
    "n_colliders": count_colliders,
    "n_root_nodes": count_root_nodes,
    "n_leaf_nodes": count_leaf_nodes,
    "n_isolated_nodes": count_isolated_nodes,
    "n_reversible_arcs": count_reversible_arcs,
}

_COMP_FNS = {
    "additions": count_additions,
    "deletions": count_deletions,
    "reversals": count_reversals,
    "shd": shd,
    "hd": hd,
    "tp": true_positives,
    "fp": false_positives,
    "fn": false_negatives,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

MetricSpec = Iterable[str] | Literal["all"] | None
PerNodeSpec = bool | Iterable[int | str]


@dataclass(frozen=True, slots=True)
class Comparison:
    """Output of :func:`compare`. Pure data; no I/O methods.

    Use :func:`to_dataframe` for a wide-format tabular view.
    """

    g1_descriptive: dict[str, float]
    """Descriptive metrics on g1 keyed by name from
    :data:`DESCRIPTIVE_METRIC_NAMES`. Empty when ``descriptive=None``."""

    g2_descriptive: dict[str, float] | None
    """Descriptive metrics on g2; ``None`` when ``g2`` is omitted or
    ``descriptive=None``."""

    comparative: dict[str, float] | None
    """Comparative metrics; ``None`` when ``g2`` is omitted or
    ``comparative=None``."""

    sid: SIDResult | None
    """SID result; ``None`` unless ``include_sid=True``."""

    per_node: dict[str | int, dict[str, float]] | None
    """Per-Markov-blanket sub-results when ``per_node`` is truthy.
    Keys are variable names (when the graph has ``var_names``) or
    integer indices. Each value is a flat dict of metric name → value
    mixing descriptive (with ``_base`` suffix for g1 when g2 is also
    present) and comparative for that blanket."""

    var_names: tuple[str, ...] | None
    """Variable names from g1 (g1 is authoritative for naming)."""


# ---- input resolution --------------------------------------------------


def _resolve_metric_set(
    spec: MetricSpec,
    available: tuple[str, ...],
    *,
    kind: str,
) -> tuple[str, ...]:
    if spec is None:
        return ()
    if spec == "all":
        return available
    out = tuple(spec)
    bad = set(out) - set(available)
    if bad:
        raise BNMInputError(
            f"compare(): unknown {kind} metric(s) {sorted(bad)}; available: {list(available)}"
        )
    # Preserve the requested order, deduped.
    seen: set[str] = set()
    deduped: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return tuple(deduped)


def _make_subgraph(
    endpoints: np.ndarray,
    indices: tuple[int, ...],
    var_names: tuple[str, ...] | None,
) -> _Graph:
    sub_n = len(indices)
    sub_ep = endpoints[np.ix_(indices, indices)].copy()
    sub_names = tuple(var_names[i] for i in indices) if var_names is not None else None
    return _Graph(n_vars=sub_n, endpoints=sub_ep, var_names=sub_names)


# ---- per-node ----------------------------------------------------------


def _compute_per_node(
    g1: object,
    g2: object | None,
    *,
    per_node: PerNodeSpec,
    desc_names: tuple[str, ...],
    comp_names: tuple[str, ...],
    include_sid: bool,
) -> dict[str | int, dict[str, float]]:
    n1, ep1, names1 = _to_endpoints(g1)
    if g2 is not None:
        _, ep2, names2 = _to_endpoints(g2)
    else:
        ep2 = None
        names2 = None

    if per_node is True:
        if names1 is not None:
            iterable: list[int | str] = list(names1)
        else:
            iterable = list(range(n1))
    else:
        iterable = list(per_node)  # type: ignore[arg-type]

    out: dict[str | int, dict[str, float]] = {}
    for var in iterable:
        v_idx = _resolve_var(var, names1, n1)
        mb_indices = markov_blanket_indices(g1, var)
        sub_g1 = _make_subgraph(ep1, mb_indices, names1)

        node_metrics: dict[str, float] = {}

        # g1 descriptive on the MB sub-graph.
        for name in desc_names:
            metric_key = f"{name}_base" if g2 is not None else name
            node_metrics[metric_key] = float(_DESC_FNS[name](sub_g1))

        if g2 is not None and ep2 is not None:
            # g2 restricted to g1's MB indices — same node set so
            # comparative + SID can run unmodified.
            sub_g2 = _make_subgraph(ep2, mb_indices, names2)
            for name in desc_names:
                node_metrics[name] = float(_DESC_FNS[name](sub_g2))
            for name in comp_names:
                node_metrics[name] = float(_COMP_FNS[name](sub_g1, sub_g2))
            if include_sid:
                # SID validity isn't always preserved on sub-graphs
                # (CIRCLE marks could surface from PAG inputs). Skip
                # silently when the sub-graph fails the DAG/CPDAG
                # contract.
                try:
                    sub_sid = sid(sub_g1, sub_g2)
                except (BNMInputError, BNMDataError):
                    pass
                else:
                    node_metrics["sid"] = float(sub_sid.sid)
                    node_metrics["sid_lower_bound"] = float(sub_sid.sid_lower_bound)
                    node_metrics["sid_upper_bound"] = float(sub_sid.sid_upper_bound)

        key: str | int = names1[v_idx] if names1 is not None else v_idx
        out[key] = node_metrics

    return out


# ---- entry point -------------------------------------------------------


def compare(
    g1: object,
    g2: object | None = None,
    *,
    descriptive: MetricSpec = "all",
    comparative: MetricSpec = "all",
    include_sid: bool = False,
    per_node: PerNodeSpec = False,
) -> Comparison:
    """Compute a multi-metric comparison.

    Args:
        g1: reference graph (any GraphLikeInput).
        g2: estimated graph; if ``None``, only descriptive metrics on
            ``g1`` are computed.
        descriptive: which descriptive metrics to include — an iterable
            of names from :data:`DESCRIPTIVE_METRIC_NAMES`, the literal
            ``"all"`` (default), or ``None`` to skip.
        comparative: which comparative metrics to include — same shape
            as ``descriptive``. Requires ``g2``.
        include_sid: also compute SID. Requires ``g2`` and that ``g1``
            is a pure DAG.
        per_node: ``True`` to compute per-Markov-blanket sub-results
            for every variable in ``g1``; an iterable of variable
            handles (names or indices) to limit to those; ``False``
            (default) to skip.

    Returns:
        :class:`Comparison`.

    Raises:
        BNMInputError on inconsistent inputs (e.g. comparative metrics
        requested but ``g2`` omitted; ``include_sid=True`` with no g2
        or with non-DAG g1; unknown metric names).
    """
    desc_names = _resolve_metric_set(descriptive, DESCRIPTIVE_METRIC_NAMES, kind="descriptive")
    comp_names = _resolve_metric_set(comparative, COMPARATIVE_METRIC_NAMES, kind="comparative")

    # Normalise once. Downstream metric calls pass the resulting `_Graph`
    # instances through `_to_endpoints`, which short-circuits validation
    # for `_Graph` inputs — so a per-node sweep over n variables runs
    # validation O(1) instead of O(n × metrics).
    g1 = to_graphlike(g1)
    if g2 is not None:
        g2 = to_graphlike(g2)

    # When g2 is omitted, the implicit default ``comparative="all"`` is
    # treated as "skip" rather than an error — single-graph mode just
    # gives back what's possible on g1 alone. An EXPLICIT request
    # like ``comparative=["shd"]`` or ``include_sid=True`` does
    # error, since the caller has clearly asked for a g1-vs-g2 metric.
    if g2 is None:
        if comparative != "all" and comp_names:
            raise BNMInputError(
                "compare(): comparative metrics requested but g2 is None. "
                "Pass g2 or set comparative=None."
            )
        comp_names = ()
        if include_sid:
            raise BNMInputError("compare(): include_sid=True but g2 is None.")

    _, _, names1 = _to_endpoints(g1)

    g1_desc = {name: float(_DESC_FNS[name](g1)) for name in desc_names}

    g2_desc: dict[str, float] | None = None
    if g2 is not None and desc_names:
        g2_desc = {name: float(_DESC_FNS[name](g2)) for name in desc_names}

    comparative_dict: dict[str, float] | None = None
    if g2 is not None and comp_names:
        comparative_dict = {name: float(_COMP_FNS[name](g1, g2)) for name in comp_names}

    sid_result: SIDResult | None = None
    if include_sid:
        sid_result = sid(g1, g2)

    per_node_dict: dict[str | int, dict[str, float]] | None = None
    if per_node:
        per_node_dict = _compute_per_node(
            g1,
            g2,
            per_node=per_node,
            desc_names=desc_names,
            comp_names=comp_names,
            include_sid=include_sid,
        )

    return Comparison(
        g1_descriptive=g1_desc,
        g2_descriptive=g2_desc,
        comparative=comparative_dict,
        sid=sid_result,
        per_node=per_node_dict,
        var_names=names1,
    )


# ---- DataFrame view ----------------------------------------------------


def to_dataframe(comparison: Comparison) -> Any:
    """Render a :class:`Comparison` as a wide-format pandas DataFrame.

    Lazy-imports pandas; raises :class:`BNMError` (with a helpful
    message) if pandas isn't installed.

    Layout (mirrors 0.1.x's ``compare_df`` shape):

      - One row per variable (when ``per_node`` was set) plus an
        ``"All"`` row for whole-graph metrics.
      - Columns:
          * g1's descriptive metric ``X``: ``X_base`` when ``g2`` is
            present, else ``X``.
          * g2's descriptive metric ``X``: ``X``.
          * Comparative metric ``X``: ``X``.
          * SID: ``sid``, ``sid_lower_bound``, ``sid_upper_bound``.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        from bnm.exceptions import BNMError  # noqa: PLC0415

        raise BNMError(
            "bnm.to_dataframe requires pandas. Install with "
            "`pip install bnm[pandas]` or `pip install pandas`."
        ) from exc

    rows: list[dict[str, Any]] = []

    has_g2 = comparison.g2_descriptive is not None or comparison.comparative is not None

    all_row: dict[str, Any] = {"node_name": "All"}
    desc_suffix_g1 = "_base" if has_g2 else ""
    for name, val in comparison.g1_descriptive.items():
        all_row[f"{name}{desc_suffix_g1}"] = val
    if comparison.g2_descriptive is not None:
        for name, val in comparison.g2_descriptive.items():
            all_row[name] = val
    if comparison.comparative is not None:
        for name, val in comparison.comparative.items():
            all_row[name] = val
    if comparison.sid is not None:
        all_row["sid"] = float(comparison.sid.sid)
        all_row["sid_lower_bound"] = float(comparison.sid.sid_lower_bound)
        all_row["sid_upper_bound"] = float(comparison.sid.sid_upper_bound)
    rows.append(all_row)

    if comparison.per_node:
        for node_key, metrics in comparison.per_node.items():
            row: dict[str, Any] = {
                "node_name": node_key if isinstance(node_key, str) else int(node_key)
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)
