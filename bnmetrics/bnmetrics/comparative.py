"""Comparative metrics: SHD, HD, F1, precision/recall, TP/FP/FN, additions,
deletions, reversals.

All functions take ``(g1, g2)`` where ``g1`` is the reference (truth) and
``g2`` is the comparison (estimate). Both are normalised through
:func:`bnmetrics.adapter._to_endpoints`. Required: equal ``n_vars`` post-
normalisation; mismatch raises :class:`bnmetrics.BNMDataError`. Variable-name
alignment: if both have ``var_names``, they must match as ordered tuples
(else :class:`BNMDataError`); positional alignment otherwise.

Edge-type semantics on the int8 endpoint matrix (matching 0.1.x):

- presence-only (additions, deletions, hd): any non-NO_EDGE pair counts.
- exact-match (TP/FP/FN): directed must match direction; undirected
  matches undirected (orientation-agnostic); bidirected matches
  bidirected.
- reversal: orientation differs even though adjacency is preserved
  (directed↔directed reversed, directed↔undirected, undirected↔directed,
  directed↔bidirected, etc.).

bnmetrics 0.1.x has no bidirected handling; the legacy-parity tests don't
exercise bidirected cases (the snapshot excludes them). v0.2's
bidirected behaviour above is the natural generalisation and is
covered by hand-computed tests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bnmetrics.adapter import _to_endpoints
from bnmetrics.exceptions import BNMDataError
from bnmetrics.marks import EndpointMark

NO_EDGE = int(EndpointMark.NO_EDGE)
TAIL = int(EndpointMark.TAIL)
ARROW = int(EndpointMark.ARROW)


# Compact edge-type codes for the upper-triangle classification.
# Used internally; values are local to this module.
_EDGE_NONE = 0
_EDGE_DIRECTED_FWD = 1  # i → j  (mark at j is ARROW, at i is TAIL)
_EDGE_DIRECTED_BWD = 2  # j → i
_EDGE_UNDIRECTED = 3
_EDGE_BIDIRECTED = 4
_EDGE_OTHER = 5  # CIRCLE-bearing or unsupported; treated as adjacency-only


def _classify_pair(mark_at_j: int, mark_at_i: int) -> int:
    """Classify an ``(i, j)`` upper-triangle edge by its endpoint marks."""
    if mark_at_j == NO_EDGE:
        return _EDGE_NONE
    if mark_at_j == ARROW and mark_at_i == TAIL:
        return _EDGE_DIRECTED_FWD
    if mark_at_j == TAIL and mark_at_i == ARROW:
        return _EDGE_DIRECTED_BWD
    if mark_at_j == TAIL and mark_at_i == TAIL:
        return _EDGE_UNDIRECTED
    if mark_at_j == ARROW and mark_at_i == ARROW:
        return _EDGE_BIDIRECTED
    return _EDGE_OTHER


def _classify_upper(endpoints: NDArray[np.int8]) -> NDArray[np.int_]:
    """Classify every upper-triangle pair to one of the _EDGE_* codes."""
    n = endpoints.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    a = endpoints[iu, ju]
    b = endpoints[ju, iu]
    out = np.full(len(iu), _EDGE_NONE, dtype=np.int_)
    out[(a == ARROW) & (b == TAIL)] = _EDGE_DIRECTED_FWD
    out[(a == TAIL) & (b == ARROW)] = _EDGE_DIRECTED_BWD
    out[(a == TAIL) & (b == TAIL)] = _EDGE_UNDIRECTED
    out[(a == ARROW) & (b == ARROW)] = _EDGE_BIDIRECTED
    # Anything else (CIRCLE-bearing) defaults to _EDGE_NONE → updated to _OTHER.
    is_other = (a != NO_EDGE) & (
        ~((a == ARROW) & (b == TAIL))
        & ~((a == TAIL) & (b == ARROW))
        & ~((a == TAIL) & (b == TAIL))
        & ~((a == ARROW) & (b == ARROW))
    )
    out[is_other] = _EDGE_OTHER
    return out


def _normalise_pair(g1: object, g2: object) -> tuple[NDArray, NDArray]:
    """Normalise (g1, g2) to a pair of (n, n) endpoint matrices and
    validate compatibility."""
    n1, ep1, names1 = _to_endpoints(g1)
    n2, ep2, names2 = _to_endpoints(g2)
    if n1 != n2:
        raise BNMDataError(f"comparative metric: g1 has {n1} variables, g2 has {n2}; must match")
    if names1 is not None and names2 is not None and names1 != names2:
        raise BNMDataError(
            "comparative metric: g1.var_names and g2.var_names differ; "
            "alignment by positional index is unsafe when names disagree"
        )
    return ep1, ep2


# ---- adjacency-only counts ---------------------------------------------


def _adjacency_mask(classes: NDArray[np.int_]) -> NDArray[np.bool_]:
    """True wherever the upper-triangle pair has an edge of any type."""
    return classes != _EDGE_NONE


def count_additions(g1: object, g2: object) -> int:
    """Edges in g2 with no adjacency in g1 (presence-only)."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    return int(np.sum(_adjacency_mask(c2) & ~_adjacency_mask(c1)))


def count_deletions(g1: object, g2: object) -> int:
    """Edges in g1 with no adjacency in g2."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    return int(np.sum(_adjacency_mask(c1) & ~_adjacency_mask(c2)))


def count_reversals(g1: object, g2: object) -> int:
    """Adjacency preserved but orientation differs.

    Counted cases (both directions, both edge types):
      directed_fwd ↔ directed_bwd
      directed_*   ↔ undirected
      directed_*   ↔ bidirected
      undirected   ↔ directed_*
      undirected   ↔ bidirected
      bidirected   ↔ directed_*
      bidirected   ↔ undirected

    Equal classifications are not reversals (TP). Adjacency-mismatch is
    not a reversal (additions/deletions cover that).
    """
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    both = _adjacency_mask(c1) & _adjacency_mask(c2)
    return int(np.sum(both & (c1 != c2)))


def shd(g1: object, g2: object) -> int:
    """Structural Hamming Distance: ``additions + deletions + reversals``."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    a1, a2 = _adjacency_mask(c1), _adjacency_mask(c2)
    additions = int(np.sum(a2 & ~a1))
    deletions = int(np.sum(a1 & ~a2))
    reversals = int(np.sum(a1 & a2 & (c1 != c2)))
    return additions + deletions + reversals


def hd(g1: object, g2: object) -> int:
    """Hamming Distance: ``additions + deletions`` (skeleton-only)."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    a1, a2 = _adjacency_mask(c1), _adjacency_mask(c2)
    return int(np.sum(a1 ^ a2))


# ---- exact-match TP/FP/FN ---------------------------------------------


def _exact_match_mask(c1: NDArray[np.int_], c2: NDArray[np.int_]) -> NDArray[np.bool_]:
    """True where both g1 and g2 have an edge AND their classifications
    agree."""
    return _adjacency_mask(c1) & (c1 == c2)


def true_positives(g1: object, g2: object) -> int:
    """Edges that match in both graphs with the same orientation/type."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    return int(np.sum(_exact_match_mask(c1, c2)))


def false_positives(g1: object, g2: object) -> int:
    """Edges in g2 that are not exact-mark matches in g1."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    return int(np.sum(_adjacency_mask(c2) & ~_exact_match_mask(c1, c2)))


def false_negatives(g1: object, g2: object) -> int:
    """Edges in g1 that are not exact-mark matches in g2."""
    ep1, ep2 = _normalise_pair(g1, g2)
    c1, c2 = _classify_upper(ep1), _classify_upper(ep2)
    return int(np.sum(_adjacency_mask(c1) & ~_exact_match_mask(c1, c2)))


def precision(g1: object, g2: object) -> float:
    """``TP / (TP + FP)`` — returns 0.0 when both are 0."""
    tp = true_positives(g1, g2)
    fp = false_positives(g1, g2)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(g1: object, g2: object) -> float:
    """``TP / (TP + FN)`` — returns 0.0 when both are 0."""
    tp = true_positives(g1, g2)
    fn = false_negatives(g1, g2)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1(g1: object, g2: object) -> float:
    """Harmonic mean of precision and recall — returns 0.0 when both are 0."""
    p = precision(g1, g2)
    r = recall(g1, g2)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---- batch helper for the snapshot-parity test path ------------------------


_METRIC_FNS = {
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
    "f1_score": f1,
}


def all_comparative(
    g1: object,
    g2: object,
    *,
    keys: Literal["legacy", "v0_2"] = "v0_2",
) -> dict[str, float]:
    """Compute every comparative metric. ``keys='legacy'`` returns
    'f1_score' (the 0.1.x naming) instead of 'f1' so the snapshot's
    'comparative' dict is directly comparable."""
    out: dict[str, float] = {}
    for name, fn in _METRIC_FNS.items():
        out[name] = fn(g1, g2)
    if keys == "v0_2":
        out["f1"] = out.pop("f1_score")
    return out
