"""Regression guards for the v0.2.1 compare() perf optimisation.

The optimisation: ``bnmetrics.compare`` normalises ``g1`` / ``g2`` to internal
``_Graph`` instances exactly once, then reuses them across every
descriptive / comparative / SID / per-node call. Downstream
``_to_endpoints`` invocations on those normalised inputs hit the
``_Graph`` fast path and skip the O(n²) ``_validate_endpoints`` work.

These tests pin two things so a regression can't quietly slip in:

  - **Structural** — ``_validate_endpoints`` runs at most once per
    distinct external GraphLike across an entire ``compare(per_node=
    True)`` call. (A regression that re-validates inside a metric or
    inside the per-node loop would push this count to O(n × metrics).)

  - **Wall-clock** — ``compare(per_node=True)`` on a 200-node external
    GraphLike completes well under one second. (Pre-fix the same call
    took multiple seconds at this size; the demo notebook had to drop
    from n=1000 to n=200 to stay tolerable. The 1.0 s bound here is a
    generous regression detector, not a precision benchmark.)
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import bnmetrics
from bnmetrics import adapter as bnm_adapter
from tests.fixtures import make_dag


def _external_graphlike(g):
    """Wrap a `_Graph` as a duck-typed external GraphLike.

    `SimpleNamespace` passes `_is_graphlike` (has n_vars / endpoints /
    var_names, isn't an ndarray) but is **not** a `_Graph`, so it goes
    through the full `_validate_endpoints` path on the first call. This
    is what cbcd's `DAG`/`CPDAG`/`PAG` instances look like to bnmetrics.
    """
    return SimpleNamespace(n_vars=g.n_vars, endpoints=g.endpoints, var_names=g.var_names)


def _chain_dag(n: int):
    """A 0→1→…→(n-1) chain with one extra fork edge per even node."""
    edges = [(i, i + 1) for i in range(n - 1)]
    edges += [(i, i + 2) for i in range(0, n - 2, 2)]
    return make_dag(n, edges, var_names=tuple(f"v{i}" for i in range(n)))


def test_validate_endpoints_called_once_per_external_input() -> None:
    g1_ext = _external_graphlike(_chain_dag(20))
    g2_ext = _external_graphlike(_chain_dag(20))

    with patch.object(
        bnm_adapter, "_validate_endpoints", wraps=bnm_adapter._validate_endpoints
    ) as spy:
        bnmetrics.compare(g1_ext, g2_ext, per_node=True)

    # One pass for g1, one for g2 — independent of n_vars and the number
    # of metrics. Pre-fix this would be O(n_vars × n_metrics).
    assert spy.call_count == 2, (
        f"_validate_endpoints was called {spy.call_count} times across a "
        f"compare(per_node=True) over two 20-node external GraphLikes; "
        f"expected exactly 2 (once per distinct external input)."
    )


def test_validate_endpoints_called_once_for_single_graph_mode() -> None:
    g_ext = _external_graphlike(_chain_dag(20))

    with patch.object(
        bnm_adapter, "_validate_endpoints", wraps=bnm_adapter._validate_endpoints
    ) as spy:
        bnmetrics.compare(g_ext, per_node=True)

    assert spy.call_count == 1


def test_compare_per_node_under_one_second_at_n200() -> None:
    """Regression guard for the n³ → n² fix.

    Pre-fix: ~minutes on n=1000, several seconds on n=200 because every
    metric inside the per-node loop re-validated the external graph.
    Post-fix: well under 1 s at n=200.
    """
    g_ext = _external_graphlike(_chain_dag(200))

    t0 = time.perf_counter()
    bnmetrics.compare(g_ext, per_node=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, (
        f"compare(per_node=True) on a 200-node external GraphLike took "
        f"{elapsed:.3f}s; regression bound is 1.0s."
    )


def test_validate_endpoints_vectorised_on_large_clean_matrix() -> None:
    """The vectorised validator handles a 1000×1000 valid matrix without
    hitting a Python-level n² loop. Generous time bound — this is a
    regression detector, not a precision benchmark."""
    n = 1000
    endpoints = np.zeros((n, n), dtype=np.int8)
    arrow = int(bnmetrics.EndpointMark.ARROW)
    tail = int(bnmetrics.EndpointMark.TAIL)
    for i in range(n - 1):
        endpoints[i, i + 1] = arrow
        endpoints[i + 1, i] = tail

    t0 = time.perf_counter()
    bnm_adapter._validate_endpoints(endpoints, source="perf test")
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, (
        f"_validate_endpoints on a 1000×1000 valid matrix took {elapsed:.3f}s; "
        f"regression bound is 0.5s."
    )
