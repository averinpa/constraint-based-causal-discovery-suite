"""DAG-with-latent fixtures and hand-written expected PAGs for FCI tests.

Each fixture returns ``(full_dag, n_observed, expected_pag)`` where:

* ``full_dag``: a ``DAG`` over observed + latent variables. Observed nodes
  have indices ``0..n_observed-1``; latents have indices
  ``n_observed..full_dag.n_vars-1``.
* ``n_observed``: the size of the observed set.
* ``expected_pag``: the PAG that ``fci()`` should recover when given the
  d-separation oracle as its CI test.
"""

from __future__ import annotations

import numpy as np

from cbcd.graph import DAG, PAG, EndpointMark


def _ep_directed(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        m[u, v] = EndpointMark.ARROW
        m[v, u] = EndpointMark.TAIL
    return m


def _ep_pag(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
CIRC = EndpointMark.CIRCLE


def y_structure() -> tuple[DAG, int, PAG]:
    """0 → 2, 1 → 2 (no latent). FCI gives 0 o→ 2 ←o 1."""
    dag = DAG(3, _ep_directed(3, [(0, 2), (1, 2)]))
    expected = PAG(
        3,
        _ep_pag(3, [(0, 2, CIRC, ARR), (1, 2, CIRC, ARR)]),
    )
    return dag, 3, expected


def chain_no_latent() -> tuple[DAG, int, PAG]:
    """0 → 1 → 2 (no latent). PC gives 0 — 1 — 2; FCI gives 0 o-o 1 o-o 2."""
    dag = DAG(3, _ep_directed(3, [(0, 1), (1, 2)]))
    expected = PAG(
        3,
        _ep_pag(3, [(0, 1, CIRC, CIRC), (1, 2, CIRC, CIRC)]),
    )
    return dag, 3, expected


def fork_no_latent() -> tuple[DAG, int, PAG]:
    """0 → 1, 0 → 2 (fork, no latent). FCI gives 1 o-o 0 o-o 2."""
    dag = DAG(3, _ep_directed(3, [(0, 1), (0, 2)]))
    expected = PAG(
        3,
        _ep_pag(3, [(0, 1, CIRC, CIRC), (0, 2, CIRC, CIRC)]),
    )
    return dag, 3, expected


def confounded_chain_through_collider() -> tuple[DAG, int, PAG]:
    """0 → 2, 1 → 2, 2 → 3, L → 1, L → 3 with L latent. Observed = {0, 1, 2, 3}.

    Pipeline:
      * Skeleton survives 0-2, 1-2, 2-3, 1-3; removes 0-1 (sepset {}), 0-3
        (sepset {1, 2}), 1-2 stays, etc.
      * Collider step finds (0, 2, 1) collider; (0, 2, 3) non-collider.
      * R1 orients 2 → 3; R2 orients 1 *→ 3; R4 (discriminating path
        ⟨0, 2, 1, 3⟩, b=1 ∈ Sepset(0, 3) = {1, 2}) orients 1 → 3 fully.
      * Final: 0 o→ 2, 1 o→ 2, 2 → 3, 1 → 3.

    The MAG of the underlying DAG is the same: L confounds 1 and 3 but
    1 → 2 → 3 makes 1 an ancestor of 3, so the MAG edge is 1 → 3 (not 1 ↔ 3).
    """
    # Latent index = 4. observed = 0..3.
    dag = DAG(
        5,
        _ep_directed(5, [(0, 2), (1, 2), (2, 3), (4, 1), (4, 3)]),
    )
    expected = PAG(
        4,
        _ep_pag(
            4,
            [
                (0, 2, CIRC, ARR),
                (1, 2, CIRC, ARR),
                (2, 3, TAIL, ARR),
                (1, 3, TAIL, ARR),
            ],
        ),
    )
    return dag, 4, expected


ALL_PAG_FIXTURES = {
    "y_structure": y_structure,
    "chain_no_latent": chain_no_latent,
    "fork_no_latent": fork_no_latent,
    "confounded_chain_through_collider": confounded_chain_through_collider,
}
