"""anytime_fci() respects the max_cond_set depth cap."""

from __future__ import annotations

import numpy as np

from cbcd import anytime_fci, fci
from cbcd.graph import EndpointMark
from tests.fixtures_pag import ALL_PAG_FIXTURES
from tests.oracle_pag import DSeparationOracleProjected


def test_anytime_fci_with_cap_zero_keeps_all_edges() -> None:
    """``max_cond_set=0`` only tries the empty conditioning set. On the
    confounded-chain fixture, the (0, 3) sepset is {1, 2} (depth 2), so with
    a cap of 0 the edge (0, 3) survives the skeleton phase and the recovered
    PAG has more adjacencies than the unbounded ``fci()``."""
    full_dag, n_observed, _ = ALL_PAG_FIXTURES["confounded_chain_through_collider"]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    data = np.zeros((10, n_observed), dtype=np.float64)
    out_capped = anytime_fci(data, 0, ci_test=oracle, alpha=0.5)
    out_full = fci(data, ci_test=oracle, alpha=0.5)
    capped_adj = out_capped.endpoints != EndpointMark.NO_EDGE
    full_adj = out_full.endpoints != EndpointMark.NO_EDGE
    # Capped run keeps strictly more (or equal) edges; in this fixture, strictly more.
    assert int(capped_adj.sum()) >= int(full_adj.sum())
    assert int(capped_adj.sum()) > int(full_adj.sum())


def test_anytime_fci_returns_valid_pag() -> None:
    full_dag, n_observed, _ = ALL_PAG_FIXTURES["chain_no_latent"]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    data = np.zeros((10, n_observed), dtype=np.float64)
    out = anytime_fci(data, 1, ci_test=oracle, alpha=0.5)
    assert out.n_vars == n_observed
