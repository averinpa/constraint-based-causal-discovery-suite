"""rfci() differs from fci() on fixtures where R5+ or refinement matter."""

from __future__ import annotations

import numpy as np

from cbcd import fci, rfci
from tests.fixtures_pag import ALL_PAG_FIXTURES
from tests.oracle_pag import DSeparationOracleProjected


def test_rfci_matches_fci_on_no_latent_fixtures() -> None:
    """For graphs with no latents, R1–R4 are sufficient; rfci() and fci() agree."""
    for name in ("y_structure", "chain_no_latent", "fork_no_latent"):
        full_dag, n_observed, _ = ALL_PAG_FIXTURES[name]()
        oracle = DSeparationOracleProjected(full_dag, n_observed)
        data = np.zeros((10, n_observed), dtype=np.float64)
        out_fci = fci(data, ci_test=oracle, alpha=0.5)
        out_rfci = rfci(data, ci_test=oracle, alpha=0.5)
        assert np.array_equal(out_fci.endpoints, out_rfci.endpoints), (
            f"{name}: fci and rfci should agree but differ"
        )


def test_rfci_returns_pag_with_only_pag_marks() -> None:
    full_dag, n_observed, _ = ALL_PAG_FIXTURES["confounded_chain_through_collider"]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    data = np.zeros((10, n_observed), dtype=np.float64)
    out = rfci(data, ci_test=oracle, alpha=0.5)
    assert out.n_vars == n_observed
