"""FCI structural regression: SHD = 0 against expected PAG with d-separation oracle."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import fci
from cbcd.graph import EndpointMark
from tests.fixtures_pag import ALL_PAG_FIXTURES
from tests.oracle_pag import DSeparationOracleProjected


def _shd_endpoints(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


@pytest.mark.parametrize("name", list(ALL_PAG_FIXTURES))
def test_fci_recovers_expected_pag(name: str) -> None:
    full_dag, n_observed, expected = ALL_PAG_FIXTURES[name]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    # FCI doesn't use ``data`` arithmetic when the CI test is supplied directly,
    # but ``_normalize_data`` checks the column count matches ``ci.n_vars``.
    dummy_data = np.zeros((10, n_observed), dtype=np.float64)
    out = fci(dummy_data, ci_test=oracle, alpha=0.5)
    assert out.n_vars == expected.n_vars
    assert _shd_endpoints(out.endpoints, expected.endpoints) == 0, (
        f"{name}: recovered\n{out.endpoints}\nexpected\n{expected.endpoints}"
    )


@pytest.mark.parametrize("name", list(ALL_PAG_FIXTURES))
def test_fci_marks_are_pag_marks(name: str) -> None:
    full_dag, n_observed, _ = ALL_PAG_FIXTURES[name]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    dummy_data = np.zeros((10, n_observed), dtype=np.float64)
    out = fci(dummy_data, ci_test=oracle, alpha=0.5)
    permitted = {
        int(EndpointMark.NO_EDGE),
        int(EndpointMark.TAIL),
        int(EndpointMark.ARROW),
        int(EndpointMark.CIRCLE),
    }
    seen = {int(m) for m in np.unique(out.endpoints)}
    assert seen <= permitted
