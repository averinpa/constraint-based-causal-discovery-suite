"""FAS produces the same Skeleton as PCStable on i.i.d. fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.skeleton import FAS, PCStable
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle


@pytest.mark.parametrize("name", list(ALL_FIXTURES))
def test_fas_matches_pc_stable_on_oracle(name: str) -> None:
    dag, _ = ALL_FIXTURES[name]()
    oracle = DSeparationOracle(dag)
    skel_fas = FAS()(oracle, alpha=0.5)
    skel_pc = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel_fas.adj, skel_pc.adj)
    assert skel_fas.sepsets == skel_pc.sepsets


def test_fas_rejects_n_jobs_gt_1() -> None:
    dag, _ = ALL_FIXTURES["chain"]()
    oracle = DSeparationOracle(dag)
    with pytest.raises(CBCDInputError):
        FAS()(oracle, alpha=0.5, n_jobs=2)
