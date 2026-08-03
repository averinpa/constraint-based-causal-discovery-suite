"""PCMCI+ oracle-recovery + contemporaneous-coverage tests.

Under the time-series d-separation oracle, ``pcmci_plus`` must recover the true ``TimeSeriesCPDAG``
*including contemporaneous edges* (the completeness upgrade over lagged-only ``pcmci``). Contemporaneous
fixtures live here (not in ``ALL_TS_FIXTURES``, which the ``pcmci`` oracle test parametrises over and
which ``pcmci`` cannot recover). The coverage test exhibits the concrete win: a contemporaneous link
``pcmci`` misses and ``pcmci_plus`` finds.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import pcmci, pcmci_plus
from cbcd.graph import EndpointMark as M
from cbcd.timeseries import LaggedDataset, TimeSeriesCPDAG, TimeSeriesDAG
from tests.timeseries.oracle import DSeparationOracleLagged


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def _contemp_vstructure() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """Contemporaneous collider 0 -> 2 <- 1 (no lag). PCMCI+ identifies the v-structure."""
    ep = np.zeros((1, 3, 3), np.int8)
    ep[0, 0, 2], ep[0, 2, 0] = M.ARROW, M.TAIL
    ep[0, 1, 2], ep[0, 2, 1] = M.ARROW, M.TAIL
    return TimeSeriesDAG(3, 0, ep), TimeSeriesCPDAG(3, 0, ep.copy())


def _contemp_oriented_by_lag() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """Contemporaneous 0 -> 1 with autoregressive 0_{t-1} -> 0_t. The lagged parent makes the
    contemporaneous edge orientable (Meek R1): 0_{t-1} -> 0_t -o 1_t, 0_{t-1} not adj 1_t => 0_t -> 1_t."""
    ep = np.zeros((2, 2, 2), np.int8)
    ep[0, 0, 1], ep[0, 1, 0] = M.ARROW, M.TAIL  # contemp 0 -> 1
    ep[1, 0, 0] = M.ARROW  # 0_{t-1} -> 0_t
    return TimeSeriesDAG(2, 1, ep), TimeSeriesCPDAG(2, 1, ep.copy())


def _contemp_undirected() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """A lone contemporaneous edge 0 - 1 with no collider or lagged parent: unorientable, stays
    undirected (o-o -> TAIL/TAIL) in the CPDAG."""
    dag_ep = np.zeros((1, 2, 2), np.int8)
    dag_ep[0, 0, 1], dag_ep[0, 1, 0] = M.ARROW, M.TAIL
    cpdag_ep = np.zeros((1, 2, 2), np.int8)
    cpdag_ep[0, 0, 1], cpdag_ep[0, 1, 0] = M.TAIL, M.TAIL
    return TimeSeriesDAG(2, 0, dag_ep), TimeSeriesCPDAG(2, 0, cpdag_ep)


_CONTEMP_FIXTURES = {
    "contemp_vstructure": _contemp_vstructure,
    "contemp_oriented_by_lag": _contemp_oriented_by_lag,
    "contemp_undirected": _contemp_undirected,
}


@pytest.mark.parametrize("name", list(_CONTEMP_FIXTURES))
def test_pcmci_plus_recovers_contemporaneous_cpdag(name: str) -> None:
    """pcmci_plus recovers the true TimeSeriesCPDAG (lagged + contemporaneous) under the oracle."""
    true_dag, expected = _CONTEMP_FIXTURES[name]()
    oracle = DSeparationOracleLagged(true_dag)
    ds = LaggedDataset(data=np.zeros((60, true_dag.n_vars)), max_lag=true_dag.max_lag)
    out = pcmci_plus(ds, ci_test=oracle, tau_max=true_dag.max_lag, alpha=0.5)
    assert out.n_vars == expected.n_vars
    assert out.max_lag == expected.max_lag
    assert _shd(out.endpoints, expected.endpoints) == 0, (
        f"{name}: recovered\n{out.endpoints}\nexpected\n{expected.endpoints}"
    )


def test_pcmci_plus_marks_are_valid_cpdag_marks() -> None:
    """Output uses only NO_EDGE / TAIL / ARROW marks."""
    true_dag, _ = _contemp_vstructure()
    oracle = DSeparationOracleLagged(true_dag)
    ds = LaggedDataset(np.zeros((60, 3)), max_lag=0)
    out = pcmci_plus(ds, ci_test=oracle, tau_max=0, alpha=0.5)
    seen = {int(m) for m in np.unique(out.endpoints)}
    assert seen <= {int(M.NO_EDGE), int(M.TAIL), int(M.ARROW)}


def test_pcmci_plus_recovers_contemporaneous_link_that_pcmci_misses() -> None:
    """The concrete reason PCMCI+ exists: a contemporaneous link that lagged-only ``pcmci`` cannot
    see. On a DGP with a true 0 -> 1 contemporaneous edge, ``pcmci`` returns *no* contemporaneous
    edge while ``pcmci_plus`` recovers (and here orients) it."""
    true_dag, _ = _contemp_oriented_by_lag()
    oracle = DSeparationOracleLagged(true_dag)
    ds = LaggedDataset(data=np.zeros((60, 2)), max_lag=1)

    lagged_only = pcmci(ds, ci_test=oracle, alpha=0.5)
    plus = pcmci_plus(ds, ci_test=oracle, tau_max=1, alpha=0.5)

    # pcmci (lagged-only) has NO contemporaneous edge; pcmci_plus DOES.
    assert lagged_only.contemporaneous_edges() == ()
    contemp = plus.contemporaneous_edges()
    assert len(contemp) == 1
    edge = contemp[0]
    assert {edge.src.var, edge.dst.var} == {0, 1}
    # It is recovered as the directed edge 0 -> 1.
    assert plus.endpoints[0, 0, 1] == M.ARROW and plus.endpoints[0, 1, 0] == M.TAIL
    # Both still agree on the lagged autoregressive edge 0_{t-1} -> 0_t.
    assert lagged_only.endpoints[1, 0, 0] == M.ARROW
    assert plus.endpoints[1, 0, 0] == M.ARROW


def test_pcmci_plus_records_run() -> None:
    from cbcd import InMemoryRecorder

    true_dag, _ = _contemp_vstructure()
    oracle = DSeparationOracleLagged(true_dag)
    rec = InMemoryRecorder()
    ds = LaggedDataset(np.zeros((60, 3)), max_lag=0)
    pcmci_plus(ds, ci_test=oracle, tau_max=0, alpha=0.5, recorder=rec, run_id="pp")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "pcmci_plus"
    assert run["run_id"] == "pp"
    assert run["n_ci_total"] > 0


def test_pcmci_plus_rejects_tau_max_mismatch() -> None:
    from cbcd.exceptions import CBCDInputError

    true_dag, _ = _contemp_oriented_by_lag()
    oracle = DSeparationOracleLagged(true_dag)
    ds = LaggedDataset(np.zeros((60, 2)), max_lag=1)
    with pytest.raises(CBCDInputError):
        pcmci_plus(ds, ci_test=oracle, tau_max=2, alpha=0.5)
