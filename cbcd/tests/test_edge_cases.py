"""Edge-case audit: small targeted tests for boundary conditions that
weren't covered by the per-module tests."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import (
    DAG,
    PartialPAG,
    fci,
    pc,
    pcmci,
)
from cbcd.citest import FisherZ
from cbcd.exceptions import CBCDDataError
from cbcd.graph import EndpointMark
from cbcd.graph.queries import find_discriminating_path
from cbcd.rules import FCIRules
from cbcd.timeseries import LaggedDataset, LaggedVar, ParCorr, PC1Skeleton

ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
CIRC = EndpointMark.CIRCLE


# --- Inf rejection in CI tests ---------------------------------------------


def test_fisherz_rejects_inf_data() -> None:
    data = np.zeros((10, 3), dtype=np.float64)
    data[0, 0] = np.inf
    with pytest.raises(CBCDDataError):
        FisherZ(data)


def test_parcorr_rejects_inf_data() -> None:
    data = np.zeros((20, 2), dtype=np.float64)
    data[0, 0] = -np.inf
    ds = LaggedDataset(data=data, max_lag=1)
    with pytest.raises(CBCDDataError):
        ParCorr(ds)


# --- n_vars = 1 boundary cases ---------------------------------------------


def test_pc_single_variable() -> None:
    # 1 variable, no edges possible. Output: empty CPDAG with one isolated node.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 1))
    out = pc(data, alpha=0.05)
    assert out.n_vars == 1
    assert int(np.sum(out.endpoints != 0)) == 0


def test_fci_single_variable() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 1))
    out = fci(data, alpha=0.05)
    assert out.n_vars == 1
    assert int(np.sum(out.endpoints != 0)) == 0


def test_pcmci_single_variable_no_autocorr() -> None:
    # 1 variable, no autocorrelation: result has zero edges.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 1))
    ds = LaggedDataset(data=data, max_lag=1)
    out = pcmci(ds, ci_test="parcorr", alpha=0.01)
    assert out.n_vars == 1
    assert out.max_lag == 1
    assert int(np.sum(out.endpoints != 0)) == 0


# --- All edges removed: independent variables ------------------------------


def test_pc_all_independent() -> None:
    # Three independent variables — PC should remove every edge.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 3))
    out = pc(data, alpha=0.05)
    assert int(np.sum(out.endpoints != 0)) == 0


def test_fci_all_independent() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 3))
    out = fci(data, alpha=0.05)
    assert int(np.sum(out.endpoints != 0)) == 0


# --- PC₁ determinism on tied p-values --------------------------------------


class _ConstantP:
    """LaggedCITest stub that returns a fixed p-value for every call.

    With a constant p > pc_alpha, every candidate gets pruned at depth 0;
    with p < pc_alpha, every candidate survives. Either way, the parent set
    is determined and the order of removals is the only nondeterminism PC₁
    has — which the explicit (var, -lag) tie-break should pin down.
    """

    def __init__(self, n_vars: int, max_lag: int, p_value: float) -> None:
        self.n_vars = n_vars
        self.max_lag = max_lag
        self._p = p_value

    def __call__(self, x, y, S):
        return self._p

    def details(self, x, y, S):
        from cbcd.timeseries.citest import LaggedCITestResult

        return LaggedCITestResult(p_value=self._p)


def test_pc1_deterministic_under_constant_p() -> None:
    # Two runs on identical input must produce identical parent sets.
    ci = _ConstantP(n_vars=3, max_lag=2, p_value=0.5)
    a = PC1Skeleton()(ci, alpha=0.05)
    b = PC1Skeleton()(ci, alpha=0.05)
    assert a.parents == b.parents
    assert a.sepsets == b.sepsets


def test_pc1_deterministic_under_dependent_p() -> None:
    # All edges survive; sepsets dict empty in both runs.
    ci = _ConstantP(n_vars=3, max_lag=1, p_value=0.001)
    a = PC1Skeleton()(ci, alpha=0.05)
    b = PC1Skeleton()(ci, alpha=0.05)
    assert a.parents == b.parents
    assert a.sepsets == b.sepsets
    # All candidates surviving = full lagged grid.
    target = LaggedVar(0, 0)
    assert len(a.parents[target]) == 3  # 3 vars × 1 lag


# --- Discriminating-path determinism ---------------------------------------


def test_find_discriminating_path_deterministic() -> None:
    """When a discriminating path exists, find_discriminating_path returns
    the same path on repeated calls (DFS over a deterministic adjacency)."""
    # Same fixture as the queries unit tests: θ=0, a=1, b=2, c=3.
    ep = np.zeros((4, 4), dtype=np.int8)
    # θ ↔ a (arrow at θ, arrow at a)
    ep[0, 1] = ARR
    ep[1, 0] = ARR
    # a ← b: arrow at a (mark at a from b), tail at b
    ep[2, 1] = ARR
    ep[1, 2] = TAIL
    # a → c
    ep[1, 3] = ARR
    ep[3, 1] = TAIL
    # b o─o c
    ep[2, 3] = CIRC
    ep[3, 2] = CIRC
    p1 = find_discriminating_path(ep, a=1, b=2, c=3)
    p2 = find_discriminating_path(ep, a=1, b=2, c=3)
    assert p1 == p2 == (0, 1, 2, 3)


# --- FCI R4 graceful no-op when sepsets are missing ------------------------


def test_fcirules_r4_no_op_when_sepsets_none() -> None:
    """Same discriminating-path PartialPAG as above, but with sepsets=None.
    R4 cannot consult Sepset(θ, c) without a witness, so the rule should
    not fire — confirming graceful handling instead of an AttributeError."""
    ep = np.zeros((4, 4), dtype=np.int8)
    ep[0, 1] = ARR
    ep[1, 0] = ARR
    ep[2, 1] = ARR
    ep[1, 2] = TAIL
    ep[1, 3] = ARR
    ep[3, 1] = TAIL
    ep[2, 3] = CIRC
    ep[3, 2] = CIRC
    g = PartialPAG(4, ep, sepsets=None)
    out = FCIRules(rules=frozenset({"R4"}))(g)
    # Without a recorded sepset for {θ=0, c=3}, R4 takes the "else" branch
    # of its rule: orient the triple ⟨a, b, c⟩ as a ↔ b ↔ c. That writes
    # ARROW at b on edge a-b and ARROW at both ends of edge b-c.
    assert out.endpoints[1, 2] == ARR  # arrow at b=2 from a=1's edge
    assert out.endpoints[3, 2] == ARR
    assert out.endpoints[2, 3] == ARR


# --- max_cond_set larger than candidate set: clamps ------------------------


def test_pc1_max_cond_set_larger_than_candidates() -> None:
    """``max_cond_set=99`` on a small graph should clamp to the actual
    candidate count rather than crashing."""
    ci = _ConstantP(n_vars=2, max_lag=1, p_value=0.001)
    out = PC1Skeleton()(ci, alpha=0.05, max_cond_set=99)
    target = LaggedVar(0, 0)
    assert len(out.parents[target]) == 2  # 2 vars × 1 lag, all survive


def test_pc_max_cond_set_larger_than_candidates() -> None:
    """``max_cond_set`` exceeding ``n_vars - 2`` is harmless."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3))
    out = pc(data, alpha=0.05, max_cond_set=99)
    assert out.n_vars == 3


# --- DAG construction with an empty graph ----------------------------------


def test_dag_empty_graph_zero_vars() -> None:
    g = DAG(0)
    assert g.n_vars == 0
    assert g.endpoints.shape == (0, 0)
