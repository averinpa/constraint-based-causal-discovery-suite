"""TimeSeriesPAG windowed-grid representation (roadmap Phase 5a)."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.timeseries import TimeSeriesPAG
from cbcd.timeseries.lagged import LaggedVar


def _example() -> TimeSeriesPAG:
    # n_series=2, max_lag=1 -> grid nodes: (v0,t)=0 (v0,t-1)=1 (v1,t)=2 (v1,t-1)=3
    ep = np.zeros((4, 4), dtype=np.int8)
    # v0_{t-1} o-> v0_t : circle at src node 1, arrow at dst node 0
    ep[1, 0] = EndpointMark.ARROW
    ep[0, 1] = EndpointMark.CIRCLE
    # v0_t <-> v1_t : bidirected (latent confounder), arrowheads both ends
    ep[0, 2] = EndpointMark.ARROW
    ep[2, 0] = EndpointMark.ARROW
    return TimeSeriesPAG(n_series=2, max_lag=1, window=PAG(4, ep))


def test_encode_decode_roundtrip() -> None:
    tspag = _example()
    assert tspag.grid_size == 4
    assert tspag.node_id(0, 0) == 0
    assert tspag.node_id(0, -1) == 1
    assert tspag.node_id(1, 0) == 2
    assert tspag.decode(1) == LaggedVar(0, -1)
    assert tspag.decode(2) == LaggedVar(1, 0)
    for var in range(2):
        for lag in (0, -1):
            assert tspag.decode(tspag.node_id(var, lag)) == LaggedVar(var, lag)


def test_edges_carry_both_marks() -> None:
    edges = {(e.src, e.dst): (e.mark_at_src, e.mark_at_dst) for e in _example().edges()}
    # lagged o-> edge: circle at the past end, arrow at the present end
    assert edges[(LaggedVar(0, -1), LaggedVar(0, 0))] == (EndpointMark.CIRCLE, EndpointMark.ARROW)
    # contemporaneous <-> edge: arrowheads at both ends
    assert edges[(LaggedVar(0, 0), LaggedVar(1, 0))] == (EndpointMark.ARROW, EndpointMark.ARROW)


def test_present_edges_drops_lag_mirror() -> None:
    ep = np.zeros((4, 4), dtype=np.int8)
    ep[1, 0] = EndpointMark.ARROW  # v0_{t-1} o-> v0_t (present-anchored, node 0 at lag 0)
    ep[0, 1] = EndpointMark.CIRCLE
    ep[0, 2] = EndpointMark.CIRCLE  # v0_t o-o v1_t (present-anchored)
    ep[2, 0] = EndpointMark.CIRCLE
    ep[1, 3] = EndpointMark.CIRCLE  # v0_{t-1} o-o v1_{t-1} (lag-mirror, both at lag -1)
    ep[3, 1] = EndpointMark.CIRCLE
    tspag = TimeSeriesPAG(n_series=2, max_lag=1, window=PAG(4, ep))

    assert len(tspag.edges()) == 3
    present = tspag.present_edges()
    assert len(present) == 2  # the lag-mirror edge is dropped
    assert {e.dst.lag for e in present} == {0}
    assert all(not (e.src.lag == -1 and e.dst.lag == -1) for e in present)


def test_validation() -> None:
    with pytest.raises(CBCDInputError):  # window size must be n_series*(max_lag+1)
        TimeSeriesPAG(n_series=2, max_lag=1, window=PAG(3, np.zeros((3, 3), dtype=np.int8)))
    tspag = _example()
    with pytest.raises(CBCDInputError):
        tspag.node_id(0, -5)  # lag out of range
    with pytest.raises(CBCDInputError):
        tspag.decode(99)  # node out of range
