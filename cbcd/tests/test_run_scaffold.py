"""Shared algorithm scaffold: iid_run / lagged_run context managers + Algorithm protocol."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import Algorithm, iid_run, lagged_run, pc
from cbcd._run import LaggedRunContext, RunContext
from cbcd.exceptions import CBCDInputError
from cbcd.recording import InMemoryRecorder
from cbcd.timeseries.lagged import LaggedDataset


def _data() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((80, 4))


def test_iid_run_validates() -> None:
    data = _data()
    with pytest.raises(CBCDInputError), iid_run(  # alpha out of range
        data, ci_test="fisherz", algorithm="t", params={}, alpha=1.5
    ):
        pass
    with pytest.raises(CBCDInputError), iid_run(  # target out of range
        data, ci_test="fisherz", algorithm="t", params={}, targets=[9]
    ):
        pass
    with pytest.raises(CBCDInputError), iid_run(  # empty targets
        data, ci_test="fisherz", algorithm="t", params={}, targets=[]
    ):
        pass


def test_iid_run_brackets_and_exposes_context() -> None:
    data = _data()
    rec = InMemoryRecorder()
    with iid_run(
        data, ci_test="fisherz", algorithm="scaf", params={"k": 1},
        targets=[2, 0], recorder=rec, run_id="r",
    ) as ctx:
        assert isinstance(ctx, RunContext)
        assert ctx.n_vars == 4 and ctx.n_samples == 80
        assert ctx.targets == [0, 2]  # deduped + sorted
        ctx.summary = {"n_edges": 3}
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "scaf" and run["run_id"] == "r"
    assert list(run["targets"]) == [0, 2]
    assert run["result_summary"] == {"n_edges": 3}


def test_iid_run_no_finish_on_exception() -> None:
    rec = InMemoryRecorder()
    with pytest.raises(ValueError), iid_run(
        _data(), ci_test="fisherz", algorithm="t", params={}, recorder=rec
    ):
        raise ValueError("boom")
    # begin_run fired, but finish_run did NOT -> no completed run row
    assert len(rec.to_frames()["runs"]) == 0


def test_lagged_run_brackets() -> None:
    data = LaggedDataset(np.random.default_rng(1).standard_normal((200, 3)), max_lag=2)
    rec = InMemoryRecorder()
    with lagged_run(data, ci_test="parcorr", algorithm="tsscaf", params={}, recorder=rec) as ctx:
        assert isinstance(ctx, LaggedRunContext)
        assert ctx.n_series == 3 and ctx.max_lag == 2 and ctx.grid_n == 9
        ctx.summary = {"n_edges": 0}
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "tsscaf" and run["n_vars"] == 9


def test_pc_satisfies_algorithm_protocol() -> None:
    assert isinstance(pc, Algorithm)
