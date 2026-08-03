"""Concrete RunRecorder implementations (Phase 0)."""

from __future__ import annotations

import json
from pathlib import Path

from cbcd.recording import (
    FileRecorder,
    InMemoryRecorder,
    NullRecorder,
    RunRecorder,
    _resolve_recorder,
    load_run,
)


def _drive(rec: RunRecorder) -> None:
    """Emit a small, deterministic event stream: 3 CI tests (one a repeat), a collider, a rule,
    and a region."""
    rec.begin_run(
        run_id="r1",
        algorithm="local",
        params={"alpha": 0.05},
        n_samples=500,
        n_vars=6,
        ci_test="fisherz",
        targets=[0, 1],
        seed=7,
    )
    rec.record_ci(x=0, y=1, S=(), p_value=0.01, depth=0, was_cache_hit=False)
    rec.record_ci(x=0, y=1, S=(2,), p_value=0.30, depth=1, was_cache_hit=False)
    rec.record_ci(x=0, y=1, S=(2,), p_value=0.30, depth=1, was_cache_hit=True)  # repeat -> cache
    rec.record_collider(triple=(0, 2, 1), classification="collider", orienter="fci:v")
    rec.record_rule(rule_set="meek", rule_name="R1", affected_edge=(2, 3))
    rec.record_region(target=0, region=[0, 2, 3], boundary=[3])
    rec.finish_run(result_summary={"n_edges": 4})


def test_null_recorder_is_noop_and_resolves() -> None:
    assert isinstance(_resolve_recorder(None), NullRecorder)
    _drive(NullRecorder())  # must not raise


def test_inmemory_counts_and_frames() -> None:
    rec = InMemoryRecorder()
    _drive(rec)
    m = rec.metrics()
    assert m["n_ci_total"] == 3
    assert m["n_ci_unique"] == 2  # (0,1,{}) and (0,1,{2}); third is a repeat
    assert m["n_cache_hits"] == 1
    assert m["redundancy"] == 1.0 - 2 / 3
    assert m["max_depth"] == 1
    assert m["n_colliders"] == 1
    assert m["n_rules"] == 1
    assert m["region_size"] == 3  # {0,2,3}
    assert m["boundary_size"] == 1

    frames = rec.to_frames()
    assert len(frames["ci_tests"]) == 3
    assert len(frames["runs"]) == 1
    run = frames["runs"].iloc[0]
    assert run["n_ci_unique"] == 2
    assert run["algorithm"] == "local"
    # seq is globally monotonic + unique across the event tables (runs is a summary, no seq)
    event_tables = ("ci_tests", "colliders", "rules", "regions")
    seqs = [row["seq"] for t in event_tables for row in rec.tables[t]]
    assert len(seqs) == len(set(seqs))  # unique
    assert sorted(seqs) == list(range(1, len(seqs) + 1))  # contiguous 1..N across all events


def test_file_recorder_streams_jsonl_and_loads(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    with FileRecorder(run_dir, run_id="r1", flush_every=1) as rec:
        _drive(rec)

    # one jsonl per non-empty table; every line is valid JSON carrying the run_id
    ci_lines = (run_dir / "ci_tests.jsonl").read_text().strip().splitlines()
    assert len(ci_lines) == 3
    assert all(json.loads(ln)["run_id"] == "r1" for ln in ci_lines)
    assert (run_dir / "runs.jsonl").exists()

    frames = load_run(run_dir)
    assert len(frames["ci_tests"]) == 3
    assert len(frames["regions"]) == 1
    assert frames["runs"].iloc[0]["n_cache_hits"] == 1


def test_pc_run_populates_recorder() -> None:
    import numpy as np

    from cbcd import InMemoryRecorder, pc

    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.standard_normal(n)
    x1 = x0 + rng.standard_normal(n)
    x2 = x1 + rng.standard_normal(n)  # chain X0 -> X1 -> X2
    data = np.column_stack([x0, x1, x2])

    rec = InMemoryRecorder()
    pc(data, recorder=rec, run_id="pc-test")

    m = rec.metrics()
    assert m["n_ci_total"] >= 1
    assert m["wallclock_s"] >= 0.0
    frames = rec.to_frames()
    assert len(frames["ci_tests"]) == m["n_ci_total"]
    assert (frames["ci_tests"]["depth"] >= 0).all()
    run = frames["runs"].iloc[0]
    assert run["run_id"] == "pc-test"
    assert run["algorithm"] == "pc"
    assert run["n_samples"] == n
    assert run["n_vars"] == 3
    assert run["ci_test"] == "fisherz"


def test_fci_run_records_algorithm_label() -> None:
    import numpy as np

    from cbcd import InMemoryRecorder, rfci

    rng = np.random.default_rng(1)
    n = 200
    z = rng.standard_normal(n)
    data = np.column_stack([z + rng.standard_normal(n), z + rng.standard_normal(n), z])

    rec = InMemoryRecorder()
    rfci(data, recorder=rec, run_id="rfci-test")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "rfci"  # labelled as rfci, not the underlying fci()
    assert rec.metrics()["n_ci_total"] >= 1


def test_pc_run_streams_to_file(tmp_path: Path) -> None:
    import numpy as np

    from cbcd import load_run, pc
    from cbcd.recording import FileRecorder

    rng = np.random.default_rng(2)
    n = 200
    a = rng.standard_normal(n)
    b = a + rng.standard_normal(n)
    data = np.column_stack([a, b, rng.standard_normal(n)])

    with FileRecorder(tmp_path / "run1") as rec:
        pc(data, recorder=rec, run_id="disk")  # begin_run's run_id is authoritative
    frames = load_run(tmp_path / "run1")
    assert len(frames["ci_tests"]) >= 1
    assert frames["runs"].iloc[0]["run_id"] == "disk"


def test_file_recorder_rejects_use_after_close(tmp_path: Path) -> None:
    rec = FileRecorder(tmp_path / "r2", run_id="r2")
    rec.begin_run(
        run_id="r2", algorithm="pc", params={}, n_samples=10, n_vars=3, ci_test="fisherz"
    )
    rec.finish_run(result_summary={})  # finalizes + closes
    try:
        rec.record_ci(x=0, y=1, S=(), p_value=0.5, depth=0, was_cache_hit=False)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on use after close")


def test_pcmci_run_populates_recorder() -> None:
    import numpy as np

    from cbcd import InMemoryRecorder, pcmci
    from cbcd.timeseries import LaggedDataset

    rng = np.random.default_rng(0)
    T, max_lag = 400, 2
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    for t in range(1, T):
        x1[t] = 0.6 * x0[t - 1] + 0.5 * rng.standard_normal()  # X0_{t-1} -> X1_t
    data = LaggedDataset(np.column_stack([x0, x1]), max_lag=max_lag)

    rec = InMemoryRecorder()
    pcmci(data, recorder=rec, run_id="pcmci-test")

    m = rec.metrics()
    assert m["n_ci_total"] >= 1
    frames = rec.to_frames()
    run = frames["runs"].iloc[0]
    assert run["algorithm"] == "pcmci"
    assert run["run_id"] == "pcmci-test"
    assert run["n_samples"] == T
    # node ids are lagged-grid encoded: within [0, n_series * (max_lag + 1))
    assert (frames["ci_tests"]["x"] < 2 * (max_lag + 1)).all()
    assert (frames["ci_tests"]["depth"] >= 0).all()
