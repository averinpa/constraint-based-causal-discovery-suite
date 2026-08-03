"""RunRecorder Protocol + concrete recorders.

Audit-trail interface from the design (see ``docs/design/recorder-schema.md``). The Protocol is the
sink every algorithm accepts via ``recorder=...``; all hooks may be no-ops.

Recorders:
  * ``NullRecorder``     -- the default; every hook a no-op, near-zero overhead.
  * ``InMemoryRecorder`` -- accumulates events in per-table lists; ``to_frames()`` / ``metrics()``.
                            For unit tests and small interactive runs.
  * ``FileRecorder``     -- streams events to ``<dir>/{runs,ci_tests,colliders,rules,regions}.jsonl``
                            (append-only, crash-safe, resumable). For the benchmark sweep.

The event hooks (``record_ci`` / ``record_collider`` / ``record_rule``) are unchanged from the
original slice. ``begin_run`` / ``record_region`` / ``finish_run`` are additive with no-op defaults,
so existing recorders and call sites keep working. Every emitted row carries a monotonic ``seq`` and
a run-relative ``ts_ns`` stamped by the recorder, so timing needs no change to the hook signatures.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_TABLES = ("runs", "ci_tests", "colliders", "rules", "regions")


@runtime_checkable
class RunRecorder(Protocol):
    """Audit-trail sink. All hooks may be no-ops."""

    def begin_run(
        self,
        *,
        run_id: str,
        algorithm: str,
        params: dict[str, Any],
        n_samples: int,
        n_vars: int,
        ci_test: str,
        targets: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None: ...

    def record_ci(
        self,
        *,
        x: int,
        y: int,
        S: tuple[int, ...],
        p_value: float,
        depth: int,
        was_cache_hit: bool,
    ) -> None: ...

    def record_collider(
        self,
        *,
        triple: tuple[int, int, int],
        classification: str,
        orienter: str,
    ) -> None: ...

    def record_rule(
        self,
        *,
        rule_set: str,
        rule_name: str,
        affected_edge: tuple[int, int],
    ) -> None: ...

    def record_region(
        self,
        *,
        target: int,
        region: Sequence[int],
        boundary: Sequence[int],
    ) -> None: ...

    def finish_run(self, *, result_summary: dict[str, Any]) -> None: ...


class NullRecorder:
    """Default recorder. Every method is a no-op; near-zero overhead."""

    def begin_run(self, **_: Any) -> None:
        return

    def record_ci(self, **_: Any) -> None:
        return

    def record_collider(self, **_: Any) -> None:
        return

    def record_rule(self, **_: Any) -> None:
        return

    def record_region(self, **_: Any) -> None:
        return

    def finish_run(self, **_: Any) -> None:
        return


class _BaseRecorder:
    """Shared accumulation: monotonic ``seq``, run-relative timing, and the running counters that
    populate the ``runs`` summary. Subclasses persist a row by overriding ``_emit``."""

    def __init__(self) -> None:
        self._seq = 0
        self._t0: int | None = None
        self._meta: dict[str, Any] = {}
        self._n_ci = 0
        self._n_unique = 0
        self._n_cache = 0
        self._max_depth = 0
        self._seen: set[tuple[int, int, frozenset[int]]] = set()
        self._region: set[int] = set()
        self._boundary: set[int] = set()

    # --- subclass persistence hooks ---------------------------------------------------------
    def _emit(self, table: str, row: dict[str, Any]) -> None:
        raise NotImplementedError

    def _finalize(self) -> None:
        return

    # --- internals --------------------------------------------------------------------------
    def _rel_ns(self) -> int:
        return 0 if self._t0 is None else time.perf_counter_ns() - self._t0

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def _run_id(self) -> str | None:
        return self._meta.get("run_id")

    # --- hooks ------------------------------------------------------------------------------
    def begin_run(
        self,
        *,
        run_id: str,
        algorithm: str,
        params: dict[str, Any],
        n_samples: int,
        n_vars: int,
        ci_test: str,
        targets: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._t0 = time.perf_counter_ns()
        self._meta = {
            "run_id": run_id,
            "algorithm": algorithm,
            "params": params,
            "ci_test": ci_test,
            "n_samples": int(n_samples),
            "n_vars": int(n_vars),
            "targets": [int(t) for t in targets] if targets is not None else None,
            "seed": seed,
            "started_epoch_ns": time.time_ns(),
        }

    def record_ci(
        self,
        *,
        x: int,
        y: int,
        S: tuple[int, ...],
        p_value: float,
        depth: int,
        was_cache_hit: bool,
    ) -> None:
        self._n_ci += 1
        a, b = (int(x), int(y)) if int(x) <= int(y) else (int(y), int(x))
        key = (a, b, frozenset(int(s) for s in S))  # unordered pair, matching the CI cache
        if key not in self._seen:
            self._seen.add(key)
            self._n_unique += 1
        if was_cache_hit:
            self._n_cache += 1
        if depth > self._max_depth:
            self._max_depth = int(depth)
        self._emit(
            "ci_tests",
            {
                "run_id": self._run_id,
                "seq": self._next(),
                "x": int(x),
                "y": int(y),
                "S": [int(s) for s in S],
                "depth": int(depth),
                "p_value": float(p_value),
                "was_cache_hit": bool(was_cache_hit),
                "ts_ns": self._rel_ns(),
            },
        )

    def record_collider(
        self,
        *,
        triple: tuple[int, int, int],
        classification: str,
        orienter: str,
    ) -> None:
        a, b, c = triple
        self._emit(
            "colliders",
            {
                "run_id": self._run_id,
                "seq": self._next(),
                "a": int(a),
                "b": int(b),
                "c": int(c),
                "classification": classification,
                "orienter": orienter,
                "ts_ns": self._rel_ns(),
            },
        )

    def record_rule(
        self,
        *,
        rule_set: str,
        rule_name: str,
        affected_edge: tuple[int, int],
    ) -> None:
        u, v = affected_edge
        self._emit(
            "rules",
            {
                "run_id": self._run_id,
                "seq": self._next(),
                "rule_set": rule_set,
                "rule_name": rule_name,
                "u": int(u),
                "v": int(v),
                "ts_ns": self._rel_ns(),
            },
        )

    def record_region(
        self,
        *,
        target: int,
        region: Sequence[int],
        boundary: Sequence[int],
    ) -> None:
        reg = [int(r) for r in region]
        bnd = [int(b) for b in boundary]
        self._region.update(reg)
        self._boundary.update(bnd)
        self._emit(
            "regions",
            {
                "run_id": self._run_id,
                "seq": self._next(),
                "target": int(target),
                "region": reg,
                "boundary": bnd,
                "ts_ns": self._rel_ns(),
            },
        )

    def finish_run(self, *, result_summary: dict[str, Any]) -> None:
        self._emit(
            "runs",
            {
                **self._meta,
                "ended_rel_ns": self._rel_ns(),
                "wallclock_s": self._rel_ns() / 1e9,
                "n_ci_total": self._n_ci,
                "n_ci_unique": self._n_unique,
                "n_cache_hits": self._n_cache,
                "redundancy": (1.0 - self._n_unique / self._n_ci) if self._n_ci else 0.0,
                "max_depth": self._max_depth,
                "region_size": len(self._region) if self._region else None,
                "boundary_size": len(self._boundary) if self._boundary else None,
                "result_summary": result_summary,
            },
        )
        self._finalize()

    def _derived_metrics(self) -> dict[str, Any]:
        return {
            "n_ci_total": self._n_ci,
            "n_ci_unique": self._n_unique,
            "n_cache_hits": self._n_cache,
            "redundancy": (1.0 - self._n_unique / self._n_ci) if self._n_ci else 0.0,
            "max_depth": self._max_depth,
            "region_size": len(self._region) if self._region else None,
            "boundary_size": len(self._boundary) if self._boundary else None,
            "wallclock_s": self._rel_ns() / 1e9,
        }


class InMemoryRecorder(_BaseRecorder):
    """Accumulates events in per-table lists. For unit tests and small interactive runs; bounded by
    run size (holds every event). Not for large sweeps -- use ``FileRecorder`` there."""

    def __init__(self) -> None:
        super().__init__()
        self._tables: dict[str, list[dict[str, Any]]] = {t: [] for t in _TABLES}

    def _emit(self, table: str, row: dict[str, Any]) -> None:
        self._tables[table].append(row)

    @property
    def tables(self) -> dict[str, list[dict[str, Any]]]:
        """Raw event rows, keyed by table name."""
        return self._tables

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """The five tables as pandas DataFrames (empty frame for a table with no events)."""
        import pandas as pd

        return {t: pd.DataFrame(rows) for t, rows in self._tables.items()}

    def metrics(self) -> dict[str, Any]:
        """Derived scalar metrics for this run (the Paper 3 frontier axes)."""
        m = self._derived_metrics()
        m["n_colliders"] = len(self._tables["colliders"])
        m["n_rules"] = len(self._tables["rules"])
        return m


class FileRecorder(_BaseRecorder):
    """Streams events to ``<directory>/<table>.jsonl`` (append-only, one JSON object per line).

    Crash-safe and resumable: each event is a self-contained line, so a killed run leaves a valid
    prefix. Run many workers by giving each a distinct ``directory`` (one run per dir); aggregation
    is then a directory scan (see ``load_run``). Use as a context manager, or call ``close()`` /
    let ``finish_run`` finalize.
    """

    def __init__(
        self,
        directory: str | Path,
        run_id: str | None = None,
        *,
        flush_every: int = 1000,
    ) -> None:
        super().__init__()
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, int(flush_every))
        self._handles: dict[str, TextIO] = {}
        self._since_flush = 0
        self._closed = False
        self._meta["run_id"] = run_id if run_id is not None else uuid.uuid4().hex

    def _handle(self, table: str) -> TextIO:
        h = self._handles.get(table)
        if h is None:
            # Persistent per-table handle held open across events for streaming appends; closed in
            # close() / finish_run(). A context manager does not fit this lifetime.
            h = open(self._dir / f"{table}.jsonl", "a", encoding="utf-8")  # noqa: SIM115
            self._handles[table] = h
        return h

    def _emit(self, table: str, row: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("FileRecorder is closed")
        line = json.dumps(row, default=str, separators=(",", ":"))
        self._handle(table).write(line + "\n")
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        for h in self._handles.values():
            h.flush()
        self._since_flush = 0

    def _finalize(self) -> None:
        self.close()

    def begin_run(self, **kwargs: Any) -> None:
        # Preserve a caller-supplied run_id if begin_run is called without one.
        prior = self._meta.get("run_id")
        super().begin_run(**kwargs)
        if not kwargs.get("run_id") and prior:
            self._meta["run_id"] = prior

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        for h in self._handles.values():
            h.close()
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> FileRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_run(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load a ``FileRecorder`` run directory into the five DataFrames (empty frame if a table is
    absent). The offline entry point for computing derived metrics / accuracy across a sweep."""
    import pandas as pd

    d = Path(directory)
    out: dict[str, pd.DataFrame] = {}
    for t in _TABLES:
        f = d / f"{t}.jsonl"
        out[t] = pd.read_json(f, lines=True) if f.exists() and f.stat().st_size else pd.DataFrame()
    return out


def _resolve_recorder(recorder: RunRecorder | None) -> RunRecorder:
    return recorder if recorder is not None else NullRecorder()
