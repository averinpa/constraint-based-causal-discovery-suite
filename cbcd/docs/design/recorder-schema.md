# RunRecorder — concrete recorders & metrics schema

Companion to `../roadmap.md` (Phase 0) and `cbcd/recording.py` (§J). Today only the no-op
`NullRecorder` ships; this doc specifies the concrete `InMemoryRecorder` and `FileRecorder` and the
event/metric schema they emit. The driving requirement is the Paper 3 **cost-vs-accuracy frontier**
for local discovery: every axis of that frontier must be derivable from what the recorder captures.

## What the frontier needs → what the recorder must capture

| Frontier axis | Captured from |
|---|---|
| CI-test count (total / unique) | `record_ci` event stream |
| Conditioning-set order distribution | `depth` on each `record_ci` |
| Redundancy (repeated CI queries) | multiplicity of `(x, y, S)` + `was_cache_hit` |
| Region size / boundary (local) | `record_region` (new) |
| Wallclock (run + per phase) | recorder-stamped timestamps + `begin_run`/`finish_run` |
| Orientation effort | `record_collider` + `record_rule` counts |
| Accuracy (SHD / F1 / orientation) | *not* the recorder's job — computed by comparing the returned graph to ground truth; the recorder stores a `result_summary` handle only |

## Event model

Keep the three existing hooks **unchanged** (backward compatible). Add three **lifecycle/region**
hooks, all with no-op defaults on `NullRecorder`, so no existing call site changes:

```python
# existing (cbcd/recording.py) — unchanged
record_ci(*, x, y, S, p_value, depth, was_cache_hit) -> None
record_collider(*, triple, classification, orienter) -> None
record_rule(*, rule_set, rule_name, affected_edge) -> None

# proposed additions
begin_run(*, run_id, algorithm, params, n_samples, n_vars, ci_test,
          targets=None, seed=None) -> None
record_region(*, target, region, boundary) -> None      # local algorithms only
finish_run(*, result_summary) -> None
```

The recorder stamps every event internally with a monotonic `seq` and a `perf_counter_ns`
timestamp, so per-event timing and wallclock come for free without widening the hook signatures.

## Output schema (long-format tables)

One `FileRecorder` writes one run into a directory `runs/<run_id>/` with four tables
(parquet; jsonl mirror optional for streaming). All share `run_id`; `seq` is the global monotonic
event counter, giving a total order across tables.

### `runs` — one row per run
| column | type | notes |
|---|---|---|
| run_id | str | uuid or caller-supplied |
| algorithm | str | e.g. `pc`, `fci`, `local`, `local_latent` |
| params | json | knobs: alpha, max_depth, orientation mode, … |
| ci_test | str | e.g. `fisherz`, `gfcm` |
| n_samples | int | |
| n_vars | int | |
| targets | list[int]/null | query set for local; null for global |
| seed | int/null | |
| started_ns / ended_ns | int | perf_counter_ns |
| wallclock_s | float | derived |
| n_ci_total / n_ci_unique / n_cache_hits | int | derived from `ci_tests` |
| max_depth | int | derived |
| region_size / boundary_size | int/null | local; derived from `regions` |
| result_summary | json | edge list / mark matrix handle, #edges, #undirected, … |

### `ci_tests` — one row per `record_ci`
| column | type |
|---|---|
| run_id, seq | str, int |
| x, y | int, int |
| S | list[int] |
| depth | int (adjacency-search depth; `len(S)` in PC) |
| p_value | float |
| was_cache_hit | bool |
| ts_ns | int (recorder-stamped) |

### `colliders` — one row per `record_collider`
| column | type |
|---|---|
| run_id, seq | str, int |
| a, b, c | int (the triple) |
| classification | str (`collider` / `noncollider` / `ambiguous`) |
| orienter | str (rule/step that decided) |

### `rules` — one row per `record_rule`
| column | type |
|---|---|
| run_id, seq | str, int |
| rule_set | str (`meek` / `fci`) |
| rule_name | str (`R1`…`R10`) |
| u, v | int (affected edge) |

### `regions` — one row per `record_region` (local only)
| column | type |
|---|---|
| run_id, seq | str, int |
| target | int |
| region | list[int] |
| boundary | list[int] |

## Recorder implementations

- **`NullRecorder`** (exists) — every hook a no-op; the default, near-zero overhead. Benchmarks that
  don't need an audit trail pay nothing.
- **`InMemoryRecorder`** — accumulates events in per-table lists; `.to_frames()` returns the four
  DataFrames, `.metrics()` returns the derived scalars below. For unit tests and small interactive
  runs. Bounded by run size; not for 10^5-cell sweeps.
- **`FileRecorder(dir, run_id, *, flush_every=...)`** — buffers and streams events to
  `runs/<run_id>/{runs,ci_tests,colliders,rules,regions}.parquet`. For the benchmark: many workers,
  one dir per run, safe to stop/resume. Never holds a whole sweep in memory.

## Derived metrics (computed offline from the tables)

Pure functions of the event tables — never recomputed inside the hot loop:
- `n_ci_total = len(ci_tests)`; `n_ci_unique = nunique((x, y, frozenset(S)))`
- `redundancy = 1 - n_ci_unique / n_ci_total` (equivalently the cache-hit rate)
- `order_hist = histogram(ci_tests.depth)` — the power-vs-conditioning-depth axis
- `region_size = len(region)`, `boundary_size = len(boundary)` per target (local)
- `wallclock_s`, and optional per-phase timing from `seq`-ordered `ts_ns` deltas
- orientation effort = `len(colliders)`, `len(rules)`

Accuracy metrics (SHD, MB-F1, orientation accuracy) are computed separately by joining
`result_summary` against ground truth — kept out of the recorder so the recorder stays a pure sink.

## Overhead & concurrency

- Hot-path cost is one attribute write + list append (InMemory) or one buffered row (File). Keep
  serialization (parquet flush) off the hot path — batch on `flush_every` and at `finish_run`.
- One recorder instance per run; do not share across worker processes. Parallel sweeps write
  disjoint `runs/<run_id>/` dirs, so aggregation is a directory scan (same pattern as the GFCM
  benchmark's per-cell parquet).
- The Protocol stays `runtime_checkable`; new hooks are additive with no-op defaults, so third-party
  recorders and existing call sites keep working.

## Open questions

1. **Per-CI timing** — stamp `ts_ns` on every `record_ci` (enables per-test cost), or only phase
   boundaries? Per-event is more informative but adds a `perf_counter_ns` call per CI test; measure
   the overhead before committing.
2. **Result storage** — store the full mark matrix in `result_summary`, or a path to a separate
   graph artifact? Lean toward a compact edge/mark encoding inline; large graphs → external handle.
3. **jsonl mirror** — worth it for live streaming/debugging, or parquet-only to keep it simple?
