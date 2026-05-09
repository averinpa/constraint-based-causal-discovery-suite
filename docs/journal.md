# dagsampler development journal

Per-package implementation history. Newest entry on top.

For cross-package and suite-level decisions, see
`../../journal.md`.

---

## 2026-05-08 — `as_ci_oracle()` + `DSeparationOracle`; bumped to 0.2.0

Added the cross-package boundary the suite design has been calling for:
`CausalDataGenerator.as_ci_oracle()` returns a `DSeparationOracle`
that satisfies the `cbcd.CITest` Protocol structurally — no `cbcd`
import, no inheritance, just the right shape.

**Surface:**

- `dagsampler.DSeparationOracle(dag, var_names)` — duck-typed CI test.
  Members:
  - `n_vars: int`
  - `var_names: tuple[str, ...]` (alphabetical, matches data column order)
  - `__call__(x: int, y: int, S: Sequence[int]) -> float` — returns
    `1.0` when `x ⫫ y | S` under d-separation, `0.0` otherwise.
  - `details(x, y, S)` returning a frozen `_CITestResult` with
    `.p_value`.
- `CausalDataGenerator.as_ci_oracle()` — convenience method binding
  the oracle to `self.graph` and the canonical column order. Raises
  `RuntimeError` if `simulate()` hasn't run yet (variable order isn't
  finalized until then).

**p-value convention:** `1.0` for d-separated, `0.0` for d-connected.
cbcd's PC compares `p > alpha` to declare independence (see
`cbcd/cbcd/skeleton.py:118`), so any `alpha ∈ (0, 1)` recovers the
oracle answer exactly.

**Why it's not just the precomputed `ci_oracle` table.** The existing
`store_ci_oracle` flag enumerates pairs up to a fixed
`ci_oracle_max_cond_set`. Wrapping that table as a `CITest` would
silently fail on larger conditioning sets that PC asks for on bigger
graphs. `DSeparationOracle` calls `networkx.is_d_separator` on demand,
so it is correct for any conditioning-set size.

**No cbcd dependency.** The Protocol is purely structural
(`@runtime_checkable` on cbcd's side, duck-typed on ours). Verified
end-to-end: with cbcd in scope, `isinstance(oracle, cbcd.CITest)`
returns `True`, and `cbcd.pc(data, ci_test=gen.as_ci_oracle())`
recovers the correct CPDAG on a chain X→Z→Y (no edge X—Y, Z linked
to both).

**Tests:** `tests/test_ci_oracle.py` — 9 tests covering structural
shape, n_vars/data alignment, chain/fork/collider d-sep semantics,
pre-`simulate()` error, index validation, and constructor input
validation. Full suite is 59 tests, all green.

**Version:** 0.1.0 → 0.2.0. CHANGELOG updated.

**Not yet pushed.** Per the no-push contract, public release of
0.2.0 to PyPI waits on explicit user instruction.
