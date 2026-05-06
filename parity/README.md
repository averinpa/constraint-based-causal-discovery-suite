# Parity harnesses

These scripts compare `cbcd`'s outputs against the reference implementations:

- `parity/pc/run.py` — `cbcd.pc()` vs `causal-learn.pc()`
- `parity/fci/run.py` — `cbcd.fci()` vs `causal-learn.fci()`
- `parity/pcmci/run.py` — `cbcd.pcmci()` vs `tigramite` PCMCI
- `parity/run_all.py` — runs all three

These are **developer tools**, not part of the cbcd package. The reference
libraries are NOT runtime or test dependencies of cbcd; they must be
installed separately.

## Setup

```bash
uv pip install causal-learn tigramite
```

These ARE pulled into the venv (so `uv run` finds them) but are NOT in
`pyproject.toml`, so a fresh `uv sync` will not re-install them. Re-run
the install command after `uv sync` if you want to repeat the parity
checks.

## Running

```bash
uv run python -m parity.pc.run      # PC family
uv run python -m parity.fci.run     # FCI family
uv run python -m parity.pcmci.run   # PCMCI family
uv run python -m parity.run_all     # all three with aggregate
```

Each harness exits 0 iff there are no "real disagreements" — cases where
both `cbcd` and the reference match the d-sep / m-sep truth but produce
different endpoint matrices. Disagreements where the reference has a
known bug (audit `docs/audit_causal_learn.md`) are reported separately.

## What the harnesses test

For each fixture from `tests/fixtures.py`, `tests/fixtures_pag.py`, or
the VAR table in `parity/pcmci/run.py`:

1. Simulate Linear-Gaussian data (or VAR data for PCMCI) from the true
   DAG with a fixed seed.
2. Run cbcd and the reference on the same data with the same alpha and
   Fisher-Z / ParCorr CI test.
3. Convert the reference's output to cbcd's endpoint convention.
4. Compute three SHDs:
   - `cbcd vs truth` — should be 0 by the d-sep regression bar.
   - `reference vs truth` — informational; may be > 0 on bugs.
   - `cbcd vs reference` — the parity number.

## Endpoint mark conventions

`parity/_compare.py` defines the conversions:

- **causal-learn / Tetrad**: `graph[i, j]` = mark at vertex `i` of edge
  `{i, j}`. Marks: `0` = no edge, `1` = ARROW, `-1` = TAIL, `2` = CIRCLE.
- **tigramite**: `graph[i, j, tau]` is a string: `''`, `'-->'`, `'<--'`,
  `'o-o'`, `'o->'`, `'<-o'`, etc., describing the edge from `i` (lag
  `-tau`) to `j` (lag 0).
- **cbcd**: `endpoints[i, j]` (i.i.d.) = mark at vertex `j`. Marks:
  `0` = NO_EDGE, `1` = TAIL, `2` = ARROW, `3` = CIRCLE. For time series,
  `endpoints[tau, i, j]` = mark at present-time end `j` of edge
  `i_{t-tau} → j_t`.
