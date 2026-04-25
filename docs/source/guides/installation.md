# Installation

## Quick Install

Install `citk` directly from GitHub:

```bash
pip install git+https://github.com/averinpa/citk.git
```

## Development Setup (uv)

```bash
uv sync --all-extras
```

This installs core, docs, dev, and all optional dependency groups declared in `pyproject.toml`.

## Optional Dependency Groups

The 19 shipped tests split across several optional dependency groups; install only what you need.

### Tigramite

For `cmiknn`, `cmiknn_mixed`, `regci`:

```bash
uv sync --extra tigramite
```

### pycomets (GCM family)

For `gcm`, `wgcm`, `pcm`. The pycomets package depends on `xgboost` at import time, so the extra installs both:

```bash
uv sync --extra pycomets
```

### R-Based Tests

R-backed tests are optional and require:

1. Python package `rpy2` (installed via the `r` extra)
2. The relevant R packages installed in your R environment:
   - `RCIT` from GitHub `ericstrobl/RCIT` — for `rcit`, `rcot`
   - `MXM` from CRAN — for `ci_mm`
   - `bnlearn` from CRAN — for `hartemink_chisq`

```bash
uv sync --extra r
```

## Local External Wrappers

The `mcmiknn` wrapper expects a local checkout of the upstream repository:

- `mcmiknn` -> `/Users/pavelaverin/Projects/vendor/mCMIkNN/src`

If the path is not available, the wrapper raises a clear `ImportError` when called.
