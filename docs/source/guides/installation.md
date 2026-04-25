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

This installs core, docs, dev, and optional R integration dependencies declared in `pyproject.toml`.

## Optional LightGBM Extra

LightGBM is not required for core `citk` functionality. Install only if you want to pass LightGBM models explicitly:

```bash
uv sync --extra ml
```

## Optional Tigramite Extra

For tigramite wrappers (`cmiknn`, `cmiknn_mixed`, `regci`):

```bash
uv sync --extra tigramite
```

## Local External Wrappers

Two optional wrappers currently expect local repositories:

1. `mcmiknn` -> `/Users/pavelaverin/Projects/vendor/mCMIkNN/src`
2. `dct` -> `/Users/pavelaverin/Projects/vendor/DCT`

If those paths are not available, these wrappers raise a clear `ImportError` when called.

## Optional R-Based Setup

R-backed tests are optional. They require:

1. Python package `rpy2` (already in the `r` extra)
2. R package `RCIT` from GitHub repository `ericstrobl/RCIT`
3. R package `bnlearn` from CRAN (for `hartemink_chisq`)

Example:

```bash
uv sync --extra r
```
