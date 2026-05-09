# Vendored Third-Party Code

This directory contains third-party source code vendored into `citk` so that
optional tests can ship without requiring the user to clone external
repositories.

## indeptests/

- **Source**: <https://github.com/hpi-epic/mCMIkNN>
- **License**: MIT (see `indeptests/LICENSE`)
- **Authors**: Johannes Hügle, Christopher Hagedorn (HPI-EPIC)
- **Paper**: Hügle, Hagedorn, Uflacker. *A kNN-based Non-Parametric Conditional
  Independence Test for Mixed Data and Application in Causal Discovery.*
  ECML PKDD 2023.
- **Vendored revision**: `5a94fcc2a3b90d343c4ea0803d82b3b90b6b293d`
- **What is vendored**: only the `indeptests/` Python package (the
  `mCMIkNN` test class and its `IndependenceTest` ABC). The upstream's
  `csl/` package, experiments, and pinned dependency manifest are
  intentionally excluded.

The `indeptests` package is vendored verbatim from the upstream repository.
The MIT licence requires preservation of the copyright notice; `LICENSE` is
copied alongside the source.

### Refresh procedure

To pull a newer revision of `indeptests/` from upstream:

1. `git clone https://github.com/hpi-epic/mCMIkNN.git`
2. Copy `src/indeptests/` over `citk/_vendor/indeptests/`, preserving the
   `LICENSE` file.
3. Update the **Vendored revision** SHA above.
4. Run `pytest tests/test_ci_smoke.py` to verify the wrapper still works.
