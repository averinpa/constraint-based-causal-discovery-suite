# Examples

Example notebooks demonstrating dagsampler usage.

These notebooks are tracked in the repo and referenced from the documentation.
They are not packaged into the wheel.

## Running

Install the `examples` extra to get Jupyter and matplotlib:

```bash
uv pip install -e ".[examples]"
jupyter lab examples/
```

Or from PyPI:

```bash
pip install "dagsampler[examples]"
jupyter lab
```

## Suggested topics

- Quick start: minimal custom DAG and inspecting `simulate()` output
- Random DAG generation with reproducible structure / data seeds
- Mixed-type nodes (continuous, binary, categorical)
- Nonlinear functional forms (`sigmoid`, `cos`, `sin`, `polynomial`, `interaction`)
- Post-nonlinear transforms (`tanh`, `sin`, `cos`, `exp_neg_abs`, `sqrt_abs`, `relu`, `sign`)
- Heavy-tailed and uniform additive noise
- Heteroskedastic noise
- Cross-type mechanisms (categorical -> continuous via `stratum_means`, continuous -> categorical via `threshold`)
- Template DAGs (`chain_config`, `fork_config`, `collider_config`, `independence_config`)
- CI oracle output for benchmarking
