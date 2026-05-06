"""End-to-end PC with Fisher-Z on synthetic linear-Gaussian data."""

from __future__ import annotations

import numpy as np

from cbcd import pc
from cbcd.graph.marks import EndpointMark
from tests.fixtures import ALL_FIXTURES


def _sample_linear_gaussian(
    dag,  # noqa: ANN001
    n_samples: int,
    rng: np.random.Generator,
    coef_low: float = 0.5,
    coef_high: float = 1.5,
    noise_sd: float = 1.0,
) -> np.ndarray:
    """Sample n rows from a linear-Gaussian SCM with the given DAG topology."""
    n = dag.n_vars
    # Topological order: parents first.
    parents = {
        i: [
            j
            for j in range(n)
            if dag.endpoints[j, i] == EndpointMark.ARROW
            and dag.endpoints[i, j] == EndpointMark.TAIL
        ]
        for i in range(n)
    }
    indeg = {i: len(parents[i]) for i in range(n)}
    order: list[int] = []
    available = [i for i in range(n) if indeg[i] == 0]
    deg = dict(indeg)
    children = {
        i: [
            j
            for j in range(n)
            if dag.endpoints[i, j] == EndpointMark.ARROW
            and dag.endpoints[j, i] == EndpointMark.TAIL
        ]
        for i in range(n)
    }
    while available:
        u = available.pop()
        order.append(u)
        for v in children[u]:
            deg[v] -= 1
            if deg[v] == 0:
                available.append(v)
    # Coefficients per (parent, child) edge.
    coefs: dict[tuple[int, int], float] = {}
    for child, ps in parents.items():
        for p in ps:
            sign = 1 if rng.random() > 0.5 else -1
            coefs[(p, child)] = sign * rng.uniform(coef_low, coef_high)

    data = np.zeros((n_samples, n), dtype=np.float64)
    for v in order:
        noise = rng.normal(0.0, noise_sd, size=n_samples)
        contribution = noise.copy()
        for p in parents[v]:
            contribution = contribution + coefs[(p, v)] * data[:, p]
        data[:, v] = contribution
    return data


def test_pc_fisherz_recovers_y_structure() -> None:
    rng = np.random.default_rng(0)
    dag, expected = ALL_FIXTURES["y_structure"]()
    data = _sample_linear_gaussian(dag, 20000, rng)
    recovered = pc(data, alpha=0.01)
    assert np.array_equal(recovered.endpoints, expected.endpoints)


def test_pc_fisherz_recovers_chain() -> None:
    rng = np.random.default_rng(1)
    dag, expected = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 20000, rng)
    recovered = pc(data, alpha=0.01)
    assert np.array_equal(recovered.endpoints, expected.endpoints)


def test_pc_fisherz_recovers_diamond() -> None:
    rng = np.random.default_rng(2)
    dag, expected = ALL_FIXTURES["diamond"]()
    data = _sample_linear_gaussian(dag, 20000, rng)
    recovered = pc(data, alpha=0.01)
    assert np.array_equal(recovered.endpoints, expected.endpoints)
