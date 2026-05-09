"""Shared base for graph types."""

from __future__ import annotations

from abc import ABC

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark


class _GraphBase(ABC):
    """Endpoint-mark int8 matrix storage shared by DAG / CPDAG / PAG types.

    Subclasses validate which marks are permitted and which edge configurations
    are well-formed.
    """

    n_vars: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
    ) -> None:
        if n_vars < 0:
            raise CBCDInputError(f"n_vars must be non-negative, got {n_vars}")
        if endpoints is None:
            endpoints = np.zeros((n_vars, n_vars), dtype=np.int8)
        else:
            endpoints = np.ascontiguousarray(endpoints, dtype=np.int8)
            if endpoints.shape != (n_vars, n_vars):
                raise CBCDInputError(
                    f"endpoints shape {endpoints.shape} does not match n_vars={n_vars}"
                )
        if var_names is not None and len(var_names) != n_vars:
            raise CBCDInputError(
                f"var_names length {len(var_names)} does not match n_vars={n_vars}"
            )
        if n_vars > 0 and np.any(np.diag(endpoints) != EndpointMark.NO_EDGE):
            raise CBCDInputError("self-loops are not allowed")
        no_edge_one_side = (endpoints == EndpointMark.NO_EDGE) ^ (
            endpoints.T == EndpointMark.NO_EDGE
        )
        if np.any(no_edge_one_side):
            raise CBCDInputError(
                "endpoint matrix is asymmetric in NO_EDGE: edge presence must agree on both sides"
            )

        self.n_vars = n_vars
        self.endpoints = endpoints
        self.var_names = var_names

    def has_edge(self, i: int, j: int) -> bool:
        return bool(self.endpoints[i, j] != EndpointMark.NO_EDGE)

    def adjacency(self) -> NDArray[np.bool_]:
        return self.endpoints != EndpointMark.NO_EDGE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _GraphBase):
            return NotImplemented
        if type(self) is not type(other):
            return False
        if self.n_vars != other.n_vars:
            return False
        return bool(np.array_equal(self.endpoints, other.endpoints))

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.n_vars, self.endpoints.tobytes()))
