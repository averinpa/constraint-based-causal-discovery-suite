"""Time-series structure, lagged d-separation oracle (Phase 1) + stationary SVAR data (Phase 2).

A stationary time-series causal graph is a structured *static* DAG once unrolled over time, so this
module reuses the same networkx d-separation machinery as :mod:`dagsampler.oracle`. It provides:

* :class:`TimeSeriesSpec` -- a lagged + contemporaneous edge structure over ``n_vars`` series with an
  optional latent suffix.
* :func:`random_ts_spec` -- a bounded-degree random generator (expected degree constant as the number
  of series grows, i.e. edge density ``degree / n_vars``).
* :func:`unroll` -- expand the spec into a static ``networkx.DiGraph`` over ``(var, t)`` nodes.
* :class:`LaggedDSeparationOracle` -- a lagged d-separation oracle that answers m-separation on the
  observed marginal (latents unrolled but never queried), duck-typed to the ``cbcd.timeseries``
  ``LaggedCITest`` Protocol (``n_vars``, ``max_lag``, ``__call__(x, y, S) -> float``, ``details``);
  no import of or dependency on ``cbcd``.

Phase 2 -- stationary mixed-type SVAR *data* generation:

* :class:`SVARParams` -- innovation distribution (gaussian / student_t / laplace / cauchy / ... , the
  same names :mod:`dagsampler.causal_sim` uses), noise model, coefficient band, per-series type mix,
  and stability policy.
* :func:`simulate_svar` -- draw one long stationary series from a :class:`TimeSeriesSpec`. Coefficients
  are tied across time slices (one weight per structural edge, shared over all ``t``); a spectral-radius
  guard on the reduced-form companion matrix keeps the linear core stationary; a burn-in discards the
  transient. Mixed-type series use identity / threshold / discretization links. Returns the observed
  ``T x n_observed`` frame plus the matching :class:`LaggedDSeparationOracle` ground truth.
* :class:`TimeSeriesDataGenerator` -- a thin config-dict wrapper mirroring
  :class:`dagsampler.causal_sim.CausalDataGenerator`'s ergonomics.

Unlike the unroll-and-vectorize-over-replicas route sketched in the design note, Phase 2 runs a genuine
``O(T)`` recurrence so the sampled series has ``n`` effective observations equal to its length -- what a
finite-sample time-series CI benchmark needs. See ``docs/design/timeseries-support.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.d_separation import is_d_separator


@dataclass(frozen=True, slots=True)
class LaggedVar:
    """A variable at a lag. ``lag <= 0``: ``0`` is the present, ``-tau`` is ``tau`` steps in the past.

    Duck-compatible with ``cbcd.timeseries.LaggedVar`` (fields ``var``, ``lag``); the oracle also
    accepts any object exposing ``.var`` and ``.lag``, so cbcd's own ``LaggedVar`` works unchanged.
    """

    var: int
    lag: int


@dataclass(frozen=True, slots=True)
class LaggedCITestResult:
    """Result type; only ``p_value`` is read by most callers. The optional fields mirror the
    ``cbcd`` result shape so the oracle can be consumed through cbcd's grid wrapper unchanged."""

    p_value: float
    statistic: float | None = None
    df: int | None = None
    n_effective: int | None = None


@dataclass
class TimeSeriesSpec:
    """A stationary time-series causal structure.

    ``n_vars`` total series (the last ``n_latent`` are latent, i.e. unobserved). ``lagged_edges`` are
    ``(i, j, tau)`` with ``tau >= 1`` meaning ``i`` at ``t - tau`` causes ``j`` at ``t``.
    ``contemp_edges`` are ``(i, j)`` meaning ``i`` causes ``j`` at the same ``t``; they must be acyclic
    within a time slice.
    """

    n_vars: int
    max_lag: int
    lagged_edges: list[tuple[int, int, int]] = field(default_factory=list)
    contemp_edges: list[tuple[int, int]] = field(default_factory=list)
    n_latent: int = 0

    def __post_init__(self) -> None:
        if self.n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {self.n_vars}")
        if not 0 <= self.n_latent < self.n_vars:
            raise ValueError(f"n_latent must be in [0, n_vars), got {self.n_latent}")
        if self.max_lag < 0:
            raise ValueError(f"max_lag must be >= 0, got {self.max_lag}")
        for i, j, tau in self.lagged_edges:
            if not (0 <= i < self.n_vars and 0 <= j < self.n_vars):
                raise ValueError(f"lagged edge ({i},{j},{tau}) out of range")
            if not 1 <= tau <= self.max_lag:
                raise ValueError(f"lagged edge tau={tau} must be in [1, {self.max_lag}]")
        for i, j in self.contemp_edges:
            if not (0 <= i < self.n_vars and 0 <= j < self.n_vars) or i == j:
                raise ValueError(f"contemp edge ({i},{j}) invalid")

    @property
    def n_observed(self) -> int:
        return self.n_vars - self.n_latent


def random_ts_spec(
    n_observed: int,
    max_lag: int,
    *,
    degree: float = 2.0,
    n_latent: int = 0,
    seed: int | None = None,
) -> TimeSeriesSpec:
    """Random bounded-degree spec: edge density ``degree / n_vars`` so expected degree stays ~constant
    as ``n_vars`` grows. Contemporaneous edges follow a random topological order (acyclic); lagged
    edges point forward in time."""
    rng = np.random.default_rng(seed)
    n = n_observed + n_latent
    density = min(degree / max(n, 1), 0.95)
    order = list(rng.permutation(n))
    pos = {v: k for k, v in enumerate(order)}
    contemp = [
        (a, b)
        for a in range(n)
        for b in range(n)
        if a != b and pos[a] < pos[b] and rng.random() < density
    ]
    lagged = [
        (i, j, tau)
        for i in range(n)
        for j in range(n)
        for tau in range(1, max_lag + 1)
        if rng.random() < density
    ]
    return TimeSeriesSpec(
        n_vars=n, max_lag=max_lag, lagged_edges=lagged, contemp_edges=contemp, n_latent=n_latent
    )


def _horizon(max_lag: int) -> tuple[int, int]:
    """(t_horizon, t_ref): a window deep enough that every queried lag maps to an interior time."""
    return max(6 * max_lag, 12), max(3 * max_lag, 6)


def unroll(spec: TimeSeriesSpec, t_horizon: int | None = None) -> nx.DiGraph:
    """Unroll the spec into a static DAG over ``(var, t)`` nodes for ``t in [0, t_horizon]``.
    Raises if the contemporaneous structure is cyclic (the unrolled graph would not be a DAG)."""
    th = _horizon(spec.max_lag)[0] if t_horizon is None else t_horizon
    g = nx.DiGraph()
    for v in range(spec.n_vars):
        for t in range(th + 1):
            g.add_node((v, t))
    for i, j in spec.contemp_edges:
        for t in range(th + 1):
            g.add_edge((i, t), (j, t))
    for i, j, tau in spec.lagged_edges:
        for t in range(tau, th + 1):
            g.add_edge((i, t - tau), (j, t))
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("contemporaneous edges induce a cycle; the unrolled graph is not a DAG")
    return g


class LaggedDSeparationOracle:
    """Lagged d-separation oracle over an unrolled time-series DAG.

    Answers m-separation on the *observed* marginal: latents are present in the unrolled graph (so
    they open/close paths correctly) but are never queried or conditioned on. ``p_value`` is 1.0 when
    ``x`` and ``y`` are d-separated given ``S`` in the unrolled graph, else 0.0. Structurally conforms
    to the ``cbcd.timeseries`` ``LaggedCITest`` Protocol without importing cbcd.
    """

    def __init__(self, spec: TimeSeriesSpec, *, t_horizon: int | None = None) -> None:
        self._spec = spec
        self.n_vars: int = spec.n_observed
        self.max_lag: int = spec.max_lag
        self._th, self._tref = _horizon(spec.max_lag)
        if t_horizon is not None:
            self._th = t_horizon
        self._g = unroll(spec, self._th)

    def _node(self, lv: LaggedVar) -> tuple[int, int]:
        if not 0 <= lv.var < self.n_vars:
            raise IndexError(f"var {lv.var} not observed (n_vars={self.n_vars})")
        if not -self.max_lag <= lv.lag <= 0:
            raise ValueError(f"lag {lv.lag} outside [-{self.max_lag}, 0]")
        t = self._tref + lv.lag
        if not 0 <= t <= self._th:
            raise ValueError(f"{lv} maps to t={t} outside [0, {self._th}]")
        return (lv.var, t)

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult:
        xn, yn = self._node(x), self._node(y)
        if xn == yn:
            raise ValueError("x and y map to the same node")
        cond = {self._node(s) for s in S}
        sep = bool(is_d_separator(self._g, {xn}, {yn}, cond))
        return LaggedCITestResult(p_value=1.0 if sep else 0.0)

    def is_ancestor(self, a: LaggedVar, b: LaggedVar) -> bool:
        """True if ``a`` is an ancestor of (or equal to) ``b`` in the unrolled graph -- the ground
        truth for endpoint-mark soundness checks."""
        an, bn = self._node(a), self._node(b)
        return an == bn or bn in nx.descendants(self._g, an)


# ============================================================================ Phase 2: SVAR data
# Innovation samplers, all zero-location (mean-zero where the mean exists) so the linear recurrence
# stays centred and does not drift. Names match dagsampler.causal_sim's continuous-distribution set.

def _innovations(dist: str, rng: np.random.Generator, shape: tuple[int, int], p: dict) -> np.ndarray:
    """Draw an ``n x k`` innovation array from ``dist`` (centred where a mean exists)."""
    if dist == "gaussian":
        return rng.normal(0.0, p.get("std", 1.0), size=shape)
    if dist == "student_t":
        return rng.standard_t(p.get("df", 5), size=shape) * p.get("scale", 1.0)
    if dist == "laplace":
        return rng.laplace(0.0, p.get("scale", 1.0), size=shape)
    if dist == "uniform":
        a = p.get("scale", 1.0)
        return rng.uniform(-a, a, size=shape)
    if dist == "cauchy":  # no finite mean/variance -- the heavy-tail stressor for GFCM-style tests
        return rng.standard_cauchy(size=shape) * p.get("scale", 1.0)
    if dist == "gamma":  # centred so the innovation is mean-zero but right-skewed
        shp, scl = p.get("shape", 2.0), p.get("scale", 1.0)
        return rng.gamma(shp, scl, size=shape) - shp * scl
    if dist == "exponential":
        scl = p.get("scale", 1.0)
        return rng.exponential(scl, size=shape) - scl
    raise ValueError(f"unknown innovation distribution {dist!r}")


@dataclass
class SVARParams:
    """Parameters for :func:`simulate_svar`.

    Coefficients are drawn once per structural edge (tied across time) from ``[coef_low, coef_high]``
    with magnitude at least ``coef_min_abs``; contemporaneous coefficients use ``contemp_coef_scale``.
    ``innovation`` selects the driving-noise distribution (``innovation_params`` carries ``std`` /
    ``df`` / ``scale`` / ``shape``). ``noise_model`` is ``additive`` or ``heteroskedastic`` (innovation
    scaled by ``0.1 + 0.5 * |predictor|``). Series types default to continuous; ``binary_frac`` /
    ``categorical_frac`` randomly assign discrete types (or pin them with ``var_types``). ``stationarity
    = 'spectral'`` rescales the lagged coefficients until the reduced-form companion spectral radius is
    ``<= target_spectral_radius``; ``'burn_in'`` relies on the transient discard alone.
    """

    n_timesteps: int = 2000
    burn_in: int = 200
    seed: int | None = None
    coef_low: float = -0.6
    coef_high: float = 0.6
    coef_min_abs: float = 0.15
    contemp_coef_scale: float = 0.4
    innovation: str = "gaussian"
    innovation_params: dict = field(default_factory=dict)
    noise_model: str = "additive"  # additive | heteroskedastic
    mechanism: str = "linear"  # linear | tanh (bounded nonlinear lag term)
    binary_frac: float = 0.0
    categorical_frac: float = 0.0
    cardinality: int = 3
    var_types: dict[int, str] | None = None  # explicit {var: 'continuous'|'binary'|'categorical'}
    stationarity: str = "spectral"  # spectral | burn_in
    target_spectral_radius: float = 0.9


def _assign_types(spec: TimeSeriesSpec, params: SVARParams, rng: np.random.Generator) -> dict[int, str]:
    """Fixed per-series type (stationary: a series keeps one type across all ``t``)."""
    if params.var_types is not None:
        types = {v: params.var_types.get(v, "continuous") for v in range(spec.n_vars)}
    else:
        types = {}
        for v in range(spec.n_vars):
            u = rng.random()
            if u < params.binary_frac:
                types[v] = "binary"
            elif u < params.binary_frac + params.categorical_frac:
                types[v] = "categorical"
            else:
                types[v] = "continuous"
    for v in range(spec.n_vars):  # invalid check
        if types[v] not in ("continuous", "binary", "categorical"):
            raise ValueError(f"var {v}: bad type {types[v]!r}")
    return types


def _contemp_order(spec: TimeSeriesSpec) -> list[int]:
    """Topological order of series under the contemporaneous edges (raises on a cycle)."""
    g = nx.DiGraph()
    g.add_nodes_from(range(spec.n_vars))
    g.add_edges_from(spec.contemp_edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("contemporaneous edges are cyclic")
    return list(nx.topological_sort(g))


def _tie_coefficients(
    spec: TimeSeriesSpec, params: SVARParams, rng: np.random.Generator
) -> tuple[dict, dict]:
    """One coefficient per structural edge (shared across all time slices)."""

    def draw(scale_lo: float, scale_hi: float) -> float:
        for _ in range(100):
            w = float(rng.uniform(scale_lo, scale_hi))
            if abs(w) >= params.coef_min_abs:
                return w
        return params.coef_min_abs if w >= 0 else -params.coef_min_abs

    lagged = {(i, j, tau): draw(params.coef_low, params.coef_high) for i, j, tau in spec.lagged_edges}
    contemp = {
        (i, j): draw(-params.contemp_coef_scale, params.contemp_coef_scale)
        for i, j in spec.contemp_edges
    }
    return lagged, contemp


def _companion_spectral_radius(
    spec: TimeSeriesSpec, lagged: dict, contemp: dict
) -> float:
    """Spectral radius of the reduced-form VAR companion matrix for the linear core.

    Structural form ``(I - Cc) X_t = sum_tau M_tau X_{t-tau} + eps`` gives reduced-form blocks
    ``B_tau = (I - Cc)^{-1} M_tau``; stationarity iff the companion of ``[B_1..B_L]`` has radius < 1.
    ``I - Cc`` is unit-triangular in contemporaneous topological order, hence always invertible.
    """
    n, L = spec.n_vars, spec.max_lag
    if L == 0:
        return 0.0
    Cc = np.zeros((n, n))
    for (i, j), c in contemp.items():
        Cc[j, i] = c
    inv = np.linalg.inv(np.eye(n) - Cc)
    blocks = []
    for tau in range(1, L + 1):
        M = np.zeros((n, n))
        for (i, j, t), a in lagged.items():
            if t == tau:
                M[j, i] = a
        blocks.append(inv @ M)
    companion = np.zeros((n * L, n * L))
    companion[:n, :] = np.hstack(blocks)
    if L > 1:
        companion[n:, : n * (L - 1)] = np.eye(n * (L - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _stabilize(spec: TimeSeriesSpec, lagged: dict, contemp: dict, target: float) -> tuple[dict, float]:
    """Rescale lagged coefficients until the companion spectral radius is ``<= target``."""
    radius = _companion_spectral_radius(spec, lagged, contemp)
    for _ in range(200):
        if radius <= target or radius == 0.0:
            break
        factor = target / radius * 0.98
        lagged = {k: v * factor for k, v in lagged.items()}
        radius = _companion_spectral_radius(spec, lagged, contemp)
    return lagged, radius


def _cat_thresholds(k: int) -> np.ndarray:
    """Fixed standard-normal cut points splitting a latent index into ``k`` roughly equal bins."""
    from scipy.stats import norm  # local import: scipy is already a dagsampler dependency

    return norm.ppf(np.arange(1, k) / k)


def simulate_svar(spec: TimeSeriesSpec, params: SVARParams | None = None) -> dict[str, Any]:
    """Draw one long stationary mixed-type SVAR series from ``spec``.

    Returns a dict with ``"data"`` (a ``T x n_observed`` :class:`pandas.DataFrame`, latent series
    dropped, columns the observed variable indices), ``"spec"``, ``"oracle"`` (the matching
    :class:`LaggedDSeparationOracle` ground truth), ``"coefficients"`` (tied lagged + contemporaneous),
    ``"var_types"``, and ``"spectral_radius"`` of the stabilized linear core.
    """
    params = params or SVARParams()
    rng_s = np.random.default_rng(params.seed)  # structure/coefficients
    rng_d = np.random.default_rng(None if params.seed is None else params.seed + 1)  # innovations

    n, L = spec.n_vars, spec.max_lag
    order = _contemp_order(spec)
    types = _assign_types(spec, params, rng_s)
    lagged, contemp = _tie_coefficients(spec, params, rng_s)
    radius = 0.0
    if params.stationarity == "spectral":
        lagged, radius = _stabilize(spec, lagged, contemp, params.target_spectral_radius)
    elif params.stationarity != "burn_in":
        raise ValueError(f"unknown stationarity policy {params.stationarity!r}")

    # incoming lagged edges grouped by target series, for the recurrence inner loop
    incoming: dict[int, list[tuple[int, int, float]]] = {v: [] for v in range(n)}
    for (i, j, tau), a in lagged.items():
        incoming[j].append((i, tau, a))
    incoming_c: dict[int, list[tuple[int, float]]] = {v: [] for v in range(n)}
    for (i, j), c in contemp.items():
        incoming_c[j].append((i, c))

    total = params.burn_in + params.n_timesteps
    eps = _innovations(params.innovation, rng_d, (total + L, n), params.innovation_params)
    thr = {v: _cat_thresholds(params.cardinality) for v in range(n) if types[v] == "categorical"}

    X = np.zeros((total + L, n))  # first L rows are the zero warm-start (absorbed by burn-in)

    def link(v: int, latent: float) -> float:
        if types[v] == "continuous":
            return latent
        if types[v] == "binary":
            return 1.0 if latent > 0.0 else 0.0
        return float(np.digitize(latent, thr[v]))  # categorical -> 0..k-1

    def lag_term(a: float, past: float) -> float:
        return a * np.tanh(past) if params.mechanism == "tanh" else a * past

    for t in range(L, total + L):
        for v in order:
            pred = 0.0
            for i, tau, a in incoming[v]:
                pred += lag_term(a, X[t - tau, i])
            for i, c in incoming_c[v]:
                pred += c * X[t, i]  # i precedes v in contemporaneous order -> already set
            noise = eps[t, v]
            if params.noise_model == "heteroskedastic":
                noise = noise * (0.1 + 0.5 * abs(pred))
            elif params.noise_model != "additive":
                raise ValueError(f"unknown noise_model {params.noise_model!r}")
            X[t, v] = link(v, pred + noise)

    series = X[params.burn_in + L :]  # discard warm-start + burn-in
    if not np.all(np.isfinite(series[:, [v for v in range(n) if types[v] == "continuous"]])):
        raise FloatingPointError("SVAR produced non-finite values; lower coefficients or use burn_in")

    observed = list(range(spec.n_observed))  # first n_observed series are observed; latents dropped
    data = pd.DataFrame(series[:, observed], columns=observed)
    return {
        "data": data,
        "spec": spec,
        "oracle": LaggedDSeparationOracle(spec),
        "coefficients": {"lagged": lagged, "contemp": contemp},
        "var_types": types,
        "spectral_radius": radius,
    }


class TimeSeriesDataGenerator:
    """Config-dict wrapper over :func:`simulate_svar`, mirroring ``CausalDataGenerator``'s ergonomics.

    ``config = {"graph_params": {...TimeSeriesSpec fields... | "spec": TimeSeriesSpec},
    "simulation_params": {...SVARParams fields...}}``. ``simulate()`` returns the same dict as
    :func:`simulate_svar`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gp = dict(config.get("graph_params", {}))
        if "spec" in gp:
            self.spec = gp["spec"]
        else:
            self.spec = TimeSeriesSpec(
                n_vars=gp["n_vars"],
                max_lag=gp["max_lag"],
                lagged_edges=list(gp.get("lagged_edges", [])),
                contemp_edges=list(gp.get("contemp_edges", [])),
                n_latent=gp.get("n_latent", 0),
            )
        sp = dict(config.get("simulation_params", {}))
        valid = {f for f in SVARParams.__dataclass_fields__}
        unknown = set(sp) - valid
        if unknown:
            raise ValueError(f"unknown simulation_params: {sorted(unknown)}")
        self.params = SVARParams(**sp)

    def simulate(self) -> dict[str, Any]:
        return simulate_svar(self.spec, self.params)
