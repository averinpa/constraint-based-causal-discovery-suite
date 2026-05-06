"""
cbcd Phase-1 API sketch — pressure-test before implementing.

This is a *design document expressed as Python stubs*. Nothing here runs.
The goal is to fix the contracts (Protocols, ABCs, dataclasses, function
signatures) before any algorithm code is written, so that adding a 20th
constraint-based method doesn't require rewriting the foundation.

DESIGN PRINCIPLES (consequences of the causal-learn audit):

1. **No magic ints.** causal-learn dispatches on `uc_rule ∈ {0,1,2}` and
   `uc_priority ∈ {-1..4}`. We replace this with strategy *objects* — adding
   a new orientation rule means writing a class, not extending an int union.

2. **Algorithms are compositions.** PC, conservative-PC, majority-PC,
   parallel-PC, RFCI, FCI+, anytime-FCI all share skeleton/orientation
   phases. They differ in *which* skeleton, *which* collider orienter,
   *which* edge-rule set. Each phase is a Protocol; algorithms wire them.

3. **CI tests are stateless w.r.t. caching.** The test object holds data
   and parameters. Caching is a *decorator* (`CachedCITest`) — separation
   of concerns and a fix for causal-learn's broken md5-on-str(ndarray) key.

4. **Graph types are distinct classes**, not one mega-class with a flag.
   `pc() -> CPDAG`, `fci() -> PAG`, etc. Type signatures express intent.

5. **Background knowledge is first-class** and plumbed through every phase
   (skeleton, collider orientation, edge orientation), not bolted on.

6. **i.i.d. and time-series have parallel but separate APIs.** Trying to
   unify them ties the i.i.d. side in knots; LaggedCITest and friends mirror
   the i.i.d. shape but with `(var, lag)` indexing.

OPEN QUESTIONS are marked OPEN: throughout. Resolve before implementing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# A. CI TEST PROTOCOL
# =============================================================================
#
# The minimal contract every CI test must satisfy. Implementations:
#   - cbcd.citest.FisherZ          (continuous Gaussian)
#   - cbcd.citest.ChisqGsq         (discrete)
#   - cbcd.citest.PartialCorr      (continuous, robust)
#   - cbcd.citest.KCI              (nonparametric)
#   - eventually citk.* (post-2026, after citk is decoupled from causal-learn)


@runtime_checkable
class CITest(Protocol):
    """A conditional independence test bound to a fixed dataset.

    Tests are *bound* (data is held internally) so that algorithms can
    pass `(x, y, S)` index tuples without re-binding data on every call.
    For multi-dataset methods (JCI/IOD), see DatasetEnsembleCITest below.
    """

    n_vars: int
    """Number of variables in the bound dataset (informational; for sanity checks)."""

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        """Return p-value for H0: x ⫫ y | S.

        Conventions:
            - S may be empty (marginal independence).
            - x, y are variable indices; order is irrelevant (test is symmetric).
            - Returned p ∈ [0, 1]. Algorithms reject H0 when p < alpha.

        Implementations MUST be deterministic for the same `(x, y, S)`.
        """
        ...

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        """Return p-value plus diagnostic info (test stat, df, n_eff, etc.).

        Algorithms only need __call__; details() is for logging and analysis.
        """
        ...

    # OPEN: do we need a `batch` method on the Protocol itself for parallel
    # backends? Or is parallelism orchestrated externally over independent
    # `(x, y, S)` queries at a given depth?  Leaning EXTERNAL — keeps the
    # Protocol minimal, lets us parallelize via joblib at the skeleton layer.


@dataclass(frozen=True, slots=True)
class CITestResult:
    """Diagnostic-rich result. Algorithms ignore everything but p_value."""

    p_value: float
    statistic: float | None = None
    df: int | None = None
    n_effective: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


# Caching + recording decorator — wraps any CITest. Caching is the primary
# concern; recording is optional and routes through the same wrapper because
# every CI call already passes through it (matches causal-learn's mental
# model where the cache *was* the record).
class CachedCITest:
    """Decorator that memoizes a CITest by `(x, y, frozenset(S))` and
    optionally emits a `CITestRecord` to a `RunRecorder` on every call.

    Cache key is the *test instance identity* + `(x, y, frozenset(S))`.
    Because the underlying test is bound to its data, instance identity
    correctly distinguishes datasets — no fragile data-content hashing,
    fixing causal-learn's md5(str(ndarray)) collision bug (cit.py:98).

    Behaviour per call:
      - On cache hit: emit `record_ci(..., was_cache_hit=True, elapsed_sec≈0)`
        and return the cached p-value.
      - On cache miss: invoke inner test, store result, emit
        `record_ci(..., was_cache_hit=False, elapsed_sec=t)`.

    If `recorder` is None, the recording path is a no-op (zero overhead);
    you still get caching. Pass `cache=False` to disable caching while
    keeping the wrapper as a transparent recording adapter.
    """

    def __init__(
        self,
        inner: CITest,
        *,
        recorder: RunRecorder | None = None,
        cache: bool = True,
    ) -> None: ...

    n_vars: int

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float: ...
    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult: ...
    def is_cached(self, x: int, y: int, S: Sequence[int]) -> bool: ...

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        ...


# =============================================================================
# B. BACKGROUND KNOWLEDGE
# =============================================================================


@dataclass(frozen=True)
class BackgroundKnowledge:
    """Constraints on the output graph, supplied by the user.

    Used during skeleton (skip CI tests on forbidden adjacencies),
    collider orientation (don't orient against required edges), and
    edge-rule application (don't reorient required edges).
    """

    forbidden_directed: frozenset[tuple[int, int]] = frozenset()
    """Directed edges the algorithm must not produce: (src, dst)."""

    required_directed: frozenset[tuple[int, int]] = frozenset()
    """Directed edges the algorithm must produce: (src, dst)."""

    forbidden_adjacent: frozenset[frozenset[int]] = frozenset()
    """Adjacencies the algorithm must not produce (skeleton-level)."""

    tiers: tuple[frozenset[int], ...] = ()
    """Temporal/causal tiers: variables in tier i cannot be ancestors of tier j<i.
    Empty tuple = no tier constraints. Each variable appears in at most one tier."""

    # OPEN: do we need "possibly directed" / "definitely-not-collider" as
    # separate kinds, or is forbidden/required enough? Tetrad has more
    # categories but most practitioners only use forbidden+required+tiers.


# =============================================================================
# C. SKELETON PHASE
# =============================================================================


@dataclass
class Skeleton:
    """Output of the skeleton-discovery phase: undirected graph + sepsets."""

    n_vars: int
    adj: NDArray[np.bool_]  # (n, n) symmetric, no self-loops
    sepsets: dict[frozenset[int], tuple[int, ...]]
    """For each *removed* edge {x, y}, a witness S such that x ⫫ y | S."""

    pvalues_max: NDArray[np.float64] | None = None
    """For maxp orientation: max p-value seen for each pair {x,y} during search.
    None if the skeleton algorithm didn't track this."""


class SkeletonAlgorithm(Protocol):
    """Strategy for producing a Skeleton from a CITest."""

    def __call__(
        self,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: BackgroundKnowledge | None = None,
        recorder: "RunRecorder | None" = None,
        n_jobs: int = 1,
    ) -> Skeleton:
        ...


# Concrete implementations (signatures only):
class PCStable:
    """PC-stable skeleton (Colombo & Maathuis 2014). The default."""
    def __init__(self, *, track_max_pvalue: bool = False) -> None: ...

class FAS:
    """Fast Adjacency Search — used by FCI family. Identical to PC-stable in
    the i.i.d. case, but kept as a separate class so FCI's signature is
    self-documenting and we can specialize later if FCI+ needs differences."""
    def __init__(self) -> None: ...


# =============================================================================
# D. GRAPH TYPES
# =============================================================================
#
# Four output graph kinds, by mathematical object:
#
#     DAG     — directed acyclic
#     CPDAG   — completed PDAG: a mix of → and — (output of PC family)
#     MAG     — maximal ancestral graph: → and ↔
#     PAG     — partial ancestral graph: marks ∈ {tail, arrow, circle} (output of FCI family)
#
# Storage: every graph is an n×n int8 matrix `endpoints` where
# `endpoints[i, j]` is the endpoint mark *at vertex j* of the edge between i
# and j. An edge exists iff endpoints[i, j] != NO_EDGE; the symmetry invariant
#     endpoints[i, j] == NO_EDGE  iff  endpoints[j, i] == NO_EDGE
# is enforced at construction.


class EndpointMark(IntEnum):
    NO_EDGE = 0
    TAIL = 1     # —
    ARROW = 2    # >
    CIRCLE = 3   # o (PAG only)


@dataclass(frozen=True, slots=True)
class Edge:
    """Read-only view of one edge — convenience for repr/debugging.

    Algorithms never *write* through Edge; they manipulate the `endpoints`
    matrix directly for speed. Construct via ``graph.edge(i, j)``.
    """

    i: int
    j: int
    mark_at_i: EndpointMark
    """The endpoint mark at vertex i (= endpoints[j, i])."""
    mark_at_j: EndpointMark
    """The endpoint mark at vertex j (= endpoints[i, j])."""

    def __str__(self) -> str:
        """Pretty-print as e.g. 'i o-> j' or 'i — j'."""
        ...

    @property
    def is_directed(self) -> bool:
        """True iff exactly one end is ARROW and the other TAIL."""
        ...

    @property
    def is_undirected(self) -> bool:
        """True iff both ends are TAIL."""
        ...

    @property
    def is_bidirected(self) -> bool:
        """True iff both ends are ARROW."""
        ...


class _GraphBase(ABC):
    """Shared storage. Not a public API — subclasses are."""

    n_vars: int
    endpoints: NDArray[np.int8]  # (n, n) — mark at j of edge {i, j}
    var_names: tuple[str, ...] | None

    @abstractmethod
    def _validate_endpoints(self) -> None:
        """Subclasses enforce which mark combinations are legal for them."""

    # Common operations (signatures only):
    def has_edge(self, i: int, j: int) -> bool: ...
    def adjacent(self, i: int) -> tuple[int, ...]: ...
    def to_adjacency_bool(self) -> NDArray[np.bool_]: ...

    def edge(self, i: int, j: int) -> Edge:
        """Read-only Edge view. Raises if no edge between i and j."""
        ...


class DAG(_GraphBase):
    """Directed acyclic. Only TAIL→ARROW edges allowed."""

    def parents(self, i: int) -> tuple[int, ...]: ...
    def children(self, i: int) -> tuple[int, ...]: ...
    def topological_order(self) -> tuple[int, ...]: ...
    def to_cpdag(self) -> CPDAG: ...


class CPDAG(_GraphBase):
    """Completed PDAG: directed (→) or undirected (—) edges only."""

    ambiguous_triples: frozenset[tuple[int, int, int]]
    """Unshielded triples (X, Z, Y) the collider orienter could not classify
    as definitely-collider or definitely-non-collider. Populated by
    ConservativeOrienter / MajorityOrienter; empty for SepsetOrienter."""

    definite_non_colliders: frozenset[tuple[int, int, int]]
    """Unshielded triples (X, Z, Y) the orienter affirmatively classified as
    *non*-colliders (Z ∈ all candidate sepsets). Distinct from "not in
    ambiguous_triples" because that conflates 'definitely not' with
    'never tested'."""

    def directed_edges(self) -> tuple[tuple[int, int], ...]: ...
    def undirected_edges(self) -> tuple[frozenset[int], ...]: ...
    def parents(self, i: int) -> tuple[int, ...]: ...
    def neighbors(self, i: int) -> tuple[int, ...]:
        """Vertices connected by *undirected* edges."""
        ...

    def to_dag_extension(self) -> DAG | None:
        """A consistent DAG extension if one exists."""
        ...


class MAG(_GraphBase):
    """Maximal ancestral graph: → or ↔ edges only."""

    def is_ancestor_of(self, i: int, j: int) -> bool: ...
    def m_separated(self, x: int, y: int, S: Sequence[int]) -> bool: ...
    def to_pag(self) -> PAG: ...


class PAG(_GraphBase):
    """Partial ancestral graph: marks ∈ {tail, arrow, circle} on either end.
    Output of the FCI family."""

    def definite_edges(self) -> tuple[tuple[int, int, EndpointMark, EndpointMark], ...]: ...
    def possibly_directed(self, i: int, j: int) -> bool: ...


# Intermediate type used between collider orientation and edge-rule application:
@dataclass
class PartialCPDAG(_GraphBase):
    """Skeleton with some colliders oriented; remaining edges undirected.
    Input to MeekRules. Carries through `ambiguous_triples` so that Meek
    rules can avoid orienting *into* an ambiguous triple."""

    ambiguous_triples: frozenset[tuple[int, int, int]] = frozenset()


@dataclass
class PartialPAG(_GraphBase):
    """Skeleton with some endpoint marks set; remaining are CIRCLE.
    Input to FCIRules."""


# =============================================================================
# E. COLLIDER ORIENTATION
# =============================================================================
#
# A ColliderOrienter classifies each unshielded triple (X, Z, Y) as one of
# {collider, non_collider, ambiguous}. It does NOT lay marks on a graph —
# that's the caller's job, since the same decisions apply to either a CPDAG
# canvas (PC family) or a PAG canvas (FCI family).
#
# Replaces causal-learn's `uc_rule ∈ {0,1,2}` × `uc_priority ∈ {-1..4}`
# matrix of magic ints. Each row of that matrix becomes one class here.


@dataclass(frozen=True, slots=True)
class ColliderDecisions:
    """Outcome of collider orientation. Pure data; no graph state.

    Triples are stored canonically as (X, Z, Y) with X < Y so that a triple
    has a unique representation regardless of which neighbour we encountered
    first. Z is always the middle vertex of the unshielded triple.
    """

    colliders: frozenset[tuple[int, int, int]]
    """Triples (X, Z, Y) where Z is judged a collider — orient X *→ Z ←* Y."""

    non_colliders: frozenset[tuple[int, int, int]]
    """Triples where Z is judged definitively *not* a collider. Distinct from
    "not in colliders" because that conflates 'definitely not' with
    'never tested' — needed by Meek/FCI rules that handle the two differently."""

    ambiguous: frozenset[tuple[int, int, int]]
    """Triples the orienter could not classify (CPC/MPC only; empty otherwise)."""

    def apply_to_cpdag(
        self,
        skeleton: Skeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialCPDAG:
        """Lay collider arrows onto a CPDAG canvas: every other edge stays
        TAIL—TAIL. Carries `ambiguous` triples through on the result so Meek
        rules can avoid re-orienting into them."""
        ...

    def apply_to_pag(
        self,
        skeleton: Skeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialPAG:
        """Lay collider arrows onto a PAG canvas: every other endpoint stays
        CIRCLE."""
        ...


class ColliderOrienter(Protocol):
    """Strategy for classifying unshielded triples.

    Concrete orienters MUST honour `background`:
      - If a directed edge X→Z is *required*, no triple (X, Z, Y) can have
        Z as collider via the X-edge — pre-classify as non_collider.
      - If a directed edge Z→X is *forbidden*, the orienter must not classify
        a triple as collider when doing so would imply Z→X.
      - Forbidden adjacencies that survive the skeleton are an upstream bug;
        the orienter may assume the skeleton respects them.
    """

    requires_max_pvalues: bool
    """If True, the algorithm pre-flight will check that the skeleton was
    built with `track_max_pvalue=True`. Lets callers fail fast at wiring time
    instead of partway through orientation."""

    def __call__(
        self,
        skeleton: Skeleton,
        ci: CITest,
        *,
        alpha: float,
        background: BackgroundKnowledge | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> ColliderDecisions:
        ...


# Concrete orienters —--------------------------------------------------------


class SepsetOrienter:
    """Standard PC R0. Triple (X, Z, Y) is a collider iff Z ∉ sepsets[{X, Y}],
    a non-collider iff Z ∈ sepsets[{X, Y}]. Never produces `ambiguous`."""

    requires_max_pvalues = False

    def __init__(self) -> None: ...


class MaxPOrienter:
    """Tie-break colliders by the maximum p-value seen during skeleton search.

    For each unshielded triple (X, Z, Y), look at all S ⊆ adj(X) ∪ adj(Y)
    that were tested during skeleton; pick the S that maximised p(X ⫫ Y | S);
    if Z ∉ that S, classify as collider. Robust to which sepset the skeleton
    happened to record first.
    """

    requires_max_pvalues = True

    def __init__(self) -> None: ...


class ConservativeOrienter:
    """Conservative PC (Ramsey 2006): re-test CI over every subset of
    adj(X) ∪ adj(Y) \\ {X, Y} up to `max_cond_set`. Classify Z as:
        collider     if Z ∉ S for every separating S
        non_collider if Z ∈ S for every separating S
        ambiguous    otherwise
    """

    requires_max_pvalues = False

    def __init__(self, *, max_cond_set: int | None = None) -> None: ...


class MajorityOrienter:
    """Majority-rule PC (Colombo & Maathuis 2014): like Conservative but with
    a majority vote across separating sepsets:
        collider     if Z ∈ S in <50% of separating S
        non_collider if Z ∈ S in >50%
        ambiguous    on exact tie or if no separating S found
    """

    requires_max_pvalues = False

    def __init__(self, *, max_cond_set: int | None = None) -> None: ...


class DefiniteMaxPOrienter:
    """causal-learn's `definite_maxp` — maxp tie-break combined with explicit
    non-collider marking when Z lies in the recorded sepset. Kept as a
    standalone class for parity reproduction during the rewrite; the same
    behaviour can be approximated by MaxPOrienter for new code."""

    requires_max_pvalues = True

    def __init__(self) -> None: ...


# =============================================================================
# F. EDGE ORIENTATION RULES
# =============================================================================
#
# After collider orientation produces a PartialCPDAG (PC family) or PartialPAG
# (FCI family), edge-orientation rules are applied to fixpoint to propagate
# orientations and produce the final CPDAG or PAG.
#
# Two Protocols (no shared union return type — different canvases need
# different rule sets):
#
#   CPDAGRules   PartialCPDAG -> CPDAG    (Meek 1995 R1-R4)
#   PAGRules     PartialPAG   -> PAG      (Zhang 2008 R1-R10)
#
# Plus an optional skeleton-refinement Protocol used between collider
# orientation and PAG edge rules in the FCI family but not RFCI.


class CPDAGRules(Protocol):
    """Apply CPDAG edge-orientation rules to fixpoint.

    Concrete implementations MUST honour `background`:
      - never re-orient an edge required by `background.required_directed`,
      - never produce an edge in `background.forbidden_directed` direction.
    """

    def __call__(
        self,
        graph: PartialCPDAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> CPDAG:
        ...


class PAGRules(Protocol):
    """Apply PAG edge-orientation rules to fixpoint.

    Same `background` honouring contract as CPDAGRules. PAG-specific:
    forbidden_directed translates to "must not produce TAIL→ARROW with this
    orientation"; required_directed pre-marks endpoints accordingly.
    """

    def __call__(
        self,
        graph: PartialPAG,
        *,
        background: BackgroundKnowledge | None = None,
        max_iterations: int | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> PAG:
        ...


class PAGSkeletonRefinement(Protocol):
    """Optional second skeleton-pruning phase, applied between collider
    orientation and PAG edge rules in the FCI family. RFCI sets this to None.

    For each pair {X, Y} still adjacent in the PartialPAG, enumerate subsets
    of Possible-D-Sep(X, Y) up to `max_cond_set` in *increasing* size; remove
    the edge on first independence found. Replaces causal-learn's
    `removeByPossibleDsep` (FCI.py:1000-1058) which had an exponential
    enumeration bug — see audit.
    """

    def __call__(
        self,
        graph: PartialPAG,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> PartialPAG:
        ...


# Concrete rule sets —--------------------------------------------------------


class MeekRules:
    """Meek 1995 R1-R4 for CPDAGs.

        R1: a → b — c with a, c non-adjacent       ⟹  b → c
        R2: a → b → c with a — c                   ⟹  a → c
        R3: a — b, c → b, d → b, a — c, a — d,
            c, d non-adjacent                       ⟹  a → b
        R4: a — b, a — c, c → d, b → d, etc.       ⟹  a → d

    Applied to fixpoint. Note: rule labels here are scoped to MeekRules and
    are *not* the same as FCIRules.R1 — Meek's and Zhang's numbering differ.
    """

    _ALL: frozenset[Literal["R1", "R2", "R3", "R4"]] = frozenset({"R1", "R2", "R3", "R4"})

    def __init__(
        self,
        *,
        rules: frozenset[Literal["R1", "R2", "R3", "R4"]] = _ALL,
    ) -> None: ...


class FCIRules:
    """Zhang 2008 R1-R10 for PAGs.

    Note: Zhang's R0 (collider rule) is handled by ColliderOrienter upstream;
    FCIRules starts at R1. R5-R10 depend on uncovered circle paths and
    discriminating paths — implementations must support these structural
    queries on the PartialPAG.

    Subsetting gives variants:
        RFCI = FCIRules(rules={"R1", "R2", "R3", "R4"})  +  refinement=None
        FCI  = FCIRules()  (all)                          +  refinement=PossibleDSepRefinement()
        GFCI = FCIRules()  (all)                          +  score-based skeleton upstream

    Rule labels here are scoped to FCIRules and are *not* the same as
    MeekRules.R1 — Meek's and Zhang's numbering differ.
    """

    _ALL: frozenset[str] = frozenset({f"R{i}" for i in range(1, 11)})

    def __init__(
        self,
        *,
        rules: frozenset[str] = _ALL,
    ) -> None: ...


class PossibleDSepRefinement:
    """Standard FCI Possible-D-Sep pruning. The default for `fci(...)`.

    For each surviving edge {X, Y}, computes Possible-D-Sep(X, Y) on the
    current PartialPAG, then enumerates subsets in increasing size up to
    `max_cond_set`, breaking on the first S with X ⫫ Y | S. Records the
    witness in the returned graph's sepsets so downstream PAG rules can
    use it.

    Differences vs causal-learn's removeByPossibleDsep:
      - increasing-size iteration with early break (audit: causal-learn
        enumerated the full powerset twice per edge, ignoring `depth`),
      - tests (X, Y) once, not (X, Y) and (Y, X) separately,
      - threads `n_jobs` for parallel CI evaluation at each subset size.
    """

    def __init__(self, *, max_cond_set: int | None = None) -> None: ...


# =============================================================================
# G. TOP-LEVEL ALGORITHM COMPOSITION
# =============================================================================
#
# PC family shape (returns CPDAG):
#
#     pc(data, *, ci_test, alpha, skeleton, collider, rules, max_cond_set,
#        background, n_jobs) -> CPDAG
#
# FCI family shape (returns PAG, with optional Possible-D-Sep refinement):
#
#     fci(data, *, ci_test, alpha, skeleton, collider, refinement, rules,
#         max_cond_set, background, n_jobs) -> PAG
#
# Variants are thin wrapper functions over these core entry points (full
# signatures preserved for IDE / docs; no functools.partial in the public
# API).
#
# n_jobs propagation policy:
#   - Threaded into `skeleton(...)` (the bulk of CI calls live here).
#   - Threaded into `refinement(...)` (per-edge powerset enumeration).
#   - Threaded into `collider(...)` *only* for ConservativeOrienter and
#     MajorityOrienter (which re-test CI over many subsets per triple).
#   - Not used by EdgeRules — fixpoint loops are sequential by nature.


# --- CI test factory + registry --------------------------------------------


def make_ci_test(
    name: str,
    data: NDArray[np.float64],
    *,
    var_names: tuple[str, ...] | None = None,
    **kwargs: object,
) -> CITest:
    """Resolve a CI-test name to a bound CITest instance.

    Built-in names (registered at import time):
        "fisherz"      -> cbcd.citest.FisherZ
        "chisq"        -> cbcd.citest.ChisqTest
        "gsq"          -> cbcd.citest.GsqTest
        "kci"          -> cbcd.citest.KCI
        "partialcorr"  -> cbcd.citest.PartialCorr

    `**kwargs` are forwarded to the CITest constructor (e.g. KCI's kernel
    bandwidth). Algorithms call this internally when given a string;
    advanced users construct CITest instances directly and bypass it.
    """
    ...


def register_ci_test(name: str, factory: Callable[..., CITest]) -> None:
    """Register a user-defined CI test under `name` for use with `make_ci_test`.

    `factory(data, *, var_names=None, **kwargs) -> CITest`. Raises if
    `name` is already registered (no silent override).
    """
    ...


# --- Data normalization ----------------------------------------------------


def _normalize_data(
    data: NDArray[np.float64] | "pd.DataFrame",
    var_names: Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
    """Internal: accept ndarray or DataFrame, return (ndarray, var_names).

    DataFrame.columns becomes var_names if not overridden. ndarray with no
    var_names returns None (algorithms fall back to integer indices).
    Run at the top of every public algorithm; not exported.
    """
    ...


# --- PC family -------------------------------------------------------------


def pc(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | Literal["fisherz", "chisq", "gsq", "kci", "partialcorr"] = "fisherz",
    alpha: float = 0.05,
    skeleton: SkeletonAlgorithm | None = None,    # default: PCStable()
    collider: ColliderOrienter | None = None,      # default: SepsetOrienter()
    rules: CPDAGRules | None = None,               # default: MeekRules()
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> CPDAG: ...


def conservative_pc(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> CPDAG:
    """Conservative PC (Ramsey 2006). Identical to pc() but uses
    ConservativeOrienter — produces ambiguous_triples for unresolved
    unshielded triples instead of guessing."""
    ...


def majority_pc(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> CPDAG:
    """Majority-rule PC (Colombo & Maathuis 2014). Same as pc() but uses
    MajorityOrienter."""
    ...


def mvpc(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    random_state: int | np.random.Generator | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> CPDAG:
    """Missing-value PC (Tu et al. 2019). Constructs virtual data to correct
    for selection bias from missingness, then runs PC.

    Separate from pc() because the skeleton phase is fundamentally different
    (skeleton-correction step over imputed data); shoehorning it into pc()
    via a flag would mirror causal-learn's branchy mvpc_alg mess
    (PC.py:254-476 in the audit).

    `random_state` is required-explicit (no global RNG fallback — fixes the
    causal-learn Helper.py:547 reproducibility bug).
    """
    ...


# --- FCI family ------------------------------------------------------------


def fci(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    skeleton: SkeletonAlgorithm | None = None,                   # default: FAS()
    collider: ColliderOrienter | None = None,                     # default: SepsetOrienter()
    refinement: PAGSkeletonRefinement | None = None,              # default: PossibleDSepRefinement()
    rules: PAGRules | None = None,                                # default: FCIRules()
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> PAG: ...


def rfci(
    data: NDArray[np.float64] | "pd.DataFrame",
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> PAG:
    """RFCI (Colombo et al. 2012). Skips the Possible-D-Sep refinement step
    and applies only Zhang's R1-R4 — faster and more conservative than FCI."""
    ...


def anytime_fci(
    data: NDArray[np.float64] | "pd.DataFrame",
    max_cond_set: int,
    *,
    ci_test: CITest | str = "fisherz",
    alpha: float = 0.05,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> PAG:
    """Anytime-FCI (Spirtes 2001). FCI with a hard depth cap on conditioning
    set size — sound but possibly incomplete. `max_cond_set` is positional and
    required to make the trade-off explicit."""
    ...


# --- CDNOD -----------------------------------------------------------------


def cdnod(
    data: NDArray[np.float64] | "pd.DataFrame",
    context_index: NDArray[np.int_],
    *,
    ci_test: CITest | str = "kci",
    alpha: float = 0.05,
    skeleton: SkeletonAlgorithm | None = None,
    collider: ColliderOrienter | None = None,
    rules: CPDAGRules | None = None,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> CPDAG:
    """CDNOD (Huang et al. 2019). Constraint-based causal discovery from
    nonstationary / heterogeneous data.

    `context_index` is a (T,) integer array giving the regime/context label
    for each sample. The algorithm augments `data` with this as an
    auxiliary variable and runs PC. Background knowledge is enforced
    automatically: the context node is forbidden from receiving any incoming
    edges, so the canonical orientation is **c_indx → X** for any adjacent
    observed variable X (resolves causal-learn's CDNOD.py:94 vs
    CDNOD.py:179 inconsistency).
    """
    ...


# --- Multi-dataset methods (JCI / IOD) -------------------------------------
#
# JCI and IOD operate on a *bank* of datasets, not one. They need a different
# CI-test shape because evidence about (X, Y | S) lives in some-but-not-all
# datasets and must be combined.


@runtime_checkable
class EnsembleCITest(Protocol):
    """A bank of CI tests over multiple datasets that share a global variable
    namespace.

    Variables are addressed in *global* index space. Each dataset covers a
    subset of the global variables (`variables_in(d)`). Calls to __call__
    return either:
      - a combined p-value across datasets that contain all of {x, y} ∪ S
        (recommended: Fisher's combined p-value or the test's native
        pooled-evidence form), or
      - the per-dataset p-values via `per_dataset(...)` for algorithms that
        want to combine evidence themselves (IOD does this).
    """

    n_datasets: int
    n_global_vars: int

    def variables_in(self, dataset_idx: int) -> frozenset[int]: ...

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        """Combined p-value, pooled across datasets that contain x, y, and S."""
        ...

    def per_dataset(
        self, x: int, y: int, S: Sequence[int]
    ) -> dict[int, float]:
        """Per-dataset p-values keyed by dataset_idx, only for datasets
        whose `variables_in` covers {x, y} ∪ S."""
        ...


@dataclass(frozen=True)
class PAGEquivalenceClass:
    """Output of IOD: a non-empty set of PAGs all consistent with the
    evidence, plus the shared skeleton they agree on."""

    members: tuple[PAG, ...]
    common_skeleton: Skeleton


def jci(
    datasets: Sequence[NDArray[np.float64]],
    intervention_targets: Sequence[frozenset[int]],
    *,
    ensemble_ci: EnsembleCITest | str = "fisherz",
    alpha: float = 0.05,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> PAG:
    """JCI (Mooij et al. 2020). Joint causal inference from observational +
    interventional data. Augments each dataset with intervention indicator
    variables (one per intervention regime), pools, and runs FCI with
    background knowledge that forbids edges into the indicators.

    `intervention_targets[d]` is the set of global variable indices targeted
    by the intervention that produced datasets[d]. Empty set = observational.
    """
    ...


def iod(
    datasets: Sequence[NDArray[np.float64]],
    variable_overlaps: Sequence[frozenset[int]],
    *,
    ensemble_ci: EnsembleCITest | str = "fisherz",
    alpha: float = 0.05,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> PAGEquivalenceClass:
    """IOD (Tillman & Spirtes 2011). Causal discovery from multiple datasets
    with overlapping but not identical variable sets.

    `variable_overlaps[d]` is the set of global variable indices observed in
    `datasets[d]`. The output is an equivalence class because IOD generally
    cannot single out one PAG.
    """
    ...


# =============================================================================
# H. TIME-SERIES API (parallel structure)
# =============================================================================
#
# Time-series constraint-based methods (PCMCI, PCMCI+, LPCMCI, tsFCI,
# SVAR-FCI, J-PCMCI) operate on lagged variables — `(var, lag)` pairs. The
# data flow is fundamentally different from i.i.d. (per-target parent
# search, MCI step that conditions on parent sets), so we define a parallel
# API rather than overloading the i.i.d. one.
#
# What's shared with i.i.d.:
#   - EndpointMark enum
#   - PAGEquivalenceClass (for multi-graph outputs)
#   - EnsembleCITest shape (J-PCMCI uses an ensemble over datasets)
#
# What's parallel-but-separate:
#   - LaggedVar / LaggedDataset / LaggedBackgroundKnowledge
#   - LaggedCITest / LaggedCITestResult
#   - TimeSeriesDAG / TimeSeriesCPDAG / TimeSeriesPAG (3D endpoints array)
#   - LaggedSkeleton / LaggedColliderDecisions / Lagged*Rules Protocols
#
# Storage convention for time-series graphs:
#   endpoints: NDArray[np.int8] of shape (max_lag + 1, n_vars, n_vars)
#   endpoints[τ, i, j] = endpoint mark at vertex j of the edge i_{t-τ} — j_t
#   The τ=0 slice is symmetric (contemporaneous edges); τ>0 slices are not
#   (lagged edges go from past to present).


# --- Lagged primitives -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class LaggedVar:
    """A variable at a specific lag: var ∈ [0, n_vars), lag ∈ [-max_lag, 0].

    Convention: lag=0 means "current time t"; lag=-1 means "t-1"; etc.
    Validation enforces lag ≤ 0 at construction. Algorithms also enforce
    lag ≥ -max_lag against the LaggedDataset.
    """

    var: int
    lag: int  # ≤ 0

    @property
    def is_contemporaneous(self) -> bool:
        return self.lag == 0


@dataclass
class LaggedDataset:
    """Multivariate stationary time series with a max-lag horizon.

    Stationarity is assumed (test parameters apply across all t). Use
    regime-switching variants for non-stationary data — out of scope for v0.
    """

    data: NDArray[np.float64]  # (T, n_vars)
    max_lag: int
    """Maximum absolute lag the user wants to consider. Algorithms validate
    max_lag < T - 1 (else there's no usable training window)."""

    var_names: tuple[str, ...] | None = None
    time_index: NDArray[np.int_] | NDArray[np.datetime64] | None = None
    """Optional, diagnostic only — used for prettier output, never for
    computation. Algorithms ignore this."""


@dataclass(frozen=True)
class LaggedBackgroundKnowledge:
    """Time-series-specific background knowledge.

    Separate from the i.i.d. BackgroundKnowledge to keep types honest:
    constraints here are over (var, lag) tuples, not bare variable indices.
    """

    forbidden_lagged: frozenset[tuple[LaggedVar, LaggedVar]] = frozenset()
    """Directed lagged edges that must not appear: (src, dst)."""

    required_lagged: frozenset[tuple[LaggedVar, LaggedVar]] = frozenset()
    """Directed lagged edges that must appear."""

    no_autoregressive: frozenset[int] = frozenset()
    """Variables for which no X_{t-τ} → X_t edge may appear at any τ > 0."""

    no_contemporaneous: frozenset[frozenset[int]] = frozenset()
    """Pairs {i, j} for which no i_t — j_t edge may appear (lag=0)."""

    contemporaneous_tiers: tuple[frozenset[int], ...] = ()
    """Tiers for contemporaneous edges only: variables in tier i cannot be
    contemporaneous ancestors of tier j<i. Lagged edges respect time."""


# --- Lagged CI test layer --------------------------------------------------


@dataclass(frozen=True, slots=True)
class LaggedCITestResult:
    """Diagnostic-rich result for a lagged CI test. Algorithms only need
    p_value; n_effective is particularly important for time series (it's
    T - max_lag, possibly less if the test does additional preprocessing)."""

    p_value: float
    statistic: float | None = None
    df: int | None = None
    n_effective: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class LaggedCITest(Protocol):
    """CI test bound to a LaggedDataset. Conditioning sets are LaggedVars.

    Convention: x.lag and y.lag may differ (e.g. test x_{t-2} ⫫ y_{t-1}).
    Algorithms typically fix one to lag=0 ("now") and probe the past for
    the other; the Protocol does not require this.

    Implementations MUST be deterministic for fixed (x, y, S) and stationary
    in the sense that p depends on (x.lag - y.lag) and the relative lags in
    S, not on absolute time t.
    """

    n_vars: int
    max_lag: int

    def __call__(
        self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
    ) -> float:
        """Return p-value for H0: x ⫫ y | S."""
        ...

    def details(
        self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
    ) -> LaggedCITestResult:
        ...


def make_lagged_ci_test(
    name: str,
    data: LaggedDataset,
    **kwargs: object,
) -> LaggedCITest:
    """Built-in names:
        "parcorr"     -> partial correlation (linear, Gaussian)
        "gpdc"        -> Gaussian process distance correlation
        "cmi_knn"     -> conditional mutual information via kNN
        "regci"       -> regression-based CI (residual independence)
    """
    ...


def register_lagged_ci_test(
    name: str, factory: Callable[..., LaggedCITest]
) -> None:
    """Register a user-defined lagged CI test."""
    ...


# --- Time-series graph types -----------------------------------------------


@dataclass(frozen=True, slots=True)
class LaggedEdge:
    """Read-only edge view for time-series graphs. Convenience for repr."""

    src: LaggedVar
    dst: LaggedVar
    mark_at_src: EndpointMark
    mark_at_dst: EndpointMark

    def __str__(self) -> str: ...
    @property
    def lag(self) -> int:
        """dst.lag - src.lag (≥ 0 by convention since edges go past→present
        for τ>0; 0 for contemporaneous)."""
        ...


class _LaggedGraphBase(ABC):
    """Shared storage for time-series graphs. Internal."""

    n_vars: int
    max_lag: int
    endpoints: NDArray[np.int8]  # (max_lag + 1, n_vars, n_vars)
    var_names: tuple[str, ...] | None

    @abstractmethod
    def _validate_endpoints(self) -> None: ...

    def has_edge(self, src: LaggedVar, dst: LaggedVar) -> bool: ...
    def edge(self, src: LaggedVar, dst: LaggedVar) -> LaggedEdge: ...
    def parents(self, v: LaggedVar) -> tuple[LaggedVar, ...]: ...
    def lagged_edges(self) -> tuple[LaggedEdge, ...]:
        """All edges with lag > 0."""
        ...
    def contemporaneous_edges(self) -> tuple[LaggedEdge, ...]:
        """All edges at lag = 0."""
        ...

    # Three correctly-typed projections (no CPDAG | PAG union antipattern):
    @abstractmethod
    def to_summary_graph(self) -> _GraphBase:
        """Project to a single graph over variables: edge i—j exists if any
        lag has an edge. Subclasses pin the return type (CPDAG/PAG)."""

    @abstractmethod
    def to_contemporaneous_graph(self) -> _GraphBase:
        """Restrict to lag=0 slice as a standalone graph."""


class TimeSeriesDAG(_LaggedGraphBase):
    """Directed time-series graph. Lagged edges always TAIL→ARROW
    (past→present); contemporaneous edges may be directed either way under
    a topological constraint."""

    def to_summary_graph(self) -> DAG: ...
    def to_contemporaneous_graph(self) -> DAG: ...
    def to_time_series_cpdag(self) -> TimeSeriesCPDAG: ...


class TimeSeriesCPDAG(_LaggedGraphBase):
    """Output of PCMCI / PCMCI+. Lagged edges all directed past→present;
    contemporaneous edges may be undirected."""

    ambiguous_triples: frozenset[tuple[LaggedVar, LaggedVar, LaggedVar]]

    def to_summary_graph(self) -> CPDAG: ...
    def to_contemporaneous_graph(self) -> CPDAG: ...


class TimeSeriesPAG(_LaggedGraphBase):
    """Output of LPCMCI / tsFCI / SVAR-FCI. Endpoint marks may be CIRCLE."""

    def to_summary_graph(self) -> PAG: ...
    def to_contemporaneous_graph(self) -> PAG: ...


@dataclass
class PartialTimeSeriesCPDAG(_LaggedGraphBase):
    """Intermediate after lagged collider orientation."""
    ambiguous_triples: frozenset[tuple[LaggedVar, LaggedVar, LaggedVar]] = frozenset()


@dataclass
class PartialTimeSeriesPAG(_LaggedGraphBase):
    """Intermediate after lagged collider orientation on a PAG canvas."""


# --- Phase Protocols (sketched; concretes firmed up after PCMCI v0) --------


@dataclass
class LaggedSkeleton:
    """Output of lagged-skeleton search: per-target parent sets + sepsets.

    PCMCI's PC₁ phase is a per-target parent search rather than a global
    skeleton search, so the data shape differs from the i.i.d. Skeleton.
    """

    n_vars: int
    max_lag: int
    parents: dict[LaggedVar, frozenset[LaggedVar]]
    """Estimated parent set for each target (typically lag=0 vars only)."""
    sepsets: dict[frozenset[LaggedVar], tuple[LaggedVar, ...]]


class LaggedSkeletonAlgorithm(Protocol):
    """Strategy for lagged-skeleton search."""

    def __call__(
        self,
        ci: LaggedCITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: LaggedBackgroundKnowledge | None = None,
        recorder: "RunRecorder | None" = None,
        n_jobs: int = 1,
    ) -> LaggedSkeleton: ...


class PC1Skeleton:
    """PCMCI's PC₁ stage: for each target Y_t, iteratively prune candidate
    lagged parents from {X_{t-τ} : 1 ≤ τ ≤ max_lag} via per-pair CI tests."""
    def __init__(self) -> None: ...


class PCMCIPlusSkeleton:
    """PCMCI+ skeleton: PC₁ for lagged + a contemporaneous PC step on
    `(var, 0)` sub-graph using the lagged parent sets as base conditioning."""
    def __init__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LaggedColliderDecisions:
    """Outcome of lagged collider orientation. Only contemporaneous triples
    can be ambiguous (lagged orientation is forced by time direction)."""

    colliders: frozenset[tuple[LaggedVar, LaggedVar, LaggedVar]]
    non_colliders: frozenset[tuple[LaggedVar, LaggedVar, LaggedVar]]
    ambiguous: frozenset[tuple[LaggedVar, LaggedVar, LaggedVar]]

    def apply_to_ts_cpdag(
        self, skeleton: LaggedSkeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialTimeSeriesCPDAG: ...

    def apply_to_ts_pag(
        self, skeleton: LaggedSkeleton,
        var_names: tuple[str, ...] | None = None,
    ) -> PartialTimeSeriesPAG: ...


class LaggedColliderOrienter(Protocol):
    requires_max_pvalues: bool

    def __call__(
        self,
        skeleton: LaggedSkeleton,
        ci: LaggedCITest,
        *,
        alpha: float,
        background: LaggedBackgroundKnowledge | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> LaggedColliderDecisions: ...


class LaggedCPDAGRules(Protocol):
    """Edge orientation rules for time-series CPDAGs."""

    def __call__(
        self,
        graph: PartialTimeSeriesCPDAG,
        *,
        background: LaggedBackgroundKnowledge | None = None,
        max_iterations: int | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> TimeSeriesCPDAG: ...


class LaggedPAGRules(Protocol):
    """Edge orientation rules for time-series PAGs (LPCMCI, tsFCI use these
    with adapted versions of Zhang's R1-R10)."""

    def __call__(
        self,
        graph: PartialTimeSeriesPAG,
        *,
        background: LaggedBackgroundKnowledge | None = None,
        max_iterations: int | None = None,
        recorder: "RunRecorder | None" = None,
    ) -> TimeSeriesPAG: ...


# --- Algorithms ------------------------------------------------------------


def pcmci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | str = "parcorr",
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesCPDAG:
    """PCMCI (Runge et al. 2019). Two-stage:
        1. PC₁ — for each target Y_t, prune lagged candidate parents.
        2. MCI — for each (X_{t-τ}, Y_t), test conditioning on parent sets.

    `pc_alpha` is the independence threshold for the PC₁ stage. If None,
    PCMCI auto-selects by minimising AIC across a small grid
    (tigramite convention).

    Returns a TimeSeriesCPDAG with all lagged edges directed past→present
    and no contemporaneous edges (PCMCI assumes contemporaneous
    independence; use pcmci_plus for contemporaneous links).
    """
    ...


def pcmci_plus(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | str = "parcorr",
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesCPDAG:
    """PCMCI+ (Runge 2020). Extends PCMCI to discover contemporaneous edges
    via an additional contemporaneous PC step on the (var, 0) layer."""
    ...


def lpcmci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | str = "parcorr",
    alpha: float = 0.05,
    n_preliminary_iterations: int = 4,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesPAG:
    """LPCMCI (Gerhardus & Runge 2020). Latent-PCMCI — handles unobserved
    common causes; output is a TimeSeriesPAG."""
    ...


def tsfci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | str = "parcorr",
    alpha: float = 0.05,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesPAG:
    """tsFCI (Entner & Hoyer 2010). Time-series FCI for stationary processes
    with possible latent confounders."""
    ...


def svar_fci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | str = "parcorr",
    alpha: float = 0.05,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesPAG:
    """SVAR-FCI (Malinsky & Spirtes 2018). Structural-VAR FCI; refines tsFCI
    using structural-VAR assumptions for stronger contemporaneous orientation."""
    ...


def j_pcmci(
    datasets: Sequence[LaggedDataset],
    *,
    ensemble_ci: EnsembleCITest | str = "parcorr",
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesCPDAG:
    """J-PCMCI (Günther et al. 2023). PCMCI on multiple datasets sharing a
    global variable namespace. Reuses the i.i.d. EnsembleCITest shape for
    evidence pooling — that Protocol is variable-based, but its
    `variables_in(d)` / `per_dataset(...)` accessors carry over to lagged
    indices via the dataset-local LaggedVar mapping handled by the
    EnsembleCITest implementation.
    """
    ...


# OPEN: regime-switching methods (RPCMCI, regime-FCI) — out of scope for
# v0; would need a non-stationarity layer that breaks the LaggedDataset
# stationarity assumption.


# =============================================================================
# J. RECORDING / AUDIT TRAIL
# =============================================================================
#
# Every algorithm accepts an optional `recorder: RunRecorder | None`.
# The recorder captures every decision the algorithm makes:
#   - each CI test invocation (variables, p-value, statistic, depth, timing)
#   - each unshielded triple classification (orienter, evidence)
#   - each edge-rule firing (which rule, which edge, which iteration)
#   - per-phase wall-clock timing
#   - skeleton checkpoints (for resuming an interrupted run)
#
# Default = NullRecorder = zero overhead. Pass an InMemoryRecorder for a
# full in-RAM trace, or a FileRecorder to stream a long run to disk and
# bundle into a single `.cbcd` archive on close.
#
# Storage format: `.cbcd` is a zip archive holding per-event-type Parquet
# files (zstd-compressed) plus a JSON manifest. One file the user moves
# around; transparent to inspect with `unzip` and to read with
# `RunRecord.from_file()`.
#
# Plumbing inside an algorithm:
#   - The algorithm wraps `ci_test` with `CachedCITest(ci, recorder=recorder)`
#     (or `CachedLaggedCITest` for time-series) so every CI call is both
#     cached and logged transparently — phases see a regular CITest. Cache
#     hits emit `was_cache_hit=True` records; misses emit
#     `was_cache_hit=False` with real `elapsed_sec`. Pass `cache=False` if
#     you want recording without caching (rare — usually keep both).
#   - The algorithm passes `recorder` as a keyword to skeleton, collider,
#     refinement, and rules Protocols so they can emit their own events.
#   - The algorithm calls `recorder.phase("skeleton")` etc. via the
#     context-manager hook for timings.
#
# Records are *flat* (no graph snapshots inside CI records, no graph state
# inside rule records — just the affected edge/triple). The final graph is
# attached to the snapshot; intermediate states can be reconstructed from
# the record stream if needed.


# --- Record types (flat, frozen, slotted) ----------------------------------


@dataclass(frozen=True, slots=True)
class CITestRecord:
    """One CI test query."""
    x: int
    y: int
    S: tuple[int, ...]
    p_value: float
    statistic: float | None
    depth: int  # |S|
    elapsed_sec: float
    decision: Literal["independent", "dependent"]
    was_cache_hit: bool = False
    """True when the value came from CachedCITest's cache (elapsed_sec≈0).
    Enables `RunRecord.cache_hit_rate()`."""


@dataclass(frozen=True, slots=True)
class LaggedCITestRecord:
    """Time-series counterpart — uses LaggedVar instead of bare int."""
    x: LaggedVar
    y: LaggedVar
    S: tuple[LaggedVar, ...]
    p_value: float
    statistic: float | None
    depth: int
    elapsed_sec: float
    decision: Literal["independent", "dependent"]
    was_cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class ColliderDecisionRecord:
    """One unshielded-triple classification by a ColliderOrienter."""
    triple: tuple[int, int, int]  # (X, Z, Y) with X < Y
    classification: Literal["collider", "non_collider", "ambiguous"]
    orienter: str  # class name, e.g. "ConservativeOrienter"
    evidence: dict[str, float]
    """Free-form numeric evidence the orienter wants to expose, e.g.
    {"max_p": 0.07, "n_separating_sets": 4, "z_in_count": 3}."""


@dataclass(frozen=True, slots=True)
class RuleFiringRecord:
    """One edge-orientation rule application."""
    rule_set: str  # "MeekRules" / "FCIRules" / "LaggedPAGRules"
    rule_name: str  # "R1" / "R3" / "R10" — class-scoped
    affected_edge: tuple[int, int]
    affected_triple: tuple[int, int, int] | None
    iteration: int  # which fixpoint round (0-indexed)


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """Wall-clock timing for one algorithm phase."""
    name: str  # "skeleton" | "refinement" | "collider" | "rules" | "mci" | ...
    started_at: datetime
    elapsed_sec: float


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Skeleton snapshot at end of a depth-d sweep, for resumption."""
    depth: int
    skeleton: Skeleton
    timestamp: datetime


# --- Recorder Protocol + concrete recorders --------------------------------


@runtime_checkable
class RunRecorder(Protocol):
    """Algorithm-side hook surface. Algorithms call these methods; the
    recorder decides what to do with the records.

    All hooks are no-ops on `NullRecorder` and compile to near-zero overhead
    so production code can keep `recorder=None` (which becomes NullRecorder
    internally) without a measurable cost.
    """

    def record_ci(
        self,
        *,
        x: int,
        y: int,
        S: tuple[int, ...],
        p_value: float,
        statistic: float | None,
        depth: int,
        elapsed_sec: float,
    ) -> None: ...

    def record_lagged_ci(
        self,
        *,
        x: LaggedVar,
        y: LaggedVar,
        S: tuple[LaggedVar, ...],
        p_value: float,
        statistic: float | None,
        depth: int,
        elapsed_sec: float,
    ) -> None: ...

    def record_collider(
        self,
        *,
        triple: tuple[int, int, int],
        classification: Literal["collider", "non_collider", "ambiguous"],
        orienter: str,
        evidence: dict[str, float],
    ) -> None: ...

    def record_rule(
        self,
        *,
        rule_set: str,
        rule_name: str,
        affected_edge: tuple[int, int],
        affected_triple: tuple[int, int, int] | None,
        iteration: int,
    ) -> None: ...

    def phase(self, name: str) -> AbstractContextManager[None]:
        """Context manager that times the wrapped block and emits a
        PhaseRecord on exit. Used by algorithms as:

            with recorder.phase("skeleton"):
                skel = skeleton(ci, alpha=alpha)
        """
        ...

    def checkpoint(self, *, depth: int, skeleton: Skeleton) -> None:
        """Save a skeleton checkpoint at end-of-depth for resumption."""

    def last_checkpoint(self) -> CheckpointRecord | None:
        """Return the most-recent checkpoint, or None. Algorithms call this
        at start to decide whether to resume."""

    def snapshot(self) -> RunRecord:
        """Return a frozen RunRecord of everything seen so far."""


class NullRecorder:
    """Default recorder: every method is a no-op. Zero overhead.

    Used internally when caller passes `recorder=None` to an algorithm.
    """
    ...


class InMemoryRecorder:
    """Records to in-memory lists. Use for medium runs where a full trace
    fits in RAM and you want to query it interactively.

        rec = InMemoryRecorder()
        graph = pc(data, recorder=rec)
        record = rec.snapshot()
        record.ci_tests_df()        # pandas DataFrame
    """

    def __init__(self) -> None: ...


class FileRecorder:
    """Streams records to disk as they happen, then bundles them into a
    single `.cbcd` archive on close.

    During the run, records stream into per-event-type Parquet writers
    inside a working directory. On `snapshot()` (or context-manager exit)
    the working directory is zipped into a single `.cbcd` archive at
    `path` and the temp dir is deleted.

    Internal layout of a `.cbcd` archive (zip):
        manifest.json              algorithm, config, seed, timestamps
        ci_tests.parquet           every CI invocation (columnar)
        lagged_ci_tests.parquet    time-series counterpart (empty for i.i.d.)
        collider_decisions.parquet
        rule_firings.parquet
        phases.parquet             wall-clock per-phase timings
        checkpoints.parquet        skeletons as binary blobs (pickled,
                                   one row per depth — recoverable for
                                   resumption via `last_checkpoint()`)

    Users see one file. Inspectors can `unzip run.cbcd` to look inside.
    `RunRecord.from_file("run.cbcd")` reads it transparently.

    If the process crashes mid-run, the working directory survives. Use
    `RunRecord.recover(working_dir) -> RunRecord` to reconstruct from a
    partial trace.

    Use for long runs (KCI on n=100, PCMCI with cmi_knn on T=10000) where
    the trace is too big for memory or where you want resumability.
    """

    def __init__(
        self,
        path: Path,
        *,
        working_dir: Path | None = None,
        flush_every_n: int = 1000,
        compression: Literal["snappy", "gzip", "zstd", None] = "zstd",
    ) -> None:
        """`path` is the final `.cbcd` archive location.
        `working_dir` defaults to a temp directory next to `path`.
        `compression` applies to per-event-type Parquet files (zstd
        gives the best size/speed tradeoff for our schemas)."""
        ...

    def __enter__(self) -> "FileRecorder": ...
    def __exit__(self, *exc: object) -> None:
        """Bundle the working dir into the `.cbcd` archive and clean up."""
        ...


def _resolve_recorder(recorder: RunRecorder | None) -> RunRecorder:
    """Internal: None -> NullRecorder. Called at the top of every algorithm."""
    ...


def enable_default_recorder(recorder: RunRecorder) -> None:
    """Set a process-wide default recorder used when an algorithm is called
    with `recorder=None`. Useful in notebooks ("trace everything for the
    rest of this session") without modifying every call site."""
    ...


# --- The lagged caching+recording adapter ----------------------------------
#
# The i.i.d. counterpart is `CachedCITest` defined in §A — same shape, same
# split of "caching primary, recording optional".


class CachedLaggedCITest:
    """Time-series counterpart of CachedCITest. Same caching + optional
    recording semantics, with LaggedVar keys.

        ci = make_lagged_ci_test("parcorr", data)
        ci = CachedLaggedCITest(ci, recorder=recorder)
        skel = skeleton(ci, alpha=alpha)
    """

    def __init__(
        self,
        inner: LaggedCITest,
        *,
        recorder: RunRecorder | None = None,
        cache: bool = True,
    ) -> None: ...

    n_vars: int
    max_lag: int

    def __call__(
        self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
    ) -> float: ...
    def details(
        self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
    ) -> LaggedCITestResult: ...
    def is_cached(
        self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
    ) -> bool: ...


# --- The frozen RunRecord (snapshot output) --------------------------------


@dataclass(frozen=True)
class RunRecord:
    """Frozen snapshot of one algorithm run. Persistable, queryable.

    Records use the algorithm-native event types: i.i.d. algorithms emit
    CITestRecord; time-series algorithms emit LaggedCITestRecord.
    `lagged_ci_tests` is empty for i.i.d. runs and vice versa.
    """

    algorithm: str
    config: dict[str, Any]
    """Serialized algorithm parameters: alpha, max_cond_set, the names of
    the chosen skeleton / collider / rules classes, etc."""

    random_seed: int | None
    started_at: datetime
    completed_at: datetime | None

    ci_tests: tuple[CITestRecord, ...] = ()
    lagged_ci_tests: tuple[LaggedCITestRecord, ...] = ()
    collider_decisions: tuple[ColliderDecisionRecord, ...] = ()
    rule_firings: tuple[RuleFiringRecord, ...] = ()
    phase_timings: tuple[PhaseRecord, ...] = ()
    checkpoints: tuple[CheckpointRecord, ...] = ()

    final_graph: _GraphBase | _LaggedGraphBase | None = None

    # --- Persistence -------------------------------------------------------

    def to_file(self, path: Path) -> None:
        """Write the run as a single `.cbcd` archive (zip of per-event-type
        Parquet files plus a JSON manifest). One file the user moves around;
        same internal layout as `FileRecorder` produces.
        """

    @classmethod
    def from_file(cls, path: Path) -> RunRecord:
        """Read a `.cbcd` archive. Transparently unzips and assembles."""
        ...

    @classmethod
    def recover(cls, working_dir: Path) -> RunRecord:
        """Reconstruct a RunRecord from a `FileRecorder` working directory
        whose run crashed before bundling. Best-effort: returns whatever
        the partial Parquet writers managed to flush."""
        ...

    # --- Analysis ----------------------------------------------------------

    def ci_tests_df(self) -> "pd.DataFrame":
        """All CI test records as a DataFrame: columns = x, y, S_size,
        p_value, statistic, depth, elapsed_sec, decision."""

    def rule_firings_df(self) -> "pd.DataFrame": ...
    def collider_decisions_df(self) -> "pd.DataFrame": ...

    def ci_tests_for(self, x: int, y: int | None = None) -> tuple[CITestRecord, ...]:
        """All CI tests involving variable x (and optionally y)."""

    def slowest_tests(self, n: int = 10) -> tuple[CITestRecord, ...]:
        """Top-n CI tests by elapsed_sec — for finding KCI hot spots."""

    def rule_firings_of(
        self, rule_set: str, rule_name: str | None = None
    ) -> tuple[RuleFiringRecord, ...]:
        """All firings of a specific rule (or rule_set if rule_name=None)."""

    def total_ci_calls(self) -> int: ...
    def cache_hit_rate(self) -> float:
        """If the underlying CITest was wrapped with CachedCITest, return
        the fraction of `record_ci` events that came from cache hits.
        Returns NaN if the test wasn't cached."""


#
# DECIDED during the design walk-through (kept here for traceability;
# implementation must respect these — change requires re-opening the design):
#
#   D1. Data input.  Accept `NDArray | pd.DataFrame` at every public entry
#       point, normalize to `(ndarray, var_names)` via `_normalize_data()`
#       in §G.  `var_names` is also accepted as an explicit kwarg.
#
#   D2. CI test instantiation.  Public factory `make_ci_test(name, data, **kw)`
#       and extension hook `register_ci_test(name, factory)` in §G.  Lagged
#       analogues `make_lagged_ci_test` / `register_lagged_ci_test` in §H.
#
#   D3. Missing data.  Separate function `mvpc()` in §G.  Not a flag on
#       `pc()` — its skeleton phase is genuinely different.
#
#   D4. Reproducibility.  Every randomized algorithm accepts
#       `random_state: int | np.random.Generator | None`.  No reliance on
#       global RNG (fixes causal-learn's Helper.py:547 bug).  `mvpc()` is
#       the canonical example; KCI and permutation-based tests follow.
#
#   D5. BackgroundKnowledge validation.  Raise at construction on
#       inconsistencies (forbidden ∩ required ≠ ∅, required-edge cycle, tier
#       contradiction).  Fail fast.
#
#   D6. JCI / IOD shape.  `EnsembleCITest` Protocol in §G with `__call__`
#       (combined p, default Fisher's method — see O3) and `per_dataset()`
#       (per-dataset p for evidence-aware algorithms like IOD).
#       `PAGEquivalenceClass` for IOD's multi-PAG output.
#
#   D7. Parallel backend.  joblib; `n_jobs=1` default, `-1` = all cores.
#       Propagation policy in §G header: skeleton + refinement always;
#       collider only for Conservative/Majority orienters; rules never.
#
#   D8. CDNOD canonical direction.  `c_indx → X` (resolves causal-learn's
#       CDNOD.py:94 vs CDNOD.py:179 inconsistency).  Documented on `cdnod()`.
#
#   D9. Time-series scope for v0.  Stationary methods only: PCMCI, PCMCI+,
#       LPCMCI, tsFCI, SVAR-FCI, J-PCMCI.  Regime-switching variants
#       (RPCMCI, regime-FCI) deferred — they break LaggedDataset's
#       stationarity assumption.
#
#   D10. Max conditioning-set parameter naming.  `max_cond_set: int | None`
#        on every algorithm and every phase Protocol — renamed from
#        causal-learn's `depth` for transparency.  None = unbounded.
#
#   D11. Audit trail / run recorder.  §J defines `RunRecorder` Protocol +
#        `NullRecorder` (default, zero overhead), `InMemoryRecorder`,
#        `FileRecorder`.  Every algorithm and phase Protocol accepts
#        `recorder: RunRecorder | None = None`.  Records cover CI tests,
#        collider decisions, rule firings, phase timings, skeleton
#        checkpoints.  `RunRecord` is persistable to a single `.cbcd`
#        archive (zip of per-event-type Parquet files + JSON manifest)
#        via `RunRecord.to_file()` / `from_file()`, and queryable via
#        DataFrame views (`ci_tests_df()`, `rule_firings_df()`, ...).
#        FileRecorder writes through a working directory and bundles into
#        the `.cbcd` archive on close; supports resumption from
#        `last_checkpoint()` and post-crash recovery via
#        `RunRecord.recover(working_dir)`.  This supersedes the earlier
#        "stdlib logging events catalogue" question — structured records
#        are the audit surface; stdlib logging is reserved for warnings
#        and unstructured progress messages.
#
#   D12. CI cache + recording fused into one wrapper.  `CachedCITest` (i.i.d.)
#        and `CachedLaggedCITest` (time-series) take an optional
#        `recorder=...` parameter and emit `CITestRecord`s on every call
#        with a `was_cache_hit: bool` field.  Drops the standalone
#        `RecordingCITest` class — every CI call already passes through
#        the cache wrapper, so threading the recorder through the same
#        wrapper avoids stacking-order ambiguity and matches causal-learn's
#        mental model where the cache *was* the record.  `cache=False`
#        disables caching while keeping the wrapper as a recording
#        adapter (rare).
#
#   D13. Two-pass FCI shape.  After `PossibleDSepRefinement` removes edges,
#        `fci()` re-runs the collider step on the refined skeleton before
#        invoking `FCIRules`.  Matches Zhang/Spirtes pseudocode and the
#        causal-learn reference: refinement can drop edges that change
#        which triples are unshielded, so the prior collider classification
#        is stale.  `rfci()` skips both refinement and the second collider
#        pass.  Settled during the FCI slice (2026-05-06).
#
#   D14. PAG collider conflict semantics.  `ColliderDecisions.apply_to_pag`
#        uses last-write semantics on the Z-endpoint mark when collider
#        triples overlap, mirroring `apply_to_cpdag`.  Conflicting writes
#        imply skeleton/sepset inconsistency upstream and should surface
#        through the recorder once `RunRecorder` is fleshed out — not
#        silently via mark preservation.  Settled during the FCI slice.
#
# -----------------------------------------------------------------------------
# OPEN — must be resolved before each item ships:
#
#   O1. **Mixed-data tests.**  Fisher-Z assumes Gaussian; χ² assumes
#       discrete.  Do we ship a default mixed-data test in `cbcd.citest`
#       (e.g. CG/CLG, or a regression-based residual test), or push users
#       to citk once it's available?  Recommendation: ship a regression-
#       residual mixed-data test in v0.1 so cbcd is usable standalone.
#
#   O2. **Default p-value combination for `EnsembleCITest`.**
#       Fisher's method (most powerful, sensitive to outliers), Stouffer's Z
#       (equal-weight, robust), or min-p with Bonferroni (most
#       conservative).  Recommendation: Fisher's as default, with the
#       choice exposed on the EnsembleCITest constructor.
#
#   O3. **`mvpc()` default missingness mechanism.**  Causal-learn supports
#       a generic virtual-data construction that's most appropriate for
#       MAR; MNAR needs a different correction.  Decide whether v0 only
#       supports MAR (with explicit guard against MNAR-looking patterns)
#       or ships separate routines per mechanism.
#
#   O4. **PCMCI `pc_alpha=None` auto-tune grid + criterion.**  Tigramite
#       uses a small grid over {0.05, 0.1, 0.2, 0.3, 0.4} and minimises
#       AIC of the residual model.  Adopt verbatim or pick a different
#       criterion (BIC, cross-validated likelihood)?
#
#   O5. **API stability commitment.**  Proposed: freeze the v0.x surface
#       after PC, FCI, and PCMCI are implemented end-to-end against this
#       design (pressure-tested against three diverse algorithm shapes).
#       Until then, breaking changes allowed at minor-version bumps.
