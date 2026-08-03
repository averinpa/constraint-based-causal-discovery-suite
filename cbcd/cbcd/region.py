"""Region-scoped orientation (roadmap Phase 0c).

Turns a region skeleton (from ``PCStable(..., variables=region)``) into a *local* CPDAG with
boundary-aware semantics:

* v-structures are oriented only at **interior** centres — nodes whose adjacency is fully explored;
* Meek closure never orients a **boundary-incident** edge (an edge touching a node outside the
  interior).

This is sound but deliberately conservative: edges touching the boundary stay undirected, because
structure outside the region could overturn them. The caller (region-grow, Phase 3) is responsible
for choosing the interior/boundary split so that every interior node's adjacency is genuinely
complete within the region.
"""

from __future__ import annotations

from collections.abc import Iterable

from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.collider import SepsetOrienter
from cbcd.graph.cpdag import CPDAG
from cbcd.recording import RunRecorder
from cbcd.rules import MeekRules
from cbcd.skeleton import Skeleton


def orient_region(
    skeleton: Skeleton,
    ci: CITest,
    *,
    interior: Iterable[int],
    alpha: float = 0.05,
    background: BackgroundKnowledge | None = None,
    var_names: tuple[str, ...] | None = None,
    recorder: RunRecorder | None = None,
) -> CPDAG:
    """Orient a region skeleton into a local CPDAG (interior oriented, boundary left undirected).

    ``interior`` is the set of fully-explored nodes; v-structures are gated on interior centres and
    Meek is frozen on boundary-incident edges. Nodes outside the skeleton's active set are isolated.
    """
    interior_set = frozenset(int(v) for v in interior)
    decisions = SepsetOrienter()(
        skeleton,
        ci,
        alpha=alpha,
        background=background,
        interior=interior_set,
        recorder=recorder,
    )
    partial = decisions.apply_to_cpdag(skeleton, var_names=var_names)
    return MeekRules()(
        partial,
        background=background,
        interior=interior_set,
        recorder=recorder,
    )
