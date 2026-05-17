"""Parity check: every comparative metric on every legacy-snapshot pair
matches bnmetrics 0.1.x output, EXCEPT where ``fixtures_legacy_v02_overrides.json``
records a documented intentional divergence.

The override file captures the 17 pairs where 0.1.x's `count_reversals`
under-counts because of an nx.DiGraph storage-direction asymmetry — see
``docs/audit.md`` bug §7. v0.2 emits the corrected values; the override
file freezes them so a future change to v0.2's reversal semantics
breaks the test loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bnmetrics.comparative import all_comparative
from tests.fixtures import from_legacy_fixture, load_legacy_snapshot

_OVERRIDES_PATH = Path(__file__).parent.parent / "fixtures_legacy_v02_overrides.json"


def _load_overrides() -> dict[str, dict[str, float]]:
    if not _OVERRIDES_PATH.exists():
        return {}
    return json.loads(_OVERRIDES_PATH.read_text())["overrides"]


def _all_pair_ids() -> list[str]:
    snapshot = load_legacy_snapshot()
    return [p["id"] for p in snapshot["pairs"]]


@pytest.mark.parametrize("pair_id", _all_pair_ids())
def test_comparative_parity(pair_id: str) -> None:
    snapshot = load_legacy_snapshot()
    overrides = _load_overrides()
    pairs_by_id = {p["id"]: p for p in snapshot["pairs"]}
    pair = pairs_by_id[pair_id]

    g1 = from_legacy_fixture(snapshot["fixtures"][pair["g1"]])
    g2 = from_legacy_fixture(snapshot["fixtures"][pair["g2"]])

    actual = all_comparative(g1, g2, keys="legacy")
    expected = pair["comparative"]
    pair_overrides = overrides.get(pair_id, {})

    for name, snapshot_value in expected.items():
        act_value = actual[name]
        exp_value = pair_overrides.get(name, snapshot_value)
        if isinstance(exp_value, float):
            assert abs(act_value - exp_value) < 1e-12, (
                f"{pair_id}.{name}: v0.2 returned {act_value}, expected {exp_value}"
            )
        else:
            assert act_value == exp_value, (
                f"{pair_id}.{name}: v0.2 returned {act_value}, expected {exp_value}"
            )
