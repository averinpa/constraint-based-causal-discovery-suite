"""Parity check: SID on every legacy-snapshot pair where 0.1.x produced
a value matches v0.2's result exactly.

The 5 pairs flagged ``"sid": {"skipped": ...}`` in the snapshot were
crashes in 0.1.x (audit bug §1) and are reclaimed via
``tests/fixtures_legacy_v02_overrides.json`` (the v0.2 SID values
become the frozen expected for those pairs going forward).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bnm
from tests.fixtures import from_legacy_fixture, load_legacy_snapshot

_OVERRIDES_PATH = Path(__file__).parent.parent / "fixtures_legacy_v02_overrides.json"


def _load_overrides() -> dict[str, dict[str, float]]:
    if not _OVERRIDES_PATH.exists():
        return {}
    return json.loads(_OVERRIDES_PATH.read_text())["overrides"]


def _pairs_with_sid() -> list[str]:
    snapshot = load_legacy_snapshot()
    return [p["id"] for p in snapshot["pairs"] if "sid" in p]


@pytest.mark.parametrize("pair_id", _pairs_with_sid())
def test_sid_parity(pair_id: str) -> None:
    snapshot = load_legacy_snapshot()
    overrides = _load_overrides()
    pairs_by_id = {p["id"]: p for p in snapshot["pairs"]}
    pair = pairs_by_id[pair_id]

    g1 = from_legacy_fixture(snapshot["fixtures"][pair["g1"]])
    g2 = from_legacy_fixture(snapshot["fixtures"][pair["g2"]])

    pair_overrides = overrides.get(pair_id, {})
    expected_legacy = pair["sid"]
    skipped_in_legacy = isinstance(expected_legacy, dict) and "skipped" in expected_legacy

    actual = bnm.sid(g1, g2)

    if skipped_in_legacy:
        # 0.1.x crashed; v0.2's result is the new source of truth and
        # must be present in the override file.
        for override_key in ("sid_sid", "sid_lower_bound", "sid_upper_bound"):
            assert override_key in pair_overrides, (
                f"{pair_id}: legacy SID was skipped (0.1.x crashed); "
                f"v0.2 expected value missing from override file under key {override_key!r}"
            )
        assert actual.sid == pair_overrides["sid_sid"]
        assert actual.sid_lower_bound == pair_overrides["sid_lower_bound"]
        assert actual.sid_upper_bound == pair_overrides["sid_upper_bound"]
        return

    expected_sid = pair_overrides.get("sid_sid", expected_legacy["sid"])
    expected_lower = pair_overrides.get("sid_lower_bound", expected_legacy["sid_lower_bound"])
    expected_upper = pair_overrides.get("sid_upper_bound", expected_legacy["sid_upper_bound"])
    assert actual.sid == expected_sid, f"{pair_id}.sid: v0.2={actual.sid}, expected={expected_sid}"
    assert actual.sid_lower_bound == expected_lower, (
        f"{pair_id}.lower: v0.2={actual.sid_lower_bound}, expected={expected_lower}"
    )
    assert actual.sid_upper_bound == expected_upper, (
        f"{pair_id}.upper: v0.2={actual.sid_upper_bound}, expected={expected_upper}"
    )
