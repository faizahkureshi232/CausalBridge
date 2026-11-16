import json
from pathlib import Path

from causalbridge.discovery.data_audit import PILOT_PAIR_IDS, data_exclusion


def test_data_exclusions_follow_the_declared_order():
    stats = {"rows": 200, "blank": 0, "bad_width": 0, "nonfinite": 0, "unique": [10, 10]}
    assert data_exclusion(1, (1, 1, 2, 2), 0, stats, 100) == "zero_weight"
    assert data_exclusion(1, (1, 2, 3, 3), 1, stats, 100) == "multicolumn"
    assert data_exclusion(1, (1, 1, 2, 2), 1, stats | {"blank": 1}, 100) == "invalid_rows"
    assert data_exclusion(1, (1, 1, 2, 2), 1, stats | {"rows": 90}, 100) == "insufficient_rows"
    assert (
        data_exclusion(1, (1, 1, 2, 2), 1, stats | {"unique": [10, 2]}, 100)
        == "insufficient_variation"
    )
    assert data_exclusion(19, (1, 1, 2, 2), 1, stats, 100) == "time_dependence"
    assert data_exclusion(5, (1, 1, 2, 2), 1, stats, 100) is None


def test_committed_manifest_keeps_the_pilot_pairs_and_assigns_every_group():
    manifest = json.loads(Path("configs/pilot-data.json").read_text())
    included = {
        pair["pair_id"]
        for pair in manifest["pairs"]
        if pair["split"] == "pilot" and "exclusion_reason" not in pair
    }
    assert included == set(PILOT_PAIR_IDS)
    groups: dict[str, set[str]] = {}
    eligible: dict[str, int] = {}
    for pair in manifest["pairs"]:
        groups.setdefault(pair["source_group"], set()).add(pair["split"])
        if "exclusion_reason" not in pair:
            eligible[pair["source_group"]] = eligible.get(pair["source_group"], 0) + 1
        else:
            assert pair["exclusion_reason"] != "reserved_not_eligibility_audited"
    assert all(len(splits) == 1 for splits in groups.values())
    for group, splits in groups.items():
        split = next(iter(splits))
        if split in {"pilot", "development", "test"}:
            assert eligible.get(group, 0) >= 1
        else:
            assert split == "ineligible"
            assert group not in eligible
    pilot_groups = {
        pair["source_group"] for pair in manifest["pairs"] if pair["pair_id"] in included
    }
    held_out = {
        pair["source_group"]
        for pair in manifest["pairs"]
        if pair["split"] in {"development", "test"}
    }
    assert pilot_groups.isdisjoint(held_out)


def test_manifest_matches_the_local_snapshot_when_present():
    snapshot = Path("runs/pilot-preparation/raw")
    if not (snapshot / "pairmeta.txt").exists():
        return
    from causalbridge.discovery.data_audit import check_manifest

    manifest = json.loads(Path("configs/pilot-data.json").read_text())
    check_manifest(manifest, snapshot)
