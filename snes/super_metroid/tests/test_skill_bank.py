"""Skill bank ingest: hop_key inventory, anchors, dual_green, best()."""

from __future__ import annotations

from pathlib import Path

from super_metroid.skill_bank import (
    DEFAULT_BANK_DIR,
    DEFAULT_BANK_PATH,
    HopSkillRecord,
    SkillBank,
    make_hop_key,
    merge_hops_into_bank,
    promote_dual_green,
    promote_from_hop_replay,
    records_from_hops_and_anchors,
    records_from_room_splits,
    records_from_tape,
)
from super_metroid.run_splits import room_splits_from_hops


def _hop(
    index: int,
    room: int,
    start: int,
    end: int,
    *,
    name: str = "",
    items: int | str | None = None,
    end_items: int | str | None = None,
) -> dict:
    row: dict = {
        "index": index,
        "room_id": room,
        "room": f"0x{room:04X}",
        "name": name or f"0x{room:04X}",
        "frame": start,
        "end_frame": end,
        "start_index": start,
        "end_index": end,
        "dwell": end - start + 1,
        "xy": [100, 200],
        "end_xy": [300, 200],
        "pose": 1,
        "end_pose": 1,
    }
    if items is not None:
        row["items"] = (
            f"0x{int(items):04X}" if isinstance(items, int) else items
        )
    if end_items is not None:
        row["end_items"] = (
            f"0x{int(end_items):04X}" if isinstance(end_items, int) else end_items
        )
    return row


def test_default_bank_path() -> None:
    assert DEFAULT_BANK_PATH.name == "bank.json"
    assert DEFAULT_BANK_DIR.name == "skill_bank"
    assert DEFAULT_BANK_PATH.parent == DEFAULT_BANK_DIR


def test_hop_key_includes_items_hex() -> None:
    hops = [
        _hop(0, 0x91F8, 0, 50, name="Landing Site", items=0),
        _hop(1, 0x9E9F, 51, 150, name="Morph Ball Room", items=0, end_items=0x4),
        _hop(2, 0x9F64, 151, 200, name="Construction Zone", items="0x0004"),
    ]
    recs = records_from_hops_and_anchors(hops, source="tape_a", run_id="tape_a")
    assert len(recs) == 3
    # Morph: from Landing, to Construction, items 0
    assert recs[1].hop_key == make_hop_key(
        0x9E9F, from_room_id=0x91F8, to_room_id=0x9F64, items=0
    )
    assert ":0x0000" in recs[1].hop_key
    # Construction inherits start items 0x4 (hex string)
    assert recs[2].hop_key == make_hop_key(
        0x9F64, from_room_id=0x9E9F, to_room_id=None, items=0x4
    )
    assert recs[2].hop_key.endswith(":0x0004")
    # First hop is start → next room
    assert recs[0].hop_key.startswith("0x91F8:start->")


def test_items_fallback_from_prev_end_items() -> None:
    hops = [
        _hop(0, 0x9E9F, 0, 100, items=0, end_items=0x4),
        # no start items — should use previous end_items
        _hop(1, 0x9F64, 101, 150),
    ]
    recs = records_from_hops_and_anchors(hops, source="t")
    assert recs[1].hop_key.endswith(":0x0004")
    assert recs[1].meta.get("items") == "0x0004"


def _touch_state(path: Path) -> str:
    """Create a dummy state file so match_anchor can resolve the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path.resolve())


def test_entry_anchor_path_when_anchors_provided(tmp_path: Path) -> None:
    boot = _touch_state(tmp_path / "boot.state")
    morph_enter = _touch_state(tmp_path / "morph_enter.state")
    morph_far = _touch_state(tmp_path / "morph_far.state")
    hops = [
        _hop(0, 0x91F8, 0, 40, name="Landing Site", items=0),
        _hop(1, 0x9E9F, 100, 180, name="Morph Ball Room", items=0x4),
    ]
    anchors = {
        "anchors": [
            {
                "kind": "boot",
                "frame": 0,
                "room_id": 0x91F8,
                "room": "0x91F8",
                "path": boot,
                "xy": [10, 20],
            },
            {
                "kind": "room_enter",
                "frame": 102,
                "room_id": 0x9E9F,
                "room": "0x9E9F",
                "path": morph_enter,
                "xy": [80, 90],
                "items": "0x0004",
            },
            # farther room_enter should lose to nearer one
            {
                "kind": "room_enter",
                "frame": 500,
                "room_id": 0x9E9F,
                "room": "0x9E9F",
                "path": morph_far,
            },
        ]
    }
    recs = records_from_hops_and_anchors(
        hops, anchors=anchors, source="human", run_id="r1"
    )
    assert recs[0].entry_anchor == boot
    assert recs[0].entry_fingerprint is not None
    assert recs[0].entry_fingerprint["kind"] == "boot"
    assert recs[1].entry_anchor == morph_enter
    assert recs[1].entry_fingerprint is not None
    assert recs[1].entry_fingerprint["kind"] == "room_enter"
    assert recs[1].entry_fingerprint["frame"] == 102
    # leave fingerprint prefers next room_enter when present
    assert recs[0].leave_fingerprint is not None


def test_prefer_room_enter_over_boot(tmp_path: Path) -> None:
    boot = _touch_state(tmp_path / "boot.state")
    enter = _touch_state(tmp_path / "enter.state")
    hops = [_hop(0, 0x91F8, 10, 50, items=0)]
    anchors = [
        {
            "kind": "boot",
            "frame": 10,
            "room_id": 0x91F8,
            "path": boot,
        },
        {
            "kind": "room_enter",
            "frame": 12,
            "room_id": 0x91F8,
            "path": enter,
        },
    ]
    recs = records_from_hops_and_anchors(hops, anchors=anchors, source="s")
    assert recs[0].entry_anchor == enter


def test_dual_green_false_by_default() -> None:
    hops = [_hop(0, 0x9E9F, 0, 80, items=0x4)]
    recs = records_from_hops_and_anchors(hops, source="human")
    assert len(recs) == 1
    assert recs[0].dual_green is False
    # alias
    assert records_from_tape is records_from_hops_and_anchors


def test_best_picks_min_frames() -> None:
    key = make_hop_key(0x9E9F, from_room_id=0x91F8, to_room_id=0x9F64, items=0x4)
    bank = SkillBank()
    bank.add(
        HopSkillRecord(
            hop_key=key,
            room_id=0x9E9F,
            name="Morph",
            frames=120,
            source="slow",
            dual_green=False,
        )
    )
    bank.add(
        HopSkillRecord(
            hop_key=key,
            room_id=0x9E9F,
            name="Morph",
            frames=90,
            source="fast",
            dual_green=False,
        )
    )
    bank.add(
        HopSkillRecord(
            hop_key=key,
            room_id=0x9E9F,
            name="Morph",
            frames=100,
            source="verified",
            dual_green=True,
        )
    )
    best = bank.best(key)
    assert best is not None
    # dual_green preferred even if slightly slower than hollow 90
    assert best.frames == 100
    assert best.dual_green is True
    best_any = bank.best(key, require_dual_green=False)
    assert best_any is not None
    assert best_any.frames == 100  # dual_green sorts first


def test_best_min_frames_among_hollow() -> None:
    key = make_hop_key(0x91F8, items=None)
    bank = SkillBank()
    bank.add(
        HopSkillRecord(hop_key=key, room_id=0x91F8, name="L", frames=50, source="a")
    )
    bank.add(
        HopSkillRecord(hop_key=key, room_id=0x91F8, name="L", frames=40, source="b")
    )
    best = bank.best(key)
    assert best is not None
    assert best.frames == 40
    assert best.source == "b"


def test_records_from_room_splits_items_per_leaf() -> None:
    hops = [
        _hop(0, 0x91F8, 0, 50, name="Landing Site"),
        _hop(1, 0x9E9F, 51, 150, name="Morph Ball Room"),
    ]
    rooms = room_splits_from_hops(hops)
    recs = records_from_room_splits(
        rooms,
        source="split",
        items_per_leaf=[0, 0x4],
    )
    assert recs[0].hop_key.endswith(":0x0000")
    assert recs[1].hop_key.endswith(":0x0004")
    # single items still works
    recs2 = records_from_room_splits(rooms, source="split", items=0x4)
    assert all(r.hop_key.endswith(":0x0004") for r in recs2)


def test_merge_hops_into_bank() -> None:
    hops_a = [_hop(0, 0x9E9F, 0, 100, items=0x4)]
    hops_b = [_hop(0, 0x9E9F, 0, 70, items=0x4)]
    bank = SkillBank()
    merge_hops_into_bank(bank, [("a", hops_a), ("b", hops_b)])
    keys = list(bank.records)
    assert len(keys) == 1
    best = bank.best(keys[0])
    assert best is not None
    assert best.frames == 71  # dwell 0..70 inclusive
    assert best.run_id == "b"
    assert best.dual_green is False
    assert ":0x0004" in keys[0]


def test_promote_dual_green_updates_existing(tmp_path: Path) -> None:
    key = make_hop_key(0x9E9F, from_room_id=0x91F8, to_room_id=0x9F64, items=0x4)
    bank_path = tmp_path / "bank.json"
    bank = SkillBank()
    bank.add(
        HopSkillRecord(
            hop_key=key,
            room_id=0x9E9F,
            name="Morph",
            frames=120,
            source="full_start_v1",
            dual_green=False,
            meta={"hop_index": 1},
        )
    )
    bank.save(bank_path)

    rec = promote_dual_green(
        key,
        bank_path=bank_path,
        source="full_start_v1",
        frames=100,
        entry_anchor="/tmp/enter.state",
    )
    assert rec.dual_green is True
    assert rec.frames == 100
    reloaded = SkillBank.load(bank_path)
    best = reloaded.best(key, require_dual_green=True)
    assert best is not None
    assert best.dual_green is True
    assert best.entry_anchor == "/tmp/enter.state"


def test_promote_from_hop_replay_matches_hop_index(tmp_path: Path) -> None:
    key = make_hop_key(0xA6E2, from_room_id=None, to_room_id=0xA59F, items=0x1105)
    bank_path = tmp_path / "bank.json"
    bank = SkillBank()
    bank.add(
        HopSkillRecord(
            hop_key=key,
            room_id=0xA6E2,
            name="Varia",
            frames=900,
            source="full_start_v1",
            dual_green=False,
            meta={"hop_index": 0},
        )
    )
    bank.save(bank_path)

    report = {
        "green": True,
        "ok": True,
        "dual": True,
        "anchor_path": "/tmp/varia_enter.state",
        "assist": True,
        "slice": {
            "hop_index": 0,
            "start_room": 0xA6E2,
            "leave_room": 0xA59F,
            "steps": 850,
            "name": "full_start_v1",
            "task": "tasks/full_start_v1.json",
            "hop": {
                "index": 0,
                "room_id": 0xA6E2,
                "name": "Varia Suit Room",
                "items": "0x1105",
                "dwell": 900,
            },
        },
    }
    rec = promote_from_hop_replay(
        report, bank_path=bank_path, source="full_start_v1"
    )
    assert rec is not None
    assert rec.hop_key == key
    assert rec.dual_green is True
    assert rec.frames == 850
    assert SkillBank.load(bank_path).best(key, require_dual_green=True) is not None
