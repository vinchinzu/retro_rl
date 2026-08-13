"""materialize_take: settled hops + run_timing + bank records (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.materialize import materialize_take
from super_metroid.skill_bank import SkillBank


def _write_mini_task(tmp: Path) -> Path:
    """Two-room take with transition frames then settle + item delta."""
    trace = []
    # Room A ordinary
    for i in range(0, 5):
        trace.append(
            {
                "frame": i,
                "room": 0x91F8,
                "x": 100 + i,
                "y": 100,
                "pose": 1,
                "items": 0,
                "door_transition": 0,
                "phase": "ordinary_gameplay",
            }
        )
    # Room B transition then settle; item pickup mid-hop
    for i in range(5, 9):
        trace.append(
            {
                "frame": i,
                "room": 0x9E9F,
                "x": 10,
                "y": 200,
                "pose": 1,
                "items": 0,
                "door_transition": 1,
                "phase": "room_transition",
            }
        )
    for i in range(9, 20):
        items = 0x4 if i >= 15 else 0
        trace.append(
            {
                "frame": i,
                "room": 0x9E9F,
                "x": 20 + i,
                "y": 200,
                "pose": 2,
                "items": items,
                "door_transition": 0,
                "phase": "ordinary_gameplay",
            }
        )
    frames = [[0] * 12 for _ in trace]
    task = {
        "name": "mini_mat",
        "frame_count": len(frames),
        "frames": frames,
        "trace": trace,
        "start_state": "scratch/x.state",
        "metadata": {
            "assist": {"unlimited_energy": True, "unlimited_ammo": True},
            "end_fingerprint": {
                "kind": "end",
                "frame": 19,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "xy": [39, 200],
                "pose": 2,
                "items": "0x0004",
            },
        },
    }
    path = tmp / "mini_mat.json"
    path.write_text(json.dumps(task), encoding="utf-8")

    # State files must exist — match_anchor only scores resolvable pins.
    boot_path = tmp / "boot.state"
    enter_path = tmp / "enter_morph.state"
    item_path = tmp / "item.state"
    for p in (boot_path, enter_path, item_path):
        p.write_bytes(b"")
    anchors = {
        "task": "mini_mat",
        "anchors_dir": str(tmp / "mini_mat_anchors"),
        "count": 3,
        "anchors": [
            {
                "kind": "boot",
                "frame": 0,
                "room": "0x91F8",
                "room_id": 0x91F8,
                "xy": [100, 100],
                "path": str(boot_path),
                "items": "0x0000",
            },
            {
                "kind": "room_enter",
                "frame": 9,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "xy": [20, 200],
                "path": str(enter_path),
                "items": "0x0000",
            },
            {
                "kind": "item_delta",
                "frame": 15,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "xy": [35, 200],
                "path": str(item_path),
                "items": "0x0004",
            },
        ],
    }
    (tmp / "mini_mat_anchors.json").write_text(
        json.dumps(anchors, indent=2) + "\n", encoding="utf-8"
    )
    return path


def test_materialize_settles_and_writes_sidecars(tmp_path: Path) -> None:
    task = _write_mini_task(tmp_path)
    bank_path = tmp_path / "skill_bank" / "bank.json"
    result = materialize_take(
        task,
        write=True,
        merge_bank=True,
        bank_path=bank_path,
    )
    assert result.run_timing_path is not None and result.run_timing_path.is_file()
    assert result.extract_path is not None and result.extract_path.is_file()
    assert result.bank_path is not None and result.bank_path.is_file()

    # Settled: morph hop starts after transition (index 9), not edge (5).
    assert len(result.hops_settled) == 2
    morph = result.hops_settled[1]
    assert morph["start_index"] == 9
    assert morph.get("transition_frames") == 4
    assert morph.get("settled_entry") is True

    # Raw edge preserved separately
    assert result.hops_raw[1]["start_index"] == 5

    timing = result.run_timing
    assert timing["summary"]["room_visits"] == 2
    assert timing["summary"]["item_splits"] >= 1

    # Hop bodies exported for hill-climb / bank seeds
    assert len(result.hop_body_paths) == 2
    for bp in result.hop_body_paths:
        assert Path(bp).is_file()
        body = json.loads(Path(bp).read_text(encoding="utf-8"))
        assert body["frame_count"] > 0
        assert "frames" in body

    # Bank records non-hollow: items in key + entry pin on morph + body_path
    assert len(result.bank_records) == 2
    morph_rec = result.bank_records[1]
    assert "0x0004" in morph_rec.hop_key or "0x0000" in morph_rec.hop_key
    assert morph_rec.entry_anchor is not None
    assert morph_rec.dual_green is False
    assert morph_rec.body_path is not None
    assert morph_rec.meta.get("source_task")

    bank = SkillBank.load(bank_path)
    assert bank.best(morph_rec.hop_key) is not None
    assert bank.best(morph_rec.hop_key).body_path is not None


def test_materialize_no_write(tmp_path: Path) -> None:
    task = _write_mini_task(tmp_path)
    result = materialize_take(task, write=False)
    assert result.run_timing_path is None
    assert result.extract_path is None
    assert len(result.hops_settled) == 2
    assert "summary" in result.run_timing
