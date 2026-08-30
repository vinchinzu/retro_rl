"""Ten item-seam lanes and non-overlapping ownership leases (no ROM)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from super_metroid.hop_id import make_hop_key
from super_metroid.leave_specs import LeaveSpec
from super_metroid.splice.cards import generate_cards
from super_metroid.splice.lanes import ITEM_SEAM_LANES, inventory_from_manifest
from super_metroid.splice.leases import (
    Lease,
    LeaseError,
    grant_lease,
    lease_for_lane,
    lease_from_card,
    rollup_candidates,
)
from super_metroid.splice.schema import LeaveSpecRef, RouteManifest

CERES = 0xDF45
LANDING = 0x91F8
SEAM_ROOMS = (
    0xCA52,
    0xCE40,
    0xAC2B,
    0xCFC9,
    0xD9AA,
    0xD2AA,
    0xB283,
    0xB6C1,
    0xB62B,
    0xB32E,
)
SEAM_SEGMENTS = (
    "s23",
    "s24",
    "s25",
    "s26",
    "s27",
    "s29",
    "s30",
    "s31",
    "s32",
    "s33",
)
LANE_NAMES = tuple(spec.name for spec in ITEM_SEAM_LANES)


def _leave(hop: str, room: int) -> dict[str, Any]:
    spec = LeaveSpec(hop=hop, room=room, x=(20, 80), y=(100, 180), pose_class="stand")
    return LeaveSpecRef.from_leave_spec(spec).to_dict()


def _edge(
    task_id: str,
    room: int,
    *,
    pred_room: int | None,
    next_room: int | None,
    leave_room: int,
    items: int | None = 0,
    goal: str | None = None,
    selected: dict[str, str] | None = None,
    allowed: tuple[str, ...] = ("tape", "controller"),
    order: int = 0,
    segment: str | None = "s1",
    tape: str | None = None,
    notes: tuple[str, ...] = ("synthetic",),
    owner: str = "snes/super_metroid/routes/kpdr",
) -> dict[str, Any]:
    hop_key = make_hop_key(
        room, from_room_id=pred_room, to_room_id=next_room, items=items, goal=goal
    )
    return {
        "task_id": task_id,
        "hop_key": hop_key,
        "room_id": room,
        "predecessor_room_id": pred_room,
        "next_room_id": next_room,
        "goal": goal,
        "required_items": items,
        "entry": {
            "fingerprint": {"room_id": room, "prior_room_id": pred_room, "items": items},
            "state_path": None,
            "state_digest": None,
        },
        "successor_leave": _leave(f"{task_id}_leave", leave_room),
        "allowed_kinds": list(allowed),
        "selected": selected or {"scaffold": "tape:board", "survival": "controller:play"},
        "owner_package": owner,
        "integration_order": order,
        "max_frames": 400,
        "max_no_progress": 200,
        "segment": segment,
        "hop_index": order,
        "tape_path": tape,
        "source_notes": list(notes),
    }


def _tiny_manifest() -> dict[str, Any]:
    e0 = _edge(
        "ceres_elev",
        CERES,
        pred_room=None,
        next_room=LANDING,
        leave_room=LANDING,
        order=0,
        segment="s1",
    )
    e1 = _edge(
        "landing",
        LANDING,
        pred_room=CERES,
        next_room=None,
        leave_room=LANDING,
        order=1,
        segment="s1",
    )
    return {"route_id": "tiny", "variant": "kpdr", "edges": [e0, e1]}


def _seam_manifest(**kwargs: Any) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    n = len(SEAM_ROOMS)
    for i, room in enumerate(SEAM_ROOMS):
        nxt = SEAM_ROOMS[i + 1] if i + 1 < n else None
        pred = SEAM_ROOMS[i - 1] if i else None
        extra = dict(kwargs.get(f"e{i}", {}))
        edges.append(
            _edge(
                extra.pop("task_id", f"seam_{i}"),
                room,
                pred_room=pred,
                next_room=nxt,
                leave_room=nxt if nxt is not None else room,
                order=i,
                segment=extra.pop("segment", SEAM_SEGMENTS[i]),
                **extra,
            )
        )
    return {"route_id": "item_seams", "variant": "kpdr", "edges": edges}


def test_ten_named_lanes_exist() -> None:
    for source in (
        RouteManifest.from_dict({"route_id": "empty", "edges": []}),
        RouteManifest.from_dict(_tiny_manifest()),
        RouteManifest.from_dict(_seam_manifest()),
    ):
        lanes = inventory_from_manifest(source)
        assert len(lanes) == 10
        assert tuple(lane.name for lane in lanes) == LANE_NAMES
        assert [lane.lane_id for lane in lanes] == [spec.lane_id for spec in ITEM_SEAM_LANES]


def test_artifact_dirs_unique() -> None:
    lanes = inventory_from_manifest(RouteManifest.from_dict(_seam_manifest()))
    dirs = [lane.artifact_dir for lane in lanes]
    assert len(dirs) == len(set(dirs))
    owners = [lane.owner_package for lane in lanes]
    assert len(owners) == len(set(owners))
    for lane in lanes:
        assert lane.artifact_dir.startswith("snes/super_metroid/recordings/splice/lanes/")
        assert not Path(lane.artifact_dir).is_absolute()
        assert not Path(lane.owner_package).is_absolute()


def test_filter_assigns_seam_segments() -> None:
    lanes = inventory_from_manifest(RouteManifest.from_dict(_seam_manifest()))
    for i, lane in enumerate(lanes):
        assert lane.task_ids == (f"seam_{i}",)
    raw = _seam_manifest()
    raw["edges"][0]["segment"] = None
    raw["edges"][0]["tape_path"] = "tasks/full_start_v1_segments/s23/tape.json"
    via_tape = inventory_from_manifest(RouteManifest.from_dict(raw))
    assert via_tape[0].task_ids == ("seam_0",)


def test_tiny_synthetic_assigns_by_hop_order() -> None:
    lanes = inventory_from_manifest(RouteManifest.from_dict(_tiny_manifest()))
    assert lanes[0].task_ids == ("ceres_elev",)
    assert lanes[1].task_ids == ("landing",)
    assert all(not lane.task_ids for lane in lanes[2:])


def test_synthetic_labels_beat_hop_order() -> None:
    raw = _tiny_manifest()
    raw["edges"][1]["task_id"] = "escape_ship"
    raw["edges"][1]["goal"] = "credits"
    raw["edges"][1]["hop_key"] = make_hop_key(
        LANDING, from_room_id=CERES, to_room_id=None, items=0, goal="credits"
    )
    lanes = inventory_from_manifest(RouteManifest.from_dict(raw))
    assert "escape_ship" in lanes[-1].task_ids
    assert lanes[0].task_ids == ("ceres_elev",)


def test_superseded_s28_skipped() -> None:
    skipped = _seam_manifest(
        e5={"segment": "s28", "task_id": "plasma_old", "notes": ("s28 superseded",)}
    )
    lanes = inventory_from_manifest(RouteManifest.from_dict(skipped))
    plasma = lanes[5]
    assert plasma.lane_id == "plasma_golden_torizo"
    assert "plasma_old" not in plasma.task_ids
    kept = _seam_manifest(e5={"segment": "s28", "task_id": "plasma_retake"})
    kept_lanes = inventory_from_manifest(RouteManifest.from_dict(kept))
    assert kept_lanes[5].task_ids == ("plasma_retake",)


def test_disjoint_lane_leases_granted() -> None:
    lanes = inventory_from_manifest(RouteManifest.from_dict(_seam_manifest()))
    existing: list[Lease] = []
    for i, lane in enumerate(lanes):
        result = grant_lease(
            lease_for_lane(lane, branch=f"worker/{lane.lane_id}", card_revision=1),
            existing,
        )
        assert result.granted, result.reason
        assert result.lease is not None
        existing.append(result.lease)
    assert len(existing) == 10


def test_overlapping_owner_paths_rejected() -> None:
    a = Lease(
        task_id="attic",
        card_revision=1,
        branch="w/a",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/attic_gravity",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/attic_gravity/",
        lane_id="attic_gravity",
    )
    nested = Lease(
        task_id="bowling",
        card_revision=1,
        branch="w/b",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/attic_gravity/bowling.py",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/bowling/",
        lane_id="attic_gravity",
    )
    result = grant_lease(nested, (a,))
    assert not result.granted
    assert "owner_paths" in result.reason


def test_overlapping_artifact_dirs_rejected() -> None:
    a = Lease(
        task_id="a",
        card_revision=1,
        branch="w/a",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/a",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/shared/",
    )
    b = Lease(
        task_id="b",
        card_revision=1,
        branch="w/b",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/b",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/shared/child/",
    )
    result = grant_lease(b, (a,))
    assert not result.granted
    assert "artifact_dir" in result.reason


def test_expired_lease_does_not_block_owner() -> None:
    old = Lease(
        task_id="old",
        card_revision=1,
        branch="w/old",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/attic_gravity",),
        expiry="2000-01-01T00:00:00+00:00",
        artifact_dir="snes/super_metroid/recordings/splice/lanes/old/",
    )
    new = Lease(
        task_id="new",
        card_revision=2,
        branch="w/new",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/attic_gravity",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/new/",
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = grant_lease(new, (old,), now=now)
    assert result.granted


def test_blank_owner_paths_rejected() -> None:
    blank = Lease(
        task_id="blank",
        card_revision=1,
        branch="w/blank",
        owner_paths=("", "  "),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/blank/",
    )
    with pytest.raises(LeaseError, match="owner_paths"):
        grant_lease(blank)
    mixed = Lease(
        task_id="mixed",
        card_revision=1,
        branch="w/mixed",
        owner_paths=("", "  ", "snes/super_metroid/routes/kpdr/seams/mixed"),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/mixed/",
    )
    result = grant_lease(mixed)
    assert result.granted
    assert result.lease is not None
    assert result.lease.owner_paths == ("snes/super_metroid/routes/kpdr/seams/mixed",)
    other = Lease(
        task_id="other",
        card_revision=1,
        branch="w/other",
        owner_paths=("snes/super_metroid/routes/kpdr/seams/mixed",),
        expiry=None,
        artifact_dir="snes/super_metroid/recordings/splice/lanes/other/",
    )
    blocked = grant_lease(other, (result.lease,))
    assert not blocked.granted
    assert "owner_paths" in blocked.reason


def test_shared_card_owner_package_rejected() -> None:
    cards = generate_cards(RouteManifest.from_dict(_tiny_manifest()))
    first = lease_from_card(cards[0], branch="w/0")
    second = lease_from_card(cards[1], branch="w/1")
    result = grant_lease(second, (first,))
    assert not result.granted
    assert "owner_paths" in result.reason


def test_rollup_selects_ids_without_bank_writes(monkeypatch: Any, tmp_path: Path) -> None:
    from super_metroid.skill_bank import DEFAULT_BANK_PATH

    manifest = RouteManifest.from_dict(_tiny_manifest())
    writes: list[str] = []
    real_write = Path.write_text

    def _track(self: Path, *args: Any, **kwargs: Any) -> Any:
        writes.append(str(self))
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _track)
    existed = DEFAULT_BANK_PATH.is_file()
    mtime = DEFAULT_BANK_PATH.stat().st_mtime if existed else None
    rollup = rollup_candidates(
        manifest,
        [
            {
                "candidate_id": "tape:slow",
                "kind": "tape",
                "implementation_id": "s1",
                "task_id": "ceres_elev",
                "entry_fingerprint": {"room_id": CERES},
                "frame_count": 80,
                "replay_rows": [{"trial": 1, "passed": True}],
            },
            {
                "candidate_id": "tape:fast",
                "kind": "tape",
                "implementation_id": "s1",
                "task_id": "ceres_elev",
                "entry_fingerprint": {"room_id": CERES},
                "frame_count": 40,
                "replay_rows": [
                    {"trial": 1, "passed": True},
                    {"trial": 2, "passed": True},
                ],
            },
        ],
        profile="scaffold",
    )
    assert rollup.as_map()["ceres_elev"] == "tape:fast"
    assert rollup.as_map()["landing"] == "tape:board"
    assert not any(Path(path).name == "bank.json" for path in writes)
    assert not (tmp_path / "bank.json").exists()
    if existed:
        assert DEFAULT_BANK_PATH.stat().st_mtime == mtime
    else:
        assert not DEFAULT_BANK_PATH.exists()
