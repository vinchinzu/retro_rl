"""splice route/task/candidate schemas (no ROM)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from super_metroid.hop_id import make_hop_key
from super_metroid.leave_specs import LeaveSpec
from super_metroid.splice import (
    FORBIDDEN_HOT_FILES,
    NON_CLAIMS,
    CandidateArtifact,
    SchemaError,
    generate_cards,
    manifest_from_board,
    repo_relative,
)
from super_metroid.splice.cards import assembly_table, format_card, format_cards
from super_metroid.splice.schema import (
    LeaveSpecRef,
    RouteManifest,
    candidate_kind,
)

CERES = 0xDF45
LANDING = 0x91F8


def _leave(hop: str, room: int) -> dict[str, Any]:
    spec = LeaveSpec(hop=hop, room=room, x=(20, 80), y=(100, 180), pose_class="stand")
    return LeaveSpecRef.from_leave_spec(spec).to_dict()


def _entry(
    room: int,
    *,
    path: str | None = None,
    digest: str | None = None,
    prior: int | None = None,
) -> dict[str, Any]:
    return {
        "fingerprint": {
            "room_id": room,
            "x": 40,
            "y": 120,
            "pose": 1,
            "velocity_x": 0,
            "sub_x": 0,
            "momentum_x": 0,
            "items": 0,
            "beams": 0,
            "prior_room_id": prior,
            "enemy_phase": "none",
        },
        "state_path": path,
        "state_digest": digest,
    }


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
    path: str | None = None,
    digest: str | None = None,
    order: int = 0,
    tape: str | None = None,
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
        "entry": _entry(room, path=path, digest=digest, prior=pred_room),
        "successor_leave": _leave(f"{task_id}_leave", leave_room),
        "allowed_kinds": list(allowed),
        "selected": selected or {"scaffold": "tape:board", "survival": "controller:play"},
        "owner_package": "snes/super_metroid/routes/kpdr",
        "integration_order": order,
        "max_frames": 400,
        "max_no_progress": 200,
        "segment": "s1",
        "hop_index": order,
        "frame_start": 10 * order,
        "frame_end": 10 * order + 50,
        "tape_path": tape,
        "tape_digest": "ab" * 32 if tape else None,
        "source_notes": ["synthetic"],
    }


def _tiny_manifest(**kwargs: Any) -> dict[str, Any]:
    e0 = _edge(
        "ceres_elev",
        CERES,
        pred_room=None,
        next_room=LANDING,
        leave_room=LANDING,
        order=0,
        **kwargs.get("e0", {}),
    )
    e1 = _edge(
        "landing",
        LANDING,
        pred_room=CERES,
        next_room=None,
        leave_room=LANDING,
        goal="credits",
        order=1,
        **kwargs.get("e1", {}),
    )
    return {"route_id": "tiny", "variant": "kpdr", "edges": [e0, e1]}


def _walk_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_walk_strings(item))
    return out


def test_round_trip_two_edge_manifest() -> None:
    raw = _tiny_manifest()
    manifest = RouteManifest.from_dict(raw)
    assert len(manifest.edges) == 2
    assert manifest.edges[0].successor_task_id == "landing"
    assert manifest.edges[1].predecessor_task_id == "ceres_elev"
    assert manifest.edges[0].hop_key == make_hop_key(
        CERES, from_room_id=None, to_room_id=LANDING, items=0
    )
    again = RouteManifest.from_dict(manifest.to_dict())
    assert again.to_dict() == manifest.to_dict()
    cand = CandidateArtifact.from_dict(
        {
            "candidate_id": "tape:ceres",
            "kind": "tape",
            "implementation_id": "s1",
            "task_id": "ceres_elev",
            "entry_fingerprint": {"room_id": CERES, "x": 1, "y": 2},
            "replay_rows": [
                {"trial": 1, "passed": True, "frames": 10},
                {"trial": 2, "passed": True, "frames": 10},
            ],
            "join_rows": [
                {
                    "trial": 1,
                    "predecessor_task_id": "start",
                    "candidate_id": "tape:ceres",
                    "successor_task_id": "landing",
                    "passed": True,
                },
                {
                    "trial": 2,
                    "predecessor_task_id": "start",
                    "candidate_id": "tape:ceres",
                    "successor_task_id": "landing",
                    "passed": True,
                },
            ],
            "memory_writes": [
                {
                    "frame": 3,
                    "address": 0x0F8C,
                    "entity": "enemy0",
                    "old": 20,
                    "new": 1,
                    "reason": "scaffold_hp_clamp",
                }
            ],
            "leftover_state_path": "snes/super_metroid/recordings/splice/ceres_elev/leftover.state",
            "parent_candidate_id": "tape:parent",
        }
    )
    assert CandidateArtifact.from_dict(cand.to_dict()).to_dict() == cand.to_dict()
    assert cand.kind == "tape"


def test_card_includes_forbidden_nonclaims_digest_join(tmp_path: Path) -> None:
    pin = tmp_path / "enter.state"
    pin.write_bytes(b"pin-bytes")
    digest = hashlib.sha256(b"pin-bytes").hexdigest()
    rel = repo_relative(pin)
    raw = _tiny_manifest(e0={"path": str(pin.resolve()), "digest": digest})
    cards = generate_cards(RouteManifest.from_dict(raw), profile="scaffold")
    assert len(cards) == 2
    card = cards[0]
    text = format_card(card)
    assert card.entry_state_digest == digest
    assert card.entry_state_path == rel
    assert card.join.leave.digest
    assert card.join.leave.room == LANDING
    assert card.join.next_entry is not None
    assert card.checkbox == "sync_green"
    for hot in FORBIDDEN_HOT_FILES:
        assert hot in card.forbidden_files
        assert hot in text
    assert any("landing-residual" in p for p in card.forbidden_files)
    for claim in NON_CLAIMS:
        assert claim in card.non_claims
        assert claim in text
    assert "join:" in text
    assert "entry_digest:" in text
    assert card.completion.next_boot_on_red.endswith("leftover.state")
    table = assembly_table(RouteManifest.from_dict(raw))
    assert [row["task_id"] for row in table] == [e.task_id for e in RouteManifest.from_dict(raw).edges]


def test_generate_cards_does_not_mutate_inputs() -> None:
    raw = _tiny_manifest()
    snapshot = json.loads(json.dumps(raw))
    cards = generate_cards(raw, profile="scaffold")
    assert raw == snapshot
    assert cards[0].task_id == "ceres_elev"
    manifest = RouteManifest.from_dict(raw)
    before = manifest.to_dict()
    generate_cards(manifest, profile="clean")
    assert manifest.to_dict() == before


def test_invalid_room_rejected_or_flagged() -> None:
    raw = _tiny_manifest()
    raw["edges"][0]["room_id"] = 0x5555
    raw["edges"][0]["hop_key"] = make_hop_key(
        0x5555, from_room_id=None, to_room_id=LANDING, items=0
    )
    raw["edges"][0]["entry"]["fingerprint"]["room_id"] = 0x5555
    manifest = RouteManifest.from_dict(raw)
    assert manifest.edges[0].invalid_room
    with pytest.raises(SchemaError) as exc:
        manifest.validate()
    assert "0x0000" in str(exc.value) or "0x5555" in str(exc.value) or "invalid room" in str(exc.value)
    raw0 = _tiny_manifest()
    raw0["edges"][1]["room_id"] = 0x0000
    raw0["edges"][1]["hop_key"] = make_hop_key(
        0x0000, from_room_id=CERES, to_room_id=None, items=0, goal="credits"
    )
    raw0["edges"][1]["entry"]["fingerprint"]["room_id"] = 0x0000
    flagged = RouteManifest.from_dict(raw0)
    assert flagged.edges[1].invalid_room
    cards = generate_cards(flagged)
    assert cards[1].invalid_room


def test_selected_candidate_must_be_allowed_kind() -> None:
    raw = _tiny_manifest(
        e0={"allowed": ("tape",), "selected": {"scaffold": "controller:ws_main"}}
    )
    with pytest.raises(SchemaError) as exc:
        RouteManifest.from_dict(raw)
    assert "allowed kinds" in str(exc.value)
    ok = _tiny_manifest(e0={"allowed": ("tape", "controller"), "selected": {"scaffold": "tape:s23"}})
    manifest = RouteManifest.from_dict(ok)
    assert candidate_kind(manifest.edges[0].selected_map()["scaffold"]) == "tape"
    empty = _tiny_manifest(e0={"selected": {"scaffold": ""}})
    with pytest.raises(SchemaError):
        RouteManifest.from_dict(empty)


def test_repo_relative_paths_only(tmp_path: Path) -> None:
    pin = tmp_path / "boot.state"
    pin.write_bytes(b"x")
    raw = _tiny_manifest(e0={"path": str(pin.resolve()), "tape": str(pin.resolve())})
    manifest = RouteManifest.from_dict(raw)
    assert manifest.edges[0].entry.state_path is not None
    assert not Path(manifest.edges[0].entry.state_path).is_absolute()
    assert not manifest.edges[0].entry.state_path.startswith("/")
    cards = generate_cards(manifest)
    for card in cards:
        payload = card.to_dict()
        for text in _walk_strings(payload):
            if "/" in text or text.endswith(".py") or text.endswith(".md"):
                assert not text.startswith("/"), text
                assert not Path(text).is_absolute() or text.startswith("snes/"), text


def test_manifest_from_synthetic_board_keeps_board_order() -> None:
    board = {
        "kind": "super_metroid_product_chain_hop_board",
        "task": "tasks/full_start_v1.json",
        "hops": [
            {
                "segment": "s1",
                "hop_index": 0,
                "room_id": CERES,
                "from_room_id": None,
                "to_room_id": LANDING,
                "items": 0,
                "dwell": 80,
                "tape": "tasks/full_start_v1_segments/s1/tape.json",
                "anchor_path": "tasks/full_start_v1_segments/s1/boot.state",
                "notes": ["ceres"],
            },
            {
                "segment": "s1",
                "hop_index": 1,
                "room_id": LANDING,
                "from_room_id": CERES,
                "to_room_id": None,
                "items": 0,
                "dwell": 40,
                "mode": "traversal",
                "policy_id": "landing_v1",
            },
        ],
    }
    manifest = manifest_from_board(board, hop_ids=("ceres_elev", "landing_site"))
    assert [e.task_id for e in manifest.edges] == ["ceres_elev", "landing_site"]
    assert manifest.edges[0].room_id == CERES
    assert manifest.edges[1].room_id == LANDING
    assert manifest.edges[0].hop_key == make_hop_key(
        CERES, from_room_id=None, to_room_id=LANDING, items=0
    )
    reversed_ids = manifest_from_board(board, hop_ids=("landing_site", "ceres_elev"))
    assert reversed_ids.edges[0].task_id == "landing_site"
    assert reversed_ids.edges[0].room_id == CERES
    cards = generate_cards(manifest)
    assert [c.task_id for c in cards] == ["ceres_elev", "landing_site"]
    assert "tape" in manifest.edges[0].allowed_kinds
    assert manifest.edges[0].selected_map() == {"scaffold": "tape:board"}
    assert "tape" not in manifest.edges[1].allowed_kinds
    assert manifest.edges[1].selected_map() == {"survival": "reactive_policy:landing_v1"}
    assert cards[0].adapter_kind == "tape"
    survival = generate_cards(manifest, profile="survival")
    assert survival[1].adapter_kind == "reactive_policy"


def test_empty_manifest_cards_do_not_claim_coverage() -> None:
    cards = generate_cards(RouteManifest.from_dict({"route_id": "empty", "edges": []}))
    assert cards == ()
    assert format_cards(cards) == "no hops inventoried"


def test_board_game_relative_pin_digest() -> None:
    from super_metroid.paths import GAME_DIR

    pin = GAME_DIR / "tasks" / "_splice_pr2_pin.state"
    tape = GAME_DIR / "tasks" / "_splice_pr2_tape.json"
    payload = b"pin-bytes-board"
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_bytes(payload)
    tape.write_bytes(b'{"frames":[]}\n')
    board = {
        "hops": [
            {
                "segment": "s1",
                "hop_index": 0,
                "room_id": CERES,
                "to_room_id": LANDING,
                "items": 0,
                "dwell": 80,
                "tape": "tasks/_splice_pr2_tape.json",
                "anchor_path": "tasks/_splice_pr2_pin.state",
            }
        ]
    }
    try:
        edge = manifest_from_board(board).edges[0]
        assert edge.entry.state_path == "snes/super_metroid/tasks/_splice_pr2_pin.state"
        assert edge.entry.state_digest == hashlib.sha256(payload).hexdigest()
        assert edge.tape_path == "snes/super_metroid/tasks/_splice_pr2_tape.json"
        assert edge.tape_digest == hashlib.sha256(b'{"frames":[]}\n').hexdigest()
        card = generate_cards(manifest_from_board(board))[0]
        assert card.entry_state_path == edge.entry.state_path
        assert card.entry_state_digest == edge.entry.state_digest
        assert not Path(str(card.entry_state_path)).is_absolute()
    finally:
        pin.unlink(missing_ok=True)
        tape.unlink(missing_ok=True)


def test_board_bad_frame_raises() -> None:
    board = {
        "hops": [
            {
                "room_id": CERES,
                "to_room_id": LANDING,
                "items": 0,
                "dwell": 10,
                "start_index": "nope",
            }
        ]
    }
    with pytest.raises(SchemaError, match="start_index"):
        manifest_from_board(board)


def test_candidate_rows_must_be_objects() -> None:
    base = {
        "candidate_id": "tape:ceres",
        "kind": "tape",
        "implementation_id": "s1",
        "task_id": "ceres_elev",
        "entry_fingerprint": {"room_id": CERES},
    }
    with pytest.raises(SchemaError):
        CandidateArtifact.from_dict({**base, "replay_rows": {"trial": 1, "passed": True}})
    with pytest.raises(SchemaError):
        CandidateArtifact.from_dict({**base, "join_rows": ["oops"]})
    with pytest.raises(SchemaError):
        CandidateArtifact.from_dict({**base, "memory_writes": "oops"})


def test_fingerprint_and_join_rooms_must_agree() -> None:
    raw = _tiny_manifest()
    raw["edges"][0]["entry"]["fingerprint"] = "nope"
    with pytest.raises(SchemaError, match="fingerprint"):
        RouteManifest.from_dict(raw)
    raw = _tiny_manifest()
    raw["edges"][0]["entry"]["fingerprint"]["room_id"] = LANDING
    with pytest.raises(SchemaError, match="fingerprint room"):
        RouteManifest.from_dict(raw)
    raw = _tiny_manifest()
    raw["edges"][0]["successor_leave"] = _leave("mismatch", CERES)
    with pytest.raises(SchemaError, match="successor_leave"):
        RouteManifest.from_dict(raw)
