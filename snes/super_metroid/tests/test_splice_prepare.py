"""splice.prepare fails closed before boot (no ROM)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from super_metroid.hop_id import make_hop_key
from super_metroid.leave_specs import LeaveSpec
from super_metroid.splice import PrepareError, PreparedTask, prepare, repo_relative
from super_metroid.splice.schema import LeaveSpecRef, RouteManifest

CERES = 0xDF45
LANDING = 0x91F8


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leave(hop: str, room: int) -> dict[str, Any]:
    spec = LeaveSpec(hop=hop, room=room, x=(20, 80), y=(100, 180), pose_class="stand")
    return LeaveSpecRef.from_leave_spec(spec).to_dict()


def _fp(
    room: int,
    *,
    prior: int | None = None,
    x: int = 40,
    y: int = 120,
    pose: int = 1,
) -> dict[str, Any]:
    return {
        "room_id": room,
        "x": x,
        "y": y,
        "pose": pose,
        "velocity_x": 0,
        "velocity_y": 0,
        "items": 0,
        "beams": 0,
        "boss_bits": 0,
        "event_bits": 0,
        "prior_room_id": prior,
        "enemy_phase": "none",
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
    path: str | None = None,
    digest: str | None = None,
    tape: str | None = None,
    tape_digest: str | None = None,
    order: int = 0,
    fingerprint: dict[str, Any] | None = None,
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
        "boss_bits": 0,
        "event_bits": 0,
        "entry": {
            "fingerprint": fingerprint or _fp(room, prior=pred_room),
            "state_path": path,
            "state_digest": digest,
        },
        "successor_leave": _leave(f"{task_id}_leave", leave_room),
        "allowed_kinds": ["tape", "controller"],
        "selected": {"scaffold": "tape:board"},
        "owner_package": "snes/super_metroid/routes/kpdr",
        "integration_order": order,
        "max_frames": 400,
        "max_no_progress": 200,
        "segment": "s1",
        "hop_index": order,
        "frame_start": 10 * order,
        "frame_end": 10 * order + 50,
        "tape_path": tape,
        "tape_digest": tape_digest,
        "source_notes": ["synthetic"],
    }


def _manifest(e0: dict[str, Any], e1: dict[str, Any] | None = None) -> dict[str, Any]:
    edges = [e0] if e1 is None else [e0, e1]
    return {"route_id": "tiny", "variant": "kpdr", "edges": edges}


def _kit(tmp_path: Path) -> dict[str, Any]:
    pin = tmp_path / "enter.state"
    pin.write_bytes(b"pin-bytes")
    tape = tmp_path / "tape.json"
    tape.write_bytes(b'{"frames":[]}\n')
    rom = tmp_path / "rom.sfc"
    rom.write_bytes(b"FAKE-ROM")
    core = tmp_path / "snes9x.so"
    core.write_bytes(b"FAKE-CORE")
    return {
        "pin": pin,
        "tape": tape,
        "rom": rom,
        "core": core,
        "pin_digest": _sha(b"pin-bytes"),
        "tape_digest": _sha(b'{"frames":[]}\n'),
    }


def _ready_edges(kit: dict[str, Any], *, pin: Path | None = None, digest: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (pin or kit["pin"]).name
    pin_digest = digest if digest is not None else kit["pin_digest"]
    tape = kit["tape"].name
    e0 = _edge(
        "ceres_elev",
        CERES,
        pred_room=None,
        next_room=LANDING,
        leave_room=LANDING,
        path=path,
        digest=pin_digest,
        tape=tape,
        tape_digest=kit["tape_digest"],
        order=0,
    )
    e1 = _edge(
        "landing",
        LANDING,
        pred_room=CERES,
        next_room=None,
        leave_room=LANDING,
        goal="credits",
        path=path,
        digest=pin_digest,
        tape=tape,
        tape_digest=kit["tape_digest"],
        order=1,
    )
    return e0, e1


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


def _prepare(tmp_path: Path, kit: dict[str, Any], raw: dict[str, Any], task_id: str = "ceres_elev"):
    return prepare(
        task_id,
        manifest=raw,
        rom_path=kit["rom"],
        core_path=kit["core"],
        repo_root=tmp_path,
    )


def test_success_synthetic_card_manifest(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    raw = _manifest(*_ready_edges(kit))
    prepared = _prepare(tmp_path, kit, raw)
    assert isinstance(prepared, PreparedTask)
    assert prepared.task_id == "ceres_elev"
    assert prepared.intervention_profile == "scaffold"
    assert prepared.entry_fingerprint.room_id == CERES
    assert prepared.entry_fingerprint.items == 0
    assert prepared.entry_fingerprint.pose == 1
    assert prepared.card.entry_state_digest == kit["pin_digest"]
    kinds = {a.kind: a for a in prepared.artifacts}
    assert kinds["rom"].digest == _sha(b"FAKE-ROM")
    assert kinds["core"].digest == _sha(b"FAKE-CORE")
    assert kinds["state"].digest == kit["pin_digest"]
    assert kinds["tape"].digest == kit["tape_digest"]
    assert all(a.exists and a.digest for a in prepared.artifacts)


def test_missing_pin(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    gone = tmp_path / "gone.state"
    e0, e1 = _ready_edges(kit, pin=gone, digest="ab" * 32)
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("entry_pin" in label or "pin" in label for label in missing)


def test_digest_mismatch(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit, digest="cd" * 32)
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("digest" in label for label in missing)


def test_invalid_room(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["room_id"] = 0x5555
    e0["hop_key"] = make_hop_key(0x5555, from_room_id=None, to_room_id=LANDING, items=0)
    e0["entry"]["fingerprint"]["room_id"] = 0x5555
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("invalid_room" in label or "5555" in label for label in missing)


def test_repo_relative_paths_only(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    prepared = _prepare(tmp_path, kit, _manifest(*_ready_edges(kit)))
    assert prepared.card.entry_state_path is not None
    assert not Path(prepared.card.entry_state_path).is_absolute()
    assert not prepared.card.entry_state_path.startswith("/")
    rel = repo_relative(kit["pin"])
    assert rel is not None
    assert not Path(rel).is_absolute()
    for text in _walk_strings(prepared.to_dict()):
        if "/" in text or text.endswith(".state") or text.endswith(".json"):
            assert not text.startswith("/"), text
            assert not Path(text).is_absolute() or text.startswith("snes/"), text


def test_missing_rom_fails_closed(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    raw = _manifest(*_ready_edges(kit))
    with pytest.raises(PrepareError) as exc:
        prepare(
            "ceres_elev",
            manifest=raw,
            rom_path=tmp_path / "no.rom",
            core_path=kit["core"],
            repo_root=tmp_path,
        )
    missing = exc.value.details.get("missing") or []
    assert any(label.startswith("rom:") for label in missing)


def test_required_tape_missing(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["tape_path"] = str((tmp_path / "missing_tape.json").resolve())
    e0["tape_digest"] = "ef" * 32
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any(label.startswith("tape:") for label in missing)


def test_unknown_task(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(*_ready_edges(kit)), task_id="nope")
    assert "nope" in str(exc.value)


def test_cli_report_only_json_strict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    kit = _kit(tmp_path)
    dest = tmp_path / "route.json"
    dest.write_text(json.dumps(_manifest(*_ready_edges(kit)), indent=2) + "\n", encoding="utf-8")
    code = main(["prepare", "nope", "--manifest", str(dest)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "PrepareError"
    code = main(["prepare", "nope", "--manifest", str(dest), "--strict"])
    assert code == 1


def test_named_pin_does_not_fallback_to_other_room_anchor(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    other = tmp_path / "other.state"
    other.write_bytes(b"pin-bytes")
    (tmp_path / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "kind": "room_enter",
                        "frame": 0,
                        "room": "0x91F8",
                        "room_id": LANDING,
                        "path": str(other),
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gone = tmp_path / "gone.state"
    e0, e1 = _ready_edges(kit, pin=gone, digest=kit["pin_digest"])
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("entry_pin" in label for label in missing)


def test_inventory_mismatch(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["required_items"] = 0
    e0["entry"]["fingerprint"]["items"] = 1
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("inventory" in label for label in missing)


def test_missing_edge_boss_bits(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["boss_bits"] = None
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any(label.startswith("boss:") for label in missing)


def test_predecessor_missing_and_mismatch(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e1["entry"]["fingerprint"]["prior_room_id"] = None
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1), task_id="landing")
    missing = exc.value.details.get("missing") or []
    assert any("predecessor:missing" in label for label in missing)
    e0, e1 = _ready_edges(kit)
    e1["entry"]["fingerprint"]["prior_room_id"] = LANDING
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1), task_id="landing")
    missing = exc.value.details.get("missing") or []
    assert any("predecessor:mismatch" in label for label in missing)


def test_unselected_survival_profile(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    raw = _manifest(*_ready_edges(kit))
    with pytest.raises(PrepareError) as exc:
        prepare(
            "ceres_elev",
            manifest=raw,
            profile="survival",
            rom_path=kit["rom"],
            core_path=kit["core"],
            repo_root=tmp_path,
        )
    missing = exc.value.details.get("missing") or []
    assert any("unselected" in label and "survival" in label for label in missing)


def test_missing_pose_and_velocity(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["entry"]["fingerprint"]["pose"] = None
    e0["entry"]["fingerprint"]["velocity_x"] = None
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any(label.startswith("pose:") for label in missing)
    assert any(label.startswith("velocity:") for label in missing)


def test_invalid_room_0000(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e0["room_id"] = 0x0000
    e0["hop_key"] = make_hop_key(0x0000, from_room_id=None, to_room_id=LANDING, items=0)
    e0["entry"]["fingerprint"]["room_id"] = 0x0000
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("invalid_room" in label or "0000" in label for label in missing)


def test_invalid_leave_room(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    e0, e1 = _ready_edges(kit)
    e1["successor_leave"] = _leave("landing_leave", 0x0000)
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1), task_id="landing")
    missing = exc.value.details.get("missing") or []
    assert any("exit" in label and ("0000" in label or "invalid_room" in label) for label in missing)


def test_catalog_room_mismatch(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    rel = "scratch/post_ws_entrance_to_main.state"
    pin = tmp_path / rel
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_bytes(b"pin-bytes")
    e0, e1 = _ready_edges(kit, pin=pin)
    e0["entry"]["state_path"] = rel
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any("catalog" in label or label.startswith("room:") for label in missing)


def test_corrupt_tape_with_declared_digest(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    blob = b"not-json"
    kit["tape"].write_bytes(blob)
    e0, e1 = _ready_edges(kit)
    e0["tape_digest"] = _sha(blob)
    e1["tape_digest"] = _sha(blob)
    with pytest.raises(PrepareError) as exc:
        _prepare(tmp_path, kit, _manifest(e0, e1))
    missing = exc.value.details.get("missing") or []
    assert any(label.startswith("tape:") and "corrupt" in label for label in missing)
