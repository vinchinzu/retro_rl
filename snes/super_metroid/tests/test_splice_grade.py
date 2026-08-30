"""splice.grade replays from the immutable start (no ROM)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from super_metroid.hop_id import make_hop_key
from super_metroid.hop_glance import LeaveMiss
from super_metroid.leave_specs import LeaveSpec
from super_metroid.paths import RECORDINGS_DIR
from super_metroid.splice import (
    GradeError,
    GradeReport,
    PreparedTask,
    grade,
    prepare,
)
from super_metroid.splice.schema import LeaveSpecRef

CERES = 0xDF45
LANDING = 0x91F8

_GREEN_STILL = {
    "room": LANDING,
    "xy": [40, 120],
    "pose": 1,
    "gs": 8,
    "dt": 0,
    "health": 99,
}
_RED_STILL = {
    "room": CERES,
    "xy": [1, 1],
    "pose": 29,
    "gs": 8,
    "dt": 0,
    "health": 99,
}


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
            "fingerprint": _fp(room, prior=pred_room),
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


def _manifest(kit: dict[str, Any]) -> dict[str, Any]:
    path = kit["pin"].name
    tape = kit["tape"].name
    e0 = _edge(
        "ceres_elev",
        CERES,
        pred_room=None,
        next_room=LANDING,
        leave_room=LANDING,
        path=path,
        digest=kit["pin_digest"],
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
        digest=kit["pin_digest"],
        tape=tape,
        tape_digest=kit["tape_digest"],
        order=1,
    )
    return {"route_id": "tiny", "variant": "kpdr", "edges": [e0, e1]}


def _prepared(tmp_path: Path, kit: dict[str, Any] | None = None) -> PreparedTask:
    kit = kit or _kit(tmp_path)
    return prepare(
        "ceres_elev",
        manifest=_manifest(kit),
        rom_path=kit["rom"],
        core_path=kit["core"],
        repo_root=tmp_path,
    )


def _bank_snapshot() -> tuple[Path, bool, bytes | None]:
    bank = RECORDINGS_DIR / "skill_bank" / "bank.json"
    existed = bank.is_file()
    payload = bank.read_bytes() if existed else None
    return bank, existed, payload


def _assert_bank_untouched(bank: Path, existed: bool, payload: bytes | None) -> None:
    if existed:
        assert bank.is_file()
        assert bank.read_bytes() == payload
    else:
        assert not bank.exists()


def test_fake_runner_green_includes_join(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    calls: list[str] = []

    def runner(prep: PreparedTask, candidate: Any) -> dict[str, Any]:
        calls.append(prep.card.entry_state_digest or "")
        assert candidate.candidate_id == "tape:board"
        return _GREEN_STILL

    before = prepared.to_dict()
    bank, existed, payload = _bank_snapshot()
    report = grade(prepared, "tape:board", runner=runner, artifact_dir=tmp_path)
    assert isinstance(report, GradeReport)
    assert report.verdict == "GREEN"
    assert report.join is not None
    assert report.join.passed
    assert report.join.join.leave.room == LANDING
    assert report.join.join.next_entry is not None
    assert report.join.misses == ()
    assert report.join.to_dict()["leave"]["room"] == LANDING
    assert report.replay_green is True
    assert report.sync_green is False
    assert report.start_digest == prepared.card.entry_state_digest
    assert calls == [prepared.card.entry_state_digest, prepared.card.entry_state_digest]
    assert not (tmp_path / "leftover.json").exists()
    assert prepared.to_dict() == before
    _assert_bank_untouched(bank, existed, payload)


def test_red_writes_leftover_package(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    board = tmp_path / "board.json"
    board.write_text('{"kind":"product"}\n', encoding="utf-8")
    before_board = board.read_bytes()
    bank, existed, payload = _bank_snapshot()

    report = grade(
        prepared,
        "tape:board",
        runner=lambda *_: _RED_STILL,
        artifact_dir=tmp_path,
    )
    assert report.verdict == "RED"
    assert report.replay_green is False
    assert report.sync_green is False
    assert report.leftover_package is not None
    assert report.leftover_package.misses
    assert any("room" in m for m in report.leftover_package.misses)
    pkg = tmp_path / "leftover.json"
    still = tmp_path / "leftover.state"
    assert pkg.is_file()
    assert still.is_file()
    data = json.loads(pkg.read_text(encoding="utf-8"))
    assert data["path"]
    assert data["misses"]
    assert data["leftover"]["xy"] == [1, 1]
    assert not Path(str(report.leftover_package.path)).is_absolute()
    assert board.read_bytes() == before_board
    assert not (tmp_path / "bank.json").exists()
    _assert_bank_untouched(bank, existed, payload)


def test_two_greens_are_replay_green(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    class Fake:
        def __init__(self) -> None:
            self.calls = 0

        def replay(self, prep: PreparedTask, candidate: Any) -> dict[str, Any]:
            self.calls += 1
            assert prep.card.entry_state_digest == prepared.card.entry_state_digest
            return {
                "leftover": _GREEN_STILL,
                "frames": 40,
                "start_digest": prep.card.entry_state_digest,
            }

    fake = Fake()
    report = grade(prepared, "tape:board", runner=fake, artifact_dir=tmp_path)
    assert fake.calls == 2
    assert report.verdict == "GREEN"
    assert report.replay_green is True
    assert report.sync_green is False
    assert [row.passed for row in report.replay_rows] == [True, True]


def test_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    cand = {
        "candidate_id": "tape:board",
        "kind": "tape",
        "implementation_id": "board",
        "task_id": "ceres_elev",
        "entry_fingerprint": prepared.entry_fingerprint.to_dict(),
        "start_state_digest": "ab" * 32,
    }
    with pytest.raises(GradeError) as exc:
        grade(
            prepared,
            cand,
            runner=lambda *_: _GREEN_STILL,
            artifact_dir=tmp_path,
        )
    assert exc.value.code == "grade.digest"
    assert not (tmp_path / "leftover.json").exists()
    assert not (tmp_path / "bank.json").exists()


def test_runner_start_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    def runner(prep: PreparedTask, candidate: Any) -> dict[str, Any]:
        return {
            "leftover": _GREEN_STILL,
            "start_digest": "cd" * 32,
        }

    with pytest.raises(GradeError) as exc:
        grade(prepared, "tape:board", runner=runner, artifact_dir=tmp_path)
    assert exc.value.code == "grade.digest"


def test_records_every_intervention(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    write = {
        "frame": 3,
        "address": 0x0F8C,
        "entity": "enemy0",
        "old": 20,
        "new": 1,
        "reason": "scaffold_hp_clamp",
    }

    def runner(prep: PreparedTask, candidate: Any) -> dict[str, Any]:
        return {
            "leftover": _GREEN_STILL,
            "interventions": [write],
            "start_digest": prep.card.entry_state_digest,
        }

    report = grade(prepared, "tape:board", runner=runner, artifact_dir=tmp_path)
    assert len(report.interventions) == 2
    first = report.interventions[0]
    assert (first.frame, first.address, first.entity, first.old, first.new, first.reason) == (
        3,
        0x0F8C,
        "enemy0",
        20,
        1,
        "scaffold_hp_clamp",
    )


def test_session_uses_hop_glance(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    class _Duck:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Session:
        def __init__(self, state: object) -> None:
            self.state = state

    state = _Duck(
        room_id=LANDING,
        samus_x=40,
        samus_y=120,
        pose=1,
        game_state=8,
        door_transition=0,
        health=99,
    )
    report = grade(
        prepared,
        "tape:board",
        session=_Session(state),
        artifact_dir=tmp_path,
    )
    assert report.verdict == "GREEN"
    assert report.join is not None and report.join.passed
    assert report.replay_green is False
    assert report.sync_green is False


def test_leave_miss_from_runner_is_red(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    def runner(prep: PreparedTask, candidate: Any) -> None:
        raise LeaveMiss("ceres_elev_leave", _RED_STILL, ["room 0xDF45 != 0x91F8"])

    report = grade(prepared, "tape:board", runner=runner, artifact_dir=tmp_path)
    assert report.verdict == "RED"
    assert report.leftover_package is not None
    assert (tmp_path / "leftover.json").is_file()


def test_refuses_without_runner(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    with pytest.raises(GradeError) as exc:
        grade(prepared, "tape:board", artifact_dir=tmp_path)
    assert exc.value.code == "grade.runner"


def test_cli_refuses_without_runner(capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    code = main(["grade", "ceres_elev", "tape:board"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "GradeError"
    assert payload["code"] == "grade.runner"


def test_mixed_trials_are_not_replay_green(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    leftover = [_GREEN_STILL, _RED_STILL]

    def runner(prep: PreparedTask, candidate: Any) -> dict[str, Any]:
        return leftover.pop(0)

    report = grade(prepared, "tape:board", runner=runner, artifact_dir=tmp_path)
    assert report.verdict == "RED"
    assert report.replay_green is False
    assert report.join is not None
    assert report.join.passed is False
    assert (tmp_path / "leftover.json").is_file()
