"""Attic→Gravity Scaffold range assemble (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from super_metroid.paths import RECORDINGS_DIR
from super_metroid.splice import (
    AssembleError,
    Assembly,
    assemble_attic_to_gravity,
    attic_to_gravity_range,
    gravity_range,
)
from super_metroid.splice.errors import PreflightError
from super_metroid.splice.ranges import (
    GRAVITY_GOAL,
    GRAVITY_TASK_ID,
    HOMING_GEEMER_ROOM,
    HOMING_GEEMER_TASK_ID,
    PANCAKES_ROOM,
    PANCAKES_TASK_ID,
    PLACEHOLDER_KIND_ID,
    PLACEHOLDER_TASKS,
    PROFILE,
    ROUTE_ID,
    TASK_ORDER,
    WEST_OCEAN_TASK_ID,
)
from super_metroid.splice.tapes import (
    ATTIC_ROOM,
    ATTIC_TASK_ID,
    BOWLING_ROOM,
    BOWLING_TASK_ID,
    GRAVITY_ROOM,
    MAIN_SHAFT_ROOM,
    WEST_OCEAN_ROOM,
)
from super_metroid.tests.test_splice_tapes import _write_s23


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


class _FakeEm:
    def __init__(self) -> None:
        self.states: list[tuple[Any, ...]] = []

    def set_state(self, *args: Any, **kwargs: Any) -> None:
        self.states.append(args)


class _FakeEnv:
    def __init__(self) -> None:
        self.loads: list[tuple[str, tuple[Any, ...]]] = []
        self.em = _FakeEm()

    def load(self, *args: Any, **kwargs: Any) -> None:
        self.loads.append(("load", args))

    def load_state(self, *args: Any, **kwargs: Any) -> None:
        self.loads.append(("load_state", args))


class _FakeSession:
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.frame = 0


def _play_recorder(calls: list[Any]) -> Any:
    def play_hops(session: Any, splits: list[Any], hops: Any, segments: Any = None) -> str:
        calls.append(
            {
                "session": session,
                "hop_ids": [hop.hop_id for hop in hops],
                "rooms": [(hop.from_room, hop.to_room) for hop in hops],
            }
        )
        if getattr(session, "env", None) is not None:
            assert session.env.loads == []
        return "ok"

    return play_hops


def test_missing_s23_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "full_start_v1_segments" / "s23"
    with pytest.raises(PreflightError) as exc:
        attic_to_gravity_range(missing)
    labels = exc.value.details.get("missing") or []
    assert any("tape" in str(label) for label in labels)
    assert exc.value.code == "preflight.missing"
    with pytest.raises(PreflightError):
        assemble_attic_to_gravity(tmp_path / "nope", assemble=lambda *a, **k: None)


def test_gravity_range_alias_and_task_order(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = gravity_range(sdir)
    assert plan is not None
    assert plan.route_id == ROUTE_ID
    assert plan.profile == PROFILE == "scaffold"
    assert plan.task_ids == TASK_ORDER
    rooms = [edge.room_id for edge in plan.manifest.edges]
    assert rooms == [
        ATTIC_ROOM,
        WEST_OCEAN_ROOM,
        PANCAKES_ROOM,
        HOMING_GEEMER_ROOM,
        BOWLING_ROOM,
        GRAVITY_ROOM,
    ]
    assert MAIN_SHAFT_ROOM not in rooms
    assert all(edge.task_id != "main_shaft" for edge in plan.manifest.edges)
    assert plan.tape_tasks == (ATTIC_TASK_ID, BOWLING_TASK_ID)
    assert plan.placeholder_tasks == PLACEHOLDER_TASKS


def test_placeholders_and_s23_tape_candidates(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = attic_to_gravity_range(sdir)
    selected = plan.selection.selected_map()
    assert selected[ATTIC_TASK_ID].startswith("tape:")
    assert selected[BOWLING_TASK_ID].startswith("tape:")
    for task in PLACEHOLDER_TASKS:
        assert selected[task] == PLACEHOLDER_KIND_ID
    by_id = {edge.task_id: edge for edge in plan.manifest.edges}
    assert by_id[ATTIC_TASK_ID].next_room_id == WEST_OCEAN_ROOM
    assert by_id[WEST_OCEAN_TASK_ID].next_room_id == PANCAKES_ROOM
    assert by_id[PANCAKES_TASK_ID].next_room_id == HOMING_GEEMER_ROOM
    assert by_id[HOMING_GEEMER_TASK_ID].next_room_id == BOWLING_ROOM
    assert by_id[BOWLING_TASK_ID].next_room_id == GRAVITY_ROOM
    assert by_id[GRAVITY_TASK_ID].next_room_id is None
    assert by_id[GRAVITY_TASK_ID].goal == GRAVITY_GOAL
    assert "bowling:entry" not in selected
    assert by_id[ATTIC_TASK_ID].predecessor_room_id == MAIN_SHAFT_ROOM


def test_hp_clamp_attic_gray_door_not_global(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = attic_to_gravity_range(sdir)
    assert plan.hp_clamp_global is False
    assert plan.hp_clamp_allowed(ATTIC_TASK_ID)
    assert not plan.hp_clamp_allowed(BOWLING_TASK_ID)
    assert not plan.hp_clamp_allowed(WEST_OCEAN_TASK_ID)
    assert not plan.hp_clamp_allowed(GRAVITY_TASK_ID)
    assert plan.hp_clamp_tasks == (ATTIC_TASK_ID,)
    assert plan.hp_clamp_allowlist
    for entry in plan.hp_clamp_allowlist:
        assert int(entry.room_id) == ATTIC_ROOM
    payload = plan.to_dict()
    assert payload["hp_clamp_global"] is False
    assert payload["hp_clamp_tasks"] == [ATTIC_TASK_ID]


def test_never_claims_survival_status_or_living_tip(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = attic_to_gravity_range(sdir)
    payload = plan.to_dict()
    assert payload["profile"] == "scaffold"
    assert payload["route_ready"] is False
    assert payload["living_tip"] is False
    assert payload["survival_claim"] is False
    joined = " ".join(plan.non_claims) + " " + " ".join(plan.source_notes)
    assert "STATUS" in joined
    assert "DEFAULT_CONTINUOUS_TIP" in joined
    assert "Survival" in joined
    assert "living Tip" in joined or "living Tip" in " ".join(plan.non_claims)
    with pytest.raises(AssembleError) as exc:
        attic_to_gravity_range(sdir, profile="survival")
    assert exc.value.code == "assemble.profile"


def test_injects_fake_assemble(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    calls: list[Any] = []

    def fake_assemble(route_id: str, selection: Any, **kwargs: Any) -> Assembly:
        calls.append({"route_id": route_id, "selection": selection, "kwargs": kwargs})
        hop_ids = tuple(task for task, _cid in selection.selected)
        return Assembly(
            route_id=route_id,
            profile=str(kwargs["profile"]),
            selected=tuple(selection.selected),
            hop_ids=hop_ids,
        )

    bank, existed, payload = _bank_snapshot()
    assembly = assemble_attic_to_gravity(sdir, assemble=fake_assemble)
    assert len(calls) == 1
    assert calls[0]["route_id"] == ROUTE_ID
    assert calls[0]["kwargs"]["profile"] == PROFILE
    assert "scaffold_allowlist" not in calls[0]["kwargs"]
    assert calls[0]["kwargs"].get("session") is None
    assert assembly.hop_ids == TASK_ORDER
    assert all("main" not in hop for hop in assembly.hop_ids)
    _assert_bank_untouched(bank, existed, payload)


def test_injects_fake_play_hops_one_session(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    session = _FakeSession()
    calls: list[Any] = []
    bank, existed, payload = _bank_snapshot()
    assembly = assemble_attic_to_gravity(
        sdir,
        play_hops=_play_recorder(calls),
        session=session,
    )
    assert isinstance(assembly, Assembly)
    assert assembly.session is session
    assert assembly.profile == PROFILE
    assert assembly.hop_ids == TASK_ORDER
    assert len(calls) == 1
    assert calls[0]["session"] is session
    assert calls[0]["hop_ids"] == list(TASK_ORDER)
    rooms = [edge.room_id for edge in attic_to_gravity_range(sdir).manifest.edges]
    assert MAIN_SHAFT_ROOM not in rooms
    assert session.env.loads == []
    _assert_bank_untouched(bank, existed, payload)


def test_mid_run_load_fails_closed(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    session = _FakeSession()

    def hop_factory(edge: Any, offer: Any) -> Any:
        def play(sess: Any) -> None:
            sess.env.em.set_state(b"room.state")

        return SimpleNamespace(hop_id=edge.task_id, play=play)

    def play_hops(sess: Any, splits: list[Any], hops: Any, segments: Any = None) -> None:
        for hop in hops:
            hop.play(sess)

    with pytest.raises(AssembleError) as exc:
        assemble_attic_to_gravity(
            sdir,
            play_hops=play_hops,
            session=session,
            hop_factory=hop_factory,
        )
    assert exc.value.code == "assemble.load"


def test_survival_profile_fails_closed(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    with pytest.raises(AssembleError) as exc:
        assemble_attic_to_gravity(
            sdir,
            profile="survival",
            play_hops=_play_recorder([]),
            session=_FakeSession(),
        )
    assert exc.value.code == "assemble.profile"


def test_cli_range_missing_s23(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    code = main(["range", "--segment", str(tmp_path / "missing"), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "PreflightError"
    assert payload["code"] == "preflight.missing"


def test_cli_range_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    sdir = _write_s23(tmp_path)
    code = main(["range", "--segment", str(sdir), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["route_id"] == ROUTE_ID
    assert payload["profile"] == PROFILE
    assert payload["living_tip"] is False
    assert payload["task_ids"] == list(TASK_ORDER)
    assert MAIN_SHAFT_ROOM not in [
        edge["room_id"] for edge in payload["manifest"]["edges"]
    ]


def test_cli_assemble_range_refuses_without_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from super_metroid.splice.__main__ import main

    sdir = _write_s23(tmp_path)
    code = main(["assemble-range", "--segment", str(sdir)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "AssembleError"
    assert payload["code"] == "assemble.session"
