"""Scaffold credits-chain assembly wiring (no ROM)."""

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
    assemble_credits,
    credits_chain,
)
from super_metroid.splice.credits import (
    CREDITS_GOAL,
    CREDITS_TASK_ID,
    PROFILE,
    ROUTE_ID,
    CreditsReport,
)
from super_metroid.splice.errors import PreflightError
from super_metroid.splice.lanes import ITEM_SEAM_LANES
from super_metroid.splice.ranges import TASK_ORDER
from super_metroid.splice.tapes import MAIN_SHAFT_ROOM
from super_metroid.tests.test_splice_tapes import _write_s23

LANE_IDS = tuple(spec.lane_id for spec in ITEM_SEAM_LANES)


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
        for hop in hops:
            splits.append(SimpleNamespace(split_id=hop.hop_id, to_dict=lambda h=hop: {"split_id": h.hop_id}))
        return "ok"

    return play_hops


def test_missing_s23_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "full_start_v1_segments" / "s23"
    with pytest.raises(PreflightError) as exc:
        credits_chain(missing)
    labels = exc.value.details.get("missing") or []
    assert any("tape" in str(label) for label in labels)
    assert exc.value.code == "preflight.missing"
    with pytest.raises(PreflightError):
        assemble_credits(tmp_path / "nope", assemble=lambda *a, **k: None)


def test_ten_lanes_and_gravity_range(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = credits_chain(sdir)
    assert plan.route_id == ROUTE_ID
    assert plan.profile == PROFILE == "scaffold"
    assert len(plan.lanes) == 10
    assert tuple(lane.lane_id for lane in plan.lanes) == LANE_IDS
    assert all(lane.task_ids for lane in plan.lanes)
    assert list(plan.task_ids[: len(TASK_ORDER)]) == list(TASK_ORDER)
    assert plan.task_ids[-1] == CREDITS_TASK_ID
    assert plan.manifest.edges[-1].goal == CREDITS_GOAL
    rooms = [edge.room_id for edge in plan.manifest.edges]
    assert MAIN_SHAFT_ROOM not in rooms
    assert all(edge.task_id != "main_shaft" for edge in plan.manifest.edges)


def test_never_claims_survival_status_or_finish(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    plan = credits_chain(sdir)
    payload = plan.to_dict()
    assert payload["profile"] == "scaffold"
    assert payload["development_only"] is True
    assert payload["route_ready"] is False
    assert payload["living_tip"] is False
    assert payload["survival_claim"] is False
    assert payload["finish_claim"] is False
    assert payload["status_claim"] is False
    assert payload["zero_state_load"] is True
    joined = " ".join(plan.non_claims) + " " + " ".join(plan.source_notes)
    assert "STATUS" in joined
    assert "DEFAULT_CONTINUOUS_TIP" in joined
    assert "Survival" in joined
    with pytest.raises(AssembleError) as exc:
        credits_chain(sdir, profile="survival")
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
    report = assemble_credits(sdir, assemble=fake_assemble)
    assert isinstance(report, CreditsReport)
    assert len(calls) == 1
    assert calls[0]["route_id"] == ROUTE_ID
    assert calls[0]["kwargs"]["profile"] == PROFILE
    assert calls[0]["kwargs"].get("session") is None
    assert calls[0]["kwargs"].get("session_factory") is None
    assert report.profile == PROFILE
    assert report.hop_ids[: len(TASK_ORDER)] == TASK_ORDER
    assert report.hop_ids[-1] == CREDITS_TASK_ID
    assert len(report.lanes) == 10
    assert tuple(lane.lane_id for lane in report.lanes) == LANE_IDS
    assert len(report.intervention_ledger) == len(report.hop_ids)
    assert len(report.room_splits) == len(report.hop_ids)
    assert report.to_dict()["development_only"] is True
    assert report.to_dict()["survival_claim"] is False
    assert all(hop != "main_shaft" and "ws_main" not in hop for hop in report.hop_ids)
    _assert_bank_untouched(bank, existed, payload)
    local_bank = Path("bank.json")
    if not local_bank.is_file():
        assert not local_bank.exists()


def test_injects_fake_play_hops_one_session(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    session = _FakeSession()
    calls: list[Any] = []
    bank, existed, payload = _bank_snapshot()
    report = assemble_credits(
        sdir,
        play_hops=_play_recorder(calls),
        session=session,
    )
    assert isinstance(report, CreditsReport)
    assert report.session is session
    assert report.profile == PROFILE
    assert report.hop_ids[-1] == CREDITS_TASK_ID
    assert len(calls) == 1
    assert calls[0]["session"] is session
    assert calls[0]["hop_ids"][-1] == CREDITS_TASK_ID
    assert session.env.loads == []
    assert len(report.intervention_ledger) == len(report.hop_ids)
    assert all("split" in row for row in report.room_splits)
    _assert_bank_untouched(bank, existed, payload)


def test_never_calls_load(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    session = _FakeSession()

    def play_hops(sess: Any, splits: list[Any], hops: Any, segments: Any = None) -> None:
        for hop in hops:
            hop.play(sess)

    assemble_credits(sdir, play_hops=play_hops, session=session)
    assert session.env.loads == []


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
        assemble_credits(
            sdir,
            play_hops=play_hops,
            session=session,
            hop_factory=hop_factory,
        )
    assert exc.value.code == "assemble.load"


def test_refuses_without_session(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    with pytest.raises(AssembleError) as exc:
        assemble_credits(sdir, play_hops=_play_recorder([]))
    assert exc.value.code == "assemble.session"


def test_survival_profile_fails_closed(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    with pytest.raises(AssembleError) as exc:
        assemble_credits(
            sdir,
            profile="survival",
            play_hops=_play_recorder([]),
            session=_FakeSession(),
        )
    assert exc.value.code == "assemble.profile"


def test_cli_credits_dry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    sdir = _write_s23(tmp_path)
    code = main(["credits", "--dry", "--segment", str(sdir), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["route_id"] == ROUTE_ID
    assert payload["profile"] == PROFILE
    assert payload["development_only"] is True
    assert payload["survival_claim"] is False
    assert payload["living_tip"] is False
    assert payload["status_claim"] is False
    assert payload["lane_ids"] == list(LANE_IDS)
    assert payload["task_ids"][-1] == CREDITS_TASK_ID
    assert MAIN_SHAFT_ROOM not in [edge["room_id"] for edge in payload["manifest"]["edges"]]


def test_cli_credits_default_is_dry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    sdir = _write_s23(tmp_path)
    code = main(["credits", "--segment", str(sdir), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["development_only"] is True
    assert payload["profile"] == "scaffold"


def test_cli_credits_missing_s23(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    code = main(["credits", "--dry", "--segment", str(tmp_path / "missing"), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "PreflightError"
    assert payload["code"] == "preflight.missing"
