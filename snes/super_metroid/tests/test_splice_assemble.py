"""splice.select / splice.assemble through play_hops (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from super_metroid.hop_id import make_hop_key
from super_metroid.leave_specs import LeaveSpec
from super_metroid.paths import RECORDINGS_DIR
from super_metroid.splice import (
    AssembleError,
    Assembly,
    CandidateOffer,
    Selection,
    assemble,
    rollback,
    select,
)
from super_metroid.splice.schema import LeaveSpecRef, RouteManifest
from super_metroid.splice.select import as_offer

CERES = 0xDF45
LANDING = 0x91F8


def _leave(hop: str, room: int) -> dict[str, Any]:
    spec = LeaveSpec(hop=hop, room=room, x=(20, 80), y=(100, 180), pose_class="stand")
    return LeaveSpecRef.from_leave_spec(spec).to_dict()


def _fp(room: int, *, prior: int | None = None) -> dict[str, Any]:
    return {
        "room_id": room,
        "x": 40,
        "y": 120,
        "pose": 1,
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
    selected: dict[str, str] | None = None,
    allowed: tuple[str, ...] = ("tape", "controller"),
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
        "entry": {"fingerprint": _fp(room, prior=pred_room)},
        "successor_leave": _leave(f"{task_id}_leave", leave_room),
        "allowed_kinds": list(allowed),
        "selected": selected or {"scaffold": "tape:a0", "survival": "controller:play"},
        "owner_package": "snes/super_metroid/routes/kpdr",
        "integration_order": order,
        "max_frames": 400,
        "max_no_progress": 200,
        "segment": "s1",
        "hop_index": order,
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


def _join(
    trial: int,
    candidate_id: str,
    *,
    pred: str,
    succ: str | None,
    passed: bool = True,
) -> dict[str, Any]:
    return {
        "trial": trial,
        "predecessor_task_id": pred,
        "candidate_id": candidate_id,
        "successor_task_id": succ,
        "passed": passed,
    }


def _cand(
    task_id: str,
    candidate_id: str,
    *,
    profile: str = "scaffold",
    frames: int | None = 40,
    replay_green: bool = True,
    join_succ: str | None = None,
    join_passed: bool = True,
    writes: list[dict[str, Any]] | None = None,
    parent: str | None = None,
    room: int = CERES,
    prior: int | None = None,
) -> dict[str, Any]:
    kind = candidate_id.split(":", 1)[0]
    impl = candidate_id.split(":", 1)[1] if ":" in candidate_id else candidate_id
    replay = (
        (
            {"trial": 1, "passed": True, "frames": frames},
            {"trial": 2, "passed": True, "frames": frames},
        )
        if replay_green
        else ({"trial": 1, "passed": False, "frames": frames, "miss": "leave"},)
    )
    joins: list[dict[str, Any]] = []
    if join_succ is not None or join_passed is False:
        succ = join_succ
        joins = [
            _join(1, candidate_id, pred="start", succ=succ, passed=join_passed),
            _join(2, candidate_id, pred="start", succ=succ, passed=join_passed),
        ]
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "kind": kind,
        "implementation_id": impl,
        "task_id": task_id,
        "entry_fingerprint": _fp(room, prior=prior),
        "intervention_profile": profile,
        "replay_rows": list(replay),
        "join_rows": joins,
        "frame_count": frames,
        "parent_candidate_id": parent,
    }
    if writes:
        payload["memory_writes"] = writes
    return payload


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
        calls.append({"session": session, "hop_ids": [hop.hop_id for hop in hops]})
        if getattr(session, "env", None) is not None:
            assert session.env.loads == []
        return "ok"

    return play_hops


def test_select_does_not_write_bank_json() -> None:
    raw = _tiny_manifest()
    snapshot = json.loads(json.dumps(raw))
    manifest = RouteManifest.from_dict(raw)
    before = manifest.to_dict()
    bank, existed, payload = _bank_snapshot()
    local_bank = Path("bank.json")
    existed_local = local_bank.is_file()
    local_payload = local_bank.read_bytes() if existed_local else None
    offers = [
        _cand("ceres_elev", "tape:a0", join_succ="landing"),
        _cand("landing", "controller:play", room=LANDING, prior=CERES),
    ]
    sel = select(manifest, offers, profile="scaffold")
    assert sel.selected_map()["ceres_elev"] == "tape:a0"
    assert sel.selected_map()["landing"] == "controller:play"
    assert raw == snapshot
    assert manifest.to_dict() == before
    _assert_bank_untouched(bank, existed, payload)
    if existed_local:
        assert local_bank.read_bytes() == local_payload
    else:
        assert not local_bank.exists()


def test_select_tape_and_controller_variants() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    offers = [
        _cand("ceres_elev", "tape:a0", join_succ="landing"),
        _cand("landing", "controller:play", room=LANDING, prior=CERES),
    ]
    sel = select(manifest, offers, profile="scaffold")
    assert sel.profile == "scaffold"
    assert sel.selected_map() == {"ceres_elev": "tape:a0", "landing": "controller:play"}


def test_new_leave_cannot_start_successor_keeps_old_and_couples() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    a1 = _cand(
        "ceres_elev",
        "tape:a1",
        frames=20,
        replay_green=True,
        join_succ="landing",
        join_passed=False,
        parent="tape:a0",
    )
    sel = select(manifest, [a1], profile="scaffold")
    assert sel.selected_map()["ceres_elev"] == "tape:a0"
    assert ("ceres_elev", "landing") in sel.coupled
    assert sel.previous_map()["ceres_elev"] == "tape:a0"


def test_sync_green_replaces_incumbent() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    a1 = _cand(
        "ceres_elev",
        "tape:a1",
        frames=20,
        replay_green=True,
        join_succ="landing",
        join_passed=True,
        parent="tape:a0",
    )
    sel = select(manifest, [a1], profile="scaffold")
    assert sel.selected_map()["ceres_elev"] == "tape:a1"
    assert sel.previous_map()["ceres_elev"] == "tape:a0"
    assert sel.coupled == ()


def test_rollback_keeps_previous_id() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    a1 = _cand(
        "ceres_elev",
        "tape:a1",
        frames=20,
        join_succ="landing",
        parent="tape:a0",
    )
    sel = select(manifest, [a1], profile="scaffold")
    assert sel.selected_map()["ceres_elev"] == "tape:a1"
    bank, existed, payload = _bank_snapshot()
    restored = rollback(sel, "ceres_elev")
    assert restored.selected_map()["ceres_elev"] == "tape:a0"
    assert restored.previous_map()["ceres_elev"] == "tape:a1"
    assert restored.selected_map()["landing"] == sel.selected_map()["landing"]
    _assert_bank_untouched(bank, existed, payload)


def test_profile_mismatch_fails_closed() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    clean = _cand("ceres_elev", "tape:clean", profile="clean", join_succ="landing")
    sel = Selection(
        route_id="tiny",
        profile="scaffold",
        selected=(("ceres_elev", "tape:clean"), ("landing", "tape:a0")),
        offers=(as_offer(clean, default_profile="clean"),),
    )
    bank, existed, payload = _bank_snapshot()
    with pytest.raises(AssembleError) as exc:
        assemble("tiny", sel, manifest=manifest, session=_FakeSession(), play_hops=_play_recorder([]))
    assert exc.value.code == "assemble.profile"
    _assert_bank_untouched(bank, existed, payload)


def test_same_id_other_profile_does_not_block_assemble() -> None:
    """Scaffold incumbent tape:a0 still assembles when a clean tape:a0 offer exists."""
    manifest = RouteManifest.from_dict(_tiny_manifest())
    clean = _cand("ceres_elev", "tape:a0", profile="clean", join_succ="landing")
    sel = select(manifest, [clean], profile="scaffold")
    assert sel.selected_map()["ceres_elev"] == "tape:a0"
    calls: list[Any] = []
    assembly = assemble(
        "tiny",
        sel,
        manifest=manifest,
        play_hops=_play_recorder(calls),
        session=_FakeSession(),
    )
    assert assembly.selected_map()["ceres_elev"] == "tape:a0"
    assert assembly.hop_ids == ("ceres_elev", "landing")
    assert len(calls) == 1


def test_assemble_rejects_mismatched_candidate_mapping() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    clean = _cand("ceres_elev", "tape:clean", profile="clean", join_succ="landing")
    landing = _cand("landing", "tape:a0", room=LANDING, prior=CERES)
    with pytest.raises(AssembleError) as exc:
        assemble(
            "tiny",
            {"ceres_elev": "tape:clean", "landing": "tape:a0"},
            manifest=manifest,
            profile="scaffold",
            candidates=[clean, landing],
            session=_FakeSession(),
            play_hops=_play_recorder([]),
        )
    assert exc.value.code == "assemble.profile"


def test_survival_rejects_scaffold_writes() -> None:
    manifest = RouteManifest.from_dict(
        _tiny_manifest(
            e0={"selected": {"survival": "tape:a0", "scaffold": "tape:a0"}},
            e1={"selected": {"survival": "tape:a0", "scaffold": "tape:a0"}},
        )
    )
    write = {
        "frame": 3,
        "address": 0x0F8C,
        "entity": "enemy0",
        "old": 20,
        "new": 1,
        "reason": "scaffold_hp_clamp",
    }
    a0 = _cand(
        "ceres_elev",
        "tape:a0",
        profile="survival",
        join_succ="landing",
        writes=[write],
    )
    landing = _cand("landing", "tape:a0", profile="survival", room=LANDING, prior=CERES)
    with pytest.raises(AssembleError) as exc:
        assemble(
            "tiny",
            {"ceres_elev": "tape:a0", "landing": "tape:a0"},
            manifest=manifest,
            profile="survival",
            candidates=[a0, landing],
            session=_FakeSession(),
            play_hops=_play_recorder([]),
        )
    assert exc.value.code == "assemble.profile"


def test_play_hops_invoked_in_order_one_session() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    offers = [
        _cand("ceres_elev", "tape:a0", join_succ="landing"),
        _cand("landing", "controller:play", room=LANDING, prior=CERES),
    ]
    sel = select(manifest, offers, profile="scaffold")
    session = _FakeSession()
    calls: list[Any] = []
    bank, existed, payload = _bank_snapshot()
    assembly = assemble(
        "tiny",
        sel,
        manifest=manifest,
        play_hops=_play_recorder(calls),
        session=session,
    )
    assert isinstance(assembly, Assembly)
    assert assembly.session is session
    assert assembly.hop_ids == ("ceres_elev", "landing")
    assert len(calls) == 1
    assert calls[0]["session"] is session
    assert calls[0]["hop_ids"] == ["ceres_elev", "landing"]
    assert session.env.loads == []
    _assert_bank_untouched(bank, existed, payload)


def test_assemble_never_calls_load() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    sel = select(manifest, profile="scaffold")
    session = _FakeSession()

    def play_hops(sess: Any, splits: list[Any], hops: Any, segments: Any = None) -> None:
        for hop in hops:
            hop.play(sess)

    assemble("tiny", sel, manifest=manifest, play_hops=play_hops, session=session)
    assert session.env.loads == []


def test_mid_run_load_fails_closed() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    sel = select(manifest, profile="scaffold")
    session = _FakeSession()

    def hop_factory(edge: Any, offer: CandidateOffer) -> Any:
        def play(sess: Any) -> None:
            sess.env.em.set_state(b"room.state")

        return SimpleNamespace(hop_id=edge.task_id, play=play)

    def play_hops(sess: Any, splits: list[Any], hops: Any, segments: Any = None) -> None:
        for hop in hops:
            hop.play(sess)

    with pytest.raises(AssembleError) as exc:
        assemble(
            "tiny",
            sel,
            manifest=manifest,
            play_hops=play_hops,
            session=session,
            hop_factory=hop_factory,
        )
    assert exc.value.code == "assemble.load"


def test_refuses_without_session() -> None:
    manifest = RouteManifest.from_dict(_tiny_manifest())
    sel = select(manifest, profile="scaffold")
    with pytest.raises(AssembleError) as exc:
        assemble("tiny", sel, manifest=manifest, play_hops=_play_recorder([]))
    assert exc.value.code == "assemble.session"


def test_cli_refuses_without_session(capsys: pytest.CaptureFixture[str]) -> None:
    from super_metroid.splice.__main__ import main

    code = main(["assemble", "tiny", "--profile", "scaffold"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "AssembleError"
    assert payload["code"] == "assemble.session"


def test_assemble_requires_manifest() -> None:
    with pytest.raises(AssembleError) as exc:
        assemble("tiny", {"ceres_elev": "tape:a0"}, play_hops=_play_recorder([]), session=_FakeSession())
    assert exc.value.code == "assemble.manifest"
