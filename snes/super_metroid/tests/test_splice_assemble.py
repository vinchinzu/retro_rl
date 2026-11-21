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


def test_clean_guards_count_attempted_loads_and_writes() -> None:
    from super_metroid.splice.assemble import _guard_loads, _guard_writes

    sess = _FakeSession()
    loads = [0]
    restore_loads = _guard_loads(sess, loads)
    try:
        with pytest.raises(AssembleError) as exc:
            sess.env.em.set_state(b"x")
        assert exc.value.code == "assemble.load"
        assert loads == [1]
    finally:
        restore_loads()

    class _Mem:
        blocks = {0: bytearray(8)}

        def assign(self, *args: Any) -> None:
            pass

    class _Data:
        memory = _Mem()

        def set_value(self, *args: Any) -> None:
            pass

    env = SimpleNamespace(data=_Data(), em=SimpleNamespace(set_ram=lambda *a: None), set_ram=lambda *a: None, set_value=lambda *a: None)
    writes = [0]
    restore_writes = _guard_writes(env, writes)
    try:
        for fn in (lambda: env.data.set_value("health", 99), lambda: env.data.memory.assign(0, "u8", 1), lambda: env.set_value("health", 99)):
            with pytest.raises(RuntimeError, match="forbids RAM writes"):
                fn()
        assert writes == [3]
    finally:
        restore_writes()


def test_clean_chain_rejects_non_vanilla_rom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid import paths as sm_paths
    from super_metroid.splice.assemble import replay_tape_chain, solve_tape_chain

    fake_rom = tmp_path / "rom.sfc"
    fake_rom.write_bytes(b"not-vanilla-rom")
    monkeypatch.setattr(sm_paths, "SHARED_ROM", fake_rom)
    stem = tmp_path / "bad_clean"
    stem.with_suffix(".json").write_text("{}\n")
    with pytest.raises(ValueError, match="vanilla ROM"):
        replay_tape_chain(stem=stem)
    with pytest.raises(ValueError, match="vanilla ROM"):
        solve_tape_chain(board=tmp_path / "board.json", stem=stem, resume=False)


@pytest.mark.rom
def test_clean_write_guard_blocks_actual_emulator_writers() -> None:
    from retro_harness.actions import idle_action
    from super_metroid.dev.common import make_dev_env
    from super_metroid.ram import write_wram_u16
    from super_metroid.splice.assemble import _guard_writes

    env = make_dev_env()
    env.reset()
    restore = _guard_writes(env)
    try:
        with pytest.raises(RuntimeError, match="forbids RAM writes"):
            write_wram_u16(env, 0x09C2, 77)
        with pytest.raises(RuntimeError, match="forbids RAM writes"):
            env.set_value("health", 77)
        block = next(iter(env.data.memory.blocks.values()))
        assert memoryview(block).readonly
        env.step(idle_action())  # The emulator itself must still update RAM.
    finally:
        restore()
        env.close()


@pytest.mark.rom
def test_clean_chain_keeps_pickup_tail_and_replays_from_power_on(tmp_path: Path) -> None:
    from super_metroid.human_tape.product_chain import write_clean_chain_board
    from super_metroid.splice.assemble import solve_tape_chain

    board = tmp_path / "board.json"
    write_clean_chain_board(out=board)
    report = solve_tape_chain(
        board=board, stem=tmp_path / "acceptance_clean",
        through_row=19, max_delay=0, resume=False,
    )
    assert report["stopped"] is None
    assert report["total_frames"] == 26_734
    assert report["final_state"]["room_id"] == 0x9F11
    assert report["final_state"]["collected_items"] & 4
    assert report["proof"]["success"]
    assert report["proof"]["all_leaves_match"]
    assert report["proof"]["all_planned_crossings_match"]
    assert report["proof"]["observed_transition_count"] == report["rooms_cleared"]
    assert report["proof"]["transitions"][0]["source_room_id"] == CERES
    assert report["proof"]["detour_rows"] == []
    assert report["proof"]["state_loads"] == report["proof"]["ram_writes"] == 0


@pytest.mark.rom
def test_failed_chain_commit_keeps_last_replayable_prefix(tmp_path: Path, monkeypatch) -> None:
    from super_metroid.human_tape.product_chain import write_clean_chain_board
    from super_metroid.routes import tips
    from super_metroid.splice.assemble import solve_tape_chain

    original = tips.play_hops

    def fail_after_play(session, splits, hops, segments=None):
        result = original(session, splits, hops, segments)
        if hops[0].hop_id == "chain:3":
            raise RuntimeError("injected commit failure")
        return result

    monkeypatch.setattr(tips, "play_hops", fail_after_play)
    board = tmp_path / "board.json"
    write_clean_chain_board(out=board)
    report = solve_tape_chain(
        board=board, stem=tmp_path / "rollback_clean",
        through_row=3, max_delay=0, resume=False,
    )
    assert report["stopped"]["row"] == 3
    assert [row["row"] for row in report["rows"]] == [1, 2]
    assert report["total_frames"] == 12_096
    assert report["final_state"]["room_id"] == 0xDF8D
    assert report["proof"]["success"]


def test_policy_coverage_separates_stored_buttons_from_controllers() -> None:
    from super_metroid.splice.__main__ import policy_coverage

    rows = [
        {"method": "tape+idle:0", "frames": 600},
        {"method": "tape+idle:3", "frames": 100},
        {"method": "tape-trim:2", "frames": 50},
        {"method": "entry-pulse:32:(0,)", "frames": 50},
        {"method": "searched-inputs", "frames": 50},
        {"method": "room-controller", "frames": 100},
        {"method": "runway-detour", "frames": 50, "controller_id": "moat_cross"},
    ]
    got = policy_coverage(rows)
    assert {k: v["frames"] for k, v in got.items() if k != "basis"} == {
        "human_verbatim": 600, "human_retimed": 150, "searched_buttons": 100, "controller": 150,
    }
    assert got["human_verbatim"]["share"] == 0.6 and got["basis"] == "ledger_labels"
    assert policy_coverage([])["controller"]["frames"] == 0


def test_policy_coverage_does_not_trust_a_tape_label_over_the_buttons() -> None:
    import numpy as np
    from super_metroid.splice.__main__ import policy_coverage

    def buttons(n: int, bit: int) -> "np.ndarray":
        out = np.zeros((n, 12), dtype=np.int8)
        out[:, bit] = 1
        return out

    human, other = buttons(10, 0), buttons(10, 1)
    rows = [
        {"row": 1, "method": "tape+idle:0", "frames": 10},   # really the human slice
        {"row": 2, "method": "tape+idle:0", "frames": 10},   # label lies: an override replaced it
        {"row": 3, "method": "tape+idle:2", "frames": 12},   # delayed human slice
        {"row": 4, "method": "override:tape+idle:0", "frames": 10},
    ]
    tape = np.concatenate((human, other, np.zeros((2, 12), dtype=np.int8), human, other))
    got = policy_coverage(rows, tape, {1: human, 2: human, 3: human, 4: human})
    assert got["basis"] == "button_diff"
    assert got["human_verbatim"]["frames"] == 10
    assert got["human_retimed"]["frames"] == 12
    assert got["searched_buttons"]["frames"] == 20


@pytest.mark.rom
def test_from_row_cuts_a_copy_and_leaves_the_source_master_alone(tmp_path: Path) -> None:
    import numpy as np
    from super_metroid.human_tape.product_chain import write_clean_chain_board
    from super_metroid.splice.assemble import solve_tape_chain

    board = tmp_path / "board.json"
    write_clean_chain_board(out=board)
    source = tmp_path / "source_clean"
    full = solve_tape_chain(board=board, stem=source, through_row=5, max_delay=0, resume=False)
    assert full["proof"]["success"] and [r["row"] for r in full["rows"]] == [1, 2, 3, 4, 5]
    before = source.with_suffix(".npz").read_bytes()

    cut = tmp_path / "cut_clean"
    again = solve_tape_chain(
        board=board, stem=cut, through_row=5, max_delay=0, from_row=4, source=source,
    )
    assert again["stopped"] is None and again["proof"]["success"]
    assert [r["row"] for r in again["rows"]] == [1, 2, 3, 4, 5]
    assert again["proof"]["state_loads"] == again["proof"]["ram_writes"] == 0
    # Search is deterministic, so re-solving rows 4-5 reproduces the buttons.
    assert np.array_equal(np.load(cut.with_suffix(".npz"))["actions"], np.load(source.with_suffix(".npz"))["actions"])
    assert source.with_suffix(".npz").read_bytes() == before  # the verified master is never rewritten

    with pytest.raises(ValueError, match="already exists"):
        solve_tape_chain(board=board, stem=cut, through_row=5, max_delay=0, from_row=4, source=source)
    with pytest.raises(ValueError, match="different _clean source"):
        solve_tape_chain(board=board, stem=source, through_row=5, max_delay=0, from_row=4, source=source)
