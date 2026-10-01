"""Argparse and --through dispatch for the opening-spine CLI (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alttp.opening_route.full_tip import FullTipResult
from alttp.opening_route.segment import SegmentEvidence
from alttp.paths import RECORDINGS_DIR
from alttp.ram import AlttpSnapshot, HYRULE_CASTLE_NW_ROOM
from alttp.route_report import RoutePhaseResult
from alttp.scripts import run_opening_spine as cli


def _snap(*, room: int = HYRULE_CASTLE_NW_ROOM, follower: int = 0) -> AlttpSnapshot:
    return AlttpSnapshot(
        game_mode=0x07,
        submodule=0,
        room_id=room,
        indoors=True,
        screen_id=0,
        link_x=376,
        link_y=3088,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=1,
        lamp_level=1,
        num_keys=0xFF,
        follower=follower,
    )


def _tip(*, ok: bool = True, follower: int = 0) -> FullTipResult:
    snap = _snap(follower=follower)
    boot = RoutePhaseResult(
        phase="boot_to_castle",
        ok=True,
        frames=5,
        snapshot=snap,
        detail="fake boot",
    )
    return FullTipResult(
        ok=ok,
        phase="verified_tip_reached" if ok else "castle_to_sword",
        frames=42,
        snapshot=snap,
        tip_node="room_50",
        boot=boot,
        blocker="" if ok else "fake segment fail",
        notes=["fake tip"],
        source="natural_boot",
    )


def test_help_lists_through() -> None:
    help_text = cli.build_parser().format_help()
    assert "--through" in help_text
    assert "room_50" in help_text
    assert "room_01" in help_text
    assert "room_72" in help_text
    assert "room_81" in help_text
    assert "zelda" in help_text
    assert "--no-video" in help_text


def test_main_help_lists_through(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--through" in out
    assert "room_50" in out
    assert "room_01" in out
    assert "room_72" in out
    assert "room_81" in out
    assert "zelda" in out
    assert "--no-video" in out


def test_default_through_is_room_50() -> None:
    args = cli.build_parser().parse_args([])
    assert args.through == "room_50"
    assert args.video is False
    assert args.report is None


def test_no_video_is_the_default() -> None:
    parser = cli.build_parser()
    assert parser.parse_args([]).video is False
    assert parser.parse_args(["--no-video"]).video is False
    assert parser.parse_args(["--video"]).video is True


def test_unknown_through_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--through", "unknown_room"])


def test_through_room_72_is_accepted() -> None:
    args = cli.build_parser().parse_args(["--through", "room_72"])
    assert args.through == "room_72"
    assert args.video is False


def test_through_room_81_is_accepted() -> None:
    args = cli.build_parser().parse_args(["--through", "room_81"])
    assert args.through == "room_81"


def test_leftover_path_is_not_verified_tip_run() -> None:
    path = cli.leftover_path("room_50")
    assert path == RECORDINGS_DIR / "opening_spine_room_50.json"
    assert path != cli.VERIFIED_TIP_RUN
    assert path.name != "verified_tip_run.json"
    room_72 = cli.leftover_path("room_72")
    assert room_72 == RECORDINGS_DIR / "opening_spine_room_72.json"
    assert room_72 != cli.VERIFIED_TIP_RUN
    assert room_72.name != "verified_tip_run.json"


def test_room_50_dispatch_calls_verified_tip() -> None:
    calls: list[tuple[object, bool]] = []
    tip = _tip()

    def fake_tip(env: object, *, close: bool = True) -> FullTipResult:
        calls.append((env, close))
        return tip

    sentinel = object()
    payload = cli.run_opening_spine(
        "room_50", env=sentinel, close=False, run_tip_fn=fake_tip
    )

    assert calls == [(sentinel, False)]
    assert payload["ok"] is True
    assert payload["through"] == "room_50"
    assert payload["continuous"] is True
    assert payload["tip_node"] == "room_50"
    assert payload["leftover"]["room_hex"] == "0x50"
    assert payload["leftover"]["follower"] == 0
    assert payload["leftover"]["follower_addr"] == "$F3CC"
    assert payload["leftover"]["xy"] == [376, 3088]
    assert payload["report"]["kind"] == "alttp_verified_tip_run"
    assert payload["video"] is None


def test_room_01_dispatch_plays_east_hop() -> None:
    seen: dict[str, object] = {}

    def fake(
        env: object,
        *,
        close: bool = True,
        segment_ids: object = None,
        success: object = None,
    ) -> FullTipResult:
        seen["env"] = env
        seen["close"] = close
        seen["stages"] = segment_ids
        seen["success"] = success
        tip = _tip()
        tip.tip_node = "room_01"
        return tip

    payload = cli.run_opening_spine("room_01", env=object(), close=False, run_tip_fn=fake)
    assert seen["close"] is False
    assert seen["success"] == "room_01"
    assert seen["stages"] == (
        "castle_to_sword",
        "sword_to_secret_entrance_clear",
        "pocket_to_main_hall",
        "castle_dungeon_prefix",
        "room_50_east_to_0x01",
    )
    assert payload["ok"] is True
    assert payload["through"] == "room_01"
    assert payload["continuous"] is True
    assert payload["tip_node"] == "room_01"
    assert payload["phase"] != "fail_closed"


def test_zelda_dispatch_requires_follower_segment() -> None:
    seen = {}

    def fake(env, **kwargs):
        seen.update(kwargs)
        tip = _tip(follower=1)
        tip.tip_node = "room_80"
        return tip

    payload = cli.run_opening_spine("zelda", env=object(), run_tip_fn=fake)
    assert payload["ok"] is True
    assert seen["segment_ids"][-1] == "room_81_to_zelda"
    assert seen["success"] == "room_80"


def test_sanctuary_dispatch_plays_escort() -> None:
    seen = {}

    def fake(env, **kwargs):
        seen.update(kwargs)
        tip = _tip()
        tip.tip_node = "sanctuary"
        return tip

    payload = cli.run_opening_spine("sanctuary", env=object(), run_tip_fn=fake)
    assert payload["ok"] is True
    assert seen["segment_ids"][-2:] == ("room_81_to_zelda", "escort_to_sanctuary")
    assert seen["success"] == "sanctuary"


def _tip_room_72(*, x: int = 1273, y: int = 3665) -> FullTipResult:
    snap = _snap(room=0x72)
    snap = AlttpSnapshot(
        game_mode=snap.game_mode,
        submodule=snap.submodule,
        room_id=0x72,
        indoors=True,
        screen_id=0,
        link_x=x,
        link_y=y,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=1,
        lamp_level=1,
        num_keys=0,
        follower=0,
    )
    boot = RoutePhaseResult(
        phase="boot_to_castle",
        ok=True,
        frames=5,
        snapshot=snap,
        detail="fake boot",
    )
    return FullTipResult(
        ok=True,
        phase="verified_tip_reached",
        frames=80,
        snapshot=snap,
        tip_node="room_72",
        boot=boot,
        notes=["fake well"],
        source="natural_boot",
    )


def test_room_72_dispatch_plays_well_and_glances() -> None:
    seen: dict[str, object] = {}

    def fake(
        env: object,
        *,
        close: bool = True,
        segment_ids: object = None,
        success: object = None,
    ) -> FullTipResult:
        seen["stages"] = segment_ids
        seen["success"] = success
        return _tip_room_72()

    payload = cli.run_opening_spine("room_72", env=object(), close=False, run_tip_fn=fake)
    assert seen["success"] == "room_72"
    assert seen["stages"][-2:] == ("room_50_east_to_0x01", "room_01_down_to_0x72")
    assert payload["ok"] is True
    assert payload["through"] == "room_72"
    assert payload["continuous"] is True
    assert payload["tip_node"] == "room_72"
    assert payload["glance"]["ok"] is True
    assert payload["glance"]["misses"] == []
    assert payload["phase"] != "fail_closed"


def test_room_72_glance_miss_is_not_continuous() -> None:
    def fake(env: object, *, close: bool = True, **_kwargs: object) -> FullTipResult:
        return _tip_room_72(x=100, y=100)

    payload = cli.run_opening_spine("room_72", env=object(), close=False, run_tip_fn=fake)
    assert payload["ok"] is False
    assert payload["continuous"] is False
    assert payload["glance"]["ok"] is False
    assert payload["blocker"].startswith("ROOM_72 glance:")


def test_room_81_dispatch_plays_b1_hops_without_a_glance() -> None:
    seen: dict[str, object] = {}

    def fake(
        env: object,
        *,
        close: bool = True,
        segment_ids: object = None,
        success: object = None,
    ) -> FullTipResult:
        seen["stages"] = segment_ids
        seen["success"] = success
        tip = _tip()
        tip.tip_node = "room_81"
        return tip

    payload = cli.run_opening_spine("room_81", env=object(), close=False, run_tip_fn=fake)
    assert seen["success"] == "room_81"
    assert seen["stages"] == (
        "castle_to_sword",
        "sword_to_secret_entrance_clear",
        "pocket_to_main_hall",
        "castle_dungeon_prefix",
        "room_50_east_to_0x01",
        "room_01_down_to_0x72",
        "room_72_south_to_0x82",
        "room_82_west_to_0x81",
    )
    assert payload["ok"] is True
    assert payload["through"] == "room_81"
    assert payload["continuous"] is True
    assert payload["tip_node"] == "room_81"
    assert "glance" not in payload
    assert payload["phase"] != "fail_closed"


def test_write_leftover_refuses_verified_tip_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banned = tmp_path / "verified_tip_run.json"
    monkeypatch.setattr(cli, "VERIFIED_TIP_RUN", banned)
    with pytest.raises(RuntimeError, match="verified_tip_run"):
        cli.write_leftover(banned, {"ok": False})
    assert not banned.exists()


def test_main_room_50_writes_opening_spine_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = SimpleNamespace(closed=False)

    def close() -> None:
        env.closed = True

    env.close = close
    monkeypatch.setattr(cli, "build_boot_env", lambda: env)

    tip = _tip()
    called: dict[str, object] = {}

    def fake_tip(active_env: object, *, close: bool = True) -> FullTipResult:
        called["env"] = active_env
        called["close"] = close
        return tip

    monkeypatch.setattr(cli, "run_to_verified_tip", fake_tip)
    report = tmp_path / "opening_spine_room_50.json"
    rc = cli.main(["--through", "room_50", "--no-video", "--report", str(report)])

    assert rc == 0
    assert env.closed is True
    assert called["env"] is env
    assert called["close"] is False
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["through"] == "room_50"
    assert payload["leftover"]["room_hex"] == "0x50"
    assert payload["leftover"]["sword"] == 1
    assert report.name != "verified_tip_run.json"


def test_main_room_01_boots_the_east_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = SimpleNamespace(closed=False)

    def close() -> None:
        env.closed = True

    env.close = close
    monkeypatch.setattr(cli, "build_boot_env", lambda: env)

    def fake_tip(active_env: object, *, close: bool = True, **_kwargs: object) -> FullTipResult:
        assert active_env is env
        assert close is False
        tip = _tip()
        tip.tip_node = "room_01"
        return tip

    monkeypatch.setattr(cli, "run_to_verified_tip", fake_tip)
    report = tmp_path / "opening_spine_room_01.json"
    rc = cli.main(["--through", "room_01", "--no-video", "--report", str(report)])
    assert rc == 0
    assert env.closed is True
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["through"] == "room_01"
    assert payload["continuous"] is True
    assert payload["video"] is None


def test_main_zelda_boots_and_dispatches_rescue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = SimpleNamespace(closed=False)
    env.close = lambda: setattr(env, "closed", True)
    monkeypatch.setattr(cli, "build_boot_env", lambda: env)
    tip = _tip(follower=1)
    tip.tip_node = "room_80"
    monkeypatch.setattr(cli, "run_to_verified_tip", lambda *_a, **_k: tip)
    report = tmp_path / "opening_spine_zelda.json"
    assert cli.main(["--through", "zelda", "--report", str(report)]) == 0
    assert env.closed
    assert json.loads(report.read_text())["ok"]


def test_main_room_72_saves_arrival_and_glances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = SimpleNamespace(closed=False, frames=9)

    def close() -> None:
        env.closed = True

    env.close = close
    monkeypatch.setattr(cli, "build_boot_env", lambda: env)
    monkeypatch.setattr(cli, "run_to_verified_tip", lambda *_a, **_k: _tip_room_72())
    saved: list[tuple[str, str]] = []

    def fake_save(active: object, prefix: str, stage: str, **_kwargs: object) -> Path:
        assert active is not None
        saved.append((prefix, stage))
        return tmp_path / f"{prefix}_{stage}.state"

    monkeypatch.setattr("alttp.scripts.run_spine.write_save_point", fake_save)
    report = tmp_path / "opening_spine_room_72.json"
    rc = cli.main(
        ["--through", "room_72", "--no-video", "--save-points", "Full", "--report", str(report)]
    )
    assert rc == 0
    assert env.closed is True
    assert ("Full", "room_72") in saved
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["continuous"] is True
    assert payload["glance"]["ok"] is True
    assert "Full_room_72.state" in payload["save_points"]
    assert report.name != "verified_tip_run.json"


def test_last_stage_before_room_50_is_castle_dungeon_prefix() -> None:
    from alttp.spine.hops import last_stage, save_point_name

    assert last_stage("room_50") == "castle_dungeon_prefix"
    assert last_stage("room_01") == "room_50_east_to_0x01"
    assert last_stage("room_72") == "room_01_down_to_0x72"
    assert last_stage("room_82") == "room_72_south_to_0x82"
    assert last_stage("room_81") == "room_82_west_to_0x81"
    assert last_stage("zelda") == "room_81_to_zelda"
    assert last_stage("sanctuary") == "escort_to_sanctuary"
    assert save_point_name("Full", "castle_dungeon_prefix") == (
        "Full_castle_dungeon_prefix"
    )


def test_save_points_offers_a_stage_lookup() -> None:
    from alttp.scripts import run_spine as spine

    seen: dict[str, object] = {}

    def fake(
        env: object,
        *,
        close: bool = True,
        segment_ids: object = None,
        get_segment_fn: object = None,
    ) -> FullTipResult:
        seen["env"] = env
        seen["close"] = close
        seen["stages"] = segment_ids
        seen["lookup"] = get_segment_fn
        return _tip()

    payload = spine.dispatch(
        "room_50",
        env=object(),
        close=False,
        run_tip_fn=fake,
        save_points="Full",
    )
    assert seen["close"] is False
    assert seen["lookup"] is not None
    assert seen["stages"] == (
        "castle_to_sword",
        "sword_to_secret_entrance_clear",
        "pocket_to_main_hall",
        "castle_dungeon_prefix",
    )
    assert payload["ok"] is True
    assert payload["continuous"] is True
    assert payload["resumed_from"] is None
    assert payload["save_points"] == []


def test_resume_unknown_stage_does_not_play() -> None:
    from alttp.scripts import run_spine as spine

    def boom(*_args: object, **_kwargs: object) -> FullTipResult:
        raise AssertionError("must not play")

    with pytest.raises(ValueError, match="unknown resume stage"):
        spine.dispatch(
            "room_50",
            env=object(),
            run_tip_fn=boom,
            save_points="Full",
            resume="nope",
        )


def test_load_save_point_resets_before_set_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alttp.scripts import run_spine as spine

    pin = tmp_path / "Full_castle_dungeon_prefix.state"
    pin.write_bytes(b"gzip-stand-in")
    order: list[str] = []

    class _Em:
        def set_state(self, data: bytes) -> None:
            order.append(f"set_state:{data!r}")

    class _Data:
        def update_ram(self) -> None:
            order.append("update_ram")

    class _Env:
        def __init__(self) -> None:
            self.em = _Em()
            self.data = _Data()

        def reset(self) -> tuple[None, dict[str, object]]:
            order.append("reset")
            return None, {}

        def step(self, *_args: object, **_kwargs: object) -> None:
            order.append("step")

    monkeypatch.setattr("retro_harness.env.state_path", lambda *_a, **_k: pin)
    monkeypatch.setattr("retro_harness.env.read_state_bytes", lambda _path: b"PIN")
    loaded = spine.load_save_point(_Env(), "Full", "castle_dungeon_prefix")
    assert loaded == pin
    assert order == ["reset", "set_state:b'PIN'", "update_ram"]


def test_main_refuses_report_path_verified_tip_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banned = tmp_path / "verified_tip_run.json"
    monkeypatch.setattr(cli, "VERIFIED_TIP_RUN", banned)
    monkeypatch.setattr(
        cli,
        "build_boot_env",
        lambda: (_ for _ in ()).throw(AssertionError("must not boot")),
    )
    with pytest.raises(RuntimeError, match="verified_tip_run"):
        cli.main(["--through", "zelda", "--report", str(banned)])
    assert not banned.exists()


def test_controller_search_discloses_intervention_in_nested_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A green controller result with restores must not publish Clean rows."""
    writes = []
    loads = []
    env = SimpleNamespace(
        closed=False,
        em=SimpleNamespace(set_state=loads.append),
        data=SimpleNamespace(memory=SimpleNamespace(assign=lambda *args: writes.append(args))),
    )
    env.close = lambda: setattr(env, "closed", True)
    monkeypatch.setattr(cli, "build_boot_env", lambda: env)
    tip = _tip()
    tip.segments.append(SegmentEvidence(
        segment_id="castle_dungeon_prefix", ok=True, frames=37,
        snapshot=tip.snapshot, source="natural_boot", phase="prefix_reached",
        development_only=False,
    ))

    def fake_tip(active_env: object, **_kwargs: object) -> FullTipResult:
        active_env.unwrapped.em.set_state(b"lookahead_restore")
        active_env.unwrapped.data.memory.assign(0x7E0022, "|u1", 42)
        return tip

    monkeypatch.setattr(cli, "run_to_verified_tip", fake_tip)
    path = tmp_path / "controller.json"
    assert cli.main(["--through", "room_50", "--report", str(path)]) == 0
    payload = json.loads(path.read_text())
    assert loads == [b"lookahead_restore"]
    assert len(writes) == 1
    assert env.closed
    assert payload["audit"]["set_state"] == 1
    assert payload["audit"]["writes"] == 1
    assert payload["ok"]
    assert not payload["continuous"]
    for report in (payload, payload["report"]):
        assert report["clean_chain"] is False
        assert report["development_only"] is True
        assert report["source"] == "controller_search"
        assert "clean" not in report["intervention"].casefold()
    for row in payload["report"]["segments"]:
        assert row["cleanChain"] is False
        assert row["developmentOnly"] is True
        assert row["source"] == "controller_search"
