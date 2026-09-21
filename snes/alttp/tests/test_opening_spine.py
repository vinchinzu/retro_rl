"""Argparse and --through dispatch for the opening-spine CLI (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alttp.opening_route.full_tip import FullTipResult
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
        cli.build_parser().parse_args(["--through", "sanctuary"])


def test_through_room_72_is_accepted() -> None:
    args = cli.build_parser().parse_args(["--through", "room_72"])
    assert args.through == "room_72"
    assert args.video is False


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


def test_room_01_fail_closed_does_not_run_tip() -> None:
    def boom(*_args: object, **_kwargs: object) -> FullTipResult:
        raise AssertionError("must not compose room_01")

    payload = cli.run_opening_spine("room_01", run_tip_fn=boom)
    assert payload["ok"] is False
    assert payload["through"] == "room_01"
    assert payload["blocker"] == "room_01 not on continuous tip"
    assert payload["continuous"] is False
    assert payload["leftover"] is None
    assert payload["phase"] == "fail_closed"
    assert payload["tip_node"] == "room_50"


def test_zelda_fail_closed_until_f3cc() -> None:
    def boom(*_args: object, **_kwargs: object) -> FullTipResult:
        raise AssertionError("must not compose zelda")

    payload = cli.run_opening_spine("zelda", run_tip_fn=boom)
    assert payload["ok"] is False
    assert payload["through"] == "zelda"
    assert payload["blocker"] == "zelda not on continuous tip; $F3CC==1 not measured"
    assert "$F3CC" in payload["blocker"]
    assert payload["continuous"] is False
    assert payload["leftover"] is None


def test_room_72_fail_closed_does_not_run_tip() -> None:
    def boom(*_args: object, **_kwargs: object) -> FullTipResult:
        raise AssertionError("must not compose room_72 stairs")

    payload = cli.run_opening_spine("room_72", run_tip_fn=boom)
    assert payload["ok"] is False
    assert payload["through"] == "room_72"
    assert payload["blocker"] == cli.ROOM_72_BLOCKER
    assert payload["blocker"] == (
        "room_72 not on continuous tip; stairs not composed into full_tip"
    )
    assert payload["continuous"] is False
    assert payload["leftover"] is None
    assert payload["phase"] == "fail_closed"
    assert payload["tip_node"] == "room_50"
    assert payload["video"] is None


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


def test_main_room_01_does_not_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "build_boot_env",
        lambda: (_ for _ in ()).throw(AssertionError("must not boot")),
    )
    monkeypatch.setattr(
        cli,
        "run_to_verified_tip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not run tip")
        ),
    )
    report = tmp_path / "opening_spine_room_01.json"
    rc = cli.main(["--through", "room_01", "--no-video", "--report", str(report)])
    assert rc == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["blocker"] == "room_01 not on continuous tip"
    assert payload["video"] is None


def test_main_zelda_does_not_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "build_boot_env",
        lambda: (_ for _ in ()).throw(AssertionError("must not boot")),
    )
    report = tmp_path / "opening_spine_zelda.json"
    rc = cli.main(["--through", "zelda", "--report", str(report)])
    assert rc == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["blocker"] == "zelda not on continuous tip; $F3CC==1 not measured"
    assert payload["ok"] is False


def test_main_room_72_does_not_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "build_boot_env",
        lambda: (_ for _ in ()).throw(AssertionError("must not boot")),
    )
    monkeypatch.setattr(
        cli,
        "run_to_verified_tip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not run tip")
        ),
    )
    report = tmp_path / "opening_spine_room_72.json"
    rc = cli.main(["--through", "room_72", "--no-video", "--report", str(report)])
    assert rc == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["through"] == "room_72"
    assert payload["blocker"] == cli.ROOM_72_BLOCKER
    assert payload["continuous"] is False
    assert payload["leftover"] is None
    assert payload["video"] is None
    assert report.name != "verified_tip_run.json"


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
        cli.main(["--through", "room_01", "--report", str(banned)])
    assert not banned.exists()
