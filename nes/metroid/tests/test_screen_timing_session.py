"""Integration tests for opt-in runner screen-timing (no ROM required)."""

from __future__ import annotations

import json
from pathlib import Path
from metroid.ram import (
    ENGINE_GAME,
    GAME_MODE_PLAYING,
    MetroidSnapshot,
)
from metroid.screen_timing_session import (
    ScreenTimingSession,
    default_timing_artifact_path,
)
from metroid.screen_timer import TimingSnapshot


def _snap(
    *,
    map_x: int,
    map_y: int = 14,
    in_door: int = 0,
    equipment: int = 0x10,
    health_hi: int = 0x03,
    game_mode: int = GAME_MODE_PLAYING,
) -> MetroidSnapshot:
    return MetroidSnapshot(
        engine_mode=ENGINE_GAME,
        game_mode=game_mode,
        paused=0,
        map_x=map_x,
        map_y=map_y,
        samus_x=100,
        samus_y=176,
        samus_dir=1,
        in_door=in_door,
        room_layout=0,
        area=0x10,
        health_lo=0x00,
        health_hi=health_hi,
        item_pause=0,
        missiles_enabled=0,
        samus_status=0,
        frame_counter=0,
        equipment=equipment,
        missiles=0,
        missile_capacity=0,
        energy_tanks=0,
    )


class _FakeEnv:
    """Minimal env surface: get_ram + optional memory extract via snapshot inject."""

    def __init__(self) -> None:
        self._snap = _snap(map_x=3)

    def set_snap(self, snap: MetroidSnapshot) -> None:
        self._snap = snap

    def get_ram(self):
        # System RAM bytes are unused when we patch read_snapshot in tests
        # that go through observe_env — we monkeypatch instead.
        return bytes(0x800)


def test_disabled_session_is_noop(monkeypatch) -> None:
    session = ScreenTimingSession(enabled=False)
    env = _FakeEnv()

    def boom(*_a, **_k):
        raise AssertionError("read_snapshot should not run when disabled")

    monkeypatch.setattr(
        "metroid.screen_timing_session.read_snapshot", boom
    )
    assert session.observe_env(env, phase="MORPH_EXIT") is None
    assert session.report() == {"enabled": False, "source": ""}
    assert session.write_report() is None


def test_observe_env_records_door_hop(monkeypatch, tmp_path: Path) -> None:
    session = ScreenTimingSession(
        enabled=True,
        source="test",
        entry_mode="after_morph",
        diagnostic_state_load="AfterMorph",
    )
    frames = [
        _snap(map_x=1),
        _snap(map_x=1),
        _snap(map_x=1, in_door=1),
        _snap(map_x=2, in_door=1),
        _snap(map_x=2),
        _snap(map_x=2),
        _snap(map_x=3),  # seamless adjacent
    ]
    idx = {"i": 0}

    def fake_read(_ram, env=None):
        snap = frames[min(idx["i"], len(frames) - 1)]
        idx["i"] += 1
        return snap

    monkeypatch.setattr(
        "metroid.screen_timing_session.read_snapshot", fake_read
    )
    env = _FakeEnv()
    phases = [
        "MORPH_EXIT",
        "MORPH_EXIT",
        "MORPH_EXIT",
        "MORPH_EXIT",
        "RETURN_STAND",
        "EAST_CORRIDOR",
        "EAST_CORRIDOR",
    ]
    for phase in phases:
        session.observe_env(env, phase=phase)

    assert len(session.timer.visits) == 2
    assert session.timer.visits[0].map_cell == (1, 14)
    assert session.timer.visits[0].dest_map_cell == (2, 14)
    assert session.timer.visits[1].map_cell == (2, 14)
    assert session.timer.visits[1].dest_map_cell == (3, 14)
    assert session.timer.visits[1].transition_frames == 0

    bn = session.bottleneck_summary()
    assert bn["visit_count"] == 2
    assert bn["longest_by_screen_frames"]["map_cell"] == [1, 14]
    assert bn["open_visit"] is not None
    assert bn["open_visit"]["map_cell"] == [3, 14]
    assert any(m["phase"] == "MORPH_EXIT" for m in bn["phase_markers"])
    assert any(m["phase"] == "EAST_CORRIDOR" for m in bn["phase_markers"])

    out = session.write_report(tmp_path / "timing.json")
    assert out is not None and out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "metroid_screen_timing"
    assert payload["enabled"] is True
    assert payload["visit_count"] == 2
    extra = payload["extra"]
    assert extra["entry_mode"] == "after_morph"
    assert extra["evaluation_class"] == "diagnostic_state_load"
    assert extra["diagnostic_state_load"] == "AfterMorph"
    assert "diagnostic_note" in extra
    assert extra["bottleneck"]["visit_count"] == 2


def test_natural_entry_evaluation_class() -> None:
    session = ScreenTimingSession(
        enabled=True,
        source="natural",
        entry_mode="natural",
        diagnostic_state_load=None,
    )
    # Empty session still reports class after finalize.
    report = session.report()
    assert report["extra"]["evaluation_class"] == "clean_natural_entry"
    assert "diagnostic_state_load" not in report["extra"]


def test_default_artifact_path_under_screen_timings() -> None:
    path = default_timing_artifact_path("natural")
    assert path.name == "first_missiles_natural_timing.json"
    assert path.parent.name == "screen_timings"


def test_bottleneck_prefers_longest_dwell() -> None:
    session = ScreenTimingSession(enabled=True, entry_mode="natural")
    # Inject visits via raw timer samples (no env).
    samples = [
        TimingSnapshot(
            frame=0, map_x=3, map_y=14, health_hi=3, area=0x10, equipment=0x10
        ),
        TimingSnapshot(
            frame=5,
            map_x=3,
            map_y=14,
            in_door=1,
            health_hi=3,
            area=0x10,
            equipment=0x10,
        ),
        TimingSnapshot(
            frame=20, map_x=4, map_y=14, health_hi=3, area=0x10, equipment=0x10
        ),
        TimingSnapshot(
            frame=200, map_x=4, map_y=14, health_hi=3, area=0x10, equipment=0x10
        ),
        TimingSnapshot(
            frame=201,
            map_x=4,
            map_y=14,
            in_door=1,
            health_hi=3,
            area=0x10,
            equipment=0x10,
        ),
        TimingSnapshot(
            frame=220, map_x=5, map_y=14, health_hi=3, area=0x10, equipment=0x10
        ),
    ]
    session.timer.observe_many(samples)
    session.absolute_frame = 220
    bn = session.bottleneck_summary()
    assert bn["longest_by_dwell_frames"]["map_cell"] == [4, 14]
    assert bn["longest_by_dwell_frames"]["dwell_frames"] == 181
    assert bn["longest_by_screen_frames"]["map_cell"] == [4, 14]


def test_run_first_missiles_cli_has_screen_timing_flag() -> None:
    """Smoke: argparse accepts --screen-timing without executing a run."""
    from metroid.scripts import run_first_missiles as mod

    parser_src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "--screen-timing" in parser_src
    assert "ScreenTimingSession" in parser_src
    assert "diagnostic_state_load" in parser_src
