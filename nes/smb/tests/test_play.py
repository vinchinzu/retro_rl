"""32-exit play recorder: presets, archive, exit clock (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path

from smb.play_record import (
    ExitClock,
    archive_existing_take,
    fmt_time,
    stage_label,
    stage_label_of,
    write_stage_pin,
)
from smb.ram import SmbSnapshot
from smb.routes import ROUTE_ALL_EXITS, ROUTE_WARP_ANY_PERCENT
from smb.scripts.play import main as play_main
from smb.start_presets import (
    POWER_ON_STARTS,
    is_power_on,
    list_start_presets,
    normalize_stage_id,
    resolve_start,
)


def _snap(
    *,
    world: int = 0,
    level: int = 0,
    oper_mode: int = 1,
    player_state: int = 8,
    lives: int = 2,
    player_x: int = 40,
    dying: bool = False,
    level_number: int | None = None,
) -> SmbSnapshot:
    return SmbSnapshot(
        frame=0,
        player_state=0x0B if dying else player_state,
        player_x=player_x,
        player_y=176,
        x_page=player_x // 256,
        x_offset=player_x % 256,
        lives=lives,
        world=world,
        level=level,
        level_id=world * 4 + level,
        oper_mode=oper_mode,
        player_power=0,
        timer_hundreds=4,
        timer=400,
        area_pointer=0,
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
        level_number=level_number,
    )


def test_power_on_aliases_and_stage_ids() -> None:
    assert is_power_on("start")
    assert is_power_on("power-on")
    assert set(POWER_ON_STARTS)
    assert normalize_stage_id("4-1") == "4-1"
    assert normalize_stage_id("smb_8_4") == "8-4"
    assert normalize_stage_id("resume") is None


def test_resolve_start_power_on(tmp_path: Path) -> None:
    resolved = resolve_start("start", task_name="t", out_dir=tmp_path)
    assert resolved.kind == "power_on"
    assert resolved.path is None
    assert resolved.route_index == 0


def test_resolve_start_missing_pin_raises(tmp_path: Path) -> None:
    try:
        resolve_start("3-1", task_name="t", out_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "3-1" in str(exc)
    else:
        raise AssertionError("expected missing 3-1 pin")


def test_list_presets_marks_missing_pins(tmp_path: Path) -> None:
    rows = list_start_presets(task_name="t", out_dir=tmp_path)
    keys = [row[0] for row in rows]
    assert keys[0] == "start"
    assert "1-1" in keys
    assert "8-4" in keys
    assert "resume" in keys
    by_key = {row[0]: row[1] for row in rows}
    assert by_key["3-2"] == "MISSING"


def test_fmt_time_and_stage_label() -> None:
    assert fmt_time(0) == "0:00.000"
    assert fmt_time(60) == "0:01.000"
    assert fmt_time(21559) == "5:59.316"
    assert stage_label(0, 0) == "1-1"
    assert stage_label(7, 3) == "8-4"


def test_exit_clock_counts_all_thirty_two() -> None:
    clock = ExitClock(ROUTE_ALL_EXITS)
    frame = 0
    for world in range(8):
        for level in range(4):
            frame += 10
            assert clock.observe(_snap(world=world, level=level), frame=frame) == "entry"
            frame += 100
            if (world, level) == (7, 3):
                event = clock.observe(
                    _snap(world=7, level=3, oper_mode=2), frame=frame
                )
            elif level < 3:
                event = clock.observe(_snap(world=world, level=level + 1), frame=frame)
            else:
                event = clock.observe(_snap(world=world + 1, level=0), frame=frame)
            assert event == "exit", (world, level, event)
    assert clock.complete
    assert [row["exit_id"] for row in clock.completed] == [
        f"{w}-{lv}" for w in range(1, 9) for lv in range(1, 5)
    ]
    assert clock.completed[-1]["successor"] == "ending"


def test_exit_clock_rewind_drops_later_events() -> None:
    clock = ExitClock(ROUTE_ALL_EXITS)
    assert clock.observe(_snap(), frame=1) == "entry"
    assert clock.observe(_snap(world=0, level=1), frame=50) == "exit"
    clock.rewind(40)
    assert clock.completed == []
    assert "1-1" in clock.entries
    assert clock.observe(_snap(world=0, level=1), frame=45) == "exit"


def test_exit_clock_death_does_not_abort() -> None:
    clock = ExitClock(ROUTE_ALL_EXITS)
    assert clock.observe(_snap(), frame=1) == "entry"
    assert clock.observe(_snap(dying=True, lives=1), frame=2) == "death"
    assert clock.observe(_snap(world=0, level=1), frame=50) == "exit"
    assert clock.completed[0]["exit_id"] == "1-1"
    assert len(clock.deaths) == 1
    assert not clock.complete


def test_exit_clock_ignores_1_2_underground_as_1_3() -> None:
    """1-2 pipe flips AreaNumber ($0760) to 2; LevelNumber ($075C) stays 1."""
    clock = ExitClock(ROUTE_ALL_EXITS)
    assert clock.observe(_snap(world=0, level=0), frame=1) == "entry"
    assert clock.observe(_snap(world=0, level=1, level_number=1), frame=50) == "exit"
    assert clock.observe(_snap(world=0, level=1, level_number=1), frame=51) == "entry"
    ug = _snap(
        world=0,
        level=2,
        level_number=1,
        player_x=160,
        dying=False,
    )
    assert clock.observe(ug, frame=80) is None
    assert [row["exit_id"] for row in clock.completed] == ["1-1"]
    assert stage_label_of(ug) == "1-2"
    real_13 = _snap(world=0, level=2, level_number=2, player_x=40)
    assert clock.observe(real_13, frame=400) == "exit"
    assert clock.completed[-1]["exit_id"] == "1-2"
    assert clock.observe(real_13, frame=401) == "entry"
    assert stage_label_of(real_13) == "1-3"


def test_exit_clock_warns_on_skipped_warp() -> None:
    clock = ExitClock(ROUTE_ALL_EXITS)
    clock.observe(_snap(), frame=1)
    event = clock.observe(_snap(world=3, level=0), frame=80)
    assert event == "off_route"
    assert clock.off_route[0]["stage"] == "4-1"
    assert clock.off_route[0]["expected"] == "1-1"
    assert clock.completed == []


def test_warp_route_still_clocks_eight() -> None:
    clock = ExitClock(ROUTE_WARP_ANY_PERCENT)
    successors = (
        _snap(world=0, level=1),
        _snap(world=3, level=0),
        _snap(world=3, level=1),
        _snap(world=7, level=0),
        _snap(world=7, level=1),
        _snap(world=7, level=2),
        _snap(world=7, level=3),
        _snap(world=7, level=3, oper_mode=2),
    )
    for frame, snap in enumerate(successors, start=1):
        assert clock.observe(snap, frame=frame) == "exit"
    assert clock.complete
    assert [row["exit_id"] for row in clock.completed] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]


def test_archive_existing_take(tmp_path: Path) -> None:
    tape = tmp_path / "all_exits_v1.json"
    tape.write_text(
        json.dumps({"name": "all_exits_v1", "frames": [[0] * 9] * 4}),
        encoding="utf-8",
    )
    dest = archive_existing_take(tape)
    assert dest is not None
    assert (dest / "tape.json").is_file()
    join = json.loads((dest / "join.json").read_text(encoding="utf-8"))
    assert join["kind"] == "smb_segment_join"
    assert join["frame_count"] == 4
    again = archive_existing_take(tape)
    assert again is not None
    assert again.name == "s1"


def test_write_stage_pin(tmp_path: Path) -> None:
    snap = _snap(world=3, level=0, player_x=48)
    path = write_stage_pin(
        task_name="all_exits_v1",
        stage_id="4-1",
        state_bytes=b"fake-state",
        snap=snap,
        frame=100,
        rta_frames=6200,
        out_dir=tmp_path,
    )
    assert path.name == "4-1.state"
    assert path.read_bytes() == b"fake-state"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["rta_frames"] == 6200
    assert meta["fingerprint"]["world"] == 3
    resolved = resolve_start("4-1", task_name="all_exits_v1", out_dir=tmp_path)
    assert resolved.kind == "state"
    assert resolved.route_index == ROUTE_ALL_EXITS.exits.index(
        next(e for e in ROUTE_ALL_EXITS.exits if e.exit_id == "4-1")
    )


def test_play_list_exits_zero() -> None:
    assert play_main(["--list"]) == 0
