"""Offline unit tests for NES Metroid screen timing (no ROM)."""

from __future__ import annotations

from metroid.ram import ENGINE_TITLE, GAME_MODE_PAUSED, GAME_MODE_PLAYING
from metroid.screen_timer import (
    DiscontinuityReason,
    ScreenTimer,
    TimingSnapshot,
    is_settled_play,
    run_offline,
    snapshots_from_json,
)


def _play(
    frame: int,
    map_x: int,
    map_y: int = 14,
    *,
    equipment: int = 0,
    missiles: int = 0,
    capacity: int = 0,
    area: int = 0x10,
) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        map_x=map_x,
        map_y=map_y,
        game_mode=GAME_MODE_PLAYING,
        in_door=0,
        area=area,
        health_lo=0x00,
        health_hi=0x03,
        equipment=equipment,
        missiles=missiles,
        missile_capacity=capacity,
    )


def _door(
    frame: int,
    map_x: int,
    map_y: int = 14,
    *,
    in_door: int = 1,
) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        map_x=map_x,
        map_y=map_y,
        game_mode=GAME_MODE_PLAYING,
        in_door=in_door,
        area=0x10,
        health_lo=0x00,
        health_hi=0x03,
    )


def test_is_settled_play_requires_clean_door_and_mode() -> None:
    assert is_settled_play(_play(0, 3))
    assert not is_settled_play(_door(0, 3))
    assert not is_settled_play(
        TimingSnapshot(
            frame=0,
            map_x=3,
            map_y=14,
            game_mode=8,
            health_lo=0x00,
            health_hi=0x03,
        )
    )
    assert not is_settled_play(
        TimingSnapshot(
            frame=0,
            map_x=3,
            map_y=14,
            paused=1,
            health_lo=0x00,
            health_hi=0x03,
        )
    )
    assert not is_settled_play(
        TimingSnapshot(
            frame=0,
            map_x=0xFF,
            map_y=14,
            health_lo=0x00,
            health_hi=0x03,
        )
    )
    assert not is_settled_play(
        TimingSnapshot(frame=0, map_x=3, map_y=14, health_lo=0, health_hi=0)
    )


def test_confirmed_door_transition_records_frame_timings() -> None:
    """Classic hop: settle (3,14) → door → settle (2,14)."""
    samples = [
        _play(0, 3, equipment=0),
        _play(20, 3, equipment=0),
        _door(21, 3, in_door=1),
        _door(35, 3, in_door=1),
        # Map may flip mid-load before settle.
        _door(45, 2, in_door=1),
        _play(60, 2, equipment=0),
    ]
    timer = ScreenTimer()
    completed = timer.observe_many(samples)

    assert len(completed) == 1
    visit = completed[0]
    assert visit.source_map_cell == (0, 0)
    assert visit.map_cell == (3, 14)
    assert visit.dest_map_cell == (2, 14)
    assert visit.entry_frame == 0
    assert visit.leave_frame == 21
    assert visit.exit_frame == 60
    assert visit.screen_frames == 60
    assert visit.dwell_frames == 21
    assert visit.transition_frames == 39
    assert visit.in_door_at_leave == 1
    assert visit.sequence_index == 0


def test_chain_of_two_hops_west_corridor() -> None:
    """Start (3,14) → (2,14) → morph (1,14)."""
    samples = [
        _play(0, 3),
        _door(5, 3),
        _play(15, 2),
        _play(18, 2),
        _door(20, 2),
        _play(40, 1, equipment=0x10),
    ]
    timer = ScreenTimer()
    visits = timer.observe_many(samples)

    assert len(visits) == 2
    assert visits[0].map_cell == (3, 14) and visits[0].dest_map_cell == (2, 14)
    assert visits[0].source_map_cell == (0, 0)
    assert visits[1].map_cell == (2, 14) and visits[1].dest_map_cell == (1, 14)
    assert visits[1].source_map_cell == (3, 14)
    assert visits[1].entry_frame == 15
    assert visits[1].leave_frame == 20
    assert visits[1].exit_frame == 40
    assert visits[1].dwell_frames == 5
    assert visits[1].transition_frames == 20
    assert visits[1].screen_frames == 25
    assert visits[1].equipment == 0
    assert visits[1].sequence_index == 1


def test_boot_and_menu_ignored_until_settled() -> None:
    samples = [
        TimingSnapshot(
            frame=0,
            map_x=0,
            map_y=0,
            engine_mode=ENGINE_TITLE,
            game_mode=0,
            health_lo=0,
            health_hi=0,
        ),
        TimingSnapshot(
            frame=10,
            map_x=0,
            map_y=0,
            engine_mode=ENGINE_TITLE,
            game_mode=0,
            health_lo=0,
            health_hi=0,
        ),
        _play(100, 3),
        _door(110, 3),
        _play(130, 2),
    ]
    report = run_offline(samples, source="boot_prefix")
    assert report["visit_count"] == 1
    visit = report["visits"][0]
    assert visit["entry_frame"] == 100
    assert visit["map_cell"] == [3, 14]
    assert visit["dest_map_cell"] == [2, 14]


def test_title_after_play_abandons_open_visit() -> None:
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 3),
            _play(5, 3),
            TimingSnapshot(
                frame=6,
                map_x=0,
                map_y=0,
                engine_mode=ENGINE_TITLE,
                health_lo=0,
                health_hi=0,
            ),
            _play(50, 5),
            _door(55, 5),
            _play(70, 6),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].map_cell == (5, 14)
    assert any(
        d.reason is DiscontinuityReason.BOOT_OR_MENU for d in timer.discontinuities
    )


def test_death_zero_energy_abandons_open_visit() -> None:
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 3),
            TimingSnapshot(
                frame=10,
                map_x=3,
                map_y=14,
                game_mode=GAME_MODE_PLAYING,
                health_lo=0,
                health_hi=0,
            ),
            _play(100, 11, map_y=13),
            _door(105, 11, map_y=13),
            _play(120, 11, map_y=12),
        ]
    )
    assert any(
        d.reason is DiscontinuityReason.DEATH_OR_RESET for d in timer.discontinuities
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].map_cell == (11, 13)
    assert timer.visits[0].dest_map_cell == (11, 12)


def test_frame_regression_is_discontinuity() -> None:
    """Save-state load / rewind should not invent a hop."""
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(100, 3),
            _play(110, 3),
            _play(50, 5),  # rewound / loaded
            _door(55, 5),
            _play(80, 6),
        ]
    )
    assert any(
        d.reason is DiscontinuityReason.FRAME_REGRESSION
        for d in timer.discontinuities
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].map_cell == (5, 14)
    assert timer.visits[0].dest_map_cell == (6, 14)
    assert timer.visits[0].entry_frame == 50


def test_seamless_adjacent_scroll_is_timed_hop() -> None:
    """Corridor screens often flip map_x with in_door still 0."""
    timer = ScreenTimer()
    visits = timer.observe_many(
        [
            _play(0, 3),
            _play(40, 3),
            _play(41, 2),  # settled adjacent; no door flag
            _play(80, 2),
            _play(81, 1),
        ]
    )
    assert len(visits) == 2
    assert visits[0].map_cell == (3, 14)
    assert visits[0].dest_map_cell == (2, 14)
    assert visits[0].entry_frame == 0
    assert visits[0].leave_frame == 41
    assert visits[0].exit_frame == 41
    assert visits[0].transition_frames == 0
    assert visits[0].dwell_frames == 41
    assert visits[0].in_door_at_leave == 0
    assert visits[1].map_cell == (2, 14)
    assert visits[1].dest_map_cell == (1, 14)
    assert not timer.discontinuities


def test_map_jump_non_adjacent_without_leave_is_not_a_timed_hop() -> None:
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 3),
            _play(10, 3),
            # Non-adjacent settled change (state load / warp).
            _play(11, 11, map_y=13),
            _door(20, 11, map_y=13),
            _play(40, 11, map_y=12),
        ]
    )
    assert any(d.reason is DiscontinuityReason.MAP_JUMP for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].map_cell == (11, 13)
    assert timer.visits[0].dest_map_cell == (11, 12)


def test_same_cell_return_after_door_cancels_leave() -> None:
    """Bounce back into the same map cell: do not complete a hop."""
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 3),
            _door(5, 3),
            _play(20, 3),  # failed / same cell settle
            _door(30, 3),
            _play(50, 4),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].entry_frame == 0
    assert timer.visits[0].leave_frame == 30
    assert timer.visits[0].dest_map_cell == (4, 14)


def test_pause_does_not_complete_transition() -> None:
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 3),
            TimingSnapshot(
                frame=5,
                map_x=3,
                map_y=14,
                game_mode=GAME_MODE_PAUSED,
                paused=1,
                health_lo=0x00,
                health_hi=0x03,
            ),
            _play(20, 3),
            _door(25, 3),
            _play(40, 4),
        ]
    )
    assert len(timer.visits) == 1
    # Pause marks leave then cancels on same-cell settle; real door at 25.
    assert timer.visits[0].leave_frame == 25
    assert timer.visits[0].dwell_frames == 25


def test_item_fanfare_mode_9_debounces_until_playing() -> None:
    """game_mode 9 (item fanfare) is unsettled; return to same cell continues."""
    timer = ScreenTimer()
    timer.observe_many(
        [
            _play(0, 1, equipment=0x10),
            TimingSnapshot(
                frame=5,
                map_x=1,
                map_y=14,
                game_mode=9,
                health_lo=0x00,
                health_hi=0x03,
                equipment=0x10,
            ),
            _play(20, 1, equipment=0x10),
            _door(25, 1),
            _play(40, 2, equipment=0x10),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].leave_frame == 25
    assert timer.visits[0].equipment == 0x10


def test_unsettled_intro_mode_ignored_before_first_settle() -> None:
    samples = [
        TimingSnapshot(
            frame=0,
            map_x=3,
            map_y=14,
            game_mode=8,
            health_lo=0,
            health_hi=0,
        ),
        TimingSnapshot(
            frame=50,
            map_x=3,
            map_y=14,
            game_mode=8,
            health_lo=0x00,
            health_hi=0x03,
        ),
        _play(100, 3),
        _door(110, 3),
        _play(130, 2),
    ]
    report = run_offline(samples, source="intro")
    assert report["visit_count"] == 1
    assert report["visits"][0]["entry_frame"] == 100


def test_report_schema_and_totals() -> None:
    samples = [
        _play(0, 3),
        _door(10, 3),
        _play(30, 4),
        _door(40, 4),
        _play(55, 5),
    ]
    report = run_offline(samples, source="totals")
    assert report["schema_version"] == 1
    assert report["kind"] == "metroid_screen_timing"
    assert report["timing_unit"] == "emulator_frames"
    assert report["timing_semantics"]["igt_or_lag"] is False
    assert report["visit_count"] == 2
    assert report["total_screen_frames"] == 30 + 25
    assert report["total_dwell_frames"] == 10 + 10
    assert report["total_transition_frames"] == 20 + 15
    assert report["open_visit"] is None  # finalized


def test_snapshots_from_json_fixture_shape() -> None:
    payload = {
        "samples": [
            {
                "frame": 0,
                "map_x": 3,
                "map_y": 14,
                "game_mode": 3,
                "in_door": 0,
                "health_hi": 3,
            },
            {
                "frame": 5,
                "map_cell": [3, 14],
                "game_mode": 3,
                "in_door": 1,
                "health_hi": 3,
            },
            {
                "frame": 20,
                "map_x": 2,
                "map_y": 14,
                "game_mode": 3,
                "in_door": 0,
                "area": 16,
                "health_hi": 3,
            },
        ]
    }
    samples = snapshots_from_json(payload)
    report = run_offline(samples, source="json_fixture")
    assert report["visit_count"] == 1
    assert report["visits"][0]["map_cell"] == [3, 14]
    assert report["visits"][0]["dest_map_cell"] == [2, 14]


def test_inventory_context_captured_at_visit() -> None:
    samples = [
        _play(0, 3, equipment=0x10, missiles=5, capacity=5),
        _door(10, 3),
        _play(30, 4, equipment=0x10, missiles=4, capacity=5),
    ]
    visits = ScreenTimer().observe_many(samples)
    assert visits[0].equipment == 0x10
    assert visits[0].missiles == 5
    assert visits[0].missile_capacity == 5


def test_from_snapshot_roundtrip() -> None:
    from metroid.ram import MetroidSnapshot

    snap = MetroidSnapshot(
        engine_mode=0,
        game_mode=3,
        paused=0,
        map_x=3,
        map_y=14,
        samus_x=128,
        samus_y=176,
        samus_dir=1,
        in_door=0,
        room_layout=0,
        area=0x10,
        health_lo=0x00,
        health_hi=0x03,
        item_pause=0,
        missiles_enabled=0,
        samus_status=0,
        frame_counter=10,
        equipment=0x10,
        missiles=0,
        missile_capacity=0,
        energy_tanks=0,
    )
    ts = TimingSnapshot.from_snapshot(snap, frame=42)
    assert ts.frame == 42
    assert ts.map_cell == (3, 14)
    assert ts.equipment == 0x10
    timer = ScreenTimer()
    assert timer.observe(snap, frame=42) is None
    open_visit = timer.report()["open_visit"]
    assert open_visit is not None
    assert open_visit["map_cell"] == [3, 14]
    assert open_visit["entry_frame"] == 42
