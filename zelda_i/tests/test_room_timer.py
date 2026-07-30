"""Offline unit tests for Zelda I screen/room timing (no ROM)."""

from __future__ import annotations

from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.room_timer import (
    DiscontinuityReason,
    GameContext,
    RoomTimer,
    TimingSnapshot,
    is_settled_play,
    run_offline,
    snapshots_from_json,
)


def _play(
    frame: int,
    screen: int,
    *,
    level: int = 0,
    sword: int = 0,
    keys: int = 0,
    triforce: int = 0,
    next_screen: int = 0,
) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        mode=PLAY_MODE,
        level=level,
        screen=screen,
        next_screen=next_screen,
        sword=sword,
        keys=keys,
        triforce=triforce,
        health=0x22,
    )


def _scroll(
    frame: int,
    screen: int,
    *,
    level: int = 0,
    mode: int = 6,
    next_screen: int = 0,
) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        mode=mode,
        level=level,
        screen=screen,
        next_screen=next_screen,
        health=0x22,
    )


def test_is_settled_play_requires_mode_five() -> None:
    assert is_settled_play(_play(0, 0x77))
    assert is_settled_play(_play(0, 0x73, level=1))
    assert not is_settled_play(_scroll(0, 0x77))
    assert not is_settled_play(
        TimingSnapshot(frame=0, mode=CAVE_MODE, level=0, screen=0x77)
    )
    assert not is_settled_play(
        TimingSnapshot(frame=0, mode=0, level=0, screen=0)
    )


def test_overworld_screen_hop_records_frame_timings() -> None:
    """Classic hop: settle A → scroll → settle B."""
    samples = [
        _play(0, 0x77, sword=1),
        _play(20, 0x77, sword=1),
        _scroll(21, 0x77, mode=6, next_screen=0x78),
        _scroll(35, 0x77, mode=7, next_screen=0x78),
        _scroll(45, 0x78, mode=7, next_screen=0x78),
        _play(60, 0x78, sword=1),
    ]
    timer = RoomTimer()
    completed = timer.observe_many(samples)

    assert len(completed) == 1
    visit = completed[0]
    assert visit.source_level == 0
    assert visit.source_screen == 0
    assert visit.level == 0
    assert visit.screen == 0x77
    assert visit.dest_level == 0
    assert visit.dest_screen == 0x78
    assert visit.context is GameContext.OVERWORLD
    assert visit.dest_context is GameContext.OVERWORLD
    assert visit.entry_frame == 0
    assert visit.leave_frame == 21
    assert visit.exit_frame == 60
    assert visit.location_frames == 60
    assert visit.dwell_frames == 21
    assert visit.transition_frames == 39
    assert visit.mode_at_leave == 6
    assert visit.next_screen_at_leave == 0x78
    assert visit.sword == 1
    assert visit.sequence_index == 0


def test_dungeon_room_hop() -> None:
    samples = [
        _play(0, 0x73, level=1, keys=0),
        _scroll(10, 0x73, level=1, mode=6, next_screen=0x74),
        _play(40, 0x74, level=1, keys=0),
    ]
    visits = RoomTimer().observe_many(samples)
    assert len(visits) == 1
    assert visits[0].context is GameContext.DUNGEON
    assert visits[0].dest_context is GameContext.DUNGEON
    assert visits[0].screen == 0x73
    assert visits[0].dest_screen == 0x74
    assert visits[0].level == 1
    assert visits[0].dest_level == 1
    assert visits[0].dwell_frames == 10
    assert visits[0].transition_frames == 30


def test_overworld_to_dungeon_entry() -> None:
    samples = [
        _play(0, 0x37, level=0, sword=1),
        _scroll(5, 0x37, level=0, mode=6),
        _play(80, 0x73, level=1, sword=1),
    ]
    visits = RoomTimer().observe_many(samples)
    assert len(visits) == 1
    assert visits[0].context is GameContext.OVERWORLD
    assert visits[0].dest_context is GameContext.DUNGEON
    assert visits[0].screen == 0x37
    assert visits[0].dest_screen == 0x73
    assert visits[0].dest_level == 1


def test_chain_of_two_overworld_hops() -> None:
    samples = [
        _play(0, 0x77),
        _scroll(5, 0x77, mode=6, next_screen=0x78),
        _play(15, 0x78),
        _play(18, 0x78),
        _scroll(20, 0x78, mode=6, next_screen=0x68),
        _play(40, 0x68),
    ]
    visits = RoomTimer().observe_many(samples)
    assert len(visits) == 2
    assert visits[0].screen == 0x77 and visits[0].dest_screen == 0x78
    assert visits[0].source_screen == 0
    assert visits[1].screen == 0x78 and visits[1].dest_screen == 0x68
    assert visits[1].source_screen == 0x77
    assert visits[1].entry_frame == 15
    assert visits[1].leave_frame == 20
    assert visits[1].exit_frame == 40
    assert visits[1].dwell_frames == 5
    assert visits[1].transition_frames == 20
    assert visits[1].location_frames == 25
    assert visits[1].sequence_index == 1


def test_boot_and_menu_ignored_until_settled() -> None:
    samples = [
        TimingSnapshot(frame=0, mode=0, level=0, screen=0),
        TimingSnapshot(frame=10, mode=1, level=0, screen=0),
        _play(100, 0x77),
        _scroll(110, 0x77, mode=6, next_screen=0x78),
        _play(130, 0x78),
    ]
    report = run_offline(samples, source="boot_prefix")
    assert report["visit_count"] == 1
    visit = report["visits"][0]
    assert visit["entry_frame"] == 100
    assert visit["screen"] == 0x77
    assert visit["dest_screen"] == 0x78


def test_soft_reset_abandons_open_visit() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x77),
            _play(5, 0x77),
            TimingSnapshot(frame=6, mode=0, level=0, screen=0),
            _play(50, 0x58),
            _scroll(55, 0x58, mode=6, next_screen=0x48),
            _play(70, 0x48),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].screen == 0x58
    assert any(d.reason is DiscontinuityReason.BOOT_OR_MENU for d in timer.discontinuities)


def test_frame_regression_is_discontinuity() -> None:
    """Save-state load / rewind should not invent a hop."""
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(100, 0x77),
            _play(110, 0x77),
            _play(50, 0x78),  # rewound / loaded
            _scroll(55, 0x78, mode=6, next_screen=0x68),
            _play(80, 0x68),
        ]
    )
    assert any(d.reason is DiscontinuityReason.FRAME_REGRESSION for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].screen == 0x78
    assert timer.visits[0].dest_screen == 0x68
    assert timer.visits[0].entry_frame == 50


def test_location_jump_without_transition_is_not_a_timed_hop() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x77),
            _play(10, 0x77),
            # Instant location change while settled (state load / warp).
            _play(11, 0x37),
            _scroll(20, 0x37, mode=6),
            _play(40, 0x38),
        ]
    )
    assert any(d.reason is DiscontinuityReason.LOCATION_JUMP for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].screen == 0x37
    assert timer.visits[0].dest_screen == 0x38


def test_same_location_return_after_scroll_cancels_leave() -> None:
    """Bounce / failed transition: do not complete a hop."""
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x77),
            _scroll(5, 0x77, mode=6),
            _play(20, 0x77),  # same location settle
            _scroll(30, 0x77, mode=6, next_screen=0x78),
            _play(50, 0x78),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].entry_frame == 0  # original entry preserved
    assert timer.visits[0].leave_frame == 30
    assert timer.visits[0].dest_screen == 0x78


def test_cave_enter_exit_same_screen_is_not_a_hop() -> None:
    """Sword cave: leave OW → mode 16/11 → return same screen."""
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x77),
            TimingSnapshot(frame=5, mode=16, level=0, screen=0x77),  # cave enter
            TimingSnapshot(frame=20, mode=CAVE_MODE, level=0, screen=0x80),
            TimingSnapshot(frame=100, mode=16, level=0, screen=0x77),
            _play(120, 0x77, sword=1),
            _scroll(130, 0x77, mode=6, next_screen=0x78),
            _play(150, 0x78, sword=1),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].screen == 0x77
    assert timer.visits[0].dest_screen == 0x78
    assert timer.visits[0].entry_frame == 0
    assert timer.visits[0].leave_frame == 130
    assert timer.visits[0].sword == 1


def test_hit_freeze_does_not_mark_leave() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x77),
            TimingSnapshot(frame=5, mode=8, level=0, screen=0x77),  # hit freeze
            _play(20, 0x77),
            _scroll(25, 0x77, mode=6, next_screen=0x78),
            _play(40, 0x78),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].leave_frame == 25
    assert timer.visits[0].dwell_frames == 25


def test_death_abandons_open_visit() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _play(0, 0x73, level=1),
            TimingSnapshot(frame=10, mode=17, level=1, screen=0x73),
            _play(100, 0x73, level=1),
            _scroll(105, 0x73, level=1, mode=6, next_screen=0x74),
            _play(120, 0x74, level=1),
        ]
    )
    assert any(d.reason is DiscontinuityReason.DEATH for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].screen == 0x73
    assert timer.visits[0].dest_screen == 0x74
    assert timer.visits[0].entry_frame == 100


def test_triforce_fanfare_then_overworld_return() -> None:
    """Mode 18 marks leave; settling on OW completes dungeon→OW hop."""
    samples = [
        _play(0, 0x36, level=1, triforce=0),
        TimingSnapshot(frame=10, mode=18, level=1, screen=0x36, triforce=1),
        TimingSnapshot(frame=100, mode=2, level=0, screen=0x37, triforce=1),
        _play(200, 0x37, level=0, triforce=1),
    ]
    visits = RoomTimer().observe_many(samples)
    assert len(visits) == 1
    assert visits[0].context is GameContext.DUNGEON
    assert visits[0].dest_context is GameContext.OVERWORLD
    assert visits[0].screen == 0x36
    assert visits[0].dest_screen == 0x37
    assert visits[0].mode_at_leave == 18
    assert visits[0].leave_frame == 10
    assert visits[0].exit_frame == 200


def test_report_schema_and_totals() -> None:
    samples = [
        _play(0, 0x77),
        _scroll(10, 0x77, mode=6, next_screen=0x78),
        _play(30, 0x78),
        _scroll(40, 0x78, mode=6, next_screen=0x68),
        _play(55, 0x68),
    ]
    report = run_offline(samples, source="totals")
    assert report["schema_version"] == 1
    assert report["kind"] == "zelda_i_screen_room_timing"
    assert report["timing_unit"] == "emulator_frames"
    assert report["timing_semantics"]["official_igt_lag"] is False
    assert report["visit_count"] == 2
    assert report["total_location_frames"] == 30 + 25
    assert report["total_dwell_frames"] == 10 + 10
    assert report["total_transition_frames"] == 20 + 15
    assert report["open_visit"] is None  # finalized


def test_snapshots_from_json_fixture_shape() -> None:
    payload = {
        "samples": [
            {"frame": 0, "mode": 5, "level": 0, "screen": 0x77},
            {
                "frame": 5,
                "mode": 6,
                "level": 0,
                "screen": 0x77,
                "next_screen": 0x78,
            },
            {"frame": 20, "mode": 5, "level": 0, "screen": 0x78},
        ]
    }
    samples = snapshots_from_json(payload)
    report = run_offline(samples, source="json_fixture")
    assert report["visit_count"] == 1
    assert report["visits"][0]["screen_hex"] == "0x77"
    assert report["visits"][0]["dest_screen_hex"] == "0x78"


def test_from_zelda_snapshot_roundtrip() -> None:
    snap = ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=0x77,
        next_screen=0,
        link_x=120,
        link_y=141,
        facing=1,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,
        triforce=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    timer = RoomTimer()
    assert timer.observe(snap, frame=42) is None
    open_visit = timer.report()["open_visit"]
    assert open_visit is not None
    assert open_visit["screen"] == 0x77
    assert open_visit["entry_frame"] == 42
    assert open_visit["context"] == "overworld"


def test_screen_zero_is_valid_overworld_location() -> None:
    """Unlike Super Metroid room_id!=0, overworld screen 0 is real."""
    samples = [
        _play(0, 0x00),
        _scroll(5, 0x00, mode=6, next_screen=0x01),
        _play(20, 0x01),
    ]
    visits = RoomTimer().observe_many(samples)
    assert len(visits) == 1
    assert visits[0].screen == 0
    assert visits[0].dest_screen == 1
