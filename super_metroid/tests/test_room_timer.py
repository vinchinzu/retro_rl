"""Offline unit tests for Super Metroid room timing (no ROM)."""

from __future__ import annotations

from super_metroid.ram import GameplayPhase
from super_metroid.room_timer import (
    DiscontinuityReason,
    RoomTimer,
    TimingSnapshot,
    is_settled_ordinary,
    rank_visits,
    run_offline,
    snapshots_from_json,
)


def _ordinary(frame: int, room: int, *, area: int = 1, items: int = 0) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        room_id=room,
        area_index=area,
        game_state=8,
        door_transition=0,
        collected_items=items,
    )


def _transition(
    frame: int,
    room: int,
    *,
    game_state: int = 9,
    door: int = 1,
    direction: int = 1,
) -> TimingSnapshot:
    return TimingSnapshot(
        frame=frame,
        room_id=room,
        area_index=1,
        game_state=game_state,
        door_transition=door,
        transition_direction=direction,
    )


def test_is_settled_ordinary_requires_clean_door_flag() -> None:
    assert is_settled_ordinary(_ordinary(0, 0x9AD9))
    assert not is_settled_ordinary(
        TimingSnapshot(frame=0, room_id=0x9AD9, game_state=8, door_transition=1)
    )
    assert not is_settled_ordinary(
        TimingSnapshot(frame=0, room_id=0, game_state=8, door_transition=0)
    )
    assert not is_settled_ordinary(_transition(0, 0x9AD9))


def test_confirmed_door_transition_records_frame_timings() -> None:
    """Classic hop: settle A → door → settle B."""
    room_a, room_b = 0x9AD9, 0x9B5B
    samples = [
        _ordinary(0, room_a, items=0x1004),
        _ordinary(20, room_a, items=0x1004),
        _transition(21, room_a, game_state=9, door=2, direction=2),
        _transition(35, room_a, game_state=11, door=2, direction=2),
        # Room id may flip mid-load before settle.
        _transition(45, room_b, game_state=11, door=1, direction=2),
        _ordinary(60, room_b, items=0x1004),
    ]
    timer = RoomTimer()
    completed = timer.observe_many(samples)

    assert len(completed) == 1
    visit = completed[0]
    assert visit.source_room_id == 0
    assert visit.room_id == room_a
    assert visit.dest_room_id == room_b
    assert visit.entry_frame == 0
    assert visit.leave_frame == 21
    assert visit.exit_frame == 60
    assert visit.room_frames == 60
    assert visit.dwell_frames == 21
    assert visit.transition_frames == 39
    assert visit.transition_direction == 2
    assert visit.door_transition_at_leave == 2
    assert visit.collected_items == 0x1004
    assert visit.sequence_index == 0


def test_chain_of_two_hops() -> None:
    a, b, c = 0x91F8, 0x92FD, 0x96BA
    samples = [
        _ordinary(0, a),
        _transition(5, a),
        _ordinary(15, b),
        _ordinary(18, b),
        _transition(20, b, direction=3),
        _ordinary(40, c),
    ]
    timer = RoomTimer()
    visits = timer.observe_many(samples)

    assert len(visits) == 2
    assert visits[0].room_id == a and visits[0].dest_room_id == b
    assert visits[0].source_room_id == 0
    assert visits[1].room_id == b and visits[1].dest_room_id == c
    assert visits[1].source_room_id == a
    assert visits[1].entry_frame == 15
    assert visits[1].leave_frame == 20
    assert visits[1].exit_frame == 40
    assert visits[1].dwell_frames == 5
    assert visits[1].transition_frames == 20
    assert visits[1].room_frames == 25
    assert visits[1].sequence_index == 1


def test_boot_and_menu_ignored_until_settled() -> None:
    samples = [
        TimingSnapshot(frame=0, room_id=0, game_state=1, phase=GameplayPhase.BOOT_OR_MENU),
        TimingSnapshot(frame=10, room_id=0, game_state=5, phase=GameplayPhase.BOOT_OR_MENU),
        _ordinary(100, 0x91F8),
        _transition(110, 0x91F8),
        _ordinary(130, 0x92FD),
    ]
    report = run_offline(samples, source="boot_prefix")
    assert report["visit_count"] == 1
    visit = report["visits"][0]
    assert visit["entry_frame"] == 100
    assert visit["room_id"] == 0x91F8
    assert visit["dest_room_id"] == 0x92FD


def test_soft_reset_abandons_open_visit() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _ordinary(0, 0x9AD9),
            _ordinary(5, 0x9AD9),
            TimingSnapshot(frame=6, room_id=0, game_state=1, phase=GameplayPhase.BOOT_OR_MENU),
            _ordinary(50, 0x91F8),
            _transition(55, 0x91F8),
            _ordinary(70, 0x92FD),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].room_id == 0x91F8
    assert any(d.reason is DiscontinuityReason.BOOT_OR_MENU for d in timer.discontinuities)


def test_frame_regression_is_discontinuity() -> None:
    """Save-state load / rewind should not invent a hop."""
    timer = RoomTimer()
    timer.observe_many(
        [
            _ordinary(100, 0x9AD9),
            _ordinary(110, 0x9AD9),
            _ordinary(50, 0x9B5B),  # rewound / loaded
            _transition(55, 0x9B5B),
            _ordinary(80, 0x9AD9),
        ]
    )
    assert any(d.reason is DiscontinuityReason.FRAME_REGRESSION for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].room_id == 0x9B5B
    assert timer.visits[0].dest_room_id == 0x9AD9
    assert timer.visits[0].entry_frame == 50


def test_room_jump_without_transition_phase_is_not_a_timed_hop() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _ordinary(0, 0x9AD9),
            _ordinary(10, 0x9AD9),
            # Instant room change while ordinary (door-warp / load).
            _ordinary(11, 0xA011),
            _transition(20, 0xA011),
            _ordinary(40, 0xA07B),
        ]
    )
    assert any(d.reason is DiscontinuityReason.ROOM_JUMP for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].room_id == 0xA011
    assert timer.visits[0].dest_room_id == 0xA07B


def test_same_room_return_after_door_cancels_leave() -> None:
    """Bounce back into the same room: do not complete a hop."""
    timer = RoomTimer()
    room = 0x9AD9
    timer.observe_many(
        [
            _ordinary(0, room),
            _transition(5, room),
            _ordinary(20, room),  # failed / same room settle
            _transition(30, room),
            _ordinary(50, 0x9B5B),
        ]
    )
    assert len(timer.visits) == 1
    assert timer.visits[0].entry_frame == 0  # original entry preserved
    assert timer.visits[0].leave_frame == 30
    assert timer.visits[0].dest_room_id == 0x9B5B


def test_pause_does_not_complete_transition() -> None:
    timer = RoomTimer()
    room = 0x9AD9
    timer.observe_many(
        [
            _ordinary(0, room),
            TimingSnapshot(
                frame=5,
                room_id=room,
                game_state=15,
                phase=GameplayPhase.PAUSE_OR_INVENTORY,
            ),
            _ordinary(20, room),
            _transition(25, room),
            _ordinary(40, 0x9B5B),
        ]
    )
    assert len(timer.visits) == 1
    # Pause counts toward dwell until the real door leave at 25.
    assert timer.visits[0].leave_frame == 25
    assert timer.visits[0].dwell_frames == 25


def test_death_abandons_open_visit() -> None:
    timer = RoomTimer()
    timer.observe_many(
        [
            _ordinary(0, 0x9AD9),
            TimingSnapshot(
                frame=10,
                room_id=0x9AD9,
                game_state=19,
                phase=GameplayPhase.DEATH_OR_GAME_OVER,
            ),
            _ordinary(100, 0x91F8),
            _transition(105, 0x91F8),
            _ordinary(120, 0x92FD),
        ]
    )
    assert any(d.reason is DiscontinuityReason.DEATH_OR_GAME_OVER for d in timer.discontinuities)
    assert len(timer.visits) == 1
    assert timer.visits[0].room_id == 0x91F8


def test_report_schema_and_totals() -> None:
    samples = [
        _ordinary(0, 0x9AD9),
        _transition(10, 0x9AD9),
        _ordinary(30, 0x9B5B),
        _transition(40, 0x9B5B),
        _ordinary(55, 0x9B9D),
    ]
    report = run_offline(samples, source="totals")
    assert report["schema_version"] == 1
    assert report["kind"] == "super_metroid_room_timing"
    assert report["timing_unit"] == "emulator_frames"
    assert report["timing_semantics"]["practice_hack_igt_lag"] is False
    assert report["visit_count"] == 2
    assert report["total_room_frames"] == 30 + 25
    assert report["total_dwell_frames"] == 10 + 10
    assert report["total_transition_frames"] == 20 + 15
    assert report["open_visit"] is None  # finalized


def test_snapshots_from_json_fixture_shape() -> None:
    payload = {
        "samples": [
            {
                "frame": 0,
                "room_id": 0x9AD9,
                "game_state": 8,
                "door_transition": 0,
            },
            {
                "frame": 5,
                "room_id": 0x9AD9,
                "game_state": 9,
                "door_transition": 1,
                "transition_direction": 1,
            },
            {
                "frame": 20,
                "room_id": 0x9B5B,
                "game_state": 8,
                "door_transition": 0,
                "area_index": 1,
            },
        ]
    }
    samples = snapshots_from_json(payload)
    report = run_offline(samples, source="json_fixture")
    assert report["visit_count"] == 1
    assert report["visits"][0]["room_id_hex"] == "0x9AD9"
    assert report["visits"][0]["dest_room_id_hex"] == "0x9B5B"


def test_game_state_8_with_door_flag_is_not_settled() -> None:
    """Matches phase_for_game_state: door_transition forces ROOM_TRANSITION."""
    snap = TimingSnapshot(frame=0, room_id=0x9AD9, game_state=8, door_transition=1)
    assert snap.resolved_phase() is GameplayPhase.ROOM_TRANSITION
    assert not is_settled_ordinary(snap)


def test_from_state_roundtrip_fields() -> None:
    from super_metroid.ram import SuperMetroidState

    state = SuperMetroidState(
        frame=42,
        game_state=8,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=0x9AD9,
        area_index=1,
        door_transition=0,
        transition_direction=0,
        samus_x=0,
        samus_y=0,
        velocity_x=0,
        velocity_y=0,
        pose=0,
        health=99,
        max_health=99,
        reserve_health=0,
        max_reserve_health=0,
        missiles=0,
        max_missiles=0,
        super_missiles=0,
        max_super_missiles=0,
        power_bombs=0,
        max_power_bombs=0,
        selected_item=0,
        equipped_items=0,
        collected_items=4,
        equipped_beams=0,
        collected_beams=0,
        timer_type=0,
        escape_timer_frames=0,
        escape_timer_seconds=0,
        escape_timer_minutes=0,
        num_enemies=0,
        enemies_killed=0,
        enemy0_x=0,
        enemy0_y=0,
        enemy0_hp=0,
        enemy0_spritemap=0,
        event_flags=(0,) * 8,
        boss_bits=(0,) * 8,
    )
    snap = TimingSnapshot.from_state(state)
    assert snap.frame == 42
    assert snap.room_id == 0x9AD9
    assert snap.collected_items == 4
    timer = RoomTimer()
    assert timer.observe(state) is None
    open_visit = timer.report()["open_visit"]
    assert open_visit is not None
    assert open_visit["room_id"] == 0x9AD9
    assert open_visit["entry_frame"] == 42


def test_rank_visits_orders_by_room_frames() -> None:
    samples = [
        _ordinary(0, 0x91F8),
        _transition(5, 0x91F8),
        _ordinary(20, 0x92FD),  # hop 0: 20 room frames
        _transition(30, 0x92FD),
        _ordinary(100, 0x96BA),  # hop 1: 80 room frames (slowest)
        _transition(105, 0x96BA),
        _ordinary(130, 0x9879),  # hop 2: 30 room frames
    ]
    report = run_offline(samples, source="rank")
    ranked = rank_visits(report["visits"], key="room_frames", limit=2)
    assert len(ranked) == 2
    assert ranked[0]["room_id"] == 0x92FD
    assert ranked[0]["room_frames"] == 80
    assert ranked[1]["room_id"] == 0x96BA
    assert ranked[1]["room_frames"] == 30
