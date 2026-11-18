"""Room timer on a synthetic module sequence. No ROM."""

from __future__ import annotations

from alttp.ram import AlttpSnapshot
from alttp.room_timer import (
    DiscontinuityReason,
    RoomTimer,
    TimingSnapshot,
    is_settled_play,
    location_key,
    transition_kind,
)


def _span(start: int, end: int, **fields: object) -> list[TimingSnapshot]:
    base = dict(
        module=0x07,
        submodule=0,
        indoors=False,
        room=0,
        dark_world=False,
        screen=0,
    )
    base.update(fields)
    return [TimingSnapshot(frame=frame, **base) for frame in range(start, end + 1)]


def test_location_key_splits_indoors_from_overworld() -> None:
    assert location_key(indoors=True, room=0x104, dark_world=False, screen=0x2C) == (
        "in",
        0x104,
    )
    light = location_key(indoors=False, room=0x104, dark_world=False, screen=0x2C)
    dark = location_key(indoors=False, room=0x104, dark_world=True, screen=0x2C)
    assert light == ("ow", False, 0x2C)
    assert dark == ("ow", True, 0x2C)
    assert type(light[1]) is bool
    assert light != dark


def test_synthetic_sequence_visits() -> None:
    # House play, dialog bounce, exit to light world, screen hop, dungeon,
    # stair bounce, then dark world. Completed hops only; the last screen stays open.
    samples: list[TimingSnapshot] = []
    samples += _span(1, 2, module=0x00, submodule=0)
    samples += _span(3, 4, module=0x01, submodule=0)
    samples += _span(10, 14, module=0x07, indoors=True, room=0x104)
    samples += _span(15, 16, module=0x0E, submodule=0x02, indoors=True, room=0x104)
    samples += _span(17, 19, module=0x07, indoors=True, room=0x104)
    samples += _span(20, 22, module=0x08, indoors=False, room=0x104, screen=0x2C)
    samples += _span(23, 27, module=0x09, indoors=False, room=0x104, screen=0x2C)
    samples += _span(28, 30, module=0x09, submodule=0x05, indoors=False, screen=0x1B)
    samples += _span(31, 34, module=0x09, indoors=False, screen=0x1B)
    samples += _span(35, 36, module=0x11, indoors=True, room=0x55)
    samples += _span(37, 40, module=0x07, indoors=True, room=0x55)
    samples += _span(41, 42, module=0x07, submodule=0x10, indoors=True, room=0x55)
    samples += _span(43, 44, module=0x07, indoors=True, room=0x55)
    samples += _span(45, 46, module=0x08, indoors=False, dark_world=True, screen=0x1B)
    samples += _span(47, 49, module=0x09, indoors=False, dark_world=True, screen=0x1B)

    timer = RoomTimer()
    timer.observe_many(samples)
    assert timer.discontinuities == []
    assert [(visit.location, visit.dest) for visit in timer.visits] == [
        (("in", 0x104), ("ow", False, 0x2C)),
        (("ow", False, 0x2C), ("ow", False, 0x1B)),
        (("ow", False, 0x1B), ("in", 0x55)),
        (("in", 0x55), ("ow", True, 0x1B)),
    ]
    house = timer.visits[0]
    assert (house.entry_frame, house.leave_frame, house.exit_frame) == (10, 20, 23)
    assert (house.dwell_frames, house.transition_frames, house.location_frames) == (
        10,
        3,
        13,
    )
    assert house.module_at_leave == 0x08
    screen = timer.visits[1]
    assert (screen.entry_frame, screen.leave_frame, screen.exit_frame) == (23, 28, 31)
    assert timer.open_location == ("ow", True, 0x1B)

    before = len(timer.visits)
    timer.finalize(frame=49)
    assert len(timer.visits) == before
    assert timer.discontinuities[-1].reason == DiscontinuityReason.SESSION_END


def test_settled_room_change_without_transition_is_not_a_visit() -> None:
    timer = RoomTimer()
    timer.observe_many(
        _span(1, 3, module=0x07, indoors=True, room=0x61)
        + _span(4, 6, module=0x07, indoors=True, room=0x50)
    )
    assert timer.visits == []
    assert len(timer.discontinuities) == 1
    assert timer.discontinuities[0].reason == DiscontinuityReason.LOCATION_JUMP
    assert timer.discontinuities[0].location == ("in", 0x61)
    assert timer.open_location == ("in", 0x50)


def test_snapshot_adapter_uses_full_room_and_play_submodule() -> None:
    house = AlttpSnapshot(
        game_mode=0x07,
        submodule=0,
        room_id=0x104,
        indoors=True,
        screen_id=0x2C,
        link_x=0,
        link_y=0,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
    )
    snap = TimingSnapshot.from_snapshot(house, frame=5)
    assert snap.location_key == ("in", 0x104)
    assert is_settled_play(snap) is True

    stairs = TimingSnapshot.from_snapshot(
        AlttpSnapshot(
            game_mode=0x07,
            submodule=0x10,
            room_id=0x55,
            indoors=True,
            screen_id=0,
            link_x=0,
            link_y=0,
            link_direction=0,
            link_action=0,
            camera_x=0,
            camera_y=0,
            dark_world=False,
        ),
        frame=6,
    )
    assert is_settled_play(stairs) is False
    assert is_settled_play(
        TimingSnapshot(
            frame=7, module=0x09, submodule=0, indoors=False, room=0x104, screen=0x1B
        )
    )


def _play(**fields: object) -> TimingSnapshot:
    base = dict(
        frame=1,
        module=0x07,
        submodule=0,
        indoors=True,
        room=0x61,
        floor=0,
        layer=0,
    )
    base.update(fields)
    return TimingSnapshot(**base)  # type: ignore[arg-type]


def test_transition_kind_on_synthetic_pairs() -> None:
    assert (
        transition_kind(_play(room=0x61), _play(frame=2, room=0x60, submodule=0x02))
        == "door"
    )
    assert (
        transition_kind(
            _play(room=0x72, floor=0xFF),
            _play(frame=2, room=0x82, floor=0xFF, submodule=0x12),
        )
        == "stairs"
    )
    # Climb out of the well. Same spiral submodule as the drop, opposite rooms.
    assert (
        transition_kind(
            _play(room=0x72, floor=0xFF, layer=0),
            _play(frame=2, room=0x01, floor=0xFF, layer=0, submodule=0x0E),
        )
        == "stairs"
    )
    assert (
        transition_kind(_play(room=0x72, layer=1), _play(frame=2, room=0x72, layer=0))
        == "layer_swap"
    )
    assert (
        transition_kind(_play(room=0x77), _play(frame=2, room=0x87, submodule=0x07))
        == "fall"
    )
    assert (
        transition_kind(_play(room=0x0C), _play(frame=2, room=0xA0, submodule=0x15))
        == "warp"
    )
    # Castle well. Room id flips on spiral 0x0E before the floor byte.
    assert (
        transition_kind(
            _play(room=0x01, floor=0, layer=1),
            _play(frame=2, room=0x72, floor=0, layer=1, submodule=0x0E),
        )
        == "fall"
    )
    assert (
        transition_kind(
            _play(room=0x01, floor=0, layer=1),
            _play(frame=3, room=0x72, floor=0xFF, layer=0, submodule=0),
        )
        == "fall"
    )


def test_room_timer_tags_well_drop_as_fall() -> None:
    samples: list[TimingSnapshot] = []
    samples += _span(1, 3, indoors=True, room=0x01, floor=0, layer=1)
    samples += _span(
        4, 6, indoors=True, room=0x72, floor=0, layer=1, submodule=0x0E
    )
    samples += _span(
        7, 8, indoors=True, room=0x72, floor=0xFF, layer=0, submodule=0x0E
    )
    samples += _span(9, 11, indoors=True, room=0x72, floor=0xFF, layer=0)
    timer = RoomTimer()
    timer.observe_many(samples)
    assert timer.discontinuities == []
    assert len(timer.visits) == 1
    drop = timer.visits[0]
    assert drop.location == ("in", 0x01)
    assert drop.dest == ("in", 0x72)
    assert drop.transition_kind == "fall"
    assert drop.submodule_at_leave == 0x0E
    assert timer.last_kind == "fall"
