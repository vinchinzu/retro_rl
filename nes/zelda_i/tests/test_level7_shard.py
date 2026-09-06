"""L7 shard leave: 0x2A -> 0x2B -> idled fanfare -> OW (`level7.shard`, 2/2).

Live evidence (rr-8t4.3, 2026-09-04): `20260904_W3`-`W6`, 943 controller
frames each, shard taken south-around the diamond floor, OW leftover `0x42`
`(96,93)` mode 5 TF `0x40`, `position_writes=0`. That TF is `0x40` and not
`0x7F` because the lineage pin starts at TF 0. The Survival packet is
filled from power-on; this factory still reports
`measured_post_l7_exit_verified=False`.
"""

from __future__ import annotations

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.anchors import TF_BIT_L7
from zelda_i.level7.hops import make_level7_shard_leave_controller
from zelda_i.level7.shard import (
    FANFARE_MODE,
    SHARD_WAYPOINTS,
    SHUTTER_MAX_FRAMES,
    Level7ShardLeaveController,
    ShardLeavePhase,
)
from zelda_i.level7.stairs import AQUAMENTUS_ROM, TRIFORCE_ROM
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 7,
    "screen": AQUAMENTUS_ROM,
    "triforce": 0x00,
}


def _snap(
    *,
    x: int,
    y: int,
    screen: int = AQUAMENTUS_ROM,
    mode: int = PLAY_MODE,
    level: int = 7,
    triforce: int = 0x00,
):
    return read_snapshot(
        make_ram(
            _DEFAULTS,
            x=x,
            y=y,
            screen=screen,
            mode=mode,
            level=level,
            triforce=triforce,
        )
    )


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def test_waypoints_match_the_measured_south_around_walk() -> None:
    assert SHARD_WAYPOINTS == ((32, 141), (32, 189), (120, 189), (128, 141))


def test_hops_factory_returns_the_live_controller() -> None:
    controller = make_level7_shard_leave_controller()
    assert isinstance(controller, Level7ShardLeaveController)
    report = controller.report()
    assert report["route_eligible"] is False
    assert report["measured_post_l7_exit_verified"] is False


def test_east_shutter_is_pushed_right_from_0x2a() -> None:
    controller = Level7ShardLeaveController()
    action = controller.step(_snap(x=120, y=141))
    assert _buttons(action) == ["RIGHT"]
    assert controller.phase is ShardLeavePhase.EAST_SHUTTER


def test_arrival_in_0x2b_starts_the_south_around_walk() -> None:
    controller = Level7ShardLeaveController()
    controller.step(_snap(x=120, y=141))
    controller.step(_snap(x=16, y=141, screen=TRIFORCE_ROM))
    assert controller.phase is ShardLeavePhase.SHARD_WALK
    action = controller.step(_snap(x=16, y=141, screen=TRIFORCE_ROM))
    assert _buttons(action) == ["RIGHT"]


def test_stalled_shutter_fails_closed() -> None:
    controller = Level7ShardLeaveController(shutter_frames=SHUTTER_MAX_FRAMES)
    controller.step(_snap(x=120, y=141))
    assert controller.failed
    assert not controller.success


def test_fanfare_is_idled_not_walked() -> None:
    controller = Level7ShardLeaveController()
    controller.step(_snap(x=120, y=141))
    action = controller.step(
        _snap(x=128, y=141, screen=TRIFORCE_ROM, mode=FANFARE_MODE)
    )
    assert _buttons(action) == []


def test_settled_overworld_after_the_shard_succeeds() -> None:
    controller = Level7ShardLeaveController()
    controller.step(_snap(x=120, y=141))
    controller.step(
        _snap(x=128, y=141, screen=TRIFORCE_ROM, triforce=TF_BIT_L7)
    )
    controller.step(_snap(x=96, y=93, screen=0x42, level=0, triforce=TF_BIT_L7))
    assert controller.success
    assert controller.leftover == {
        "screen": 0x42,
        "link_x": 96,
        "link_y": 93,
        "mode": PLAY_MODE,
        "triforce": TF_BIT_L7,
    }


def test_success_needs_the_rising_edge_of_the_shard_bit() -> None:
    """A pin that already carries TF 0x40 must not green on an OW frame."""
    controller = Level7ShardLeaveController()
    controller.step(_snap(x=120, y=141, triforce=TF_BIT_L7))
    controller.step(_snap(x=96, y=93, screen=0x42, level=0, triforce=TF_BIT_L7))
    assert controller.failed
    assert not controller.success
    assert "overworld_without_shard" in controller.notes


def test_leaving_0x2b_for_another_dungeon_room_fails() -> None:
    controller = Level7ShardLeaveController()
    controller.step(_snap(x=120, y=141))
    controller.step(_snap(x=16, y=141, screen=TRIFORCE_ROM))
    controller.step(_snap(x=208, y=141, screen=AQUAMENTUS_ROM))
    assert controller.failed
