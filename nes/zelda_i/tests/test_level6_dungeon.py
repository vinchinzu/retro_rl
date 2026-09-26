"""Unit tests for Level 6 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon.ids import (
    KEESE_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6.dungeon import (
    LEVEL6_COMPASS_BIT,
    ROOM_29_SPEC,
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
    ROOM_L6_COMPASS,
    ROOM_L6_DARK_29,
    ROOM_L6_EAST_KEY,
    ROOM_L6_ENTRY,
    ROOM_L6_HARD_38,
    ROOM_L6_KEESE,
    ROOM_L6_MAP,
    ROOM_L6_ROD_WIZZ,
    ROOM_L6_WEST_WIZZROBE,
    ROOM_L6_WIZZROBE_28,
    level6_room_09_clear_success,
    level6_room_19_clear_success,
    level6_room_28_clear_success,
    level6_room_38_clear_success,
    level6_room_58_clear_success,
    level6_room_68_compass_success,
    level6_room_78_clear_success,
    level6_room_7a_key_success,
)
from zelda_i.level6.overworld import (
    Level6WestKeyDoorController,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import (
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "screen": ROOM_L6_ENTRY,
    "x": 120,
    "y": 205,
    "keys": 0,
}


def _ram(
    *,
    level: int = 6,
    room: int = ROOM_L6_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    wizzrobes: int = 0,
    hp: int = 64,
    **fields: int,
) -> np.ndarray:
    ram = make_ram(
        _DEFAULTS,
        level=level,
        screen=room,
        x=x,
        y=y,
        mode=mode,
        keys=keys,
        **fields,
    )
    for slot in range(1, wizzrobes + 1):
        ram[ADDR_OBJ_TYPE + slot] = WIZZROBE_ORANGE_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    return ram


def test_east_key_hunts_center_key_when_cleared() -> None:
    """Clean leftover (118,141): key on the floor, 2px west of pickup."""
    from zelda_i.level6.wizzrobe import make_east_key_controller

    ram = _ram(room=ROOM_L6_EAST_KEY, x=118, y=141, wizzrobes=0)
    ctl = make_east_key_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_key"
    assert list(action.action) == list(nes_action("RIGHT"))


def test_east_key_reason_histogram_records_without_steering() -> None:
    """The trace must be records-only: same action, plus a reason count.

    rr-d6v lost a trial to a 12000-frame stage whose only leftover was a
    tile. The histogram names the rule that held the frames.
    """
    from zelda_i.level6.wizzrobe import make_east_key_controller

    ram = _ram(room=ROOM_L6_EAST_KEY, x=118, y=141, wizzrobes=0)
    plain = make_east_key_controller()
    traced = make_east_key_controller()
    snap = read_snapshot(ram)
    assert list(traced.step(snap).action) == list(plain.step(snap).action)
    report = traced.report()
    assert report["reason_counts"] == {"wizzrobe_key": 1}
    assert report["tail"] and "(118,141)" in report["tail"][0]


def test_east_key_leaves_west_block_pocket() -> None:
    """Clean leftover (64,117): RIGHT off the (64,112) block, not UP/DOWN."""
    from zelda_i.level6.wizzrobe import make_east_key_controller

    ram = _ram(room=ROOM_L6_EAST_KEY, x=64, y=117, wizzrobes=1)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ram[ADDR_LINK_X + 1] = 160
    ram[ADDR_LINK_Y + 1] = 141
    ctl = make_east_key_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_leave_block"
    assert list(action.action) == list(nes_action("RIGHT"))
    assert (64, 109) not in ROOM_7A_SPEC.combat.patrol
    assert (80, 109) in ROOM_7A_SPEC.combat.patrol


def test_east_key_0x24_e_contact_pose_is_not_dodgeable() -> None:
    """Census f460: Link (88,133) vs parked 0x24 (96,125), d=8, ttc=0."""
    from zelda_i.dungeon.threat import MIN_DODGE_BODY, assess, dodgeable
    from zelda_i.dungeon.tracking import ObjectTracker

    ram = _ram(room=ROOM_L6_EAST_KEY, x=88, y=133, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 96
    ram[ADDR_LINK_Y + 1] = 125
    ram[ADDR_LINK_FACING + 1] = 0x02
    snap = read_snapshot(ram)
    tracked = ObjectTracker().observe(snap)
    impact = assess((88, 133), tracked)
    assert impact.frames == 0
    assert dodgeable(impact) is False
    assert impact.frames < MIN_DODGE_BODY


def test_east_key_engages_undodgeable_0x24_pose() -> None:
    """Measured pose: still engage. In-place slash / south-stand died in 0x7a."""
    from zelda_i.level6.wizzrobe import make_east_key_controller

    ram = _ram(room=ROOM_L6_EAST_KEY, x=88, y=133, wizzrobes=1, facing=0x01)
    ram[ADDR_LINK_X + 1] = 96
    ram[ADDR_LINK_Y + 1] = 125
    ram[ADDR_LINK_FACING + 1] = 0x02
    ctl = make_east_key_controller()
    action = ctl.step(read_snapshot(ram))
    assert "engage" in action.reason
    assert list(action.action) == list(nes_action("RIGHT"))


@pytest.mark.parametrize("x,y", [(189, 141), (192, 140)])
def test_west_east_mouth_inland_dash_left(x: int, y: int) -> None:
    """v4/v5 leftover: live 0x24 on the east lip → LEFT inland, not engage."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    assert ROOM_78_SPEC.combat.inland_dash > 0
    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=x, y=y, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 80
    ram[ADDR_LINK_Y + 1] = 141
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_inland_dash"
    assert list(action.action) == list(nes_action("LEFT"))
    assert "engage" not in action.reason
    objs = ctl.report()["objects"]
    assert any(int(o["type"]) == WIZZROBE_ORANGE_TYPE for o in objs)


def test_west_south_commit_sidestep_not_up() -> None:
    """v8 leftover (160,149): 0x24 same x + 0x59 y=141 → LEFT/RIGHT, not UP."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=160, y=149, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 160
    ram[ADDR_LINK_Y + 1] = 141
    ram[ADDR_OBJ_TYPE + 2] = 0x59
    ram[ADDR_LINK_X + 2] = 165
    ram[ADDR_LINK_Y + 2] = 141
    ram[ADDR_OBJ_HP + 2] = 128
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_sidestep"
    assert list(action.action) in (
        list(nes_action("LEFT")),
        list(nes_action("RIGHT")),
    )
    assert list(action.action) != list(nes_action("UP"))


def test_west_beam_59_keeps_leaving_y125() -> None:
    """v7 leftover (120,125) + 0x59@(133,125) → DOWN off that band, not engage."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=120, y=125, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 160
    ram[ADDR_LINK_Y + 1] = 125
    ram[ADDR_OBJ_TYPE + 2] = 0x59
    ram[ADDR_LINK_X + 2] = 133
    ram[ADDR_LINK_Y + 2] = 125
    ram[ADDR_OBJ_HP + 2] = 128
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_beam_peel"
    assert list(action.action) == list(nes_action("DOWN"))
    assert "engage" not in action.reason
    assert list(action.action) != list(nes_action("LEFT"))


def test_west_beam_59_peel_off_waist_not_left() -> None:
    """v6 leftover (176,141) + 0x59@(187,141) → UP/DOWN, never LEFT."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=176, y=141, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 208
    ram[ADDR_LINK_Y + 1] = 141
    ram[ADDR_OBJ_TYPE + 2] = 0x59
    ram[ADDR_LINK_X + 2] = 187
    ram[ADDR_LINK_Y + 2] = 141
    ram[ADDR_OBJ_HP + 2] = 128
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_beam_peel"
    assert list(action.action) in (
        list(nes_action("UP")),
        list(nes_action("DOWN")),
    )
    assert list(action.action) != list(nes_action("LEFT"))
    assert list(action.action) != list(nes_action("RIGHT"))


def test_west_waist_dodge_off_axis_not_left() -> None:
    """Inland of the east lip: inbound y-band shot → UP/DOWN, not LEFT."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=160, y=141, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 80
    ram[ADDR_LINK_Y + 1] = 141
    ram[ADDR_OBJ_TYPE + 2] = 0x55
    ram[ADDR_LINK_X + 2] = 120
    ram[ADDR_LINK_Y + 2] = 141
    ram[ADDR_LINK_FACING + 2] = 0x01
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_waist_dodge"
    assert list(action.action) in (
        list(nes_action("UP")),
        list(nes_action("DOWN")),
    )
    assert list(action.action) != list(nes_action("LEFT"))
    assert list(action.action) != list(nes_action("RIGHT"))


def test_west_block_pocket_still_right_with_waist_shot() -> None:
    """West block pocket: RIGHT even with an inbound y-band projectile."""
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller

    ram = _ram(room=ROOM_L6_WEST_WIZZROBE, x=64, y=117, wizzrobes=1)
    ram[ADDR_LINK_X + 1] = 160
    ram[ADDR_LINK_Y + 1] = 141
    ram[ADDR_OBJ_TYPE + 2] = 0x55
    ram[ADDR_LINK_X + 2] = 80
    ram[ADDR_LINK_Y + 2] = 117
    ram[ADDR_LINK_FACING + 2] = 0x02
    ctl = make_west_wizzrobe_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wizzrobe_leave_block"
    assert list(action.action) == list(nes_action("RIGHT"))


def test_room_7a_reward_waypoints_recover_from_blocked_leftover() -> None:
    """Regression (live 2026-09-04): ``ROOM_7A_SPEC.reward`` had a plain
    ``target=(136, 141)`` with no waypoints. Live power-on recon showed the
    combat backstep policy can leave Link at (64, 93), with a cart-WRAM
    tilemap block cell at (64, 112) directly south — the no-waypoints
    ``_collect_reward`` branch has no stuck-escape, so it pressed DOWN into
    the block for the full 12,000-frame room timeout and the key was never
    collected. Waypoints (reused from the combat patrol ring, ending at the
    verified pickup spot (120, 141)) give the existing 24-frame stuck-skip
    a real route. This pins Link at that exact frozen leftover and asserts
    the controller eventually gives up hunting rather than spinning the
    same blocked direction forever.
    """
    assert ROOM_7A_SPEC.reward.waypoints
    assert ROOM_7A_SPEC.reward.target == (120, 141)

    controller = GenericDungeonRoomController(ROOM_7A_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 4
    # Live-measured freeze position: north wall, directly above the (64,112)
    # block. Static x/y (no walk physics here) stands in for "blocked".
    ram = _ram(room=ROOM_L6_EAST_KEY, x=64, y=93, keys=4, room_all_dead=24)
    n = len(ROOM_7A_SPEC.reward.waypoints)
    action = None
    for _ in range(n * 30):
        action = controller.step(read_snapshot(ram))
        if action.reason == "collect_wait":
            break
    assert action is not None
    assert action.reason == "collect_wait"
    assert np.array_equal(action.action, nes_idle_action())
    assert controller._collect_skips >= n


def test_live_wizzrobes_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=64))
    assert len(ROOM_7A_SPEC.live_enemies(snap)) == 5
    snap_dead = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=0))
    assert len(ROOM_7A_SPEC.live_enemies(snap_dead)) == 0
    snap78 = read_snapshot(_ram(room=ROOM_L6_WEST_WIZZROBE, wizzrobes=5, hp=64))
    assert len(ROOM_78_SPEC.live_enemies(snap78)) == 5


@pytest.mark.parametrize(
    "success_fn, room, extra, live_type, live_hp",
    [
        (level6_room_7a_key_success, ROOM_L6_EAST_KEY, {"keys": 1}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_78_clear_success, ROOM_L6_WEST_WIZZROBE, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_68_compass_success, ROOM_L6_COMPASS, {"compass": LEVEL6_COMPASS_BIT}, ZOL_OBJECT_TYPE, 64),
        (level6_room_19_clear_success, ROOM_L6_MAP, {}, ZOL_OBJECT_TYPE, 32),
        (level6_room_09_clear_success, ROOM_L6_ROD_WIZZ, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_58_clear_success, ROOM_L6_KEESE, {}, KEESE_OBJECT_TYPE, 0),
        (level6_room_38_clear_success, ROOM_L6_HARD_38, {}, LIKE_LIKE_OBJECT_TYPE, 64),
        (level6_room_28_clear_success, ROOM_L6_WIZZROBE_28, {}, WIZZROBE_ORANGE_TYPE, 64),
    ],
    ids=["7a", "78", "68", "19", "09", "58", "38", "28"],
)
def test_clear_success(success_fn, room, extra, live_type, live_hp) -> None:
    ram = _ram(room=room, **extra)
    assert success_fn(ram)
    ram[ADDR_OBJ_TYPE + 1] = live_type
    ram[ADDR_OBJ_HP + 1] = live_hp
    assert not success_fn(ram)


def test_west_key_door_controller_from_east_edge() -> None:
    ctl = Level6WestKeyDoorController()
    # East door channel after free return from 0x7a — must LEFT first
    # (vertical blocked at x≈224).
    snap = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=224, y=141, keys=1)
    )
    act = ctl.step(snap)
    assert act.reason == "leave_east_door_channel"

    # At fire-wall column, adjust y before crossing.
    ctl2 = Level6WestKeyDoorController()
    snap2 = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=208, y=141, keys=1)
    )
    act2 = ctl2.step(snap2)
    assert act2.reason == "east_to_wall_y"

    arrived = Level6WestKeyDoorController()
    arrived.step(read_snapshot(
        _ram(room=ROOM_L6_WEST_WIZZROBE, x=224, y=141, keys=0)
    ))
    assert arrived.success


def _29_ram(*, x: int, y: int, enemy_x: int, enemy_y: int) -> np.ndarray:
    ram = _ram(room=ROOM_L6_DARK_29, x=x, y=y, wizzrobes=1, hp=64)
    ram[ADDR_LINK_X + 1] = enemy_x
    ram[ADDR_LINK_Y + 1] = enemy_y
    return ram


def _clear29_fight(*, x: int, y: int, enemy_x: int, enemy_y: int):
    ctl = GenericDungeonRoomController(ROOM_29_SPEC)
    ctl.phase = DungeonPhase.FIGHT
    ctl.combat_frames = 24
    snap = read_snapshot(_29_ram(x=x, y=y, enemy_x=enemy_x, enemy_y=enemy_y))
    return ctl.step(snap)


def test_clear29_patrol_omits_sw_trap() -> None:
    assert ROOM_29_SPEC.reward.target == (120, 189)
    assert ROOM_29_SPEC.combat.occupancy_bounds == (16, 216, 77, 141)
    blocked = set(ROOM_29_SPEC.combat.occupancy_blocked)
    assert (48, 157) not in ROOM_29_SPEC.combat.patrol
    assert (56, 133) not in ROOM_29_SPEC.combat.patrol
    assert (120, 189) in ROOM_29_SPEC.combat.patrol
    assert (63, 133) in blocked
    assert (63, 132) in blocked
    assert (56, 157) in blocked
    assert (120, 141) not in blocked
    assert (128, 125) in blocked
    assert (120, 189) not in blocked
    assert (80, 109) not in blocked


def test_clear29_downs_inland_from_north_mouth() -> None:
    """Red 3: LEFT at (120,77) is the door channel. DOWN to y=109 first."""
    act = _clear29_fight(x=120, y=77, enemy_x=184, enemy_y=144)
    assert act.reason.startswith("leave_wall")
    assert list(act.action) == list(nes_action("DOWN")) or list(act.action) == list(
        nes_action("DOWN", "A")
    )
    assert list(act.action) != list(nes_action("LEFT"))


@pytest.mark.parametrize(
    "x,y",
    [
        pytest.param(120, 109, id="north_band_is_inland"),
        pytest.param(48, 133, id="may_chase_east_from_west_aisle"),
    ],
)
def test_clear29_does_not_west_peel(x: int, y: int) -> None:
    """y=109 is inland (no LEFT-peel of the door channel); west-only chase at
    (48,133) left two wizzrobes live for 15000f (reds 1-2) — neither is a peel."""
    act = _clear29_fight(x=x, y=y, enemy_x=184, enemy_y=144)
    assert act.reason != "west_peel"
    assert list(act.action) != list(nes_action("LEFT"))


@pytest.mark.parametrize(
    "x,y",
    [
        pytest.param(96, 109, id="clips_right_down_to_waist"),
        pytest.param(104, 131, id="peels_up_from_plus_interior"),
    ],
)
def test_clear29_leftover_clip(x: int, y: int) -> None:
    """Cardinal RIGHT boxed at x=96,y=109; inside the plus north of the waist
    at (104,131) — both take the same RIGHT+DOWN open-axis clip."""
    ctl, act = _cleared_29(x=x, y=y)
    assert not ctl.success
    assert act.reason == "leftover_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))


def test_clear29_leftover_waist_goes_south() -> None:
    ctl, act = _cleared_29(x=120, y=141)
    assert act.reason == "leftover_south"
    assert list(act.action) == list(nes_action("DOWN"))
    ctl, act = _cleared_29(x=176, y=141)
    assert act.reason == "leftover_align"
    assert list(act.action) == list(nes_action("LEFT"))




def test_clear29_does_not_walk_deeper_into_sw_trap() -> None:
    act = _clear29_fight(x=48, y=157, enemy_x=48, enemy_y=173)
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("DOWN", "A"))


def _cleared_29(*, x: int, y: int):
    ctl = GenericDungeonRoomController(ROOM_29_SPEC)
    ctl.phase = DungeonPhase.FIGHT
    ctl.max_live_enemies = 5
    snap = read_snapshot(_ram(room=ROOM_L6_DARK_29, x=x, y=y, wizzrobes=0))
    return ctl, ctl.step(snap)


def test_clear29_rejects_sw_trap_as_success() -> None:
    """Minimized repro: (56,157) with five dead must not be reason=done."""
    ctl, act = _cleared_29(x=56, y=157)
    assert not ctl.success
    assert act.reason != "done"


def test_clear29_rejects_island_face_as_success() -> None:
    """(55,133) / (63,133) are tile 244. Leftover is the south door."""
    for x, y in ((55, 133), (63, 133)):
        ctl, act = _cleared_29(x=x, y=y)
        assert not ctl.success
        assert act.reason != "done"


def test_clear29_accepts_south_door_leftover() -> None:
    ctl, act = _cleared_29(x=120, y=189)
    assert ctl.success
    assert act.reason == "done"


def test_clear29_plus_center_is_not_leftover() -> None:
    """Waist (120,141) is not the spine leftover; south door is."""
    ctl, act = _cleared_29(x=120, y=141)
    assert not ctl.success
    assert act.reason == "leftover_south"


def test_clear29_leftover_walk_from_north_mouth() -> None:
    ctl, act = _cleared_29(x=128, y=93)
    assert not ctl.success
    assert act.reason == "leftover_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))


def test_clear29_plus_seed_paths_around_to_south_door() -> None:
    """l6_south29_door2 boxed at (128,125). Seeded island must still path."""
    from zelda_i.walk.physics import OccupancyGrid

    bounds = ROOM_29_SPEC.combat.occupancy_bounds
    assert bounds is not None
    xmin, xmax, ymin, _ymax = bounds
    grid = OccupancyGrid(
        blocked=set(ROOM_29_SPEC.combat.occupancy_blocked),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=205,
    )
    assert grid.shortest_path((104, 116), (120, 189)) is not None
    assert grid.shortest_path((128, 116), (120, 189)) is not None
    assert grid.shortest_path((120, 77), (120, 189)) is not None
    ctl, act = _cleared_29(x=128, y=125)
    assert not ctl.success
    assert act.reason == "leftover_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
