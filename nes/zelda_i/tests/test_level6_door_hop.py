"""Shared L6 dest-hop controller factories and dest-hop RAM walks."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_door_hop import Level6DoorHopController
from zelda_i.level6_east29 import make_east29_controller
from zelda_i.level6_east39 import make_east39_controller
from zelda_i.level6_south09 import make_south09_controller
from zelda_i.level6_south18 import make_south18_controller
from zelda_i.level6_south19 import make_south19_controller
from zelda_i.level6_south1d import make_south1d_controller
from zelda_i.level6_south29 import make_south29_controller
from zelda_i.level6_west19 import make_west19_controller
from zelda_i.level6_north2c import make_north2c_controller
from zelda_i.level6_west2d import make_west2d_controller
from zelda_i.level6_spine import (
    L6_THROUGH,
    level6_east29_stages,
    level6_east29_success,
    level6_east39_stages,
    level6_east39_success,
    level6_south09_stages,
    level6_south09_success,
    level6_south18_stages,
    level6_south18_success,
    level6_south19_stages,
    level6_south19_success,
    level6_south29_stages,
    level6_south29_success,
    level6_west19_stages,
    level6_west19_success,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAP,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x09)
    ram[ADDR_LINK_X] = fields.get("x", 192)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_BOMBS] = fields.get("bombs", 0)
    ram[ADDR_MAP] = fields.get("map", 0)
    return ram


def test_door_hop_factories_return_shared_controller() -> None:
    makers = (
        make_south09_controller,
        make_south19_controller,
        make_south29_controller,
        make_east29_controller,
        make_east39_controller,
        make_west19_controller,
        make_south18_controller,
        make_south1d_controller,
        make_west2d_controller,
        make_north2c_controller,
    )
    ids = (
        "level6_south_0x09",
        "level6_south_0x19",
        "level6_south_0x29",
        "level6_east_0x29",
        "level6_east_0x39",
        "level6_west_0x19",
        "level6_south_0x18",
        "level6_south_0x1d",
        "level6_west_0x2d",
        "level6_north_0x2c",
    )
    for make, spec_id in zip(makers, ids, strict=True):
        ctl = make()
        assert isinstance(ctl, Level6DoorHopController)
        assert ctl.spec_id == spec_id


def test_south09_ram_walk_on_shared_controller() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    leftover = _ram(screen=0x09, x=192, y=141)
    ctl = make_south09_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("DOWN"))
    halt = make_south09_controller()
    act = halt.step(read_snapshot(_ram(screen=0x09, x=192, y=109)))
    assert act.reason == "south_north_halt"
    assert list(act.action) == list(nes_idle_action())
    dest = _ram(screen=0x19, x=120, y=93)
    arrive = make_south09_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_19"


def test_west19_ram_walk_on_shared_controller() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(screen=0x19, x=120, y=205)
    ctl = make_west19_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("UP"))
    dest = _ram(screen=0x18, x=208, y=141)
    arrive = make_west19_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_18"
    key_up = _ram(screen=0x09, x=120, y=205, keys=3)
    north = make_west19_controller()
    north.keys = 4
    act = north.step(read_snapshot(key_up))
    assert north.failed
    assert act.reason.startswith("key_up_09")


def test_level6_south09_occupancy_then_down() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_south09 import (
        SOUTH_DOOR_X,
        SOUTH_DOOR_Y,
        make_south09_controller,
    )
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_south09_stages()
    assert [name for name, _, _ in stages] == ["level6_south_0x09"]
    leftover = _ram(level=6, screen=0x09, x=192, y=141)
    leftover[ADDR_ROD] = 1
    ctl = make_south09_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_path"
    # BFS explores DOWN before LEFT; leftover LEFT is the remaining 0x68.
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    north = _ram(level=6, screen=0x09, x=192, y=109)
    north[ADDR_ROD] = 1
    halt = make_south09_controller()
    act = halt.step(read_snapshot(north))
    assert act.reason == "south_north_halt"
    assert list(act.action) == list(nes_idle_action())
    band = _ram(level=6, screen=0x09, x=192, y=181)
    band[ADDR_ROD] = 1
    align = make_south09_controller()
    act = align.step(read_snapshot(band))
    assert act.reason == "south_align"
    assert list(act.action) == list(nes_action("LEFT"))
    door = _ram(level=6, screen=0x09, x=SOUTH_DOOR_X, y=SOUTH_DOOR_Y)
    door[ADDR_ROD] = 1
    push = make_south09_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "south_push"
    assert list(act.action) == list(nes_action("DOWN"))
    dest = _ram(level=6, screen=0x19, x=120, y=93)
    dest[ADDR_ROD] = 1
    arrive = make_south09_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_19"
    assert level6_south09_success(read_snapshot(dest))
    other = _ram(level=6, screen=0x1A, x=16, y=141)
    other[ADDR_ROD] = 1
    assert level6_south09_success(read_snapshot(other))
    still = _ram(level=6, screen=0x09, x=192, y=141)
    still[ADDR_ROD] = 1
    assert not level6_south09_success(read_snapshot(still))
    cellar = _ram(level=6, screen=0x75, x=136, y=141, mode=9)
    cellar[ADDR_ROD] = 1
    assert not level6_south09_success(read_snapshot(cellar))
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0
    dest[ADDR_BOW] = 1
    dest[ADDR_ARROWS] = 1
    armed = read_snapshot(dest)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-south09", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_south_0x09"
    assert "level6-south09" in L6_THROUGH


def test_level6_south19_occupancy_then_down() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_south19 import (
        SOUTH_DOOR_X,
        SOUTH_DOOR_Y,
        make_south19_controller,
    )
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_south19_stages()
    assert [name for name, _, _ in stages] == ["level6_south_0x19"]
    leftover = _ram(level=6, screen=0x19, x=120, y=77)
    leftover[ADDR_ROD] = 1
    ctl = make_south19_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    door = _ram(level=6, screen=0x19, x=SOUTH_DOOR_X, y=SOUTH_DOOR_Y)
    door[ADDR_ROD] = 1
    push = make_south19_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "south_push"
    assert list(act.action) == list(nes_action("DOWN"))
    dest = _ram(level=6, screen=0x29, x=120, y=77)
    dest[ADDR_ROD] = 1
    arrive = make_south19_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_29"
    assert level6_south19_success(read_snapshot(dest))
    other = _ram(level=6, screen=0x1A, x=16, y=141)
    other[ADDR_ROD] = 1
    assert level6_south19_success(read_snapshot(other))
    still = _ram(level=6, screen=0x19, x=120, y=77)
    still[ADDR_ROD] = 1
    assert not level6_south19_success(read_snapshot(still))
    gleeok = _ram(level=6, screen=0x18, x=208, y=141)
    gleeok[ADDR_ROD] = 1
    assert level6_south19_success(read_snapshot(gleeok))
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0
    dest[ADDR_BOW] = 1
    dest[ADDR_ARROWS] = 1
    armed = read_snapshot(dest)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-south19", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_south_0x19"
    assert "level6-south19" in L6_THROUGH


def test_level6_east29_y_align_then_right() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_east29 import (
        EAST_DOOR_X,
        EAST_DOOR_Y,
        make_east29_controller,
    )
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_east29_stages()
    assert [name for name, _, _ in stages] == ["level6_east_0x29"]
    leftover = _ram(level=6, screen=0x29, x=55, y=133)
    leftover[ADDR_ROD] = 1
    ctl = make_east29_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "east_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))
    door = _ram(level=6, screen=0x29, x=EAST_DOOR_X, y=EAST_DOOR_Y)
    door[ADDR_ROD] = 1
    push = make_east29_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "east_push"
    assert list(act.action) == list(nes_action("RIGHT"))
    dest = _ram(level=6, screen=0x2A, x=16, y=141)
    dest[ADDR_ROD] = 1
    arrive = make_east29_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_2a"
    assert level6_east29_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x29, x=55, y=133)
    still[ADDR_ROD] = 1
    assert not level6_east29_success(read_snapshot(still))
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0
    dest[ADDR_BOW] = 1
    dest[ADDR_ARROWS] = 1
    armed = read_snapshot(dest)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-east29", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_east_0x29"
    assert "level6-east29" in L6_THROUGH


def test_level6_south29_clips_then_down() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_south29 import (
        SOUTH_DOOR_X,
        SOUTH_DOOR_Y,
        make_south29_controller,
    )
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_south29_stages()
    assert [name for name, _, _ in stages] == ["level6_south_0x29"]
    leftover = _ram(level=6, screen=0x29, x=55, y=133)
    leftover[ADDR_ROD] = 1
    ctl = make_south29_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert list(act.action) != list(nes_action("DOWN"))
    mid = _ram(level=6, screen=0x29, x=80, y=141)
    mid[ADDR_ROD] = 1
    path = make_south29_controller()
    act = path.step(read_snapshot(mid))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    band = _ram(level=6, screen=0x29, x=64, y=181)
    band[ADDR_ROD] = 1
    face = make_south29_controller()
    act = face.step(read_snapshot(band))
    assert act.reason == "south_face"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("RIGHT", "DOWN"))
    door = _ram(level=6, screen=0x29, x=SOUTH_DOOR_X, y=SOUTH_DOOR_Y)
    door[ADDR_ROD] = 1
    push = make_south29_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "south_push"
    assert list(act.action) == list(nes_action("DOWN"))
    dest = _ram(level=6, screen=0x39, x=120, y=77)
    dest[ADDR_ROD] = 1
    arrive = make_south29_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_39"
    assert level6_south29_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x29, x=55, y=133)
    still[ADDR_ROD] = 1
    assert not level6_south29_success(read_snapshot(still))
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0
    dest[ADDR_BOW] = 1
    dest[ADDR_ARROWS] = 1
    armed = read_snapshot(dest)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-south29", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_south_0x29"
    assert "level6-south29" in L6_THROUGH


def test_level6_east39_y_align_then_right() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_east39 import (
        EAST_DOOR_X,
        EAST_DOOR_Y,
        make_east39_controller,
    )
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_east39_stages()
    assert [name for name, _, _ in stages] == ["level6_east_0x39"]
    leftover = _ram(level=6, screen=0x39, x=136, y=173)
    leftover[ADDR_ROD] = 1
    ctl = make_east39_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "east_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    mid = _ram(level=6, screen=0x39, x=176, y=EAST_DOOR_Y)
    mid[ADDR_ROD] = 1
    hold = make_east39_controller()
    act = hold.step(read_snapshot(mid))
    assert act.reason == "east_hold"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_idle_action())
    door = _ram(level=6, screen=0x39, x=EAST_DOOR_X, y=EAST_DOOR_Y)
    door[ADDR_ROD] = 1
    push = make_east39_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "east_push"
    assert list(act.action) == list(nes_action("RIGHT"))
    dest = _ram(level=6, screen=0x3A, x=16, y=141)
    dest[ADDR_ROD] = 1
    arrive = make_east39_controller()
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert act.reason == "arrived_3a"
    assert level6_east39_success(read_snapshot(dest))
    still = _ram(level=6, screen=0x39, x=136, y=173)
    still[ADDR_ROD] = 1
    assert not level6_east39_success(read_snapshot(still))
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0
    dest[ADDR_BOW] = 1
    dest[ADDR_ARROWS] = 1
    armed = read_snapshot(dest)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-east39", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_east_0x39"
    assert "level6-east39" in L6_THROUGH


def test_level6_west19_occupancy_left_at_y141() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_west19 import WEST_DOOR_X, WEST_DOOR_Y, make_west19_controller
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_west19_stages()
    assert [name for name, _, _ in stages] == ["level6_west_0x19"]
    leftover = _ram(level=6, screen=0x19, x=120, y=205, keys=4)
    leftover[ADDR_ROD] = 1
    ctl = make_west19_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    aisle = _ram(level=6, screen=0x19, x=120, y=141, keys=4)
    aisle[ADDR_ROD] = 1
    path = make_west19_controller()
    act = path.step(read_snapshot(aisle))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("LEFT"))
    door = _ram(level=6, screen=0x19, x=WEST_DOOR_X, y=WEST_DOOR_Y, keys=4)
    door[ADDR_ROD] = 1
    push = make_west19_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))
    north = _ram(level=6, screen=0x19, x=120, y=109, keys=4)
    north[ADDR_ROD] = 1
    halt = make_west19_controller()
    act = halt.step(read_snapshot(north))
    assert act.reason == "north_key_halt"
    assert list(act.action) == list(nes_idle_action())
    dest = _ram(level=6, screen=0x18, x=208, y=141, keys=4)
    dest[ADDR_ROD] = 1
    arrive = make_west19_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_18"
    assert level6_west19_success(read_snapshot(dest))
    spent = _ram(level=6, screen=0x18, x=208, y=141, keys=3)
    spent[ADDR_ROD] = 1
    key_hop = make_west19_controller()
    key_hop.keys = 4
    act = key_hop.step(read_snapshot(spent))
    assert any(n.startswith("key_spent_19_to_18_4->3") for n in key_hop.notes)
    still = _ram(level=6, screen=0x19, x=32, y=141, keys=4)
    still[ADDR_ROD] = 1
    assert not level6_west19_success(read_snapshot(still))
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    assert not level6_west19_success(read_snapshot(key_up))
    north_fail = make_west19_controller()
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(key_up))
    assert north_fail.failed
    assert act.reason.startswith("key_up_09")
    back = _ram(level=6, screen=0x29, x=120, y=77, keys=4)
    back[ADDR_ROD] = 1
    south = make_west19_controller()
    act = south.step(read_snapshot(back))
    assert south.failed
    assert act.reason.startswith("backtrack_29")
    snap = read_snapshot(leftover)
    assert snap.bow == 0
    assert snap.arrows == 0
    leftover[ADDR_BOW] = 1
    leftover[ADDR_ARROWS] = 1
    armed = read_snapshot(leftover)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-west19", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x19"
    assert "level6-west19" in L6_THROUGH
    assert L6_THROUGH[-8:] == (
        "level6-south1d",
        "level6-west2d",
        "level6-north2c",
        "level6-east3a",
        "level6-north39",
        "level6-inland29",
        "level6-west19",
        "level6-south18",
    )


def test_level6_south18_occupancy_down_from_east_mouth() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level6_south18 import SOUTH_DOOR_X, SOUTH_DOOR_Y, make_south18_controller
    from zelda_i.ram import ADDR_ARROWS, ADDR_BOW, ADDR_ROD

    stages = level6_south18_stages()
    assert [name for name, _, _ in stages] == ["level6_south_0x18"]
    leftover = _ram(level=6, screen=0x18, x=208, y=141, keys=4)
    leftover[ADDR_ROD] = 1
    ctl = make_south18_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    aisle = _ram(level=6, screen=0x18, x=120, y=141, keys=4)
    aisle[ADDR_ROD] = 1
    path = make_south18_controller()
    act = path.step(read_snapshot(aisle))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("DOWN"))
    door = _ram(level=6, screen=0x18, x=SOUTH_DOOR_X, y=SOUTH_DOOR_Y, keys=4)
    door[ADDR_ROD] = 1
    push = make_south18_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "south_push"
    assert list(act.action) == list(nes_action("DOWN"))
    north = _ram(level=6, screen=0x18, x=120, y=109, keys=4)
    north[ADDR_ROD] = 1
    halt = make_south18_controller()
    act = halt.step(read_snapshot(north))
    assert act.reason == "north_hole_halt"
    assert list(act.action) == list(nes_idle_action())
    dest = _ram(level=6, screen=0x28, x=120, y=77, keys=4)
    dest[ADDR_ROD] = 1
    arrive = make_south18_controller()
    arrive.keys = 4
    act = arrive.step(read_snapshot(dest))
    assert arrive.success
    assert not arrive.failed
    assert act.reason == "arrived_28"
    assert level6_south18_success(read_snapshot(dest))
    spent = _ram(level=6, screen=0x28, x=120, y=77, keys=3)
    spent[ADDR_ROD] = 1
    key_hop = make_south18_controller()
    key_hop.keys = 4
    act = key_hop.step(read_snapshot(spent))
    assert any(n.startswith("key_spent_18_to_28_4->3") for n in key_hop.notes)
    still = _ram(level=6, screen=0x18, x=120, y=189, keys=4)
    still[ADDR_ROD] = 1
    assert not level6_south18_success(read_snapshot(still))
    key_up = _ram(level=6, screen=0x09, x=120, y=205, keys=3)
    key_up[ADDR_ROD] = 1
    assert not level6_south18_success(read_snapshot(key_up))
    north_fail = make_south18_controller()
    north_fail.keys = 4
    act = north_fail.step(read_snapshot(key_up))
    assert north_fail.failed
    assert act.reason.startswith("key_up_09")
    back = _ram(level=6, screen=0x19, x=32, y=141, keys=4)
    back[ADDR_ROD] = 1
    east = make_south18_controller()
    act = east.step(read_snapshot(back))
    assert east.failed
    assert act.reason.startswith("backtrack_19")
    cellar = _ram(level=6, screen=0x18, x=120, y=96, keys=4, mode=9)
    cellar[ADDR_ROD] = 1
    warp = make_south18_controller()
    act = warp.step(read_snapshot(cellar))
    assert warp.failed
    assert act.reason.startswith("warped_cellar")
    snap = read_snapshot(leftover)
    assert snap.bow == 0
    assert snap.arrows == 0
    leftover[ADDR_BOW] = 1
    leftover[ADDR_ARROWS] = 1
    armed = read_snapshot(leftover)
    assert armed.bow == 1
    assert armed.arrows == 1
    run = SpineRun(through="level6-south18", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_south_0x18"
    assert "level6-south18" in L6_THROUGH
    assert L6_THROUGH[-8:] == (
        "level6-south1d",
        "level6-west2d",
        "level6-north2c",
        "level6-east3a",
        "level6-north39",
        "level6-inland29",
        "level6-west19",
        "level6-south18",
    )
