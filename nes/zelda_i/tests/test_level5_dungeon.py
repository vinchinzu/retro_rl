"""Unit tests for Level 5 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import GenericDungeonRoomController
from zelda_i.level5.dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    Level5PolsVoiceController,
    level5_room_66_cleared,
    level5_room_77_key_success,
)
from zelda_i.level5.path import make_pols_south_controller, make_room66_controller
from zelda_i.level5.spine import ROOM_66_SPINE_SPEC
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
    read_snapshot,
)


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    doors: int = 0,
    all_dead: int = 0,
    enemy_type: int = GIBDO_OBJECT_TYPE,
    enemies: int = 0,
    hp: int = 112,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[ADDR_ROOM_ALL_DEAD] = all_dead
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 64 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def test_room_66_combat_uses_occupancy_across_river() -> None:
    """TF suffix leftover (79,165): cardinal patrol never crossed the river."""
    assert ROOM_66_SPEC.combat.occupancy_patrol is True
    assert ROOM_66_SPEC.combat.occupancy_bounds == (16, 216, 77, 205)
    assert ROOM_66_SPEC.combat.contact_backstep == 16
    assert ROOM_66_SPINE_SPEC.combat.contact_backstep == 16


def test_room_66_peels_contact_gibdo_at_leftover() -> None:
    """Death (128,133): peel a north Gibdo; do not greedy-close UP into the body."""
    ram = _ram(room=ROOM_L5_GIBDO_66, x=128, y=133, enemies=1, hp=112)
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 125
    act = GenericDungeonRoomController(spec=ROOM_66_SPINE_SPEC).step(
        read_snapshot(ram)
    )
    assert act.reason == "combat_backstep"
    assert act.action == nes_action("DOWN")
    assert act.action != nes_action("UP")
    assert act.action != nes_action("UP", "A")


def test_room_66_leaves_nw_pocket_south() -> None:
    """Timeout (64,120): 1 east Gibdo — DOWN off the pocket, not idle wait."""
    ram = _ram(room=ROOM_L5_GIBDO_66, x=64, y=120, enemies=1, hp=112)
    ram[ADDR_LINK_X + 1] = 192
    ram[ADDR_LINK_Y + 1] = 149
    act = make_room66_controller(spec=ROOM_66_SPINE_SPEC).step(read_snapshot(ram))
    assert act.reason == "66_pocket_south"
    assert act.action == nes_action("DOWN")
    assert act.action != nes_idle_action()


def test_room_66_cleared_predicate() -> None:
    assert level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x08, all_dead=20)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=3, doors=0x08, all_dead=20, hp=112)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x00, all_dead=20)
    )


def test_room_77_key_success() -> None:
    assert level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=1, enemies=0))
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=0, enemies=0))
    assert not level5_room_77_key_success(
        _ram(
            room=ROOM_L5_POLS_77,
            keys=1,
            enemies=2,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=160,
        )
    )
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_ENTRY, keys=1, enemies=0))


def test_pols_voice_controller_is_solid() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    # (153, 189) is under the right 2x3 cluster on the south wall - MUST be solid
    assert ctrl._is_solid(153, 189) is True
    # (200, 189) is in the east column on the south wall - open
    assert ctrl._is_solid(200, 189) is False
    # x in 152..184 at y >= 109 is strictly blocked
    for x in (152, 160, 176, 184):
        for y in (109, 141, 165, 189):
            assert ctrl._is_solid(x, y) is True
    # Southwest dead-end pocket (x < 88 and y > 145) is solid
    for x in (44, 48, 64, 84):
        for y in (146, 160, 173, 182, 189):
            assert ctrl._is_solid(x, y) is True
    # West door entrance and north aisle are open
    assert ctrl._is_solid(48, 141) is False
    assert ctrl._is_solid(48, 109) is False
    assert ctrl._is_solid(56, 109) is False
    # Left cluster (56..88, 109 < y <= 164) is solid
    assert ctrl._is_solid(56, 141) is True
    assert ctrl._is_solid(88, 141) is True


def test_pols_voice_controller_west_door_routes_north() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=48,
        y=141,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 141
    snap = read_snapshot(ram)

    # At (48, 141) with enemy in central aisle, Link routes UP towards y <= 109
    act = ctrl.step(snap)
    assert act.action == nes_action("UP")
    assert act.reason == "route_north_from_west_door"
    assert ctrl.last_dir == "UP"

    # Once at y <= 109 (e.g. 48, 109), Link can move RIGHT into central aisle
    ram[ADDR_LINK_Y] = 109
    snap_109 = read_snapshot(ram)
    act_109 = ctrl.step(snap_109)
    assert act_109.action == nes_action("RIGHT")
    assert ctrl.last_dir == "RIGHT"


def test_pols_voice_controller_occupancy_miss_and_stand() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    # Pre-block DOWN, LEFT, RIGHT so only UP is considered from (120, 141)
    ctrl.blocked_cells.add((120, 145))  # DOWN
    ctrl.blocked_cells.add((116, 141))  # LEFT
    ctrl.blocked_cells.add((124, 141))  # RIGHT

    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=120,
        y=141,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 200
    ram[ADDR_LINK_Y + 1] = 93
    snap = read_snapshot(ram)

    # Frame 1: Link is at (120, 141), moves UP
    act1 = ctrl.step(snap)
    assert act1.action == nes_action("UP")
    assert ctrl.last_dir == "UP"

    # Frames 2-4: Link remains at (120, 141)
    ctrl.step(snap)
    ctrl.step(snap)
    ctrl.step(snap)

    # Frame 5: 4th stuck frame -> occupancy miss records cell ahead (120, 137)
    act5 = ctrl.step(snap)
    assert (120, 137) in ctrl.blocked_cells
    assert ctrl.misses == 1
    # Now all directions are blocked -> controller stands on no path
    assert act5.reason == "stand_no_path"
    assert act5.action == nes_idle_action()


def test_pols_voice_does_not_chase_leaping_body() -> None:
    """Leftover (133,157): a hopper in the 12-24 band is not a strike/chase."""
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (133, 130)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=133,
        y=157,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 133
    ram[ADDR_LINK_Y + 1] = 140
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("UP", "A")
    assert act.action != nes_action("DOWN", "A")
    assert act.action != nes_action("DOWN")
    assert act.reason.startswith("evade_leap_")


def test_pols_voice_leaves_waist_south() -> None:
    """Death (101,141): leaping Pols in the waist — south, not A."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (110, 141)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=101,
        y=141,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 141
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("LEFT", "A")
    assert act.action != nes_action("RIGHT", "A")
    assert act.action == nes_action("DOWN")
    assert act.reason in {"77_waist_south", "evade_leap_DOWN"}


def test_pols_voice_leaves_sw_pocket_east() -> None:
    """Death (61,173): west of the west island — RIGHT into the south aisle."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=61,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("LEFT")
    assert act.action != nes_action("LEFT", "A")
    assert act.action == nes_action("RIGHT")
    assert act.reason == "77_aisle_east"


def test_pols_voice_leaves_south_lip() -> None:
    """Death (142,181): south wall / right island — UP or LEFT, not A/DOWN."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=142,
        y=181,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("DOWN")
    assert act.action != nes_action("DOWN", "A")
    assert act.action in (nes_action("UP"), nes_action("LEFT"))
    assert act.reason in {"77_lip_north", "77_aisle_west"}


def test_pols_voice_leaves_south_lip_at_179() -> None:
    """Death (136,179): y=179 slipped under y=181 peel — UP or LEFT, not A/DOWN."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=136,
        y=179,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("DOWN")
    assert act.action != nes_action("DOWN", "A")
    assert act.action in (nes_action("UP"), nes_action("LEFT"))
    assert act.reason in {"77_lip_north", "77_aisle_west"}


def test_pols_voice_peels_hold_row_hopper() -> None:
    """Death (136,173): leaping Pols on the south aisle — LEFT/RIGHT, not A."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (104, 173)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=136,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("UP")
    assert act.action != nes_action("DOWN")
    assert act.action != nes_action("UP", "A")
    assert act.action != nes_action("DOWN", "A")
    assert act.action in (nes_action("LEFT"), nes_action("RIGHT"))
    assert act.reason in {"77_hold_peel_LEFT", "77_hold_peel_RIGHT"}


def test_pols_voice_does_not_walk_into_east_island() -> None:
    """Death (139,173): x>=132 never RIGHT into the east 2x3 — LEFT or idle."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (104, 173)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=139,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("RIGHT")
    assert act.action != nes_action("RIGHT", "A")
    assert act.action != nes_action("DOWN")
    assert act.action != nes_action("UP")
    assert act.action in (nes_action("LEFT"), nes_idle_action())
    assert act.reason in {"77_hold_peel_LEFT", "stand_no_path"}


def test_pols_voice_holds_aisle_center() -> None:
    """Death (131,173): LEFT toward x=120, not RIGHT, not A."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (104, 173)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=131,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert "A" not in act.reason
    assert act.action != nes_action("A")
    assert act.action != nes_action("RIGHT")
    assert act.action != nes_action("RIGHT", "A")
    assert act.action != nes_action("DOWN")
    assert act.action != nes_action("UP")
    assert act.action == nes_action("LEFT")
    assert act.reason == "77_hold_peel_LEFT"


def test_pols_voice_peels_hold_center_hopper() -> None:
    """Death (120,173): leaping Pols at x≈120 — L/R not A/idle; no recenter."""
    # (lx, pol_x, prev, expect) expect=None means either L or R.
    cases = (
        (120, 136, (104, 173), None),  # disp off-column
        (120, 120, (120, 149), None),  # leaping on x=120
        (116, 120, (120, 149), "LEFT"),  # do not RIGHT back under landing
        (124, 120, (120, 149), "RIGHT"),  # do not LEFT back under landing
    )
    for lx, px, prev, expect in cases:
        ctrl = make_pols_south_controller()
        ctrl.entered_central = True
        ctrl.enemy_prev_pos[1] = prev
        ram = _ram(
            room=ROOM_L5_POLS_77,
            x=lx,
            y=173,
            enemies=1,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=160,
        )
        ram[ADDR_LINK_X + 1] = px
        ram[ADDR_LINK_Y + 1] = 173
        act = ctrl.step(read_snapshot(ram))
        assert "A" not in act.reason
        assert act.action != nes_action("A")
        assert act.action != nes_idle_action()
        assert act.action != nes_action("DOWN")
        assert act.action != nes_action("UP")
        if expect is None:
            assert act.action in (nes_action("LEFT"), nes_action("RIGHT"))
        else:
            assert act.action == nes_action(expect)
        assert act.reason in {"77_hold_peel_LEFT", "77_hold_peel_RIGHT"}
    # state==1 at leftover pose, zero displacement still peels.
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=120,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 173
    ram[ADDR_OBJ_STATE + 1] = 1
    act = ctrl.step(read_snapshot(ram))
    assert act.action in (nes_action("LEFT"), nes_action("RIGHT"))
    assert act.reason in {"77_hold_peel_LEFT", "77_hold_peel_RIGHT"}
    assert act.action != nes_idle_action()


def test_pols_voice_slashes_landed_at_hold_center() -> None:
    """Hold (120,173): landed Pols (state==0, no displacement) still slashes."""
    ctrl = make_pols_south_controller()
    ctrl.entered_central = True
    ctrl.enemy_prev_pos[1] = (136, 173)
    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=120,
        y=173,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 136
    ram[ADDR_LINK_Y + 1] = 173
    act = ctrl.step(read_snapshot(ram))
    assert act.action == nes_action("A")
    assert act.reason == "77_hold_slash"

