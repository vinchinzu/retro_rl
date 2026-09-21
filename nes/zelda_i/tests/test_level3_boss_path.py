"""Unit tests for Level 3 boss path library (no emulator)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallPhase
from zelda_i.dungeon.door_hop import door_band_goal
from zelda_i.dungeon.ops import (
    GEL_SPLIT_OBJECT_TYPE,
    live_killables,
)
from zelda_i.level3.boss_path import (
    MANHANDLA_FIGHT_Y_MIN,
    MANHANDLA_RETREAT,
    RIGHT_5C_SPEC,
    UP_5D_SPEC,
    L3DoorHopController,
    Level3BossPathController,
    Level3Clear5cController,
    Level3ManhandlaController,
    Level3SpawnClearController,
    level3_boss_suffix_stages,
    make_l3_bomb_5b,
    prep_5d_still_killable,
)
from zelda_i.dungeon.engine import GenericDungeonRoomController
from zelda_i.level3.dungeon import (
    DARKNUT_OBJECT_TYPE,
    GEL_OBJECT_TYPE,
    INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    MANHANDLA_OBJECT_TYPE,
    ROOM_5C_SPEC,
    ROOM_5D_SPEC,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_DARKNUTS,
    ROOM_L3_TF,
    ZOL_OBJECT_TYPE,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram


_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 3,
    "x": 120,
    "y": 141,
    "triforce": 0x03,
    "bombs": 4,
    "keys": 1,
    "health": 0x22,
}


def _ram(
    *,
    level: int = 3,
    room: int = ROOM_L3_BOSS_PREP,
    x: int = 120,
    y: int = 141,
    mode: int = PLAY_MODE,
    **fields: int,
) -> np.ndarray:
    ram = make_ram(
        _DEFAULTS, mode=mode, level=level, screen=room, x=x, y=y, **fields
    )
    return ram


def test_prep_killables_ignore_0x2b_slots_1_12() -> None:
    ram = _ram(room=ROOM_L3_BOSS_PREP)
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 32
    ram[ADDR_OBJ_TYPE + 2] = INVULN_MOVER_0X2B
    ram[ADDR_OBJ_HP + 2] = 240
    ram[ADDR_OBJ_TYPE + 3] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 3] = 0
    # Gel residual in slot 11 (LIVE seal on UP shutter)
    ram[ADDR_OBJ_TYPE + 11] = GEL_SPLIT_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 11] = 0
    snap = read_snapshot(ram)
    killable = prep_5d_still_killable(snap)
    types = {o.type_id for o in killable}
    slots = {o.slot for o in killable}
    assert ZOL_OBJECT_TYPE in types
    assert KEESE_OBJECT_TYPE in types
    assert GEL_SPLIT_OBJECT_TYPE in types
    assert INVULN_MOVER_0X2B not in types
    assert 11 in slots

    # live_killables with only darknuts must not pick 0x2b
    assert live_killables(snap, (0x0B,)) == []


def test_continuous_controller_forbids_state_restore() -> None:
    ctl = Level3BossPathController(continuous_mode=True)
    em = SimpleNamespace(set_state=lambda state: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(RuntimeError, match="forbids"):
        ctl._restore_state(SimpleNamespace(em=em), object())
    assert ctl.state_restores == 0


def test_path_to_5d_has_no_5b_return_fight_clear() -> None:
    src = inspect.getsource(Level3BossPathController.path_to_5d)
    assert "inspect_5b_return" not in src
    assert "clear_5b_return" not in src
    assert "failed_clear_5b_return" not in src
    assert "make_l3_bomb_5b" in src
    assert "DoorHopController" in src
    assert "push_dir(" not in src
    assert "idle(env, assist, total, 60)" not in src
    assert "idle(env, assist, total, 40)" not in src
    assert "idle(env, assist, total, 110)" not in src


def _snap(*, screen: int, **fields: int):
    return read_snapshot(_ram(room=screen, **fields))


def test_right_5c_diamond_waist_uses_occupancy() -> None:
    """(96,141) is a 0x5c diamond. cardinal_hold RIGHT never moves; occupancy must."""
    assert RIGHT_5C_SPEC.cardinal_hold is False
    assert RIGHT_5C_SPEC.align == "dest"
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    leftover = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=96, y=141)
    first = ctl.step(leftover)
    assert not ctl.failed
    assert first.reason != "east_hold"
    assert list(first.action) != list(nes_idle_action())
    stuck = [ctl.step(leftover).reason for _ in range(12)]
    assert "east_hold" not in stuck
    assert ctl.walker.misses >= 1


def test_right_5c_tilemap_seed_goes_around_waist() -> None:
    """bind_env seeds $6530 BLOCK_TILES; first step is not RIGHT into the diamond."""
    from zelda_i.dungeon.tilemap import (
        ADDR_ROOM_TILE_MAP,
        TILE_ROWS,
        WRAM_BASE,
        WRAM_RAM_OFFSET,
    )

    cpu = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=96, y=141, doors=3, bombs=4)
    ram = np.zeros(10240, dtype=np.uint8)
    ram[:0x800] = cpu
    base = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    # Waist diamond at (96,144): 2x2 BLOCK_TILES.
    col, row = 96 // 8, (144 - 64) // 8
    for dc, dr, value in ((0, 0, 0xB0), (1, 0, 0xB2), (0, 1, 0xB1), (1, 1, 0xB3)):
        ram[base + (col + dc) * TILE_ROWS + (row + dr)] = value
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    assert ctl._blocks_seeded
    assert (97, 141) in ctl.walker.grid.blocked
    assert (97, 141) not in ctl.walker.grid.inferred
    assert ctl.walker.grid.passable(208, 141)
    first = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert list(first.action) != list(nes_action("RIGHT"))
    assert list(first.action) != list(nes_idle_action())


def test_right_5c_leftover_relative_goal() -> None:
    """Wave 0 door_band_goal: on-band leftover keeps y; south mouth uses door y."""
    goal = (208, 141)
    assert door_band_goal("RIGHT", (120, 143), goal) == (208, 143)
    assert door_band_goal("RIGHT", (120, 181), goal) == (208, 141)
    assert door_band_goal("RIGHT", (208, 141), goal) == (208, 141)
    on_band = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=143)
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    first = ctl.step(on_band)
    assert ctl.goal == (208, 143)
    assert list(first.action) != list(nes_idle_action())
    south = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=181)
    ctl2 = L3DoorHopController(RIGHT_5C_SPEC)
    ctl2.step(south)
    assert ctl2.goal == (208, 141)


def test_up_5d_off_column_binds_door_x() -> None:
    assert door_band_goal("UP", (208, 157), (120, 93)) == (120, 109)
    assert door_band_goal("UP", (118, 157), (120, 93)) == (118, 109)
    leftover = _snap(screen=ROOM_L3_BOSS_PREP, x=208, y=157)
    ctl = L3DoorHopController(UP_5D_SPEC)
    first = ctl.step(leftover)
    assert ctl.goal[0] == 120
    assert list(first.action) != list(nes_idle_action())


def test_right_5c_jitter_still_arrives_dest() -> None:
    """Entry-frame jitter is not the hop; dest is RAM 0x5d play.

    idle_frames=223 is the only value worth pinning here: the wait loop is
    straight-line code with no branch on iteration count (mode=6 does not
    change what the loop does), so 0/30/90 measured as byte-identical
    coverage to this case and to the file's other no-idle door-hop tests —
    collapsed 2026-09-16 (see A_coverage_redundant.txt)."""
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    wait = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=141, mode=6)
    for _ in range(223):
        act = ctl.step(wait)
        assert not ctl.success
        assert not ctl.failed
        assert list(act.action) == list(nes_action("RIGHT"))
    origin = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=141)
    ctl.step(origin)
    dest = _snap(screen=ROOM_L3_BOSS_PREP, x=32, y=141)
    done = ctl.step(dest)
    assert ctl.success
    assert done.reason.startswith("arrived_")


def test_bomb_5b_zero_bombs_fails_without_poke() -> None:
    ctl = make_l3_bomb_5b()
    assert getattr(ctl, "select_item", None) == 1
    empty = _snap(screen=ROOM_L3_DARKNUTS, x=192, y=141, bombs=0)
    ctl.step(empty)
    assert ctl.phase is BombWallPhase.FAILED
    assert any("no_bombs" in n for n in ctl.notes)


def test_bomb_5b_at_stand_faces_right() -> None:
    ctl = make_l3_bomb_5b()
    stand = _snap(screen=ROOM_L3_DARKNUTS, x=192, y=141, bombs=4)
    act = ctl.step(stand)
    assert ctl.phase is not BombWallPhase.FAILED
    assert act.reason == "face_right"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_bomb_5b_off_stand_walks_to_dest() -> None:
    ctl = make_l3_bomb_5b()
    inland = _snap(screen=ROOM_L3_DARKNUTS, x=120, y=141, bombs=4)
    act = ctl.step(inland)
    assert ctl.phase is not BombWallPhase.FAILED
    assert list(act.action) != list(nes_idle_action())


def test_bomb_5b_inland_176_125_drops_to_waist() -> None:
    """Live leftover (176,125) must y-first to stand y=141, not RIGHT into a tile."""
    ctl = make_l3_bomb_5b()
    leftover = _snap(screen=ROOM_L3_DARKNUTS, x=176, y=125, bombs=10)
    act = ctl.step(leftover)
    assert ctl.phase is not BombWallPhase.FAILED
    assert act.reason in {"approach_y", "south_band"}
    assert list(act.action) == list(nes_action("DOWN"))


def _plant_manhandla(ram: np.ndarray, *, x: int = 128, y: int = 112) -> None:
    ram[ADDR_OBJ_TYPE + 1] = MANHANDLA_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ram[ADDR_LINK_X + 1] = x
    ram[ADDR_LINK_Y + 1] = y


def test_manhandla_south_mouth_climbs() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=205, bombs=4)
    _plant_manhandla(ram)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert act.reason in {"climb", "approach", "fight_up"}
    assert list(act.action) == list(nes_action("UP"))


def test_manhandla_spawn_waits_not_hc() -> None:
    """Empty 0x4d is spawn, not HC collect. Serial red 4 walked into the flower."""
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=189, bombs=4)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert not ctl.saw_heads
    assert act.reason in {"climb", "spawn_wait"}
    assert act.reason != "hc_align_y"


def test_manhandla_far_centroid_approaches() -> None:
    """c_dist > bomb max is approach, not a NameError. ROM red 1 crashed here."""
    ram = _ram(room=ROOM_L3_BOSS, x=176, y=157, bombs=4)
    _plant_manhandla(ram, x=80, y=109)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert act.reason == "approach"
    assert list(act.action) != list(nes_idle_action())


def test_manhandla_near_head_places_bomb() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=141, bombs=4)
    _plant_manhandla(ram, x=120, y=109)
    ctl = Level3ManhandlaController()
    reasons = [ctl.step(read_snapshot(ram)).reason for _ in range(8)]
    assert not ctl.failed
    assert any(r in {"place_bomb", "approach"} for r in reasons)


def test_manhandla_contact_dodges_south() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=141, bombs=4)
    _plant_manhandla(ram, x=120, y=125)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "combat_backstep"
    assert list(act.action) == list(nes_action("DOWN"))


def test_manhandla_does_not_chase_north_of_waist() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=82, y=125, bombs=2)
    _plant_manhandla(ram, x=82, y=93)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "stay_south"
    assert list(act.action) == list(nes_action("DOWN"))
    assert ram[ADDR_LINK_Y] < MANHANDLA_FIGHT_Y_MIN or act.reason == "stay_south"


def test_manhandla_retreats_after_bomb() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=141, bombs=4)
    _plant_manhandla(ram, x=120, y=109)
    ctl = Level3ManhandlaController()
    first = ctl.step(read_snapshot(ram))
    assert first.reason == "place_bomb"
    second = ctl.step(read_snapshot(ram))
    assert second.reason == "retreat_bomb"
    assert list(second.action) == list(nes_action("DOWN"))


def test_manhandla_retreat_at_y_max_not_east_or_south_door() -> None:
    """ROM death (184,173): away-RIGHT at y=MAX; y=189 is the south door."""
    ram = _ram(room=ROOM_L3_BOSS, x=136, y=141, bombs=4)
    _plant_manhandla(ram, x=120, y=109)
    ctl = Level3ManhandlaController()
    assert ctl.step(read_snapshot(ram)).reason == "place_bomb"
    ram[ADDR_LINK_X] = 160
    ram[ADDR_LINK_Y] = 173
    east = ctl.step(read_snapshot(ram))
    assert east.reason == "retreat_bomb"
    assert list(east.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_Y] = 189
    south = ctl.step(read_snapshot(ram))
    assert south.reason == "retreat_bomb"
    assert list(south.action) != list(nes_action("DOWN"))
    assert list(south.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 80
    ram[ADDR_LINK_Y] = 173
    west = ctl.step(read_snapshot(ram))
    assert list(west.action) == list(nes_action("RIGHT"))


def test_manhandla_after_retreat_does_not_reenter_flower() -> None:
    """ROM death: retreat (104,163) then approach UP to waist (104,142). Stay south."""
    ram = _ram(room=ROOM_L3_BOSS, x=136, y=141, bombs=4)
    _plant_manhandla(ram, x=120, y=109)
    ctl = Level3ManhandlaController()
    assert ctl.step(read_snapshot(ram)).reason == "place_bomb"
    ram[ADDR_LINK_X] = 104
    ram[ADDR_LINK_Y] = 163
    for _ in range(MANHANDLA_RETREAT):
        act = ctl.step(read_snapshot(ram))
        assert act.reason == "retreat_bomb"
        assert list(act.action) == list(nes_action("DOWN"))
    leftover = _ram(room=ROOM_L3_BOSS, x=104, y=163, bombs=3)
    _plant_manhandla(leftover, x=120, y=109)
    for _ in range(16):
        act = ctl.step(read_snapshot(leftover))
        assert not ctl.failed
        assert list(act.action) != list(nes_action("UP"))
        assert act.reason in {"strafe", "approach", "place_bomb", "retreat_bomb"}


def test_manhandla_after_kill_collects_hc() -> None:
    live = _ram(room=ROOM_L3_BOSS, x=120, y=141, bombs=2)
    _plant_manhandla(live, x=120, y=109)
    ctl = Level3ManhandlaController()
    ctl.step(read_snapshot(live))
    assert ctl.saw_heads
    dead = _ram(room=ROOM_L3_BOSS, x=120, y=141, bombs=2)
    act = ctl.step(read_snapshot(dead))
    assert not ctl.failed
    assert act.reason.startswith("hc")


def test_manhandla_tf_bit_is_dest() -> None:
    ram = _ram(room=ROOM_L3_TF, x=120, y=141, triforce=0x07)
    ram[ADDR_TRIFORCE] = 0x07
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "tf04"


def test_clean_suffix_stages_do_not_poke_inventory() -> None:
    names = []
    for name, ctl, max_frames in level3_boss_suffix_stages():
        names.append(name)
        assert max_frames > 0
        assert getattr(ctl, "poke_bombs", None) in (None, False)
        assert getattr(ctl, "route_eligible", False) is False
        src = inspect.getsource(type(ctl))
        assert "poke_bombs(" not in src
        assert "poke_keys(" not in src
    assert names == [
        "bomb_5b",
        "clear_5c",
        "right_5d",
        "clear_5d",
        "up_4d",
        "manhandla_tf",
    ]
    src = inspect.getsource(level3_boss_suffix_stages)
    assert "poke_bombs" not in src
    assert "make_l3_bomb_5b" in src


def _plant_darknut(
    ram: np.ndarray, slot: int = 1, *, x: int = 80, y: int = 141, facing: int = 2
) -> None:
    ram[ADDR_OBJ_TYPE + slot] = DARKNUT_OBJECT_TYPE
    ram[ADDR_OBJ_HP + slot] = 64
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y
    # Facing in Zelda 1 RAM: $0098 + slot (1=R, 2=L, 4=D, 8=U)
    ram[0x0098 + slot] = facing


@pytest.mark.parametrize(
    "x,y",
    [
        pytest.param(98, 157, id="west_leftover"),
        pytest.param(192, 181, id="se_corner"),
    ],
)
def test_clear_5c_leftover_is_dest_hop(x: int, y: int) -> None:
    """Combat leftover on a diamond is success; dest hop BFS owns the leave."""
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=x, y=y, bombs=4, doors=3)
    ctl = Level3SpawnClearController(ROOM_5C_SPEC)
    ctl.saw_live = True
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "done"


def test_clear_5c_east_corridor_is_leave() -> None:
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=192, y=141, bombs=4, doors=3)
    ctl = Level3SpawnClearController(ROOM_5C_SPEC)
    ctl.saw_live = True
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "done"


def test_spawn_clear_5c_wires_clear5c_controller() -> None:
    ctl = Level3SpawnClearController(ROOM_5C_SPEC)
    assert isinstance(ctl.combat, Level3Clear5cController)


def test_clear_5c_spends_bombs_when_available() -> None:
    """Carried bombs > 4: spend bomb at waist when Darknut approaches."""
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=64, y=141, bombs=7)
    _plant_darknut(ram, slot=1, x=96, y=141, facing=2)
    ctl = Level3Clear5cController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "place_bomb"
    assert ctl.bomb_cd > 0
    assert ctl.retreat_frames > 0


def test_clear_5c_conserves_bombs_for_manhandla() -> None:
    """Carried bombs <= 4: preserve bombs for Manhandla; use sword/flank."""
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=64, y=141, bombs=4)
    _plant_darknut(ram, slot=1, x=96, y=141, facing=2)
    ctl = Level3Clear5cController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason != "place_bomb"


def test_clear_5c_does_not_chase_north_onto_diamonds() -> None:
    """Link must stay south of waist; do not chase north into y<=109 diamond trap."""
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=64, y=141, bombs=4)
    _plant_darknut(ram, slot=1, x=64, y=93, facing=4)
    ctl = Level3Clear5cController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason in {"stay_south", "patrol_waist", "wait_for_darknut"}
    assert list(act.action) != list(nes_action("UP"))


def test_5d_nw_death_corner_peels_off_wall() -> None:
    """Trial leftover (49,93) is the north-west door; do not chase UP."""
    ram = _ram(room=ROOM_L3_BOSS_PREP, x=49, y=93, bombs=4)
    ram[ADDR_OBJ_TYPE + 1] = GEL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 16
    ram[ADDR_LINK_X + 1] = 80
    ram[ADDR_LINK_Y + 1] = 141
    ctl = GenericDungeonRoomController(ROOM_5D_SPEC)
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "leave_wall"
    assert list(act.action) != list(nes_action("UP"))


def test_clear_5c_avoids_direct_shield_slash() -> None:
    """Enemy facing Link directly: flank perpendicularly instead of slashing into shield."""
    ram = _ram(room=ROOM_L3_BOMB_SHORTCUT, x=64, y=141, bombs=4)
    _plant_darknut(ram, slot=1, x=90, y=141, facing=2)  # facing 2 = LEFT (towards Link)
    ctl = Level3Clear5cController()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "flank"
    assert act.reason != "sword_slash"

