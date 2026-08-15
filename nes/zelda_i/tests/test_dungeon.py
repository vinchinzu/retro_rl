from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.chain import run_controller_stage
from zelda_i.level1_dungeon import ROOM_45_SURVIVAL_SPEC
from zelda_i.dungeon import (
    BLUE_GORIYA_OBJECT_TYPE,
    DungeonPhase,
    GenericDungeonRoomController,
    LEVEL_2,
    ROOM_23_SPEC,
    ROOM_35_SPEC,
    ROOM_33_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_44_SPEC,
    ROOM_45_SPEC,
    ROOM_52_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
    ROOM_4E_SPEC,
    ROOM_4F_SPEC,
    ROOM_5E_SPEC,
    ROOM_6C_SPEC,
    ROOM_6D_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    ROOM_7D_SPEC,
    ROOM_7E_SPEC,
    GEL_OBJECT_TYPE,
    GORIYA_OBJECT_TYPE,
    ROPE_OBJECT_TYPE,
    dungeon_room_cleared,
    level2_room_4e_key_success,
    level2_room_4f_magic_boomerang_success,
    level2_room_4f_ready,
    level2_room_5e_cleared,
    level2_room_5f_ready,
    level2_room_6c_key_success,
    level2_room_6d_cleared,
    level2_room_6e_cleared,
    level2_room_6f_compass_success,
    level2_room_7e_key_success,
    BOMB_N_STAND,
    BOOM_BOMB_N_STAND,
    BombNorthPhase,
    BoomBombNorthPhase,
    Level2BombNorthController,
    Level2BoomBombNorthController,
    ROOM_L2_BOMB_N,
    ROOM_L2_BOOM_CANDIDATE,
    ROOM_L2_ROPES_NORTH,
)
from zelda_i.dungeon_ids import object_name, ram_symbol, room_item_name
from zelda_i.dungeon_lab import LabRequest
from zelda_i.dungeon_trace import (
    TraceRecorder,
    first_trace_divergence,
    ram_delta_report,
    write_state_provenance,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_COMPASS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _room_ram(
    *,
    room: int,
    enemy_type: int = 0,
    enemies: int = 0,
    hp: int = 0,
    x: int = 120,
    y: int = 141,
    keys: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x20
    ram[ADDR_KEYS] = keys
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 80 + slot * 8
        ram[ADDR_LINK_Y + slot] = 93 + slot * 8
    return ram


def test_level2_east_of_ropes_room_id_and_boomerang_addrs() -> None:
    """Post-west-key live graph + Magical Boomerang inventory map (rr-ep8/rr-ebe)."""
    from zelda_i.dungeon import (
        ROOM_L2_COMPASS,
        ROOM_L2_EAST_KEY,
        ROOM_L2_EAST_OF_ROPES,
    )
    from zelda_i.ram import ADDR_BOOMERANG, ADDR_MAGIC_BOOMERANG

    from zelda_i.dungeon import ROOM_L2_BOMB_N, ROOM_L2_GORIYA_WEST

    assert ROOM_L2_EAST_OF_ROPES == 0x6E
    assert ROOM_L2_EAST_KEY == 0x7E  # entry-east 5 ropes + key (diamond-nav)
    assert ROOM_L2_COMPASS == 0x6F  # key-RIGHT from 0x6e (rr-c6b)
    assert ROOM_L2_BOMB_N == 0x5F  # bomb N from 0x6f @ (120,101)
    assert ROOM_L2_GORIYA_WEST == 0x5E  # key-LEFT from 0x5f
    assert ADDR_BOOMERANG == 0x0674
    assert ADDR_MAGIC_BOOMERANG == 0x0675
    assert ram_symbol(ADDR_BOOMERANG) == "boomerang"
    assert ram_symbol(ADDR_MAGIC_BOOMERANG) == "magical_boomerang"


def test_diamond_east_phase_advances() -> None:
    """Reusable diamond-blocked east door phases (nav_common)."""
    from zelda_i.nav_common import DIAMOND_BAND_6E, DIAMOND_BAND_7D, diamond_east_phase
    from zelda_i.ram import ZeldaSnapshot

    def snap(x: int, y: int) -> ZeldaSnapshot:
        return ZeldaSnapshot(
            mode=5,
            level=2,
            screen=0x7D,
            next_screen=0x7D,
            link_x=x,
            link_y=y,
            facing=1,
            sword=1,
            bombs=0,
            rupees=0,
            keys=1,
            health=0x2F,
            triforce=1,
            compass=0,
            dialog_timer=0,
            colliding_tile=0,
            room_item_id=3,
            room_all_dead=0,
            room_obj_count=0,
            cur_opened_doors=0,
            open_doorway_mask=0,
            objects=(),
        )

    _, ph = diamond_east_phase(snap(120, 157), phase="band", band_y=DIAMOND_BAND_7D)
    assert ph == "wall"
    _, ph = diamond_east_phase(snap(200, 157), phase="wall", band_y=DIAMOND_BAND_7D)
    assert ph == "door_y"
    _, ph = diamond_east_phase(
        snap(200, 141), phase="door_y", band_y=DIAMOND_BAND_7D, cycle=0
    )
    assert ph == "push"
    assert DIAMOND_BAND_6E == 113


def test_level2_room_specs_and_stop_predicates() -> None:
    """L2 Moon rooms: recon IDs + isolated pure stop predicates."""
    assert ROPE_OBJECT_TYPE == 0x28
    assert object_name(0x28) == "rope"
    assert ROOM_7D_SPEC.level == LEVEL_2
    assert ROOM_7D_SPEC.room_id == 0x7D
    assert ROOM_7D_SPEC.expected_enemy_count == 0
    assert ROOM_7D_SPEC.room_item_id == 0x03
    assert ROOM_6D_SPEC.level == LEVEL_2
    assert ROOM_6D_SPEC.room_id == 0x6D
    assert ROOM_6D_SPEC.source_room == 0x7D
    assert ROOM_6D_SPEC.enemy_types == (0x28,)
    assert ROOM_6D_SPEC.expected_enemy_count == 5
    assert ROOM_6D_SPEC.required_open_doors == 0x02
    assert ROOM_6D_SPEC.entry.direction == "UP"
    assert ROOM_6D_SPEC.combat.engage_distance == 64
    assert ROOM_6D_SPEC.combat.attack_phase == 4
    assert ROOM_6C_SPEC.room_id == 0x6C
    assert ROOM_6C_SPEC.expected_enemy_count == 6
    assert ROOM_6C_SPEC.room_item_id == 0x19
    assert ROOM_6C_SPEC.reward.inventory_field == "keys"
    assert ROOM_6C_SPEC.reward.target == (136, 141)
    assert ROOM_6C_SPEC.combat.engage_distance == 64
    assert ROOM_6C_SPEC.combat.attack_phase == 2
    assert ROOM_7E_SPEC.room_id == 0x7E
    assert ROOM_7E_SPEC.source_room == 0x7D
    assert ROOM_7E_SPEC.expected_enemy_count == 5
    assert ROOM_7E_SPEC.room_item_id == 0x19
    assert ROOM_7E_SPEC.reward.inventory_field == "keys"
    assert ROOM_7E_SPEC.reward.target == (136, 141)
    assert ROOM_7E_SPEC.entry.direction == "RIGHT"
    assert ROOM_7E_SPEC.entry.waypoints[0] == (120, 157)
    assert ROOM_7E_SPEC.combat.engage_distance == 64
    assert ROOM_7E_SPEC.combat.attack_phase == 4
    assert ROOM_6E_SPEC.room_id == 0x6E
    assert ROOM_6E_SPEC.expected_enemy_count == 3
    assert ROOM_6E_SPEC.alive_rule.value == "hp"
    assert ROOM_6F_SPEC.room_id == 0x6F
    assert ROOM_6F_SPEC.source_room == 0x6E
    assert ROOM_6F_SPEC.enemy_types == (GEL_OBJECT_TYPE,)
    assert ROOM_6F_SPEC.expected_enemy_count == 6
    assert ROOM_6F_SPEC.alive_rule.value == "type"
    assert ROOM_6F_SPEC.room_item_id == 0x16
    assert ROOM_6F_SPEC.reward.inventory_field == "compass"
    assert ROOM_6F_SPEC.reward.target == (208, 101)
    assert ROOM_6F_SPEC.reward.waypoints[0] == (192, 101)
    assert ROOM_6F_SPEC.entry.waypoints[0] == (120, 113)

    live = read_snapshot(
        _room_ram(room=0x6D, enemy_type=0x28, enemies=5, hp=0x10)
    )
    # Force level 2 on synthetic RAM for live_enemies only needs types/hp.
    assert len(ROOM_6D_SPEC.live_enemies(live)) == 5
    dead_hp = read_snapshot(
        _room_ram(room=0x6D, enemy_type=0x28, enemies=5, hp=0)
    )
    assert len(ROOM_6D_SPEC.live_enemies(dead_hp)) == 0

    clear = _room_ram(room=0x6D, enemies=0)
    clear[ADDR_LEVEL] = LEVEL_2
    clear[ADDR_ROOM_ALL_DEAD] = 20
    clear[0x00EE] = 0x02  # ADDR_CUR_OPENED_DOORS left bit
    assert dungeon_room_cleared(clear, ROOM_6D_SPEC)
    assert level2_room_6d_cleared(clear)

    # Incomplete: missing door bit or all_dead
    no_door = clear.copy()
    no_door[0x00EE] = 0
    assert not level2_room_6d_cleared(no_door)

    key_room = _room_ram(room=0x6C, enemies=0, keys=1)
    key_room[ADDR_LEVEL] = LEVEL_2
    assert level2_room_6c_key_success(key_room)
    # Key held but ropes still live → not done
    still_fight = _room_ram(
        room=0x6C, enemy_type=0x28, enemies=2, hp=0x10, keys=1
    )
    still_fight[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_6c_key_success(still_fight)
    # Cleared but no key yet
    no_key = _room_ram(room=0x6C, enemies=0, keys=0)
    no_key[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_6c_key_success(no_key)

    east_key = _room_ram(room=0x7E, enemies=0, keys=1)
    east_key[ADDR_LEVEL] = LEVEL_2
    assert level2_room_7e_key_success(east_key)
    east_live = _room_ram(
        room=0x7E, enemy_type=0x28, enemies=3, hp=0x10, keys=1
    )
    east_live[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_7e_key_success(east_live)
    east_no_key = _room_ram(room=0x7E, enemies=0, keys=0)
    east_no_key[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_7e_key_success(east_no_key)

    ropes_6e = _room_ram(room=0x6E, enemies=0)
    ropes_6e[ADDR_LEVEL] = LEVEL_2
    ropes_6e[ADDR_ROOM_ALL_DEAD] = 20
    assert level2_room_6e_cleared(ropes_6e)

    # Gels: TYPE-only (hp=0 still live); compass bit1 = level 2.
    from zelda_i.ram import ADDR_COMPASS

    gel_live = _room_ram(room=0x6F, enemy_type=0x15, enemies=6, hp=0)
    gel_live[ADDR_LEVEL] = LEVEL_2
    assert len(ROOM_6F_SPEC.live_enemies(read_snapshot(gel_live))) == 6
    compass_ok = _room_ram(room=0x6F, enemies=0)
    compass_ok[ADDR_LEVEL] = LEVEL_2
    compass_ok[ADDR_COMPASS] = 0x02  # level-2 bit
    assert level2_room_6f_compass_success(compass_ok)
    no_compass = compass_ok.copy()
    no_compass[ADDR_COMPASS] = 0
    assert not level2_room_6f_compass_success(no_compass)
    gels_remain = _room_ram(room=0x6F, enemy_type=0x15, enemies=2, hp=0)
    gels_remain[ADDR_LEVEL] = LEVEL_2
    gels_remain[ADDR_COMPASS] = 0x02
    assert not level2_room_6f_compass_success(gels_remain)

    # Bomb-north stop: play-ready on 0x5f.
    ready_5f = _room_ram(room=0x5F, enemies=0)
    ready_5f[ADDR_LEVEL] = LEVEL_2
    assert level2_room_5f_ready(ready_5f)
    still_6f = _room_ram(room=0x6F, enemies=0)
    still_6f[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_5f_ready(still_6f)
    wrong_mode = ready_5f.copy()
    wrong_mode[ADDR_MODE] = 7
    assert not level2_room_5f_ready(wrong_mode)

    # 0x5e Goriya west: 5× type 0x06 TYPE_AND_HP; CLEAR_ONLY stop.
    assert GORIYA_OBJECT_TYPE == 0x06
    assert ROOM_5E_SPEC.room_id == 0x5E
    assert ROOM_5E_SPEC.source_room == 0x5F
    assert ROOM_5E_SPEC.enemy_types == (GORIYA_OBJECT_TYPE,)
    assert ROOM_5E_SPEC.expected_enemy_count == 5
    assert ROOM_5E_SPEC.alive_rule.value == "hp"
    assert ROOM_5E_SPEC.entry.direction == "LEFT"
    assert ROOM_5E_SPEC.entry.waypoints[0] == (120, 141)
    assert ROOM_5E_SPEC.reward.kind.value == "clear"
    goriya_live = _room_ram(
        room=0x5E, enemy_type=0x06, enemies=5, hp=0x30
    )
    goriya_live[ADDR_LEVEL] = LEVEL_2
    assert len(ROOM_5E_SPEC.live_enemies(read_snapshot(goriya_live))) == 5
    goriya_hp0 = _room_ram(room=0x5E, enemy_type=0x06, enemies=5, hp=0)
    goriya_hp0[ADDR_LEVEL] = LEVEL_2
    assert len(ROOM_5E_SPEC.live_enemies(read_snapshot(goriya_hp0))) == 0
    cleared_5e = _room_ram(room=0x5E, enemies=0)
    cleared_5e[ADDR_LEVEL] = LEVEL_2
    cleared_5e[ADDR_ROOM_ALL_DEAD] = 20
    assert level2_room_5e_cleared(cleared_5e)
    assert dungeon_room_cleared(cleared_5e, ROOM_5E_SPEC)
    still_goriya = _room_ram(
        room=0x5E, enemy_type=0x06, enemies=2, hp=0x30
    )
    still_goriya[ADDR_LEVEL] = LEVEL_2
    still_goriya[ADDR_ROOM_ALL_DEAD] = 20
    assert not level2_room_5e_cleared(still_goriya)
    no_settle = cleared_5e.copy()
    no_settle[ADDR_ROOM_ALL_DEAD] = 5
    assert not level2_room_5e_cleared(no_settle)

    # 0x4e ropes north + key; 0x4f magical boomerang (rr-bsq / rr-ebe).
    assert ROOM_L2_ROPES_NORTH == 0x4E
    assert ROOM_L2_BOOM_CANDIDATE == 0x4F
    assert BLUE_GORIYA_OBJECT_TYPE == 0x05
    assert ROOM_4E_SPEC.room_id == 0x4E
    assert ROOM_4E_SPEC.source_room == 0x5E
    assert ROOM_4E_SPEC.enemy_types == (ROPE_OBJECT_TYPE,)
    assert ROOM_4E_SPEC.expected_enemy_count == 5
    assert ROOM_4E_SPEC.room_item_id == 0x19
    assert ROOM_4E_SPEC.reward.inventory_field == "keys"
    assert ROOM_4F_SPEC.room_id == 0x4F
    assert ROOM_4F_SPEC.source_room == 0x5F
    assert ROOM_4F_SPEC.enemy_types == (BLUE_GORIYA_OBJECT_TYPE,)
    assert ROOM_4F_SPEC.expected_enemy_count == 3
    assert ROOM_4F_SPEC.alive_rule.value == "hp"
    assert ROOM_4F_SPEC.room_item_id == 0x1E
    assert ROOM_4F_SPEC.reward.inventory_field == "magical_boomerang"
    assert ROOM_4F_SPEC.reward.target == (136, 135)
    assert BOOM_BOMB_N_STAND == (120, 101)

    boom_live = _room_ram(room=0x4F, enemy_type=0x05, enemies=3, hp=0x50)
    boom_live[ADDR_LEVEL] = LEVEL_2
    assert len(ROOM_4F_SPEC.live_enemies(read_snapshot(boom_live))) == 3
    boom_dead = _room_ram(room=0x4F, enemy_type=0x05, enemies=3, hp=0)
    boom_dead[ADDR_LEVEL] = LEVEL_2
    assert len(ROOM_4F_SPEC.live_enemies(read_snapshot(boom_dead))) == 0

    ready_4f = _room_ram(room=0x4F, enemies=0)
    ready_4f[ADDR_LEVEL] = LEVEL_2
    assert level2_room_4f_ready(ready_4f)
    not_4f = _room_ram(room=0x5F, enemies=0)
    not_4f[ADDR_LEVEL] = LEVEL_2
    assert not level2_room_4f_ready(not_4f)

    from zelda_i.ram import ADDR_MAGIC_BOOMERANG

    no_boom = ready_4f.copy()
    no_boom[ADDR_MAGIC_BOOMERANG] = 0
    assert not level2_room_4f_magic_boomerang_success(no_boom)
    got_boom = ready_4f.copy()
    got_boom[ADDR_MAGIC_BOOMERANG] = 1
    assert level2_room_4f_magic_boomerang_success(got_boom)
    # Inventory stop does not require still standing in 0x4f.
    got_boom_elsewhere = got_boom.copy()
    got_boom_elsewhere[ADDR_SCREEN] = 0x5F
    assert level2_room_4f_magic_boomerang_success(got_boom_elsewhere)

    # FIXED_INVENTORY success helper for 0x4e keys.
    key_4e = _room_ram(room=0x4E, enemies=0)
    key_4e[ADDR_LEVEL] = LEVEL_2
    key_4e[ADDR_KEYS] = 1
    assert level2_room_4e_key_success(key_4e)
    no_key_4e = key_4e.copy()
    no_key_4e[ADDR_KEYS] = 0
    assert not level2_room_4e_key_success(no_key_4e)


def test_level2_boom_bomb_north_controller_phases() -> None:
    """0x5f bomb-N → 0x4f: stand geometry + success on dest play."""
    assert BOOM_BOMB_N_STAND == (120, 101)

    ctrl = Level2BoomBombNorthController(clear_gels=False)
    ram = _room_ram(room=0x5F, enemies=0)
    ram[ADDR_LEVEL] = LEVEL_2
    ram[ADDR_MODE] = 5
    ram[ADDR_BOMBS] = 0
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141
    ctrl.step(read_snapshot(ram))
    assert ctrl.phase is BoomBombNorthPhase.FAILED

    ctrl2 = Level2BoomBombNorthController()
    ram4 = _room_ram(room=0x4F, enemies=0)
    ram4[ADDR_LEVEL] = LEVEL_2
    ram4[ADDR_MODE] = 5
    act = ctrl2.step(read_snapshot(ram4))
    assert ctrl2.success
    assert ctrl2.phase is BoomBombNorthPhase.DONE
    assert act.reason == "done"

    ctrl3 = Level2BoomBombNorthController(clear_gels=False)
    ram_stand = _room_ram(room=0x5F, enemies=0)
    ram_stand[ADDR_LEVEL] = LEVEL_2
    ram_stand[ADDR_MODE] = 5
    ram_stand[ADDR_BOMBS] = 3
    ram_stand[ADDR_LINK_X] = 120
    ram_stand[ADDR_LINK_Y] = 101
    s = read_snapshot(ram_stand)
    ctrl3.step(s)
    assert ctrl3.phase in (BoomBombNorthPhase.TO_STAND, BoomBombNorthPhase.FACE)
    for _ in range(10):
        if ctrl3.phase in (
            BoomBombNorthPhase.PLACE,
            BoomBombNorthPhase.WAIT,
            BoomBombNorthPhase.FAILED,
        ):
            break
        act = ctrl3.step(s)
    while ctrl3.phase is BoomBombNorthPhase.FACE:
        act = ctrl3.step(s)
    if ctrl3.phase is BoomBombNorthPhase.PLACE:
        act = ctrl3.step(s)
    assert ctrl3.phase is BoomBombNorthPhase.WAIT
    assert act.reason == "place_bomb"
    assert ctrl3.bombs_before_place == 3


def test_level2_bomb_north_controller_phases_and_fail_no_bombs() -> None:
    """Pure phase machine: stand geometry + fail cleanly without bombs."""
    assert BOMB_N_STAND == (120, 101)
    assert ROOM_L2_BOMB_N == 0x5F

    ctrl = Level2BombNorthController()
    # Already clear, bombs=0 → fail on leave settle / to_stand.
    ram = _room_ram(room=0x6F, enemies=0)
    ram[ADDR_LEVEL] = LEVEL_2
    ram[ADDR_MODE] = 5
    ram[ADDR_COMPASS] = 0x02
    ram[ADDR_BOMBS] = 0
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141

    snap = read_snapshot(ram)
    act = ctrl.step(snap)
    assert ctrl.phase is BombNorthPhase.FAILED
    assert "no_bombs" in ctrl.notes[-1] or act.reason == "no_bombs"

    # Success when already on 0x5f play.
    ctrl2 = Level2BombNorthController()
    ram5 = _room_ram(room=0x5F, enemies=0)
    ram5[ADDR_LEVEL] = LEVEL_2
    ram5[ADDR_MODE] = 5
    act2 = ctrl2.step(read_snapshot(ram5))
    assert ctrl2.success
    assert ctrl2.phase is BombNorthPhase.DONE
    assert act2.reason == "done"

    # At stand with bombs: advances FACE → PLACE (B+UP).
    ctrl3 = Level2BombNorthController()
    ram_stand = _room_ram(room=0x6F, enemies=0)
    ram_stand[ADDR_LEVEL] = LEVEL_2
    ram_stand[ADDR_MODE] = 5
    ram_stand[ADDR_COMPASS] = 0x02
    ram_stand[ADDR_BOMBS] = 4
    ram_stand[ADDR_LINK_X] = 120
    ram_stand[ADDR_LINK_Y] = 101
    # Settle → TO_STAND (clear) → at stand → FACE
    s = read_snapshot(ram_stand)
    ctrl3.step(s)
    assert ctrl3.phase in (BombNorthPhase.TO_STAND, BombNorthPhase.FACE)
    # Drive to face/place
    for _ in range(8):
        if ctrl3.phase in (
            BombNorthPhase.PLACE,
            BombNorthPhase.WAIT,
            BombNorthPhase.FAILED,
        ):
            break
        act = ctrl3.step(s)
    assert ctrl3.phase in (
        BombNorthPhase.FACE,
        BombNorthPhase.PLACE,
        BombNorthPhase.WAIT,
    )
    # Complete face frames then place
    while ctrl3.phase is BombNorthPhase.FACE:
        act = ctrl3.step(s)
    if ctrl3.phase is BombNorthPhase.PLACE:
        act = ctrl3.step(s)
    assert ctrl3.phase is BombNorthPhase.WAIT
    assert act.reason == "place_bomb"
    assert ctrl3.bombs_before_place == 4


def test_room_specs_support_hp_and_type_only_liveness() -> None:
    stalfos = read_snapshot(
        _room_ram(room=0x53, enemy_type=0x2A, enemies=5, hp=0x20)
    )
    keese = read_snapshot(
        _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    )
    assert len(ROOM_53_SPEC.live_enemies(stalfos)) == 5
    assert len(ROOM_54_SPEC.live_enemies(keese)) == 8
    assert ROOM_52_SPEC.expected_enemy_count == 6
    assert ROOM_52_SPEC.entry.direction == "LEFT"
    assert ROOM_42_SPEC.enemy_types == (0x15,)
    assert ROOM_42_SPEC.entry.direction == "UP"
    assert ROOM_43_SPEC.expected_enemy_count == 5
    assert ROOM_33_SPEC.reward.inventory_field == "keys"
    assert ROOM_33_SPEC.combat.engage_distance == 24
    assert ROOM_33_SPEC.combat.attack_phase == 4
    assert ROOM_23_SPEC.enemy_types == (0x06,)
    assert ROOM_23_SPEC.combat.engage_distance == 96
    assert ROOM_23_SPEC.combat.attack_phase == 2
    assert (176, 149) in ROOM_23_SPEC.reward.waypoints
    assert (114, 117) in ROOM_23_SPEC.reward.waypoints
    assert (96, 181) in ROOM_23_SPEC.reward.waypoints
    assert ROOM_44_SPEC.room_item_id == 0x1D
    assert ROOM_44_SPEC.combat.engage_distance == 64
    assert ROOM_44_SPEC.combat.attack_phase == 7
    assert ROOM_45_SPEC.enemy_types == (0x27,)
    assert ROOM_45_SPEC.combat.engage_distance == 80
    assert ROOM_45_SPEC.combat.engage_dominant_axis is True
    assert ROOM_45_SPEC.combat.attack_phase == 0
    assert ROOM_45_SPEC.combat.avoid_walls is False
    assert (32, 117) in ROOM_45_SPEC.combat.patrol
    assert ROOM_35_SPEC.enemy_types == (0x3D,)


def test_generic_controller_routes_and_clears_type_only_room() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    source = read_snapshot(_room_ram(room=0x53, x=120, y=109))
    action = controller.step(source)
    assert action.reason == "entry_route"

    live_ram = _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    action = controller.step(read_snapshot(live_ram))
    assert controller.phase is DungeonPhase.FIGHT
    assert action.reason.startswith("combat_")
    assert controller.max_live_enemies == 8

    clear_ram = _room_ram(room=0x54, enemies=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 20
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert controller.phase is DungeonPhase.DONE
    assert action.reason == "done"


def test_generic_controller_collects_fixed_inventory_reward() -> None:
    controller = GenericDungeonRoomController(ROOM_53_SPEC)
    controller.phase = DungeonPhase.FIGHT
    live_ram = _room_ram(
        room=0x53,
        enemy_type=0x2A,
        enemies=5,
        hp=0x20,
        keys=0,
    )
    controller.step(read_snapshot(live_ram))

    clear_ram = _room_ram(room=0x53, x=88, y=141, keys=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(clear_ram))
    assert controller.phase is DungeonPhase.COLLECT_REWARD
    assert action.reason == "collect_reward"

    clear_ram[ADDR_KEYS] = 1
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert action.reason == "done"


def test_settle_in_target_room_holds_entry_direction() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.ENTER
    ram = _room_ram(room=0x45, x=16, y=141)
    ram[ADDR_MODE] = 7
    action = controller.step(read_snapshot(ram))
    assert action.reason == "settle_target_room"


def test_room45_collects_key_after_clear() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.initial_inventory = 0
    controller.max_live_enemies = 8
    live = _room_ram(
        room=0x45, x=160, y=141, enemy_type=0x27, enemies=8, hp=0x20, keys=0
    )
    controller.step(read_snapshot(live))
    clear_ram = _room_ram(room=0x45, x=160, y=141, keys=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(clear_ram))
    assert controller.phase is DungeonPhase.COLLECT_REWARD
    assert action.reason == "collect_reward"
    clear_ram[ADDR_KEYS] = 1
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert action.reason == "done"


def test_room45_survival_collect_leaves_mid_band_via_east_column() -> None:
    """Continuous combat ends ~(132,157); x=160 south is solid, x=208 is not."""
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.clear_signal_seen = True
    ram = _room_ram(room=0x45, x=132, y=157, keys=0)
    ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(ram))
    assert action.reason == "collect_reward"
    assert controller.waypoint_index == 0
    # First Survival waypoint is (208, 157): walk RIGHT along the mid band.
    assert np.array_equal(action.action, nes_action("RIGHT"))


def test_room45_dashes_inland_from_west_door() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x45, x=16, y=141, enemy_type=0x27, enemies=8, hp=0x20
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason.startswith("leave_wall")


def test_room45_leaves_west_door_before_engage() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.combat_frames = ROOM_45_SURVIVAL_SPEC.combat.inland_dash
    ram = _room_ram(
        room=0x45, x=32, y=141, enemy_type=0x27, enemies=8, hp=0x20
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason.startswith("leave_wall")
    off_row = _room_ram(
        room=0x45, x=32, y=157, enemy_type=0x27, enemies=8, hp=0x20
    )
    off_row_action = controller.step(read_snapshot(off_row))
    assert off_row_action.reason.startswith("leave_wall")
    inland = _room_ram(
        room=0x45, x=120, y=141, enemy_type=0x27, enemies=8, hp=0x20
    )
    inland_action = controller.step(read_snapshot(inland))
    assert inland_action.reason != "leave_wall"


def test_room45_engage_does_not_walk_onto_west_wall() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.combat_frames = ROOM_45_SURVIVAL_SPEC.combat.inland_dash
    ram = _room_ram(
        room=0x45, x=56, y=141, enemy_type=0x27, enemies=1, hp=0x20
    )
    ram[ADDR_LINK_X + 1] = 32
    ram[ADDR_LINK_Y + 1] = 141
    action = controller.step(read_snapshot(ram))
    assert action.reason != "leave_wall"
    assert action.reason.startswith("combat_engage")


def test_controller_fails_fast_after_leaving_target_room() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.clear_signal_seen = True
    ram = _room_ram(room=0x73, x=160, y=173, keys=0)
    action = controller.step(read_snapshot(ram))
    assert controller.success is False
    assert controller.phase is DungeonPhase.FAILED
    assert action.reason == "left_target_room"
    assert "left_target_room" in controller.notes


def test_collect_reward_skips_waypoint_after_stuck_frames() -> None:
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    ram = _room_ram(room=0x23, x=112, y=93, keys=0)
    ram[ADDR_ROOM_ALL_DEAD] = 24
    first = controller.waypoint_index
    last_index = first
    for _ in range(30):
        controller.step(read_snapshot(ram))
        last_index = controller.waypoint_index
    assert last_index != first
    assert any(note.startswith("collect_skip_") for note in controller.notes)


def test_trace_diff_and_ram_delta_are_symbolic() -> None:
    ram = _room_ram(room=0x54)
    snap = read_snapshot(ram)
    trace = TraceRecorder(tail_frames=1)
    trace.record(
        frame=0,
        phase="FIGHT",
        reason="idle",
        action=nes_idle_action(),
        snap=snap,
    )
    trace.record(
        frame=1,
        phase="FIGHT",
        reason="move",
        action=nes_idle_action(),
        snap=snap,
    )
    assert len(trace.tail) == 1

    left = trace.frames
    right = [dict(left[0]), {**left[1], "reason": "different"}]
    divergence = first_trace_divergence(left, right)
    assert divergence is not None
    assert divergence["index"] == 1
    assert divergence["changed_fields"] == ["reason"]

    after = ram.copy()
    after[ADDR_KEYS] = 1
    after[0x0200] = 7
    report = ram_delta_report(ram, after)
    assert any(row["symbol"] == "keys" for row in report["known"])
    assert any(row["address"] == 0x0200 for row in report["unknown"])


def test_symbolic_registry_distinguishes_correlated_and_unknown_ids() -> None:
    assert object_name(0x1B) == "keese"
    assert room_item_name(0x16) == "compass_walkthrough_correlated"
    assert room_item_name(0x17) == "dungeon_map_walkthrough_correlated"
    assert room_item_name(0x1D) == "boomerang_walkthrough_correlated"
    assert room_item_name(0x1E) == "magical_boomerang_room_item_residual"
    assert object_name(0x05) == "goriya_blue_or_residual"
    assert object_name(0x55) == "fireball_or_statue_projectile"
    assert object_name(0xFE) == "unknown_object_0xfe"
    assert ram_symbol(ADDR_OBJ_TYPE + 2) == "obj_type[2]"


def test_state_provenance_hashes_source_and_output(tmp_path) -> None:
    source = tmp_path / "source.state"
    output = tmp_path / "output.state"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    sidecar = write_state_provenance(
        output,
        source_state_path=source,
        request={"room": 0x54},
        selected_trial={"success": True},
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["state_sha256"] != payload["source_state_sha256"]
    assert payload["development_only"] is True


class _Phase(Enum):
    RUNNING = auto()
    DONE = auto()


@dataclass
class _FakeController:
    success: bool = False
    phase: _Phase = _Phase.RUNNING
    frames: int = 0

    def step(self, _snap) -> FrameAction:
        self.frames += 1
        if self.frames == 2:
            self.success = True
            self.phase = _Phase.DONE
        return FrameAction(nes_idle_action(), "fake")

    def report(self) -> dict:
        return {"frames": self.frames}


class _FakeEnv:
    def __init__(self) -> None:
        self.ram = _room_ram(room=0x53)

    def get_ram(self):
        return self.ram

    def step(self, _action):
        return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, False, False, {}


def test_milestone_stage_runner_reuses_standard_loop() -> None:
    controller = _FakeController()
    obs, result = run_controller_stage(
        _FakeEnv(),
        None,
        name="fake",
        controller=controller,
        max_frames=10,
    )
    assert obs.shape == (2, 2, 3)
    assert result.success is True
    assert result.frames == 2


def test_lab_request_is_serializable() -> None:
    request = LabRequest(
        state="Level1Cleared53",
        room_id=0x54,
        alive_rule=None,
    )
    assert request.to_dict()["room_id_hex"] == "0x54"


def _combat_room_ram(
    *,
    room: int = 0x54,
    link_x: int = 120,
    link_y: int = 141,
    enemy_x: int | None = None,
    enemy_y: int | None = None,
    enemy_type: int = 0x1B,
    enemies: int = 1,
    hp: int = 0,
) -> np.ndarray:
    """Synthetic play RAM with optional one placed enemy for combat policy tests."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_HEALTH] = 0x20
    if enemies <= 0:
        return ram
    # Default far-right of Link when not specified.
    ex = link_x + 40 if enemy_x is None else enemy_x
    ey = link_y if enemy_y is None else enemy_y
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = ex + (slot - 1) * 4
        ram[ADDR_LINK_Y + slot] = ey
    return ram


def test_engage_far_enemy_walks_without_slash() -> None:
    """Chase within engage distance but outside sword reach: no A pulse."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    # Keese type-only; 40 px east is under engage_distance (48) but outside sword.
    ram = _combat_room_ram(
        room=0x54,
        link_x=120,
        link_y=141,
        enemy_x=120 + 40,
        enemy_y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
    )
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert action.reason == "combat_engage"
    assert not action.reason.endswith("_slash")
    # Several frames still no slash while approaching far target.
    for _ in range(6):
        action = controller.step(snap)
        assert action.reason == "combat_engage"
        assert "_slash" not in action.reason


def test_engage_enemy_in_sword_hitbox_slashes() -> None:
    """Enemy in blade rectangle → combat_engage_slash on attack hold frames."""
    from dataclasses import replace

    # attack_phase=0, hold covers first frames after combat_frames increments.
    tuning = replace(
        ROOM_54_SPEC.combat,
        engage_distance=64,
        engage_attack_period=8,
        engage_attack_hold=4,
        attack_phase=0,
    )
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    # 12 px east of Link, facing RIGHT when engaging → in sword hitbox.
    ram = _combat_room_ram(
        room=0x54,
        link_x=120,
        link_y=141,
        enemy_x=120 + 12,
        enemy_y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
    )
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert action.reason == "combat_engage_slash"


def test_patrol_does_not_slash() -> None:
    """Patrol (enemies beyond engage distance, or none) walks without A."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    # Enemy far beyond engage_distance (48) → patrol.
    ram = _combat_room_ram(
        room=0x54,
        link_x=120,
        link_y=141,
        enemy_x=120 + 80,
        enemy_y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
    )
    snap = read_snapshot(ram)
    for _ in range(16):
        action = controller.step(snap)
        assert action.reason == "combat_patrol"
        assert "_slash" not in action.reason
