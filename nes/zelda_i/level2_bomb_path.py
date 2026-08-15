"""Level 2 bomb-wall path factories over ``bomb_wall_path.BombWallController``.

Canonical API: ``make_*_controller()`` only. Geometry from
``level2_puzzles.BombWall``; inventory poke constants live in ``dungeon_ops``.
"""

from __future__ import annotations

from zelda_i.bomb_wall_path import (
    BOMB_N_MAX_FRAMES,
    BOMB_N_STAND_TOL,
    BOMB_N_STEP_BACK,
    BOMB_N_WAIT_BLAST,
    BombNorth1EPhase,
    BombNorthPhase,
    BombWallController,
    BombWallPhase,
    BoomBombNorthPhase,
    PostBoomBombNorthPhase,
)
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import ADDR_SELECTED_ITEM, B_ITEM_BOMB
from zelda_i.level2_dungeon import (
    LEVEL_2,
    ROOM_6F_SPEC,
    ROOM_L2_BOMB_N,
    ROOM_L2_COMPASS,
)
from zelda_i.level2_puzzles import (
    BOMB_WALL_1E_NORTH,
    BOMB_WALL_4F_NORTH,
    BOMB_WALL_5F_NORTH,
    BOMB_WALL_6F_NORTH,
)
from zelda_i.ram import ZeldaSnapshot

# Shared stand (all L2 north bomb walls).
BOMB_N_STAND = BOMB_WALL_6F_NORTH.stand  # (120, 101)
BOOM_BOMB_N_STAND = BOMB_N_STAND
BOOM_BOMB_N_MAX_FRAMES = BOMB_N_MAX_FRAMES

# Probe-local gel clear on 0x5f before bomb-N to boom (not a STATUS room).
_ROOM_5F_GEL_CLEAR = DungeonRoomSpec(
    spec_id="level2_room5f_gel_clear_for_boom",
    source_room=ROOM_L2_COMPASS,
    room_id=ROOM_L2_BOMB_N,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=(
            (120, 141),
            (168, 141),
            (168, 109),
            (120, 109),
            (72, 109),
            (72, 141),
            (72, 173),
            (120, 173),
            (168, 173),
            (120, 141),
        ),
        engage_distance=56,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=10),
    max_frames=10000,
    level=LEVEL_2,
)


def _need_clear_6f(snap: ZeldaSnapshot) -> bool:
    """Compass room: clear gels or missing L2 compass bit before bomb-N."""
    level_bit = 1 << (LEVEL_2 - 1)
    gels = ROOM_6F_SPEC.live_enemies(snap)
    return bool(gels) or (snap.compass & level_bit) == 0


def make_bomb_north_controller() -> BombWallController:
    """0x6f compass → bomb north → 0x5f (Clean geometry; no inventory poke)."""
    return BombWallController(
        wall=BOMB_WALL_6F_NORTH,
        level=LEVEL_2,
        clear_spec=ROOM_6F_SPEC,
        clear_when=_need_clear_6f,
        face_frames=4,
        step_back=BOMB_N_STEP_BACK,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=True,
        wait_hold_face=False,
    )


def make_boom_bomb_north_controller(
    *, clear_gels: bool = True
) -> BombWallController:
    """0x5f → bomb north → 0x4f boom room."""
    return BombWallController(
        wall=BOMB_WALL_5F_NORTH,
        level=LEVEL_2,
        clear_spec=_ROOM_5F_GEL_CLEAR if clear_gels else None,
        clear_when=(
            (lambda s: bool(_ROOM_5F_GEL_CLEAR.live_enemies(s)))
            if clear_gels
            else None
        ),
        face_frames=4,
        step_back=BOMB_N_STEP_BACK,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=True,
        wait_hold_face=False,
    )


def make_bomb_north_5f_controller() -> BombWallController:
    """0x5f geometry-only hop (no gel clear) → 0x4f."""
    return make_boom_bomb_north_controller(clear_gels=False)


def make_post_boom_bomb_north_controller() -> BombWallController:
    """0x4f (boom collected) → bomb north → 0x3f traps+Keese."""
    return BombWallController(
        wall=BOMB_WALL_4F_NORTH,
        level=LEVEL_2,
        clear_spec=None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
    )


def make_bomb_north_1e_controller() -> BombWallController:
    """0x1e cleared Goriya → bomb north → Dodongo 0x0e."""
    return BombWallController(
        wall=BOMB_WALL_1E_NORTH,
        level=LEVEL_2,
        clear_spec=None,
        south_band_first=True,
        south_band_y=170,
        south_band_max_frames=80,
        south_center_max_frames=200,
        stand_tol=12,
        approach_waypoints=(
            (96, 189),
            (176, 189),
            (176, 93),
            (120, 93),
        ),
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=12000,
    )


# Back-compat aliases (class-shaped names → make_*). Prefer make_* in new code.
Level2BombNorthController = make_bomb_north_controller
Level2BoomBombNorthController = make_boom_bomb_north_controller
Level2BombNorth5FController = make_bomb_north_5f_controller
Level2PostBoomBombNorthController = make_post_boom_bomb_north_controller
Level2BombNorth1EController = make_bomb_north_1e_controller


__all__ = [
    "ADDR_SELECTED_ITEM",
    "B_ITEM_BOMB",
    "BOMB_N_MAX_FRAMES",
    "BOMB_N_STAND",
    "BOMB_N_STAND_TOL",
    "BOMB_N_STEP_BACK",
    "BOMB_N_WAIT_BLAST",
    "BOOM_BOMB_N_MAX_FRAMES",
    "BOOM_BOMB_N_STAND",
    "BombNorth1EPhase",
    "BombNorthPhase",
    "BombWallController",
    "BombWallPhase",
    "BoomBombNorthPhase",
    "Level2BombNorth1EController",
    "Level2BombNorth5FController",
    "Level2BombNorthController",
    "Level2BoomBombNorthController",
    "Level2PostBoomBombNorthController",
    "PostBoomBombNorthPhase",
    "make_bomb_north_1e_controller",
    "make_bomb_north_5f_controller",
    "make_bomb_north_controller",
    "make_boom_bomb_north_controller",
    "make_post_boom_bomb_north_controller",
]
