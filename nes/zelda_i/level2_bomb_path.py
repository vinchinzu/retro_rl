"""Level 2 bomb-wall path factories over ``bomb_wall_path.BombWallController``.

Canonical API: ``make_*_controller()`` only. Geometry from
``level2_puzzles.BombWall``; inventory poke constants live in ``dungeon_ops``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
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
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

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


# Bow-splice clear1e leftover (120, 117). Spine v8 waypoint (120, 93) is
# the closed bomb wall — UP no-op 12000f. Peel west at north-band y, then
# stand (120, 101). Do not walk to y=93 before the blast.
BOMB_1E_SPINE_APPROACH: tuple[tuple[int, int], ...] = (
    (96, 117),
    (96, 101),
    (120, 101),
)


# v9 leftover (96, 101): west peel reached stand Y, cardinal RIGHT solid
# (Goriya / north-wall face). RIGHT+UP slides the wall toward the stand.
_STAND_CLIP_X = (80, 110)
_STAND_CLIP_Y = (93, 109)


@dataclass
class Level2BombNorth1eSpineController:
    """West peel to (96,101), then RIGHT+UP clip to the 0x1e bomb stand."""

    inner: BombWallController = field(init=False)
    dest_room: int = 0x0E

    def __post_init__(self) -> None:
        self.inner = make_bomb_north_1e_controller(
            approach_waypoints=BOMB_1E_SPINE_APPROACH
        )

    @property
    def max_frames(self) -> int:
        return self.inner.max_frames

    @property
    def success(self) -> bool:
        return self.inner.success

    @property
    def phase(self):
        return self.inner.phase

    @property
    def approach_waypoints(self):
        return self.inner.approach_waypoints

    @property
    def to_room(self) -> int:
        return self.inner.to_room

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        if (
            snap.mode == PLAY_MODE
            and snap.screen == 0x1E
            and not snap.transitioning
            and _STAND_CLIP_X[0] <= x <= _STAND_CLIP_X[1]
            and _STAND_CLIP_Y[0] <= y <= _STAND_CLIP_Y[1]
            and not self.inner._at_stand(snap)
        ):
            self.inner.step(snap)
            return FrameAction(nes_action("RIGHT", "UP"), "stand_clip")
        return self.inner.step(snap)

    def report(self) -> dict[str, Any]:
        payload = self.inner.report()
        payload["policy"] = (
            "west peel (96,117)->(96,101); RIGHT+UP clip to stand (120,101)"
        )
        return payload


def make_bomb_north_1e_controller(
    *,
    approach_waypoints: tuple[tuple[int, int], ...] | None = None,
) -> BombWallController:
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
        approach_waypoints=approach_waypoints
        if approach_waypoints is not None
        else (
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
    "BOMB_1E_SPINE_APPROACH",
    "BOOM_BOMB_N_MAX_FRAMES",
    "BOOM_BOMB_N_STAND",
    "BombNorth1EPhase",
    "BombNorthPhase",
    "BombWallController",
    "BombWallPhase",
    "BoomBombNorthPhase",
    "Level2BombNorth1EController",
    "Level2BombNorth1eSpineController",
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
