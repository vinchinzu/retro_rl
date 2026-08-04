"""Super Metroid platformer LevelConfig segments (game-owned).

Registers into ``retro_harness.platformer.level_config.LEVEL_REGISTRY`` on
import. Primary SM progression still lives under ``super_metroid`` adventure /
route stacks; these configs are the platformer-optimizer view of early rooms.

All SM segments share the same RAM layout (SM_RAM). Each segment covers
one room transition on the ZebesStart -> Bomb Torizo route.

Progress is tracked via player_x/player_y (Samus's absolute position)
rather than camera scroll, since SM rooms are not horizontally scrolling
levels like DKC.
"""

from retro_harness.platformer.level_config import PlatformerRAM, LevelConfig, register_level
from retro_harness.platformer.actions import NUM_BUTTONS

# -- SNES button indices (retro env order) -----------------------------------
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)

# Shared RAM layout for all SM segments
SM_RAM = PlatformerRAM(
    # SM doesn't use camera for progress; player position is absolute
    player_x=(0x0AF6, "u16"),       # Samus X position
    player_y=(0x0AFA, "u16"),       # Samus Y position
    level_id=(0x079B, "u16"),       # Current room ID (2-byte pointer)
    extras={
        "health": (0x09C2, "u16"),  # Samus energy (health)
        "max_health": (0x09C4, "u16"),
        "missiles": (0x09C6, "u16"),
        "max_missiles": (0x09C8, "u16"),
        "super_missiles": (0x09CA, "u16"),
        "max_super_missiles": (0x09CC, "u16"),
        "game_state": (0x0998, "u16"),
        "door_transition": (0x0797, "u16"),
    },
)

# SM discrete action table (matches train_curriculum.py DISCRETE_ACTIONS)
# 26 actions covering directional movement, jumping, shooting, morph ball
SM_ACTIONS: list[list[int]] = []


def _sm_make(button_dict: dict[int, int]) -> list[int]:
    """Create a 12-element action array from a {button_idx: 1} dict."""
    action = [0] * NUM_BUTTONS
    for idx, val in button_dict.items():
        action[idx] = val
    return action


# Build SM_ACTIONS matching DISCRETE_ACTIONS from train_curriculum.py
_SM_ACTION_DEFS = [
    {_LEFT: 1}, {_RIGHT: 1},                                           # 0-1: move
    {_LEFT: 1, _X: 1}, {_RIGHT: 1, _X: 1},                           # 2-3: run + shoot
    {_X: 1},                                                           # 4: shoot
    {_UP: 1, _X: 1}, {_UP: 1, _LEFT: 1, _X: 1}, {_UP: 1, _RIGHT: 1, _X: 1},  # 5-7: aim up
    {_A: 1, _X: 1}, {_A: 1}, {_A: 1, _LEFT: 1}, {_A: 1, _RIGHT: 1},  # 8-11: jump
    {_B: 1, _LEFT: 1}, {_B: 1, _RIGHT: 1},                           # 12-13: dash
    {_DOWN: 1}, {_DOWN: 1, _X: 1}, {_DOWN: 1, _LEFT: 1}, {_DOWN: 1, _RIGHT: 1},  # 14-17: down
    {_A: 1, _UP: 1, _X: 1}, {_A: 1, _LEFT: 1, _X: 1}, {_A: 1, _RIGHT: 1, _X: 1},  # 18-20: jump+aim
    {_B: 1, _LEFT: 1, _X: 1}, {_B: 1, _RIGHT: 1, _X: 1},            # 21-22: dash+shoot
    {_B: 1, _A: 1, _LEFT: 1}, {_B: 1, _A: 1, _RIGHT: 1},            # 23-24: dash+jump
    {},                                                                # 25: nothing
    {_SELECT: 1},                                                      # 26: select (weapon toggle)
]
SM_ACTIONS = [_sm_make(d) for d in _SM_ACTION_DEFS]


# -- Room IDs from world_map.json (hex string -> int) -----------------------
_ROOMS = {
    "Landing Site": 0x91F8,
    "Parlor and Alcatraz": 0x92FD,
    "Climb": 0x96BA,
    "Pit Room": 0x975C,
    "Blue Brinstar Elevator Room": 0x97B5,
    "Morph Ball Room": 0x9E9F,
    "Construction Zone": 0x9F11,
    "First Missile Room": 0xA107,
    "Flyway": 0x9879,
    "Bomb Torizo Room": 0x9804,
}


def _sm_config(
    level_id: str,
    display_name: str,
    start_state: str,
    start_room_id: int,
    exit_room_id: int = 0,
    progress_axis: str = "player_y",
    progress_direction: int = 1,
    max_stall_frames: int = 360,
    waypoints: list[tuple[float, float]] | None = None,
) -> LevelConfig:
    """Convenience constructor for SM segments (all share common defaults).

    Args:
        start_room_id: Room ID where Samus starts.
        exit_room_id: Room ID that counts as completion. If 0, any room change
            is accepted (for simple rooms with only one exit).
    """
    return LevelConfig(
        level_id=level_id,
        display_name=display_name,
        game_name="SuperMetroid-Snes",
        game_dir_name="super_metroid",
        start_state=start_state,
        ram=SM_RAM,
        target_level_id=start_room_id,
        progress_axis="waypoints" if waypoints else progress_axis,
        progress_direction=progress_direction,
        waypoints=waypoints or [],
        death_signals=["health_zero"],
        # The published SM start states are route anchors, not easy death labs.
        # Skip the generic platformer death probe unless a real sequence is published.
        selftest_expect_death=False,
        completion_signal="level_id_change",
        completion_min_progress=0.0,
        completion_level_ids=[exit_room_id] if exit_room_id else [],
        action_table=SM_ACTIONS,
        max_stall_frames=max_stall_frames,
    )


# =============================================================================
# DESCENT PHASE: Landing Site -> Morph Ball
# =============================================================================

register_level(
    _sm_config(
        "sm_landing_site",
        "Landing Site -> Parlor",
        "ZebesStart",
        _ROOMS["Landing Site"],
        exit_room_id=_ROOMS["Parlor and Alcatraz"],
        progress_axis="player_x",
        progress_direction=-1,  # going left
        # Auto-generated from navigation/waypoint_gen: go left then down to door
        waypoints=[
            (1100, 900), (896, 896), (640, 896),
            (640, 1152), (384, 1152), (8, 1152),
        ],
    ),
    "landing_site",
    "sm_landing",
)

register_level(
    _sm_config(
        "sm_parlor_descent",
        "Parlor -> Climb (descent)",
        "Parlor and Alcatraz [from Landing Site]",
        _ROOMS["Parlor and Alcatraz"],
        exit_room_id=_ROOMS["Climb"],
        progress_axis="waypoints",
        progress_direction=1,
        # Parlor 5x5: enter top-right, go left across top, fall down left side
        # Traced from recording: (1272,139)→(972,80)→(656,187)→(353,171)→(407,235)→(437,390)→(427,492)→(403,729)→(393,1248)
        waypoints=[
            (1272, 139), (972, 100), (656, 170), (400, 170),
            (420, 350), (430, 500), (420, 730), (400, 930),
            (393, 1248),
        ],
    ),
    "parlor_descent",
    "sm_parlor_down",
)

register_level(
    _sm_config(
        "sm_climb_descent",
        "Climb -> Pit Room (descent)",
        "Climb [from Parlor and Alcatraz]",
        _ROOMS["Climb"],
        exit_room_id=_ROOMS["Pit Room"],
        progress_axis="waypoints",
        progress_direction=1,
        # Climb 3x9: enter top, fall straight down at x~475
        # Traced: (393,41)→(475,152)→(475,349)→...→(475,1859)→(493,2187)
        waypoints=[
            (400, 50), (475, 200), (475, 450), (475, 700),
            (475, 950), (475, 1200), (475, 1500), (475, 1800),
            (490, 2100),
        ],
    ),
    "climb_descent",
    "sm_climb_down",
)

register_level(
    _sm_config(
        "sm_pit_room_descent",
        "Pit Room -> Elevator (descent)",
        "Pit Room [from Climb]",
        _ROOMS["Pit Room"],
        exit_room_id=_ROOMS["Blue Brinstar Elevator Room"],
        progress_axis="player_y",
        progress_direction=1,  # going down
    ),
    "pit_room_descent",
    "sm_pit_down",
)

register_level(
    _sm_config(
        "sm_elevator_descent",
        "Elevator -> Morph Ball Room",
        "Blue Brinstar Elevator Room [from Pit Room]",
        _ROOMS["Blue Brinstar Elevator Room"],
        exit_room_id=_ROOMS["Morph Ball Room"],
        progress_axis="player_y",
        progress_direction=1,  # going down
    ),
    "elevator_descent",
    "sm_elevator_down",
)

register_level(
    _sm_config(
        "sm_morph_ball_collect",
        "Morph Ball Room -> Collect",
        "Morph Ball Room [from Blue Brinstar Elevator Room]",
        _ROOMS["Morph Ball Room"],
        exit_room_id=_ROOMS["Construction Zone"],
        progress_axis="player_x",
        progress_direction=1,  # going right to item
    ),
    "morph_ball_collect",
    "sm_morph_collect",
)

# =============================================================================
# MISSILE DETOUR: Morph Ball -> Construction Zone -> First Missile -> back
# =============================================================================

register_level(
    _sm_config(
        "sm_morph_to_construction",
        "Morph Ball -> Construction Zone",
        "Morph Ball Room [from Blue Brinstar Elevator Room]",
        _ROOMS["Morph Ball Room"],
        exit_room_id=_ROOMS["Construction Zone"],
        progress_axis="player_x",
        progress_direction=1,  # going right through room
    ),
    "morph_to_construction",
    "sm_morph_constr",
)

register_level(
    _sm_config(
        "sm_construction_to_missile",
        "Construction Zone -> First Missile Room",
        "Construction Zone [from Morph Ball Room]",
        _ROOMS["Construction Zone"],
        exit_room_id=_ROOMS["First Missile Room"],
        progress_axis="player_y",
        progress_direction=-1,  # going up
    ),
    "construction_to_missile",
    "sm_constr_missile",
)

register_level(
    _sm_config(
        "sm_missile_to_construction",
        "First Missile Room -> Construction Zone",
        "First Missile Room [from Construction Zone]",
        _ROOMS["First Missile Room"],
        exit_room_id=_ROOMS["Construction Zone"],
        progress_axis="player_x",
        progress_direction=-1,  # going left/back
    ),
    "missile_to_construction",
    "sm_missile_constr",
)

register_level(
    _sm_config(
        "sm_construction_to_morph",
        "Construction Zone -> Morph Ball Room",
        "Construction Zone [from First Missile Room]",
        _ROOMS["Construction Zone"],
        exit_room_id=_ROOMS["Morph Ball Room"],
        progress_axis="player_y",
        progress_direction=1,  # going down/back
    ),
    "construction_to_morph",
    "sm_constr_morph",
)

# =============================================================================
# RETURN PHASE: Morph Ball -> Bomb Torizo
# =============================================================================

register_level(
    _sm_config(
        "sm_morph_ball_return",
        "Morph Ball -> Elevator (return)",
        "Morph Ball Room [from Construction Zone]",
        _ROOMS["Morph Ball Room"],
        exit_room_id=_ROOMS["Blue Brinstar Elevator Room"],
        progress_axis="player_y",
        progress_direction=-1,  # going up (Y decreases)
    ),
    "morph_ball_return",
    "sm_morph_return",
)

register_level(
    _sm_config(
        "sm_elevator_return",
        "Elevator -> Pit Room (return)",
        "Blue Brinstar Elevator Room [from Morph Ball Room]",
        _ROOMS["Blue Brinstar Elevator Room"],
        exit_room_id=_ROOMS["Pit Room"],
        progress_axis="player_y",
        progress_direction=-1,  # going up
    ),
    "elevator_return",
    "sm_elevator_up",
)

register_level(
    _sm_config(
        "sm_pit_room_return",
        "Pit Room -> Climb (return)",
        "Pit Room [from Blue Brinstar Elevator Room]",
        _ROOMS["Pit Room"],
        exit_room_id=_ROOMS["Climb"],
        progress_axis="player_x",
        progress_direction=-1,  # spawn right (x~783), walk left to exit
    ),
    "pit_room_return",
    "sm_pit_up",
)

register_level(
    _sm_config(
        "sm_climb_return",
        "Climb -> Parlor (return)",
        "Climb [from Pit Room]",
        _ROOMS["Climb"],
        exit_room_id=_ROOMS["Parlor and Alcatraz"],
        progress_axis="waypoints",
        progress_direction=1,
        max_stall_frames=600,  # tall room, long climb
        # Climb return: spawn at bottom (521,2187) after door transition, climb up
        # Traced: (521,2187)→(400,2020)→(340,1860)→(340,1500)→(320,1090)→(360,730)→(330,480)→(340,210)→(340,67)
        waypoints=[
            (500, 2187), (400, 2020), (340, 1860), (340, 1500),
            (320, 1090), (360, 730), (330, 480), (340, 210),
            (340, 67),
        ],
    ),
    "climb_return",
    "sm_climb_up",
)

register_level(
    _sm_config(
        "sm_parlor_to_flyway",
        "Parlor -> Flyway",
        "Parlor and Alcatraz [from Climb]",
        _ROOMS["Parlor and Alcatraz"],
        exit_room_id=_ROOMS["Flyway"],
        progress_axis="player_x",
        progress_direction=1,  # going right
    ),
    "parlor_to_flyway",
    "sm_parlor_flyway",
)

register_level(
    _sm_config(
        "sm_flyway_to_torizo",
        "Flyway -> Bomb Torizo",
        "Flyway [from Parlor and Alcatraz]",
        _ROOMS["Flyway"],
        exit_room_id=_ROOMS["Bomb Torizo Room"],
        progress_axis="player_x",
        progress_direction=1,  # going right
    ),
    "flyway_to_torizo",
    "sm_flyway_torizo",
)
