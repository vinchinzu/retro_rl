"""Super Mario Bros (NES) level configurations.

NES has 9 buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
SNES has 12:      [B, Y,    SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]

Indices 0-8 align between NES and SNES, so our 12-element action tables
work as-is -- the evaluator truncates elements 9-11 for NES envs.

SMB's X position is split across two RAM bytes (x_page * 256 + x_offset),
and level identity across world/level bytes (world * 4 + level). These are
combined via computed_values so the framework sees `player_x` and `level_id`.

NES SMB buttons:
  B (index 0) = run / fireball
  A (index 8) = jump
"""

from retro_harness.platformer.level_config import PlatformerRAM, LevelConfig, register_level
from retro_harness.platformer.actions import _make, SNES_RIGHT, SNES_LEFT, SNES_DOWN, SNES_B, SNES_A

# -- RAM layout (shared by all SMB levels) ------------------------------------

SMB_RAM = PlatformerRAM(
    # player_x and level_id are computed from extras below
    player_y=(0x00CE, "u8"),
    lives=(0x075A, "u8"),
    extras={
        "x_page": (0x006D, "u8"),       # level page (256-pixel chunks)
        "x_offset": (0x0086, "u8"),     # position within page
        "x_speed": (0x0057, "s8"),      # signed horizontal speed
        "y_speed": (0x009F, "s8"),      # signed vertical speed
        "facing": (0x0033, "u8"),       # 1=right, 2=left
        "screen_page": (0x071A, "u8"),  # camera page
        "screen_x_off": (0x071C, "u8"), # camera X within page
        "world": (0x075F, "u8"),        # world number (0-indexed)
        "level": (0x0760, "u8"),        # level number (0-indexed)
        "player_status": (0x000E, "u8"),  # 0x0B = dying
        "player_power": (0x0756, "u8"), # 0=small, 1=big, 2=fire
        "area_pointer": (0x0750, "u8"), # area/venue within a level (8-4 pipe transitions)
        "game_mode": (0x0770, "u8"),    # 0=demo, 1=playing, 2=end world, 3=game over
        "timer_hundreds": (0x07F8, "u8"),
        "timer_tens": (0x07F9, "u8"),
        "timer_ones": (0x07FA, "u8"),
    },
)

SMB_COMPUTED = {
    "player_x": lambda v: v["x_page"] * 256 + v["x_offset"],
    "level_id": lambda v: v["world"] * 4 + v["level"],
    "screen_x": lambda v: v["screen_page"] * 256 + v["screen_x_off"],
    "timer": lambda v: v["timer_hundreds"] * 100 + v["timer_tens"] * 10 + v["timer_ones"],
}

# For 8-4 segments: use area_pointer as level_id so pipe transitions
# are visible to the completion detector.
SMB_84_COMPUTED = {
    "player_x": lambda v: v["x_page"] * 256 + v["x_offset"],
    "level_id": lambda v: v["area_pointer"],
}

# -- Action table -------------------------------------------------------------
# Uses SNES button constants (indices 0-8 match NES).
# NES B = run/fire (SNES_B = 0), NES A = jump (SNES_A = 8).
# Elements 9-11 (X, L, R) are truncated by the evaluator for NES envs.

SMB_ACTIONS = [
    _make(buttons=[]),                                    # 0: NOTHING
    _make(buttons=[SNES_RIGHT]),                          # 1: RIGHT (walk)
    _make(buttons=[SNES_RIGHT, SNES_B]),                  # 2: RIGHT + B (run)
    _make(buttons=[SNES_RIGHT, SNES_B, SNES_A]),          # 3: RIGHT + B + A (run + jump)
    _make(buttons=[SNES_RIGHT, SNES_A]),                  # 4: RIGHT + A (walk + jump)
    _make(buttons=[SNES_A]),                              # 5: JUMP
    _make(buttons=[SNES_LEFT]),                           # 6: LEFT
    _make(buttons=[SNES_LEFT, SNES_B]),                   # 7: LEFT + B (run left)
    _make(buttons=[SNES_LEFT, SNES_B, SNES_A]),           # 8: LEFT + B + A (run left + jump)
    _make(buttons=[SNES_LEFT, SNES_A]),                   # 9: LEFT + A (walk left + jump)
    _make(buttons=[SNES_DOWN]),                           # 10: DOWN (duck/pipe)
]


# -- Helper to register an SMB level -----------------------------------------

def _smb_level(
    world: int,
    level: int,
    state: str,
    *aliases: str,
    completion_min_progress: float = 2500.0,
    level_id_aliases: list[int] | None = None,
    completion_level_ids: list[int] | None = None,
) -> None:
    """Register one SMB level. world/level are 1-indexed (human-readable)."""
    target_id = (world - 1) * 4 + (level - 1)
    level_id_str = f"smb_{world}_{level}"
    display = f"Super Mario Bros {world}-{level}"

    register_level(
        LevelConfig(
            level_id=level_id_str,
            display_name=display,
            game_name="SuperMarioBros-Nes-v0",
            game_dir_name="smb",
            start_state=state,
            ram=SMB_RAM,
            target_level_id=target_id,
            level_id_aliases=level_id_aliases or [],
            progress_axis="player_x",
            progress_direction=1,
            death_signals=["lives_drop"],
            completion_signal="level_id_change",
            completion_min_progress=completion_min_progress,
            completion_level_ids=completion_level_ids or [],
            action_table=SMB_ACTIONS,
            max_stall_frames=600,  # generous: underground pipe transitions take time
            computed_values=SMB_COMPUTED,
            bk2_to_env=[8 - i for i in range(9)],  # NES 9-button reversed
        ),
        *aliases,
    )


# -- Level registrations (states from stable-retro built-ins) -----------------
# SMB underground levels (1-2, 4-2) change level_id mid-level when entering
# the underground area. level_id_aliases tells the evaluator these are the
# same level, not sub-levels. Values discovered via --trace on recordings.

# Flagpole slide + score tally can stall progress for ~700f; keep max_stall high.
register_level(
    LevelConfig(
        level_id="smb_1_1",
        display_name="Super Mario Bros 1-1",
        game_name="SuperMarioBros-Nes-v0",
        game_dir_name="smb",
        start_state="Level1_1",
        ram=SMB_RAM,
        target_level_id=0,
        level_id_aliases=[],
        progress_axis="player_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=2500.0,
        completion_level_ids=[],
        action_table=SMB_ACTIONS,
        max_stall_frames=900,
        computed_values=SMB_COMPUTED,
        bk2_to_env=[8 - i for i in range(9)],
    ),
    "smb_11",
)

# 1-1 flagpole segment: from pipe exit to end of level.
# Used for focused optimization of just the flagpole approach.
register_level(
    LevelConfig(
        level_id="smb_1_1_flagpole",
        display_name="Super Mario Bros 1-1 (flagpole segment)",
        game_name="SuperMarioBros-Nes-v0",
        game_dir_name="smb",
        start_state="Level1_1_PipeExit",
        ram=SMB_RAM,
        target_level_id=0,   # world 1-1 = (1-1)*4+(1-1) = 0
        level_id_aliases=[],
        progress_axis="player_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=500.0,  # must advance from pipe exit (x=2616)
        completion_level_ids=[],
        action_table=SMB_ACTIONS,
        max_stall_frames=750,  # high flagpole grab: slide ~60f + wait ~340f + walk ~200f + level transition ~100f
        computed_values=SMB_COMPUTED,
        bk2_to_env=[8 - i for i in range(9)],
    ),
    "smb_11_flag",
)
_smb_level(1, 2, "Level1_2", "smb_1_2", "smb_12",
           completion_min_progress=0.0,
           level_id_aliases=[2],  # underground area = world*4+level = 0*4+2
           completion_level_ids=[12])  # warp zone exit must go to world 4 (4-1 = 3*4+0 = 12)
_smb_level(4, 1, "Level4_1", "smb_4_1", "smb_41",
           completion_min_progress=2500.0)
_smb_level(4, 2, "Level4_2", "smb_4_2", "smb_42",
           completion_min_progress=0.0,
           level_id_aliases=[14],  # underground area = 3*4+2
           completion_level_ids=[28])  # warp zone exit must go to world 8 (8-1 = 7*4+0 = 28)
_smb_level(8, 1, "Level8_1", "smb_8_1", "smb_81",
           completion_min_progress=2500.0)
_smb_level(8, 2, "Level8_2", "smb_8_2", "smb_82",
           completion_min_progress=2000.0)
_smb_level(8, 3, "Level8_3", "smb_8_3", "smb_83",
           completion_min_progress=2000.0)
# 8-4 uses area_pointer as level_id so pipe/venue transitions are visible.
# First pass: register the full level for discovery (play + trace to find all areas).
# Once areas are known, register individual segments below.

def _smb_84_segment(
    seg: int,
    state: str,
    start_area: int,
    aliases: list[str] | None = None,
    completion_level_ids: list[int] | None = None,
    completion_debounce_frames: int = 0,
    completion_signal: str = "level_id_change",
    completion_ram_key: str = "",
    completion_ram_value: int = 0,
    max_stall_frames: int = 600,
) -> None:
    """Register one segment of SMB 8-4."""
    level_id_str = f"smb_8_4_{seg}"
    display = f"Super Mario Bros 8-4 seg{seg}"

    register_level(
        LevelConfig(
            level_id=level_id_str,
            display_name=display,
            game_name="SuperMarioBros-Nes-v0",
            game_dir_name="smb",
            start_state=state,
            ram=SMB_RAM,
            target_level_id=start_area,
            level_id_aliases=[],
            progress_axis="player_x",
            progress_direction=1,
            death_signals=["lives_drop"],
            completion_signal=completion_signal,
            completion_min_progress=0.0,
            completion_level_ids=completion_level_ids or [],
            completion_debounce_frames=completion_debounce_frames,
            completion_ram_key=completion_ram_key,
            completion_ram_value=completion_ram_value,
            action_table=SMB_ACTIONS,
            max_stall_frames=max_stall_frames,
            computed_values=SMB_84_COMPUTED,
            bk2_to_env=[8 - i for i in range(9)],
        ),
        *(aliases or []),
    )

# Full 8-4 for chain-live and discovery.
# Uses area_pointer as level_id (pipe transitions visible).
# Completion = Bowser defeated (game_mode=2), NOT level_id_change.
# All sub-areas (pipe=0xE5, underwater=0x02) listed as aliases so
# progress tracks through the whole level.
register_level(
    LevelConfig(
        level_id="smb_8_4",
        display_name="Super Mario Bros 8-4 (full)",
        game_name="SuperMarioBros-Nes-v0",
        game_dir_name="smb",
        start_state="Level8_4",
        ram=SMB_RAM,
        target_level_id=0x65,  # starting area_pointer
        level_id_aliases=[0xE5, 0x02],  # pipe area, underwater area
        progress_axis="player_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="ram_flag",
        completion_ram_key="game_mode",
        completion_ram_value=2,
        completion_min_progress=0.0,
        action_table=SMB_ACTIONS,
        max_stall_frames=600,
        computed_values=SMB_84_COMPUTED,
        bk2_to_env=[8 - i for i in range(9)],
    ),
    "smb_84",
)

# === 8-4 SEGMENTS ===
# Area sequence: 0x65 → 0xE5 → 0x65 → 0x02 → 0x65 → 0xE5(end)
# 1-frame glitches at transitions need debounce=5 for 0xE5/0x65 boundaries.

# Seg 1: Castle start → first pipe (0x65 → 0xE5)
_smb_84_segment(1, "Level8_4", 0x65, ["smb_841"],
                completion_level_ids=[0xE5], completion_debounce_frames=5)
# Seg 2: Pipe section → castle (0xE5 → 0x65)
_smb_84_segment(2, "Level8_4_seg2", 0xE5, ["smb_842"],
                completion_level_ids=[0x65], completion_debounce_frames=5)
# Seg 3: Castle maze → underwater (0x65 → 0x02, no glitches to 0x02)
_smb_84_segment(3, "Level8_4_seg3", 0x65, ["smb_843"],
                completion_level_ids=[0x02])
# Seg 4: Underwater → castle (0x02 → 0x65, clean transition)
_smb_84_segment(4, "Level8_4_seg4", 0x02, ["smb_844"],
                completion_level_ids=[0x65], max_stall_frames=900)
# Seg 5: Final castle + Bowser (0x65 → game_mode=2 when Bowser defeated)
_smb_84_segment(5, "Level8_4_seg5", 0x65, ["smb_845"],
                completion_signal="ram_flag",
                completion_ram_key="game_mode", completion_ram_value=2)


# -- Any% route (warp zones: 1-2→W4, 4-2→W8) --------------------------------

from retro_harness.platformer.route import RouteConfig, RouteSegment, register_route

register_route(
    RouteConfig(
        route_id="smb_any_percent",
        display_name="Super Mario Bros Any% (Warp Zone)",
        segments=[
            RouteSegment("smb_1_1",     label="1-1"),
            RouteSegment("smb_1_2",     label="1-2 (→W4)"),
            RouteSegment("smb_4_1",     label="4-1"),
            RouteSegment("smb_4_2",     label="4-2 (→W8)"),
            RouteSegment("smb_8_1",     label="8-1"),
            RouteSegment("smb_8_2",     label="8-2"),
            RouteSegment("smb_8_3",     label="8-3"),
            # 8-4: human recording segs 1-4 + neuro plays Bowser fight live
            RouteSegment("smb_8_4",     label="8-4",
                         neuro_checkpoint="../smb_8_4_5/neuro/neuro_best.json"),
        ],
    ),
    "smb_any",
    "smb",
)

# Showcase / stitch routes (video builder lives in smb.full_run; these keep
# retro_harness.platformer aware of the same exit lists).
register_route(
    RouteConfig(
        route_id="smb_warp_any_percent",
        display_name="Super Mario Bros Any% (Warp → 8 Exit)",
        segments=[
            RouteSegment("smb_1_1", label="1-1"),
            RouteSegment("smb_1_2", label="1-2 (→W4)"),
            RouteSegment("smb_4_1", label="4-1"),
            RouteSegment("smb_4_2", label="4-2 (→W8)"),
            RouteSegment("smb_8_1", label="8-1"),
            RouteSegment("smb_8_2", label="8-2"),
            RouteSegment("smb_8_3", label="8-3"),
            RouteSegment("smb_8_4", label="8-4"),
        ],
    ),
    "warp",
    "warp8",
)

# All 32 main-game exits (1-1 … 8-4). Levels beyond the warp route still need
# LevelConfig registrations + recordings before optimizer chain-video works;
# smb.full_run already stitches whatever legal_stitch / optimizer sources exist.
_all_exit_segments = [
    RouteSegment(f"smb_{world}_{level}", label=f"{world}-{level}")
    for world in range(1, 9)
    for level in range(1, 5)
]
register_route(
    RouteConfig(
        route_id="smb_all_exits",
        display_name="Super Mario Bros All 32 Exits",
        segments=_all_exit_segments,
    ),
    "all_exits",
    "smb_100",
)
