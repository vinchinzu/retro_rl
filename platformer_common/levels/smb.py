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

from platformer_common.level_config import PlatformerRAM, LevelConfig, register_level
from platformer_common.actions import _make, SNES_RIGHT, SNES_LEFT, SNES_DOWN, SNES_B, SNES_A

# -- RAM layout (shared by all SMB levels) ------------------------------------

SMB_RAM = PlatformerRAM(
    # player_x and level_id are computed from extras below
    player_y=(0x00CE, "u8"),
    lives=(0x075A, "u8"),
    extras={
        "x_page": (0x006D, "u8"),       # level page (256-pixel chunks)
        "x_offset": (0x0086, "u8"),     # position within page
        "world": (0x075F, "u8"),        # world number (0-indexed)
        "level": (0x0760, "u8"),        # level number (0-indexed)
        "player_status": (0x000E, "u8"),  # 0x0B = dying
    },
)

SMB_COMPUTED = {
    "player_x": lambda v: v["x_page"] * 256 + v["x_offset"],
    "level_id": lambda v: v["world"] * 4 + v["level"],
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
            game_dir_name="super_mario_bros",
            start_state=state,
            ram=SMB_RAM,
            target_level_id=target_id,
            level_id_aliases=level_id_aliases or [],
            progress_axis="player_x",
            progress_direction=1,
            death_signals=["lives_drop"],
            completion_signal="level_id_change",
            completion_min_progress=completion_min_progress,
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

_smb_level(1, 1, "Level1_1", "smb_1_1", "smb_11",
           completion_min_progress=2500.0)
_smb_level(1, 2, "Level1_2", "smb_1_2", "smb_12",
           completion_min_progress=0.0,
           level_id_aliases=[2])  # underground area = world*4+level = 0*4+2
_smb_level(4, 1, "Level4_1", "smb_4_1", "smb_41",
           completion_min_progress=2500.0)
_smb_level(4, 2, "Level4_2", "smb_4_2", "smb_42",
           completion_min_progress=0.0,
           level_id_aliases=[14])  # underground area = 3*4+2 (TBD: verify with --trace)
_smb_level(8, 1, "Level8_1", "smb_8_1", "smb_81",
           completion_min_progress=2500.0)
_smb_level(8, 2, "Level8_2", "smb_8_2", "smb_82",
           completion_min_progress=2000.0)
_smb_level(8, 3, "Level8_3", "smb_8_3", "smb_83",
           completion_min_progress=2000.0)
