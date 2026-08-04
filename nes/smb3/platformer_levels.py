"""Super Mario Bros. 3 (NES) platformer LevelConfig segments (game-owned).

Registers into ``retro_harness.platformer.level_config.LEVEL_REGISTRY`` on
import.

NES buttons (fceumm): [B, null, Select, Start, Up, Down, Left, Right, A]
  B (0) = run
  A (8) = jump

Progress uses coarse horizontal units at 0x75 (8-block steps) combined with
on-screen X at 0x90. Completion is the auto-control flag set when Mario hits
the goal card (0x0559 != 0) after enough progress.
"""

from __future__ import annotations

from retro_harness.platformer.actions import (
    SNES_A,
    SNES_B,
    SNES_DOWN,
    SNES_LEFT,
    SNES_RIGHT,
    _make,
)
from retro_harness.platformer.level_config import LevelConfig, PlatformerRAM, register_level

# Indices 0-8 match NES; evaluator truncates trailing SNES slots.
SMB3_RAM = PlatformerRAM(
    player_y=(0x00A2, "u8"),
    lives=(0x0736, "u8"),
    extras={
        "x_page": (0x0075, "u8"),  # coarse horizontal (8-block units in levels)
        "x_offset": (0x0090, "u8"),  # on-screen X
        "world": (0x0727, "u8"),  # world number - 1
        "in_air": (0x00D8, "u8"),
        "hvel": (0x00BD, "u8"),
        "auto_control": (0x0559, "u8"),  # non-zero during goal / cutscene
        "return_map": (0x0014, "u8"),
        "form": (0x0746, "u8"),
    },
)

SMB3_COMPUTED = {
    # Coarse page can jump to map values (~0x20+) on death/return; clamp for progress.
    "player_x": lambda v: (
        (v["x_page"] * 256 + v["x_offset"]) if v["x_page"] < 0x18 else 0
    ),
    "level_id": lambda v: int(v["world"]),
    # Goal grab / auto walk sets this; treat any non-zero as complete flag bit.
    "goal_auto": lambda v: 1 if v["auto_control"] != 0 else 0,
}

SMB3_ACTIONS = [
    _make(buttons=[]),  # 0: NOTHING
    _make(buttons=[SNES_RIGHT]),  # 1: RIGHT
    _make(buttons=[SNES_RIGHT, SNES_B]),  # 2: RIGHT + B (run)
    _make(buttons=[SNES_RIGHT, SNES_B, SNES_A]),  # 3: run + jump
    _make(buttons=[SNES_RIGHT, SNES_A]),  # 4: walk + jump
    _make(buttons=[SNES_A]),  # 5: JUMP
    _make(buttons=[SNES_LEFT]),  # 6: LEFT
    _make(buttons=[SNES_LEFT, SNES_B]),  # 7: LEFT + B
    _make(buttons=[SNES_LEFT, SNES_B, SNES_A]),  # 8: LEFT + B + A
    _make(buttons=[SNES_LEFT, SNES_A]),  # 9: LEFT + A
    _make(buttons=[SNES_DOWN]),  # 10: DOWN (pipe)
]


register_level(
    LevelConfig(
        level_id="smb3_1_1",
        display_name="Super Mario Bros. 3 World 1-1",
        game_name="SuperMarioBros3-Nes",
        game_dir_name="smb3",
        start_state="Level1_1",
        ram=SMB3_RAM,
        target_level_id=0,  # world 0
        progress_axis="player_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="ram_flag",
        completion_ram_key="goal_auto",
        completion_ram_value=1,
        # ~2816 px level; page unit ~256px → clear past ~2000
        completion_min_progress=1800.0,
        action_table=SMB3_ACTIONS,
        max_stall_frames=450,
        computed_values=SMB3_COMPUTED,
        bk2_to_env=[8 - i for i in range(9)],
        population_size=40,
        num_generations=80,
    ),
    "smb3_11",
    "smb3",
)
