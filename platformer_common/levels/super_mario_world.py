"""Super Mario World level configurations.

This is the first-pass SMW integration for the shared platformer tooling.
It starts with stable-retro's built-in level states and a RAM layout backed by
SMWDisX symbols. Chained any% states can be published under SMW later without
changing the shared runner.
"""

from platformer_common.actions import (
    SNES_A,
    SNES_B,
    SNES_DOWN,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_UP,
    SNES_Y,
    _make,
)
from platformer_common.level_config import LevelConfig, PlatformerRAM, register_level


# SMW is a LoROM SNES game. Addresses here are offsets inside stable-retro's
# WRAM array, so $7E00D1 is represented as 0x00D1.
SMW_RAM = PlatformerRAM(
    camera_x=(0x001A, "u16"),        # Layer 1 X scroll
    camera_y=(0x001C, "u16"),        # Layer 1 Y scroll
    player_x=(0x00D1, "u16"),        # Mario X position, current frame
    player_y=(0x00D3, "u16"),        # Mario Y position, current frame
    lives=(0x0DBE, "s8"),            # Current player's lives
    level_id=(0x0100, "u8"),         # GameMode; 0x14 while in active level
    timer_frames=(0x0F30, "u8"),
    extras={
        "true_frame": (0x0013, "u8"),
        "effective_frame": (0x0014, "u8"),
        "powerup": (0x0019, "u8"),
        "player_animation": (0x0071, "u8"),
        "player_in_air": (0x0072, "u8"),
        "player_direction": (0x0076, "u8"),
        "player_blocked_dir": (0x0077, "u8"),
        "player_x_speed": (0x007A, "s16"),
        "player_y_speed": (0x007C, "s16"),
        "player_x_next": (0x0094, "u16"),
        "player_y_next": (0x0096, "u16"),
        "coins": (0x0DBF, "u8"),
        "item_box": (0x0DC2, "u8"),
        "level_timer_hundreds": (0x0F31, "u8"),
        "level_timer_tens": (0x0F32, "u8"),
        "level_timer_ones": (0x0F33, "u8"),
        "translevel": (0x13BF, "u8"),
        "current_submap": (0x13C3, "u8"),
        "midway_flag": (0x13CE, "u8"),
        "p_meter": (0x13E4, "u8"),
        "on_ground": (0x13EF, "u8"),
        "active_boss": (0x13FC, "u8"),
        "camera_scrolling": (0x13FD, "u8"),
    },
)


# SMW speed table. Y is run/grab/fire; B is normal jump; A is spin jump.
SMW_SPEED_ACTIONS = [
    _make(buttons=[]),                            # 0: nothing
    _make(buttons=[SNES_RIGHT, SNES_Y]),          # 1: run right
    _make(buttons=[SNES_RIGHT, SNES_Y, SNES_B]),  # 2: run+jump right
    _make(buttons=[SNES_RIGHT, SNES_B]),          # 3: walk+jump right
    _make(buttons=[SNES_B]),                      # 4: normal jump
    _make(buttons=[SNES_RIGHT, SNES_Y, SNES_A]),  # 5: run+spin right
    _make(buttons=[SNES_A]),                      # 6: spin jump
    _make(buttons=[SNES_LEFT, SNES_Y]),           # 7: run left
    _make(buttons=[SNES_LEFT, SNES_Y, SNES_B]),   # 8: run+jump left
    _make(buttons=[SNES_LEFT, SNES_B]),           # 9: walk+jump left
    _make(buttons=[SNES_DOWN]),                   # 10: duck / pipe
    _make(buttons=[SNES_UP]),                     # 11: climb / door
    _make(buttons=[SNES_RIGHT]),                  # 12: right without Y
    _make(buttons=[SNES_LEFT]),                   # 13: left without Y
]


_LEVEL_GAME_MODE = 0x14
_OVERWORLD_MODES = [0x0B, 0x0C, 0x0D, 0x0E]


def _smw_level(
    level_id: str,
    display_name: str,
    start_state: str,
    *aliases: str,
    completion_min_progress: float = 1200.0,
    progress_axis: str = "camera_x",
    progress_direction: int = 1,
) -> None:
    """Register a stable-retro SMW level state."""
    register_level(
        LevelConfig(
            level_id=level_id,
            display_name=display_name,
            game_name="SuperMarioWorld-Snes-v0",
            game_dir_name="SMW",
            start_state=start_state,
            ram=SMW_RAM,
            target_level_id=_LEVEL_GAME_MODE,
            progress_axis=progress_axis,  # type: ignore[arg-type]
            progress_direction=progress_direction,  # type: ignore[arg-type]
            death_signals=["lives_drop"],
            completion_signal="level_id_change",
            completion_min_progress=completion_min_progress,
            completion_level_ids=_OVERWORLD_MODES,
            completion_exclude_ids=[0x15, 0x16, 0x17],
            action_table=SMW_SPEED_ACTIONS,
            max_stall_frames=600,
        ),
        *aliases,
    )


_smw_level("smw_yoshi_island_1", "Super Mario World - Yoshi's Island 1", "YoshiIsland1", "yi1", "smw_yi1")
_smw_level("smw_yoshi_island_2", "Super Mario World - Yoshi's Island 2", "YoshiIsland2", "yi2", "smw_yi2")
_smw_level("smw_yoshi_island_3", "Super Mario World - Yoshi's Island 3", "YoshiIsland3", "yi3", "smw_yi3")
_smw_level("smw_yoshi_island_4", "Super Mario World - Yoshi's Island 4", "YoshiIsland4", "yi4", "smw_yi4")

_smw_level("smw_donut_plains_1", "Super Mario World - Donut Plains 1", "DonutPlains1", "dp1", "smw_dp1")
_smw_level("smw_donut_plains_2", "Super Mario World - Donut Plains 2", "DonutPlains2", "dp2", "smw_dp2")
_smw_level("smw_donut_plains_3", "Super Mario World - Donut Plains 3", "DonutPlains3", "dp3", "smw_dp3")
_smw_level("smw_donut_plains_4", "Super Mario World - Donut Plains 4", "DonutPlains4", "dp4", "smw_dp4")
_smw_level("smw_donut_plains_5", "Super Mario World - Donut Plains 5", "DonutPlains5", "dp5", "smw_dp5")

_smw_level("smw_vanilla_dome_1", "Super Mario World - Vanilla Dome 1", "VanillaDome1", "vd1", "smw_vd1")
_smw_level("smw_vanilla_dome_2", "Super Mario World - Vanilla Dome 2", "VanillaDome2", "vd2", "smw_vd2")
_smw_level("smw_vanilla_dome_3", "Super Mario World - Vanilla Dome 3", "VanillaDome3", "vd3", "smw_vd3")
_smw_level("smw_vanilla_dome_4", "Super Mario World - Vanilla Dome 4", "VanillaDome4", "vd4", "smw_vd4")
_smw_level("smw_vanilla_dome_5", "Super Mario World - Vanilla Dome 5", "VanillaDome5", "vd5", "smw_vd5")

_smw_level("smw_bridges_1", "Super Mario World - Bridges 1", "Bridges1", "br1", "smw_br1")
_smw_level("smw_bridges_2", "Super Mario World - Bridges 2", "Bridges2", "br2", "smw_br2")

_smw_level("smw_forest_1", "Super Mario World - Forest 1", "Forest1", "forest1", "smw_forest1")
_smw_level("smw_forest_2", "Super Mario World - Forest 2", "Forest2", "forest2", "smw_forest2")
_smw_level("smw_forest_3", "Super Mario World - Forest 3", "Forest3", "forest3", "smw_forest3")
_smw_level("smw_forest_4", "Super Mario World - Forest 4", "Forest4", "forest4", "smw_forest4")
_smw_level("smw_forest_5", "Super Mario World - Forest 5", "Forest5", "forest5", "smw_forest5")

_smw_level("smw_chocolate_island_1", "Super Mario World - Chocolate Island 1", "ChocolateIsland1", "ci1", "smw_ci1")
_smw_level("smw_chocolate_island_2", "Super Mario World - Chocolate Island 2", "ChocolateIsland2", "ci2", "smw_ci2")
_smw_level("smw_chocolate_island_3", "Super Mario World - Chocolate Island 3", "ChocolateIsland3", "ci3", "smw_ci3")
