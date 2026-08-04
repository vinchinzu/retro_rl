"""Donkey Kong Country platformer LevelConfig segments (game-owned).

Registers into ``retro_harness.platformer.level_config.LEVEL_REGISTRY`` on
import. All DKC levels share the same RAM layout (DKC_RAM). Adding a new
level is just a register_level() call with a different start_state and
target_level_id.
"""

from retro_harness.platformer.level_config import PlatformerRAM, LevelConfig, register_level
from retro_harness.platformer.actions import _make, SNES_RIGHT, SNES_LEFT, SNES_UP, SNES_DOWN, SNES_B, SNES_Y

# Shared RAM layout for all DKC levels
DKC_RAM = PlatformerRAM(
    camera_x=(0x00B2, "u16"),       # Absolute camera X (monotonic through level)
    player_x=(0x00B4, "u16"),       # Camera-relative X (wraps, NOT for progress)
    player_y=(0x00B6, "u16"),
    lives=(0x0575, "u8"),
    level_id=(0x003E, "u8"),        # Stable level ID (0x76 is volatile)
    timer_frames=(0x0046, "u16"),
    timer_minutes=(0x0048, "u16"),
    extras={"bonus_timer": (0x13F3, "u8")},
)

# DKC speed action table: prioritizes Y (run button) but allows Y-release
# for cartwheel re-taps (RIGHT+Y → RIGHT → RIGHT+Y = cartwheel attack).
# Removes useless walk+jump combos (RIGHT+B, LEFT+B) and A button.
# Running (Y) = 2.4 px/frame. Walking without Y = 0 px/frame.
DKC_SPEED_ACTIONS = [
    _make(buttons=[]),                              # 0: NOTHING
    _make(buttons=[SNES_RIGHT, SNES_Y]),            # 1: RIGHT + Y (run right)
    _make(buttons=[SNES_RIGHT, SNES_Y, SNES_B]),   # 2: RIGHT + Y + B (run + jump)
    _make(buttons=[SNES_B]),                        # 3: JUMP (precision)
    _make(buttons=[SNES_LEFT, SNES_Y]),             # 4: LEFT + Y (run left)
    _make(buttons=[SNES_LEFT, SNES_Y, SNES_B]),    # 5: LEFT + Y + B (run left + jump)
    _make(buttons=[SNES_DOWN]),                     # 6: DOWN (duck/dismount)
    _make(buttons=[SNES_UP]),                       # 7: UP (enter door)
    _make(buttons=[SNES_RIGHT]),                    # 8: RIGHT (release Y for cartwheel re-tap)
    _make(buttons=[SNES_LEFT]),                     # 9: LEFT (release Y for cartwheel re-tap)
    _make(buttons=[SNES_RIGHT, SNES_B]),            # 10: RIGHT + B (walk-jump right, no run)
    _make(buttons=[SNES_LEFT, SNES_B]),             # 11: LEFT + B (walk-jump left, no run)
]

# Mapping from old 14-action DEFAULT_PLATFORMER_ACTIONS indices to DKC_SPEED_ACTIONS indices.
# Used to convert seeds built with the old table.
OLD_TO_SPEED = {
    0: 0,   # NOTHING → NOTHING
    1: 8,   # RIGHT → RIGHT (keep as Y-release)
    2: 1,   # RIGHT+Y → RIGHT+Y
    3: 2,   # RIGHT+Y+B → RIGHT+Y+B
    4: 2,   # RIGHT+B → RIGHT+Y+B (upgrade walk+jump to run+jump)
    5: 3,   # B → B
    6: 9,   # LEFT → LEFT (keep as Y-release)
    7: 4,   # LEFT+Y → LEFT+Y
    8: 5,   # LEFT+Y+B → LEFT+Y+B
    9: 5,   # LEFT+B → LEFT+Y+B (upgrade walk+jump to run+jump)
    10: 6,  # DOWN → DOWN
    11: 0,  # A → NOTHING (no A in speed table)
    12: 1,  # RIGHT+A → RIGHT+Y (convert to run right)
    13: 7,  # UP → UP
}


def convert_old_to_speed(actions: list[int]) -> list[int]:
    """Convert action sequence from old 14-action table to DKC_SPEED_ACTIONS."""
    return [OLD_TO_SPEED.get(a, 0) for a in actions]


# -- Winky's Walkway ---------------------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_winkys_walkway",
        display_name="Winky's Walkway",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="WinkysWalkway",
        ram=DKC_RAM,
        target_level_id=0xD9,      # 217
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=4000.0,  # min progress before checking completion
        completion_level_ids=[0x2E],   # real exit to overworld (bonus rooms are 0x51 etc)
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "winkys",
    "dkc_winkys",
    "winkys_walkway",
)

# -- Ropey Rampage (World 1-2) ------------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_ropey_rampage",
        display_name="Ropey Rampage",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="RopeyRampage",
        ram=DKC_RAM,
        target_level_id=0x0C,      # 12
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=5000.0,  # level progress ~7000; require ~70%
        # Bonus rooms use 0xEE (entered/exited multiple times mid-level).
        # Real exit likely 0x72 (level complete screen, same as Jungle Hijinxs).
        completion_exclude_ids=[0xEE],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "ropey_rampage",
    "dkc_ropey",
    "ropey",
)

# -- Jungle Hijinxs (World 1-1) -----------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_jungle_hijinks",
        display_name="Jungle Hijinxs",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="JungleHijinks",
        ram=DKC_RAM,
        target_level_id=0x16,      # 22 (SAME as W1 overworld!)
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=4000.0,  # level is ~5300 units; require ~75% progress
        # Level shares level_id 0x16 with W1 overworld.
        # Bonus rooms: 0x25, 0x06 (Rambi bonus cave).
        # Real exit: 0x72 (level complete screen).
        completion_level_ids=[0x72],
        completion_exclude_ids=[0x25, 0x06],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "jungle_hijinks",
    "dkc_jungle",
    "jungle_hijinxs",
)


# -- Reptile Rumble (World 1-3) -----------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_reptile_rumble",
        display_name="Reptile Rumble",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="ReptileRumble",
        ram=DKC_RAM,
        target_level_id=0x01,      # 1
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=2000.0,  # level ~2500 progress; exits to 0xBF (Coral Capers)
        completion_exclude_ids=[0x25, 0xEE],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "reptile_rumble",
    "dkc_reptile",
    "reptile",
)

# -- Coral Capers (World 1-4) ------------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_coral_capers",
        display_name="Coral Capers",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="CoralCapers",
        ram=DKC_RAM,
        target_level_id=0xBF,      # 191
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=3000.0,
        completion_exclude_ids=[0x25, 0xEE],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "coral_capers",
    "dkc_coral",
    "coral",
)

# -- Barrel Cannon Canyon (World 1-5) ----------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_barrel_cannon_canyon",
        display_name="Barrel Cannon Canyon",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="BarrelCannonCanyon",
        ram=DKC_RAM,
        target_level_id=0x17,      # 23
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=3000.0,
        completion_exclude_ids=[0x25, 0xEE, 0xFA],  # 0xFA = bonus room
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "barrel_cannon",
    "dkc_barrel",
    "barrel_cannon_canyon",
)

# -- Stop & Go Station (World 2-1) -------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_stop_and_go_station",
        display_name="Stop & Go Station",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="StopAndGoStation",
        ram=DKC_RAM,
        target_level_id=0x31,      # 49
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=3000.0,
        completion_exclude_ids=[0x25, 0xEE, 0xCD],  # 0xCD = bonus room
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "stop_and_go",
    "dkc_stop_go",
    "stop_go",
)

# -- Mine Cart Carnage (World 2-2) --------------------------------------------
# Autoscroller: mine cart moves automatically, player mainly jumps.
# Camera progresses without player input, but stall detection still works.

register_level(
    LevelConfig(
        level_id="dkc_mine_cart_carnage",
        display_name="Mine Cart Carnage",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="MineCartCarnage",
        ram=DKC_RAM,
        target_level_id=0x2E,      # 46
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=3000.0,
        completion_exclude_ids=[0x25, 0xEE],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "mine_cart",
    "dkc_mine_cart",
    "mine_cart_carnage",
)


# -- Bouncy Bonanza (World 2-3) -----------------------------------------------

register_level(
    LevelConfig(
        level_id="dkc_bouncy_bonanza",
        display_name="Bouncy Bonanza",
        game_name="DonkeyKongCountry-Snes",
        game_dir_name="donkey_kong_country",
        start_state="BouncyBonanza",
        ram=DKC_RAM,
        target_level_id=0x07,
        progress_axis="camera_x",
        progress_direction=1,
        death_signals=["lives_drop"],
        completion_signal="level_id_change",
        completion_min_progress=3000.0,
        completion_exclude_ids=[0x25, 0xEE],
        max_stall_frames=360,
        action_table=DKC_SPEED_ACTIONS,
    ),
    "bouncy_bonanza",
    "dkc_bouncy",
    "bouncy",
)
