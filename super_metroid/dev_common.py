"""Shared development-only helpers for Super Metroid mid/endgame work.

These utilities door-warp, place Samus, and boot from local save states so
room slices can be iterated without a continuous power-on prefix. Continuous
acceptance still forbids progression writes and post-boot state loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action
from retro_harness.env import make_env, read_state_bytes, write_state_bytes
from super_metroid.paths import GAME, GAME_DIR
from super_metroid.ram import (
    ADDR_DOOR_DEF_PTR,
    ADDR_GAME_STATE,
    ADDR_INVINCIBILITY_TIMER,
    ADDR_KNOCKBACK_TIMER,
    GameplayPhase,
    parse_env_state,
    write_wram_u16,
)


def make_dev_env() -> Any:
    """Create a headless RGB Super Metroid env for development probes."""
    return make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")


def boot_from_state(env: Any, source: Path, *, settle_frames: int = 5) -> Any:
    """Reset, load a save state, and idle a few frames."""
    env.reset()
    env.em.set_state(read_state_bytes(source))
    state = parse_env_state(env)
    for _ in range(settle_frames):
        env.step(idle_action())
        state = parse_env_state(env)
    return state


def door_warp(
    env: Any,
    door_ptr: int,
    *,
    settle_frames: int = 900,
    expected_room: int | None = None,
) -> Any:
    """Trigger a door transition via ``door_def_ptr`` + game state 9.

    Waits until ordinary gameplay (game state 8) so multi-screen loads finish.
    Early exit on ``phase == ordinary`` alone is too aggressive: many warps sit
    in game state 11 with ``door_transition != 0`` for 50–100+ frames.
    """
    write_wram_u16(env, ADDR_DOOR_DEF_PTR, door_ptr)
    write_wram_u16(env, ADDR_GAME_STATE, 9)
    state = parse_env_state(env)
    for frame in range(settle_frames):
        env.step(idle_action())
        state = parse_env_state(env, frame=frame)
        room_ok = expected_room is None or state.room_id == expected_room
        if (
            room_ok
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 20
        ):
            break
    return state


def place_samus(env: Any, x: int, y: int, *, camera: bool = True) -> None:
    """Teleport Samus inside the current room (development only)."""
    write_wram_u16(env, 0x0AF6, x)
    write_wram_u16(env, 0x0AFA, y)
    if camera:
        # Keep camera roughly near Samus on wide rooms.
        write_wram_u16(env, 0x0911, max(0, x - 128))
        write_wram_u16(env, 0x0915, max(0, y - 112))


def free_place_if_stuck(
    env: Any,
    x: int,
    y: int,
    *,
    settle_frames: int = 30,
) -> Any:
    """If Samus is off-map/in wall (x/y wrap), place her in free air."""
    state = parse_env_state(env)
    if state.samus_x < 60000 and state.samus_y < 60000:
        return state
    place_samus(env, x, y)
    for _ in range(settle_frames):
        apply_dev_survivability(env)
        env.step(idle_action())
        state = parse_env_state(env)
    return state


def apply_dev_survivability(env: Any) -> None:
    """Refill energy/ammo and hold invincibility for spray-and-pray probes."""
    state = parse_env_state(env)
    if state.max_health > 0:
        write_wram_u16(env, 0x09C2, state.max_health)
    if state.max_missiles > 0:
        write_wram_u16(env, 0x09C6, state.max_missiles)
    if state.max_super_missiles > 0:
        write_wram_u16(env, 0x09CA, state.max_super_missiles)
    if state.max_power_bombs > 0:
        write_wram_u16(env, 0x09CE, state.max_power_bombs)
    write_wram_u16(env, ADDR_INVINCIBILITY_TIMER, 0x7FFF)
    write_wram_u16(env, ADDR_KNOCKBACK_TIMER, 0)


def select_weapon(env: Any, selected: int) -> None:
    """Set ``selected_item`` (0 beam, 1 missiles, 2 supers, 3 power bombs)."""
    write_wram_u16(env, 0x09D2, selected)


def enemy_hps(env: Any, n: int = 8) -> list[int]:
    """Read first ``n`` enemy HP words from low WRAM."""
    ram = env.get_ram()
    return [
        int(ram[0x0F8C + i * 0x40]) | (int(ram[0x0F8D + i * 0x40]) << 8)
        for i in range(n)
    ]


def save_dev_state(env: Any, path: Path) -> Path:
    """Write the current emulator state to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_state_bytes(path, env.em.get_state())
    return path


def state_summary(env: Any, *, frame: int = 0) -> dict[str, object]:
    """Compact room/inventory snapshot for probe reports."""
    state = parse_env_state(env, frame=frame)
    return {
        "roomIdHex": f"0x{state.room_id:04X}",
        "samusX": state.samus_x,
        "samusY": state.samus_y,
        "itemsHex": f"0x{state.collected_items:04X}",
        "missiles": f"{state.missiles}/{state.max_missiles}",
        "supers": f"{state.super_missiles}/{state.max_super_missiles}",
        "powerBombs": f"{state.power_bombs}/{state.max_power_bombs}",
        "health": f"{state.health}/{state.max_health}",
        "enemy0Hp": state.enemy0_hp,
        "numEnemies": state.num_enemies,
        "pose": state.pose,
        "developmentOnly": True,
    }
