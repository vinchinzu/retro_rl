"""Development helpers for Kraid fight and post-Kraid route toward Phantoon.

Development-only: door-warps and placement teleports. Continuous acceptance
still requires natural entry from the power-on prefix.

Proven so far (dev states under custom_integrations/SuperMetroid-Snes/):

- ``dev_kraid_eye_at_eye.state`` — Kraid Eye Door with Supers, door open side
- ``dev_kraid_room_natural.state`` — inside Kraid arena after door-warp + place
- ``dev_kraid_defeated.state`` — Kraid dead (Brinstar boss bits include bit 0)
- ``dev_varia_equipped_dev.state`` — Varia equipped (PLM collect flaky after warp)

Fight recipe: face right mid-arena, pulse Super Missiles; body HP 1000→0.

Shared primitives: :mod:`super_metroid.dev.common`.
Ship route / Power Bombs / Phantoon: :mod:`super_metroid.dev.phantoon_dev`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import (
    boot_from_state,
    door_warp,
    enemy_hps,
    make_dev_env,
    place_samus,
    save_dev_state,
    select_weapon,
)
from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import GameplayPhase, parse_env_state, parse_state, read_bank7e_wram

VARIA_MASK = 0x0001
ROOM_KRAID_EYE = 0xA56B
ROOM_KRAID = 0xA59F
ROOM_VARIA = 0xA6E2
DOOR_EYE_TO_KRAID = 0x91B6
DOOR_KRAID_TO_VARIA = 0x91DA

EYE_STATE = INTEGRATION_DIR / "dev_kraid_eye_at_eye.state"
KRAID_NATURAL = INTEGRATION_DIR / "dev_kraid_room_natural.state"
KRAID_DEFEATED = INTEGRATION_DIR / "dev_kraid_defeated.state"
VARIA_STATE = INTEGRATION_DIR / "dev_varia_equipped_dev.state"


def brinstar_boss_bits(env: Any) -> int:
    return int(read_bank7e_wram(env)[0xD829])


def kraid_defeated(env: Any) -> bool:
    """True when Brinstar boss bit 0 is set (Kraid)."""
    return bool(brinstar_boss_bits(env) & 0x01)


def door_warp_to_kraid(env: Any, source: Path = EYE_STATE) -> Any:
    """Load eye-door inventory state and door-warp into Kraid's Room."""
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, source)
    for _ in range(3):
        state = parse_env_state(env)
        assist.apply(env.data, state)
        env.step(idle_action())
    state = door_warp(env, DOOR_EYE_TO_KRAID, expected_room=ROOM_KRAID)
    if state.room_id != ROOM_KRAID or state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
        raise RuntimeError(f"failed to warp into Kraid: room 0x{state.room_id:04X}")
    # Warp lands in the left wall; place on the arena floor.
    place_samus(env, 120, 395)
    for _ in range(15):
        state = parse_env_state(env)
        assist.apply(env.data, state)
        env.step(idle_action())
    return parse_env_state(env)


def fight_kraid_action(state, frame: int, *, dead: bool) -> list[int]:
    """Controller action for one frame of the Kraid spray-and-pray fight."""
    x = state.samus_x
    if x > 60000:
        return idle_action()
    if dead:
        if state.pose in (137, 138):
            return buttons("LEFT") if (frame // 30) % 2 == 0 else buttons("RIGHT", "A", "B")
        return buttons("RIGHT", "B", "A")
    if x > 280:
        return buttons("LEFT", "B")
    if x < 50:
        return buttons("RIGHT", "B")
    if frame % 50 < 10:
        return buttons("RIGHT", "A", "X")
    if frame % 12 < 6:
        return buttons("RIGHT", "X")
    return buttons("RIGHT")


def run_kraid_fight(
    *,
    max_frames: int = 9000,
    source: Path = EYE_STATE,
    save_defeated: Path = KRAID_DEFEATED,
) -> dict[str, object]:
    """Door-warp into Kraid and spray Supers until Brinstar boss bit 0 sets."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        door_warp_to_kraid(env, source)
        save_dev_state(env, KRAID_NATURAL)
        dead_frame = None
        min_body = 10**9
        body_zero_frame: int | None = None
        for frame in range(max_frames):
            st = parse_state(env.get_ram(), frame=frame)
            assist.apply(env.data, st)
            select_weapon(env, 2)  # Super Missiles
            body = enemy_hps(env)[0]
            if body < 60000:
                min_body = min(min_body, body)
            if body == 0 and body_zero_frame is None:
                body_zero_frame = frame
            # Boss bit lags HP-zero by a few hundred frames (death animation).
            dead = kraid_defeated(env)
            dying = body == 0 or dead
            if dead and dead_frame is None:
                dead_frame = frame
                save_dev_state(env, save_defeated)
                break
            if st.samus_x > 60000:
                place_samus(env, 120, 395)
            env.step(fight_kraid_action(st, frame, dead=dying))
        st = parse_env_state(env)
        return {
            "success": dead_frame is not None,
            "deadFrame": dead_frame,
            "bodyZeroFrame": body_zero_frame,
            "minBodyHp": min_body if min_body < 10**9 else None,
            "bossBitsBrinstar": brinstar_boss_bits(env),
            "finalRoomIdHex": f"0x{st.room_id:04X}",
            "itemsHex": f"0x{st.collected_items:04X}",
            "statePath": str(save_defeated.resolve()) if dead_frame is not None else None,
            "developmentOnly": True,
        }
    finally:
        env.close()


def try_natural_varia(
    *,
    source: Path = KRAID_DEFEATED,
    output: Path = VARIA_STATE,
    max_frames: int = 3600,
) -> dict[str, object]:
    """From defeated Kraid, attempt rear-door exit + Varia PLM collect.

    Camera/door unlock after Kraid death is still flaky; this records whether
    the Varia bit (``0x0001``) sets without a capacity write.
    """
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        collected_frame: int | None = None
        rooms: list[str] = []
        last_room = None
        for frame in range(max_frames):
            state = parse_env_state(env, frame=frame)
            assist.apply(env.data, state)
            if state.room_id != last_room:
                rooms.append(f"0x{state.room_id:04X}")
                last_room = state.room_id
            if state.collected_items & VARIA_MASK:
                collected_frame = frame
                save_dev_state(env, output)
                break
            # After death: push right toward rear door / Varia.
            if state.room_id == ROOM_KRAID:
                if frame % 60 < 20:
                    action = buttons("RIGHT", "B", "A")
                elif frame % 60 < 40:
                    action = buttons("RIGHT", "B")
                else:
                    action = buttons("RIGHT", "X")
            elif state.room_id == ROOM_VARIA:
                action = buttons("RIGHT", "B")
            else:
                action = buttons("RIGHT", "B")
            if state.samus_x > 60000 and state.room_id == ROOM_KRAID:
                place_samus(env, 280, 395)
            env.step(action)
        state = parse_env_state(env)
        return {
            "success": collected_frame is not None,
            "collectedFrame": collected_frame,
            "itemsHex": f"0x{state.collected_items:04X}",
            "finalRoomIdHex": f"0x{state.room_id:04X}",
            "rooms": rooms,
            "statePath": str(output.resolve()) if collected_frame is not None else None,
            "developmentOnly": True,
        }
    finally:
        env.close()
