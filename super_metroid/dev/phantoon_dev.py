"""Development helpers: Power Bombs → ship route → Phantoon.

Development-only: door-warps, placement teleports, and optional capacity
grants for route scaffolding. Continuous acceptance still requires natural
item collection and natural room entry.

Route (with Power Bombs):

```text
Pink Brinstar Power Bomb Room 0x9E11
→ Big Pink → GHZ → Noob Bridge → Red Tower
→ Hellway → Caterpillar → Elevator → Crateria Kihunter
→ Moat → West Ocean → Wrecked Ship → Phantoon 0xCD13
```

Door pointers are bank ``$83`` definitions that land in the destination room.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import (
    apply_dev_survivability,
    boot_from_state,
    door_warp,
    enemy_hps,
    free_place_if_stuck,
    make_dev_env,
    place_samus,
    save_dev_state,
    select_weapon,
    state_summary,
)
from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import parse_env_state, read_bank7e_wram, write_wram_u16

# Rooms
ROOM_BIG_PINK = 0x9D19
ROOM_PINK_PB = 0x9E11
ROOM_RED_TOWER = 0xA253
ROOM_HELLWAY = 0xA2F7
ROOM_CATERPILLAR = 0xA322
ROOM_ELEV_TO_CAT = 0x962A
ROOM_CRATERIA_KIHUNTER = 0x948C
ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE
ROOM_WS_ENTRANCE = 0xCA08
ROOM_WS_MAIN = 0xCAF6
ROOM_WS_BASEMENT = 0xCC6F
ROOM_PHANTOON = 0xCD13

# Bank $83 door defs → destination room
DOOR_BIG_PINK_TO_PB_TOP = 0x8DDE
DOOR_BIG_PINK_TO_PB_BOTTOM = 0x8E02
DOOR_PB_BOTTOM_TO_BIG_PINK = 0x8E6E
DOOR_RED_TOWER_TO_HELLWAY = 0x901E
DOOR_HELLWAY_TO_CATERPILLAR = 0x908A
DOOR_CATERPILLAR_TO_ELEV = 0x90BA
DOOR_ELEV_TO_CRATERIA_KIHUNTER = 0x8AF6
DOOR_CRATERIA_KIHUNTER_TO_MOAT = 0x8A36
DOOR_MOAT_TO_WEST_OCEAN = 0x8AEA
DOOR_WEST_OCEAN_TO_WS = 0x89D6
DOOR_WS_ENTRANCE_TO_MAIN = 0xA1BC
DOOR_WS_MAIN_TO_BASEMENT = 0xA21C
DOOR_WS_BASEMENT_TO_PHANTOON = 0xA2AC

# Ship route hops after Power Bombs (dev door-warp chain).
SHIP_ROUTE: tuple[tuple[str, int, int, int, int], ...] = (
    ("hellway", DOOR_RED_TOWER_TO_HELLWAY, ROOM_HELLWAY, 80, 180),
    ("caterpillar", DOOR_HELLWAY_TO_CATERPILLAR, ROOM_CATERPILLAR, 100, 180),
    ("elev_to_cat", DOOR_CATERPILLAR_TO_ELEV, ROOM_ELEV_TO_CAT, 128, 100),
    ("crateria_kihunter", DOOR_ELEV_TO_CRATERIA_KIHUNTER, ROOM_CRATERIA_KIHUNTER, 200, 180),
    ("moat", DOOR_CRATERIA_KIHUNTER_TO_MOAT, ROOM_MOAT, 100, 180),
    ("west_ocean", DOOR_MOAT_TO_WEST_OCEAN, ROOM_WEST_OCEAN, 200, 200),
    ("ws_entrance", DOOR_WEST_OCEAN_TO_WS, ROOM_WS_ENTRANCE, 100, 180),
    ("ws_main", DOOR_WS_ENTRANCE_TO_MAIN, ROOM_WS_MAIN, 128, 300),
    ("ws_basement", DOOR_WS_MAIN_TO_BASEMENT, ROOM_WS_BASEMENT, 200, 180),
    ("phantoon", DOOR_WS_BASEMENT_TO_PHANTOON, ROOM_PHANTOON, 140, 180),
)

BIG_PINK_STATE = INTEGRATION_DIR / "dev_big_pink_mainshaft.state"
RED_TOWER_STATE = INTEGRATION_DIR / "dev_red_tower_stable.state"
PB_COLLECTED_STATE = INTEGRATION_DIR / "dev_power_bombs_collected.state"
PHANTOON_ENTRY_STATE = INTEGRATION_DIR / "dev_phantoon_entry.state"
PHANTOON_DEFEATED_STATE = INTEGRATION_DIR / "dev_phantoon_defeated.state"


def wrecked_ship_boss_bits(env: Any) -> int:
    """Wrecked Ship boss bits (area index 3) at ``$7E:D82B``."""
    return int(read_bank7e_wram(env)[0xD82B])


def phantoon_defeated(env: Any) -> bool:
    """True when Wrecked Ship boss bit 0 is set (Phantoon)."""
    return bool(wrecked_ship_boss_bits(env) & 0x01)


def grant_power_bombs_dev(env: Any, *, capacity: int = 5) -> None:
    """Development-only Power Bomb capacity grant (not continuous-legal)."""
    write_wram_u16(env, 0x09D0, capacity)
    write_wram_u16(env, 0x09CE, capacity)


def collect_power_bombs(
    *,
    source: Path = BIG_PINK_STATE,
    output: Path = PB_COLLECTED_STATE,
    door: int = DOOR_BIG_PINK_TO_PB_TOP,
    item_x: int = 120,
    item_y: int = 380,
    max_frames: int = 600,
) -> dict[str, object]:
    """Door-warp into Pink PB room, place near the item, wait for capacity.

    PLM collect works reliably when Samus is placed on the item tile after a
    clean door-warp (unlike post-Kraid Varia after a messy arena exit).
    """
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        state = door_warp(env, door, expected_room=ROOM_PINK_PB)
        if state.room_id != ROOM_PINK_PB:
            raise RuntimeError(
                f"expected Pink PB 0x{ROOM_PINK_PB:04X}, got 0x{state.room_id:04X}"
            )
        place_samus(env, item_x, item_y)
        collected_frame: int | None = None
        for frame in range(max_frames):
            state = parse_env_state(env, frame=frame)
            assist.apply(env.data, state)
            apply_dev_survivability(env)
            if state.max_power_bombs > 0 and collected_frame is None:
                collected_frame = frame
                if state.game_state == 8 and frame > collected_frame:
                    break
            if collected_frame is not None and state.game_state == 8 and frame > collected_frame + 20:
                break
            env.step(buttons("LEFT") if frame % 20 < 10 else idle_action())
        state = parse_env_state(env)
        if state.max_power_bombs <= 0:
            raise RuntimeError("Power Bomb capacity still zero after probe")
        save_dev_state(env, output)
        summary = state_summary(env)
        summary.update(
            {
                "success": True,
                "collectedFrame": collected_frame,
                "statePath": str(output.resolve()),
            }
        )
        return summary
    finally:
        env.close()


def door_warp_ship_route(
    env: Any,
    *,
    place_free: bool = True,
    save_hops: bool = True,
) -> list[dict[str, object]]:
    """Door-warp the Red Tower → Phantoon ship route in-place on ``env``."""
    hops: list[dict[str, object]] = []
    for name, door, room, px, py in SHIP_ROUTE:
        state = door_warp(env, door, expected_room=room)
        if state.room_id != room:
            hops.append(
                {
                    "name": name,
                    "success": False,
                    "expectedRoomHex": f"0x{room:04X}",
                    "gotRoomHex": f"0x{state.room_id:04X}",
                    "gameState": state.game_state,
                }
            )
            break
        if place_free:
            free_place_if_stuck(env, px, py)
            if parse_env_state(env).samus_x > 60000:
                place_samus(env, px, py)
                for _ in range(20):
                    apply_dev_survivability(env)
                    env.step(idle_action())
        if save_hops:
            save_dev_state(env, INTEGRATION_DIR / f"dev_route_{name}.state")
        hops.append(
            {
                "name": name,
                "success": True,
                "roomIdHex": f"0x{room:04X}",
                **{k: v for k, v in state_summary(env).items() if k != "developmentOnly"},
            }
        )
    return hops


def capture_phantoon_entry(
    *,
    source: Path = RED_TOWER_STATE,
    output: Path = PHANTOON_ENTRY_STATE,
    grant_pbs: bool = True,
) -> dict[str, object]:
    """From Red Tower (or similar), grant PBs if needed and warp to Phantoon."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        for _ in range(3):
            state = parse_env_state(env)
            assist.apply(env.data, state)
            env.step(idle_action())
        if grant_pbs and parse_env_state(env).max_power_bombs <= 0:
            grant_power_bombs_dev(env)
        hops = door_warp_ship_route(env, place_free=True, save_hops=True)
        state = parse_env_state(env)
        if state.room_id != ROOM_PHANTOON:
            return {
                "success": False,
                "hops": hops,
                "finalRoomIdHex": f"0x{state.room_id:04X}",
                "developmentOnly": True,
            }
        place_samus(env, 140, 180)
        for _ in range(30):
            apply_dev_survivability(env)
            env.step(idle_action())
        save_dev_state(env, output)
        summary = state_summary(env)
        summary.update(
            {
                "success": True,
                "hops": hops,
                "statePath": str(output.resolve()),
                "bossBitsWreckedShip": wrecked_ship_boss_bits(env),
            }
        )
        return summary
    finally:
        env.close()


def fight_phantoon_action(state: Any, frame: int, *, enemy_x: int, enemy_y: int) -> list[int]:
    """Track Phantoon and spray missiles; open-eye damage still WIP."""
    if state.samus_x > 60000:
        return idle_action()
    face_right = enemy_x >= state.samus_x
    names: list[str] = ["RIGHT" if face_right else "LEFT"]
    if enemy_y + 20 < state.samus_y:
        names.append("UP")
    if frame % 90 < 12:
        names.append("A")
    names.extend(["B", "X"])
    return buttons(*names)


def run_phantoon_fight(
    *,
    source: Path = PHANTOON_ENTRY_STATE,
    save_defeated: Path = PHANTOON_DEFEATED_STATE,
    max_frames: int = 12000,
    capture_if_missing: bool = True,
) -> dict[str, object]:
    """Load Phantoon entry and spray until Wrecked Ship boss bit 0 sets.

    Damage is unreliable until open-eye timing is scripted; this records the
    furthest probe (min HP, boss bits) for iteration.
    """
    if not source.exists() and capture_if_missing:
        capture_phantoon_entry(output=source)

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        place_samus(env, 140, 180)
        for _ in range(20):
            apply_dev_survivability(env)
            env.step(idle_action())

        dead_frame: int | None = None
        min_body = 10**9
        for frame in range(max_frames):
            state = parse_env_state(env, frame=frame)
            assist.apply(env.data, state)
            apply_dev_survivability(env)
            select_weapon(env, 1)  # missiles
            hps = enemy_hps(env, 4)
            body = hps[0]
            if body < 60000:
                min_body = min(min_body, body)
            if phantoon_defeated(env):
                dead_frame = frame
                save_dev_state(env, save_defeated)
                break
            if state.samus_x > 60000:
                place_samus(env, 140, 180)
            ram = env.get_ram()
            enemy_x = int(ram[0x0F7A]) | (int(ram[0x0F7B]) << 8)
            enemy_y = int(ram[0x0F7E]) | (int(ram[0x0F7F]) << 8)
            env.step(
                fight_phantoon_action(
                    state, frame, enemy_x=enemy_x, enemy_y=enemy_y
                )
            )
        state = parse_env_state(env)
        return {
            "success": dead_frame is not None,
            "deadFrame": dead_frame,
            "minBodyHp": min_body if min_body < 10**9 else None,
            "bossBitsWreckedShip": wrecked_ship_boss_bits(env),
            "finalRoomIdHex": f"0x{state.room_id:04X}",
            "powerBombs": f"{state.power_bombs}/{state.max_power_bombs}",
            "statePath": (
                str(save_defeated.resolve()) if dead_frame is not None else None
            ),
            "developmentOnly": True,
            "note": (
                "Phantoon open-eye damage not yet reliable; entry + route proven."
                if dead_frame is None
                else "Phantoon defeated"
            ),
        }
    finally:
        env.close()
