"""Historical competitive-order KPDR door-warp topology.

Door-warp topology + optional item grants. **Not continuous evidence.**

Hops follow competitive KPDR (Kraid → Varia → Hi-Jump) for diagnostics. The
active controller route in ``docs/routes/ROUTE_KPDR.md`` uses the safer
Hi-Jump-before-Kraid order; do not infer play order from this warp table.

Door pointers for Hi-Jump:

- Business Center ``0xA7DE`` → Hi-Jump Shaft ``0xAA41``: ``0x92D6``
- Hi-Jump Shaft → Hi-Jump Room ``0xA9E5``: ``0x9426``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import (
    apply_dev_survivability,
    boot_from_state,
    door_warp,
    free_place_if_stuck,
    make_dev_env,
    place_samus,
    save_dev_state,
    state_summary,
)
from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import parse_env_state, write_wram_u16

# Item / beam bits
ITEM_VARIA = 0x0001
ITEM_MORPH = 0x0004
ITEM_HI_JUMP = 0x0100
ITEM_BOMBS = 0x1000

# Rooms
ROOM_BIG_PINK = 0x9D19
ROOM_GHZ = 0x9E52
ROOM_NOOB = 0x9FBA
ROOM_RED_TOWER = 0xA253
ROOM_BAT = 0xA3DD
ROOM_BELOW_SPAZER = 0xA408
ROOM_WEST_TUNNEL = 0xCF54
ROOM_GLASS = 0xCEFB
ROOM_EAST_TUNNEL = 0xCF80
ROOM_WAREHOUSE = 0xA6A1
ROOM_ZEELA = 0xA471
ROOM_WAREHOUSE_KIHUNTER = 0xA4DA
ROOM_BABY_KRAID = 0xA521
ROOM_KRAID_EYE = 0xA56B
ROOM_KRAID = 0xA59F
ROOM_VARIA = 0xA6E2
ROOM_BUSINESS = 0xA7DE
ROOM_HJ_SHAFT = 0xAA41
ROOM_HJ = 0xA9E5

# Bank $83 door defs → destination
DOOR_BIG_PINK_TO_GHZ = 0x8DEA
DOOR_GHZ_TO_NOOB = 0x8E92
DOOR_NOOB_TO_RED = 0x8F0A
DOOR_RED_TO_BAT = 0x9042
DOOR_BAT_TO_BELOW_SPAZER = 0x9102
DOOR_BELOW_SPAZER_TO_WEST = 0x911A
DOOR_WEST_TO_GLASS = 0xA360
DOOR_GLASS_TO_EAST = 0xA348
DOOR_EAST_TO_WAREHOUSE = 0xA384
DOOR_WAREHOUSE_TO_ZEELA = 0x923A
DOOR_ZEELA_TO_KIHUNTER = 0x9156
DOOR_KIHUNTER_TO_BABY = 0x917A
DOOR_BABY_TO_EYE = 0x919E
DOOR_EYE_TO_KRAID = 0x91B6
DOOR_KRAID_TO_VARIA = 0x91DA
# Reverse Kraid lair → Business Center → Hi-Jump
DOOR_VARIA_TO_KRAID = 0x9252
DOOR_KRAID_TO_EYE = 0x91CE
DOOR_EYE_TO_BABY = 0x91AA
DOOR_BABY_TO_KIHUNTER = 0x9192
DOOR_KIHUNTER_TO_ZEELA = 0x916E
DOOR_ZEELA_TO_WAREHOUSE = 0x913E
DOOR_WAREHOUSE_TO_BUSINESS = 0x9246
DOOR_BUSINESS_TO_HJ_SHAFT = 0x92D6
DOOR_HJ_SHAFT_TO_HJ = 0x9426

# Named hop table: (id, door, room, place_x, place_y)
# From Big Pink main toward Hi-Jump (includes Kraid forward + reverse).
KPDR_TO_HIJUMP: tuple[tuple[str, int, int, int, int], ...] = (
    ("ghz", DOOR_BIG_PINK_TO_GHZ, ROOM_GHZ, 120, 200),
    ("noob", DOOR_GHZ_TO_NOOB, ROOM_NOOB, 100, 200),
    ("red_tower", DOOR_NOOB_TO_RED, ROOM_RED_TOWER, 80, 400),
    ("bat", DOOR_RED_TO_BAT, ROOM_BAT, 120, 180),
    ("below_spazer", DOOR_BAT_TO_BELOW_SPAZER, ROOM_BELOW_SPAZER, 120, 180),
    ("west_tunnel", DOOR_BELOW_SPAZER_TO_WEST, ROOM_WEST_TUNNEL, 100, 180),
    ("glass", DOOR_WEST_TO_GLASS, ROOM_GLASS, 200, 180),
    ("east_tunnel", DOOR_GLASS_TO_EAST, ROOM_EAST_TUNNEL, 100, 180),
    ("warehouse", DOOR_EAST_TO_WAREHOUSE, ROOM_WAREHOUSE, 200, 180),
    ("zeela", DOOR_WAREHOUSE_TO_ZEELA, ROOM_ZEELA, 120, 180),
    ("warehouse_kihunter", DOOR_ZEELA_TO_KIHUNTER, ROOM_WAREHOUSE_KIHUNTER, 120, 180),
    ("baby_kraid", DOOR_KIHUNTER_TO_BABY, ROOM_BABY_KRAID, 200, 180),
    ("kraid_eye", DOOR_BABY_TO_EYE, ROOM_KRAID_EYE, 200, 200),
    ("kraid", DOOR_EYE_TO_KRAID, ROOM_KRAID, 120, 300),
    ("varia", DOOR_KRAID_TO_VARIA, ROOM_VARIA, 80, 180),
    # reverse out of lair
    ("kraid_exit", DOOR_VARIA_TO_KRAID, ROOM_KRAID, 200, 300),
    ("eye_exit", DOOR_KRAID_TO_EYE, ROOM_KRAID_EYE, 200, 200),
    ("baby_exit", DOOR_EYE_TO_BABY, ROOM_BABY_KRAID, 200, 180),
    ("kihunter_exit", DOOR_BABY_TO_KIHUNTER, ROOM_WAREHOUSE_KIHUNTER, 120, 180),
    ("zeela_exit", DOOR_KIHUNTER_TO_ZEELA, ROOM_ZEELA, 120, 180),
    ("warehouse_exit", DOOR_ZEELA_TO_WAREHOUSE, ROOM_WAREHOUSE, 200, 180),
    ("business", DOOR_WAREHOUSE_TO_BUSINESS, ROOM_BUSINESS, 120, 400),
    ("hj_shaft", DOOR_BUSINESS_TO_HJ_SHAFT, ROOM_HJ_SHAFT, 400, 180),
    ("hj_room", DOOR_HJ_SHAFT_TO_HJ, ROOM_HJ, 80, 180),
)

HOP_BY_ID = {h[0]: h for h in KPDR_TO_HIJUMP}

BIG_PINK_MAIN = INTEGRATION_DIR / "dev_b1_bigpink_main_controller.state"
VARIA_STATE = INTEGRATION_DIR / "dev_varia_equipped_dev.state"
HJ_ENTRY = INTEGRATION_DIR / "dev_hijump_room_entry.state"
HJ_COLLECTED_DEV = INTEGRATION_DIR / "dev_hijump_collected_dev.state"

# Intermediate anchors written by route-to-hijump when save_hops=True
ANCHOR_NAMES: dict[str, str] = {
    "ghz": "dev_kpdr_ghz.state",
    "noob": "dev_kpdr_noob.state",
    "red_tower": "dev_kpdr_red_tower.state",
    "warehouse": "dev_kpdr_warehouse.state",
    "kraid_eye": "dev_kpdr_kraid_eye.state",
    "kraid": "dev_kpdr_kraid_entry.state",
    "varia": "dev_kpdr_varia.state",
    "business": "dev_kpdr_business.state",
    "hj_shaft": "dev_kpdr_hj_shaft.state",
    "hj_room": "dev_hijump_room_entry.state",
}


def ensure_missiles(env: Any, *, capacity: int = 15) -> None:
    """Red doors (Hi-Jump) need missiles."""
    state = parse_env_state(env)
    if state.max_missiles < capacity:
        write_wram_u16(env, 0x09C8, capacity)
    write_wram_u16(env, 0x09C6, max(capacity, state.missiles))


def ensure_supers(env: Any, *, capacity: int = 5) -> None:
    state = parse_env_state(env)
    if state.max_super_missiles < capacity:
        write_wram_u16(env, 0x09CC, capacity)
        write_wram_u16(env, 0x09CA, capacity)


def grant_hi_jump_dev(env: Any) -> None:
    """Development-only Hi-Jump collected+equipped bit (not continuous-legal)."""
    state = parse_env_state(env)
    write_wram_u16(env, 0x09A4, state.collected_items | ITEM_HI_JUMP)
    write_wram_u16(env, 0x09A2, state.equipped_items | ITEM_HI_JUMP)


def grant_varia_dev(env: Any) -> None:
    state = parse_env_state(env)
    write_wram_u16(env, 0x09A4, state.collected_items | ITEM_VARIA)
    write_wram_u16(env, 0x09A2, state.equipped_items | ITEM_VARIA)


def _settle_place(env: Any, px: int, py: int, *, frames: int = 10) -> None:
    free_place_if_stuck(env, px, py)
    place_samus(env, px, py)
    write_wram_u16(env, 0x0A1C, 1)
    for _ in range(frames):
        apply_dev_survivability(env)
        env.step(idle_action())


def door_warp_hops(
    env: Any,
    hops: tuple[tuple[str, int, int, int, int], ...],
    *,
    place_free: bool = True,
    save_hops: bool = False,
    until: str | None = None,
) -> list[dict[str, object]]:
    """Run door-warp hops; optionally stop after hop id ``until`` (inclusive)."""
    results: list[dict[str, object]] = []
    for name, door, room, px, py in hops:
        ensure_missiles(env)
        ensure_supers(env)
        state = door_warp(env, door, expected_room=room)
        ok = state.room_id == room
        if place_free:
            _settle_place(env, px, py)
        state = parse_env_state(env)
        row: dict[str, object] = {
            "name": name,
            "success": ok and state.room_id == room,
            "roomIdHex": f"0x{state.room_id:04X}",
            "expectedRoomIdHex": f"0x{room:04X}",
            "doorIdHex": f"0x{door:04X}",
            "samusX": state.samus_x,
            "samusY": state.samus_y,
            "itemsHex": f"0x{state.collected_items:04X}",
        }
        if save_hops and name in ANCHOR_NAMES:
            path = INTEGRATION_DIR / ANCHOR_NAMES[name]
            save_dev_state(env, path)
            row["statePath"] = str(path.resolve())
        results.append(row)
        if not row["success"]:
            break
        if until is not None and name == until:
            break
    return results


def route_to_hijump(
    *,
    source: Path = BIG_PINK_MAIN,
    output: Path = HJ_ENTRY,
    save_hops: bool = True,
    grant_varia: bool = True,
    grant_hijump: bool = False,
) -> dict[str, object]:
    """Door-warp Big Pink → … → Kraid → Varia → Business → Hi-Jump room.

    Marks ``developmentOnly``. Optionally grants Varia at the Varia hop and
    Hi-Jump at the final room (PLM collect is flaky after warps).
    """
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env))
        ensure_supers(env)
        ensure_missiles(env)
        hops = door_warp_hops(env, KPDR_TO_HIJUMP, place_free=True, save_hops=save_hops)
        # After varia hop, ensure bit for heat safety on later work
        if grant_varia:
            for row in hops:
                if row["name"] == "varia" and row["success"]:
                    # Re-apply if we continued past varia (still in env)
                    pass
            grant_varia_dev(env)
        final = parse_env_state(env)
        if final.room_id != ROOM_HJ:
            return {
                "success": False,
                "hops": hops,
                "finalRoomIdHex": f"0x{final.room_id:04X}",
                "developmentOnly": True,
            }
        if grant_hijump:
            grant_hi_jump_dev(env)
            for _ in range(8):
                apply_dev_survivability(env)
                env.step(idle_action())
            save_dev_state(env, HJ_COLLECTED_DEV)
        save_dev_state(env, output)
        summary = state_summary(env)
        summary.update(
            {
                "success": True,
                "hops": hops,
                "hopCount": len(hops),
                "hopSuccess": sum(1 for h in hops if h["success"]),
                "statePath": str(output.resolve()),
                "hiJumpGranted": grant_hijump,
                "developmentOnly": True,
            }
        )
        return summary
    finally:
        env.close()


def route_varia_to_hijump(
    *,
    source: Path = VARIA_STATE,
    output: Path = HJ_ENTRY,
    grant_hijump: bool = True,
) -> dict[str, object]:
    """Shorter chain: existing Varia state → Hi-Jump room."""
    reverse_and_hj = tuple(
        h
        for h in KPDR_TO_HIJUMP
        if h[0]
        in (
            "kraid_exit",
            "eye_exit",
            "baby_exit",
            "kihunter_exit",
            "zeela_exit",
            "warehouse_exit",
            "business",
            "hj_shaft",
            "hj_room",
        )
    )
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env))
        ensure_missiles(env)
        ensure_supers(env)
        hops = door_warp_hops(env, reverse_and_hj, place_free=True, save_hops=True)
        final = parse_env_state(env)
        ok = final.room_id == ROOM_HJ and all(h["success"] for h in hops)
        if ok and grant_hijump:
            grant_hi_jump_dev(env)
            for _ in range(8):
                apply_dev_survivability(env)
                env.step(idle_action())
            save_dev_state(env, HJ_COLLECTED_DEV)
        if ok:
            save_dev_state(env, output)
        summary = state_summary(env)
        summary.update(
            {
                "success": ok,
                "hops": hops,
                "statePath": str(output.resolve()) if ok else None,
                "hiJump": bool(parse_env_state(env).collected_items & ITEM_HI_JUMP),
                "developmentOnly": True,
            }
        )
        return summary
    finally:
        env.close()


def hop_once(
    *,
    hop_id: str,
    source: Path,
    output: Path | None = None,
) -> dict[str, object]:
    """Single door-warp hop from a save state."""
    if hop_id not in HOP_BY_ID:
        raise KeyError(f"unknown hop {hop_id!r}; choose from {sorted(HOP_BY_ID)}")
    name, door, room, px, py = HOP_BY_ID[hop_id]
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        for _ in range(4):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env))
        ensure_missiles(env)
        ensure_supers(env)
        state = door_warp(env, door, expected_room=room)
        _settle_place(env, px, py)
        state = parse_env_state(env)
        ok = state.room_id == room
        if ok and output is not None:
            save_dev_state(env, output)
        summary = state_summary(env)
        summary.update(
            {
                "success": ok,
                "hop": name,
                "doorIdHex": f"0x{door:04X}",
                "developmentOnly": True,
                "statePath": str(output.resolve()) if output and ok else None,
            }
        )
        return summary
    finally:
        env.close()
