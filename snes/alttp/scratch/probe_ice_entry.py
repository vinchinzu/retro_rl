"""Ice Palace entry pin (rr-3yxm): poke kit, real room-load, first hop.

Load HyruleCastleGrounds, poke entrance loadout + $F3C5=3, trigger
Module_PreDungeon with which_entrance=0x2C (vanilla Ice Palace, room 0x0E).
Indoor pin only after a real room-load (controllable, Ice Palace overlay).

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ice_entry.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image

from alttp.paths import GAME_DIR, INTEGRATION, INTEGRATION_DIR, RECORDINGS_DIR
from alttp.primitives import Waypoint, active_sprites, move_to, settle_control
from alttp.ram import room_label, wram_index
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state, write_state_bytes

OUT_DIR = RECORDINGS_DIR / "probe_ice_entry"
BOOT = "HyruleCastleGrounds"
PIN = "IcePalaceEntry"
ICE_ENTRANCE = 0x2D  # kEntranceNames: [Room #014] Ice Palace (0x2C is Lost Woods 0xE1)
HYPOTHESIS_ROOM = 0x0E
MODULE_PREDUNGEON = 0x06
MODULE_SPOTLIGHT = 0x0F
SAVED_MODULE_PREDUNGEON = 0x06

# WRAM $F3xx inventory / progress. Offset is WRAM, not get_ram() index.
POKES: tuple[tuple[int, int, str], ...] = (
    (0xF345, 1, "fire_rod"),
    (0xF354, 2, "titans"),
    (0xF355, 1, "boots"),
    (0xF356, 1, "flippers"),
    (0xF357, 1, "moon_pearl"),
    (0xF359, 2, "master_sword"),
    (0xF35B, 0, "armor_green"),  # must NOT be blue mail
    (0xF36C, 0x40, "max_hp_8"),
    (0xF36D, 0x40, "hp_8"),
    (0xF36E, 0x80, "magic_full"),
    (0xF379, 0x1E, "ability_lift2_swim_dash"),
    (0xF3C5, 3, "progress_aga1"),
)

LOADOUT_READS: tuple[tuple[int, str], ...] = (
    (0xF345, "F345_firerod"),
    (0xF354, "F354_gloves"),
    (0xF355, "F355_boots"),
    (0xF356, "F356_flippers"),
    (0xF357, "F357_pearl"),
    (0xF359, "F359_sword"),
    (0xF35B, "F35B_armor"),
    (0xF36C, "F36C_max_hp"),
    (0xF36D, "F36D_hp"),
    (0xF36E, "F36E_magic"),
    (0xF36F, "F36F_keys"),
    (0xF379, "F379_ability"),
    (0xF3C5, "F3C5_progress"),
    (0xF3CA, "F3CA_darkworld"),
    (0xF3CC, "F3CC_follower"),
)


SNES_WRAM_BANK = 0x7E0000


def _mem(env: object) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return unwrapped.data.memory


def bus_addr(offset: int) -> int:
    """WRAM offset → stable-retro assign address (bank $7E for >= $2000)."""
    off = int(offset)
    return SNES_WRAM_BANK + off if off >= 0x2000 else off


def assign_u8(env: object, offset: int, value: int) -> None:
    """Write one WRAM byte. High WRAM uses $7E:F3xx, not get_ram() index."""
    _mem(env).assign(bus_addr(offset), "|u1", int(value) & 0xFF)


def read_wram(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    out = {name: int(ram[wram_index(off)]) for off, name in LOADOUT_READS}
    out["00EE_layer"] = int(ram[0x00EE])
    out["040C_dungeon"] = int(ram[0x040C])
    out["010E_entrance"] = int(ram[0x010E])
    out["0303_yitem"] = int(ram[0x0303])
    out["0307_rod"] = int(ram[0x0307])
    out["006C_doorway"] = int(ram[0x006C])
    out["0FFF_dw"] = int(ram[0x0FFF])
    out["2F_dir"] = int(ram[0x2F])
    out["5D_act"] = int(ram[0x5D])
    return out


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    rec = leftover_from_snapshot(snap)
    rec.update(
        {
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "room_label": room_label(snap.room_base_id),
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "link_action": int(snap.link_action),
            "has_control": bool(snap.has_control),
            "dark_world": bool(snap.dark_world),
            "leftover": leftover_bytes(env),
            "sprites": [
                {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
                for s in active_sprites(env)
            ][:12],
        }
    )
    return rec


def brief(g: dict[str, Any]) -> str:
    return (
        f"{g['room_hex']} ({g['link_x']},{g['link_y']}) "
        f"mod={g['module_hex']} sub={g['submodule']} "
        f"ctrl={int(bool(g['has_control']))} sword={g['sword']} "
        f"keys={g['keys']} F3CC={g['follower']}"
    )


def save_png(env: object, path: Path, title: str) -> None:
    img = overlay_from_env(env, include_all_sprites=True, title=title)
    Image.fromarray(img).save(path)


def apply_pokes(env: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, value, name in POKES:
        before = read_wram(env, offset)
        assign_u8(env, offset, value)
        after = read_wram(env, offset)
        rows.append(
            {
                "name": name,
                "offset": f"0x{offset:04X}",
                "want": value,
                "before": before,
                "after": after,
                "ok": after == value,
            }
        )
    step_frames(env, no_action(), 2)
    for row in rows:
        row["after_settle"] = read_wram(env, int(row["offset"], 16))
        row["ok"] = row["after_settle"] == row["want"]
    return rows


def trigger_entrance(env: object, entrance: int, *, module: int) -> None:
    assign_u8(env, 0x010E, entrance)
    assign_u8(env, 0x010F, 0)
    assign_u8(env, 0x010C, SAVED_MODULE_PREDUNGEON)
    assign_u8(env, 0x0011, 0)
    assign_u8(env, 0x00B0, 0)
    assign_u8(env, 0x0010, module)


def wait_indoor_control(env: object, *, max_frames: int = 900) -> dict[str, Any]:
    frames = 0
    idle = no_action()
    while frames < max_frames:
        step_frames(env, idle, 4)
        frames += 4
        snap = snapshot_env(env)
        if snap.is_text_mode:
            step_frames(env, action_for("A"), 2)
            step_frames(env, idle, 2)
            frames += 4
            continue
        if (
            snap.indoors
            and snap.has_control
            and not snap.is_hold_up_item
            and snap.game_mode == 0x07
        ):
            return {"ok": True, "frames": frames, "glance": glance(env)}
    return {"ok": False, "frames": frames, "glance": glance(env)}


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int = 280,
    start_room: int,
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    events: list[dict[str, Any]] = []
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule)
    idle = 0
    while frames < max_frames:
        step_frames(env, action, 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule)
        if cur != prev:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "sub": int(snap.submodule),
                    "xy": [snap.link_x, snap.link_y],
                    "indoors": bool(snap.indoors),
                    "ctrl": bool(snap.has_control),
                }
            )
            prev = cur
            if snap.room_base_id != start_room or (not snap.indoors):
                settle_control(env, max_frames=480)
                return {
                    "ok": True,
                    "frames": frames,
                    "events": events[-12:],
                    "end": glance(env),
                }
        else:
            idle += 4
            # Intra-room (sub=1) idles during the scroll; don't abort.
            if idle >= 48 and int(snap.submodule) == 0:
                break
    return {
        "ok": False,
        "frames": frames,
        "events": events[-12:],
        "end": glance(env),
    }


FREEZOR = 0xA1
FIRE_ROD_Y = 5  # $0303: 5=fire rod, 6=ice rod (LinkItem_Rod)


def freezor_alive(env: object) -> bool:
    return any(s.sprite_type == FREEZOR and s.hp > 0 for s in active_sprites(env))


def freezor_hp(env: object) -> list[int]:
    return [s.hp for s in active_sprites(env) if s.sprite_type == FREEZOR]


def shoot_fire_rod(env: object) -> None:
    """Equip fire rod ($0303) and tap Y. Does not fire in a doorway ($6C)."""
    assign_u8(env, 0x0303, FIRE_ROD_Y)
    assign_u8(env, 0x0307, 1)
    if read_wram(env, 0xF36E) < 0x20:
        assign_u8(env, 0xF36E, 0x80)
    step_frames(env, no_action(), 2)
    step_frames(env, action_for("UP"), 4)  # face the wall sculpture
    step_frames(env, action_for("Y"), 8)
    step_frames(env, no_action(), 40)


def first_hop(env: object, spawn_state: bytes, out: Path, room: int) -> dict[str, Any]:
    """Leave the south door, melt Freezor, LEFT through Ice Lobby WS."""
    env.em.set_state(spawn_state)
    settle_control(env, max_frames=60)
    rec: dict[str, Any] = {"start": glance(env), "freezorHp": []}
    # Doorway ($6C) blocks LinkItem_Rod. Walk onto the ice, aligned with Freezor.
    walk = move_to(
        env,
        Waypoint(7520, 328, tolerance=10, room=room, label="freezor_stand"),
        max_frames=240,
    )
    rec["stand"] = {"ok": walk.ok, "reason": walk.reason, "glance": glance(env)}
    save_png(env, out / "hop_stand.png", f"stand {brief(rec['stand']['glance'])}")
    shots = 0
    while freezor_alive(env) and shots < 6:
        shoot_fire_rod(env)
        shots += 1
        rec["freezorHp"].append(freezor_hp(env))
        save_png(env, out / f"hop_shot{shots}.png", f"shot{shots} hp={rec['freezorHp'][-1]}")
    rec["shots"] = shots
    rec["freezorDead"] = not freezor_alive(env)
    rec["afterShots"] = glance(env)
    save_png(env, out / "hop_freezor.png", f"freezor {brief(rec['afterShots'])}")
    cleared = env.em.get_state()
    sweeps: list[dict[str, Any]] = []
    rec["ok"] = False
    rec["end"] = rec["afterShots"]
    rec["via"] = "west"
    # West lip is x=7464; y≈360–376 actually crossed the mid-line (x=7423).
    for y in (360, 368, 376, 352, 344):
        env.em.set_state(cleared)
        settle_control(env, max_frames=30)
        move_to(
            env,
            Waypoint(7464, y, tolerance=6, room=room, label=f"west_y{y}"),
            max_frames=220,
        )
        posed = glance(env)
        west = hold(env, ("LEFT",), max_frames=520, start_room=room)
        hit = {
            "y": y,
            "pose": [posed["link_x"], posed["link_y"]],
            "left": west["ok"],
            "endRoom": west["end"]["room_hex"],
            "endXy": [west["end"]["link_x"], west["end"]["link_y"]],
            "indoors": west["end"]["indoors"],
        }
        sweeps.append(hit)
        print(
            f"  WEST y={y} pose={hit['pose']} → {hit['endRoom']} {hit['endXy']} "
            f"left={west['ok']}"
        )
        crossed = west["end"]["link_x"] < 7400 or west["end"]["room"] != room
        if west["end"]["indoors"] and west["end"]["module"] == 0x07 and crossed:
            rec["ok"] = True
            rec["end"] = west["end"]
            rec["westY"] = y
            rec["approachXy"] = hit["pose"]
            rec["intraRoom"] = west["end"]["room"] == room
            save_png(env, out / "hop_west.png", f"west y={y} {brief(west['end'])}")
            break
    else:
        env.em.set_state(cleared)
        settle_control(env, max_frames=30)
        save_png(env, out / "hop_west.png", f"west miss {brief(glance(env))}")
        rec["end"] = glance(env)
    rec["sweeps"] = sweeps
    return rec


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ice Palace entry pin rr-3yxm")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", action="store_false", dest="save")
    parser.add_argument("--entrance", type=lambda s: int(s, 0), default=ICE_ENTRANCE)
    args = parser.parse_args(argv)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema": "alttp_ice_entry",
        "schemaVersion": 1,
        "bead": "rr-3yxm",
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "boot": BOOT,
        "pin": PIN,
        "hypothesisRoom": HYPOTHESIS_ROOM,
        "entranceId": args.entrance,
    }

    env = build_boot_env(BOOT, render_mode="rgb_array")
    try:
        env.reset()
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, BOOT):
            raise RuntimeError(f"failed to load {BOOT}")
        settle_control(env)
        start = glance(env)
        payload["bootGlance"] = start
        save_png(env, out / "boot.png", f"boot {brief(start)}")
        print(f"BOOT {brief(start)}")

        pokes = apply_pokes(env)
        payload["pokes"] = pokes
        poke_ok = all(r["ok"] for r in pokes)
        print("POKES " + ("ok" if poke_ok else "FAIL"))
        for row in pokes:
            mark = "ok" if row["ok"] else "MISS"
            print(
                f"  {mark} {row['name']} {row['offset']} "
                f"{row['before']}->{row['after_settle']} want={row['want']}"
            )
        if not poke_ok:
            payload["halt"] = "poke_mismatch"
            payload["end"] = glance(env)
            write_json(out / "leftover.json", payload)
            return 1

        after_poke = glance(env)
        payload["afterPoke"] = after_poke
        save_png(env, out / "after_poke.png", f"poked {brief(after_poke)}")

        trigger_entrance(env, args.entrance, module=MODULE_PREDUNGEON)
        load = wait_indoor_control(env)
        payload["load"] = {
            "ok": load["ok"],
            "frames": load["frames"],
            "module": MODULE_PREDUNGEON,
            "end": load["glance"],
        }
        print(f"LOAD predungeon {load['ok']} f={load['frames']} {brief(load['glance'])}")
        if not load["ok"] or load["glance"]["room"] != HYPOTHESIS_ROOM:
            # Spotlight-close path used by Overworld_UseEntrance.
            env.reset()
            resync_custom_state(env, GAME_DIR, INTEGRATION, BOOT)
            settle_control(env)
            apply_pokes(env)
            trigger_entrance(env, args.entrance, module=MODULE_SPOTLIGHT)
            load2 = wait_indoor_control(env, max_frames=1800)
            payload["loadSpotlight"] = {
                "ok": load2["ok"],
                "frames": load2["frames"],
                "module": MODULE_SPOTLIGHT,
                "end": load2["glance"],
            }
            print(
                f"LOAD spotlight {load2['ok']} f={load2['frames']} "
                f"{brief(load2['glance'])}"
            )
            load = load2

        pin_g = load["glance"]
        save_png(env, out / "pin.png", f"pin {brief(pin_g)}")
        payload["pinGlance"] = pin_g
        indoor_ok = (
            bool(pin_g["indoors"])
            and bool(pin_g["has_control"])
            and pin_g["module"] == 0x07
            and pin_g["submodule"] == 0
            and pin_g["room"] == HYPOTHESIS_ROOM
            and pin_g["leftover"]["F359_sword"] == 2
            and pin_g["leftover"]["F354_gloves"] == 2
            and pin_g["leftover"]["F357_pearl"] == 1
            and pin_g["leftover"]["F345_firerod"] == 1
            and pin_g["leftover"]["F356_flippers"] == 1
            and pin_g["leftover"]["F355_boots"] == 1
            and pin_g["leftover"]["F35B_armor"] == 0
            and pin_g["leftover"]["040C_dungeon"] == 0x12
        )
        payload["pinOk"] = indoor_ok
        if not indoor_ok:
            payload["halt"] = "pin_not_ice"
            payload["end"] = pin_g
            write_json(out / "leftover.json", payload)
            return 1

        spawn_state = env.em.get_state()
        if args.save:
            write_state_bytes(INTEGRATION_DIR / f"{PIN}.state", spawn_state)
            print(f"WROTE {PIN}.state")

        # Independent cardinals from spawn (restore between rays).
        rays: list[dict[str, Any]] = []
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            env.em.set_state(spawn_state)
            settle_control(env, max_frames=60)
            ray = hold(env, (direction,), max_frames=360, start_room=HYPOTHESIS_ROOM)
            ray["dir"] = direction
            rays.append(ray)
            g = ray["end"]
            save_png(env, out / f"hold_{direction.lower()}.png", f"{direction} {brief(g)}")
            print(
                f"  {direction} → {g['room_hex']} ({g['link_x']},{g['link_y']}) "
                f"left={ray['ok']}"
            )
        payload["cardinals"] = [
            {
                "dir": r["dir"],
                "left": r["ok"],
                "frames": r["frames"],
                "end": {
                    "room": r["end"]["room_hex"],
                    "xy": [r["end"]["link_x"], r["end"]["link_y"]],
                    "indoors": r["end"]["indoors"],
                    "module": r["end"]["module_hex"],
                },
            }
            for r in rays
        ]

        hop = first_hop(env, spawn_state, out, HYPOTHESIS_ROOM)
        payload["firstHop"] = hop
        print(
            f"HOP via={hop.get('via')} ok={hop.get('ok')} "
            f"freezorDead={hop.get('freezorDead')} shots={hop.get('shots')} "
            f"{brief(hop['end'])}"
        )

        env.em.set_state(spawn_state)
        settle_control(env, max_frames=60)
        end = glance(env)
        payload["end"] = end
        write_json(out / "leftover.json", payload)
        print(f"END {brief(end)}")
        return 0 if indoor_ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
