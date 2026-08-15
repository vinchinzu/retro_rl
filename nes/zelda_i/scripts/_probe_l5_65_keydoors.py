"""From Level5Cleared65: dump already done. Try each door; spend a key if locked.

No pokes, no candle, no bomb walls. Walk any dest that is not 0x55.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared65"
ROOM_55 = 0x55
ROOM_65 = 0x65
DIRS = ("RIGHT", "LEFT", "DOWN", "UP")  # UP last: back to 0x55


def decode(mask: int) -> dict:
    v = int(mask) & 0x0F
    return {
        "raw": v,
        "raw_hex": f"0x{v:02x}",
        "east": bool(v & DoorDir.RIGHT),
        "west": bool(v & DoorDir.LEFT),
        "south": bool(v & DoorDir.DOWN),
        "north": bool(v & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(v)),
    }


def dump_live(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    c = compact_snapshot(snap)
    c["doors"] = decode(snap.cur_opened_doors)
    c["doorway_mask"] = decode(snap.open_doorway_mask)
    c["room_hex"] = f"0x{snap.screen:02x}"
    c["next_room_hex"] = f"0x{snap.next_screen:02x}"
    c["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    c["objects"] = [
        {
            "slot": o.slot,
            "type_id": o.type_id,
            "type_hex": f"0x{o.type_id:02x}",
            "type_name": object_name(o.type_id),
            "x": o.x,
            "y": o.y,
            "hp": o.hp,
        }
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    return c


def open_from_bytes(state_data: bytes):
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    env.em.set_state(state_data)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def try_dir(state_data: bytes, direction: str, total: list[int]) -> dict:
    env = None
    try:
        env, assist, obs = open_from_bytes(state_data)
        idle(env, assist, total, 8)
        snap0 = read_snapshot(env.get_ram())
        keys0 = int(snap0.keys)
        bombs0 = int(snap0.bombs)
        hop = exit_door(env, assist, total, direction, push=180)
        # extra hold in case unlock animation needs more
        room0 = snap0.screen
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0 and snap.mode == PLAY_MODE:
                break
            obs, *_ = env.step(nes_action(direction))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 24)
        snap = read_snapshot(env.get_ram())
        # wait play if transitioned
        if snap.screen != room0:
            for _ in range(180):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and not snap.transitioning:
                    idle(env, assist, total, 16)
                    break
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                assist.apply_env(env, frame=total[0])
        d = dump_live(env)
        png = RECORDINGS_DIR / f"l5_65_key_{direction.lower()}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        dest = d["room_hex"]
        changed = dest != "0x65"
        keys1 = int(d["keys"])
        rec = {
            "direction": direction,
            "changed": changed,
            "dest_room": dest if changed else None,
            "sealed": not changed,
            "keys_in": keys0,
            "keys_out": keys1,
            "key_spent": keys1 < keys0,
            "bombs_in": bombs0,
            "bombs_out": int(d["bombs"]),
            "mode": d["mode"],
            "xy": [d["x"], d["y"]],
            "doors": d["doors"],
            "doorway_mask": d["doorway_mask"],
            "objects": d["objects"],
            "room_item_id": d.get("room_item_id"),
            "screenshot": str(png.resolve()),
            "hop_result": hop.get("result"),
        }
        if changed:
            write_json_report(
                RECORDINGS_DIR / f"l5_{snap.screen:02x}_from65.json",
                {
                    "via": f"0x65 {direction}",
                    "ok": True,
                    "key_spent": rec["key_spent"],
                    "dump": d,
                    "screenshot": str(png.resolve()),
                    "status_claim": None,
                    "pokes": False,
                },
            )
            rec["dump_path"] = str((RECORDINGS_DIR / f"l5_{snap.screen:02x}_from65.json").resolve())
        return rec
    finally:
        if env is not None:
            env.close()


def main() -> dict:
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [0]
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 20)
        start = dump_live(env)
        print(
            "START",
            start["room_hex"],
            "mode",
            start["mode"],
            "doors",
            start["doors"],
            "mask",
            start["doorway_mask"],
            "keys",
            start["keys"],
            "xy",
            [start["x"], start["y"]],
            flush=True,
        )
        state_bytes = env.em.get_state()
        env.close()
        env = None
    finally:
        if env is not None:
            env.close()

    tries = []
    entered = []
    key_door = None
    for direction in DIRS:
        rec = try_dir(state_data=state_bytes, direction=direction, total=total)
        tries.append(rec)
        print(
            "TRY",
            direction,
            "dest" if rec["changed"] else "sealed",
            rec.get("dest_room"),
            "key_spent",
            rec["key_spent"],
            "keys",
            rec["keys_in"],
            "->",
            rec["keys_out"],
            "xy",
            rec["xy"],
            flush=True,
        )
        if rec["key_spent"]:
            key_door = {
                "direction": direction,
                "dest": rec.get("dest_room"),
                "keys_in": rec["keys_in"],
                "keys_out": rec["keys_out"],
                "dumped_first": True,
            }
        if rec["changed"] and rec.get("dest_room") not in (None, "0x55", "0x65"):
            entered.append(rec)

    exits = {
        "from_room": "0x65",
        "from_state": STATE,
        "all_5_dead": True,
        "doors": start["doors"],
        "doorway_mask": start["doorway_mask"],
        "dumped_before_key_spend": True,
        "probes": tries,
        "next_rooms": [
            {
                "via": f"0x65 {r['direction']}",
                "room": r.get("dest_room"),
                "key_spent": r["key_spent"],
                "objects": r.get("objects"),
                "doors": r.get("doors"),
                "screenshot": r.get("screenshot"),
            }
            for r in entered
        ],
        "key_door": key_door,
        "status_claim": None,
        "pokes": False,
        "whistle_0x065C": start.get("whistle_0x065C"),
    }
    write_json_report(RECORDINGS_DIR / "l5_65_exits.json", exits)
    write_json_report(RECORDINGS_DIR / "l5_65_keydoors.json", exits)
    return exits


if __name__ == "__main__":
    r = main()
    print("DOORS", r.get("doors"), "MASK", r.get("doorway_mask"))
    print("PROBES")
    for p in r.get("probes") or []:
        print(" ", p["direction"], "dest" if p["changed"] else "sealed", p.get("dest_room"), "key", p["key_spent"])
    print("NEXT", r.get("next_rooms"))
    print("KEY_DOOR", r.get("key_door"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("status_claim", None)
