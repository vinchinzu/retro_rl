"""Live recon from Level5North56: dump 0x56, walk 0x57 Zols, key UP 0x46.

Controller-only. No key/door pokes. No bomb walls. No Whistle route.
Not Clean STATUS. Does not redo east-key.
"""
from __future__ import annotations

from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ids import GEL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, ZOL_OBJECT_TYPE
from zelda_i.dungeon_ops import exit_door, fight_clear, idle, live_killables, room_fields
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

STATE = "Level5North56"
ROOM_56 = 0x56
ROOM_57 = 0x57
ROOM_46 = 0x46
CANDLE_NAMES = {0: "none", 1: "blue", 2: "red"}
DIR_NAME = {DoorDir.RIGHT: "RIGHT", DoorDir.LEFT: "LEFT", DoorDir.DOWN: "DOWN", DoorDir.UP: "UP"}


def decode_doors(mask: int) -> dict:
    value = int(mask) & 0x0F
    return {
        "raw": value,
        "raw_hex": f"0x{value:02x}",
        "east": bool(value & DoorDir.RIGHT),
        "west": bool(value & DoorDir.LEFT),
        "south": bool(value & DoorDir.DOWN),
        "north": bool(value & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(value)),
    }


def candle_block(ram) -> dict:
    raw = read_u8(ram, ADDR_CANDLE)
    selected = read_u8(ram, ADDR_SELECTED_ITEM)
    return {
        "addr": "0x065B",
        "raw": raw,
        "name": CANDLE_NAMES.get(raw, f"unknown_{raw}"),
        "present": raw > 0,
        "selected_item": selected,
        "sword_0x0657": read_u8(ram, ADDR_SWORD),
        "whistle_0x065C": read_u8(ram, ADDR_WHISTLE),
    }


def dump_live(snap, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["candle"] = candle_block(ram)
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    compact["item_id"] = snap.room_item_id
    compact["item_name"] = compact.get("room_item_name")
    return compact


def open_env(state: str):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def settle(env, assist, total, frames=45):
    idle(env, assist, total, frames)
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def live_zols(snap) -> list:
    return live_killables(snap, (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE))


def classify_room(dump: dict) -> str:
    objs = dump.get("objects") or []
    combat_types = []
    for obj in objs:
        tid = obj.get("type_id", obj.get("type"))
        hp = obj.get("hp", 0)
        if tid in (0, 0xFF, None):
            continue
        if tid in (0x40, 0x49, 0x4E, 0x4F, 0x55, 0x2B):
            continue
        if hp > 0 or tid in (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE, 0x1B):
            combat_types.append(tid)
    item_id = dump.get("item_id") or dump.get("room_item_id") or 0
    if combat_types:
        return "combat"
    if item_id:
        return "item"
    return "empty_or_unsettled"


def natural_next_dirs(dump: dict, entry_dir: str) -> list[str]:
    """Open doors in cur_opened_doors except the entry we came through."""
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    skip = opposite.get(entry_dir)
    doors = dump.get("doors") or {}
    out = []
    for name, flag in (
        ("RIGHT", doors.get("east")),
        ("LEFT", doors.get("west")),
        ("UP", doors.get("north")),
        ("DOWN", doors.get("south")),
    ):
        if flag and name != skip:
            out.append(name)
    return out


def session_56_and_57() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png_56 = RECORDINGS_DIR / "l5_56_recon.png"
    png_57 = RECORDINGS_DIR / "l5_57_recon.png"
    env, assist, obs = open_env(STATE)
    total = [1]
    checkpoints: list[str] = []
    try:
        obs = settle(env, assist, total, 50)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump56 = dump_live(snap, ram)
        save_rgb_png(obs, png_56)
        report56 = {
            "ok": snap.level == 5 and snap.screen == ROOM_56 and snap.mode == PLAY_MODE,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "bomb_or_candle": False,
            "dump": dump56,
            "screenshot": str(png_56.resolve()),
            "room_fields": room_fields(snap, ram),
        }
        write_json_report(RECORDINGS_DIR / "l5_56_recon.json", report56)

        walked57 = False
        cleared57 = False
        clear_report = None
        dump57 = None
        checkpoint57 = None
        if report56["ok"]:
            east = exit_door(env, assist, total, "RIGHT")
            walked57 = bool(east.get("changed_room")) and east.get("after", {}).get("screen") == ROOM_57
            obs = settle(env, assist, total, 70)
            ram = env.get_ram()
            snap = read_snapshot(ram)
            dump57 = dump_live(snap, ram)
            save_rgb_png(obs, png_57)
            if snap.screen == ROOM_57 and snap.mode == PLAY_MODE:
                clear_report = fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=(ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE),
                    max_frames=10000,
                    level=5,
                )
                obs = settle(env, assist, total, 40)
                ram = env.get_ram()
                snap = read_snapshot(ram)
                dump57 = dump_live(snap, ram)
                save_rgb_png(obs, png_57)
                live = live_zols(snap)
                cleared57 = (
                    snap.screen == ROOM_57
                    and snap.mode == PLAY_MODE
                    and not live
                    and bool(clear_report.get("ok"))
                )
                if cleared57:
                    path = save_state(env, GAME_DIR, GAME, "Level5Cleared57")
                    write_state_provenance(
                        path,
                        source_state_path=(
                            GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                        ),
                        request={
                            "segment": "level5_clear57",
                            "predecessor_entry": True,
                            "start_state": STATE,
                        },
                        selected_trial={
                            "success": True,
                            "frames": clear_report.get("frames"),
                            "room": ROOM_57,
                            "keys": snap.keys,
                            "x": snap.link_x,
                            "y": snap.link_y,
                            "live_zols": 0,
                        },
                        natural_entry=False,
                    )
                    checkpoint57 = "Level5Cleared57"
                    checkpoints.append(checkpoint57)

        report57 = {
            "ok": walked57,
            "walked": walked57,
            "cleared": cleared57,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "bomb_or_candle": False,
            "clear": (
                {k: v for k, v in (clear_report or {}).items() if k != "final"}
                if clear_report
                else None
            ),
            "dump": dump57,
            "screenshot": str(png_57.resolve()),
            "checkpoint": checkpoint57,
            "frames_total": total[0],
        }
        write_json_report(RECORDINGS_DIR / "l5_57_recon.json", report57)
        return {
            "dump56": dump56,
            "report56": report56,
            "report57": report57,
            "checkpoints": checkpoints,
            "assist": assist.report(),
        }
    finally:
        env.close()


def session_46() -> dict:
    configure_headless()
    png_46 = RECORDINGS_DIR / "l5_46_recon.png"
    env, assist, obs = open_env(STATE)
    total = [1]
    checkpoints: list[str] = []
    try:
        obs = settle(env, assist, total, 30)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        start = dump_live(snap, ram)
        arrived46 = False
        dump46 = None
        next_room = None
        if snap.level == 5 and snap.screen == ROOM_56:
            north = exit_door(env, assist, total, "UP")
            arrived46 = bool(north.get("changed_room")) and north.get("after", {}).get("screen") == ROOM_46
            keys_before = start.get("keys")
            obs = settle(env, assist, total, 90)
            ram = env.get_ram()
            snap = read_snapshot(ram)
            dump46 = dump_live(snap, ram)
            save_rgb_png(obs, png_46)
            arrived46 = snap.level == 5 and snap.screen == ROOM_46 and snap.mode == PLAY_MODE
            if arrived46:
                path = save_state(env, GAME_DIR, GAME, "Level5Entered46")
                write_state_provenance(
                    path,
                    source_state_path=(
                        GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                    ),
                    request={
                        "segment": "level5_enter46",
                        "predecessor_entry": True,
                        "start_state": STATE,
                    },
                    selected_trial={
                        "success": True,
                        "arrived": True,
                        "frames": total[0],
                        "room": ROOM_46,
                        "keys": snap.keys,
                        "x": snap.link_x,
                        "y": snap.link_y,
                        "keys_before": keys_before,
                    },
                    natural_entry=False,
                )
                checkpoints.append("Level5Entered46")

                nxt_dirs = natural_next_dirs(dump46, "UP")
                if nxt_dirs:
                    direction = nxt_dirs[0]
                    hop = exit_door(env, assist, total, direction)
                    obs = settle(env, assist, total, 70)
                    ram = env.get_ram()
                    snap = read_snapshot(ram)
                    png_next = RECORDINGS_DIR / f"l5_{snap.screen:02x}_recon.png"
                    save_rgb_png(obs, png_next)
                    next_room = {
                        "direction": direction,
                        "changed": hop.get("changed_room"),
                        "dump": dump_live(snap, ram),
                        "screenshot": str(png_next.resolve()),
                        "hop": {k: v for k, v in hop.items() if k not in {"before", "at_door", "after"}},
                    }
                    write_json_report(
                        RECORDINGS_DIR / f"l5_{snap.screen:02x}_recon.json",
                        {
                            "ok": hop.get("changed_room"),
                            "via": f"0x46 {direction}",
                            "status_claim": None,
                            "pokes": False,
                            "dump": next_room["dump"],
                            "screenshot": next_room["screenshot"],
                        },
                    )
            else:
                save_rgb_png(obs, png_46)
        else:
            save_rgb_png(obs, png_46)
            dump46 = start

        kind = classify_room(dump46) if dump46 else "unknown"
        report46 = {
            "ok": arrived46,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "bomb_or_candle": False,
            "combat_vs_item": kind,
            "start_56": start,
            "dump": dump46,
            "screenshot": str(png_46.resolve()),
            "checkpoint": "Level5Entered46" if arrived46 else None,
            "next_room": next_room,
            "frames_total": total[0],
        }
        write_json_report(RECORDINGS_DIR / "l5_46_recon.json", report46)
        return {
            "report46": report46,
            "checkpoints": checkpoints,
            "assist": assist.report(),
        }
    finally:
        env.close()


def main() -> dict:
    a = session_56_and_57()
    b = session_46()
    summary = {
        "status_claim": None,
        "from_state": STATE,
        "pokes": False,
        "checkpoints_written": a["checkpoints"] + b["checkpoints"],
        "whistle_0x065C": (b["report46"].get("dump") or {}).get("whistle_0x065C"),
        "session_56_57": {
            "dump56": a["dump56"],
            "report57": {
                "walked": a["report57"]["walked"],
                "cleared": a["report57"]["cleared"],
                "checkpoint": a["report57"]["checkpoint"],
                "dump": a["report57"]["dump"],
            },
        },
        "session_46": b["report46"],
    }
    write_json_report(RECORDINGS_DIR / "l5_56_57_46_summary.json", summary)
    return summary


if __name__ == "__main__":
    summary = main()
    d56 = summary["session_56_57"]["dump56"]
    print("=== 0x56 ===")
    print(
        "room", d56.get("room_hex"),
        "xy", d56.get("x"), d56.get("y"),
        "keys", d56.get("keys"),
        "doors", d56.get("doors"),
        "doorway", d56.get("doorway_mask"),
        "item", d56.get("item_id"),
        "whistle", d56.get("whistle_0x065C"),
        "candle", (d56.get("candle") or {}).get("name"),
    )
    print("OBJECTS56", d56.get("objects"))
    print("PNG56", RECORDINGS_DIR / "l5_56_recon.png")
    r57 = summary["session_56_57"]["report57"]
    d57 = r57.get("dump") or {}
    print("=== 0x57 ===")
    print("walked", r57.get("walked"), "cleared", r57.get("cleared"), "ckpt", r57.get("checkpoint"))
    print(
        "room", d57.get("room_hex"),
        "xy", d57.get("x"), d57.get("y"),
        "keys", d57.get("keys"),
        "doors", d57.get("doors"),
    )
    print("OBJECTS57", d57.get("objects"))
    r46 = summary["session_46"]
    d46 = r46.get("dump") or {}
    print("=== 0x46 ===")
    print("arrived", r46.get("ok"), "kind", r46.get("combat_vs_item"), "ckpt", r46.get("checkpoint"))
    print(
        "room", d46.get("room_hex"),
        "xy", d46.get("x"), d46.get("y"),
        "keys", d46.get("keys"),
        "doors", d46.get("doors"),
        "item", d46.get("item_id"), d46.get("item_name"),
        "whistle", d46.get("whistle_0x065C"),
    )
    print("OBJECTS46", d46.get("objects"))
    nxt = r46.get("next_room")
    if nxt:
        nd = nxt.get("dump") or {}
        print("=== next ===", nxt.get("direction"), nd.get("room_hex"), nd.get("objects"))
    else:
        print("=== next === none (no natural extra door, or hop skipped)")
    print("CHECKPOINTS", summary["checkpoints_written"])
    print("WHISTLE", summary["whistle_0x065C"])
    print("status_claim", None)
