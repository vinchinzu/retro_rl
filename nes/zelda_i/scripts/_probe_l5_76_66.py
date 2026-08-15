"""One-off live recon: Level5EastKey 0x77 WEST→0x76, then NORTH→0x66.

Controller-only. No pokes. No UP from 0x77. No Whistle route.
Bomb walls skipped: none documented on 0x77/0x76/0x66.
Not a route runner. Not Clean STATUS.

Drive style matches dungeon_lab._drive_exit (DoorRoute waypoints, then hold).
"""
from __future__ import annotations

from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon import DoorRoute
from zelda_i.dungeon_trace import compact_snapshot
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
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

STATE = "Level5EastKey"
ROOM_77 = 0x77
ROOM_76 = 0x76
ROOM_66 = 0x66
PRIOR_66_DOORS = 0x08
CANDLE_NAMES = {0: "none", 1: "blue", 2: "red"}


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
        "blue": raw == 1,
        "red": raw == 2,
        "present": raw > 0,
        "selected_item": selected,
        "selected_item_addr": "0x0656",
        "selected_is_candle": selected == 4,
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
    return compact


def drive_exit_on_env(env, *, spec_room: int, route: DoorRoute, max_frames: int) -> dict:
    """In-env clone of dungeon_lab._drive_exit; keeps the same env for chaining."""
    waypoint_index = 0
    entered = False
    play_frames = 0
    obs = None
    for frame in range(max_frames):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if snap.screen != spec_room and snap.mode == PLAY_MODE:
            play_frames += 1
            if play_frames >= 40:
                return {
                    "success": True,
                    "direction": route.direction,
                    "frames": frame,
                    "room": snap.screen,
                    "dump": dump_live(snap, ram),
                    "obs": obs,
                }
            action = nes_idle_action()
        elif snap.screen == spec_room and snap.mode == PLAY_MODE and not entered:
            if waypoint_index < len(route.waypoints):
                tx, ty = route.waypoints[waypoint_index]
                dx = tx - snap.link_x
                dy = ty - snap.link_y
                if abs(dx) <= 2 and abs(dy) <= 2:
                    waypoint_index += 1
                    action = nes_idle_action()
                elif abs(dx) > 2:
                    action = nes_action("RIGHT" if dx > 0 else "LEFT")
                else:
                    action = nes_action("DOWN" if dy > 0 else "UP")
            else:
                entered = True
                action = nes_action(route.direction)
        elif snap.transitioning or entered:
            action = nes_action(route.direction)
        else:
            action = nes_idle_action()
        obs, *_ = env.step(action)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "success": False,
        "direction": route.direction,
        "frames": max_frames,
        "room": snap.screen,
        "mode": snap.mode,
        "x": snap.link_x,
        "y": snap.link_y,
        "dump": dump_live(snap, ram),
        "obs": obs,
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png_76 = RECORDINGS_DIR / "l5_76_recon.png"
    png_66 = RECORDINGS_DIR / "l5_66_recon.png"
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        for _ in range(20):
            obs, *_ = env.step(nes_idle_action())
        ram = env.get_ram()
        start = dump_live(read_snapshot(ram), ram)

        west_route = DoorRoute("LEFT", ((32, 165), (32, 141)))
        west = drive_exit_on_env(
            env, spec_room=ROOM_77, route=west_route, max_frames=1200
        )
        if west.get("obs") is not None:
            save_rgb_png(west["obs"], png_76)
        landed_76 = bool(west["success"]) and west.get("room") == ROOM_76

        north = None
        if landed_76:
            north_route = DoorRoute(
                "UP",
                ((208, 141), (208, 157), (120, 157), (120, 61)),
            )
            north = drive_exit_on_env(
                env, spec_room=ROOM_76, route=north_route, max_frames=1500
            )
            if north.get("obs") is not None:
                save_rgb_png(north["obs"], png_66)
    finally:
        env.close()

    bomb_note = {
        "tried": False,
        "reason": (
            "No documented bomb walls on 0x77/0x76/0x66 in "
            "docs/LEVEL5_ROUTE.md or AGENTS.md. Wiki bomb-skip is later "
            "Dodongo rooms, not these. Skipped."
        ),
    }
    report_76 = {
        "ok": landed_76,
        "status_claim": None,
        "from_state": STATE,
        "pokes": False,
        "bomb_or_candle": False,
        "bomb_attempt": bomb_note,
        "start_77": start,
        "west_attempt": {
            "direction": "LEFT",
            "method": "dungeon_lab._drive_exit-style DoorRoute LEFT ((32,165),(32,141))  # south of 0x77 block clusters, then door band",
            "opened": landed_76,
            "sealed": not landed_76,
            "dest_room": f"0x{west['room']:02x}" if west.get("room") is not None else None,
            "frames": west.get("frames"),
            "lab": {k: v for k, v in west.items() if k not in {"dump", "obs"}},
        },
        "dump": west.get("dump"),
        "screenshot": str(png_76.resolve()),
    }
    write_json_report(RECORDINGS_DIR / "l5_76_recon.json", report_76)

    report_66 = None
    if north is not None:
        entered_66 = bool(north["success"]) and north.get("room") == ROOM_66
        dump66 = north.get("dump")
        if entered_66 and dump66 is not None:
            doors_now = dump66["doors"]["raw"]
            key_door_note = (
                f"0x66 doors now 0x{doors_now:02x} "
                f"(prior post-clear 0x{PRIOR_66_DOORS:02x}); "
                f"north_bit={dump66['doors']['north']} "
                f"west_bit={dump66['doors']['west']}; "
                + (
                    "door bits changed vs prior 0x08"
                    if doors_now != PRIOR_66_DOORS
                    else "no new door bits vs prior 0x08"
                )
            )
        else:
            key_door_note = "did_not_enter_0x66"
        report_66 = {
            "ok": entered_66,
            "status_claim": None,
            "from_state": STATE,
            "via": "0x77 WEST 0x76 NORTH",
            "pokes": False,
            "bomb_or_candle": False,
            "bomb_attempt": bomb_note,
            "north_attempt": {
                "direction": "UP",
                "method": (
                    "dungeon_lab._drive_exit-style DoorRoute UP "
                    "((208,141),(208,157),(120,157),(120,61))"
                ),
                "opened": bool(north["success"]),
                "sealed": not north["success"],
                "dest_room": (
                    f"0x{north['room']:02x}" if north.get("room") is not None else None
                ),
                "entered_66": entered_66,
                "frames": north.get("frames"),
                "lab": {k: v for k, v in north.items() if k not in {"dump", "obs"}},
            },
            "dump": dump66,
            "prior_66_doors": f"0x{PRIOR_66_DOORS:02x}",
            "key_door_newly_opened": key_door_note,
            "screenshot": str(png_66.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_66_recon.json", report_66)

    return {"start": start, "west": report_76, "north": report_66}


if __name__ == "__main__":
    report = main()
    start = report["start"]
    west = report["west"]
    north = report["north"]
    print(
        "START77",
        start.get("room_hex"),
        "xy",
        start.get("x"),
        start.get("y"),
        "keys",
        start.get("keys"),
        "candle",
        start.get("candle"),
    )
    dump76 = west.get("dump") or {}
    print(
        "ROOM76",
        dump76.get("room_hex"),
        "xy",
        dump76.get("x"),
        dump76.get("y"),
        "keys",
        dump76.get("keys"),
        "doors",
        dump76.get("doors"),
        "candle",
        dump76.get("candle"),
    )
    print("OBJECTS76", dump76.get("objects"))
    print(
        "WEST opened",
        west["west_attempt"]["opened"],
        "dest",
        west["west_attempt"]["dest_room"],
    )
    print("PNG76", west["screenshot"])
    print("JSON76", RECORDINGS_DIR / "l5_76_recon.json")
    if north is None:
        print("NORTH skipped (WEST did not land 0x76)")
    else:
        dump66 = north.get("dump") or {}
        print(
            "ROOM66",
            dump66.get("room_hex"),
            "xy",
            dump66.get("x"),
            dump66.get("y"),
            "keys",
            dump66.get("keys"),
            "doors",
            dump66.get("doors"),
            "candle",
            dump66.get("candle"),
        )
        print("OBJECTS66", dump66.get("objects"))
        print(
            "NORTH opened",
            north["north_attempt"]["opened"],
            "entered_66",
            north["north_attempt"]["entered_66"],
            "dest",
            north["north_attempt"]["dest_room"],
        )
        print("KEYDOOR", north.get("key_door_newly_opened"))
        print("PNG66", north["screenshot"])
        print("JSON66", RECORDINGS_DIR / "l5_66_recon.json")
    print("BOMB skipped (none documented on 0x77/0x76/0x66)")
    print("status_claim", None)
