"""L5 backtrack: 0x56 push/bomb, 0x57 exits, 0x46 blocks, 0x66 west-if-keyhole.

Reuses dungeon_ops.goto/push_dir/exit_door/idle and dungeon_lab._drive_exit.
No pokes. No candle. No Clean STATUS. No east67. No 0x65 bombs.
Bomb only a cracked or ROM-documented wall on the rooms this hunt names.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon import DoorRoute
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_ops import exit_door, goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_stairs import (
    BLOCK_PUSH_STANDS,
    BLOCK_STAIRS_X,
    BLOCK_STAIRS_Y,
    STAIR_STANDS,
    stair_transition_modes,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_COLLIDING_TILE,
    ADDR_MAP,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

# L1-6 Q1 ROM 0x18700 (room 0x65 matched N=shutter S=wall W/E=bomb secret=none).
ROM_Q1 = {
    0x46: {"N": "wall", "S": "key", "W": "wall", "E": "open", "secret": "none"},
    0x56: {"N": "key", "S": "open", "W": "shutter", "E": "open", "secret": "foes_item"},
    0x57: {"N": "open", "S": "wall", "W": "open", "E": "wall", "secret": "foes_item"},
    0x66: {"N": "shutter", "S": "open", "W": "bomb", "E": "key", "secret": "all_dead"},
}

EXIT_ROUTES = {
    "UP": DoorRoute("UP", ((120, 141), (120, 93))),
    "DOWN": DoorRoute("DOWN", ((120, 141), (120, 205))),
    "LEFT": DoorRoute("LEFT", ((120, 141), (32, 141))),
    "RIGHT": DoorRoute("RIGHT", ((120, 141), (208, 141))),
}

# Classic post-shutter / center-or-side block stands (reuse L9 + L5-65 grid).
PUSH_TRIALS = (
    # stand just right of center, push LEFT (classic)
    *((stand, "LEFT") for stand in BLOCK_PUSH_STANDS),
    # stand left of center, push RIGHT
    ((104, 144), "RIGHT"),
    ((96, 144), "RIGHT"),
    ((80, 141), "RIGHT"),
    # toward center from south / north
    ((120, 173), "UP"),
    ((120, 157), "UP"),
    ((120, 109), "DOWN"),
    ((120, 125), "DOWN"),
    # side blocks into a recess
    ((80, 125), "RIGHT"),
    ((160, 125), "LEFT"),
    ((80, 157), "RIGHT"),
    ((160, 157), "LEFT"),
    ((96, 125), "DOWN"),
    ((144, 125), "DOWN"),
    ((96, 157), "UP"),
    ((144, 157), "UP"),
)


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


def dump(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    c = compact_snapshot(snap)
    c["doors"] = decode(snap.cur_opened_doors)
    c["doorway_mask"] = decode(snap.open_doorway_mask)
    c["room_hex"] = f"0x{snap.screen:02x}"
    c["next_room_hex"] = f"0x{snap.next_screen:02x}"
    c["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    c["candle_0x065B"] = int(read_u8(ram, ADDR_CANDLE))
    c["map_0x0668"] = int(read_u8(ram, ADDR_MAP))
    c["selected_item_0x0656"] = int(read_u8(ram, ADDR_SELECTED_ITEM))
    c["colliding_tile"] = int(read_u8(ram, ADDR_COLLIDING_TILE))
    c["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
            "state": obj.state,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]
    return c


def open_env(state: str):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def snap_sig(env) -> tuple:
    s = read_snapshot(env.get_ram())
    return (
        s.screen,
        s.mode,
        int(s.cur_opened_doors) & 0x0F,
        int(s.open_doorway_mask) & 0x0F,
        s.room_item_id,
    )


def push_from(env, assist, total, stand, direction, frames=100) -> dict:
    sx, sy = stand
    room0 = read_snapshot(env.get_ram()).screen
    goto(env, assist, total, sx, sy, tol=3, max_f=400)
    if read_snapshot(env.get_ram()).screen != room0:
        return {"ok": False, "left_room": True, "stand": [sx, sy], "direction": direction}
    sig0 = snap_sig(env)
    tile0 = int(read_u8(env.get_ram(), ADDR_COLLIDING_TILE))
    x0 = read_snapshot(env.get_ram()).link_x
    y0 = read_snapshot(env.get_ram()).link_y
    for _ in range(frames):
        step(env, assist, total, nes_action(direction))
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or snap.screen != room0:
            d = dump(env)
            return {
                "ok": True,
                "kind": "stairs_or_room",
                "stand": [sx, sy],
                "direction": direction,
                "dump": d,
            }
        sig = (
            snap.screen,
            snap.mode,
            int(snap.cur_opened_doors) & 0x0F,
            int(snap.open_doorway_mask) & 0x0F,
            snap.room_item_id,
        )
        if sig != sig0:
            d = dump(env)
            return {
                "ok": True,
                "kind": "door_or_item",
                "stand": [sx, sy],
                "direction": direction,
                "sig0": list(sig0),
                "sig": list(sig),
                "dump": d,
            }
    snap = read_snapshot(env.get_ram())
    return {
        "ok": False,
        "stand": [sx, sy],
        "direction": direction,
        "xy": [snap.link_x, snap.link_y],
        "moved": [snap.link_x - x0, snap.link_y - y0],
        "tile": int(read_u8(env.get_ram(), ADDR_COLLIDING_TILE)),
        "tile0": tile0,
    }


def hunt_pushes(env, assist, total, room: int) -> dict:
    opened = None
    tried = []
    for stand, direction in PUSH_TRIALS:
        if read_snapshot(env.get_ram()).screen != room:
            break
        rec = push_from(env, assist, total, stand, direction, frames=90)
        tried.append({k: rec[k] for k in rec if k != "dump"})
        if rec.get("ok"):
            opened = rec
            print("PUSH_OPEN", room, rec.get("kind"), stand, direction, flush=True)
            break
    return {"opened": opened, "tried_n": len(tried), "tried": tried}


def walk_stair_stands(env, assist, total, room: int) -> dict:
    """After pushes, stand on engine stair spawn + other hole candidates."""
    hits = []
    for tx, ty in ((BLOCK_STAIRS_X, BLOCK_STAIRS_Y),) + STAIR_STANDS:
        if read_snapshot(env.get_ram()).screen != room:
            d = dump(env)
            return {"left_room": True, "dump": d, "hits": hits}
        goto(env, assist, total, tx, ty, tol=2, max_f=300)
        idle(env, assist, total, 20)
        snap = read_snapshot(env.get_ram())
        tile = int(read_u8(env.get_ram(), ADDR_COLLIDING_TILE))
        row = {
            "stand": [tx, ty],
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "tile": tile,
            "stair_mode": stair_transition_modes(snap.mode),
        }
        if stair_transition_modes(snap.mode) or snap.screen != room:
            hits.append(row)
            return {"left_room": True, "hit": row, "hits": hits, "dump": dump(env)}
        # stair tiles 0x70-0x73, black mouth 0x24
        if 0x70 <= tile <= 0x73 or tile == 0x24:
            hits.append(row)
    return {"left_room": False, "hits": hits, "dump": dump(env)}


def wall_shots(env, assist, total, stem: str) -> dict:
    """Stand at each mouth and screenshot. No bombs here."""
    shots = {}
    room0 = read_snapshot(env.get_ram()).screen
    for name, (x, y), face in (
        ("north", (120, 93), "UP"),
        ("south", (120, 205), "DOWN"),
        ("west", (32, 141), "LEFT"),
        ("east", (208, 141), "RIGHT"),
    ):
        if read_snapshot(env.get_ram()).screen != room0:
            break
        goto(env, assist, total, x, y, tol=4, max_f=400)
        for _ in range(6):
            step(env, assist, total, nes_action(face))
        obs, *_ = env.step(nes_idle_action())
        path = RECORDINGS_DIR / f"{stem}_{name}.png"
        save_rgb_png(obs, path)
        snap = read_snapshot(env.get_ram())
        shots[name] = {
            "png": str(path.resolve()),
            "xy": [snap.link_x, snap.link_y],
            "room": f"0x{snap.screen:02x}",
            "tile": int(read_u8(env.get_ram(), ADDR_COLLIDING_TILE)),
            "doors": decode(snap.cur_opened_doors),
            "mask": decode(snap.open_doorway_mask),
        }
    return shots


def probe_four(state_bytes, room: int, stem: str) -> list:
    probes = []
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        raw = _drive_exit(
            state_bytes,
            spec_room=room,
            route=EXIT_ROUTES[direction],
            screenshot_path=RECORDINGS_DIR / f"{stem}_{direction.lower()}.png",
            max_frames=900,
        )
        dest = None
        if raw.get("success"):
            dest = (raw.get("room_hex") or f"0x{raw.get('room', 0):02x}").lower()
        probes.append(
            {
                "direction": direction,
                "success": bool(raw.get("success")),
                "sealed": not raw.get("success"),
                "dest_room": dest,
                "objects": raw.get("objects"),
                "x": raw.get("x"),
                "y": raw.get("y"),
                "screenshot": raw.get("screenshot"),
            }
        )
        print("EXIT", f"0x{room:02x}", direction, dest or "sealed", flush=True)
    return probes


def session_56() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Cleared56")
    total = [1]
    try:
        idle(env, assist, total, 30)
        start = dump(env)
        save_rgb_png(obs if obs is not None else env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_56_push.png")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_56_push.png")
        print("56_START", start["room_hex"], start["doors"], start["doorway_mask"], start["keys"], start["bombs"], flush=True)

        pushes = hunt_pushes(env, assist, total, 0x56)
        after_push = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_56_after_push.png")

        stairs = walk_stair_stands(env, assist, total, 0x56)
        after_stairs = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_56_stairs_check.png")

        walls = {}
        if read_snapshot(env.get_ram()).screen == 0x56:
            walls = wall_shots(env, assist, total, "l5_56_wall")

        # ROM 0x56 has no bomb walls (N=key S=open W=shutter E=open). Do not invent.
        bomb = {
            "tried": False,
            "reason": "ROM L1-6 Q1 0x56 N=key S=open W=shutter E=open secret=foes_item; no bomb door. Bomb only if a wall shot shows a crack.",
            "rom": ROM_Q1[0x56],
        }

        dest = None
        if after_stairs.get("left_room") or stair_transition_modes(after_stairs.get("dump", {}).get("mode", 5)):
            dest = after_stairs.get("dump", {}).get("room_hex")

        report = {
            "from_state": "Level5Cleared56",
            "pokes": False,
            "status_claim": None,
            "rom": ROM_Q1[0x56],
            "start": start,
            "push": {
                "opened": bool(pushes.get("opened")),
                "kind": (pushes.get("opened") or {}).get("kind"),
                "stand": (pushes.get("opened") or {}).get("stand"),
                "direction": (pushes.get("opened") or {}).get("direction"),
                "tried_n": pushes.get("tried_n"),
            },
            "after_push": after_push,
            "stairs": {
                "left_room": stairs.get("left_room"),
                "hit": stairs.get("hit"),
                "hits": stairs.get("hits"),
                "dest": dest,
            },
            "walls": walls,
            "bomb": bomb,
            "whistle_0x065C": after_stairs.get("dump", after_push).get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_56_push.json", report)
        return report
    finally:
        env.close()


def session_57() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Cleared57")
    total = [1]
    try:
        idle(env, assist, total, 40)
        start = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_57_exits.png")
        print("57_START", start["room_hex"], start["doors"], start["doorway_mask"], start["keys"], flush=True)
        state_bytes = env.em.get_state()
        env.close()
        env = None
        probes = probe_four(state_bytes, 0x57, "l5_57_exit")
        report = {
            "from_state": "Level5Cleared57",
            "pokes": False,
            "status_claim": None,
            "rom": ROM_Q1[0x57],
            "start": start,
            "doors": start["doors"],
            "doorway_mask": start["doorway_mask"],
            "probes": probes,
            "whistle_0x065C": start.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_57_exits.json", report)
        return report
    finally:
        if env is not None:
            env.close()


def session_46() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Entered46")
    total = [1]
    try:
        idle(env, assist, total, 30)
        start = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_blocks.png")
        print("46_START", start["room_hex"], start["doors"], start["doorway_mask"], start["item_id"] if "item_id" in start else start.get("room_item_id"), start["map_0x0668"], flush=True)

        # Recapture map if still on the floor (item 0x17 / 23).
        map_pick = None
        if start.get("room_item_id") in (0x17, 23) or start.get("map_0x0668", 0) == 0:
            goto(env, assist, total, 120, 141, tol=3, max_f=400)
            idle(env, assist, total, 80)
            after = dump(env)
            map_pick = {
                "map_before": start.get("map_0x0668"),
                "map_after": after.get("map_0x0668"),
                "item_after": after.get("room_item_id"),
                "xy": [after.get("x"), after.get("y")],
            }
            print("46_MAP", map_pick, flush=True)
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / "l5_46_after_map.png")

        pushes = hunt_pushes(env, assist, total, 0x46)
        after_push = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_after_push.png")
        stairs = walk_stair_stands(env, assist, total, 0x46)

        walls = {}
        if read_snapshot(env.get_ram()).screen == 0x46:
            walls = wall_shots(env, assist, total, "l5_46_wall")

        # ROM E=open not bomb. User: bomb east ONLY if cracked.
        bomb = {
            "tried": False,
            "reason": "ROM L1-6 Q1 0x46 E=open (not bomb) secret=none. East bomb skipped unless wall shot shows a crack.",
            "rom": ROM_Q1[0x46],
        }

        report = {
            "from_state": "Level5Entered46",
            "pokes": False,
            "status_claim": None,
            "rom": ROM_Q1[0x46],
            "start": start,
            "map_pick": map_pick,
            "push": {
                "opened": bool(pushes.get("opened")),
                "kind": (pushes.get("opened") or {}).get("kind"),
                "stand": (pushes.get("opened") or {}).get("stand"),
                "direction": (pushes.get("opened") or {}).get("direction"),
                "tried_n": pushes.get("tried_n"),
            },
            "after_push": after_push,
            "stairs": {
                "left_room": stairs.get("left_room"),
                "hit": stairs.get("hit"),
                "dest": (stairs.get("dump") or {}).get("room_hex") if stairs.get("left_room") else None,
            },
            "walls": walls,
            "bomb": bomb,
            "whistle_0x065C": after_push.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_46_blocks.json", report)
        return report
    finally:
        env.close()


def session_66() -> dict:
    """From Cleared56 walk DOWN to 0x66. West only if keyhole visible."""
    configure_headless()
    env, assist, obs = open_env("Level5Cleared56")
    total = [1]
    try:
        idle(env, assist, total, 20)
        start56 = dump(env)
        print("66_FROM56", start56["room_hex"], start56["keys"], start56["doors"], flush=True)
        hop = exit_door(env, assist, total, "DOWN")
        idle(env, assist, total, 40)
        at66 = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_66_from56.png")
        print("66_ARRIVE", at66["room_hex"], at66["doors"], at66["doorway_mask"], at66["keys"], flush=True)

        # Face west wall and screenshot.
        goto(env, assist, total, 32, 141, tol=3, max_f=500)
        for _ in range(8):
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 8)
        west = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_66_west.png")

        # ROM W=bomb, not key. User: try west with keys ONLY if keyhole visible.
        # Do not invent a key door. Do not bomb here (hunt said keyhole-only).
        key_try = None
        west_action = "skip_solid_or_no_keyhole"
        # Live bits: if west already open, walk it. If doorway west, it is not a keyhole.
        if west["doors"].get("west") or west["doorway_mask"].get("west"):
            hopw = exit_door(env, assist, total, "LEFT")
            idle(env, assist, total, 40)
            after = dump(env)
            key_try = {
                "reason": "west_bit_already_open_or_doorway",
                "changed": hopw.get("changed_room"),
                "dest": after.get("room_hex"),
                "keys": after.get("keys"),
            }
            west_action = "walked_existing_west"
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / "l5_66_west_dest.png")
        else:
            # No west bit. ROM says bomb, live previously solid at keys=2.
            # Screenshot is the keyhole evidence; do not spend a key on a solid wall.
            west_action = "skip_no_keyhole_bit"
            key_try = {
                "reason": "no_west_door_bit; ROM W=bomb not key; skip unless screenshot shows keyhole",
                "keys": west.get("keys"),
                "rom_west": "bomb",
            }

        report = {
            "from_state": "Level5Cleared56",
            "pokes": False,
            "status_claim": None,
            "rom": ROM_Q1[0x66],
            "hop_down": {
                "changed": hop.get("changed_room"),
                "result": hop.get("result"),
                "after": hop.get("after"),
            },
            "at66": at66,
            "west": west,
            "west_action": west_action,
            "key_try": key_try,
            "whistle_0x065C": west.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_66_west.json", report)
        return report
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_backtrack.py",
    ]
    r56 = session_56()
    r57 = session_57()
    r46 = session_46()
    r66 = session_66()

    new_dests = []
    if r56["push"]["opened"] or r56["stairs"]["dest"]:
        new_dests.append({"room": "0x56", "via": "push/stairs", "dest": r56["stairs"]["dest"]})
    for p in r57["probes"]:
        if p["success"] and p["dest_room"] not in (None, "0x57", "0x56"):
            new_dests.append({"room": "0x57", "via": p["direction"], "dest": p["dest_room"]})
        elif p["success"]:
            new_dests.append({"room": "0x57", "via": p["direction"], "dest": p["dest_room"], "known": True})
    if r46["push"]["opened"] or r46["stairs"]["dest"]:
        new_dests.append({"room": "0x46", "via": "push/stairs", "dest": r46["stairs"]["dest"]})
    if (r66.get("key_try") or {}).get("changed"):
        new_dests.append({"room": "0x66", "via": "west", "dest": r66["key_try"].get("dest")})

    summary = {
        "status_claim": None,
        "pokes": False,
        "commands": commands,
        "whistle_0x065C": r56.get("whistle_0x065C"),
        "candle_0x065B": 0,
        "0x56": {
            "push": r56["push"],
            "bomb": r56["bomb"],
            "stairs_dest": r56["stairs"]["dest"],
            "doors": r56["start"]["doors"],
            "rom": r56["rom"],
        },
        "0x57": {
            "doors": r57["doors"],
            "doorway_mask": r57["doorway_mask"],
            "exits": [{"direction": p["direction"], "dest": p["dest_room"] or "sealed"} for p in r57["probes"]],
            "rom": r57["rom"],
        },
        "0x46": {
            "push": r46["push"],
            "bomb": r46["bomb"],
            "map_pick": r46["map_pick"],
            "rom": r46["rom"],
        },
        "0x66": {
            "west_action": r66["west_action"],
            "key_try": r66["key_try"],
            "doors": (r66.get("at66") or {}).get("doors"),
            "keys": (r66.get("at66") or {}).get("keys"),
            "rom": r66["rom"],
        },
        "new_dests": new_dests,
        "which_room_opened_new_dest": new_dests[0]["room"] if new_dests else "none",
    }
    write_json_report(RECORDINGS_DIR / "l5_backtrack_summary.json", summary)
    return summary


if __name__ == "__main__":
    s = main()
    print("=== SUMMARY ===", flush=True)
    print("56 push", s["0x56"]["push"], "stairs", s["0x56"]["stairs_dest"], "bomb", s["0x56"]["bomb"]["tried"], flush=True)
    print("57 exits", s["0x57"]["exits"], flush=True)
    print("46 push", s["0x46"]["push"], "bomb", s["0x46"]["bomb"]["tried"], flush=True)
    print("66", s["0x66"]["west_action"], s["0x66"]["key_try"], flush=True)
    print("NEW", s["which_room_opened_new_dest"], s["new_dests"], flush=True)
    print("WHISTLE", s["whistle_0x065C"], flush=True)
    print("status_claim", None, flush=True)
