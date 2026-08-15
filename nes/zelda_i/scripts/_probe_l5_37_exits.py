"""From Level5Cleared37: unstick, grab compass 0x16, walk N/S exits."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_COMPASS, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared37"
ROOM = 0x37
LEVEL5_COMPASS_BIT = 0x10
KNOWN = {0x37, 0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}

# Center-south hunt; screenshot showed compass south of mid.
COMPASS_GRID = tuple(
    (x, y)
    for y in (141, 149, 157, 165, 173, 181, 189)
    for x in (80, 96, 112, 120, 128, 144, 160)
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
    c["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    c["compass_0x0667"] = int(read_u8(ram, ADDR_COMPASS))
    c["compass_l5"] = bool(int(read_u8(ram, ADDR_COMPASS)) & LEVEL5_COMPASS_BIT)
    c["colliding_tile"] = snap.colliding_tile
    c["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
            "type_name": object_name(obj.type_id),
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
            "state": obj.state,
            "facing": obj.facing,
        }
        for obj in snap.objects
        if obj.type_id not in (0, 0xFF)
    ]
    return c


def open_env(state: str = STATE):
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


def wait_play(env, assist, total, room: int, max_f: int = 240) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.screen == room and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 16)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def unstick(env, assist, total) -> dict:
    """Leave a left-wall pinch: south band then x=120."""
    snap0 = read_snapshot(env.get_ram())
    start = [snap0.link_x, snap0.link_y]
    # Nudge cardinals if colliding.
    for direction in ("DOWN", "RIGHT", "UP", "LEFT", "DOWN", "RIGHT"):
        for _ in range(12):
            step(env, assist, total, nes_action(direction))
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - start[0]) > 4 or abs(snap.link_y - start[1]) > 4:
            break
    # Prefer south band (open like 0x47) then center x=120.
    goto(env, assist, total, snap0.link_x, 189, tol=4, max_f=400)
    goto(env, assist, total, 120, 189, tol=3, max_f=500)
    goto(env, assist, total, 120, 141, tol=3, max_f=500)
    snap = read_snapshot(env.get_ram())
    return {
        "start": start,
        "end": [snap.link_x, snap.link_y],
        "moved": [snap.link_x, snap.link_y] != start,
        "colliding_tile": snap.colliding_tile,
    }


def grab_compass(env, assist, total) -> dict:
    compass0 = int(read_u8(env.get_ram(), ADDR_COMPASS))
    tried = []
    for tx, ty in COMPASS_GRID:
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        if compass & LEVEL5_COMPASS_BIT:
            break
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM or snap.mode != PLAY_MODE:
            break
        ok = goto(env, assist, total, tx, ty, tol=2, max_f=280)
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        tried.append(
            {
                "stand": [tx, ty],
                "ok": ok,
                "xy": [snap.link_x, snap.link_y],
                "l5": bool(compass & LEVEL5_COMPASS_BIT),
            }
        )
        if compass & LEVEL5_COMPASS_BIT:
            break
    compass1 = int(read_u8(env.get_ram(), ADDR_COMPASS))
    snap = read_snapshot(env.get_ram())
    return {
        "compass_in": compass0,
        "compass_out": compass1,
        "grabbed": bool(compass1 & LEVEL5_COMPASS_BIT)
        and not bool(compass0 & LEVEL5_COMPASS_BIT),
        "have_l5": bool(compass1 & LEVEL5_COMPASS_BIT),
        "tried_n": len(tried),
        "xy": [snap.link_x, snap.link_y],
        "hits": [t for t in tried if t["l5"] or t["ok"]][:8],
        "last": tried[-1] if tried else None,
        "pokes": False,
    }


def enter(direction: str, via, mouth, face: str, tag: str) -> dict:
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        stuck = unstick(env, assist, total)
        for tx, ty in via:
            goto(env, assist, total, tx, ty, tol=3, max_f=450)
        goto(env, assist, total, mouth[0], mouth[1], tol=2, max_f=400)
        for _ in range(24):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - mouth[0]) <= 1 and abs(snap.link_y - mouth[1]) <= 1:
                break
            if abs(snap.link_x - mouth[0]) > 1:
                step(
                    env,
                    assist,
                    total,
                    nes_action("RIGHT" if snap.link_x < mouth[0] else "LEFT"),
                )
            else:
                step(
                    env,
                    assist,
                    total,
                    nes_action("DOWN" if snap.link_y < mouth[1] else "UP"),
                )
        at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, face, frames=160)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 24)
        after = dump(env)
        png = RECORDINGS_DIR / f"{tag}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        dest = after.get("room_hex")
        changed = dest != "0x37"
        json_path = RECORDINGS_DIR / f"{tag}.json"
        body = {
            "via": f"0x37 {direction}",
            "ok": changed,
            "unstick": stuck,
            "at_mouth": at,
            "dump": after,
            "screenshot": str(png.resolve()),
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(json_path, body)
        print(
            "DEST",
            direction,
            dest,
            "changed",
            changed,
            "mouth",
            at,
            "xy",
            [after.get("x"), after.get("y")],
            "item",
            after.get("room_item_id"),
            "objs",
            after.get("objects"),
            flush=True,
        )
        return {
            "direction": direction,
            "changed": changed,
            "sealed": not changed,
            "dest_room": dest if changed else None,
            "known": (after.get("room") in KNOWN) if changed else None,
            "at_mouth": at,
            "unstick": stuck,
            "dump": after,
            "screenshot": str(png.resolve()),
            "dump_path": str(json_path.resolve()),
        }
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump(env)
        print(
            "START",
            start.get("room_hex"),
            "xy",
            [start.get("x"), start.get("y")],
            "compass",
            start.get("compass_0x0667"),
            "l5",
            start.get("compass_l5"),
            "tile",
            start.get("colliding_tile"),
            "objs",
            start.get("objects"),
            flush=True,
        )
        stuck = unstick(env, assist, total)
        print("UNSTICK", stuck, flush=True)
        pick = grab_compass(env, assist, total)
        print("COMPASS", pick, flush=True)
        after = dump(env)
        png = RECORDINGS_DIR / "l5_37_compass.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        write_json_report(
            RECORDINGS_DIR / "l5_37_compass.json",
            {
                "start": start,
                "unstick": stuck,
                "pick": pick,
                "after": after,
                "screenshot": str(png.resolve()),
                "status_claim": None,
                "pokes": False,
            },
        )
    finally:
        env.close()

    north = enter("UP", ((120, 141), (120, 93)), (120, 93), "UP", "l5_27_from37")
    south = enter("DOWN", ((120, 141), (120, 205)), (120, 205), "DOWN", "l5_47_from37")
    # If north failed, try north-band slide like 0x47 dests.
    if not north.get("changed"):
        north2 = enter(
            "UP",
            ((128, 93), (120, 93)),
            (120, 93),
            "UP",
            "l5_27_from37",
        )
        if north2.get("changed"):
            north = north2
        else:
            north = {**north, "retry": {k: v for k, v in north2.items() if k != "dump"}}
    if not south.get("changed"):
        south2 = enter(
            "DOWN",
            ((128, 189), (120, 205)),
            (120, 205),
            "DOWN",
            "l5_47_from37",
        )
        if south2.get("changed"):
            south = south2
        else:
            south = {**south, "retry": {k: v for k, v in south2.items() if k != "dump"}}

    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "compass": pick,
        "N": {
            "dest": north.get("dest_room"),
            "known": north.get("known"),
            "sealed": not north.get("changed"),
            "at_mouth": north.get("at_mouth"),
        },
        "S": {
            "dest": south.get("dest_room"),
            "known": south.get("known"),
            "sealed": not south.get("changed"),
            "at_mouth": south.get("at_mouth"),
        },
        "E": "wall_skipped",
        "W": "wall_skipped",
        "next_dest": north.get("dest_room")
        if north.get("changed") and not north.get("known")
        else (south.get("dest_room") if south.get("changed") and not south.get("known") else None),
        "whistle_0x065C": (north.get("dump") or {}).get("whistle_0x065C"),
    }
    write_json_report(RECORDINGS_DIR / "l5_37_exits.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("COMPASS", r.get("compass"))
    print("N", r["N"])
    print("S", r["S"])
    print("NEXT", r["next_dest"])
    print("WHISTLE", r["whistle_0x065C"])
