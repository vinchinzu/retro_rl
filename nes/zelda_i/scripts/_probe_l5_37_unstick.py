"""Y-first unstick from x=56 column on Level5Cleared37, then compass + N/S."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_COMPASS, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared37"
LEVEL5_COMPASS_BIT = 0x10
KNOWN = {0x37, 0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}


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
            "type_hex": f"0x{obj.type_id:02x}",
            "type_name": object_name(obj.type_id),
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]
    return c


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
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


def hold(env, assist, total, direction: str, frames: int) -> None:
    for _ in range(frames):
        step(env, assist, total, nes_action(direction))


def walk_axis(env, assist, total, axis: str, target: int, max_f: int = 400) -> bool:
    """Walk only one axis to target (y-first / x-second)."""
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
    return False


def wait_play(env, assist, total, room: int, max_f: int = 240) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.screen == room and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 16)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def to_center(env, assist, total) -> dict:
    """From x=56 column: north or south band, then slide to x=120."""
    snap = read_snapshot(env.get_ram())
    start = [snap.link_x, snap.link_y]
    log = [{"at": start, "step": "start"}]
    # Try north band first (y=93 is door corridor).
    ok_y = walk_axis(env, assist, total, "y", 93, max_f=300)
    snap = read_snapshot(env.get_ram())
    log.append({"at": [snap.link_x, snap.link_y], "ok_y93": ok_y})
    ok_x = walk_axis(env, assist, total, "x", 120, max_f=400)
    snap = read_snapshot(env.get_ram())
    log.append({"at": [snap.link_x, snap.link_y], "ok_x120_from_n": ok_x})
    if abs(snap.link_x - 120) <= 2:
        walk_axis(env, assist, total, "y", 141, max_f=300)
        snap = read_snapshot(env.get_ram())
        log.append({"at": [snap.link_x, snap.link_y], "centered_via": "north_band"})
        return {"ok": True, "via": "north_band", "log": log, "xy": [snap.link_x, snap.link_y]}

    # South band fallback.
    ok_y = walk_axis(env, assist, total, "y", 189, max_f=400)
    snap = read_snapshot(env.get_ram())
    log.append({"at": [snap.link_x, snap.link_y], "ok_y189": ok_y})
    ok_x = walk_axis(env, assist, total, "x", 120, max_f=400)
    snap = read_snapshot(env.get_ram())
    log.append({"at": [snap.link_x, snap.link_y], "ok_x120_from_s": ok_x})
    if abs(snap.link_x - 120) <= 2:
        walk_axis(env, assist, total, "y", 141, max_f=300)
        snap = read_snapshot(env.get_ram())
        log.append({"at": [snap.link_x, snap.link_y], "centered_via": "south_band"})
        return {"ok": True, "via": "south_band", "log": log, "xy": [snap.link_x, snap.link_y]}

    # Mid bands y=109 / 173.
    for ty in (109, 117, 173, 157, 205, 85):
        walk_axis(env, assist, total, "y", ty, max_f=250)
        walk_axis(env, assist, total, "x", 120, max_f=250)
        snap = read_snapshot(env.get_ram())
        log.append({"at": [snap.link_x, snap.link_y], "try_y": ty})
        if abs(snap.link_x - 120) <= 2:
            return {"ok": True, "via": f"y{ty}", "log": log, "xy": [snap.link_x, snap.link_y]}
    snap = read_snapshot(env.get_ram())
    return {"ok": False, "via": None, "log": log, "xy": [snap.link_x, snap.link_y]}


def hunt_compass(env, assist, total) -> dict:
    compass0 = int(read_u8(env.get_ram(), ADDR_COMPASS))
    # y-first then x, dense center-south.
    spots = []
    for ty in (141, 157, 165, 173, 181, 125, 117):
        for tx in (120, 112, 128, 96, 144, 80, 160):
            spots.append((tx, ty))
    tried = []
    for tx, ty in spots:
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        if compass & LEVEL5_COMPASS_BIT:
            break
        walk_axis(env, assist, total, "y", ty, max_f=200)
        walk_axis(env, assist, total, "x", tx, max_f=200)
        idle(env, assist, total, 6)
        snap = read_snapshot(env.get_ram())
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        tried.append(
            {
                "stand": [tx, ty],
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
        "first_hit": next((t for t in tried if t["l5"]), None),
        "pokes": False,
    }


def walk_door(env, assist, total, direction: str) -> dict:
    mouth = {"UP": (120, 93), "DOWN": (120, 205)}[direction]
    # Always re-center via bands, then align mouth, then push.
    center = to_center(env, assist, total)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    walk_axis(env, assist, total, "y", mouth[1], max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    for _ in range(16):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 1 and abs(snap.link_y - mouth[1]) <= 1:
            break
        if abs(snap.link_x - 120) > 1:
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < 120 else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < mouth[1] else "UP"))
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, direction, frames=170)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room0:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 20)
    after = dump(env)
    dest = after.get("room_hex")
    changed = dest != "0x37"
    tag = f"l5_{snap.screen:02x}_from37" if changed else f"l5_37_exit_{direction.lower()}"
    png = RECORDINGS_DIR / f"{tag}.png"
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, png)
    write_json_report(
        RECORDINGS_DIR / f"{tag}.json",
        {
            "via": f"0x37 {direction}",
            "ok": changed,
            "center": center,
            "at_mouth": at,
            "dump": after,
            "screenshot": str(png.resolve()),
            "status_claim": None,
            "pokes": False,
        },
    )
    return {
        "direction": direction,
        "changed": changed,
        "sealed": not changed,
        "dest_room": dest if changed else None,
        "known": (after.get("room") in KNOWN) if changed else None,
        "at_mouth": at,
        "center": center,
        "xy": [after.get("x"), after.get("y")],
        "objects": after.get("objects"),
        "room_item_id": after.get("room_item_id"),
        "doors": after.get("doors"),
        "screenshot": str(png.resolve()),
        "dump": after,
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        start = dump(env)
        print("START", start.get("xy") if False else [start.get("x"), start.get("y")],
              "tile", start.get("colliding_tile"), flush=True)
        center = to_center(env, assist, total)
        print("CENTER", center, flush=True)
        pick = hunt_compass(env, assist, total)
        print("COMPASS", pick, flush=True)
        after = dump(env)
        png = RECORDINGS_DIR / "l5_37_compass.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        write_json_report(
            RECORDINGS_DIR / "l5_37_compass.json",
            {
                "start": start,
                "center": center,
                "pick": pick,
                "after": {
                    k: after[k]
                    for k in (
                        "room_hex",
                        "x",
                        "y",
                        "compass_0x0667",
                        "compass_l5",
                        "room_item_id",
                        "objects",
                    )
                },
                "screenshot": str(png.resolve()),
                "status_claim": None,
                "pokes": False,
            },
        )
    finally:
        env.close()

    # Fresh env per exit so a failed N doesn't poison S.
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 8)
        north = walk_door(env, assist, total, "UP")
        print("N", north.get("dest_room"), north.get("at_mouth"), north.get("xy"), flush=True)
    finally:
        env.close()

    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 8)
        south = walk_door(env, assist, total, "DOWN")
        print("S", south.get("dest_room"), south.get("at_mouth"), south.get("xy"), flush=True)
    finally:
        env.close()

    next_dest = None
    if north.get("changed") and not north.get("known"):
        next_dest = north.get("dest_room")
    elif south.get("changed") and not south.get("known"):
        next_dest = south.get("dest_room")
    elif north.get("changed"):
        next_dest = north.get("dest_room")

    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "center": center,
        "compass": pick,
        "N": {
            "dest": north.get("dest_room"),
            "known": north.get("known"),
            "sealed": north.get("sealed"),
            "at_mouth": north.get("at_mouth"),
            "xy": north.get("xy"),
            "objects": north.get("objects"),
            "room_item_id": north.get("room_item_id"),
        },
        "S": {
            "dest": south.get("dest_room"),
            "known": south.get("known"),
            "sealed": south.get("sealed"),
            "at_mouth": south.get("at_mouth"),
            "xy": south.get("xy"),
            "objects": south.get("objects"),
            "room_item_id": south.get("room_item_id"),
        },
        "E": "wall_rom",
        "W": "wall_rom",
        "next_dest": next_dest,
        "whistle_0x065C": (north.get("dump") or {}).get("whistle_0x065C"),
    }
    write_json_report(RECORDINGS_DIR / "l5_37_exits.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("CENTER_OK", (r.get("center") or {}).get("ok"), (r.get("center") or {}).get("via"))
    print("COMPASS", r.get("compass"))
    print("N", r["N"])
    print("S", r["S"])
    print("NEXT", r["next_dest"])
