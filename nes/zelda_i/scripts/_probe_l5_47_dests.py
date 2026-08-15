"""Dump 0x47 dests: N=0x37 (new), S=0x57, retry W. From Level5Cleared47."""
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
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared47"
KNOWN = {0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}


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
    c["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
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


def wait_play(env, assist, total, room: int, max_f: int = 240) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.screen == room and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 20)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def enter(direction: str, via, mouth, face: str, tag: str) -> dict:
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        for tx, ty in via:
            goto(env, assist, total, tx, ty, tol=3, max_f=400)
        goto(env, assist, total, mouth[0], mouth[1], tol=2, max_f=400)
        for _ in range(24):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - mouth[0]) <= 1 and abs(snap.link_y - mouth[1]) <= 1:
                break
            if abs(snap.link_x - mouth[0]) > 1:
                step(env, assist, total, nes_action("RIGHT" if snap.link_x < mouth[0] else "LEFT"))
            else:
                step(env, assist, total, nes_action("DOWN" if snap.link_y < mouth[1] else "UP"))
        at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, face, frames=150)
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
        changed = dest != "0x47"
        json_path = RECORDINGS_DIR / f"{tag}.json"
        body = {
            "via": f"0x47 {direction}",
            "ok": changed,
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
            "mode",
            after.get("mode"),
            "xy",
            [after.get("x"), after.get("y")],
            "doors",
            after.get("doors"),
            "item",
            after.get("room_item_id"),
            "objs",
            after.get("objects"),
            flush=True,
        )
        return {
            "direction": direction,
            "changed": changed,
            "dest_room": dest if changed else None,
            "known": (after.get("room") in KNOWN) if changed else None,
            "at_mouth": at,
            "dump": after,
            "screenshot": str(png.resolve()),
            "dump_path": str(json_path.resolve()),
        }
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    north = enter("UP", ((128, 93),), (120, 93), "UP", "l5_37_from47")
    south = enter("DOWN", ((128, 189),), (120, 205), "DOWN", "l5_57_from47")
    # West: first climb to the corridor that previously reached x=32, then slide y.
    west = enter(
        "LEFT",
        ((128, 141), (96, 141), (64, 141), (32, 149)),
        (32, 141),
        "LEFT",
        "l5_47_west",
    )
    # If still in 0x47, try a north-band then west (around the C).
    if not west.get("changed"):
        west2 = enter(
            "LEFT",
            ((128, 93), (80, 93), (80, 141), (32, 141)),
            (32, 141),
            "LEFT",
            "l5_47_west",
        )
        west = {**west, "retry": west2}
        if west2.get("changed"):
            west = west2
    walks = [north, south, west]
    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "N": {"dest": north.get("dest_room"), "known": north.get("known"), "dump": north.get("dump")},
        "S": {"dest": south.get("dest_room"), "known": south.get("known"), "dump": south.get("dump")},
        "W": {"dest": west.get("dest_room"), "known": west.get("known"), "sealed": not west.get("changed"), "dump": west.get("dump")},
        "E": "wall_skipped",
        "next_dest": north.get("dest_room") if north.get("changed") and not north.get("known") else None,
        "pocket": False,
        "whistle_0x065C": (north.get("dump") or {}).get("whistle_0x065C"),
    }
    write_json_report(RECORDINGS_DIR / "l5_47_exits.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("N", r["N"]["dest"], "known", r["N"]["known"])
    print("S", r["S"]["dest"], "known", r["S"]["known"])
    print("W", r["W"]["dest"], "sealed", r["W"]["sealed"])
    print("NEXT", r["next_dest"], "POCKET", r["pocket"])
