"""Align to exact door tiles on 0x47 (C-blocks pinch x=128) then push N/S/W."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared47"
ROOM = 0x47
KNOWN = {0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}

# Prior walk: UP reached (128,93), DOWN (128,189), LEFT (32,149).
# Door mouths are (120,93) / (120,205) / (32,141). Slide the last 8px then push.
APPROACH = {
    "UP": ((128, 93), (120, 93), "UP"),
    "DOWN": ((128, 189), (120, 189), "DOWN"),
    "LEFT": ((32, 149), (32, 141), "LEFT"),
}


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
            idle(env, assist, total, 16)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def walk_dir(direction: str) -> dict:
    stand, mouth, face = APPROACH[direction]
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        start = dump(env)
        goto(env, assist, total, stand[0], stand[1], tol=3, max_f=500)
        at_stand = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        goto(env, assist, total, mouth[0], mouth[1], tol=2, max_f=400)
        at_mouth = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        # extra nudge toward exact mouth
        for _ in range(20):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - mouth[0]) <= 1 and abs(snap.link_y - mouth[1]) <= 1:
                break
            if abs(snap.link_x - mouth[0]) > 1:
                step(env, assist, total, nes_action("RIGHT" if snap.link_x < mouth[0] else "LEFT"))
            else:
                step(env, assist, total, nes_action("DOWN" if snap.link_y < mouth[1] else "UP"))
        at_mouth2 = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, face, frames=140)
        idle(env, assist, total, 20)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=200)
        idle(env, assist, total, 16)
        after = dump(env)
        png = RECORDINGS_DIR / f"l5_47_align_{direction.lower()}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        dest = after.get("room_hex")
        changed = dest != "0x47"
        print(
            "ALIGN",
            direction,
            "stand",
            at_stand,
            "mouth",
            at_mouth,
            at_mouth2,
            "dest",
            dest,
            "mode",
            after.get("mode"),
            "xy",
            [after.get("x"), after.get("y")],
            flush=True,
        )
        return {
            "direction": direction,
            "changed": changed,
            "sealed": not changed,
            "dest_room": dest if changed else None,
            "dest_room_id": after.get("room") if changed else None,
            "known": (after.get("room") in KNOWN) if changed else None,
            "at_stand": at_stand,
            "at_mouth": at_mouth,
            "at_mouth_nudge": at_mouth2,
            "after_xy": [after.get("x"), after.get("y")],
            "after_mode": after.get("mode"),
            "after_doors": after.get("doors"),
            "after_objects": after.get("objects"),
            "room_item_id": after.get("room_item_id"),
            "screenshot": str(png.resolve()),
            "dump": after,
        }
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    walks = [walk_dir(d) for d in ("UP", "DOWN", "LEFT")]
    new = [w for w in walks if w.get("changed") and not w.get("known")]
    known = [w for w in walks if w.get("changed") and w.get("known")]
    sealed = [w["direction"] for w in walks if w.get("sealed")]
    pocket = all((not w.get("changed")) or w.get("known") for w in walks)
    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "method": "align_then_push",
        "walks": walks,
        "new_dests": [{"direction": w["direction"], "dest": w["dest_room"]} for w in new],
        "known_dests": [{"direction": w["direction"], "dest": w["dest_room"]} for w in known],
        "sealed": sealed,
        "pocket": pocket,
        "whistle_0x065C": (walks[0].get("dump") or {}).get("whistle_0x065C") if walks else 0,
    }
    write_json_report(RECORDINGS_DIR / "l5_47_exits.json", report)
    write_json_report(RECORDINGS_DIR / "l5_47_exits3.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("NEW", r["new_dests"])
    print("KNOWN", r["known_dests"])
    print("SEALED", r["sealed"])
    print("POCKET", r["pocket"])
