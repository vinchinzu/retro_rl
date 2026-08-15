"""Re-walk 0x47 N/S/W from Level5Cleared47. Center tile between C-blocks stuck _drive_exit."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared47"
ROOM = 0x47
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
            idle(env, assist, total, 12)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def recenter(env, assist, total) -> dict:
    """Leave the C-block pinch: go to x=120, then a side recess if still stuck."""
    snap0 = read_snapshot(env.get_ram())
    xy0 = [snap0.link_x, snap0.link_y]
    goto(env, assist, total, 120, 141, tol=2, max_f=400)
    snap = read_snapshot(env.get_ram())
    if abs(snap.link_x - 120) > 4 or abs(snap.link_y - 141) > 4:
        # side recesses of the C shapes
        for tx, ty in ((80, 141), (64, 141), (160, 141), (176, 141), (80, 173), (80, 109)):
            goto(env, assist, total, tx, ty, tol=3, max_f=350)
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM:
                break
            if abs(snap.link_x - tx) <= 4:
                break
    snap = read_snapshot(env.get_ram())
    return {"from": xy0, "to": [snap.link_x, snap.link_y], "room": f"0x{snap.screen:02x}"}


def walk_dir(direction: str) -> dict:
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump(env)
        rec = recenter(env, assist, total)
        print("RECENTER", direction, rec, flush=True)
        hop = exit_door(env, assist, total, direction)
        dest_id = hop.get("after", {}).get("screen")
        if hop.get("changed_room") and dest_id is not None:
            wait_play(env, assist, total, dest_id, max_f=200)
        idle(env, assist, total, 20)
        after = dump(env)
        png = RECORDINGS_DIR / f"l5_47_walk_{direction.lower()}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        dest = after.get("room_hex")
        changed = hop.get("changed_room") and dest != "0x47"
        report = {
            "direction": direction,
            "changed": bool(changed),
            "sealed": not changed,
            "dest_room": dest if changed else None,
            "dest_room_id": after.get("room") if changed else None,
            "known": (after.get("room") in KNOWN) if changed else None,
            "recenter": rec,
            "start_xy": [start.get("x"), start.get("y")],
            "after_xy": [after.get("x"), after.get("y")],
            "after_mode": after.get("mode"),
            "after_doors": after.get("doors"),
            "after_objects": after.get("objects"),
            "room_item_id": after.get("room_item_id"),
            "hop_result": hop.get("result"),
            "screenshot": str(png.resolve()),
            "dump": after,
        }
        print(
            "WALK",
            direction,
            "changed" if changed else "sealed",
            dest,
            after.get("mode"),
            [after.get("x"), after.get("y")],
            flush=True,
        )
        return report
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    walks = [walk_dir(d) for d in ("UP", "DOWN", "LEFT")]
    # east only if a walk showed live east
    new = [w for w in walks if w.get("changed") and not w.get("known")]
    known = [w for w in walks if w.get("changed") and w.get("known")]
    sealed = [w for w in walks if w.get("sealed")]
    pocket = all((not w.get("changed")) or w.get("known") for w in walks)
    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "walks": walks,
        "new_dests": [{"direction": w["direction"], "dest": w["dest_room"]} for w in new],
        "known_dests": [{"direction": w["direction"], "dest": w["dest_room"]} for w in known],
        "sealed": [w["direction"] for w in sealed],
        "pocket": pocket,
        "whistle_0x065C": (walks[0].get("dump") or {}).get("whistle_0x065C") if walks else 0,
    }
    write_json_report(RECORDINGS_DIR / "l5_47_exits.json", {**report, "note": "rewalk from Level5Cleared47 after center-tile stick"})
    write_json_report(RECORDINGS_DIR / "l5_47_exits2.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("NEW", r["new_dests"])
    print("KNOWN", r["known_dests"])
    print("SEALED", r["sealed"])
    print("POCKET", r["pocket"])
    print("WHISTLE", r["whistle_0x065C"])
