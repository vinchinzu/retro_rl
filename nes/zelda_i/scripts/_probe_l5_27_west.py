"""From Level5Cleared27: leave x=160 ladder, spend west key, dump dest."""
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
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared27"
ROOM = 0x27


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


def walk_axis(env, assist, total, axis: str, target: int, max_f: int = 400) -> bool:
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


def try_path(env, assist, total, name: str, steps: list) -> dict:
    log = []
    snap = read_snapshot(env.get_ram())
    log.append({"step": "start", "xy": [snap.link_x, snap.link_y], "tile": snap.colliding_tile})
    for kind, a, b in steps:
        if kind == "axis":
            ok = walk_axis(env, assist, total, a, b, max_f=400)
        elif kind == "hold":
            for _ in range(b):
                step(env, assist, total, nes_action(a))
            ok = True
        else:
            ok = False
        snap = read_snapshot(env.get_ram())
        log.append(
            {
                "step": f"{kind}:{a}:{b}",
                "ok": ok,
                "xy": [snap.link_x, snap.link_y],
                "tile": snap.colliding_tile,
                "room": f"0x{snap.screen:02x}",
            }
        )
        print(name, log[-1], flush=True)
    return {"name": name, "log": log, "xy": [snap.link_x, snap.link_y]}


def push_west(env, assist, total) -> dict:
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 32, max_f=400)
    for _ in range(16):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 32) <= 2 and abs(snap.link_y - 141) <= 2:
            break
        if abs(snap.link_y - 141) > 2:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < 141 else "UP"))
        else:
            step(env, assist, total, nes_action("LEFT" if snap.link_x > 32 else "RIGHT"))
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    keys0 = int(read_snapshot(env.get_ram()).keys)
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, "LEFT", frames=220)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room0:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 20)
    after = dump(env)
    return {
        "at_mouth": at,
        "keys_in": keys0,
        "keys_out": after.get("keys"),
        "key_spent": int(after.get("keys") or 0) < keys0,
        "changed": after.get("room_hex") != "0x27",
        "dest": after.get("room_hex"),
        "dump": after,
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        (
            "off_ladder_down_south173",
            [
                ("hold", "DOWN", 40),
                ("axis", "y", 173),
                ("axis", "x", 64),
                ("axis", "y", 173),
                ("axis", "x", 32),
                ("axis", "y", 141),
            ],
        ),
        (
            "off_ladder_up_north109",
            [
                ("hold", "UP", 40),
                ("axis", "y", 109),
                ("axis", "x", 64),
                ("axis", "y", 109),
                ("axis", "x", 32),
                ("axis", "y", 141),
            ],
        ),
        (
            "south189_then_west",
            [
                ("hold", "DOWN", 50),
                ("axis", "y", 189),
                ("axis", "x", 40),
                ("axis", "y", 141),
            ],
        ),
        (
            "y157_then_wall",
            [
                ("hold", "DOWN", 30),
                ("axis", "y", 157),
                ("axis", "x", 32),
                ("axis", "y", 141),
            ],
        ),
    ]

    results = []
    west = None
    for name, steps in paths:
        env, assist, obs = open_env()
        total = [1]
        try:
            idle(env, assist, total, 40)  # item-get freeze
            start = dump(env)
            print("START", name, [start.get("x"), start.get("y")], "tile", start.get("colliding_tile"), flush=True)
            nav = try_path(env, assist, total, name, steps)
            pushed = push_west(env, assist, total)
            print(
                "PUSH",
                name,
                "changed",
                pushed.get("changed"),
                "dest",
                pushed.get("dest"),
                "mouth",
                pushed.get("at_mouth"),
                "keys",
                pushed.get("keys_in"),
                "->",
                pushed.get("keys_out"),
                flush=True,
            )
            rec = {
                "name": name,
                "start_xy": [start.get("x"), start.get("y")],
                "nav": nav,
                "at_mouth": pushed.get("at_mouth"),
                "changed": pushed.get("changed"),
                "dest": pushed.get("dest"),
                "keys_in": pushed.get("keys_in"),
                "keys_out": pushed.get("keys_out"),
                "key_spent": pushed.get("key_spent"),
                "objects": (pushed.get("dump") or {}).get("objects"),
                "doors": (pushed.get("dump") or {}).get("doors"),
                "item": (pushed.get("dump") or {}).get("room_item_id"),
            }
            results.append(rec)
            if pushed.get("changed"):
                png = RECORDINGS_DIR / "l5_27_west.png"
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, png)
                write_json_report(
                    RECORDINGS_DIR / "l5_27_west.json",
                    {
                        "via": "0x27 LEFT key",
                        "ok": True,
                        "path": name,
                        "at_mouth": pushed.get("at_mouth"),
                        "keys_in": pushed.get("keys_in"),
                        "keys_out": pushed.get("keys_out"),
                        "key_spent": pushed.get("key_spent"),
                        "dump": pushed.get("dump"),
                        "screenshot": str(png.resolve()),
                        "status_claim": None,
                        "pokes": False,
                    },
                )
                west = rec
                west["dump"] = pushed.get("dump")
                west["screenshot"] = str(png.resolve())
                break
        finally:
            env.close()

    if west is None:
        # last attempt screenshot of failure
        env, assist, obs = open_env()
        total = [1]
        try:
            idle(env, assist, total, 20)
            png = RECORDINGS_DIR / "l5_27_west.png"
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, png)
        finally:
            env.close()

    report = {
        "from_state": STATE,
        "pokes": False,
        "status_claim": None,
        "tried": [{k: v for k, v in r.items() if k != "dump"} for r in results],
        "ok": bool(west and west.get("changed")),
        "dest": (west or {}).get("dest"),
        "path": (west or {}).get("name"),
        "whistle_0x065C": 0,
    }
    write_json_report(RECORDINGS_DIR / "l5_27_west_tries.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "DEST", r.get("dest"), "PATH", r.get("path"))
    for t in r.get("tried") or []:
        print(" ", t.get("name"), "changed", t.get("changed"), "mouth", t.get("at_mouth"), "dest", t.get("dest"))
