"""Post-clear 0x65: settle, push 4-block, re-dump doors, walk exits. No pokes."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon import DoorRoute
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Cleared65"
ROOM = 0x65

BLOCKS = (
    (96, 125),
    (144, 125),
    (96, 157),
    (144, 157),
    (80, 125),
    (160, 125),
    (80, 157),
    (160, 157),
    (112, 125),
    (128, 125),
    (112, 157),
    (128, 157),
)

EXIT_ROUTES = {
    "UP": DoorRoute("UP", ((120, 141), (120, 93))),
    "DOWN": DoorRoute("DOWN", ((120, 141), (120, 205))),
    "LEFT": DoorRoute("LEFT", ((120, 141), (32, 141))),
    "RIGHT": DoorRoute("RIGHT", ((120, 141), (208, 141))),
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
    return c


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def push_from(env, assist, total, stand, direction, frames=120) -> dict:
    sx, sy = stand
    goto(env, assist, total, sx, sy, tol=3, max_f=400)
    doors0 = read_snapshot(env.get_ram()).cur_opened_doors
    mask0 = read_snapshot(env.get_ram()).open_doorway_mask
    x0, y0 = read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y
    for _ in range(frames):
        step(env, assist, total, nes_action(direction))
        snap = read_snapshot(env.get_ram())
        if snap.cur_opened_doors != doors0 or snap.open_doorway_mask != mask0:
            return {
                "ok": True,
                "stand": [sx, sy],
                "direction": direction,
                "doors": decode(snap.cur_opened_doors),
                "mask": decode(snap.open_doorway_mask),
                "xy": [snap.link_x, snap.link_y],
                "moved": [snap.link_x - x0, snap.link_y - y0],
            }
    snap = read_snapshot(env.get_ram())
    return {
        "ok": False,
        "stand": [sx, sy],
        "direction": direction,
        "doors": decode(snap.cur_opened_doors),
        "mask": decode(snap.open_doorway_mask),
        "xy": [snap.link_x, snap.link_y],
        "moved": [snap.link_x - x0, snap.link_y - y0],
    }


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
        start = dump(env)
        print("START", start["room_hex"], start["doors"], start["doorway_mask"], start["x"], start["y"], flush=True)

        samples = []
        for n in (30, 60, 120, 180):
            idle(env, assist, total, 30)
            d = dump(env)
            samples.append({"idle": n, "doors": d["doors"], "mask": d["doorway_mask"], "all_dead": d["room_all_dead"]})
            print("SETTLE", n, d["doors"], d["doorway_mask"], flush=True)

        # Hard UP from north mouth (return to 0x55).
        goto(env, assist, total, 120, 93, tol=2, max_f=500)
        room0 = read_snapshot(env.get_ram()).screen
        keys0 = read_snapshot(env.get_ram()).keys
        for _ in range(200):
            step(env, assist, total, nes_action("UP"))
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0:
                break
        up_try = dump(env)
        up_try["changed"] = up_try["room"] != room0
        up_try["keys_delta"] = up_try["keys"] - keys0
        print("UP_TRY", up_try["room_hex"], up_try["changed"], up_try["xy"] if False else [up_try["x"], up_try["y"]], up_try["doors"], flush=True)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_65_up_try.png")

        # If still in 0x65, try block pushes.
        pushes = []
        opened = None
        if read_snapshot(env.get_ram()).screen == ROOM:
            # Approach each block from the opposite side of the push.
            approaches = {
                "LEFT": (16, 0),
                "RIGHT": (-16, 0),
                "UP": (0, 16),
                "DOWN": (0, -16),
            }
            for bx, by in BLOCKS:
                for direction, (dx, dy) in approaches.items():
                    rec = push_from(env, assist, total, (bx + dx, by + dy), direction, frames=90)
                    pushes.append(rec)
                    if rec["ok"]:
                        opened = rec
                        print("PUSH_OPEN", rec, flush=True)
                        break
                if opened:
                    break
            print("PUSH_TRIED", len(pushes), "opened", bool(opened), flush=True)

        after_push = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_65_after_push.png")
        print("AFTER_PUSH", after_push["doors"], after_push["doorway_mask"], [after_push["x"], after_push["y"]], flush=True)

        # If still sealed, spend a key on north AFTER dump (dump already in l5_65_clear).
        key_try = None
        if (
            read_snapshot(env.get_ram()).screen == ROOM
            and after_push["doors"]["raw"] == 0
            and after_push["doorway_mask"]["raw"] == 0
            and read_snapshot(env.get_ram()).keys >= 1
        ):
            keys_before = int(read_snapshot(env.get_ram()).keys)
            hop = exit_door(env, assist, total, "UP", push=160)
            snap = read_snapshot(env.get_ram())
            key_try = {
                "direction": "UP",
                "keys_before": keys_before,
                "keys_after": int(snap.keys),
                "changed": hop.get("changed_room"),
                "after": hop.get("after"),
                "result": hop.get("result"),
            }
            print("KEY_UP", key_try, flush=True)
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / "l5_65_key_up.png")

        state_bytes = env.em.get_state()
        env.close()
        env = None

        probes = []
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            raw = _drive_exit(
                state_bytes,
                spec_room=read_snapshot  # placeholder, fixed below
                    and ROOM,
                route=EXIT_ROUTES[direction],
                screenshot_path=RECORDINGS_DIR / f"l5_65_exit2_{direction.lower()}.png",
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
        print("PROBES2", [(p["direction"], p["dest_room"] or "sealed") for p in probes], flush=True)

        report = {
            "from_state": STATE,
            "pokes": False,
            "status_claim": None,
            "start": start,
            "settle": samples,
            "up_try": {
                "changed": up_try.get("changed"),
                "room": up_try.get("room_hex"),
                "doors": up_try.get("doors"),
                "mask": up_try.get("doorway_mask"),
                "xy": [up_try.get("x"), up_try.get("y")],
                "keys_delta": up_try.get("keys_delta"),
            },
            "pushes_ok": [p for p in pushes if p.get("ok")],
            "pushes_n": len(pushes),
            "opened": opened,
            "after_push": {
                "doors": after_push["doors"],
                "mask": after_push["doorway_mask"],
                "xy": [after_push["x"], after_push["y"]],
                "room": after_push["room_hex"],
            },
            "key_try": key_try,
            "probes": probes,
            "whistle_0x065C": after_push.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_65_exits.json", {
            "from_room": "0x65",
            "doors": after_push["doors"],
            "doorway_mask": after_push["doorway_mask"],
            "probes": probes,
            "real_dests": [{"direction": p["direction"], "dest": p["dest_room"]} for p in probes if p.get("success")],
            "new_dests": [
                {"direction": p["direction"], "dest": p["dest_room"]}
                for p in probes
                if p.get("success") and p.get("dest_room") not in (None, "0x65", "0x55")
            ],
            "key_try": key_try,
            "block_push": opened,
            "status_claim": None,
            "pokes": False,
        })
        write_json_report(RECORDINGS_DIR / "l5_65_exits2.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("OPENED", r.get("opened"))
    print("AFTER", r.get("after_push"))
    print("KEY", r.get("key_try"))
    print("PROBES", [(p["direction"], p.get("dest_room") or "sealed") for p in r.get("probes") or []])
