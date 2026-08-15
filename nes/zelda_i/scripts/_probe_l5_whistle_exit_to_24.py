"""From Level5Whistle: exit 0x04 via short ladder, then scan 0x05 toward 0x24."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import exit_whistle_04, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"slot": o.slot, "type": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    return {
        "L": s.level,
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "item": s.room_item_id,
        "tile": int(s.colliding_tile),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "objs": objs,
    }


def wait_play(env, assist, total, n=240):
    for _ in range(n):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 8)
            return
        step(env, assist, total, nes_idle_action())


def try_door(env, assist, total, direction, ax, ay):
    room0 = read_snapshot(env.get_ram()).screen
    walk_axis(env, assist, total, "y", ay, max_f=400)
    walk_axis(env, assist, total, "x", ax, max_f=400)
    push_dir(env, assist, total, direction, frames=240)
    idle(env, assist, total, 12)
    wait_play(env, assist, total)
    s = read_snapshot(env.get_ram())
    rec = {"dir": direction, "changed": s.screen != room0, **dump(env)}
    print("DOOR", rec, flush=True)
    return rec


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        log.append({"tag": "start", **dump(env)})
        print("START", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_whistle_exit_start.png")

        ex = exit_whistle_04(env, assist, total)
        log.append({"tag": "exit04", **{k: ex[k] for k in ex if k != "log"}, "steps": ex.get("log")})
        print("EXIT04", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_whistle_exit_after.png")
        log.append({"tag": "after_exit", **dump(env)})

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen != 0x04:
            path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle05"), env.em.get_state())
            write_state_provenance(
                path,
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle.state",
                request={
                    "segment": "Level5Whistle05",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x04 short-ladder pit left-mouth",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle05 room", f"0x{s.screen:02x}", flush=True)

            if s.screen == 0x05:
                for direction, ax, ay in (
                    ("RIGHT", 224, 141),
                    ("DOWN", 120, 205),
                    ("UP", 120, 93),
                    ("LEFT", 32, 141),
                ):
                    if read_snapshot(env.get_ram()).screen != 0x05:
                        break
                    rec = try_door(env, assist, total, direction, ax, ay)
                    log.append(rec)
                    save_rgb_png(
                        env.step(nes_idle_action())[0],
                        RECORDINGS_DIR / f"l5_whistle05_{direction}.png",
                    )
                    if rec["changed"]:
                        break

        body = {
            "ok": read_snapshot(env.get_ram()).mode == PLAY_MODE
            and read_snapshot(env.get_ram()).screen != 0x04,
            "log": log,
            "final": dump(env),
            "pokes": False,
            "status_claim": None,
        }
        write_json_report(RECORDINGS_DIR / "l5_whistle_exit_to_24.json", body)
        print("FINAL", body["final"], flush=True)
        print("OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
