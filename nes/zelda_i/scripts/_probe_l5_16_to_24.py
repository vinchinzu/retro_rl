"""From Level5Whistle16: south toward 0x26/0x25/0x24. Whistle already 1."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis, walk_west_from_25, walk_west_from_26
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle16"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"t": f"0x{o.type_id:02x}", "hp": o.hp, "xy": [o.x, o.y]}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": s.keys,
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "item": s.room_item_id,
        "objs": objs[:8],
    }


def wait_play(env, assist, total):
    for _ in range(240):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 8)
            return
        step(env, assist, total, nes_action("DOWN"))


def door(env, assist, total, direction, ax, ay):
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


def save(env, name, via):
    s = read_snapshot(env.get_ram())
    write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        state_path(GAME_DIR, GAME, name),
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": STATE,
            "via": via,
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
        natural_entry=False,
    )


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
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w16.png")

        # Prefer SOUTH (0x26)
        rec = door(env, assist, total, "DOWN", 120, 205)
        log.append(rec)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w16_south.png")
        if not rec["changed"]:
            for direction, ax, ay in (("RIGHT", 224, 141), ("LEFT", 32, 141), ("UP", 120, 93)):
                if read_snapshot(env.get_ram()).screen != 0x16:
                    break
                rec = door(env, assist, total, direction, ax, ay)
                log.append(rec)
                if rec["changed"]:
                    break

        s = read_snapshot(env.get_ram())
        if s.screen != 0x16:
            save(env, f"Level5Whistle{s.screen:02X}", f"0x16 {log[-1].get('dir')}")
            print("DEST", dump(env), flush=True)

        # If 0x26, go west to 0x25 then 0x24
        if read_snapshot(env.get_ram()).screen == 0x26:
            w = walk_west_from_26(env, assist, total)
            print("W26", w, flush=True)
            log.append({"tag": "west26", **w, **dump(env)})
        if read_snapshot(env.get_ram()).screen == 0x25:
            save(env, "Level5Whistle25", "0x16 south -> 0x26 west")
            w = walk_west_from_25(env, assist, total)
            print("W25", w, flush=True)
            log.append({"tag": "west25", **w, **dump(env)})
        if read_snapshot(env.get_ram()).screen == 0x24:
            save(env, "Level5Whistle24", "west to Digdogger door")
            print("AT24", dump(env), flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w24.png")

        write_json_report(RECORDINGS_DIR / "l5_16_to_24.json", {"log": log, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
