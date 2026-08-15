"""Level5Whistle64 → east bomb 0x65 → north 0x55 → east 0x56. No pokes."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle64"


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"t": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF) and o.type_id < 0x40
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "objs": objs,
    }


def door(env, assist, total, direction, ax, ay, frames=240):
    walk_axis(env, assist, total, "y", ay, max_f=400)
    walk_axis(env, assist, total, "x", ax, max_f=400)
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, direction, frames=frames)
    idle(env, assist, total, 16)
    for _ in range(200):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen != room0:
            break
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    return dump(env)


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

        # Leave 0x64 diamond south then east wall, y=141 bomb hole.
        for axis, tgt in (("y", 189), ("x", 208), ("y", 141), ("x", 224)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=450)
            print("NAV64", axis, tgt, ok, dump(env), flush=True)
        rec = door(env, assist, total, "RIGHT", 224, 141)
        log.append({"tag": "east64", **rec})
        print("EAST64", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w64_east.png")

        if read_snapshot(env.get_ram()).screen == 0x65:
            rec = door(env, assist, total, "UP", 120, 93)
            log.append({"tag": "north65", **rec})
            print("NORTH65", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_north.png")

        if read_snapshot(env.get_ram()).screen == 0x55:
            rec = door(env, assist, total, "RIGHT", 224, 141)
            log.append({"tag": "east55", **rec})
            print("EAST55", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w55_east.png")

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x55, 0x56, 0x65):
            name = f"Level5Whistle{s.screen:02X}"
            write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / f"{name}.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle64.state",
                request={"segment": name, "via": "0x64 bomb-east 0x65 north 0x55 east", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED", name, flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x56, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w64_to_56.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
