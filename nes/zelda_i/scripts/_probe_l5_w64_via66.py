"""Level5Whistle64 → 0x65 bomb-east 0x66 → UP 0x56. North shutter is one-way."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import bomb_east_from_65, walk_axis, walk_east_from_64
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle64"


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [{"t": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
            for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF) and o.type_id < 0x40]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "objs": objs,
    }


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        print("START", dump(env), flush=True)

        rec = walk_east_from_64(env, assist, total)
        log.append({"tag": "east64", **rec, **dump(env)})
        print("EAST64", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w64_via66_65.png")

        rec = bomb_east_from_65(env, assist, total)
        log.append({"tag": "bomb_east", **rec, **dump(env)})
        print("BOMB_EAST", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w64_via66_66.png")

        s = read_snapshot(env.get_ram())
        if s.screen == 0x66 and s.mode == PLAY_MODE:
            walk_axis(env, assist, total, "y", 109, max_f=400)
            walk_axis(env, assist, total, "x", 120, max_f=400)
            walk_axis(env, assist, total, "y", 93, max_f=300)
            push_dir(env, assist, total, "UP", frames=280)
            idle(env, assist, total, 20)
            for _ in range(200):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen == 0x56:
                    break
                env.step(nes_idle_action())
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            rec = {"tag": "up66", **dump(env)}
            log.append(rec)
            print("UP66", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w64_via66_56.png")

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x66, 0x56, 0x57) and int(read_u8(env.get_ram(), ADDR_WHISTLE)) == 1:
            name = f"Level5Whistle{s.screen:02X}"
            write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, name),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={
                    "segment": name,
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x64 east, 0x65 bomb-east 0x66 (north shutter one-way), up 0x56",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1, "xy": [s.link_x, s.link_y]},
                natural_entry=False,
            )
            print("SAVED", name, dump(env), flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x56, "pokes": False, "log": log, "final": dump(env)}
        write_json_report(RECORDINGS_DIR / "l5_w64_via66.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
