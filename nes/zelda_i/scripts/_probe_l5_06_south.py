"""From Level5Whistle05: EAST 0x06, key SOUTH, dump dest. Fallback south-gap stairs."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import take_center_stairs_64, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": s.keys,
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
    }


def wait_play(env, assist, total):
    for _ in range(240):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 8)
            return
        step(env, assist, total, nes_action("DOWN"))


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
        print("START", dump(env), flush=True)
        walk_axis(env, assist, total, "y", 141, max_f=400)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        push_dir(env, assist, total, "RIGHT", frames=220)
        idle(env, assist, total, 12)
        wait_play(env, assist, total)
        log.append({"tag": "at06", **dump(env)})
        print("AT06", log[-1], flush=True)

        # Pull off west door, go south band, then south door
        walk_axis(env, assist, total, "x", 48, max_f=300)
        walk_axis(env, assist, total, "y", 189, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 205, max_f=300)
        log.append({"tag": "south_stand", **dump(env)})
        print("SOUTH_STAND", log[-1], flush=True)
        keys0 = read_snapshot(env.get_ram()).keys
        push_dir(env, assist, total, "DOWN", frames=260)
        idle(env, assist, total, 16)
        wait_play(env, assist, total)
        log.append({"tag": "after_south", "keys0": keys0, **dump(env)})
        print("AFTER_SOUTH", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_south.png")

        s = read_snapshot(env.get_ram())
        if s.screen != 0x06:
            save(env, f"Level5Whistle{s.screen:02X}", "0x06 key south")
            print("SAVED dest", dump(env), flush=True)
        else:
            # south-gap stairs like 0x64
            print("SOUTH_LOCKED_TRY_STAIRS", flush=True)
            walk_axis(env, assist, total, "y", 173, max_f=400)
            walk_axis(env, assist, total, "x", 120, max_f=400)
            walk_axis(env, assist, total, "y", 157, max_f=300)
            walk_axis(env, assist, total, "x", 120, max_f=200)
            for d in ("UP", "DOWN", "UP"):
                for _ in range(90):
                    s = read_snapshot(env.get_ram())
                    if s.screen != 0x06 or s.mode != PLAY_MODE:
                        break
                    if s.link_x > 160:
                        step(env, assist, total, nes_action("LEFT"))
                    elif s.link_x < 80:
                        step(env, assist, total, nes_action("RIGHT"))
                    else:
                        step(env, assist, total, nes_action(d))
                if read_snapshot(env.get_ram()).screen != 0x06:
                    break
            wait_play(env, assist, total)
            log.append({"tag": "stairs", **dump(env)})
            print("STAIRS", log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_stairs2.png")

        write_json_report(RECORDINGS_DIR / "l5_06_south.json", {"log": log, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
