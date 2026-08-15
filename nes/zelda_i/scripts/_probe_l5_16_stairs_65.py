"""Level5Whistle16 UP to 0x06, south-gap center stairs, 0x07 left to 0x64, EAST 0x65."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis
from zelda_i.level9_stairs import stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle16"


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
    }


def wait_play(env, assist, total):
    for _ in range(240):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 8)
            return
        step(env, assist, total, nes_action("UP"))


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


def stairs_06(env, assist, total):
    walk_axis(env, assist, total, "y", 173, max_f=400)
    walk_axis(env, assist, total, "x", 120, max_f=500)
    walk_axis(env, assist, total, "y", 157, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    print("GAP", dump(env), flush=True)
    for direction in ("UP", "DOWN", "UP"):
        for _ in range(100):
            s = read_snapshot(env.get_ram())
            if stair_transition_modes(s.mode) or s.screen != 0x06:
                break
            if s.link_x > 160:
                step(env, assist, total, nes_action("LEFT"))
            elif s.link_x < 80:
                step(env, assist, total, nes_action("RIGHT"))
            else:
                step(env, assist, total, nes_action(direction))
        s = read_snapshot(env.get_ram())
        print("NUDGE", direction, dump(env), flush=True)
        if stair_transition_modes(s.mode) or s.screen != 0x06:
            break
    wait_play(env, assist, total)
    idle(env, assist, total, 16)
    return dump(env)


def main():
    configure_headless()
    env = None
    total = [1]
    hops = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        print("START", dump(env), flush=True)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 12)
        wait_play(env, assist, total)
        hops.append({"hop": "16_north", **dump(env)})
        print("AT06", hops[-1], flush=True)

        end = stairs_06(env, assist, total)
        hops.append({"hop": "06_stairs", **end})
        print("STAIRS", end, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_stairs_ok.png")

        s = read_snapshot(env.get_ram())
        if s.mode in (9, 10, 11) or s.screen == 0x07:
            cellar = cellar_other_mouth(env, assist, total)
            hops.append({"hop": "07_mouth", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", cellar, flush=True)
        print("AFTER", dump(env), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w_after07.png")

        s = read_snapshot(env.get_ram())
        if s.screen == 0x64:
            walk_axis(env, assist, total, "y", 141, max_f=400)
            walk_axis(env, assist, total, "x", 224, max_f=500)
            push_dir(env, assist, total, "RIGHT", frames=240)
            idle(env, assist, total, 12)
            wait_play(env, assist, total)
            hops.append({"hop": "64_east", **dump(env)})
            print("AT65", hops[-1], flush=True)
            save(env, "Level5Whistle65", "0x16 north 0x06 stairs 0x07 0x64 east")
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65.png")

        write_json_report(RECORDINGS_DIR / "l5_16_stairs_65.json", {"hops": hops, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
