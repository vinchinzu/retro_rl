"""0x06: around diamond, push 0x68, stand on revealed stairs -> 0x07 -> 0x64."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import exit_door, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8


def dump(env):
    s = read_snapshot(env.get_ram())
    blocks = [
        {"x": o.x, "y": o.y, "hp": o.hp}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id == 0x68
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "blocks": blocks,
    }


def wait_play(env, assist, total, max_f=240):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            return
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])


def in_cellar(env):
    s = read_snapshot(env.get_ram())
    return s.mode in (9, 10, 11, 16) or s.screen == 0x07


def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle05", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        exit_door(env, assist, total, "RIGHT", push=200)
        wait_play(env, assist, total)
        print("ARRIVE", dump(env), flush=True)

        # North around to east of block, push LEFT.
        for axis, tgt in (("x", 96), ("y", 93), ("x", 160), ("y", 144), ("x", 128)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
            rec = {"tag": f"{axis}{tgt}", "ok": ok, **dump(env)}
            log.append(rec)
            print("WALK", rec, flush=True)
            if in_cellar(env):
                break

        if not in_cellar(env):
            # Face left and push the 0x68
            for _ in range(12):
                env.step(nes_action("LEFT"))
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            push_dir(env, assist, total, "LEFT", frames=120)
            rec = {"tag": "push_left", **dump(env)}
            log.append(rec)
            print("PUSH", rec, flush=True)
            idle(env, assist, total, 16)

        if not in_cellar(env):
            # Stand on typical stairs / former block tile
            for tx, ty in ((112, 144), (120, 141), (96, 144), (112, 141), (120, 144)):
                walk_axis(env, assist, total, "y", ty, max_f=300)
                walk_axis(env, assist, total, "x", tx, max_f=300)
                idle(env, assist, total, 10)
                rec = {"tag": f"stand_{tx}_{ty}", **dump(env)}
                log.append(rec)
                print("STAND", rec, flush=True)
                if in_cellar(env):
                    break
                for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                    push_dir(env, assist, total, d, frames=16)
                    if in_cellar(env):
                        break
                if in_cellar(env):
                    break

        # Fallback: south around + push UP from below
        if not in_cellar(env):
            print("FALLBACK south", flush=True)
            for axis, tgt in (("x", 160), ("y", 189), ("x", 112), ("y", 160)):
                walk_axis(env, assist, total, axis, tgt, max_f=400)
                print("  ", axis, tgt, dump(env), flush=True)
            push_dir(env, assist, total, "UP", frames=120)
            print("PUSH_UP", dump(env), flush=True)
            walk_axis(env, assist, total, "y", 144, max_f=200)
            walk_axis(env, assist, total, "x", 112, max_f=200)
            idle(env, assist, total, 12)
            print("STAND112", dump(env), flush=True)

        wait_play(env, assist, total, 80)
        after = dump(env)
        print("AFTER", after, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_push.png")

        dest = after
        if in_cellar(env) or after["sc"] == "0x07":
            cellar = cellar_other_mouth(env, assist, total)
            print("CELLAR", {k: cellar[k] for k in cellar if k != "start"}, flush=True)
            wait_play(env, assist, total)
            dest = dump(env)
            print("DEST", dest, flush=True)
            if dest["sc"] != "0x64":
                walk_axis(env, assist, total, "y", 189, max_f=400)
                sx = read_snapshot(env.get_ram()).link_x
                tx = 48 if sx > 128 else 192
                walk_axis(env, assist, total, "x", tx, max_f=500)
                push_dir(env, assist, total, "UP", frames=260)
                wait_play(env, assist, total)
                dest = dump(env)
                print("DEST2", dest, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_64_from06.png")
            if dest["sc"] == "0x64" and dest["mode"] == 5:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": "Level5Whistle05",
                        "via": "0x06 push 0x68 stairs, 0x07 other mouth",
                        "key_poke": False, "door_poke": False, "bomb_count_poke": False, "selected_item_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dest, flush=True)

        write_json_report(RECORDINGS_DIR / "l5_06_push.json", {"log": log, "final": dest, "pokes": False})
        print("FINAL", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
