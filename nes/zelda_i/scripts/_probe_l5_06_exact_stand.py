"""After 0x06 north-push 0x68, stand EXACT (120,141) for CheckWarp. tol=0."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis, walk_east_from_05
from zelda_i.level9_stairs import on_stair_tile, on_warp_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5WhistleFloor"
CELLAR = (9, 10, 11, 16)


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    blocks = [{"x": o.x, "y": o.y} for o in s.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "tile_h": f"0x{int(s.colliding_tile):02x}",
        "stair": bool(on_stair_tile(s)),
        "warp": bool(on_warp_tile(s)),
        "blocks": blocks,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def exact(env, assist, total, tx, ty, max_f=240):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen != 0x06:
            return True
        if s.link_x == tx and s.link_y == ty:
            return True
        if s.link_x != tx:
            step(env, assist, total, nes_action("RIGHT" if s.link_x < tx else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if s.link_y < ty else "UP"))
    return False


def cellar_raw(env, assist, total, axis, target, max_f=500):
    last = None
    stall = 0
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x64, 0x06, 0x05):
            return True
        cur = s.link_x if axis == "x" else s.link_y
        if abs(cur - target) <= 1:
            return True
        if axis == "x":
            step(env, assist, total, nes_action("RIGHT" if s.link_x < target else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if s.link_y < target else "UP"))
        s2 = read_snapshot(env.get_ram())
        pos = (s2.link_x, s2.link_y, s2.mode)
        if pos == last:
            stall += 1
            if stall >= 80:
                return False
        else:
            stall = 0
        last = pos
    return False


def warped(env):
    s = read_snapshot(env.get_ram())
    return s.mode in CELLAR or s.screen == 0x07


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
        print("EAST", walk_east_from_05(env, assist, total), flush=True)

        # 0x64 method
        walk_axis(env, assist, total, "x", 48, max_f=200)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        walk_axis(env, assist, total, "x", 64, max_f=300)
        walk_axis(env, assist, total, "y", 160, max_f=400)
        s = read_snapshot(env.get_ram())
        blocks = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
        bx = blocks[0].x if blocks else 96
        by = blocks[0].y if blocks else 144
        walk_axis(env, assist, total, "x", bx, max_f=300)
        walk_axis(env, assist, total, "y", min(by + 16, 170), max_f=300)
        push_dir(env, assist, total, "UP", frames=160)
        idle(env, assist, total, 12)
        rec = {"tag": "pushed", **dump(env)}
        log.append(rec)
        print("PUSHED", rec, flush=True)

        # Exact stands. CheckWarp is tol=0.
        for tx, ty in ((120, 141), (128, 141), (112, 141), (120, 133), (120, 125), (96, 133)):
            if warped(env):
                break
            # back off then walk onto the pixel
            walk_axis(env, assist, total, "y", ty, max_f=200)
            walk_axis(env, assist, total, "x", max(tx - 8, 80), max_f=200)
            exact(env, assist, total, tx, ty)
            rec = {"tag": f"exact_{tx}_{ty}", **dump(env)}
            log.append(rec)
            print("EXACT", rec, flush=True)
            for i in range(90):
                if warped(env):
                    rec = {"tag": "warped", "f": i, "tgt": [tx, ty], **dump(env)}
                    log.append(rec)
                    print("WARPED", rec, flush=True)
                    break
                step(env, assist, total, nes_idle_action())
            if warped(env):
                break
            # one-pixel nudges
            for d in ("RIGHT", "LEFT", "UP", "DOWN"):
                if warped(env):
                    break
                step(env, assist, total, nes_action(d))
                idle(env, assist, total, 8)
                rec = {"tag": f"nudge_{d}_from_{tx}_{ty}", **dump(env)}
                log.append(rec)
                print("NUDGE", rec, flush=True)
                if warped(env):
                    break
                # undo
                opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[d]
                step(env, assist, total, nes_action(opp))

        after = dump(env)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_exact_stand.png")
        print("AFTER_STAND", after, flush=True)

        dest64 = None
        if warped(env):
            print("CELLAR", dump(env), flush=True)
            cellar_raw(env, assist, total, "y", 189, max_f=500)
            cellar_raw(env, assist, total, "x", 48, max_f=600)
            print("LEFT_MOUTH", dump(env), flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_exact_cellar.png")
            push_dir(env, assist, total, "UP", frames=280)
            idle(env, assist, total, 16)
            for _ in range(300):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen == 0x64:
                    break
                if s.mode in CELLAR:
                    step(env, assist, total, nes_action("UP"))
                else:
                    step(env, assist, total, nes_idle_action())
            idle(env, assist, total, 16)
            dest64 = dump(env)
            print("DEST64", dest64, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_exact_64.png")
            if dest64["sc"] == "0x64" and dest64["mode"] == 5 and dest64["whistle"] == 1:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": STATE,
                        "via": "0x06 push 0x68 NORTH, exact (120,141) CheckWarp, cellar left 0x64",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1, "xy": dest64["xy"]},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", flush=True)

        body = {
            "ok": bool(dest64 and dest64["sc"] == "0x64"),
            "warp_06": warped(env) or (dest64 is not None),
            "reached_64": bool(dest64 and dest64["sc"] == "0x64"),
            "pokes": False,
            "status_claim": None,
            "log": log,
            "after": after,
            "dest64": dest64,
            "final": dump(env),
        }
        write_json_report(RECORDINGS_DIR / "l5_06_exact_stand.json", body)
        print("FINAL", body["final"], "WARP", body["warp_06"], "R64", body["reached_64"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
