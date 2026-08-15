"""0x06 push 0x68 NORTH, exact (128,141) CheckWarp, cellar LEFT mouth → 0x64."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis, walk_east_from_05
from zelda_i.level9_stairs import on_stair_tile
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
        "blocks": blocks,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
    }


def exact(env, assist, total, tx, ty, max_f=300):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen == 0x07:
            return True
        if s.link_x == tx and s.link_y == ty:
            return True
        if s.link_x != tx:
            step(env, assist, total, nes_action("RIGHT" if s.link_x < tx else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if s.link_y < ty else "UP"))
    return False


def raw_axis(env, assist, total, axis, target, max_f=700):
    """Keep stepping in mode 9. Do not use walk_axis."""
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
        pos = (s2.link_x, s2.link_y)
        if pos == last:
            stall += 1
            if stall >= 200:
                return False
        else:
            stall = 0
        last = pos
    return False


def wait_mouth(env, assist, total, max_f=240):
    """Idle through mode 16 until mode 9 on a mouth column."""
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == 9 and (s.link_x <= 64 or s.link_x >= 176):
            return dump(env)
        if s.mode == PLAY_MODE and s.screen in (0x64, 0x06):
            return dump(env)
        step(env, assist, total, nes_idle_action())
    return dump(env)


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
        east = walk_east_from_05(env, assist, total)
        print("EAST", east, flush=True)

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

        # Exact CheckWarp stand is (128,141), not (120,141)
        walk_axis(env, assist, total, "y", 141, max_f=200)
        walk_axis(env, assist, total, "x", 112, max_f=200)
        exact(env, assist, total, 128, 141)
        rec = {"tag": "stand_128_141", **dump(env)}
        log.append(rec)
        print("STAND", rec, flush=True)
        for i in range(120):
            s = read_snapshot(env.get_ram())
            if s.mode in CELLAR or s.screen == 0x07:
                rec = {"tag": "warped", "f": i, **dump(env)}
                log.append(rec)
                print("WARPED", rec, flush=True)
                break
            step(env, assist, total, nes_idle_action())
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_to64_warp.png")

        s = read_snapshot(env.get_ram())
        warp_ok = s.mode in CELLAR or s.screen == 0x07
        dest64 = None
        if warp_ok:
            mouth = wait_mouth(env, assist, total)
            log.append({"tag": "mouth_settle", **mouth})
            print("MOUTH", mouth, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_to64_mouth.png")

            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x64:
                dest64 = dump(env)
            else:
                # From 0x06 we land on the RIGHT mouth (192). Floor-cross to LEFT (48).
                if s.link_x >= 128:
                    print("CROSS from right", dump(env), flush=True)
                    raw_axis(env, assist, total, "y", 189, max_f=700)
                    log.append({"tag": "floor", **dump(env)})
                    print("FLOOR", log[-1], flush=True)
                    raw_axis(env, assist, total, "x", 48, max_f=800)
                    log.append({"tag": "left_col", **dump(env)})
                    print("LEFT_COL", log[-1], flush=True)
                else:
                    print("ALREADY left", dump(env), flush=True)
                save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_to64_left.png")
                # Climb left mouth. Raw UP, do not walk_axis.
                for _ in range(320):
                    s = read_snapshot(env.get_ram())
                    if s.mode == PLAY_MODE and s.screen == 0x64:
                        break
                    if s.mode == PLAY_MODE and s.screen not in (0x07, 0x06):
                        break
                    if abs(s.link_x - 48) > 4 and s.mode in CELLAR:
                        step(env, assist, total, nes_action("LEFT" if s.link_x > 48 else "RIGHT"))
                    else:
                        step(env, assist, total, nes_action("UP"))
                idle(env, assist, total, 20)
                dest64 = dump(env)
            print("DEST64", dest64, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_to64_dest.png")
            if dest64 and dest64["sc"] == "0x64" and dest64["mode"] == 5 and dest64["whistle"] == 1:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": STATE,
                        "via": "0x06 push 0x68 NORTH, exact (128,141), cellar left mouth 0x64",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1, "xy": dest64["xy"]},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dest64, flush=True)

        body = {
            "ok": bool(dest64 and dest64["sc"] == "0x64" and dest64["mode"] == 5),
            "warp_06": warp_ok,
            "reached_64": bool(dest64 and dest64["sc"] == "0x64" and dest64["mode"] == 5),
            "pokes": False,
            "status_claim": None,
            "log": log,
            "dest64": dest64,
            "final": dump(env),
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        }
        write_json_report(RECORDINGS_DIR / "l5_06_to_64.json", body)
        print("FINAL", body["final"], "WARP", body["warp_06"], "R64", body["reached_64"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
