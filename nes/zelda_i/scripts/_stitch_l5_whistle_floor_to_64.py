"""Level5WhistleFloor 0x05 → east 0x06 → block stairs 0x07 → dest 0x64.

One fceumm session. Survival assist only. No key/door/item pokes.
Does not replay 0x04. Does not fight Digdogger. Does not claim Level5Complete.
If 0x07 fails, stop honestly — do not fake 0x64.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level5_path import cellar_07_to_64, take_center_stairs_06, walk_east_from_05
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

TAG = "l5_whistle_floor_to_64_stitch"
START = "Level5WhistleFloor"
ROOM_NAMES = {
    0x04: "Whistle basement / Recorder cellar",
    0x05: "six-Darknut whistle room",
    0x06: "empty whistle passage (diamond)",
    0x07: "whistle cellar passage",
    0x16: "south key drop (not the return)",
    0x24: "Digdogger",
    0x64: "Blue Darknut stairs",
}


def pin(env):
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "room_name": ROOM_NAMES.get(s.screen, f"L5 room 0x{s.screen:02x}"),
        "x": s.link_x,
        "y": s.link_y,
        "keys": s.keys,
        "bombs": s.bombs,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce_0x0671": tf,
        "tf_l5_bit": bool(tf & 0x10),
        "doors": int(s.cur_opened_doors),
        "cellar": s.mode != PLAY_MODE or s.screen in (0x04, 0x07),
    }


def slim(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    return {k: rec[k] for k in rec if k != "log"}


def save_pin(env, name: str, via: str, after: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{START}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": START,
            "via": via,
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        selected_trial={
            "success": True,
            "room": after["screen"],
            "mode": after["mode"],
            "whistle_0x065C": after["whistle"],
            "xy": [after["x"], after["y"]],
        },
        natural_entry=False,
    )
    print("PINNED", name, after, flush=True)
    return name


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle_floor_to_64"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    start = None
    east = None
    stairs = None
    cellar = None
    after_east = None
    after_stairs = None
    after_64 = None
    png = None
    blocker = None
    ckpt07 = None
    ckpt64 = None
    dest_ok = False

    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        start = pin(env)
        print("DUMP_FLOOR", start, flush=True)
        write_json_report(RECORDINGS_DIR / "l5_whistle_floor_dump.json", {"start_state": START, **start})
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_whistle_floor_dump.png")
        n[0] += 1
        print("START", start, flush=True)
        if not (
            start["level"] == LEVEL_5
            and start["screen"] == 0x05
            and start["mode"] == PLAY_MODE
            and start["whistle"] == 1
        ):
            blocker = "start_not_whistle_floor_0x05"
        else:
            east = walk_east_from_05(env, assist, n)
            idle(env, assist, n, 8)
            after_east = pin(env)
            print("EAST", slim(east), after_east, flush=True)
            if not (
                after_east["screen"] == 0x06
                and after_east["mode"] == PLAY_MODE
                and after_east["whistle"] == 1
            ):
                blocker = "east_did_not_enter_06"
            else:
                stairs = take_center_stairs_06(env, assist, n)
                idle(env, assist, n, 8)
                after_stairs = pin(env)
                print("STAIRS", slim(stairs), after_stairs, flush=True)
                in_07 = bool(stairs and stairs.get("success")) and (
                    after_stairs["cellar"] or after_stairs["screen"] == 0x07
                )
                if after_stairs["screen"] == 0x16:
                    blocker = "south_key_drop_0x16_not_return"
                elif not in_07:
                    blocker = "stairs_not_taken_0x07"
                else:
                    ckpt07 = save_pin(
                        env,
                        "Level5Whistle07",
                        "0x05 east 0x06, push 0x68 UP, idle (120,141) tile 0x71 cellar 0x07",
                        after_stairs,
                    )
                    cellar = cellar_07_to_64(env, assist, n)
                    idle(env, assist, n, 12)
                    after_64 = pin(env)
                    print("CELLAR", slim(cellar), after_64, flush=True)
                    for row in (cellar or {}).get("log") or []:
                        print("  ", row, flush=True)
                    dest_ok = (
                        after_64["level"] == LEVEL_5
                        and after_64["screen"] == 0x64
                        and after_64["mode"] == PLAY_MODE
                        and after_64["whistle"] == 1
                        and not after_64["tf_l5_bit"]
                    )
                    if dest_ok:
                        ckpt64 = save_pin(
                            env,
                            "Level5Whistle64",
                            "0x06 push68 idle (120,141) → 0x07 right-drop left-climb → 0x64",
                            after_64,
                        )
                    else:
                        blocker = "cellar_did_not_enter_64"

        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        png = out / f"{TAG}_final.png"
        save_rgb_png(obs, png)

        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    bk2 = str(bk2s[-1]) if bk2s else None
    final = after_64 or after_stairs or after_east or start
    report = {
        "ok": bool(dest_ok),
        "segment": TAG,
        "walkers": ["walk_east_from_05", "take_center_stairs_06", "cellar_07_to_64"],
        "path": "05_east_06_push68_idle_120_141_07_right192_left48_64",
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "pokes": False,
        "start_state": START,
        "start": start,
        "east": slim(east),
        "after_east": after_east,
        "stairs": slim(stairs),
        "after_stairs": after_stairs,
        "cellar": slim(cellar),
        "after_64": after_64,
        "final": final,
        "dest_screen": None if final is None else final["screen_hex"],
        "dest_room_name": None if final is None else final["room_name"],
        "whistle_still_1": None if final is None else final["whistle"] == 1,
        "tf_l5_bit": None if final is None else final["tf_l5_bit"],
        "did_not_replay_0x04": True,
        "did_not_fight_digdogger": True,
        "level5_complete_claimed": False,
        "total_frames": n[0],
        "png": str(png) if png else None,
        "bk2": bk2,
        "checkpoint_07": ckpt07,
        "checkpoint_64": ckpt64 if dest_ok else None,
        "blocker": blocker,
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} dest={final} ok={dest_ok} blocker={blocker}",
        flush=True,
    )
    if dest_ok:
        return 0
    if blocker == "stairs_not_taken_0x07":
        return 2
    if blocker == "cellar_did_not_enter_64":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
