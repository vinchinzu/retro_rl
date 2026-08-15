"""Level5Whistle cellar 0x04 → play-mode L5 dungeon (typical 0x05 Blue Darknut stairs).

One fceumm session. Survival assist only. No key/door/item pokes.
Does not start Digdogger from inside the cellar. Does not claim Level5Complete.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level5_path import leave_whistle_cellar
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

TAG = "l5_whistle_cellar_exit_stitch"
ROOM_NAMES = {
    0x04: "Whistle basement / Recorder cellar",
    0x05: "Blue Darknut stairs",
    0x06: "Whistle passage",
    0x07: "Whistle basement passage",
    0x15: "Blue Darknut stairs (alt)",
    0x24: "Digdogger",
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
        "cellar": s.mode != PLAY_MODE or s.screen == 0x04,
    }


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle_cellar_exit"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    start = None
    hop = None
    after = None
    obs = None
    png = None
    left = False
    dest_ok = False
    ckpt = None

    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        start = pin(env)
        print("START", start, flush=True)

        hop = leave_whistle_cellar(env, assist, n)
        idle(env, assist, n, 12)
        after = pin(env)
        print("LEAVE", {k: hop[k] for k in hop if k != "log"}, after, flush=True)

        left = bool(hop.get("left_cellar")) or (
            after["mode"] == PLAY_MODE
            and after["level"] == LEVEL_5
            and after["screen"] != 0x04
        )
        dest_ok = left and after["whistle"] == 1 and not after["tf_l5_bit"]
        ckpt = None
        if dest_ok:
            name = "Level5WhistleFloor"
            path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                path,
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle.state",
                request={
                    "segment": name,
                    "predecessor_entry": True,
                    "start_state": "Level5Whistle",
                    "via": "0x04 short ladder x=176 DOWN, pit y=189, left mouth x=48 UP",
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
            ckpt = name
            print("PINNED", name, after, flush=True)
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
    report = {
        "ok": bool(dest_ok),
        "segment": TAG,
        "walker": "leave_whistle_cellar",
        "path": "alcove_ladder176_pit189_left48",
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "pokes": False,
        "start_state": "Level5Whistle",
        "start": start,
        "hop": {k: hop[k] for k in hop} if hop else None,
        "after": after,
        "exit_landed": bool(left),
        "dest_screen": None if after is None else after["screen_hex"],
        "dest_room_name": None if after is None else after["room_name"],
        "whistle_still_1": None if after is None else after["whistle"] == 1,
        "tf_l5_bit": None if after is None else after["tf_l5_bit"],
        "did_not_fight_digdogger": True,
        "level5_complete_claimed": False,
        "total_frames": n[0],
        "png": str(png) if png else None,
        "bk2": bk2,
        "checkpoint": ckpt if dest_ok else None,
        "blocker": None if dest_ok else ("still_in_cellar_0x04" if not left else "whistle_or_tf_unexpected"),
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} left={left} dest={after} ok={dest_ok}",
        flush=True,
    )
    return 0 if dest_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
