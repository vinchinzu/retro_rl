"""Level5Whistle24 Digdogger → whistle-shrink → north 0x14 TF bit 0x10.

Suffix tape after the 65→37 session that landed 0x57 key-north.
One env, assisted, record= BK2, stop_record. No Cleared25. No STATUS.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("w65tf", HERE.parent / "_probe_l5_w65_east_to_tf.py")
w65 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w65)

TAG = "l5_w24_tf"
START = "Level5Whistle24"
TF_BIT = 0x10


def pin(env) -> dict:
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce": tf,
        "tf_hex": hex(tf),
        "tf_l5_bit": bool(tf & TF_BIT),
    }


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_w24_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        start = pin(env)
        print("START24", start, flush=True)
        if start["whistle"] != 1 or start["screen"] != 0x24:
            blocker = f"not_24_whistle1_got_0x{start['screen']:02x}"
            boss = None
        else:
            boss = w65.digdogger(env, assist, n)
            print("BOSS", {k: boss[k] for k in boss if k != "log"}, flush=True)
            blocker = None if boss.get("tf_l5") else "tf_bit_0x10_not_set"
        final = pin(env)
        save_rgb_png(env.step(nes_idle_action())[0], out / f"{TAG}_final.png")
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
    tf_l5 = bool(final.get("tf_l5_bit"))
    report = {
        "ok": tf_l5 and blocker is None,
        "segment": TAG,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "level5_complete_claim": False,
        "start_state": START,
        "whistle_0x065C": final.get("whistle"),
        "triforce_0x0671": final.get("triforce"),
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": None if boss is None else boss.get("killed"),
        "boss": None if boss is None else {k: boss[k] for k in boss if k != "log"},
        "total_frames": n[0],
        "start": start,
        "final": final,
        "blocker": blocker,
        "bk2": str(bk2s[-1]) if bk2s else None,
        "png": str(out / f"{TAG}_final.png"),
        "pokes": False,
        "path_note": "Suffix: Level5Whistle24 Digdogger whistle-shrink → north 0x14 TF. Stitched after 65→37 key-north tape.",
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(f"wrote {path} frames={n[0]} tf={final.get('triforce')} tf_l5={tf_l5} blocker={blocker}", flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
