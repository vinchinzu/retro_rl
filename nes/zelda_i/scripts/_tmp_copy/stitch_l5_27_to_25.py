"""Cleared37 → 27 → 26 → 25. One session from named pins. Not STATUS."""
from __future__ import annotations
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level5_dungeon import (
    ROOM_25_SPEC, ROOM_26_SPEC, ROOM_27_SPEC,
    level5_room_25_cleared, level5_room_26_cleared, level5_room_27_cleared,
    Level5PolsVoiceController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

TAG = "l5_cleared27_to_25_stitch"

def pin(env):
    s = read_snapshot(env.get_ram())
    return {"mode": s.mode, "level": s.level, "screen": s.screen, "screen_hex": f"0x{s.screen:02x}",
            "x": s.link_x, "y": s.link_y, "keys": s.keys}

def run_spec(env, spec, assist, n, controller=None):
    ctl = controller or GenericDungeonRoomController(spec)
    obs = None
    for _ in range(spec.max_frames):
        if assist: assist.apply_env(env, frame=n[0])
        obs, *_ = env.step(ctl.step(read_snapshot(env.get_ram())).action)
        n[0] += 1
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    return obs, ctl

def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_37_to_25"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    env = make_env(GAME, "Level5Cleared27", GAME_DIR, render_mode="rgb_array", record=str(movie))
    seams = []
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action()); n[0] += 1
        assist.apply_env(env, frame=0)
        print("start", pin(env), flush=True)
        obs, c26 = run_spec(env, ROOM_26_SPEC, assist, n)
        ok26 = level5_room_26_cleared(env.get_ram())
        seams.append({"name": "Level5Cleared26", "ok": ok26, "ctl": c26.report(), **pin(env)})
        print("26", ok26, pin(env), flush=True)
        if not ok26: raise SystemExit("fail_26")
        obs, c25 = run_spec(env, ROOM_25_SPEC, assist, n, controller=Level5PolsVoiceController(ROOM_25_SPEC))
        ok25 = level5_room_25_cleared(env.get_ram())
        final = pin(env)
        seams.append({"name": "Level5Cleared25", "ok": ok25, "ctl": c25.report(), **final})
        print("25", ok25, final, flush=True)
        if not ok25: raise SystemExit("fail_25")
        shot = out / f"{TAG}_final.png"
        save_rgb_png(obs, shot)
        if hasattr(env, "stop_record"): env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"): env.stop_record()
        except Exception:
            pass
        env.close()
    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    bk2 = bk2s[-1] if bk2s else None
    report = {"ok": True, "segment": TAG, "continuous_emulator_session": True, "track": "assisted",
              "status_claim": False, "start_state": "Level5Cleared27", "end_claim": "Level5Cleared25",
              "total_frames": n[0], "seams": seams, "final": final, "bk2": str(bk2) if bk2 else None}
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(f"ok frames={n[0]} wrote {path} bk2={bk2}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
