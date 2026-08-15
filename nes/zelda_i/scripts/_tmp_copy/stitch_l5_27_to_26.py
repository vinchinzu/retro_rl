"""Cleared27 west key → clear 0x26. Do not continue to 25."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level5_dungeon import ROOM_26_SPEC, level5_in_room_26, level5_room_26_cleared
from zelda_i.level5_path import walk_west_from_27
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

TAG = "l5_cleared27_to_26_stitch"

def pin(env):
    s = read_snapshot(env.get_ram())
    return {"mode": s.mode, "level": s.level, "screen": s.screen,
            "screen_hex": f"0x{s.screen:02x}", "x": s.link_x, "y": s.link_y, "keys": s.keys}

def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_27_to_26"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    env = make_env(GAME, "Level5Cleared27", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action()); n[0] += 1
        assist.apply_env(env, frame=0)
        print("start", pin(env), flush=True)
        hop = walk_west_from_27(env, assist, n)
        print("hop", hop, pin(env), flush=True)
        # mode 7 during scroll is fine — dest 0x26 + key spent is the hop.
        for _ in range(240):
            s = read_snapshot(env.get_ram())
            if s.screen == 0x26 and s.mode == 5 and not s.transitioning:
                break
            obs, *_ = env.step(nes_idle_action()); n[0] += 1
            assist.apply_env(env, frame=n[0])
        if not level5_in_room_26(env.get_ram()):
            print("after settle", pin(env), flush=True)
            raise SystemExit("fail_hop")
        fight = GenericDungeonRoomController(ROOM_26_SPEC)
        for _ in range(ROOM_26_SPEC.max_frames):
            assist.apply_env(env, frame=n[0])
            obs, *_ = env.step(fight.step(read_snapshot(env.get_ram())).action)
            n[0] += 1
            if fight.success or fight.phase is DungeonPhase.FAILED:
                break
        ok = level5_room_26_cleared(env.get_ram())
        final = pin(env)
        print("clear26", ok, fight.report().get("phase"), final, flush=True)
        if not ok:
            raise SystemExit("fail_26")
        shot = out / f"{TAG}_final.png"
        save_rgb_png(obs, shot)
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
    bk2 = bk2s[-1] if bk2s else None
    report = {
        "ok": True, "segment": TAG, "continuous_emulator_session": True,
        "track": "assisted", "status_claim": False,
        "start_state": "Level5Cleared27", "end_claim": "Level5Cleared26",
        "did_not_continue_to_25": True, "total_frames": n[0],
        "hop": hop, "fight": fight.report(), "final": final,
        "bk2": str(bk2) if bk2 else None,
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(f"ok frames={n[0]} wrote {path} bk2={bk2}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
