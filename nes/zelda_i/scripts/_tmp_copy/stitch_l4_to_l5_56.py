"""L4Complete → EastKey → 0x56. One session. Extends EastKey tape. Not STATUS."""
from __future__ import annotations
from typing import Any
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level5_dungeon import (
    ROOM_66_SPEC, ROOM_77_SPEC, ROOM_L5_POLS_77,
    level5_room_66_cleared, level5_room_77_key_success, level5_room_56_arrived,
    make_pols_voice_controller,
)
from zelda_i.level5_overworld import POST_L4_TO_LEVEL5_HOPS, OverworldToLevel5Controller, level5_entrance_success
from zelda_i.level5_path import level5_east_key_step, make_west65_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, ADDR_RAFT, PLAY_MODE, read_snapshot, read_u8

POST_L4_RETURN, SETTLE_MAX, PATH_MAX, ENTER77_MAX = 0x45, 1800, 40000, 2500
TAG = "l4_complete_to_l5_56_stitch"

def pin(env):
    s = read_snapshot(env.get_ram())
    return {"mode": s.mode, "level": s.level, "screen": s.screen, "screen_hex": f"0x{s.screen:02x}",
            "x": s.link_x, "y": s.link_y, "keys": s.keys, "triforce": s.triforce,
            "raft": int(read_u8(env.get_ram(), ADDR_RAFT)), "ladder": int(read_u8(env.get_ram(), ADDR_LADDER))}

def step(env, action, assist, n):
    obs, *_ = env.step(action)
    n[0] += 1
    if assist: assist.apply_env(env, frame=n[0])
    return obs

def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_l4_to_56"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    env = make_env(GAME, "Level4Complete", GAME_DIR, render_mode="rgb_array", record=str(movie))
    seams = []
    try:
        obs, _ = reset_obs(env)
        obs = step(env, nes_idle_action(), assist, n)
        for _ in range(SETTLE_MAX):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.level == 0 and s.screen == POST_L4_RETURN and not s.transitioning:
                break
            obs = step(env, nes_idle_action(), assist, n)
        ow = OverworldToLevel5Controller(hops=POST_L4_TO_LEVEL5_HOPS, max_frames=PATH_MAX)
        while not ow.success:
            s = read_snapshot(env.get_ram())
            if s.mode == 17 or ow.phase.name == "FAILED": break
            obs = step(env, ow.step(s).action, assist, n)
        ok76 = level5_entrance_success(env.get_ram()) and ow.success
        seams.append({"name": "Level5EntranceFromL4", "ok": ok76, **pin(env)})
        print("76", ok76, pin(env), flush=True)
        if not ok76: raise SystemExit("fail_76")
        c66 = GenericDungeonRoomController(ROOM_66_SPEC)
        for _ in range(ROOM_66_SPEC.max_frames):
            obs = step(env, c66.step(read_snapshot(env.get_ram())).action, assist, n)
            if c66.success or c66.phase is DungeonPhase.FAILED: break
        ok66 = level5_room_66_cleared(env.get_ram())
        seams.append({"name": "Level5Cleared66", "ok": ok66, **pin(env)})
        print("66", ok66, pin(env), flush=True)
        if not ok66: raise SystemExit("fail_66")
        for _ in range(ENTER77_MAX):
            s = read_snapshot(env.get_ram())
            if s.level == 5 and s.screen == ROOM_L5_POLS_77 and s.mode == PLAY_MODE:
                for _ in range(40): obs = step(env, nes_idle_action(), assist, n)
                break
            obs = step(env, level5_east_key_step(s).action, assist, n)
        pols = make_pols_voice_controller()
        for _ in range(ROOM_77_SPEC.max_frames):
            obs = step(env, pols.step(read_snapshot(env.get_ram())).action, assist, n)
            if pols.success or pols.phase is DungeonPhase.FAILED: break
        ok77 = level5_room_77_key_success(env.get_ram())
        seams.append({"name": "Level5EastKey", "ok": ok77, **pin(env)})
        print("77", ok77, pin(env), flush=True)
        if not ok77: raise SystemExit("fail_77")
        nav = make_west65_controller()
        for _ in range(nav.max_frames):
            obs = step(env, nav.step(read_snapshot(env.get_ram())).action, assist, n)
            if nav.success or nav.failed: break
        ok56 = level5_room_56_arrived(env.get_ram()) and nav.success
        final = pin(env)
        seams.append({"name": "Level5North56", "ok": ok56, **final})
        print("56", ok56, final, flush=True)
        if not ok56: raise SystemExit("fail_56")
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
              "status_claim": False, "start_state": "Level4Complete", "end_claim": "Level5North56",
              "total_frames": n[0], "seams": seams, "final": final,
              "bk2": str(bk2) if bk2 else None, "screenshot": str(shot)}
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(f"ok frames={n[0]} wrote {path} bk2={bk2}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
