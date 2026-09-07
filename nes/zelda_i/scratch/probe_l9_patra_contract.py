"""Run the join from L9PostArrowsReal to Patra room 0x52, pin a savestate the
frame Link lands there, then report each `level9_live_patra_stop` term every
few frames for 600 frames -- long enough to tell a slow eye spawn apart from a
genuinely unmet term.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_patra_contract.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import (
    FULL_TRIFORCE, LEVEL9, MAGICAL_SWORD, ROOM_FINAL_PATRA, SILVER_ARROWS,
    level9_live_patra_stop,
)
from zelda_i.level9.patra import PATRA_EYE_COUNT, final_patra_live, patra_eyes
from zelda_i.level9.ganon import NORTH_DOOR
from zelda_i.level9.natural_path import make_natural_patra_join_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9PatraArrivalReal"
PLAY_MODE = 5


def terms(snap) -> dict[str, object]:
    return {
        "level9": snap.level == LEVEL9,
        "play_mode": snap.mode == PLAY_MODE,
        "not_transitioning": not snap.transitioning,
        "triforce": snap.triforce == FULL_TRIFORCE,
        "bow": snap.bow,
        "arrows": snap.arrows,
        "arrows_ok": snap.arrows == SILVER_ARROWS,
        "screen_ok": snap.screen == ROOM_FINAL_PATRA,
        "sword": snap.sword,
        "sword_ok": snap.sword >= MAGICAL_SWORD,
        "patra_live": final_patra_live(snap),
        "eyes": len(patra_eyes(snap)),
        "eyes_ok": len(patra_eyes(snap)) == PATRA_EYE_COUNT,
        "north_shut": not (snap.cur_opened_doors & NORTH_DOOR),
    }


def main() -> int:
    configure_headless()
    env = make_env(GAME, "L9PostArrowsReal", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_natural_patra_join_controller()
    frame = 0
    while frame < 30000:
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_FINAL_PATRA and snap.mode == PLAY_MODE and not snap.transitioning:
            break
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"arrived f{frame} room=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})", flush=True)
    save_state(env, GAME_DIR, GAME, STATE_NAME)
    print(f"pinned {STATE_NAME}", flush=True)
    for i in range(601):
        snap = read_snapshot(env.get_ram())
        if i % 40 == 0 or level9_live_patra_stop(snap):
            print(f"  +{i}: stop={level9_live_patra_stop(snap)} {terms(snap)}", flush=True)
            if level9_live_patra_stop(snap):
                break
        obs, *_ = env.step(ctl.step(snap).action)
        assist.apply_env(env, frame=frame + i)
    snap = read_snapshot(env.get_ram())
    print("objects:", flush=True)
    for o in snap.objects:
        if o.type_id:
            print(f"  slot={o.slot} type=0x{o.type_id:02x} xy=({o.x},{o.y}) hp={o.hp} st={o.state}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
