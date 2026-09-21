"""Dest diagnostic: Map-1 into 0x79, LEFT out, skirt to 0x7A. Hunt off."""

from __future__ import annotations

import json

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.shop_p7 import PRE_L1_BOMB_HOPS
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist

OUT = RECORDINGS_DIR / "scratch_79_back"
JOIN = 0x7B
CAP = 16000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    obs, _ = reset_obs(env)
    obs, boot = boot_to_ready(env, first_playthrough=True, assist=assist)
    sword = SwordCaveController()
    frames = 0
    trail = []
    last = None
    ctl = None
    while frames < CAP:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            halt = "death"
            break
        if sword.success and ctl is None:
            ctl = OverworldPathController(
                hops=PRE_L1_BOMB_HOPS,
                require_sword=True,
                farm_below_hearts=0,
                need_rupees=0,
                evade=True,
                hunt_destination=False,
                max_frames=CAP,
            )
        driver = ctl if ctl is not None else sword
        act = driver.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frames)
        snap = read_snapshot(env.get_ram())
        key = (int(snap.screen), int(snap.mode))
        if key != last:
            last = key
            path = OUT / f"s{snap.screen:02X}_m{snap.mode:02d}_f{frames}.png"
            save_rgb_png(obs, path)
            rec = {
                "f": frames,
                "screen": int(snap.screen),
                "mode": int(snap.mode),
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "hop": None if ctl is None else ctl.hop_index,
                "reason": act.reason,
            }
            trail.append(rec)
            print(rec, flush=True)
        if (
            ctl is not None
            and snap.mode == PLAY_MODE
            and snap.screen == JOIN
            and not snap.transitioning
        ):
            halt = "hit_7b"
            break
        if getattr(driver, "success", False) and ctl is not None:
            halt = "path_complete"
            break
        phase = getattr(driver, "phase", None)
        if getattr(phase, "name", "") == "FAILED":
            halt = "failed"
            break
    else:
        halt = "timeout"
        snap = read_snapshot(env.get_ram())
    save_rgb_png(obs, OUT / "final.png")
    leftover = {
        "halt": halt,
        "boot": boot,
        "frames": frames,
        "screen": int(snap.screen),
        "mode": int(snap.mode),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "hop": None if ctl is None else ctl.hop_index,
        "notes": [] if ctl is None else list(ctl.notes),
        "trail": trail,
    }
    (OUT / "result.json").write_text(json.dumps(leftover, indent=2) + "\n")
    print(json.dumps({k: leftover[k] for k in ("halt", "frames", "screen", "x", "y", "hop")}, indent=2))


if __name__ == "__main__":
    main()
