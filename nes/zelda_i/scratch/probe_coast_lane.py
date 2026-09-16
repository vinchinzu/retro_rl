"""Scratch probe: y-sweep every Map-1 coast screen's EAST exit, one boot.

The overlay paints the corridor at y~130; on the south coast that row is the
rocky bowl, and ``shop_p7`` already carries one hand-measured exception
(``SCREEN_79_BEACH_Y``). This measures the rest instead of guessing: on
arrival the emulator state is saved, then for each candidate row Link walks
to it and pushes east. A row that scrolls is a lane; the cheapest one wins,
is replayed for real, and the sweep moves to the next screen.

Do not BFS the OccupancyWalker across these screens: 0x78 is a tree maze and
the 1px learned grid walls its own start cell in (measured, ``c1``).

Health assist is ON — this is geometry, not a survival claim.
Not a production CLI.
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.shop_p7 import SHOP_P7_HOPS
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist

OUT = RECORDINGS_DIR / "scratch_coast_lane"
LEAD_HOPS = 2  # 0x77 -> 0x78 -> 0x79; both cross clean on the live table
ALIGN_CAP = 420
PUSH_CAP = 520
ALIGN_TOL = 3
CANDIDATES = tuple(range(77, 206, 8))


def _run(env, obs, controller, assist, cap: int, frame_base: int):
    frames = 0
    while frames < cap:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, frames, snap
        act = controller.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frame_base + frames)
        if getattr(controller, "success", False):
            return obs, frames, read_snapshot(env.get_ram())
        if getattr(getattr(controller, "phase", None), "name", "") == "FAILED":
            return obs, frames, read_snapshot(env.get_ram())
    return obs, frames, read_snapshot(env.get_ram())


def _try_lane(env, assist, screen: int, want_y: int) -> dict:
    """Walk to row ``want_y``, then hold EAST. Did the screen scroll?"""
    snap = read_snapshot(env.get_ram())
    entry = (int(snap.link_x), int(snap.link_y))
    aligned = False
    align_frames = 0
    for align_frames in range(1, ALIGN_CAP + 1):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE or int(snap.screen) != screen:
            break
        dy = int(snap.link_y) - want_y
        if abs(dy) <= ALIGN_TOL:
            aligned = True
            break
        env.step(nes_action("DOWN" if dy < 0 else "UP"))
        if assist is not None:
            assist.apply_env(env)
    snap = read_snapshot(env.get_ram())
    stood = (int(snap.link_x), int(snap.link_y))
    crossed = ""
    push_frames = 0
    obs = None
    for push_frames in range(1, PUSH_CAP + 1):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            crossed = "death"
            break
        if snap.mode == PLAY_MODE and int(snap.screen) != screen:
            crossed = f"0x{int(snap.screen):02X}"
            break
        obs, *_ = env.step(nes_action("RIGHT"))
        if assist is not None:
            assist.apply_env(env)
    snap = read_snapshot(env.get_ram())
    return {
        "want_y": want_y,
        "entry": entry,
        "aligned": aligned,
        "stood": stood,
        "align_frames": align_frames,
        "push_frames": push_frames,
        "crossed": crossed,
        "final": (int(snap.link_x), int(snap.link_y)),
        "final_screen": f"0x{int(snap.screen):02X}",
        "frames": align_frames + push_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", default="0x7F")
    parser.add_argument("--tag", default="l1")
    args = parser.parse_args()
    stop = int(args.stop, 0)

    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    screens: list[dict] = []
    try:
        obs, _ = reset_obs(env)
        obs, boot = boot_to_ready(env, first_playthrough=True, assist=assist)
        sword = SwordCaveController()
        obs, sword_f, snap = _run(env, obs, sword, assist, SWORD_MAX, boot)
        if not sword.success:
            raise SystemExit(f"sword failed 0x{int(snap.screen):02X}")
        walk = OverworldPathController(
            hops=SHOP_P7_HOPS[:LEAD_HOPS],
            require_sword=True,
            farm_below_hearts=0,
            need_rupees=0,
            evade=True,
            max_frames=6000,
        )
        obs, hop_f, snap = _run(env, obs, walk, assist, 6000, boot + sword_f)
        if not (snap.mode == PLAY_MODE and int(snap.screen) == 0x79):
            raise SystemExit(f"lead hops missed 0x79: 0x{int(snap.screen):02X}")

        screen = 0x79
        while screen <= stop:
            state = env.em.get_state()
            snap = read_snapshot(env.get_ram())
            rows = []
            for want_y in CANDIDATES:
                env.em.set_state(state)
                row = _try_lane(env, assist, screen, want_y)
                rows.append(row)
                print(json.dumps({"screen": f"0x{screen:02X}", **row}))
            good = [r for r in rows if r["crossed"].startswith("0x")]
            rec = {
                "screen": f"0x{screen:02X}",
                "arrival": (int(snap.link_x), int(snap.link_y)),
                "rows": rows,
                "lanes": [(r["want_y"], r["stood"][1], r["crossed"], r["frames"]) for r in good],
            }
            screens.append(rec)
            (OUT / f"{args.tag}.json").write_text(
                json.dumps({"screens": screens}, indent=2) + "\n"
            )
            if not good:
                rec["halt"] = "no_lane"
                break
            best = min(good, key=lambda r: r["frames"])
            rec["best"] = best["want_y"]
            env.em.set_state(state)
            replay = _try_lane(env, assist, screen, best["want_y"])
            save_rgb_png(env.render(), OUT / f"{args.tag}_{screen:02X}.png")
            if not replay["crossed"].startswith("0x"):
                rec["halt"] = f"replay_failed_{replay['crossed'] or 'stall'}"
                break
            screen = int(replay["crossed"], 16)
        (OUT / f"{args.tag}.json").write_text(
            json.dumps({"screens": screens}, indent=2) + "\n"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
