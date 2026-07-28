"""Boot Metroid (NES) from reset and save a controllable Level1 state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metroid.menus import boot_to_level1_script
from metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR
from metroid.ram import is_level1_ready, parse_game_state, read_snapshot
from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action
from snes_oneshot.segment_runner import configure_headless, save_rgb_png

STABLE_FRAMES = 20
MIN_FRAME = 40


def run_probe(*, save_level1: bool = True, walk_frames: int = 40) -> int:
    """Reach Brinstar, verify readiness, walk, and save checkpoint."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        stable = 0
        ready_at = None
        log: list[dict] = []
        for scripted in boot_to_level1_script():
            obs, *_ = env.step(scripted.action)
            frame += 1
            mean = float(obs.mean())
            ram = env.get_ram()
            snap = read_snapshot(ram, env=env)
            if frame % 60 == 0 or (frame >= MIN_FRAME and is_level1_ready(ram, mean)):
                log.append(
                    {
                        "frame": frame,
                        "mean": round(mean, 1),
                        "engine": snap.engine_mode,
                        "game_mode": snap.game_mode,
                        "map": [snap.map_x, snap.map_y],
                        "xy": [snap.samus_x, snap.samus_y],
                        "equipment": snap.equipment,
                        "ready": is_level1_ready(ram, mean),
                    }
                )
            if frame >= MIN_FRAME and is_level1_ready(ram, obs_mean=mean):
                stable += 1
            else:
                stable = 0
            if stable >= STABLE_FRAMES:
                ready_at = frame
                break

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if ready_at is None:
            png = save_rgb_png(obs, RECORDINGS_DIR / "boot_level1.png")
            print(
                f"LEVEL1 frame={frame} ready=False mean={float(obs.mean()):.1f} "
                f"screenshot={png}"
            )
            (RECORDINGS_DIR / "boot_probe_log.json").write_text(
                json.dumps(log, indent=2), encoding="utf-8"
            )
            return 1

        # Motion check (does not persist into Level1 — save first at spawn).
        snap = read_snapshot(env.get_ram(), env=env)
        mean = float(obs.mean())
        ready = is_level1_ready(env.get_ram(), obs_mean=mean)
        state = parse_game_state(env.get_ram(), frame=frame, obs_mean=mean, env=env)
        if save_level1 and ready:
            path = save_state(env, GAME_DIR, GAME, "Level1")
            print(f"saved {path}")

        before = env.get_ram().copy()
        for _ in range(walk_frames):
            obs, *_ = env.step(nes_action("RIGHT"))
            frame += 1
        after = env.get_ram()
        changed = int((before != after).sum())
        snap_walk = read_snapshot(after, env=env)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_level1.png")
        print(
            f"LEVEL1 frame={frame} mode={state.mode.name} ready={ready} "
            f"map=({snap.map_x},{snap.map_y}) xy=({snap.samus_x},{snap.samus_y}) "
            f"equip=0x{snap.equipment:02X} walk_map=({snap_walk.map_x},{snap_walk.map_y}) "
            f"changed={changed} mean={mean:.1f} screenshot={png}"
        )
        (RECORDINGS_DIR / "boot_probe_log.json").write_text(
            json.dumps(
                log + [{"ready_at": ready_at, "final": state.extras}],
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0 if ready else 1
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--walk-frames", type=int, default=40)
    args = parser.parse_args()
    raise SystemExit(
        run_probe(save_level1=not args.no_save, walk_frames=args.walk_frames)
    )


if __name__ == "__main__":
    main()
