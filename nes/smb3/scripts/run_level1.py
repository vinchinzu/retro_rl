"""Boot Super Mario Bros. 3, enter World 1-1, and clear it.

Natural-entry path: title → World 1 map → 1-1 node → scripted clear.

Usage::

    uv run python smb3/scripts/run_level1.py
    uv run python smb3/scripts/run_level1.py --from-state Level1_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smb3.menus import boot_to_level1_script
from smb3.paths import GAME, GAME_DIR, RECORDINGS_DIR
from smb3.policy import Level1Policy, enter_level1_script
from smb3.ram import (
    is_goal_auto,
    is_level1_ready,
    is_in_level,
    parse_game_state,
    player_progress_x,
)
from retro_harness.env import make_env, read_state_bytes, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png

BOOT_STABLE_FRAMES = 40
BOOT_MIN_FRAME = 400
LEVEL_LOAD_MAX = 300
POST_GOAL_MAX = 800


def _boot_to_map(env) -> tuple[object, int]:
    """Reset already done; run title script to World 1 map control."""
    frame = 0
    stable = 0
    obs = None
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        mean = float(obs.mean())
        if frame >= BOOT_MIN_FRAME and is_level1_ready(env.get_ram(), obs_mean=mean):
            stable += 1
        else:
            stable = 0
        if stable >= BOOT_STABLE_FRAMES:
            break
    if obs is None:
        raise RuntimeError("boot produced no frames")
    for _ in range(90):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
    return obs, frame


def _enter_and_wait_level(env, frame: int) -> tuple[object, int]:
    """Walk to 1-1 on the map, press A, wait until in-level."""
    obs = None
    for scripted in enter_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
    for _ in range(LEVEL_LOAD_MAX):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        ram = env.get_ram()
        if is_in_level(ram) and float(obs.mean()) > 140:
            for _ in range(10):
                obs, *_ = env.step(nes_idle_action())
                frame += 1
            return obs, frame
    raise RuntimeError("failed to enter World 1-1 from map")


def run_level1(
    *,
    from_state: str | None = None,
    policy_path: Path | str | None = None,
    save_after: bool = True,
) -> int:
    """Clear World 1-1; return 0 on success."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    state_name = from_state if from_state else "NONE"
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0

        if from_state is None:
            obs, frame = _boot_to_map(env)
            save_rgb_png(obs, RECORDINGS_DIR / "level1_map.png")
            print(f"MAP frame={frame} mean={float(obs.mean()):.1f}")
            obs, frame = _enter_and_wait_level(env, frame)
            save_rgb_png(obs, RECORDINGS_DIR / "level1_start.png")
        else:
            # fceumm via retro.make(state=...) differs slightly from
            # em.get_state()/set_state bytes (desyncs frame-perfect policies).
            # Reload the on-disk state through em.set_state for TAS fidelity.
            raw = read_state_bytes(state_path(GAME_DIR, GAME, from_state))
            env.em.set_state(raw)
            obs, *_ = env.step(nes_idle_action())
            # The idle above advances one frame — re-load so policy frame 0
            # matches the saved settle point exactly.
            env.em.set_state(raw)
            frame = 0

        ram = env.get_ram()
        lives0 = int(ram[0x0736])
        print(
            f"LEVEL_START frame={frame} progress={player_progress_x(ram)} "
            f"lives={lives0} mean={float(obs.mean()):.1f}"
        )

        policy = Level1Policy.from_file(policy_path)
        max_prog = 0.0
        goal_frame = None
        for _ in range(len(policy) + 50):
            fa = policy.tick()
            obs, *_ = env.step(np.asarray(fa.action, dtype=np.int8))
            frame += 1
            ram = env.get_ram()
            prog = player_progress_x(ram)
            max_prog = max(max_prog, prog)
            if int(ram[0x0736]) < lives0:
                png = save_rgb_png(obs, RECORDINGS_DIR / "level1_death.png")
                print(f"DEATH frame={frame} max_prog={max_prog:.0f} screenshot={png}")
                return 1
            if is_goal_auto(ram) and max_prog >= 1500:
                goal_frame = frame
                save_rgb_png(obs, RECORDINGS_DIR / "level1_goal.png")
                print(f"GOAL frame={frame} max_prog={max_prog:.0f}")
                break
        else:
            png = save_rgb_png(obs, RECORDINGS_DIR / "level1_fail.png")
            print(f"FAIL no_goal frame={frame} max_prog={max_prog:.0f} screenshot={png}")
            return 1

        map_return = False
        for j in range(POST_GOAL_MAX):
            obs, *_ = env.step(nes_idle_action())
            frame += 1
            mean = float(obs.mean())
            ram = env.get_ram()
            if (
                j > 80
                and 90.0 < mean < 150.0
                and int(ram[0x0736]) == lives0
                and int(ram[0x0090]) == 0
            ):
                map_return = True
                break

        state = parse_game_state(env.get_ram(), frame=frame, obs_mean=float(obs.mean()))
        png = save_rgb_png(obs, RECORDINGS_DIR / "level1_clear.png")
        print(
            f"CLEAR goal_frame={goal_frame} end_frame={frame} max_prog={max_prog:.0f} "
            f"lives={state.lives} map_return={map_return} mean={float(obs.mean()):.1f} "
            f"screenshot={png}"
        )
        if save_after and map_return:
            path = save_state(env, GAME_DIR, GAME, "AfterLevel1")
            print(f"saved {path}")
        return 0 if map_return else 1
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-state",
        default=None,
        help="Skip boot/map and load this integration state (e.g. Level1_1)",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Policy JSON path")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_level1(
            from_state=args.from_state,
            policy_path=args.policy,
            save_after=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
