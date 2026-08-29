"""Clear a World 1 SMB3 stage from natural entry.

1-1: title → World 1 map → 1-1 node → scripted clear.
1-2: AfterLevel1 map → two RIGHT hops → 1-2 node → scripted clear.

Usage::

    uv run python nes/smb3/scripts/run_level1.py
    uv run python nes/smb3/scripts/run_level1.py --from-state Level1_1
    uv run python nes/smb3/scripts/run_level1.py --level 1-2
    uv run python nes/smb3/scripts/run_level1.py --level 1-2 --from-state Level1_2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smb3.menus import boot_to_level1_script
from smb3.paths import GAME, GAME_DIR, RECORDINGS_DIR
from smb3.policy import STAGES, Level1Policy, StageSpec
from smb3.ram import (
    ADDR_MAP_MOVE,
    is_goal_auto,
    is_level1_ready,
    is_in_level,
    is_map_controllable,
    parse_game_state,
    player_progress_x,
)
from retro_harness.env import make_env, read_state_bytes, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png

BOOT_STABLE_FRAMES = 40
BOOT_MIN_FRAME = 400
LEVEL_LOAD_MAX = 300
POST_GOAL_MAX = 800
MAP_IDLE_MAX = 250


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


def _wait_map_controllable(env, frame: int) -> tuple[object, int]:
    """Idle until Map_Operation is normal move/enter."""
    obs = None
    for _ in range(MAP_IDLE_MAX):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        ram = env.get_ram()
        if is_map_controllable(ram) and int(ram[ADDR_MAP_MOVE]) == 0:
            return obs, frame
    raise RuntimeError("map never reached Map_Operation $0D")


def _enter_and_wait_level(env, frame: int, stage: StageSpec) -> tuple[object, int]:
    """Play the stage enter script and wait until in-level."""
    obs = None
    if stage.enter is not None:
        for scripted in stage.enter():
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
    raise RuntimeError(f"failed to enter World {stage.id} from map")


def _load_state(env, name: str) -> object:
    """Load a custom integration state through em.set_state (TAS fidelity)."""
    raw = read_state_bytes(state_path(GAME_DIR, GAME, name))
    env.em.set_state(raw)
    obs, *_ = env.step(nes_idle_action())
    env.em.set_state(raw)
    return obs


def run_level1(
    *,
    from_state: str | None = None,
    policy_path: Path | str | None = None,
    save_after: bool = True,
    level: str = "1-1",
) -> int:
    """Clear a World 1 stage; return 0 on success."""
    if level not in STAGES:
        raise ValueError(f"unknown level {level!r}; expected {sorted(STAGES)}")
    stage = STAGES[level]
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = stage.recordings_prefix
    if from_state:
        load_name = from_state
    elif level == "1-1":
        load_name = "NONE"
    else:
        load_name = "AfterLevel1"
    env = make_env(GAME, load_name, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        frame = 0
        in_level_direct = from_state == stage.start_state

        if level == "1-1" and from_state is None:
            obs, frame = _boot_to_map(env)
            save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_map.png")
            print(f"MAP frame={frame} mean={float(obs.mean()):.1f}")
            obs, frame = _enter_and_wait_level(env, frame, stage)
            save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_start.png")
        elif in_level_direct:
            obs = _load_state(env, from_state)
            frame = 0
        else:
            map_state = from_state or "AfterLevel1"
            obs = _load_state(env, map_state)
            obs, frame = _wait_map_controllable(env, 0)
            save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_map.png")
            print(f"MAP frame={frame} mean={float(obs.mean()):.1f}")
            obs, frame = _enter_and_wait_level(env, frame, stage)
            save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_start.png")
            if save_after:
                path = save_state(env, GAME_DIR, GAME, stage.start_state)
                print(f"saved {path}")

        ram = env.get_ram()
        lives0 = int(ram[0x0736])
        print(
            f"LEVEL_START frame={frame} progress={player_progress_x(ram)} "
            f"lives={lives0} mean={float(obs.mean()):.1f}"
        )

        policy = Level1Policy.from_file(policy_path or stage.policy_file)
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
                png = save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_death.png")
                print(f"DEATH frame={frame} max_prog={max_prog:.0f} screenshot={png}")
                return 1
            if is_goal_auto(ram) and max_prog >= stage.completion_min_progress:
                goal_frame = frame
                save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_goal.png")
                print(f"GOAL frame={frame} max_prog={max_prog:.0f}")
                break
        else:
            png = save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_fail.png")
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
        png = save_rgb_png(obs, RECORDINGS_DIR / f"{prefix}_clear.png")
        print(
            f"CLEAR goal_frame={goal_frame} end_frame={frame} max_prog={max_prog:.0f} "
            f"lives={state.lives} map_return={map_return} mean={float(obs.mean()):.1f} "
            f"screenshot={png}"
        )
        if save_after and map_return:
            path = save_state(env, GAME_DIR, GAME, stage.after_state)
            print(f"saved {path}")
        return 0 if map_return else 1
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=sorted(STAGES),
        default="1-1",
        help="World 1 stage to clear (default: 1-1)",
    )
    parser.add_argument(
        "--from-state",
        default=None,
        help="Skip boot/map and load this integration state (e.g. Level1_1, AfterLevel1)",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Policy JSON path")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_level1(
            from_state=args.from_state,
            policy_path=args.policy,
            save_after=not args.no_save,
            level=args.level,
        )
    )


if __name__ == "__main__":
    main()
