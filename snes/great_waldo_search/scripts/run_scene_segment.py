"""Run a scene-segment policy: move cursor toward a target and confirm.

Uses Scene1.state when available. Supports optional players=2 parental P2-A
assist (seek objective), then P1 A confirm at the landing / target.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.scene_policy import (
    CursorPose,
    CursorTarget,
    cursor_from_state,
    playing_state,
    step_toward_target,
)
from great_waldo_search.targets import (
    CONFIRMED_FIND_POINTS,
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    FOUND_FLAG_ADDR,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.env import get_available_states, make_env
from retro_harness.actions import buttons_multi, idle_action, idle_action_multi

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")

def run_segment(
    *,
    target_x: int,
    target_y: int,
    frames: int = 200,
    cursor_x_addr: int | None = None,
    cursor_y_addr: int | None = None,
    state: str | None = None,
    use_p2_assist: bool = False,
    p2a_frames: int = 300,
    fast: bool = True,
) -> dict:
    """Step the segment policy; optionally P2-A seek first."""
    _configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state or ("Scene1" if "Scene1" in available else "NONE")
    target = CursorTarget(x=target_x, y=target_y)
    players = 2 if use_p2_assist else 1
    x_addr = cursor_x_addr if cursor_x_addr is not None else CURSOR_X_ADDR
    y_addr = cursor_y_addr if cursor_y_addr is not None else CURSOR_Y_ADDR

    env = make_env(
        game=GAME,
        state=chosen,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=players if use_p2_assist else None,
    )
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    try:
        obs, _info = env.reset()
        if use_p2_assist:
            for _ in range(10):
                env.step(idle_action_multi(players=2))
            for _ in range(p2a_frames):
                env.step(buttons_multi(p2=("A",)))
                reasons.append("p2a_assist")
            for _ in range(4):
                env.step(idle_action_multi(players=2))

        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        pose = CursorPose(x=int(ram[x_addr]), y=int(ram[y_addr]))
        score_before = score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR])
        found_before = int(ram[FOUND_FLAG_ADDR])

        for frame_i in range(frames):
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            pose = CursorPose(x=int(ram[x_addr]), y=int(ram[y_addr]))
            state_obj = playing_state(
                frame=frame_i,
                cursor_x=pose.x,
                cursor_y=pose.y,
            )
            assert cursor_from_state(state_obj) == pose
            action = step_toward_target(
                pose,
                target,
                fast_button="Y" if fast else None,
            )
            reasons.append(action.reason)
            if use_p2_assist:
                multi = idle_action_multi(players=2)
                multi[:12] = list(action.action)
                env.step(multi)
            else:
                env.step(action.action)
            if action.reason == "confirm_at_target":
                confirm = (
                    buttons_multi(p1=("A",))
                    if use_p2_assist
                    else action.action
                )
                for _ in range(6):
                    env.step(confirm)
                idle = (
                    idle_action_multi(players=2)
                    if use_p2_assist
                    else idle_action()
                )
                for _ in range(80):
                    obs, *_rest = env.step(idle)
                break

        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        pose = CursorPose(x=int(ram[x_addr]), y=int(ram[y_addr]))
        score_after = score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR])
        found_after = int(ram[FOUND_FLAG_ADDR])
        out_png = RECORDINGS_DIR / "segment_last.png"
        Image.fromarray(obs).save(out_png)
        summary = {
            "state": chosen,
            "frames_run": len(reasons),
            "final_pose": {"x": pose.x, "y": pose.y},
            "target": {"x": target_x, "y": target_y},
            "last_reasons": reasons[-10:],
            "reached": bool(reasons and reasons[-1] == "confirm_at_target"),
            "score_before": score_before,
            "score_after": score_after,
            "found_before": found_before,
            "found_after": found_after,
            "find_points": score_after - score_before,
            "confirmed_find": (score_after - score_before)
            >= CONFIRMED_FIND_POINTS,
            "png": str(out_png),
        }
        return summary
    finally:
        env.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=int, default=32)
    parser.add_argument("--y", type=int, default=100)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--cursor-x-addr", type=int, default=None)
    parser.add_argument("--cursor-y-addr", type=int, default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--p2-assist",
        action="store_true",
        help="Hold controller-2 A to seek, then drive/confirm",
    )
    parser.add_argument("--p2a-frames", type=int, default=300)
    parser.add_argument("--no-fast", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI for the scene-segment runner."""
    args = _build_parser().parse_args(argv)
    summary = run_segment(
        target_x=args.x,
        target_y=args.y,
        frames=args.frames,
        cursor_x_addr=args.cursor_x_addr,
        cursor_y_addr=args.cursor_y_addr,
        state=args.state,
        use_p2_assist=args.p2_assist,
        p2a_frames=args.p2a_frames,
        fast=not args.no_fast,
    )
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
