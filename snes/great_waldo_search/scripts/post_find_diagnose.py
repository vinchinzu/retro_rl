"""Diagnose why post-+1000 A-clicks do not score.

Checks: long settle, button variants, camera pan, mid-click score
samples, and Waldo-first (skip scroll) candidate clicks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.targets import (
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    FOUND_FLAG_ADDR,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.env import make_env
from retro_harness.actions import buttons_multi, idle_action_multi
from retro_harness.cursor import CursorPose, CursorTarget, step_toward_target
from retro_harness.ram_state import diff_changed, snapshot


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _idle(env: object, frames: int) -> np.ndarray:
    if frames <= 0:
        obs, *_rest = env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
        return np.asarray(obs)
    obs = None
    for _ in range(frames):
        obs, *_rest = env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
    assert obs is not None
    return np.asarray(obs)


def _hold_p2a(env: object, frames: int) -> CursorPose:
    for _ in range(frames):
        env.step(buttons_multi(p2=("A",)))  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _drive(env: object, target: CursorTarget, frames: int = 500) -> CursorPose:
    for _ in range(frames):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
        action = step_toward_target(pose, target, fast_button="Y")
        if action.reason == "confirm_at_target":
            return pose
        multi = idle_action_multi(players=2)
        multi[:12] = list(action.action)
        env.step(multi)  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _metrics(env: object) -> dict:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return {
        "score": score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        "found": int(ram[FOUND_FLAG_ADDR]),
        "x": int(ram[CURSOR_X_ADDR]),
        "y": int(ram[CURSOR_Y_ADDR]),
        "ram_c3": int(ram[0x00C3]),
        "ram_1bc": int(ram[0x01BC]),
        "ram_1bd": int(ram[0x01BD]),
        "ram_1be": int(ram[0x01BE]),
        "ram_1bf": int(ram[0x01BF]),
    }


def _trace_click(env: object, *, hold: int, settle: int) -> dict:
    before = snapshot(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    timeline: list[dict] = []
    for i in range(hold):
        env.step(buttons_multi(p1=("A",)))  # type: ignore[attr-defined]
        if i in (0, 1, 2, 5, hold - 1):
            m = _metrics(env)
            m["t"] = f"a{i}"
            timeline.append(m)
    for i in range(settle):
        env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
        if i in (0, 10, 20, 40, 80, 120, settle - 1) or i % 30 == 0:
            m = _metrics(env)
            m["t"] = f"i{i}"
            timeline.append(m)
    after = snapshot(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    deltas = [
        {"address": d.address, "before": d.before, "after": d.after}
        for d in diff_changed(before, after, limit=None)
        if d.address < 0x400
    ]
    return {"timeline": timeline, "low_deltas": deltas[:60]}


def run_diagnose() -> dict:
    """Run post-find diagnostics and Waldo-first probes."""
    _configure_headless()
    out = RECORDINGS_DIR / "post_find_diagnose"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"cases": []}

    # Case A: AfterFind + long wait + click landing
    env = make_env(
        game=GAME,
        state="Scene1_AfterFind1000",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        env.reset()
        Image.fromarray(_idle(env, 4)).save(out / "A0_start.png")
        land = _hold_p2a(env, 300)
        _idle(env, 4)
        report["second_landing"] = {"x": land.x, "y": land.y}
        for wait in (0, 60, 180, 360):
            env.close()
            env = make_env(
                game=GAME,
                state="Scene1_AfterFind1000",
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 8)
            _hold_p2a(env, 300)
            _idle(env, wait)
            before = _metrics(env)
            traced = _trace_click(env, hold=8, settle=140)
            after = _metrics(env)
            Image.fromarray(_idle(env, 2)).save(out / f"A_wait{wait}.png")
            row = {
                "case": f"afterfind_wait{wait}",
                "before": before,
                "after": after,
                "trace": traced,
            }
            report["cases"].append(row)
            print(
                f"[diag] wait{wait} score {before['score']}->{after['score']} "
                f"found {before['found']}->{after['found']}"
            )

        # Case B: pan camera RIGHT then P2-A + click
        env.close()
        env = make_env(
            game=GAME,
            state="Scene1_AfterFind1000",
            game_dir=GAME_DIR,
            render_mode="rgb_array",
            players=2,
        )
        env.reset()
        _idle(env, 8)
        for _ in range(400):
            # P1 RIGHT+Y to pan/move cursor toward right edge.
            env.step(buttons_multi(p1=("RIGHT", "Y")))
        Image.fromarray(_idle(env, 2)).save(out / "B_pan.png")
        land = _hold_p2a(env, 350)
        _idle(env, 8)
        before = _metrics(env)
        traced = _trace_click(env, hold=8, settle=140)
        after = _metrics(env)
        Image.fromarray(_idle(env, 2)).save(out / "B_after_click.png")
        report["cases"].append(
            {
                "case": "pan_then_assist",
                "landing": {"x": land.x, "y": land.y},
                "before": before,
                "after": after,
                "trace": traced,
            }
        )
        print(
            f"[diag] pan+assist land=({land.x},{land.y}) "
            f"{before['score']}->{after['score']}"
        )

        # Case C: button B then A (clock/RNG), and START
        for label, seq in (
            ("press_b_then_a", [("B", 8), ("idle", 20), ("A", 8)]),
            ("press_start", [("START", 4), ("idle", 40)]),
            ("press_x", [("X", 8), ("idle", 40)]),
        ):
            env.close()
            env = make_env(
                game=GAME,
                state="Scene1_AfterFind1000",
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 8)
            _hold_p2a(env, 300)
            _idle(env, 4)
            before = _metrics(env)
            for kind, n in seq:
                for _ in range(n):
                    if kind == "idle":
                        env.step(idle_action_multi(players=2))
                    else:
                        env.step(buttons_multi(p1=(kind,)))
            after = _metrics(env)
            Image.fromarray(_idle(env, 60)).save(out / f"C_{label}.png")
            after2 = _metrics(env)
            report["cases"].append(
                {
                    "case": label,
                    "before": before,
                    "after": after,
                    "after_settle": after2,
                }
            )
            print(
                f"[diag] {label} {before['score']}->{after2['score']} "
                f"found {before['found']}->{after2['found']}"
            )
    finally:
        env.close()

    # Case D: Waldo-first — skip scroll, click candidates from Scene1
    waldo_first_targets = [
        (206, 100),
        (198, 94),
        (200, 82),
        (220, 28),
        (20, 115),
        (98, 75),
        (205, 125),
        (214, 50),
        (32, 80),
        (40, 100),
        (180, 100),
        (160, 90),
        (120, 70),
        (240, 100),
    ]
    env = make_env(
        game=GAME,
        state="Scene1",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        for x, y in waldo_first_targets:
            env.close()
            env = make_env(
                game=GAME,
                state="Scene1",
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 10)
            # Do NOT take P2-A scroll; drive manually.
            final = _drive(env, CursorTarget(x=x, y=y, deadzone=2))
            before = _metrics(env)
            traced = _trace_click(env, hold=6, settle=120)
            after = _metrics(env)
            Image.fromarray(_idle(env, 2)).save(
                out / f"D_first_{x}_{y}_s{after['score']}.png"
            )
            row = {
                "case": "waldo_first",
                "target": {"x": x, "y": y},
                "pose": {"x": final.x, "y": final.y},
                "before": before,
                "after": after,
                "delta": after["score"] - before["score"],
                "trace_scores": [t["score"] for t in traced["timeline"]],
            }
            report["cases"].append(row)
            print(
                f"[diag] waldo_first ({x},{y}) "
                f"{before['score']}->{after['score']} "
                f"found {before['found']}->{after['found']}"
            )
    finally:
        env.close()

    # Case E: P2-A but cancel before click — move to other target first
    env = make_env(
        game=GAME,
        state="Scene1",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        env.reset()
        _idle(env, 10)
        land = _hold_p2a(env, 300)
        # Nudge away then to secondary-looking coords without confirming scroll
        for target in ((206, 100), (198, 94), (220, 28), (20, 115)):
            env.close()
            env = make_env(
                game=GAME,
                state="Scene1",
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 10)
            _hold_p2a(env, 300)
            _idle(env, 4)
            _drive(env, CursorTarget(x=target[0], y=target[1], deadzone=2))
            before = _metrics(env)
            traced = _trace_click(env, hold=6, settle=120)
            after = _metrics(env)
            Image.fromarray(_idle(env, 2)).save(
                out / f"E_nudge_{target[0]}_{target[1]}_s{after['score']}.png"
            )
            report["cases"].append(
                {
                    "case": "assist_then_nudge",
                    "from_assist": {"x": land.x, "y": land.y},
                    "target": {"x": target[0], "y": target[1]},
                    "before": before,
                    "after": after,
                    "delta": after["score"] - before["score"],
                }
            )
            print(
                f"[diag] assist_nudge {target} "
                f"{before['score']}->{after['score']} "
                f"found->{after['found']}"
            )
    finally:
        env.close()

    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[diag] wrote {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI for post-find diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    run_diagnose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
