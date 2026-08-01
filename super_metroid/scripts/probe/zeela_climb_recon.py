#!/usr/bin/env python3
"""Diagnostic input grid for the lower Zeela reverse climb.

Every trial boots a fresh copy of the continuous-like Zeela source and uses
only ordinary controller input plus the resource-only assist.  The probe is
intentionally diagnostic: it does not place Samus, write progression state, or
claim a route/pure-green result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402

ZEELA_ROOM = 0xA471
DEFAULT_FRAMES = 360
DEFAULT_TARGET_X = (60, 120, 190)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "zeela_climb_recon.json"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pin(state: Any) -> dict[str, object]:
    return {
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "door_transition": state.door_transition,
    }


class ReconSession:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="full")
        self.samples: list[dict[str, object]] = []

    def step(self, action: Any) -> None:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        self.samples.append(
            {
                "frame": self.frame,
                "room": f"0x{self.state.room_id:04X}",
                "pose": self.state.pose,
                "x": self.state.samus_x,
                "y": self.state.samus_y,
                "door_transition": self.state.door_transition,
            }
        )

    def hold(self, frames: int, *names: str) -> None:
        action = buttons(*names) if names else idle_action()
        for _ in range(max(0, frames)):
            self.step(action)


def roll_to_band(session: ReconSession, target_x: int, budget: int) -> None:
    """Use a short natural morph roll to make the candidate x bands comparable."""
    session.hold(min(8, budget - session.frame), "DOWN", "X")
    direction = "LEFT" if session.state.samus_x > target_x else "RIGHT"
    while session.frame < budget and abs(session.state.samus_x - target_x) > 12:
        session.step(buttons(direction, "X"))
    session.hold(min(8, budget - session.frame))


def strategy_actions(name: str, frame: int) -> tuple[str, ...]:
    phase = frame % 24
    if name == "standing_a_spam":
        return ("A",)
    if name == "crouch_load_a":
        return ("DOWN",) if phase < 8 else ("A",)
    if name == "left_wallrun_hi_jump":
        return ("LEFT", "A") if phase < 18 else ("LEFT", "B", "A")
    if name == "right_wallrun_hi_jump":
        return ("RIGHT", "A") if phase < 18 else ("RIGHT", "B", "A")
    if name == "morph_bomb_cycle":
        return ("DOWN", "X") if phase < 5 else (("X",) if phase < 8 else ("A",))
    if name == "morph_left_bomb_cycle":
        return ("LEFT", "DOWN", "X") if phase < 5 else (("LEFT", "X") if phase < 8 else ("LEFT", "A"))
    if name == "forward_drop_reverse_shot":
        return ("UP", "X") if phase < 6 else (("A",) if phase < 12 else ("LEFT", "A"))
    if name == "wall_run_crouch_load":
        return ("LEFT", "DOWN") if phase < 8 else ("LEFT", "A")
    raise KeyError(name)


STRATEGIES: tuple[str, ...] = (
    "standing_a_spam",
    "crouch_load_a",
    "left_wallrun_hi_jump",
    "right_wallrun_hi_jump",
    "morph_bomb_cycle",
    "morph_left_bomb_cycle",
    "forward_drop_reverse_shot",
    "wall_run_crouch_load",
)


def run_trial(source: Path, target_x: int, strategy: str, frames: int) -> dict[str, object]:
    env = make_dev_env()
    try:
        boot_from_state(env, source)
        session = ReconSession(env)
        start = pin(session.state)
        setup_frames = min(48, frames)
        roll_to_band(session, target_x, setup_frames)
        candidate_start = pin(session.state)
        climb_start_frame = session.frame
        while session.frame < frames and session.state.room_id == ZEELA_ROOM:
            session.step(buttons(*strategy_actions(strategy, session.frame)))
        room_samples = [row for row in session.samples if row["room"] == f"0x{ZEELA_ROOM:04X}"]
        ys = [int(row["y"]) for row in room_samples]
        transitions = [row for row in session.samples if int(row["door_transition"]) != 0]
        return {
            "target_x": target_x,
            "start_x": start["x"],
            "strategy_start_x": candidate_start["x"],
            "strategy": strategy,
            "min_y": min(ys) if ys else None,
            "end": pin(session.state),
            "door_transition": bool(transitions),
            "frames": session.frame,
            "climb_frames": session.frame - climb_start_frame,
            "rooms_observed": list(dict.fromkeys(row["room"] for row in session.samples)),
            "assist": session.assist.report(),
        }
    finally:
        env.close()


def run_probe(source: Path, target_x: tuple[int, ...], frames: int, output: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not 1 <= frames <= 2000:
        raise ValueError("--frames must be between 1 and 2000")
    if not target_x or any(value < 0 for value in target_x):
        raise ValueError("at least one non-negative --x value is required")
    trials = [
        run_trial(source, x, strategy, frames)
        for x in target_x
        for strategy in STRATEGIES
    ]
    best = min((trial["min_y"] for trial in trials if trial["min_y"] is not None), default=None)
    report: dict[str, object] = {
        "kind": "zeela_lower_climb_diagnostic_recon",
        "developmentOnly": True,
        "source": display_path(source),
        "room": f"0x{ZEELA_ROOM:04X}",
        "framesPerTrial": frames,
        "targetX": list(target_x),
        "strategies": list(STRATEGIES),
        "bestMinY": best,
        "trials": trials,
        "nonClaims": [
            "Diagnostic recon only; not pure-green evidence",
            "Not continuous evidence and no STATUS promotion",
            "No Samus placement or progression/capacity/door/event/boss RAM writes",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for trial in trials:
        print(
            f"x={trial['target_x']:3} strategy={trial['strategy']:<27} "
            f"start_x={trial['strategy_start_x']:3} min_y={trial['min_y']} "
            f"end={trial['end']} door_transition={trial['door_transition']} frames={trial['frames']}"
        )
    print(f"best_min_y={best}")
    print(f"output={display_path(output)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--x", dest="target_x", type=int, nargs="+", default=DEFAULT_TARGET_X)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(args.source, tuple(args.target_x), args.frames, args.output)


if __name__ == "__main__":
    main()
