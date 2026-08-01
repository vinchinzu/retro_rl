#!/usr/bin/env python3
"""Sweep natural lower-alcove launch inputs in the Kihunter room.

Each trial boots a fresh copy of the post-Baby source state.  The probe only
uses controller inputs and the resource-only assist; it does not position
Samus or write progression, door, inventory-capacity, event, or boss state.
The result is diagnostic data for the lower-alcove climb, not route evidence.
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

KIHUNTER_ROOM = 0xA4DA
BABY_ROOM = 0xA521
DEFAULT_FRAMES = 480
DEFAULT_LEFT_FRAMES = tuple(range(0, 33, 4))
DEFAULT_RIGHT_CAPS = (450, 460, 470, 475)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kihunter_climb_recon.json"

# The patterns differ only in shot cadence; horizontal control remains capped.
SHOT_PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "up_shot": (
        ("A", "B", "UP", "X"),
        ("A", "B", "UP", "X"),
        ("RIGHT", "A", "B", "UP", "X"),
        ("RIGHT", "A", "B", "X"),
    ),
    "alternating_shot": (
        ("A", "B", "UP", "X"),
        ("RIGHT", "A", "B", "X"),
        ("A", "B", "UP", "X"),
        ("LEFT", "A", "B", "X"),
    ),
    "held_up_shot": (
        ("A", "B", "UP", "X"),
        ("A", "B", "UP", "X"),
        ("A", "B", "UP", "X"),
        ("RIGHT", "A", "B", "UP", "X"),
    ),
}


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


class TrialSession:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="full")
        self.samples: list[dict[str, int | str]] = []

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


def select_beam(session: TrialSession, frames: int) -> None:
    """Mirror the route's ordinary SELECT cycle without inventory writes."""
    for _ in range(8):
        if session.state.selected_item == 0 or session.frame >= frames:
            return
        session.step(buttons("SELECT"))
        for _ in range(min(25, frames - session.frame)):
            session.step(idle_action())


def action_for(state: Any, right_cap: int, pattern: tuple[tuple[str, ...], ...], frame: int) -> Any:
    # Stay inside the requested cap; the left brake also prevents an east-door
    # false positive when momentum carries Samus past the cap.
    if state.samus_x >= right_cap:
        horizontal = "LEFT"
    elif state.samus_x <= 360:
        horizontal = "RIGHT"
    else:
        horizontal = "RIGHT"
    names = [horizontal, *pattern[frame % len(pattern)]]
    return buttons(*names)


def run_trial(source: Path, left_frames: int, right_cap: int, pattern_name: str, frames: int) -> dict[str, object]:
    env = make_dev_env()
    try:
        boot_from_state(env, source)
        session = TrialSession(env)
        start = pin(session.state)

        select_beam(session, frames)
        # Let the source settle, then use the launch's only swept knob: brief
        # natural LEFT before entering the capped climb pattern.
        for _ in range(min(12, frames)):
            session.step(idle_action())
        for _ in range(min(left_frames, max(0, frames - session.frame))):
            session.step(buttons("LEFT"))

        pattern = SHOT_PATTERNS[pattern_name]
        while session.frame < frames and session.state.room_id == KIHUNTER_ROOM:
            session.step(action_for(session.state, right_cap, pattern, session.frame))

        samples = session.samples
        room_samples = [row for row in samples if row["room"] == f"0x{KIHUNTER_ROOM:04X}"]
        ys = [int(row["y"]) for row in room_samples]
        xs = [int(row["x"]) for row in room_samples]
        upper_land = any(
            row["room"] == f"0x{KIHUNTER_ROOM:04X}"
            and int(row["y"]) < 280
            and int(row["door_transition"]) == 0
            for row in samples
        )
        transitions = [row for row in samples if int(row["door_transition"]) != 0]
        return {
            "left_frames": left_frames,
            "right_cap": right_cap,
            "shot_pattern": pattern_name,
            "start": start,
            "min_y": min(ys) if ys else None,
            "min_x": min(xs) if xs else None,
            "max_x": max(xs) if xs else None,
            "final": pin(session.state),
            "upper_land": upper_land,
            "rooms_observed": list(dict.fromkeys(row["room"] for row in samples)),
            "transition_rooms": list(dict.fromkeys(row["room"] for row in transitions)),
            "frames_run": session.frame,
            "assist": session.assist.report(),
        }
    finally:
        env.close()


def run_probe(source: Path, left_frames: tuple[int, ...], right_caps: tuple[int, ...], frames: int, output: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not 1 <= frames <= 2000:
        raise ValueError("--frames must be between 1 and 2000")
    if any(value < 0 for value in left_frames):
        raise ValueError("--left-frames values must be non-negative")
    if any(value < 1 for value in right_caps):
        raise ValueError("--right-cap values must be positive")

    trials = [
        run_trial(source, left, cap, pattern, frames)
        for left in left_frames
        for cap in right_caps
        for pattern in SHOT_PATTERNS
    ]
    report: dict[str, object] = {
        "kind": "kihunter_lower_alcove_climb_recon",
        "developmentOnly": True,
        "source": display_path(source),
        "framesPerTrial": frames,
        "leftFrames": list(left_frames),
        "rightCaps": list(right_caps),
        "shotPatterns": list(SHOT_PATTERNS),
        "rooms": {"kihunter": f"0x{KIHUNTER_ROOM:04X}", "baby": f"0x{BABY_ROOM:04X}"},
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
            f"left={trial['left_frames']:2} cap={trial['right_cap']} pattern={trial['shot_pattern']:<18} "
            f"min_y={trial['min_y']} x={trial['min_x']}..{trial['max_x']} "
            f"final={trial['final']} upper_land={trial['upper_land']}"
        )
    print(f"output={display_path(output)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--left-frames", type=int, nargs="+", default=DEFAULT_LEFT_FRAMES)
    parser.add_argument("--right-cap", type=int, nargs="+", default=DEFAULT_RIGHT_CAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(args.source, tuple(args.left_frames), tuple(args.right_cap), args.frames, args.output)


if __name__ == "__main__":
    main()
