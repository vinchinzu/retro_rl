#!/usr/bin/env python3
"""Predict Super Metroid trajectories from start state + input tape.

Demonstrates physics predictor client interface for route planning and
trajectory analysis without requiring full emulation for every candidate.

Usage:
    # Use stub predictor with synthetic inputs
    uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \\
        --frames 60 --predictor stub

    # Load state from pin file (when available)
    uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \\
        --state snes/super_metroid/custom_integrations/SuperMetroid-Snes/pin.state \\
        --frames 120 --predictor stub --output trajectory.json

    # Use sm_rev backend (when sm_rev binary is available)
    uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \\
        --frames 60 --predictor sm_rev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from super_metroid.physics_sim import (
    FrameInput,
    SimState,
    Trajectory,
    load_predictor,
)


def _default_start_state() -> SimState:
    """Default start state for testing (Landing Site).

    Simple grounded state for smoke testing without ROM/state file.
    """
    return SimState(
        frame=0,
        room_id=0x91F8,  # Landing Site
        samus_x=400,
        samus_y=200,
        samus_x_sub=0,
        samus_y_sub=0,
        velocity_x=0,
        velocity_y=0,
        velocity_x_sub=0,
        velocity_y_sub=0,
        momentum_x=0,
        momentum_x_sub=0,
        pose=0,
        facing=0x08,  # Right
        movement_type=0,  # Grounded
        speed_counter=0,
        speed_flag=0,
        shinespark_timer=0,
    )


def _synthetic_inputs(num_frames: int) -> list[FrameInput]:
    """Generate synthetic input sequence for testing.

    Creates a simple walk-right + jump pattern for demonstration.
    """
    inputs: list[FrameInput] = []
    for i in range(num_frames):
        buttons = 0
        # Hold RIGHT for movement
        if i < num_frames - 10:
            buttons |= 0x40  # RIGHT
        # Jump on frames 10, 30, 50
        if i in (10, 30, 50):
            buttons |= 0x01  # B (jump)
        inputs.append(FrameInput(buttons=buttons))
    return inputs


def _load_state_from_file(path: Path) -> SimState:
    """Load state from save state file (placeholder).

    Currently not implemented - would require emulator state parsing.
    Use default state for now.
    """
    # TODO: Implement state loading from .state files when needed
    # This would require stable-retro state parsing utilities
    raise NotImplementedError(
        "State loading from .state files not yet implemented. "
        "Use --predictor stub without --state for testing."
    )


def _load_inputs_from_file(path: Path) -> list[FrameInput]:
    """Load input tape from JSON file.

    Expected format:
        {"inputs": [{"buttons": 0x40}, {"buttons": 0x41}, ...]}
    """
    with path.open("r") as f:
        data = json.load(f)
    return [FrameInput.from_dict(inp) for inp in data["inputs"]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict Super Metroid trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Load start state from file (currently not implemented)",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Load input tape from JSON file (default: synthetic)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to predict (default: 60)",
    )
    parser.add_argument(
        "--predictor",
        choices=["stub", "sm_rev"],
        default="stub",
        help="Predictor backend (default: stub)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write trajectory JSON to file (default: stdout)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary instead of full trajectory",
    )

    args = parser.parse_args()

    # Load or create start state
    if args.state:
        try:
            start = _load_state_from_file(args.state)
        except NotImplementedError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        start = _default_start_state()

    # Load or generate inputs
    if args.inputs:
        inputs = _load_inputs_from_file(args.inputs)
    else:
        inputs = _synthetic_inputs(args.frames)

    # Load predictor
    try:
        predictor = load_predictor(args.predictor)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Predict trajectory
    try:
        trajectory = predictor.predict(start, inputs)
    except RuntimeError as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        return 1

    # Output
    if args.summary:
        _print_summary(trajectory)
    else:
        output = json.dumps(trajectory.to_dict(), indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Wrote trajectory to {args.output}", file=sys.stderr)
        else:
            print(output)

    return 0


def _print_summary(trajectory: Trajectory) -> None:
    """Print human-readable trajectory summary."""
    start = trajectory.start
    print(f"Predictor: {trajectory.predictor}")
    print(f"Frames: {len(trajectory.frames)}")
    print(f"Start: room 0x{start.room_id:04X} @ ({start.samus_x}, {start.samus_y})")

    if trajectory.frames:
        first = trajectory.frames[0]
        last = trajectory.frames[-1]
        dx = last.samus_x - start.samus_x
        dy = last.samus_y - start.samus_y
        print(f"First frame: ({first.samus_x}, {first.samus_y})")
        print(f"Last frame: ({last.samus_x}, {last.samus_y})")
        print(f"Displacement: ({dx:+d}, {dy:+d}) pixels")

        # Find peak/min Y
        peak_y = min(f.samus_y for f in trajectory.frames)
        print(f"Peak Y: {peak_y} (apex {start.samus_y - peak_y:+d} from start)")


if __name__ == "__main__":
    sys.exit(main())
