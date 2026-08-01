#!/usr/bin/env python3
"""Measure the Kihunter upper down-door band without changing route state.

Each x target boots a fresh copy of the named post-Baby source, repeats the
existing shot-block climb, moves toward that target with ordinary inputs, and
samples the result while aiming DOWN.  The probe records the room selected by
the game (including the neighboring Baby Kraid hatch), but never writes room,
door, progression, capacity, event, or boss state.
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
from super_metroid.dev.common import boot_from_state, make_dev_env, place_samus  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402

KIHUNTER_ROOM = 0xA4DA
ZEELA_ROOM = 0xA471
BABY_ROOM = 0xA521
DEFAULT_FRAMES = 1200
DEFAULT_X_VALUES = (64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kihunter_zeela_recon.json"


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


def sample(state: Any, phase: str) -> dict[str, object]:
    return {
        "frame": state.frame,
        "phase": phase,
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "door_transition": state.door_transition,
        "transition_direction": state.transition_direction,
        "game_state": state.game_state,
        "velocity_x": state.velocity_x,
        "velocity_y": state.velocity_y,
    }


class ReconSession:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.phase = "boot"
        self.state = parse_env_state(env, frame=0, mode="full")
        self.samples: list[dict[str, object]] = []

    def record(self) -> None:
        self.samples.append(sample(self.state, self.phase))

    def step(self, action: Any) -> Any:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        self.record()
        return self.state

    def hold(self, frames: int, *names: str) -> None:
        action = buttons(*names) if names else idle_action()
        for _ in range(max(0, frames)):
            self.step(action)


def select_beam(session: ReconSession, budget: int) -> None:
    """Mirror the route's normal SELECT cycle without writing inventory RAM."""
    session.phase = "select_beam"
    for _ in range(8):
        if session.state.selected_item == 0 or session.frame >= budget:
            return
        session.step(buttons("SELECT"))
        session.hold(min(25, budget - session.frame))


def climb_to_upper(session: ReconSession, budget: int) -> bool:
    """Use the controller's climb only; return once the upper y band is reached."""
    session.phase = "entry_setup"
    session.hold(min(12, budget - session.frame))
    session.hold(min(8, budget - session.frame), "RIGHT")
    session.hold(min(8, budget - session.frame))
    if session.state.room_id != KIHUNTER_ROOM:
        return False
    session.phase = "climb"
    for cycle in range(10):
        for frame in range(72):
            if session.frame >= budget or session.state.room_id != KIHUNTER_ROOM:
                return False
            phase = frame % 24
            if phase < 5:
                names = ("A", "B", "UP", "X")
            elif phase < 10:
                names = ("RIGHT", "A", "B", "UP", "X")
            elif phase < 14:
                names = ("RIGHT", "A", "B", "X")
            elif phase < 18:
                names = ("RIGHT", "A", "B", "UP", "X")
            else:
                names = ("RIGHT", "B", "UP", "X")
            session.step(buttons(*names))
            if session.state.room_id != KIHUNTER_ROOM:
                return False
            if session.state.samus_y < 280:
                return True
        session.phase = "climb_replant"
        session.hold(min(8, budget - session.frame))
        session.phase = "climb"
    return False


def run_trial(source: Path, target_x: int, frames: int) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        session = ReconSession(env, assist)
        session.record()
        start = pin(session.state)
        select_beam(session, frames)
        climbed = climb_to_upper(session, frames)
        climb_pin = pin(session.state)
        natural_climb_samples = session.samples
        natural_climb_rooms = list(dict.fromkeys(row["room"] for row in natural_climb_samples))
        natural_transitions = [row for row in natural_climb_samples if int(row["door_transition"]) != 0]
        natural_climb = {
            "rooms_observed": natural_climb_rooms,
            "transition_rooms": list(dict.fromkeys(row["room"] for row in natural_transitions)),
            "final": climb_pin,
        }

        # The source currently falls into a door transition before the climb
        # reaches the upper band.  A development-only position helper keeps
        # the x sweep useful without forging any progression state.
        upper_warped = False
        if not climbed or session.state.room_id != KIHUNTER_ROOM:
            boot_from_state(env, source)
            session = ReconSession(env, assist)
            session.record()
            select_beam(session, frames)
            session.phase = "upper_warp"
            place_samus(env, target_x, 240)
            for _ in range(min(12, max(0, frames - session.frame))):
                session.step(idle_action())
            upper_warped = session.state.room_id == KIHUNTER_ROOM
            climb_pin = pin(session.state)

        session.phase = "x_position"
        direction = "RIGHT" if session.state.samus_x < target_x else "LEFT"
        for _ in range(180):
            if session.frame >= frames or session.state.room_id != KIHUNTER_ROOM:
                break
            if abs(session.state.samus_x - target_x) <= 3:
                break
            session.step(buttons(direction, "B"))

        session.phase = "down_probe"
        for index in range(max(0, frames - session.frame)):
            if session.state.room_id != KIHUNTER_ROOM:
                break
            if index % 24 < 6:
                session.step(buttons("DOWN", "X"))
            elif index % 24 < 14:
                session.step(buttons("DOWN", "A"))
            else:
                session.step(buttons("DOWN"))

        samples = session.samples
        rooms = list(dict.fromkeys(row["room"] for row in samples))
        transitions = [row for row in samples if int(row["door_transition"]) != 0]
        transition_rooms = list(dict.fromkeys(row["room"] for row in transitions))
        down_samples = [row for row in samples if row["phase"] == "down_probe"]
        down_x = [int(row["x"]) for row in down_samples]
        final = samples[-1] if samples else {}
        return {
            "target_x": target_x,
            "climbed": climbed,
            "upper_warped": upper_warped,
            "upper_entry_method": "climb" if climbed else "development_position_warp",
            "start": start,
            "post_climb": climb_pin,
            "natural_climb_attempt": natural_climb,
            "rooms_observed": rooms,
            "transition_rooms": transition_rooms,
            "door_transition_observed": bool(transitions),
            "down_x_min": min(down_x) if down_x else None,
            "down_x_max": max(down_x) if down_x else None,
            "final": {key: final.get(key) for key in ("room", "pose", "x", "y", "door_transition")},
            "samples": samples,
            "assist": assist.report(),
        }
    finally:
        env.close()


def run_probe(source: Path, x_values: tuple[int, ...], frames: int, output: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not 1 <= frames <= 2000:
        raise ValueError("--frames must be between 1 and 2000")
    if not x_values:
        raise ValueError("at least one x target is required")

    trials = [run_trial(source, target_x, frames) for target_x in x_values]
    report: dict[str, object] = {
        "kind": "kihunter_zeela_door_band_recon",
        "developmentOnly": True,
        "source": display_path(source),
        "framesPerTrial": frames,
        "targetXValues": list(x_values),
        "rooms": {"kihunter": f"0x{KIHUNTER_ROOM:04X}", "zeela": f"0x{ZEELA_ROOM:04X}", "baby": f"0x{BABY_ROOM:04X}"},
        "trials": trials,
        "nonClaims": [
            "Not pure-green evidence",
            "Not continuous evidence",
            "No STATUS promotion",
            "No PLM, door-BTS, or door-state RAM determination",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for trial in trials:
        print(
            f"target_x={trial['target_x']} climbed={trial['climbed']} "
            f"post_climb={trial['post_climb']} transition_rooms={trial['transition_rooms']} "
            f"down_x={trial['down_x_min']}..{trial['down_x_max']} final={trial['final']}"
        )
    print(f"output={display_path(output)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--x", dest="x_values", type=int, nargs="+", default=DEFAULT_X_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(args.source, tuple(args.x_values), args.frames, args.output)


if __name__ == "__main__":
    main()
