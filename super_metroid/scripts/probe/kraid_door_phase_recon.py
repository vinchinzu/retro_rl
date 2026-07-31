#!/usr/bin/env python3
"""Phase-instrumented diagnostic for the Kraid left blue door.

This probe mirrors ``play_kraid_to_eye_return`` without calling the controller
itself, so each choreography phase can be inspected independently. It boots
only the named save state and applies the resource-only assist. It never writes
progression, capacity, event, boss, room, or position state.
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

def _display_path(path: Path) -> str:
    """Prefer repo-relative paths in stdout (no machine home prefixes)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import unmorph  # noqa: E402

KRAID_ROOM = 0xA59F
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kraid_door_phase_recon.json"


def sample(state: Any, phase: str) -> dict[str, object]:
    """Serialize only read-only fields useful for diagnosing the door."""
    return {
        "frame": state.frame,
        "phase": phase,
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "game_state": state.game_state,
        "door_transition": state.door_transition,
        "transition_direction": state.transition_direction,
        "enemy0_hp": state.enemy0_hp,
        "boss_bits": list(state.boss_bits),
        "selected_item": state.selected_item,
        "velocity_y": state.velocity_y,
    }


class ReconSession:
    """Small ControllerSession-compatible wrapper with full-state sampling."""

    def __init__(self, env: Any, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="full")
        self.samples: list[dict[str, object]] = []
        self.phase = "boot"

    def record(self) -> None:
        self.samples.append(sample(self.state, self.phase))

    def step(self, action: Any, reason: str = "probe") -> Any:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        self.record()
        return self.state

    def hold(self, frames: int, *names: str, reason: str) -> None:
        action = buttons(*names) if names else idle_action()
        for _ in range(frames):
            self.step(action, reason)


def set_phase(session: ReconSession, phase: str) -> None:
    session.phase = phase


def run_phases(env: Any, source: Path, budget: int) -> dict[str, object]:
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, source)
    session = ReconSession(env, assist)
    session.record()
    phase_ranges: dict[str, tuple[int, int]] = {}

    def begin(name: str) -> int:
        set_phase(session, name)
        return session.frame

    def end(name: str, start: int) -> None:
        phase_ranges[name] = (start, session.frame)

    start = begin("approach")
    for _ in range(420):
        if session.state.samus_x <= 180 or session.frame >= budget:
            break
        session.step(buttons("LEFT", "B", "A"), "kraid_return_approach")
    end("approach", start)

    start = begin("lip_backoff")
    if session.frame < budget:
        session.hold(min(10, budget - session.frame), "RIGHT", reason="kraid_return_lip_backoff")
    end("lip_backoff", start)

    start = begin("unmorph")
    # ``unmorph`` can spend up to 26 frames; do not exceed a small CLI budget.
    if budget - session.frame >= 26:
        # Use the production primitive, but retain this probe's phase label.
        unmorph(session)  # type: ignore[arg-type]
    end("unmorph", start)

    start = begin("face_left_release")
    if session.frame < budget:
        session.hold(min(8, budget - session.frame), "LEFT", reason="kraid_return_face_left")
    if session.frame < budget:
        session.hold(min(6, budget - session.frame), reason="kraid_return_release")
    end("face_left_release", start)

    start = begin("door_shots")
    for _ in range(4):
        if session.frame >= budget:
            break
        session.hold(min(4, budget - session.frame), "LEFT", "X", reason="kraid_return_door_shot")
        if session.frame < budget:
            session.hold(min(18, budget - session.frame), reason="kraid_return_door_fuse")
    end("door_shots", start)

    start = begin("spin_push")
    for _ in range(720):
        if session.frame >= budget:
            break
        state = session.step(buttons("LEFT", "B", "A"), "kraid_return_exit")
        if state.room_id != KRAID_ROOM:
            break
        if state.pose in (137, 138) and state.samus_x <= 80:
            if session.frame < budget:
                session.hold(min(4, budget - session.frame), reason="kraid_return_lip_release")
            if session.frame < budget:
                session.hold(min(4, budget - session.frame), "RIGHT", reason="kraid_return_lip_backoff")
            if session.frame < budget:
                session.hold(min(4, budget - session.frame), "LEFT", reason="kraid_return_reface")
            if session.frame < budget:
                session.hold(min(4, budget - session.frame), "LEFT", "X", reason="kraid_return_lip_reshot")
            if session.frame < budget:
                session.hold(min(14, budget - session.frame), reason="kraid_return_lip_fuse")
    end("spin_push", start)

    return finish_report("phases", source, session, assist, phase_ranges)


def _attempt_sequence(session: ReconSession, variant: str, budget: int) -> None:
    """Run one normal-input Y approach, then the common door attempt."""
    set_phase(session, f"{variant}:approach")
    jump_frames = {"floor_walk": 0, "short_hop": 18, "medium_hop": 36}[variant]
    for index in range(420):
        if session.frame >= budget or session.state.samus_x <= 180:
            break
        names = ("LEFT",) if index >= jump_frames else ("LEFT", "A")
        session.step(buttons(*names), f"y_sweep_{variant}_approach")
    set_phase(session, f"{variant}:door_shots")
    for _ in range(4):
        if session.frame >= budget:
            break
        session.hold(min(4, budget - session.frame), "LEFT", "X", reason=f"y_sweep_{variant}_shot")
        if session.frame < budget:
            session.hold(min(18, budget - session.frame), reason=f"y_sweep_{variant}_fuse")
    set_phase(session, f"{variant}:brief_spin")
    if session.frame < budget:
        session.hold(min(80, budget - session.frame), "LEFT", "B", "A", reason=f"y_sweep_{variant}_spin")


def run_y_sweep(env_factory: Any, source: Path, budget: int) -> dict[str, object]:
    variants = ("floor_walk", "short_hop", "medium_hop")
    attempts: list[dict[str, object]] = []
    per_attempt = max(1, budget // len(variants))
    for variant in variants:
        env = env_factory()
        assist = UnlimitedResourcesAssist()
        try:
            boot_from_state(env, source)
            session = ReconSession(env, assist)
            session.record()
            _attempt_sequence(session, variant, per_attempt)
            attempts.append(finish_report(variant, source, session, assist, {}))
        finally:
            env.close()
    return {
        "kind": "kraid_door_phase_recon",
        "mode": "y-sweep",
        "developmentOnly": True,
        "source": _display_path(source),
        "budget": budget,
        "attempts": attempts,
        "roomsObserved": sorted({room for attempt in attempts for room in attempt["roomsObserved"]}),
        "doorTransitionObserved": any(attempt["doorTransitionObserved"] for attempt in attempts),
        "assist": [attempt["assist"] for attempt in attempts],
    }


def finish_report(
    mode: str,
    source: Path,
    session: ReconSession,
    assist: UnlimitedResourcesAssist,
    phase_ranges: dict[str, tuple[int, int]],
) -> dict[str, object]:
    samples = session.samples
    rooms = list(dict.fromkeys(str(item["room"]) for item in samples))
    transition_samples = [item for item in samples if int(item["door_transition"]) != 0]
    return {
        "kind": "kraid_door_phase_recon",
        "mode": mode,
        "developmentOnly": True,
        "source": _display_path(source),
        "frames": session.frame,
        "start": samples[0] if samples else None,
        "end": samples[-1] if samples else None,
        "lastPin": {
            "room": samples[-1]["room"] if samples else None,
            "pose": samples[-1]["pose"] if samples else None,
            "x": samples[-1]["x"] if samples else None,
            "y": samples[-1]["y"] if samples else None,
        },
        "roomsObserved": rooms,
        "roomChanged": len(rooms) > 1,
        "doorTransitionObserved": bool(transition_samples),
        "doorTransitionMax": max((int(item["door_transition"]) for item in samples), default=0),
        "phaseRanges": {name: {"start": start, "end": end} for name, (start, end) in phase_ranges.items()},
        "samples": samples,
        "assist": assist.report(),
    }


def write_output(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict[str, object], output: Path) -> None:
    print(f"mode={report['mode']} frames={report.get('frames', 'per-attempt')} output={_display_path(output)}")
    if report["mode"] == "phases":
        for name, span in report["phaseRanges"].items():
            rows = [row for row in report["samples"] if row["phase"] == name]
            print(f"{name}: frames={span['start']}..{span['end']} start/end={rows[0] if rows else None} / {rows[-1] if rows else None}")
        print(f"door_transition!=0: {report['doorTransitionObserved']}; final pin: {report['lastPin']}")
    else:
        for attempt in report["attempts"]:
            print(f"{attempt['mode']}: final pin={attempt['lastPin']} rooms={attempt['roomsObserved']} door_transition!=0={attempt['doorTransitionObserved']}")
        print(f"any door_transition!=0: {report['doorTransitionObserved']}; rooms={report['roomsObserved']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("phases", "y-sweep"), default="phases")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.frames <= 2000:
        parser.error("--frames must be between 1 and 2000")

    if args.mode == "phases":
        env = make_dev_env()
        try:
            report = run_phases(env, args.source, args.frames)
        finally:
            env.close()
    else:
        report = run_y_sweep(make_dev_env, args.source, args.frames)
    write_output(report, args.output)
    print_summary(report, args.output)


if __name__ == "__main__":
    main()
