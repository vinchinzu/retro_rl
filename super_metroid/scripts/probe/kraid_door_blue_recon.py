#!/usr/bin/env python3
"""Every-frame blue-door diagnostic for the Kraid-room left exit.

This probe is deliberately diagnostic rather than route code.  It boots one
named state, applies only the resource assist, mirrors the existing Kraid
return approach, and records read-only door-related RAM while firing four
standing left-facing shots.  It never writes progression, capacity, room,
event, boss, or door state.
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
from super_metroid.ram import (  # noqa: E402
    ADDR_DOOR_DEF_PTR,
    ADDR_INVINCIBILITY_TIMER,
    ADDR_KNOCKBACK_TIMER,
    parse_env_state,
    peek_wram,
)

KRAID_ROOM = 0xA59F
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kraid_door_blue_recon.json"
DEFAULT_FRAMES = 1200


def sample(state: Any, env: Any, phase: str) -> dict[str, object]:
    """Serialize every exposed field relevant to the door attempt."""
    extra = peek_wram(
        env,
        {
            "door_definition_ptr": ADDR_DOOR_DEF_PTR,
            "invincibility_timer": ADDR_INVINCIBILITY_TIMER,
            "knockback_timer": ADDR_KNOCKBACK_TIMER,
        },
    )
    return {
        "frame": state.frame,
        "phase": phase,
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "velocity_x": state.velocity_x,
        "velocity_y": state.velocity_y,
        "door_transition": state.door_transition,
        "transition_direction": state.transition_direction,
        "game_state": state.game_state,
        "enemy0_hp": state.enemy0_hp,
        "enemy0_x": state.enemy0_x,
        "enemy0_y": state.enemy0_y,
        "enemy0_spritemap": state.enemy0_spritemap,
        "selected_item": state.selected_item,
        **extra,
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
        self.samples.append(sample(self.state, self.env, self.phase))

    def step(self, action: Any) -> Any:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        self.record()
        return self.state

    def hold(self, frames: int, *names: str) -> None:
        action = buttons(*names) if names else idle_action()
        for _ in range(frames):
            self.step(action)


def _changed_fields(before: dict[str, object], after: dict[str, object]) -> list[str]:
    ignored = {"frame", "phase"}
    return sorted(
        key for key in after if key not in ignored and before.get(key) != after.get(key)
    )


def _transition_windows(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group contiguous frames with a nonzero door transition value."""
    windows: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    for row in samples:
        value = int(row["door_transition"])
        if value:
            if active is None:
                active = {
                    "start": row["frame"],
                    "end": row["frame"],
                    "values": [value],
                    "phases": [row["phase"]],
                }
            else:
                active["end"] = row["frame"]
                active["values"].append(value)  # type: ignore[union-attr]
                if row["phase"] not in active["phases"]:  # type: ignore[operator]
                    active["phases"].append(row["phase"])  # type: ignore[union-attr]
        elif active is not None:
            windows.append(active)
            active = None
    if active is not None:
        windows.append(active)
    return windows


def run_probe(source: Path, frames: int, output: Path) -> dict[str, object]:
    if not 1 <= frames <= 2000:
        raise ValueError("--frames must be between 1 and 2000")

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        session = ReconSession(env, assist)
        session.record()

        session.phase = "approach"
        for _ in range(min(420, frames - session.frame)):
            if session.state.samus_x <= 180:
                break
            session.step(buttons("LEFT", "B", "A"))

        session.phase = "lip_backoff"
        session.hold(min(10, frames - session.frame), "RIGHT")

        session.phase = "unmorph"
        session.hold(min(8, frames - session.frame), "UP")
        session.hold(min(8, frames - session.frame), "A")
        session.hold(min(10, frames - session.frame))

        session.phase = "face_left_release"
        session.hold(min(8, frames - session.frame), "LEFT")
        session.hold(min(6, frames - session.frame))

        for shot in range(1, 5):
            if session.frame >= frames:
                break
            session.phase = f"shot_{shot}"
            session.hold(min(4, frames - session.frame), "LEFT", "X")
            session.phase = f"shot_{shot}_fuse"
            session.hold(min(18, frames - session.frame))

        session.phase = "post_shots"
        session.hold(frames - session.frame, "LEFT", "B", "A")

        samples = session.samples
        transition_windows = _transition_windows(samples)
        shot_samples = [row for row in samples if str(row["phase"]).startswith("shot_")]
        changes_during_shots: list[dict[str, object]] = []
        for previous, current in zip(shot_samples, shot_samples[1:]):
            changed = _changed_fields(previous, current)
            if changed:
                changes_during_shots.append(
                    {"from_frame": previous["frame"], "to_frame": current["frame"], "fields": changed}
                )

        rooms = list(dict.fromkeys(str(row["room"]) for row in samples))
        report: dict[str, object] = {
            "kind": "kraid_door_blue_recon",
            "developmentOnly": True,
            "source": _display_path(source),
            "frames": session.frame,
            "sampleEveryFrame": True,
            "roomsObserved": rooms,
            "roomChanged": len(rooms) > 1,
            "doorTransitionObserved": bool(transition_windows),
            "doorTransitionWindows": transition_windows,
            "doorTransitionValues": sorted({int(row["door_transition"]) for row in samples}),
            "shotFieldChanges": changes_during_shots,
            "lastPin": {
                key: samples[-1][key] if samples else None
                for key in ("room", "pose", "x", "y")
            },
            "harnessFields": {
                "sampled": sorted(samples[0].keys()) if samples else [],
                "unavailable": [
                    "door_open_state / door state machine internals",
                    "PLM records and PLM activation state",
                    "door BTS / tile collision metadata",
                ],
            },
            "assist": assist.report(),
            "samples": samples,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"frames={session.frame} rooms={rooms} "
            f"door_transition!=0={bool(transition_windows)} output={_display_path(output)}"
        )
        print(f"last_pin={report['lastPin']}")
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(args.source, args.frames, args.output)


if __name__ == "__main__":
    main()
