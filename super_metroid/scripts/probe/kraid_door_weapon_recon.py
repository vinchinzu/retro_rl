#!/usr/bin/env python3
"""Compare beam, missile, and Super shots at the Kraid-room left door.

This is a development-only diagnostic.  Each mode boots a fresh copy of the
named source state, uses only the resource assist, and samples navigation,
weapon, and ammo state after every emulator frame.  It does not write room,
door, event, boss, progression, or capacity state.
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

DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kraid_door_weapon_recon.json"
DEFAULT_FRAMES = 1200
WEAPONS = {"beam": 0, "missile": 1, "super": 2}


def pin(state: Any) -> dict[str, object]:
    return {
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
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
        "selected_item": state.selected_item,
        "missiles": state.missiles,
        "max_missiles": state.max_missiles,
        "super_missiles": state.super_missiles,
        "max_super_missiles": state.max_super_missiles,
        "power_bombs": state.power_bombs,
        "max_power_bombs": state.max_power_bombs,
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

    def step(self, action: Any) -> None:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        self.record()

    def hold(self, frames: int, *names: str) -> None:
        action = buttons(*names) if names else idle_action()
        for _ in range(max(0, frames)):
            self.step(action)


def select_weapon(session: ReconSession, target: int) -> bool:
    """Use the normal SELECT cycle, without writing selected-item RAM."""
    for _ in range(4):
        if session.state.selected_item == target:
            return True
        session.phase = "select_weapon"
        session.step(buttons("SELECT"))
        session.hold(25)
    return session.state.selected_item == target


def run_mode(source: Path, mode: str, frames: int) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        session = ReconSession(env, assist)
        session.record()
        source_state = session.state
        target = WEAPONS[mode]
        available = (
            target == 0
            or (target == 1 and source_state.max_missiles > 0)
            or (target == 2 and source_state.max_super_missiles > 0)
        )
        selected = available and select_weapon(session, target)

        if selected:
            session.phase = "approach"
            while session.frame < frames and session.state.samus_x > 180:
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
                session.phase = f"{mode}_shot_{shot}"
                session.hold(min(4, frames - session.frame), "LEFT", "X")
                session.phase = f"{mode}_shot_{shot}_fuse"
                session.hold(min(18, frames - session.frame))
            session.phase = "brief_spin"
            session.hold(frames - session.frame, "LEFT", "B", "A")

        rooms = list(dict.fromkeys(str(row["room"]) for row in session.samples))
        transitions = [row for row in session.samples if row["door_transition"]]
        final = session.samples[-1] if session.samples else {}
        return {
            "mode": mode,
            "weaponId": target,
            "availableOnSource": available,
            "selected": selected,
            "frames": session.frame,
            "roomsObserved": rooms,
            "roomChanged": len(rooms) > 1,
            "doorTransitionObserved": bool(transitions),
            "doorTransitionValues": sorted({int(row["door_transition"]) for row in session.samples}),
            "sourceAmmo": {
                "missiles": source_state.missiles,
                "maxMissiles": source_state.max_missiles,
                "supers": source_state.super_missiles,
                "maxSupers": source_state.max_super_missiles,
                "powerBombs": source_state.power_bombs,
                "maxPowerBombs": source_state.max_power_bombs,
            },
            "lastPin": {key: final.get(key) for key in ("room", "pose", "x", "y")},
            "assist": assist.report(),
            "samples": session.samples,
        }
    finally:
        env.close()


def run_probe(source: Path, modes: list[str], frames: int, output: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not 1 <= frames <= 2000:
        raise ValueError("--frames must be between 1 and 2000")
    results = [run_mode(source, mode, frames) for mode in modes]
    report: dict[str, object] = {
        "kind": "kraid_door_weapon_recon",
        "developmentOnly": True,
        "source": _display_path(source),
        "sampleEveryFrame": True,
        "modes": results,
        "nonClaims": ["not pure green", "not continuous", "not STATUS promotion"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"mode={result['mode']} available={result['availableOnSource']} "
            f"selected={result['selected']} frames={result['frames']} "
            f"rooms={result['roomsObserved']} "
            f"door_transition!=0={result['doorTransitionObserved']} "
            f"last_pin={result['lastPin']}"
        )
    print(f"output={_display_path(output)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("beam", "missile", "super", "all"), default="all")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    modes = list(WEAPONS) if args.mode == "all" else [args.mode]
    run_probe(args.source, modes, args.frames, args.output)


if __name__ == "__main__":
    main()
