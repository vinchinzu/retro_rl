#!/usr/bin/env python3
"""Bounded RAM recon for the Kraid-room left blue door.

This is a diagnostic probe only.  It boots one named state, applies the
contracted resource assist, walks left, and tries four beam shots.  It does
not write room, item, event, boss, or capacity state.

Example (from the monorepo root)::

    uv run python super_metroid/scripts/probe/kraid_left_door_recon.py \
      --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
      --frames 600
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


def _sample(state: Any) -> dict[str, object]:
    """Return only fields needed to diagnose the door attempt."""
    area = state.area_index
    return {
        "frame": state.frame,
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "game_state": state.game_state,
        "door_transition": state.door_transition,
        "transition_direction": state.transition_direction,
        "enemy0_hp": state.enemy0_hp,
        "boss_bits_area": state.boss_bits[area] if 0 <= area < len(state.boss_bits) else None,
        "boss_bits": list(state.boss_bits),
        "selected_item": state.selected_item,
    }


class _ReconSession:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="full")

    def step(self, action: Any) -> Any:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        return self.state


def run_recon(
    *, source: Path, frames: int, sample_every: int, output: Path, shots: int
) -> dict[str, object]:
    if frames < 1 or frames > 2000:
        raise ValueError("--frames must be between 1 and 2000")
    if sample_every < 1:
        raise ValueError("--sample-every must be positive")
    if shots < 0 or shots > 4:
        raise ValueError("--shots must be between 0 and 4")

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    samples: list[dict[str, object]] = []
    try:
        boot_from_state(env, source)
        session = _ReconSession(env, assist)

        def record(force: bool = False) -> None:
            if force or session.frame % sample_every == 0:
                sample = _sample(session.state)
                samples.append(sample)
                print(json.dumps(sample, sort_keys=True))

        record(force=True)
        approach_frames = min(frames, 360)
        for _ in range(approach_frames):
            session.step(buttons("LEFT"))
            record()

        for _ in range(shots):
            for _ in range(min(4, frames - session.frame)):
                session.step(buttons("LEFT", "X"))
                record()
            for _ in range(min(18, frames - session.frame)):
                session.step(idle_action())
                record()

        while session.frame < frames:
            session.step(idle_action())
            record()
        record(force=True)

        rooms = list(dict.fromkeys(sample["room"] for sample in samples))
        trajectory = {
            "start": samples[0] if samples else None,
            "end": samples[-1] if samples else None,
            "min_x": min(int(sample["x"]) for sample in samples),
            "max_x": max(int(sample["x"]) for sample in samples),
            "min_y": min(int(sample["y"]) for sample in samples),
            "max_y": max(int(sample["y"]) for sample in samples),
        }
        report: dict[str, object] = {
            "kind": "kraid_left_door_recon",
            "developmentOnly": True,
            "source": str(source),
            "frames": session.frame,
            "sampleEvery": sample_every,
            "shots": shots,
            "roomsObserved": rooms,
            "roomChanged": len(rooms) > 1,
            "trajectory": trajectory,
            "samples": samples,
            "assist": assist.report(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"summary": report["trajectory"], "roomChanged": report["roomChanged"], "output": str(output)}))
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "super_metroid" / "debug" / "kraid_left_door_recon.json",
    )
    args = parser.parse_args()
    run_recon(
        source=args.source,
        frames=args.frames,
        sample_every=args.sample_every,
        output=args.output,
        shots=args.shots,
    )


if __name__ == "__main__":
    main()
