#!/usr/bin/env python3
"""Verify Red Tower Ice checkpoint edges from the natural Bat handoff.

The default command runs the first edge twice plus an enemy-phase sweep.  It
does not claim the full Red Tower -> Hellway room clear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import write_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.kpdr.k5.red_ice_climb import (  # noqa: E402
    LOWER_RIPPER_1,
    POLICY_ID,
    checkpoint_supported,
    play_bottom_to_ripper1,
    ripper_at_height,
)

DEFAULT_SOURCE = INTEGRATION_DIR / "scratch" / "post_ice_bat_to_red_pure.state"
DEFAULT_OUTPUT = INTEGRATION_DIR / "scratch" / "red_ice_lower_ripper1_pure.state"


class ProbeSession:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str):
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        return self.state


def run_once(source: Path, phase_offset: int, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        for _ in range(max(0, int(phase_offset))):
            session.step(idle_action(), "red_ice_phase_perturb")
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_bottom_to_ripper1(session)
        except Exception as exc:  # report the full sweep instead of stopping early
            error = str(exc)
        enemy = ripper_at_height(env, 2376)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        green = error is None and checkpoint_supported(env, session.state, LOWER_RIPPER_1)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "phaseOffset": int(phase_offset),
            "policyFrames": policy_frames,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def _offsets(value: str) -> list[int]:
    if value == "full":
        return list(range(0, 241, 8))
    return [int(part.strip(), 0) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--phase-offsets",
        default="full",
        help="comma-separated idle offsets, or 'full' for 0..240 step 8",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_OUTPUT,
        type=Path,
        help="save the offset-0 checkpoint state (default scratch path)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    offsets = _offsets(args.phase_offsets)
    # Two independent offset-zero boots are the exact dual check.  The sweep
    # follows, excluding duplicate zero in the compact report.
    runs = [run_once(args.source, 0, save=args.save), run_once(args.source, 0)]
    runs.extend(run_once(args.source, value) for value in offsets if value != 0)
    green = all(row["green"] for row in runs)
    exact_fields = ("green", "policyFrames", "room", "xy", "pose", "freezeTimer", "enemyX", "error")
    dual_exact = all(runs[0][field] == runs[1][field] for field in exact_fields)
    report = {
        "policy": POLICY_ID,
        "scope": "bottom_floor->lower_ripper_1",
        "green": green,
        "dualExact": dual_exact,
        "runs": runs,
        "policyFrames": {
            "min": min(row["policyFrames"] for row in runs),
            "max": max(row["policyFrames"] for row in runs),
        },
        "fps": {
            "min": min(row["fps"] for row in runs),
            "max": max(row["fps"] for row in runs),
        },
        "saved": str(args.save) if args.save is not None and runs[0]["green"] else None,
        "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        mark = "GREEN" if green else "RED"
        print(
            f"{mark} {POLICY_ID} dual={report['dualExact']} "
            f"runs={len(runs)} policy_frames="
            f"{report['policyFrames']['min']}..{report['policyFrames']['max']} "
            f"fps={report['fps']['min']:.0f}..{report['fps']['max']:.0f}"
        )
        for row in runs:
            if not row["green"]:
                print(f"  RED offset={row['phaseOffset']}: {row['error']} {row['xy']}")
        if report["saved"]:
            print(f"  checkpoint -> {report['saved']}")
        print("  partial only: lower_ripper_1; Hellway remains RED")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
