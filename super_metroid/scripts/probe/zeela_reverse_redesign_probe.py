#!/usr/bin/env python3
"""Planner geometry probe for Zeela reverse (floor → middle → upper).

Diagnostic only. Boots ``post_kihunter_to_zeela_return``, uses ordinary
inputs + resource assist, and reports min_y / end pin per named strategy.
Not pure-green or continuous evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402

ZEELA = 0xA471
DEFAULT_SOURCE = (
    ROOT
    / "super_metroid/custom_integrations/SuperMetroid-Snes/scratch"
    / "post_kihunter_to_zeela_return.state"
)
DEFAULT_OUTPUT = ROOT / "super_metroid/debug/zeela_reverse_redesign_probe.json"


def pin(state: Any) -> dict[str, object]:
    return {
        "room": f"0x{state.room_id:04X}",
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "vx": state.velocity_x,
        "vy": state.velocity_y,
        "door_transition": state.door_transition,
    }


class Session:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="full")
        self.min_y = self.state.samus_y
        self.trace: list[dict[str, object]] = []

    def step(self, *names: str) -> None:
        action = buttons(*names) if names else idle_action()
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, self.state)
        if self.state.room_id == ZEELA:
            self.min_y = min(self.min_y, self.state.samus_y)
        if self.frame % 15 == 0 or self.state.door_transition:
            self.trace.append({"f": self.frame, **pin(self.state)})

    def hold(self, n: int, *names: str) -> None:
        for _ in range(max(0, n)):
            self.step(*names)

    def morph(self) -> None:
        if self.state.pose not in (0x1D, 0x1E, 0x1F, 0x79, 0x7A, 0x7B):
            self.hold(8, "DOWN")
            self.hold(4)

    def unmorph(self) -> None:
        if self.state.pose in (0x1D, 0x1E, 0x1F, 0x79, 0x7A, 0x7B, 39, 40, 137, 138):
            self.hold(8, "UP")
            self.hold(6)


def roll_left_to(session: Session, x_max: int, y_min: int, budget: int) -> bool:
    """Morph-roll left until x<=x_max and y>=y_min (floor band)."""
    session.morph()
    for _ in range(budget):
        s = session.state
        if s.room_id != ZEELA:
            return False
        if s.door_transition and s.samus_y > 250:
            return False
        if s.samus_x <= x_max and s.samus_y >= y_min:
            return True
        # Bomb occasionally if snagged on debris
        if session.frame % 50 < 2:
            session.step("LEFT", "X")
        else:
            session.step("LEFT")
    return session.state.samus_x <= x_max


def align_x(session: Session, lo: int, hi: int, budget: int = 120) -> None:
    session.unmorph()
    for _ in range(budget):
        s = session.state
        if lo <= s.samus_x <= hi:
            break
        if s.samus_x < lo:
            session.step("RIGHT")
        else:
            session.step("LEFT")
    session.hold(6)


def strategy_hj_under_hole(session: Session, frames: int) -> None:
    """Crouch-load Hi-Jump under hole, occasional UP+X."""
    session.unmorph()
    session.hold(8, "DOWN")
    session.hold(2)
    for f in range(frames):
        if session.state.room_id != ZEELA:
            return
        phase = f % 28
        if phase < 4:
            session.step("UP", "X")
        elif phase < 14:
            session.step("A")
        elif phase < 18:
            session.step("A", "UP")
        else:
            session.step()
        if session.state.samus_y <= 320 and session.state.velocity_y == 0:
            return


def strategy_bomb_jump_hole(session: Session, frames: int) -> None:
    """Morph bomb-jump cycle under hole (kihunter-style)."""
    session.morph()
    end = session.frame + frames
    while session.frame < end:
        if session.state.room_id != ZEELA:
            return
        if session.state.samus_y <= 320:
            return
        session.hold(2, "X")
        wait = 48 if session.state.samus_y > 360 else 30
        for _ in range(wait):
            session.step()
            if session.state.samus_y <= 320:
                return


def strategy_reverse_shot(session: Session, frames: int, keep_x_min: int = 55) -> None:
    """UP+X open then jump/left, with hard x floor to avoid Energy Tank door."""
    session.unmorph()
    session.hold(6, "UP", "X")
    for f in range(frames):
        if session.state.room_id != ZEELA:
            return
        if session.state.samus_y <= 320:
            return
        phase = f % 24
        if phase < 6:
            session.step("UP", "X")
        elif phase < 12:
            session.step("A")
        elif session.state.samus_x <= keep_x_min:
            session.step("RIGHT", "A")
        else:
            session.step("LEFT", "A")


def strategy_spin_up(session: Session, frames: int) -> None:
    """Spin-jump with brief left/right wiggle under hole."""
    session.unmorph()
    for f in range(frames):
        if session.state.room_id != ZEELA:
            return
        if session.state.samus_y <= 320:
            return
        phase = f % 20
        if phase < 8:
            session.step("A", "B")
        elif phase < 12:
            session.step("LEFT", "A", "B")
        elif phase < 16:
            session.step("RIGHT", "A", "B")
        else:
            session.step("UP", "X")


def strategy_wall_plant_left(session: Session, frames: int) -> None:
    """Run left into wall, plant, Hi-Jump + right drift (kihunter mid-ledge class)."""
    session.unmorph()
    for _ in range(40):
        if session.state.samus_x <= 40:
            break
        session.step("LEFT", "B")
    session.hold(4)
    session.hold(8, "DOWN")
    for f in range(frames):
        if session.state.room_id != ZEELA:
            return
        if session.state.samus_y <= 320:
            return
        if f < 25:
            session.step("A")
        elif f < 40:
            session.step("A", "UP", "X")
        elif session.state.samus_y <= 350:
            session.step("RIGHT", "A", "B")
        else:
            session.step("RIGHT", "B")


STRATEGIES: dict[str, Callable[[Session, int], None]] = {
    "hj_under_hole": strategy_hj_under_hole,
    "bomb_jump_hole": strategy_bomb_jump_hole,
    "reverse_shot": strategy_reverse_shot,
    "spin_up": strategy_spin_up,
    "wall_plant_left": strategy_wall_plant_left,
}

# Candidate x bands under the second-drop hole (forward lands mid ~x=105).
HOLE_BANDS = (
    (90, 120),
    (100, 130),
    (115, 145),
    (140, 170),
    (160, 200),
    (200, 250),
    (250, 290),
)


def run_trial(
    source: Path,
    band: tuple[int, int],
    strategy: str,
    climb_frames: int,
) -> dict[str, object]:
    env = make_dev_env()
    try:
        boot_from_state(env, source)
        s = Session(env)
        start = pin(s.state)
        lo, hi = band
        mid = (lo + hi) // 2
        # Full setup: roll to floor-left near band, then align.
        ok = roll_left_to(s, x_max=max(mid + 40, hi + 20), y_min=350, budget=900)
        setup_after_roll = pin(s.state)
        if ok and s.state.room_id == ZEELA and not s.state.door_transition:
            align_x(s, lo, hi, budget=160)
        setup = pin(s.state)
        s.min_y = s.state.samus_y
        climb_start = s.frame
        if s.state.room_id == ZEELA and not (s.state.door_transition and s.state.samus_y > 250):
            STRATEGIES[strategy](s, climb_frames)
        end = pin(s.state)
        return {
            "band": list(band),
            "strategy": strategy,
            "start": start,
            "after_roll": setup_after_roll,
            "setup": setup,
            "min_y": s.min_y,
            "end": end,
            "climb_frames": s.frame - climb_start,
            "frames": s.frame,
            "mid_band": s.min_y <= 325,
            "upper_band": s.min_y <= 200,
            "wrong_door": end["room"] != f"0x{ZEELA:04X}",
            "trace_tail": s.trace[-8:],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--climb-frames", type=int, default=420)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only bands near forward mid-land and best strategies",
    )
    args = parser.parse_args()
    if not args.source.exists():
        raise SystemExit(f"missing source: {args.source}")

    bands = HOLE_BANDS[:3] if args.quick else HOLE_BANDS
    strategies = (
        ("bomb_jump_hole", "hj_under_hole", "reverse_shot")
        if args.quick
        else tuple(STRATEGIES)
    )

    trials: list[dict[str, object]] = []
    for band in bands:
        for name in strategies:
            trial = run_trial(args.source, band, name, args.climb_frames)
            trials.append(trial)
            flag = (
                "MID"
                if trial["mid_band"]
                else ("DOOR" if trial["wrong_door"] else "---")
            )
            print(
                f"band={band} strat={name:<16} setup_x={trial['setup']['x']:3} "
                f"min_y={trial['min_y']:3} end={trial['end']['x']}/{trial['end']['y']} "
                f"{flag} room={trial['end']['room']}"
            )

    mids = [t for t in trials if t["mid_band"]]
    best = min((t["min_y"] for t in trials), default=None)
    report = {
        "kind": "zeela_reverse_redesign_probe",
        "developmentOnly": True,
        "source": str(args.source.relative_to(ROOT)),
        "bestMinY": best,
        "midBandHits": len(mids),
        "trials": trials,
        "nonClaims": [
            "Diagnostic only — not pure-green",
            "Not continuous / no STATUS",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"best_min_y={best} mid_hits={len(mids)}")
    print(f"output={args.output.relative_to(ROOT)}")
    if mids:
        for t in sorted(mids, key=lambda r: r["min_y"]):
            print(
                f"  HIT band={t['band']} strat={t['strategy']} "
                f"min_y={t['min_y']} setup={t['setup']}"
            )


if __name__ == "__main__":
    main()
