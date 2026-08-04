"""Capture Raphael fight-ready save states for continuous-faithful grinding.

Existing FullHard* checkpoints are Leonardo (char 2). Continuous production
selects Raphael (char 8), so Leo probe KEEPs do not transfer.

Two modes:

1. **power-on** (default): production boot + policy; dumps Technodrome
   entry / duo / tank. Prefer stage-entry dumps — mid-tank Raph states can
   soft-lock when captured on an empty foot frame.

2. **from-state**: resume an existing Raph stage entry and dump Slash /
   later bosses (e.g. ``--from-state RaphDiagStage5`` → RaphFullHardBoss5).

Saves under custom_integrations/TMNTIV-Snes/:

- RaphFullHardStage4  — Technodrome stage entry (byte 3)
- RaphFullHardDuo     — first Tokka/Rahzar live frame
- RaphFullHardTank    — Shredder tank event 0x18 (first frame)
- RaphFullHardBoss5   — first Slash (0x50) live frame (stage 4)
- RaphFullHardStage5  — Prehistoric stage entry (byte 4)
- RaphFullHardStage7  — Wounded Knee stage entry (byte 6)
- RaphFullHardBoss9   — Super Shredder form 1 (0x52, stage ≥ 8)

Usage:
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.capture_raph_states
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.capture_raph_states \\
      --from-state RaphDiagStage5 --max-frames 40000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state  # noqa: E402
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.ram_state import GameMode  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402
from tmnt_iv.scripts.record_full_hard_run import (  # noqa: E402
    _BOOT_ACTIONS,
    _EMERGENCY_HP_RESTORE,
    _EMERGENCY_HP_THRESHOLD,
    _boot_action,
)

_RAPH_CHAR = 8
_SLASH = 0x50
_TOKKA = 0x48
_RAHZAR = 0xA0
_SHREDDER_F1 = 0x52
_TANK_EVENT = 0x18


@dataclass
class CapturePoint:
    """One named dump request."""

    name: str
    description: str
    saved: bool = False
    frame: int | None = None
    stage: int | None = None
    event: int | None = None
    health: int | None = None
    lives: int | None = None
    path: str | None = None


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def _has_kind(state: Any, kind: int) -> bool:
    return any(e.kind == kind and e.health > 0 for e in state.living_enemies)


def run_capture(
    *,
    max_frames: int = 220_000,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Power-on → capture Raph fight states; stop after Super Shredder f1."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    _reset(env)

    points = {
        "stage4": CapturePoint(
            "RaphFullHardStage4", "Technodrome stage entry"
        ),
        "duo": CapturePoint("RaphFullHardDuo", "Tokka/Rahzar first live"),
        "tank": CapturePoint("RaphFullHardTank", "Shredder tank event 0x18"),
        "stage5": CapturePoint(
            "RaphFullHardStage5", "Prehistoric stage entry"
        ),
        "slash": CapturePoint("RaphFullHardBoss5", "Slash first live"),
        "stage7": CapturePoint(
            "RaphFullHardStage7", "Wounded Knee stage entry"
        ),
        "shredder": CapturePoint(
            "RaphFullHardBoss9", "Super Shredder form 1 first live"
        ),
    }
    prev_stage = -1
    prev_health: int | None = None
    heals = 0
    damage = 0
    char_seen: int | None = None
    done_reason = "max_frames"

    def dump(key: str, state: Any, frame: int) -> None:
        point = points[key]
        if point.saved:
            return
        path = save_state(env, GAME_DIR, GAME, point.name)
        point.saved = True
        point.frame = frame
        point.stage = state.stage
        point.event = int(state.extras.get("event", -1))
        point.health = state.health
        point.lives = state.lives
        point.path = str(path)
        print(
            f"saved {point.name} @ f={frame} stage={state.stage} "
            f"event={hex(point.event)} hp={state.health} lives={state.lives} "
            f"char={state.extras.get('char_id')} → {path}",
            flush=True,
        )

    try:
        for frame in range(0, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            menu = int(state.extras.get("menu", -1))
            event = int(state.extras.get("event", -1))
            char_id = int(state.extras.get("char_id", -1))
            if char_id == _RAPH_CHAR:
                char_seen = char_id

            active = (
                menu == 6
                and state.player_x > 0
                and state.stage <= 9
            )
            if (
                active
                and prev_health is not None
                and 0 <= state.health <= 0x60
                and prev_health <= 0x60
                and state.health < prev_health
            ):
                damage += prev_health - max(0, state.health)
            if active and 0 < state.health <= 0x60:
                if state.health <= _EMERGENCY_HP_THRESHOLD:
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    prev_health = state.health
                else:
                    prev_health = state.health
            elif active and state.health == 0:
                env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                heals += 1
                state = parse_game_state(env.get_ram(), frame=frame)
                prev_health = state.health

            # Stage entry dumps (same moment as full-run splits).
            if (
                state.stage != prev_stage
                and state.stage >= 0
                and menu == 6
                and state.player_x > 0
            ):
                if state.stage == 3:
                    dump("stage4", state, frame)
                elif state.stage == 4:
                    dump("stage5", state, frame)
                elif state.stage == 6:
                    dump("stage7", state, frame)
                prev_stage = state.stage
                policy.reset()
                prev_health = (
                    state.health if 0 < state.health <= 0x60 else None
                )

            if char_seen == _RAPH_CHAR:
                # Gate by stage — entity kind IDs are reused outside the
                # intended fights (e.g. 0x52 mid-Technodrome is not form 1).
                if (
                    not points["duo"].saved
                    and state.stage == 3
                    and (_has_kind(state, _TOKKA) or _has_kind(state, _RAHZAR))
                ):
                    dump("duo", state, frame)
                if (
                    not points["tank"].saved
                    and state.stage == 3
                    and event == _TANK_EVENT
                ):
                    dump("tank", state, frame)
                if (
                    not points["slash"].saved
                    and state.stage == 4
                    and _has_kind(state, _SLASH)
                ):
                    dump("slash", state, frame)
                if (
                    not points["shredder"].saved
                    and state.stage >= 8
                    and _has_kind(state, _SHREDDER_F1)
                ):
                    dump("shredder", state, frame)
                    done_reason = "shredder_captured"
                    break

            if frame <= max(_BOOT_ACTIONS):
                action = _boot_action(frame)
            elif state.player_x == 0 or state.mode in {
                GameMode.CUTSCENE,
                GameMode.CONTINUE,
            }:
                action = idle_action()
            else:
                tick = policy.tick(state)
                action = (
                    tick.action.action
                    if tick.action is not None
                    else idle_action()
                )
            env.step(action)

            if frame and frame % 10000 == 0:
                print(
                    f"frame {frame} stage={state.stage} event={hex(event)} "
                    f"dmg={damage} heals={heals} "
                    f"saved={[k for k, p in points.items() if p.saved]}",
                    flush=True,
                )
    finally:
        env.close()

    report = {
        "char_seen": char_seen,
        "done_reason": done_reason,
        "damage_taken": damage,
        "heals": heals,
        "captures": {k: asdict(v) for k, v in points.items()},
    }
    out = report_path or (
        RECORDINGS_DIR / "raph_state_capture" / "capture_report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=220_000)
    parser.add_argument(
        "--report",
        type=Path,
        default=RECORDINGS_DIR / "raph_state_capture" / "capture_report.json",
    )
    args = parser.parse_args(argv)
    report = run_capture(max_frames=args.max_frames, report_path=args.report)
    saved = sum(1 for p in report["captures"].values() if p["saved"])
    if report.get("char_seen") != _RAPH_CHAR:
        print("ERROR: Raphael (char 8) never appeared", file=sys.stderr)
        return 2
    if saved < 4:
        print(f"ERROR: only {saved} captures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
