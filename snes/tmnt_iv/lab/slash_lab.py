"""Slash (char 0x50) attack-pattern lab from FullHardBoss5.

Standalone reference controllers for the implementer to port into policy.
Does **not** import or mutate ``tmnt_iv.policy`` — keep that free for the
production agent.

Examples::

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.lab.slash_lab --list

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.lab.slash_lab \\
        --pattern classic_thrash --heal emergency

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.lab.slash_lab --all --heal full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from retro_harness.env import make_env, reset_obs  # noqa: E402
from tmnt_iv.assist import apply_emergency_hp  # noqa: E402
from tmnt_iv.observe import HpDelta, living_hp  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402
from tmnt_iv.lab.slash_patterns import (  # noqa: E402
    ClassicThrash,
    HybridStickAndMove,
    HybridWhiplash,
    IframeAggressive,
    JumpSlashPunish,
    ProductionSlash,
    SlashPattern,
    StatusAware,
    ThrashFleeSpin,
    VulnReactive,
    _assert_no_a,
    _slash_enemy,
)

_DEFAULT_STATE = "FullHardBoss5"
_MAX_FRAMES_DEFAULT = 35_000
_FULL_HEAL_HP = 96

PATTERNS: dict[str, Callable[[], SlashPattern]] = {
    ProductionSlash.name: ProductionSlash,
    ClassicThrash.name: ClassicThrash,
    ThrashFleeSpin.name: ThrashFleeSpin,
    StatusAware.name: StatusAware,
    IframeAggressive.name: IframeAggressive,
    JumpSlashPunish.name: JumpSlashPunish,
    HybridWhiplash.name: HybridWhiplash,
    HybridStickAndMove.name: HybridStickAndMove,
    VulnReactive.name: VulnReactive,
}

# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    pattern: str
    heal_mode: str
    state: str
    outcome: str
    frames: int
    boss_hp_start: int
    boss_hp_end: int
    dmg_taken: int
    heals: int
    min_hp: int | None
    end_hp: int
    end_stage: int
    event: str
    elapsed_s: float
    top_reasons: list[tuple[str, int]] = field(default_factory=list)
    description: str = ""

    @property
    def boss_damage(self) -> int:
        return max(0, self.boss_hp_start - self.boss_hp_end)

    @property
    def dps(self) -> float:
        if self.frames <= 0:
            return 0.0
        return self.boss_damage * 60.0 / self.frames

    @property
    def dmg_per_boss_hp(self) -> float:
        bd = self.boss_damage
        if bd <= 0:
            return float("inf") if self.dmg_taken else 0.0
        return self.dmg_taken / bd


def run_trial(
    *,
    pattern: SlashPattern,
    state_name: str = _DEFAULT_STATE,
    max_frames: int = _MAX_FRAMES_DEFAULT,
    heal_mode: str = "emergency",
    stop_stage_gt: int = 4,
) -> TrialResult:
    """Run one controller from ``state_name`` and collect metrics.

    heal_mode:
      - ``none``: no HP writes (pure survival stress)
      - ``emergency``: restore to 80 when HP <= 16 (production-like)
      - ``full``: restore to 96 on any damage (pure DPS ranking)
    """
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    pattern.reset()
    reset_obs(env)
    start = parse_game_state(env.get_ram(), frame=0)
    meter = HpDelta.start(start.health, count_zero=True)
    prev_lives = start.lives
    heals = 0
    reasons: dict[str, int] = {}
    boss_hp_start = int(start.extras.get("boss_hp", 0))
    # Also track Slash entity HP directly (boss_hp extras may drop when
    # status filters flicker).
    slash0 = _slash_enemy(start)
    if slash0 is not None:
        boss_hp_start = max(boss_hp_start, slash0.health)
    final = start
    outcome = "timeout"
    t0 = time.perf_counter()
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            # Natural damage from HP drops (before any heal write).
            meter.note(state.health)

            # Heal assists.
            if heal_mode == "full" and living_hp(state.health):
                if state.health < _FULL_HEAL_HP:
                    env.set_value("player_hp", _FULL_HEAL_HP)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    meter.note(state.health)
            elif heal_mode == "emergency":
                if apply_emergency_hp(env, state.health):
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    meter.note(state.health)

            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives

            if state.stage > stop_stage_gt:
                outcome = "stage_advance"
                break

            slash = _slash_enemy(state)
            if (
                start.boss_active
                and not state.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19}
            ) or (
                slash is None
                and start.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19, 0x04}
            ):
                outcome = "boss_down"
            if outcome == "boss_down" and frame % 30 == 0:
                if state.stage > start.stage or int(
                    state.extras.get("event", 0)
                ) in {0x19, 0x04, 0x0B}:
                    if slash is None or slash.health <= 0:
                        outcome = "cleared"
                        break

            if state.mode.name in {"CONTINUE", "GAME_OVER"}:
                if heal_mode != "none":
                    apply_emergency_hp(env, 0)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    meter.note(state.health)
                else:
                    outcome = "life_loss"
                    break

            fa = pattern.next(state)
            action = fa.action if fa is not None else idle_action()
            reason = fa.reason if fa is not None else "idle"
            reasons[reason] = reasons.get(reason, 0) + 1
            _assert_no_a(action)
            env.step(action)
        else:
            if outcome not in {"cleared", "boss_down"}:
                outcome = "timeout"
    finally:
        env.close()

    elapsed = time.perf_counter() - t0
    slash_f = _slash_enemy(final)
    boss_hp_end = (
        slash_f.health
        if slash_f is not None
        else int(final.extras.get("boss_hp", 0))
    )
    if outcome in {"cleared", "stage_advance", "boss_down"} and slash_f is None:
        boss_hp_end = 0

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:10]
    return TrialResult(
        pattern=pattern.name,
        heal_mode=heal_mode,
        state=state_name,
        outcome=outcome,
        frames=final.frame,
        boss_hp_start=boss_hp_start,
        boss_hp_end=boss_hp_end,
        dmg_taken=meter.damage,
        heals=heals,
        min_hp=meter.min_hp,
        end_hp=final.health,
        end_stage=final.stage,
        event=hex(int(final.extras.get("event", -1))),
        elapsed_s=round(elapsed, 2),
        top_reasons=top,
        description=pattern.description,
    )


def _score(r: TrialResult) -> tuple:
    """Rank key: clear first, then less damage, fewer frames, fewer heals."""
    cleared = r.outcome in {"cleared", "stage_advance", "boss_down"}
    # Prefer full clear / stage advance over mere boss_down
    tier = {
        "stage_advance": 3,
        "cleared": 3,
        "boss_down": 2,
        "timeout": 1,
        "life_loss": 0,
        "forbidden_a": -1,
    }.get(r.outcome, 0)
    return (
        tier,
        r.boss_damage,  # more boss damage better when not cleared
        -r.dmg_taken,
        -r.frames,
        -r.heals,
    )


def _print_result(r: TrialResult) -> None:
    print(
        f"[{r.pattern}/{r.heal_mode}] outcome={r.outcome} "
        f"frames={r.frames} boss={r.boss_hp_start}->{r.boss_hp_end} "
        f"dmg_taken={r.dmg_taken} heals={r.heals} "
        f"dps={r.dps:.2f} dmg/bossHP={r.dmg_per_boss_hp:.2f} "
        f"min_hp={r.min_hp} event={r.event} ({r.elapsed_s}s)"
    )
    if r.top_reasons:
        brief = ", ".join(f"{k}:{v}" for k, v in r.top_reasons[:5])
        print(f"  reasons: {brief}")


def _markdown_table(results: list[TrialResult]) -> str:
    lines = [
        "| pattern | heal | frames | boss_hp | dmg_taken | heals | outcome | dps | dmg/bhp |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in sorted(results, key=_score, reverse=True):
        boss = f"{r.boss_hp_start}->{r.boss_hp_end}"
        if r.outcome in {"cleared", "stage_advance"} and r.boss_hp_end == 0:
            boss = f"{r.boss_hp_start}->0 (clear)"
        lines.append(
            f"| `{r.pattern}` | {r.heal_mode} | {r.frames} | {boss} | "
            f"{r.dmg_taken} | {r.heals} | {r.outcome} | {r.dps:.2f} | "
            f"{r.dmg_per_boss_hp:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="pattern name (repeatable). default: all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every registered pattern",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list patterns and exit",
    )
    parser.add_argument(
        "--heal",
        choices=("none", "emergency", "full"),
        action="append",
        dest="heals",
        default=None,
        help="heal mode (repeatable). default: emergency + full",
    )
    parser.add_argument("--state", default=_DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=_MAX_FRAMES_DEFAULT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write full JSON results",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="optional path to write markdown table only",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in PATTERNS.items():
            inst = cls()
            print(f"  {name:22s}  {inst.description}")
        return 0

    names = args.patterns
    if args.all or not names:
        names = list(PATTERNS.keys())
    for n in names:
        if n not in PATTERNS:
            print(f"unknown pattern: {n}", file=sys.stderr)
            print(f"choose from: {', '.join(PATTERNS)}", file=sys.stderr)
            return 2

    heals = args.heals or ["emergency", "full"]
    results: list[TrialResult] = []
    for heal in heals:
        for name in names:
            print(f"\n=== RUN {name} heal={heal} ===", flush=True)
            ctrl = PATTERNS[name]()
            result = run_trial(
                pattern=ctrl,
                state_name=args.state,
                max_frames=args.max_frames,
                heal_mode=heal,
            )
            _print_result(result)
            results.append(result)

    print("\n======== SUMMARY ========")
    print(_markdown_table(results))

    ranked = sorted(results, key=_score, reverse=True)
    if ranked:
        best = ranked[0]
        print(
            f"\nWINNER: {best.pattern} ({best.heal_mode}) "
            f"outcome={best.outcome} frames={best.frames} "
            f"dmg_taken={best.dmg_taken} heals={best.heals} "
            f"boss={best.boss_hp_start}->{best.boss_hp_end}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_markdown_table(results) + "\n")
        print(f"wrote {args.md_out}")

    # Non-zero only if every trial life-lost / forbidden.
    if results and all(
        r.outcome in {"life_loss", "forbidden_a"} for r in results
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
