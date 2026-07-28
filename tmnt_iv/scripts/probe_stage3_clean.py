"""Stage 3 Clean proof: heal=none, pizza-only HP recovery, Sewer clear.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage3_clean_track/``.

Copy of ``probe_stage2_clean.py`` for Sewer Surfin' (stage byte 2). See
``docs/CLEAN_PLAYBOOK.md``.

**Entry notes (2026-07-27):**

- ``Stage3`` / ``Boss3`` saves are last-life (lives=0) and die on the
  post-kill ``event=0x0B`` fade even after Rat King HP hits 0 — known
  checkpoint artifact (STATUS: dies ~444f into 0x0B). Prefer
  ``LiveHardStage3`` (lives=2) for full stage_advance proof.
- Spike props char ``0x1C``/``0x2C`` (−16) are in ``HAZARD_CHAR_IDS``;
  policy jumps when near (see ``SewerSpikeAvoid``).

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage3_clean --state LiveHardStage3
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --state Boss3
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --from-stage2-clear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env  # noqa: E402
from snes_oneshot.actions import idle_action  # noqa: E402
from snes_oneshot.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

# LiveHard (lives=2) is the faithful full-stage gate. Boss3 is combat-only
# (last-life fade is broken). Stage3 last-life checkpoint is secondary.
_SUITE_STATES: tuple[str, ...] = (
    "LiveHardStage3",
    "Boss3",
    "Stage3",
)


def _is_live_sewer(state: Any) -> bool:
    """True once Sewer Surfin' gameplay is live (not cutscene / despawn)."""
    return (
        state.mode.name == "PLAYING"
        and state.stage == 2
        and 20 <= state.health <= 96
        and 0 < state.player_x < 400
        and int(state.extras.get("event", 0)) >= 0x0A
    )


def run_clean_probe(
    *,
    state_name: str = "LiveHardStage3",
    max_frames: int = 25000,
    stop_stage_gt: int = 2,
    from_stage2_clear: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    configure_headless()
    start_label = "Stage2_Clear" if from_stage2_clear else state_name
    env = make_env(GAME, start_label, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    result = env.reset()
    if isinstance(result, tuple):
        pass

    in_play = not from_stage2_clear
    play_frame0 = 0

    start = parse_game_state(env.get_ram(), frame=0)
    prev_hp = start.health if 0 < start.health <= 0x60 else None
    prev_lives = start.lives
    damage = 0
    max_hit = 0
    min_hp = prev_hp
    pizza_heals: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    final = start
    outcome = "timeout"
    rat_king_entry_hp: int | None = None
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            if from_stage2_clear and not in_play:
                if _is_live_sewer(state):
                    in_play = True
                    play_frame0 = frame
                    prev_hp = state.health
                    prev_lives = state.lives
                    min_hp = state.health
                    policy = Stage1Policy()
                else:
                    tick = policy.tick(state)
                    action = (
                        tick.action.action
                        if tick.action is not None
                        else idle_action()
                    )
                    env.step(action)
                    continue

            if 0 < state.health <= 0x60:
                if prev_hp is not None and state.health < prev_hp:
                    hit = prev_hp - state.health
                    damage += hit
                    max_hit = max(max_hit, hit)
                    hits.append(
                        {
                            "frame": frame - play_frame0,
                            "hit": hit,
                            "hp": state.health,
                            "player_x": state.player_x,
                            "boss": state.boss_active,
                            "progress": int(
                                state.extras.get("progress_x", 0)
                            ),
                            "hazards": bool(state.extras.get("hazards")),
                        }
                    )
                elif prev_hp is not None and state.health > prev_hp:
                    pizza_heals.append(
                        {
                            "frame": frame - play_frame0,
                            "from_hp": prev_hp,
                            "to_hp": state.health,
                            "player_x": state.player_x,
                        }
                    )
                prev_hp = state.health
                if min_hp is None or state.health < min_hp:
                    min_hp = state.health

            if (
                state.boss_active
                and rat_king_entry_hp is None
                and 0 < state.health <= 0x60
            ):
                rat_king_entry_hp = state.health

            # Real failure: game over / death / life loss.
            if state.mode.name in {"GAME_OVER", "TITLE"} or state.player_dead:
                outcome = "life_loss"
                break
            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives
            # Real success: stage advances while still playing with HP.
            if (
                state.stage > stop_stage_gt
                and state.mode.name == "PLAYING"
                and 0 < state.health <= 0x60
            ):
                outcome = "stage_advance"
                break

            tick = policy.tick(state)
            action = (
                tick.action.action
                if tick.action is not None
                else idle_action()
            )
            reason = (
                tick.action.reason
                if tick.action is not None
                else tick.reason or "idle"
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            if action[8]:
                outcome = "forbidden_a"
                break
            env.step(action)
        else:
            outcome = "timeout"
    finally:
        env.close()

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:16]
    wave_dmg = sum(h["hit"] for h in hits if not h["boss"])
    boss_dmg = sum(h["hit"] for h in hits if h["boss"])
    label = "stage2_clear" if from_stage2_clear else state_name
    real_pizza = []
    for p in pizza_heals:
        if p.get("player_x", 0) <= 0:
            continue
        if (
            outcome == "life_loss"
            and p.get("frame", -1) >= (final.frame - play_frame0)
        ):
            continue
        real_pizza.append(p)
    return {
        "state": label,
        "heal_mode": "none",
        "assist": "pizza_only",
        "outcome": outcome,
        "success": outcome == "stage_advance",
        "frames": final.frame - play_frame0 if play_frame0 else final.frame,
        "total_frames": final.frame,
        "start_hp": start.health if not from_stage2_clear else 80,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "wave_damage": wave_dmg,
        "boss_damage": boss_dmg,
        "max_hit": max_hit,
        "pizza_heals": real_pizza,
        "pizza_heal_count": len(real_pizza),
        "rat_king_entry_hp": rat_king_entry_hp,
        "lives": f"{prev_lives}->{final.lives}",
        "boss_hp": (
            f"{int(start.extras.get('boss_hp', 0))}"
            f"->{int(final.extras.get('boss_hp', 0))}"
        ),
        "event": hex(int(final.extras.get("event", -1))),
        "end_stage": final.stage,
        "top_reasons": top,
        "hits": hits,
    }


def run_suite(*, max_frames: int = 25000) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + Stage2_Clear bridge."""
    results: list[dict[str, Any]] = []
    for name in _SUITE_STATES:
        try:
            report = run_clean_probe(state_name=name, max_frames=max_frames)
        except Exception as exc:  # noqa: BLE001 — suite continues
            report = {
                "state": name,
                "outcome": "error",
                "success": False,
                "error": str(exc),
            }
        results.append(report)
        print(
            f"  [{name}] outcome={report.get('outcome')} "
            f"dmg={report.get('damage_taken', '?')} "
            f"min_hp={report.get('min_hp', '?')} "
            f"entry={report.get('rat_king_entry_hp', '?')} "
            f"pizza={report.get('pizza_heal_count', '?')} "
            f"frames={report.get('frames', '?')}"
        )
    bridge = run_clean_probe(
        from_stage2_clear=True, max_frames=max_frames + 4000
    )
    results.append(bridge)
    print(
        f"  [stage2_clear] outcome={bridge.get('outcome')} "
        f"dmg={bridge.get('damage_taken', '?')} "
        f"min_hp={bridge.get('min_hp', '?')} "
        f"entry={bridge.get('rat_king_entry_hp', '?')} "
        f"pizza={bridge.get('pizza_heal_count', '?')} "
        f"frames={bridge.get('frames', '?')}"
    )
    ok = sum(1 for r in results if r.get("success"))
    return {
        "assist": "pizza_only",
        "suite_size": len(results),
        "passed": ok,
        "failed": len(results) - ok,
        "all_passed": ok == len(results),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry for Stage 3 Clean (pizza-only) probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="LiveHardStage3")
    parser.add_argument("--max-frames", type=int, default=25000)
    parser.add_argument("--stop-stage-gt", type=int, default=2)
    parser.add_argument(
        "--from-stage2-clear",
        action="store_true",
        help="Start from Stage2_Clear and measure Sewer only",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Multi-entry Clean suite (path RNG coverage)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (default under stage3_clean_track/)",
    )
    args = parser.parse_args(argv)
    out_dir = RECORDINGS_DIR / "stage3_clean_track"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.suite:
        report = run_suite(max_frames=args.max_frames)
        out = args.out or (out_dir / "clean_suite.json")
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"suite passed={report['passed']}/{report['suite_size']} "
            f"all_passed={report['all_passed']}"
        )
        print(f"report={out}")
        return 0 if report["all_passed"] else 1

    report = run_clean_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        stop_stage_gt=args.stop_stage_gt,
        from_stage2_clear=args.from_stage2_clear,
    )
    label = "stage2_clear" if args.from_stage2_clear else args.state.lower()
    out = args.out or (out_dir / f"clean_{label}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"dmg={report['damage_taken']} (wave={report['wave_damage']} "
        f"boss={report['boss_damage']}) min_hp={report['min_hp']} "
        f"pizza_heals={report['pizza_heal_count']} "
        f"rat_king_entry_hp={report['rat_king_entry_hp']} "
        f"max_hit={report['max_hit']}"
    )
    print(f"report={out}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
