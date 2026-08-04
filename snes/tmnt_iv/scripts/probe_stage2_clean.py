"""Stage 2 Clean proof: heal=none, pizza-only HP recovery, Alleycat clear.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage2_clean_track/``.

Copy of ``probe_stage1_clean.py`` for Alleycat Blues (stage byte 1). See
``docs/CLEAN_PLAYBOOK.md``.

Status (2026-07-27): early/mid waves still fail heal=none; Metalhead and
pre-boss (w17) already clear pizza-only. Suite tracks both.

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Boss2
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --from-stage1-clear
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
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

# Full-stage checkpoint + boss + late mid. Early mids optional.
_SUITE_STATES: tuple[str, ...] = (
    "Stage2",
    "Stage2_Clear_w17_cam27882",
    "Boss2",
)


def _is_live_alleycat(state: Any) -> bool:
    """True once Alleycat gameplay is live (not transition / despawn)."""
    return (
        state.mode.name == "PLAYING"
        and state.stage == 1
        and 20 <= state.health <= 96
        and 0 < state.player_x < 400
        and int(state.extras.get("event", 0)) >= 0x0A
    )


def run_clean_probe(
    *,
    state_name: str = "Stage2",
    max_frames: int = 25000,
    stop_stage_gt: int = 1,
    from_stage1_clear: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    configure_headless()
    start_label = "Stage1_Clear" if from_stage1_clear else state_name
    env = make_env(GAME, start_label, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    result = env.reset()
    if isinstance(result, tuple):
        pass

    in_play = not from_stage1_clear
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
    metalhead_entry_hp: int | None = None
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            if from_stage1_clear and not in_play:
                if _is_live_alleycat(state):
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
                and metalhead_entry_hp is None
                and 0 < state.health <= 0x60
            ):
                metalhead_entry_hp = state.health

            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives
            if state.stage > stop_stage_gt:
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
    label = "stage1_clear" if from_stage1_clear else state_name
    # Respawn after life_loss restores HP to 80 and looks like pizza — drop
    # any heal on the final frame when the outcome is life_loss.
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
        "start_hp": start.health if not from_stage1_clear else 80,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "wave_damage": wave_dmg,
        "boss_damage": boss_dmg,
        "max_hit": max_hit,
        "pizza_heals": real_pizza,
        "pizza_heal_count": len(real_pizza),
        "metalhead_entry_hp": metalhead_entry_hp,
        "lives": f"{prev_lives}->{final.lives}",
        "boss_hp": (
            f"{int(start.extras.get('boss_hp', 0))}"
            f"->{int(final.extras.get('boss_hp', 0))}"
        ),
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": top,
        "hits": hits,
    }


def run_suite(*, max_frames: int = 25000) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + Stage1_Clear bridge."""
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
            f"pizza={report.get('pizza_heal_count', '?')} "
            f"frames={report.get('frames', '?')}"
        )
    bridge = run_clean_probe(from_stage1_clear=True, max_frames=max_frames + 4000)
    results.append(bridge)
    print(
        f"  [stage1_clear] outcome={bridge.get('outcome')} "
        f"dmg={bridge.get('damage_taken', '?')} "
        f"min_hp={bridge.get('min_hp', '?')} "
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
    """CLI entry for Stage 2 Clean (pizza-only) probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage2")
    parser.add_argument("--max-frames", type=int, default=25000)
    parser.add_argument("--stop-stage-gt", type=int, default=1)
    parser.add_argument(
        "--from-stage1-clear",
        action="store_true",
        help="Start from Stage1_Clear and measure Alleycat only",
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
        help="JSON report path (default under stage2_clean_track/)",
    )
    args = parser.parse_args(argv)
    out_dir = RECORDINGS_DIR / "stage2_clean_track"
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
        from_stage1_clear=args.from_stage1_clear,
    )
    label = "stage1_clear" if args.from_stage1_clear else args.state.lower()
    out = args.out or (out_dir / f"clean_{label}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"dmg={report['damage_taken']} (wave={report['wave_damage']} "
        f"boss={report['boss_damage']}) min_hp={report['min_hp']} "
        f"pizza_heals={report['pizza_heal_count']} "
        f"metalhead_entry_hp={report['metalhead_entry_hp']} "
        f"max_hit={report['max_hit']}"
    )
    print(f"report={out}")
    top = report.get("top_reasons") or []
    if top:
        print("reasons: " + ", ".join(f"{k}={v}" for k, v in top[:10]))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
