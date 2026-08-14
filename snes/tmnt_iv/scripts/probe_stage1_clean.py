"""Stage 1 Clean proof: heal=none, pizza-only HP recovery, stage advance.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage1_clean_track/``.

**Template for later-stage Clean suites** — copy this module pattern
(heal=none, multi-entry, pizza counts) per ``docs/CLEAN_PLAYBOOK.md``.
Do not relearn Stage 1 traps; extend stage allowlists carefully.

Path-RNG suite (checkpoint + Baxter + power-on):

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage1_clean --state Stage1
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --power-on
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.menus import boot_to_stage1_script  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

# Path-RNG suite: full-stage entries + Baxter. Mid-wave Clear_w* states are
# historical wave locks with different spawn tables; they are not required
# for Clean proof (power-on + Stage1 already stress full Big Apple).
_SUITE_STATES: tuple[str, ...] = (
    "Stage1",
    "Stage1_BeforeBoss",
    "Boss",
)

def _is_live_stage1(state: Any) -> bool:
    """True once Big Apple gameplay is live (not menus / despawn X)."""
    return (
        state.mode.name == "PLAYING"
        and state.stage == 0
        and 40 <= state.health <= 96
        and 0 < state.player_x < 400
        and int(state.extras.get("event", 0)) >= 0x0A
    )

def run_clean_probe(
    *,
    state_name: str = "Stage1",
    max_frames: int = 20000,
    stop_stage_gt: int = 0,
    power_on: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    configure_headless()
    start_label = "NONE" if power_on else state_name
    env = make_env(GAME, start_label, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    reset_obs(env)

    boot_actions = (
        [fa.action for fa in boot_to_stage1_script()] if power_on else []
    )
    boot_i = 0
    in_play = not power_on
    play_frame0 = 0

    start = parse_game_state(env.get_ram(), frame=0)
    prev_hp = start.health if 0 < start.health <= 0x60 else None
    prev_lives = start.lives if start.lives > 0 else 2
    damage = 0
    max_hit = 0
    min_hp = prev_hp
    pizza_heals: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    final = start
    outcome = "timeout"
    baxter_entry_hp: int | None = None
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            if power_on and not in_play:
                if _is_live_stage1(state):
                    in_play = True
                    play_frame0 = frame
                    prev_hp = state.health
                    prev_lives = state.lives
                    min_hp = state.health
                    policy = Stage1Policy()  # fresh policy at true play start
                elif boot_i < len(boot_actions):
                    env.step(boot_actions[boot_i])
                    boot_i += 1
                    continue
                else:
                    env.step(
                        buttons("START") if frame % 40 == 0 else idle_action()
                    )
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
                and baxter_entry_hp is None
                and 0 < state.health <= 0x60
            ):
                baxter_entry_hp = state.health

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
    label = "power_on" if power_on else state_name
    return {
        "state": label,
        "heal_mode": "none",
        "assist": "pizza_only",
        "outcome": outcome,
        "success": outcome == "stage_advance",
        "frames": final.frame - play_frame0 if play_frame0 else final.frame,
        "total_frames": final.frame,
        "start_hp": start.health if not power_on else 80,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "wave_damage": wave_dmg,
        "boss_damage": boss_dmg,
        "max_hit": max_hit,
        "pizza_heals": pizza_heals,
        "pizza_heal_count": len(
            [p for p in pizza_heals if p.get("player_x", 0) > 0]
        ),
        "baxter_entry_hp": baxter_entry_hp,
        "lives": f"{prev_lives}->{final.lives}",
        "boss_hp": (
            f"{int(start.extras.get('boss_hp', 0))}"
            f"->{int(final.extras.get('boss_hp', 0))}"
        ),
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": top,
        "hits": hits,
    }

def run_suite(*, max_frames: int = 22000) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + power-on."""
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
            f"frames={report.get('frames', '?')}"
        )
    power = run_clean_probe(power_on=True, max_frames=max_frames + 4000)
    results.append(power)
    print(
        f"  [power_on] outcome={power.get('outcome')} "
        f"dmg={power.get('damage_taken', '?')} "
        f"min_hp={power.get('min_hp', '?')} "
        f"frames={power.get('frames', '?')}"
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
    """CLI entry for Stage 1 Clean (pizza-only) probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage1")
    parser.add_argument("--max-frames", type=int, default=22000)
    parser.add_argument("--stop-stage-gt", type=int, default=0)
    parser.add_argument(
        "--power-on",
        action="store_true",
        help="Boot from NONE through menus, then Clean Stage 1",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Multi-entry + power-on Clean suite (path RNG coverage)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (default under stage1_clean_track/)",
    )
    args = parser.parse_args(argv)
    out_dir = RECORDINGS_DIR / "stage1_clean_track"
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
        power_on=args.power_on,
    )
    label = "power_on" if args.power_on else args.state.lower()
    out = args.out or (out_dir / f"clean_{label}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"dmg={report['damage_taken']} (wave={report['wave_damage']} "
        f"boss={report['boss_damage']}) min_hp={report['min_hp']} "
        f"pizza_heals={report['pizza_heal_count']} "
        f"baxter_entry_hp={report['baxter_entry_hp']} "
        f"max_hit={report['max_hit']}"
    )
    print(f"report={out}")
    top = report.get("top_reasons") or []
    if top:
        print("reasons: " + ", ".join(f"{k}={v}" for k, v in top[:10]))
    return 0 if report.get("success") else 1

if __name__ == "__main__":
    raise SystemExit(main())
