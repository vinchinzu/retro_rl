"""Shared Clean (heal=none, pizza-only) probe loop for TMNT IV stages 1–3.

``scripts/probe_stage{N}_clean.py`` used to copy the same boot / HP-delta /
forbidden-A loop. Import ``run_clean_probe`` / ``CleanProbeSpec`` instead.

Stage byte 0 = Big Apple, 1 = Alleycat, 2 = Sewer. No emergency HP writes
and no form-2 iframe writes. Path-RNG suite = checkpoints + extra entry
(power-on or previous-clear bridge).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from tmnt_iv.menus import boot_to_stage1_script
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state

ExtraEntry = Literal["power_on", "from_stage1_clear", "from_stage2_clear"]


def _is_live(
    state: Any, *, stage_byte: int, min_hp: int
) -> bool:
    """True once the stage's gameplay is live (not menus / despawn X)."""
    return (
        state.mode.name == "PLAYING"
        and state.stage == stage_byte
        and min_hp <= state.health <= 96
        and 0 < state.player_x < 400
        and int(state.extras.get("event", 0)) >= 0x0A
    )


def is_live_big_apple(state: Any) -> bool:
    """True once Big Apple gameplay is live (not menus / despawn X)."""
    return _is_live(state, stage_byte=0, min_hp=40)


def is_live_alleycat(state: Any) -> bool:
    """True once Alleycat gameplay is live (not transition / despawn)."""
    return _is_live(state, stage_byte=1, min_hp=20)


def is_live_sewer(state: Any) -> bool:
    """True once Sewer Surfin' gameplay is live (not cutscene / despawn)."""
    return _is_live(state, stage_byte=2, min_hp=20)


@dataclass(frozen=True)
class CleanProbeSpec:
    """Per-stage names and knobs for the shared Clean (heal=none) loop."""

    stage_byte: int
    suite_states: tuple[str, ...]
    default_state: str
    stop_stage_gt: int
    default_max_frames: int
    cli_max_frames: int
    evidence_dir: str
    is_live: Callable[[Any], bool]
    extra_entry: ExtraEntry
    extra_start_state: str
    extra_label: str
    extra_help: str
    boss_entry_hp_key: str
    min_live_hp: int = 20
    lives_fallback: int | None = None
    filter_respawn_pizza: bool = True
    hit_hazards: bool = False
    hit_progress: bool = False
    # Stage 3 last-life checkpoints die on the post-kill 0x0B fade.
    detect_game_over: bool = False
    strict_advance: bool = False
    include_end_stage: bool = False
    suite_help: str = "Multi-entry Clean suite (path RNG coverage)"
    suite_print_pizza: bool = False
    suite_print_entry: bool = False
    cli_print_reasons: bool = True


STAGE1_CLEAN = CleanProbeSpec(
    stage_byte=0,
    suite_states=("Stage1", "Stage1_BeforeBoss", "Boss"),
    default_state="Stage1",
    stop_stage_gt=0,
    default_max_frames=20000,
    cli_max_frames=22000,
    evidence_dir="stage1_clean_track",
    is_live=is_live_big_apple,
    extra_entry="power_on",
    extra_start_state="NONE",
    extra_label="power_on",
    extra_help="Boot from NONE through menus, then Clean Stage 1",
    boss_entry_hp_key="baxter_entry_hp",
    min_live_hp=40,
    lives_fallback=2,
    filter_respawn_pizza=False,
    hit_hazards=True,
    suite_help="Multi-entry + power-on Clean suite (path RNG coverage)",
)

STAGE2_CLEAN = CleanProbeSpec(
    stage_byte=1,
    suite_states=("Stage2", "Stage2_Clear_w17_cam27882", "Boss2"),
    default_state="Stage2",
    stop_stage_gt=1,
    default_max_frames=25000,
    cli_max_frames=25000,
    evidence_dir="stage2_clean_track",
    is_live=is_live_alleycat,
    extra_entry="from_stage1_clear",
    extra_start_state="Stage1_Clear",
    extra_label="stage1_clear",
    extra_help="Start from Stage1_Clear and measure Alleycat only",
    boss_entry_hp_key="metalhead_entry_hp",
    hit_progress=True,
    suite_print_pizza=True,
)

STAGE3_CLEAN = CleanProbeSpec(
    stage_byte=2,
    suite_states=("LiveHardStage3", "Boss3", "Stage3"),
    default_state="LiveHardStage3",
    stop_stage_gt=2,
    default_max_frames=25000,
    cli_max_frames=25000,
    evidence_dir="stage3_clean_track",
    is_live=is_live_sewer,
    extra_entry="from_stage2_clear",
    extra_start_state="Stage2_Clear",
    extra_label="stage2_clear",
    extra_help="Start from Stage2_Clear and measure Sewer only",
    boss_entry_hp_key="rat_king_entry_hp",
    hit_hazards=True,
    hit_progress=True,
    detect_game_over=True,
    strict_advance=True,
    include_end_stage=True,
    suite_print_pizza=True,
    suite_print_entry=True,
    cli_print_reasons=False,
)

CLEAN_SPECS: dict[int, CleanProbeSpec] = {
    0: STAGE1_CLEAN,
    1: STAGE2_CLEAN,
    2: STAGE3_CLEAN,
}


def _hp_ok(health: int) -> bool:
    return 0 < health <= 0x60


def _entry_flags(
    *,
    power_on: bool,
    from_stage1_clear: bool,
    from_stage2_clear: bool,
) -> tuple[bool, str | None, str | None]:
    """Return (waiting, env start label, report state label)."""
    if power_on:
        return True, "NONE", "power_on"
    if from_stage1_clear:
        return True, "Stage1_Clear", "stage1_clear"
    if from_stage2_clear:
        return True, "Stage2_Clear", "stage2_clear"
    return False, None, None


def _policy_action(policy: Stage1Policy, state: Any) -> tuple[Any, str]:
    tick = policy.tick(state)
    if tick.action is not None:
        return tick.action.action, tick.action.reason
    return idle_action(), tick.reason or "idle"


def _filter_pizza(
    pizza_heals: list[dict[str, Any]],
    *,
    outcome: str,
    play_frames: int,
    drop_respawn: bool,
) -> list[dict[str, Any]]:
    real: list[dict[str, Any]] = []
    for pizza in pizza_heals:
        if pizza.get("player_x", 0) <= 0:
            continue
        if (
            drop_respawn
            and outcome == "life_loss"
            and pizza.get("frame", -1) >= play_frames
        ):
            continue
        real.append(pizza)
    return real


def _hit_row(
    spec: CleanProbeSpec, state: Any, *, frame: int, hit: int
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "hit": hit,
        "hp": state.health,
        "player_x": state.player_x,
        "boss": state.boss_active,
    }
    if spec.hit_progress:
        row["progress"] = int(state.extras.get("progress_x", 0))
    if spec.hit_hazards:
        row["hazards"] = bool(state.extras.get("hazards"))
    return row


def run_clean_probe(
    spec: CleanProbeSpec,
    *,
    state_name: str | None = None,
    max_frames: int | None = None,
    stop_stage_gt: int | None = None,
    power_on: bool = False,
    from_stage1_clear: bool = False,
    from_stage2_clear: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    configure_headless()
    name = spec.default_state if state_name is None else state_name
    frames = spec.default_max_frames if max_frames is None else max_frames
    stop_gt = spec.stop_stage_gt if stop_stage_gt is None else stop_stage_gt
    waiting, extra_start, extra_label = _entry_flags(
        power_on=power_on,
        from_stage1_clear=from_stage1_clear,
        from_stage2_clear=from_stage2_clear,
    )
    start_label = extra_start if extra_start is not None else name
    env = make_env(GAME, start_label, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    reset_obs(env)

    boot_actions = (
        [fa.action for fa in boot_to_stage1_script()] if power_on else []
    )
    boot_i = 0
    in_play = not waiting
    play_frame0 = 0

    start = parse_game_state(env.get_ram(), frame=0)
    prev_hp = start.health if _hp_ok(start.health) else None
    if spec.lives_fallback is not None:
        prev_lives = start.lives if start.lives > 0 else spec.lives_fallback
    else:
        prev_lives = start.lives
    damage = 0
    max_hit = 0
    min_hp = prev_hp
    pizza_heals: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    final = start
    outcome = "timeout"
    boss_entry_hp: int | None = None
    try:
        for frame in range(1, frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state

            if waiting and not in_play:
                if spec.is_live(state):
                    in_play = True
                    play_frame0 = frame
                    prev_hp = state.health
                    prev_lives = state.lives
                    min_hp = state.health
                    policy = Stage1Policy()
                elif boot_i < len(boot_actions):
                    env.step(boot_actions[boot_i])
                    boot_i += 1
                    continue
                elif power_on:
                    env.step(
                        buttons("START") if frame % 40 == 0 else idle_action()
                    )
                    continue
                else:
                    action, _reason = _policy_action(policy, state)
                    env.step(action)
                    continue

            if _hp_ok(state.health):
                if prev_hp is not None and state.health < prev_hp:
                    hit = prev_hp - state.health
                    damage += hit
                    max_hit = max(max_hit, hit)
                    hits.append(
                        _hit_row(
                            spec,
                            state,
                            frame=frame - play_frame0,
                            hit=hit,
                        )
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
                and boss_entry_hp is None
                and _hp_ok(state.health)
            ):
                boss_entry_hp = state.health

            if spec.detect_game_over and (
                state.mode.name in {"GAME_OVER", "TITLE"} or state.player_dead
            ):
                outcome = "life_loss"
                break
            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives
            if state.stage > stop_gt:
                if (not spec.strict_advance) or (
                    state.mode.name == "PLAYING" and _hp_ok(state.health)
                ):
                    outcome = "stage_advance"
                    break

            action, reason = _policy_action(policy, state)
            reasons[reason] = reasons.get(reason, 0) + 1
            if action[8]:
                outcome = "forbidden_a"
                break
            env.step(action)
        else:
            outcome = "timeout"
    finally:
        env.close()

    play_frames = final.frame - play_frame0 if play_frame0 else final.frame
    real_pizza = _filter_pizza(
        pizza_heals,
        outcome=outcome,
        play_frames=play_frames,
        drop_respawn=spec.filter_respawn_pizza,
    )
    report: dict[str, Any] = {
        "state": extra_label if extra_label is not None else name,
        "heal_mode": "none",
        "assist": "pizza_only",
        "outcome": outcome,
        "success": outcome == "stage_advance",
        "frames": play_frames,
        "total_frames": final.frame,
        "start_hp": start.health if not waiting else 80,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "wave_damage": sum(h["hit"] for h in hits if not h["boss"]),
        "boss_damage": sum(h["hit"] for h in hits if h["boss"]),
        "max_hit": max_hit,
        "pizza_heals": real_pizza if spec.filter_respawn_pizza else pizza_heals,
        "pizza_heal_count": len(real_pizza),
        spec.boss_entry_hp_key: boss_entry_hp,
        "lives": f"{prev_lives}->{final.lives}",
        "boss_hp": (
            f"{int(start.extras.get('boss_hp', 0))}"
            f"->{int(final.extras.get('boss_hp', 0))}"
        ),
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": sorted(reasons.items(), key=lambda kv: -kv[1])[:16],
        "hits": hits,
    }
    if spec.include_end_stage:
        report["end_stage"] = final.stage
    return report


def _print_suite_row(
    spec: CleanProbeSpec, name: str, report: dict[str, Any]
) -> None:
    parts = [
        f"  [{name}] outcome={report.get('outcome')}",
        f"dmg={report.get('damage_taken', '?')}",
        f"min_hp={report.get('min_hp', '?')}",
    ]
    if spec.suite_print_entry:
        parts.append(f"entry={report.get(spec.boss_entry_hp_key, '?')}")
    if spec.suite_print_pizza:
        parts.append(f"pizza={report.get('pizza_heal_count', '?')}")
    parts.append(f"frames={report.get('frames', '?')}")
    print(" ".join(parts))


def run_suite(
    spec: CleanProbeSpec, *, max_frames: int | None = None
) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + extra (power-on / bridge)."""
    frames = spec.cli_max_frames if max_frames is None else max_frames
    results: list[dict[str, Any]] = []
    for name in spec.suite_states:
        try:
            report = run_clean_probe(spec, state_name=name, max_frames=frames)
        except Exception as exc:  # noqa: BLE001 — suite continues
            report = {
                "state": name,
                "outcome": "error",
                "success": False,
                "error": str(exc),
            }
        results.append(report)
        _print_suite_row(spec, name, report)
    extra = run_clean_probe(
        spec, max_frames=frames + 4000, **{spec.extra_entry: True}
    )
    results.append(extra)
    _print_suite_row(spec, spec.extra_label, extra)
    ok = sum(1 for r in results if r.get("success"))
    return {
        "assist": "pizza_only",
        "suite_size": len(results),
        "passed": ok,
        "failed": len(results) - ok,
        "all_passed": ok == len(results),
        "results": results,
    }


def build_clean_parser(
    spec: CleanProbeSpec, description: str
) -> argparse.ArgumentParser:
    """Standard flags shared by ``probe_stage{N}_clean`` CLIs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--state", default=spec.default_state)
    parser.add_argument("--max-frames", type=int, default=spec.cli_max_frames)
    parser.add_argument("--stop-stage-gt", type=int, default=spec.stop_stage_gt)
    flag = "--" + spec.extra_entry.replace("_", "-")
    parser.add_argument(flag, action="store_true", help=spec.extra_help)
    parser.add_argument("--suite", action="store_true", help=spec.suite_help)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSON report path (default under {spec.evidence_dir}/)",
    )
    return parser


def clean_main(
    spec: CleanProbeSpec,
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> int:
    """Parse argv, run one probe or the suite, write JSON under evidence_dir."""
    desc = description if description is not None else (
        f"Stage {spec.stage_byte + 1} Clean (pizza-only) probes."
    )
    args = build_clean_parser(spec, desc).parse_args(argv)
    out_dir = RECORDINGS_DIR / spec.evidence_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.suite:
        report = run_suite(spec, max_frames=args.max_frames)
        out = args.out or (out_dir / "clean_suite.json")
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"suite passed={report['passed']}/{report['suite_size']} "
            f"all_passed={report['all_passed']}"
        )
        print(f"report={out}")
        return 0 if report["all_passed"] else 1

    extra_on = bool(getattr(args, spec.extra_entry, False))
    report = run_clean_probe(
        spec,
        state_name=args.state,
        max_frames=args.max_frames,
        stop_stage_gt=args.stop_stage_gt,
        **{spec.extra_entry: extra_on},
    )
    label = spec.extra_label if extra_on else args.state.lower()
    out = args.out or (out_dir / f"clean_{label}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"dmg={report['damage_taken']} (wave={report['wave_damage']} "
        f"boss={report['boss_damage']}) min_hp={report['min_hp']} "
        f"pizza_heals={report['pizza_heal_count']} "
        f"{spec.boss_entry_hp_key}={report[spec.boss_entry_hp_key]} "
        f"max_hit={report['max_hit']}"
    )
    print(f"report={out}")
    top = report.get("top_reasons") or []
    if spec.cli_print_reasons and top:
        print("reasons: " + ", ".join(f"{k}={v}" for k, v in top[:10]))
    return 0 if report.get("success") else 1
