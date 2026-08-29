"""Per-frame helpers for the continuous full Hard run."""

from __future__ import annotations

from typing import Any

from retro_harness.actions import idle_action
from retro_harness.controls import SNES_START
from retro_harness.env import save_state
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.assist import apply_emergency_hp
from tmnt_iv.menus import RAPH_HARD_BOOT_LAST, raph_hard_boot_action
from tmnt_iv.observe import HpDelta, living_hp, policy_input
from tmnt_iv.paths import GAME, GAME_DIR
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state
from tmnt_iv.run.metrics import (
    HARD_VALUE,
    STAGE_NAMES,
    RunMetrics,
    StageSplit,
    format_duration,
)


def is_gameplay_active(state: GameState, metrics: RunMetrics) -> bool:
    """True during live stages before the credits byte is seen."""
    menu = int(state.extras.get("menu", -1))
    return (
        metrics.credits_start_frame is None
        and menu == 6
        and state.player_x > 0
        and state.stage <= 9
    )


def mark_gameplay_start(
    state: GameState,
    *,
    hp: HpDelta,
    metrics: RunMetrics,
) -> int:
    """Seed lives / HP on the first active frame. Returns current lives."""
    metrics.lives_start = state.lives
    metrics.lives_peak = state.lives
    if living_hp(state.health):
        hp.prev = state.health
        metrics.min_health_seen = (
            hp.min_hp if hp.min_hp is not None else state.health
        )
    return state.lives


def guard_hard_route(
    env: Any,
    state: GameState,
    *,
    frame: int,
    menu: int,
    last_stage: int,
) -> tuple[bool, int]:
    """Confirm Hard WRAM and reject title/stage regression."""
    difficulty = int(env.get_ram()[0x1FEE])
    hard_ok = difficulty == HARD_VALUE
    if not hard_ok and frame > 2500:
        raise RuntimeError(f"difficulty changed from Hard: {difficulty}")
    if menu == 0:
        raise RuntimeError(f"unexpected return to title at frame {frame}")
    if state.stage < last_stage:
        raise RuntimeError(
            f"stage regressed {last_stage}->{state.stage} at frame {frame}"
        )
    return hard_ok, max(last_stage, state.stage)


def guard_lives(
    state: GameState,
    *,
    frame: int,
    previous_lives: int,
    metrics: RunMetrics,
) -> int:
    """Abort on a life decrement; otherwise track peak/end lives."""
    if state.lives < previous_lives:
        metrics.life_losses += previous_lives - state.lives
        raise RuntimeError(
            f"life loss at frame {frame}: "
            f"{previous_lives}->{state.lives} "
            f"stage={state.stage} dmg={metrics.total_damage_taken}"
        )
    metrics.lives_peak = max(metrics.lives_peak or state.lives, state.lives)
    metrics.lives_end = state.lives
    return state.lives


def freeze_is_armed(
    state: GameState, *, started: bool, metrics: RunMetrics
) -> bool:
    """True while playing with no enemies before credits (freeze-abort arm)."""
    return (
        started
        and metrics.credits_start_frame is None
        and state.mode is GameMode.PLAYING
        and state.player_x > 0
        and not state.living_enemies
    )


def record_active_hp(
    env: Any,
    state: GameState,
    *,
    frame: int,
    active: bool,
    emergency_hp: bool,
    hp: HpDelta,
    metrics: RunMetrics,
) -> GameState:
    """Count natural HP drops; optional emergency heal. May re-parse RAM."""
    if not active:
        hp.prev = None
        return state
    hit = hp.note(state.health)
    if hit:
        metrics.damage_by_stage[state.stage] = (
            metrics.damage_by_stage.get(state.stage, 0) + hit
        )
    metrics.total_damage_taken = hp.damage
    metrics.max_single_frame_damage = hp.max_hit
    if hp.min_hp is not None:
        metrics.min_health_seen = hp.min_hp
    if emergency_hp and apply_emergency_hp(env, state.health):
        metrics.health_guard_interventions += 1
        state = parse_game_state(env.get_ram(), frame=frame)
        # Restored bar must not count as a later drop from the pre-heal value.
        hp.prev = state.health
    return state


def select_full_run_action(
    *,
    frame: int,
    state: GameState,
    policy: Stage1Policy,
    metrics: RunMetrics,
    started: bool,
) -> tuple[Any, str]:
    """Boot plan, credits/transition idle, or policy. Never START after boot."""
    if frame <= RAPH_HARD_BOOT_LAST:
        action = raph_hard_boot_action(frame)
        reason = "boot_menu" if any(action) else "boot_idle"
    elif metrics.credits_start_frame is not None:
        action = idle_action()
        reason = "credits_idle"
    elif started and (
        state.player_x == 0
        or state.mode in {GameMode.CUTSCENE, GameMode.CONTINUE}
    ):
        # Stage loads briefly look like CONTINUE because HP/X are zero.
        # Do not press START after gameplay begins.
        action = idle_action()
        reason = "transition_idle"
    else:
        action, reason = policy_input(policy, state)
    if frame > RAPH_HARD_BOOT_LAST and action[SNES_START]:
        action = idle_action()
        reason = "suppressed_start"
    if action[8]:
        raise RuntimeError(f"forbidden A special at frame {frame}")
    return action, reason


def record_stage_split(
    env: Any,
    state: GameState,
    *,
    frame: int,
    fps: float,
    metrics: RunMetrics,
    policy: Stage1Policy,
    hp: HpDelta,
    split_stages: set[int],
    entry_state_prefix: str | None,
) -> None:
    """Reset combat phase and record the first playable frame of a stage."""
    if state.stage in split_stages:
        return
    split_stages.add(state.stage)
    # Stateful combat phases must not leak across natural stage
    # transitions. This also makes each continuous stage match its
    # independently verified checkpoint probe.
    policy.reset()
    hp.prev = state.health if living_hp(state.health) else None
    name = STAGE_NAMES.get(state.stage, "UNKNOWN")
    metrics.stage_splits.append(
        StageSplit(
            stage=state.stage,
            name=name,
            frame=frame,
            elapsed_seconds=frame / fps,
        )
    )
    if entry_state_prefix:
        save_state(
            env,
            GAME_DIR,
            GAME,
            f"{entry_state_prefix}Stage{state.stage + 1}",
        )
    print(
        f"stage {state.stage + 1:02d} {name} "
        f"at {format_duration(frame / fps)} "
        f"dmg={metrics.total_damage_taken}",
        flush=True,
    )


def log_progress(
    state: GameState,
    *,
    frame: int,
    event: int,
    metrics: RunMetrics,
    reason: str,
) -> None:
    """Periodic RAM snapshot for long dry-runs."""
    if not frame or frame % 10_000 != 0:
        return
    targets = [
        (hex(enemy.kind), enemy.health, enemy.x, enemy.y)
        for enemy in state.living_enemies
    ]
    print(
        f"frame {frame}  stage={state.stage} event={event:#04x} "
        f"damage={metrics.total_damage_taken} lives={state.lives} "
        f"p=({state.player_x},{state.player_y}) "
        f"hp={state.health} char={state.extras.get('char_id')} "
        f"reason={reason} pickups={state.extras.get('pickups')} "
        f"targets={targets}",
        flush=True,
    )
