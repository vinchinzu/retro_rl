"""Leftover probe step loop: checkpoint saves and stall abort."""

from __future__ import annotations

import gzip
from pathlib import Path

from retro_harness import TaskResult, TaskStatus, WorldState
from retro_harness.headed import headed_emu_repeat

from harvest.core.shipping_credit import shipping_scene_needs_dismiss
from harvest.paths import GAME_DIR
from harvest.planner.d2_work import leftover_chain_decision, phase_already_clear
from harvest.tasks.farm_clear_quota import ClearQuota, DebrisCounts, count_debris
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import dismiss_dialogue_result


def save_emulator_state(env, state_name: str) -> Path:
    out_state = GAME_DIR / f"{state_name}.state"
    with gzip.open(out_state, "wb", compresslevel=9) as handle:
        handle.write(env.em.get_state())
    return out_state


def print_leftover_table(
    start: dict, end: dict, cleared: dict, wanted: ClearQuota, frames: int
) -> None:
    clock = (end.get("clock") or {}).get("clock", "?")
    rows = [
        ("Weeds", f"{cleared.get('weeds', 0)} / {wanted.weeds}  ({start['debris']['weeds']}→{end['debris']['weeds']})"),
        ("Fences", f"{cleared.get('fences', 0)} / {wanted.fences}  ({start['debris']['fences']}→{end['debris']['fences']})"),
        ("Stones", f"{cleared.get('stones', 0)} / {wanted.stones}  ({start['debris']['stones']}→{end['debris']['stones']})"),
        ("Small06", f"{cleared.get('small_rocks', 0)} / {wanted.small_rocks}  ({start['debris']['small_rocks']}→{end['debris']['small_rocks']})"),
        ("Boulders", f"{cleared.get('large_rocks', 0)} / {wanted.large_rocks}  ({start['debris']['large_rocks']}→{end['debris']['large_rocks']})"),
        ("Stumps", f"{cleared.get('stumps', 0)} / {wanted.stumps}  ({start['debris']['stumps']}→{end['debris']['stumps']})"),
        ("Stamina", f"{start['stamina']['current']}→{end['stamina']['current']} / {end['stamina']['maximum']}"),
        ("Frames", f"{frames} ({clock})"),
    ]
    print()
    print(f"{'Check':<10} {'Result'}")
    print("-" * 56)
    for name, result in rows:
        print(f"{name:<10} {result}")


def _debris_key(ram) -> tuple:
    counts = count_debris(ram).as_dict()
    return tuple(counts[k] for k in sorted(counts))


def _should_abort_stall(frame: int, last_progress: int, stall_frames: int) -> bool:
    """True when debris counts have not changed for stall_frames."""
    return stall_frames > 0 and frame - last_progress >= stall_frames


def _is_spa_phase(spec) -> bool:
    phase = str(getattr(spec, "phase", "") or "")
    kind = str(getattr(spec, "kind", "") or "")
    return phase == "HOT_SPRING_STAMINA" or kind == "hot_spring"


def _phase_timeout(spec, remaining: int) -> int:
    """timeout<=0 or spa spends remaining; other estimates cap."""
    params = spec.params or {}
    if "timeout" in params:
        timeout = int(params["timeout"])
        if timeout <= 0:
            return remaining
        return min(timeout, remaining)
    if _is_spa_phase(spec):
        return remaining
    estimated = getattr(spec.contract, "estimated_frames", None)
    return min(int(estimated or 8000), remaining)


def _phase_timeout_result(result: TaskResult | None, timeout: int) -> TaskResult:
    """FAILURE when the phase frame budget elapsed with no terminal status."""
    if result is None or result.status == TaskStatus.RUNNING:
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"phase timeout {timeout}f",
        )
    return result


def run_leftover_task(
    env,
    task,
    *,
    timeout: int,
    start_frame: int,
    checkpoint_state: str | None = None,
    checkpoint_every: int = 15_000,
    stall_frames: int = 24_000,
):
    """Step until SUCCESS/FAILURE, timeout, or no-debris stall."""
    obs = None
    result = None
    frame = start_frame
    last_key = _debris_key(env.get_ram())
    last_progress = start_frame
    last_checkpoint = start_frame
    while frame <= start_frame + timeout:
        budget = headed_emu_repeat(env)
        stopped = False
        for _ in range(budget):
            ram = env.get_ram()
            if shipping_scene_needs_dismiss(ram):
                dismiss = dismiss_dialogue_result(
                    frame, buttons=("a",), pulse_every=2, reason="shipping scene"
                )
                action = dismiss.action.action
            else:
                world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
                result = task.step(world)
                if result.status != TaskStatus.RUNNING:
                    stopped = True
                    break
                action = (
                    result.action.action if result.action is not None else make_action()
                )
            obs, _reward, _term, _trunc, _info = env.step(action)
            frame += 1
            poll = (
                frame % 60 == 0
                or (
                    checkpoint_state
                    and checkpoint_every > 0
                    and frame - last_checkpoint >= checkpoint_every
                )
            )
            if poll:
                ram = env.get_ram()
                key = _debris_key(ram)
                if key != last_key:
                    last_key = key
                    last_progress = frame
                if (
                    checkpoint_state
                    and checkpoint_every > 0
                    and frame - last_checkpoint >= checkpoint_every
                ):
                    saved = save_emulator_state(env, checkpoint_state)
                    print(
                        f"[LEFTOVER] checkpoint {checkpoint_state} f={frame} "
                        f"debris={list(key)} -> {saved}"
                    )
                    last_checkpoint = frame
                if _should_abort_stall(frame, last_progress, stall_frames):
                    if checkpoint_state:
                        save_emulator_state(env, checkpoint_state)
                    print(
                        f"[LEFTOVER] no debris progress {stall_frames}f "
                        f"(last_progress={last_progress}); aborting phase"
                    )
                    result = TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            f"no debris progress {stall_frames}f "
                            f"(last_progress={last_progress})"
                        ),
                    )
                    stopped = True
                    break
            if frame > start_frame + timeout:
                stopped = True
                break
        if stopped:
            break
    return frame, _phase_timeout_result(result, timeout), env.get_ram()
