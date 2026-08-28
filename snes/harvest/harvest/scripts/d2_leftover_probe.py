#!/usr/bin/env python3
"""D2 leftover smash from a live pin (not a plant tape).

10 bushes (pick+toss) → dump fence posts in ponds → toss remaining stones
in ponds → hammer remaining 2×2 boulders → axe 2 stumps.
Isolated leftover pin (rr-w14t / rr-20w.2.8). Do not redo power-on here.

    HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \\
      --state Y1_After_Buy_Potato --out recordings/d2_leftover_smash.json
    HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe --dump
    uv run python -m harvest.scripts.d2_leftover_probe --headed --section fences \\
      --state Y1_D2_After_Bushes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState
from retro_harness.headed import (
    add_headed_flag,
    attach_headed,
    idle_headed,
)

from harvest.clock_glance import d2_leftover_spec, leftover_json
from harvest.core.carry import backpack_tool, selected_tool
from harvest.core.game_clock import clock_from_ram, format_segment_time
from harvest.core.ram_catalog import read_ram_value
from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    CLEARABLE_DEBRIS_TYPES,
    DebrisType,
)
from harvest.planner.d2_work import (
    bush_clear_phase,
    d2_leftover_phases,
    ensure_axe_phase,
    ensure_hammer_phase,
    fence_dump_phase,
    needs_spa_before_next_smash,
    rock_clear_phase,
    should_spa_retry,
    stone_pond_phase,
    stump_clear_phase,
)
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.planner.day_phase_stamina import full_restore_spa_phase
from harvest.planner.day_phase_types import DayPlannerPolicy
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.runtime.watch_display import configure_headless
from harvest.scripts.leftover_exec import (
    phase_already_clear,
    print_leftover_table,
    run_leftover_task,
    save_emulator_state,
)
from harvest.tasks.farm_clear_quota import (
    ClearQuota,
    DebrisCounts,
    classify_target,
    count_debris,
    quota_counts_met,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram


_SECTIONS = ("all", "bushes", "fences", "stones", "rocks", "stumps")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_After_Buy_Potato")
    p.add_argument("--timeout", type=int, default=400_000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "d2_leftover_smash.json",
    )
    p.add_argument(
        "--save-end-state",
        type=str,
        default=None,
        help="Write a gzip save under custom_integrations when the section is empty.",
    )
    p.add_argument(
        "--save-partial-state",
        type=str,
        default=None,
        help="Write a debug-only gzip save when the selected section is still red.",
    )
    p.add_argument(
        "--checkpoint-state",
        type=str,
        default="Y1_D2_Leftover_Checkpoint",
        help="Overwrite this gzip save on a progress interval (empty disables).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=15_000,
        help="Frames between leftover checkpoint saves.",
    )
    p.add_argument(
        "--stall-frames",
        type=int,
        default=24_000,
        help="Abort a phase if debris counts stay unchanged this long.",
    )
    p.add_argument(
        "--section",
        choices=_SECTIONS,
        default="all",
        help="Run one leftover section or the full smash.",
    )
    p.add_argument(
        "--dump",
        action="store_true",
        help="Snapshot debris/stamina/clock and exit (no smash).",
    )
    p.add_argument("--no-spa", action="store_true", help="Never insert HOT_SPRING_STAMINA.")
    add_headed_flag(
        p,
        help=(
            "Watch-only pygame window (no human/bot toggle). Inspect and F5-record "
            "with: python -m harvest.runtime.harvest_bot play --state STATE "
            "--no-day-plan --record NAME"
        ),
    )
    return p.parse_args()


def _headed_hud(env) -> str:
    ram = env.get_ram()
    clock = clock_from_ram(ram)
    pos = get_pos_from_ram(ram)
    tm = int(read_ram_value(ram, "tilemap") or 0)
    return f"BOT leftover {clock} map={hex(tm)} ({pos.x // 16},{pos.y // 16})"


def _run_task(
    env,
    task,
    *,
    timeout: int,
    start_frame: int,
    checkpoint_state: str | None = None,
    checkpoint_every: int = 0,
    stall_frames: int = 0,
):
    return run_leftover_task(
        env,
        task,
        timeout=timeout,
        start_frame=start_frame,
        checkpoint_state=checkpoint_state or None,
        checkpoint_every=checkpoint_every,
        stall_frames=stall_frames,
    )


def _carry(ram) -> dict:
    return {
        "selected": int(selected_tool(ram)),
        "backpack": int(backpack_tool(ram)),
    }


def _snapshot(ram) -> dict:
    pos = get_pos_from_ram(ram)
    stam = Stamina.from_ram(ram)
    clock = clock_from_ram(ram)
    counts = count_debris(ram)
    samples = {key: [] for key in counts.as_dict()}
    scan_types = set(CLEARABLE_DEBRIS_TYPES) | {DebrisType.FENCE}
    for target in TileScanner().scan(ram, types=scan_types):
        key = classify_target(int(target.tile_id), target.debris_type)
        if key in samples and len(samples[key]) < 16:
            samples[key].append([target.tile[0], target.tile[1], hex(int(target.tile_id))])
    return {
        "tilemap": hex(int(read_ram_value(ram, "tilemap") or 0)),
        "pos": [pos.x, pos.y],
        "tile": [pos.x // 16, pos.y // 16],
        "clock": clock.to_dict(),
        "stamina": stam.to_dict(),
        "carry": _carry(ram),
        "debris": counts.as_dict(),
        "samples": samples,
    }


def _wanted_quota(section: str) -> ClearQuota:
    """Day 2 work contract; the oversized fence quota caps to all present."""
    if section == "bushes":
        return ClearQuota(weeds=10)
    if section == "fences":
        return ClearQuota(fences=10_000)
    if section == "stones":
        return ClearQuota(stones=10_000)
    if section == "rocks":
        return ClearQuota(large_rocks=10_000)
    if section == "stumps":
        return ClearQuota(stumps=2)
    return ClearQuota(
        weeds=10,
        stones=10_000,
        large_rocks=10_000,
        stumps=2,
        fences=10_000,
    )


def _section_complete(
    section: str,
    start: DebrisCounts,
    end: DebrisCounts,
) -> bool:
    """True when this pass removes the bounded D2 quota from its own start."""
    return quota_counts_met(start, end, _wanted_quota(section))


def _phases_for(section: str, *, stamina: Stamina, include_spa: bool):
    if section == "all":
        policy = DayPlannerPolicy(include_spa=include_spa)
        return d2_leftover_phases(stamina=stamina, policy=policy)
    phases = []
    if include_spa and section in {"rocks", "stumps"} and not stamina.can_finish_multi_hit():
        phases.append(full_restore_spa_phase())
    if section == "bushes":
        phases.append(bush_clear_phase())
    elif section == "fences":
        phases.append(fence_dump_phase())
    elif section == "stones":
        phases.append(stone_pond_phase())
    elif section == "rocks":
        phases.extend([ensure_hammer_phase(), rock_clear_phase()])
    elif section == "stumps":
        phases.extend([ensure_axe_phase(), stump_clear_phase()])
    return phases


def _phase_timeout(spec, remaining: int) -> int:
    """timeout <= 0 means exhaustive: spend the leftover probe budget."""
    params = spec.params or {}
    if "timeout" in params:
        timeout = int(params["timeout"])
        if timeout <= 0:
            return remaining
        return min(timeout, remaining)
    estimated = getattr(spec.contract, "estimated_frames", None)
    return min(int(estimated or 8000), remaining)


def _save_emulator_state(env, state_name: str) -> Path:
    return save_emulator_state(env, state_name)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def _try_snapshot(ram) -> dict:
    if ram is None:
        return {}
    try:
        return _snapshot(ram)
    except Exception:
        return {}


def _emit(
    path: Path, *, ram, section: str, ok: bool, done: bool | None = None, **fields
) -> None:
    spec = d2_leftover_spec(section, done=ok if done is None else done)
    fields.setdefault("section", section)
    _write_payload(path, leftover_json(_try_snapshot(ram), spec, ok=ok, **fields))


def main() -> int:
    args = _parse_args()
    headed = bool(getattr(args, "headed", False))
    if not headed:
        configure_headless()
    env = make_harvest_env(state=args.state)
    journal = []
    pygame_mod = None
    ram = None
    start = None
    ctx = TaskBuildContext(state_name=args.state)
    try:
        env.reset()
        if headed:
            pygame_mod = attach_headed(
                env, title="Harvest D2 leftover", hud=_headed_hud, speed=4.0
            )
        ram = env.get_ram()
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        frame = 0
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if is_house_tilemap(tilemap) or not is_farm_tilemap(tilemap):
            exit_task = ExitToFarmTask()
            exit_task.reset(world)
            frame, result, ram = _run_task(
                env, exit_task, timeout=2_000, start_frame=0, stall_frames=0
            )
            journal.append(
                {
                    "phase": "exit_to_farm",
                    "status": result.status.value if result is not None else "none",
                    "reason": result.reason if result is not None else "",
                    "frames": frame,
                }
            )
            if result is None or result.status != TaskStatus.SUCCESS:
                _emit(
                    args.out,
                    ram=ram,
                    section=args.section,
                    ok=False,
                    journal=journal,
                )
                return 1

        start = _snapshot(ram)
        if args.dump:
            _emit(
                args.out,
                ram=ram,
                section=args.section,
                ok=True,
                done=False,
                start=start,
                journal=journal,
                dump=True,
            )
            return 0

        wanted = _wanted_quota(args.section)
        include_spa = not args.no_spa
        stam = Stamina.from_ram(ram)
        pending = list(
            _phases_for(args.section, stamina=stam, include_spa=include_spa)
        )
        ok = True
        while pending:
            spec = pending.pop(0)
            remaining = max(200, args.timeout - frame)
            timeout = _phase_timeout(spec, remaining)
            world = WorldState(frame=frame, ram=ram, info={}, obs=None)
            live_counts = count_debris(ram)
            if phase_already_clear(spec.phase, live_counts):
                journal.append(
                    {
                        "phase": spec.phase,
                        "status": "skipped",
                        "reason": "already clear on pin",
                        "frames": frame,
                        "debris_after": live_counts.as_dict(),
                    }
                )
                continue
            task = build_phase_task(ctx, spec, world)
            if task is None:
                journal.append({"phase": spec.phase, "status": "none", "reason": "no task"})
                ok = False
                break
            before = count_debris(ram).as_dict()
            stam_before = Stamina.from_ram(ram).to_dict()
            task.reset(world)
            ckpt = (args.checkpoint_state or "").strip() or None
            frame, result, ram = _run_task(
                env,
                task,
                timeout=timeout,
                start_frame=frame,
                checkpoint_state=ckpt,
                checkpoint_every=int(args.checkpoint_every),
                stall_frames=int(args.stall_frames),
            )
            after = count_debris(ram).as_dict()
            reason = result.reason if result is not None else ""
            row = {
                "phase": spec.phase,
                "status": result.status.value if result is not None else "none",
                "reason": reason,
                "frames": frame,
                "timeout": timeout,
                "cleared_count": getattr(task, "cleared_count", None),
                "carry": _carry(ram),
                "stamina": Stamina.from_ram(ram).to_dict(),
                "stamina_before": stam_before,
                "debris_before": before,
                "debris_after": after,
                "clock": clock_from_ram(ram).to_dict(),
            }
            journal.append(row)
            _emit(
                args.out,
                ram=ram,
                section=args.section,
                ok=False,
                start=start,
                journal=journal,
                partial=True,
            )
            live_stam = Stamina.from_ram(ram)
            if result is not None and result.status == TaskStatus.SUCCESS:
                if needs_spa_before_next_smash(
                    spec.phase,
                    live_stam,
                    include_spa=include_spa,
                    remaining_phases=[p.phase for p in pending],
                ):
                    pending.insert(0, full_restore_spa_phase())
                continue
            if should_spa_retry(
                spec.phase, reason, live_stam, include_spa=include_spa
            ):
                pending.insert(0, spec)
                pending.insert(0, full_restore_spa_phase())
                continue
            ok = False
            break

        end = _snapshot(ram)
        cleared = DebrisCounts(**start["debris"]).cleared_since(
            DebrisCounts(**end["debris"])
        )
        start_counts = DebrisCounts(**start["debris"])
        end_counts = DebrisCounts(**end["debris"])
        ok = ok and _section_complete(args.section, start_counts, end_counts)
        required_empty = []
        if args.section in {"fences", "all"}:
            required_empty.append("fences")
        if args.section in {"stones", "all"}:
            required_empty.append("stones")
        if args.section in {"rocks", "all"}:
            required_empty.append("large_rocks")
        saved = None
        if args.save_end_state and ok:
            saved = _save_emulator_state(env, args.save_end_state)
            print(f"[LEFTOVER] Saved end state -> {saved}")
        elif args.save_partial_state and not ok:
            saved = _save_emulator_state(env, args.save_partial_state)
            print(f"[LEFTOVER] Saved partial debug state -> {saved}")
        extra = {}
        if saved is not None:
            extra["saved_state"] = str(saved)
        _emit(
            args.out,
            ram=ram,
            section=args.section,
            ok=ok,
            start=start,
            journal=journal,
            end=end,
            cleared=cleared.as_dict(),
            required_empty=required_empty,
            required_quota=_wanted_quota(args.section).__dict__,
            frames=frame,
            time=format_segment_time(frame),
            **extra,
        )
        print_leftover_table(start, end, cleared.as_dict(), wanted, frame)
        return 0 if ok else 1
    except KeyboardInterrupt:
        extra = {"journal": journal, "reason": "headed window closed"}
        if start is not None:
            extra["start"] = start
        _emit(args.out, ram=ram, section=args.section, ok=False, **extra)
        return 1
    finally:
        if headed and pygame_mod is not None:
            idle_headed(env, pygame_mod)
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
