#!/usr/bin/env python3
"""D2 leftover quota smash from Y1_After_Buy_Potato (not a plant tape).

10 bushes (pick+toss) → dump fence posts in ponds → toss 10 stones in ponds
→ hammer 4 boulders → axe 2 stumps.
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

from harvest.paths import GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState
from retro_harness.headed import (
    add_headed_flag,
    attach_headed,
    headed_emu_repeat,
    idle_headed,
)

from harvest.clock_glance import d2_leftover_spec, leftover_json
from harvest.core.carry import backpack_tool, selected_tool
from harvest.core.game_clock import clock_from_ram, format_segment_time
from harvest.core.ram_catalog import read_ram_value
from harvest.core.shipping_credit import shipping_scene_needs_dismiss
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
    rock_clear_phase,
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
from harvest.tasks.farm_clear_quota import (
    ClearQuota,
    DebrisCounts,
    classify_target,
    count_debris,
    quota_counts_met,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import dismiss_dialogue_result


_SECTIONS = ("all", "bushes", "fences", "stones", "rocks", "stumps")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_After_Buy_Potato")
    p.add_argument("--timeout", type=int, default=200_000)
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
    add_headed_flag(p)
    return p.parse_args()


def _headed_hud(env) -> str:
    ram = env.get_ram()
    clock = clock_from_ram(ram)
    pos = get_pos_from_ram(ram)
    tm = int(read_ram_value(ram, "tilemap") or 0)
    return f"BOT leftover {clock} map={hex(tm)} ({pos.x // 16},{pos.y // 16})"


def _run_task(env, task, *, timeout: int, start_frame: int):
    obs = None
    result = None
    frame = start_frame
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
                action = result.action.action if result.action is not None else make_action()
            obs, _reward, _term, _trunc, _info = env.step(action)
            frame += 1
            if frame > start_frame + timeout:
                stopped = True
                break
        if stopped:
            break
    return frame, result, env.get_ram()


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
        return ClearQuota(stones=10)
    if section == "rocks":
        return ClearQuota(large_rocks=4)
    if section == "stumps":
        return ClearQuota(stumps=2)
    return ClearQuota(
        weeds=10,
        stones=10,
        large_rocks=4,
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
    import gzip

    out_state = GAME_DIR / f"{state_name}.state"
    with gzip.open(out_state, "wb", compresslevel=9) as handle:
        handle.write(env.em.get_state())
    return out_state


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


def _print_table(start: dict, end: dict, cleared: dict, wanted: ClearQuota, frames: int) -> None:
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
                env, exit_task, timeout=2_000, start_frame=0
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
        stam = Stamina.from_ram(ram)
        phases = _phases_for(
            args.section, stamina=stam, include_spa=not args.no_spa
        )
        ok = True
        for spec in phases:
            remaining = max(200, args.timeout - frame)
            timeout = _phase_timeout(spec, remaining)
            world = WorldState(frame=frame, ram=ram, info={}, obs=None)
            task = build_phase_task(ctx, spec, world)
            if task is None:
                journal.append({"phase": spec.phase, "status": "none", "reason": "no task"})
                ok = False
                break
            before = count_debris(ram).as_dict()
            stam_before = Stamina.from_ram(ram).to_dict()
            task.reset(world)
            frame, result, ram = _run_task(
                env, task, timeout=timeout, start_frame=frame
            )
            after = count_debris(ram).as_dict()
            row = {
                "phase": spec.phase,
                "status": result.status.value if result is not None else "none",
                "reason": result.reason if result is not None else "",
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
            if result is None or result.status != TaskStatus.SUCCESS:
                ok = False
                break

        end = _snapshot(ram)
        cleared = DebrisCounts(**start["debris"]).cleared_since(
            DebrisCounts(**end["debris"])
        )
        start_counts = DebrisCounts(**start["debris"])
        end_counts = DebrisCounts(**end["debris"])
        ok = ok and _section_complete(args.section, start_counts, end_counts)
        required_empty = ["fences"] if args.section in {"fences", "all"} else []
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
        _print_table(start, end, cleared.as_dict(), wanted, frame)
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
