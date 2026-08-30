#!/usr/bin/env python3
"""D2 leftover smash from a live pin (not a plant tape).

10 bushes (pick+toss) → dump fence posts in ponds → toss remaining stones
in ponds → hammer remaining 2×2 boulders → axe remaining stumps.
Smash types run as four farm chunks (nw/ne/sw/se) so a last-cell stall
cannot eat the whole farm. Isolated leftover pin (rr-w14t / rr-20w.2.8).
Do not redo power-on here.

    HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \\
      --state Y1_After_Buy_Potato --out recordings/d2_leftover_smash.json
    HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe --dump
    uv run python -m harvest.scripts.d2_leftover_probe --headed --section fences \\
      --state Y1_D2_After_Bushes
    HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \\
      --section stones --chunk sw --state Y1_D2_After_Stones \\
      --out recordings/d2_leftover_stones_sw.json
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
from harvest.planner.d2_farm_chunks import (
    FARM_CHUNK_ORDER,
    chunk_bounds,
    resolve_chunks,
    section_complete,
    smash_done_empty,
    wanted_quota,
)
from harvest.planner.d2_work import D2FarmClearTactic, leftover_section_phases, observe_d2_farm
from harvest.planner.day_phase_registry import TaskBuildContext
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.day_plan_tasks import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.runtime.watch_display import configure_headless
from harvest.scripts.leftover_exec import (
    _phase_timeout,
    leftover_chain_decision,
    phase_already_clear,
    print_leftover_table,
    run_leftover_task,
    save_emulator_state,
)
from harvest.tasks.farm_clear_quota import (
    DebrisCounts,
    classify_target,
    count_debris,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import get_pos_from_ram


_SECTIONS = ("all", "bushes", "fences", "stones", "rocks", "stumps")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_After_Buy_Potato")
    p.add_argument("--timeout", type=int, default=2_000_000)
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
        "--chunk",
        choices=("all",) + FARM_CHUNK_ORDER,
        default="all",
        help="Farm quadrant for stones/rocks/stumps (default: chain all four).",
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


def _scan_bounds(section: str, chunk: str):
    """Clip completion counts to one quadrant; None is the whole farm."""
    if chunk == "all":
        return None
    if section not in {"stones", "rocks", "stumps", "all"}:
        return None
    names = resolve_chunks(chunk)
    if len(names) != 1:
        return None
    return chunk_bounds(names[0])


def _wanted_quota(section: str):
    return wanted_quota(section)


def _section_complete(
    section: str,
    start: DebrisCounts,
    end: DebrisCounts,
) -> bool:
    """True when this pass removes the bounded D2 quota from its own start."""
    return section_complete(section, start, end)


def _phases_for(section: str, *, stamina: Stamina, include_spa: bool, chunk: str = "all"):
    return leftover_section_phases(
        section, stamina=stamina, include_spa=include_spa, chunk=chunk
    )


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
        scan_bounds = _scan_bounds(args.section, args.chunk)
        start_counts = count_debris(ram, scan_bounds)
        world = WorldState(frame=frame, ram=ram, info={}, obs=None)
        tactic = D2FarmClearTactic(
            section=args.section,
            chunk=args.chunk,
            include_spa=include_spa,
            ctx=ctx,
        )
        tactic.reset(world)
        ckpt = (args.checkpoint_state or "").strip() or None
        frame, result, ram = _run_task(
            env,
            tactic,
            timeout=max(200, args.timeout - frame),
            start_frame=frame,
            checkpoint_state=ckpt,
            checkpoint_every=int(args.checkpoint_every),
            stall_frames=int(args.stall_frames),
        )
        journal.extend(tactic.journal)
        farm = tactic.farm_status or observe_d2_farm(ram, tactic.journal)
        ok = result is not None and result.status == TaskStatus.SUCCESS
        _emit(
            args.out,
            ram=ram,
            section=args.section,
            ok=False,
            start=start,
            journal=journal,
            farm_status=farm.outcome.value if farm is not None else "",
            partial=True,
        )

        end = _snapshot(ram)
        cleared = DebrisCounts(**start["debris"]).cleared_since(
            DebrisCounts(**end["debris"])
        )
        end_counts = count_debris(ram, scan_bounds)
        ok = ok and _section_complete(args.section, start_counts, end_counts)
        whole_farm = args.chunk == "all"
        required_empty = list(smash_done_empty(args.section) if whole_farm else ())
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
        glance_done = ok and whole_farm
        _emit(
            args.out,
            ram=ram,
            section=args.section,
            ok=ok,
            done=glance_done,
            start=start,
            journal=journal,
            end=end,
            cleared=cleared.as_dict(),
            required_empty=required_empty,
            required_quota=_wanted_quota(args.section).__dict__,
            chunk=args.chunk,
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
