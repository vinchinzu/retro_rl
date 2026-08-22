#!/usr/bin/env python3
"""Natural-entry mountain hot-spring probe (no stamina RAM poke).

Drains stamina with real tool use, then runs ``HotSpringStaminaTask`` through
farm → mountain → **upper outdoor pond** (tilemap 0x10, lip ~(619,201),
water 0xF7) → A+direction bath → optional return to farm.

Cave 0x29 is MapMountainCave (not the spa). Camp tent pond is wrong water.

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.hot_spring_probe
    HEADLESS=1 uv run python -m harvest.scripts.hot_spring_probe \\
      --state Y1_D2_Night_Farm --min-stamina full --target-stamina 70
    HEADLESS=1 uv run python -m harvest.scripts.hot_spring_probe \\
      --state Y1_Inside_House --min-stamina 40 --target-stamina 25
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.planner.day_plan_status import is_farm_tilemap
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import make_action
from harvest.tasks.farm_ops import (
    use_tool,
    cycle_tool,
)
from harvest.core.stamina import Stamina
from harvest.tasks.hot_spring import (
    HotSpringStaminaTask,
    SPA_TILEMAP,
    MOUNTAIN_TILEMAP,
    PATH_TILEMAP,
    PLAYER_ACTION_JUMP,
    read_stamina,
    read_max_stamina,
    read_player_action,
)

# Farm outdoor with tools (natural multi-map entry). House states exit first.
DEFAULT_STATE = "Y1_D2_Night_Farm"


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    # Never re-fill stamina during this probe.
    os.environ.pop("INFINITE_STAMINA", None)


def _configure_watch() -> None:
    os.environ.pop("HEADLESS", None)
    os.environ.pop("SDL_VIDEODRIVER", None)
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state",
        default=DEFAULT_STATE,
        help="Named save state (default: Y1_D2_Night_Farm)",
    )
    p.add_argument(
        "--min-stamina",
        default="full",
        help="Soak target: 'full' (current==max) or an integer threshold",
    )
    p.add_argument(
        "--target-stamina",
        type=int,
        default=35,
        help="Drain with tools until stamina is at or below this (must be < min-stamina)",
    )
    p.add_argument(
        "--return-to-farm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After soak, navigate mountain_to_farm (default: true)",
    )
    p.add_argument(
        "--drain-frames",
        type=int,
        default=8000,
        help="Max frames spent on natural tool drain",
    )
    p.add_argument(
        "--task-timeout",
        type=int,
        default=24000,
        help="HotSpringStaminaTask overall frame budget",
    )
    p.add_argument(
        "--max-jump-cycles",
        type=int,
        default=10,
        help="A+direction bath cycles at upper pond (recording ~5; budget extras)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "hot_spring_probe.json",
    )
    p.add_argument(
        "--no-drain",
        action="store_true",
        help="Skip tool drain (only when stamina is already below min-stamina)",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Open a pygame window and blit each frame (no HEADLESS)",
    )
    p.add_argument("--watch-scale", type=int, default=3, help="Watch window integer scale")
    return p.parse_args()


def _parse_min_stamina(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"full", "max", "all", ""}:
        return None
    return int(text, 0)


def _snap(ram, frame: int, phase: str = "") -> dict:
    stam = Stamina.from_ram(ram)
    return {
        "frame": frame,
        "phase": phase,
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "tilemap_hex": f"0x{int(read_ram_value(ram, 'tilemap')):02X}",
        "stamina": stam.to_dict(),
        "stamina_current": stam.current,
        "max_stamina": stam.maximum,
        "player_x": int(read_ram_value(ram, "player_x")),
        "player_y": int(read_ram_value(ram, "player_y")),
        "player_action": int(read_player_action(ram)),
        "tool": int(read_ram_value(ram, "tool_selected")),
        "input_lock": int(read_ram_value(ram, "input_lock")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
    }


def _step(env, action) -> object:
    step = env.step(action)
    # gymnasium: obs, reward, terminated, truncated, info
    return step[0]


def _watch_begin(scale: int):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    pygame.display.set_caption("Harvest spa corridor  [Esc to close]")
    screen = pygame.display.set_mode((256 * scale, 224 * scale))
    clock = pygame.time.Clock()
    return pygame, screen, clock, scale


def _watch_frame(pygame, screen, clock, scale, obs) -> bool:
    """Blit obs. Return False when the viewer closed the window."""
    import numpy as np

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    if obs is None:
        return True
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        frame = arr[..., :3]
        h, w = frame.shape[0], frame.shape[1]
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (w * scale, h * scale))
        if screen.get_size() != (w * scale, h * scale):
            pygame.display.set_mode((w * scale, h * scale))
            screen = pygame.display.get_surface()
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    return True


def _ensure_outdoor(env, log: list, max_frames: int = 4000) -> dict:
    """Leave house/shed/shop so tool drain and farm_to_spa can run."""
    ram = env.get_ram()
    tilemap = int(read_ram_value(ram, "tilemap"))
    if (
        is_farm_tilemap(tilemap)
        or tilemap in (PATH_TILEMAP, MOUNTAIN_TILEMAP, SPA_TILEMAP)
    ):
        return {"ok": True, "frames": 0, "tilemap": tilemap, "skipped": True}

    print(f"[EXIT] not outdoor (map=0x{tilemap:02X}); ExitToFarmTask")
    from retro_harness import WorldState

    task = ExitToFarmTask()
    obs = None
    world = WorldState(frame=0, ram=ram, info={}, obs=obs)
    task.reset(world)
    frame = 0
    status = TaskStatus.RUNNING
    while frame < max_frames and status == TaskStatus.RUNNING:
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = task.step(world)
        status = result.status
        action = make_action()
        if result.action is not None:
            action = getattr(result.action, "action", result.action)
        step = env.step(action)
        obs = step[0]
        frame += 1
        if frame % 200 == 0:
            print(
                f"[EXIT] f={frame} map=0x{int(read_ram_value(env.get_ram(), 'tilemap')):02X} "
                f"status={status.value}"
            )
    ram = env.get_ram()
    tilemap = int(read_ram_value(ram, "tilemap"))
    ok = is_farm_tilemap(tilemap) or tilemap == PATH_TILEMAP
    out = {
        "ok": ok,
        "frames": frame,
        "tilemap": tilemap,
        "status": status.value,
        "skipped": False,
    }
    log.append({"event": "exit_outdoor", **out, **_snap(ram, frame)})
    print(f"[EXIT] done ok={ok} map=0x{tilemap:02X} frames={frame}")
    return out


def _drain_stamina(env, *, target: int, max_frames: int, log: list, watch=None) -> dict:
    """Use tools for real stamina cost. Never pokes RAM."""
    ram = env.get_ram()
    start = Stamina.from_ram(ram)
    print(f"[DRAIN] start {start} target<={target} (no RAM poke)")
    log.append({"event": "drain_start", **_snap(ram, 0)})

    stalled = 0
    last = start
    frame = 0
    queue: list = []

    while frame < max_frames:
        ram = env.get_ram()
        stam = read_stamina(ram)
        if stam <= target:
            print(f"[DRAIN] reached stamina={stam} at frame={frame}")
            out = {
                "ok": True,
                "frames": frame,
                "stamina": stam,
                "start": int(start),
            }
            log.append({"event": "drain_done", **_snap(ram, frame)})
            return out

        if not queue:
            # Hold Y for a swing, idle for animation, occasional X to cycle tools
            # if the can is empty / wrong tool is selected.
            if frame > 0 and frame % 400 == 0 and stam == last:
                queue.extend(cycle_tool())
                queue.extend(use_tool(frames=12, cooldown=18))
            else:
                queue.extend(use_tool(frames=10, cooldown=14))

        action = queue.pop(0)
        obs = _step(env, action)
        frame += 1
        if watch is not None:
            pygame, screen, clock, scale = watch
            if not _watch_frame(pygame, screen, clock, scale, obs):
                return {
                    "ok": False,
                    "frames": frame,
                    "stamina": read_stamina(env.get_ram()),
                    "start": int(start),
                    "reason": "watch window closed",
                }

        if frame % 100 == 0:
            ram = env.get_ram()
            stam = read_stamina(ram)
            print(
                f"[DRAIN] f={frame} stam={stam} tool={int(read_ram_value(ram, 'tool_selected')):02X} "
                f"map=0x{int(read_ram_value(ram, 'tilemap')):02X}"
            )
            if stam >= last:
                stalled += 1
            else:
                stalled = 0
                last = stam
            # If tool use is not draining, try a few more cycles then fail cleanly.
            if stalled >= 12 and stam > target:
                print(f"[DRAIN] stalled at stamina={stam} after {frame} frames")
                log.append({"event": "drain_stalled", **_snap(ram, frame)})
                return {
                    "ok": False,
                    "frames": frame,
                    "stamina": stam,
                    "start": int(start),
                    "reason": "tool use not draining stamina",
                }

    ram = env.get_ram()
    stam = read_stamina(ram)
    print(f"[DRAIN] timeout stamina={stam}")
    log.append({"event": "drain_timeout", **_snap(ram, frame)})
    return {
        "ok": stam <= target,
        "frames": frame,
        "stamina": stam,
        "start": int(start),
        "reason": "drain frame budget",
    }


def _run_task(
    env,
    task: HotSpringStaminaTask,
    log: list,
    *,
    watch=None,
) -> dict:
    obs = None
    ram = env.get_ram()
    world = WorldState(frame=0, ram=ram, info={}, obs=obs)
    task.reset(world)

    events: list[dict] = []
    tilemaps_seen: list[int] = []
    spa_entered = False
    spa_entry_frame: int | None = None
    soak_start: int | None = None
    soak_peak: int | None = None
    soak_gains: list[dict] = []
    jump_frames: list[int] = []
    was_jumping = False
    last_phase = ""
    last_stam = int(Stamina.from_ram(ram))
    status = TaskStatus.RUNNING
    reason = ""
    frame = 0

    print(
        f"[SPA] task start {last_stam} min={task.min_stamina} "
        f"return_to_farm={task.return_to_farm} max_jump_cycles={task.max_jump_cycles}"
    )
    log.append({"event": "task_start", **_snap(ram, 0, task.phase_text)})

    while frame < task.timeout:
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = task.step(world)
        status = result.status
        reason = result.reason or ""

        action = None
        if result.action is not None:
            # ActionResult.action is the button array; tolerate raw ndarray too.
            action = getattr(result.action, "action", result.action)
        if action is None:
            action = make_action()
        obs = _step(env, action)
        frame += 1
        if watch is not None:
            pygame, screen, clock, scale = watch
            if not _watch_frame(pygame, screen, clock, scale, obs):
                print("[SPA] watch window closed")
                status = TaskStatus.FAILURE
                reason = "watch window closed"
                break

        ram = env.get_ram()
        tilemap = int(read_ram_value(ram, "tilemap"))
        stam = read_stamina(ram)
        pact = read_player_action(ram)
        phase = task.phase_text

        if not tilemaps_seen or tilemaps_seen[-1] != tilemap:
            tilemaps_seen.append(tilemap)
            row = {
                "event": "tilemap",
                **_snap(ram, frame, phase),
            }
            events.append(row)
            log.append(row)
            print(
                f"[SPA] tilemap → 0x{tilemap:02X} f={frame} phase={phase} "
                f"stam={stam} pos=({int(read_ram_value(ram, 'player_x'))},"
                f"{int(read_ram_value(ram, 'player_y'))})"
            )

        if phase != last_phase:
            row = {"event": "phase", **_snap(ram, frame, phase)}
            events.append(row)
            log.append(row)
            print(f"[SPA] phase → {phase} f={frame} stam={stam}")
            last_phase = phase

        # Outdoor spa is mountain 0x10; only count "entered" once soak phase runs
        # near the upper pond (SPA_TILEMAP == MOUNTAIN_TILEMAP).
        if phase == "soak" and not spa_entered:
            spa_entered = True
            spa_entry_frame = frame
            soak_start = stam
            soak_peak = stam
            print(
                f"[SPA] ENTERED outdoor soak on 0x{tilemap:02X} at f={frame} "
                f"stamina={stam} pos=({int(read_ram_value(ram, 'player_x'))},"
                f"{int(read_ram_value(ram, 'player_y'))})"
            )
            log.append({"event": "spa_enter", **_snap(ram, frame, phase)})

        if spa_entered and phase == "soak":
            if soak_peak is None or stam > soak_peak:
                soak_peak = stam
            if stam > last_stam:
                gain = {"frame": frame, "from": last_stam, "to": stam, "player_action": pact}
                soak_gains.append(gain)
                print(f"[SPA] soak gain {last_stam} → {stam} @ f={frame} action={pact}")
                log.append({"event": "soak_gain", **gain, **_snap(ram, frame, phase)})
            if pact == PLAYER_ACTION_JUMP:
                if not was_jumping:
                    jump_frames.append(frame)
                    print(f"[SPA] water/jump anim action=3 @ f={frame} stam={stam}")
                    log.append({"event": "jump_start", **_snap(ram, frame, phase)})
                was_jumping = True
            else:
                was_jumping = False

        if frame % 300 == 0:
            print(
                f"[SPA] f={frame} phase={phase} map=0x{tilemap:02X} "
                f"stam={stam} action={pact} jumps={len(jump_frames)} "
                f"status={status.value}"
            )

        last_stam = stam

        if status != TaskStatus.RUNNING:
            break

    final = _snap(env.get_ram(), frame, task.phase_text)
    summary = {
        "status": status.value,
        "reason": reason,
        "frames": frame,
        "spa_entered": spa_entered,
        "spa_entry_frame": spa_entry_frame,
        "soak_start": soak_start,
        "soak_peak": soak_peak,
        "soak_gains": soak_gains,
        "jump_starts": len(jump_frames),
        "jump_frames": jump_frames[:40],
        "task_jumps_seen": getattr(task, "_jumps_seen", None),
        "task_jump_cycles": getattr(task, "_jump_cycles", None),
        "tilemaps_seen": [f"0x{t:02X}" for t in tilemaps_seen],
        "final": final,
        "saw_mountain": MOUNTAIN_TILEMAP in tilemaps_seen,
        "saw_spa": SPA_TILEMAP in tilemaps_seen,
        "returned_farm": final["tilemap"] in (0x00, 0x0C) and status == TaskStatus.SUCCESS,
    }
    log.append({"event": "task_end", **summary})
    print(
        f"[SPA] DONE status={status.value} reason={reason!r} "
        f"spa_entered={spa_entered} soak={soak_start}→{soak_peak} "
        f"maps={summary['tilemaps_seen']} final_map=0x{final['tilemap']:02X} "
        f"final_stam={final['stamina_current']}/{final['max_stamina']}"
    )
    return summary


def main() -> int:
    args = _parse_args()
    if args.watch:
        _configure_watch()
    else:
        _configure_headless()
    min_stamina = _parse_min_stamina(args.min_stamina)

    if min_stamina is not None and args.target_stamina >= min_stamina:
        print(
            f"ERROR: --target-stamina ({args.target_stamina}) must be < "
            f"--min-stamina ({min_stamina}) so the task routes to the spa"
        )
        return 2

    t0 = time.time()
    log: list = []
    env = make_harvest_env(args.state)
    try:
        env.reset()
        ram = env.get_ram()
        boot = _snap(ram, 0)
        print(
            f"BOOT state={args.state} map=0x{boot['tilemap']:02X} "
            f"stam={boot['stamina_current']}/{boot['max_stamina']} "
            f"object={boot['stamina']} "
            f"tool=0x{boot['tool']:02X} pos=({boot['player_x']},{boot['player_y']})"
        )
        log.append({"event": "boot", "state": args.state, **boot})

        watch = _watch_begin(args.watch_scale) if args.watch else None
        if watch is not None:
            ram0 = env.get_ram()
            # First blit of the loaded pin.
            try:
                obs0 = env.get_screen() if hasattr(env, "get_screen") else None
            except Exception:
                obs0 = None
            if obs0 is not None:
                pygame, screen, clock, scale = watch
                _watch_frame(pygame, screen, clock, scale, obs0)

        outdoor = _ensure_outdoor(env, log)
        if not outdoor.get("ok"):
            report = {
                "ok": False,
                "state": args.state,
                "outdoor": outdoor,
                "drain": None,
                "task": None,
                "elapsed_s": round(time.time() - t0, 2),
                "log": log,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"FAIL outdoor exit: {outdoor}")
            print(f"report={args.out}")
            return 1

        skip_drain = bool(args.no_drain)
        if skip_drain:
            ram = env.get_ram()
            stam = Stamina.from_ram(ram)
            drain = {
                "ok": True,
                "skipped": True,
                "frames": 0,
                "stamina": stam.current,
                "start": stam.current,
            }
            log.append({"event": "drain_skipped", **_snap(ram, 0)})
            print(f"[DRAIN] skipped {stam}")
        else:
            drain = _drain_stamina(
                env,
                target=args.target_stamina,
                max_frames=args.drain_frames,
                log=log,
                watch=watch,
            )
        if not drain.get("ok"):
            report = {
                "ok": False,
                "state": args.state,
                "drain": drain,
                "task": None,
                "elapsed_s": round(time.time() - t0, 2),
                "log": log,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"FAIL drain: {drain}")
            print(f"report={args.out}")
            return 1

        task = HotSpringStaminaTask(
            min_stamina=min_stamina,
            return_to_farm=args.return_to_farm,
            timeout=args.task_timeout,
            soak_timeout=3600,
            soak_plateau_frames=180,
            max_jump_cycles=args.max_jump_cycles,
        )
        summary = _run_task(env, task, log, watch=watch)
        if watch is not None:
            watch[0].quit()
        final_stam = Stamina.from_mapping(summary["final"]["stamina"])
        restored = (
            final_stam.is_full
            if min_stamina is None
            else final_stam.current >= min_stamina
        )

        ok = (
            summary["status"] == TaskStatus.SUCCESS.value
            and summary["spa_entered"]
            and restored
            and summary["soak_peak"] is not None
            and summary["soak_start"] is not None
            and summary["soak_peak"] > summary["soak_start"]
        )
        # Soft ok: entered spa and gained even if return nav flaked.
        soft_ok = (
            summary["spa_entered"]
            and summary["soak_peak"] is not None
            and summary["soak_start"] is not None
            and summary["soak_peak"] > summary["soak_start"]
        )

        report = {
            "ok": ok,
            "soft_ok": soft_ok,
            "state": args.state,
            "min_stamina": min_stamina,
            "min_stamina_arg": args.min_stamina,
            "target_stamina": args.target_stamina,
            "final_stamina": final_stam.to_dict(),
            "restored": restored,
            "return_to_farm": args.return_to_farm,
            "poke": False,
            "drain": drain,
            "task": summary,
            "elapsed_s": round(time.time() - t0, 2),
            "log_tail": log[-40:],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"RESULT ok={ok} soft_ok={soft_ok} elapsed={report['elapsed_s']}s "
            f"report={args.out}"
        )
        return 0 if ok else (0 if soft_ok else 1)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
