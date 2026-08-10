#!/usr/bin/env python3
"""ROM probe: y=31 fence gap south transit strategies (rr-3q27 tip).

Loads ``Y1_Test_Crops_Planted_Dry``, opens the pond corridor, then trials
scripted south-cross patterns. Writes JSON under ``recordings/``.

Prefer Clean. Evidence-only — does not close beads.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.crop_planter import CropWaterTask, pond_access_blocking_fences
from harvest.tasks.farm_clearer import TILE_SIZE, get_pos_from_ram, get_tile_at, make_action
from harvest.tasks.fence_flow import (
    ACTION_CARRYING_BIT,
    ADDR_PLAYER_STATE,
    FenceClearLoopTask,
)

DEFAULT_STATE = "Y1_Test_Crops_Planted_Dry"
DEFAULT_OUT = PROJECT_DIR / "recordings" / "gap_transit_probe.json"
ADDR_INPUT_LOCK = 0x019A


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _world(env, frame: int) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)


def _tile(ram) -> Tuple[int, int]:
    pos = get_pos_from_ram(ram)
    return (int(pos.x) // TILE_SIZE, int(pos.y) // TILE_SIZE)


def _px(ram) -> Tuple[int, int]:
    pos = get_pos_from_ram(ram)
    return (int(pos.x), int(pos.y))


def _carrying(ram) -> bool:
    return bool(int(ram[ADDR_PLAYER_STATE]) & ACTION_CARRYING_BIT)


def _gap_xs(ram) -> List[int]:
    """x columns on y=31 that are not fence (0x05)."""
    open_xs: List[int] = []
    for x in range(11, 30):
        tid = int(get_tile_at(ram, x, 31))
        if tid != 0x05:
            open_xs.append(x)
    return open_xs


def _snap(env, frame: int, note: str = "") -> dict[str, Any]:
    ram = env.get_ram()
    tx, ty = _tile(ram)
    px, py = _px(ram)
    return {
        "f": frame,
        "note": note,
        "tile": [tx, ty],
        "px": [px, py],
        "carry": _carrying(ram),
        "gap_xs": _gap_xs(ram),
        "fences": len(pond_access_blocking_fences(ram)),
        "can": int(CropWaterTask._water_level(ram)),
        "input_lock": int(ram[ADDR_INPUT_LOCK]) if len(ram) > ADDR_INPUT_LOCK else -1,
    }


def _step(env, action, frame: int) -> int:
    env.step(action)
    return frame + 1


def _idle(env, frame: int, n: int = 8) -> int:
    z = make_action()
    for _ in range(n):
        frame = _step(env, z, frame)
    return frame


def _hold(env, frame: int, n: int, **buttons) -> int:
    a = make_action(**buttons)
    for _ in range(n):
        frame = _step(env, a, frame)
    return frame


def _wait_input(env, frame: int, max_frames: int = 120) -> int:
    z = make_action()
    for _ in range(max_frames):
        ram = env.get_ram()
        if int(ram[ADDR_INPUT_LOCK]) == 1 and not (
            int(ram[ADDR_PLAYER_STATE]) & 0x08  # anim-ish
        ):
            # Also mash A/B briefly if locked.
            break
        if int(ram[ADDR_INPUT_LOCK]) != 1:
            frame = _step(env, make_action(a=True) if frame % 2 == 0 else make_action(b=True), frame)
        else:
            frame = _step(env, z, frame)
    return frame


def _walk_to_tile(
    env,
    frame: int,
    target: Tuple[int, int],
    *,
    timeout: int = 900,
    b_run: bool = True,
) -> Tuple[int, bool]:
    """Greedy pixel walk toward tile center. Returns (frame, reached)."""
    tgt_x = target[0] * TILE_SIZE + 8
    tgt_y = target[1] * TILE_SIZE + 8
    last = _tile(env.get_ram())
    stasis = 0
    for _ in range(timeout):
        ram = env.get_ram()
        if int(ram[ADDR_INPUT_LOCK]) != 1:
            frame = _step(
                env,
                make_action(a=True) if frame % 2 == 0 else make_action(b=True),
                frame,
            )
            continue
        px, py = _px(ram)
        if abs(px - tgt_x) <= 2 and abs(py - tgt_y) <= 2:
            return frame, True
        cur = _tile(ram)
        if cur == last:
            stasis += 1
        else:
            stasis = 0
            last = cur
        if stasis > 180:
            return frame, False
        dx = tgt_x - px
        dy = tgt_y - py
        kwargs: dict[str, bool] = {}
        if abs(dx) >= abs(dy):
            kwargs["right" if dx > 0 else "left"] = True
        else:
            kwargs["down" if dy > 0 else "up"] = True
        if b_run:
            kwargs["b"] = True
        frame = _step(env, make_action(**kwargs), frame)
    return frame, False


def _run_fence_open(
    env,
    frame: int,
    *,
    max_fences: int = 2,
    timeout: int = 5000,
    corridor_only: bool = True,
) -> Tuple[int, dict[str, Any]]:
    task = FenceClearLoopTask(
        max_fences=max_fences,
        max_steps_per_fence=1600,
        corridor_only=corridor_only,
        debug=False,
    )
    w = _world(env, frame)
    task.reset(w)
    start = frame
    last_status = "running"
    while frame - start < timeout:
        w = _world(env, frame)
        result = task.step(w)
        last_status = result.status.name if hasattr(result.status, "name") else str(result.status)
        if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.BLOCKED):
            break
        action = result.action.action if result.action is not None else make_action()
        frame = _step(env, action, frame)
    return frame, {
        "status": last_status,
        "cleared": int(getattr(task, "cleared_count", 0)),
        "state": getattr(task, "_state", None),
        "snap": _snap(env, frame, "after_fence_open"),
    }


def _drop_local(env, frame: int) -> int:
    """Multi-face A-drop to clear hands."""
    for face in ("down", "left", "right", "up"):
        frame = _hold(env, frame, 8, **{face: True})
        frame = _hold(env, frame, 16, **{face: True, "a": True})
        frame = _idle(env, frame, 10)
        if not _carrying(env.get_ram()):
            break
    return frame


StrategyFn = Callable[[Any, int], Tuple[int, dict[str, Any]]]


def strategy_empty_charge_x12(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """Re-seat (12,29) then pure B-down charge."""
    log: List[dict[str, Any]] = []
    if _carrying(env.get_ram()):
        frame = _drop_local(env, frame)
    frame, ok = _walk_to_tile(env, frame, (12, 29), timeout=1200)
    log.append(_snap(env, frame, f"at_12_29 ok={ok}"))
    # Nudge up then long south charge
    frame = _hold(env, frame, 20, up=True)
    frame = _idle(env, frame, 6)
    frame = _hold(env, frame, 160, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_empty_charge_x12")
    log.append(snap)
    return frame, {
        "name": "empty_charge_x12",
        "crossed": snap["tile"][1] >= 32,
        "log": log,
    }


def strategy_empty_charge_x13(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """Re-seat (13,29) then pure B-down charge (often soft-blocks)."""
    log: List[dict[str, Any]] = []
    if _carrying(env.get_ram()):
        frame = _drop_local(env, frame)
    frame, ok = _walk_to_tile(env, frame, (13, 29), timeout=1200)
    log.append(_snap(env, frame, f"at_13_29 ok={ok}"))
    frame = _hold(env, frame, 20, up=True)
    frame = _idle(env, frame, 6)
    frame = _hold(env, frame, 160, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_empty_charge_x13")
    log.append(snap)
    return frame, {
        "name": "empty_charge_x13",
        "crossed": snap["tile"][1] >= 32,
        "log": log,
    }


def strategy_empty_charge_align_gap(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """Align to first gap x at y=29 then charge; if stuck on y=31, strafe+down."""
    log: List[dict[str, Any]] = []
    if _carrying(env.get_ram()):
        frame = _drop_local(env, frame)
    gaps = _gap_xs(env.get_ram())
    gx = gaps[0] if gaps else 12
    frame, ok = _walk_to_tile(env, frame, (gx, 29), timeout=1200)
    log.append(_snap(env, frame, f"at_gap_x29 x={gx} ok={ok}"))
    frame = _hold(env, frame, 24, up=True)
    frame = _idle(env, frame, 6)
    # Charge with brief left/right wiggle if stasis on y=31
    for i in range(200):
        ram = env.get_ram()
        ty = _tile(ram)[1]
        if ty >= 32:
            break
        if ty == 31 and i > 40 and i % 40 < 8:
            # strafe one tile west then resume down
            frame = _hold(env, frame, 1, left=True, b=True)
        elif ty == 31 and i > 40 and i % 40 < 16:
            frame = _hold(env, frame, 1, right=True, b=True)
        else:
            frame = _hold(env, frame, 1, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_align_gap_charge")
    log.append(snap)
    return frame, {
        "name": "empty_charge_align_gap_wiggle",
        "crossed": snap["tile"][1] >= 32,
        "gap_x": gx,
        "log": log,
    }


def strategy_double_gap_then_charge(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """If only 1 gap open, clear another adjacent fence, then charge x12."""
    log: List[dict[str, Any]] = []
    gaps = _gap_xs(env.get_ram())
    log.append(_snap(env, frame, f"pre_double gaps={gaps}"))
    if len(gaps) < 2:
        # Force another corridor clear (not corridor_only success-after-1)
        frame, info = _run_fence_open(env, frame, max_fences=2, timeout=4500)
        log.append({"fence2": info})
        if _carrying(env.get_ram()):
            frame = _drop_local(env, frame)
    gaps = _gap_xs(env.get_ram())
    log.append(_snap(env, frame, f"post_double gaps={gaps}"))
    # Prefer contiguous pair
    charge_x = 12
    for x in gaps:
        if (x + 1) in gaps:
            charge_x = x
            break
        charge_x = x
    frame, ok = _walk_to_tile(env, frame, (charge_x, 29), timeout=1200)
    log.append(_snap(env, frame, f"at_charge ok={ok} x={charge_x}"))
    frame = _hold(env, frame, 24, up=True)
    frame = _idle(env, frame, 6)
    frame = _hold(env, frame, 180, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_double_gap_charge")
    log.append(snap)
    return frame, {
        "name": "double_gap_then_charge",
        "crossed": snap["tile"][1] >= 32,
        "gaps": gaps,
        "charge_x": charge_x,
        "log": log,
    }


def strategy_carry_south_from_gap(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """Pick nearest fence, carry, long south charge while holding post."""
    log: List[dict[str, Any]] = []
    # If not carrying, lift nearest fence on y=31 near player
    if not _carrying(env.get_ram()):
        if _tile(env.get_ram())[1] > 30:
            frame, _ = _walk_to_tile(env, frame, (12, 29), timeout=900)
        # Find fence adjacent
        ram = env.get_ram()
        px, py = _tile(ram)
        fences = pond_access_blocking_fences(ram)
        if not fences:
            return frame, {"name": "carry_south", "crossed": False, "reason": "no_fence"}
        target = min(fences, key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        # Approach north of fence
        approach = (target[0], target[1] - 1)
        frame, ok = _walk_to_tile(env, frame, approach, timeout=1200)
        log.append(_snap(env, frame, f"approach {approach} ok={ok}"))
        # Face down + A lift
        frame = _hold(env, frame, 12, down=True)
        frame = _hold(env, frame, 28, down=True, a=True)
        frame = _idle(env, frame, 30)
        frame = _wait_input(env, frame)
        log.append(_snap(env, frame, "after_lift"))
    # South charge while carrying
    for i in range(180):
        ram = env.get_ram()
        if int(ram[ADDR_INPUT_LOCK]) != 1:
            frame = _step(
                env,
                make_action(a=True) if frame % 2 == 0 else make_action(b=True),
                frame,
            )
            continue
        if _tile(ram)[1] >= 32:
            break
        frame = _hold(env, frame, 1, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_carry_south")
    log.append(snap)
    crossed = snap["tile"][1] >= 32
    if crossed and _carrying(env.get_ram()):
        frame = _drop_local(env, frame)
        log.append(_snap(env, frame, "dropped_south"))
    return frame, {
        "name": "carry_south_from_gap",
        "crossed": crossed,
        "log": log,
    }


def strategy_pixel_nudge_then_charge(env, frame: int) -> Tuple[int, dict[str, Any]]:
    """From north of gap: align pixel x to gap center, then down with no B first."""
    log: List[dict[str, Any]] = []
    if _carrying(env.get_ram()):
        frame = _drop_local(env, frame)
    gaps = _gap_xs(env.get_ram())
    gx = gaps[0] if gaps else 12
    frame, ok = _walk_to_tile(env, frame, (gx, 29), timeout=1200)
    log.append(_snap(env, frame, f"align ok={ok}"))
    # Center pixel on gap
    tgt_x = gx * TILE_SIZE + 8
    for _ in range(40):
        px, _py = _px(env.get_ram())
        if abs(px - tgt_x) <= 1:
            break
        if px < tgt_x:
            frame = _hold(env, frame, 1, right=True)
        else:
            frame = _hold(env, frame, 1, left=True)
    frame = _idle(env, frame, 8)
    # Walk without B first (slow) then B
    frame = _hold(env, frame, 80, down=True)
    frame = _hold(env, frame, 100, down=True, b=True)
    frame = _idle(env, frame, 10)
    snap = _snap(env, frame, "after_pixel_charge")
    log.append(snap)
    return frame, {
        "name": "pixel_nudge_then_charge",
        "crossed": snap["tile"][1] >= 32,
        "gap_x": gx,
        "log": log,
    }


STRATEGIES: Sequence[Tuple[str, StrategyFn]] = (
    ("empty_charge_x12", strategy_empty_charge_x12),
    ("empty_charge_x13", strategy_empty_charge_x13),
    ("empty_charge_align_gap_wiggle", strategy_empty_charge_align_gap),
    ("double_gap_then_charge", strategy_double_gap_then_charge),
    ("carry_south_from_gap", strategy_carry_south_from_gap),
    ("pixel_nudge_then_charge", strategy_pixel_nudge_then_charge),
)


def main() -> int:
    _configure_headless()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--strategies",
        default="all",
        help="Comma list of strategy names or 'all'",
    )
    p.add_argument("--max-fences", type=int, default=2)
    p.add_argument("--skip-fence-open", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    env = make_harvest_env(args.state, require_rom=True)
    frame = 0
    env.reset()
    frame = _idle(env, frame, 30)

    report: dict[str, Any] = {
        "state": args.state,
        "start": _snap(env, frame, "start"),
        "fence_open": None,
        "strategies": [],
        "winners": [],
    }

    if not args.skip_fence_open:
        # Stage to (12,29) first
        frame, ok = _walk_to_tile(env, frame, (12, 29), timeout=1500)
        report["staged"] = _snap(env, frame, f"staged ok={ok}")
        frame, info = _run_fence_open(
            env, frame, max_fences=args.max_fences, timeout=5500
        )
        report["fence_open"] = info
        if _carrying(env.get_ram()):
            frame = _drop_local(env, frame)
        # Re-seat north
        frame = _hold(env, frame, 48, up=True)
        frame = _idle(env, frame, 12)
        report["after_reseat"] = _snap(env, frame, "after_reseat")

    # Snapshot env state for reloads between strategies
    try:
        base_state = env.em.get_state()
    except Exception:
        base_state = None

    wanted = (
        [n for n, _ in STRATEGIES]
        if args.strategies.strip().lower() == "all"
        else [s.strip() for s in args.strategies.split(",") if s.strip()]
    )
    name_to_fn = {n: fn for n, fn in STRATEGIES}

    for name in wanted:
        fn = name_to_fn.get(name)
        if fn is None:
            report["strategies"].append({"name": name, "error": "unknown"})
            continue
        if base_state is not None:
            env.em.set_state(base_state)
            frame = _idle(env, frame, 8)
        try:
            frame, result = fn(env, frame)
        except Exception as exc:  # pragma: no cover - ROM probe
            result = {"name": name, "crossed": False, "error": repr(exc)}
        report["strategies"].append(result)
        if result.get("crossed"):
            report["winners"].append(name)
            print(f"[GAP] WIN {name} at {_tile(env.get_ram())}")
        else:
            print(f"[GAP] FAIL {name} at {_tile(env.get_ram())} snap={result.get('log', result)[-1] if result.get('log') else result}")

    report["wall_s"] = round(time.time() - t0, 2)
    report["any_cross"] = bool(report["winners"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "winners": report["winners"],
        "fence": report.get("fence_open"),
        "after_reseat": report.get("after_reseat"),
        "out": str(args.out),
        "wall_s": report["wall_s"],
    }, indent=2))
    env.close()
    return 0 if report["any_cross"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
