"""Pickup-to-ship frame bench: run ``HarvestTask`` alone on a farm pin and print its ship log.

The pin must stand on the farm with ripe crops (``LN_saddle_whatif_harvest_y2d22``: Y2 Summer D22
12:11, 48 ripe). Nothing is written to RAM; the task drives the pad from the pin's frame on.

    HEADLESS=1 .venv/bin/python -m harvest.scripts.ship_bench LN_saddle_whatif_harvest_y2d22 \
        --frames 6000 --trace 2 --out recordings/ship_bench/<tag>.json

``--trace N`` keeps a per-frame row (pad, carry bit, action, pixel) for the first N crops, which is
where the pick and throw frames go. ``harvest.tools.ship_ledger`` reads the ``--out`` file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(pin: str, frames: int, trace_crops: int, exit_house: bool = False) -> dict:
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.harvest_task import SHIP_LOG_SCHEMA, HarvestTask, clock_tick
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.ship_bench"))
    trace: list[dict] = []
    try:
        env.reset()
        if exit_house:
            from harvest.planner.tasks.inventory_exit import ExitToFarmTask

            leave = ExitToFarmTask()
            leave.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
            for _ in range(6000):
                result = leave.step(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action is not None else make_action())
        ram = env.get_ram()
        start = {"ship_raw": int(read_ram_value(ram, "shipping_money", raw=True)), "tick": clock_tick(ram)}
        task = HarvestTask()
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        status, used, reason = TaskStatus.RUNNING, 0, ""
        for used in range(frames):
            ram = env.get_ram()
            result = task.step(WorldState(frame=used, ram=ram, info={}, obs=None))
            status, reason = result.status, result.reason or ""
            if status != TaskStatus.RUNNING:
                break
            action = result.action.action if result.action is not None else make_action()
            if len(task.ship_log) < trace_crops:
                trace.append({
                    "f": used, "pad": [int(v) for v in action], "phase": task._phase,
                    "carry": int(read_ram_value(ram, "player_state", raw=True)) & 0x02,
                    "act": int(read_ram_value(ram, "player_action", raw=True)),
                    "x": int(read_ram_value(ram, "player_x", raw=True)),
                    "y": int(read_ram_value(ram, "player_y", raw=True)),
                    "tick": clock_tick(ram),
                })
            env.step(action)
        ram = env.get_ram()
        audit = env.audit()
        return {
            "schema": SHIP_LOG_SCHEMA, "pin": pin, "status": str(status), "reason": reason, "frames": used,
            "shipped_g": (int(read_ram_value(ram, "shipping_money", raw=True)) - start["ship_raw"]) * 10,
            "ticks": clock_tick(ram) - start["tick"],
            "ship_log": task.ship_log, "trace": trace,
            "audit": {"ram_writes": audit.ram_writes, "mid_run_loads": audit.mid_run_loads},
        }
    finally:
        env.close()


def summary(report: dict) -> str:
    log = report["ship_log"]
    lines = [f"{report['pin']}: {len(log)} shipped, {report['shipped_g']}G, {report['frames']} frames, "
             f"{report['ticks']} clock ticks, {report['status']} {report['reason'][:80]}"]
    prev = None
    for rec in log:
        start = rec["f_select"] if prev is None else prev["f_ship"]
        lines.append(
            f"  {rec['crop']:6s} {tuple(rec['target'])} stand={tuple(rec.get('stand', ()))} "
            f"ship={tuple(rec.get('ship_stand', ()))}{rec.get('ship_face', '')[:1]} "
            f"approach={rec['f_pick'] - rec['f_select']:4d} carry={rec['f_ship'] - rec['f_pick']:4d} "
            f"cycle={rec['f_ship'] - start:4d} paid={rec['paid_g']}")
        prev = rec
    if log:
        cycle = (log[-1]["f_ship"] - log[0]["f_select"]) / len(log)
        lines.append(f"  mean cycle {cycle:.0f} frames/crop")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pin")
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--trace", type=int, default=0, help="per-frame rows for the first N crops")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exit-house", action="store_true", help="the pin is a morning in the house: walk out first")
    args = parser.parse_args()
    report = run(args.pin, args.frames, args.trace, args.exit_house)
    print(summary(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
