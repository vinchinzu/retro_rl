#!/usr/bin/env python3
"""Probe Phantoon loot + left-door exit to WS Basement.

Default pin is the in-room kill leave. Dual reloads the pin.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/phantoon_leave.py leave --assist \
  --report snes/super_metroid/scratch/phantoon_loot_exit.json --save-state
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.phantoon import list_pickups
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.common import save_dev_state
from super_metroid.dev.phantoon_dev import phantoon_defeated, wrecked_ship_boss_bits
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.k6.phantoon_leave import play_phantoon_loot_exit
from super_metroid.routes.kpdr.room_ids import ROOM_WS_BASEMENT

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_phantoon_poweron.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_phantoon_leave.state"

_NAMED_STATES: dict[str, Path] = {
    "kill": DEFAULT_ENTRY,
    "post_phantoon_poweron": DEFAULT_ENTRY,
    "natural": DEFAULT_ENTRY,
}


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint="Need scratch/post_phantoon_poweron.state (in-room kill leave).",
    )


def _snapshot(session: ProbeSession) -> dict[str, object]:
    st = session.state
    return {
        "room_id_hex": f"0x{st.room_id:04X}",
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "facing": st.facing,
        "health": st.health,
        "game_state": st.game_state,
        "door_transition": st.door_transition,
        "missiles": st.missiles,
        "super_missiles": st.super_missiles,
        "selected_item": st.selected_item,
        "enemy0_hp": st.enemy0_hp,
        "pickups": [p.__dict__ for p in list_pickups(session.env)],
    }


def _run_leave(session: ProbeSession) -> dict[str, object]:
    entry = _snapshot(session)
    play_phantoon_loot_exit(session)
    timing = format_segment_time(session.frame)
    st = session.state
    success = (
        int(st.room_id) == ROOM_WS_BASEMENT
        and int(st.game_state) == 8
        and int(st.door_transition) == 0
        and phantoon_defeated(session.env)
    )
    return {
        "success": success,
        "entry": entry,
        "final": _snapshot(session),
        "timing": timing,
        "frames": timing["frames"],
        "seconds": timing["seconds"],
        "clock": timing["clock"],
        "boss_bits_wrecked_ship": wrecked_ship_boss_bits(session.env),
        "phantoon_defeated": phantoon_defeated(session.env),
        "reasons": dict(session.action_reasons),
    }


def cmd_leave(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(
        unlimited_energy=args.assist,
        unlimited_ammo=args.assist,
    )
    try:
        session = ProbeSession(env, assist)
        result = _run_leave(session)
        out_path = None
        if result["success"] and args.save_state:
            out = Path(args.save_state) if args.save_state is not True else DEFAULT_OUT
            out.parent.mkdir(parents=True, exist_ok=True)
            save_dev_state(env, out)
            out_path = out
        report = {
            "command": "leave",
            "state": loaded,
            "assist_enabled": bool(args.assist),
            "saved_state": str(out_path) if out_path is not None else None,
            **result,
        }
        write_json_report(report, args.report)
        return 0 if result["success"] else 1
    finally:
        env.close()


def cmd_bench(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    runs = []
    for i in range(2):
        env, loaded = _open_env(state_path)
        assist = UnlimitedResourcesAssist(
            unlimited_energy=args.assist,
            unlimited_ammo=args.assist,
        )
        try:
            session = ProbeSession(env, assist)
            result = _run_leave(session)
            result["run"] = i + 1
            result["state"] = loaded
            runs.append(result)
        finally:
            env.close()
    success = all(r["success"] for r in runs) and runs[0]["frames"] == runs[1]["frames"]
    report = {
        "command": "bench",
        "success": success,
        "assist_enabled": bool(args.assist),
        "frames": [r["frames"] for r in runs],
        "seconds": [r["seconds"] for r in runs],
        "clock": [r["clock"] for r in runs],
        "runs": runs,
    }
    write_json_report(report, args.report)
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    for name, func in (("leave", cmd_leave), ("bench", cmd_bench)):
        p = sub.add_parser(name)
        p.add_argument("--state", default="kill")
        p.add_argument("--assist", action="store_true")
        p.add_argument("--report", type=Path, default=None)
        if name == "leave":
            p.add_argument("--save-state", nargs="?", const=True, default=False)
        p.set_defaults(func=func)
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
