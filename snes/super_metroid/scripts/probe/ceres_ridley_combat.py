#!/usr/bin/env python3
"""Ceres Ridley combat probe: capture enter pin, dump, strategy, bench.

Always print segment time as frames + seconds (@ 60.0988) + mm:ss.cc.
Bench runs **wait** then **tail_tank** from the same pin.

```bash
# Power-on → Ridley door; write scratch pin
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py capture

# Idle dump (geometry / hits)
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py dump --frames 400

# One policy from the pin
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py strategy --policy wait
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py strategy --policy tail_tank

# Before/after from the same pin
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py bench
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.env import make_env
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.ceres_ridley import (
    ROOM_CERES_RIDLEY,
    CeresRidleyStrategy,
    play_ceres_ridley_fight,
)
from super_metroid.combat.features import ceres_ridley_catalog
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.common import save_dev_state
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.progression import MORPH_GRAPH
from super_metroid.ram import ADDR_INVINCIBILITY_TIMER, ADDR_KNOCKBACK_TIMER
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.ceres.outbound import play_ceres_to_ridley_door
from super_metroid.routes.kpdr.early_spine import play_boot_to_ceres
from super_metroid.routes.runtime import RouteSession

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "ceres_ridley_enter.state"
DEFAULT_REPORT = (
    GAME_DIR / "scratch" / "ceres_ridley_bench.json"
)

_NAMED_STATES: dict[str, Path] = {
    "enter": DEFAULT_ENTRY,
    "entry": DEFAULT_ENTRY,
    "ceres_ridley_enter": DEFAULT_ENTRY,
}


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint="Capture first: ceres_ridley_combat.py capture",
    )


def _snapshot(session, extra: dict | None = None) -> dict[str, object]:
    state = session.state
    env = getattr(session, "env", None)
    invuln = kb = 0
    if env is not None:
        ram = env.get_ram()
        invuln = _u16(ram, ADDR_INVINCIBILITY_TIMER)
        kb = _u16(ram, ADDR_KNOCKBACK_TIMER)
    out: dict[str, object] = {
        "room_id_hex": f"0x{state.room_id:04X}",
        "samus_x": state.samus_x,
        "samus_y": state.samus_y,
        "pose": state.pose,
        "health": state.health,
        "max_health": state.max_health,
        "timer_type": state.timer_type,
        "escape_timer_seconds": state.escape_timer_seconds,
        "enemy0_x": state.enemy0_x,
        "enemy0_y": state.enemy0_y,
        "enemy0_hp": state.enemy0_hp,
        "enemy0_spritemap": f"0x{state.enemy0_spritemap:04X}",
        "invuln": invuln,
        "knockback_timer": kb,
        "game_state": state.game_state,
    }
    if extra:
        out.update(extra)
    return out


def _print_report(report: dict, path: Path | None) -> None:
    write_json_report(report, path)


def cmd_capture(args: argparse.Namespace) -> int:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    assist = UnlimitedResourcesAssist()
    try:
        session = RouteSession(env, writer=None, assist=assist, graph=MORPH_GRAPH)
        play_boot_to_ceres(session)
        play_ceres_to_ridley_door(session)
        if session.state.room_id != ROOM_CERES_RIDLEY or session.state.game_state != 8:
            report = {
                "command": "capture",
                "success": False,
                "outcome": "not_in_ridley_room",
                "final": _snapshot(session),
                "frame": session.frame,
            }
            _print_report(report, args.report)
            return 1
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        save_dev_state(env, out)
        report = {
            "command": "capture",
            "success": True,
            "saved_state": str(out),
            "entry": _snapshot(session),
            "frame": session.frame,
            "timing": format_segment_time(session.frame),
            "notes": "Power-on → Ridley ordinary settle. Fight not played.",
        }
        _print_report(report, args.report)
        return 0
    finally:
        env.close()


def cmd_dump(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    assist = UnlimitedResourcesAssist()
    try:
        session = ProbeSession(env, assist)
        samples: list[dict[str, object]] = []
        last_health = session.state.health
        for i in range(args.frames):
            snap = _snapshot(session, {"frame": session.frame, "i": i})
            if i == 0 or i + 1 == args.frames or session.state.health != last_health:
                samples.append(snap)
            last_health = session.state.health
            session.step([0] * 12, "dump_idle")
            if session.state.timer_type == 3:
                samples.append(_snapshot(session, {"frame": session.frame, "i": i + 1}))
                break
        report = {
            "command": "dump",
            "state": loaded,
            "entry": samples[0] if samples else _snapshot(session),
            "samples": samples,
            "final": _snapshot(session),
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
        }
        _print_report(report, args.report)
        return 0
    finally:
        env.close()


def _run_policy(env, policy: str, max_frames: int):
    assist = UnlimitedResourcesAssist()
    session = ProbeSession(env, assist)
    if session.state.room_id != ROOM_CERES_RIDLEY:
        return session, None, {
            "success": False,
            "outcome": "wrong_room",
            "room_id_hex": f"0x{session.state.room_id:04X}",
        }
    entry = _snapshot(session)
    evidence = play_ceres_ridley_fight(
        session,
        strategy=CeresRidleyStrategy(policy=policy, max_fight_frames=max_frames),
    )
    payload = evidence.to_dict()
    return session, evidence, {
        "success": evidence.outcome == "ceres_ridley_countdown",
        "entry": entry,
        "fight": payload,
        "final": _snapshot(session),
        "timing": {
            "frames": payload["action_frames"],
            "seconds": payload["seconds"],
            "clock": payload["clock"],
            "ntsc_fps": payload["ntsc_fps"],
        },
    }


def cmd_strategy(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    try:
        _session, _ev, body = _run_policy(env, args.policy, args.max_frames)
        report = {
            "command": "strategy",
            "state": loaded,
            "policy": args.policy,
            "catalog": ceres_ridley_catalog().boss_id,
            **body,
        }
        _print_report(report, args.report)
        return 0 if body.get("success") else 1
    finally:
        env.close()


def cmd_bench(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    policies = ("wait", "tail_tank")
    rows: dict[str, dict] = {}
    for policy in policies:
        env, loaded = _open_env(state_path)
        try:
            _session, _ev, body = _run_policy(env, policy, args.max_frames)
            rows[policy] = {"state": loaded, **body}
        finally:
            env.close()

    before = rows["wait"]["timing"]
    after = rows["tail_tank"]["timing"]
    delta_frames = int(after["frames"]) - int(before["frames"])
    delta = format_segment_time(abs(delta_frames))
    report = {
        "command": "bench",
        "state": str(state_path),
        "before": {"policy": "wait", **rows["wait"]},
        "after": {"policy": "tail_tank", **rows["tail_tank"]},
        "delta": {
            "frames": delta_frames,
            "seconds": round(delta_frames / float(delta["ntsc_fps"]), 3),
            "clock": ("-" if delta_frames < 0 else "+") + str(delta["clock"]),
            "ntsc_fps": delta["ntsc_fps"],
        },
        "winner": "tail_tank" if delta_frames < 0 else "wait",
        "notes": (
            "Same enter pin. Negative delta = tail_tank is faster. "
            "wiki.supermetroid.run/Ridley#Ceres_Station: five tail hits."
        ),
    }
    out = args.report if args.report is not None else DEFAULT_REPORT
    _print_report(report, out)
    before_ok = rows["wait"].get("success")
    after_ok = rows["tail_tank"].get("success")
    return 0 if before_ok and after_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="Power-on → Ridley door pin")
    p_cap.add_argument("--save-state", default=str(DEFAULT_ENTRY))
    p_cap.add_argument("--report", type=Path, default=None)
    p_cap.set_defaults(func=cmd_capture)

    p_dump = sub.add_parser("dump", help="Idle dump from enter pin")
    p_dump.add_argument("--state", default="enter")
    p_dump.add_argument("--frames", type=int, default=400)
    p_dump.add_argument("--report", type=Path, default=None)
    p_dump.set_defaults(func=cmd_dump)

    p_st = sub.add_parser("strategy", help="Run one policy from the enter pin")
    p_st.add_argument("--state", default="enter")
    p_st.add_argument("--policy", choices=("wait", "tail_tank"), default="tail_tank")
    p_st.add_argument("--max-frames", type=int, default=6_000)
    p_st.add_argument("--report", type=Path, default=None)
    p_st.set_defaults(func=cmd_strategy)

    p_b = sub.add_parser("bench", help="wait vs tail_tank from the same pin")
    p_b.add_argument("--state", default="enter")
    p_b.add_argument("--max-frames", type=int, default=6_000)
    p_b.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p_b.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
