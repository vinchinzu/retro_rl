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
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get("_SNES_IMPORT_ROOT", ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.combat.ceres_ridley import (  # noqa: E402
    ROOM_CERES_RIDLEY,
    CeresRidleyStrategy,
    play_ceres_ridley_fight,
)
from super_metroid.combat.features import ceres_ridley_catalog  # noqa: E402
from super_metroid.dev.common import save_dev_state  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR  # noqa: E402
from super_metroid.progression import MORPH_GRAPH  # noqa: E402
from super_metroid.ram import (  # noqa: E402
    ADDR_INVINCIBILITY_TIMER,
    ADDR_KNOCKBACK_TIMER,
    parse_state,
)
from super_metroid.room_timer import format_segment_time  # noqa: E402
from super_metroid.routes.kpdr.ceres.outbound import (  # noqa: E402
    play_ceres_to_ridley_door,
)
from super_metroid.routes.kpdr.early_spine import play_boot_to_ceres  # noqa: E402
from super_metroid.routes.runtime import RouteSession  # noqa: E402

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "ceres_ridley_enter.state"
DEFAULT_REPORT = (
    GAME_DIR / "scratch" / "ceres_ridley_bench.json"
)

_NAMED_STATES: dict[str, Path] = {
    "enter": DEFAULT_ENTRY,
    "entry": DEFAULT_ENTRY,
    "ceres_ridley_enter": DEFAULT_ENTRY,
}


class _Session:
    """Minimal ControllerSession for pin-local fight probes."""

    def __init__(self, env: object, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.action_reasons: Counter[str] = Counter()
        self.state = parse_state(env.get_ram(), frame=0)  # type: ignore[attr-defined]

    def step(self, action, reason: str):
        self.env.step(action)  # type: ignore[attr-defined]
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)  # type: ignore[attr-defined]
        self.assist.apply(self.env.data, self.state)  # type: ignore[attr-defined]
        self.action_reasons[reason] += 1
        return self.state


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _resolve_state(name: str) -> Path:
    key = name.strip()
    if key in _NAMED_STATES:
        return _NAMED_STATES[key]
    path = Path(key)
    if path.suffix == ".state" or "/" in key or path.exists():
        if not path.is_absolute():
            for candidate in (
                path,
                GAME_DIR / path,
                SCRATCH_STATE_DIR / path.name,
            ):
                if candidate.exists():
                    return candidate
        return path
    for candidate in (
        SCRATCH_STATE_DIR / f"{key}.state",
        GAME_DIR / "tasks" / f"{key}.state",
    ):
        if candidate.exists():
            return candidate
    return path


def _open_env(state_path: Path):
    if not state_path.exists():
        raise FileNotFoundError(
            f"Ceres Ridley enter pin not found: {state_path}\n"
            "Capture first: ceres_ridley_combat.py capture"
        )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(state_path))
    for _ in range(4):
        env.step([0] * 12)
    return env, str(state_path)


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
    text = json.dumps(report, indent=2)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


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
        session = _Session(env, assist)
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
    session = _Session(env, assist)
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
