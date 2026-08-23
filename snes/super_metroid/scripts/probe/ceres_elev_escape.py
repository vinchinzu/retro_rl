#!/usr/bin/env python3
"""Ceres elevator escape hop: capture enter pin, dump, climb, bench.

Hop: ``0xDF45:0xDF8D->0x91F8:0x0000`` (Ceres Elevator, Falling → Landing).
Enter pin is the first ordinary frame in the elevator after a tail-tank
countdown. Seconds use NTSC 60.0988 via ``format_segment_time``.

```bash
uv run python snes/super_metroid/scripts/probe/ceres_elev_escape.py capture
uv run python snes/super_metroid/scripts/probe/ceres_elev_escape.py dump --frames 200
uv run python snes/super_metroid/scripts/probe/ceres_elev_escape.py strategy
uv run python snes/super_metroid/scripts/probe/ceres_elev_escape.py pin
uv run python snes/super_metroid/scripts/probe/ceres_elev_escape.py bench
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env, read_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.probe import open_state_env, write_json_report
from super_metroid.combat.ceres_ridley import (
    CeresRidleyStrategy,
    play_ceres_ridley_fight,
)
from super_metroid.dev.common import save_dev_state
from super_metroid.hop_id import make_hop_key
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.progression import MORPH_GRAPH
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_arm_pump_until,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.kpdr.ceres.elev_escape import (
    _ceres_elev_leaving,
    _ceres_elev_ship_band,
    _ceres_reactive_elev_climb,
)
from super_metroid.routes.kpdr.ceres.magnet import (
    _ceres_reactive_falling,
    _ceres_reactive_magnet_escape,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_LANDING_SITE,
)
from super_metroid.routes.runtime import ActionSpan, RouteSession  # noqa: E402

RIDLEY_ENTER = SCRATCH_STATE_DIR / "ceres_ridley_enter.state"
DEFAULT_ENTRY = SCRATCH_STATE_DIR / "ceres_elev_enter.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "ceres_elev_bench.json"
HOP_KEY = make_hop_key(
    ROOM_CERES_ELEVATOR,
    from_room_id=ROOM_CERES_FALLING,
    to_room_id=ROOM_LANDING_SITE,
    items=0,
)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint="Capture first: ceres_elev_escape.py capture",
    )


def _session(env) -> RouteSession:
    return RouteSession(
        env, writer=None, assist=UnlimitedResourcesAssist(), graph=MORPH_GRAPH
    )


def _snapshot(session: RouteSession, extra: dict | None = None) -> dict[str, object]:
    state = session.state
    out: dict[str, object] = {
        "room_id_hex": f"0x{state.room_id:04X}",
        "samus_x": int(state.samus_x),
        "samus_y": int(state.samus_y),
        "pose": int(state.pose),
        "health": int(state.health),
        "game_state": int(state.game_state),
        "timer_type": int(state.timer_type),
        "escape_timer_seconds": int(state.escape_timer_seconds),
        "enemy0_x": int(state.enemy0_x),
        "enemy0_y": int(state.enemy0_y),
        "enemy0_hp": int(state.enemy0_hp),
        "num_enemies": int(state.num_enemies),
        "velocity_y": int(state.velocity_y),
    }
    if extra:
        out.update(extra)
    return out


def _print_report(report: dict, path: Path | None) -> None:
    write_json_report(report, path)


def _play_tail_tank_to_elev(session: RouteSession) -> None:
    """Ridley enter pin → tail-tank → reverse rooms → elev ordinary."""
    if session.state.room_id != ROOM_CERES_RIDLEY:
        raise RuntimeError(
            f"expected Ridley room, got 0x{session.state.room_id:04X}"
        )
    play_ceres_ridley_fight(
        session, strategy=CeresRidleyStrategy(policy="tail_tank")
    )
    for _ in range(40):
        if not is_knockback(session.state):
            break
        session.step(idle_action(), "ceres_ridley_settle")
    session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
    if session.state.room_id != ROOM_CERES_MAGNET:
        _ceres_arm_pump_until(
            session,
            "LEFT",
            reason="ceres_reverse_arm_pump",
            max_frames=700,
            done=lambda s: s.room_id == ROOM_CERES_MAGNET,
        )
    _ceres_reactive_magnet_escape(session)
    _ceres_reactive_falling(session)
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    for _ in range(160):
        st = session.state
        if (
            st.room_id == ROOM_CERES_ELEVATOR
            and st.game_state == 8
            and int(st.samus_y) >= 620
        ):
            break
        session.step(idle_action(), "ceres_elev_entry_idle")


def cmd_capture(args: argparse.Namespace) -> int:
    if not RIDLEY_ENTER.exists():
        raise FileNotFoundError(
            f"Ridley enter pin missing: {RIDLEY_ENTER}\n"
            "Capture first: ceres_ridley_combat.py capture"
        )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(RIDLEY_ENTER))
    for _ in range(4):
        env.step([0] * 12)
    try:
        session = _session(env)
        _play_tail_tank_to_elev(session)
        ok = (
            session.state.room_id == ROOM_CERES_ELEVATOR
            and session.state.game_state == 8
        )
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        if ok:
            save_dev_state(env, out)
        report = {
            "command": "capture",
            "success": ok,
            "hop_key": HOP_KEY,
            "saved_state": str(out) if ok else None,
            "entry": _snapshot(session),
            "frame": session.frame,
            "timing": format_segment_time(session.frame),
            "notes": "Tail-tank + reverse → first ordinary elev floor.",
        }
        _print_report(report, args.report)
        return 0 if ok else 1
    except Exception as exc:
        report = {
            "command": "capture",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _print_report(report, args.report)
        return 1
    finally:
        env.close()


def cmd_dump(args: argparse.Namespace) -> int:
    env, loaded = _open_env(Path(args.state) if args.state != "enter" else DEFAULT_ENTRY)
    try:
        session = _session(env)
        samples: list[dict[str, object]] = []
        for i in range(args.frames):
            snap = _snapshot(session, {"frame": session.frame, "i": i})
            if i == 0 or i + 1 == args.frames or i % 20 == 0:
                samples.append(snap)
            session.step(idle_action(), "dump_idle")
            if _ceres_elev_leaving(session.state):
                samples.append(_snapshot(session, {"frame": session.frame}))
                break
        report = {
            "command": "dump",
            "hop_key": HOP_KEY,
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


def _run_climb(env) -> tuple[RouteSession, dict]:
    session = _session(env)
    session.ceres_shaft_trace = []  # type: ignore[attr-defined]
    entry = _snapshot(session)
    start = session.frame
    error = None
    try:
        _ceres_reactive_elev_climb(session)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    frames = session.frame - start
    left = _ceres_elev_leaving(session.state) or session.state.room_id == ROOM_LANDING_SITE
    ship = _ceres_elev_ship_band(session.state)
    # Ship-band y is not leave — gs 32 / Landing is the only GREEN.
    success = left
    trace = list(getattr(session, "ceres_shaft_trace", []) or [])
    min_y = min((int(s["y"]) for s in trace), default=int(session.state.samus_y))
    return session, {
        "success": bool(success),
        "entry": entry,
        "final": _snapshot(session),
        "error": error,
        "timing": format_segment_time(frames),
        "best_y": min_y,
        "left_elev": bool(left),
        "ship_band": bool(ship),
        "trace": trace,
    }


def cmd_strategy(args: argparse.Namespace) -> int:
    state_path = Path(args.state) if args.state != "enter" else DEFAULT_ENTRY
    env, loaded = _open_env(state_path)
    try:
        _session_obj, body = _run_climb(env)
        report = {
            "command": "strategy",
            "hop_key": HOP_KEY,
            "state": loaded,
            **body,
        }
        _print_report(report, args.report)
        return 0 if body.get("success") else 1
    finally:
        env.close()


_SPIN = frozenset({25, 26, 27, 28})
_GUN = frozenset({81, 82, 83, 84, 47, 48})
_CROUCH = frozenset({37, 38, 39, 40, 41, 42})
_STAND = frozenset({1, 2, 5, 6, 7, 8, 9, 10})
_LEDGE_Y = 571
_CENTER_Y = 475


def _is_planted(state) -> bool:
    pose = int(state.pose)
    return (
        pose not in _SPIN
        and pose not in _GUN
        and abs(int(state.velocity_y)) <= 1
    )


def _sweep_hop(
    env,
    mid_bytes: bytes,
    *,
    target_y: int,
    sides: tuple[str, ...] = ("RIGHT", "LEFT"),
) -> list[dict[str, object]]:
    """From a planted land, try runup × hold × side. Record peak and higher land."""
    trials: list[dict[str, object]] = []
    src_y = None
    for side in sides:
        for runup in (0, 4, 8, 12, 16, 20):
            for hold in (8, 16, 24, 32, 40):
                env.em.set_state(mid_bytes)
                session = _session(env)
                if src_y is None:
                    src_y = int(session.state.samus_y)
                peak_y = int(session.state.samus_y)
                peak_x = int(session.state.samus_x)
                land = None
                for _ in range(16):
                    pose = int(session.state.pose)
                    if pose in _STAND:
                        break
                    if pose in _CROUCH:
                        session.step(buttons("UP"), "pin_uncrouch")
                    else:
                        # Interior of the 571 ledge is left of the x=115 plant.
                        session.step(buttons("LEFT"), "pin_stand")
                for _ in range(runup):
                    session.step(buttons(side, "B"), "pin_runup")
                for _ in range(hold):
                    session.step(buttons(side, "B", "A"), "pin_jump")
                    y = int(session.state.samus_y)
                    if y < peak_y:
                        peak_y = y
                        peak_x = int(session.state.samus_x)
                for i in range(55):
                    st = session.state
                    y = int(st.samus_y)
                    if y < peak_y:
                        peak_y = y
                        peak_x = int(st.samus_x)
                    if _is_planted(st) and y <= target_y + 16 and y < src_y - 24:
                        land = {
                            "x": int(st.samus_x),
                            "y": y,
                            "pose": int(st.pose),
                            "i": runup + hold + i,
                        }
                        break
                    if y > src_y + 30:
                        break
                    session.step(buttons(side), "pin_fall")
                trials.append(
                    {
                        "side": side,
                        "runup": runup,
                        "hold": hold,
                        "peak_x": peak_x,
                        "peak_y": peak_y,
                        "land": land,
                    }
                )
    trials.sort(
        key=lambda t: (
            t["land"] is None,
            int((t["land"] or {}).get("y", t["peak_y"])),
            int(t["peak_y"]),
        )
    )
    return trials


def _materialize_trial(env, source: bytes, trial: dict[str, object], target_y: int):
    """Replay one sweep winner and return its first planted target state."""
    env.em.set_state(source)
    session = _session(env)
    side = str(trial["side"])
    for _ in range(16):
        pose = int(session.state.pose)
        if pose in _STAND:
            break
        if pose in _CROUCH:
            session.step(buttons("UP"), "pin_win_uncrouch")
        else:
            session.step(buttons("LEFT"), "pin_win_stand")
    for _ in range(int(trial["runup"])):
        session.step(buttons(side, "B"), "pin_win_runup")
    for _ in range(int(trial["hold"])):
        session.step(buttons(side, "B", "A"), "pin_win_jump")
    for _ in range(70):
        st = session.state
        if _is_planted(st) and abs(int(st.samus_y) - target_y) <= 18:
            return {"bytes": env.em.get_state(), "snap": _snapshot(session)}
        session.step(buttons(side), "pin_win_fall")
    return None


def _sweep_ship(env, source: bytes) -> list[dict[str, object]]:
    """Top y171 landing → right-wall contact → ship-pad recipe sweep."""
    trials: list[dict[str, object]] = []
    for stand_left in (4, 8, 12):
        for idle in (0, 3, 6):
            for land_side in ("LEFT", "RIGHT", "IDLE"):
                for boost in (38,):
                    env.em.set_state(source)
                    session = _session(env)
                    best_y = int(session.state.samus_y)
                    for _ in range(stand_left):
                        session.step(buttons("LEFT"), "ship_sweep_stand")
                    seek_right = 0
                    for _ in range(40):
                        if is_knockback(session.state):
                            break
                        session.step(buttons("RIGHT"), "ship_sweep_wall")
                        seek_right += 1
                    for _ in range(idle):
                        session.step(idle_action(), "ship_sweep_idle")
                    pre = _snapshot(session, {"knockback": is_knockback(session.state)})
                    for _ in range(boost):
                        session.step(buttons("LEFT", "A"), "ship_sweep_boost")
                        best_y = min(best_y, int(session.state.samus_y))
                    for _ in range(80):
                        if _ceres_elev_leaving(session.state):
                            break
                        names = () if land_side == "IDLE" else (land_side,)
                        session.step(
                            buttons(*names) if names else idle_action(),
                            "ship_sweep_pad",
                        )
                        best_y = min(best_y, int(session.state.samus_y))
                    trials.append(
                        {
                            "stand_left": stand_left,
                            "seek_right": seek_right,
                            "idle": idle,
                            "boost": boost,
                            "land_side": land_side,
                            "pre": pre,
                            "best_y": best_y,
                            "left": _ceres_elev_leaving(session.state),
                            "final": _snapshot(session),
                        }
                    )
    trials.sort(key=lambda t: (not bool(t["left"]), int(t["best_y"])))
    return trials


def cmd_pin(args: argparse.Namespace) -> int:
    """Seat the 571 ledge, then sweep hops to 475 and the platform above it."""
    state_path = Path(args.state) if args.state != "enter" else DEFAULT_ENTRY
    env, loaded = _open_env(state_path)
    try:
        session = _session(env)
        landings: list[dict[str, object]] = []
        ledge: dict[str, object] = {}
        last_ground_y = 800
        orig_step = session.step

        def step_with_y(action, reason: str = ""):
            nonlocal last_ground_y
            result = orig_step(action, reason)
            st = session.state
            if int(st.game_state) != 8:
                return result
            y = int(st.samus_y)
            if _is_planted(st) and abs(y - last_ground_y) > 8:
                landings.append(
                    {
                        "frame": session.frame,
                        "x": int(st.samus_x),
                        "y": y,
                        "pose": int(st.pose),
                    }
                )
                last_ground_y = y
            if (
                ledge.get("bytes") is None
                and _is_planted(st)
                and abs(y - _LEDGE_Y) <= 8
                and int(st.samus_x) <= 60
            ):
                ledge["bytes"] = env.em.get_state()
                ledge["snap"] = _snapshot(session)
                raise _Seated()
            return result

        session.step = step_with_y  # type: ignore[method-assign]
        error = None
        try:
            _ceres_reactive_elev_climb(session)
        except _Seated:
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        trials_475: list[dict[str, object]] = []
        trials_next: list[dict[str, object]] = []
        next_from = None
        if ledge.get("bytes") is not None:
            trials_475 = _sweep_hop(
                env, ledge["bytes"], target_y=_CENTER_Y, sides=("RIGHT",)  # type: ignore[arg-type]
            )
            winner = next((t for t in trials_475 if t.get("land")), None)
            if winner is not None:
                next_from = _materialize_trial(
                    env, ledge["bytes"], winner, _CENTER_Y  # type: ignore[arg-type]
                )
        upper_from = None
        if next_from is not None:
            trials_next = _sweep_hop(
                env, next_from["bytes"], target_y=360, sides=("RIGHT", "LEFT")
            )
            winner = next((t for t in trials_next if t.get("land")), None)
            if winner is not None:
                upper_from = _materialize_trial(
                    env, next_from["bytes"], winner, 363
                )
        trials_upper: list[dict[str, object]] = []
        top_from = None
        if upper_from is not None:
            trials_upper = _sweep_hop(
                env, upper_from["bytes"], target_y=267, sides=("LEFT", "RIGHT")
            )
            winner = next((t for t in trials_upper if t.get("land")), None)
            if winner is not None:
                top_from = _materialize_trial(env, upper_from["bytes"], winner, 267)
        trials_top: list[dict[str, object]] = []
        ship_trials: list[dict[str, object]] = []
        if top_from is not None:
            trials_top = _sweep_hop(
                env, top_from["bytes"], target_y=171, sides=("RIGHT", "LEFT")
            )
            winner = next((t for t in trials_top if t.get("land")), None)
            if winner is not None:
                ship_from = _materialize_trial(env, top_from["bytes"], winner, 171)
                if ship_from is not None:
                    ship_trials = _sweep_ship(env, ship_from["bytes"])
        landed_475 = [t for t in trials_475 if t.get("land")]
        landed_next = [t for t in trials_next if t.get("land")]
        report = {
            "command": "pin",
            "hop_key": HOP_KEY,
            "state": loaded,
            "success": bool(landed_next),
            "error": error,
            "landings": landings,
            "ledge": ledge.get("snap"),
            "center": None if next_from is None else next_from.get("snap"),
            "upper": None if upper_from is None else upper_from.get("snap"),
            "top_approach": None if top_from is None else top_from.get("snap"),
            "best_to_475": trials_475[:8],
            "best_above_475": trials_next[:8],
            "best_above_363": trials_upper[:8],
            "best_to_top": trials_top[:8],
            "best_to_ship": ship_trials[:8],
            "n_landed_475": len(landed_475),
            "n_landed_next": len(landed_next),
            "notes": (
                "Geometry pin from first planted 571 land. "
                "Ceres Station clock starts at first elev control."
            ),
        }
        out = args.report if args.report is not None else (
            GAME_DIR / "scratch" / "ceres_elev_pin.json"
        )
        _print_report(report, out)
        return 0 if landed_next else 1
    finally:
        env.close()


class _Seated(Exception):
    """Stop the product climb once the 571 ledge is planted."""


def cmd_bench(args: argparse.Namespace) -> int:
    """Reload the same pin and run the live climb (spine body)."""
    state_path = Path(args.state) if args.state != "enter" else DEFAULT_ENTRY
    env, loaded = _open_env(state_path)
    try:
        _session_obj, body = _run_climb(env)
    finally:
        env.close()
    report = {
        "command": "bench",
        "hop_key": HOP_KEY,
        "state": loaded,
        "policy": "reactive_shaft",
        **body,
        "notes": (
            "Same elev enter pin after tail-tank. Success = ship pad or Ceres "
            "leave (gs 32) / Landing Site."
        ),
    }
    out = args.report if args.report is not None else DEFAULT_REPORT
    _print_report(report, out)
    return 0 if body.get("success") else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="Tail-tank + reverse → elev pin")
    p_cap.add_argument("--save-state", default=str(DEFAULT_ENTRY))
    p_cap.add_argument("--report", type=Path, default=None)
    p_cap.set_defaults(func=cmd_capture)

    p_dump = sub.add_parser("dump", help="Idle dump from elev enter pin")
    p_dump.add_argument("--state", default="enter")
    p_dump.add_argument("--frames", type=int, default=200)
    p_dump.add_argument("--report", type=Path, default=None)
    p_dump.set_defaults(func=cmd_dump)

    p_st = sub.add_parser("strategy", help="Run live elev climb from the pin")
    p_st.add_argument("--state", default="enter")
    p_st.add_argument("--report", type=Path, default=None)
    p_st.set_defaults(func=cmd_strategy)

    p_pin = sub.add_parser("pin", help="Land 475 then sweep the next hop")
    p_pin.add_argument("--state", default="enter")
    p_pin.add_argument("--report", type=Path, default=None)
    p_pin.set_defaults(func=cmd_pin)

    p_b = sub.add_parser("bench", help="Climb from the same pin")
    p_b.add_argument("--state", default="enter")
    p_b.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p_b.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
