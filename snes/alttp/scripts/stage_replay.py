"""Replay one controller from a save pin. Not a continuous run.

``TARGET`` is ``module:Name``. A ``run(env)`` or ``step(snap)`` controller
is driven here. A function whose first parameter is ``env`` is called as
``fn(env, *ARGS)`` (the room-engine edge is that shape).

``--idle N`` holds N idle frames after the pin is controllable and before
the controller (one RNG offset). ``--window A-B`` prints those controller
frames: pose, buttons, reason. ``--last-heart`` reloads a session copy of
the pin and sets current HP to one heart (``$F36D = 8``). The pin file is
not written. That poke is a what-if, not a route result.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        CastleMain alttp.opening_route.room_engine:run_room_edge room_61 west_to_0x60
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alttp.dungeon.bosses import FightResult
from alttp.ram import LINK_HP, assign_u8, wram_index
from alttp.spine.ledger import RunLedger
from alttp.startup import (
    BOOT_STATE,
    build_boot_env,
    no_action,
    snapshot_env,
)
from alttp.primitives import settle_control
from retro_harness.controls import pressed_snes_buttons
from retro_harness.input_script import FrameAction
from retro_harness.segment_runner import configure_headless

# $F36D counts in eighths. 8 is one heart, not one HP point.
ONE_HEART_HP = 0x08
_DEFAULT_STEP_FRAMES = 6000


@dataclass(frozen=True)
class ControllerRun:
    """One pin pass. ``result`` is the controller outcome.

    ``hp_lost`` and ``hits`` come from ``RunLedger``, not from ``FightResult``.
    """

    result: FightResult
    hp_lost: int
    hits: str
    room: int
    link_x: int
    link_y: int
    hp_in: Any
    hp_out: Any


def _positional(fn: Any) -> list[inspect.Parameter]:
    params = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if params and params[0].name == "self":
        return params[1:]
    return params


def _required(params: list[inspect.Parameter]) -> int:
    return sum(1 for p in params if p.default is inspect.Parameter.empty)


def _has_step(obj: Any) -> bool:
    if inspect.isfunction(obj) or inspect.ismethod(obj) or isinstance(obj, type):
        return False
    return callable(getattr(obj, "step", None))


def _buttons(action: Any) -> str:
    raw = action.action if isinstance(action, FrameAction) else action
    if hasattr(raw, "action") and not isinstance(raw, (list, tuple, np.ndarray)):
        raw = raw.action
    try:
        values = [int(v) for v in list(raw)[:12]]
    except TypeError:
        return "?"
    names = pressed_snes_buttons(values)
    return "+".join(names) if names else "-"


def _reason(action: Any) -> str:
    if isinstance(action, FrameAction):
        return str(action.reason or "")
    return str(getattr(action, "reason", "") or "")


def _read_hp(env: Any) -> int:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    return int(ram[wram_index(LINK_HP)])


def _last_heart(env: Any) -> None:
    """One heart on a reloaded copy of the pin bytes. Does not write the pin file."""
    blob = bytes(env.em.get_state())
    fd, name = tempfile.mkstemp(prefix="Whatif_last_heart_", suffix=".state")
    os.close(fd)
    path = Path(name)
    try:
        path.write_bytes(blob)
        env.em.set_state(path.read_bytes())
    finally:
        path.unlink(missing_ok=True)
    assign_u8(env, LINK_HP, ONE_HEART_HP)
    got = _read_hp(env)
    if got != ONE_HEART_HP:
        raise SystemExit(
            f"last-heart copy hp=0x{got:02X}, wanted 0x{ONE_HEART_HP:02X}"
        )
    print(f"whatif=last-heart hp=0x{got:02X} pin-file-unchanged")


def _drive(env: Any, ctl: Any, max_frames: int) -> FightResult:
    limit = max_frames if max_frames > 0 else int(getattr(ctl, "max_frames", 0) or _DEFAULT_STEP_FRAMES)
    for frame in range(1, limit + 1):
        if getattr(ctl, "success", False):
            return FightResult(True, frame - 1)
        snap = snapshot_env(env)
        act = ctl.step(snap)
        raw = act.action if isinstance(act, FrameAction) else getattr(act, "action", act)
        env.step(raw if isinstance(act, FrameAction) or hasattr(act, "action") else act)
        if getattr(ctl, "success", False):
            return FightResult(True, frame)
        if getattr(ctl, "failed", False):
            return FightResult(False, frame, "failed")
    return FightResult(bool(getattr(ctl, "success", False)), limit, "frame cap")


def _invoke(obj: Any, env: Any, extra: list[str], max_frames: int, depth: int = 0) -> Any:
    if depth > 3:
        raise SystemExit("controller factory nested too deep")
    if isinstance(obj, type):
        try:
            made = obj(*extra) if extra else obj()
        except TypeError:
            made = obj()
        return _invoke(made, env, [], max_frames, depth + 1)
    if _has_step(obj):
        return _drive(env, obj, max_frames)
    run = getattr(obj, "run", None)
    if callable(run) and not inspect.isfunction(obj):
        return run(env, *extra) if extra else run(env)
    if not callable(obj):
        raise SystemExit(f"{obj!r} is not a controller")
    params = _positional(obj)
    takes_env = bool(params) and params[0].name in {"env", "environment"}
    if takes_env:
        return obj(env, *extra)
    if _required(params) == 0 and not extra:
        return _invoke(obj(), env, [], max_frames, depth + 1)
    return obj(env, *extra)


def _load_target(target: str) -> Any:
    module_name, sep, name = target.partition(":")
    if not sep or not module_name or not name:
        raise SystemExit(f"controller must be module:Name, got {target!r}")
    return getattr(importlib.import_module(module_name), name)


def _ok_frames(result: Any) -> tuple[bool, int, str]:
    """``FightResult`` and ``SegmentResult`` both expose these attributes."""
    ok = False
    for attr in ("ok", "success"):
        if hasattr(result, attr):
            ok = bool(getattr(result, attr))
            break
    frames = int(getattr(result, "frames", 0) or 0)
    blocker = str(getattr(result, "blocker", "") or "")
    return ok, frames, blocker


def _as_fight(result: Any) -> FightResult:
    if isinstance(result, FightResult):
        return result
    ok, frames, blocker = _ok_frames(result)
    return FightResult(ok=ok, frames=frames, blocker=blocker)


def _hits_text(causes: dict[str, int]) -> str:
    if not causes:
        return "-"
    return ",".join(f"{name}:{int(count)}" for name, count in sorted(causes.items()))


def _window(text: str | None) -> tuple[int, int]:
    if not text:
        return 1, 0
    lo_s, sep, hi_s = text.partition("-")
    if not sep:
        raise SystemExit("--window must be A-B")
    return int(lo_s), int(hi_s)


def _open_pin(name: str) -> Any:
    path = Path(name)
    if path.suffix == ".state" and path.is_file():
        env = build_boot_env(BOOT_STATE)
        env.reset()
        env.em.set_state(path.read_bytes())
        return env
    env = build_boot_env(name)
    env.reset()
    return env


def _trace(env: Any, lo: int, hi: int) -> dict[str, int]:
    seen = {"frame": 0}
    inner = env.step

    def step(action: Any, *args: Any, **kwargs: Any) -> Any:
        seen["frame"] += 1
        frame = seen["frame"]
        if lo <= frame <= hi:
            snap = snapshot_env(env)
            reason = _reason(action)
            line = f"f{frame} ({snap.link_x},{snap.link_y}) {_buttons(action)}"
            if reason:
                line += f" {reason}"
            print(line, flush=True)
        return inner(action, *args, **kwargs)

    env.step = step
    return seen


def run_controller(
    pin: str,
    target: str,
    call_args: list[str],
    *,
    idle: int = 0,
    last_heart: bool = False,
    frames: int = 0,
    window: str | None = None,
) -> ControllerRun:
    """One pin, one controller. ``hp_lost`` and ``hits`` are the ledger, not the fight."""
    configure_headless()
    lo, hi = _window(window)
    env = _open_pin(pin)
    try:
        if last_heart:
            _last_heart(env)
        settle_control(env)
        for _ in range(max(0, idle)):
            env.step(no_action())
        _trace(env, lo, hi)
        snap0 = snapshot_env(env)
        ledger = RunLedger(print_on_close=False)
        ledger._hp = int(snap0.hp)
        ledger.begin_stage(target)
        ledger.attach(env)
        raw = _invoke(_load_target(target), env, list(call_args), frames)
        damage = ledger.damage_report()
        snap = snapshot_env(env)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return ControllerRun(
        result=_as_fight(raw),
        hp_lost=int(damage.get("damage") or 0),
        hits=_hits_text(dict(damage.get("hits_by_cause") or {})),
        room=int(snap.room_base_id),
        link_x=int(snap.link_x),
        link_y=int(snap.link_y),
        hp_in=damage.get("hp_in"),
        hp_out=damage.get("hp_out"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pin", help="save-state name, or a .state path")
    parser.add_argument("target", help="module:Name controller")
    parser.add_argument("call_args", nargs="*", help="extra positionals for fn(env, *args)")
    parser.add_argument("--idle", type=int, default=0, help="idle frames before the controller")
    parser.add_argument("--window", default=None, metavar="A-B", help="print controller frames A..B")
    parser.add_argument(
        "--last-heart",
        action="store_true",
        help="what-if copy: set HP to one heart; does not write the pin",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="step-controller cap (default: controller max_frames or 6000)",
    )
    args = parser.parse_args(argv)
    row = run_controller(
        args.pin,
        args.target,
        list(args.call_args),
        idle=args.idle,
        last_heart=args.last_heart,
        frames=args.frames,
        window=args.window,
    )
    result = row.result
    ok = result.ok
    frames = result.frames
    blocker = result.blocker
    print(
        f"frames={frames} success={ok} room=0x{row.room:02X} "
        f"({row.link_x},{row.link_y}) hp {row.hp_in}->{row.hp_out} hp_lost={row.hp_lost}"
    )
    if blocker and not ok:
        print(f"blocker={blocker}")
    print(f"hits_by_cause: {row.hits}")
    print(
        f"RESULT pass={1 if ok else 0} frames={frames} hp_lost={row.hp_lost} hits={row.hits}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
