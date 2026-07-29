#!/usr/bin/env python3
"""Capture and probe Mother Brain / Tourian escape development states.

Examples:

```bash
uv run python super_metroid/scripts/probe/mother_brain.py capture-mb
uv run python super_metroid/scripts/probe/mother_brain.py capture-escape1
uv run python super_metroid/scripts/probe/mother_brain.py spray-mb --frames 3600
uv run python super_metroid/scripts/probe/mother_brain.py run-escape --frames 12000
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes, write_state_bytes  # noqa: E402
from super_metroid.dev.mother_brain_dev import (  # noqa: E402
    ESCAPE1_STATE,
    MB_ENTRY_STATE,
    ROOM_ESCAPE_1,
    ROOM_ESCAPE_2,
    ROOM_ESCAPE_3,
    ROOM_ESCAPE_4,
    ROOM_LANDING_SITE,
    ROOM_MOTHER_BRAIN,
    apply_dev_survivability,
    capture_escape_room1,
    capture_mother_brain_entry,
    place_samus,
    start_escape_timer,
)
from super_metroid.paths import GAME, GAME_DIR  # noqa: E402
from super_metroid.ram import GameplayPhase, parse_env_state, write_wram_u16  # noqa: E402


def _screenshot(env, path: Path) -> None:
    import cv2

    obs = env.get_screen() if hasattr(env, "get_screen") else None
    if obs is None:
        obs, *_ = env.step(idle_action())
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))


def _enemy_hps(env, n: int = 6) -> list[int]:
    ram = env.get_ram()
    out = []
    for i in range(n):
        o = 0x0F8C + i * 0x40
        out.append(int(ram[o]) | (int(ram[o + 1]) << 8))
    return out


def spray_mb(*, frames: int, state_path: Path) -> dict[str, object]:
    """Spray missiles/supers at Mother Brain from a captured entry state."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    rooms: list[tuple] = []
    last_room = None
    min_body = 10**9
    min_brain = 10**9
    try:
        env.reset()
        env.em.set_state(read_state_bytes(state_path))
        # Stay off the right-door edge.
        place_samus(env, 860, 180)
        for frame in range(frames):
            state = parse_env_state(env, frame=frame)
            apply_dev_survivability(env)
            hps = _enemy_hps(env)
            min_body = min(min_body, hps[0])
            min_brain = min(min_brain, hps[1] if len(hps) > 1 else min_brain)
            if state.room_id != last_room:
                rooms.append(
                    (
                        frame,
                        f"0x{state.room_id:04X}",
                        state.samus_x,
                        state.samus_y,
                        state.phase.value,
                    )
                )
                last_room = state.room_id
            if state.phase is GameplayPhase.ENDING_OR_CREDITS:
                break
            if state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
                break

            # Pulse supers while near zebetites; face left without jamming.
            if state.pose in (0x8A, 0x89):  # ran into wall
                write_wram_u16(env, 0x09D2, 2)
                action = buttons("X") if frame % 6 < 3 else buttons("RIGHT")
            elif state.samus_x > 300:
                write_wram_u16(env, 0x09D2, 2)
                if frame % 20 < 6:
                    action = buttons("LEFT", "B")
                elif frame % 8 < 3:
                    action = buttons("LEFT", "X")
                else:
                    action = buttons("X")
            else:
                write_wram_u16(env, 0x09D2, 1)
                action = buttons("LEFT", "X") if frame % 8 < 4 else buttons("LEFT")
            env.step(action)
            if frame % 200 == 0:
                state = parse_env_state(env, frame=frame)
                print(
                    frame,
                    f"0x{state.room_id:04X}",
                    "xy",
                    state.samus_x,
                    state.samus_y,
                    "pose",
                    state.pose,
                    "hps",
                    _enemy_hps(env),
                    "min",
                    min_body,
                    min_brain,
                )
        final = parse_env_state(env)
        out = GAME_DIR / "debug" / "mother_brain" / "mb_spray_final.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _screenshot(env, out)
        write_state_bytes(
            GAME_DIR
            / "custom_integrations"
            / "SuperMetroid-Snes"
            / "dev_mother_brain_probe_end.state",
            env.em.get_state(),
        )
        return {
            "rooms": rooms,
            "final": final.to_dict(),
            "minBodyHp": min_body if min_body < 10**9 else None,
            "minBrainHp": min_brain if min_brain < 10**9 else None,
            "screenshot": str(out),
        }
    finally:
        env.close()


def run_escape(*, frames: int, state_path: Path) -> dict[str, object]:
    """Coarse escape navigation probe from Escape Room 1."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    rooms: list[tuple] = []
    last_room = None
    try:
        env.reset()
        env.em.set_state(read_state_bytes(state_path))
        place_samus(env, 400, 180)
        for frame in range(frames):
            state = parse_env_state(env, frame=frame)
            apply_dev_survivability(env)
            if state.timer_type == 0 or (
                state.escape_timer_minutes == 0 and state.escape_timer_seconds < 20
            ):
                start_escape_timer(env)
            if state.room_id != last_room:
                rooms.append(
                    (
                        frame,
                        f"0x{state.room_id:04X}",
                        state.samus_x,
                        state.samus_y,
                        state.phase.value,
                        state.game_state,
                    )
                )
                print("ROOM", rooms[-1])
                last_room = state.room_id
            if state.phase is GameplayPhase.ENDING_OR_CREDITS:
                print("CREDITS", frame, state.game_state)
                break
            if state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
                print("DEAD", frame)
                break

            room = state.room_id
            x, y = state.samus_x, state.samus_y
            if room == ROOM_ESCAPE_1:
                action = (
                    buttons("LEFT", "B", "A")
                    if x > 100
                    else buttons("DOWN", "B", "A")
                )
            elif room == ROOM_ESCAPE_2:
                action = (
                    buttons("UP", "A", "B") if y > 100 else buttons("LEFT", "B", "A")
                )
            elif room in (ROOM_ESCAPE_3, ROOM_ESCAPE_4):
                action = buttons("LEFT", "B", "A")
            elif room == ROOM_LANDING_SITE:
                if x > 500:
                    action = buttons("LEFT", "B", "A")
                else:
                    action = buttons("UP") if frame % 30 < 15 else buttons("LEFT")
            else:
                action = buttons("LEFT", "B", "A")
            env.step(action)
            if frame % 300 == 0:
                state = parse_env_state(env, frame=frame)
                print(
                    frame,
                    f"0x{state.room_id:04X}",
                    "xy",
                    state.samus_x,
                    state.samus_y,
                    "timer",
                    f"{state.escape_timer_minutes}:{state.escape_timer_seconds:02d}",
                    state.phase.value,
                )
        final = parse_env_state(env)
        out = GAME_DIR / "debug" / "mother_brain" / "escape_run_final.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _screenshot(env, out)
        return {"rooms": rooms, "final": final.to_dict(), "screenshot": str(out)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("capture-mb")
    sub.add_parser("capture-escape1")

    spray = sub.add_parser("spray-mb")
    spray.add_argument("--frames", type=int, default=3600)
    spray.add_argument("--state", type=Path, default=MB_ENTRY_STATE)

    esc = sub.add_parser("run-escape")
    esc.add_argument("--frames", type=int, default=12000)
    esc.add_argument("--state", type=Path, default=ESCAPE1_STATE)

    args = parser.parse_args()
    if args.command == "capture-mb":
        print(json.dumps(capture_mother_brain_entry(), indent=2))
    elif args.command == "capture-escape1":
        print(json.dumps(capture_escape_room1(), indent=2))
    elif args.command == "spray-mb":
        if not args.state.is_file():
            capture_mother_brain_entry(output=args.state)
        print(json.dumps(spray_mb(frames=args.frames, state_path=args.state), indent=2))
    elif args.command == "run-escape":
        if not args.state.is_file():
            capture_escape_room1(output=args.state)
        print(
            json.dumps(run_escape(frames=args.frames, state_path=args.state), indent=2)
        )


if __name__ == "__main__":
    main()
