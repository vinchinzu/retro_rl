"""Replay short controller recipes from a local suffix-development state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, globals().get('_SNES_IMPORT_ROOT', _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import (  # noqa: E402
    make_env,
    read_state_bytes,
    write_state_bytes,
)
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR  # noqa: E402
from super_metroid.ram import parse_state  # noqa: E402


def _span(value: str) -> tuple[int, tuple[str, ...]]:
    frame_text, _, button_text = value.partition(":")
    names = tuple(name.upper() for name in button_text.split(",") if name)
    return int(frame_text), names


def _pulse(value: str) -> tuple[int, int, int, tuple[str, ...]]:
    count_text, on_text, off_text, button_text = value.split(":", 3)
    names = tuple(name.upper() for name in button_text.split(",") if name)
    return int(count_text), int(on_text), int(off_text), names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--span", action="append", default=[], type=_span)
    parser.add_argument("--pulse", action="append", default=[], type=_pulse)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--trace-room", action="store_true")
    parser.add_argument("--trace-y", action="store_true")
    args = parser.parse_args()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist()
    frame = 0
    try:
        observation, _ = env.reset()
        env.em.set_state(read_state_bytes(args.state))
        state = parse_state(env.get_ram(), frame=frame)
        print("start", json.dumps(state.to_dict()))
        previous_room = state.room_id
        previous_y = state.samus_y
        for frames, names in args.span:
            action = buttons(*names) if names else idle_action()
            min_y = state.samus_y
            max_y = state.samus_y
            for _ in range(frames):
                observation, _, _, _, _ = env.step(action)
                frame += 1
                state = parse_state(env.get_ram(), frame=frame)
                assist.apply(env.data, state)
                min_y = min(min_y, state.samus_y)
                max_y = max(max_y, state.samus_y)
                if args.trace_y and state.samus_y != previous_y:
                    print(
                        f"y frame={frame} y={state.samus_y} "
                        f"vy={state.velocity_y} pose={state.pose}"
                    )
                    previous_y = state.samus_y
                if args.trace_room and state.room_id != previous_room:
                    print(
                        f"transition frame={frame} "
                        f"0x{previous_room:04X}->0x{state.room_id:04X}"
                    )
                    previous_room = state.room_id
            print(
                f"span {frames}:{','.join(names) or '-'}",
                json.dumps(state.to_dict()),
                f"y_range={min_y}..{max_y}",
            )
        for count, on_frames, off_frames, names in args.pulse:
            action = buttons(*names) if names else idle_action()
            min_y = state.samus_y
            max_y = state.samus_y
            for _ in range(count):
                for pulse_action, frames in (
                    (action, on_frames),
                    (idle_action(), off_frames),
                ):
                    for _ in range(frames):
                        observation, _, _, _, _ = env.step(pulse_action)
                        frame += 1
                        state = parse_state(env.get_ram(), frame=frame)
                        assist.apply(env.data, state)
                        min_y = min(min_y, state.samus_y)
                        max_y = max(max_y, state.samus_y)
                        if args.trace_y and state.samus_y != previous_y:
                            print(
                                f"y frame={frame} y={state.samus_y} "
                                f"vy={state.velocity_y} pose={state.pose}"
                            )
                            previous_y = state.samus_y
                        if args.trace_room and state.room_id != previous_room:
                            print(
                                f"transition frame={frame} "
                                f"0x{previous_room:04X}->0x{state.room_id:04X}"
                            )
                            previous_room = state.room_id
            print(
                f"pulse {count}:{on_frames}:{off_frames}:"
                f"{','.join(names) or '-'}",
                json.dumps(state.to_dict()),
                f"y_range={min_y}..{max_y}",
            )
        if args.save is not None:
            write_state_bytes(args.save, env.em.get_state())
        if args.screenshot is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(args.screenshot),
                cv2.cvtColor(observation, cv2.COLOR_RGB2BGR),
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
