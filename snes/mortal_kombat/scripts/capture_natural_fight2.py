#!/usr/bin/env python3
"""Capture a model-free Match 2 continuation from the natural Fight 1 pin.

Replays NATURAL_FIGHT1_RLE from power-on, snapshots the emulator, then drives
Match 2 with the same offline oracle family as Fight 1 (deterministic Match5
v3, then per-stage v3, then scripted). Runtime artifacts are RLE only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "snes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.snapshot import get_emulator_state, set_emulator_state  # noqa: E402
from mortal_kombat.natural_fight1_tape import NATURAL_FIGHT1_RLE  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    LIU_KANG_ID,
    Screen,
    char_name,
    parse_ram,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask  # noqa: E402
from mortal_kombat.tournament import TournamentRunner  # noqa: E402


def mask_from_buttons(buttons: np.ndarray) -> int:
    mask = 0
    for index, value in enumerate(np.asarray(buttons).reshape(-1)[:12]):
        if int(value):
            mask |= 1 << index
    return mask


def rle_encode(masks: list[int]) -> list[tuple[int, int]]:
    encoded: list[tuple[int, int]] = []
    for mask in masks:
        if encoded and encoded[-1][0] == mask:
            encoded[-1] = (mask, encoded[-1][1] + 1)
        else:
            encoded.append((mask, 1))
    return encoded


def format_rle(pairs: list[tuple[int, int]], width: int = 8) -> str:
    chunks = [f"({mask}, {count})" for mask, count in pairs]
    lines = ["NATURAL_FIGHT2_RLE: tuple[tuple[int, int], ...] = ("]
    for start in range(0, len(chunks), width):
        lines.append("    " + ", ".join(chunks[start : start + width]) + ",")
    lines.append(")")
    return "\n".join(lines)


class RecordingEnv:
    """Record 12-button masks while forwarding to a live retro env."""

    def __init__(self, env):
        self.env = env
        self.masks: list[int] = []

    def step(self, action):
        self.masks.append(mask_from_buttons(action))
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def replay_fight1(env) -> None:
    reset_obs(env)
    for mask, count in NATURAL_FIGHT1_RLE:
        buttons = buttons_from_mask(mask)
        for _ in range(count):
            env.step(buttons)


def describe(snap) -> str:
    return (
        f"screen={snap.screen.name} match={snap.match_counter} "
        f"char={snap.p1_character}/{char_name(snap.p1_character)} "
        f"p2={snap.p2_character}/{char_name(snap.p2_character)} "
        f"hp={snap.p1_health}/{snap.p2_health} "
        f"rounds={snap.p1_rounds}-{snap.p2_rounds} timer={snap.timer}"
    )


def capture_from_pin(
    env,
    pin,
    *,
    ladder_model: str | None,
    force_scripted: bool,
    max_frames: int,
) -> tuple[bool, list[int], object]:
    set_emulator_state(env.unwrapped, pin)
    recorder = RecordingEnv(env)
    p1_kos = p2_kos = 0
    prev_health: tuple[int, int] | None = None
    fight_started = False
    last_snap = parse_ram(env.unwrapped.get_ram())

    def on_frame(_env, frame, snap, _prev) -> bool:
        nonlocal p1_kos, p2_kos, prev_health, fight_started, last_snap
        last_snap = snap
        health = (snap.p1_health, snap.p2_health)
        fight_started = fight_started or (
            snap.match_counter == 1
            and snap.p1_character == LIU_KANG_ID
            and snap.timer > 50
            and health == (161, 161)
        )
        if fight_started and snap.match_counter == 1 and prev_health is not None:
            p1_kos += int(prev_health[1] > 0 and health[1] == 0)
            p2_kos += int(prev_health[0] > 0 and health[0] == 0)
        prev_health = health if fight_started and snap.match_counter == 1 else None
        if frame % 500 == 0:
            print(f"  f={frame} kos={p1_kos}-{p2_kos} {describe(snap)}", flush=True)
        if snap.screen is Screen.CONTINUE:
            return True
        return snap.match_counter >= 2 and snap.p1_character == LIU_KANG_ID

    runner = TournamentRunner(
        deterministic=True,
        force_scripted=force_scripted,
        ladder_model=ladder_model,
        on_frame=on_frame,
    )
    result = runner.run_on(recorder, max_frames=max_frames)
    won = last_snap.match_counter >= 2 and last_snap.p1_character == LIU_KANG_ID
    print(
        f"  done won={won} frames={len(recorder.masks)} kos={p1_kos}-{p2_kos} "
        f"furthest={result.furthest} {describe(last_snap)}"
    )
    if result.swaps:
        for swap in result.swaps:
            print(f"  swap {swap}")
    return won, recorder.masks, last_snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=25_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("snes/mortal_kombat/scratch/natural_fight2_rle.py"),
    )
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replaying natural fight 1…", flush=True)
        replay_fight1(env)
        pin_snap = parse_ram(env.unwrapped.get_ram())
        print(f"pin {describe(pin_snap)}", flush=True)
        if pin_snap.match_counter < 1 or pin_snap.p1_character != LIU_KANG_ID:
            print("fight 1 pin is not at Match 2")
            return 1
        pin = get_emulator_state(env.unwrapped)

        attempts = (
            ("match5-ladder", "mk1_v3_Match5_ppo_final.zip", False),
            ("per-stage-v3", None, False),
            ("scripted", None, True),
        )
        for label, ladder, scripted in attempts:
            print(f"oracle {label}", flush=True)
            won, masks, snap = capture_from_pin(
                env,
                pin,
                ladder_model=ladder,
                force_scripted=scripted,
                max_frames=args.max_frames,
            )
            if not won:
                continue
            encoded = rle_encode(masks)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            body = (
                f"NATURAL_FIGHT2_FRAMES = {len(masks)}\n"
                f"{format_rle(encoded)}\n"
            )
            args.out.write_text(body)
            print(f"wrote {args.out} frames={len(masks)} rle={len(encoded)}")
            print(f"end {describe(snap)}")
            return 0
        print("no oracle reached match_counter=2")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
