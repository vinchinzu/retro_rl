#!/usr/bin/env python3
"""Capture a model-free Match 5 continuation from the natural Fight 4 pin.

Replays NATURAL_THROUGH_FIGHT4_RLE from power-on, snapshots the emulator,
identifies the live Match 5 opponent (leftover HUD is ignored), then drives
Match 5 with RAM specialists first. Pixel ladder-ft is the fallback after
RAM oracles miss. Runtime artifacts are RLE only.

Liu Kang CPU walkthrough (IceMaster / LWang): fireball F,F,HP and flying
kick F,F,HK. Jump-kick into flying kick; fireball on wakeup. Do not jump
into Sub-Zero ice or Scorpion spear. Live Match 5 after Cage → Sonya →
Sub-Zero → Raiden is Kano (id 1); leftover pin HUD still shows Raiden.
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
from mortal_kombat.boot import BootController, action_from_buttons  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    LIU_KANG_ID,
    Screen,
    char_name,
    is_fight_ready,
    parse_ram,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask  # noqa: E402
from mortal_kombat.scripts.replay_natural_fight4 import (  # noqa: E402
    NATURAL_THROUGH_FIGHT4_RLE,
)
from dataclasses import replace  # noqa: E402

from mortal_kombat.tournament import TournamentRunner  # noqa: E402

MATCH5 = 4
MATCH6 = 5


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
    lines = ["NATURAL_FIGHT5_RLE: tuple[tuple[int, int], ...] = ("]
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


def replay_through_fight4(env) -> None:
    reset_obs(env)
    for mask, count in NATURAL_THROUGH_FIGHT4_RLE:
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


def save_rgb(env, path: Path) -> None:
    rgb = env.render()
    if rgb is None:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb)).save(path)
    print(f"wrote {path}")


def identify_live_match5(env, pin, *, max_frames: int, screenshot: Path | None):
    """Idle through VS/load until Match 5 is actually fight-ready."""
    set_emulator_state(env.unwrapped, pin)
    boot = BootController(allow_continue=False)
    last = parse_ram(env.unwrapped.get_ram())
    print(f"identify pin {describe(last)}", flush=True)
    if last.match_counter == MATCH5 and is_fight_ready(last):
        if screenshot is not None:
            save_rgb(env, screenshot)
        return 0, last
    for frame in range(1, max_frames + 1):
        _phase, names = boot.decide(last, frame)
        env.step(action_from_buttons(names))
        last = parse_ram(env.unwrapped.get_ram())
        if frame % 200 == 0:
            print(f"  identify f={frame} {describe(last)}", flush=True)
        if last.match_counter == MATCH5 and is_fight_ready(last):
            if screenshot is not None:
                save_rgb(env, screenshot)
            return frame, last
        if last.screen is Screen.CONTINUE:
            break
    return max_frames, last


def capture_from_pin(
    env,
    pin,
    *,
    ladder_model: str | None,
    force_scripted: bool,
    max_frames: int,
    pixel_model: str | None = None,
    deterministic: bool = True,
) -> tuple[bool, list[int], object]:
    set_emulator_state(env.unwrapped, pin)
    recorder = RecordingEnv(env)
    p1_kos = p2_kos = 0
    prev_health: tuple[int, int] | None = None
    fight_started = False
    live_p2: int | None = None
    last_snap = parse_ram(env.unwrapped.get_ram())

    def on_frame(_env, frame, snap, _prev) -> bool:
        nonlocal p1_kos, p2_kos, prev_health, fight_started, last_snap, live_p2
        last_snap = snap
        health = (snap.p1_health, snap.p2_health)
        fight_started = fight_started or (
            snap.match_counter == MATCH5
            and snap.p1_character == LIU_KANG_ID
            and snap.timer > 50
            and health == (161, 161)
        )
        if fight_started and live_p2 is None and snap.match_counter == MATCH5:
            live_p2 = snap.p2_character
            print(
                f"  live opponent id={live_p2}/{char_name(live_p2)} "
                f"f={frame} {describe(snap)}",
                flush=True,
            )
        if fight_started and snap.match_counter == MATCH5 and prev_health is not None:
            p1_kos += int(prev_health[1] > 0 and health[1] == 0)
            p2_kos += int(prev_health[0] > 0 and health[0] == 0)
        prev_health = health if fight_started and snap.match_counter == MATCH5 else None
        if frame % 500 == 0:
            print(f"  f={frame} kos={p1_kos}-{p2_kos} {describe(snap)}", flush=True)
        if snap.screen is Screen.CONTINUE:
            return True
        return snap.match_counter >= MATCH6 and snap.p1_character == LIU_KANG_ID

    runner = TournamentRunner(
        deterministic=deterministic,
        force_scripted=force_scripted,
        ladder_model=ladder_model,
        on_frame=on_frame,
    )
    if pixel_model:
        runner.slots = [
            replace(slot, model=pixel_model, kind="pixel") for slot in runner.slots
        ]
    result = runner.run_on(recorder, max_frames=max_frames)
    won = last_snap.match_counter >= MATCH6 and last_snap.p1_character == LIU_KANG_ID
    print(
        f"  done won={won} frames={len(recorder.masks)} kos={p1_kos}-{p2_kos} "
        f"live_p2={live_p2}/{char_name(live_p2) if live_p2 is not None else '?'} "
        f"furthest={result.furthest} {describe(last_snap)}"
    )
    if result.swaps:
        for swap in result.swaps:
            print(f"  swap {swap}")
    return won, recorder.masks, last_snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=25_000)
    parser.add_argument("--max-identify-frames", type=int, default=4_000)
    parser.add_argument("--identify-only", action="store_true")
    parser.add_argument(
        "--oracles",
        nargs="*",
        default=None,
        help="Subset of oracle labels to try (default: all).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions (tape is still exact RLE once captured).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Attempts per oracle (useful with --stochastic).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("snes/mortal_kombat/scratch/natural_fight5_rle.py"),
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("snes/mortal_kombat/recordings/natural_match5_start.png"),
    )
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replaying natural fight 1+2+3+4…", flush=True)
        replay_through_fight4(env)
        pin_snap = parse_ram(env.unwrapped.get_ram())
        print(f"pin {describe(pin_snap)}", flush=True)
        if pin_snap.match_counter < MATCH5 or pin_snap.p1_character != LIU_KANG_ID:
            print("fight 4 pin is not at Match 5")
            return 1
        pin = get_emulator_state(env.unwrapped)

        ident_frames, ident_snap = identify_live_match5(
            env,
            pin,
            max_frames=args.max_identify_frames,
            screenshot=args.screenshot,
        )
        live = ident_snap.match_counter == MATCH5 and is_fight_ready(ident_snap)
        print(
            f"identify live={live} frames={ident_frames} {describe(ident_snap)}",
            flush=True,
        )
        if not live:
            print("Match 5 fight never became ready; leftover HUD is not the opponent")
            return 1
        print(
            f"live Match 5 opponent is {ident_snap.p2_character}/"
            f"{char_name(ident_snap.p2_character)} "
            "(pin HUD may still show Raiden)"
        )
        if args.identify_only:
            return 0

        # RAM specialists first. Pixel ladder-ft closed Match 4 after RAM
        # oracles lost Raiden; try it here only after Match5 v3 / scripted miss.
        attempts = (
            ("match5-v3", "mk1_v3_Match5_ppo_final.zip", False, None),
            ("fight-v3", "mk1_v3_Fight_ppo_final.zip", False, None),
            ("match2-v3", "mk1_v3_Match2_ppo_final.zip", False, None),
            ("match6-v3", "mk1_v3_Match6_ppo_final.zip", False, None),
            ("match7-v3", "mk1_v3_Match7_ppo_final.zip", False, None),
            ("per-stage-v3", None, False, None),
            ("scripted", None, True, None),
            ("ladder-ft-pixel", None, False, "mk1_ladder_ft_ppo_final.zip"),
            ("ladder-pixel", None, False, "mk1_ladder_ppo_final.zip"),
            ("fresh-pixel", None, False, "mk1_fresh_ppo_final.zip"),
            ("multichar-pixel", None, False, "mk1_multichar_ppo_final.zip"),
            ("match4-pixel", None, False, "mk1_match4_ppo_final.zip"),
            ("match7-pixel", None, False, "mk1_match7_ppo_9500000_steps.zip"),
        )
        if args.oracles:
            wanted = set(args.oracles)
            attempts = tuple(item for item in attempts if item[0] in wanted)
        for label, ladder, scripted, pixel in attempts:
            for attempt in range(args.repeats):
                tag = label if args.repeats == 1 else f"{label}#{attempt + 1}"
                print(f"oracle {tag} det={not args.stochastic}", flush=True)
                won, masks, snap = capture_from_pin(
                    env,
                    pin,
                    ladder_model=ladder,
                    force_scripted=scripted,
                    max_frames=args.max_frames,
                    pixel_model=pixel,
                    deterministic=not args.stochastic,
                )
                if not won:
                    continue
                encoded = rle_encode(masks)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                body = (
                    f"NATURAL_FIGHT5_FRAMES = {len(masks)}\n"
                    f"{format_rle(encoded)}\n"
                )
                args.out.write_text(body)
                print(f"wrote {args.out} frames={len(masks)} rle={len(encoded)}")
                print(f"end {describe(snap)}")
                return 0
        print("no oracle reached match_counter=5")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
