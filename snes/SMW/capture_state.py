"""Capture a custom SMW start state for human segment recording.

Play from a known anchor (e.g. YoshiIsland4), reach the next level start,
press **F5** to write ``custom_integrations/.../<Name>.state``, then record
with ``python -m SMW -l <alias> play``.

Examples::

    # Iggy: use a fresh wipe (package YI4 alone does not open the castle path;
    # north pipe is Donut Plains 1). Play YI1–4, enter castle, F5 at room start.
    uv run python -m SMW capture-state --from NONE --name IggysCastle

    # Fresh boot → any overworld/level entry
    uv run python -m SMW capture-state --from NONE --name MyAnchor

    # Then record
    uv run python -m SMW -l iggy play
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Prefer native Wayland when present (matches PlaySession).
if "SDL_VIDEODRIVER" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "wayland"
    else:
        os.environ["SDL_VIDEODRIVER"] = "x11"

ROOT = Path(__file__).resolve().parent
GAME = "SuperMarioWorld-Snes-v0"
INTEGRATION = ROOT / "custom_integrations" / GAME

# Level alias → stable-retro / custom state name for --from
_FROM_ALIASES: dict[str, str] = {
    "yi1": "YoshiIsland1",
    "yi2": "YoshiIsland2",
    "yi3": "YoshiIsland3",
    "yi4": "YoshiIsland4",
    "iggy": "IggysCastle",
    "dp1": "DonutPlains1",
    "none": "NONE",
    "boot": "NONE",
    "title": "NONE",
}


def _resolve_from_state(value: str) -> str:
    key = value.strip()
    if key.upper() == "NONE":
        return "NONE"
    return _FROM_ALIASES.get(key.lower(), key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SMW human state capture (controller-friendly F5 save)",
    )
    parser.add_argument(
        "--from",
        dest="from_state",
        default="yi4",
        help="Start state or alias (yi4, yi3, NONE, …). Default: yi4",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="State name to write (no .state), e.g. IggysCastle",
    )
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args(argv)

    start = _resolve_from_state(args.from_state)
    name = args.name.removesuffix(".state")

    from retro_harness.env import make_env, save_state
    from retro_harness.play_session import PlaySession

    env = make_env(
        game=GAME,
        state=None if start == "NONE" else start,
        game_dir=ROOT,
        render_mode="rgb_array",
    )

    save_count = 0
    out_path = INTEGRATION / f"{name}.state"

    def on_hud(info: dict) -> list[str]:
        ram = env.get_ram()
        # GameMode $0100, translevel $13BF, player x/y
        mode = int(ram[0x0100]) if len(ram) > 0x0100 else -1
        trans = int(ram[0x13BF]) if len(ram) > 0x13BF else -1
        px = int(ram[0x00D1]) | (int(ram[0x00D2]) << 8) if len(ram) > 0x00D2 else 0
        py = int(ram[0x00D3]) | (int(ram[0x00D4]) << 8) if len(ram) > 0x00D4 else 0
        return [
            f"Capture → {name}.state  (saves={save_count})",
            f"mode={mode:#x} trans={trans:#x} x={px} y={py}",
            "F5=save this spot   TAB=turbo   ESC=quit",
            "Tip: save when mode=0x14 (in level), near intended start",
        ]

    session = PlaySession(
        env,
        game_dir=str(ROOT),
        game=GAME,
        scale=args.scale,
        title=f"SMW capture: {name}",
    )
    session.on_hud = on_hud

    original = session.on_key_down

    def on_key(key):
        import pygame

        nonlocal save_count
        if key == pygame.K_F5:
            path = save_state(env, ROOT, GAME, name)
            save_count += 1
            ram = env.get_ram()
            mode = int(ram[0x0100]) if len(ram) > 0x0100 else -1
            trans = int(ram[0x13BF]) if len(ram) > 0x13BF else -1
            print(f"\n[SAVED] {path}")
            print(f"  save #{save_count}  GameMode={mode:#x} translevel={trans:#x}")
            if mode != 0x14:
                print(
                    "  note: GameMode is not 0x14 (active level). "
                    "For segment play, re-save once inside the level."
                )
            else:
                print("  next: uv run python -m SMW -l iggy play   # if name=IggysCastle")
                print(f"        (level must map start_state={name!r})")
            print()
            return True
        return original(key) if original else False

    session.on_key_down = on_key

    print(f"Start state: {start}")
    print(f"Will write:  {out_path}")
    print("Play with controller. F5 saves; ESC quits.\n")
    try:
        session.run()
    finally:
        env.close()

    if out_path.exists():
        print(f"\nState on disk: {out_path}")
        print("Record: uv run python -m SMW -l iggy play   # after IggysCastle capture")
    else:
        print("\nNo state saved (F5 was not pressed).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
