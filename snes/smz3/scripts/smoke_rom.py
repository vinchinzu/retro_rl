"""Smoke-check a built SMZ3 combo ROM (size + title bytes, optional boot).

  uv run python smz3/scripts/smoke_rom.py smz3/seeds/test_seed/smz3.sfc
  SDL_VIDEODRIVER=dummy uv run python smz3/scripts/smoke_rom.py --boot ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smz3.paths import COMBO_ROM_SIZE, TEST_SEED_DIR  # noqa: E402
from smz3.rom_builder import rom_title_bytes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rom",
        nargs="?",
        type=Path,
        default=TEST_SEED_DIR / "smz3.sfc",
        help="Path to combo ROM",
    )
    parser.add_argument(
        "--boot",
        action="store_true",
        help="Attempt stable-retro boot via SMZ3-Snes integration (if wired)",
    )
    parser.add_argument(
        "--controllable",
        action="store_true",
        help="With --boot: run power-on → first SM controllable frame (M1)",
    )
    args = parser.parse_args(argv)

    rom_path = args.rom
    if not rom_path.is_file():
        print(f"Missing ROM: {rom_path}", file=sys.stderr)
        return 1

    data = rom_path.read_bytes()
    print(f"path: {rom_path}")
    print(f"size: {len(data)} (expected {COMBO_ROM_SIZE})")
    if len(data) != COMBO_ROM_SIZE:
        print("WARN: unexpected size", file=sys.stderr)

    title = rom_title_bytes(data)
    try:
        title_txt = title.decode("ascii", errors="replace")
    except Exception:
        title_txt = repr(title)
    print(f"title_probe: {title_txt!r}")

    # ZSM version string is written by the seed patch.
    if b"ZSM" in data:
        idx = data.find(b"ZSM")
        print(f"ZSM marker at {idx:#x}: {data[idx:idx+21]!r}")
    else:
        print("WARN: no ZSM marker found in ROM", file=sys.stderr)

    if args.boot:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from smz3.paths import GAME_DIR, INTEGRATION_DIR
        from retro_harness.env import make_env
        from retro_harness.actions import idle_action

        link = INTEGRATION_DIR / "rom.sfc"
        if not link.exists():
            print(
                f"Missing {link}; run: uv run python smz3/scripts/wire_integration_rom.py {rom_path}",
                file=sys.stderr,
            )
            return 1
        env = make_env("SMZ3-Snes", None, GAME_DIR, render_mode="rgb_array")
        try:
            obs, _info = env.reset()
            if args.controllable:
                from smz3.boot import boot_to_controllable

                result = boot_to_controllable(env, close=False)
                print(
                    f"controllable: ok={result.ok} frames={result.frames} "
                    f"world={result.world.value} detail={result.detail}"
                )
                if not result.ok:
                    return 2
                obs = env.render()
            else:
                idle = idle_action()
                for _ in range(60):
                    obs, *_rest = env.step(idle)
                print(f"boot frames: 60 ok; frame shape={getattr(obs, 'shape', None)}")
        finally:
            env.close()

    print("smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
