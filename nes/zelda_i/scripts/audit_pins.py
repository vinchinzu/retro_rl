"""Audit every named save state for a ``$066F`` a real Link cannot hold.

``$066F`` is ``hi = containers - 1``, ``lo = whole hearts``, so a coherent
byte always has ``lo <= hi`` (``ram.health_byte_is_coherent``). The ROM does
not clamp a byte that breaks this, so an incoherent pin hands the lane that
loads it several times a real Link's damage budget without saying so —
``Level6Entrance`` held ``0x2F`` (15 hearts in 3 containers) and four sittings
of Clean L6 room tuning were graded against it before anyone read the byte.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/audit_pins.py

Exit status is 1 when any pin is incoherent, so this can gate a lane that is
about to quote hearts. Reads only; loads one state per emulator process
because stable-retro keeps one emulator per process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from retro_harness.env import integration_dir
from zelda_i.paths import GAME, GAME_DIR

_CHILD = """
import json, sys
from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import health_byte_is_coherent, read_snapshot

name = sys.argv[1]
configure_headless()
env = make_env(GAME, name, GAME_DIR, render_mode=None)
reset_obs(env)
resync_custom_state(env, GAME_DIR, GAME, name)
snap = read_snapshot(env.get_ram())
health = int(snap.health)
print(json.dumps({
    "pin": name,
    "health": health,
    "health_hex": f"0x{health:02x}",
    "containers": (health >> 4) + 1,
    "hearts": health & 0x0F,
    "coherent": health_byte_is_coherent(health),
    "triforce": int(snap.triforce),
    "level": int(snap.level),
    "room": int(snap.screen),
}))
env.close()
"""


# The pins a Clean lane starts a level from. These are the ones whose health
# byte becomes a heart budget; the other ~380 states are mid-room probes.
ENTRANCE_GLOBS = ("Level*Entrance*", "At[0-9A-F][0-9A-F]")


def pin_names(*, everything: bool = False) -> list[str]:
    integration = integration_dir(GAME_DIR, GAME)
    if everything:
        return sorted(p.stem for p in integration.glob("*.state"))
    found: set[str] = set()
    for pattern in ENTRANCE_GLOBS:
        found.update(p.stem for p in integration.glob(f"{pattern}.state"))
    return sorted(found)


def read_pin(name: str) -> dict:
    """One emulator per process: stable-retro will not load two in one."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, name],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    if proc.returncode != 0:
        return {"pin": name, "error": (proc.stderr or "").strip()[-200:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="append", help="audit only these pins")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every *.state, not just the level-entrance pins (slow: one "
        "emulator process each)",
    )
    parser.add_argument("--json", action="store_true", help="emit the rows as JSON")
    args = parser.parse_args(argv)

    names = args.pin or pin_names(everything=args.all)
    rows = [read_pin(name) for name in names]
    bad = [r for r in rows if r.get("coherent") is False]
    if args.json:
        print(json.dumps({"rows": rows, "incoherent": bad}, indent=2))
    else:
        for row in rows:
            if "error" in row:
                print(f"{row['pin']:<40} ERROR {row['error']}")
                continue
            flag = "  " if row["coherent"] else "  <-- INCOHERENT"
            print(
                f"{row['pin']:<40} {row['health_hex']}  "
                f"{row['hearts']}/{row['containers']} hearts  "
                f"tf=0x{row['triforce']:02x}{flag}"
            )
        print(f"\n{len(rows)} pins, {len(bad)} incoherent")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
