"""Independent audit of the Clean master tape: raw buttons from power-on.

Deliberately imports nothing from ``super_metroid.splice``, ``assist``,
``combat`` or ``routes``. It builds a stable-retro env with no start state,
feeds each 12-button row of ``master_tape_clean.npz`` to ``env.step`` once, and
never calls set_state/set_ram/assign. It then compares the end state with the
build-time report. If the tape reaches the recorded state this way, no state
load, RAM write, assist or harness code can have contributed.

    uv run python snes/super_metroid/scripts/tools/audit_master_tape.py
    uv run python snes/super_metroid/scripts/tools/audit_master_tape.py --headed
    uv run python snes/super_metroid/scripts/tools/audit_master_tape.py --headed --from-frame 220000 --fps 60

``--headed`` opens a window and runs at ``--fps`` (default 60 = real time).
Skipped frames before ``--from-frame`` still run, unrendered and unthrottled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import retro

GAME = "SuperMetroid-Snes"
GAME_DIR = Path(__file__).resolve().parents[2]
REC = GAME_DIR / "recordings"
VANILLA_SHA1 = "da957f0d63d14cb441d215462904c4fa8519c613"


def word(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr]) | int(ram[addr + 1]) << 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="master_tape_clean")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--from-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    report = json.loads((REC / f"{args.stem}.json").read_text())
    tape = np.load(REC / f"{args.stem}.npz")["actions"]
    assert tape.ndim == 2 and tape.shape[1] == 12 and np.isin(tape, [0, 1]).all()
    assert hashlib.sha256(tape.tobytes()).hexdigest() == report["tape_sha256"], "tape digest"

    integ = GAME_DIR / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integ))
    rom = (integ / GAME / "rom.sfc").resolve()
    sha1 = hashlib.sha1(rom.read_bytes()).hexdigest()
    print(f"rom {rom} sha1 {sha1} vanilla={sha1 == VANILLA_SHA1}")

    env = retro.make(
        GAME,
        state=retro.State.NONE,  # true power-on, no save state
        inttype=retro.data.Integrations.CUSTOM,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="human" if args.headed else "rgb_array",
    )
    env.reset()
    frames = len(tape) if not args.max_frames else min(args.max_frames, len(tape))
    period = 1.0 / args.fps if args.headed else 0.0
    t0 = time.perf_counter()
    deaths = 0
    for i in range(frames):
        env.step(tape[i])
        if i % 30 == 0:
            ram = env.get_ram()
            gs = word(ram, 0x0998)
            if 0x13 <= gs <= 0x1A:
                deaths += 1
        if args.headed and i >= args.from_frame:
            env.render()
            wait = t0 + (i - args.from_frame + 1) * period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        elif args.headed and i == args.from_frame - 1:
            t0 = time.perf_counter()
        if i % 20000 == 0:
            ram = env.get_ram()
            print(f"frame {i:>7} room 0x{word(ram, 0x79B):04X} "
                  f"({word(ram, 0xAF6)},{word(ram, 0xAFA)}) energy {word(ram, 0x9C2)}", flush=True)

    ram = env.get_ram()
    got = {
        "room": word(ram, 0x79B), "x": word(ram, 0xAF6), "y": word(ram, 0xAFA),
        "energy": word(ram, 0x9C2), "max_energy": word(ram, 0x9C4),
        "missiles": word(ram, 0x9C6), "max_missiles": word(ram, 0x9C8),
        "supers": word(ram, 0x9CA), "max_supers": word(ram, 0x9CC),
        "pbs": word(ram, 0x9CE), "max_pbs": word(ram, 0x9D0),
        "items": word(ram, 0x9A4), "beams": word(ram, 0x9A8),
        "game_state": word(ram, 0x998),
    }
    print("final", {k: (hex(v) if k in ("room", "items", "beams") else v) for k, v in got.items()})
    ok = sha1 == VANILLA_SHA1 and deaths == 0
    if not (sha1 == VANILLA_SHA1):
        print(f"FAIL: non-vanilla ROM sha1 {sha1} != {VANILLA_SHA1}")
    if deaths:
        print(f"FAIL: observed {deaths} death-state samples")
    if frames == len(tape):
        endpoint = hashlib.sha256(env.em.get_state()).hexdigest()
        same = endpoint == report["endpoint_sha256"]
        print(f"emulator snapshot sha256 matches build report: {same}")
        ok &= same
        last = report["rows"][-1]
        want_room = int(last["room"], 16) if not last.get("to_room") else int(last["to_room"], 16)
        want = (want_room, last["leave_x"], last["leave_y"], last["energy"])
        print("report leave (room, x, y, energy):", tuple(hex(want[0]) if i == 0 else v for i, v in enumerate(want)))
        ok &= (got["room"], got["x"], got["y"], got["energy"]) == want
        fs = report.get("final_state") or {}
        if fs:
            field_pairs = [
                ("room", "room_id"), ("x", "samus_x"), ("y", "samus_y"),
                ("energy", "health"), ("max_energy", "max_health"),
                ("missiles", "missiles"), ("max_missiles", "max_missiles"),
                ("supers", "super_missiles"), ("max_supers", "max_super_missiles"),
                ("pbs", "power_bombs"), ("max_pbs", "max_power_bombs"),
                ("items", "collected_items"), ("beams", "collected_beams"),
                ("game_state", "game_state"),
            ]
            mismatches = {k: (got[k], fs[fk]) for k, fk in field_pairs if fk in fs and got[k] != fs[fk]}
            if mismatches:
                print(f"FAIL: final_state mismatches: {mismatches}")
                ok = False
        ok &= got["energy"] > 0
    if frames != len(tape):
        print(f"frames {frames}  PARTIAL RUN: nothing compared (full tape is {len(tape)} frames)")
        env.close()
        return 0
    print(f"frames {frames}  RESULT {'PASS' if ok else 'MISMATCH'}")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
