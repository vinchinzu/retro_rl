"""Clean Level 9 damage per hop over RNG offsets, in parallel. Scratch.

Runs ``l9_probe.py`` from the ``<PREFIX>_<step>_*`` pin once per offset with
a what-if heart write (``--hearts``; no beam), and prints ok / frames /
guard overrides / damage / hits by cause per run and a mean per step.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/eval_l9_hop_offsets.py \
        L9S5 s17 s23 --guard --hearts 10 --offsets 0 3 7 11 --jobs 8

Uses this interpreter and inherits ``PYTHONPATH``, so it runs a worktree's
code when launched with ``PYTHONPATH=$W:$W/snes:$W/nes``.
The guard defaults on; use ``--no-guard`` for the A/B baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1]
ROOT = GAME_DIR.parents[1]
PINS = GAME_DIR / "custom_integrations" / "LegendOfZelda-Nes"
PROBE = GAME_DIR / "scratch" / "l9_probe.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("steps", nargs="+")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 3, 7, 11])
    ap.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True,
                    help="Level 9 shot guard (default on; --no-guard for the baseline)")
    ap.add_argument("--hearts", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--to", default=None, help="stop after this step (default: the start step)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to l9_probe")
    a = ap.parse_args()
    hv = 0xE0 | (a.hearts - 1)

    def run(step: str, off: int):
        pins = sorted(f for f in os.listdir(PINS) if f.startswith(f"{a.prefix}_{step}_"))
        if not pins:
            return step, off, [], f"no pin {a.prefix}_{step}_*"
        pin = pins[0][: -len(".state")]
        cmd = [
            sys.executable, str(PROBE), pin, "--from", step, "--to", a.to or step,
            "--idle", str(off), "--set", f"0x066F={hv}", "--set", "0x0670=0xFF",
        ]
        cmd.append("--guard" if a.guard else "--no-guard")
        cmd.extend(a.extra)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        err = "" if rows else proc.stderr[-400:]
        return step, off, rows, err

    jobs = [(s, o) for s in a.steps for o in a.offsets]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda so: run(*so), jobs))
    summary: dict[str, list[tuple[bool, float]]] = {}
    for step, off, rows, err in results:
        dmg = sum(r["damage"] or 0 for r in rows)
        ok = bool(rows) and all(r["ok"] for r in rows)
        frames = sum(r["frames"] for r in rows)
        g = sum(r.get("guard") or 0 for r in rows)
        causes: dict[str, float] = {}
        for r in rows:
            for k, v in (r.get("by_cause") or {}).items():
                causes[k] = causes.get(k, 0) + v
        notes = [n for r in rows for n in (r.get("notes") or [])]
        end = rows[-1]["end"] if rows else ""
        print(f"{step} o{off:<3d} ok={ok!s:5s} f={frames:6d} guard={g:5d} dmg={dmg:5.2f} {causes} {notes} {end} {err}")
        summary.setdefault(step, []).append((ok, dmg))
    for step, v in summary.items():
        n = len(v)
        print(f"== {step}: ok {sum(1 for o, _ in v if o)}/{n} mean dmg {sum(d for _, d in v) / n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
