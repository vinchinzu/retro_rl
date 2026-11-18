"""Log boss sprites for one pin. Load only; the pin file is not written.

Sprites are recorded every frame. ``--byte NAME=HEX`` names a per-slot
WRAM byte (slot ``i`` is ``HEX+i``). A ``--run`` reloads the pin and
either idles, calls ``shoot_arrow``, or plays ``DIR:N`` presses. That
reload is the rollout sweep. ``$F3CC`` is not written.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boss_probe.py \\
        Whatif_Eastern_armos --frames 1800 --byte hp=0x0E50 --byte phase=0x0D80 \\
        --out alttp/recordings/probe_AP-103/boss_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alttp.primitives import active_ancillae, active_sprites, shoot_arrow
from alttp.ram import read_snapshot
from alttp.scripts.whatif_pin import _refresh, open_pin
from alttp.startup import action_for, no_action, step_frames

_SLOT_COUNT = 16


def _parse_byte(text: str) -> tuple[str, int]:
    name, sep, raw = text.partition("=")
    name = name.strip()
    if not sep or not name or not raw.strip():
        raise SystemExit(f"--byte must be name=hex, got {text!r}")
    try:
        addr = int(raw.strip(), 0)
    except ValueError:
        raise SystemExit(f"--byte {name} is not an int: {raw!r}") from None
    if addr < 0 or addr + _SLOT_COUNT > 0x2000:
        raise SystemExit(f"--byte {name}={addr:#x} is not a low-WRAM slot byte")
    return name, addr


def _parse_press(text: str) -> tuple[str, int]:
    name, sep, raw = text.partition(":")
    if not sep:
        raise SystemExit(f"--press must be DIR:N, got {text!r}")
    try:
        frames = int(raw)
    except ValueError:
        raise SystemExit(f"--press must be DIR:N, got {text!r}") from None
    if frames < 0:
        raise SystemExit(f"--press frames must be >= 0, got {text!r}")
    return name.strip().upper(), frames


def _sample(env: object, bases: dict[str, int]) -> list[list[int]]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    rows: list[list[int]] = []
    for spr in active_sprites(env):
        slot = int(spr.slot)
        named = [int(ram[bases[name] + slot]) for name in bases]
        rows.append(
            [slot, int(spr.sprite_type), int(spr.state), int(spr.x), int(spr.y), *named]
        )
    return rows


def _ancilla(env: object) -> list[list[int]]:
    return [
        [int(row.slot), int(row.ancilla_type), int(row.x), int(row.y)]
        for row in active_ancillae(env)
    ]


def _play_run(
    env: object,
    *,
    frames: int,
    presses: list[tuple[str, int]],
    arrow: bool,
    bases: dict[str, int],
) -> dict[str, object]:
    idle = no_action()
    samples: list[list[object]] = []
    ancilla_changes: list[dict[str, object]] = []
    seen_ancilla: tuple[tuple[int, int, int, int], ...] | None = None
    used = 0

    def tick(count: int, action: object) -> None:
        nonlocal used, seen_ancilla
        for _ in range(count):
            if used >= frames:
                return
            step_frames(env, action, 1)  # type: ignore[arg-type]
            _refresh(env)
            snap = read_snapshot(env.get_ram())  # type: ignore[attr-defined]
            sprites = _sample(env, bases)
            samples.append([used, snap.link_x, snap.link_y, snap.hp, sprites])
            now = tuple(tuple(row) for row in _ancilla(env))
            if now != seen_ancilla:
                ancilla_changes.append({"frame": used, "ancilla": [list(row) for row in now]})
                seen_ancilla = now
            used += 1

    for name, count in presses:
        action = no_action() if name in ("", "NONE") else action_for(name)
        tick(count, action)
    if arrow and used < frames:
        shot = shoot_arrow(env)
        _refresh(env)
        # shoot_arrow already stepped. Catch up one sample for that call.
        snap = read_snapshot(env.get_ram())  # type: ignore[attr-defined]
        samples.append(
            [used, snap.link_x, snap.link_y, snap.hp, _sample(env, bases)]
        )
        ancilla_changes.append(
            {
                "frame": used,
                "arrow": shot.reason,
                "arrow_ok": bool(shot.ok),
                "arrow_frames": int(shot.frames),
                "ancilla": _ancilla(env),
            }
        )
        used += int(shot.frames)
    tick(frames - used, idle)
    return {
        "frames": used,
        "samples": samples,
        "ancilla": ancilla_changes,
        "changes": _changes(samples, list(bases)),
    }


def _changes(samples: list[list[object]], names: list[str]) -> list[dict[str, object]]:
    """HP/phase (and any other named byte) edges, plus new sprite types."""
    previous: dict[int, list[int]] = {}
    first_types: set[int] = set()
    if samples:
        for row in samples[0][4]:  # type: ignore[index]
            first_types.add(int(row[1]))
    born: list[dict[str, int]] = []
    edges: list[dict[str, object]] = []
    for sample in samples:
        frame = int(sample[0])
        for row in sample[4]:  # type: ignore[index]
            slot = int(row[0])
            kind = int(row[1])
            named = [int(value) for value in row[5:]]
            if kind not in first_types and slot not in previous:
                born.append({"frame": frame, "slot": slot, "type": kind})
            before = previous.get(slot)
            if before is not None and before != named:
                edges.append(
                    {
                        "frame": frame,
                        "slot": slot,
                        "type": kind,
                        "from": {names[i]: before[i] for i in range(len(names))},
                        "to": {names[i]: named[i] for i in range(len(names))},
                    }
                )
            previous[slot] = named
    return [{"born": born, "edges": edges}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pin", help="save-state name, not modified")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument(
        "--byte",
        action="append",
        default=[],
        metavar="NAME=HEX",
        help="per-slot WRAM byte (repeatable)",
    )
    parser.add_argument(
        "--press",
        action="append",
        default=[],
        metavar="DIR:N",
        help="buttons before the idle tail (repeatable, in order)",
    )
    parser.add_argument(
        "--arrow",
        action="store_true",
        help="call shoot_arrow after --press, then idle",
    )
    parser.add_argument("--out", required=True, help="JSON path")
    args = parser.parse_args(argv)
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if not args.byte:
        raise SystemExit("need at least one --byte name=hex")
    bases = dict(_parse_byte(item) for item in args.byte)
    presses = [_parse_press(item) for item in args.press]
    env = open_pin(args.pin)
    try:
        report = _play_run(
            env,
            frames=args.frames,
            presses=presses,
            arrow=bool(args.arrow),
            bases=bases,
        )
    finally:
        env.close()
    payload = {
        "pin": args.pin,
        "frames": args.frames,
        "bytes": {name: f"0x{addr:04X}" for name, addr in bases.items()},
        "press": [f"{name}:{count}" for name, count in presses],
        "arrow": bool(args.arrow),
        "run": report,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    changes = report["changes"][0]
    print(
        f"{args.pin} frames={report['frames']} "
        f"edges={len(changes['edges'])} born={len(changes['born'])} "
        f"out={path}"
    )
    for edge in changes["edges"][:24]:
        print(
            f"  f={edge['frame']} slot={edge['slot']} type=0x{edge['type']:02X} "
            f"{edge['from']} -> {edge['to']}"
        )
    for born in changes["born"][:12]:
        print(f"  born f={born['frame']} slot={born['slot']} type=0x{born['type']:02X}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
