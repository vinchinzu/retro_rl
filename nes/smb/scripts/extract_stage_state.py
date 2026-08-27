#!/usr/bin/env python3
"""Promote ./play all-exits stage pins into integration practice start states.

The 32-exit track (`smb_all_exits`) has no boot states for non-warp stages.
The human tape pins under ``recordings/human/<task>_pins/`` are verified
control-entry snapshots written by ``./play smb``; this tool copies one into
the SuperMarioBros-Nes-v0 custom integration as a real start state (e.g.
``Level1_3``) and verifies it end-to-end against the pin fingerprint. It is
standalone: nothing here touches the warp any% line (reactive_12 fragments,
natural_82 seed, HL slices stay byte-for-byte).

Fingerprints must match exactly on load (full emulator state). After a short
idle settle the stage must be live: oper_mode playing, player walking,
timer counting down from 400. Any mismatch aborts without writing.

```bash
uv run python -m smb.scripts.extract_stage_state --list
uv run python -m smb.scripts.extract_stage_state 1-3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.extract_stage_state 1-4 2-1
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.env import (  # noqa: E402
    make_env,
    read_state_bytes,
    write_state_bytes,
)
from smb.paths import GAME_DIR, GAME_V0, RECORDINGS_DIR  # noqa: E402
from smb.paths import INTEGRATION_V0_DIR  # noqa: E402
from smb.ram import SmbSnapshot, read_snapshot  # noqa: E402
from smb.start_presets import HUMAN_DIR, normalize_stage_id  # noqa: E402

# Fingerprint fields compared verbatim between pin meta and loaded snapshot.
_CHECKED_FIELDS = (
    "world",
    "level",
    "area_pointer",
    "oper_mode",
    "player_state",
    "player_x",
    "player_y",
    "x_speed",
    "y_speed",
    "lives",
)


def pins_root(task: str) -> Path:
    return HUMAN_DIR if task == "all_exits_v1" else RECORDINGS_DIR / task


def pin_paths(stage_id: str, task: str) -> tuple[Path, Path]:
    root = pins_root(task) / f"{task}_pins"
    return root / f"{stage_id}.state", root / f"{stage_id}.json"


def _snapshot_fingerprint(snap: SmbSnapshot) -> dict[str, int]:
    return {
        "world": snap.world,
        "level": snap.level,
        "area_pointer": snap.area_pointer,
        "oper_mode": snap.oper_mode,
        "player_state": snap.player_state,
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "x_speed": int(snap.x_speed),
        "y_speed": int(snap.y_speed),
        "lives": snap.lives,
    }


def _mismatch(loaded: dict[str, int], expect: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in _CHECKED_FIELDS:
        want = expect.get(field)
        got = loaded.get(field)
        if want is not None and int(want) != int(got):
            issues.append(f"{field}: pin={want} loaded={got}")
    return issues


def extract(stage_id: str, *, task: str, settle: int, probe: int) -> dict[str, Any]:
    """Verify + install one stage's pin as an integration start state."""
    world, level = (int(part) for part in stage_id.split("-", 1))
    state_name = f"Level{world}_{level}"
    pin_state, pin_meta = pin_paths(stage_id, task)
    if not pin_state.is_file():
        raise FileNotFoundError(f"no pin state for {stage_id}: {pin_state}")
    meta: dict[str, Any] = json.loads(pin_meta.read_text())
    fingerprint: dict[str, Any] = meta.get("fingerprint", {})

    data = read_state_bytes(pin_state)

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        # Stable-retro advances one blank frame on load/reset; apply twice so
        # the observed frame is the exact pinned control entry.
        env.em.set_state(data)
        env.reset()
        env.em.set_state(data)
        snap = read_snapshot(env.get_ram(), frame=0)
        issues = _mismatch(_snapshot_fingerprint(snap), fingerprint)

        # Idle settle then confirm the stage is actually live/controllable.
        action = [0] * 9
        max_x = snap.player_x
        for step_idx in range(settle + probe):
            obs, reward, done, info = env.step(action)[:4]
            current = read_snapshot(env.get_ram(), frame=step_idx + 1)
            max_x = max(max_x, current.player_x)
            if step_idx >= settle and current.oper_mode != 1:
                issues.append(f"oper_mode={current.oper_mode} @f{step_idx + 1}")
            del obs, reward, done, info

        settled = read_snapshot(env.get_ram(), frame=settle)
        if settled.dash_level != level - 1:
            issues.append(f"dash_level={settled.dash_level} expected {level - 1}")
        if settled.timer <= 0:
            issues.append("timer not running after settle")
        report: dict[str, Any] = {
            "stage": stage_id,
            "state_name": state_name,
            "pin": str(pin_state),
            "entry_snapshot": _snapshot_fingerprint(snap),
            "settled_snapshot": {
                "dash_level": settled.dash_level,
                "timer": settled.timer,
                "player_x": settled.player_x,
                "player_state": settled.player_state,
                "lives": settled.lives,
            },
            "probe_max_player_x": max_x,
            "issues": issues,
            "ok": not issues,
        }
        if issues:
            return report

        out_path = INTEGRATION_V0_DIR / f"{state_name}.state"
        if out_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing start state {out_path}"
            )
        write_state_bytes(out_path, data)
        report["written"] = str(out_path)

        # Round-trip: the installed name must boot through make_env directly.
        boot_env = make_env(GAME_V0, state_name, GAME_DIR, render_mode="rgb_array")
        try:
            booted = read_snapshot(boot_env.get_ram(), frame=0)
            report["roundtrip"] = {
                "dash_level": booted.dash_level,
                "lives": booted.lives,
                "ok": booted.dash_level == level - 1 and booted.oper_mode == 1,
            }
        finally:
            boot_env.close()
        if not report["roundtrip"]["ok"]:
            report["issues"].append(
                f"roundtrip dash_level={booted.dash_level} oper_mode={booted.oper_mode}"
            )
            report["ok"] = False
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", help="stages to promote (e.g. 1-3)")
    parser.add_argument("--task", default="all_exits_v1")
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--probe", type=int, default=90)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        root = pins_root(args.task) / f"{args.task}_pins"
        for path in sorted(root.glob("*.json")):
            if path.stem in ("resume",):
                continue
            meta = json.loads(path.read_text())
            fp = meta.get("fingerprint", {})
            print(
                f"{path.stem:8s} kind={meta.get('kind'):8s} "
                f"x={fp.get('player_x')} lives={fp.get('lives')}"
            )
        return 0

    records = []
    failures = 0
    for arg in args.stages:
        stage_id = normalize_stage_id(arg)
        if stage_id is None:
            print(f"[FAIL] unknown stage {arg!r}")
            failures += 1
            continue
        record = extract(stage_id, task=args.task, settle=args.settle, probe=args.probe)
        records.append(record)
        status = "[OK]" if record["ok"] else "[FAIL]"
        print(f"{status} {stage_id} -> {record.get('written', 'not written')}")
        if record["issues"]:
            print("   ", "; ".join(record["issues"]))
        failures += 0 if record["ok"] else 1

    out_dir = RECORDINGS_DIR / "segments_all_exits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "extract_stage_state_report.json"
    out_path.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"report → {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
