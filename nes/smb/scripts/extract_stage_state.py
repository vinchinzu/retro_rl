#!/usr/bin/env python3
"""Promote ./play all-exits stage pins into integration practice start states.

The 32-exit track (`smb_all_exits`) has no boot states for non-warp stages.
Human pins under ``recordings/human/<task>_pins/`` are copied into the
SuperMarioBros-Nes-v0 integration as a real start state (e.g. ``Level1_3``).
Nothing here touches the warp any% line (reactive_12, natural_82, HL seeds).

Load fingerprints must match every ``snapshot_fingerprint`` key. After idle
settle the stage must pass ``level_control_gate``, timer > 0, and
``player_x <= CONTROL_X_MAX``. Any mismatch aborts without writing; a failed
roundtrip deletes the file just written.

```bash
uv run python -m smb.scripts.extract_stage_state --list
uv run python -m smb.scripts.extract_stage_state 1-3
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR  # noqa: E402
from smb.ram import SmbSnapshot, read_snapshot  # noqa: E402
from smb.reactive_route import level_control_gate, snapshot_fingerprint  # noqa: E402
from smb.routes import ExitSegment, ROUTE_ALL_EXITS  # noqa: E402
from smb.start_presets import (  # noqa: E402
    normalize_stage_id, pin_meta_path, pin_state_path, pins_dir,
)

# Spawn x at genuine control entry is ~40px; mid-stage pins (2-1 at 2431) fail.
CONTROL_X_MAX = 80


def _all_exits_segment(stage_id: str) -> ExitSegment | None:
    return next((seg for seg in ROUTE_ALL_EXITS.exits if seg.exit_id == stage_id), None)


def pin_promotion_issues(
    stage_id: str,
    *,
    loaded: SmbSnapshot,
    settled: SmbSnapshot,
    fingerprint: Mapping[str, Any],
) -> list[str]:
    """Return promotion blockers; empty means the pin is safe to write.

    After settle, ``player_x`` must be a control spawn (``<= CONTROL_X_MAX``),
    not a mid-stage pose that happens to share world/dash identity.
    """
    issues: list[str] = []
    got = snapshot_fingerprint(loaded)
    for field, actual in got.items():
        want = fingerprint.get(field)
        if want is None:
            issues.append(f"{field}: missing from pin fingerprint")
        elif int(want) != int(actual):
            issues.append(f"{field}: pin={want} loaded={actual}")

    exit_seg = _all_exits_segment(stage_id)
    if exit_seg is None:
        issues.append(f"{stage_id} is not on all_exits")
    elif not level_control_gate(exit_seg).matches(settled):
        issues.append(
            "control gate failed: "
            f"world={settled.world} dash_level={settled.dash_level} "
            f"oper_mode={settled.oper_mode} player_state={settled.player_state} "
            f"(want world={exit_seg.world - 1} dash_level={exit_seg.level - 1})"
        )
    if settled.timer <= 0:
        issues.append("timer not running after settle")
    if settled.player_x > CONTROL_X_MAX:
        issues.append(
            f"player_x={settled.player_x} exceeds control spawn max {CONTROL_X_MAX}"
        )
    return issues


def extract(stage_id: str, *, task: str, settle: int, probe: int) -> dict[str, Any]:
    """Verify + install one stage's pin as an integration start state."""
    from retro_harness.env import make_env, read_state_bytes, write_state_bytes

    world, level = (int(part) for part in stage_id.split("-", 1))
    state_name = f"Level{world}_{level}"
    pin_state = pin_state_path(task, stage_id)
    pin_meta = pin_meta_path(task, stage_id)
    if not pin_state.is_file():
        raise FileNotFoundError(f"no pin state for {stage_id}: {pin_state}")
    fingerprint = json.loads(pin_meta.read_text()).get("fingerprint", {})
    data = read_state_bytes(pin_state)

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        # Stable-retro advances one blank frame on load/reset; apply twice so
        # the observed frame is the exact pinned control entry.
        env.em.set_state(data)
        env.reset()
        env.em.set_state(data)
        loaded = read_snapshot(env.get_ram(), frame=0)
        action = [0] * 9
        for _ in range(settle):
            env.step(action)
        settled = read_snapshot(env.get_ram(), frame=settle)
        issues = pin_promotion_issues(
            stage_id, loaded=loaded, settled=settled, fingerprint=fingerprint
        )
        max_x = max(loaded.player_x, settled.player_x)
        for step_idx in range(probe):
            env.step(action)
            current = read_snapshot(env.get_ram(), frame=settle + step_idx + 1)
            max_x = max(max_x, current.player_x)
            if current.oper_mode != 1:
                issues.append(f"oper_mode={current.oper_mode} @f{settle + step_idx + 1}")

        report: dict[str, Any] = {
            "stage": stage_id,
            "state_name": state_name,
            "pin": str(pin_state),
            "entry_snapshot": snapshot_fingerprint(loaded),
            "settled_snapshot": {
                "dash_level": settled.dash_level, "timer": settled.timer,
                "player_x": settled.player_x, "player_state": settled.player_state,
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
        # fceumm allows one emulator per process; close the pin env first.
        env.close()
        env = None
        try:
            boot_env = make_env(GAME_V0, state_name, GAME_DIR, render_mode="rgb_array")
            try:
                boot_env.reset()
                booted = read_snapshot(boot_env.get_ram(), frame=0)
                exit_seg = _all_exits_segment(stage_id)
                rt_ok = (
                    exit_seg is not None
                    and booted.dash_level == level - 1
                    and booted.oper_mode == 1
                    and level_control_gate(exit_seg).matches(booted)
                )
                report["roundtrip"] = {
                    "dash_level": booted.dash_level,
                    "lives": booted.lives,
                    "ok": rt_ok,
                }
            finally:
                boot_env.close()
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
        if report["roundtrip"]["ok"]:
            report["written"] = str(out_path)
        else:
            out_path.unlink(missing_ok=True)
            report["issues"].append(
                f"roundtrip dash_level={booted.dash_level} oper_mode={booted.oper_mode}"
            )
            report["ok"] = False
        return report
    finally:
        if env is not None:
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
        for path in sorted(pins_dir(args.task).glob("*.json")):
            if path.stem == "resume":
                continue
            meta = json.loads(path.read_text())
            fp = meta.get("fingerprint", {})
            print(
                f"{path.stem:8s} kind={meta.get('kind'):8s} "
                f"x={fp.get('player_x')} lives={fp.get('lives')}"
            )
        return 0
    if not args.stages:
        parser.error("stage id required (or pass --list)")

    records: list[dict[str, Any]] = []
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
