#!/usr/bin/env python3
"""Human record SMB from a *natural reactive* handoff (no W4 pad).

Auto-plays the verified stairs 1-1 + polished reactive 1-2 (+ optional
control-relative 4-1 continuation) to a named gate, then hands the stick to
you. Recordings are control-relative-friendly: full NES-9 buttons + per-frame
RAM trace so ``parse_human_recording`` can pull jump/skill chunks for
hillclimb.

```bash
# Late-route practice: bot to natural 4-1 control, then you play
uv run python -m smb.scripts.record_human --from 4-1 --name late_v1

# Start at World-4 pipe (pre-control) after polished 1-2
uv run python -m smb.scripts.record_human --from w4 --name w4_v1

# Bot drives retimed continuation; press ~ anytime to take over
uv run python -m smb.scripts.record_human --from auto --name pickup_v1

# Full human from Level1_1 (no bot prefix)
uv run python -m smb.scripts.record_human --from 1-1 --name full_v1

# Parse later (or --parse after save)
uv run python -m smb.scripts.parse_human_recording \\
  nes/smb/recordings/human/late_v1.json --export-skills
```

Controls (PlaySession):
  F5 / F1   Save recording + end state, exit
  ESC / Q   Cancel without saving
  ~         Hot-swap bot ↔ human (useful with --from auto)
  [ ] TAB   Speed / turbo
  Arrows    D-pad · Z=B · X=A
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from retro_harness.env import make_env, read_state_bytes
from retro_harness.nes import NES_ACTION_SIZE
from retro_harness.play_session import PlaySession
from retro_harness.segment_runner import configure_headless
from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    compress_nes9_rle,
    expand_nes9_rle,
    load_nes9_rle_seed,
)
from smb.ram import read_snapshot
from smb.reactive_12 import Reactive12Policy, play_reactive_12
from smb.reactive_route import level_control_gate, snapshot_fingerprint
from smb.routes import ROUTE_WARP_ANY_PERCENT
from smb.policy import DEFAULT_STAIRS_1_1 as STAIRS_1_1, play_1_1_until_clear as _play_1_1_until_clear
from smb.reactive_route import (
    DEFAULT_CONTINUATION,
    DEFAULT_CONTINUATION_START,
    KNOWN_41_CONTROL_RESUME,
    continuation_frames as _continuation_frames,
)

LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
HUMAN_DIR = RECORDINGS_DIR / "human"

Handoff = Literal["1-1", "1-2", "w4", "4-1", "4-2", "8-1", "auto"]

HANDOFF_HELP = {
    "1-1": "Level1_1 settle only — full human",
    "1-2": "stairs 1-1 clear → natural 1-2 surface control",
    "w4": "stairs + reactive 1-2 → World-4 pipe detect",
    "4-1": "… → natural 4-1 control (recommended late-route start)",
    "4-2": "… + control-rel 4-1 body → natural 4-2 control",
    "8-1": "… through 4-2 → natural 8-1 control",
    "auto": "bot plays retimed W4+ continuation; ~ to take over anytime",
}


def _stage_label(world: int, level: int) -> str:
    return f"{world + 1}-{level + 1}"


def _trace_row(snap, action: list[int], *, rec_frame: int) -> dict[str, Any]:
    # NES layout: [B, hole, Select, Start, Up, Down, Left, Right, A]
    names_full = ["B", "_", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
    pressed: list[str] = []
    for i, name in enumerate(names_full):
        if name == "_":
            continue
        if i < len(action) and int(action[i]) != 0:
            pressed.append(name)
    return {
        "frame": rec_frame,
        "world": snap.world,
        "level": snap.level,
        "stage": _stage_label(snap.world, snap.level),
        "x": snap.player_x,
        "y": snap.player_y,
        "xs": snap.x_speed,
        "ys": snap.y_speed,
        "player_state": snap.player_state,
        "oper_mode": snap.oper_mode,
        "timer": snap.timer,
        "lives": snap.lives,
        "area_pointer": snap.area_pointer,
        "in_air": bool(snap.in_air),
        "buttons": pressed,
    }


def _play_to_handoff(
    env,
    *,
    handoff: Handoff,
    max_tail: int = 20_000,
) -> dict[str, Any]:
    """Advance env from Level1_1 settle to the requested natural gate.

    Returns prefix metadata + handoff fingerprint. Leaves env at that state.
    """
    idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
    meta: dict[str, Any] = {
        "handoff": handoff,
        "prefix_frames": 0,
        "stages": {},
        "entry": None,
    }

    if handoff == "1-1":
        snap = read_snapshot(env.get_ram())
        meta["entry"] = snapshot_fingerprint(snap)
        meta["prefix_frames"] = 0
        return meta

    stairs = expand_nes9_rle(load_nes9_rle_seed(STAIRS_1_1))
    stage_11 = _play_1_1_until_clear(env, stairs)
    meta["stages"]["1-1"] = {
        "success": stage_11["success"],
        "frames": stage_11["frames"],
        "outcome": stage_11["outcome"],
    }
    if not stage_11["success"]:
        raise RuntimeError(f"1-1 prefix failed: {stage_11['outcome']}")
    meta["prefix_frames"] = int(stage_11["frames"])

    if handoff == "1-2":
        # Idle until surface control on 1-2.
        gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[1])
        for i in range(600):
            snap = read_snapshot(env.get_ram())
            if gate.matches(snap):
                meta["entry"] = snapshot_fingerprint(snap)
                meta["prefix_frames"] += i
                meta["stages"]["1-2_wait"] = i
                return meta
            env.step(idle)
        raise RuntimeError("timeout waiting for 1-2 surface control")

    stage_12 = play_reactive_12(
        env, policy=Reactive12Policy(action_size=int(env.action_space.shape[0]))
    )
    meta["stages"]["1-2"] = {
        "success": stage_12["success"],
        "frames": stage_12["frames"],
        "outcome": stage_12.get("outcome"),
    }
    if not stage_12["success"]:
        raise RuntimeError(f"1-2 prefix failed: {stage_12.get('outcome')}")
    meta["prefix_frames"] += int(stage_12["frames"])

    if handoff == "w4":
        snap = read_snapshot(env.get_ram())
        meta["entry"] = snapshot_fingerprint(snap)
        return meta

    # Remaining handoffs need W4 continuation with control-relative 4-1.
    cont = _continuation_frames(
        DEFAULT_CONTINUATION,
        start=DEFAULT_CONTINUATION_START,
        drop_at=None,
        drop_count=0,
    )
    gate_4_1 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[2])
    gate_4_2 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[3])
    gate_8_1 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[4])
    aligning = True
    source = 0
    last = read_snapshot(env.get_ram())
    target_gate = {
        "4-1": gate_4_1,
        "4-2": gate_4_2,
        "8-1": gate_8_1,
        "auto": None,  # leave at w4; bot continues live
    }[handoff]

    if handoff == "auto":
        meta["entry"] = snapshot_fingerprint(last)
        meta["continuation_ready"] = True
        return meta

    for tail in range(1, max_tail + 1):
        if aligning:
            if gate_4_1.matches(last):
                aligning = False
                source = KNOWN_41_CONTROL_RESUME
                if handoff == "4-1":
                    meta["entry"] = snapshot_fingerprint(last)
                    meta["align_4_1_tail"] = tail
                    meta["prefix_frames"] += tail - 1
                    # Do not consume the resume frame yet — human/bot starts here.
                    return meta
                raw = cont[source]
                source += 1
            else:
                raw = [0] * 9
        else:
            if source >= len(cont):
                raise RuntimeError("continuation exhausted before handoff")
            raw = cont[source]
            source += 1

        action = list(raw[: int(env.action_space.shape[0])])
        if len(action) < int(env.action_space.shape[0]):
            action.extend([0] * (int(env.action_space.shape[0]) - len(action)))
        env.step(np.asarray(action, dtype=np.int8))
        last = read_snapshot(env.get_ram(), frame=tail)

        if last.lives < 2 or last.dying:
            raise RuntimeError(
                f"death during prefix at tail={tail} "
                f"{_stage_label(last.world, last.level)} x={last.player_x}"
            )

        if not aligning and target_gate is not None and target_gate.matches(last):
            meta["entry"] = snapshot_fingerprint(last)
            meta["prefix_frames"] += tail
            meta["continuation_source_index"] = source
            return meta

    raise RuntimeError(f"timeout reaching handoff {handoff!r}")


class _ContinuationBot:
    """Frame-perfect continuation bot with control-relative 4-1 align."""

    def __init__(self, frames: list[list[int]], *, resume_at_control: bool = True):
        self.frames = frames
        self.index = 0
        self.resume_at_control = resume_at_control
        self.aligning = resume_at_control
        self.gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[2])
        self.done = False
        self.align_meta: dict[str, Any] | None = None

    def __call__(self, obs, info) -> list[int]:
        del obs, info
        # PlaySession calls bot before step; we need current RAM from env.
        # Bot is bound after env exists — use bound method with env ref.
        env = self._env
        snap = read_snapshot(env.get_ram())
        if self.aligning:
            if self.gate.matches(snap):
                self.aligning = False
                self.index = KNOWN_41_CONTROL_RESUME
                self.align_meta = {
                    "resume_index": KNOWN_41_CONTROL_RESUME,
                    "entry": snapshot_fingerprint(snap),
                }
                if self.index >= len(self.frames):
                    self.done = True
                    return [0] * 9
                raw = self.frames[self.index]
                self.index += 1
                return list(raw[:9])
            return [0] * 9
        if self.index >= len(self.frames):
            self.done = True
            return [0] * 9
        raw = self.frames[self.index]
        self.index += 1
        return list(raw[:9])

    def bind(self, env) -> None:
        self._env = env


def record_human(
    *,
    handoff: Handoff = "4-1",
    name: str | None = None,
    scale: int = 3,
    out_dir: Path | None = None,
    parse_after: bool = False,
) -> Path | None:
    """Interactive human record. Returns saved JSON path or None if cancelled."""
    if not LEVEL1_1_STATE.exists():
        raise FileNotFoundError(f"missing {LEVEL1_1_STATE}")
    if not STAIRS_1_1.exists():
        raise FileNotFoundError(f"missing {STAIRS_1_1}")

    out = out_dir or HUMAN_DIR
    out.mkdir(parents=True, exist_ok=True)
    task_name = name or f"human_{handoff}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_path = out / f"{task_name}.json"
    end_state_path = out / f"{task_name}_end.state"

    print(f"[REC] building natural prefix → handoff={handoff} …")
    # Build prefix headless (fast), capture state, then open visible session.
    configure_headless()
    boot_env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        boot_env.reset()
        boot_env.em.set_state(read_state_bytes(LEVEL1_1_STATE))
        idle = np.zeros(int(boot_env.action_space.shape[0]), dtype=np.int8)
        for _ in range(CONTINUOUS_SETTLE_FRAMES):
            boot_env.step(idle)
        prefix_meta = _play_to_handoff(boot_env, handoff=handoff)
        handoff_bytes = boot_env.em.get_state()
        handoff_snap = read_snapshot(boot_env.get_ram())
    finally:
        boot_env.close()

    print(
        f"[REC] handoff ready: {_stage_label(handoff_snap.world, handoff_snap.level)} "
        f"x={handoff_snap.player_x} y={handoff_snap.player_y} "
        f"state={handoff_snap.player_state} timer={handoff_snap.timer} "
        f"prefix≈{prefix_meta.get('prefix_frames')}f"
    )
    print(f"[REC] entry fingerprint: {prefix_meta.get('entry')}")

    # Visible session — must NOT use configure_headless permanently.
    # Drop dummy drivers if we set them.
    import os

    for key in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        if os.environ.get(key) == "dummy":
            del os.environ[key]

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    cont_frames = _continuation_frames(
        DEFAULT_CONTINUATION,
        start=DEFAULT_CONTINUATION_START,
        drop_at=None,
        drop_count=0,
    )
    bot: _ContinuationBot | None = None
    if handoff == "auto":
        bot = _ContinuationBot(cont_frames, resume_at_control=True)
        bot.bind(env)

    recorded: list[list[int]] = []
    trace: list[dict[str, Any]] = []
    saved = {"ok": False}
    live: dict[str, Any] = {
        "stage": _stage_label(handoff_snap.world, handoff_snap.level),
        "x": handoff_snap.player_x,
        "y": handoff_snap.player_y,
        "mode": "BOT" if bot else "HUMAN",
    }

    def on_step(obs, reward, done, info) -> None:
        del obs, reward, done, info
        action = list(session.last_action_post_sanitize[:NES_ACTION_SIZE])
        if len(action) < NES_ACTION_SIZE:
            action.extend([0] * (NES_ACTION_SIZE - len(action)))
        # Only record human frames (or all frames if always-human handoff).
        # With auto bot, still record everything so the full path is usable;
        # mark bot frames in trace.
        snap = read_snapshot(env.get_ram(), frame=len(recorded))
        row = _trace_row(snap, action, rec_frame=len(recorded))
        row["source"] = "bot" if session.bot_active else "human"
        recorded.append(action)
        trace.append(row)
        live["stage"] = row["stage"]
        live["x"] = row["x"]
        live["y"] = row["y"]
        live["mode"] = "BOT" if session.bot_active else "HUMAN"

    def on_hud(info) -> list[str]:
        del info
        n_human = sum(1 for r in trace if r.get("source") == "human")
        return [
            f"[REC] {task_name}  F5=save  ESC=cancel  ~=bot/human",
            f"{live['mode']}  {live['stage']}  xy=({live['x']},{live['y']})  "
            f"frames={len(recorded)} human={n_human}",
            f"handoff={handoff}  prefix≈{prefix_meta.get('prefix_frames')}f",
        ]

    def on_key_down(key: int) -> bool:
        import pygame

        if key in (pygame.K_F5, pygame.K_F1):
            _finalize(save=True)
            session.running = False
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            print("[REC] cancelled")
            session.running = False
            return True
        return False

    def _finalize(*, save: bool) -> None:
        if not save or saved["ok"]:
            return
        if not recorded:
            print("[REC] nothing recorded")
            return
        try:
            end_bytes = env.em.get_state()
            end_state_path.write_bytes(end_bytes)
        except Exception as exc:
            print(f"[REC] end-state capture failed: {exc}")
            end_bytes = None

        human_only = [
            frame
            for frame, row in zip(recorded, trace)
            if row.get("source") == "human"
        ]
        payload = {
            "format": "smb_human_nes9",
            "name": task_name,
            "recorded_at": datetime.now().isoformat(),
            "handoff": handoff,
            "handoff_entry": prefix_meta.get("entry"),
            "prefix": prefix_meta,
            "start_state": "Level1_1",
            "settle_frames": CONTINUOUS_SETTLE_FRAMES,
            "game_name": GAME_V0,
            "button_layout": ["B", "_", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"],
            "num_frames": len(recorded),
            "num_human_frames": len(human_only),
            "frames": recorded,
            "human_frames": human_only,
            "trace": trace,
            "segments_rle": compress_nes9_rle(recorded),
            "human_segments_rle": compress_nes9_rle(human_only) if human_only else [],
            "end_state": str(end_state_path.relative_to(GAME_DIR))
            if end_state_path.exists()
            else None,
        }
        task_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        saved["ok"] = True
        print(f"[REC] saved {task_path} ({len(recorded)} frames, human={len(human_only)})")
        if end_state_path.exists():
            print(f"[REC] end state → {end_state_path}")
        print(
            f"[REC] parse: uv run python -m smb.scripts.parse_human_recording "
            f"{task_path} --export-skills"
        )
        if parse_after:
            from smb.scripts.parse_human_recording import parse_and_export

            parse_and_export(task_path, do_export=True)

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME_V0,
        scale=scale,
        title=f"SMB human REC: {task_name} [{handoff}]",
        bot=bot,
        action_size=NES_ACTION_SIZE,
        base_fps=60,
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down

    # PlaySession.run() always reset() first — re-inject handoff state.
    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_handoff(e):
        obs, info = _orig_reset(e)
        e.em.set_state(handoff_bytes)
        # No extra idle: handoff_bytes is already a controllable natural gate.
        # (An injected settle desyncs deterministic replay of human frames.)
        snap = read_snapshot(e.get_ram())
        live["stage"] = _stage_label(snap.world, snap.level)
        live["x"] = snap.player_x
        live["y"] = snap.player_y
        print(
            f"[BOOT] {_stage_label(snap.world, snap.level)} "
            f"xy=({snap.player_x},{snap.player_y}) handoff={handoff}"
        )
        return obs, info

    ps_mod.reset_env = _reset_then_handoff  # type: ignore[assignment]
    try:
        print("[REC] window open — F5 save · ESC cancel · ~ bot/human")
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]

    return task_path if saved["ok"] else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from",
        dest="handoff",
        default="4-1",
        choices=list(HANDOFF_HELP),
        help="Natural reactive handoff (default: 4-1)",
    )
    parser.add_argument("--name", default=None, help="Recording stem under recordings/human/")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output dir (default: {HUMAN_DIR})",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Run skill parser immediately after a successful save",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List handoff presets and exit",
    )
    args = parser.parse_args()
    if args.list:
        print("Handoff presets (all natural / control-relative — no W4 pad):\n")
        for key, desc in HANDOFF_HELP.items():
            print(f"  {key:6s}  {desc}")
        return
    path = record_human(
        handoff=args.handoff,
        name=args.name,
        scale=args.scale,
        out_dir=args.out_dir,
        parse_after=args.parse,
    )
    if path is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
