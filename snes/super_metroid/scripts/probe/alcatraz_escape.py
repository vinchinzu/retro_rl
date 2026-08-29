#!/usr/bin/env python3
"""Post-Torizo Alcatraz left-chimney WJ + instant-morph roll-out.

Pin: ``scratch/post_torizo_parlor_continuous.state`` (Flyway door in Parlor).
Goal: land the shaft lip (y<=210) and roll left through the morph opening.

```bash
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py run --no-video
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py dual --no-video
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py record
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py search
```

Overwrite ``scratch/alcatraz_escape_dual.json``. Do not STATUS-promote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from retro_harness.actions import idle_action
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.probe import open_state_env, write_json_report
from super_metroid.dev.common import save_dev_state
from super_metroid.hop_glance import final_from_state
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr import alcatraz_escape as alcatraz
from super_metroid.routes.kpdr.alcatraz_escape import (
    SHAFT_LIP_Y,
    at_alcatraz_rollout,
    at_left_wall_base,
    at_mid_ledge,
    play_alcatraz_escape,
)
from super_metroid.routes.runtime import hold

DEFAULT_SOURCE = SCRATCH_STATE_DIR / "post_torizo_parlor_continuous.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_alcatraz_escape.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "alcatraz_escape.json"
DEFAULT_DUAL = GAME_DIR / "scratch" / "alcatraz_escape_dual.json"
DEFAULT_VIDEO = RECORDINGS_DIR / "alcatraz_escape.mp4"
DEFAULT_SHOT = GAME_DIR / "scratch" / "alcatraz_escape_leave.png"
BOOT_SETTLE = 5
HOP = "alcatraz_escape"


class _Sess:
    def __init__(
        self,
        env: Any,
        assist: UnlimitedResourcesAssist | None,
        writer: VideoRecorder | None = None,
    ) -> None:
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.min_y = int(self.state.samus_y)
        self.min_y_xy = [int(self.state.samus_x), int(self.state.samus_y)]
        self.trace: list[dict[str, Any]] = []

    def step(self, action, reason: str = ""):
        obs, *_ = self.env.step(action)
        self.frame += 1
        if self.assist is not None:
            st = parse_env_state(self.env, frame=self.frame, mode="nav")
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        y = int(self.state.samus_y)
        if y < self.min_y:
            self.min_y = y
            self.min_y_xy = [int(self.state.samus_x), y]
        if self.frame % 6 == 0 or int(self.state.pose) in (131, 132) or "morph" in reason:
            self.trace.append(
                {
                    "f": self.frame,
                    "x": int(self.state.samus_x),
                    "y": y,
                    "p": int(self.state.pose),
                    "r": reason,
                }
            )
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state


def _save_shot(env: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(path)
    return path


def _snap(st: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = final_from_state(st)
    out.update(
        {
            "items": f"0x{int(st.collected_items):04X}",
            "pose": int(st.pose),
            "facing": int(st.facing),
            "vx": int(st.velocity_x),
            "vy": int(st.velocity_y),
            "probePin": probe_pin(st),
        }
    )
    if extra:
        out.update(extra)
    return out


def _open(source: Path, *, settle: int = BOOT_SETTLE):
    return open_state_env(
        source,
        settle=settle,
        missing_hint="Need post_torizo_parlor_continuous.state",
    )


def run_once(
    source: Path,
    *,
    video_path: Path | None = None,
    out_pin: Path | None = None,
    scale: int = 2,
    crf: int = 20,
) -> dict[str, Any]:
    env, resolved = _open(source)
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    error: str | None = None
    t0 = time.perf_counter()
    session: _Sess | None = None
    try:
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            obs = env.render()
            writer = VideoRecorder(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=VideoCaptureConfig(
                    fps=60, scale=scale, crf=crf, audio=False, footer=True
                ),
                audio_rate=None,
            )
            writer.write_from_env(
                env,
                obs,
                action=None,
                frame_index=0,
                room_id=int(parse_env_state(env, mode="nav").room_id),
            )
        session = _Sess(env, assist, writer)
        start = _snap(session.state)
        try:
            evidence = play_alcatraz_escape(session)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            evidence = None
        final = _snap(
            session.state,
            extra={
                "minY": session.min_y,
                "minYxy": session.min_y_xy,
                "frames": session.frame,
            },
        )
        png = DEFAULT_SHOT if error is None else GAME_DIR / "scratch" / "alcatraz_escape_red.png"
        _save_shot(env, png)
        if out_pin is not None and error is None:
            save_dev_state(env, out_pin)
    finally:
        encoded = 0
        if writer is not None:
            encoded = writer.frames
            writer.close()
        env.close()

    assert session is not None
    rolled = at_alcatraz_rollout(session.state)
    lip = session.min_y <= SHAFT_LIP_Y
    success = error is None and rolled
    timing = format_segment_time(session.frame)
    report = {
        "kind": HOP,
        "source": resolved,
        "success": success,
        "error": error,
        "frames": session.frame,
        "seconds": timing["seconds"],
        "clock": timing["clock"],
        "minY": session.min_y,
        "minYxy": session.min_y_xy,
        "lipClass": lip,
        "rolledOut": rolled,
        "start": start,
        "final": final,
        "evidence": None if evidence is None else evidence.to_dict(),
        "video": None if video_path is None else str(video_path),
        "encodedFrames": encoded,
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "shot": str(png),
        "trace": session.trace[-80:],
    }
    write_json_report(report, None)
    return report


def _boot_session(source: Path) -> tuple[Any, _Sess]:
    env, _resolved = _open(source)
    session = _Sess(env, UnlimitedResourcesAssist(), None)
    return env, session


def _one_jump_base(session: _Sess, *, run: int = 30, jump: int = 16) -> None:
    hold(session, 2, "LEFT", reason="face")
    hold(session, run, "LEFT", "B", reason="run")
    hold(session, jump, "LEFT", "A", reason="hop")
    hold(session, 16, reason="land")
    alcatraz._unmorph_probe_pose(session)


def _ledge_cycle(session: _Sess, spin: int, latch: int) -> bool:
    grounded = {1, 2, 5, 6, 7, 8, 9, 10}
    hold(session, 2, "RIGHT", reason="face_r")
    for _ in range(spin):
        hold(session, 1, "RIGHT", "A", reason="ledge_spin")
        if int(session.state.samus_y) <= 470 and int(session.state.pose) in grounded:
            return True
    hold(session, 2, "LEFT", reason="face_l")
    for _ in range(latch):
        hold(session, 1, "LEFT", "A", reason="ledge_latch")
        if int(session.state.samus_y) <= 470 and int(session.state.pose) in grounded:
            return True
    return at_mid_ledge(session.state)


def search_approach(source: Path) -> dict[str, Any]:
    """Commit the one-jump + 3-cycle ledge + unmorph, then WJ."""
    env, session = _boot_session(source)
    try:
        hold(session, 2, "LEFT", reason="face")
        hold(session, 30, "LEFT", "B", reason="run")
        hold(session, 18, "LEFT", "A", reason="hop")
        hold(session, 16, reason="land")
        alcatraz._unmorph_probe_pose(session)
        hold(session, 2, "RIGHT", reason="face_r")
        for i in range(3):
            hold(session, 40, "RIGHT", "A", reason=f"spin{i}")
            hold(session, 2, "LEFT", reason="face_l")
            hold(session, 28, "LEFT", "A", reason=f"latch{i}")
        hold(session, 16, reason="settle")
        alcatraz._unmorph_probe_pose(session)
        hold(session, 8, reason="stand")
        st = session.state
        ledge = {
            "f": session.frame,
            "xy": [int(st.samus_x), int(st.samus_y)],
            "pose": int(st.pose),
            "mid": at_mid_ledge(st),
        }
        _save_shot(env, GAME_DIR / "scratch" / "alcatraz_tight_ledge.png")
        try:
            wj, morph = alcatraz._climb_chimney(session)
            error = None
        except Exception as exc:  # noqa: BLE001
            wj, morph = -1, -1
            error = str(exc)
        st = session.state
        payload = {
            "kind": "alcatraz_approach_search",
            "ledge": ledge,
            "rolled": error is None and at_alcatraz_rollout(st),
            "f": session.frame,
            "minY": session.min_y,
            "exit": [int(st.samus_x), int(st.samus_y)],
            "pose": int(st.pose),
            "wj": wj,
            "morph": morph,
            "error": error,
        }
        _save_shot(env, GAME_DIR / "scratch" / "alcatraz_tight_final.png")
    finally:
        env.close()
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload
    variants: list[dict[str, Any]] = []
    recipes = (
        {"run": 30, "jump": 16, "cycles": 2, "settle": 8},
        {"run": 30, "jump": 16, "cycles": 3, "settle": 8},
        {"run": 30, "jump": 18, "cycles": 2, "settle": 8},
        {"run": 30, "jump": 18, "cycles": 3, "settle": 8},
        {"run": 26, "jump": 22, "cycles": 2, "settle": 8},
        {"run": 26, "jump": 22, "cycles": 3, "settle": 8},
        {"run": 30, "jump": 16, "cycles": 3, "settle": 25},
        {"run": 30, "jump": 18, "cycles": 3, "settle": 25},
    )
    grounded = {1, 2, 5, 6, 7, 8, 9, 10}
    for rec in recipes:
        env, session = _boot_session(source)
        try:
            hold(session, 2, "LEFT", reason="face")
            hold(session, rec["run"], "LEFT", "B", reason="run")
            hold(session, rec["jump"], "LEFT", "A", reason="hop")
            hold(session, 16, reason="land")
            alcatraz._unmorph_probe_pose(session)
            hold(session, 2, "RIGHT", reason="face_r")
            landed = False
            used = 0
            for i in range(rec["cycles"]):
                used = i + 1
                hold(session, 40, "RIGHT", "A", reason=f"spin{i}")
                hold(session, 2, "LEFT", reason="face_l")
                hold(session, 28, "LEFT", "A", reason=f"latch{i}")
                if int(session.state.samus_y) <= 470 and int(session.state.pose) in grounded:
                    landed = True
                    break
            hold(session, rec["settle"], reason="settle")
            st = session.state
            ledge = at_mid_ledge(st) or landed
            row: dict[str, Any] = {
                **rec,
                "used": used,
                "ledge": ledge,
                "baseF": 2 + rec["run"] + rec["jump"] + 16,
                "xy": [int(st.samus_x), int(st.samus_y)],
                "pose": int(st.pose),
                "rolled": False,
                "error": None,
            }
            if ledge:
                try:
                    wj, morph = alcatraz._climb_chimney(session)
                    row.update(
                        {
                            "rolled": at_alcatraz_rollout(session.state),
                            "minY": session.min_y,
                            "wj": wj,
                            "morph": morph,
                            "exit": [
                                int(session.state.samus_x),
                                int(session.state.samus_y),
                            ],
                            "f": session.frame,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    row["error"] = str(exc)
                    row["f"] = session.frame
            else:
                row["f"] = session.frame
            variants.append(row)
            _save_shot(env, GAME_DIR / "scratch" / f"alcatraz_r{rec['run']}_j{rec['jump']}_c{rec['cycles']}.png")
        finally:
            env.close()
    variants.sort(key=lambda r: (not r.get("rolled"), r.get("f", 9999)))
    payload = {
        "kind": "alcatraz_approach_search",
        "variants": variants,
        "best": variants[0] if variants else None,
    }
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload
    variants: list[dict[str, Any]] = []
    for approach, extra in (
        ("direct", ()),
        ("nudge", (("LEFT", 4),)),
        ("into", (("LEFT", 3), ("LEFT", "B", "A", 12), ("LEFT", 2))),
        ("into22", (("LEFT", 3), ("LEFT", "B", "A", 22), ("LEFT", 4))),
    ):
        env, session = _boot_session(source)
        try:
            _one_jump_base(session)
            for spec in extra:
                *buttons, frames = spec
                hold(session, int(frames), *buttons, reason="wj_setup")
            try:
                walljump_frame, morph_frame = alcatraz._climb_chimney(session)
                error = None
            except Exception as exc:  # noqa: BLE001
                walljump_frame = morph_frame = -1
                error = str(exc)
            st = session.state
            row = {
                "approach": approach,
                "rolled": error is None and at_alcatraz_rollout(st),
                "f": session.frame,
                "minY": session.min_y,
                "xy": [int(st.samus_x), int(st.samus_y)],
                "pose": int(st.pose),
                "wj": walljump_frame,
                "morph": morph_frame,
                "error": error,
            }
            variants.append(row)
            _save_shot(env, GAME_DIR / "scratch" / f"alcatraz_{approach}.png")
        finally:
            env.close()

    # Also: one-jump + 2-cycle mid-ledge (product-like, no third bonk).
    env, session = _boot_session(source)
    try:
        _one_jump_base(session)
        hold(session, 2, "RIGHT", reason="face_r")
        for i in range(2):
            hold(session, 40, "RIGHT", "A", reason=f"spin{i}")
            hold(session, 2, "LEFT", reason="face_l")
            hold(session, 28, "LEFT", "A", reason=f"latch{i}")
        hold(session, 12, reason="settle")
        st = session.state
        ledge = at_mid_ledge(st) or int(st.samus_y) <= 475
        row = {
            "approach": "two_cycle",
            "ledge": ledge,
            "f": session.frame,
            "xy": [int(st.samus_x), int(st.samus_y)],
            "pose": int(st.pose),
            "rolled": False,
            "error": None,
        }
        if ledge:
            try:
                wj, morph = alcatraz._climb_chimney(session)
                row.update(
                    {
                        "rolled": at_alcatraz_rollout(session.state),
                        "minY": session.min_y,
                        "wj": wj,
                        "morph": morph,
                        "xy": [int(session.state.samus_x), int(session.state.samus_y)],
                        "pose": int(session.state.pose),
                        "f": session.frame,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                row["f"] = session.frame
        variants.append(row)
        _save_shot(env, GAME_DIR / "scratch" / "alcatraz_two_cycle.png")
    finally:
        env.close()

    variants.sort(key=lambda r: (not r.get("rolled"), r.get("f", 9999)))
    payload = {
        "kind": "alcatraz_approach_search",
        "base": {"run": 30, "jump": 16, "xy": [809, 549]},
        "variants": variants,
        "best": variants[0] if variants else None,
    }
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload
    grounded = {1, 2, 5, 6, 7, 8, 9, 10}
    env, session = _boot_session(source)
    try:
        _one_jump_base(session)
        _save_shot(env, GAME_DIR / "scratch" / "alcatraz_base.png")
        save_dev_state(env, SCRATCH_STATE_DIR / "post_alcatraz_base.state")
        base_xy = [int(session.state.samus_x), int(session.state.samus_y)]
        base_pose = int(session.state.pose)
        base_f = session.frame
    finally:
        env.close()

    ledges: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    for walk in (0, 2, 4, 6, 8, 10):
        for jump in (8, 10, 12, 14, 16, 18, 20, 24):
            for dir_ in (("RIGHT", "A"), ("A",), ("RIGHT", "A", "B")):
                env, session = _boot_session(source)
                try:
                    _one_jump_base(session)
                    if walk:
                        hold(session, walk, "RIGHT", reason="walk_r")
                    hold(session, jump, *dir_, reason="ledge_hop")
                    hold(session, 16, reason="ledge_land")
                    alcatraz._unmorph_probe_pose(session)
                    st = session.state
                    ok = at_mid_ledge(st) or (
                        int(st.samus_y) <= 475
                        and int(st.pose) in grounded
                        and 800 <= int(st.samus_x) <= 860
                    )
                    row = {
                        "walk": walk,
                        "jump": jump,
                        "dir": list(dir_),
                        "f": session.frame,
                        "xy": [int(st.samus_x), int(st.samus_y)],
                        "pose": int(st.pose),
                        "ledge": ok,
                    }
                    ledges.append(row)
                    if not ok:
                        continue
                    try:
                        walljump_frame, morph_frame = alcatraz._climb_chimney(session)
                        rolled = at_alcatraz_rollout(session.state)
                        full.append(
                            {
                                **row,
                                "rolled": rolled,
                                "minY": session.min_y,
                                "exit": [
                                    int(session.state.samus_x),
                                    int(session.state.samus_y),
                                ],
                                "exitPose": int(session.state.pose),
                                "wj": walljump_frame,
                                "morph": morph_frame,
                                "total": session.frame,
                                "error": None,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        full.append(
                            {
                                **row,
                                "rolled": False,
                                "total": session.frame,
                                "error": str(exc),
                            }
                        )
                finally:
                    env.close()
    ledges.sort(key=lambda r: (not r["ledge"], r["xy"][1], r["f"]))
    full.sort(key=lambda r: (not r.get("rolled"), r.get("total", 9999)))
    payload = {
        "kind": "alcatraz_approach_search",
        "base": {"f": base_f, "xy": base_xy, "pose": base_pose, "run": 30, "jump": 16},
        "bestHeight": ledges[:8],
        "ledgeHits": [r for r in ledges if r["ledge"]],
        "fullHits": full[:8],
        "best": full[0] if full else None,
    }
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload
    grounded = {1, 2, 5, 6, 7, 8, 9, 10}
    plats: list[dict[str, Any]] = []
    for run in (8, 10, 12, 14, 16, 18, 20, 22):
        for jump in (10, 12, 14, 16, 18, 20, 22):
            env, session = _boot_session(source)
            try:
                hold(session, 2, "LEFT", reason="face")
                hold(session, run, "LEFT", "B", reason="run")
                hold(session, jump, "LEFT", "A", reason="hop")
                hold(session, 14, reason="land")
                alcatraz._unmorph_probe_pose(session)
                st = session.state
                x, y, p = int(st.samus_x), int(st.samus_y), int(st.pose)
                plat = (
                    int(st.pose) in grounded
                    and 860 <= x <= 930
                    and 520 <= y <= 560
                )
                plats.append(
                    {
                        "run": run,
                        "jump": jump,
                        "f": session.frame,
                        "xy": [x, y],
                        "pose": p,
                        "plat": plat,
                    }
                )
            finally:
                env.close()
    plats.sort(key=lambda r: (not r["plat"], abs(r["xy"][0] - 895), r["f"]))
    hits = [r for r in plats if r["plat"]]

    ledges: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    for hop in hits[:5]:
        for wait in (0, 4):
            for hop2 in (12, 16, 20, 24, 28):
                env, session = _boot_session(source)
                try:
                    hold(session, 2, "LEFT", reason="face")
                    hold(session, hop["run"], "LEFT", "B", reason="run")
                    hold(session, hop["jump"], "LEFT", "A", reason="hop")
                    hold(session, 14, reason="land")
                    alcatraz._unmorph_probe_pose(session)
                    if wait:
                        hold(session, wait, reason="plat_settle")
                    hold(session, hop2, "LEFT", "A", reason="ledge_hop")
                    hold(session, 14, reason="ledge_land")
                    alcatraz._unmorph_probe_pose(session)
                    st = session.state
                    ok = at_mid_ledge(st) or (
                        int(st.samus_y) <= 475
                        and int(st.pose) in grounded
                        and 810 <= int(st.samus_x) <= 870
                    )
                    row = {
                        "run": hop["run"],
                        "jump": hop["jump"],
                        "wait": wait,
                        "hop2": hop2,
                        "f": session.frame,
                        "xy": [int(st.samus_x), int(st.samus_y)],
                        "pose": int(st.pose),
                        "ledge": ok,
                    }
                    ledges.append(row)
                    if not ok:
                        continue
                    try:
                        walljump_frame, morph_frame = alcatraz._climb_chimney(session)
                        rolled = at_alcatraz_rollout(session.state)
                        full.append(
                            {
                                **row,
                                "rolled": rolled,
                                "minY": session.min_y,
                                "exit": [
                                    int(session.state.samus_x),
                                    int(session.state.samus_y),
                                ],
                                "exitPose": int(session.state.pose),
                                "wj": walljump_frame,
                                "morph": morph_frame,
                                "total": session.frame,
                                "error": None,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        full.append(
                            {
                                **row,
                                "rolled": False,
                                "total": session.frame,
                                "error": str(exc),
                            }
                        )
                finally:
                    env.close()
    full.sort(key=lambda r: (not r.get("rolled"), r.get("total", 9999)))
    payload = {
        "kind": "alcatraz_approach_search",
        "platHits": hits[:8],
        "platNear": plats[:6],
        "ledgeHits": [r for r in ledges if r["ledge"]],
        "ledgeMisses": [r for r in ledges if not r["ledge"]][:8],
        "fullHits": full[:8],
        "best": full[0] if full else None,
    }
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload
    grounded = {1, 2, 5, 6, 7, 8, 9, 10}
    ledges: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    for settle in (0, 6):
        for cycles in (1, 2):
            for spin, latch in ((28, 24), (32, 26), (36, 28), (40, 28), (44, 30)):
                env, session = _boot_session(source)
                try:
                    _one_jump_base(session)
                    if settle:
                        hold(session, settle, reason="base_settle")
                    ok = False
                    for _ in range(cycles):
                        if _ledge_cycle(session, spin, latch):
                            ok = True
                            break
                    hold(session, 8, reason="ledge_settle")
                    st = session.state
                    ok = ok or at_mid_ledge(st) or (
                        int(st.samus_y) <= 475 and int(st.pose) in grounded
                    )
                    row = {
                        "settle": settle,
                        "cycles": cycles,
                        "spin": spin,
                        "latch": latch,
                        "f": session.frame,
                        "xy": [int(st.samus_x), int(st.samus_y)],
                        "pose": int(st.pose),
                        "ledge": ok,
                    }
                    ledges.append(row)
                    if not ok:
                        continue
                    try:
                        walljump_frame, morph_frame = alcatraz._climb_chimney(session)
                        rolled = at_alcatraz_rollout(session.state)
                        full.append(
                            {
                                **row,
                                "rolled": rolled,
                                "minY": session.min_y,
                                "exit": [
                                    int(session.state.samus_x),
                                    int(session.state.samus_y),
                                ],
                                "exitPose": int(session.state.pose),
                                "wj": walljump_frame,
                                "morph": morph_frame,
                                "total": session.frame,
                                "error": None,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        full.append(
                            {
                                **row,
                                "rolled": False,
                                "total": session.frame,
                                "error": str(exc),
                            }
                        )
                finally:
                    env.close()
    full.sort(key=lambda r: (not r.get("rolled"), r.get("total", 9999)))
    payload = {
        "kind": "alcatraz_approach_search",
        "base": {"run": 30, "jump": 16},
        "ledgeMisses": [r for r in ledges if not r["ledge"]][:6],
        "ledgeHits": [r for r in ledges if r["ledge"]],
        "fullHits": full[:8],
        "best": full[0] if full else None,
    }
    path = GAME_DIR / "scratch" / "alcatraz_approach_search.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


def dual(source: Path, *, video: bool) -> dict[str, Any]:
    rows = []
    for i in range(2):
        video_path = None
        if video:
            video_path = RECORDINGS_DIR / f"alcatraz_escape_dual{i}.mp4"
        row = run_once(source, video_path=video_path)
        rows.append(row)
        if not row["success"]:
            break
    match = len(rows) == 2 and all(r["success"] for r in rows)
    payload = {
        "kind": "alcatraz_escape_dual",
        "match": match,
        "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows],
    }
    DEFAULT_DUAL.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DUAL.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("run", "dual", "record", "search"))
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    video = None
    if args.cmd == "record" or (args.cmd == "run" and not args.no_video):
        video = DEFAULT_VIDEO
    if args.cmd == "search":
        payload = search_approach(args.source)
        ok = bool(payload.get("rolled") or (payload.get("best") or {}).get("rolled"))
        raise SystemExit(0 if ok else 1)
    if args.cmd == "dual":
        payload = dual(args.source, video=not args.no_video)
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
        for i, row in enumerate(payload["rows"]):
            slim = {k: v for k, v in row.items() if k not in ("trace", "start")}
            print(f"row{i}: {json.dumps(slim, indent=2)}")
        raise SystemExit(0 if payload["match"] else 1)

    out_pin = args.out if args.cmd != "record" else None
    report = run_once(args.source, video_path=video, out_pin=out_pin)
    DEFAULT_REPORT.write_text(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trace", "start")}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
