#!/usr/bin/env python3
"""Bubble Mountain Save → runway walljump practice gym.

Boots the **bubble-save** pin (0xB0DD save station, items 0x1105 from
``full_start_v1`` end) and gives live terminal feedback on walljump timing:

  EARLY  A pressed N frames before wall latch (pose 132)
  LATE   A pressed N frames after latch window ended / delayed into latch
  ON     A within the ideal early window of latch
  MISS   latched but never jumped

```bash
# Multi-take gym (recommended)
uv run python snes/super_metroid/scripts/probe/bubble_save_practice.py

# Named series under tasks/bubble_save_practice/<series>/
uv run python snes/super_metroid/scripts/probe/bubble_save_practice.py \\
  --series bubble_wj_v1

# Free-record from same pin (anchors + materialize; no live WJ grades)
./snes/super_metroid/play bubble-save full_start_v1
```

Controls:
  **SELECT+R2**  save checkpoint 1 (practice mid seat)
  **SELECT+L2**  load checkpoint 1  (boot seeds CP1 = pin)
  **R**          hard reload pin + re-seed CP1
  **F5**         save take + diagnose + reload pin
  **F1**         save take + diagnose (stay put)
  **ESC/Q**      quit
  ``[`` ``]`` / TAB  speed / turbo

Recipe (leave trap → climb):
  1. Unmorph, walk RIGHT out of Save 0xB0DD (door)
  2. On Bubble 0xACB3 save-runway y≈395, seat max-left x∈[25,32]
  3. RIGHT+B dash ~20f → spin-glide RIGHT+B+A → wall contact
  4. On latch (pose 132): press A (into wall) then flip — **double WJ**
  5. Goal: Phase D x≥300 y≤200 → Super door top-right
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX  # noqa: E402
from retro_harness.env import make_env, read_state_bytes, write_state_bytes  # noqa: E402
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from retro_harness.task_recording import RecordedTask, pressed_buttons  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import POSE_WALL_LATCH  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_PIN = SCRATCH / "bubble_save.state"
FALLBACK_PINS = (
    SCRATCH / "full_start_v1_bubble_save.state",
    GAME_DIR / "tasks" / "bubble_save.state",
    GAME_DIR / "tasks" / "full_start_v1_end.state",
    GAME_DIR / "tasks" / "full_start_v1_anchors" / "f011198_end_0xB0DD.state",
)
ROOM_SAVE = 0xB0DD
ROOM_BUBBLE = 0xACB3
TASKS_ROOT = GAME_DIR / "tasks" / "bubble_save_practice"

# Ideal: press A on the first 0..IDEAL_MAX frames of wall latch.
IDEAL_MAX = 2
# A within this many frames before latch counts as EARLY (not random earlier A).
EARLY_LOOKBACK = 8
# A after latch ends within this window counts as LATE (missed window).
LATE_LOOKAHEAD = 20
# Phase D / height class (same as bubble_to_bat policy).
PHASE_D_X = 300
PHASE_D_Y = 200
HEIGHT_CLASS_Y = 280


# ---------------------------------------------------------------------------
# Diagnosis (pure — unit-testable)
# ---------------------------------------------------------------------------


def _btn_set(row: dict[str, Any]) -> set[str]:
    return {str(b).upper() for b in (row.get("buttons") or [])}


def _latch_windows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Continuous pose-132 spans with A timing vs latch start/end."""
    windows: list[dict[str, Any]] = []
    start_i: int | None = None
    for i, row in enumerate(trace):
        latched = int(row.get("pose") or 0) == POSE_WALL_LATCH
        if latched and start_i is None:
            start_i = i
        elif not latched and start_i is not None:
            windows.append(_finish_latch_window(trace, start_i, i - 1))
            start_i = None
    if start_i is not None:
        windows.append(_finish_latch_window(trace, start_i, len(trace) - 1))
    return windows


def _finish_latch_window(
    trace: list[dict[str, Any]], start_i: int, end_i: int
) -> dict[str, Any]:
    stretch = trace[start_i : end_i + 1]
    start = stretch[0]
    end = stretch[-1]
    f0 = int(start.get("frame") or start_i)
    f1 = int(end.get("frame") or end_i)

    a_during: dict[str, Any] | None = None
    for r in stretch:
        if "A" in _btn_set(r):
            a_during = r
            break

    a_before: dict[str, Any] | None = None
    for r in reversed(trace[max(0, start_i - EARLY_LOOKBACK) : start_i]):
        if "A" in _btn_set(r):
            a_before = r
            break

    a_after: dict[str, Any] | None = None
    for r in trace[end_i + 1 : end_i + 1 + LATE_LOOKAHEAD]:
        if "A" in _btn_set(r):
            a_after = r
            break

    grade = "MISS"
    frames_off: int | None = None
    note = "latched but no A near window"
    a_frame: int | None = None

    if a_during is not None:
        a_frame = int(a_during.get("frame") or 0)
        frames_off = a_frame - f0
        if frames_off <= IDEAL_MAX:
            grade = "ON"
            note = f"A on latch +{frames_off}f (ideal 0..{IDEAL_MAX})"
        else:
            grade = "LATE"
            note = f"A late into latch +{frames_off}f (ideal 0..{IDEAL_MAX})"
    elif a_before is not None:
        a_frame = int(a_before.get("frame") or 0)
        frames_off = a_frame - f0  # negative
        grade = "EARLY"
        note = f"A {abs(frames_off)}f BEFORE latch — hold into wall, A on 132"
    elif a_after is not None:
        a_frame = int(a_after.get("frame") or 0)
        # late relative to latch end (missed the window)
        frames_off = a_frame - f1
        grade = "LATE"
        note = (
            f"A {frames_off}f AFTER latch ended "
            f"(window was f{f0}–f{f1}, hold {f1 - f0 + 1}f)"
        )

    return {
        "latch_start": f0,
        "latch_end": f1,
        "latch_frames": end_i - start_i + 1,
        "xy_start": (int(start.get("x") or 0), int(start.get("y") or 0)),
        "xy_end": (int(end.get("x") or 0), int(end.get("y") or 0)),
        "a_frame": a_frame,
        "frames_off": frames_off,
        "grade": grade,
        "note": note,
        "room": int(start.get("room") or 0),
        "room_hex": start.get("room_hex")
        or f"0x{int(start.get('room') or 0):04X}",
    }


def diagnose_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Grade a practice take from per-frame rows (pose/buttons/xy/room)."""
    if not trace:
        return {
            "ok": False,
            "grade": "EMPTY",
            "failures": ["no frames"],
            "cues": ["boot pin and attempt the runway walljump"],
            "windows": [],
            "recipe": _recipe(),
        }

    windows = _latch_windows(trace)
    min_y = min(int(r.get("y") or 9999) for r in trace)
    max_x = max(int(r.get("x") or 0) for r in trace)
    rooms = sorted({int(r.get("room") or 0) for r in trace})
    left_save = ROOM_BUBBLE in rooms or any(
        int(r.get("room") or 0) == ROOM_BUBBLE for r in trace
    )
    end = trace[-1]
    end_room = int(end.get("room") or 0)
    end_xy = (int(end.get("x") or 0), int(end.get("y") or 0))

    phase_d = any(
        int(r.get("room") or 0) == ROOM_BUBBLE
        and int(r.get("x") or 0) >= PHASE_D_X
        and int(r.get("y") or 0) <= PHASE_D_Y
        for r in trace
    )
    height_class = min_y <= HEIGHT_CLASS_Y and left_save

    failures: list[str] = []
    cues: list[str] = []
    if not left_save and end_room == ROOM_SAVE:
        failures.append("still in Save 0xB0DD — walk RIGHT to door, leave trap")
        cues.append("Unmorph → RIGHT to exit → Bubble save-runway y≈395")
    if left_save and not windows:
        failures.append("reached Bubble but no wall latch (pose 132)")
        cues.append(
            "From fire seat x∈[25,32] y≈395: RIGHT+B ~20f then RIGHT+B+A spin into wall"
        )
    if windows:
        early_n = sum(1 for w in windows if w["grade"] == "EARLY")
        late_n = sum(1 for w in windows if w["grade"] == "LATE")
        miss_n = sum(1 for w in windows if w["grade"] == "MISS")
        on_n = sum(1 for w in windows if w["grade"] == "ON")
        if early_n:
            failures.append(f"{early_n} EARLY walljump(s) — A before latch")
            cues.append("Wait for pose 132, then A (into wall) immediately")
        if late_n:
            failures.append(f"{late_n} LATE walljump(s) — A after window")
            cues.append("A sooner on latch (ideal 0–2f after 132)")
        if miss_n:
            failures.append(f"{miss_n} MISS — latched without A")
            cues.append("Hold toward wall, tap A on latch, then flip away+A")
        if on_n and len(windows) == 1 and not phase_d:
            cues.append("Need **double** WJ — one latch is not enough for Phase D")
        if on_n >= 2 and not phase_d:
            cues.append("Double WJ landed but short of Phase D — more spin follow / seat")

    if phase_d:
        grade = "GREEN"
        ok = True
        cues = ["Phase D reached — Super door top-right next"]
        failures = []
    elif height_class and windows and any(w["grade"] == "ON" for w in windows):
        grade = "YELLOW"
        ok = False
        if not failures:
            failures.append("height class ok but not Phase D (x≥300 y≤200)")
    elif windows:
        grade = "RED"
        ok = False
    elif left_save:
        grade = "RED"
        ok = False
    else:
        grade = "RED"
        ok = False

    return {
        "ok": ok,
        "grade": grade,
        "phase_d": phase_d,
        "height_class": height_class,
        "min_y": min_y,
        "max_x": max_x,
        "rooms": [f"0x{r:04X}" for r in rooms],
        "end_room": f"0x{end_room:04X}",
        "end_xy": end_xy,
        "windows": windows,
        "failures": failures,
        "cues": cues,
        "recipe": _recipe(),
        "n_latch": len(windows),
        "n_on": sum(1 for w in windows if w["grade"] == "ON"),
        "n_early": sum(1 for w in windows if w["grade"] == "EARLY"),
        "n_late": sum(1 for w in windows if w["grade"] == "LATE"),
        "n_miss": sum(1 for w in windows if w["grade"] == "MISS"),
    }


def _recipe() -> list[str]:
    return [
        "Leave Save RIGHT → Bubble runway y≈395 seat x∈[25,32]",
        "RIGHT+B dash ~21f → RIGHT+B+A spin-glide into right wall",
        "On pose 132: A immediately (0–2f) into wall, then flip+A — do this TWICE",
        "Phase D: x≥300 y≤200  |  SELECT+L2 reload pin  |  R hard reset",
    ]


def format_diagnosis(diag: dict[str, Any], *, take: str = "") -> str:
    lines = [
        f"[GRADE {diag.get('grade')}] {take}".rstrip(),
        f"  min_y={diag.get('min_y')} max_x={diag.get('max_x')} "
        f"phase_d={diag.get('phase_d')} height={diag.get('height_class')} "
        f"rooms={diag.get('rooms')}",
        f"  end={diag.get('end_room')} xy={diag.get('end_xy')} "
        f"latches={diag.get('n_latch')} "
        f"ON={diag.get('n_on')} EARLY={diag.get('n_early')} "
        f"LATE={diag.get('n_late')} MISS={diag.get('n_miss')}",
    ]
    for i, w in enumerate(diag.get("windows") or [], 1):
        off = w.get("frames_off")
        off_s = f"{off:+d}f" if isinstance(off, int) else "n/a"
        lines.append(
            f"  WJ{i} [{w.get('grade')}] latch f{w.get('latch_start')}–"
            f"{w.get('latch_end')} ({w.get('latch_frames')}f) "
            f"xy={w.get('xy_start')}  frames_off={off_s}"
        )
        lines.append(f"       {w.get('note')}")
    for f in diag.get("failures") or []:
        lines.append(f"  FAIL: {f}")
    for c in diag.get("cues") or []:
        lines.append(f"  → {c}")
    return "\n".join(lines)


def _live_grade_event(window: dict[str, Any]) -> str:
    off = window.get("frames_off")
    off_s = f"{off:+d}f" if isinstance(off, int) else "?"
    g = window["grade"]
    if g == "ON":
        tag = "ON TIME"
    elif g == "EARLY":
        tag = "EARLY"
    elif g == "LATE":
        tag = "LATE"
    else:
        tag = "MISS"
    return (
        f"[WJ {tag}] frames_off={off_s}  "
        f"latch f{window['latch_start']}–{window['latch_end']} "
        f"xy={window['xy_start']}  {window['note']}"
    )


# ---------------------------------------------------------------------------
# Pin resolve
# ---------------------------------------------------------------------------


def resolve_pin(source: Path | None = None) -> Path:
    if source is not None:
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"source pin missing: {p}")
        return p
    if DEFAULT_PIN.is_file():
        return DEFAULT_PIN
    for cand in FALLBACK_PINS:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "No bubble-save pin. Expected scratch/bubble_save.state "
        "(from full_start_v1 end 0xB0DD)."
    )


def _trace_row(env: Any, frame: int, action: Any) -> dict[str, Any]:
    st = parse_env_state(env, frame=frame, mode="nav")
    return {
        "frame": frame,
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "room": int(st.room_id),
        "room_hex": f"0x{int(st.room_id):04X}",
        "pose": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
        "buttons": pressed_buttons(action),
        "energy": int(st.health),
        "items": int(st.collected_items),
    }


def _list_takes_dir(out_dir: Path) -> None:
    paths = sorted(out_dir.glob("take*.json"))
    if not paths:
        print(f"  (no takes yet in {out_dir})")
        return
    for p in paths[-8:]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"  {p.name}: unreadable")
            continue
        diag = data.get("diagnosis") or {}
        start = data.get("attempt_start_state") or data.get("start_state")
        print(
            f"  {p.name}: grade={diag.get('grade')} "
            f"min_y={diag.get('min_y')} latches={diag.get('n_latch')} "
            f"phase_d={diag.get('phase_d')}  start={start}"
        )


def buttons_to_snes12(names: list[str] | None, *, strip_menu: bool = True) -> list[int]:
    """Button name list → SNES-12 ints. Strips SELECT/START by default (CP chords)."""
    action = [0] * 12
    for n in names or []:
        idx = SNES_BUTTON_NAME_TO_INDEX.get(str(n).upper())
        if idx is not None:
            action[idx] = 1
    if strip_menu:
        action[2] = 0  # SELECT
        action[3] = 0  # START
    return action


def body_start_index(buttons_stream: list[list[str]]) -> int:
    """Skip leading SELECT-only / idle frames after a CP load chord."""
    move = {"RIGHT", "LEFT", "A", "B", "X", "Y", "UP", "DOWN", "L", "R"}
    for i, b in enumerate(buttons_stream):
        s = {str(x).upper() for x in (b or [])}
        if s & move:
            return i
    return 0


def dump_state(env: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_state_bytes(path, env.em.get_state())


# ---------------------------------------------------------------------------
# Human multi-take
# ---------------------------------------------------------------------------


def cmd_human(args: argparse.Namespace) -> int:
    pin = resolve_pin(Path(args.source) if args.source else None)
    series = args.series or f"bs_{datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) if args.out_dir else TASKS_ROOT / series
    out_dir.mkdir(parents=True, exist_ok=True)

    state_bytes = read_state_bytes(pin)
    assist = UnlimitedResourcesAssist() if not args.no_assist else None

    take_i = 1
    while (out_dir / f"take{take_i:02d}.json").is_file():
        take_i += 1

    print("=" * 60)
    print("BUBBLE SAVE PRACTICE  — leave Save → runway walljump")
    print(f"  pin:    {pin}")
    print(f"  series: {series}  → {out_dir}")
    print(f"  next:   take{take_i:02d}")
    print("  SELECT+R2 = save CP1 mid seat · SELECT+L2 = load CP1")
    print("  R = hard reload pin (re-seeds CP1) · F5 = save take + reload")
    print("  F1 = save take stay · ESC/Q = quit")
    print("  Feedback: EARLY / LATE / ON / MISS + frames_off on each latch")
    print("  Recipe: leave RIGHT → seat x25–32 y395 → dash+spin → double WJ")
    print("=" * 60)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    live: dict[str, Any] = {
        "take": take_i,
        "pose": 0,
        "x": 0,
        "y": 0,
        "room": 0,
        "phase": "boot",
        "last_grade": "",
        "last_wj": "",
        "min_y": 9999,
        "n_latch": 0,
        "latch_open": False,
        "latch_start_frame": -1,
        "latch_start_i": -1,
        "a_in_latch": False,
        "trace": [],
        # Bytes at the start of the *current* attempt (boot pin or last CP load).
        # F5 dumps this so open-loop can re-seat exactly (subpixel-critical).
        "attempt_start_bytes": None,
        "attempt_start_label": "boot",
    }
    task_holder: dict[str, Any] = {"task": None}

    def new_task() -> RecordedTask:
        name = f"{series}_take{live['take']:02d}"
        t = RecordedTask(
            name=name,
            start_state=str(pin.relative_to(INTEGRATION_DIR))
            if pin.is_relative_to(INTEGRATION_DIR)
            else str(pin),
        )
        t.metadata["series"] = series
        t.metadata["take"] = int(live["take"])
        t.metadata["pin"] = str(pin)
        t.metadata["goal"] = "bubble_save_runway_walljump"
        t.metadata["room"] = "0xB0DD → 0xACB3 Phase D"
        return t

    task_holder["task"] = new_task()

    def _capture_attempt_start(e: Any, label: str) -> None:
        live["attempt_start_bytes"] = e.em.get_state()
        live["attempt_start_label"] = label

    def _seed_cp1(session: PlaySession, label: str = "pin") -> None:
        session.save_checkpoint(1)
        print(f"[CP1] seeded ({label}) — SELECT+L2 reloads this seat", flush=True)

    def boot_pin(e: Any, session: PlaySession | None = None) -> None:
        e.em.set_state(state_bytes)
        for _ in range(12):
            step_env(e, idle_action())
            if assist is not None:
                st = parse_env_state(e, mode="nav")
                try:
                    assist.apply(e.data, st)
                except Exception:  # noqa: BLE001
                    try:
                        assist.apply(e, st)
                    except Exception:  # noqa: BLE001
                        pass
        st = parse_env_state(e, mode="nav")
        live.update(
            x=int(st.samus_x),
            y=int(st.samus_y),
            pose=int(st.pose),
            room=int(st.room_id),
            phase="SAVE  unmorph → RIGHT leave door",
            min_y=int(st.samus_y),
            n_latch=0,
            latch_open=False,
            latch_start_frame=-1,
            latch_start_i=-1,
            a_in_latch=False,
            last_wj="",
        )
        live["trace"] = []
        _capture_attempt_start(e, "boot_bubble_save")
        print(
            f"[BOOT take{live['take']:02d}] room=0x{st.room_id:04X} "
            f"xy=({st.samus_x},{st.samus_y}) pose={st.pose} "
            f"items=0x{int(st.collected_items):04X}",
            flush=True,
        )
        if session is not None:
            _seed_cp1(session, "boot pin")

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env
    session_box: dict[str, PlaySession | None] = {"s": None}

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        boot_pin(e, session_box["s"])
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]

    def save_take(*, reload_pin: bool) -> None:
        task: RecordedTask = task_holder["task"]
        spark_trace = list(live["trace"])
        # close open latch window into trace for diagnosis
        diag = diagnose_trace(spark_trace)
        task.metadata["diagnosis"] = diag
        task.metadata["recorded_at"] = datetime.now(timezone.utc).isoformat()
        task.metadata["frame_count"] = len(spark_trace)

        take_stem = f"take{live['take']:02d}"
        take_path = out_dir / f"{take_stem}.json"
        btn_stream = [row.get("buttons") or [] for row in spark_trace]
        frames_snes12 = [buttons_to_snes12(b) for b in btn_stream]
        body0 = body_start_index(btn_stream)

        # Exact attempt-start pin (boot or last SELECT+L2 seat) — required for
        # open-loop. Without this, nearby pure seats desync (subpixel).
        attempt_rel: str | None = None
        start_blob = live.get("attempt_start_bytes")
        if start_blob:
            start_path = out_dir / f"{take_stem}_start.state"
            write_state_bytes(start_path, start_blob)
            attempt_rel = str(start_path.relative_to(GAME_DIR))
            print(
                f"[PIN] attempt start ({live.get('attempt_start_label')}) → {start_path}",
                flush=True,
            )
        try:
            end_path = out_dir / f"{take_stem}_end.state"
            dump_state(env, end_path)
            print(f"[PIN] end → {end_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[PIN] end dump failed: {exc}", flush=True)
            end_path = None

        rooms_seq: list[str] = []
        for row in spark_trace:
            rh = row.get("room_hex") or f"0x{int(row.get('room') or 0):04X}"
            if not rooms_seq or rooms_seq[-1] != rh:
                rooms_seq.append(str(rh))

        payload = {
            "name": task.name,
            # Series boot pin (Save station) — NOT always the open-loop seat.
            "start_state": task.start_state,
            # Actual attempt start after last R / L2 / boot (use this for replay).
            "attempt_start_state": attempt_rel,
            "attempt_start_label": live.get("attempt_start_label"),
            "end_state": (
                str(end_path.relative_to(GAME_DIR)) if end_path is not None else None
            ),
            "frame_count": len(spark_trace),
            "body_start_index": body0,
            "recorded_at": task.metadata["recorded_at"],
            "metadata": {
                k: v for k, v in task.metadata.items() if k != "wj_trace"
            },
            "room_sequence": rooms_seq,
            "trace": spark_trace,
            "diagnosis": diag,
            "buttons_stream": btn_stream,
            "frames": frames_snes12,
            "note": (
                "SELECT+L2 clears the attempt tape — F5 only keeps the last "
                "attempt since boot/load. Door-to-door stitch needs one clean "
                "run without mid-take CP load, or hop-compose of separate pieces."
            ),
        }
        take_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(format_diagnosis(diag, take=take_path.name), flush=True)
        print(f"[SAVED] {take_path}  body_start={body0} frames={len(frames_snes12)}", flush=True)
        if attempt_rel:
            print(
                f"  open-loop:  uv run python snes/super_metroid/scripts/probe/"
                f"bubble_save_practice.py replay {take_path}",
                flush=True,
            )
        else:
            print(
                "  WARN: no attempt_start pin — open-loop will desync "
                "(upgrade: re-record; R2 now also dumps cp1_seat.state)",
                flush=True,
            )
        if rooms_seq and rooms_seq[0] != "0xB0DD":
            print(
                f"  NOTE: tape starts in {rooms_seq[0]} (not Save) — "
                f"not door-to-door; seat→exit hop only.",
                flush=True,
            )
        live["last_grade"] = str(diag.get("grade"))

        live["take"] = int(live["take"]) + 1
        task_holder["task"] = new_task()
        if reload_pin and session_box["s"] is not None:
            boot_pin(env, session_box["s"])
            print(f"[RELOAD] ready for take{live['take']:02d}", flush=True)

    def _close_latch_if_open(trace: list[dict], end_i: int) -> None:
        if not live["latch_open"]:
            return
        start_i = int(live["latch_start_i"])
        if start_i < 0 or start_i > end_i:
            live["latch_open"] = False
            return
        w = _finish_latch_window(trace, start_i, end_i)
        live["n_latch"] = int(live["n_latch"]) + 1
        live["last_wj"] = (
            f"{w['grade']} {w.get('frames_off', '?')}f  "
            f"(#{live['n_latch']})"
        )
        print(_live_grade_event(w), flush=True)
        live["latch_open"] = False
        live["a_in_latch"] = False

    def on_step(obs, reward, done, info) -> None:
        del obs, reward, done, info
        action = session.last_action_post_sanitize
        frame = session.frame_count
        st = parse_env_state(env, frame=frame, mode="nav")
        if assist is not None:
            try:
                assist.apply(env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    assist.apply(env, st)
                except Exception:  # noqa: BLE001
                    pass
        row = _trace_row(env, frame - 1 if frame > 0 else 0, action)
        task: RecordedTask = task_holder["task"]
        task.append_frame(action, trace_row=row)
        live["trace"].append(row)
        i = len(live["trace"]) - 1

        live["x"] = row["x"]
        live["y"] = row["y"]
        live["pose"] = row["pose"]
        live["room"] = row["room"]
        live["min_y"] = min(int(live["min_y"]), int(row["y"]))

        latched = int(row["pose"]) == POSE_WALL_LATCH
        has_a = "A" in _btn_set(row)

        if latched and not live["latch_open"]:
            live["latch_open"] = True
            live["latch_start_frame"] = int(row["frame"])
            live["latch_start_i"] = i
            live["a_in_latch"] = has_a
            print(
                f"[LATCH] f{row['frame']} pose=132 xy=({row['x']},{row['y']}) "
                f"— press A NOW (ideal 0–{IDEAL_MAX}f)",
                flush=True,
            )
        elif latched and live["latch_open"]:
            if has_a:
                live["a_in_latch"] = True
        elif not latched and live["latch_open"]:
            _close_latch_if_open(live["trace"], i - 1)

        # Live phase string
        rid = int(row["room"])
        if rid == ROOM_SAVE:
            live["phase"] = "SAVE  unmorph → walk RIGHT → door"
        elif rid == ROOM_BUBBLE:
            if latched:
                live["phase"] = "!!! LATCH 132 — press A NOW !!!"
            elif int(row["y"]) <= PHASE_D_Y and int(row["x"]) >= PHASE_D_X:
                live["phase"] = "PHASE D  Super door top-right"
            elif int(row["y"]) <= HEIGHT_CLASS_Y:
                live["phase"] = "height class — keep climbing / 2nd WJ"
            elif int(row["x"]) <= 90 and 380 <= int(row["y"]) <= 430:
                live["phase"] = "RUNWAY seat — RIGHT+B dash then spin"
            else:
                live["phase"] = "BUBBLE  find save-runway y≈395 left"
        else:
            live["phase"] = f"room 0x{rid:04X}"

    def on_hud(_info) -> list[str]:
        grade = live.get("last_grade") or ""
        gbit = f" last={grade}" if grade else ""
        wj = live.get("last_wj") or ""
        wjbit = f"  wj={wj}" if wj else ""
        return [
            f"[BUBBLE-SAVE] take{live['take']:02d}{gbit}  "
            f"L2=load R2=save R=reset F5=take+reload",
            f"0x{int(live['room']):04X} xy=({live['x']},{live['y']}) "
            f"p={live['pose']} min_y={live['min_y']} latches={live['n_latch']}"
            f"{wjbit}",
            f"phase: {live['phase']}",
        ]

    def on_key_down(key: int) -> bool:
        try:
            import pygame as pg
        except ImportError:
            return False
        if key == pg.K_F5:
            save_take(reload_pin=True)
            return True
        if key == pg.K_F1:
            save_take(reload_pin=False)
            return True
        if key == pg.K_r:
            # Hard reset pin mid-take (discard current attempt frames in live)
            if session_box["s"] is not None:
                boot_pin(env, session_box["s"])
                task_holder["task"] = new_task()
                print("[RESET] pin reloaded (take counter unchanged)", flush=True)
            return True
        return False

    def on_trigger_save(slot: int) -> None:
        frame = session.save_checkpoint(slot)
        st = parse_env_state(env, mode="nav")
        # Disk pin so a later GREEN attempt can open-loop from this exact seat.
        seat_path = out_dir / "cp1_seat.state"
        try:
            dump_state(env, seat_path)
            seat_note = f" + disk {seat_path.name}"
        except Exception as exc:  # noqa: BLE001
            seat_note = f" (disk dump fail: {exc})"
        print(
            f"[CP SAVE {slot}] room=0x{int(st.room_id):04X} "
            f"xy=({int(st.samus_x)},{int(st.samus_y)}) f{frame} "
            f"— SELECT+L2 reloads here{seat_note}",
            flush=True,
        )

    def on_trigger_load(slot: int) -> None:
        frame = session.load_checkpoint(slot)
        if frame is None:
            print(f"[CP LOAD {slot}] empty — use R to hard-reset pin", flush=True)
            return
        st = parse_env_state(env, mode="nav")
        live.update(
            x=int(st.samus_x),
            y=int(st.samus_y),
            pose=int(st.pose),
            room=int(st.room_id),
            min_y=int(st.samus_y),
            n_latch=0,
            latch_open=False,
            last_wj="",
        )
        live["trace"] = []
        _capture_attempt_start(env, f"cp{slot}_load")
        task_holder["task"] = new_task()
        print(
            f"[CP LOAD {slot}] room=0x{int(st.room_id):04X} "
            f"xy=({int(st.samus_x)},{int(st.samus_y)}) f{frame}  "
            f"— attempt cleared (only this run is kept on F5)",
            flush=True,
        )

    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title=f"Bubble save practice — {series}",
            action_size=12,
            base_fps=60,
            initial_speed=args.speed,
            headless=False,
        )
        session_box["s"] = session
        session.on_hud = on_hud
        session.on_step = on_step
        session.on_key_down = on_key_down
        session.on_trigger_save = on_trigger_save
        session.on_trigger_load = on_trigger_load
        session.quiet_checkpoints = True  # we print our own CP lines
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    n = live["take"] - 1
    print(f"session end — takes under {out_dir} (last saved index ~{n})")
    _list_takes_dir(out_dir)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    path = Path(args.take)
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = data.get("trace") or data.get("wj_trace") or []
    diag = diagnose_trace(trace)
    print(format_diagnosis(diag, take=path.name))
    return 0 if diag.get("ok") else 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Open-loop replay a practice take from its attempt_start pin."""
    import numpy as np

    take_path = Path(args.take)
    data = json.loads(take_path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not frames:
        stream = data.get("buttons_stream") or []
        frames = [buttons_to_snes12(b) for b in stream]
    if not frames:
        print("ERROR: take has no frames / buttons_stream", file=sys.stderr)
        return 1

    body0 = int(data.get("body_start_index") or body_start_index(
        data.get("buttons_stream") or []
    ))
    if args.from_frame is not None:
        body0 = int(args.from_frame)

    start_rel = data.get("attempt_start_state")
    if args.source:
        start_path = Path(args.source)
    elif start_rel:
        start_path = GAME_DIR / start_rel
        if not start_path.is_file():
            start_path = take_path.parent / Path(start_rel).name
    else:
        # Legacy takes (pre pin-dump): cannot dual-green reliably.
        print(
            "ERROR: take has no attempt_start_state pin.\n"
            "  take01 from the first session only kept buttons after CP load;\n"
            "  the exact (24,395) seat was memory-only and is gone.\n"
            "  Re-record: F5 after GREEN now dumps takeNN_start.state.\n"
            "  Door-to-door: ./play bubble-save full_start_v1 (no mid L2).",
            file=sys.stderr,
        )
        return 2

    if not start_path.is_file():
        print(f"ERROR: start pin missing: {start_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"REPLAY  {take_path.name}")
    print(f"  start: {start_path}")
    print(f"  body:  frames[{body0}:{len(frames)}]  ({len(frames) - body0}f)")
    print("=" * 60)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(read_state_bytes(start_path))
        for _ in range(int(args.settle)):
            env.step(idle_action())
        st0 = parse_env_state(env, mode="nav")
        print(
            f"[BOOT] room=0x{st0.room_id:04X} xy=({st0.samus_x},{st0.samus_y}) "
            f"pose={st0.pose}",
            flush=True,
        )
        for i in range(body0, len(frames)):
            env.step(np.array(frames[i], dtype=np.int8))
        st = parse_env_state(env, mode="nav")
        # Prefer Bat leave or Phase D band
        phase_d = (
            int(st.room_id) == ROOM_BUBBLE
            and int(st.samus_x) >= PHASE_D_X
            and int(st.samus_y) <= PHASE_D_Y
        )
        bat = int(st.room_id) == 0xB07A
        mark = "GREEN" if bat or phase_d else "RED"
        print(
            f"[{mark}] end room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"pose={st.pose}  bat={bat} phase_d={phase_d}",
            flush=True,
        )
        if args.dual:
            env.em.set_state(read_state_bytes(start_path))
            for _ in range(int(args.settle)):
                env.step(idle_action())
            for i in range(body0, len(frames)):
                env.step(np.array(frames[i], dtype=np.int8))
            st2 = parse_env_state(env, mode="nav")
            bat2 = int(st2.room_id) == 0xB07A
            phase_d2 = (
                int(st2.room_id) == ROOM_BUBBLE
                and int(st2.samus_x) >= PHASE_D_X
                and int(st2.samus_y) <= PHASE_D_Y
            )
            mark2 = "GREEN" if bat2 or phase_d2 else "RED"
            print(
                f"[{mark2}] dual end room=0x{st2.room_id:04X} "
                f"xy=({st2.samus_x},{st2.samus_y}) bat={bat2} phase_d={phase_d2}",
                flush=True,
            )
            return 0 if (bat or phase_d) and (bat2 or phase_d2) else 1
        return 0 if bat or phase_d else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p_h = sub.add_parser("human", help="Multi-take practice gym (default)")
    p_h.add_argument("--source", default=None, help="Override pin path")
    p_h.add_argument("--series", default=None)
    p_h.add_argument("--out-dir", default=None)
    p_h.add_argument("--scale", type=int, default=3)
    p_h.add_argument("--speed", type=float, default=1.0)
    p_h.add_argument("--no-assist", action="store_true")

    p_d = sub.add_parser("diagnose", help="Re-grade a saved take JSON")
    p_d.add_argument("take", type=str)

    p_r = sub.add_parser("replay", help="Open-loop from attempt_start pin")
    p_r.add_argument("take", type=str, help="takeNN.json path")
    p_r.add_argument("--source", default=None, help="Override start pin")
    p_r.add_argument("--from-frame", type=int, default=None)
    p_r.add_argument("--settle", type=int, default=0)
    p_r.add_argument("--dual", action="store_true", help="Run twice")

    # Default to human when no subcommand
    parser.add_argument("--source", default=None)
    parser.add_argument("--series", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-assist", action="store_true")

    args = parser.parse_args()
    if args.cmd == "diagnose":
        return cmd_diagnose(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    # human (explicit or default)
    return cmd_human(args)


if __name__ == "__main__":
    raise SystemExit(main())
