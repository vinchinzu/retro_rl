#!/usr/bin/env python3
"""Shinespark practice gym — Landing Site left-of-ship + attempt diagnose.

Safe dual-track pin (same Speed loadout as Moat / Kihunter work):

  ``scratch/landing_site_speed_practice.state``  room ``0x91F8`` ~(899,1163)
  built from pre-Moat Kihunter by walking left a few rooms (not escape finish).

Practice goal (horizontal RIGHT spark):
  1. RIGHT+B until echoes=4 (blue suit / speed boost) — ~90f flat runway
  2. **While still holding RIGHT+B**, press DOWN — ``$0A68`` arms ≈179
     (Do NOT release direction first — idle/B-only dumps echoes in 1f.)
  3. Brief UP (or neutral) ~4f so pose leaves crouch windup
  4. RIGHT+A to activate (horizontal) — spark poses 199–202

Harness (do not swap with VOD A/B labels):
  **B** = dash / charge · **A** = jump / shine activate · **DOWN** = store

```bash
# STORE DRILL (recommended if you cannot arm $0A68 yet)
# Bot charges for you and holds RIGHT+B; you only press DOWN when HUD yells.
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill

# Multi-take human gym (F5 = save take + diagnose + reload pin)
uv run python snes/super_metroid/scripts/probe/shine_practice.py human

# Named series under tasks/shine_practice/<series>/
uv run python snes/super_metroid/scripts/probe/shine_practice.py human \\
  --series ls_edge_v1

# Bot demo of the full green recipe
uv run python snes/super_metroid/scripts/probe/shine_practice.py demo

# Re-diagnose a saved take JSON
uv run python snes/super_metroid/scripts/probe/shine_practice.py diagnose \\
  snes/super_metroid/tasks/shine_practice/ls_edge_v1/take03.json
```

Keys (human): **F5** save take + diagnose + reload · **ESC/Q** quit ·
``[`` ``]`` speed · TAB turbo · **F1** save without reload
Keys (drill): **DOWN** when prompted · **R** retry · **ESC** quit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from retro_harness.task_recording import RecordedTask, pressed_buttons  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.skills import shinespark as spark  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_PIN = SCRATCH / "landing_site_speed_practice.state"
PRE_MOAT = SCRATCH / "post_kihunter_pre_moat_spark.state"
ROOM_LANDING = 0x91F8
TASKS_ROOT = GAME_DIR / "tasks" / "shine_practice"
DEBUG_DIR = GAME_DIR / "debug" / "shine_practice"

# Soft success: saw a real spark pose while timer was armed / draining.
SPARK_OK_POSES = spark.SPARK_POSES


# ---------------------------------------------------------------------------
# Diagnosis (pure function — unit-testable from a frame trace)
# ---------------------------------------------------------------------------


def _charge_windows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find continuous spans where speed_echoes >= 4; note DOWN inside/after."""
    windows: list[dict[str, Any]] = []
    start_i: int | None = None
    for i, row in enumerate(trace):
        e = int(row.get("speed_echoes") or 0)
        if e >= spark.ECHOES_FULL and start_i is None:
            start_i = i
        elif e < spark.ECHOES_FULL and start_i is not None:
            windows.append(_finish_charge_window(trace, start_i, i - 1))
            start_i = None
    if start_i is not None:
        windows.append(_finish_charge_window(trace, start_i, len(trace) - 1))
    return windows


def _finish_charge_window(
    trace: list[dict[str, Any]], start_i: int, end_i: int
) -> dict[str, Any]:
    stretch = trace[start_i : end_i + 1]
    start = stretch[0]
    end = stretch[-1]
    downs_in = [
        r for r in stretch if "DOWN" in [str(b).upper() for b in (r.get("buttons") or [])]
    ]
    # first DOWN after window ends
    lag: int | None = None
    late_down: dict[str, Any] | None = None
    for r in trace[end_i + 1 :]:
        btns = [str(b).upper() for b in (r.get("buttons") or [])]
        if "DOWN" in btns:
            lag = int(r.get("frame") or 0) - int(end.get("frame") or 0)
            late_down = r
            break
    start_btns = [str(b).upper() for b in (start.get("buttons") or [])]
    end_btns = [str(b).upper() for b in (end.get("buttons") or [])]
    next_btns: list[str] = []
    if end_i + 1 < len(trace):
        next_btns = [str(b).upper() for b in (trace[end_i + 1].get("buttons") or [])]
    kept_dash = all(
        "B" in [str(b).upper() for b in (r.get("buttons") or [])]
        for r in stretch[-min(5, len(stretch)) :]
    )
    # How the boost died (measured): idle or B-without-direction dumps e=4→0 in 1f.
    kill = "unknown"
    if "DOWN" in next_btns:
        kill = "down_next"  # rare if window already ended
    elif not next_btns:
        kill = "released_all"  # idle — kills echoes instantly
    elif "B" in next_btns and "RIGHT" not in next_btns and "LEFT" not in next_btns:
        kill = "released_direction_kept_B"  # classic fail: B alone
    elif "RIGHT" in next_btns and "B" not in next_btns:
        kill = "released_B_kept_RIGHT"  # e can survive; OK-ish
    elif "LEFT" in next_btns:
        kill = "turned_around"
    return {
        "start_frame": int(start.get("frame") or 0),
        "end_frame": int(end.get("frame") or 0),
        "frames": len(stretch),
        "start_xy": (int(start.get("x") or 0), int(start.get("y") or 0)),
        "start_pose": int(start.get("pose") or 0),
        "start_buttons": start_btns,
        "end_buttons": end_btns,
        "next_buttons": next_btns,
        "boost_kill": kill,
        "charged_left": "LEFT" in start_btns and "RIGHT" not in start_btns,
        "charged_right": "RIGHT" in start_btns,
        "downs_while_full": len(downs_in),
        "kept_holding_B_at_end": kept_dash,
        "down_lag_after_charge_died": lag,
        "late_down_pose": int(late_down.get("pose") or 0) if late_down else None,
        "late_down_echoes": int(late_down.get("speed_echoes") or 0) if late_down else None,
    }


def diagnose_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a practice attempt from per-frame WRAM/nav rows.

    Expected keys per row (missing treated as 0 / None):
      frame, x, y, pose, vx, buttons (list[str]),
      speed_echoes, spark_timer, speed_flag
    """
    if not trace:
        return {
            "ok": False,
            "grade": "EMPTY",
            "failures": ["no frames recorded"],
            "cues": ["Press buttons — nothing was recorded."],
            "peaks": {},
            "milestones": {},
        }

    peak_echoes = 0
    peak_timer = 0
    first_echo4: int | None = None
    first_store: int | None = None
    first_spark_pose: int | None = None
    first_spark_timer_while_pose: int | None = None
    store_while_spin = False
    down_before_echo4 = False
    activate_without_store = False
    activate_from_crouch_walk = False  # RIGHT+A while timer>0 but no spark pose soon
    pre_stand_up_seen = False
    spark_travel_frames = 0
    max_x = int(trace[0].get("x") or 0)
    min_y = int(trace[0].get("y") or 0)
    armed_then_zero_without_spark = False
    saw_timer_gt0 = False
    saw_spark_after_arm = False
    down_frames_total = 0
    down_while_echoes_zero = 0
    crouch_no_timer = 0  # pose 39/53 while timer=0 (fake crouch)

    for row in trace:
        f = int(row.get("frame") or 0)
        echoes = int(row.get("speed_echoes") or 0)
        timer = int(row.get("spark_timer") or 0)
        pose = int(row.get("pose") or 0)
        btns = [str(b).upper() for b in (row.get("buttons") or [])]
        x = int(row.get("x") or 0)
        y = int(row.get("y") or 0)
        if x < 60000:
            max_x = max(max_x, x)
        if y > 0:
            min_y = min(min_y, y)

        peak_echoes = max(peak_echoes, echoes)
        peak_timer = max(peak_timer, timer)

        if first_echo4 is None and echoes >= spark.ECHOES_FULL:
            first_echo4 = f
        if first_store is None and timer > 0:
            first_store = f
            saw_timer_gt0 = True
        if timer > 0:
            saw_timer_gt0 = True

        if "DOWN" in btns:
            down_frames_total += 1
            if echoes == 0 and timer == 0:
                down_while_echoes_zero += 1

        if pose in (39, 40, 53, 54) and timer == 0 and echoes == 0:
            crouch_no_timer += 1

        if pose in spark.STORE_WIPE_POSES and "DOWN" in btns:
            store_while_spin = True

        if "DOWN" in btns and first_echo4 is None and echoes < spark.ECHOES_FULL:
            down_before_echo4 = True

        if first_spark_pose is None and spark.is_spark_pose(pose):
            first_spark_pose = f
            first_spark_timer_while_pose = timer
            if saw_timer_gt0:
                saw_spark_after_arm = True

        if spark.is_spark_pose(pose):
            spark_travel_frames += 1

        if "UP" in btns and saw_timer_gt0 and first_spark_pose is None:
            pre_stand_up_seen = True

        # A while not armed
        if "A" in btns and not saw_timer_gt0 and ("RIGHT" in btns or "LEFT" in btns):
            # only flag if they thought they were sparking (had some echoes)
            if peak_echoes >= 2:
                activate_without_store = True

    charge_wins = _charge_windows(trace)
    # Pattern: full charge windows with zero DOWN inside, DOWN a few frames later
    missed_store_windows = [
        w
        for w in charge_wins
        if w["downs_while_full"] == 0 and w.get("down_lag_after_charge_died") is not None
    ]
    kept_running_through_charge = sum(
        1 for w in charge_wins if w["downs_while_full"] == 0 and w.get("kept_holding_B_at_end")
    )
    left_charges = sum(1 for w in charge_wins if w.get("charged_left"))
    right_charges = sum(1 for w in charge_wins if w.get("charged_right"))
    late_store = (
        first_store is None
        and peak_echoes >= spark.ECHOES_FULL
        and len(missed_store_windows) > 0
        and down_frames_total > 0
    )
    # median lag for messaging
    lags = [
        int(w["down_lag_after_charge_died"])
        for w in missed_store_windows
        if w.get("down_lag_after_charge_died") is not None
    ]
    typical_lag = sorted(lags)[len(lags) // 2] if lags else None

    # After arm, if A+direction held for ≥8f while timer>0 and never spark pose
    if saw_timer_gt0 and first_spark_pose is None:
        armed_window = [r for r in trace if int(r.get("spark_timer") or 0) > 0]
        a_hold = 0
        for r in armed_window:
            btns = [str(b).upper() for b in (r.get("buttons") or [])]
            if "A" in btns and ("RIGHT" in btns or "LEFT" in btns or "UP" in btns):
                a_hold += 1
            pose = int(r.get("pose") or 0)
            if pose in (39, 40, 53, 54) and "A" in btns and "RIGHT" in btns:
                activate_from_crouch_walk = True
        if a_hold >= 8:
            activate_from_crouch_walk = True
        # timer drained fully without spark
        if armed_window and int(armed_window[-1].get("spark_timer") or 0) == 0:
            armed_then_zero_without_spark = True
        elif peak_timer > 0 and int(trace[-1].get("spark_timer") or 0) == 0:
            armed_then_zero_without_spark = True

    ok = first_spark_pose is not None and spark_travel_frames >= 3
    failures: list[str] = []
    cues: list[str] = []

    if peak_echoes < spark.ECHOES_FULL:
        failures.append(f"charge incomplete (peak echoes={peak_echoes}, need ≥4)")
        cues.append("Hold RIGHT+B on flat ground ~90f until blue echoes fill (echoes=4).")
        cues.append("Do not mash A while charging — 1f A taps kill echo build.")
    elif first_store is None:
        failures.append("never crouch-stored ($0A68 stayed 0)")
        if late_store:
            failures.append(
                f"late crouch: charged {len(charge_wins)}× but DOWN never during echoes=4 "
                f"({len(missed_store_windows)} windows; typical DOWN ~{typical_lag}f AFTER charge died)"
            )
            kill_counts: dict[str, int] = {}
            for w in missed_store_windows:
                k = str(w.get("boost_kill") or "unknown")
                kill_counts[k] = kill_counts.get(k, 0) + 1
            top_kill = max(kill_counts, key=kill_counts.get) if kill_counts else "unknown"
            if top_kill == "released_direction_kept_B":
                failures.append(
                    "boost killed by releasing LEFT/RIGHT while keeping B "
                    f"({kill_counts.get(top_kill)} windows)"
                )
                cues.append(
                    "CRITICAL (measured): B alone or idle dumps echoes 4→0 in ONE frame. "
                    "You let go of RIGHT, then crouched. Too late — charge is already dead."
                )
                cues.append(
                    "CORRECT: while still holding RIGHT+B and HUD shows echoes=4, "
                    "ALSO press DOWN (DOWN+RIGHT+B is fine). Do NOT release direction first."
                )
            elif top_kill == "released_all":
                failures.append(
                    f"boost killed by releasing all buttons ({kill_counts.get(top_kill)} windows)"
                )
                cues.append(
                    "CRITICAL: idle for even 1f after full charge wipes echoes. "
                    "Press DOWN on the same frames you are still dashing RIGHT+B."
                )
            else:
                cues.append(
                    "CRITICAL: press DOWN while echoes are still 4. "
                    f"Boost died via '{top_kill}' before your crouch."
                )
                cues.append(
                    "Do NOT: release stick/B, wait, then crouch. "
                    "DO: DOWN while still running RIGHT+B (blue)."
                )
            if crouch_no_timer > 10:
                cues.append(
                    f"You crouched {crouch_no_timer}f with timer=0 (pose 39/53) — ordinary "
                    "crouch, not a shine-store. Store only arms from live speed boost."
                )
            cues.append(
                "Drill: uv run python snes/super_metroid/scripts/probe/shine_practice.py drill"
            )
        else:
            cues.append("At full charge (pose 9), press DOWN once — timer should jump to ~179.")
            cues.append("Do not DOWN while spinning (poses 25/166 wipe echoes).")
        if left_charges > right_charges and left_charges > 0:
            cues.append(
                f"You charged LEFT more often ({left_charges}× left vs {right_charges}× right). "
                "Pin runway faces RIGHT (toward open beach / ship). Prefer RIGHT+B only."
            )
        if down_while_echoes_zero > 20 and not late_store:
            cues.append(
                f"{down_while_echoes_zero}f of DOWN with echoes=0 — crouching cold does nothing."
            )
    elif first_spark_pose is None:
        failures.append("stored but never entered spark pose")
        if store_while_spin:
            failures.append("DOWN pressed during spin (store wipe risk)")
            cues.append("Land / stand (pose 9) before DOWN.")
        if activate_from_crouch_walk:
            failures.append("RIGHT+A from crouch walked instead of sparking")
            cues.append(
                "After store: release to UP or neutral ~4f (crystal flash), "
                "THEN RIGHT+A. Immediate crouch RIGHT+A only walks."
            )
        elif not pre_stand_up_seen:
            cues.append("Try UP ×4 after store before RIGHT+A (pre-stand).")
        if armed_then_zero_without_spark:
            cues.append("Store timer drained to 0 — you waited too long (~170f max).")
        if activate_without_store:
            cues.append("You pressed A before arming — store first (DOWN).")
        if not cues:
            cues.append("Recipe: RIGHT+B → echoes4 → DOWN → UP×4 → RIGHT+A hold.")
    elif spark_travel_frames < 3:
        failures.append(f"spark pose flickered only ({spark_travel_frames}f)")
        cues.append("Hold RIGHT+A a bit longer through the crystal flash.")

    if down_before_echo4 and peak_echoes < spark.ECHOES_FULL:
        cues.append("You pressed DOWN before full charge — finish the runway first.")

    if ok:
        grade = "GREEN"
        cues = ["Spark pose seen — good. Repeat for consistency, then try diagonal UP+RIGHT."]
    elif peak_echoes >= spark.ECHOES_FULL and first_store is not None:
        grade = "YELLOW"  # charge+store ok, activate miss
    elif peak_echoes >= spark.ECHOES_FULL:
        grade = "ORANGE"  # charge only
    else:
        grade = "RED"

    return {
        "ok": ok,
        "grade": grade,
        "failures": failures,
        "cues": cues,
        "peaks": {
            "echoes": peak_echoes,
            "spark_timer": peak_timer,
            "max_x": max_x,
            "min_y": min_y,
            "spark_travel_frames": spark_travel_frames,
            "charge_windows": len(charge_wins),
            "missed_store_windows": len(missed_store_windows),
            "down_frames": down_frames_total,
            "down_while_echoes_zero": down_while_echoes_zero,
            "crouch_no_timer_frames": crouch_no_timer,
            "typical_down_lag_after_charge": typical_lag,
            "left_charges": left_charges,
            "right_charges": right_charges,
        },
        "milestones": {
            "first_echo4_frame": first_echo4,
            "first_store_frame": first_store,
            "first_spark_pose_frame": first_spark_pose,
            "spark_timer_at_first_pose": first_spark_timer_while_pose,
            "pre_stand_up_seen": pre_stand_up_seen,
            "store_while_spin": store_while_spin,
            "activate_from_crouch_walk": activate_from_crouch_walk,
            "activate_without_store": activate_without_store,
            "late_store_after_charge_died": late_store,
            "kept_running_through_charge": kept_running_through_charge,
        },
        "charge_windows": charge_wins[:12],  # compact for JSON
        "recipe": [
            "RIGHT+B until echoes=4 (~90f, pose 9)",
            "While still holding RIGHT (+B ok), press DOWN → $0A68≈179",
            "UP ×4 (or neutral ×5) — do not skip",
            "RIGHT+A hold → pose 199/201 horizontal spark",
        ],
        "harness": {"dash": "B", "activate": "A", "store": "DOWN"},
    }


def format_diagnosis(diag: dict[str, Any], *, take: str = "") -> str:
    lines = [
        f"=== DIAGNOSE {take}  grade={diag.get('grade')}  "
        f"{'OK' if diag.get('ok') else 'FAIL'} ===",
    ]
    peaks = diag.get("peaks") or {}
    ms = diag.get("milestones") or {}
    lines.append(
        f"  peaks: echoes={peaks.get('echoes')} timer={peaks.get('spark_timer')} "
        f"spark_travel={peaks.get('spark_travel_frames')}f "
        f"max_x={peaks.get('max_x')} min_y={peaks.get('min_y')}"
    )
    lines.append(
        f"  charge_windows={peaks.get('charge_windows')} "
        f"missed_store={peaks.get('missed_store_windows')} "
        f"DOWN_total={peaks.get('down_frames')} "
        f"DOWN@echo0={peaks.get('down_while_echoes_zero')} "
        f"lag≈{peaks.get('typical_down_lag_after_charge')}f"
    )
    lines.append(
        f"  milestones: echo4@{ms.get('first_echo4_frame')} "
        f"store@{ms.get('first_store_frame')} "
        f"spark_pose@{ms.get('first_spark_pose_frame')} "
        f"pre_UP={ms.get('pre_stand_up_seen')} "
        f"late_store={ms.get('late_store_after_charge_died')}"
    )
    wins = diag.get("charge_windows") or []
    if wins and not diag.get("ok"):
        # show first 3 windows briefly
        for w in wins[:3]:
            lines.append(
                f"  window f={w.get('start_frame')}-{w.get('end_frame')} "
                f"({w.get('frames')}f) dir="
                f"{'L' if w.get('charged_left') else 'R' if w.get('charged_right') else '?'} "
                f"DOWN_in={w.get('downs_while_full')} "
                f"DOWN_after=+{w.get('down_lag_after_charge_died')}f"
            )
        if len(wins) > 3:
            lines.append(f"  … +{len(wins) - 3} more charge windows")
    for f in diag.get("failures") or []:
        lines.append(f"  FAIL: {f}")
    for c in diag.get("cues") or []:
        lines.append(f"  → {c}")
    if not diag.get("ok"):
        lines.append("  recipe: " + " · ".join(diag.get("recipe") or []))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pin bootstrap (delegates to landing_shine_practice)
# ---------------------------------------------------------------------------


def ensure_landing_pin(source: Path | None = None) -> Path:
    if source is not None and source.is_file():
        return source
    if DEFAULT_PIN.is_file():
        # quick clean check
        env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        try:
            env.reset()
            env.em.set_state(read_state_bytes(DEFAULT_PIN))
            for _ in range(12):
                env.step(idle_action())
            st = parse_env_state(env, mode="nav")
            if (
                st.room_id == ROOM_LANDING
                and (st.equipped_items & 0x2000)
                and st.equipped_items != 0xF32F
            ):
                return DEFAULT_PIN
        finally:
            env.close()
    # rebuild via landing_shine_practice.bootstrap (same directory)
    import importlib.util

    lsp_path = Path(__file__).resolve().parent / "landing_shine_practice.py"
    spec = importlib.util.spec_from_file_location("landing_shine_practice", lsp_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {lsp_path}")
    lsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lsp)

    print("bootstrapping Landing Site practice pin from pre-Moat Kihunter …", flush=True)
    summary = lsp.bootstrap_practice_pin(DEFAULT_PIN, source=PRE_MOAT)
    print(
        f"  room={summary.get('room_hex')} xy=({summary.get('x')},{summary.get('y')}) "
        f"items={summary.get('equipped_items_hex')}",
        flush=True,
    )
    return DEFAULT_PIN


# ---------------------------------------------------------------------------
# Trace row helper
# ---------------------------------------------------------------------------


def _trace_row(env: Any, frame: int, action: Any) -> dict[str, Any]:
    st = parse_env_state(env, frame=frame, mode="nav")
    w = spark.read_spark_wram(env)
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
        "speed_echoes": w["speed_echoes"],
        "spark_timer": w["spark_timer"],
        "speed_flag": w["speed_flag"],
        "speed_counter_word": w["speed_counter_word"],
        "speed_boosting": bool(st.speed_boosting),
        "shinesparking": bool(st.shinesparking),
    }


def _live_phase(echoes: int, timer: int, pose: int) -> str:
    if spark.is_spark_pose(pose) or (timer > 0 and pose in (199, 200, 201, 202, 203, 204)):
        return "SPARK"
    if timer > 0:
        if pose in (39, 40, 53, 54):
            return "STORED (need UP then A)"
        return "ARMED"
    if echoes >= spark.ECHOES_FULL:
        return "!!! e=4  PRESS DOWN NOW (keep RIGHT held) !!!"
    if echoes >= 1:
        return f"CHARGING echoes={echoes}/4  (keep RIGHT+B)"
    return "RUNWAY  hold RIGHT+B"


# ---------------------------------------------------------------------------
# human multi-take
# ---------------------------------------------------------------------------


def cmd_human(args: argparse.Namespace) -> int:
    pin = ensure_landing_pin(Path(args.source) if args.source else None)
    series = args.series or f"ls_{datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) if args.out_dir else TASKS_ROOT / series
    out_dir.mkdir(parents=True, exist_ok=True)

    state_bytes = read_state_bytes(pin)
    assist = UnlimitedResourcesAssist() if not args.no_assist else None

    # find next take index
    take_i = 1
    while (out_dir / f"take{take_i:02d}.json").is_file():
        take_i += 1

    print("=" * 60)
    print("SHINE PRACTICE  Landing Site left-of-ship runway")
    print(f"  pin:    {pin}")
    print(f"  series: {series}  → {out_dir}")
    print(f"  next:   take{take_i:02d}")
    print("  F5 = save + diagnose + reload pin")
    print("  F1 = save + diagnose (stay in place)")
    print("  ESC/Q = quit")
    print("  Recipe: RIGHT+B → e=4 → DOWN while still holding RIGHT → UP×4 → RIGHT+A")
    print("  TRAP: releasing RIGHT (B alone / idle) kills charge in 1f — then crouch fails")
    print("  Harness: B=dash  A=activate  DOWN=store")
    print("  Stuck on store?  use:  shine_practice.py drill")
    print("=" * 60)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    live: dict[str, Any] = {
        "take": take_i,
        "echoes": 0,
        "timer": 0,
        "pose": 0,
        "x": 0,
        "y": 0,
        "phase": "boot",
        "last_grade": "",
    }
    task_holder: dict[str, Any] = {"task": None}
    quit_flag = {"done": False}

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
        t.metadata["goal"] = "horizontal_right_shinespark"
        t.metadata["room"] = "0x91F8 Landing Site"
        return t

    task_holder["task"] = new_task()

    def boot_pin(e: Any) -> None:
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
        live["x"] = st.samus_x
        live["y"] = st.samus_y
        live["pose"] = st.pose
        live["echoes"] = 0
        live["timer"] = 0
        live["phase"] = "RUNWAY  hold RIGHT+B"
        print(
            f"[BOOT take{live['take']:02d}] room=0x{st.room_id:04X} "
            f"xy=({st.samus_x},{st.samus_y}) pose={st.pose}",
            flush=True,
        )

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        boot_pin(e)
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]

    def save_take(*, reload_pin: bool) -> None:
        task: RecordedTask = task_holder["task"]
        trace = list(task.trace) if hasattr(task, "trace") else []
        # RecordedTask may store via frames — rebuild from metadata if needed
        if not trace and hasattr(task, "frames"):
            # fall back: diagnosis needs our rows; we append into task.metadata
            trace = list(task.metadata.get("spark_trace") or [])

        spark_trace = list(task.metadata.get("spark_trace") or [])
        diag = diagnose_trace(spark_trace)
        task.metadata["diagnosis"] = diag
        task.metadata["recorded_at"] = datetime.now(timezone.utc).isoformat()
        task.metadata["frame_count"] = len(spark_trace)

        take_path = out_dir / f"take{live['take']:02d}.json"
        # Write rich JSON (task + spark_trace + diagnosis)
        payload = {
            "name": task.name,
            "start_state": task.start_state,
            "frame_count": len(spark_trace),
            "recorded_at": task.metadata["recorded_at"],
            "metadata": {
                k: v
                for k, v in task.metadata.items()
                if k != "spark_trace"
            },
            "trace": spark_trace,
            "diagnosis": diag,
        }
        # also store compact button stream for bot imitation
        payload["buttons_stream"] = [row.get("buttons") or [] for row in spark_trace]
        take_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(format_diagnosis(diag, take=take_path.name), flush=True)
        print(f"[SAVED] {take_path}", flush=True)
        live["last_grade"] = str(diag.get("grade"))

        live["take"] = int(live["take"]) + 1
        task_holder["task"] = new_task()
        if reload_pin:
            boot_pin(env)
            print(f"[RELOAD] ready for take{live['take']:02d}", flush=True)

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
        spark_trace = task.metadata.setdefault("spark_trace", [])
        spark_trace.append(row)
        live["x"] = row["x"]
        live["y"] = row["y"]
        live["pose"] = row["pose"]
        live["echoes"] = row["speed_echoes"]
        live["timer"] = row["spark_timer"]
        live["phase"] = _live_phase(row["speed_echoes"], row["spark_timer"], row["pose"])

    def on_hud(_info) -> list[str]:
        grade = live.get("last_grade") or ""
        gbit = f"  last={grade}" if grade else ""
        e = int(live.get("echoes") or 0)
        t = int(live.get("timer") or 0)
        if e >= 4 and t == 0:
            tip = ">>> KEEP RIGHT — press DOWN now (do not release stick) <<<"
        elif t > 0:
            tip = "stored! UP×4 then RIGHT+A"
        else:
            tip = "RIGHT+B to blue, then DOWN while still holding RIGHT"
        return [
            f"[SHINE] take{live['take']:02d}{gbit}  F5=save+reload  F1=save  ESC=quit",
            f"xy=({live['x']},{live['y']}) p={live['pose']}  "
            f"echoes={live['echoes']}/4  $0A68={live['timer']}",
            f"phase: {live['phase']}",
            tip,
            "TRAP: release RIGHT/idle = charge dies in 1f  |  B=dash A=act DOWN=store",
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
        return False

    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title=f"Shine practice — {series}",
            action_size=12,
            base_fps=60,
            initial_speed=args.speed,
            headless=False,
        )
        session.on_hud = on_hud
        session.on_step = on_step
        session.on_key_down = on_key_down
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        quit_flag["done"] = True

    n = live["take"] - 1
    print(f"session end — takes saved under {out_dir} (last index attempted {n})")
    _list_takes_dir(out_dir)
    return 0


# ---------------------------------------------------------------------------
# drill — bot charges + holds RIGHT+B; human only adds DOWN
# ---------------------------------------------------------------------------


def cmd_drill(args: argparse.Namespace) -> int:
    """Bot keeps boost alive; you only practice the store tap.

    Measured: idle or B-without-direction dumps echoes 4→0 in one frame.
    Drill holds RIGHT+B after full charge so DOWN still works.
    """
    pin = ensure_landing_pin(Path(args.source) if args.source else None)
    state_bytes = read_state_bytes(pin)
    assist = UnlimitedResourcesAssist() if not args.no_assist else None
    hold_budget = int(args.hold_budget)

    print("=" * 60)
    print("STORE DRILL  — bot runs; YOU only press DOWN")
    print(f"  pin: {pin}")
    print("  Bot keeps RIGHT+B (boost stays alive after e=4)")
    print("  You: arrow DOWN when HUD says PRESS DOWN")
    print("  Success = $0A68 > 0   |   R = next rep   |   ESC = quit")
    print("  Do NOT release stick first — that kills the charge")
    print("=" * 60)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    live: dict[str, Any] = {
        "phase": "charge",
        "echoes": 0,
        "timer": 0,
        "pose": 0,
        "x": 0,
        "y": 0,
        "rep": 1,
        "ok": 0,
        "fail": 0,
        "msg": "charging…",
        "hold_left": 0,
        "human_down": False,
    }

    def boot_pin(e: Any) -> None:
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
        live["phase"] = "charge"
        live["msg"] = "bot charging RIGHT+B…"
        live["hold_left"] = 0
        live["human_down"] = False
        st = parse_env_state(e, mode="nav")
        live.update(
            x=st.samus_x,
            y=st.samus_y,
            pose=st.pose,
            echoes=0,
            timer=0,
        )
        print(f"[DRILL rep{live['rep']}] go — wait for PRESS DOWN", flush=True)

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        boot_pin(e)
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]

    def bot(_obs, _info):
        # Read keyboard DOWN even while bot owns the action stream.
        try:
            import pygame as pg

            keys = pg.key.get_pressed()
            live["human_down"] = bool(keys[pg.K_DOWN])
        except Exception:  # noqa: BLE001
            live["human_down"] = False

        phase = live["phase"]
        if phase == "charge":
            return buttons("RIGHT", "B")
        if phase == "hold":
            live["hold_left"] = max(0, int(live["hold_left"]) - 1)
            if live["human_down"]:
                # Store while boost still held — the correct motion.
                return buttons("RIGHT", "B", "DOWN")
            if int(live["hold_left"]) <= 0 and int(live["timer"]) == 0:
                live["phase"] = "done"
                live["fail"] = int(live["fail"]) + 1
                live["msg"] = "FAIL — no DOWN in time  (R=retry)"
                print("[DRILL] FAIL window expired", flush=True)
                return idle_action()
            return buttons("RIGHT", "B")
        # done — freeze
        return idle_action()

    def on_step(_o, _r, _d, _i) -> None:
        st = parse_env_state(env, mode="nav")
        if assist is not None:
            try:
                assist.apply(env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    assist.apply(env, st)
                except Exception:  # noqa: BLE001
                    pass
        w = spark.read_spark_wram(env)
        live["x"] = st.samus_x
        live["y"] = st.samus_y
        live["pose"] = st.pose
        live["echoes"] = w["speed_echoes"]
        live["timer"] = w["spark_timer"]

        if live["phase"] == "charge" and w["speed_echoes"] >= spark.ECHOES_FULL:
            live["phase"] = "hold"
            live["hold_left"] = hold_budget
            live["msg"] = "PRESS DOWN NOW (bot keeps RIGHT+B)"
            print("[DRILL] echoes=4 — PRESS DOWN", flush=True)

        if live["phase"] == "hold" and w["spark_timer"] > 0:
            live["phase"] = "done"
            live["ok"] = int(live["ok"]) + 1
            live["msg"] = f"SUCCESS $0A68={w['spark_timer']}  R=next"
            print(
                f"[DRILL] SUCCESS timer={w['spark_timer']} pose={st.pose} "
                f"ok={live['ok']}",
                flush=True,
            )

    def on_hud(_info) -> list[str]:
        e, t = int(live["echoes"]), int(live["timer"])
        hl = int(live.get("hold_left") or 0)
        urgent = live["phase"] == "hold" and t == 0
        return [
            f"[DRILL] rep{live['rep']}  ok={live['ok']} fail={live['fail']}  "
            f"R=retry  ESC=quit",
            f"xy=({live['x']},{live['y']}) p={live['pose']}  "
            f"echoes={e}/4  $0A68={t}  hold={hl}f",
            f"{'>>> PRESS DOWN NOW <<<' if urgent else live['msg']}",
            "bot holds RIGHT+B  |  you only add DOWN  |  do not release first",
            f"phase={live['phase']}  down_key={live.get('human_down')}",
        ]

    def on_key_down(key: int) -> bool:
        try:
            import pygame as pg
        except ImportError:
            return False
        if key == pg.K_r:
            live["rep"] = int(live["rep"]) + 1
            boot_pin(env)
            return True
        return False

    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title="Shine STORE drill",
            bot=bot,
            action_size=12,
            base_fps=60,
            initial_speed=args.speed,
            headless=False,
        )
        session.on_hud = on_hud
        session.on_step = on_step
        session.on_key_down = on_key_down
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    print(f"drill end  ok={live['ok']} fail={live['fail']}")
    return 0 if int(live["ok"]) > 0 else 1


# ---------------------------------------------------------------------------
# demo (bot green recipe)
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    pin = ensure_landing_pin(Path(args.source) if args.source else None)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist() if not args.no_assist else None
    env.reset()
    env.em.set_state(read_state_bytes(pin))

    class _Sess:
        def __init__(self) -> None:
            self.env = env
            self.frame = 0
            self.state = parse_env_state(env, mode="nav")

        def step(self, action, reason: str = ""):
            del reason
            env.step(action)
            self.frame += 1
            if assist is not None:
                st0 = parse_env_state(env, mode="nav")
                try:
                    assist.apply(env.data, st0)
                except Exception:  # noqa: BLE001
                    assist.apply(env, st0)
            self.state = parse_env_state(env, mode="nav")
            return self.state

    sess = _Sess()
    for _ in range(12):
        sess.step(idle_action())

    print(f"[DEMO] pin {pin} xy=({sess.state.samus_x},{sess.state.samus_y})")
    charge = spark.charge_until_boost(sess, "RIGHT", budget=500)
    print(f"  charge ok={charge.get('ok')} frames={charge.get('frames')} "
          f"boost={charge.get('boost')}")
    if not charge.get("ok"):
        env.close()
        return 1
    store = spark.crouch_store(sess)
    print(f"  store ok={store.get('ok')} peak={store.get('peak_timer_during_store')} "
          f"armed={store.get('armed')}")
    if not store.get("ok"):
        env.close()
        return 1
    act = spark.activate_shinespark(
        sess,
        "RIGHT",
        pre_stand_frames=4,
        pre_stand_buttons=("UP",),
        hold_frames=16,
        travel_budget=200,
    )
    print(
        f"  activate ok={act.get('ok')} spark_pose={act.get('spark_pose_seen')} "
        f"max_x={act.get('max_x')} min_y={act.get('min_y')} "
        f"final={act.get('final')}"
    )
    # synthetic trace for diagnose
    # (demo is open-loop; build minimal milestones for print)
    ok = bool(act.get("ok") and act.get("spark_pose_seen"))
    print(f"{'GREEN' if ok else 'RED'} demo horizontal spark")
    env.close()
    return 0 if ok else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"missing: {path}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = data.get("trace") or data.get("spark_trace") or []
    diag = diagnose_trace(trace)
    print(format_diagnosis(diag, take=path.name))
    # rewrite diagnosis into file if --write
    if args.write:
        data["diagnosis"] = diag
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"updated {path}")
    return 0 if diag.get("ok") else 1


def _list_takes_dir(d: Path) -> None:
    paths = sorted(d.glob("take*.json"))
    if not paths:
        print(f"  (no takes in {d})")
        return
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"  {p.name}: unreadable")
            continue
        diag = data.get("diagnosis") or diagnose_trace(data.get("trace") or [])
        peaks = diag.get("peaks") or {}
        print(
            f"  {p.name}: grade={diag.get('grade')} frames={data.get('frame_count')} "
            f"echoes={peaks.get('echoes')} timer={peaks.get('spark_timer')} "
            f"spark_f={peaks.get('spark_travel_frames')}"
        )


def cmd_list(args: argparse.Namespace) -> int:
    series = args.series
    if not series:
        # list series dirs
        if not TASKS_ROOT.is_dir():
            print(f"No practice root yet: {TASKS_ROOT}")
            return 1
        for d in sorted(TASKS_ROOT.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("take*.json")))
                print(f"  {d.name}: {n} takes")
        return 0
    d = Path(args.out_dir) if args.out_dir else TASKS_ROOT / series
    print(f"Series {series}  {d}")
    _list_takes_dir(d)
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    del args
    pin = ensure_landing_pin(None)
    print(f"pin ready: {pin}")
    return 0 if pin.is_file() else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("human", help="Multi-take Landing Site practice + diagnose")
    p_h.add_argument("--source", type=Path, default=None)
    p_h.add_argument("--series", type=str, default=None)
    p_h.add_argument("--out-dir", type=Path, default=None)
    p_h.add_argument("--scale", type=int, default=3)
    p_h.add_argument("--speed", type=float, default=1.0)
    p_h.add_argument("--no-assist", action="store_true")
    p_h.set_defaults(func=cmd_human)

    p_dr = sub.add_parser(
        "drill",
        help="STORE only: bot charges+holds RIGHT+B; you press DOWN",
    )
    p_dr.add_argument("--source", type=Path, default=None)
    p_dr.add_argument("--scale", type=int, default=3)
    p_dr.add_argument("--speed", type=float, default=1.0)
    p_dr.add_argument("--no-assist", action="store_true")
    p_dr.add_argument(
        "--hold-budget",
        type=int,
        default=90,
        help="Frames bot keeps RIGHT+B after e=4 waiting for your DOWN (default 90)",
    )
    p_dr.set_defaults(func=cmd_drill)

    p_d = sub.add_parser("demo", help="Bot plays green horizontal spark recipe")
    p_d.add_argument("--source", type=Path, default=None)
    p_d.add_argument("--no-assist", action="store_true")
    p_d.set_defaults(func=cmd_demo)

    p_g = sub.add_parser("diagnose", help="Re-run diagnosis on a take JSON")
    p_g.add_argument("path", type=Path)
    p_g.add_argument("--write", action="store_true")
    p_g.set_defaults(func=cmd_diagnose)

    p_l = sub.add_parser("list", help="List series / takes")
    p_l.add_argument("--series", type=str, default=None)
    p_l.add_argument("--out-dir", type=Path, default=None)
    p_l.set_defaults(func=cmd_list)

    p_b = sub.add_parser("bootstrap", help="Rebuild Landing Site pin from pre-Moat")
    p_b.set_defaults(func=cmd_bootstrap)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
