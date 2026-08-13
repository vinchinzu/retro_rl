#!/usr/bin/env python3
"""Moat shinespark probe — live watch, WRAM store window, door spark.

Reusable CLI (no one-off pin scripts). Default source is the cleared runway
pin when present, else human end.

```bash
# Live bot (clear → runway → charge → store → spark)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py watch

# Human play + live WRAM HUD (F5 saves current state to --out)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py human

# Charge → crouch-store → measure $0A68 window → optional mid-door spark
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py measure \\
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state

# Sweep idle frames after store before activate (window leeway)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py measure --sweep 0:90:5

# Store→stand→spin-hop→unspin→activate (Moat carry)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py hop \\
  --stand 8 --hop-f 13 --unspin UP --unspin-f 4 --activate RIGHT+A

# Hop grid around the MOAT-partial band
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py hop --sweep \\
  'stand=6,8,10;hop=12:14:1;unspin_f=3,4,5;run=0,2;travel=RIGHT+A,A,RIGHT+UP+A'

# Headless full controller once
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure

# Annotated practice MP4 (velocity, Δv, shine bar, phase cues)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py record
```

Keys (watch/human): ``[`` ``]`` speed · TAB turbo · ``~`` bot/human · ESC quit ·
F5 save (human mode)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from retro_harness.video import (  # noqa: E402
    VideoCaptureConfig,
    VideoRecorder,
    format_snes_buttons,
)
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import save_dev_state  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state
from super_metroid.routes.kpdr.moat import (  # noqa: E402
    ROOM_KIHUNTER,
    ROOM_MOAT,
    ROOM_WEST_OCEAN,
    _LEFT_DOOR_LIP_X,
    _WALK_STOP_X,
    list_air_enemies,
    near_left_door,
    play_moat_shinespark,
)
from super_metroid.routes.skills.knockback import (  # noqa: E402
    escape_knockback_spin,
    is_knockback,
)
from super_metroid.routes.skills import shinespark as spark_skill  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_SOURCE_CANDIDATES = (
    SCRATCH / "post_kihunter_pre_moat_spark.state",
    SCRATCH / "pre_moat_shinespark_runway.state",
    SCRATCH / "speed_with_spazer_human_end.state",
)
DEFAULT_REPORT_DIR = Path("snes/super_metroid/debug/moat_spark")
DEFAULT_PRACTICE_VIDEO = Path(
    "snes/super_metroid/recordings/moat_shinespark_practice_hud.mp4"
)
# Typical full arm after crouch-store (for shine countdown bar fill).
_SPARK_TIMER_FULL = spark_skill.TYPICAL_ARM_TIMER
_ECHOES_FULL = spark_skill.ECHOES_FULL


def default_source() -> Path:
    for p in DEFAULT_SOURCE_CANDIDATES:
        if p.is_file():
            return p
    return DEFAULT_SOURCE_CANDIDATES[-1]


spark_wram = spark_skill.read_spark_wram
snap = spark_skill.spark_snapshot


class _Sess:
    """Minimal ControllerSession + frame log."""

    def __init__(self, env, assist: UnlimitedResourcesAssist | None):
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="nav")
        self.log: list[dict[str, Any]] = []
        self.phase = "boot"
        self.detail = ""
        self.done = False
        self.success = False
        self.error: str | None = None
        # live bot fields (watch mode)
        self._live_i = 0
        self._live_dir = "LEFT"
        self._live_stage = "clear"
        self._store_left = 0
        self._spark_left = 0
        self._boost_seen = False
        self._clear_ok = 0
        self._plant_face: str | None = None

    def step(self, action, reason: str = "", *, record: bool = False):
        self.env.step(action)
        self.frame += 1
        if self.assist is not None:
            st0 = parse_env_state(self.env, frame=self.frame, mode="nav")
            try:
                self.assist.apply(self.env.data, st0)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st0)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        if reason:
            if reason.startswith("kihunter_clear"):
                self.phase = "clear"
            elif reason.startswith("kihunter_runway"):
                self.phase = "runway"
            elif reason.startswith("kihunter_charge") or reason.startswith("charge"):
                self.phase = "charge"
            elif "store" in reason:
                self.phase = "store"
            elif "spark" in reason or "travel" in reason or "activate" in reason:
                self.phase = "spark"
            elif reason.startswith("idle"):
                self.phase = "idle"
            self.detail = reason
        if record:
            row = snap(self.env, self.frame)
            row["phase"] = self.phase
            row["reason"] = reason
            self.log.append(row)
        return self.state

    def hold(self, n: int, *btns: str, reason: str = "", record: bool = False):
        act = buttons(*btns) if btns else idle_action()
        for _ in range(n):
            self.step(act, reason=reason, record=record)
        return self.state


def boot_env(source: Path, *, assist: bool = True):
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    a = UnlimitedResourcesAssist() if assist else None
    env.reset()
    env.em.set_state(read_state_bytes(source))
    sess = _Sess(env, a)
    for _ in range(10):
        sess.step(idle_action(), reason="boot")
    sess.frame = 0
    sess.log.clear()
    sess.state = parse_env_state(env, mode="nav")
    return env, sess


# ---------------------------------------------------------------------------
# measure: charge → store → window → activate → door
# ---------------------------------------------------------------------------


def _charge_until_boost(
    sess: _Sess,
    *,
    budget: int = 500,
    record: bool = True,
) -> dict[str, Any]:
    """RIGHT+B until grounded speed_boosting (echoes ≥4)."""
    first_echo: dict[str, Any] | None = None
    boost_row: dict[str, Any] | None = None
    for _ in range(budget):
        st = sess.state
        if st.room_id not in (ROOM_KIHUNTER, ROOM_MOAT):
            break
        w = spark_wram(sess.env)
        if first_echo is None and w["speed_echoes"] >= 1:
            first_echo = snap(sess.env, sess.frame)
        # Store candidate: grounded full echoes
        if st.speed_boosting and st.velocity_y == 0 and st.pose not in (137, 138):
            boost_row = snap(sess.env, sess.frame)
            break
        x = st.samus_x
        if 545 <= x <= 575 and st.velocity_y == 0:
            sess.hold(1, "RIGHT", "B", "A", reason="charge_trap_hop", record=record)
        else:
            sess.hold(1, "RIGHT", "B", reason="charge_run", record=record)
    if boost_row is None:
        boost_row = snap(sess.env, sess.frame)
    return {"first_echo": first_echo, "boost": boost_row}


def _crouch_store(
    sess: _Sess,
    *,
    store_frames: int = 18,
    record: bool = True,
) -> dict[str, Any]:
    """Hold DOWN; capture first frame spark_timer > 0 (armed store)."""
    armed: dict[str, Any] | None = None
    peak = 0
    for i in range(store_frames):
        sess.hold(1, "DOWN", reason=f"store_{i}", record=record)
        w = spark_wram(sess.env)
        peak = max(peak, w["spark_timer"])
        if armed is None and w["spark_timer"] > 0:
            armed = snap(sess.env, sess.frame)
            armed["store_frame_index"] = i
    return {
        "armed": armed,
        "peak_timer_during_store": peak,
        "after": snap(sess.env, sess.frame),
    }


def _idle_window(
    sess: _Sess,
    *,
    frames: int,
    hold_down: bool = True,
    record: bool = True,
) -> dict[str, Any]:
    """Sit after store; track $0A68 until zero or budget ends."""
    series: list[dict[str, Any]] = []
    alive = 0
    for i in range(frames):
        if hold_down:
            sess.hold(1, "DOWN", reason=f"idle_store_{i}", record=record)
        else:
            sess.hold(1, reason=f"idle_{i}", record=record)
        row = snap(sess.env, sess.frame)
        series.append(row)
        if row["spark_timer"] > 0:
            alive += 1
        elif i > 0 and series[0]["spark_timer"] > 0:
            # timer died
            break
    nonzero = [r for r in series if r["spark_timer"] > 0]
    return {
        "requested_frames": frames,
        "hold_down": hold_down,
        "frames_timer_gt0": alive,
        "timer_start": series[0]["spark_timer"] if series else 0,
        "timer_end": series[-1]["spark_timer"] if series else 0,
        "first_zero_frame": next(
            (r["frame"] for r in series if r["spark_timer"] == 0), None
        ),
        "samples": series if len(series) <= 120 else series[:: max(1, len(series) // 60)],
        "sample_count": len(series),
    }


def _parse_hold(spec: str) -> tuple[str, ...]:
    """Parse hold string into harness button names.

    Accepts ``RIGHT+A``, ``R+A``, ``A``, ``RIGHT+UP+A``, ``UP``, ``X``, ``L``.
    Short aliases: R=RIGHT, L stays L (shoulder), U=UP, D=DOWN.
    """
    raw = (spec or "").strip()
    if not raw or raw.lower() in ("none", "idle", "-"):
        return ()
    alias = {
        "R": "RIGHT",
        "RIGHT": "RIGHT",
        "L": "L",
        "LEFT": "LEFT",
        "U": "UP",
        "UP": "UP",
        "D": "DOWN",
        "DOWN": "DOWN",
        "A": "A",
        "B": "B",
        "X": "X",
        "Y": "Y",
        "SELECT": "SELECT",
        "START": "START",
    }
    parts: list[str] = []
    for tok in raw.replace(",", "+").split("+"):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in alias:
            raise ValueError(f"unknown hold button token {tok!r} in {spec!r}")
        parts.append(alias[t])
    return tuple(parts)


def _is_spark_pose(pose: int) -> bool:
    return pose in (199, 200, 201, 202)


def _activate_and_travel(
    sess: _Sess,
    *,
    travel_budget: int = 700,
    record: bool = True,
    activate_hold: tuple[str, ...] = ("RIGHT", "A"),
    travel_hold: tuple[str, ...] | None = None,
    activate_frames: int = 12,
    retap_a_every: int = 0,
    moat_sample_every: int = 10,
    door_open: bool = True,
) -> dict[str, Any]:
    """Activate shinespark and hold travel toward West Ocean.

    Harness: A=jump/shine-activate, B=dash. Using B after store only walks.
    ``retap_a_every``: if >0, on every Nth Moat frame force a 1f A-only press
    (then resume travel_hold) to re-kick a stalling spark.
    ``moat_sample_every``: log $0A68/pose/xy every N frames while in Moat.
    ``door_open``: after spark dies in Moat (~x475 wall), walk RIGHT+X to open
    the blue door into West Ocean (verified pure path).
    """
    if travel_hold is None:
        travel_hold = activate_hold
    activate_snap = None
    spark_pose = False
    for i in range(activate_frames):
        sess.hold(1, *activate_hold, reason=f"activate_{i}", record=record)
        st = sess.state
        w = spark_wram(sess.env)
        if _is_spark_pose(st.pose):
            spark_pose = True
        if activate_snap is None and (spark_pose or w["spark_timer"] > 0):
            activate_snap = snap(sess.env, sess.frame)
        if sess.state.room_id in (ROOM_MOAT, ROOM_WEST_OCEAN):
            break

    door_entry: dict[str, Any] | None = None
    moat_entry: dict[str, Any] | None = None
    west: dict[str, Any] | None = None
    moat_samples: list[dict[str, Any]] = []
    moat_max_x = 0
    moat_min_y: int | None = None
    moat_max_y: int | None = None
    moat_frames = 0
    spark_died_at: dict[str, Any] | None = None
    last_spark_timer = -1
    post_spark_frames = 0

    for i in range(travel_budget):
        st = sess.state
        if moat_entry is None and st.room_id == ROOM_MOAT and st.game_state == 8:
            moat_entry = snap(sess.env, sess.frame)
        if (
            door_entry is None
            and st.room_id == ROOM_KIHUNTER
            and (st.door_transition != 0 or st.game_state != 8 or st.samus_x > 740)
        ):
            door_entry = snap(sess.env, sess.frame)
        if st.room_id == ROOM_WEST_OCEAN and st.door_transition == 0 and st.game_state == 8:
            west = snap(sess.env, sess.frame)
            break
        # Hold RIGHT through unfinished West transition
        if st.room_id == ROOM_WEST_OCEAN:
            sess.hold(1, "RIGHT", reason=f"west_trans_{i}", record=record)
            continue

        # Dense Moat telemetry
        if st.room_id == ROOM_MOAT and st.game_state == 8:
            moat_frames += 1
            moat_max_x = max(moat_max_x, st.samus_x)
            moat_min_y = st.samus_y if moat_min_y is None else min(moat_min_y, st.samus_y)
            moat_max_y = st.samus_y if moat_max_y is None else max(moat_max_y, st.samus_y)
            w = spark_wram(sess.env)
            if moat_sample_every > 0 and (
                moat_frames == 1 or moat_frames % moat_sample_every == 0
            ):
                row = snap(sess.env, sess.frame)
                row["moat_f"] = moat_frames
                moat_samples.append(row)
            if (
                spark_died_at is None
                and last_spark_timer > 0
                and w["spark_timer"] == 0
                and not _is_spark_pose(st.pose)
            ):
                spark_died_at = snap(sess.env, sess.frame)
                spark_died_at["moat_f"] = moat_frames
            last_spark_timer = w["spark_timer"]

            sparking = _is_spark_pose(st.pose)
            if sparking:
                if retap_a_every > 0 and moat_frames % retap_a_every == 0:
                    sess.hold(1, "A", reason=f"retap_a_{i}", record=record)
                    continue
                sess.hold(1, *travel_hold, reason=f"travel_{i}", record=record)
                continue

            # Spark dead in Moat — open blue door + walk into West
            post_spark_frames += 1
            if door_open:
                # Pulse shoot so blue door opens; keep RIGHT for contact
                if post_spark_frames % 8 < 5:
                    sess.hold(1, "RIGHT", "X", reason=f"door_open_{i}", record=record)
                else:
                    sess.hold(1, "RIGHT", reason=f"door_walk_{i}", record=record)
            else:
                sess.hold(1, *travel_hold, reason=f"travel_{i}", record=record)
            continue

        # Early abort: timer dead, not sparkling, still in Kihunter trench
        if (
            i > 40
            and st.room_id == ROOM_KIHUNTER
            and st.game_state == 8
            and st.door_transition == 0
            and spark_wram(sess.env)["spark_timer"] == 0
            and not _is_spark_pose(st.pose)
            and st.samus_x < 700
        ):
            break

        sess.hold(1, *travel_hold, reason=f"travel_{i}", record=record)

    # Settle a few frames if we just landed West mid-transition edge
    if (
        west is None
        and sess.state.room_id == ROOM_WEST_OCEAN
    ):
        for j in range(30):
            sess.hold(1, "RIGHT", reason=f"west_settle_{j}", record=record)
            if (
                sess.state.room_id == ROOM_WEST_OCEAN
                and sess.state.door_transition == 0
                and sess.state.game_state == 8
            ):
                west = snap(sess.env, sess.frame)
                break

    final = snap(sess.env, sess.frame)
    if west is None and final["room"] == ROOM_WEST_OCEAN and final["door_trans"] == 0 and final["gs"] == 8:
        west = final
    return {
        "activate": activate_snap or final,
        "door_entry": door_entry,
        "moat_entry": moat_entry,
        "west_ocean": west,
        "final": final,
        "reached_moat": final["room"] == ROOM_MOAT or moat_entry is not None,
        "reached_west": west is not None,
        "moat_samples": moat_samples,
        "moat_max_x": moat_max_x,
        "moat_min_y": moat_min_y,
        "moat_max_y": moat_max_y,
        "moat_frames": moat_frames,
        "spark_died_at": spark_died_at,
        "activate_hold": list(activate_hold),
        "travel_hold": list(travel_hold),
        "retap_a_every": retap_a_every,
        "door_open": door_open,
        "post_spark_frames": post_spark_frames,
    }


# ---------------------------------------------------------------------------
# hop: store → stand → micro-run → spin hop → unspin → activate mid-air
# ---------------------------------------------------------------------------


def _hop_unspin_activate(
    sess: _Sess,
    *,
    stand: int = 8,
    micro_run: int = 0,
    hop_f: int = 13,
    hop_a_f: int = -1,
    hop_hold: tuple[str, ...] = ("RIGHT", "B", "A"),
    hop_coast_hold: tuple[str, ...] = ("RIGHT", "B"),
    unspin: tuple[str, ...] = ("UP",),
    unspin_f: int = 4,
    activate_hold: tuple[str, ...] = ("RIGHT", "A"),
    travel_hold: tuple[str, ...] | None = None,
    activate_frames: int = 16,
    retap_a_every: int = 0,
    travel_budget: int = 700,
    record: bool = True,
    moat_sample_every: int = 10,
    door_open: bool = True,
) -> dict[str, Any]:
    """After store is armed: stand, optional micro-run, spin hop, unspin, spark.

    Sequence (wiki-aligned store-first):
      idle stand ~stand f
      RIGHT+B ~micro_run f   (required to leave crouch pose 39 → pose 9)
      hop_hold for hop_a_f frames (default all hop_f) then hop_coast for rest
      unspin buttons ~unspin_f f  (UP mid-air → often pose 199 activate)
      activate_hold then travel_hold through Moat
      after spark dies at ~x475: RIGHT+X door open into West (if door_open)

    ``hop_a_f``: frames of A-including hop_hold at hop start. -1 means hold
    hop_hold for entire hop_f (legacy). Use e.g. hop_a_f=2 so A only starts
    the spin jump, then coast RIGHT+B without re-tapping A (avoids early
    spark-activate into the x555 wall). Continuous hop with micro_run≥2 is
    the verified Moat band.
    """
    if travel_hold is None:
        travel_hold = activate_hold
    if hop_a_f < 0:
        hop_a_f = hop_f

    # Stand up from crouch-store (neutral idle so pose returns to standing)
    for i in range(stand):
        sess.hold(1, reason=f"stand_{i}", record=record)
    after_stand = snap(sess.env, sess.frame)

    for i in range(micro_run):
        sess.hold(1, "RIGHT", "B", reason=f"micro_run_{i}", record=record)
    after_run = snap(sess.env, sess.frame) if micro_run else after_stand

    a_frames = max(0, min(hop_a_f, hop_f))
    coast_frames = max(0, hop_f - a_frames)
    for i in range(a_frames):
        sess.hold(1, *hop_hold, reason=f"hop_a_{i}", record=record)
    for i in range(coast_frames):
        sess.hold(1, *hop_coast_hold, reason=f"hop_coast_{i}", record=record)
    after_hop = snap(sess.env, sess.frame)

    for i in range(unspin_f):
        if unspin:
            sess.hold(1, *unspin, reason=f"unspin_{i}", record=record)
        else:
            sess.hold(1, reason=f"unspin_idle_{i}", record=record)
    after_unspin = snap(sess.env, sess.frame)

    travel = _activate_and_travel(
        sess,
        travel_budget=travel_budget,
        record=record,
        activate_hold=activate_hold,
        travel_hold=travel_hold,
        activate_frames=activate_frames,
        retap_a_every=retap_a_every,
        moat_sample_every=moat_sample_every,
        door_open=door_open,
    )
    return {
        "after_stand": after_stand,
        "after_run": after_run,
        "after_hop": after_hop,
        "after_unspin": after_unspin,
        "travel": travel,
        "params": {
            "stand": stand,
            "micro_run": micro_run,
            "hop_f": hop_f,
            "hop_a_f": hop_a_f,
            "hop_hold": list(hop_hold),
            "hop_coast_hold": list(hop_coast_hold),
            "unspin": list(unspin),
            "unspin_f": unspin_f,
            "activate_hold": list(activate_hold),
            "travel_hold": list(travel_hold),
            "activate_frames": activate_frames,
            "retap_a_every": retap_a_every,
            "door_open": door_open,
        },
    }


def run_hop_once(
    source: Path,
    *,
    store_frames: int = 18,
    stand: int = 8,
    micro_run: int = 0,
    hop_f: int = 13,
    hop_a_f: int = -1,
    hop_hold: tuple[str, ...] = ("RIGHT", "B", "A"),
    hop_coast_hold: tuple[str, ...] = ("RIGHT", "B"),
    unspin: tuple[str, ...] = ("UP",),
    unspin_f: int = 4,
    activate_hold: tuple[str, ...] = ("RIGHT", "A"),
    travel_hold: tuple[str, ...] | None = None,
    activate_frames: int = 16,
    retap_a_every: int = 0,
    door_open: bool = True,
    record_all: bool = True,
    assist: bool = True,
    save_west: Path | None = None,
) -> dict[str, Any]:
    """Full pin → charge → store → hop-unspin-activate → Moat/West."""
    env, sess = boot_env(source, assist=assist)
    report: dict[str, Any] = {
        "source": str(source),
        "boot": snap(env, 0),
        "mode": "hop",
        "params": {
            "store_frames": store_frames,
            "stand": stand,
            "micro_run": micro_run,
            "hop_f": hop_f,
            "hop_a_f": hop_a_f,
            "hop_hold": list(hop_hold),
            "hop_coast_hold": list(hop_coast_hold),
            "unspin": list(unspin),
            "unspin_f": unspin_f,
            "activate_hold": list(activate_hold),
            "travel_hold": list(travel_hold or activate_hold),
            "activate_frames": activate_frames,
            "retap_a_every": retap_a_every,
            "door_open": door_open,
        },
    }
    try:
        report["charge"] = _charge_until_boost(sess, record=record_all)
        if not report["charge"]["boost"].get("speed_boosting"):
            report["ok"] = False
            report["error"] = "never reached speed_boosting (echoes≥4)"
            report["final"] = snap(env, sess.frame)
            return report

        report["store"] = _crouch_store(
            sess, store_frames=store_frames, record=record_all
        )
        armed = report["store"]["armed"]
        if armed is None:
            report["ok"] = False
            report["error"] = (
                f"store never armed $0A68 "
                f"(peak={report['store']['peak_timer_during_store']})"
            )
            report["final"] = snap(env, sess.frame)
            return report

        hop = _hop_unspin_activate(
            sess,
            stand=stand,
            micro_run=micro_run,
            hop_f=hop_f,
            hop_a_f=hop_a_f,
            hop_hold=hop_hold,
            hop_coast_hold=hop_coast_hold,
            unspin=unspin,
            unspin_f=unspin_f,
            activate_hold=activate_hold,
            travel_hold=travel_hold,
            activate_frames=activate_frames,
            retap_a_every=retap_a_every,
            door_open=door_open,
            record=record_all,
        )
        report["hop"] = {
            "after_stand": hop["after_stand"],
            "after_run": hop["after_run"],
            "after_hop": hop["after_hop"],
            "after_unspin": hop["after_unspin"],
            "params": hop["params"],
        }
        travel = hop["travel"]
        report["travel"] = travel
        report["ok"] = bool(travel["reached_west"])
        if not report["ok"] and travel["reached_moat"]:
            report["ok"] = "partial_moat"
        fin = travel["final"]
        report["error"] = None if report["ok"] is True else (
            f"stopped room={fin['room_hex']} xy=({fin['x']},{fin['y']}) "
            f"pose={fin['pose']} spark={fin['spark_timer']} "
            f"moat_max_x={travel.get('moat_max_x')} "
            f"act_y={(travel.get('activate') or {}).get('y')}"
        )
        report["final"] = fin
        report["log_len"] = len(sess.log)
        # Cap log in report for single runs that request it
        if record_all and len(sess.log) <= 800:
            report["log"] = sess.log
        else:
            report["log"] = sess.log[:: max(1, len(sess.log) // 200)] if sess.log else []

        if report["ok"] is True and save_west is not None:
            save_dev_state(env, save_west)
            report["west_state"] = str(save_west)
        return report
    finally:
        env.close()


def _hop_summary(rep: dict[str, Any]) -> dict[str, Any]:
    travel = rep.get("travel") or {}
    hop = rep.get("hop") or {}
    act = travel.get("activate") or {}
    fin = travel.get("final") or rep.get("final") or {}
    unspin = hop.get("after_unspin") or {}
    params = rep.get("params") or {}
    return {
        "params": params,
        "ok": rep.get("ok"),
        "error": rep.get("error"),
        "boost_xy": (
            (rep.get("charge") or {}).get("boost", {}).get("x"),
            (rep.get("charge") or {}).get("boost", {}).get("y"),
        ),
        "armed_timer": ((rep.get("store") or {}).get("armed") or {}).get("spark_timer"),
        "after_hop_xy": (
            (hop.get("after_hop") or {}).get("x"),
            (hop.get("after_hop") or {}).get("y"),
        ),
        "after_hop_pose": (hop.get("after_hop") or {}).get("pose"),
        "after_unspin_xy": (unspin.get("x"), unspin.get("y")),
        "after_unspin_pose": unspin.get("pose"),
        "after_unspin_spark": unspin.get("spark_timer"),
        "activate_xy": (act.get("x"), act.get("y")),
        "activate_pose": act.get("pose"),
        "activate_spark": act.get("spark_timer"),
        "reached_moat": travel.get("reached_moat"),
        "reached_west": travel.get("reached_west"),
        "moat_entry_xy": (
            (travel.get("moat_entry") or {}).get("x"),
            (travel.get("moat_entry") or {}).get("y"),
        ),
        "moat_entry_spark": (travel.get("moat_entry") or {}).get("spark_timer"),
        "moat_max_x": travel.get("moat_max_x"),
        "moat_min_y": travel.get("moat_min_y"),
        "moat_max_y": travel.get("moat_max_y"),
        "moat_frames": travel.get("moat_frames"),
        "moat_samples": travel.get("moat_samples") or [],
        "spark_died_at": travel.get("spark_died_at"),
        "final_room": fin.get("room_hex"),
        "final_xy": (fin.get("x"), fin.get("y")),
        "final_pose": fin.get("pose"),
        "final_spark": fin.get("spark_timer"),
    }


# ---------------------------------------------------------------------------
# Practice HUD overlay + annotated proof video
# ---------------------------------------------------------------------------


def _phase_cue(reason: str) -> tuple[str, str]:
    """Map step reason → (phase label, human practice cue)."""
    r = reason or ""
    if r.startswith("boot"):
        return "BOOT", "settle"
    if r.startswith("charge_trap"):
        return "CHARGE", "RIGHT+B+A hop trap · keep dash"
    if r.startswith("charge"):
        return "CHARGE", "HOLD RIGHT+B  (dash / speed)"
    if r.startswith("store"):
        return "STORE", "HOLD DOWN  (crouch-store shine)"
    if r.startswith("stand"):
        return "STAND", "release · leave crouch"
    if r.startswith("micro_run"):
        return "MICRO", "RIGHT+B  leave crouch → pose 9"
    if r.startswith("hop_a"):
        return "HOP", "RIGHT+B+A  spin over x555 wall"
    if r.startswith("hop_coast"):
        return "COAST", "RIGHT+B  coast spin (no early A)"
    if r.startswith("unspin"):
        return "UNSPIN", "HOLD UP  unspin mid-air"
    if r.startswith("activate"):
        return "ACTIVATE", "RIGHT+A  spark activate"
    if r.startswith("retap_a"):
        return "RETAP", "tap A"
    if r.startswith("door_open"):
        return "DOOR", "RIGHT+X  open blue door"
    if r.startswith("door_walk"):
        return "DOOR", "RIGHT  walk into door"
    if r.startswith("west"):
        return "WEST", "West Ocean settle"
    if r.startswith("travel"):
        return "SPARK", "HOLD RIGHT+A  across Moat"
    if r.startswith("idle"):
        return "IDLE", "wait"
    if "store" in r:
        return "STORE", "HOLD DOWN"
    if "spark" in r or "travel" in r:
        return "SPARK", "HOLD RIGHT+A"
    return (r.split("_", 1)[0].upper() if r else "—"), r or "—"


def _phase_budget(reason: str, params: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return (remaining, total) open-loop frames for the current phase, if known."""
    r = reason or ""

    def _idx(prefix: str) -> int | None:
        if not r.startswith(prefix):
            return None
        tail = r[len(prefix) :].lstrip("_")
        if tail.isdigit():
            return int(tail)
        return 0

    mapping = (
        ("store_", "store_frames"),
        ("stand_", "stand"),
        ("micro_run_", "micro_run"),
        ("hop_a_", "hop_f"),
        ("hop_coast_", "hop_f"),
        ("unspin_", "unspin_f"),
        ("activate_", "activate_frames"),
    )
    for prefix, key in mapping:
        i = _idx(prefix)
        if i is None:
            continue
        total = int(params.get(key) or 0)
        if total <= 0:
            return None, None
        # hop_a uses hop_a_f when set; remaining is against its own length
        if prefix == "hop_a_":
            hop_a = int(params.get("hop_a_f") or -1)
            total_a = total if hop_a < 0 else max(0, min(hop_a, total))
            left = max(0, total_a - i - 1)
            return left, total_a
        if prefix == "hop_coast_":
            hop_a = int(params.get("hop_a_f") or -1)
            a_frames = total if hop_a < 0 else max(0, min(hop_a, total))
            coast = max(0, total - a_frames)
            left = max(0, coast - i - 1)
            return left, coast
        left = max(0, total - i - 1)
        return left, total
    return None, None


def _bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    frac: float,
    *,
    fill: tuple[int, int, int],
    bg: tuple[int, int, int] = (28, 32, 40),
    border: tuple[int, int, int] = (90, 100, 120),
) -> None:
    frac = max(0.0, min(1.0, float(frac)))
    draw.rectangle([x, y, x + w, y + h], fill=bg, outline=border)
    fw = int(round((w - 2) * frac))
    if fw > 0:
        draw.rectangle([x + 1, y + 1, x + 1 + fw, y + h - 1], fill=fill)


def _echo_pips(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    echoes: int,
    *,
    full: int = _ECHOES_FULL,
    pip_w: int = 10,
    pip_h: int = 7,
    gap: int = 2,
) -> None:
    for i in range(full):
        px = x + i * (pip_w + gap)
        on = i < max(0, int(echoes))
        fill = (80, 200, 255) if on else (35, 40, 50)
        outline = (180, 230, 255) if on else (70, 80, 95)
        draw.rectangle([px, y, px + pip_w, y + pip_h], fill=fill, outline=outline)


def _load_hud_fonts() -> tuple[Any, Any]:
    """Prefer a real mono TTF so practice text is readable at 2× scale."""
    candidates = (
        "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-SemiCondensedBold.ttf",
        "/usr/share/fonts/noto/NotoSansMono-SemiCondensed.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return (
                    ImageFont.truetype(path, 11),
                    ImageFont.truetype(path, 10),
                )
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=10), ImageFont.load_default(size=8)
    except TypeError:
        f = ImageFont.load_default()
        return f, f


def render_practice_overlay(
    obs: np.ndarray,
    *,
    row: dict[str, Any],
    prev_vx: int | None,
    prev_vy: int | None,
    reason: str,
    params: dict[str, Any],
    action: Any = None,
) -> np.ndarray:
    """Burn velocity / shine bars / phase countdown onto one RGB frame."""
    rgb = np.asarray(obs, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    font, font_sm = _load_hud_fonts()

    phase, cue = _phase_cue(reason)
    left, total = _phase_budget(reason, params)
    vx = int(row.get("vx") or 0)
    vy = int(row.get("vy") or 0)
    dvx = 0 if prev_vx is None else vx - int(prev_vx)
    dvy = 0 if prev_vy is None else vy - int(prev_vy)
    echoes = int(row.get("speed_echoes") or 0)
    spark_t = int(row.get("spark_timer") or 0)
    spark_full = max(_SPARK_TIMER_FULL, spark_t, 1)
    room = row.get("room_hex") or f"0x{int(row.get('room') or 0):04X}"
    btn = format_snes_buttons(action)
    frame = int(row.get("frame") or 0)
    x = int(row.get("x") or 0)
    y = int(row.get("y") or 0)
    pose = int(row.get("pose") or 0)

    # Top translucent panel
    panel_h = 58
    draw.rectangle([0, 0, w, panel_h], fill=(0, 0, 0, 185))

    # Title / phase
    draw.text((4, 2), f"F{frame:04d}  {phase}", fill=(255, 220, 120), font=font)
    draw.text((96, 3), cue, fill=(180, 255, 200), font=font_sm)

    # Room / pose / xy
    draw.text(
        (4, 16),
        f"{room}  xy=({x},{y})  p={pose}",
        fill=(170, 185, 205),
        font=font_sm,
    )

    # Velocity + Δv
    def _sgn(v: int) -> str:
        return f"{v:+d}" if v != 0 else "0"

    draw.text(
        (4, 28),
        f"vx={vx:+d}  Δvx={_sgn(dvx)}   vy={vy:+d}  Δvy={_sgn(dvy)}",
        fill=(120, 220, 255),
        font=font_sm,
    )

    # Echoes power pips
    draw.text((4, 40), "SPD", fill=(150, 200, 255), font=font_sm)
    _echo_pips(draw, 32, 41, echoes, full=_ECHOES_FULL)
    draw.text(
        (32 + _ECHOES_FULL * 12 + 6, 40),
        f"{echoes}/{_ECHOES_FULL}",
        fill=(150, 200, 255),
        font=font_sm,
    )

    # Shine timer bar (right half of panel)
    bar_x = w // 2
    draw.text((bar_x, 16), "SHINE $0A68", fill=(255, 200, 120), font=font_sm)
    shine_frac = spark_t / float(spark_full)
    # color: armed yellow → sparking gold → empty gray
    if spark_t <= 0:
        fill = (70, 70, 80)
    elif int(row.get("pose") or 0) in spark_skill.SPARK_POSES:
        fill = (255, 210, 60)
    else:
        fill = (255, 140, 40)
    _bar(draw, bar_x, 28, w - bar_x - 6, 9, shine_frac, fill=fill)
    draw.text(
        (bar_x, 40),
        f"{spark_t}f left  / arm~{_SPARK_TIMER_FULL}",
        fill=(255, 210, 160),
        font=font_sm,
    )

    # Phase countdown bar under shine bar when open-loop budget known
    if left is not None and total is not None and total > 0:
        done = total - left
        frac = done / float(total)
        _bar(
            draw,
            bar_x,
            50,
            w - bar_x - 6,
            6,
            frac,
            fill=(100, 220, 140),
            bg=(30, 40, 35),
            border=(80, 120, 90),
        )
        draw.text(
            (4, 50),
            f"phase {done}/{total}  ({left}f left)",
            fill=(140, 230, 160),
            font=font_sm,
        )

    # Side dash reminder strip (bottom)
    side_h = 16
    draw.rectangle([0, h - side_h, w, h], fill=(0, 0, 0, 195))
    # Side-specific dash cue
    if phase == "CHARGE":
        side_cue = "DASH SIDE: RIGHT  ·  hold B the whole runway"
    elif phase in ("MICRO", "HOP", "COAST"):
        side_cue = "DASH SIDE: RIGHT  ·  B stays down through hop"
    elif phase == "ACTIVATE":
        side_cue = "SPARK SIDE: RIGHT+A  (A = shine activate in harness)"
    elif phase == "SPARK":
        side_cue = "TRAVEL: keep RIGHT+A  ·  dies ~x475 then door"
    elif phase == "DOOR":
        side_cue = "AFTER SPARK: RIGHT+X to open Moat blue door → West"
    elif phase == "STORE":
        side_cue = "STORE WINDOW: DOWN only  ·  $0A68 arms ~179"
    elif phase == "UNSPIN":
        side_cue = "UNSPIN: UP  ·  then RIGHT+A activate"
    else:
        side_cue = f"btn {btn}  ·  harness B=dash A=jump/shine"
    draw.text((4, h - side_h + 2), side_cue, fill=(230, 230, 240), font=font_sm)
    bw = draw.textbbox((0, 0), btn, font=font_sm)[2]
    draw.text((w - bw - 4, h - side_h + 2), btn, fill=(255, 220, 100), font=font_sm)

    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def run_record_hop(
    source: Path,
    *,
    video_path: Path = DEFAULT_PRACTICE_VIDEO,
    store_frames: int = 18,
    stand: int = 4,
    micro_run: int = 2,
    hop_f: int = 14,
    hop_a_f: int = -1,
    hop_hold: tuple[str, ...] = ("RIGHT", "B", "A"),
    hop_coast_hold: tuple[str, ...] = ("RIGHT", "B"),
    unspin: tuple[str, ...] = ("UP",),
    unspin_f: int = 3,
    activate_hold: tuple[str, ...] = ("RIGHT", "A"),
    travel_hold: tuple[str, ...] | None = None,
    activate_frames: int = 16,
    retap_a_every: int = 0,
    door_open: bool = True,
    pad_end: int = 45,
    assist: bool = True,
    scale: int = 2,
    save_west: Path | None = None,
) -> dict[str, Any]:
    """Replay verified Moat hop path and write practice-HUD MP4."""
    if travel_hold is None:
        travel_hold = activate_hold
    params = {
        "store_frames": store_frames,
        "stand": stand,
        "micro_run": micro_run,
        "hop_f": hop_f,
        "hop_a_f": hop_a_f,
        "hop_hold": list(hop_hold),
        "hop_coast_hold": list(hop_coast_hold),
        "unspin": list(unspin),
        "unspin_f": unspin_f,
        "activate_hold": list(activate_hold),
        "travel_hold": list(travel_hold),
        "activate_frames": activate_frames,
        "retap_a_every": retap_a_every,
        "door_open": door_open,
    }

    env, sess = boot_env(source, assist=assist)
    obs = env.render()
    if obs is None:
        env.step(idle_action())
        obs = env.render()
    assert obs is not None

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    config = VideoCaptureConfig(
        fps=60,
        scale=scale,
        crf=18,
        preset="veryfast",
        audio=False,
        footer=True,
    )
    writer = VideoRecorder(
        video_path,
        width=int(obs.shape[1]),
        height=int(obs.shape[0]),
        config=config,
    )

    prev_vx: int | None = None
    prev_vy: int | None = None
    last_reason = "boot"
    last_action: Any = None
    phase_marks: list[dict[str, Any]] = []
    last_phase = ""

    def _write(action: Any = None, reason: str = "") -> None:
        nonlocal prev_vx, prev_vy, last_reason, last_action, last_phase
        frame = env.render()
        if frame is None:
            return
        row = snap(env, sess.frame)
        row["phase"] = sess.phase
        row["reason"] = reason or last_reason
        if reason:
            last_reason = reason
        if action is not None:
            last_action = action
        phase, cue = _phase_cue(last_reason)
        if phase != last_phase:
            phase_marks.append(
                {
                    "frame": sess.frame,
                    "phase": phase,
                    "cue": cue,
                    "reason": last_reason,
                    "xy": (row["x"], row["y"]),
                    "vx": row["vx"],
                    "spark_timer": row["spark_timer"],
                    "echoes": row["speed_echoes"],
                }
            )
            last_phase = phase
        decorated = render_practice_overlay(
            frame,
            row=row,
            prev_vx=prev_vx,
            prev_vy=prev_vy,
            reason=last_reason,
            params=params,
            action=last_action,
        )
        writer.write(
            decorated,
            action=last_action,
            frame_index=sess.frame,
            room_id=int(row["room"]),
        )
        prev_vx = int(row["vx"])
        prev_vy = int(row["vy"])

    orig_step = sess.step

    def _step_rec(action, reason: str = "", *, record: bool = False):
        st = orig_step(action, reason=reason, record=record)
        _write(action=action, reason=reason)
        return st

    sess.step = _step_rec  # type: ignore[method-assign]

    report: dict[str, Any] = {
        "mode": "record",
        "source": str(source),
        "video": str(video_path),
        "params": params,
        "boot": snap(env, 0),
    }
    try:
        # Opening freeze for readability
        _write(action=None, reason="boot")
        report["charge"] = _charge_until_boost(sess, record=True)
        if not (report["charge"].get("boost") or {}).get("speed_boosting"):
            report["ok"] = False
            report["error"] = "never reached speed_boosting"
            report["final"] = snap(env, sess.frame)
            return report

        report["store"] = _crouch_store(
            sess, store_frames=store_frames, record=True
        )
        if report["store"].get("armed") is None:
            report["ok"] = False
            report["error"] = "store never armed"
            report["final"] = snap(env, sess.frame)
            return report

        hop = _hop_unspin_activate(
            sess,
            stand=stand,
            micro_run=micro_run,
            hop_f=hop_f,
            hop_a_f=hop_a_f,
            hop_hold=hop_hold,
            hop_coast_hold=hop_coast_hold,
            unspin=unspin,
            unspin_f=unspin_f,
            activate_hold=activate_hold,
            travel_hold=travel_hold,
            activate_frames=activate_frames,
            retap_a_every=retap_a_every,
            door_open=door_open,
            record=True,
        )
        report["hop"] = {
            "after_stand": hop["after_stand"],
            "after_run": hop["after_run"],
            "after_hop": hop["after_hop"],
            "after_unspin": hop["after_unspin"],
            "params": hop["params"],
        }
        travel = hop["travel"]
        report["travel"] = {
            k: v
            for k, v in travel.items()
            if k != "moat_samples"
        }
        report["ok"] = bool(travel.get("reached_west"))
        if not report["ok"] and travel.get("reached_moat"):
            report["ok"] = "partial_moat"
        report["final"] = travel.get("final") or snap(env, sess.frame)
        report["error"] = None if report["ok"] is True else (
            f"stopped room={(report['final'] or {}).get('room_hex')} "
            f"moat_max_x={travel.get('moat_max_x')}"
        )
        if pad_end > 0:
            for i in range(pad_end):
                sess.hold(1, reason=f"west_pad_{i}", record=False)
        if report["ok"] is True and save_west is not None:
            save_dev_state(env, save_west)
            report["west_state"] = str(save_west)
        report["phase_marks"] = phase_marks
        report["frames_written"] = writer.frames_written
        report["summary"] = _hop_summary(
            {
                **report,
                "travel": travel,
                "params": params,
            }
        )
        return report
    finally:
        writer.close()
        env.close()


def cmd_record(args: argparse.Namespace) -> int:
    """Record Moat hop with velocity / shine-bar practice HUD burned in."""
    source = Path(args.source)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2
    try:
        hop_hold = _parse_hold(args.hop_hold)
        hop_coast = _parse_hold(args.hop_coast)
        unspin = _parse_hold(args.unspin)
        activate_hold = _parse_hold(args.activate)
        travel_hold = _parse_hold(args.travel) if args.travel else activate_hold
    except ValueError as exc:
        print(f"hold parse error: {exc}", file=sys.stderr)
        return 2

    video = Path(args.video) if args.video else DEFAULT_PRACTICE_VIDEO
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_west = Path(args.save_west) if args.save_west else None

    print(
        f"recording practice HUD → {video}  source={source.name}",
        flush=True,
    )
    rep = run_record_hop(
        source,
        video_path=video,
        store_frames=args.store_frames,
        stand=args.stand,
        micro_run=args.micro_run,
        hop_f=args.hop_f,
        hop_a_f=args.hop_a_f,
        hop_hold=hop_hold if hop_hold else ("RIGHT", "B", "A"),
        hop_coast_hold=hop_coast if hop_coast else ("RIGHT", "B"),
        unspin=unspin,
        unspin_f=args.unspin_f,
        activate_hold=activate_hold,
        travel_hold=travel_hold,
        activate_frames=args.activate_frames,
        retap_a_every=args.retap_a_every,
        door_open=not args.no_door_open,
        pad_end=args.pad_end,
        assist=not args.no_assist,
        scale=args.video_scale,
        save_west=save_west,
    )
    path = out_dir / "record_practice.json"
    # Keep JSON light
    light = {
        k: v
        for k, v in rep.items()
        if k not in ("log",)
    }
    if isinstance(light.get("travel"), dict):
        light["travel"] = {
            k: v
            for k, v in light["travel"].items()
            if k not in ("moat_samples",)
        }
    path.write_text(json.dumps(light, indent=2) + "\n")
    flag = (
        "GREEN"
        if rep.get("ok") is True
        else ("MOAT" if rep.get("ok") == "partial_moat" else "RED")
    )
    print(
        f"{flag} video={video} frames={rep.get('frames_written')} "
        f"ok={rep.get('ok')} err={rep.get('error')}"
    )
    print(f"wrote {path}")
    if rep.get("phase_marks"):
        print("phase marks:")
        for m in rep["phase_marks"]:
            print(
                f"  F{m['frame']:04d} {m['phase']:8s} "
                f"xy={m['xy']} vx={m['vx']} spark={m['spark_timer']} "
                f"| {m['cue']}"
            )
    return 0 if rep.get("ok") is True else 1


def cmd_hop(args: argparse.Namespace) -> int:
    """Store-first hop-unspin-activate measure / grid sweep."""
    source = Path(args.source)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        unspin = _parse_hold(args.unspin)
        activate_hold = _parse_hold(args.activate)
        travel_hold = _parse_hold(args.travel) if args.travel else activate_hold
    except ValueError as exc:
        print(f"hold parse error: {exc}", file=sys.stderr)
        return 2

    # --- grid sweep ---
    if args.sweep:
        # Format: stand=6,8,10;hop=12,13,14;unspin_f=3,4,5;run=0,2;travel=RIGHT+A,A,RIGHT+UP+A
        # or simple hop=12:16:1 style ranges
        grid = _parse_hop_sweep(args.sweep, defaults={
            "stand": [args.stand],
            "hop": [args.hop_f],
            "hop_a": [args.hop_a_f],
            "unspin_f": [args.unspin_f],
            "run": [args.micro_run],
            "unspin": [args.unspin],
            "activate": [args.activate],
            "travel": [args.travel or args.activate],
            "retap": [args.retap_a_every],
            "hop_hold": [args.hop_hold],
            "hop_coast": [args.hop_coast],
        })
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        n = 0
        for combo in _iter_hop_grid(grid):
            n += 1
            try:
                us = _parse_hold(combo["unspin"])
                ah = _parse_hold(combo["activate"])
                th = _parse_hold(combo["travel"])
                hh = _parse_hold(combo["hop_hold"])
                hc = _parse_hold(combo["hop_coast"])
            except ValueError as exc:
                print(f"skip bad combo {combo}: {exc}")
                continue
            rep = run_hop_once(
                source,
                store_frames=args.store_frames,
                stand=int(combo["stand"]),
                micro_run=int(combo["run"]),
                hop_f=int(combo["hop"]),
                hop_a_f=int(combo["hop_a"]),
                hop_hold=hh if hh else ("RIGHT", "B", "A"),
                hop_coast_hold=hc if hc else ("RIGHT", "B"),
                unspin=us,
                unspin_f=int(combo["unspin_f"]),
                activate_hold=ah,
                travel_hold=th,
                activate_frames=args.activate_frames,
                retap_a_every=int(combo["retap"]),
                door_open=not args.no_door_open,
                record_all=False,
                assist=not args.no_assist,
                save_west=Path(args.save_west) if args.save_west else None,
            )
            summary = _hop_summary(rep)
            # drop heavy samples from row unless --full-log
            if not args.full_log:
                samples = summary.get("moat_samples") or []
                summary["moat_sample_n"] = len(samples)
                summary["moat_samples_head"] = samples[:3]
                summary["moat_samples_tail"] = samples[-3:] if samples else []
                summary.pop("moat_samples", None)
            rows.append(summary)
            flag = (
                "GREEN"
                if summary["ok"] is True
                else ("MOAT" if summary["ok"] == "partial_moat" else "RED")
            )
            score = (
                1_000_000
                if summary["ok"] is True
                else (
                    100_000 + int(summary.get("moat_max_x") or 0)
                    if summary["ok"] == "partial_moat"
                    else int(summary.get("moat_max_x") or 0)
                )
            )
            if best is None or score > best["_score"]:
                best = {**summary, "_score": score}
            print(
                f"[{n}] {flag:5s} stand={combo['stand']} hop={combo['hop']} "
                f"a={combo['hop_a']} run={combo['run']} "
                f"us={combo['unspin']}@{combo['unspin_f']} "
                f"act={combo['activate']} trv={combo['travel']} retap={combo['retap']} "
                f"act_y={summary['activate_xy'][1]} "
                f"moat_x={summary['moat_max_x']} "
                f"room={summary['final_room']} xy={summary['final_xy']} "
                f"spark={summary['final_spark']}"
            )
            if summary["ok"] is True:
                print(f"  *** WEST OCEAN *** params={combo}")
                break  # first green is enough for this residual

        path = out_dir / "hop_sweep.json"
        payload = {
            "source": str(source),
            "n": len(rows),
            "rows": rows,
            "best": {k: v for k, v in (best or {}).items() if k != "_score"},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}")
        if best:
            print(
                f"BEST ok={best.get('ok')} moat_max_x={best.get('moat_max_x')} "
                f"params={best.get('params')} act_y={best.get('activate_xy')}"
            )
        good = [r for r in rows if r["ok"] is True]
        moat = [r for r in rows if r["ok"] == "partial_moat"]
        if good:
            return 0
        if moat:
            mx = max(int(r.get("moat_max_x") or 0) for r in moat)
            print(f"leeway: {len(moat)} partial Moat, best moat_max_x={mx}")
            return 1
        print("leeway: none reached Moat/West")
        return 1

    # --- single hop run ---
    save_west = Path(args.save_west) if args.save_west else (
        SCRATCH / "post_moat_west_ocean_spark.state"
    )
    try:
        hop_hold = _parse_hold(args.hop_hold)
        hop_coast = _parse_hold(args.hop_coast)
    except ValueError as exc:
        print(f"hold parse error: {exc}", file=sys.stderr)
        return 2
    rep = run_hop_once(
        source,
        store_frames=args.store_frames,
        stand=args.stand,
        micro_run=args.micro_run,
        hop_f=args.hop_f,
        hop_a_f=args.hop_a_f,
        hop_hold=hop_hold if hop_hold else ("RIGHT", "B", "A"),
        hop_coast_hold=hop_coast if hop_coast else ("RIGHT", "B"),
        unspin=unspin,
        unspin_f=args.unspin_f,
        activate_hold=activate_hold,
        travel_hold=travel_hold,
        activate_frames=args.activate_frames,
        retap_a_every=args.retap_a_every,
        door_open=not args.no_door_open,
        record_all=True,
        assist=not args.no_assist,
        save_west=save_west,  # only written inside on West green
    )
    summary = _hop_summary(rep)
    boost = (rep.get("charge") or {}).get("boost") or {}
    armed = (rep.get("store") or {}).get("armed") or {}
    hop = rep.get("hop") or {}
    travel = rep.get("travel") or {}
    print(
        f"boot  xy=({rep['boot']['x']},{rep['boot']['y']}) "
        f"room={rep['boot']['room_hex']}"
    )
    print(
        f"boost xy=({boost.get('x')},{boost.get('y')}) "
        f"echoes={boost.get('speed_echoes')}"
    )
    print(
        f"store armed=$0A68={armed.get('spark_timer')} "
        f"xy=({armed.get('x')},{armed.get('y')}) pose={armed.get('pose')}"
    )
    print(
        f"hop stand={args.stand} run={args.micro_run} hop_f={args.hop_f} "
        f"hop_a_f={args.hop_a_f} hold={args.hop_hold} coast={args.hop_coast} "
        f"unspin={args.unspin}@{args.unspin_f} act={args.activate} "
        f"trv={args.travel or args.activate} retap={args.retap_a_every}"
    )
    ah = hop.get("after_hop") or {}
    au = hop.get("after_unspin") or {}
    print(
        f"  after_hop   xy=({ah.get('x')},{ah.get('y')}) pose={ah.get('pose')} "
        f"spark={ah.get('spark_timer')}"
    )
    print(
        f"  after_unspin xy=({au.get('x')},{au.get('y')}) pose={au.get('pose')} "
        f"spark={au.get('spark_timer')}"
    )
    act = travel.get("activate") or {}
    print(
        f"  activate    xy=({act.get('x')},{act.get('y')}) pose={act.get('pose')} "
        f"spark={act.get('spark_timer')}"
    )
    fin = travel.get("final") or {}
    print(
        f"travel moat={travel.get('reached_moat')} west={travel.get('reached_west')} "
        f"moat_max_x={travel.get('moat_max_x')} moat_y=[{travel.get('moat_min_y')},"
        f"{travel.get('moat_max_y')}] frames={travel.get('moat_frames')}"
    )
    print(
        f"  final room={fin.get('room_hex')} xy=({fin.get('x')},{fin.get('y')}) "
        f"pose={fin.get('pose')} spark={fin.get('spark_timer')}"
    )
    # Print Moat $0A68 samples every 10f
    samples = travel.get("moat_samples") or []
    if samples:
        print(f"  moat $0A68 samples (n={len(samples)}):")
        for s in samples:
            print(
                f"    moat_f={s.get('moat_f'):4d} xy=({s['x']},{s['y']}) "
                f"pose={s['pose']} $0A68={s['spark_timer']} vx={s.get('vx')} "
                f"vy={s.get('vy')}"
            )
    if travel.get("spark_died_at"):
        d = travel["spark_died_at"]
        print(
            f"  spark_died_at moat_f={d.get('moat_f')} "
            f"xy=({d['x']},{d['y']}) pose={d['pose']}"
        )
    if rep.get("west_state"):
        print(f"  west state → {rep['west_state']}")
    flag = "GREEN" if rep.get("ok") is True else (
        "MOAT" if rep.get("ok") == "partial_moat" else "RED"
    )
    print(f"{flag} {rep.get('error') or ''}".rstrip())

    path = out_dir / "hop_measure.json"
    # Cap samples in written report already handled; write full summary+travel
    write_rep = {k: v for k, v in rep.items() if k != "log"}
    write_rep["summary"] = summary
    path.write_text(json.dumps(write_rep, indent=2) + "\n")
    print(f"wrote {path} (log_len={rep.get('log_len')})")
    return 0 if rep.get("ok") is True else 1


def _parse_range_or_list(spec: str) -> list:
    """Parse ``12,13,14`` or ``12:16:1`` into a list of ints or keep strings."""
    spec = spec.strip()
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        if len(parts) == 3:
            start, stop, step = (int(p) for p in parts)
            return list(range(start, stop + 1, step))
        if len(parts) == 2:
            start, stop = (int(p) for p in parts)
            return list(range(start, stop + 1))
    out: list = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            out.append(tok)
    return out


def _parse_hop_sweep(spec: str, defaults: dict[str, list]) -> dict[str, list]:
    """Parse ``stand=6,8;hop=12:14:1;travel=RIGHT+A,A,RIGHT+UP+A`` grid."""
    grid = {k: list(v) for k, v in defaults.items()}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"sweep chunk needs key=vals, got {chunk!r}")
        key, vals = chunk.split("=", 1)
        key = key.strip()
        # normalize aliases
        key = {
            "hop_f": "hop",
            "hop_a_f": "hop_a",
            "micro_run": "run",
            "retap_a": "retap",
            "retap_a_every": "retap",
            "act": "activate",
            "trv": "travel",
            "us": "unspin",
            "us_f": "unspin_f",
            "hop_coast_hold": "hop_coast",
        }.get(key, key)
        if key not in grid:
            raise ValueError(
                f"unknown sweep key {key!r}; "
                f"known={sorted(grid)}"
            )
        grid[key] = _parse_range_or_list(vals)
    return grid


def _iter_hop_grid(grid: dict[str, list]):
    """Cartesian product over hop sweep grid (deterministic order)."""
    keys = [
        "stand",
        "hop",
        "hop_a",
        "run",
        "unspin_f",
        "unspin",
        "activate",
        "travel",
        "retap",
        "hop_hold",
        "hop_coast",
    ]
    lists = [grid[k] for k in keys]
    from itertools import product

    for vals in product(*lists):
        yield dict(zip(keys, vals))


def run_measure_once(
    source: Path,
    *,
    store_frames: int = 18,
    idle_after_store: int = 0,
    hold_down_idle: bool = True,
    do_spark: bool = True,
    save_store: Path | None = None,
    record_all: bool = True,
    assist: bool = True,
) -> dict[str, Any]:
    env, sess = boot_env(source, assist=assist)
    report: dict[str, Any] = {
        "source": str(source),
        "boot": snap(env, 0),
        "params": {
            "store_frames": store_frames,
            "idle_after_store": idle_after_store,
            "hold_down_idle": hold_down_idle,
            "do_spark": do_spark,
        },
    }
    try:
        report["charge"] = _charge_until_boost(sess, record=record_all)
        if not report["charge"]["boost"].get("speed_boosting"):
            report["ok"] = False
            report["error"] = "never reached speed_boosting (echoes≥4)"
            report["final"] = snap(env, sess.frame)
            return report

        report["store"] = _crouch_store(
            sess, store_frames=store_frames, record=record_all
        )
        armed = report["store"]["armed"]
        if armed is None:
            report["ok"] = False
            report["error"] = (
                f"store never armed $0A68 "
                f"(peak={report['store']['peak_timer_during_store']})"
            )
            report["final"] = snap(env, sess.frame)
            return report

        if save_store is not None:
            save_dev_state(env, save_store)
            report["store_state"] = str(save_store)

        if idle_after_store > 0:
            report["window"] = _idle_window(
                sess,
                frames=idle_after_store,
                hold_down=hold_down_idle,
                record=record_all,
            )
        else:
            # Still sample 1f to record armed timer value post-store
            report["window"] = {
                "requested_frames": 0,
                "frames_timer_gt0": 1 if armed["spark_timer"] > 0 else 0,
                "timer_start": armed["spark_timer"],
                "timer_end": armed["spark_timer"],
            }

        if do_spark:
            # If idle burned the timer, re-charge is out of scope — just try activate
            report["travel"] = _activate_and_travel(sess, record=record_all)
            report["ok"] = bool(report["travel"]["reached_west"])
            if not report["ok"] and report["travel"]["reached_moat"]:
                report["ok"] = "partial_moat"
            report["error"] = None if report["ok"] is True else (
                f"stopped room={report['travel']['final']['room_hex']} "
                f"xy=({report['travel']['final']['x']},{report['travel']['final']['y']}) "
                f"spark={report['travel']['final']['spark_timer']}"
            )
        else:
            report["ok"] = True
            report["final"] = snap(env, sess.frame)

        report["log_len"] = len(sess.log)
        report["log"] = sess.log
        return report
    finally:
        env.close()


def cmd_measure(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        # start:stop:step  e.g. 0:90:5
        parts = args.sweep.split(":")
        if len(parts) != 3:
            print("--sweep needs start:stop:step", file=sys.stderr)
            return 2
        start, stop, step = (int(p) for p in parts)
        rows = []
        for idle in range(start, stop + 1, step):
            save = None
            if args.save_store and idle == start:
                save = Path(args.save_store)
            rep = run_measure_once(
                source,
                store_frames=args.store_frames,
                idle_after_store=idle,
                hold_down_idle=not args.idle_neutral,
                do_spark=not args.no_spark,
                save_store=save,
                record_all=args.full_log,
                assist=not args.no_assist,
            )
            # Drop heavy log for sweep summary
            summary = {
                "idle_after_store": idle,
                "ok": rep.get("ok"),
                "error": rep.get("error"),
                "boost_x": (rep.get("charge") or {}).get("boost", {}).get("x"),
                "boost_echoes": (rep.get("charge") or {}).get("boost", {}).get(
                    "speed_echoes"
                ),
                "armed_timer": ((rep.get("store") or {}).get("armed") or {}).get(
                    "spark_timer"
                ),
                "armed_x": ((rep.get("store") or {}).get("armed") or {}).get("x"),
                "window_timer_start": (rep.get("window") or {}).get("timer_start"),
                "window_timer_end": (rep.get("window") or {}).get("timer_end"),
                "window_gt0": (rep.get("window") or {}).get("frames_timer_gt0"),
                "reached_moat": (rep.get("travel") or {}).get("reached_moat"),
                "reached_west": (rep.get("travel") or {}).get("reached_west"),
                "final_room": (rep.get("travel") or {}).get("final", {}).get(
                    "room_hex"
                )
                or (rep.get("final") or {}).get("room_hex"),
                "final_xy": (
                    (rep.get("travel") or {}).get("final", {}).get("x"),
                    (rep.get("travel") or {}).get("final", {}).get("y"),
                ),
            }
            rows.append(summary)
            flag = (
                "GREEN"
                if summary["ok"] is True
                else ("MOAT" if summary["ok"] == "partial_moat" else "RED")
            )
            print(
                f"idle={idle:3d} {flag:5s} armed_t={summary['armed_timer']} "
                f"win={summary['window_timer_start']}→{summary['window_timer_end']} "
                f"gt0={summary['window_gt0']} "
                f"room={summary['final_room']} xy={summary['final_xy']}"
            )
        path = out_dir / "measure_sweep.json"
        path.write_text(json.dumps({"source": str(source), "rows": rows}, indent=2) + "\n")
        print(f"wrote {path}")
        # Print leeway: max idle that still reaches west
        good = [r["idle_after_store"] for r in rows if r["ok"] is True]
        moat = [
            r["idle_after_store"]
            for r in rows
            if r["ok"] is True or r["ok"] == "partial_moat"
        ]
        if good:
            print(f"leeway West Ocean: idle_after_store ≤ {max(good)}f (of tested)")
        elif moat:
            print(f"leeway Moat only: idle_after_store ≤ {max(moat)}f (of tested)")
        else:
            print("leeway: none of the sweep reached Moat/West")
        return 0 if good else 1

    save = Path(args.save_store) if args.save_store else (
        SCRATCH / "pre_moat_shine_store.state"
    )
    rep = run_measure_once(
        source,
        store_frames=args.store_frames,
        idle_after_store=args.idle,
        hold_down_idle=not args.idle_neutral,
        do_spark=not args.no_spark,
        save_store=save,
        record_all=True,
        assist=not args.no_assist,
    )
    # Compact stdout
    armed = (rep.get("store") or {}).get("armed") or {}
    boost = (rep.get("charge") or {}).get("boost") or {}
    win = rep.get("window") or {}
    travel = rep.get("travel") or {}
    print(
        f"boot  xy=({rep['boot']['x']},{rep['boot']['y']}) "
        f"room={rep['boot']['room_hex']}"
    )
    print(
        f"boost xy=({boost.get('x')},{boost.get('y')}) "
        f"echoes={boost.get('speed_echoes')} sc_word={boost.get('speed_counter_word')} "
        f"flag={boost.get('speed_flag')}"
    )
    print(
        f"store armed=$0A68={armed.get('spark_timer')} "
        f"xy=({armed.get('x')},{armed.get('y')}) pose={armed.get('pose')} "
        f"at_frame={armed.get('frame')} idx={armed.get('store_frame_index')}"
    )
    if save:
        print(f"store state → {save}")
    print(
        f"window idle={args.idle}f hold_down={not args.idle_neutral} "
        f"timer {win.get('timer_start')}→{win.get('timer_end')} "
        f"gt0_frames={win.get('frames_timer_gt0')}"
    )
    if travel:
        fin = travel.get("final") or {}
        print(
            f"travel moat={travel.get('reached_moat')} west={travel.get('reached_west')} "
            f"final room={fin.get('room_hex')} xy=({fin.get('x')},{fin.get('y')}) "
            f"spark={fin.get('spark_timer')}"
        )
        if travel.get("moat_entry"):
            m = travel["moat_entry"]
            print(f"  moat entry xy=({m['x']},{m['y']}) spark={m['spark_timer']}")
        if travel.get("west_ocean"):
            w = travel["west_ocean"]
            print(f"  west ocean xy=({w['x']},{w['y']})")
    flag = "GREEN" if rep.get("ok") is True else (
        "MOAT" if rep.get("ok") == "partial_moat" else "RED"
    )
    print(f"{flag} {rep.get('error') or ''}".rstrip())

    path = out_dir / "measure.json"
    # Keep log but cap if huge
    path.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"wrote {path} (log_len={rep.get('log_len')})")
    return 0 if rep.get("ok") is True else 1


# ---------------------------------------------------------------------------
# watch / pure / human (existing live paths)
# ---------------------------------------------------------------------------


class _WatchSession(_Sess):
    """Live bot session (same as before)."""

    def live_action(self) -> list[int]:
        if self.done:
            return idle_action()

        st = parse_env_state(self.env, frame=self.frame, mode="nav")
        if self.assist is not None:
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        st = self.state
        self._live_i += 1

        if st.room_id == ROOM_WEST_OCEAN and st.door_transition == 0:
            self.done = True
            self.success = True
            self.phase = "done"
            self.detail = "West Ocean"
            return idle_action()

        if (
            self._live_stage == "clear"
            and st.room_id != ROOM_KIHUNTER
            and st.door_transition == 0
        ):
            self.done = True
            self.error = f"left Kihunter into 0x{st.room_id:04X}"
            self.phase = "fail"
            return idle_action()

        if st.room_id not in (ROOM_KIHUNTER, ROOM_MOAT, ROOM_WEST_OCEAN):
            self.done = True
            self.error = f"unexpected room 0x{st.room_id:04X}"
            self.phase = "fail"
            return idle_action()

        if st.room_id == ROOM_KIHUNTER and self._live_stage in (
            "clear",
            "runway",
            "charge",
            "store",
        ):
            if near_left_door(st) or st.samus_x < _LEFT_DOOR_LIP_X:
                self.detail = f"AVOID TUBE door x={st.samus_x}"
                return buttons("RIGHT")
            if self._live_stage != "charge" and st.samus_x > 720:
                if self._live_stage in ("clear", "runway"):
                    self.detail = f"AVOID MOAT door x={st.samus_x}"
                    return buttons("LEFT")

        if is_knockback(st):
            self.detail = "knockback"
            if self._live_stage in ("clear", "runway"):
                if st.samus_x < _WALK_STOP_X:
                    return buttons("RIGHT", "B")
                return buttons("LEFT", "B", "X")
            return buttons(self._live_dir, "B", "X")

        enemies = list_air_enemies(self.env)

        # Skip clear if already clean and on left side
        if self._live_stage == "clear" and not enemies and st.samus_x < 250:
            self._live_stage = "runway"
            self.detail = "already clear → runway"

        if self._live_stage == "clear":
            self.phase = "clear"
            self._live_dir = "LEFT"
            if not enemies:
                self._clear_ok = getattr(self, "_clear_ok", 0) + 1
                if self._clear_ok > 25:
                    self._live_stage = "runway"
                    self.detail = "cleared → runway"
                    self._live_i = 0
                    self._plant_face = None
                    return idle_action()
                return idle_action()
            self._clear_ok = 0
            self.detail = (
                f"walk LEFT + shoot  n={len(enemies)} x={st.samus_x} "
                f"ids={[hex(e[4]) for e in enemies]}"
            )
            if st.samus_x <= _WALK_STOP_X:
                mean_x = sum(e[1] for e in enemies) / len(enemies)
                want = "RIGHT" if mean_x >= st.samus_x - 8 else "LEFT"
                if want == "LEFT" and st.samus_x < _LEFT_DOOR_LIP_X + 40:
                    want = "RIGHT"
                prev = getattr(self, "_plant_face", None)
                if prev != want or (self._live_i % 50) < 2:
                    self._plant_face = want
                    if (self._live_i % 50) < 2:
                        return buttons(want)
                pulse = self._live_i % 32
                if pulse < 26:
                    return buttons("X")
                return idle_action()
            self._plant_face = None
            pulse = self._live_i % 30
            if pulse < 24:
                return buttons("LEFT", "X")
            if pulse < 26:
                return buttons("LEFT")
            return buttons("LEFT", "X")

        if self._live_stage == "runway":
            self.phase = "runway"
            self.detail = f"runway band x={st.samus_x}"
            if st.samus_x < 140:
                # Already left of band (user pin x≈39) — face right and charge
                if st.samus_x < 90:
                    return buttons("RIGHT")
                self._live_stage = "charge"
                self._live_i = 0
                self.detail = "charge from left pin"
                return buttons("RIGHT")
            if (
                140 <= st.samus_x <= 200
                and st.samus_y <= 180
                and st.velocity_y == 0
            ):
                self._live_stage = "charge"
                self._live_i = 0
                self.detail = "charge start"
                return buttons("RIGHT")
            return buttons("LEFT", "B")

        if self._live_stage == "charge":
            self.phase = "charge"
            w = spark_wram(self.env)
            self.detail = (
                f"sc={w['speed_echoes']} word={w['speed_counter_word']} "
                f"boost={st.speed_boosting} spark={w['spark_timer']} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
            if st.room_id == ROOM_MOAT:
                self._live_stage = "store"
                self._store_left = 18
                return buttons("DOWN")
            if st.speed_boosting and st.velocity_y == 0:
                self._boost_seen = True
                self._live_stage = "store"
                self._store_left = 18
                self.detail = "STORE"
                return buttons("DOWN")
            x = st.samus_x
            if 545 <= x <= 575 and st.velocity_y == 0:
                return buttons("RIGHT", "B", "A")
            return buttons("RIGHT", "B")

        if self._live_stage == "store":
            self.phase = "store"
            w = spark_wram(self.env)
            self.detail = (
                f"store left={self._store_left} $0A68={w['spark_timer']} "
                f"sc={w['speed_echoes']}"
            )
            if self._store_left > 0:
                self._store_left -= 1
                return buttons("DOWN")
            self._live_stage = "spark"
            self._spark_left = 450
            self.detail = "SPARK (A=jump)"
            return buttons("RIGHT", "A")

        if self._live_stage == "spark":
            self.phase = "spark"
            w = spark_wram(self.env)
            self.detail = (
                f"$0A68={w['spark_timer']} pose={st.pose} room=0x{st.room_id:04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
            self._spark_left -= 1
            if self._spark_left <= 0:
                self.done = True
                self.error = "spark travel timeout"
                self.phase = "fail"
                return idle_action()
            return buttons("RIGHT", "A")

        return idle_action()


def _make_bot(watch: _WatchSession):
    def bot(_obs, _info):
        act = watch.live_action()
        watch.frame += 1
        return act

    bot.mission_status = lambda: (  # type: ignore[attr-defined]
        f"{watch.phase} | {watch.detail}"
        + (" | OK" if watch.success else "")
        + (f" | ERR {watch.error}" if watch.error else "")
    )
    return bot


def cmd_watch(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist() if not args.no_assist else None
    state_bytes = read_state_bytes(source)
    watch = _WatchSession(env, assist)

    def on_hud(_info):
        st = watch.state
        w = spark_wram(env)
        enemies = list_air_enemies(env) if not watch.done else []
        return [
            f"phase={watch.phase}  {watch.detail}",
            f"room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) p={st.pose}",
            (
                f"$0B3E echoes={w['speed_echoes']} word={w['speed_counter_word']} "
                f"flag={w['speed_flag']}  $0A68 spark={w['spark_timer']}"
            ),
            f"air={len(enemies)} frame={watch.frame}",
            (
                "SUCCESS → West Ocean"
                if watch.success
                else (watch.error or "clear → runway → charge → store → spark")
            ),
        ]

    if args.headless or args.mode == "pure":
        env.reset()
        env.em.set_state(state_bytes)
        for _ in range(10):
            env.step(idle_action())
            if assist is not None:
                st = parse_env_state(env, mode="nav")
                assist.apply(env, st)
        watch.state = parse_env_state(env, mode="nav")
        watch.frame = 0
        try:
            # If already clear on left, skip clear
            skip = (
                not list_air_enemies(env)
                and watch.state.room_id == ROOM_KIHUNTER
                and watch.state.samus_x < 250
            )
            play_moat_shinespark(watch, skip_clear=skip)
            print(
                f"GREEN room=0x{watch.state.room_id:04X} "
                f"xy=({watch.state.samus_x},{watch.state.samus_y}) frames={watch.frame}"
            )
            out = SCRATCH / "post_moat_west_ocean_spark.state"
            save_dev_state(env, out)
            print(f"saved {out}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"RED {exc}")
            print(
                f"pin room=0x{watch.state.room_id:04X} "
                f"xy=({watch.state.samus_x},{watch.state.samus_y}) "
                f"sc={watch.state.speed_counter} spark={watch.state.shinespark_timer}"
            )
            print(f"enemies={list_air_enemies(env)} wram={spark_wram(env)}")
            return 1
        finally:
            env.close()

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        e.em.set_state(state_bytes)
        for _ in range(10):
            obs, *_rest, info = step_env(e, idle_action())
            if assist is not None:
                st = parse_env_state(e, mode="nav")
                assist.apply(e, st)
        watch.state = parse_env_state(e, mode="nav")
        watch.frame = 0
        # Skip clear stage if pin is already clean
        if not list_air_enemies(e) and watch.state.samus_x < 250:
            watch._live_stage = "runway"
        print(
            f"[BOOT] room=0x{watch.state.room_id:04X} "
            f"xy=({watch.state.samus_x},{watch.state.samus_y}) from {source.name}"
        )
        print("[BOT] charge/store/spark — HUD shows $0B3E / $0A68")
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]
    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title="Moat shinespark watch",
            bot=_make_bot(watch),
            action_size=12,
            base_fps=60,
            initial_speed=args.speed,
            headless=False,
        )
        session.on_hud = on_hud
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    if watch.success:
        print(f"SUCCESS frames≈{watch.frame}")
        return 0
    print(f"ENDED phase={watch.phase} err={watch.error} frame={watch.frame}")
    return 1


def cmd_human(args: argparse.Namespace) -> int:
    """Human play with WRAM HUD; F5 saves --out."""
    source = Path(args.source)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2
    out = Path(args.out)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist() if not args.no_assist else None
    state_bytes = read_state_bytes(source)
    saved = {"ok": False}
    pin: dict[str, Any] = {}

    def refresh():
        st = parse_env_state(env, mode="nav")
        if assist is not None:
            try:
                assist.apply(env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    assist.apply(env, st)
                except Exception:  # noqa: BLE001
                    pass
            st = parse_env_state(env, mode="nav")
        pin["st"] = st
        pin["w"] = spark_wram(env)
        return st

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        e.em.set_state(state_bytes)
        for _ in range(10):
            obs, *_rest, info = step_env(e, idle_action())
            if assist is not None:
                st = parse_env_state(e, mode="nav")
                try:
                    assist.apply(e.data, st)
                except Exception:  # noqa: BLE001
                    try:
                        assist.apply(e, st)
                    except Exception:  # noqa: BLE001
                        pass
        refresh()
        st = pin["st"]
        print(
            f"[BOOT] room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"from {source.name}"
        )
        print(f"[HUMAN] F5 saves → {out}")
        print("        run RIGHT+B dash, DOWN store, watch $0A68, RIGHT+A spark")
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]

    def on_hud(_info):
        st = pin.get("st") or parse_env_state(env, mode="nav")
        w = pin.get("w") or spark_wram(env)
        return [
            "HUMAN spark lab  |  F5 save pin  ·  ESC quit",
            f"room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) p={st.pose}",
            (
                f"$0B3E echoes={w['speed_echoes']} word={w['speed_counter_word']} "
                f"flag={w['speed_flag']}"
            ),
            (
                f"$0A68 spark_timer={w['spark_timer']}  "
                f"boost={st.speed_boosting} shining={st.shinesparking}"
            ),
            f"out → {out.name}",
        ]

    def on_step(_o, _r, _d, _i):
        refresh()

    def on_key_down(key: int) -> bool:
        try:
            import pygame as pg
        except ImportError:
            return False
        if key != pg.K_F5:
            return False
        st = refresh()
        w = pin["w"]
        save_dev_state(env, out)
        print(
            f"[SAVED] {out}\n"
            f"  xy=({st.samus_x},{st.samus_y}) pose={st.pose} "
            f"$0A68={w['spark_timer']} echoes={w['speed_echoes']}"
        )
        saved["ok"] = True
        return True

    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title="Moat spark human lab",
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
    return 0 if saved["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--source",
            type=Path,
            default=None,
            help="Start state (default: first existing runway/human pin)",
        )
        p.add_argument("--no-assist", action="store_true")
        p.add_argument("--scale", type=int, default=3)
        p.add_argument("--speed", type=float, default=1.0)

    p_watch = sub.add_parser("watch", help="Live bot window")
    add_common(p_watch)
    p_watch.add_argument(
        "--headless",
        action="store_true",
        help="No window; run controller once",
    )
    p_watch.set_defaults(func=cmd_watch)

    p_pure = sub.add_parser("pure", help="Headless controller once")
    add_common(p_pure)
    p_pure.add_argument("--headless", action="store_true", default=True)
    p_pure.set_defaults(func=cmd_watch, headless=True)

    p_human = sub.add_parser("human", help="Human play + WRAM HUD + F5 save")
    add_common(p_human)
    p_human.add_argument(
        "--out",
        type=Path,
        default=SCRATCH / "moat_spark_human_pin.state",
        help="F5 save path",
    )
    p_human.set_defaults(func=cmd_human)

    p_m = sub.add_parser(
        "measure",
        help="Charge→store WRAM window (+ optional mid-door spark)",
    )
    add_common(p_m)
    p_m.add_argument(
        "--store-frames",
        type=int,
        default=18,
        help="DOWN frames for crouch-store",
    )
    p_m.add_argument(
        "--idle",
        type=int,
        default=0,
        help="Frames to wait after store before activate (window probe)",
    )
    p_m.add_argument(
        "--idle-neutral",
        action="store_true",
        help="During idle, release DOWN (default: hold DOWN)",
    )
    p_m.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Sweep idle delays start:stop:step e.g. 0:90:5",
    )
    p_m.add_argument(
        "--no-spark",
        action="store_true",
        help="Only charge+store (+idle); do not activate/travel",
    )
    p_m.add_argument(
        "--save-store",
        type=Path,
        default=None,
        help="Where to write armed-store state (default scratch/pre_moat_shine_store.state)",
    )
    p_m.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    p_m.add_argument(
        "--full-log",
        action="store_true",
        help="Keep per-frame log in sweep rows (heavy)",
    )
    p_m.set_defaults(func=cmd_measure)

    p_h = sub.add_parser(
        "hop",
        help="Store→stand→spin-hop→unspin→activate (Moat carry probe)",
    )
    add_common(p_h)
    p_h.add_argument("--store-frames", type=int, default=18)
    p_h.add_argument(
        "--stand",
        type=int,
        default=4,
        help="Idle frames after store (verified green band: 4–8)",
    )
    p_h.add_argument(
        "--micro-run",
        type=int,
        default=2,
        help="RIGHT+B frames before hop (need ≥2 to leave crouch pose 39)",
    )
    p_h.add_argument(
        "--hop-f",
        type=int,
        default=14,
        help="Total hop frames (13–16 → Moat y≈115–122, max x≈475)",
    )
    p_h.add_argument(
        "--hop-a-f",
        type=int,
        default=-1,
        help="Frames of hop_hold (w/ A) at hop start; -1 = all hop_f",
    )
    p_h.add_argument(
        "--hop-hold",
        type=str,
        default="RIGHT+B+A",
        help="Buttons during hop A-phase",
    )
    p_h.add_argument(
        "--hop-coast",
        type=str,
        default="RIGHT+B",
        help="Buttons after hop A-phase (coast over wall)",
    )
    p_h.add_argument(
        "--unspin",
        type=str,
        default="UP",
        help="Unspin hold e.g. UP, X, L, UP+X",
    )
    p_h.add_argument("--unspin-f", type=int, default=3, help="Unspin hold frames")
    p_h.add_argument(
        "--activate",
        type=str,
        default="RIGHT+A",
        help="Activate hold e.g. RIGHT+A, A, RIGHT+UP+A",
    )
    p_h.add_argument(
        "--travel",
        type=str,
        default=None,
        help="Travel hold (default=activate). e.g. RIGHT, A, RIGHT+UP+A",
    )
    p_h.add_argument(
        "--activate-frames",
        type=int,
        default=16,
        help="Frames of activate hold before pure travel loop",
    )
    p_h.add_argument(
        "--retap-a-every",
        type=int,
        default=0,
        help="If >0, 1f A-only press every N Moat frames",
    )
    p_h.add_argument(
        "--no-door-open",
        action="store_true",
        help="Disable post-spark RIGHT+X blue-door open (default: on)",
    )
    p_h.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            "Grid: stand=6,8,10;hop=12:14:1;unspin_f=3,4,5;"
            "run=0,2;travel=RIGHT+A,A,RIGHT+UP+A;unspin=UP,X;retap=0,20"
        ),
    )
    p_h.add_argument(
        "--save-west",
        type=Path,
        default=None,
        help="On West green, save state here "
        f"(default single-run: {SCRATCH / 'post_moat_west_ocean_spark.state'})",
    )
    p_h.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    p_h.add_argument(
        "--full-log",
        action="store_true",
        help="Keep full moat_samples in sweep JSON",
    )
    p_h.set_defaults(func=cmd_hop)

    p_r = sub.add_parser(
        "record",
        help="Write practice-HUD MP4 (vx/Δv, shine bar, phase cues)",
    )
    add_common(p_r)
    p_r.add_argument("--store-frames", type=int, default=18)
    p_r.add_argument("--stand", type=int, default=4)
    p_r.add_argument("--micro-run", type=int, default=2)
    p_r.add_argument("--hop-f", type=int, default=14)
    p_r.add_argument("--hop-a-f", type=int, default=-1)
    p_r.add_argument("--hop-hold", type=str, default="RIGHT+B+A")
    p_r.add_argument("--hop-coast", type=str, default="RIGHT+B")
    p_r.add_argument("--unspin", type=str, default="UP")
    p_r.add_argument("--unspin-f", type=int, default=3)
    p_r.add_argument("--activate", type=str, default="RIGHT+A")
    p_r.add_argument("--travel", type=str, default=None)
    p_r.add_argument("--activate-frames", type=int, default=16)
    p_r.add_argument("--retap-a-every", type=int, default=0)
    p_r.add_argument("--no-door-open", action="store_true")
    p_r.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_PRACTICE_VIDEO,
        help=f"Output mp4 (default: {DEFAULT_PRACTICE_VIDEO})",
    )
    p_r.add_argument("--video-scale", type=int, default=2)
    p_r.add_argument("--pad-end", type=int, default=45, help="Idle frames after West")
    p_r.add_argument("--save-west", type=Path, default=None)
    p_r.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p_r.set_defaults(func=cmd_record)

    return ap


def main(argv: list[str] | None = None) -> int:
    # Backward-compat: bare flags without subcommand → watch
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] not in (
        "watch",
        "pure",
        "human",
        "measure",
        "hop",
        "record",
        "-h",
        "--help",
    ):
        raw = ["watch", *raw]
    elif not raw:
        raw = ["watch"]

    ap = build_parser()
    args = ap.parse_args(raw)
    if getattr(args, "source", None) is None:
        args.source = default_source()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
