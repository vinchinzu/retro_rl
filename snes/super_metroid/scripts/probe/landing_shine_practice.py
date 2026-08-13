#!/usr/bin/env python3
"""Landing Site shinespark practice — dual-track charge/store/activate windows.

Not continuous evidence. Default pin is a Speed-equipped Landing Site runway
under ``scratch/``, reached by walking **left a few rooms** from the pre-Moat
Kihunter pin (same loadout as Moat spark work — **not** Parlor, **not** escape):

  ``scratch/post_kihunter_pre_moat_spark.state``  (room ``0x948C``, items ``0x3105``)
    → step off left lip, shoot if needed
    → LEFT door → intermediate ``0x95D4``
    → LEFT → Landing Site ``0x91F8``
    → walk to bottom-floor runway for RIGHT charges / diagonal sparks

**Never** use ``dev_route_anchor_landing_site_finish`` (ship countdown / ``0xF32F``).
**Never** bootstrap via Parlor Torizo continuous — wrong side of the map.

```bash
# Charge → crouch-store → activate; dump thresholds JSON
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py measure

# Diagonal UP+RIGHT sparks
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py diagonal

# Re-bootstrap: pre-moat → left few rooms → Landing Site
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py bootstrap

# Alias of bootstrap
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py route

# Record diagonal shine proof video (no room-exit success criteria)
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py record-diagonal
```

Harness: **B**=dash charge, **A**=jump/shine activate, DOWN=store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from retro_harness.video import VideoCaptureConfig, VideoRecorder  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import place_samus, save_dev_state  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import (  # noqa: E402
    ADDR_EVENT_FLAGS,
    ADDR_TIMER_TYPE,
    EVENT_MOTHER_BRAIN_DEFEATED,
    parse_env_state,
    read_bank7e_wram,
    write_wram_u16,
)
from super_metroid.routes.skills import shinespark as spark  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
ROOM_LANDING = 0x91F8
ROOM_KIHUNTER = 0x948C  # pre-Moat Crateria Kihunter
ROOM_WEST_HALL = 0x95D4  # one room left of Kihunter → Landing Site
DEFAULT_PIN = SCRATCH / "landing_site_speed_practice.state"
# Pre-Moat Speed pin (same as Moat residual) — walk left a few rooms to LS.
PRE_MOAT_SOURCE = SCRATCH / "post_kihunter_pre_moat_spark.state"
# After natural LS entry (right side ~x2200), free-place onto bottom runway.
# Room load is already non-escape; place only sets charge geometry.
RUNWAY_X = 900
RUNWAY_Y_DROP = 900
DEFAULT_REPORT_DIR = Path("snes/super_metroid/debug/landing_shine")
DEFAULT_DIAGONAL_VIDEO = Path(
    "snes/super_metroid/recordings/landing_site_diagonal_shine_proof.mp4"
)

# ---------------------------------------------------------------------------
# Minimal session (ControllerSession + env for spark WRAM)
# ---------------------------------------------------------------------------


class _Sess:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist | None):
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="nav")
        self.log: list[dict[str, Any]] = []

    def step(self, action: Any, reason: str = "") -> Any:
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
            row = spark.spark_snapshot(self.env, self.frame)
            row["reason"] = reason
            self.log.append(row)
        return self.state

    def hold(self, n: int, *btns: str, reason: str = "") -> Any:
        act = buttons(*btns) if btns else idle_action()
        for _ in range(n):
            self.step(act, reason=reason)
        return self.state


def boot_env(source: Path, *, assist: bool = True) -> tuple[Any, _Sess]:
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


def _escape_snapshot(env: Any, st: Any | None = None) -> dict[str, Any]:
    """Escape timer + MB event bit for verify paste."""
    if st is None:
        st = parse_env_state(env, mode="nav")
    byte_index = EVENT_MOTHER_BRAIN_DEFEATED >> 3
    bit = 1 << (EVENT_MOTHER_BRAIN_DEFEATED & 7)
    addr = ADDR_EVENT_FLAGS + byte_index
    event_byte = int(read_bank7e_wram(env)[addr])
    ram = env.get_ram()
    return {
        "timer_type": int(ram[ADDR_TIMER_TYPE]),
        "escape_min": int(st.escape_timer_minutes),
        "escape_sec": int(st.escape_timer_seconds),
        "escape_frames": int(st.escape_timer_frames),
        "event_0e_mb_dead": bool(event_byte & bit),
        "event_byte": event_byte,
        "escape_active": bool(
            int(ram[ADDR_TIMER_TYPE]) != 0
            or (event_byte & bit)
        ),
    }


def _pin_is_clean(env: Any, st: Any) -> bool:
    """True when LS + Speed + mid-route loadout + no escape."""
    if st.room_id != ROOM_LANDING:
        return False
    if not (st.equipped_items & 0x2000):
        return False
    # Reject corrupted escape-finish full loadout.
    if st.equipped_items == 0xF32F or st.collected_items == 0xF32F:
        return False
    esc = _escape_snapshot(env, st)
    return not esc["escape_active"]


def _hold_env(env: Any, n: int, *btns: str, assist: UnlimitedResourcesAssist | None = None):
    act = buttons(*btns) if btns else idle_action()
    for _ in range(n):
        env.step(act)
        if assist is not None:
            assist.apply(env, parse_env_state(env, mode="nav"))
    return parse_env_state(env, mode="nav")


def bootstrap_practice_pin(
    out: Path = DEFAULT_PIN,
    *,
    source: Path = PRE_MOAT_SOURCE,
    runway_x: int = RUNWAY_X,
    runway_y_drop: int = RUNWAY_Y_DROP,
) -> dict[str, Any]:
    """Walk left from pre-Moat Kihunter to Landing Site (dual-track).

    Route recipe (measured)
    -----------------------
    1. Load ``post_kihunter_pre_moat_spark.state`` (``0x948C``, items ``0x3105``, Speed).
    2. Step RIGHT off the left door lip, shoot, hold LEFT.
    3. Room ``0x95D4`` (one hall), continue LEFT → Landing Site ``0x91F8``.
    4. Free-place onto bottom-floor runway (~x900) for RIGHT charges (room is
       already natural LS load from the door — not escape-finish geometry).
    5. Face right; save pin.

    No Parlor. No item grants. No escape-finish anchor.
    """
    if not source.is_file():
        raise FileNotFoundError(f"missing pre-moat source: {source}")

    assist = UnlimitedResourcesAssist()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(source))
    for _ in range(10):
        env.step(idle_action())
        assist.apply(env, parse_env_state(env, mode="nav"))
    st0 = parse_env_state(env, mode="nav")
    if st0.room_id != ROOM_KIHUNTER:
        env.close()
        raise RuntimeError(
            f"pre-moat source not in Kihunter 0x948C: room=0x{st0.room_id:04X} "
            f"path={source}"
        )
    if not (st0.equipped_items & 0x2000):
        env.close()
        raise RuntimeError(
            f"pre-moat source missing Speed: items=0x{st0.equipped_items:04X}"
        )

    room_chain: list[dict[str, Any]] = [
        {
            "frame": 0,
            "room_hex": f"0x{st0.room_id:04X}",
            "x": st0.samus_x,
            "y": st0.samus_y,
        }
    ]
    # Off lip (pin starts x≈39 on left door), face + shoot, then LEFT through doors.
    _hold_env(env, 24, "RIGHT", assist=assist)
    _hold_env(env, 4, "LEFT", assist=assist)
    _hold_env(env, 16, "X", assist=assist)

    last_room = st0.room_id
    entry_xy: tuple[int, int] | None = None
    for f in range(2500):
        st = parse_env_state(env, mode="nav")
        assist.apply(env, st)
        if (
            st.room_id != last_room
            and st.door_transition == 0
            and st.game_state == 8
        ):
            room_chain.append(
                {
                    "frame": f,
                    "room_hex": f"0x{st.room_id:04X}",
                    "x": st.samus_x,
                    "y": st.samus_y,
                }
            )
            last_room = st.room_id
            if st.room_id == ROOM_LANDING:
                entry_xy = (st.samus_x, st.samus_y)
                break
        env.step(buttons("LEFT"))
    else:
        st = parse_env_state(env, mode="nav")
        env.close()
        raise TimeoutError(
            f"never reached Landing Site from pre-moat "
            f"(room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"chain={[r['room_hex'] for r in room_chain]})"
        )

    # Natural door lands ~right side (~x2264 y1163). Place mid runway for charge.
    place_samus(env, runway_x, runway_y_drop)
    write_wram_u16(env, 0x18AA, 0)
    write_wram_u16(env, 0x18A8, 0x400)
    for i in range(400):
        env.step(idle_action())
        st = parse_env_state(env, mode="nav")
        assist.apply(env, st)
        if st.velocity_y == 0 and st.samus_y > 1000 and i > 20:
            break

    for _ in range(6):
        env.step(buttons("RIGHT"))
        assist.apply(env, parse_env_state(env, mode="nav"))
    for _ in range(12):
        env.step(idle_action())
        assist.apply(env, parse_env_state(env, mode="nav"))

    st = parse_env_state(env, mode="nav")
    esc = _escape_snapshot(env, st)

    idle_rooms: set[int] = set()
    idle_gs: set[int] = set()
    for _ in range(120):
        env.step(idle_action())
        st_i = parse_env_state(env, mode="nav")
        assist.apply(env, st_i)
        idle_rooms.add(int(st_i.room_id))
        idle_gs.add(int(st_i.game_state))
    st = parse_env_state(env, mode="nav")
    esc = _escape_snapshot(env, st)

    save_dev_state(env, out)
    summary: dict[str, Any] = {
        "path": str(out),
        "room_hex": f"0x{st.room_id:04X}",
        "x": st.samus_x,
        "y": st.samus_y,
        "pose": st.pose,
        "facing": st.facing,
        "equipped_items_hex": f"0x{st.equipped_items:04X}",
        "collected_items_hex": f"0x{st.collected_items:04X}",
        "beams_hex": f"0x{st.equipped_beams:04X}",
        "speed_equipped": bool(st.equipped_items & 0x2000),
        "not_endgame_f32f": st.equipped_items != 0xF32F,
        "escape": esc,
        "idle120_rooms": sorted(f"0x{r:04X}" for r in idle_rooms),
        "idle120_game_states": sorted(idle_gs),
        "idle_stable": idle_rooms == {ROOM_LANDING} and 8 in idle_gs,
        "room_chain": room_chain,
        "natural_entry_xy": list(entry_xy) if entry_xy else None,
        "route": {
            "source": str(source),
            "source_room": f"0x{ROOM_KIHUNTER:04X}",
            "via_room": f"0x{ROOM_WEST_HALL:04X}",
            "dest_room": f"0x{ROOM_LANDING:04X}",
            "method": (
                "open-loop LEFT from pre-moat → natural LS door; "
                "then runway place for charge (no parlor, no grant, no escape finish)"
            ),
            "runway_place": [runway_x, runway_y_drop],
        },
        "development_only": True,
        "note": (
            "Landing Site from pre-Moat Kihunter by walking left a few rooms. "
            "Same Speed loadout as Moat pin (0x3105). Not escape-finish."
        ),
    }
    env.close()
    if st.room_id != ROOM_LANDING:
        raise RuntimeError(f"bootstrap failed: {summary}")
    return summary


def ensure_source(source: Path | None) -> Path:
    """Return a usable pin; bootstrap DEFAULT_PIN when missing / unclean."""
    if source is not None and source.is_file():
        env, sess = boot_env(source, assist=True)
        ok = _pin_is_clean(env, sess.state)
        env.close()
        if ok:
            return source
        # Caller forced a non-default path — still use it (measure reports room).
        if source != DEFAULT_PIN:
            return source
        # Default pin is corrupt (escape / endgame) — fall through to rebuild.

    if DEFAULT_PIN.is_file():
        env, sess = boot_env(DEFAULT_PIN, assist=True)
        ok = _pin_is_clean(env, sess.state)
        env.close()
        if ok:
            return DEFAULT_PIN

    print("bootstrapping clean Landing Site Speed practice pin …", flush=True)
    summary = bootstrap_practice_pin(DEFAULT_PIN)
    print(json.dumps(summary, indent=2), flush=True)
    return DEFAULT_PIN


def run_record_diagonal(
    source: Path,
    *,
    video_path: Path = DEFAULT_DIAGONAL_VIDEO,
    idle_after_store: int = 2,
    travel_budget: int = 250,
    pad_end: int = 60,
    assist: bool = True,
) -> dict[str, Any]:
    """Charge RIGHT → store → diagonal RIGHT+UP spark; write proof mp4.

    No room-exit success criteria — only that a spark pose is seen. Short
    neutral pad at the end for watchability.
    """
    env, sess = boot_env(source, assist=assist)
    boot = spark.spark_snapshot(env, 0)
    esc0 = _escape_snapshot(env, sess.state)
    obs = env.render()
    if obs is None:
        env.step(idle_action())
        obs = env.render()
    assert obs is not None

    config = VideoCaptureConfig(
        fps=60,
        scale=2,
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
    # Opening freeze.
    writer.write(
        obs,
        action=None,
        frame_index=0,
        room_id=int(sess.state.room_id),
    )

    orig_step = sess.step

    def _step_rec(action: Any, reason: str = "") -> Any:
        st = orig_step(action, reason=reason)
        frame = env.render()
        if frame is not None:
            writer.write(
                frame,
                action=action,
                frame_index=sess.frame,
                room_id=int(st.room_id),
            )
        return st

    sess.step = _step_rec  # type: ignore[method-assign]

    try:
        charge = spark.charge_until_boost(sess, "RIGHT")
        store = spark.crouch_store(sess)
        window = None
        if idle_after_store > 0:
            window = spark.wait_store_window(
                sess, idle_after_store, hold_down=False
            )
        act = spark.activate_shinespark(
            sess,
            "RIGHT",
            "UP",
            travel_budget=travel_budget,
            label="diag_vid",
        )
        if pad_end > 0:
            sess.hold(pad_end, reason="pad_end")
        final = spark.spark_snapshot(env, sess.frame)
        esc1 = _escape_snapshot(env, sess.state)
        report: dict[str, Any] = {
            "mode": "record-diagonal",
            "source": str(source),
            "video": str(video_path),
            "boot": boot,
            "boot_escape": esc0,
            "charge_ok": charge.get("ok"),
            "charge_frames": charge.get("frames"),
            "store_ok": store.get("ok"),
            "armed_timer": (store.get("armed") or {}).get("spark_timer"),
            "window": window,
            "spark_ok": act.get("ok"),
            "spark_pose_seen": act.get("spark_pose_seen"),
            "activate_pose": (act.get("activate") or {}).get("pose"),
            "min_y": act.get("min_y"),
            "max_x": act.get("max_x"),
            "height_gain": (
                (charge.get("boost") or {}).get("y", 0) - act.get("min_y", 0)
                if charge.get("boost") and act.get("min_y") is not None
                else None
            ),
            "final": final,
            "final_escape": esc1,
            "frames_written": writer.frames_written,
            "ok": bool(act.get("ok")),
        }
        return report
    finally:
        writer.close()
        env.close()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_measure(
    source: Path,
    *,
    direction: str = "RIGHT",
    store_frames: int = spark.DEFAULT_STORE_MAX_FRAMES,
    idle_after_store: int = 0,
    hold_down_idle: bool = False,
    aim: tuple[str, ...] = ("RIGHT",),
    travel_budget: int = 220,
    do_spark: bool = True,
    assist: bool = True,
) -> dict[str, Any]:
    env, sess = boot_env(source, assist=assist)
    report: dict[str, Any] = {
        "mode": "measure",
        "source": str(source),
        "boot": spark.spark_snapshot(env, 0),
        "params": {
            "direction": direction,
            "store_frames": store_frames,
            "idle_after_store": idle_after_store,
            "hold_down_idle": hold_down_idle,
            "aim": list(aim),
            "travel_budget": travel_budget,
            "do_spark": do_spark,
        },
        "constants": {
            "ECHOES_FULL": spark.ECHOES_FULL,
            "TYPICAL_ARM_TIMER": spark.TYPICAL_ARM_TIMER,
            "TYPICAL_CHARGE_FRAMES": spark.TYPICAL_CHARGE_FRAMES,
            "SPARK_POSES": sorted(spark.SPARK_POSES),
        },
    }
    try:
        if do_spark:
            pipe = spark.charge_store_activate(
                sess,
                direction=direction,  # type: ignore[arg-type]
                store_max_frames=store_frames,
                idle_after_store=idle_after_store,
                hold_down_idle=hold_down_idle,
                aim_buttons=aim,
                travel_budget=travel_budget,
                label="ls",
            )
            report.update(pipe)
        else:
            report["charge"] = spark.charge_until_boost(
                sess, direction=direction  # type: ignore[arg-type]
            )
            report["store"] = spark.crouch_store(
                sess, max_frames=store_frames
            )
            if idle_after_store > 0:
                report["window"] = spark.wait_store_window(
                    sess, idle_after_store, hold_down=hold_down_idle
                )
            report["ok"] = bool(
                report["charge"].get("ok") and report["store"].get("ok")
            )
        report["log_len"] = len(sess.log)
        # Cap log for disk
        report["log"] = sess.log[:: max(1, len(sess.log) // 200)] if sess.log else []
        report["final"] = spark.spark_snapshot(env, sess.frame)
        return report
    finally:
        env.close()


def run_diagonal(
    source: Path,
    *,
    variants: list[tuple[str, ...]] | None = None,
    idle_after_store: int = 2,
    travel_budget: int = 250,
    assist: bool = True,
) -> dict[str, Any]:
    """Try several aim recipes after store; log peak height / room exit."""
    if variants is None:
        # Aim only (activate adds A). Pre-stand UP is applied inside activate.
        variants = [
            ("RIGHT", "UP"),
            ("UP", "RIGHT"),
            ("UP",),
            ("RIGHT",),
            ("LEFT", "UP"),
        ]
    # Normalize: aim without A (activate adds A)
    cleaned: list[tuple[str, ...]] = []
    for v in variants:
        aim = tuple(b for b in v if b != "A")
        if aim not in cleaned:
            cleaned.append(aim)

    rows: list[dict[str, Any]] = []
    for aim in cleaned:
        env, sess = boot_env(source, assist=assist)
        try:
            charge = spark.charge_until_boost(sess, "RIGHT")
            store = spark.crouch_store(sess)
            window = None
            if idle_after_store > 0:
                window = spark.wait_store_window(
                    sess, idle_after_store, hold_down=False
                )
            act = spark.activate_shinespark(
                sess,
                *aim,
                travel_budget=travel_budget,
                label="diag",
            )
            row = {
                "aim": list(aim),
                "charge_ok": charge.get("ok"),
                "charge_frames": charge.get("frames"),
                "boost": charge.get("boost"),
                "store_ok": store.get("ok"),
                "armed_timer": (store.get("armed") or {}).get("spark_timer"),
                "armed_pose": (store.get("armed") or {}).get("pose"),
                "window": window,
                "spark_ok": act.get("ok"),
                "spark_pose_seen": act.get("spark_pose_seen"),
                "activate_pose": (act.get("activate") or {}).get("pose"),
                "min_y": act.get("min_y"),
                "max_x": act.get("max_x"),
                "min_x": act.get("min_x"),
                "start_y": (charge.get("boost") or {}).get("y"),
                "height_gain": (
                    (charge.get("boost") or {}).get("y", 0) - act.get("min_y", 0)
                    if charge.get("boost") and act.get("min_y") is not None
                    else None
                ),
                "room_changed": act.get("room_changed"),
                "end_room": act.get("end_room"),
                "final": act.get("final"),
            }
            rows.append(row)
            flag = "SPARK" if row["spark_ok"] else "MISS"
            print(
                f"aim={'+'.join(aim) or 'neutral':12s} {flag:5s} "
                f"arm_t={row['armed_timer']} pose_act={row['activate_pose']} "
                f"min_y={row['min_y']} max_x={row['max_x']} "
                f"Δh={row['height_gain']} room=0x{(row['end_room'] or 0):04X}"
            )
        finally:
            env.close()

    best = max(
        (r for r in rows if r.get("spark_ok")),
        key=lambda r: (r.get("height_gain") or 0, r.get("max_x") or 0),
        default=None,
    )
    return {
        "mode": "diagonal",
        "source": str(source),
        "rows": rows,
        "best": best,
        "ok": any(r.get("spark_ok") for r in rows),
    }


def run_sweep(
    source: Path,
    *,
    idle_range: tuple[int, int, int] = (0, 90, 5),
    activate_delay_range: tuple[int, int, int] | None = None,
    hold_down_idle: bool = False,
    aim: tuple[str, ...] = ("RIGHT",),
    assist: bool = True,
) -> dict[str, Any]:
    """Sweep idle_after_store and optional pre-activate delays."""
    start, stop, step = idle_range
    idle_rows: list[dict[str, Any]] = []
    for idle in range(start, stop + 1, step):
        rep = run_measure(
            source,
            idle_after_store=idle,
            hold_down_idle=hold_down_idle,
            aim=aim,
            assist=assist,
        )
        armed = (rep.get("store") or {}).get("armed") or {}
        act = rep.get("activate") or {}
        charge = rep.get("charge") or {}
        summary = {
            "idle_after_store": idle,
            "ok": rep.get("ok"),
            "charge_frames": charge.get("frames"),
            "boost_x": (charge.get("boost") or {}).get("x"),
            "boost_pose": (charge.get("boost") or {}).get("pose"),
            "armed_timer": armed.get("spark_timer"),
            "armed_pose": armed.get("pose"),
            "window_timer_end": (rep.get("window") or {}).get("timer_end"),
            "window_gt0": (rep.get("window") or {}).get("frames_timer_gt0"),
            "spark_ok": act.get("ok"),
            "spark_pose": (act.get("activate") or {}).get("pose"),
            "min_y": act.get("min_y"),
            "max_x": act.get("max_x"),
            "final_pose": (act.get("final") or {}).get("pose"),
            "final_xy": (
                (act.get("final") or {}).get("x"),
                (act.get("final") or {}).get("y"),
            ),
        }
        idle_rows.append(summary)
        flag = "GREEN" if summary["spark_ok"] else "RED"
        print(
            f"idle={idle:3d} {flag:5s} charge_f={summary['charge_frames']} "
            f"arm_t={summary['armed_timer']} pose={summary['armed_pose']} "
            f"spark_pose={summary['spark_pose']} "
            f"min_y={summary['min_y']} max_x={summary['max_x']}"
        )

    delay_rows: list[dict[str, Any]] = []
    if activate_delay_range is not None:
        d0, d1, ds = activate_delay_range
        for delay in range(d0, d1 + 1, ds):
            # charge+store then wait `delay` neutral then activate
            env, sess = boot_env(source, assist=assist)
            try:
                charge = spark.charge_until_boost(sess, "RIGHT")
                store = spark.crouch_store(sess)
                if delay > 0:
                    win = spark.wait_store_window(
                        sess, delay, hold_down=hold_down_idle
                    )
                else:
                    win = {"timer_end": (store.get("armed") or {}).get("spark_timer")}
                act = spark.activate_shinespark(
                    sess, *aim, travel_budget=180, label="delay"
                )
                delay_rows.append(
                    {
                        "activate_delay": delay,
                        "charge_ok": charge.get("ok"),
                        "store_ok": store.get("ok"),
                        "timer_at_activate": win.get("timer_end"),
                        "spark_ok": act.get("ok"),
                        "spark_pose": (act.get("activate") or {}).get("pose"),
                        "min_y": act.get("min_y"),
                        "max_x": act.get("max_x"),
                    }
                )
                flag = "GREEN" if act.get("ok") else "RED"
                print(
                    f"delay={delay:3d} {flag:5s} "
                    f"t={win.get('timer_end')} pose={delay_rows[-1]['spark_pose']} "
                    f"min_y={act.get('min_y')} max_x={act.get('max_x')}"
                )
            finally:
                env.close()

    good_idle = [r["idle_after_store"] for r in idle_rows if r.get("spark_ok")]
    good_delay = [
        r["activate_delay"] for r in delay_rows if r.get("spark_ok")
    ]
    return {
        "mode": "sweep",
        "source": str(source),
        "idle_rows": idle_rows,
        "delay_rows": delay_rows,
        "max_idle_still_spark": max(good_idle) if good_idle else None,
        "max_activate_delay_still_spark": max(good_delay) if good_delay else None,
        "ok": bool(good_idle) or bool(good_delay),
    }


def _print_measure(rep: dict[str, Any]) -> None:
    boot = rep.get("boot") or {}
    charge = rep.get("charge") or {}
    boost = charge.get("boost") or {}
    store = rep.get("store") or {}
    armed = store.get("armed") or {}
    win = rep.get("window") or {}
    act = rep.get("activate") or {}
    print(
        f"boot  room={boot.get('room_hex')} xy=({boot.get('x')},{boot.get('y')}) "
        f"pose={boot.get('pose')}"
    )
    print(
        f"charge frames={charge.get('frames')} ok={charge.get('ok')} "
        f"boost xy=({boost.get('x')},{boost.get('y')}) pose={boost.get('pose')} "
        f"echoes={boost.get('speed_echoes')}"
    )
    print(
        f"store armed=$0A68={armed.get('spark_timer')} "
        f"pose={armed.get('pose')} idx={armed.get('store_frame_index')} "
        f"xy=({armed.get('x')},{armed.get('y')}) start_pose={store.get('start_pose')}"
    )
    print(
        f"window timer {win.get('timer_start')}→{win.get('timer_end')} "
        f"gt0={win.get('frames_timer_gt0')} drain/f={win.get('drain_per_frame')}"
    )
    if act:
        fin = act.get("final") or {}
        print(
            f"activate ok={act.get('ok')} spark_pose_seen={act.get('spark_pose_seen')} "
            f"act_pose={(act.get('activate') or {}).get('pose')} "
            f"min_y={act.get('min_y')} max_x={act.get('max_x')} "
            f"final pose={fin.get('pose')} xy=({fin.get('x')},{fin.get('y')}) "
            f"room={fin.get('room_hex')}"
        )
    flag = "GREEN" if rep.get("ok") else "RED"
    print(f"{flag} {rep.get('error') or ''}".rstrip())


def _parse_range(s: str) -> tuple[int, int, int]:
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("need start:stop:step")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _parse_aim(s: str) -> tuple[str, ...]:
    if not s or s.lower() in ("none", "neutral", "-"):
        return ()
    return tuple(p.strip().upper() for p in s.replace("+", ",").split(",") if p.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "measure",
            "diagonal",
            "sweep",
            "bootstrap",
            "route",
            "record-diagonal",
            "verify",
        ),
        help="Practice mode",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Save state (default: {DEFAULT_PIN} or auto-bootstrap)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="JSON report directory",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_DIAGONAL_VIDEO,
        help=f"Output mp4 for record-diagonal (default: {DEFAULT_DIAGONAL_VIDEO})",
    )
    parser.add_argument("--direction", choices=("LEFT", "RIGHT"), default="RIGHT")
    parser.add_argument("--store-frames", type=int, default=spark.DEFAULT_STORE_MAX_FRAMES)
    parser.add_argument("--idle", type=int, default=0, help="idle frames after store")
    parser.add_argument(
        "--idle-down",
        action="store_true",
        help="hold DOWN during idle window (default neutral)",
    )
    parser.add_argument(
        "--aim",
        type=str,
        default="RIGHT",
        help="activate aim buttons, e.g. RIGHT, RIGHT+UP, UP",
    )
    parser.add_argument("--travel", type=int, default=220)
    parser.add_argument("--no-spark", action="store_true")
    parser.add_argument("--no-assist", action="store_true")
    parser.add_argument(
        "--idle-sweep",
        type=str,
        default="0:100:5",
        help="sweep idle_after_store start:stop:step",
    )
    parser.add_argument(
        "--delay-sweep",
        type=str,
        default="0:120:10",
        help="sweep activate_delay start:stop:step (empty to skip)",
    )
    parser.add_argument(
        "--no-delay-sweep",
        action="store_true",
        help="only idle_after_store sweep",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("bootstrap", "route"):
        summary = bootstrap_practice_pin(DEFAULT_PIN)
        path = out_dir / "bootstrap.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"wrote {path}")
        ok = (
            summary.get("speed_equipped")
            and summary.get("room_hex") == "0x91F8"
            and summary.get("not_endgame_f32f")
            and summary.get("idle_stable")
            and not (summary.get("escape") or {}).get("escape_active")
        )
        print(
            f"VERIFY room={summary.get('room_hex')} "
            f"items={summary.get('equipped_items_hex')} "
            f"escape_active={(summary.get('escape') or {}).get('escape_active')} "
            f"idle_stable={summary.get('idle_stable')} "
            f"{'GREEN' if ok else 'RED'}"
        )
        return 0 if ok else 1

    if args.mode == "verify":
        source = ensure_source(args.source)
        env, sess = boot_env(source, assist=True)
        st = sess.state
        esc = _escape_snapshot(env, st)
        rooms: set[int] = set()
        gs: set[int] = set()
        for _ in range(120):
            sess.step(idle_action())
            rooms.add(int(sess.state.room_id))
            gs.add(int(sess.state.game_state))
        st = sess.state
        esc = _escape_snapshot(env, st)
        env.close()
        ok = (
            st.room_id == ROOM_LANDING
            and bool(st.equipped_items & 0x2000)
            and st.equipped_items != 0xF32F
            and not esc["escape_active"]
            and rooms == {ROOM_LANDING}
            and gs == {8}
        )
        print(f"room=0x{st.room_id:04X}")
        print(f"items=0x{st.equipped_items:04X} collected=0x{st.collected_items:04X}")
        print(f"xy=({st.samus_x},{st.samus_y}) pose={st.pose} facing={st.facing}")
        print(
            f"escape event/timer inactive={not esc['escape_active']} "
            f"timer_type={esc['timer_type']} "
            f"t={esc['escape_min']}:{esc['escape_sec']}.{esc['escape_frames']} "
            f"event_0e={esc['event_0e_mb_dead']}"
        )
        print(
            f"no explosions for 120 idle frames: "
            f"rooms={[f'0x{r:04X}' for r in sorted(rooms)]} gs={sorted(gs)}"
        )
        print(f"{'GREEN' if ok else 'RED'} pin={source}")
        path = out_dir / "verify.json"
        path.write_text(
            json.dumps(
                {
                    "ok": ok,
                    "source": str(source),
                    "room_hex": f"0x{st.room_id:04X}",
                    "items_hex": f"0x{st.equipped_items:04X}",
                    "xy": [st.samus_x, st.samus_y],
                    "escape": esc,
                    "idle_rooms": sorted(f"0x{r:04X}" for r in rooms),
                    "idle_gs": sorted(gs),
                },
                indent=2,
            )
            + "\n"
        )
        return 0 if ok else 1

    source = ensure_source(args.source)
    aim = _parse_aim(args.aim)

    if args.mode == "measure":
        rep = run_measure(
            source,
            direction=args.direction,
            store_frames=args.store_frames,
            idle_after_store=args.idle,
            hold_down_idle=args.idle_down,
            aim=aim,
            travel_budget=args.travel,
            do_spark=not args.no_spark,
            assist=not args.no_assist,
        )
        _print_measure(rep)
        path = out_dir / "measure.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"wrote {path} (log_len={rep.get('log_len')})")
        return 0 if rep.get("ok") else 1

    if args.mode == "diagonal":
        rep = run_diagonal(
            source,
            idle_after_store=max(args.idle, 2),
            travel_budget=args.travel,
            assist=not args.no_assist,
        )
        path = out_dir / "diagonal.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"wrote {path}")
        if rep.get("best"):
            b = rep["best"]
            print(
                f"best aim={'+'.join(b['aim'])} height_gain={b.get('height_gain')} "
                f"min_y={b.get('min_y')} max_x={b.get('max_x')}"
            )
        return 0 if rep.get("ok") else 1

    if args.mode == "record-diagonal":
        source = ensure_source(args.source)
        rep = run_record_diagonal(
            source,
            video_path=args.video,
            idle_after_store=max(args.idle, 2),
            travel_budget=max(args.travel, 250),
            assist=not args.no_assist,
        )
        path = out_dir / "record_diagonal.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        vpath = Path(rep["video"])
        size = vpath.stat().st_size if vpath.is_file() else 0
        print(
            f"room={(rep.get('boot') or {}).get('room_hex')} "
            f"spark_ok={rep.get('spark_ok')} pose={rep.get('activate_pose')} "
            f"min_y={rep.get('min_y')} height_gain={rep.get('height_gain')}"
        )
        print(f"video={vpath} size={size} frames={rep.get('frames_written')}")
        print(f"wrote {path}")
        flag = "GREEN" if rep.get("ok") and size > 1000 else "RED"
        print(flag)
        return 0 if rep.get("ok") and size > 1000 else 1

    if args.mode == "sweep":
        idle_r = _parse_range(args.idle_sweep)
        delay_r = None if args.no_delay_sweep else _parse_range(args.delay_sweep)
        rep = run_sweep(
            source,
            idle_range=idle_r,
            activate_delay_range=delay_r,
            hold_down_idle=args.idle_down,
            aim=aim,
            assist=not args.no_assist,
        )
        path = out_dir / "sweep.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"wrote {path}")
        print(f"max idle_after_store still spark: {rep.get('max_idle_still_spark')}")
        print(
            f"max activate_delay still spark: "
            f"{rep.get('max_activate_delay_still_spark')}"
        )
        return 0 if rep.get("ok") else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
