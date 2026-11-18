"""Structured progress snapshots for autoplay watchdog and diagnostics.

``ProgressSnapshot.signature()`` is a *semantic* stall key: task identity,
phase, child signature, and non-tick details. Elapsed ``step_count`` stays on
the snapshot for UI/diagnostics but is not progress, including any
``("step_count", ...)`` pair smuggled in ``details``.

Two independent stall windows sit beside the snapshot for later D2 / PlaySession
use (leftover_exec still has its own comparator this slice):

- Motion liveness (``motion_liveness_key``): target / approach / player
  position. Short window: ``MOTION_STALL_FRAMES`` (6s at 60fps).
- Goal progress (``goal_progress_key``): debris counts, crop planted/wet,
  carry, stamina. Long window: ``GOAL_STALL_FRAMES`` (leftover_exec default).

``stalled(last_frame, now, window)`` is the shared comparator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


MOTION_STALL_FRAMES = 360
GOAL_STALL_FRAMES = 24_000
# A no-button, no-movement stretch this long is listed in the work report (2 s).
DEAD_RUN_MIN_FRAMES = 120
_TICK_DETAIL_KEYS = frozenset({"step_count"})


def _semantic_details(details: Sequence[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(pair for pair in details if pair[0] not in _TICK_DETAIL_KEYS)


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable progress view for a task tree node."""

    task_name: str
    phase_text: str = ""
    phase_index: Optional[int] = None
    step_count: Optional[int] = None
    details: Tuple[Tuple[str, Any], ...] = ()
    child: Optional["ProgressSnapshot"] = None

    def signature(self) -> Tuple[Any, ...]:
        """Hashable semantic signature for stall detection.

        Elapsed ``step_count`` is diagnostic-only and is excluded.
        """
        return (
            self.task_name,
            self.phase_text,
            self.phase_index,
            _semantic_details(self.details),
            self.child.signature() if self.child is not None else None,
        )

    def to_record(self) -> dict[str, Any]:
        """JSON-safe tree for a run sidecar (includes diagnostic step_count)."""
        return {
            "task_name": self.task_name,
            "phase_text": self.phase_text,
            "phase_index": self.phase_index,
            "step_count": self.step_count,
            "details": {str(key): jsonable(value) for key, value in self.details},
            "child": self.child.to_record() if self.child is not None else None,
        }


def motion_liveness_key(*, target: Any, approach: Any, pos: Any) -> Tuple[Any, ...]:
    """Short-window key: navigation target / approach / player position."""
    return (target, approach, pos)


def goal_progress_key(
    *,
    debris: Any,
    planted: Any,
    wet: Any,
    carry: Any,
    stamina: Any,
) -> Tuple[Any, ...]:
    """Long-window key: debris / crop planted-wet / carry / stamina."""
    return (debris, planted, wet, carry, stamina)


def stalled(last_frame: int, now: int, window: int) -> bool:
    """True when ``now - last_frame`` has reached a positive ``window``."""
    return window > 0 and now - last_frame >= window


# A walking task that has not left a 24 px box for 10 s is wedged: a walk covers 1-2 px a frame. The
# corridor's own recoveries (close-range stall, yield, pixel-stuck) finish by about 400 f, so this never pre-empts them.
NAV_WEDGE_BOX_PX = 24
NAV_WEDGE_FRAMES = 600
# Never closer to the target by NAV_PROGRESS_PX for 20 s: pacing, not a detour (the whole farm is ~1,000 px, a
# long detour 750 f). Within NAV_NEAR_PX of the target it is 10 s: that is the farmer circling a plot centre it
# cannot reach (rec1 Y1 D12: 30-38 px out for 9,000 f, never in one 24 px box, so the box alone missed it).
NAV_PROGRESS_PX = 8
NAV_NEAR_PX = 64
NAV_NO_PROGRESS_FRAMES = 1200
NAV_NO_PROGRESS_NEAR_FRAMES = 600
# A multi-map leg that gets no closer to its waypoint (8 px) for this long fails, however far the farmer wobbles:
# fdet_20261115 swung 50 px up and down inside the fishing camp for 11,001 f, outside any 24 px box.
NAV_WAYPOINT_NO_PROGRESS_FRAMES = 2400


class MotionBox:
    """Frames a walking task has stayed inside a small box on one map.

    ``observe`` returns 0 whenever the farmer leaves the box or changes map, so a wedge of any
    shape (pushing a wall, a two-pixel wiggle, a dead pad) reads as one rising number.
    """

    def __init__(self, box_px: int = NAV_WEDGE_BOX_PX) -> None:
        self.box_px = box_px
        self.anchor: tuple[int, int, int] | None = None
        self.since = 0

    def observe(self, step: int, tilemap: int, x: int, y: int) -> int:
        a = self.anchor
        if a is None or a[0] != tilemap or max(abs(x - a[1]), abs(y - a[2])) > self.box_px:
            self.anchor = (tilemap, x, y)
            self.since = step
            return 0
        return step - self.since

    def restart(self, step: int) -> None:
        self.since = step

    def clear(self) -> None:
        self.anchor = None
        self.since = 0


def _read_attr(task: object, name: str) -> Any:
    try:
        return getattr(task, name)
    except AttributeError:
        return None


def task_progress_snapshot(task: object, *, depth: int = 0) -> Optional[ProgressSnapshot]:
    """Build a progress snapshot from a task, using public APIs when present."""
    if task is None or depth > 4:
        return None

    progress = getattr(task, "progress_snapshot", None)
    if callable(progress):
        snap = progress()
        if isinstance(snap, ProgressSnapshot):
            return snap

    phase_text = _read_attr(task, "phase_text")
    if phase_text is None:
        phase = _read_attr(task, "_phase")
        if phase is not None:
            phase_text = str(phase).upper()

    details: list[tuple[str, Any]] = []
    for attr in (
        "phase_index",
        "step_count",
        "_wp_index",
        "_plot_phase",
        "_water_index",
        "_target_tile",
        "_approach_tile",
        "_target_cow_slot",
    ):
        value = _read_attr(task, attr)
        if value is not None:
            details.append((attr, value))

    child_task = _read_attr(task, "current_task")
    if child_task is None:
        child_task = _read_attr(task, "_current_task")
    if child_task is None:
        child_task = _read_attr(task, "_task")
    if child_task is None:
        child_task = _read_attr(task, "_nav")
    if child_task is None:
        child_task = _read_attr(task, "_inner")
    child = None
    if child_task is not None and child_task is not task:
        child = task_progress_snapshot(child_task, depth=depth + 1)

    return ProgressSnapshot(
        task_name=task.__class__.__name__,
        phase_text=str(phase_text or ""),
        phase_index=_read_attr(task, "phase_index"),
        step_count=_read_attr(task, "step_count"),
        details=tuple(details),
        child=child,
    )


def task_progress_chain(task: object) -> Tuple[Any, ...]:
    """Return a hashable chain for watchdog comparisons."""
    snap = task_progress_snapshot(task)
    return () if snap is None else (snap.signature(),)


def jsonable(value: Any) -> Any:
    """Convert snapshot / RAM diagnostics to JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    name = getattr(value, "name", None)
    if isinstance(name, str) and not isinstance(value, type):
        return name
    return str(value)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON so readers never observe a partial sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(jsonable(payload), indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def walk_task_tree(task: object) -> Iterator[object]:
    """Depth-first walk of planner/tactic/child wrappers."""
    seen: set[int] = set()
    stack = [task]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in (
            "current_task",
            "_current_task",
            "_child",
            "_inner",
            "_task",
            "_nav",
            "_sub",
        ):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)


def work_snapshot(ram, *, census: bool = False) -> dict[str, Any]:
    """Read-only work receipt. Crop gates require a settled, loaded farm."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.farm_clear_quota import farm_map_loaded
    from harvest.tasks.crop_geometry import count_crop_survival

    keys = ("year", "season", "day", "hour", "minute", "tilemap", "player_x",
            "player_y", "stamina", "watering_can", "money", "shipping_money",
            "potato_seeds", "turnip_seeds", "corn_seeds", "tomato_seeds")
    row = {key: int(read_ram_value(ram, key) or 0) for key in keys}
    if census:
        loaded = farm_map_loaded(ram)
        row["crops"] = count_crop_survival(ram) if loaded else None
        row["gates"] = {"farm_loaded": loaded, "all_growing_crops_wet":
                        row["crops"]["dry"] == 0 if loaded else None}
    return row


def work_receipt(before: Mapping, ram, task: object) -> dict[str, Any]:
    """Structured counters and RAM deltas, including partial failed work."""
    after = work_snapshot(ram, census=True)
    counts = {key: int(getattr(task, key)) for key in
              ("planted_count", "watered_count", "skipped_water", "harvested_count",
               "shipped_count", "skipped_count") if hasattr(task, key)}
    return {"before": dict(before), "after": after, "counts": counts,
            "delta": {key: after[key] - before[key] for key in
                      ("stamina", "watering_can", "money", "shipping_money",
                       "potato_seeds", "turnip_seeds", "corn_seeds", "tomato_seeds")
                      if key in before}}


class WorkFrameLedger:
    """Passive frame attribution through the existing Composer, never a driver.

    ``observe`` runs immediately before each emulator step, after the task has
    chosen its action. Intervals are [start_frame, end_frame); no emulator step
    means no charged frame. Targets and task states, not changing positions or
    counters, delimit intervals. A/B comparisons retain the exact visit order.

    With ``action`` it also counts idle time. A frame is *dead* when no button
    is held and the map and pixel position equal the previous frame's: the run
    asked for nothing and nothing moved. ``input_frames`` / ``dead_frames`` sit
    on each segment, and ``report()["idle"]`` lists every dead run of
    ``DEAD_RUN_MIN_FRAMES`` or more with its phase and date.
    """

    def __init__(self) -> None:
        self.segments: list[dict[str, Any]] = []
        self._key: tuple | None = None
        self._nodes: tuple = ()
        self._pose: tuple | None = None
        self._dead: dict[str, Any] | None = None
        self.dead_runs: list[dict[str, Any]] = []
        self.idle_totals = {"input_frames": 0, "idle_frames": 0, "dead_frames": 0}

    def observe(self, *, frame: int, ram, task: object, action=None) -> None:
        from harvest.planner.day_plan_status import read_world_ymd

        nodes = tuple(walk_task_tree(task))
        composer = next((n for n in nodes if hasattr(n, "runtime_phases")), task)
        phase = str(getattr(composer, "phase_text", getattr(composer, "_phase", "")))
        context = []
        for node in nodes:
            if not type(node).__module__.startswith("harvest."):
                continue
            state = getattr(node, "_phase", getattr(node, "_state", ""))
            target = getattr(node, "target_tile", None) or getattr(node, "_target_tile", None)
            current = getattr(node, "_current", None)
            target = target or getattr(current, "target", None)
            px = getattr(node, "target_px", None)
            if hasattr(px, "x"):
                px = (px.x, px.y)
            context.append((type(node).__name__, str(getattr(node, "name", "")),
                            str(state), tuple(target) if target is not None else None,
                            tuple(px) if px is not None else None))
        key = (read_world_ymd(ram), phase, tuple(context))
        if key != self._key:
            snap = work_snapshot(ram)
            if self.segments:
                self._close(snap)
            self.segments.append({"date": list(key[0]), "phase": phase,
                                  "context": jsonable(context), "start_frame": int(frame),
                                  "end_frame": int(frame), "frames": 0, "before": snap})
            self._key = key
        self._nodes = nodes
        row = self.segments[-1]
        row["end_frame"] = int(frame) + 1
        row["frames"] += 1
        if action is not None:
            self._count_idle(frame, ram, action, row, phase, key[0])

    def _count_idle(self, frame: int, ram, action, row: dict, phase: str, date) -> None:
        from harvest.core.ram_catalog import read_ram_value

        pressed = bool(np.any(action))
        pose = (read_ram_value(ram, "tilemap"), read_ram_value(ram, "player_x"),
                read_ram_value(ram, "player_y"))
        dead = not pressed and pose == self._pose
        self._pose = pose
        totals = self.idle_totals
        totals["input_frames" if pressed else "idle_frames"] += 1
        row["input_frames"] = row.get("input_frames", 0) + int(pressed)
        if not dead:
            self._close_dead()
            return
        totals["dead_frames"] += 1
        row["dead_frames"] = row.get("dead_frames", 0) + 1
        if self._dead is None:
            self._dead = {"start_frame": int(frame), "frames": 0, "phase": phase,
                          "date": list(date), "tilemap": int(pose[0])}
        self._dead["frames"] += 1

    def _close_dead(self) -> None:
        if self._dead is not None and self._dead["frames"] >= DEAD_RUN_MIN_FRAMES:
            self.dead_runs.append(self._dead)
        self._dead = None

    def _close(self, snapshot: dict) -> None:
        row = self.segments[-1]
        row["after"] = snapshot
        row["counts"] = {str(getattr(n, "name", type(n).__name__)): {
            key: int(getattr(n, key)) for key in
            ("planted_count", "watered_count", "harvested_count", "shipped_count",
             "skipped_count", "skipped_water") if hasattr(n, key)} for n in self._nodes
            if any(hasattr(n, key) for key in ("planted_count", "harvested_count"))}

    def report(self, ram) -> dict[str, Any]:
        if self.segments:
            self._close(work_snapshot(ram))
        totals: dict[str, int] = {}
        for row in self.segments:
            totals[row["phase"]] = totals.get(row["phase"], 0) + row["frames"]
        report = {"schema_version": 1, "source": "emulator_step_intervals",
                  "frames": sum(totals.values()), "phase_frames": totals,
                  "segments": self.segments}
        if any(self.idle_totals.values()):
            self._close_dead()
            report["idle"] = {**self.idle_totals, "dead_run_min_frames": DEAD_RUN_MIN_FRAMES,
                              "dead_runs": list(self.dead_runs)}
        return report


def watchdog_counters(task: object) -> dict[str, Any]:
    """Motion/goal stall ages from the first tactic that owns those windows."""
    empty = {
        "task": None,
        "step": None,
        "motion_age": None,
        "goal_age": None,
        "unobservable_frames": None,
        "motion_window": MOTION_STALL_FRAMES,
        "goal_window": GOAL_STALL_FRAMES,
    }
    for node in walk_task_tree(task):
        if not hasattr(node, "_motion_at") or not hasattr(node, "_goal_at"):
            continue
        step = int(getattr(node, "_step", 0) or 0)
        motion_at = int(getattr(node, "_motion_at", step) or step)
        goal_at = int(getattr(node, "_goal_at", step) or step)
        return {
            "task": node.__class__.__name__,
            "step": step,
            "motion_age": step - motion_at,
            "goal_age": step - goal_at,
            "unobservable_frames": int(getattr(node, "_unobs", 0) or 0),
            "motion_window": MOTION_STALL_FRAMES,
            "goal_window": GOAL_STALL_FRAMES,
        }
    return empty


def _d2_record(d2_status: object | None) -> dict[str, Any] | None:
    if d2_status is None:
        return None
    to_record = getattr(d2_status, "to_record", None)
    if callable(to_record):
        row = dict(to_record())
    elif isinstance(d2_status, Mapping):
        row = dict(d2_status)
    else:
        return None
    stam = getattr(d2_status, "stamina", None)
    if stam is not None and "stamina" not in row:
        row["stamina"] = {
            "current": int(getattr(stam, "current", 0) or 0),
            "maximum": int(getattr(stam, "maximum", 0) or 0),
        }
    return jsonable(row)


def build_run_progress_record(
    *,
    frame: int,
    planner_frames: int,
    wall_seconds: float,
    ram,
    task: object,
    d2_status: object | None = None,
) -> dict[str, Any]:
    """Sidecar payload: date/map/pose/phase/task/D2/debris/stamina/watchdogs."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.stamina import Stamina
    from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, TILE_SIZE

    year = int(read_ram_value(ram, "year") or 0)
    season = int(read_ram_value(ram, "season") or 0)
    day = int(read_ram_value(ram, "day") or 0)
    hour = int(read_ram_value(ram, "hour") or 0)
    minute = int(read_ram_value(ram, "minute") or 0)
    money = int(read_ram_value(ram, "money") or 0)
    px = int(read_ram_value(ram, "player_x") or 0)
    py = int(read_ram_value(ram, "player_y") or 0)
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else -1
    lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
    stam = Stamina.from_ram(ram)
    snap = task_progress_snapshot(task)
    d2 = _d2_record(d2_status)
    debris = None
    if d2 is not None:
        debris = {
            "weeds": d2.get("weeds"),
            "fences": d2.get("fences"),
            "stones": d2.get("stones"),
            "small_rocks": d2.get("small_rocks"),
            "large_rocks": d2.get("large_rocks"),
            "trees_or_stumps": d2.get("trees_or_stumps"),
        }
        if d2.get("stamina") is None:
            d2["stamina"] = {"current": stam.current, "maximum": stam.maximum}
    phase = _read_attr(task, "phase_text")
    if phase is None:
        phase = _read_attr(task, "_phase")
    return {
        "frame": int(frame),
        "planner_frames": int(planner_frames),
        "wall_seconds": round(float(wall_seconds), 1),
        "date": {
            "year": year,
            "season": season,
            "day": day,
            "hour": hour,
            "minute": minute,
            "money": money,
        },
        "map": {"tilemap": tilemap},
        "position": {
            "x": px,
            "y": py,
            "tile": [px // TILE_SIZE, py // TILE_SIZE],
        },
        "phase": str(phase or ""),
        "progress_text": str(_read_attr(task, "progress_text") or ""),
        "task": snap.to_record() if snap is not None else None,
        "d2": d2,
        "debris": debris,
        "stamina": {"current": stam.current, "maximum": stam.maximum},
        "input_stable": lock == 1,
        "watchdog": watchdog_counters(task),
    }


def stall_progress_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Semantic stall key: clock ticks are not progress."""
    task = record.get("task") or {}
    return (
        checkpoint_progress_key(record),
        task.get("phase_text"),
        jsonable(task.get("details")),
        jsonable((task.get("child") or {}).get("details")),
        (task.get("child") or {}).get("phase_text"),
    )


def checkpoint_progress_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Coarse key for safe emulator pins (phase/debris/crops, not nav target)."""
    date = record.get("date") or {}
    d2 = record.get("d2") or {}
    debris = record.get("debris") or {}
    return (
        date.get("year"),
        date.get("season"),
        date.get("day"),
        record.get("phase"),
        ((record.get("task") or {}).get("child") or {}).get("phase_text"),
        debris.get("weeds", d2.get("weeds")),
        debris.get("fences", d2.get("fences")),
        debris.get("stones", d2.get("stones")),
        debris.get("large_rocks", d2.get("large_rocks")),
        debris.get("trees_or_stumps", d2.get("trees_or_stumps")),
        d2.get("planted"),
        d2.get("wet"),
        d2.get("outcome"),
    )


def safe_to_checkpoint(record: Mapping[str, Any]) -> bool:
    """Skip mid-swing / dialogue frames; those pins are not resume-safe."""
    d2 = record.get("d2") or {}
    input_stable = d2.get("input_stable", record.get("input_stable", True))
    animating = bool(d2.get("animating", False))
    return bool(input_stable) and not animating


def format_run_progress_line(record: Mapping[str, Any]) -> str:
    date = record.get("date") or {}
    debris = record.get("debris") or {}
    stam = record.get("stamina") or {}
    stall = record.get("stall") or {}
    debris_txt = ""
    if debris:
        debris_txt = (
            f" debris=w{debris.get('weeds')}/f{debris.get('fences')}"
            f"/s{debris.get('stones')}/r{debris.get('large_rocks')}"
            f"/u{debris.get('trees_or_stumps')}"
        )
    # Year 1 keeps the legacy ``S<n>D<n>`` shape (log parsers); later years
    # carry a 1-based ``Y<n>`` prefix so a rollover is visible.
    year0 = int(date.get("year") or 0)
    year_txt = f"Y{year0 + 1}" if year0 else ""
    return (
        f"[RUN] f={record.get('frame')} date={year_txt}S{date.get('season')}D{date.get('day')} "
        f"{int(date.get('hour') or 0):02d}:{int(date.get('minute') or 0):02d} "
        f"${date.get('money')} phase={record.get('phase')} "
        f"{record.get('progress_text') or ''}"
        f"{debris_txt} stam={stam.get('current')}/{stam.get('maximum')} "
        f"stall={stall.get('semantic_age')}"
    ).rstrip()


@dataclass
class RunProgressSidecar:
    """Atomic live sidecar plus optional safe checkpoints on semantic progress."""

    path: Path
    checkpoint_dir: Path | None = None
    checkpoint_on_progress: bool = False
    keep: int = 4
    last_record: dict[str, Any] | None = None
    last_stall_key: Tuple[Any, ...] | None = None
    last_stall_frame: int = 0
    last_checkpoint_key: Tuple[Any, ...] | None = None
    pending_checkpoint: bool = False
    named_checkpoints: list[Path] | None = None

    def write(
        self,
        *,
        frame: int,
        planner_frames: int,
        wall_seconds: float,
        ram,
        task: object,
        d2_status: object | None = None,
        save_state: Callable[[Path], Path] | None = None,
    ) -> dict[str, Any]:
        record = build_run_progress_record(
            frame=frame,
            planner_frames=planner_frames,
            wall_seconds=wall_seconds,
            ram=ram,
            task=task,
            d2_status=d2_status,
        )
        stall_key = stall_progress_key(record)
        if stall_key != self.last_stall_key:
            self.last_stall_key = stall_key
            self.last_stall_frame = frame
        record["stall"] = {
            "semantic_age": int(frame) - int(self.last_stall_frame),
            "last_progress_frame": int(self.last_stall_frame),
        }
        ck = checkpoint_progress_key(record)
        if ck != self.last_checkpoint_key:
            self.last_checkpoint_key = ck
            self.pending_checkpoint = True
        record["checkpoint"] = None
        want_pin = bool(self.checkpoint_on_progress or self.checkpoint_dir)
        if want_pin and self.pending_checkpoint and save_state is not None:
            if safe_to_checkpoint(record):
                paths = self._save_checkpoints(frame, save_state)
                record["checkpoint"] = {
                    "paths": [str(path) for path in paths],
                    "frame": int(frame),
                }
                self.pending_checkpoint = False
        atomic_write_json(self.path, record)
        self.last_record = record
        return record

    def _save_checkpoints(
        self, frame: int, save_state: Callable[[Path], Path]
    ) -> list[Path]:
        import shutil

        if self.checkpoint_dir is None:
            return []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        named = self.checkpoint_dir / f"progress_f{int(frame)}.state"
        save_state(named)
        latest = self.checkpoint_dir / "latest.state"
        tmp = latest.with_name(latest.name + ".tmp")
        shutil.copy2(named, tmp)
        os.replace(tmp, latest)
        stored = self.named_checkpoints
        if stored is None:
            stored = []
            self.named_checkpoints = stored
        stored.append(named)
        while len(stored) > self.keep:
            old = stored.pop(0)
            if old != named:
                old.unlink(missing_ok=True)
        return [latest, named]
