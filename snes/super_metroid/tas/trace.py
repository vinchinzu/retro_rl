"""Replay Super Metroid TAS button streams under stable-retro and annotate WRAM.

Loads SNES-12 frames (movie / slice / raw list), steps the harness without
directional sanitize, and records:

* event timeline (:class:`~super_metroid.tas.annotate.Annotator`)
* room hops via :class:`~super_metroid.room_timer.RoomTimer`
* optional dense / strided kinematics series
* optional ``.state`` dumps at room_enter / every N frames

```bash
# Prefer the CLI:
uv run python -m super_metroid.tas.replay --slice sniq_any_menu --annotate
```
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Sequence

import numpy as np

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import SNES_BUTTON_NAMES
from retro_harness.env import make_env, write_state_bytes
from super_metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR
from super_metroid.ram import SuperMetroidState, parse_env_state, probe_pin
from super_metroid.room_timer import RoomTimer
from super_metroid.tas.annotate import Annotator, TraceEvent
from super_metroid.tas.rle import expand_snes12_rle, load_snes12_rle_seed
from super_metroid.tas.slice import SLICE_CATALOG, load_movie_frames, slice_frames

ProgressCb = Callable[[int, int, SuperMetroidState, int], None]

DEFAULT_OUT_ROOT = RECORDINGS_DIR / "tas_import"


def _pad12(frame: Sequence[int]) -> list[int]:
    buttons = [1 if int(b) else 0 for b in frame[:SNES_ACTION_SIZE]]
    if len(buttons) < SNES_ACTION_SIZE:
        buttons.extend([0] * (SNES_ACTION_SIZE - len(buttons)))
    return buttons


def frame_button_names(frame: Sequence[int]) -> list[str]:
    """Return pressed SNES button names for one env frame (no sanitize)."""
    fr = _pad12(frame)
    return [SNES_BUTTON_NAMES[i] for i, v in enumerate(fr) if v]


def action_array(frame: Sequence[int]) -> np.ndarray:
    """NumPy int8 action for ``env.step`` — preserves L+R."""
    return np.asarray(_pad12(frame), dtype=np.int8)


@dataclass
class FrameSample:
    """One strided kinematics + input sample."""

    frame: int
    buttons: list[str]
    room_id: int
    game_state: int
    phase: str
    pose: int
    x: int
    y: int
    x_sub: int
    y_sub: int
    velocity_x: int
    velocity_y: int
    momentum_x: int
    speed_counter: int
    facing: int
    movement_type: int
    shinespark_timer: int
    items: int
    beams: int
    health: int = 0
    door_transition: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room_id_hex"] = f"0x{self.room_id:04X}"
        return d

    @classmethod
    def from_state(
        cls,
        state: SuperMetroidState,
        buttons: Sequence[str],
    ) -> FrameSample:
        return cls(
            frame=int(state.frame),
            buttons=list(buttons),
            room_id=int(state.room_id),
            game_state=int(state.game_state),
            phase=state.phase.value if hasattr(state.phase, "value") else str(state.phase),
            pose=int(state.pose),
            x=int(state.samus_x),
            y=int(state.samus_y),
            x_sub=int(state.samus_x_sub),
            y_sub=int(state.samus_y_sub),
            velocity_x=int(state.velocity_x),
            velocity_y=int(state.velocity_y),
            momentum_x=int(state.momentum_x),
            speed_counter=int(state.speed_counter),
            facing=int(state.facing),
            movement_type=int(state.movement_type),
            shinespark_timer=int(state.shinespark_timer),
            items=int(state.collected_items),
            beams=int(state.collected_beams),
            health=int(state.health),
            door_transition=int(state.door_transition),
        )


@dataclass
class MovieTrace:
    """Full annotated timeline for one TAS replay under the harness."""

    source: str
    start_mode: str
    num_frames: int
    frames_played: int
    events: list[TraceEvent] = field(default_factory=list)
    rooms: list[dict[str, Any]] = field(default_factory=list)
    room_timer_report: dict[str, Any] = field(default_factory=dict)
    series: list[FrameSample] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    state_dumps: list[dict[str, Any]] = field(default_factory=list)
    annotate_summary: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "super_metroid_tas_trace_v1",
            "source": self.source,
            "start_mode": self.start_mode,
            "num_frames": self.num_frames,
            "frames_played": self.frames_played,
            "events": [e.to_dict() for e in self.events],
            "rooms": self.rooms,
            "room_timer": self.room_timer_report,
            "series_len": len(self.series),
            "series": [s.to_dict() for s in self.series],
            "final": self.final,
            "state_dumps": self.state_dumps,
            "annotate": self.annotate_summary,
            "meta": self.meta,
        }

    def summary(self) -> dict[str, Any]:
        """Compact stdout-oriented summary (no full series)."""
        room_enters = [e for e in self.events if e.kind == "room_enter"]
        unique_rooms: list[dict[str, Any]] = []
        seen: set[int] = set()
        for e in room_enters:
            if e.room_id not in seen:
                seen.add(e.room_id)
                unique_rooms.append(
                    {
                        "frame": e.frame,
                        "room_id": e.room_id,
                        "room_id_hex": f"0x{e.room_id:04X}",
                        "pose": e.pose,
                        "x": e.x,
                        "y": e.y,
                    }
                )
        return {
            "source": self.source,
            "start_mode": self.start_mode,
            "num_frames": self.num_frames,
            "frames_played": self.frames_played,
            "event_count": len(self.events),
            "annotate": self.annotate_summary,
            "room_visits": len(self.rooms),
            "room_enter_count": len(room_enters),
            "unique_rooms": unique_rooms,
            "room_timeline": [
                {
                    "frame": e.frame,
                    "detail": e.detail,
                    "room_id_hex": f"0x{e.room_id:04X}",
                    "pose": e.pose,
                    "x": e.x,
                    "y": e.y,
                }
                for e in room_enters
            ],
            "series_len": len(self.series),
            "state_dumps": len(self.state_dumps),
            "final": self.final,
            "meta": self.meta,
            "first_events": [e.to_dict() for e in self.events[:30]],
            "last_events": [e.to_dict() for e in self.events[-15:]],
        }


def resolve_frames(
    *,
    frames: Sequence[Sequence[int]] | None = None,
    movie: Path | str | None = None,
    slice_id: str | None = None,
    seed_path: Path | str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> tuple[list[list[int]], str]:
    """Load SNES-12 frames from one of the supported sources."""
    if frames is not None:
        body = [_pad12(f) for f in frames]
        source = "inline"
    elif slice_id is not None:
        if slice_id not in SLICE_CATALOG:
            raise KeyError(f"unknown slice {slice_id!r}; known={sorted(SLICE_CATALOG)}")
        seed = GAME_DIR / "tas" / "slices" / f"{slice_id}.json"
        if seed.exists():
            data = load_snes12_rle_seed(seed)
            body = expand_snes12_rle(data)
            source = f"slice:{slice_id}"
        else:
            # Fall back to live parse of the ref movie.
            spec = SLICE_CATALOG[slice_id]
            if not spec.movie.exists():
                raise FileNotFoundError(f"missing movie for slice {slice_id}: {spec.movie}")
            all_fr = load_movie_frames(spec.movie, spec.kind)
            stop = spec.resolve_end(len(all_fr))
            body = slice_frames(all_fr, spec.start, stop)
            source = f"slice:{slice_id}:movie"
    elif seed_path is not None:
        path = Path(seed_path)
        data = load_snes12_rle_seed(path)
        body = expand_snes12_rle(data)
        source = str(path)
    elif movie is not None:
        path = Path(movie)
        body = load_movie_frames(path)
        source = str(path)
    else:
        raise ValueError("provide frames, slice_id, seed_path, or movie")

    lo = 0 if start is None else int(start)
    hi = len(body) if end is None else int(end)
    if lo < 0 or lo > len(body):
        raise ValueError(f"start {lo} out of range for {len(body)} frames")
    hi = min(max(hi, lo), len(body))
    if lo or hi != len(body):
        body = body[lo:hi]
        source = f"{source}[{lo}:{hi}]"
    return body, source


def trace_frames(
    frames: Sequence[Sequence[int]],
    *,
    source: str = "inline",
    state_name: str | None = None,
    state_path: Path | str | None = None,
    game_dir: Path | None = None,
    max_frames: int | None = None,
    series_stride: int = 0,
    parse_mode: str = "nav",
    stall_frames: int = 90,
    dump_states_on: Sequence[str] = (),
    dump_every: int = 0,
    states_dir: Path | str | None = None,
    keep_series: bool = True,
    progress_every: int = 0,
    on_progress: ProgressCb | None = None,
) -> MovieTrace:
    """Replay *frames* and collect annotation + optional series / state dumps.

    Parameters
    ----------
    state_name:
        Integration state stem, or ``None`` / ``\"NONE\"`` for power-on.
    state_path:
        Explicit ``.state`` path (loaded after reset via ``em.set_state``).
    series_stride:
        ``0`` = no series; ``1`` = every frame; ``N`` = every N frames.
    dump_states_on:
        Event kinds that trigger a ``.state`` write (e.g. ``room_enter``,
        ``control``, ``item_gain``).
    dump_every:
        Also dump every N frames when > 0.
    progress_every / on_progress:
        Heartbeat for long runs. Callback args:
        ``(frame_i, total, state, event_count)``.
    """
    game_dir = game_dir or GAME_DIR
    buttons = [_pad12(f) for f in frames]
    if max_frames is not None:
        buttons = buttons[: max(0, int(max_frames))]

    start_mode = "poweron"
    env_state: str | None = "NONE"
    if state_path is not None:
        start_mode = f"state_path:{Path(state_path).name}"
        env_state = "NONE"
    elif state_name is not None and str(state_name).upper() not in ("", "NONE", "NULL"):
        start_mode = f"state:{state_name}"
        env_state = str(state_name)

    dump_kinds = frozenset(dump_states_on or ())
    states_dir_p = Path(states_dir) if states_dir else None
    if states_dir_p is not None:
        states_dir_p.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, env_state, game_dir, render_mode="rgb_array")
    try:
        env.reset()
        if state_path is not None:
            from retro_harness.env import read_state_bytes

            raw = read_state_bytes(state_path)
            env.em.set_state(raw)

        annotator = Annotator(stall_frames=stall_frames)
        timer = RoomTimer()
        series: list[FrameSample] = []
        state_dumps: list[dict[str, Any]] = []
        final_state: SuperMetroidState | None = None
        total = len(buttons)

        for i, fr in enumerate(buttons):
            # Critical: do NOT call sanitize_action — TAS uses L+R / angles.
            env.step(action_array(fr))
            frame_i = i + 1
            state = parse_env_state(env, frame=frame_i, mode=parse_mode)
            final_state = state
            names = frame_button_names(fr)

            new_events = annotator.observe(state, buttons=names)
            timer.observe(state)

            if keep_series and series_stride > 0 and (frame_i % series_stride == 0):
                series.append(FrameSample.from_state(state, names))

            should_dump = False
            dump_reason = ""
            if states_dir_p is not None:
                if dump_every > 0 and frame_i % dump_every == 0:
                    should_dump = True
                    dump_reason = f"every_{dump_every}"
                for ev in new_events:
                    if ev.kind in dump_kinds:
                        should_dump = True
                        dump_reason = ev.kind
                        break
            if should_dump and states_dir_p is not None:
                stem = f"f{frame_i:06d}_{dump_reason}_r{state.room_id:04X}"
                path = states_dir_p / f"{stem}.state"
                write_state_bytes(path, env.em.get_state())
                state_dumps.append(
                    {
                        "frame": frame_i,
                        "reason": dump_reason,
                        "path": str(path),
                        "pin": probe_pin(state),
                    }
                )

            if on_progress is not None and progress_every > 0:
                if frame_i % progress_every == 0 or frame_i == total:
                    on_progress(frame_i, total, state, len(annotator.events))

        timer.finalize(frame=len(buttons) if buttons else 0)
        room_report = timer.report(source=source)
        visits = [v.to_dict() for v in timer.visits]

        final_dict: dict[str, Any] = {}
        if final_state is not None:
            final_dict = {
                **probe_pin(final_state),
                "game_state": final_state.game_state,
                "phase": final_state.phase.value,
                "items": f"0x{final_state.collected_items:04X}",
                "beams": f"0x{final_state.collected_beams:04X}",
                "health": final_state.health,
                "max_health": final_state.max_health,
                "missiles": final_state.missiles,
                "max_missiles": final_state.max_missiles,
            }

        return MovieTrace(
            source=source,
            start_mode=start_mode,
            num_frames=len(frames) if max_frames is None else min(len(frames), len(buttons)),
            frames_played=len(buttons),
            events=list(annotator.events),
            rooms=visits,
            room_timer_report=room_report,
            series=series,
            final=final_dict,
            state_dumps=state_dumps,
            annotate_summary=annotator.summary(),
            meta={
                "parse_mode": parse_mode,
                "series_stride": series_stride,
                "stall_frames": stall_frames,
                "dump_states_on": sorted(dump_kinds),
                "dump_every": dump_every,
                "game": GAME,
            },
        )
    finally:
        env.close()


def write_trace_artifacts(
    trace: MovieTrace,
    out_dir: Path | str,
    *,
    write_series: bool = True,
    write_summary: bool = True,
) -> dict[str, Path]:
    """Write ``trace.json`` (+ optional series / summary) under *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Full artifact may be large; strip series into jsonl when requested.
    payload = trace.to_dict()
    series = payload.pop("series", [])
    payload["series_len"] = len(series)
    trace_path = out / "trace.json"
    trace_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    written["trace"] = trace_path

    if write_series and series:
        series_path = out / "series.jsonl"
        with series_path.open("w", encoding="utf-8") as fh:
            for row in series:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        written["series"] = series_path

    if write_summary:
        summary_path = out / "summary.json"
        summary_path.write_text(
            json.dumps(trace.summary(), indent=2) + "\n", encoding="utf-8"
        )
        written["summary"] = summary_path

    # Pins at room_enter / control for pure-card residual work.
    pins = [
        e.to_dict()
        for e in trace.events
        if e.kind
        in (
            "room_enter",
            "control",
            "item_gain",
            "beam_gain",
            "capacity_gain",
            "desync_suspect",
            "death",
            "ending",
        )
    ]
    pins_path = out / "pins.json"
    pins_path.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    written["pins"] = pins_path

    # Compact room timeline CSV for spreadsheet / agent skim.
    room_enters = [e for e in trace.events if e.kind == "room_enter"]
    if room_enters:
        csv_path = out / "room_timeline.csv"
        lines = ["frame,room_id_hex,pose,x,y,detail"]
        for e in room_enters:
            detail = e.detail.replace(",", ";")
            lines.append(
                f"{e.frame},0x{e.room_id:04X},{e.pose},{e.x},{e.y},{detail}"
            )
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["room_timeline"] = csv_path

    return written


__all__ = [
    "DEFAULT_OUT_ROOT",
    "FrameSample",
    "MovieTrace",
    "action_array",
    "frame_button_names",
    "resolve_frames",
    "trace_frames",
    "write_trace_artifacts",
]
