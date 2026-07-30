"""Opt-in screen-timing session for Metroid segment runners.

Reuses :class:`metroid.screen_timer.ScreenTimer` as a passive observer. Does
not own input policy, route graphs, or controller state machines — runners
call :meth:`observe_env` after each ``env.step`` when timing is enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from metroid.paths import SCREEN_TIMINGS_DIR
from metroid.ram import read_snapshot
from metroid.screen_timer import ScreenTimer, ScreenVisit


@dataclass
class PhaseMarker:
    """Controller/route phase annotation at an absolute emulator frame."""

    frame: int
    phase: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "phase": self.phase,
            "note": self.note,
        }


@dataclass
class ScreenTimingSession:
    """Passive hop timer attached to a natural morph / first-missiles run.

    Frame index is the absolute emulator step count for the session (boot +
    morph + missiles when natural-entry is used). Controllers keep their own
    relative frame counters unchanged.
    """

    enabled: bool = True
    source: str = ""
    entry_mode: str = ""  # natural | after_morph | level1
    diagnostic_state_load: str | None = None
    timer: ScreenTimer = field(default_factory=ScreenTimer)
    absolute_frame: int = 0
    phase_markers: list[PhaseMarker] = field(default_factory=list)
    _last_phase: str | None = field(default=None, repr=False)
    completed_visits: list[ScreenVisit] = field(default_factory=list)

    def observe_env(
        self,
        env: Any,
        *,
        phase: str | None = None,
        note: str = "",
    ) -> ScreenVisit | None:
        """Read RAM after a step and feed the timer. No-op when disabled."""
        if not self.enabled:
            return None
        self.absolute_frame += 1
        if phase is not None and phase != self._last_phase:
            self.phase_markers.append(
                PhaseMarker(
                    frame=self.absolute_frame,
                    phase=phase,
                    note=note,
                )
            )
            self._last_phase = phase
        snap = read_snapshot(env.get_ram(), env=env)
        visit = self.timer.observe(snap, frame=self.absolute_frame)
        if visit is not None:
            self.completed_visits.append(visit)
        return visit

    def finalize(self) -> None:
        if not self.enabled:
            return
        self.timer.finalize(frame=self.absolute_frame)

    def bottleneck_summary(self) -> dict[str, Any]:
        """Identify the longest completed hop and open-visit dwell.

        Used to pick the first *measured* screen-level bottleneck on a timed
        run. Does not invent route progress past the controller frontier.
        """
        visits = self.timer.visits
        open_visit = None
        if self.timer._open is not None:
            ov = self.timer._open
            open_dwell = self.absolute_frame - ov.entry_frame
            open_visit = {
                "map_cell": [ov.map_x, ov.map_y],
                "entry_frame": ov.entry_frame,
                "open_dwell_frames": open_dwell,
                "in_transition": ov.in_transition,
            }

        if not visits:
            return {
                "visit_count": 0,
                "longest_by_screen_frames": None,
                "longest_by_dwell_frames": None,
                "open_visit": open_visit,
                "phase_markers": [m.to_dict() for m in self.phase_markers],
                "interpretation": (
                    "no completed map-cell hops; see open_visit / discontinuities"
                ),
            }

        by_screen = max(visits, key=lambda v: v.screen_frames)
        by_dwell = max(visits, key=lambda v: v.dwell_frames)
        return {
            "visit_count": len(visits),
            "longest_by_screen_frames": {
                "map_cell": list(by_screen.map_cell),
                "dest_map_cell": list(by_screen.dest_map_cell),
                "screen_frames": by_screen.screen_frames,
                "dwell_frames": by_screen.dwell_frames,
                "transition_frames": by_screen.transition_frames,
                "sequence_index": by_screen.sequence_index,
                "entry_frame": by_screen.entry_frame,
                "exit_frame": by_screen.exit_frame,
            },
            "longest_by_dwell_frames": {
                "map_cell": list(by_dwell.map_cell),
                "dest_map_cell": list(by_dwell.dest_map_cell),
                "screen_frames": by_dwell.screen_frames,
                "dwell_frames": by_dwell.dwell_frames,
                "transition_frames": by_dwell.transition_frames,
                "sequence_index": by_dwell.sequence_index,
                "entry_frame": by_dwell.entry_frame,
                "exit_frame": by_dwell.exit_frame,
            },
            "open_visit": open_visit,
            "phase_markers": [m.to_dict() for m in self.phase_markers],
            "total_screen_frames": sum(v.screen_frames for v in visits),
            "total_dwell_frames": sum(v.dwell_frames for v in visits),
            "total_transition_frames": sum(v.transition_frames for v in visits),
        }

    def report(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "source": self.source,
            }
        # Capture open-visit dwell before session finalize abandons it.
        bottleneck = self.bottleneck_summary()
        self.finalize()
        payload_extra: dict[str, Any] = {
            "entry_mode": self.entry_mode,
            "absolute_frames": self.absolute_frame,
            "phase_markers": [m.to_dict() for m in self.phase_markers],
            "bottleneck": bottleneck,
            "evaluation_class": (
                "clean_natural_entry"
                if self.entry_mode == "natural" and not self.diagnostic_state_load
                else "diagnostic_state_load"
                if self.diagnostic_state_load
                else self.entry_mode or "unspecified"
            ),
        }
        if self.diagnostic_state_load:
            payload_extra["diagnostic_state_load"] = self.diagnostic_state_load
            payload_extra["diagnostic_note"] = (
                "Development save-state load — not Clean natural-entry evidence. "
                "Use for timing diagnostics only."
            )
        if extra:
            payload_extra.update(dict(extra))
        report = self.timer.report(source=self.source, extra=payload_extra)
        report["enabled"] = True
        return report

    def write_report(
        self,
        path: Path | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write timing JSON under ``SCREEN_TIMINGS_DIR``. Returns path or None."""
        if not self.enabled:
            return None
        report = self.report(extra=extra)
        if path is None:
            mode = self.entry_mode or "session"
            path = SCREEN_TIMINGS_DIR / f"first_missiles_{mode}_timing.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return path


def default_timing_artifact_path(entry_mode: str) -> Path:
    return SCREEN_TIMINGS_DIR / f"first_missiles_{entry_mode}_timing.json"
