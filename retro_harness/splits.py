"""Generic segment timing / split tracking for speedrun games.

Replaces per-game implementations (DKC autosplit, Super Metroid Leaderboard)
with a single reusable tracker that handles PB detection and JSONL logging.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple

from retro_harness.recordings import append_jsonl


class SplitResult(NamedTuple):
    """Returned by on_segment_change for the completed segment."""
    segment_id: str
    elapsed_seconds: float | None
    is_pb: bool
    pb_seconds: float | None
    diff_seconds: float | None


class SplitTracker:
    """Track segment times with PB detection and JSONL logging."""

    def __init__(
        self,
        log_path: Path | None = None,
        best_times_path: Path | None = None,
        session_id: str | None = None,
    ) -> None:
        self.log_path = log_path
        self.best_times_path = best_times_path
        self.session_id = session_id or str(int(time.time()))
        self._splits: list[dict] = []
        self._current_segment: str | None = None
        self._current_name: str | None = None
        self._segment_start_time: float = time.monotonic()
        self._best_times: dict[str, float] = {}
        if best_times_path:
            self._load_best_times()

    def _load_best_times(self) -> None:
        if self.best_times_path and self.best_times_path.exists():
            try:
                self._best_times = json.loads(self.best_times_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._best_times = {}

    def _save_best_times(self) -> None:
        if not self.best_times_path:
            return
        self.best_times_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_times_path.write_text(
            json.dumps(self._best_times, indent=2) + "\n", encoding="utf-8")

    def on_segment_change(
        self, segment_id: str, *, frame: int | None = None,
        elapsed_seconds: float | None = None, segment_name: str | None = None,
        info: dict | None = None,
    ) -> SplitResult | None:
        """Record transition to a new segment.

        Args:
            segment_id: Unique segment identifier (e.g. level_id, room_name)
            frame: Current frame number (optional)
            elapsed_seconds: Wall/game-time seconds for the completed segment
            segment_name: Human-readable name (defaults to str(segment_id))
            info: Extra metadata dict to log

        Returns a SplitResult for the completed segment, or None if this is
        the first segment of the run.
        """
        result: SplitResult | None = None
        if self._current_segment is not None:
            result = self._finish_segment(
                elapsed_seconds=elapsed_seconds, frame=frame, info=info)
        self._current_segment = segment_id
        self._current_name = segment_name or str(segment_id)
        self._segment_start_time = time.monotonic()
        return result

    def finish_run(self, *, info: dict | None = None) -> None:
        """Mark the current run as complete. Log total time."""
        total = sum(s.get("elapsed_seconds", 0) or 0 for s in self._splits)
        entry: dict = {
            "event": "run_complete", "session_id": self.session_id,
            "total_seconds": round(total, 3), "segments": len(self._splits),
            "timestamp": time.time(),
        }
        if info:
            entry.update(info)
        if self.log_path:
            append_jsonl(self.log_path, entry)

    def hud_lines(self, max_lines: int = 4) -> list[str]:
        """Return formatted split times for HUD overlay."""
        lines: list[str] = []
        start = max(0, len(self._splits) - (max_lines - 1))
        for s in self._splits[start:]:
            name = s.get("segment_name", s["segment_id"])
            secs = s.get("elapsed_seconds")
            text = f"{name}: {secs:.1f}s" if secs is not None else f"{name}: --"
            pb = self._best_times.get(s["segment_id"])
            if pb is not None and secs is not None:
                diff = secs - pb
                sign = "+" if diff >= 0 else ""
                text += f" (PB: {pb:.1f}s {sign}{diff:.1f}s)"
            elif pb is not None:
                text += f" (PB: {pb:.1f}s)"
            lines.append(text)
        if self._current_segment is not None and len(lines) < max_lines:
            elapsed = time.monotonic() - self._segment_start_time
            name = self._current_name or str(self._current_segment)
            lines.append(f"Current: {name} +{elapsed:.1f}s")
        return lines

    @property
    def current_segment(self) -> str | None:
        return self._current_segment

    @property
    def splits(self) -> list[dict]:
        return list(self._splits)

    def _finish_segment(
        self, *, elapsed_seconds: float | None, frame: int | None,
        info: dict | None,
    ) -> SplitResult:
        seg_id = self._current_segment
        assert seg_id is not None
        seg_name = self._current_name or str(seg_id)
        pb = self._best_times.get(seg_id)
        is_pb = False
        diff: float | None = None
        if elapsed_seconds is not None:
            if pb is None or elapsed_seconds < pb:
                is_pb = True
                self._best_times[seg_id] = elapsed_seconds
                self._save_best_times()
            if pb is not None:
                diff = elapsed_seconds - pb
        split_record: dict = {
            "segment_id": seg_id, "segment_name": seg_name,
            "elapsed_seconds": elapsed_seconds, "frame": frame, "is_pb": is_pb,
        }
        self._splits.append(split_record)
        if self.log_path:
            entry: dict = {
                "event": "split", "session_id": self.session_id,
                "timestamp": time.time(), **split_record,
            }
            if info:
                entry.update(info)
            append_jsonl(self.log_path, entry)
        return SplitResult(
            segment_id=seg_id, elapsed_seconds=elapsed_seconds, is_pb=is_pb,
            pb_seconds=self._best_times.get(seg_id), diff_seconds=diff,
        )
