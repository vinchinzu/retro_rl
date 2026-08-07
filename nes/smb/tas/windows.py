"""Discover optimizable frame windows from a 1-1 SeedTrace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from smb.tas.trace import SeedTrace


@dataclass(frozen=True)
class TasWindow:
    """Half-open frame interval [start, end) to mutate during hill-climb."""

    start: int
    end: int
    label: str
    reason: str = ""
    priority: int = 0  # higher = try first

    def clamp(self, total: int) -> "TasWindow":
        s = max(0, min(self.start, total))
        e = max(s + 1, min(self.end, total)) if total > s else s
        return TasWindow(s, e, self.label, self.reason, self.priority)

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["length"] = self.length
        return d


# Hand-tuned isolated Level1_1 windows (measured 2026-08 on stairs/clear seeds).
# Indices are into isolated seeds starting at Level1_1 (not continuous settle=14).
ISOLATED_1_1_WINDOWS: tuple[TasWindow, ...] = (
    TasWindow(1050, 1280, "stairs", "end-stairs / pre-flag approach", priority=100),
    TasWindow(300, 520, "first-pipe", "first pipe approach / enter", priority=80),
    TasWindow(520, 900, "mid-level", "pits / mid pipes / enemies", priority=50),
    TasWindow(0, 200, "accel", "startup accel / first jump timing", priority=40),
    TasWindow(900, 1100, "pre-stairs", "approach to end stairs", priority=60),
)


def discover_windows(
    trace: SeedTrace,
    *,
    seed_len: int | None = None,
    include_static: bool = True,
    pad: int = 24,
    min_window: int = 30,
    max_windows: int = 8,
) -> list[TasWindow]:
    """Build a priority-sorted list of polish windows from *trace*.

    Combines static isolated-1-1 windows (clamped before flag) with dynamic
    regions around wall-slams, long zero-speed runs, and pre-flag stalls.
    """
    total = seed_len if seed_len is not None else trace.num_frames
    flag = trace.flag_frame or total
    # Never mutate post-flag automation — inputs are ignored / phase-risk
    hard_end = max(1, flag - 2)

    windows: list[TasWindow] = []
    seen: set[tuple[int, int]] = set()

    def _add(w: TasWindow) -> None:
        c = w.clamp(hard_end)
        if c.length < min_window:
            return
        key = (c.start, c.end)
        if key in seen:
            return
        # Reject pure post-flag
        if c.start >= hard_end:
            return
        seen.add(key)
        windows.append(c)

    if include_static:
        for w in ISOLATED_1_1_WINDOWS:
            _add(w)

    # Wall-slam clusters → windows centered on slam frames
    for slam in trace.wall_slams:
        start = max(0, slam.frame - pad * 2)
        end = min(hard_end, slam.frame + pad)
        _add(
            TasWindow(
                start,
                end,
                f"slam@{slam.frame}",
                f"wall_slam x={slam.x} {slam.detail}",
                priority=90,
            )
        )

    for run in trace.xs_zero_runs:
        if run.get("length", 0) < 6:
            continue
        start = max(0, int(run["start"]) - pad)
        end = min(hard_end, int(run["start"]) + int(run["length"]) + pad)
        _add(
            TasWindow(
                start,
                end,
                f"xs0@{run['start']}",
                f"xs_zero len={run['length']} x={run.get('x')}",
                priority=70,
            )
        )

    for stall in trace.stalls:
        if stall.get("length", 0) < min_window:
            continue
        start = max(0, int(stall["start"]) - 8)
        end = min(hard_end, int(stall["start"]) + int(stall["length"]) + 8)
        _add(
            TasWindow(
                start,
                end,
                f"stall@{stall['start']}",
                f"no_progress len={stall['length']}",
                priority=55,
            )
        )

    # Full pre-flag body as a low-priority fallback
    if hard_end > min_window:
        _add(TasWindow(0, hard_end, "pre-flag", "entire controllable body", priority=10))

    windows.sort(key=lambda w: (-w.priority, w.start))
    return windows[:max_windows]


def windows_from_labels(
    labels: Sequence[str],
    *,
    seed_len: int,
    flag_frame: int | None = None,
) -> list[TasWindow]:
    """Resolve CLI labels (``stairs``, ``first-pipe``, ``start:end``)."""
    hard_end = flag_frame - 2 if flag_frame else seed_len
    hard_end = max(1, min(hard_end, seed_len))
    by_label = {w.label: w for w in ISOLATED_1_1_WINDOWS}
    out: list[TasWindow] = []
    for lab in labels:
        lab = lab.strip()
        if not lab:
            continue
        if lab in by_label:
            out.append(by_label[lab].clamp(hard_end))
            continue
        if ":" in lab:
            a, b = lab.split(":", 1)
            out.append(TasWindow(int(a), int(b), lab, "cli", priority=100).clamp(hard_end))
            continue
        raise ValueError(
            f"unknown window {lab!r}; known: {sorted(by_label)} or start:end"
        )
    return out
