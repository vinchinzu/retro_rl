"""Full-run metrics, stage splits, and Hard credits detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from retro_harness.ram_state import GameState

METRIC_HOLD_FRAMES = 600
FINAL_SCENE_SETTLE_FRAMES = 1200
HARD_VALUE = 2
FINAL_CREDITS_EVENT = 0x1A

STAGE_NAMES = {
    0: "BIG APPLE",
    1: "ALLEYCAT BLUES",
    2: "SEWER SURFIN'",
    3: "TECHNODROME",
    4: "PREHISTORIC",
    5: "SKULL & CROSSBONES",
    6: "WOUNDED KNEE",
    7: "NEON NIGHT RIDERS",
    8: "STARBASE",
    9: "FINAL SHELL SHOCK",
}


@dataclass
class StageSplit:
    """First playable frame for one stage byte."""

    stage: int
    name: str
    frame: int
    elapsed_seconds: float


@dataclass
class RunMetrics:
    """Integrity and outcome metrics accumulated during a run."""

    total_damage_taken: int = 0
    max_single_frame_damage: int = 0
    health_guard_interventions: int = 0
    final_boss_iframe_guard_frames: int = 0
    life_losses: int = 0
    lives_start: int | None = None
    lives_peak: int | None = None
    lives_end: int | None = None
    min_health_seen: int | None = None
    credits_start_frame: int | None = None
    credits_complete_frame: int | None = None
    final_scene_start_frame: int | None = None
    hard_credits_event_seen: bool = False
    stage_splits: list[StageSplit] = field(default_factory=list)
    action_reasons: Counter[str] = field(default_factory=Counter)
    damage_by_stage: dict[int, int] = field(default_factory=dict)


class CreditsTracker:
    """Recognize the complete Hard staff/cast roll and final Splinter scene."""

    def __init__(self) -> None:
        self._last_playing = False
        self._stage9_playing_entries = 0

    def update(
        self,
        state: GameState,
        *,
        frame: int,
        metrics: RunMetrics,
    ) -> None:
        """Update credits evidence and completion frames."""
        event = int(state.extras.get("event", -1))
        menu = int(state.extras.get("menu", -1))
        if state.stage >= 10 and metrics.credits_start_frame is None:
            metrics.credits_start_frame = frame
        if metrics.credits_start_frame is None:
            return
        if event == FINAL_CREDITS_EVENT:
            metrics.hard_credits_event_seen = True

        playing = (
            event == FINAL_CREDITS_EVENT
            and state.stage == 9
            and menu == 6
            and state.player_x > 0
            and state.health > 0
        )
        if playing and not self._last_playing:
            self._stage9_playing_entries += 1
        self._last_playing = playing

        final_transition = (
            self._stage9_playing_entries >= 2
            and event == FINAL_CREDITS_EVENT
            and state.stage == 9
            and state.player_x == 0
        )
        if final_transition and metrics.final_scene_start_frame is None:
            metrics.final_scene_start_frame = frame
        if (
            metrics.final_scene_start_frame is not None
            and metrics.credits_complete_frame is None
            and frame
            >= metrics.final_scene_start_frame + FINAL_SCENE_SETTLE_FRAMES
        ):
            metrics.credits_complete_frame = frame


def format_duration(seconds: float) -> str:
    """Format an elapsed duration as HH:MM:SS.mmm."""
    millis = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def metrics_dict(metrics: RunMetrics, *, fps: float) -> dict[str, Any]:
    """Convert metrics to JSON-friendly values with readable timestamps."""
    payload = asdict(metrics)
    payload["action_reasons"] = dict(metrics.action_reasons.most_common())
    complete = metrics.credits_complete_frame
    if complete is not None:
        payload["power_on_to_credits_seconds"] = complete / fps
        payload["power_on_to_credits"] = format_duration(complete / fps)
    start = metrics.credits_start_frame
    if start is not None:
        payload["credits_start_seconds"] = start / fps
    return payload
