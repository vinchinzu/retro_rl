"""Mission/status helpers for bots that support human takeover and resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MissionSnapshot:
    """Human-readable mission state for HUD/debug output."""

    mission_id: str
    phase: str
    objective: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        text = f"{self.mission_id}: {self.phase}"
        if self.objective:
            return f"{text} | {self.objective}"
        return text


class MissionAware(Protocol):
    """Optional bot protocol for human/autopilot handoff."""

    def mission_status(self) -> MissionSnapshot | str | None:
        ...

    def on_human_takeover(self) -> None:
        ...

    def on_autopilot_resume(self) -> None:
        ...
