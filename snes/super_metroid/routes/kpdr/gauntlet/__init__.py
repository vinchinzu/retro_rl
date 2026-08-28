"""Morph+Bombs Gauntlet side-quest: post-Torizo Parlor → Gauntlet Entrance."""

from super_metroid.routes.kpdr.gauntlet.landing_to_gauntlet import (
    play_landing_to_gauntlet,
)
from super_metroid.routes.kpdr.gauntlet.parlor_to_landing import play_parlor_to_landing
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import PhaseStop


def play_parlor_to_gauntlet(
    session: ControllerSession,
    *,
    stop_at: str | None = None,
) -> None:
    """Post-BT Flyway door → Gauntlet Entrance (two rooms, Morph+Bombs)."""
    parlor_stops = {"flyway", "parlor_top"}
    if stop_at in parlor_stops:
        play_parlor_to_landing(session, stop_at=stop_at)
        return
    play_parlor_to_landing(session, stop_at=None)
    if stop_at == "landing":
        raise PhaseStop("landing", session.state, label="parlor_to_gauntlet")
    play_landing_to_gauntlet(session, stop_at=stop_at)


__all__ = [
    "play_landing_to_gauntlet",
    "play_parlor_to_gauntlet",
    "play_parlor_to_landing",
]
