"""Mission profile: play mode, club set, and difficulty axis.

Harder difficulties (Pro / tournament) should extend ``Difficulty`` and select
route tables via ``MissionProfile``, not new bools bolted onto the mission FSM.
"""

from __future__ import annotations

from dataclasses import dataclass

from hals_golf.tasks.menus import ClubSet, Difficulty, PlayMode
from hals_golf.tasks.routes.tables import VS_HAL_MATCH_HOLES

__all__ = [
    "Difficulty",
    "MissionProfile",
    "resolve_club_set",
]


@dataclass(frozen=True)
class MissionProfile:
    """Immutable mode / club / difficulty contract for planning + bootstrap."""

    play_mode: PlayMode = PlayMode.STROKE_PLAY
    club_set: ClubSet = ClubSet.STANDARD
    difficulty: Difficulty = Difficulty.AMATEUR
    max_holes: int | None = None

    @property
    def is_vs_hal(self) -> bool:
        return self.play_mode is PlayMode.VS_HAL

    @property
    def uses_metal(self) -> bool:
        return self.club_set is ClubSet.METAL

    @property
    def is_pro(self) -> bool:
        return self.difficulty is Difficulty.PRO

    def resolved_max_holes(self) -> int:
        if self.max_holes is not None:
            return self.max_holes
        if self.is_vs_hal:
            return VS_HAL_MATCH_HOLES
        return 18


def resolve_club_set(
    *,
    club_set_arg: str,
    play_mode: PlayMode,
    skip_bootstrap: bool,
) -> ClubSet:
    """Map CLI club-set choice to a concrete ClubSet.

    ``auto`` uses metal only for a fresh VS HAL boot so existing in-round
    standard-club save states stay on the verified calibration.
    """
    if club_set_arg == "metal":
        return ClubSet.METAL
    if club_set_arg == "standard":
        return ClubSet.STANDARD
    if play_mode is PlayMode.VS_HAL and not skip_bootstrap:
        return ClubSet.METAL
    return ClubSet.STANDARD
