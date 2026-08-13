"""Calibrated REST-band corrections applied after stroke-index plans.

Later matching rules win, matching the historical sequential ``if`` chain.
Keep this table next to the other route overlays — ``shot_policy`` only
applies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hals_golf.tasks.menus import ClubSet

if TYPE_CHECKING:
    from hals_golf.tasks.profile import MissionProfile
    from hals_golf.tasks.shot_policy import ShotSituation


@dataclass(frozen=True)
class RestBandOverride:
    """Ordered REST-band correction applied after stroke-index plans.

    ``requires_vs_hal``:
      * ``True`` — VS HAL only (default; preserves older match-only fixes)
      * ``False`` — stroke play only
      * ``None`` — both modes (leave-shaped metal recoveries)
    """

    hole: int
    rest_min: int
    rest_max: int
    club_downs: int
    power: int
    aim: int
    club_set: ClubSet | None = None
    lie: int | None = None
    requires_vs_hal: bool | None = True

    def matches(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> bool:
        if self.requires_vs_hal is True:
            if not profile.is_vs_hal:
                return False
        elif self.requires_vs_hal is False:
            if profile.is_vs_hal:
                return False
        if self.club_set is not None and profile.club_set is not self.club_set:
            return False
        if situation.hole != self.hole:
            return False
        if not (self.rest_min <= situation.rest <= self.rest_max):
            return False
        if self.lie is not None and situation.lie != self.lie:
            return False
        return True


# Calibrated VS HAL / metal REST-band fixes. Later matches win.
REST_BAND_OVERRIDES: tuple[RestBandOverride, ...] = (
    RestBandOverride(6, 25, 40, 0, 28, 4, lie=2),
    RestBandOverride(
        6, 70, 85, 12, 30, 4, club_set=ClubSet.METAL, lie=0
    ),
    RestBandOverride(
        2, 85, 100, 11, 32, 0, club_set=ClubSet.METAL, lie=2
    ),
    # Stroke metal H3: after 4W leave ~62y (lie=0), 7I hangs; soft driver
    # 38/-2 reaches a one-yard green.
    RestBandOverride(
        3,
        55,
        75,
        0,
        38,
        -2,
        club_set=ClubSet.METAL,
        lie=0,
        requires_vs_hal=False,
    ),
    # Leave-shaped metal recoveries: apply in stroke play and VS HAL.
    RestBandOverride(
        7,
        240,
        270,
        1,
        44,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        7,
        130,
        160,
        0,
        44,
        -2,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        7,
        18,
        35,
        0,
        26,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        7,
        190,
        220,
        3,
        44,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    # Stroke-play metal H8: fairway leave after the bunker-avoiding tee.
    RestBandOverride(
        8,
        95,
        120,
        9,
        38,
        0,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    # VS HAL metal: SW from ~100–115y fairway.
    RestBandOverride(
        8,
        100,
        115,
        12,
        40,
        2,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=True,
    ),
    # Short H8 leftovers after a long iron — soft SW chip.
    RestBandOverride(
        8,
        40,
        70,
        12,
        32,
        2,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    # H9 metal: short fairway/bunker leftovers after a missed green — soft
    # PW, not SW (SW at ~27y hung the Jul-21 clear for 100k+ frames).
    RestBandOverride(
        9,
        15,
        40,
        11,
        32,
        0,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        9,
        15,
        40,
        11,
        32,
        0,
        club_set=ClubSet.METAL,
        lie=3,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        10,
        120,
        160,
        10,
        38,
        0,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        11,
        200,
        240,
        1,
        44,
        -8,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        11,
        140,
        160,
        9,
        38,
        0,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        11,
        80,
        100,
        12,
        32,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=None,
    ),
    RestBandOverride(
        11,
        50,
        70,
        12,
        30,
        4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        12,
        120,
        150,
        8,
        38,
        -2,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        12,
        20,
        30,
        0,
        26,
        -8,
        club_set=ClubSet.METAL,
        lie=0,
        requires_vs_hal=None,
    ),
    RestBandOverride(7, 230, 280, 0, 36, 6, lie=0),
    # Stroke-play H13 birdie leaves H14 ~172y (baseline ~163y). The stock 9I
    # stops ~107y; a second driver reaches ~48y for the SW finish below.
    RestBandOverride(
        14, 168, 185, 0, 42, 0, lie=2, requires_vs_hal=False
    ),
    RestBandOverride(
        14, 40, 55, 12, 30, 3, lie=2, requires_vs_hal=False
    ),
    RestBandOverride(
        15,
        150,
        180,
        1,
        42,
        -4,
        club_set=ClubSet.METAL,
        lie=0,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        15,
        150,
        180,
        1,
        42,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    # After H13 birdie the H15 second shot stops ~100y fairway; stock
    # driver finish finds sand. Soft SW reaches a one-yard tap-in.
    RestBandOverride(
        15, 95, 110, 12, 39, 2, lie=2, requires_vs_hal=False
    ),
    RestBandOverride(
        16,
        250,
        290,
        0,
        42,
        4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        16,
        130,
        170,
        0,
        44,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        16,
        18,
        35,
        0,
        26,
        -8,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    RestBandOverride(
        18,
        165,
        220,
        0,
        44,
        -5,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
    # After H13 birdie the H18 third shot is ~35y fairway; stock SW misses.
    # Soft driver reaches a one-yard tap-in.
    RestBandOverride(
        18, 30, 40, 0, 29, 3, lie=2, requires_vs_hal=False
    ),
    # Stroke-play metal H6: recover the ~142–175y fairway leave with PW.
    RestBandOverride(
        6,
        130,
        185,
        11,
        42,
        -4,
        club_set=ClubSet.METAL,
        lie=2,
        requires_vs_hal=False,
    ),
)
