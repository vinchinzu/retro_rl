"""Shared guide polyline helpers and colors for KPDR human routes."""

from __future__ import annotations

from retro_harness.path_overlay import GuidePoint

# Crateria / extra rooms not always on the Norfair K4 room_ids export.
ROOM_BUBBLE_SAVE = 0xB0DD  # Bubble Mountain Save (wrong-door trap)
ROOM_PARLOR = 0x92FD  # Parlor and Alcatraz (post-Bomb Torizo)
ROOM_KIHUNTER = 0x948C
ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE
ROOM_WS_ENTRANCE = 0xCA08

_C = {
    "entrance": (120, 200, 255),
    "cathedral": (80, 255, 120),
    "rising": (255, 200, 80),
    "bubble": (200, 120, 255),
    "trap": (255, 60, 60),
    "bat": (255, 100, 100),
    "parlor": (80, 220, 255),
    "charge": (255, 180, 60),
    "ghz": (120, 255, 160),
    "spazer": (255, 140, 220),
    "below_spazer": (220, 90, 140),
    "speed": (80, 200, 255),
    "wave": (180, 120, 255),
    "ice": (100, 200, 255),
    "moat": (255, 160, 80),
    "west_ocean": (80, 200, 255),
    "ws": (200, 180, 100),
}


def _pts(*pairs: tuple[int, int, str]) -> tuple[GuidePoint, ...]:
    return tuple(GuidePoint(x, y, label) for x, y, label in pairs)
