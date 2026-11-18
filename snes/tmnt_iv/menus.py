"""Power-on boot through the real menus: Hard difficulty, then a turtle.

Frame-accurate schedule. Two DOWN presses enter Options, RIGHT sets Level to
Hard, two UP presses return to 1 Player, then 0-3 five-frame RIGHT bursts on
character select pick the turtle and START confirms. Control arrives at
frame 3520 for every turtle (measured 2026-11-14).

Turtle facts (power-on probe, 2026-11-14): starting HP differs and hit sizes
do not (4-24 per hit for everyone), so a bigger bar is free Clean margin.

    turtle  rights  char id  HP
    leo     0       2        80
    mike    1       4        96
    don     2       6        64
    raph    3       8        48
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action

TURTLES: dict[str, int] = {"leo": 0, "mike": 1, "don": 2, "raph": 3}
TURTLE_CHAR: dict[str, int] = {"leo": 2, "mike": 4, "don": 6, "raph": 8}
TURTLE_HP: dict[str, int] = {"leo": 80, "mike": 96, "don": 64, "raph": 48}
# Leo: the base policy's poke bands were tuned on his katana. Mike's 96 HP
# lost to reach at Metalhead (12 forced hits, 88 HP) where Leo took 0
# (power-on runs, 2026-11-14).
DEFAULT_TURTLE = "leo"

_MENU: dict[int, tuple[str, ...]] = {
    300: ("START",),
    700: ("DOWN",),
    720: ("DOWN",),
    750: ("START",),
    950: ("RIGHT",),
    1000: ("START",),
    1200: ("UP",),
    1220: ("UP",),
    1250: ("START",),
    1490: ("START",),
}
_BURSTS = ((1440, 1444), (1452, 1456), (1464, 1468))
BOOT_LAST = 1490


def boot_schedule(turtle: str = DEFAULT_TURTLE) -> dict[int, tuple[str, ...]]:
    """Frame -> buttons for one power-on boot to ``turtle`` on Hard."""
    rights = TURTLES[turtle]
    sched = dict(_MENU)
    for lo, hi in _BURSTS[:rights]:
        for frame in range(lo, hi + 1):
            sched[frame] = ("RIGHT",)
    return sched


def hard_boot_action(frame: int, turtle: str = DEFAULT_TURTLE) -> list[int]:
    """Scheduled menu input for one power-on frame."""
    names = boot_schedule(turtle).get(frame)
    return buttons(*names) if names else idle_action()
