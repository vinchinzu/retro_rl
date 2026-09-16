"""The full-health sword shot — the "flying sword" — as a ranged attack.

One ROM mechanism: its gate, its geometry, where to stand for it, and its
census. ``combat`` owns the blade's own hitbox; this owns the thing the blade
throws, because it is a *different* weapon to route around — it reaches the
far side of the screen and it switches itself off the first time Link is hit.

ROM (aldonunez ``Z_07.asm`` ``MakeSwordShot``): the shot lives in object slot
``$0E`` and spawns when the blade (slot ``$0D``) reaches state 3, the low
nibble of ``HeartValues`` equals the high nibble, and ``HeartPartial >= $80``.
``Z_01.asm CheckMonsterSwordShotOrMagicShotCollision`` hands it the blade's
own damage points ($10 wooden, $20 white, $40 magical) and the blade's damage
type (1), and a beam kill runs ``HandleMonsterDied`` — so it feeds
``WorldKillCount`` / ``HelpDropCount`` and drops exactly like a melee kill.
It is free reach, not a different economy.

Kinematics measured 2026-09-15 (``scratch/probe_beam.py``, screen 0x77, hp
``0x22``/``0xFF``): see :data:`BEAM_SPEED` and :data:`BEAM_MUZZLE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = [
    "BEAM_EDGE_MAX_X",
    "BEAM_EDGE_MIN_X",
    "BEAM_HALF_WIDTH",
    "BEAM_MUZZLE",
    "BEAM_PARTIAL_MIN",
    "BEAM_REACH",
    "BEAM_SPEED",
    "BEAM_SPREAD_FRAMES",
    "BEAM_STAND_FRAMES",
    "BEAM_STAND_OFF",
    "BeamCensus",
    "BeamPolicy",
    "beam_aim",
    "beam_live",
    "beam_offsets",
    "beam_ready",
    "beam_spawns",
    "beam_stand",
    "in_beam_lane",
]


# --- The full-health sword shot ("flying sword") ----------------------- #
# ROM (aldonunez ``Z_07.asm`` ``MakeSwordShot``): the shot lives in slot
# ``$0E`` and spawns when the blade (slot ``$0D``) reaches state 3, the low
# nibble of ``HeartValues`` equals the high nibble, and ``HeartPartial >=
# $80``. ``Z_01.asm CheckMonsterSwordShotOrMagicShotCollision`` gives it the
# blade's own damage points ($10 wooden) and damage type (1), and a beam kill
# runs ``HandleMonsterDied`` — so it feeds ``WorldKillCount`` /
# ``HelpDropCount`` and drops exactly like a melee kill. It is free reach, not
# a different weapon.
BEAM_PARTIAL_MIN = 0x80
# Measured 2026-09-15 (``scratch/probe_beam.py``, 0x77, hp 0x22/0xFF): 3.0
# px/frame in all four directions, flying state even ($10), spreading state
# odd ($11) for ~22f, and the shot flies until the room bound — there is no
# distance decay, so "range" is the screen.
BEAM_SPEED = 3.0
BEAM_SPREAD_FRAMES = 22
BEAM_REACH = 224
# Measured spawn offset from Link's ($70,$84). Vertical is asymmetric because
# Link's y is the top of his sprite.
BEAM_MUZZLE = {"RIGHT": (19, 0), "LEFT": (-19, 0), "UP": (0, -16), "DOWN": (0, 30)}
# ``DoObjectsCollideWithThresholds`` with threshold $0C on both axes, against
# monster mid ``(x+8, y+8)`` and shot mid ``(x+8, y+6)`` / ``(x+6, y+8)``: the
# true window is offset by 2 px. 9 is the symmetric half that always lands.
BEAM_HALF_WIDTH = 9
# Stand this far off a body to fire. Past ``MIN_DODGE_BODY`` and past a
# tektite hop, still inside the beam's screen-long reach.
BEAM_STAND_OFF = 56
BEAM_STAND_FRAMES = 150  # then close with the blade; a beam lane can be walled
# ``SetUpWeaponWithState``: a *horizontal* shot is placed, then deactivated on
# the spot if its x is < $14 or >= $EC. A hop arrives at x=0, so a travelling
# press near a scroll line is spent for nothing unless this is checked.
BEAM_EDGE_MIN_X = 0x14
BEAM_EDGE_MAX_X = 0xEC


def beam_live(snap: ZeldaSnapshot) -> bool:
    """True while slot ``$0E`` is occupied — the beam's own cooldown."""
    return int(snap.sword_shot.state) != 0


def beam_ready(snap: ZeldaSnapshot) -> bool:
    """True when the next A edge will spawn a sword shot.

    ``$0670`` is the gate that bites: ``Link_BeHarmed`` subtracts the damage
    from ``HeartPartial`` first, so one wooden octorok chip ($80) takes a full
    ``$FF`` to ``$7F`` and the beam is gone until a heart or fairy refills it.
    The beam is an *at full health* weapon, not a "most of the time" one.
    """
    return (
        int(snap.sword) >= 1
        and snap.health_is_full
        and int(snap.heart_partial) >= BEAM_PARTIAL_MIN
        and not beam_live(snap)
    )


def beam_spawns(link_x: int, direction: str) -> bool:
    """False where ``SetUpWeaponWithState`` kills the shot the frame it is made."""
    if direction not in ("LEFT", "RIGHT"):
        return True
    x = int(link_x) + BEAM_MUZZLE[direction][0]
    return BEAM_EDGE_MIN_X <= x < BEAM_EDGE_MAX_X


def beam_offsets(
    link_x: int, link_y: int, direction: str, enemy_x: int, enemy_y: int
) -> tuple[int, int] | None:
    """``(along, perpendicular)`` px from Link to the body for ``direction``.

    ``along`` is negative when the body is behind Link. ``None`` for a
    direction name the sword does not have.
    """
    dx = int(enemy_x) - int(link_x)
    dy = int(enemy_y) - int(link_y)
    if direction == "RIGHT":
        return dx, dy
    if direction == "LEFT":
        return -dx, dy
    if direction == "DOWN":
        return dy, dx
    if direction == "UP":
        return -dy, dx
    return None


def in_beam_lane(
    link_x: int,
    link_y: int,
    direction: str,
    enemy_x: int,
    enemy_y: int,
    *,
    reach: int = BEAM_REACH,
    half_width: int = BEAM_HALF_WIDTH,
) -> bool:
    """True if a shot fired ``direction`` from Link would cross the body.

    Walls are not modelled: ``MoveShot`` spreads the shot on a blocked cell,
    which costs the A edge and the slot, never health.
    """
    offsets = beam_offsets(link_x, link_y, direction, enemy_x, enemy_y)
    if offsets is None:
        return False
    along, perp = offsets
    return 0 < along <= int(reach) and abs(perp) <= int(half_width)


def beam_aim(
    link_x: int,
    link_y: int,
    bodies: Iterable[ZeldaObject],
    *,
    reach: int = BEAM_REACH,
    half_width: int = BEAM_HALF_WIDTH,
    prefer: str | None = None,
) -> tuple[str, ZeldaObject] | None:
    """Nearest body already standing in a beam lane, and the way to face it.

    ``prefer`` (Link's current facing) wins ties so the swing costs no turn.
    """
    best: tuple[str, ZeldaObject] | None = None
    best_cost = 10**9
    for obj in bodies:
        for direction in ("RIGHT", "LEFT", "UP", "DOWN"):
            if not beam_spawns(link_x, direction):
                continue
            if not in_beam_lane(
                link_x, link_y, direction, int(obj.x), int(obj.y),
                reach=reach, half_width=half_width,
            ):
                continue
            offsets = beam_offsets(link_x, link_y, direction, int(obj.x), int(obj.y))
            assert offsets is not None
            cost = offsets[0] - (1 if direction == prefer else 0)
            if cost < best_cost:
                best_cost = cost
                best = (direction, obj)
    return best


@dataclass
class BeamCensus:
    """What the shot did, kept off the policy's tuning surface.

    ``PreyPolicy`` is ``frozen=True`` with no counters on it and is the
    easiest object in this cluster to read; the counters are what stops the
    rest from being that. They live here so ``BeamPolicy``'s own fields are
    all knobs, and reach the outside through :meth:`BeamPolicy.report` (or
    the read-through properties the probes already name).
    """

    fired: int = 0
    aimed: int = 0  # ready *and* something in a lane: the press was available
    pressed: int = 0  # A edges spent on a shot; ``fired`` is what the ROM made of them
    impacts: int = 0
    ready_frames: int = 0
    stand_frames: int = 0
    by_screen: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.fired = 0
        self.aimed = 0
        self.pressed = 0
        self.impacts = 0
        self.ready_frames = 0
        self.stand_frames = 0
        self.by_screen.clear()

    def report(self) -> dict[str, Any]:
        return {
            "beam_fired": self.fired,
            "beam_aimed": self.aimed,
            "beam_pressed": self.pressed,
            "beam_impacts": self.impacts,
            "beam_ready_frames": self.ready_frames,
            "beam_stand_frames": self.stand_frames,
            "beam_fired_by_screen": {
                f"{k:#04x}": v for k, v in sorted(self.by_screen.items())
            },
        }


@dataclass
class BeamPolicy:
    """The sword shot as a weapon the hunt can plan around, plus its census.

    Sibling of ``hunt.ShotPolicy``: it owns one ROM mechanism and reports on
    it. :meth:`observe` is the honest count — slot ``$0E`` going live is a
    shot that exists, not an A press we hoped landed. :meth:`stand` is the
    only part with a budget: a lane can be walled, and a shot that never
    lands must not hold the chase open.

    Every field here is a knob; the accounting is on :class:`BeamCensus`.
    """

    enabled: bool = True
    stand_off: int = BEAM_STAND_OFF
    stand_max_frames: int = BEAM_STAND_FRAMES
    reach: int = BEAM_REACH
    half_width: int = BEAM_HALF_WIDTH
    census: BeamCensus = field(default_factory=BeamCensus)
    _state: int = field(default=0, repr=False)
    _stand_slot: int | None = field(default=None, repr=False)
    _stand_spent: int = field(default=0, repr=False)

    # Read-through to the census: probes and tests already name these
    # (``hunter.beam.pressed``, ``policy.by_screen``), so the counters stay
    # readable where they always were without sitting on the tuning surface.
    @property
    def fired(self) -> int:
        return self.census.fired

    @property
    def aimed(self) -> int:
        return self.census.aimed

    @property
    def pressed(self) -> int:
        return self.census.pressed

    @property
    def impacts(self) -> int:
        return self.census.impacts

    @property
    def ready_frames(self) -> int:
        return self.census.ready_frames

    @property
    def stand_frames(self) -> int:
        return self.census.stand_frames

    @property
    def by_screen(self) -> dict[int, int]:
        return self.census.by_screen

    def observe(self, snap: ZeldaSnapshot, box: tuple[int, int, int, int]) -> None:
        """Census slot ``$0E``: fires, and where each shot stopped flying.

        ``UpdateSwordShotOrMagicShot`` spreads the shot (odd state) the frame
        its move is blocked, so an even->odd flip well inside ``box`` is the
        shot meeting a body or a wall rather than running out the screen.
        """
        shot = snap.sword_shot
        was, now = int(self._state), int(shot.state)
        self._state = now
        census = self.census
        if self.enabled and beam_ready(snap):
            census.ready_frames += 1
        if was == 0 and now != 0:
            census.fired += 1
            screen = int(snap.screen)
            census.by_screen[screen] = census.by_screen.get(screen, 0) + 1
        elif was and was % 2 == 0 and now % 2 == 1:
            xlo, xhi, ylo, yhi = box
            if xlo <= int(shot.x) <= xhi and ylo <= int(shot.y) <= yhi:
                census.impacts += 1

    def ready(self, snap: ZeldaSnapshot) -> bool:
        return self.enabled and beam_ready(snap)

    def aim(
        self,
        snap: ZeldaSnapshot,
        bodies: Iterable[ZeldaObject],
        *,
        prefer: str | None = None,
    ) -> tuple[str, ZeldaObject] | None:
        """Which way to face so this frame's A edge also lands a shot."""
        aim = beam_aim(
            int(snap.link_x), int(snap.link_y), bodies,
            reach=self.reach, half_width=self.half_width, prefer=prefer,
        )
        if aim is not None:
            self.census.aimed += 1
        return aim

    def stand(
        self,
        snap: ZeldaSnapshot,
        target: ZeldaObject,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int] | None:
        """Where to wait for ``target``, or ``None`` once the budget is spent.

        The caller decides *which* bodies are worth waiting for (that is a
        behaviour question, and this module does not import behaviours).
        """
        slot = int(target.slot)
        if slot != self._stand_slot:
            self._stand_slot = slot
            self._stand_spent = 0
        self._stand_spent += 1
        if self._stand_spent > self.stand_max_frames:
            return None
        self.census.stand_frames += 1
        return beam_stand(
            int(snap.link_x), int(snap.link_y), target, box,
            stand_off=self.stand_off,
        )

    def enter(self) -> None:
        """New screen: the held stand target does not survive a scroll."""
        self._stand_slot = None
        self._stand_spent = 0

    def press(self) -> None:
        """An A edge was spent on a shot. ``fired`` says whether one spawned."""
        self.census.pressed += 1

    def reset(self) -> None:
        self.census.reset()
        self._state = 0
        self.enter()

    def report(self) -> dict[str, Any]:
        return self.census.report()


def beam_stand(
    link_x: int,
    link_y: int,
    obj: ZeldaObject,
    box: tuple[int, int, int, int],
    *,
    stand_off: int = BEAM_STAND_OFF,
) -> tuple[int, int]:
    """Cell in the body's own row/column, ``stand_off`` px away on Link's side.

    The point of the beam is that this cell is *not* contact range: a blue
    tektite's hop is what costs health on the coast, and a body lined up at
    56 px dies without ever reaching Link.
    """
    ox, oy = int(obj.x), int(obj.y)
    xlo, xhi, ylo, yhi = box
    dx, dy = int(link_x) - ox, int(link_y) - oy
    if abs(dx) >= abs(dy):
        cell = (ox + (stand_off if dx >= 0 else -stand_off), oy)
    else:
        cell = (ox, oy + (stand_off if dy >= 0 else -stand_off))
    return (max(xlo, min(xhi, cell[0])), max(ylo, min(yhi, cell[1])))
