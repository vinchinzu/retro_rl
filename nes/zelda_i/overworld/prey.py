"""What a body pays, what a hit costs, and whether either is worth a walk.

``overworld.hunt`` picked the manhattan-nearest live slot and chased it to a
per-target budget. On the pre-L1 corridor that is the wrong question twice
over, and the per-screen bill (``docs/PRE_L1.md``, ``tables1``) says so in
both directions:

* **Not every body is worth the same.** A blue tektite is ROM drop row 1
  (0.891 R/kill, the only table with two 5-rupees); a red octorok is row 0
  (0.156). Six of ``0x4A``'s tektites are 5.3R of the corridor's 8.4R, and
  the walk spent its health on the row-0 screens before it got there.
* **The streak is the actual wage.** Ten unbroken kills force a 5-rupee and
  sixteen force a fairy (``scratch/drop_mechanics_rom.md``), so *any* kill is
  worth :data:`STREAK_RUPEES` on top of its own table — which is more than
  row 0's own drop is worth. That is why "skip the red octoroks" cannot mean
  "walk past them": it means do not spend the screen budget walking *to* one.

The asset this module exists to protect is the streak, because the only
gameplay write of 0 to ``$0627`` is ``Link_BeHarmed``. One contact at streak
9 throws away 4.5R of forced-drop progress — more than every random drop the
five octorok screens paid put together.

No emulator and no snapshot: this is the arithmetic
``scratch/bomb_budget.py`` was working out, promoted to where policy can read
it. ``bomb_budget`` keeps the reporting CLI and imports the tables from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.dungeon.ids import (
    ARMOS_OBJECT_TYPE,
    BOULDER_OBJECT_TYPE,
    OBJECT_NAMES,
    ZORA_OBJECT_TYPE,
)
from zelda_i.ram import ZeldaObject

__all__ = [
    "BOMB",
    "CHASE_FLOOR",
    "CLOCK",
    "DROP_ROWS",
    "FAIRY",
    "FIVE_RUPEE",
    "FREE_REACH",
    "HEART",
    "RICH_CHASE_RADIUS",
    "RUPEE",
    "SKIP_TYPES",
    "STREAK_FAIRY",
    "STREAK_FORCED",
    "STREAK_RUPEES",
    "THRIFTY_BELOW_HEARTS",
    "THRIFTY_CHASE_RADIUS",
    "PreyPolicy",
    "drop_row",
    "forced_drop_kills",
    "kill_value",
    "prey_name",
    "random_rupees",
    "streak_forfeit",
]

BOMB, FIVE_RUPEE, RUPEE, CLOCK, HEART, FAIRY = 0x00, 0x0F, 0x18, 0x21, 0x22, 0x23
_RUPEE_VALUE = {RUPEE: 1, FIVE_RUPEE: 5}

# ``Z_04.asm`` ``DropItemRates`` + ``Types0..3``, transcribed in
# ``scratch/drop_mechanics_rom.md``. row -> (rate byte /256, 10-column table,
# ObjTypes that select the row). Anything unlisted falls to row 3, which is
# what the ROM does.
DROP_ROWS: dict[int, tuple[int, tuple[int, ...], tuple[int, ...]]] = {
    0: (
        0x50,
        (HEART, RUPEE, HEART, RUPEE, FAIRY, RUPEE, HEART, HEART, RUPEE, RUPEE),
        (0x07, 0x08, 0x0E, 0x04, 0x0F),
    ),
    1: (
        0x98,
        (FIVE_RUPEE, RUPEE, HEART, RUPEE, FIVE_RUPEE, HEART, CLOCK, RUPEE, RUPEE, RUPEE),
        (0x0D, 0x10, 0x21, 0x22, 0x13, 0x28, 0x2A),
    ),
    2: (
        0x68,
        (HEART, BOMB, RUPEE, CLOCK, RUPEE, HEART, BOMB, RUPEE, BOMB, HEART),
        (0x09, 0x0A, 0x03, 0x01),
    ),
    3: (
        0x68,
        (HEART, HEART, FAIRY, RUPEE, HEART, FAIRY, HEART, HEART, HEART, RUPEE),
        (),
    ),
}
_ROW_BY_TYPE = {t: row for row, (_, _, types) in DROP_ROWS.items() for t in types}

# ``SetUpDroppedItem``: ``$0627 == 16`` forces a fairy (and zeroes ``$50``
# only), ``$50 >= 10`` forces a 5-rupee. The fairy spends six kills of
# 5-rupee progress, so a clean streak pays at 10, 26, 36, 46 — not every ten.
STREAK_FORCED = 10
STREAK_FAIRY = 16
# Rupees one kill is worth purely for advancing ``$0627``/``$0050``, taken
# over the first 26 kills of a clean streak (two forced 5-rupees, one fairy):
# 10R / 26. Deliberately the *pessimistic* end of the 0.4-0.5 range — it is
# already larger than row 0's own drop, which is the whole point.
STREAK_RUPEES = 10.0 / 26.0


def drop_row(type_id: int) -> int:
    """``Types0..3`` row for an ObjType. Unlisted types fall to row 3."""
    return _ROW_BY_TYPE.get(int(type_id) & 0xFF, 3)


def random_rupees(row: int) -> float:
    """Expected rupees from one random drop roll on ``row``."""
    rate, table, _ = DROP_ROWS[int(row)]
    return (rate / 256.0) * (
        sum(_RUPEE_VALUE.get(item, 0) for item in table) / len(table)
    )


def kill_value(type_id: int) -> float:
    """Expected rupees from killing one body: its own table plus the streak.

    Both halves are real money and they are not the same size. Row 0's table
    pays 0.156; the streak pays :data:`STREAK_RUPEES` on *every* kill, red
    octorok included. A policy that reads only the table would walk past the
    cheap kills that fund the forced 5-rupee.
    """
    return random_rupees(drop_row(type_id)) + STREAK_RUPEES


def forced_drop_kills(limit: int = 50) -> tuple[int, ...]:
    """Kill numbers of a clean streak that force a 5-rupee (10, 26, 36 ...).

    ``$0627 == 16`` is tested first and forces a fairy, zeroing ``$0050``
    with six kills of progress still on it.
    """
    out: list[int] = []
    help_count = 0
    for world in range(1, int(limit) + 1):
        help_count = min(help_count + 1, STREAK_FORCED)
        if world == STREAK_FAIRY:
            help_count = 0
        elif help_count >= STREAK_FORCED:
            out.append(world)
            help_count = 0
    return tuple(out)


def streak_forfeit(streak: int) -> float:
    """Rupees a ``Link_BeHarmed`` throws away at this ``$0627`` value.

    Every kill already banked toward the next forced drop is worth
    :data:`STREAK_RUPEES` again, because it has to be paid for twice. At the
    measured pre-L1 peak of 7 that is ~2.7R — more than the five octorok
    screens' random drops put together.
    """
    return max(0, int(streak)) * STREAK_RUPEES


def prey_name(type_id: int) -> str:
    return OBJECT_NAMES.get(int(type_id) & 0xFF, f"unk_{int(type_id):#04x}")


# Bodies the wooden sword should never be walked toward on this corridor.
# A Zora is not a kill: ``UpdateZora`` submerges it on its own schedule
# (``DestroyMonster``, no ``HandleMonsterDied``), so the slot vanishing is not
# a streak tick and the chase is the walk that stands Link on its firing row.
# Armos and Boulder are drop row X in ``locations.py`` — no drop code at all.
SKIP_TYPES = frozenset(
    {ZORA_OBJECT_TYPE, ARMOS_OBJECT_TYPE, BOULDER_OBJECT_TYPE}
)
# A body this close is already paid for: one or two swings, no walk, and
# stepping away from it is how the reactive layer used to trade a kill for a
# frame of separation. Chebyshev, like every other contact test here.
FREE_REACH = 40
# Row 1 is 1.275 R/kill with the streak in it; rows 0, 2 and 3 are 0.541,
# 0.507 and 0.466. The gap is wide enough that the constant is not tuned to a
# boundary case — anything at or above a rupee a kill is row 1.
CHASE_FLOOR = 1.0
# **At full health a red octorok is worth chasing, and it is not close.**
# One chase frame costs the corridor 0.00137R of walk time (8.4R billed over
# 6117 live frames) plus 0.00101R of streak risk (4 hits over those frames at
# ``streak_forfeit`` of the mean streak, 4) — 0.0024R a frame against a body
# worth 0.541R. That pays back a 227 px walk for a red and 536 px for a blue
# tektite, and the interior box is only 182 px wide. There is no distance
# inside it at which walking away from a red is the better trade, and
# ``tables1`` measured nine of them landing 0.00 hearts. A flat "skip the red
# octoroks" radius would be tuning against the arithmetic; what the drop rows
# are for here is the *order*, not a refusal.
#
# **On short health the same chase stops paying at 113 px.** A contact then
# costs the streak *and* a heart, and the measured price of that heart is not
# half a heart of health: Link reached ``0x4A``'s six blue tektites — 5.3R of
# the corridor's 8.4R — on 1.49 hearts, spent 808 of the screen's 2401 frames
# guarding instead of killing, and left two of the six alive. 808 frames of
# walk time plus two row-1 bodies is 3.66R, which triples the cost of a chase
# frame to 0.0048R and halves every break-even with it.
#
# So the gate is health, not distance. Above ``thrifty_below_hearts`` every
# body in the box is fair game; at or below it a cheap body is only worth the
# walk inside ``THRIFTY_CHASE_RADIUS``, and a row-1 body still is not capped
# (its short-health break-even, 267 px, is wider than the box).
RICH_CHASE_RADIUS = 10**6
THRIFTY_CHASE_RADIUS = 113
# Whole hearts (``ram.whole_hearts``), not the raw ``$066F`` nibble. Link
# starts this corridor on 3 of 3, so this arms on the first whole heart lost.
THRIFTY_BELOW_HEARTS = 2


@dataclass(frozen=True)
class PreyPolicy:
    """Which body to hold, and how far the hunt may walk to hold it.

    Three gates and an order, all reading the same ROM drop row.

    :meth:`skipped` drops the bodies that are not kills at all. ``rich``
    (:meth:`chase_radius`) is the value gate, and it only tightens when Link
    is short of hearts — see :data:`THRIFTY_CHASE_RADIUS` for why distance
    alone does not justify walking away from a red octorok. ``budget`` is
    plain honesty: a chase longer than the frames left on this screen ends
    in a retire, so it should never start. :meth:`score` then orders whatever
    survives, which is where "focus on the tektites" actually happens.
    """

    free_reach: int = FREE_REACH
    chase_floor: float = CHASE_FLOOR
    rich_radius: int = RICH_CHASE_RADIUS
    thrifty_radius: int = THRIFTY_CHASE_RADIUS
    thrifty_below_hearts: int = THRIFTY_BELOW_HEARTS
    skip_types: frozenset[int] = SKIP_TYPES
    enabled: bool = True

    def skipped(self, obj: ZeldaObject) -> bool:
        """True for a body that is never a target at any distance."""
        return self.enabled and (int(obj.type_id) & 0xFF) in self.skip_types

    def rich(self, obj: ZeldaObject) -> bool:
        """True for a body on a drop row worth leaving a lane for."""
        return kill_value(int(obj.type_id)) >= self.chase_floor

    def chase_radius(self, obj: ZeldaObject, hearts: int = 99) -> int:
        """Chebyshev px the hunt may walk to reach this body.

        ``hearts`` is whole hearts (``ram.whole_hearts``), not the raw
        ``$066F`` nibble, which reads one low.
        """
        if not self.enabled:
            return self.rich_radius
        if self.skipped(obj):
            return 0
        if self.rich(obj) or int(hearts) > self.thrifty_below_hearts:
            return self.rich_radius
        return self.thrifty_radius

    def worth_chasing(
        self,
        obj: ZeldaObject,
        pad: int,
        *,
        hearts: int = 99,
        budget_left: int = 10**6,
    ) -> bool:
        """True when a body ``pad`` px away is worth holding as a target.

        ``free_reach`` is not a value test: a body already in contact range
        is answered with the blade whatever its drop row says, and that is
        the ladder in :meth:`ScreenHunter.step`, not a chase.
        """
        if self.skipped(obj):
            return False
        pad = int(pad)
        if pad <= self.free_reach:
            return True
        if self.enabled and pad > int(budget_left):
            # Link walks ~1 px/frame, so a body further away than the screen
            # has frames left cannot be reached before the budget retires it.
            return False
        return pad <= self.chase_radius(obj, hearts)

    def score(self, obj: ZeldaObject, pad: int) -> float:
        """Rupees per screen-frame, roughly: value over the walk that buys it.

        Link walks ~1 px/frame, so ``pad`` *is* the frame cost of the chase.
        The constant keeps a body already at Link's feet from dividing by
        zero and from outranking a richer one a few pixels further out.
        """
        return kill_value(int(obj.type_id)) / (float(max(int(pad), 0)) + 24.0)
