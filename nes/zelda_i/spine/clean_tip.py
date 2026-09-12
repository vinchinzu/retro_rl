"""The Clean campaign ladder: what is proven, what is blocked, and why.

Clean progress lived in six residual documents and a hand-carried note
("W1 open, W2/W3/W4 blocked"). Working out the next open hop meant re-reading
all of them, and the blocker classes were prose, so three lanes spent a
sitting each on the same root cause without noticing it was the same.

This table is that state as rows. It is a *status ladder*, not a dispatcher —
the Composer is still ``spine/hops.py``. Rungs follow the evidence ladder in
``docs/tasks/rr-npv-clean-parallel.md``; a row only moves up when its lane's
residual says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

__all__ = (
    "Rung",
    "Blocker",
    "CleanStep",
    "CLEAN_LADDER",
    "TOOL_FOR_BLOCKER",
    "tip",
    "next_open",
    "blocked",
    "by_blocker",
    "tool_for",
    "render",
)


class Rung(IntEnum):
    """Evidence ladder. Parallel lanes stop at FIXTURE_LIVE; the integrator
    promotes."""

    HYPOTHESIS = 1
    FIXTURE_LIVE = 2
    NATURAL = 3
    SPINE_GREEN = 4


class Blocker(str, Enum):
    """Why a row is not on its next rung — the class, not the symptom.

    The first three are answered by ``dungeon.threat``; naming them is what
    tells the next sitting to stop writing another position table.
    """

    NONE = "none"
    SHOT_UNDODGEABLE = "shot_undodgeable"
    BODY_UNDODGEABLE = "body_undodgeable"
    FIRING_LINE = "firing_line"
    OCCUPANCY_STALL = "occupancy_stall"
    INVENTORY_GAP = "inventory_gap"
    PICKUP_MISS = "pickup_miss"


TOOL_FOR_BLOCKER: dict[Blocker, str] = {
    Blocker.SHOT_UNDODGEABLE: (
        "threat.dodgeable is False at the stand cell — no peel tuning can "
        "work. Answer with threat.off_line_step (shoot and scoot) or a "
        "stand cell that was never on the axis."
    ),
    Blocker.BODY_UNDODGEABLE: (
        "threat.MIN_DODGE_BODY (16 frames) is not available on the hold "
        "line. Peel earlier off tracking velocity, or trade the sword."
    ),
    Blocker.FIRING_LINE: (
        "threat.in_firing_line is true before the shot spawns; step off the "
        "shooter's axis instead of reacting to the beam."
    ),
    Blocker.OCCUPANCY_STALL: (
        "walk.physics occupancy: miss → block cell → replan; no path → "
        "stand. Dump the tile map (dungeon.tilemap), never $049E."
    ),
    Blocker.INVENTORY_GAP: (
        "The pin cannot hold the item. Needs a natural-segment entry, not a "
        "poke."
    ),
    Blocker.PICKUP_MISS: (
        "Dest is RAM: walk the pickup tile axis-first, then confirm the "
        "inventory delta before leaving."
    ),
    Blocker.NONE: "",
}


@dataclass(frozen=True)
class CleanStep:
    """One Clean segment and the evidence standing behind it."""

    id: str
    bead: str
    segment: str
    rung: Rung
    blocker: Blocker = Blocker.NONE
    room: str = ""
    pose: str = ""
    residual: str = ""
    note: str = ""

    @property
    def open(self) -> bool:
        return self.rung < Rung.SPINE_GREEN

    @property
    def proven(self) -> bool:
        return self.rung >= Rung.FIXTURE_LIVE and self.blocker is Blocker.NONE


# Ordered by run order, not by lane number. Sources are the lane residuals
# under docs/tasks; keep each row's `residual` pointing at its own.
CLEAN_LADDER: tuple[CleanStep, ...] = (
    CleanStep(
        id="l1_tf",
        bead="M5",
        segment="power-on → L1 Triforce",
        rung=Rung.NATURAL,
        blocker=Blocker.PICKUP_MISS,
        room="L1 0x33",
        pose="(88,165) health 0x21 keys 1 deaths 0, 6000f cap at 16648",
        residual="docs/tasks/rr-npv-reactive-combat.md",
        note=(
            "MEASURED RED 2026-09-11, 2/2 deterministic: clear52→clear42→"
            "exit42→clear43 pass, clear33_key times out. Link stands on the "
            "0x33 key tile row and mashes RIGHT into the east block instead "
            "of dropping south. Room33ScoopController._scoop_if_low walks a "
            "4-way delta toward the heart drop with no occupancy awareness, "
            "so it can walk into a wall forever and never re-check clear."
        ),
    ),
    CleanStep(
        id="l1_exit_ow_l2",
        bead="rr-ps7.3",
        segment="L1 leave → OW walk → L2 east mouth 0x4C",
        rung=Rung.NATURAL,
        blocker=Blocker.OCCUPANCY_STALL,
        room="L1 0x23 / OW 0x4C",
        pose="0x23 (144,149) 4627 occupancy misses / 6000f",
        residual="docs/tasks/rr-8t4.4-residual.md",
        note="door hops green in isolation; power-on red before the hop",
    ),
    CleanStep(
        id="l2_tf",
        bead="rr-4oz",
        segment="L2 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        room="L2 0x0e / 0x4f",
        residual="docs/tasks/rr-4oz-residual.md",
        note="Dodongo bomb placement on stable mouth only",
    ),
    CleanStep(
        id="l3_tf",
        bead="rr-npv.1",
        segment="L3 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        residual="docs/tasks/rr-npv.1-residual.md",
        note="all dest hops green, TF 0x04, deaths 0",
    ),
    CleanStep(
        id="l4_tf",
        bead="rr-bxzj",
        segment="L4 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        residual="docs/tasks/rr-bxzj-residual.md",
        note="31 contiguous stages, deaths 0, heart-safe Gleeok",
    ),
    CleanStep(
        id="l5_tf",
        bead="rr-npv.2",
        segment="L5 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        blocker=Blocker.BODY_UNDODGEABLE,
        room="L5 0x77",
        pose="(120,173) mode 17, third time on the hold line",
        residual="docs/tasks/rr-npv.2-residual.md",
        note="Pols Voice lands on the stand cell; peel starts too late",
    ),
    CleanStep(
        id="l6_tf",
        bead="rr-d6v",
        segment="L6 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        blocker=Blocker.FIRING_LINE,
        room="L6 0x78",
        pose="(144,141) mode 17, 3/3 on the east waist",
        residual="docs/tasks/rr-d6v-residual.md",
        note=(
            "0x78 is a 5-wizzrobe crossfire, but the health is spent "
            "upstream: postmortem counts 6 hits in the *green* "
            "level6_east_key_0x7a stage (4x 0x24 from the east)"
        ),
    ),
    CleanStep(
        id="l7_tf",
        bead="rr-npv.3",
        segment="L7 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        blocker=Blocker.INVENTORY_GAP,
        room="L7 hungry Goriya",
        pose="Food 0 from the recon pin",
        residual="docs/tasks/rr-npv.3-residual.md",
        note="needs the natural bait-shop hop; never poke ADDR_FOOD",
    ),
    CleanStep(
        id="l8_tf",
        bead="rr-npv.4",
        segment="L8 Entrance → Triforce",
        rung=Rung.FIXTURE_LIVE,
        blocker=Blocker.SHOT_UNDODGEABLE,
        room="L8 0x1E",
        pose="(128,181) mode 17, 3/3 oscillating 128↔112",
        residual="docs/tasks/rr-npv.4-residual.md",
        note="Blue Gohma fires down the arrow-alignment column",
    ),
    CleanStep(
        id="l9_credits",
        bead="rr-npv.5",
        segment="L9 Entrance → Ganon → credits",
        rung=Rung.FIXTURE_LIVE,
        residual="docs/tasks/rr-npv.5-residual.md",
        note="Patra / Ganon / credits green from recon pins, deaths 0",
    ),
)


def tip() -> CleanStep | None:
    """Deepest row still contiguous with power-on proof.

    ``None`` when the very first row is not green — the honest answer when
    the power-on run itself is red. Do not report a tip the ROM does not
    support.
    """
    last: CleanStep | None = None
    for step in CLEAN_LADDER:
        if step.rung < Rung.SPINE_GREEN:
            return last
        last = step
    return last


def next_open() -> CleanStep | None:
    """First row below spine-green: the hop a sitting should take next."""
    return next((step for step in CLEAN_LADDER if step.open), None)


def blocked() -> tuple[CleanStep, ...]:
    return tuple(step for step in CLEAN_LADDER if step.blocker is not Blocker.NONE)


def by_blocker() -> dict[Blocker, tuple[CleanStep, ...]]:
    """Blocked rows grouped by root cause, so shared causes are visible."""
    groups: dict[Blocker, list[CleanStep]] = {}
    for step in blocked():
        groups.setdefault(step.blocker, []).append(step)
    return {key: tuple(value) for key, value in groups.items()}


def tool_for(blocker: Blocker) -> str:
    return TOOL_FOR_BLOCKER.get(blocker, "")


def render() -> str:
    """Plain-text ladder for a CLI or a residual paste."""
    top = tip()
    head = (
        f"clean tip: {top.id} ({top.segment})"
        if top is not None
        else "clean tip: NONE — the power-on run is red at the first row"
    )
    lines = [
        head,
        "",
        f"{'id':<16} {'rung':<13} {'blocker':<18} room / pose",
    ]
    for step in CLEAN_LADDER:
        where = f"{step.room} {step.pose}".strip()
        lines.append(
            f"{step.id:<16} {step.rung.name.lower():<13} "
            f"{step.blocker.value:<18} {where}"
        )
    nxt = next_open()
    if nxt is not None:
        lines += ["", f"next open: {nxt.id} — {nxt.segment}"]
        if nxt.note:
            lines.append(f"  note: {nxt.note}")
        if nxt.blocker is not Blocker.NONE:
            lines.append(f"  tool: {tool_for(nxt.blocker)}")
    groups = by_blocker()
    if groups:
        lines += ["", "blockers by class:"]
        for blocker, steps in groups.items():
            ids = ", ".join(step.id for step in steps)
            lines.append(f"  {blocker.value}: {ids}")
            lines.append(f"    {tool_for(blocker)}")
    return "\n".join(lines)
