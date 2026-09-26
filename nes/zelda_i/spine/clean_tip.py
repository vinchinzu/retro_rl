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
    "SHARED_MECHANISMS",
    "TOOL_FOR_BLOCKER",
    "adoption",
    "tip",
    "next_open",
    "blocked",
    "by_blocker",
    "render_adoption",
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
    # The lane is not blocked by the ROM; it is blocked by what it measures
    # against. Distinct from INVENTORY_GAP (a pin that is merely poorer than a
    # real arrival): here the pin holds a state normal play cannot reach, so
    # every number taken from it — hearts above all — is against a fake
    # denominator and no amount of room tuning can be trusted.
    INVALID_PIN = "invalid_pin"


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
    Blocker.INVALID_PIN: (
        "Check the pin before the room. $066F is hi=containers-1, lo=whole "
        "hearts, so a coherent byte always has lo <= hi (ram.full_health_byte). "
        "Rebuild from a measured arrival — scripts/fixtures/"
        "capture_level6_entrance_fixture.py is the pattern — then re-measure; "
        "do not tune a room against the old numbers."
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
    # The fixture this row's evidence was measured from. Named because an
    # invalid one is invisible otherwise: rr-d6v spent four sittings tuning
    # L6 rooms against a pin holding 15 hearts in 3 containers, and the pin
    # was nowhere in this table.
    pin: str = ""

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
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="L1 0x36",
        pose="triforce=0x01 end 18909",
        pin="power-on (no pin)",
        residual="docs/tasks/rr-npv-reactive-combat.md",
        note=(
            "2026-09-14 (later): 2/2 natural-entry, triforce=0x01, "
            "18909f. 6ca2a9a0 landed the shortest_path goal guard "
            "(`and` -> `or`) on unit tests alone and took M5 red: Link "
            "died in 0x23 at f1453 to a goriya body while parrying in "
            "place, because a blocked chase goal now correctly returns "
            "no path and _engage stood on the None. Fixed by retargeting "
            "the chase goal and the collect waypoint to grid.nearest_open "
            "locally (not the walker-wide retarget_blocked_goal), and by "
            "counting collect_skip_unreachable toward the one-lap guard "
            "(0x45 burned its whole 9000f budget alternating two skips). "
            "18909f is the new oracle; 19416f was the pre-6ca2a9a0 tree. "
            "2026-09-14: re-verified 2/2 natural-entry, triforce=0x01, "
            "19416f — the same frame count as 2026-09-12, with the "
            "entry-route stall guard, reward nudge and engine reason "
            "histogram in the tree. "
            "2026-09-12: 2/2 natural-entry, triforce=0x01, 19416f. "
            "clear45_key 1568f 0 hits after scoop yield + collect "
            "stale-skip. Upstream still 4 hits (0x52 1, 0x23 1, 0x44 2); "
            "Link arrives 0x45 on half a heart and lives. Hearts on the "
            "0x23/0x44 floor are still unbanked. Do not grid-wire the "
            "evader: that capped 0x23 at 6000f. "
            "2026-09-14: pre-L1 is Zelda Dungeon The Gathering "
            "(docs/PRE_L1.md, row pre_l1) so the next L1 measure is 6 HC "
            "+ White Sword, not another 3HC wooden sit."
        ),
    ),
    CleanStep(
        id="pre_l1",
        bead="rr-ps7.4",
        segment="power-on → sword → bombs → 3 OW hearts → White Sword → L1 mouth",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="OW 0x77 → 0x6F / 0x7B / 0x2C / 0x0C / 0x0A / 0x47",
        pose="stop on 0x37 with 6 HC, bombs, candle, White Sword",
        residual="docs/PRE_L1.md",
        note=(
            "Zelda Dungeon The Gathering. Spec is docs/PRE_L1.md. "
            "Bombs are the south coast in shop_p7.SHOP_P7_HOPS: "
            "0x77, 0x78, 0x79 beach y=165, 0x7A band (133, 141), "
            "0x7B and 0x7C any row, the hop that leaves 0x7D carries "
            "SCREEN_7E_EAST_BAND (137, 145), then 0x7F UP into 0x6F. "
            "Not 0x68, not the 0x5C maze, not candle 0x5E, not arrow "
            "cave 0x4A. Stop is ADDR_BOMBS >= 1. Later: heart 0x7B, "
            "heart 0x2C, candle 0x0C, White Sword 0x0A around Lost "
            "Hills 0x1B, burn heart 0x47, shield 0x46, optional arrows "
            "0x4A, Blue Ring 0x34, then the 0x37 mouth. --through "
            "pre-l1 forces assist off. Do not poke bombs, rupees, "
            "candle, or $066F. A 2026-09-20 flagged --rollout trial "
            "bought bombs. That is not the default arm and not STATUS."
        ),
    ),
    CleanStep(
        id="l1_exit_ow_l2",
        bead="rr-ps7.3",
        segment="L1 leave → OW walk → L2 east mouth 0x4C",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="OW 0x48 / 0x49 hop lanes",
        pose="dies 0x4C (120,133) mode 17 hearts 0/4, door_death, 5 live octoroks",
        residual="docs/tasks/rr-8t4.4-residual.md",
        note=(
            "2026-09-12 (b): opt-in threat.decide is now wired into "
            "OverworldPathController (default off; on for the L2 controller "
            "only). It bought a heart and half the frames — prefix to 0x4A "
            "went 5010f hp 0x32 2/4 -> 2907f hp 0x33 3/4, 5/5 identical, and "
            "with evade=False the engine reproduces the old baseline "
            "frame-for-frame. farm_short is gone (farm_h3). The leg now runs "
            "four screens further and dies on 0x4C. "
            "RE-CLASSED body_undodgeable -> occupancy_stall: both surviving "
            "hits are lane problems, not dodge problems. The 0x48 leever sits "
            "at (112,205) with vx=vy=0 and surfaces UNDER Link (TTC 0, never "
            "dodgeable) — the real cause is a wedge, since align_and_push "
            "drops align_x below y=205 (80 < link_y < 205), so Link hammers "
            "DOWN into the wall 8px west of the x=120 gap for 159 frames "
            "while track_stuck never arms (he flips 112<->113 every frame). "
            "The 0x49 octorok is already inside the 16px pad the first frame "
            "it registers. 2026-09-14: 0x48 DOWN hop now strafes onto "
            "align_x=120 at y>=205 (live prefix 2072f, 0x48 hit gone). "
            "Occupied-lane on LEFT/RIGHT hops (L2 only) removed the 0x49 "
            "hit on a separate trial; DOWN occupied-lane added 0x38 hits "
            "and was reverted. Do NOT re-try an in-pad peel or an idle "
            "lane wait. Combined prefix not re-run this sitting. "
            "2026-09-14 0x4C census (one Clean trial, --farm-hearts 0): "
            "door_death 0x4C (120,133) mode 17 hp 0x30 0/4. Last playable "
            "(121,133) facing W, hop 5 UP align_x=112, stuck=0, 30× hop5_ax "
            "walking 160→121 at y=133. Killer slot 1 octorok_fast 0x08 at "
            "(112,124) vx=0 vy=+0.8 cheb 9 in pad, TTC 0 dodgeable False, "
            "body; four more octoroks live, no rocks. Not occupancy stall, "
            "not a y-band miss, not the east-mouth timeout. Class "
            "body_undodgeable — occupied door column, same 0x49 shape. "
            "Arrive 0x4C already 0/4; the body spent the partial. Pre-L1 "
            "6 HC is the budget; occupied-lane on hop 5 is the lane."
        ),
    ),
    CleanStep(
        id="l2_tf",
        bead="rr-4oz",
        segment="L2 Entrance → Triforce",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="L2 0x0e / 0x4f",
        pin="Level2Entrance $066F=0x3f (15 hearts in 4 containers)",
        residual="docs/tasks/rr-4oz-residual.md",
        note=(
            "Dodongo bomb placement on stable mouth only. 2026-09-14: "
            "scripts/audit_pins.py finds the pin incoherent, so the green "
            "was run on ~4x a real budget and the room findings are not "
            "yet a Clean measurement."
        ),
    ),
    CleanStep(
        id="l3_tf",
        bead="rr-npv.1",
        segment="L3 Entrance → Triforce",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        pin="Level3Entrance $066F=0x7f (15 hearts in 8 containers)",
        residual="docs/tasks/rr-npv.1-residual.md",
        note=(
            "all dest hops green, TF 0x04, deaths 0 — but measured from an "
            "incoherent pin (2026-09-14 audit), so the deaths-0 claim is "
            "against ~2x a real budget."
        ),
    ),
    CleanStep(
        id="l4_tf",
        bead="rr-bxzj",
        segment="L4 Entrance → Triforce",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        pin="Level4Entrance $066F=0x6f (15 hearts in 7 containers)",
        residual="docs/tasks/rr-bxzj-residual.md",
        note=(
            "31 contiguous stages, deaths 0, heart-safe Gleeok — measured "
            "from an incoherent pin (2026-09-14 audit). 'heart-safe' is the "
            "claim most exposed to the fake denominator."
        ),
    ),
    CleanStep(
        id="l5_tf",
        bead="rr-npv.2",
        segment="L5 Entrance → Triforce",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="L5 0x77",
        pose="(120,173) mode 17, third time on the hold line",
        pin="Level5Entrance $066F=0x3f (15 hearts in 4 containers), TF 0x00",
        residual="docs/tasks/rr-npv.2-residual.md",
        note=(
            "Room class is body_undodgeable: Pols Voice lands on the stand "
            "cell and the peel starts too late. But the pin is incoherent "
            "AND holds TF 0x00, which no L5 arrival can (L1-L4 is 0x0F), so "
            "fix the pin before trusting the death. Do not tune the peel "
            "against this budget."
        ),
    ),
    CleanStep(
        id="l6_tf",
        bead="rr-d6v",
        segment="L6 Entrance → Triforce",
        rung=Rung.SPINE_GREEN,
        blocker=Blocker.NONE,
        room="L6 0x0C",
        pose="triforce 0x3F, sword 3, 13 containers, 235596f",
        pin="power-on clean_poweron83",
        residual="docs/tasks/rr-d6v-residual.md",
        note=(
            "clean_poweron83. 0x78 and 0x28 are walked, 0x28 bombs east "
            "into 0x29, and 0x7a is off the route. Next open row is L7."
        ),
    ),
    CleanStep(
        id="l7_tf",
        bead="rr-rgum",
        segment="L7 Entrance → Triforce",
        rung=Rung.HYPOTHESIS,
        blocker=Blocker.NONE,
        room="L7 0x42",
        pose="triforce 0x7F after Aquamentus",
        residual="docs/tasks/rr-npv.3-residual.md",
        note=(
            "rr-rgum: --clean --through level7 from power-on. Food is "
            "already carried from the gather. Do not poke ADDR_FOOD."
        ),
    ),
    CleanStep(
        id="l8_tf",
        bead="rr-npv.4",
        segment="L8 Entrance → Triforce",
        rung=Rung.HYPOTHESIS,
        blocker=Blocker.INVENTORY_GAP,
        room="L8 0x1E",
        pose="(128,181) mode 17, 3/3 oscillating 128↔112",
        pin="Level8EntranceReconFixture $066F=0x2f (15 hearts in 3 containers)",
        residual="docs/tasks/rr-npv.4-residual.md",
        note=(
            "rr-npv.4 waits on Level 7 and on rr-awh6. 0x3E clears its "
            "wall-bombed flag on every screen change, so the 0x4C bomb "
            "can be spent twice."
        ),
    ),
    CleanStep(
        id="l9_credits",
        bead="rr-npv.5",
        segment="L9 Entrance → Ganon → credits",
        rung=Rung.HYPOTHESIS,
        residual="docs/tasks/rr-npv.5-residual.md",
        note=(
            "rr-npv.5: one --clean --through level9-credits power-on. "
            "That tape is the STATUS row. Recon pins are not it."
        ),
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


# Engine mechanisms that replaced a hand-written per-room rule. A level that
# has not adopted one is not "differently tuned" — it is running the version
# the mechanism was written to fix, and its lane will re-derive the same
# finding. Keyed by the ``CombatTuning`` field so this cannot drift from the
# spec table: the report is measured off the live specs, not written here.
SHARED_MECHANISMS: dict[str, str] = {
    "occupancy_patrol": (
        "walk.physics: predict 1px, grade, block the missed cell, replan. "
        "The alternative is a patrol that walks a wall until stuck-escape "
        "guesses."
    ),
    "occupancy_from_tilemap": (
        "Seed those walls from the live $6530 map (dungeon.tilemap) instead "
        "of a hand-written occupancy_blocked box. The L1 0x23 box walled 84 "
        "cells of real floor; the walker burned 2438 frames on it."
    ),
    "evade": (
        "Run threat.decide before the position rules in _combat. A pose rule "
        "that returns first silences the reactive layer for that frame — the "
        "dominant damage bug in this tree."
    ),
}


def adoption() -> dict[int, dict[str, tuple[int, int]]]:
    """Per level: ``{mechanism: (rooms_with, rooms_total)}`` off the live specs."""
    from zelda_i.dungeon.engine import _ROOM_SPECS_BY_LEVEL, ensure_default_specs

    ensure_default_specs()
    rows: dict[int, dict[str, tuple[int, int]]] = {}
    for (level, _room), spec in sorted(_ROOM_SPECS_BY_LEVEL.items()):
        counts = rows.setdefault(
            int(level), {name: (0, 0) for name in SHARED_MECHANISMS}
        )
        for name in SHARED_MECHANISMS:
            have, total = counts[name]
            counts[name] = (
                have + bool(getattr(spec.combat, name, False)),
                total + 1,
            )
    return rows


def render_adoption() -> str:
    """Which levels run the shared mechanisms and which still do not."""
    rows = adoption()
    names = list(SHARED_MECHANISMS)
    lines = [
        "shared mechanisms by level (rooms adopted / rooms with a spec):",
        "",
        "level  " + "  ".join(f"{name:<22}" for name in names),
    ]
    for level, counts in sorted(rows.items()):
        cells = []
        for name in names:
            have, total = counts[name]
            cells.append(f"{f'{have}/{total}':<22}")
        lines.append(f"L{level:<5}  " + "  ".join(cells))
    lines.append("")
    for name, why in SHARED_MECHANISMS.items():
        lines.append(f"  {name}: {why}")
    return "\n".join(lines)
