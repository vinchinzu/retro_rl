"""Autobot progress for the twelve dungeons.

Reads checkbox marks in ``docs/PORT_PLAN.md`` and prints one row per dungeon.
Does not edit the plan and does not boot a ROM. A row is not a Clean claim.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from alttp.paths import DOCS_DIR

_TITLE = re.compile(r"\*\*(AP-\d+) · ([^*]+)\*\*")
_MARK = re.compile(r"- \[([ x!~])\]")
_LOOKBACK = 4000

# slug, display name, first AP number, last AP number. Eight steps each.
DUNGEONS: tuple[tuple[str, str, int, int], ...] = (
    ("eastern", "Eastern Palace", 99, 106),
    ("desert", "Desert Palace", 107, 114),
    ("hera", "Tower of Hera", 115, 122),
    ("tower", "Agahnim's Tower", 123, 130),
    ("pod", "Palace of Darkness", 131, 138),
    ("swamp", "Swamp Palace", 139, 146),
    ("skull", "Skull Woods", 147, 154),
    ("thieves", "Thieves' Town", 155, 162),
    ("ice", "Ice Palace", 163, 170),
    ("mire", "Misery Mire", 171, 178),
    ("turtle", "Turtle Rock", 179, 186),
    ("gt", "Ganon's Tower", 187, 194),
)

# Printed while this id is still open or claimed. That id must not boot.
_HALT = {
    "AP-105": (
        "Play_eog9_7 is room 0xB9 at (4856, 6104), keys 0. "
        "The north door has no approach point. "
        "The west door has no floor path. "
        "The east door is type 0x1C. "
        "Do not boot the hall. Do not press A on the rupee chest. "
        "opening_route/room_engine.py is past 1000 lines. Do not add a sibling module."
    ),
    "AP-113": (
        "Claimed by desert-tablet. Do not steal it. Do not dash the plaque again."
    ),
    "AP-152": "Mothula room 0x29 was not entered. Do not boot that room.",
    "AP-168": "Claimed by kholdstare-fight. Do not send another stand.",
    "AP-182": "No underworld room id for the laser bridge. Do not boot.",
    "AP-190": (
        "Room 0x6B door type 0x1E stayed shut and the big key stayed 0. "
        "Do not walk it."
    ),
    "AP-192": "The earlier Agahnim step is blocked. Do not start the clone fight.",
}


@dataclass(frozen=True)
class Step:
    ap_id: str
    number: int
    mark: str
    title: str


@dataclass(frozen=True)
class DungeonRow:
    slug: str
    name: str
    steps: tuple[Step, ...]

    @property
    def done(self) -> int:
        return sum(step.mark == "x" for step in self.steps)

    @property
    def open_count(self) -> int:
        return sum(step.mark == " " for step in self.steps)

    @property
    def blocked(self) -> int:
        return sum(step.mark == "!" for step in self.steps)

    @property
    def claimed(self) -> int:
        return sum(step.mark == "~" for step in self.steps)

    @property
    def next_open(self) -> Step | None:
        for step in self.steps:
            if step.mark == " ":
                return step
        return None

    @property
    def may_boot(self) -> bool:
        """True only for an interior step with no earlier block and no halt."""
        nxt = self.next_open
        if nxt is None or nxt.ap_id in _HALT or nxt.number >= 7:
            return False
        return not any(
            step.mark == "!" and step.number < nxt.number for step in self.steps
        )

    def gate(self) -> str:
        """One boot instruction for the next open step."""
        halts = [
            _HALT[step.ap_id]
            for step in self.steps
            if step.mark in (" ", "~") and step.ap_id in _HALT
        ]
        nxt = self.next_open
        if nxt is not None and nxt.ap_id in _HALT:
            return " ".join([*halts, "Boot: no."])
        blocked = [step.ap_id for step in self.steps if step.mark == "!"]
        claimed = [step.ap_id for step in self.steps if step.mark == "~"]
        parts: list[str] = []
        if nxt is None:
            parts.append("No open step.")
        elif nxt.number >= 7 and blocked:
            parts.append(
                f"{nxt.ap_id} is step {nxt.number}. "
                f"Blocked earlier: {', '.join(blocked)}. "
                "Do not walk the chain."
            )
        elif nxt.number >= 7:
            parts.append(
                f"{nxt.ap_id} is step {nxt.number}. "
                "It needs the previous dungeon step 8. Do not power on."
            )
        else:
            parts.append(f"{nxt.ap_id} is step {nxt.number}.")
            if blocked:
                parts.append(f"Blocked earlier: {', '.join(blocked)}.")
        if claimed:
            parts.append(f"Claimed: {', '.join(claimed)}. Do not steal.")
        parts.extend(halts)
        if not self.may_boot:
            parts.append("Boot: no.")
        return " ".join(parts)


def parse_marks(text: str) -> dict[str, tuple[str, str]]:
    """Map ``AP-###`` to ``(checkbox mark, title)`` from heading text."""
    found: dict[str, tuple[str, str]] = {}
    for match in _TITLE.finditer(text):
        ap_id = match.group(1)
        title = match.group(2).strip().rstrip(".")
        window = text[max(0, match.start() - _LOOKBACK) : match.start()]
        marks = list(_MARK.finditer(window))
        if not marks:
            raise ValueError(
                f"{ap_id} has no checkbox within {_LOOKBACK} characters"
            )
        found[ap_id] = (marks[-1].group(1), title)
    return found


def load_rows(plan_text: str | None = None) -> tuple[DungeonRow, ...]:
    """Twelve dungeon rows from the plan text, or from ``PORT_PLAN.md``."""
    text = (
        plan_text
        if plan_text is not None
        else (DOCS_DIR / "PORT_PLAN.md").read_text(encoding="utf-8")
    )
    marks = parse_marks(text)
    rows: list[DungeonRow] = []
    for slug, name, lo, hi in DUNGEONS:
        steps: list[Step] = []
        for number, ap_num in enumerate(range(lo, hi + 1), start=1):
            ap_id = f"AP-{ap_num:03d}"
            if ap_id not in marks:
                raise ValueError(f"{ap_id} is missing from the plan")
            mark, title = marks[ap_id]
            steps.append(Step(ap_id, number, mark, title))
        rows.append(DungeonRow(slug, name, tuple(steps)))
    return tuple(rows)


def format_rows(rows: tuple[DungeonRow, ...]) -> str:
    """Text table plus one gate line per dungeon."""
    lines = [
        f"{'dungeon':<22} {'done':>4} {'open':>4} {'blk':>4} {'claim':>5}  "
        f"{'next':<8} boot"
    ]
    for row in rows:
        nxt = row.next_open.ap_id if row.next_open else "-"
        boot = "yes" if row.may_boot else "no"
        lines.append(
            f"{row.name:<22} {row.done:4d} {row.open_count:4d} {row.blocked:4d} "
            f"{row.claimed:5d}  {nxt:<8} {boot}"
        )
        lines.append(f"  {row.gate()}")
    lines.append(
        f"{'dungeon steps':<22} "
        f"{sum(row.done for row in rows):4d} "
        f"{sum(row.open_count for row in rows):4d} "
        f"{sum(row.blocked for row in rows):4d} "
        f"{sum(row.claimed for row in rows):5d}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    del argv
    print(format_rows(load_rows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
