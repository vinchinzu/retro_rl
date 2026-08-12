"""Classify a live A-press or tape held-change without a new recording.

Ground forage (grape/mushroom) sets held and opens Eat / Don't eat.
NPC talk locks input with held still 0. Farm bushes are a third class
(sustained A, no keep-menu). Do not cargo-cult one onto another.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

# HM-Decomp held-item table (0x091D).
FORAGE_HELD = {
    0x01: "mushroom",
    0x02: "poison_mushroom",
    0x03: "grapes",
    0x04: "green_fruit",
    0x05: "flower",
    0x07: "fish",
    0x08: "power_berry",
}
_LOCKED = frozenset({0, 2, 4})


def held_name(held: int) -> str:
    hid = int(held)
    if hid == 0:
        return "empty"
    return FORAGE_HELD.get(hid, f"0x{hid:02X}")


def classify_interact(
    *,
    held_before: int,
    held_after: int,
    lock_after: int,
    text_choices: Sequence[str] = (),
    npc_in_face: bool = False,
) -> str:
    """Name the interact from RAM deltas. Not vibes, not a screenshot guess."""
    before, after = int(held_before), int(held_after)
    if after and after != before:
        blob = " ".join(text_choices).lower()
        if "don't eat" in blob or "dont eat" in blob:
            return "forage_keep_menu"
        if after in FORAGE_HELD:
            return "forage_held"
        return "item_held"
    if int(lock_after) in _LOCKED and after == 0:
        return "npc_talk" if npc_in_face else "dialogue_empty_hands"
    return "no_interact"


def first_held_change(trace: Iterable[dict]) -> Optional[dict]:
    """First RAM-trace row where held_item changes. Offline — no emulator."""
    prev: Optional[int] = None
    for row in trace:
        held = int(row.get("held_item") or 0)
        if prev is not None and held != prev:
            out = dict(row)
            out["held_before"] = prev
            out["held_after"] = held
            out["held_name"] = held_name(held)
            return out
        prev = held
    return None


__all__ = [
    "FORAGE_HELD",
    "classify_interact",
    "first_held_change",
    "held_name",
]
