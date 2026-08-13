"""KPDR Any% split table (kennycason super_metroid_auto_tracker layout).

Same 24 splits as ``SplitProfiles.KPDR_ANY``:
https://github.com/kennycason/super_metroid_auto_tracker

Timing flags (match tracker / SuperMetroid.asl where the hop timeline
can):

* **RTA zero** — first Ceres Elevator ordinary control (gs=8). Tracker
  default auto-start is title Start (gs 2→31) and *includes* the intro
  cinematic (~40s). We stay on first-control so HUD/PB is gameplay RTA.
* **Ceres Station** — ASL ``ceresEscape``: leave ordinary in Ceres
  Elevator toward Landing Site (gs 8→32). Not first Landing Site settle.
* **Later rows** — still room-entry proxies (item/boss bits are not on
  the hop timeline). Tracker fires item/boss *bits*; same-order auto-skip
  means an earlier HJ pickup still prints as a 0-segment after Varia.

Columns match the tracker's splits view:

* **BEST** — product-line cumulative (this stitched run / PB)
* **TIME** — Best Possible: sum of hop-PB dwells along the product path
* **Best +/-** — TIME − BEST (negative = golds beat the product line)

MB1 / MB2 / Ship are a coarse splice: product line to Mother Brain room
entry, then ``g4_tourian_human_bb`` + ``_mb`` finish and escape (±10s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.rta_clock import fmt_time


# (id, display name, room_id, 1-based visit count on the product line)
# Charge is the second Big Pink visit (post-Super return) — no Charge room id.
# Ceres uses elevator-leave (see ``_ceres_escape_hit``), not 0x91F8 entry.
KPDR_ANY_SPLITS: tuple[tuple[str, str, int | None, int], ...] = (
    ("ceres_station", "Ceres Station", 0xDF45, 2),
    ("morph_ball", "Morph Ball", 0x9E9F, 1),
    ("first_missile", "First Missiles", 0xA107, 1),
    ("bomb", "Bomb", 0x9804, 1),
    ("first_super", "First Super", 0x9B5B, 1),
    ("charge_beam", "Charge Beam", 0x9D19, 2),
    ("spazer", "Spazer", 0xA447, 1),
    ("kraid", "Kraid", 0xA59F, 1),
    ("varia_suit", "Varia Suit", 0xA6E2, 1),
    ("hi_jump", "Hi-Jump Boots", 0xA9E5, 1),
    ("speed_booster", "Speed Booster", 0xAD1B, 1),
    ("wave_beam", "Wave Beam", 0xADDE, 1),
    ("ice_beam", "Ice Beam", 0xA890, 1),
    ("first_power_bomb", "First Power Bomb", 0xA3AE, 1),
    ("phantoon", "Phantoon", 0xCD13, 1),
    ("gravity_suit", "Gravity Suit", 0xCE40, 1),
    ("draygon", "Draygon", 0xDA60, 1),
    ("space_jump", "Space Jump", 0xD9AA, 1),
    ("plasma_beam", "Plasma Beam", 0xD2AA, 1),
    ("ridley", "Ridley", 0xB32E, 1),
    ("golden_four", "G4", 0xA66A, 1),
    ("mother_brain_1", "Mother Brain 1", None, 1),
    ("mother_brain_2", "Mother Brain 2", None, 1),
    ("ship", "Ship", None, 1),
)

ROOM_CERES_ELEVATOR = 0xDF45
ROOM_LANDING_SITE = 0x91F8
ROOM_MOTHER_BRAIN = 0xDD58

# Coarse tail from 2026-08-10 tapes (not the dying full_start_v1 MB hop).
# Offsets are frames after settled MB room enter (bb f005415 / product enter).
# bb mid-lockstep f008135 is already rainbow (pose 84) → both phases in ~45s.
# mb tape end f024220 is Landing Site ship / ending (gs 39).
_G4_BB_ENTER = 5415
_G4_BB_END = 19972
_G4_MB_SHIP = 24220
MB_ESCAPE_TAIL: dict[str, tuple[int, str]] = {
    "mother_brain_1": (1200, "g4_tourian_human_bb"),
    "mother_brain_2": (2520, "g4_tourian_human_bb"),
    "ship": ((_G4_BB_END - _G4_BB_ENTER) + _G4_MB_SHIP, "g4_tourian_human_mb"),
}


@dataclass(frozen=True)
class KpdrSplitRow:
    split_id: str
    name: str
    best_frames: int | None
    gold_frames: int | None
    delta_frames: int | None
    source: str
    hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.split_id,
            "name": self.name,
            "best_frames": self.best_frames,
            "best": fmt_tracker(self.best_frames) if self.best_frames is not None else None,
            "gold_frames": self.gold_frames,
            "time": fmt_tracker(self.gold_frames) if self.gold_frames is not None else None,
            "delta_frames": self.delta_frames,
            "delta": fmt_tracker_delta(self.delta_frames),
            "source": self.source,
            "hit": self.hit,
        }


def fmt_tracker(frames: int | None) -> str:
    """Tracker clock: ``mm:ss.cc`` (60 fps → centiseconds; minutes may exceed 59)."""
    if frames is None:
        return "—"
    frames = max(0, int(frames))
    cs = (frames * 100 + 30) // 60
    minutes, cs = divmod(cs, 6000)
    seconds, cs = divmod(cs, 100)
    return f"{minutes:02d}:{seconds:02d}.{cs:02d}"


def fmt_tracker_delta(frames: int | None) -> str:
    if frames is None:
        return "—"
    sign = "+" if frames > 0 else "-" if frames < 0 else "+"
    body = fmt_tracker(abs(int(frames)))
    if body.startswith("00:"):
        body = body[3:]  # ss.cc for sub-minute deltas
    return f"{sign}{body}"


def _nth_room_hit(
    timeline: Sequence[Mapping[str, Any]],
    room_id: int,
    visit: int,
    *,
    not_before: int = 0,
) -> Mapping[str, Any] | None:
    """Nth visit, or the first visit at/after *not_before* if already passed."""
    seen = 0
    earlier: Mapping[str, Any] | None = None
    for row in timeline:
        if int(row.get("room_id") or 0) != int(room_id):
            continue
        seen += 1
        entry = int(row.get("abs_entry") or 0)
        if seen == int(visit):
            if entry >= int(not_before):
                return row
            earlier = row
        if entry >= int(not_before) and seen >= int(visit):
            return row
    return earlier


def _gold_up_to(
    timeline: Sequence[Mapping[str, Any]],
    hop_pb: Mapping[str, int],
    abs_entry: int,
) -> int:
    """Product abs_entry minus hop-PB savings (keeps door transitions)."""
    saved = 0
    for row in timeline:
        entry = int(row.get("abs_entry") or 0)
        if entry > int(abs_entry):
            break
        key = str(row.get("hop_key") or "")
        dwell = int(row.get("dwell") or 0)
        pb = int(hop_pb.get(key, dwell))
        if 0 <= pb < dwell:
            saved += dwell - pb
    return max(0, int(abs_entry) - saved)


def _leave_abs(row: Mapping[str, Any]) -> int:
    """Ordinary leave (abs_entry + dwell) — RoomTimer leave_frame / gs≠8."""
    return int(row.get("abs_entry") or 0) + max(0, int(row.get("dwell") or 0))


def _ceres_escape_hit(
    timeline: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """ASL ceresEscape: Ceres Elevator hop that leaves toward Landing Site.

    Tracker: room==CeresElevator and gs 8→32. On our leaf timeline that is
    the elevator visit whose dest is Landing Site (leave ordinary = cutscene
    start). Falls back to the second elevator visit.
    """
    visits: list[Mapping[str, Any]] = []
    dest_hit: Mapping[str, Any] | None = None
    for row in timeline:
        if int(row.get("room_id") or 0) != ROOM_CERES_ELEVATOR:
            continue
        visits.append(row)
        dest = row.get("dest_room_id")
        if dest is not None and int(dest) == ROOM_LANDING_SITE:
            dest_hit = row
    if dest_hit is not None:
        return dest_hit
    if len(visits) >= 2:
        return visits[1]
    return visits[0] if visits else None


def _mb_entry_row(timeline: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return _nth_room_hit(timeline, ROOM_MOTHER_BRAIN, 1)


def build_kpdr_split_rows(
    timeline: Sequence[Mapping[str, Any]],
    *,
    hop_pb: Mapping[str, int] | None = None,
) -> list[KpdrSplitRow]:
    """Match KPDR Any% splits on the product timeline."""
    pbs = dict(hop_pb or {})
    mb_enter = _mb_entry_row(timeline)
    mb_abs = int(mb_enter.get("abs_entry") or 0) if mb_enter is not None else None
    mb_gold = _gold_up_to(timeline, pbs, mb_abs) if mb_abs is not None else None
    rows: list[KpdrSplitRow] = []
    prev_best = 0
    for split_id, name, room_id, visit in KPDR_ANY_SPLITS:
        tail = MB_ESCAPE_TAIL.get(split_id)
        if tail is not None:
            if mb_abs is None or mb_gold is None:
                rows.append(
                    KpdrSplitRow(
                        split_id=split_id,
                        name=name,
                        best_frames=None,
                        gold_frames=None,
                        delta_frames=None,
                        source="",
                        hit=False,
                    )
                )
                continue
            offset, source = tail
            best = mb_abs + int(offset)
            gold = mb_gold + int(offset)
            prev_best = best
            rows.append(
                KpdrSplitRow(
                    split_id=split_id,
                    name=name,
                    best_frames=best,
                    gold_frames=gold,
                    delta_frames=gold - best,
                    source=source,
                    hit=True,
                )
            )
            continue
        if split_id == "ceres_station":
            hit = _ceres_escape_hit(timeline)
            if hit is None:
                rows.append(
                    KpdrSplitRow(
                        split_id=split_id,
                        name=name,
                        best_frames=None,
                        gold_frames=None,
                        delta_frames=None,
                        source="",
                        hit=False,
                    )
                )
                continue
            raw = _leave_abs(hit)
            best = max(raw, prev_best)
            gold = _gold_up_to(timeline, pbs, best)
            prev_best = best
            rows.append(
                KpdrSplitRow(
                    split_id=split_id,
                    name=name,
                    best_frames=best,
                    gold_frames=gold,
                    delta_frames=gold - best,
                    source=str(hit.get("source") or ""),
                    hit=True,
                )
            )
            continue
        if room_id is None:
            rows.append(
                KpdrSplitRow(
                    split_id=split_id,
                    name=name,
                    best_frames=None,
                    gold_frames=None,
                    delta_frames=None,
                    source="",
                    hit=False,
                )
            )
            continue
        hit = _nth_room_hit(timeline, room_id, visit, not_before=prev_best)
        if hit is None:
            rows.append(
                KpdrSplitRow(
                    split_id=split_id,
                    name=name,
                    best_frames=None,
                    gold_frames=None,
                    delta_frames=None,
                    source="",
                    hit=False,
                )
            )
            continue
        raw = int(hit.get("abs_entry") or 0)
        # Already collected before this row in the KPDR list → 0-segment skip.
        best = max(raw, prev_best)
        gold = _gold_up_to(timeline, pbs, best)
        prev_best = best
        rows.append(
            KpdrSplitRow(
                split_id=split_id,
                name=name,
                best_frames=best,
                gold_frames=gold,
                delta_frames=gold - best,
                source=str(hit.get("source") or ""),
                hit=True,
            )
        )
    return rows


def format_kpdr_split_table(
    rows: Sequence[KpdrSplitRow],
    *,
    product_frames: int,
    grade: str = "EST",
) -> str:
    """Tracker-style KPDR Any% table (BEST / Best +/- / TIME)."""
    hits = [r for r in rows if r.hit and r.best_frames is not None]
    last_gold = hits[-1].gold_frames if hits else None
    last_best = hits[-1].best_frames if hits else None
    ship_hit = any(r.split_id == "ship" and r.hit for r in rows)
    leftover = 0
    if last_best is not None and not ship_hit:
        leftover = max(0, int(product_frames) - int(last_best))
    gold_total = (int(last_gold) + leftover) if last_gold is not None else None
    pb_frames = int(last_best) if ship_hit and last_best is not None else int(product_frames)
    status = (
        "assisted · spliced MB finish + escape"
        if ship_hit
        else "assisted · incomplete · no credits"
    )
    gold_note = (
        "hop-PB gold + same MB/escape tail · theoretical"
        if ship_hit
        else "hop-PB gold + unfinished tail · theoretical"
    )
    pb_note = (
        f"{grade} · product to MB + g4_tourian_bb/mb"
        if ship_hit
        else f"{grade} · product RTA {fmt_time(product_frames)}"
    )
    lines = [
        "-" * 72,
        f"KPDR ANY%  ·  {grade} ({status})",
        f"{'':<22} {'BEST':>10} {'Best +/-':>10} {'TIME':>10}",
    ]
    for row in rows:
        best = fmt_tracker(row.best_frames) if row.hit else "—"
        gold = fmt_tracker(row.gold_frames) if row.hit else "—"
        delta = fmt_tracker_delta(row.delta_frames) if row.hit else "—"
        lines.append(f"{row.name:<22} {best:>10} {delta:>10} {gold:>10}")
    lines.append("")
    pb_s = fmt_tracker(pb_frames)
    gold_s = fmt_tracker(gold_total) if gold_total is not None else "—"
    lines.append(f"{'Personal Best':<22} {pb_s:>10}    ({pb_note})")
    lines.append(f"{'Best Possible':<22} {gold_s:>10}    ({gold_note})")
    lines.append("BEST = this product line  ·  TIME = hop-PB gold  ·  Best +/- = TIME − BEST")
    lines.append("-" * 72)
    return "\n".join(lines)
