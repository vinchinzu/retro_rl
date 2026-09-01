"""Ceres station spine: boot, outbound, escape. Not the morph file.

Morph composes :data:`CERES_SPINE` as its prefix. Ceres 1–3 always use the
reactive moonfall / magnet-feet / jump-before-ledge policies. There is no
legacy tape fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from retro_harness.actions import buttons, idle_action
from super_metroid.paths import GAME_DIR
from super_metroid.progression.types import (
    DoorEdge,
    ProgressCondition,
    ProgressionMilestone,
)
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
    ROOM_LANDING_SITE,
)
from super_metroid.routes.kpdr.spine import SpineHop
from super_metroid.routes.runtime import ActionSpan, RouteSession

# Open-loop title/intro mash. TAS-style START/A mash is play_boot_to_ceres_tas
# (probe only) — it is not a product fallback.
_BOOT_MENU_MASH_FRAMES = 400
_BOOT_MAX_FRAMES = 12_000

CERES_SCIENTIST_MAX_FRAMES = 400
_TAS_CERES_HOPS_PATH = GAME_DIR / "tas" / "bodies" / "sniq_100_ceres_lsnes_hops.json"
TAS_CLOCK_ROOM_FLIP = "room_flip"
TAS_CLOCK_SETTLED_GS8 = "settled_gs8"
DEFAULT_TAS_CLOCK = TAS_CLOCK_SETTLED_GS8


def load_tas_ceres_hops(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Sniq 100% lsnes hop table (scratch alignment, not STATUS)."""
    raw = json.loads((path or _TAS_CERES_HOPS_PATH).read_text())
    hops = raw.get("hops") if isinstance(raw, dict) else raw
    return [dict(h) for h in hops]


def tas_hop_clock(
    hop: Mapping[str, Any] | None,
    clock: str = DEFAULT_TAS_CLOCK,
) -> dict[str, Any]:
    """Named TAS clock blob. RoomTimer product rows are settled gs=8.

    ``room_flip`` is gs=11 enter-to-enter. ``settled_gs8`` is dest gs=8 minus
    source gs=8 (elev pad y=72). Missing clock blobs fall back to hop.frames
    and set ``fallback`` so 344 is never silently treated as 345.
    """
    if hop is None:
        return {}
    blob = hop.get(clock)
    if isinstance(blob, Mapping) and blob.get("frames") is not None:
        return dict(blob)
    frames = hop.get("frames")
    if frames is None:
        return {}
    return {"frames": int(frames), "fallback": "room_flip_frames"}


def _visit_int(visit: Mapping[str, Any], key: str) -> int | None:
    raw = visit.get(key)
    return None if raw is None else int(raw)


def ceres_hops_vs_tas(
    visits: Sequence[Mapping[str, Any]],
    *,
    landing_frame: int | None = None,
    first_control_frame: int | None = None,
    tas_hops: Sequence[Mapping[str, Any]] | None = None,
    clock: str = DEFAULT_TAS_CLOCK,
) -> dict[str, Any]:
    """Per-room vs TAS on a named clock, including last elev→landing.

    Product RoomTimer ``room_frames`` is dest gs=8 settle (dwell + fade).
    Default ``clock`` is ``settled_gs8``. ``room_flip`` is TAS enter-to-enter
    only — comparing it to RoomTimer totals is labeled ``clock_mismatch``.
    RoomTimer previously dropped elev→landing: Zebes gs=6 is a cinematic.
    Pass ``landing_frame`` to synthesize it when the visit list is short.
    """
    tas = list(tas_hops) if tas_hops is not None else load_tas_ceres_hops()
    tas_by_edge = {(str(h["from"]), str(h["to"])): h for h in tas}
    rows: list[dict[str, Any]] = []
    for visit in visits:
        src = str(visit.get("room_id_hex") or f"0x{int(visit['room_id']):04X}")
        dst = str(visit.get("dest_room_id_hex") or f"0x{int(visit['dest_room_id']):04X}")
        tas_hop = tas_by_edge.get((src, dst))
        clock_blob = tas_hop_clock(tas_hop, clock)
        our = int(visit["room_frames"])
        tas_frames = None if clock_blob.get("frames") is None else int(clock_blob["frames"])
        our_dwell = _visit_int(visit, "dwell_frames")
        our_trans = _visit_int(visit, "transition_frames")
        tas_dwell = clock_blob.get("dwell_frames")
        tas_trans = clock_blob.get("transition_frames")
        row = {
            "name": None if tas_hop is None else tas_hop.get("name"),
            "from": src,
            "to": dst,
            "clock": clock,
            "our_clock": TAS_CLOCK_SETTLED_GS8,
            "our_frames": our,
            "our_dwell_frames": our_dwell,
            "our_transition_frames": our_trans,
            "tas_frames": tas_frames,
            "tas_dwell_frames": None if tas_dwell is None else int(tas_dwell),
            "tas_transition_frames": None if tas_trans is None else int(tas_trans),
            "delta_frames": None if tas_frames is None else our - tas_frames,
            "delta_dwell_frames": None
            if tas_dwell is None or our_dwell is None
            else our_dwell - int(tas_dwell),
            "delta_transition_frames": None
            if tas_trans is None or our_trans is None
            else our_trans - int(tas_trans),
            "tas_source_frames": {
                k: clock_blob[k]
                for k in (
                    "source_gs8",
                    "dest_gs8",
                    "door_gs9",
                    "source_enter",
                    "dest_enter",
                    "fallback",
                )
                if k in clock_blob
            },
        }
        if clock != TAS_CLOCK_SETTLED_GS8:
            row["clock_mismatch"] = True
        rows.append(row)

    last_edge = (f"0x{ROOM_CERES_ELEVATOR:04X}", f"0x{ROOM_LANDING_SITE:04X}")
    has_last = any((r["from"], r["to"]) == last_edge for r in rows)
    last_entry = None
    if visits:
        falling_to_elev = [
            v
            for v in visits
            if int(v["room_id"]) == ROOM_CERES_FALLING
            and int(v["dest_room_id"]) == ROOM_CERES_ELEVATOR
        ]
        if falling_to_elev:
            last_entry = int(falling_to_elev[-1]["exit_frame"])
        last_visit = visits[-1]
        if int(last_visit["room_id"]) == ROOM_CERES_ELEVATOR and int(
            last_visit["dest_room_id"]
        ) == ROOM_LANDING_SITE:
            last_entry = int(last_visit["entry_frame"])
    if not has_last and landing_frame is not None and last_entry is not None:
        tas_hop = tas_by_edge.get(last_edge)
        clock_blob = tas_hop_clock(tas_hop, clock)
        our = int(landing_frame) - last_entry
        tas_frames = None if clock_blob.get("frames") is None else int(clock_blob["frames"])
        syn = {
            "name": "elev_to_landing",
            "from": last_edge[0],
            "to": last_edge[1],
            "clock": clock,
            "our_clock": TAS_CLOCK_SETTLED_GS8,
            "our_frames": our,
            "tas_frames": tas_frames,
            "delta_frames": None if tas_frames is None else our - tas_frames,
            "synthesized": True,
        }
        if clock != TAS_CLOCK_SETTLED_GS8:
            syn["clock_mismatch"] = True
        rows.append(syn)

    up_to_last = None
    last_room = None
    if last_entry is not None:
        if first_control_frame is not None:
            up_to_last = format_segment_time(last_entry - int(first_control_frame))
        if landing_frame is not None:
            last_room = format_segment_time(int(landing_frame) - last_entry)
        elif has_last:
            last_row = next(r for r in rows if (r["from"], r["to"]) == last_edge)
            last_room = format_segment_time(int(last_row["our_frames"]))

    return {
        "hops": rows,
        "clock": clock,
        "up_to_last_room": up_to_last,
        "last_room": last_room,
        "last_room_entry_frame": last_entry,
    }


def _boot_spans() -> list[ActionSpan]:
    """Maintained open-loop boot prefix (product Morph dual GREEN @ 24,187f)."""
    spans = [
        ActionSpan((), 2100, "boot_title_wait"),
        ActionSpan(("A",), 10, "boot_title_confirm"),
        ActionSpan((), 120, "boot_file_menu_wait"),
        ActionSpan(("A",), 10, "boot_file_confirm"),
        ActionSpan((), 300, "boot_prologue_wait"),
        ActionSpan(("A",), 10, "boot_prologue_confirm"),
        ActionSpan((), 30, "boot_prologue_settle"),
    ]
    for _ in range(69):
        spans.append(ActionSpan(("A",), 10, "boot_intro_mash"))
        spans.append(ActionSpan((), 110, "boot_intro_wait"))
    return spans


def play_boot_to_ceres_tas(session: RouteSession) -> None:
    """TAS-style boot → settled Ceres elev pad (probe / residual rr-14u)."""
    reached = False
    for i in range(_BOOT_MAX_FRAMES):
        st = session.state
        if st.room_id == ROOM_CERES_ELEVATOR and st.game_state == 8:
            reached = True
            break
        if i < _BOOT_MENU_MASH_FRAMES:
            name = "START" if (i % 2) == 0 else "A"
            session.step(buttons(name), "boot_menu_mash")
        elif (i % 2) == 0:
            session.step(buttons("A"), "boot_cutscene_mash")
        else:
            session.step(idle_action(), "boot_cutscene_wait")
    if not reached:
        raise RuntimeError(
            f"TAS boot missed Ceres control after {_BOOT_MAX_FRAMES}f: {session.state}"
        )
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR
        and s.game_state == 8
        and int(s.samus_y) >= 60
        and abs(int(s.velocity_y)) <= 1,
        timeout=200,
        reason="boot_elev_settle",
    )
    for _ in range(4):
        session.step(idle_action(), "boot_elev_plant")


def play_boot_to_ceres(session: RouteSession) -> None:
    """Power-on → first controllable Ceres elevator frame."""
    session.spans(_boot_spans())
    if not (session.state.room_id == ROOM_CERES_ELEVATOR and session.state.game_state == 8):
        raise RuntimeError(f"boot missed first Ceres control: {session.state}")


CERES_SPINE: tuple[SpineHop, ...] = (
    SpineHop(
        "first_ceres_control",
        play_boot_to_ceres,
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_ELEVATOR,
        "Ceres Elevator",
        "morph",
        use_transition_split=False,
    ),
    SpineHop(
        "ridley_countdown",
        play_ceres_outbound_to_ridley,
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_RIDLEY,
        "Ceres Ridley",
        "morph",
        use_transition_split=False,
        policy_id="ceres_outbound",
    ),
    SpineHop(
        "zebes_landing",
        play_ceres_escape_to_landing,
        ROOM_CERES_RIDLEY,
        ROOM_LANDING_SITE,
        "Landing Site",
        "morph",
        use_transition_split=False,
        policy_id="ceres_escape",
    ),
)

CERES_DOOR_EDGES: tuple[DoorEdge, ...] = (
    DoorEdge(
        "ceres_elevator_to_falling",
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_FALLING,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_falling_to_magnet",
        ROOM_CERES_FALLING,
        ROOM_CERES_MAGNET,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_magnet_to_scientist",
        ROOM_CERES_MAGNET,
        ROOM_CERES_SCIENTIST,
        "bottom_right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_scientist_to_flat",
        ROOM_CERES_SCIENTIST,
        ROOM_CERES_FLAT,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_flat_to_ridley",
        ROOM_CERES_FLAT,
        ROOM_CERES_RIDLEY,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_ridley_to_flat",
        ROOM_CERES_RIDLEY,
        ROOM_CERES_FLAT,
        "left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_flat_to_scientist",
        ROOM_CERES_FLAT,
        ROOM_CERES_SCIENTIST,
        "left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_scientist_to_magnet",
        ROOM_CERES_SCIENTIST,
        ROOM_CERES_MAGNET,
        "left",
        "bottom_right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_magnet_to_falling",
        ROOM_CERES_MAGNET,
        ROOM_CERES_FALLING,
        "upper_left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_falling_to_elevator",
        ROOM_CERES_FALLING,
        ROOM_CERES_ELEVATOR,
        "left",
        "bottom",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_to_landing",
        ROOM_CERES_ELEVATOR,
        ROOM_LANDING_SITE,
        "elevator",
        "ship",
        policy_id="ceres_escape",
        verification="continuous",
    ),
)

CERES_MILESTONES: tuple[ProgressionMilestone, ...] = (
    ProgressionMilestone(
        "first_ceres_control",
        "First controllable Ceres frame",
        ProgressCondition(room_id=ROOM_CERES_ELEVATOR, game_states=frozenset({8})),
        timeout_frames=12_000,
        policy_id="power_on_boot",
    ),
    ProgressionMilestone(
        "ridley_countdown",
        "Natural Ceres countdown",
        ProgressCondition(room_id=ROOM_CERES_RIDLEY, game_states=frozenset({8})),
        timeout_frames=7_000,
        policy_id="ceres_ridley_tail_tank",
    ),
    ProgressionMilestone(
        "zebes_landing",
        "Zebes Landing Site control",
        ProgressCondition(room_id=ROOM_LANDING_SITE, game_states=frozenset({8})),
        timeout_frames=8_000,
        policy_id="ceres_escape",
    ),
)

__all__ = [
    "CERES_SPINE",
    "CERES_DOOR_EDGES",
    "CERES_MILESTONES",
    "CERES_SCIENTIST_MAX_FRAMES",
    "play_boot_to_ceres",
    "play_boot_to_ceres_tas",
    "load_tas_ceres_hops",
    "tas_hop_clock",
    "ceres_hops_vs_tas",
    "DEFAULT_TAS_CLOCK",
    "TAS_CLOCK_ROOM_FLIP",
    "TAS_CLOCK_SETTLED_GS8",
    "_boot_spans",
    "_BOOT_MENU_MASH_FRAMES",
    "_BOOT_MAX_FRAMES",
]
