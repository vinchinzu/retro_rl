"""Control-relative room stage table for Super Metroid TAS adapt.

Mirrors the *concept* of ``smb.tas.stages.StageSpec`` (control gate → body →
goal) without cloning NES FM2 plumbing. Stages are **room-ID gated** under
stable-retro settle rules; movie frame indices are **hints only**.

Hard rules (see ``docs/TAS_ADAPT.md``):

* Never sanitize L+R on TAS bodies (raw SNES-12).
* Do not STATUS-claim from movie indices alone — re-anchor first.
* Product pure owns multi-room continuity; movie bodies are splice candidates.

Usage
-----

* Catalog early Ceres + morph-spine product pins for re-slice export.
* ``export_room_body`` (stub API) slices movie frames between room_enter pins
  when an annotated ``pins.json`` / ``trace.json`` is available.
* Future chain player loads a ``.state`` at control, plays the body, checks goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from super_metroid.paths import GAME_DIR, RECORDINGS_DIR
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
    ROOM_CLIMB,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.tas.annotate import is_settled_control

TAS_DIR = GAME_DIR / "tas"
REF_ANY = TAS_DIR / "ref" / "sniq_any_3653M.lsmv"
REF_100 = TAS_DIR / "ref" / "sniq_100p.bk2"
TAS_IMPORT = RECORDINGS_DIR / "tas_import"

# Verified harness pins (power-on any% annotate, 2026-08-07).
# Movie indices are *hints*; control is room_id + settle.
ANY_FIRST_CONTROL_FRAME = 11_182
ANY_FIRST_CONTROL_ROOM = ROOM_CERES_ELEVATOR  # 0xDF45

# Product morph spine (continuous pure; not movie indices).
PRODUCT_LANDING_FRAME = 21_548  # approximate continuous morph dual pin
PRODUCT_MORPH_FRAME = 26_824

# Known-good TAS movie splice (Landing → Parlor under Sniq any%).
ANY_LANDING_MOVIE_START = 15_000
ANY_LANDING_BODY_HINT = 12_000


# ---------------------------------------------------------------------------
# Control / goal predicates
# ---------------------------------------------------------------------------

StateLike = Any  # SuperMetroidState or pin dict with room_id / game_state / …


def _room_id(state: StateLike) -> int:
    if hasattr(state, "room_id"):
        return int(state.room_id)
    if isinstance(state, Mapping):
        if "room_id" in state:
            return int(state["room_id"])
        if "roomId" in state:
            return int(state["roomId"])
        room = state.get("room")
        if isinstance(room, str) and room.startswith("0x"):
            return int(room, 16)
    raise TypeError(f"cannot read room_id from {type(state)!r}")


def is_room_settled(state: StateLike, room_id: int) -> bool:
    """True when *state* is ordinary control in *room_id*.

    Settle rule matches RoomTimer / Annotator: gs==8, door_transition==0,
    room_id!=0, ordinary phase when available.
    """
    rid = int(room_id)
    if hasattr(state, "phase") or hasattr(state, "game_state"):
        try:
            if not is_settled_control(state):  # type: ignore[arg-type]
                return False
        except Exception:
            # Pin dicts / partial objects: fall through to field checks.
            pass
        return _room_id(state) == rid

    if isinstance(state, Mapping):
        gs = state.get("game_state", state.get("gs", 8))
        door = state.get("door_transition", 0)
        phase = state.get("phase", "ORDINARY_GAMEPLAY")
        if int(gs) != 8 or int(door) != 0:
            return False
        if phase not in (None, "ORDINARY_GAMEPLAY", 8, "8"):
            # Allow missing phase; reject known non-ordinary strings.
            if isinstance(phase, str) and phase not in (
                "ORDINARY_GAMEPLAY",
                "ordinary",
            ):
                return False
        return _room_id(state) == rid

    return _room_id(state) == rid


def control_in(room_id: int) -> Callable[[StateLike], bool]:
    """Factory: control predicate for a single room."""

    def _pred(state: StateLike) -> bool:
        return is_room_settled(state, room_id)

    _pred.__name__ = f"control_0x{room_id:04X}"
    _pred.__doc__ = f"Settled control in room 0x{room_id:04X}."
    return _pred


def goal_enter(room_id: int) -> Callable[[StateLike], bool]:
    """Factory: goal when settled in *room_id* (door exit completed)."""

    def _pred(state: StateLike) -> bool:
        return is_room_settled(state, room_id)

    _pred.__name__ = f"goal_enter_0x{room_id:04X}"
    _pred.__doc__ = f"Settled enter room 0x{room_id:04X}."
    return _pred


class GoalKind(str, Enum):
    """How a room-body probe decides success."""

    ENTER_ROOM = "enter_room"  # settled in goal_room_id
    ITEM_BIT = "item_bit"  # collected_items & mask
    BEAM_BIT = "beam_bit"  # collected_beams & mask
    CUSTOM = "custom"  # use StageSpec.goal callable only


@dataclass(frozen=True)
class RoomStageSpec:
    """One control-relative TAS / product body leg (room hop or tech window).

    Fields
    ------
    id:
        Stable stage id (``ceres_elev_to_falling``, ``landing_to_parlor``, …).
    room_id:
        Control room (body starts when settled here).
    goal_room_id:
        Default goal: settled enter of this room (when goal_kind is ENTER_ROOM).
    movie_start / body_frames:
        **Hints** from Sniq movies — never STATUS evidence alone.
    track:
        ``product`` | ``tas_any`` | ``tas_100`` | ``hybrid`` — write-root isolation.
    """

    id: str
    room_id: int
    goal_room_id: int | None = None
    goal_kind: GoalKind = GoalKind.ENTER_ROOM
    goal_mask: int = 0
    movie: Path | None = None
    movie_start: int | None = None
    body_frames: int | None = None
    seed_path: Path | None = None
    state_hint: Path | None = None  # dumped .state under tas_import if known
    predecessor: str = ""
    track: str = "tas_any"
    tags: tuple[str, ...] = ()
    tech: tuple[str, ...] = ()  # mockball, door_speed, shine, arm_pump, wj, …
    note: str = ""
    # Optional explicit callables (default: room settle factories).
    control_fn: Callable[[StateLike], bool] | None = field(default=None, repr=False)
    goal_fn: Callable[[StateLike], bool] | None = field(default=None, repr=False)

    def control(self, state: StateLike) -> bool:
        if self.control_fn is not None:
            return bool(self.control_fn(state))
        return is_room_settled(state, self.room_id)

    def goal(self, state: StateLike) -> bool:
        if self.goal_fn is not None:
            return bool(self.goal_fn(state))
        if self.goal_kind is GoalKind.ENTER_ROOM:
            if self.goal_room_id is None:
                return False
            return is_room_settled(state, self.goal_room_id)
        if self.goal_kind is GoalKind.ITEM_BIT:
            items = int(getattr(state, "collected_items", 0) or 0)
            if isinstance(state, Mapping):
                raw = state.get("collected_items", state.get("items", 0))
                if isinstance(raw, str):
                    items = int(raw, 16) if raw.startswith("0x") else int(raw)
                else:
                    items = int(raw or 0)
            return bool(items & self.goal_mask)
        if self.goal_kind is GoalKind.BEAM_BIT:
            beams = int(getattr(state, "collected_beams", 0) or 0)
            if isinstance(state, Mapping):
                raw = state.get("collected_beams", state.get("beams", 0))
                if isinstance(raw, str):
                    beams = int(raw, 16) if raw.startswith("0x") else int(raw)
                else:
                    beams = int(raw or 0)
            return bool(beams & self.goal_mask)
        return False

    def room_hex(self) -> str:
        return f"0x{self.room_id:04X}"

    def goal_room_hex(self) -> str | None:
        if self.goal_room_id is None:
            return None
        return f"0x{self.goal_room_id:04X}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "room_id": self.room_id,
            "room_id_hex": self.room_hex(),
            "goal_room_id": self.goal_room_id,
            "goal_room_id_hex": self.goal_room_hex(),
            "goal_kind": self.goal_kind.value,
            "goal_mask": self.goal_mask,
            "movie": str(self.movie) if self.movie else None,
            "movie_start": self.movie_start,
            "body_frames": self.body_frames,
            "seed_path": str(self.seed_path) if self.seed_path else None,
            "state_hint": str(self.state_hint) if self.state_hint else None,
            "predecessor": self.predecessor,
            "track": self.track,
            "tags": list(self.tags),
            "tech": list(self.tech),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Catalog — early Ceres (TAS power-on usable) + product morph spine
# ---------------------------------------------------------------------------

_ANY_OPEN = TAS_IMPORT / "sniq_any_full"
_PRODUCT = TAS_IMPORT / "product_morph_annotate"


def _stage(
    id: str,
    room: int,
    goal: int | None,
    *,
    movie: Path | None = REF_ANY,
    movie_start: int | None = None,
    body_frames: int | None = None,
    predecessor: str = "",
    track: str = "tas_any",
    tags: tuple[str, ...] = (),
    tech: tuple[str, ...] = (),
    note: str = "",
    state_hint: Path | None = None,
    goal_kind: GoalKind = GoalKind.ENTER_ROOM,
    goal_mask: int = 0,
) -> RoomStageSpec:
    return RoomStageSpec(
        id=id,
        room_id=room,
        goal_room_id=goal,
        goal_kind=goal_kind,
        goal_mask=goal_mask,
        movie=movie,
        movie_start=movie_start,
        body_frames=body_frames,
        predecessor=predecessor,
        track=track,
        tags=tags,
        tech=tech,
        note=note,
        state_hint=state_hint,
    )


STAGE_CATALOG: dict[str, RoomStageSpec] = {
    # --- Ceres (power-on any% first_control → thrash; still useful early) ---
    "ceres_first_control": _stage(
        "ceres_first_control",
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_FALLING,
        movie_start=ANY_FIRST_CONTROL_FRAME,
        body_frames=6_640,  # ~to first Falling enter under any% annotate
        tags=("ceres", "boot", "control"),
        tech=("walljump",),
        note=(
            "Harness first_control @ f11182 room 0xDF45 (any%). "
            "Usable pin for TAS boot residual; body desyncs mid-Ceres."
        ),
        state_hint=_ANY_OPEN / "states" / "f011182_control_rDF45.state",
    ),
    "ceres_elev_to_falling": _stage(
        "ceres_elev_to_falling",
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_FALLING,
        movie_start=ANY_FIRST_CONTROL_FRAME,
        body_frames=6_640,
        predecessor="ceres_first_control",
        tags=("ceres", "hop"),
        tech=("walljump", "arm_pump"),
        note="Elev shaft → Falling Tile. Product shaft owns multi-room; TAS for residual only.",
    ),
    "ceres_falling_to_magnet": _stage(
        "ceres_falling_to_magnet",
        ROOM_CERES_FALLING,
        ROOM_CERES_MAGNET,
        tags=("ceres", "hop"),
        note="Falling → Magnet Stairs. Prefer product pure for continuity.",
    ),
    "ceres_magnet_to_scientist": _stage(
        "ceres_magnet_to_scientist",
        ROOM_CERES_MAGNET,
        ROOM_CERES_SCIENTIST,
        tags=("ceres", "hop"),
    ),
    "ceres_scientist_to_flat": _stage(
        "ceres_scientist_to_flat",
        ROOM_CERES_SCIENTIST,
        ROOM_CERES_FLAT,
        tags=("ceres", "hop"),
    ),
    "ceres_flat_to_ridley": _stage(
        "ceres_flat_to_ridley",
        ROOM_CERES_FLAT,
        ROOM_CERES_RIDLEY,
        tags=("ceres", "hop", "boss"),
        note="Ceres Ridley room enter — combat deferred; door path only.",
    ),
    # --- Product morph spine (owns multi-room Zebes; movie optional splice) ---
    "landing_to_parlor": _stage(
        "landing_to_parlor",
        ROOM_LANDING_SITE,
        ROOM_PARLOR,
        movie_start=ANY_LANDING_MOVIE_START,
        body_frames=ANY_LANDING_BODY_HINT,
        track="hybrid",
        tags=("zebes", "morph_spine", "splice"),
        tech=("door_speed",),
        note=(
            "Product Landing pin + Sniq movie@15000 enters Parlor (verified). "
            "Later thrash without Climb pin."
        ),
    ),
    "parlor_to_climb": _stage(
        "parlor_to_climb",
        ROOM_PARLOR,
        ROOM_CLIMB,
        track="product",
        tags=("zebes", "morph_spine"),
        note="Prefer product pure; open-loop movie Climb needs tighter pin.",
    ),
    "climb_to_pit": _stage(
        "climb_to_pit",
        ROOM_CLIMB,
        ROOM_PIT,
        track="product",
        tags=("zebes", "morph_spine"),
        note="Skip TAS for Climb (mostly-fall seed).",
    ),
    "pit_to_elev": _stage(
        "pit_to_elev",
        ROOM_PIT,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        track="product",
        tags=("zebes", "morph_spine"),
        tech=("dash_jump",),
        note=(
            "Product first-jump land (195,123) pose 9 then seed tail f198–809. "
            "Do not open-loop full movie body in Pit."
        ),
    ),
    "elev_to_morph": _stage(
        "elev_to_morph",
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        ROOM_MORPH,
        track="product",
        tags=("zebes", "morph_spine", "item"),
        tech=("morph",),
        goal_kind=GoalKind.ENTER_ROOM,
        note="BB elev → Morph ball room; morph bit is separate item goal.",
    ),
    # --- Product Ice path gap (P0 pure; TAS informs residual only) ---
    "ice_gate_to_acid": _stage(
        "ice_gate_to_acid",
        ROOM_ICE_GATE,
        ROOM_ICE_ACID,
        movie=None,
        track="product",
        tags=("norfair", "ice", "product_p0"),
        note="Gate→Acid dual GREEN (rr-9t4). TAS not required.",
    ),
    "ice_acid_to_snake": _stage(
        "ice_acid_to_snake",
        ROOM_ICE_ACID,
        ROOM_ICE_SNAKE,
        movie=None,
        track="product",
        tags=("norfair", "ice", "product_p0"),
        tech=(),
        note="Acid→Snake dual GREEN (rr-5cf) horizontal RLE. 2WJ is Snake→Ice (rr-5if).",
    ),
}


def get_stage(stage_id: str) -> RoomStageSpec:
    try:
        return STAGE_CATALOG[stage_id]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_CATALOG))
        raise KeyError(f"unknown stage {stage_id!r}; known: {known}") from exc


def stages_for_track(track: str) -> list[RoomStageSpec]:
    return [s for s in STAGE_CATALOG.values() if s.track == track]


def stages_with_tag(tag: str) -> list[RoomStageSpec]:
    return [s for s in STAGE_CATALOG.values() if tag in s.tags]


# ---------------------------------------------------------------------------
# Control-relative body export (from annotated hops — open TAS_ADAPT item)
# ---------------------------------------------------------------------------


def movie_window_from_pins(
    pins: Sequence[Mapping[str, Any]],
    *,
    from_room: int,
    to_room: int,
    after_frame: int = 0,
) -> tuple[int, int] | None:
    """Return ``(start_frame, end_frame)`` for first hop from→to after *after_frame*.

    Uses ``room_enter`` pins: start = enter of *from_room* (or prior leave),
    end = enter of *to_room*. Returns None if the hop is not found.

    This is a **movie-index hint** extracted from a harness annotate pass —
    re-validate under a control-relative state load before pure adoption.
    """
    enters = [
        p
        for p in pins
        if p.get("kind") == "room_enter" and int(p.get("frame", 0)) >= after_frame
    ]
    # Walk consecutive enters for from_room → to_room.
    for i, pin in enumerate(enters):
        rid = int(pin.get("room_id") or 0)
        if rid != int(from_room):
            continue
        start = int(pin["frame"])
        for nxt in enters[i + 1 :]:
            if int(nxt.get("room_id") or 0) == int(to_room):
                end = int(nxt["frame"])
                if end > start:
                    return start, end
                break
    return None


def export_room_body_spec(
    stage: RoomStageSpec,
    pins: Sequence[Mapping[str, Any]],
    *,
    after_frame: int = 0,
) -> dict[str, Any]:
    """Build a control-relative export descriptor for *stage* from annotate pins.

    Does **not** write movie bytes — returns a machine-readable plan:

    * control room + settle rule
    * movie window hint (if pins contain the hop)
    * state dump path (if stage.state_hint)
    * goal predicate description

    Downstream ``export_slices`` / RLE writers can materialize seeds from this.
    """
    window = None
    if stage.goal_room_id is not None:
        window = movie_window_from_pins(
            pins,
            from_room=stage.room_id,
            to_room=stage.goal_room_id,
            after_frame=after_frame,
        )
    start_hint = stage.movie_start
    body_hint = stage.body_frames
    if window is not None:
        start_hint = window[0]
        body_hint = window[1] - window[0]

    return {
        "schema": "sm_tas_room_body_v1",
        "stage_id": stage.id,
        "track": stage.track,
        "control": {
            "room_id": stage.room_id,
            "room_id_hex": stage.room_hex(),
            "settle": "gs==8 && door_transition==0 && room_id!=0 && ordinary",
        },
        "goal": {
            "kind": stage.goal_kind.value,
            "room_id": stage.goal_room_id,
            "room_id_hex": stage.goal_room_hex(),
            "mask": stage.goal_mask,
        },
        "movie": str(stage.movie) if stage.movie else None,
        "movie_start": start_hint,
        "body_frames": body_hint,
        "window_from_pins": list(window) if window else None,
        "state_hint": str(stage.state_hint) if stage.state_hint else None,
        "tech": list(stage.tech),
        "tags": list(stage.tags),
        "note": stage.note,
        "status": "plan_only",  # not pure-proven; not STATUS
        "hard_rules": [
            "never_sanitize_L+R",
            "assist_off",
            "re_anchor_before_status",
            "pure_first_before_graph_edge",
        ],
    }


__all__ = [
    "ANY_FIRST_CONTROL_FRAME",
    "ANY_FIRST_CONTROL_ROOM",
    "ANY_LANDING_BODY_HINT",
    "ANY_LANDING_MOVIE_START",
    "GoalKind",
    "REF_100",
    "REF_ANY",
    "RoomStageSpec",
    "STAGE_CATALOG",
    "control_in",
    "export_room_body_spec",
    "get_stage",
    "goal_enter",
    "is_room_settled",
    "movie_window_from_pins",
    "stages_for_track",
    "stages_with_tag",
]
