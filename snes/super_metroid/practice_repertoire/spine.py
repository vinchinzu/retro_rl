"""Product route spine: session map, route edges, hop keys, recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from super_metroid.practice_repertoire.catalog import (
    GRADES,
    PRODUCT_CATEGORY,
    RepertoireSession,
    get_session,
    neighbors,
    route_sessions,
    sessions,
)

try:
    from super_metroid.hop_id import make_hop_key
except ImportError:
    from super_metroid.skill_bank import make_hop_key

# Map practice-hack session ids → guided_human start presets / living pins.
# Prefer full_start_v1 item seams so demos, policies, and route edges share lineage.
PRODUCT_SESSION_MAP: dict[str, dict[str, str]] = {
    "kpdr25/crateria/ceres_elevator": {
        "start_preset": "start",
        "note": "power-on / Ceres elevator (no state)",
    },
    "kpdr25/crateria/ship": {
        "start_preset": "start",
        "note": "Landing Site after Ceres — use power-on continuous prefix",
    },
    "kpdr25/crateria/morph": {
        "start_preset": "morph",
        "state": "scratch/full_start_v1_morph.state",
    },
    "kpdr25/crateria/bomb_torizo": {
        "start_preset": "bomb",
        "state": "scratch/full_start_v1_bomb.state",
    },
    "kpdr25/brinstar/big_pink": {
        "start_preset": "big-pink",
        "state": "dev_b1_bigpink_main_controller.state",
    },
    "kpdr25/brinstar/below_spazer": {
        "start_preset": "below-spazer",
        "state": "scratch/post_below_spazer_with_charge_continuous.state",
    },
    "kpdr25/brinstar/spazer": {
        "start_preset": "post-spazer",
        "state": "scratch/post_spazer_collect_pure.state",
    },
    "kpdr25/kraid/leaving_varia": {
        "start_preset": "varia",
        "state": "scratch/full_start_v1_varia.state",
    },
    "kpdr25/upper_norfair/business_center_postelev": {
        "start_preset": "business",
        "state": "scratch/post_business_continuous.state",
    },
    "kpdr25/upper_norfair/hijump_etank": {
        "start_preset": "hj",
        "state": "scratch/full_start_v1_hj.state",
    },
    "kpdr25/upper_norfair/bubble_mountain": {
        "start_preset": "bubble-human",
        "state": "scratch/full_start_v1_bubble.state",
    },
    "kpdr25/upper_norfair/bat_cave": {
        "start_preset": "bat-cave",
        "state": "scratch/full_start_v1_bat.state",
    },
    "kpdr25/upper_norfair/speed_hallway": {
        "start_preset": "speed-hall",
        "state": "scratch/post_speed_hall_pre_speed_with_spazer.state",
    },
    "kpdr25/upper_norfair/double_chamber": {
        "start_preset": "double-chamber",
        "state": "scratch/post_single_to_double_chamber_continuous_like.state",
    },
    "kpdr25/upper_norfair/ice_escape": {
        "start_preset": "wave",
        "note": "post-Ice on practice route; product pin is full_start wave/ice",
        "state": "scratch/full_start_v1_wave.state",
    },
    "kpdr25/red_brinstar/alpha_power_bombs": {
        "start_preset": "alpha-pb",
        "state": "scratch/full_start_v1_alpha_pb.state",
    },
    "kpdr25/wrecked_ship/phantoon": {
        "start_preset": "phantoon",
        "state": "scratch/full_start_v1_phantoon_mid.state",
    },
    "kpdr25/wrecked_ship/leaving_gravity": {
        "start_preset": "gravity",
        "state": "scratch/full_start_v1_gravity.state",
    },
    "kpdr25/wrecked_ship/entering_wrecked_ship": {
        "start_preset": "ws-entrance",
        "state": "scratch/post_west_ocean_ws_spark.state",
    },
    "kpdr25/maridia/draygon": {
        "start_preset": "post-draygon",
        "state": "scratch/post_draygon_precious.state",
        "note": "practice entry ≈ post-Draygon Precious pin",
    },
    "kpdr25/maridia/botwoon": {
        "start_preset": "main-street",
        "state": "scratch/full_start_v1_main_street.state",
        "note": "full_start_v1 Grapple→Main Street living pin",
    },
    "kpdr25/maridia/plasma_beam": {
        "start_preset": "plasma-beam",
        "state": "scratch/full_start_v1_plasma.state",
        "note": "product pin is Plasma Room post-collect (0xD2AA beams 0x100F)",
    },
    "kpdr25/red_brinstar/caterpillars_up": {
        "start_preset": "post-gravity",
        "state": "scratch/post_gravity_caterpillar.state",
    },
}


@dataclass(frozen=True)
class RouteEdge:
    """One route edge: entry session → hop → leave/next session."""

    from_session: str
    to_session: str
    hop_key: str
    room_id: int | None
    from_room_id: int | None
    to_room_id: int | None
    items: int | None
    entry_state: str | None
    leave_state: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.room_id is not None:
            d["room_hex"] = f"0x{self.room_id:04X}"
        return d


@dataclass(frozen=True)
class RecoveryHint:
    """Autopilot / thrash reseed suggestion from live room+items."""

    session_id: str
    room_id: int
    items: int | None
    state_path: str | None
    hop_key: str | None
    next_session_id: str | None
    grade: str
    score: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hop_key_for_session(
    session: RepertoireSession,
    *,
    next_session: RepertoireSession | None = None,
    prev_session: RepertoireSession | None = None,
) -> str | None:
    """Hop identity from this pin toward the next route pin."""
    if session.room_id is None:
        return None
    if next_session is None and prev_session is None:
        prev_session, next_session = neighbors(session.id, category=session.category)
    to_room = next_session.room_id if next_session is not None else None
    from_room = prev_session.room_id if prev_session is not None else None
    return make_hop_key(
        session.room_id,
        from_room_id=from_room,
        to_room_id=to_room,
        items=session.items,
    )


def route_edge(session_id: str) -> RouteEdge | None:
    """Edge from this session to the next route session."""
    s = get_session(session_id)
    prev_s, next_s = neighbors(session_id)
    if next_s is None:
        return None
    entry = s.resolve_state_path()
    leave = next_s.resolve_state_path()
    return RouteEdge(
        from_session=s.id,
        to_session=next_s.id,
        hop_key=hop_key_for_session(s, next_session=next_s, prev_session=prev_s) or "",
        room_id=s.room_id,
        from_room_id=prev_s.room_id if prev_s else None,
        to_room_id=next_s.room_id,
        items=s.items,
        entry_state=str(entry) if entry else None,
        leave_state=str(leave) if leave else None,
    )


def product_route_edges(category: str = PRODUCT_CATEGORY) -> list[RouteEdge]:
    """All ordered route edges for a category route."""
    route = route_sessions(category)
    out: list[RouteEdge] = []
    for s in route:
        edge = route_edge(s.id)
        if edge is not None:
            out.append(edge)
    return out


def recover_session(
    room_id: int,
    items: int | None = None,
    *,
    category: str = PRODUCT_CATEGORY,
    prefer_living: bool = True,
) -> RecoveryHint | None:
    """Pick the best repertoire pin for autopilot / thrash recovery.

    Ranking: exact room → inventory distance → living pin → route order.
    """
    # Lazy import: graduation_status lives in board (board imports spine).
    from super_metroid.practice_repertoire.board import graduation_status

    room_id = int(room_id)
    cands = [s for s in route_sessions(category) if s.room_id == room_id]
    if not cands:
        # Fall back: any category session in this room (still KPDR-ish)
        cands = [s for s in sessions(category=category) if s.room_id == room_id]
    if not cands:
        return None

    def score(s: RepertoireSession) -> tuple:
        inv_dist = 0
        if items is not None and s.items is not None:
            inv_dist = bin(int(items) ^ int(s.items)).count("1")
        elif items is not None:
            inv_dist = 8
        living = s.living_state_path()
        has_living = bool(living and living.is_file())
        has_canon = s.canonical_state_path.is_file()
        grade = graduation_status(s)
        grade_rank = GRADES.index(grade) if grade in GRADES else 0
        # Prefer living product pins when inventory is close (≤2 bits off),
        # otherwise exact inventory match wins for thrash reseed fidelity.
        inv_bucket = 0 if inv_dist <= 2 else inv_dist
        return (
            inv_bucket,
            -grade_rank,
            0 if (prefer_living and has_living) else 1,
            inv_dist,
            0 if has_canon else 1,
            s.route_index if s.route_index >= 0 else 9999,
            s.id,
        )

    best = min(cands, key=score)
    prev_s, next_s = neighbors(best.id, category=category)
    state = best.resolve_state_path()
    hop = hop_key_for_session(best, next_session=next_s, prev_session=prev_s)
    inv_bits = 0
    if items is not None and best.items is not None:
        inv_bits = bin(int(items) ^ int(best.items)).count("1")
    return RecoveryHint(
        session_id=best.id,
        room_id=room_id,
        items=best.items,
        state_path=str(state) if state else None,
        hop_key=hop,
        next_session_id=next_s.id if next_s else None,
        grade=graduation_status(best),
        score=float(inv_bits),
        detail=(
            f"repertoire recovery {best.id} grade={graduation_status(best)} "
            f"inv_xor_bits={inv_bits}"
        ),
    )


def recovery_hint_for_state(state: Any, *, category: str = PRODUCT_CATEGORY) -> RecoveryHint | None:
    """Convenience: accept SuperMetroidState (or duck-typed room/items)."""
    room = int(getattr(state, "room_id", 0) or 0)
    if not room:
        return None
    items = getattr(state, "collected_items", None)
    if items is None:
        items = getattr(state, "items", None)
    return recover_session(room, int(items) if items is not None else None, category=category)
