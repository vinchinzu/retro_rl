"""Multi-truth route anchors for the ALTTP opening continuous path.

Separates **route**, **approach**, and **trigger** tiers (see
``docs/TRIGGER_HANDOFF.md`` and root ``ARCHITECTURE_AND_CLEANUP_PLAN.md``).

Each anchor may carry:

- **RAM** — screen/room/inventory/position window (gameplay authority)
- **map/Yaze** — hole tile, entrance id, world coords (nav association)
- **visual** — optional screenshot path under ``recordings/``

Do not treat screen-id alone as a route checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    HYRULE_CASTLE_SCREEN,
    SANCTUARY_ROOM,
    SECRET_HOLE_APPROACH_TOLERANCE,
    SECRET_HOLE_WORLD_X,
    SECRET_HOLE_WORLD_Y,
    SECRET_PASSAGE_ROOM,
    ZELDA_CELL_ROOM,
    AlttpSnapshot,
)
from alttp.room_map import load_room_map

# Room 0x55 multi-chamber: south guards chamber is clearly past uncle y.
# Used only for continuous tip resolution when the fighter-sword anchor matches
# without a position window (same room_base_id for uncle/sword/south).
# Shared with secret_entrance_clear.approach_south_chamber success y.
ROOM_55_SOUTH_Y_MIN = 2850

# Outdoor landing after secret-entrance stairs (measured 2026-07-30).
COURTYARD_SECRET_POCKET_X = 2248
COURTYARD_SECRET_POCKET_Y = 1755
COURTYARD_SECRET_POCKET_TOLERANCE = 48

# South-chamber stairs alignment (trigger tier).
STAIRS_ALIGN_X = 2672
STAIRS_ALIGN_Y = 2916
STAIRS_ALIGN_TOLERANCE = 6

# Main castle door (measured 2026-07-30 from CastleMain exit + pocket entry).
MAIN_DOOR_APPROACH_X = 2040
MAIN_DOOR_APPROACH_Y = 1790
MAIN_DOOR_APPROACH_TOLERANCE = 24

AnchorTier = str  # "route" | "approach" | "trigger"


@dataclass(frozen=True)
class PositionWindow:
    """World (or room) coordinate acceptance window."""

    x: int
    y: int
    tolerance: int = 32
    label: str = ""

    def contains(self, x: int, y: int) -> bool:
        return (
            abs(int(x) - self.x) <= self.tolerance
            and abs(int(y) - self.y) <= self.tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "tolerance": self.tolerance,
            "label": self.label,
        }


def _door_approach_window(
    map_id: str,
    door_label: str,
    *,
    tolerance: int,
    label: str,
) -> PositionWindow:
    """Build a position window from map door approach (geometry authority)."""
    door = load_room_map(map_id).door(door_label)
    if door is None:
        raise KeyError(f"door {door_label!r} not in map {map_id!r}")
    ax, ay = door.approach_xy
    return PositionWindow(ax, ay, tolerance, label=label)


@dataclass(frozen=True)
class MultiTruthAnchor:
    """Named semantic anchor with RAM + optional map + visual evidence.

    ``anchor_id`` is stable and meaning-bearing (not just a screen name).
    """

    anchor_id: str
    name: str
    tier: AnchorTier
    # RAM predicates
    screen_id: int | None = None
    room_base_id: int | None = None
    require_indoors: bool | None = None
    require_fighter_sword: bool = False
    require_control: bool = True
    position: PositionWindow | None = None
    # Map / Yaze association (not gameplay authority alone)
    yaze_entrance_id: int | None = None
    map_note: str = ""
    # Visual
    screenshot_hint: str = ""
    # Graph node linkage
    graph_node_id: str = ""
    notes: tuple[str, ...] = ()
    # Optional custom matcher (for complex cases)
    _extra_match: Callable[[AlttpSnapshot], bool] | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def matches(self, snapshot: AlttpSnapshot) -> bool:
        """Return True when RAM (and optional extras) satisfy this anchor."""
        if self.require_control and not snapshot.has_control:
            return False
        if self.screen_id is not None and int(snapshot.screen_id) != self.screen_id:
            if not snapshot.indoors:  # outdoor anchors require screen
                return False
            if self.require_indoors is not True:
                return False
        if self.room_base_id is not None:
            if not snapshot.indoors or snapshot.room_base_id != self.room_base_id:
                return False
        if self.require_indoors is True and not snapshot.indoors:
            return False
        if self.require_indoors is False and snapshot.indoors:
            return False
        if self.require_fighter_sword and not snapshot.has_fighter_sword:
            return False
        if self.position is not None and not self.position.contains(
            snapshot.link_x, snapshot.link_y
        ):
            return False
        if self._extra_match is not None and not self._extra_match(snapshot):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchorId": self.anchor_id,
            "name": self.name,
            "tier": self.tier,
            "screenId": self.screen_id,
            "roomBaseId": self.room_base_id,
            "requireIndoors": self.require_indoors,
            "requireFighterSword": self.require_fighter_sword,
            "requireControl": self.require_control,
            "position": self.position.to_dict() if self.position else None,
            "yazeEntranceId": self.yaze_entrance_id,
            "mapNote": self.map_note,
            "screenshotHint": self.screenshot_hint,
            "graphNodeId": self.graph_node_id,
            "notes": list(self.notes),
        }


def _outdoor_screen(snap: AlttpSnapshot, screen: int) -> bool:
    return (not snap.indoors) and (not snap.dark_world) and snap.screen_id == screen


# ---------------------------------------------------------------------------
# Opening-route anchors (semantic names)
# ---------------------------------------------------------------------------


def opening_anchors() -> tuple[MultiTruthAnchor, ...]:
    """Canonical multi-truth anchors for the continuous opening spine."""
    return (
        MultiTruthAnchor(
            anchor_id="HyruleCastle_GroundsSpawn_Controllable",
            name="Castle grounds spawn (controllable)",
            tier="route",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            graph_node_id="castle_grounds",
            map_note="Screen 0x1B only — not a precise bridge/turn anchor",
            screenshot_hint="recordings/castle_grounds.png",
            notes=(
                "Dev state HyruleCastleGrounds means controllable on screen 0x1B;",
                "not 'bridge turn east' or 'secret-hole approach'.",
            ),
            _extra_match=lambda s: _outdoor_screen(s, HYRULE_CASTLE_SCREEN),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretPassageApproach",
            name="Secret-hole approach (near bush)",
            tier="approach",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            position=PositionWindow(
                SECRET_HOLE_WORLD_X,
                SECRET_HOLE_WORLD_Y,
                SECRET_HOLE_APPROACH_TOLERANCE,
                label="secret_hole_approach",
            ),
            yaze_entrance_id=0x7D,
            map_note="Yaze entrance 0x7D @ world (2432,1696)",
            screenshot_hint="recordings/debug_nav/exact_hole.png",
            graph_node_id="castle_grounds",
            notes=("Approach pocket before bush-lift; not yet in room 0x55.",),
            _extra_match=lambda s: s.near_secret_hole,
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretPassageExactTile",
            name="Secret-hole bush-lift + drop trigger",
            tier="trigger",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            position=PositionWindow(
                SECRET_HOLE_WORLD_X,
                SECRET_HOLE_WORLD_Y,
                16,
                label="secret_hole_exact",
            ),
            yaze_entrance_id=0x7D,
            map_note="Hitbox: face UP, A×4, wait 20, UP×56 → room 0x55",
            notes=(
                "Trigger/hitbox problem — see docs/TRIGGER_HANDOFF.md.",
                "Proven SECRET_HOLE_ENTRY_SCRIPT; min UP after A/wait = 40.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretEntrance_UncleChamber",
            name="Secret entrance uncle chamber (room 0x55)",
            tier="route",
            room_base_id=SECRET_PASSAGE_ROOM,
            require_indoors=True,
            graph_node_id="room_55_uncle",
            notes=("RAM room base 0x55 after hole drop; sword may still be 0.",),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretEntrance_FighterSword",
            name="Post-uncle fighter sword (room 0x55)",
            tier="route",
            room_base_id=SECRET_PASSAGE_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            graph_node_id="room_55_sword",
            notes=(
                "Dev state FighterSword is a checkpoint only — not natural-chain proof.",
                "Hold-up-item ($5D==21) needs ~95 LEFT frames to dismiss.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretEntrance_SouthChamber",
            name="Secret entrance south combat chamber",
            tier="approach",
            room_base_id=SECRET_PASSAGE_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            position=PositionWindow(2680, 2925, 80, label="south_chamber"),
            graph_node_id="room_55_south",
            notes=("Guards chamber; LEFT×100 + DOWN×250 from uncle corridor.",),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_SecretEntrance_StairsAlign",
            name="South stairs exact alignment",
            tier="trigger",
            room_base_id=SECRET_PASSAGE_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            position=PositionWindow(
                STAIRS_ALIGN_X,
                STAIRS_ALIGN_Y,
                STAIRS_ALIGN_TOLERANCE,
                label="stairs_align",
            ),
            map_note="Off-center y≥2960 soft-locks indoors without transition",
            graph_node_id="room_55_south",
            notes=("Trigger: align then DOWN → outdoors pocket.",),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_Courtyard_SecretStairsPocket",
            name="Outdoor hedge pocket after secret stairs",
            tier="route",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            require_fighter_sword=True,
            position=PositionWindow(
                COURTYARD_SECRET_POCKET_X,
                COURTYARD_SECRET_POCKET_Y,
                COURTYARD_SECRET_POCKET_TOLERANCE,
                label="secret_stairs_pocket",
            ),
            graph_node_id="courtyard_secret_pocket",
            screenshot_hint="recordings/probe_secret_exit/clear/",
            notes=(
                "Secret-entrance clear landing.",
                "UP re-enters stairs; escape requires bush-cut S/W.",
            ),
            _extra_match=lambda s: _outdoor_screen(s, HYRULE_CASTLE_SCREEN)
            and s.has_fighter_sword,
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_Courtyard_OpenGardens",
            name="Open courtyard after hedge escape (route)",
            tier="route",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            require_fighter_sword=True,
            # Loose window: flower gardens / open court south-west of pocket.
            position=PositionWindow(2180, 1910, 80, label="open_gardens"),
            graph_node_id="courtyard_secret_pocket",
            screenshot_hint="recordings/probe_courtyard_door/spiral/",
            notes=(
                "Route-tier: bush-cut south-west out of hedge pocket.",
                "Walk-only stays boxed ~48×64 at landing.",
            ),
            _extra_match=lambda s: _outdoor_screen(s, HYRULE_CASTLE_SCREEN)
            and s.has_fighter_sword
            and s.link_y >= 1880,
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainDoorApproach",
            name="Main castle door approach (outdoor)",
            tier="approach",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            require_fighter_sword=True,
            position=PositionWindow(
                MAIN_DOOR_APPROACH_X,
                MAIN_DOOR_APPROACH_Y,
                MAIN_DOOR_APPROACH_TOLERANCE,
                label="main_door_approach",
            ),
            graph_node_id="courtyard_secret_pocket",
            map_note="Reverse-measured CastleMain exit ~(2040,1779); entry ~(2040,1790)",
            screenshot_hint="recordings/probe_courtyard_door/south_door/",
            notes=(
                "Approach via south corridor y≈2024 then west to x≈2040 then north.",
                "Soldiers on the approach path — fight_nearby as needed.",
            ),
            _extra_match=lambda s: _outdoor_screen(s, HYRULE_CASTLE_SCREEN)
            and s.has_fighter_sword,
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainDoorTrigger",
            name="Main castle door entry trigger",
            tier="trigger",
            screen_id=HYRULE_CASTLE_SCREEN,
            require_indoors=False,
            require_fighter_sword=True,
            position=PositionWindow(
                MAIN_DOOR_APPROACH_X,
                MAIN_DOOR_APPROACH_Y,
                16,
                label="main_door_exact",
            ),
            map_note="Hitbox: align x≈2040, hold UP → room 0x61",
            notes=(
                "Trigger/hitbox — see docs/TRIGGER_HANDOFF.md.",
                "Proven 2026-07-30 headless from pocket predecessor.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainHall",
            name="Hyrule Castle main hall (room 0x61)",
            tier="route",
            room_base_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            graph_node_id="room_61",
            notes=(
                "Indoors after main door. Next: clear hostiles → west door → 0x60.",
                "Dev state CastleMain is a checkpoint only.",
                "Geometry: maps/room_61.json via room_map.load_room_map.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainWest_0x60",
            name="Hyrule Castle main west (room 0x60)",
            tier="route",
            room_base_id=HYRULE_CASTLE_MAIN_WEST_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            graph_node_id="room_60",
            notes=(
                "Continuous clean prefix west exit from main hall.",
                "Next: north door → room 0x50 (maps/room_60.json north_to_0x50).",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_NW_0x50",
            name="Hyrule Castle NW chamber (room 0x50)",
            tier="route",
            room_base_id=HYRULE_CASTLE_NW_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            graph_node_id="room_50",
            notes=(
                "Continuous clean-prefix north exit from 0x60. Geometry: maps/room_50.json.",
                "Next: isolate the physical exit after 0x50 before asserting Zelda path.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainHall_WestDoorApproach",
            name="Main hall west door approach (side corridor)",
            tier="approach",
            room_base_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            position=_door_approach_window(
                "room_61",
                "west_to_0x60",
                tolerance=24,
                label="main_hall_west_door",
            ),
            graph_node_id="room_61",
            map_note="maps/room_61.json door west_to_0x60; LEFT → room 0x60",
            screenshot_hint="recordings/probe_main_hall/",
            notes=(
                "Approach xy from map door (geometry authority).",
                "Pure UP on carpet mid-line wedges at y≈3352 — use side corridor.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_MainHall_WestDoorTrigger",
            name="Main hall west door exit trigger",
            tier="trigger",
            room_base_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            position=_door_approach_window(
                "room_61",
                "west_to_0x60",
                tolerance=16,
                label="main_hall_west_exact",
            ),
            map_note="Hitbox: align corridor, hold LEFT → room base 0x60",
            notes=(
                "Trigger/hitbox — see docs/TRIGGER_HANDOFF.md.",
                "Proven 2026-07-31 headless 3/3 from CastleMain.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_ZeldaCell",
            name="Zelda cell (room 0x80)",
            tier="route",
            room_base_id=ZELDA_CELL_ROOM,
            require_indoors=True,
            require_fighter_sword=True,
            graph_node_id="room_80",
            notes=(
                "Planned continuous tip — rescue not yet continuous.",
                "RAM room base 0x80; follower may still be 0 until rescue.",
            ),
        ),
        MultiTruthAnchor(
            anchor_id="HyruleCastle_Sanctuary",
            name="Sanctuary (room 0x12)",
            tier="route",
            room_base_id=SANCTUARY_ROOM,
            require_indoors=True,
            graph_node_id="sanctuary",
            notes=(
                "Planned continuous tip — escort not yet continuous.",
                "RAM room base 0x12.",
            ),
        ),
    )


# Most-specific first for continuous tip resolution (graph_node_id).
# Mantle has no RAM signature yet — stay on the graph only, not tip order.
TIP_ANCHOR_ORDER: tuple[str, ...] = (
    # Continuous post-main-hall nodes listed before earlier spine anchors.
    "HyruleCastle_NW_0x50",
    "HyruleCastle_MainWest_0x60",
    "HyruleCastle_MainHall",  # earlier continuous spine checkpoint
    "HyruleCastle_ZeldaCell",  # planned
    "HyruleCastle_Sanctuary",  # planned
    "HyruleCastle_Courtyard_SecretStairsPocket",
    "HyruleCastle_SecretEntrance_StairsAlign",
    "HyruleCastle_SecretEntrance_SouthChamber",
    "HyruleCastle_SecretEntrance_FighterSword",
    "HyruleCastle_SecretEntrance_UncleChamber",
    "HyruleCastle_MainDoorApproach",
    "HyruleCastle_GroundsSpawn_Controllable",
)


def resolve_continuous_tip_node(snapshot: AlttpSnapshot) -> str:
    """Return ``graph_node_id`` of the most specific matching tip anchor.

    Outdoor + main hall + Zelda cell + Sanctuary are decided by anchors in
    :data:`TIP_ANCHOR_ORDER`. Room ``0x55`` is multi-chamber (uncle / sword /
    south share the same ``room_base_id``); the fighter-sword route anchor has
    no position window, so when that anchor wins we apply a single y-threshold
    (``ROOM_55_SOUTH_Y_MIN``) to prefer ``room_55_south`` over ``room_55_sword``.
    South-chamber / stairs-align anchors already carry position windows and
    win earlier in the order when they match.

    Returns ``\"unknown\"`` when nothing matches.
    """
    by_id = {a.anchor_id: a for a in opening_anchors()}
    for aid in TIP_ANCHOR_ORDER:
        anchor = by_id.get(aid)
        if anchor is None or not anchor.matches(snapshot):
            continue
        node = anchor.graph_node_id
        if not node:
            continue
        # Chamber split for room 0x55 when only the no-position sword anchor matched.
        if node == "room_55_sword" and int(snapshot.link_y) >= ROOM_55_SOUTH_Y_MIN:
            return "room_55_south"
        return node
    # Fall back: first matched anchor that carries a graph node id.
    for a in match_anchors(snapshot):
        if a.graph_node_id:
            return a.graph_node_id
    return "unknown"


def anchor_by_id(anchor_id: str) -> MultiTruthAnchor | None:
    for a in opening_anchors():
        if a.anchor_id == anchor_id:
            return a
    return None


def match_anchors(snapshot: AlttpSnapshot) -> list[MultiTruthAnchor]:
    """Return all anchors currently satisfied by ``snapshot``."""
    return [a for a in opening_anchors() if a.matches(snapshot)]


def anchors_to_report(snapshot: AlttpSnapshot) -> list[dict[str, Any]]:
    """Evidence-friendly list of matched anchor ids + tiers."""
    return [
        {"anchorId": a.anchor_id, "tier": a.tier, "name": a.name}
        for a in match_anchors(snapshot)
    ]


# Semantic meaning of on-disk save states (filename → intended meaning).
# Filenames stay short for retro integration; meaning lives here.
STATE_SEMANTICS: Mapping[str, str] = {
    "YazeSlot000": "Title/boot slot (power-on predecessor)",
    "LinksHouseWake": "Link's House indoor wake (dev)",
    "HyruleCastleGrounds": (
        "HyruleCastle_GroundsSpawn_Controllable — outdoor screen 0x1B with "
        "control; NOT bridge-turn or secret-hole approach"
    ),
    "FirstAction": "Alias/checkpoint near first outdoor control (dev)",
    "FighterSword": (
        "HyruleCastle_SecretEntrance_FighterSword — room 0x55 post-uncle; "
        "state-load only, not natural-chain proof"
    ),
    "FighterSwordLamp": "FighterSword + lamp inventory (dev escort prep)",
    "Castle_55": "Generic room 0x55 checkpoint (ambiguous chamber — prefer semantic)",
    "CastleMain": (
        "HyruleCastle_MainHall — room 0x61 checkpoint (dev; outdoor door "
        "landing ~2040,1779 when exiting south)"
    ),
    "CourtyardSecretPocket": (
        "HyruleCastle_Courtyard_SecretStairsPocket — outdoor 0x1B after "
        "stairs exit (dev pocket reload; prefer segment chain)"
    ),
    "CastleMantleZelda": "Mantle/escort prep with Zelda (dev; confirm follower RAM)",
    "CastleZeldaFollower": "Zelda follower set (dev; confirm $F3CC==1)",
}
