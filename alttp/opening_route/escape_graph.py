"""RAM-first indoor route graph: castle escape → Zelda → Sanctuary.

Authority is stable-retro RAM (``room_base_id``, sword/lamp inventory,
dungeon keys, follower). Nodes use RAM-stable string ids; multi-screen
chambers of room ``0x55`` are split into uncle / post-sword / south nodes
that share ``meta.room_base_id``.

z3-json-data escape region names (e.g. ``Sewers (Dark)``, ``Sanctuary``)
may appear in node/edge ``meta`` as optional logic labels only — they are
**not** execution authority and must not be treated as screen coordinates.

Verified transitions (see ``docs/STATUS.md``):

- castle grounds → secret hole (secret entrance, RAM ``0x55``)
- hole → fighter sword (uncle dialogue)
- post-sword uncle corridor → south combat chamber
- south chamber stairs → outdoors screen ``0x1B`` (secret-entrance clear)

Courtyard pocket → main hall is measured (natural-entry / continuous spine
through ``room_50``). Main hall west → room ``0x60`` and north → room
``0x50`` are continuous via the clean power-on prefix. After ``0x50`` → Zelda
cell / escort / Sanctuary remain planned (map seeds under ``maps/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from adventure_common.graph import (
    GraphEdge,
    GraphNode,
    PlannedLeg,
    RouteGraph,
    RouteLeg,
    normalize_capability,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    HYRULE_CASTLE_SCREEN,
    SANCTUARY_ROOM,
    SECRET_PASSAGE_ROOM,
    ZELDA_CELL_ROOM,
    AlttpSnapshot,
)

# ---------------------------------------------------------------------------
# Capability tokens (snake_case; GraphEdge normalizes via normalize_capability)
# ---------------------------------------------------------------------------

CAP_FIGHTER_SWORD = "fighter_sword"
CAP_SMALL_KEY = "small_key"
CAP_LAMP = "lamp"
CAP_ZELDA_FOLLOWER = "zelda_follower"

# Natural boot collects the Link's House lamp before leaving for the castle.
NATURAL_HOUSE_EXIT_CAPABILITIES: frozenset[str] = frozenset({CAP_LAMP})

# ---------------------------------------------------------------------------
# Node ids (stable string keys)
# ---------------------------------------------------------------------------

N_CASTLE_GROUNDS = "castle_grounds"
N_ROOM_55_UNCLE = "room_55_uncle"
N_ROOM_55_SWORD = "room_55_sword"
N_ROOM_55_SOUTH = "room_55_south"
N_ROOM_55_KEYED = "room_55_keyed"
# Outdoor hedge pocket after secret-entrance stairs exit (screen 0x1B).
N_COURTYARD_SECRET_POCKET = "courtyard_secret_pocket"
N_ROOM_61 = "room_61"
# West of main hall (continuous clean prefix 2026-08-01).
N_ROOM_60 = "room_60"
# North of 0x60 (continuous clean prefix 2026-08-01).
N_ROOM_50 = "room_50"
N_ROOM_80 = "room_80"
N_CASTLE_MANTLE = "castle_mantle"
N_SEWERS_DARK = "sewers_dark"
N_SANCTUARY = "sanctuary"  # indoor room 0x12; abstract escort goal name

# Alias for callers that prefer the room_id pattern.
N_ROOM_12 = N_SANCTUARY

# Edge verification ladder (parallel to Super Metroid DoorEdge.verification).
VERIFICATION_PLANNED = "planned"
VERIFICATION_ISOLATED = "isolated"  # works from a save-state, not natural-entry
VERIFICATION_NATURAL_ENTRY = "natural_entry"  # from real predecessor segment
VERIFICATION_CONTINUOUS = "continuous"  # on continuous spine claim

# Route-path tags (edge meta ``path``; missing defaults to primary).
PATH_PRIMARY = "primary"
PATH_INTERNAL_KEY = "internal_key"
_BOTH_PATHS: frozenset[str] = frozenset({PATH_PRIMARY, PATH_INTERNAL_KEY})
_PRIMARY_ONLY: frozenset[str] = frozenset({PATH_PRIMARY})
_KEY_ONLY: frozenset[str] = frozenset({PATH_INTERNAL_KEY})


def _room_meta(
    room_base_id: int | None,
    *,
    z3_label: str = "",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    meta: dict[str, object] = {"authority": "stable_retro_ram"}
    if room_base_id is not None:
        meta["room_base_id"] = int(room_base_id)
        meta["room_hex"] = f"0x{int(room_base_id) & 0xFF:02X}"
    if z3_label:
        meta["z3_label"] = z3_label
    if extra:
        meta.update(extra)
    return meta


def escape_route_graph() -> RouteGraph:
    """Build the modest castle-escape capability graph (~10 nodes)."""
    nodes = (
        GraphNode(
            node_id=N_CASTLE_GROUNDS,
            name="Hyrule Castle grounds",
            area="light_world",
            tags=frozenset({"overworld", "opening", "escape"}),
            meta=_room_meta(
                None,
                extra={
                    "screen_id": HYRULE_CASTLE_SCREEN,
                    "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
                    "z3_label": "Hyrule Castle Courtyard",
                },
            ),
        ),
        GraphNode(
            node_id=N_ROOM_55_UNCLE,
            name="Secret passage — uncle chamber",
            area="hyrule_castle",
            tags=frozenset({"indoors", "room_55", "escape"}),
            meta=_room_meta(
                SECRET_PASSAGE_ROOM,
                z3_label="Hyrule Castle Secret Entrance",
                extra={"chamber": "uncle"},
            ),
        ),
        GraphNode(
            node_id=N_ROOM_55_SWORD,
            name="Secret passage — post fighter sword",
            area="hyrule_castle",
            tags=frozenset({"indoors", "room_55", "escape", "sword"}),
            meta=_room_meta(
                SECRET_PASSAGE_ROOM,
                z3_label="Hyrule Castle Secret Entrance",
                extra={"chamber": "post_sword"},
            ),
        ),
        GraphNode(
            node_id=N_ROOM_55_SOUTH,
            name="Secret passage — south combat chamber",
            area="hyrule_castle",
            tags=frozenset({"indoors", "room_55", "escape", "combat"}),
            meta=_room_meta(
                SECRET_PASSAGE_ROOM,
                z3_label="Hyrule Castle Secret Entrance",
                extra={"chamber": "south"},
            ),
        ),
        GraphNode(
            node_id=N_ROOM_55_KEYED,
            name="Secret passage — small key ready",
            area="hyrule_castle",
            tags=frozenset({"indoors", "room_55", "escape", "key"}),
            meta=_room_meta(
                SECRET_PASSAGE_ROOM,
                z3_label="Hyrule Castle Secret Entrance",
                extra={"chamber": "keyed"},
            ),
        ),
        GraphNode(
            node_id=N_COURTYARD_SECRET_POCKET,
            name="Courtyard — secret-stairs outdoor pocket",
            area="light_world",
            tags=frozenset({"overworld", "opening", "escape", "pocket"}),
            meta=_room_meta(
                None,
                extra={
                    "screen_id": HYRULE_CASTLE_SCREEN,
                    "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
                    "world_xy_approx": (2248, 1755),
                    "z3_label": "Hyrule Castle Courtyard",
                    "note": (
                        "Tight hedge pocket after stairs exit; UP re-enters "
                        "secret entrance. Escape: bush-cut S/W → open court."
                    ),
                },
            ),
        ),
        GraphNode(
            node_id=N_ROOM_61,
            name="Hyrule Castle main hall",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape"}),
            meta=_room_meta(
                HYRULE_CASTLE_MAIN_HALL_ROOM,
                z3_label="Hyrule Castle",
                extra={"door_approach_xy": (2040, 1790)},
            ),
        ),
        GraphNode(
            node_id=N_ROOM_60,
            name="Hyrule Castle main west (0x60)",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "continuous"}),
            meta=_room_meta(
                HYRULE_CASTLE_MAIN_WEST_ROOM,
                z3_label="Hyrule Castle",
                extra={
                    "note": "West of main hall; geometry maps/room_60.json",
                    "map_id": "room_60",
                },
            ),
        ),
        GraphNode(
            node_id=N_ROOM_50,
            name="Hyrule Castle NW chamber (0x50)",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "continuous"}),
            meta=_room_meta(
                HYRULE_CASTLE_NW_ROOM,
                z3_label="Hyrule Castle",
                extra={
                    "note": "North of 0x60 on Zelda path; geometry maps/room_50.json",
                    "map_id": "room_50",
                },
            ),
        ),
        GraphNode(
            node_id=N_ROOM_80,
            name="Zelda's cell",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "zelda", "planned"}),
            meta=_room_meta(
                ZELDA_CELL_ROOM,
                z3_label="Hyrule Castle - Zelda's Chest",
                extra={"map_id": "room_80"},
            ),
        ),
        GraphNode(
            node_id=N_CASTLE_MANTLE,
            name="Castle mantle / throne escort",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "escort", "planned"}),
            meta=_room_meta(
                None,
                z3_label="Throne Room",
                extra={"note": "rear mantle path; lamp + Zelda required"},
            ),
        ),
        GraphNode(
            node_id=N_SEWERS_DARK,
            name="Sewers (dark)",
            area="sewers",
            tags=frozenset({"indoors", "escape", "dark", "planned"}),
            meta=_room_meta(
                None,
                z3_label="Sewers (Dark)",
                extra={"note": "abstract dark-sewer hop; lamp required"},
            ),
        ),
        GraphNode(
            node_id=N_SANCTUARY,
            name="Sanctuary",
            area="sanctuary",
            tags=frozenset({"indoors", "escape", "sanctuary", "planned"}),
            meta=_room_meta(
                SANCTUARY_ROOM,
                z3_label="Sanctuary",
                extra={"node_alias": "room_12"},
            ),
        ),
    )

    edges = (
        # --- continuous primary outdoor path (STATUS-proven) ----------------
        GraphEdge(
            source_id=N_CASTLE_GROUNDS,
            target_id=N_ROOM_55_UNCLE,
            edge_id="grounds_to_hole",
            direction="down",
            verification=VERIFICATION_CONTINUOUS,
            provenance="castle_to_sword.SECRET_HOLE_ENTRY_SCRIPT",
            meta={
                "path": PATH_PRIMARY,
                "status_fact": "grounds→hole",
                "z3_label": "Hyrule Castle Secret Entrance Drop",
            },
        ),
        GraphEdge(
            source_id=N_ROOM_55_UNCLE,
            target_id=N_ROOM_55_SWORD,
            edge_id="hole_to_sword",
            direction="npc",
            verification=VERIFICATION_CONTINUOUS,
            provenance="castle_to_sword.uncle_dialogue",
            meta={"path": PATH_PRIMARY, "status_fact": "hole→sword"},
        ),
        GraphEdge(
            source_id=N_ROOM_55_SWORD,
            target_id=N_ROOM_55_SOUTH,
            edge_id="sword_to_south_chamber",
            direction="south",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="secret_entrance_clear.SWORD_TO_SOUTH_CHAMBER_SCRIPT",
            meta={
                "path": PATH_PRIMARY,
                "status_fact": "sword→south chamber",
                "script": "LEFT×100 + DOWN×250",
            },
        ),
        # Free multi-screen hop when already sword-capable is the same edge;
        # reverse is not modeled (stair pocket soft-trap risk).
        GraphEdge(
            source_id=N_ROOM_55_SOUTH,
            target_id=N_COURTYARD_SECRET_POCKET,
            edge_id="south_stairs_to_courtyard_pocket",
            direction="down",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="secret_entrance_clear.exit_secret_entrance_stairs",
            meta={
                "path": PATH_PRIMARY,
                "status_fact": "secret-entrance clear (stairs → outdoors)",
                "tier": "trigger",
            },
        ),
        # --- continuous: courtyard pocket → main door (measured 2026-07-30) ---
        GraphEdge(
            source_id=N_COURTYARD_SECRET_POCKET,
            target_id=N_ROOM_61,
            edge_id="pocket_to_main_hall",
            direction="north",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="pocket_to_main_hall.run_from_pocket",
            meta={
                "path": PATH_PRIMARY,
                "status_fact": "pocket→main hall 0x61",
                "to_room_base_id": HYRULE_CASTLE_MAIN_HALL_ROOM,
                "door_approach_xy": (2040, 1790),
                "south_corridor_y": 2024,
                "tier": "route+approach+trigger",
                "note": (
                    "Bush-cut S/W out of hedges; south corridor y≈2024; "
                    "west to x≈2040; north; UP into door."
                ),
            },
        ),
        # --- planned alternate: leave 0x55 via key/shutter (work queue) ---
        GraphEdge(
            source_id=N_ROOM_55_SOUTH,
            target_id=N_ROOM_55_KEYED,
            edge_id="south_clear_small_key",
            direction="combat",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.sword_to_zelda_key",
            meta={
                "note": "alternate: clear soldiers; collect small key in 0x55",
                "path": PATH_INTERNAL_KEY,
            },
        ),
        GraphEdge(
            source_id=N_ROOM_55_KEYED,
            target_id=N_ROOM_61,
            edge_id="keyed_exit_to_main_hall",
            direction="door",
            requires=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.room_55_key_door",
            meta={
                "note": "alternate key/shutter path out of 0x55",
                "to_room_base_id": HYRULE_CASTLE_MAIN_HALL_ROOM,
                "path": PATH_INTERNAL_KEY,
            },
        ),
        # --- continuous: main hall west door → room 0x60 ----------------------
        # Geometry (approach/landing) lives only in maps/room_61.json.
        GraphEdge(
            source_id=N_ROOM_61,
            target_id=N_ROOM_60,
            edge_id="main_hall_west_to_0x60",
            direction="west",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="castle_dungeon.MAIN_HALL_TO_NW_PREFIX",
            meta={
                "path": PATH_PRIMARY,
                "to_room_base_id": HYRULE_CASTLE_MAIN_WEST_ROOM,
                "door_label": "west_to_0x60",
                "map_id": "room_61",
                "note": "Clean power-on prefix: clear hostiles + recovery-aware side corridor + LEFT push",
            },
        ),
        # --- continuous: room 0x60 north → room 0x50 ---------------------------
        GraphEdge(
            source_id=N_ROOM_60,
            target_id=N_ROOM_50,
            edge_id="room_60_north_to_0x50",
            direction="north",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="castle_dungeon.MAIN_HALL_TO_NW_PREFIX",
            meta={
                "path": PATH_PRIMARY,
                "to_room_base_id": HYRULE_CASTLE_NW_ROOM,
                "door_label": "north_to_0x50",
                "map_id": "room_60",
                "note": "Clean power-on prefix: north shaft → UP (maps/room_60.json)",
            },
        ),
        # --- planned primary: after 0x50 → Zelda cell / escort / Sanctuary ----
        GraphEdge(
            source_id=N_ROOM_50,
            target_id=N_ROOM_80,
            edge_id="room_50_to_zelda_cell",
            direction="east",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.b1_to_zelda_cell",
            meta={
                "path": PATH_PRIMARY,
                "to_room_base_id": ZELDA_CELL_ROOM,
                "note": (
                    "Measured next hop candidates: 0x50→0x01→…; B1 0x81/0x82/0x72 "
                    "maps exist. Intermediate rooms not yet continuous."
                ),
            },
        ),
        GraphEdge(
            source_id=N_ROOM_80,
            target_id=N_CASTLE_MANTLE,
            edge_id="rescue_escort_to_mantle",
            direction="escort",
            # Follower is acquired on this leg (cell rescue); subsequent
            # mantle/sewer edges gate on zelda_follower.
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.zelda_escort_mantle",
            meta={
                "path": PATH_PRIMARY,
                "note": "rescue Zelda then escort to rear mantle / throne",
                "z3_label": "Throne Room",
            },
        ),
        GraphEdge(
            source_id=N_CASTLE_MANTLE,
            target_id=N_SEWERS_DARK,
            edge_id="mantle_to_dark_sewers",
            direction="north",
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.mantle_sewer_push",
            meta={
                "path": PATH_PRIMARY,
                "note": "mantle checks lamp + Zelda; opens dark sewers",
                "z3_label": "Throne Room → Sewers (Dark)",
            },
        ),
        GraphEdge(
            source_id=N_SEWERS_DARK,
            target_id=N_SANCTUARY,
            edge_id="sewers_to_sanctuary",
            direction="north",
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.sewers_sanctuary_push",
            meta={
                "path": PATH_PRIMARY,
                "to_room_base_id": SANCTUARY_ROOM,
                "z3_label": "Sanctuary Push Door",
            },
        ),
    )
    return RouteGraph(nodes, edges)


# ---------------------------------------------------------------------------
# Single hop table — primary + key-path Sanctuary plans share one definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EscapeHop:
    """One planned hop; ``paths`` selects which Sanctuary plans include it."""

    leg_id: str
    source_id: str
    target_id: str
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    goal: str = ""
    paths: frozenset[str] = field(default_factory=lambda: _PRIMARY_ONLY)


# Ordered along each plan. Outdoor continuous + key alternate diverge after
# room_55_south; both rejoin at room_61 for the shared Zelda→Sanctuary tail.
_ESCAPE_HOPS: tuple[_EscapeHop, ...] = (
    # --- shared through south chamber ---------------------------------------
    _EscapeHop(
        leg_id="grounds_to_hole",
        source_id=N_CASTLE_GROUNDS,
        target_id=N_ROOM_55_UNCLE,
        goal="enter_secret_passage_0x55",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="uncle_fighter_sword",
        source_id=N_ROOM_55_UNCLE,
        target_id=N_ROOM_55_SWORD,
        acquires=frozenset({CAP_FIGHTER_SWORD}),
        goal="fighter_sword_equip_ram",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="sword_to_south_chamber",
        source_id=N_ROOM_55_SWORD,
        target_id=N_ROOM_55_SOUTH,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="room_55_south_chamber",
        paths=_BOTH_PATHS,
    ),
    # --- primary outdoor continuous path ------------------------------------
    _EscapeHop(
        leg_id="south_stairs_to_courtyard_pocket",
        source_id=N_ROOM_55_SOUTH,
        target_id=N_COURTYARD_SECRET_POCKET,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="secret_entrance_exited",
        paths=_PRIMARY_ONLY,
    ),
    _EscapeHop(
        leg_id="pocket_to_main_hall",
        source_id=N_COURTYARD_SECRET_POCKET,
        target_id=N_ROOM_61,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="enter_main_castle_door",
        paths=_PRIMARY_ONLY,
    ),
    # --- alternate internal key path ----------------------------------------
    _EscapeHop(
        leg_id="south_clear_small_key",
        source_id=N_ROOM_55_SOUTH,
        target_id=N_ROOM_55_KEYED,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        acquires=frozenset({CAP_SMALL_KEY}),
        goal="room_55_small_key",
        paths=_KEY_ONLY,
    ),
    _EscapeHop(
        leg_id="exit_to_main_hall",
        source_id=N_ROOM_55_KEYED,
        target_id=N_ROOM_61,
        requires=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
        goal="reach_room_61",
        paths=_KEY_ONLY,
    ),
    # --- shared post-main-hall (west + north continuous; Zelda planned) -----
    _EscapeHop(
        leg_id="main_hall_west_to_0x60",
        source_id=N_ROOM_61,
        target_id=N_ROOM_60,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="reach_room_60_west",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="room_60_north_to_0x50",
        source_id=N_ROOM_60,
        target_id=N_ROOM_50,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="reach_room_50_nw",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="room_50_to_zelda_cell",
        source_id=N_ROOM_50,
        target_id=N_ROOM_80,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        goal="reach_zelda_cell_0x80",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="rescue_zelda",
        source_id=N_ROOM_80,
        target_id=N_CASTLE_MANTLE,
        requires=frozenset({CAP_FIGHTER_SWORD}),
        acquires=frozenset({CAP_ZELDA_FOLLOWER}),
        goal="follower_indicator_zelda",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="mantle_to_dark_sewers",
        source_id=N_CASTLE_MANTLE,
        target_id=N_SEWERS_DARK,
        requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
        goal="enter_dark_sewers",
        paths=_BOTH_PATHS,
    ),
    _EscapeHop(
        leg_id="sewers_to_sanctuary",
        source_id=N_SEWERS_DARK,
        target_id=N_SANCTUARY,
        requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
        goal="reach_sanctuary_0x12",
        paths=_BOTH_PATHS,
    ),
)


def _hop_to_leg(hop: _EscapeHop) -> RouteLeg:
    return RouteLeg(
        leg_id=hop.leg_id,
        source_id=hop.source_id,
        target_id=hop.target_id,
        requires=hop.requires,
        acquires=hop.acquires,
        goal=hop.goal,
    )


def _edge_path(edge: GraphEdge) -> str:
    """Return edge meta path; missing defaults to primary."""
    raw = edge.meta.get("path", PATH_PRIMARY)
    return str(raw) if raw is not None else PATH_PRIMARY


def _planned_legs_for_path(path: str) -> tuple[RouteLeg, ...]:
    """Build a contiguous Sanctuary plan for ``path`` from the shared hop table."""
    return tuple(_hop_to_leg(hop) for hop in _ESCAPE_HOPS if path in hop.paths)


def continuous_spine_legs() -> tuple[RouteLeg, ...]:
    """Verified continuous tip: grounds → secret clear → NW chamber (0x50).

    Derived from primary-path hops whose matching graph edge is continuous.
    Further legs toward Zelda/Sanctuary remain planned (see
    :func:`escape_route_legs`).
    """
    graph = escape_route_graph()
    legs: list[RouteLeg] = []
    for hop in _ESCAPE_HOPS:
        if PATH_PRIMARY not in hop.paths:
            continue
        edge = graph.edge_for(hop.source_id, hop.target_id)
        if edge is None:
            continue
        if edge.verification != VERIFICATION_CONTINUOUS:
            continue
        if _edge_path(edge) != PATH_PRIMARY:
            continue
        legs.append(_hop_to_leg(hop))
    return tuple(legs)


def escape_route_legs() -> tuple[RouteLeg, ...]:
    """Contiguous legs: castle grounds → Sanctuary with capability acquires.

    Primary plan uses the outdoor continuous path through the courtyard
    pocket, then planned B1 / Zelda / escort legs. The internal key/shutter
    path remains on the graph as an alternate (``path: internal_key``;
    demoted in the work queue) but is not the default Sanctuary plan.

    Initial inventory after a natural house exit is typically just ``lamp``.
    ``fighter_sword`` is acquired at the uncle leg; ``zelda_follower`` when
    the cell rescue completes.
    """
    return _planned_legs_for_path(PATH_PRIMARY)


def escape_route_legs_key_path() -> tuple[RouteLeg, ...]:
    """Alternate Sanctuary plan via internal 0x55 key/shutter (work queue).

    Post-``room_50`` legs are the same shared hop-table tail as the primary
    plan (not a hand-copied Sanctuary sequence).
    """
    return _planned_legs_for_path(PATH_INTERNAL_KEY)


def escape_route_legs_from_room_55() -> tuple[RouteLeg, ...]:
    """Legs starting at post-drop uncle (skip grounds→hole)."""
    legs = escape_route_legs()
    return tuple(leg for leg in legs if leg.source_id != N_CASTLE_GROUNDS)


def plan_escape_to_sanctuary(
    capabilities: frozenset[str] | Iterable[str] | None = None,
    *,
    legs: Iterable[RouteLeg] | None = None,
    graph: RouteGraph | None = None,
) -> tuple[PlannedLeg, ...]:
    """Plan escape legs with capability bookkeeping.

    Default initial capabilities match a natural house exit (``lamp`` only).
    Pass an explicit set (e.g. ``{fighter_sword, lamp}``) when resuming from a
    fighter-sword checkpoint. Raises ``ValueError`` when required caps are
    missing and not acquired earlier on the leg chain.
    """
    g = graph or escape_route_graph()
    route_legs = tuple(legs) if legs is not None else escape_route_legs()
    if capabilities is None:
        initial: frozenset[str] = NATURAL_HOUSE_EXIT_CAPABILITIES
    else:
        initial = frozenset(normalize_capability(v) for v in capabilities)
    return g.plan_legs(route_legs, initial_capabilities=initial)


def capabilities_from_snapshot(snapshot: AlttpSnapshot) -> frozenset[str]:
    """Map an :class:`AlttpSnapshot` to route-planning capability tokens."""
    caps: set[str] = set()
    if snapshot.has_fighter_sword:
        caps.add(CAP_FIGHTER_SWORD)
    if snapshot.has_lamp:
        caps.add(CAP_LAMP)
    if snapshot.has_zelda_follower:
        caps.add(CAP_ZELDA_FOLLOWER)
    keys = snapshot.dungeon_key_count
    if keys is not None and keys >= 1:
        caps.add(CAP_SMALL_KEY)
    return frozenset(caps)
