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

Courtyard pocket → main hall is measured (natural-entry / continuous tip
through ``room_61``). Zelda cell / escort / Sanctuary remain planned.
"""

from __future__ import annotations

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
            node_id=N_ROOM_80,
            name="Zelda's cell",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "zelda", "planned"}),
            meta=_room_meta(
                ZELDA_CELL_ROOM,
                z3_label="Hyrule Castle - Zelda's Chest",
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
        # --- continuous (STATUS-proven) ------------------------------------
        GraphEdge(
            source_id=N_CASTLE_GROUNDS,
            target_id=N_ROOM_55_UNCLE,
            edge_id="grounds_to_hole",
            direction="down",
            verification=VERIFICATION_CONTINUOUS,
            provenance="castle_to_sword.SECRET_HOLE_ENTRY_SCRIPT",
            meta={
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
            meta={"status_fact": "hole→sword"},
        ),
        GraphEdge(
            source_id=N_ROOM_55_SWORD,
            target_id=N_ROOM_55_SOUTH,
            edge_id="sword_to_south_chamber",
            direction="south",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_CONTINUOUS,
            provenance="sword_to_zelda.SWORD_TO_SOUTH_CHAMBER_SCRIPT",
            meta={
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
            provenance="sword_to_zelda.exit_secret_entrance_stairs",
            meta={
                "status_fact": "secret-entrance clear (stairs → outdoors)",
                "stairs_align_xy": (2672, 2916),
                "landing_xy_approx": (2248, 1755),
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
                "path": "internal_key",
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
                "path": "internal_key",
            },
        ),
        GraphEdge(
            source_id=N_ROOM_61,
            target_id=N_ROOM_80,
            edge_id="main_hall_to_zelda_cell",
            direction="east",
            requires=frozenset({CAP_FIGHTER_SWORD}),
            verification=VERIFICATION_PLANNED,
            provenance="planned.castle_to_zelda_cell",
            meta={
                "to_room_base_id": ZELDA_CELL_ROOM,
                "note": "coarse hop; intermediate B1 rooms not expanded",
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
                "to_room_base_id": SANCTUARY_ROOM,
                "z3_label": "Sanctuary Push Door",
            },
        ),
    )
    return RouteGraph(nodes, edges)


def continuous_spine_legs() -> tuple[RouteLeg, ...]:
    """Verified continuous tip: grounds → secret clear → main hall (0x61).

    This is the truthful continuous claim as of STATUS. Further legs toward
    Zelda/Sanctuary remain planned (see :func:`escape_route_legs`).
    """
    return (
        RouteLeg(
            leg_id="grounds_to_hole",
            source_id=N_CASTLE_GROUNDS,
            target_id=N_ROOM_55_UNCLE,
            goal="enter_secret_passage_0x55",
        ),
        RouteLeg(
            leg_id="uncle_fighter_sword",
            source_id=N_ROOM_55_UNCLE,
            target_id=N_ROOM_55_SWORD,
            acquires=frozenset({CAP_FIGHTER_SWORD}),
            goal="fighter_sword_equip_ram",
        ),
        RouteLeg(
            leg_id="sword_to_south_chamber",
            source_id=N_ROOM_55_SWORD,
            target_id=N_ROOM_55_SOUTH,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="room_55_south_chamber",
        ),
        RouteLeg(
            leg_id="south_stairs_to_courtyard_pocket",
            source_id=N_ROOM_55_SOUTH,
            target_id=N_COURTYARD_SECRET_POCKET,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="secret_entrance_exited",
        ),
        RouteLeg(
            leg_id="pocket_to_main_hall",
            source_id=N_COURTYARD_SECRET_POCKET,
            target_id=N_ROOM_61,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="enter_main_castle_door",
        ),
    )


def escape_route_legs() -> tuple[RouteLeg, ...]:
    """Contiguous legs: castle grounds → Sanctuary with capability acquires.

    Primary plan uses the outdoor continuous path through the courtyard
    pocket, then planned main-door / B1 / Zelda / escort legs. The internal
    key/shutter path remains on the graph as an alternate (work-queue focus)
    but is not the default Sanctuary plan.

    Initial inventory after a natural house exit is typically just ``lamp``.
    ``fighter_sword`` is acquired at the uncle leg; ``zelda_follower`` when
    the cell rescue completes.
    """
    return continuous_spine_legs() + (
        RouteLeg(
            leg_id="main_hall_to_zelda_cell",
            source_id=N_ROOM_61,
            target_id=N_ROOM_80,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="reach_zelda_cell_0x80",
        ),
        RouteLeg(
            leg_id="rescue_zelda",
            source_id=N_ROOM_80,
            target_id=N_CASTLE_MANTLE,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            acquires=frozenset({CAP_ZELDA_FOLLOWER}),
            goal="follower_indicator_zelda",
        ),
        RouteLeg(
            leg_id="mantle_to_dark_sewers",
            source_id=N_CASTLE_MANTLE,
            target_id=N_SEWERS_DARK,
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            goal="enter_dark_sewers",
        ),
        RouteLeg(
            leg_id="sewers_to_sanctuary",
            source_id=N_SEWERS_DARK,
            target_id=N_SANCTUARY,
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            goal="reach_sanctuary_0x12",
        ),
    )


def escape_route_legs_key_path() -> tuple[RouteLeg, ...]:
    """Alternate Sanctuary plan via internal 0x55 key/shutter (work queue)."""
    return (
        RouteLeg(
            leg_id="grounds_to_hole",
            source_id=N_CASTLE_GROUNDS,
            target_id=N_ROOM_55_UNCLE,
            goal="enter_secret_passage_0x55",
        ),
        RouteLeg(
            leg_id="uncle_fighter_sword",
            source_id=N_ROOM_55_UNCLE,
            target_id=N_ROOM_55_SWORD,
            acquires=frozenset({CAP_FIGHTER_SWORD}),
            goal="fighter_sword_equip_ram",
        ),
        RouteLeg(
            leg_id="sword_to_south_chamber",
            source_id=N_ROOM_55_SWORD,
            target_id=N_ROOM_55_SOUTH,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="room_55_south_chamber",
        ),
        RouteLeg(
            leg_id="south_clear_small_key",
            source_id=N_ROOM_55_SOUTH,
            target_id=N_ROOM_55_KEYED,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            acquires=frozenset({CAP_SMALL_KEY}),
            goal="room_55_small_key",
        ),
        RouteLeg(
            leg_id="exit_to_main_hall",
            source_id=N_ROOM_55_KEYED,
            target_id=N_ROOM_61,
            requires=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
            goal="reach_room_61",
        ),
        RouteLeg(
            leg_id="main_hall_to_zelda_cell",
            source_id=N_ROOM_61,
            target_id=N_ROOM_80,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            goal="reach_zelda_cell_0x80",
        ),
        RouteLeg(
            leg_id="rescue_zelda",
            source_id=N_ROOM_80,
            target_id=N_CASTLE_MANTLE,
            requires=frozenset({CAP_FIGHTER_SWORD}),
            acquires=frozenset({CAP_ZELDA_FOLLOWER}),
            goal="follower_indicator_zelda",
        ),
        RouteLeg(
            leg_id="mantle_to_dark_sewers",
            source_id=N_CASTLE_MANTLE,
            target_id=N_SEWERS_DARK,
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            goal="enter_dark_sewers",
        ),
        RouteLeg(
            leg_id="sewers_to_sanctuary",
            source_id=N_SEWERS_DARK,
            target_id=N_SANCTUARY,
            requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
            goal="reach_sanctuary_0x12",
        ),
    )


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
