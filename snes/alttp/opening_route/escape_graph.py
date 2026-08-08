"""RAM-first indoor route graph: castle escape → Zelda → Sanctuary.

Authority is stable-retro RAM (``room_base_id``, sword/lamp inventory,
dungeon keys, follower). Nodes use RAM-stable string ids; multi-screen
chambers of room ``0x55`` are split into uncle / post-sword / south nodes
that share ``meta.room_base_id``.

**Single hop table:** :class:`EscapeHop` is the only route definition.
:func:`escape_route_graph` and plan/spine leg helpers are generated from it
so edge ids and leg ids cannot drift.

z3-json-data escape region names may appear in node/edge ``meta`` as optional
logic labels only — not execution authority or screen coordinates.

Verified transitions (see ``docs/STATUS.md``): continuous through NW chamber
``0x50``; ``0x50`` east → ``0x01`` is natural_entry; Zelda/escort/Sanctuary
remain planned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from retro_harness.adventure.graph import (
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
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
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
# East of 0x50 north connector (natural_entry 2026-08-02).
N_ROOM_01 = "room_01"
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


# ---------------------------------------------------------------------------
# Single hop table — generates GraphEdge + RouteLeg (one id, no dual truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscapeHop:
    """One route hop: shared definition for graph edges and plan legs.

    ``hop_id`` is both ``GraphEdge.edge_id`` and ``RouteLeg.leg_id``.
    ``paths`` selects which Sanctuary plans include the hop.
    ``acquires`` is plan-only (edges do not acquire); ``verification`` /
    ``direction`` / ``provenance`` / ``meta`` feed the graph edge.
    """

    hop_id: str
    source_id: str
    target_id: str
    direction: str = ""
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    verification: str = VERIFICATION_PLANNED
    provenance: str = ""
    goal: str = ""
    paths: frozenset[str] = field(default_factory=lambda: _PRIMARY_ONLY)
    meta: Mapping[str, object] = field(default_factory=dict)

    def path_tag(self) -> str:
        """Edge meta path: key-only hops are internal_key; else primary."""
        if self.paths == _KEY_ONLY:
            return PATH_INTERNAL_KEY
        return PATH_PRIMARY

    def to_edge(self) -> GraphEdge:
        edge_meta: dict[str, object] = {"path": self.path_tag(), **dict(self.meta)}
        return GraphEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            edge_id=self.hop_id,
            direction=self.direction,
            requires=self.requires,
            verification=self.verification,
            provenance=self.provenance,
            meta=edge_meta,
        )

    def to_leg(self) -> RouteLeg:
        return RouteLeg(
            leg_id=self.hop_id,
            source_id=self.source_id,
            target_id=self.target_id,
            requires=self.requires,
            acquires=self.acquires,
            goal=self.goal,
        )


# Ordered along each plan. Outdoor continuous + key alternate diverge after
# room_55_south; both rejoin at room_61 for the shared Zelda→Sanctuary tail.
_ESCAPE_HOPS: tuple[EscapeHop, ...] = (
    # --- shared through south chamber ---------------------------------------
    EscapeHop(
        hop_id="grounds_to_hole",
        source_id=N_CASTLE_GROUNDS,
        target_id=N_ROOM_55_UNCLE,
        direction="down",
        verification=VERIFICATION_CONTINUOUS,
        provenance="castle_to_sword.SECRET_HOLE_ENTRY_SCRIPT",
        goal="enter_secret_passage_0x55",
        paths=_BOTH_PATHS,
        meta={
            "status_fact": "grounds→hole",
            "z3_label": "Hyrule Castle Secret Entrance Drop",
        },
    ),
    EscapeHop(
        hop_id="hole_to_sword",
        source_id=N_ROOM_55_UNCLE,
        target_id=N_ROOM_55_SWORD,
        direction="npc",
        acquires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="castle_to_sword.uncle_dialogue",
        goal="fighter_sword_equip_ram",
        paths=_BOTH_PATHS,
        meta={"status_fact": "hole→sword"},
    ),
    EscapeHop(
        hop_id="sword_to_south_chamber",
        source_id=N_ROOM_55_SWORD,
        target_id=N_ROOM_55_SOUTH,
        direction="south",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="secret_entrance_clear.SWORD_TO_SOUTH_CHAMBER_SCRIPT",
        goal="room_55_south_chamber",
        paths=_BOTH_PATHS,
        meta={
            "status_fact": "sword→south chamber",
            "script": "LEFT×100 + DOWN×250",
        },
    ),
    # --- primary outdoor continuous path ------------------------------------
    EscapeHop(
        hop_id="south_stairs_to_courtyard_pocket",
        source_id=N_ROOM_55_SOUTH,
        target_id=N_COURTYARD_SECRET_POCKET,
        direction="down",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="secret_entrance_clear.exit_secret_entrance_stairs",
        goal="secret_entrance_exited",
        paths=_PRIMARY_ONLY,
        meta={
            "status_fact": "secret-entrance clear (stairs → outdoors)",
            "tier": "trigger",
        },
    ),
    EscapeHop(
        hop_id="pocket_to_main_hall",
        source_id=N_COURTYARD_SECRET_POCKET,
        target_id=N_ROOM_61,
        direction="north",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="pocket_to_main_hall.run_from_pocket",
        goal="enter_main_castle_door",
        paths=_PRIMARY_ONLY,
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
    # --- alternate internal key path ----------------------------------------
    EscapeHop(
        hop_id="south_clear_small_key",
        source_id=N_ROOM_55_SOUTH,
        target_id=N_ROOM_55_KEYED,
        direction="combat",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        acquires=frozenset({CAP_SMALL_KEY}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.room_55_small_key",
        goal="room_55_small_key",
        paths=_KEY_ONLY,
        meta={
            "note": "alternate: clear soldiers; collect small key in 0x55",
        },
    ),
    EscapeHop(
        hop_id="keyed_exit_to_main_hall",
        source_id=N_ROOM_55_KEYED,
        target_id=N_ROOM_61,
        direction="door",
        requires=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.room_55_key_door",
        goal="reach_room_61",
        paths=_KEY_ONLY,
        meta={
            "note": "alternate key/shutter path out of 0x55",
            "to_room_base_id": HYRULE_CASTLE_MAIN_HALL_ROOM,
        },
    ),
    # --- shared post-main-hall (west + north continuous; Zelda planned) -----
    EscapeHop(
        hop_id="main_hall_west_to_0x60",
        source_id=N_ROOM_61,
        target_id=N_ROOM_60,
        direction="west",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="castle_dungeon.MAIN_HALL_TO_NW_PREFIX",
        goal="reach_room_60_west",
        paths=_BOTH_PATHS,
        meta={
            "to_room_base_id": HYRULE_CASTLE_MAIN_WEST_ROOM,
            "door_label": "west_to_0x60",
            "map_id": "room_61",
            "note": (
                "Clean power-on prefix: clear hostiles + recovery-aware "
                "side corridor + LEFT push"
            ),
        },
    ),
    EscapeHop(
        hop_id="room_60_north_to_0x50",
        source_id=N_ROOM_60,
        target_id=N_ROOM_50,
        direction="north",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_CONTINUOUS,
        provenance="castle_dungeon.MAIN_HALL_TO_NW_PREFIX",
        goal="reach_room_50_nw",
        paths=_BOTH_PATHS,
        meta={
            "to_room_base_id": HYRULE_CASTLE_NW_ROOM,
            "door_label": "north_to_0x50",
            "map_id": "room_60",
            "note": "Clean power-on prefix: north shaft → UP (maps/room_60.json)",
        },
    ),
    EscapeHop(
        hop_id="room_50_east_to_0x01",
        source_id=N_ROOM_50,
        target_id=N_ROOM_01,
        direction="east",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_NATURAL_ENTRY,
        provenance="room_engine.room_50.east_to_0x01+natural_room_50_east",
        goal="reach_room_01_connector",
        paths=_BOTH_PATHS,
        meta={
            "to_room_base_id": HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
            "door_label": "east_to_0x01",
            "map_id": "room_50",
            "note": (
                "Exhaustive 2026-08-02 probe: only physical forward exit from "
                "continuous tip 0x50. Natural entry from real 0x50 predecessor."
            ),
        },
    ),
    EscapeHop(
        hop_id="room_01_to_zelda_cell",
        source_id=N_ROOM_01,
        target_id=N_ROOM_80,
        direction="east",
        requires=frozenset({CAP_FIGHTER_SWORD}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.b1_to_zelda_cell",
        goal="reach_zelda_cell_0x80",
        paths=_BOTH_PATHS,
        meta={
            "to_room_base_id": ZELDA_CELL_ROOM,
            "note": (
                "Measured exploration chain: 0x01→0x52→0x62 (clear required in "
                "0x52). B1 stairs not yet isolated; maps/room_70.json seed exists."
            ),
        },
    ),
    EscapeHop(
        hop_id="rescue_escort_to_mantle",
        source_id=N_ROOM_80,
        target_id=N_CASTLE_MANTLE,
        direction="escort",
        # Follower is acquired on this leg (cell rescue); subsequent
        # mantle/sewer edges gate on zelda_follower.
        requires=frozenset({CAP_FIGHTER_SWORD}),
        acquires=frozenset({CAP_ZELDA_FOLLOWER}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.zelda_escort_mantle",
        goal="follower_indicator_zelda",
        paths=_BOTH_PATHS,
        meta={
            "note": "rescue Zelda then escort to rear mantle / throne",
            "z3_label": "Throne Room",
        },
    ),
    EscapeHop(
        hop_id="mantle_to_dark_sewers",
        source_id=N_CASTLE_MANTLE,
        target_id=N_SEWERS_DARK,
        direction="north",
        requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.mantle_sewer_push",
        goal="enter_dark_sewers",
        paths=_BOTH_PATHS,
        meta={
            "note": "mantle checks lamp + Zelda; opens dark sewers",
            "z3_label": "Throne Room → Sewers (Dark)",
        },
    ),
    EscapeHop(
        hop_id="sewers_to_sanctuary",
        source_id=N_SEWERS_DARK,
        target_id=N_SANCTUARY,
        direction="north",
        requires=frozenset({CAP_ZELDA_FOLLOWER, CAP_LAMP}),
        verification=VERIFICATION_PLANNED,
        provenance="planned.sewers_sanctuary_push",
        goal="reach_sanctuary_0x12",
        paths=_BOTH_PATHS,
        meta={
            "to_room_base_id": SANCTUARY_ROOM,
            "z3_label": "Sanctuary Push Door",
        },
    ),
)


def escape_route_hops() -> tuple[EscapeHop, ...]:
    """Return the single ordered hop table (graph + plans are derived from it)."""
    return _ESCAPE_HOPS


def _escape_nodes() -> tuple[GraphNode, ...]:
    return (
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
            node_id=N_ROOM_01,
            name="Hyrule Castle north connector (0x01)",
            area="hyrule_castle",
            tags=frozenset({"indoors", "escape", "natural_entry"}),
            meta=_room_meta(
                HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
                z3_label="Hyrule Castle",
                extra={
                    "note": (
                        "East of 0x50; only physical forward exit from tip "
                        "(2026-08-02)"
                    ),
                    "map_id": "room_01",
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


def escape_route_graph() -> RouteGraph:
    """Build the castle-escape capability graph from the hop table."""
    edges = tuple(hop.to_edge() for hop in _ESCAPE_HOPS)
    return RouteGraph(_escape_nodes(), edges)


def _planned_legs_for_path(path: str) -> tuple[RouteLeg, ...]:
    """Build a contiguous Sanctuary plan for ``path`` from the hop table."""
    return tuple(hop.to_leg() for hop in _ESCAPE_HOPS if path in hop.paths)


def continuous_spine_legs() -> tuple[RouteLeg, ...]:
    """Verified continuous tip: grounds → secret clear → NW chamber (0x50).

    Derived from primary-path hops whose verification is continuous.
    Further legs toward Zelda/Sanctuary remain planned (see
    :func:`escape_route_legs`).
    """
    return tuple(
        hop.to_leg()
        for hop in _ESCAPE_HOPS
        if PATH_PRIMARY in hop.paths and hop.verification == VERIFICATION_CONTINUOUS
    )


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
