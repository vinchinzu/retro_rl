"""Opening-route checkpoint data: Link's House → Hyrule Castle grounds.

Maps the confirmed boot goal (title → fresh file → Link's House exit →
controllable on light-world screen ``0x1B``) onto local ``z3-json-data``
regions/connections for developer validation and progress artifacts.

**Authority split (do not collapse these):**

- Gameplay routing uses stable-retro RAM fields from ``alttp.ram`` /
  ``alttp.overworld`` (screen id, room id, indoors, control).
- z3 room/node names are randomizer-oriented **logic labels**. They are
  *associated* with route segments for naming and graph checks; they are
  **not** exact stable-retro screen coordinates or RAM screen IDs.

Never auto-downloads. Uses ``alttp.z3_json_data`` only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.adventure.graph import GraphEdge, GraphNode, RouteGraph, RouteLeg
from alttp.paths import (
    RECORDINGS_DIR,
    Z3_JSON_DATA_PIN,
)
from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_ROOM,
    LINKS_HOUSE_SCREEN,
)
from alttp.z3_json_data import OPENING_ROUTE_ROOM_NAMES

CATALOG_KIND = "alttp_opening_route_catalog"
CATALOG_VERSION = 1
DEFAULT_ARTIFACT = RECORDINGS_DIR / "opening_route_catalog.json"

DISCLAIMER = (
    "z3 room/node names are logic labels from vg-json-data/z3-json-data; "
    "they are associated with route segments and are NOT exact stable-retro "
    "screen coordinates or RAM screen IDs. Gameplay proof requires RAM "
    "snapshots from a real boot/route run (alttp.ram / boot_to_castle)."
)

# Overworld screen path used by the scripted BFS (authoritative gameplay IDs).
# Labels are local; not z3 names.
OVERWORLD_SCREEN_PATH: tuple[dict[str, Any], ...] = (
    {
        "screen_id": LINKS_HOUSE_SCREEN,
        "screen_hex": f"0x{LINKS_HOUSE_SCREEN:02X}",
        "label": "links_house",
    },
    {
        "screen_id": 0x24,
        "screen_hex": "0x24",
        "label": "north_field",
    },
    {
        "screen_id": 0x1C,
        "screen_hex": "0x1C",
        "label": "castle_approach",
    },
    {
        "screen_id": HYRULE_CASTLE_SCREEN,
        "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
        "label": "hyrule_castle",
    },
)


def _ow_node_id(screen_id: int) -> str:
    return f"ow_{int(screen_id) & 0xFF:02x}"


def opening_overworld_route_graph() -> RouteGraph:
    """Catalog-only overworld path as a :class:`RouteGraph` (RAM screen ids).

    Pure data export for tooling / future escape work. **Not** the boot
    executor — ``boot_to_castle`` / overworld BFS own live movement.

    z3 room/node names are *not* encoded here — they live on checkpoints as
    logic associations only.
    """
    nodes = [
        GraphNode(
            node_id=_ow_node_id(int(step["screen_id"])),
            name=str(step["label"]),
            area="light_world",
            tags=frozenset({"overworld", "opening_route"}),
            meta={
                "screen_id": int(step["screen_id"]),
                "screen_hex": step["screen_hex"],
                "authority": "stable_retro_ram",
            },
        )
        for step in OVERWORLD_SCREEN_PATH
    ]
    edges: list[GraphEdge] = []
    for prev, nxt in zip(OVERWORLD_SCREEN_PATH, OVERWORLD_SCREEN_PATH[1:]):
        src = int(prev["screen_id"])
        dst = int(nxt["screen_id"])
        edges.append(
            GraphEdge(
                source_id=_ow_node_id(src),
                target_id=_ow_node_id(dst),
                direction="north",
                verification="continuous",
                provenance="alttp_opening_bfs",
                meta={
                    "from_screen": src,
                    "to_screen": dst,
                    "from_label": prev["label"],
                    "to_label": nxt["label"],
                },
            )
        )
    return RouteGraph(nodes, edges)


def opening_overworld_route_legs() -> tuple[RouteLeg, ...]:
    """Catalog-only directed legs for Link's House → castle screen path.

    Not wired into boot execution; pairs with
    :func:`opening_overworld_route_graph` as data for tooling.
    """
    legs: list[RouteLeg] = []
    for prev, nxt in zip(OVERWORLD_SCREEN_PATH, OVERWORLD_SCREEN_PATH[1:]):
        src = int(prev["screen_id"])
        dst = int(nxt["screen_id"])
        legs.append(
            RouteLeg(
                leg_id=f"ow_{src:02x}_to_{dst:02x}",
                source_id=_ow_node_id(src),
                target_id=_ow_node_id(dst),
                goal=f"reach_screen_{dst:02X}",
            )
        )
    return tuple(legs)


@dataclass(frozen=True)
class ExpectedConnection:
    """A directed z3 connection we care about for the opening route."""

    origin: str
    destination: str
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class ExpectedNode:
    """A named door/item node expected inside a z3 room."""

    room_name: str
    node_name: str
    required: bool = True


@dataclass(frozen=True)
class OpeningCheckpoint:
    """One actionable checkpoint on Link's House → castle grounds.

    ``gameplay_*`` fields describe the stable-retro RAM acceptance for that
    segment when observed. ``z3_*`` fields are logic associations only.
    """

    id: str
    label: str
    role: str  # start | transit | goal | post_goal_context
    gameplay: dict[str, Any]
    z3_rooms: tuple[str, ...] = ()
    z3_nodes: tuple[ExpectedNode, ...] = ()
    z3_connections: tuple[ExpectedConnection, ...] = ()
    notes: str = ""


def opening_checkpoints() -> tuple[OpeningCheckpoint, ...]:
    """Return the curated Link's House → castle grounds checkpoint list."""
    return (
        OpeningCheckpoint(
            id="links_house_interior",
            label="Link's House (interior)",
            role="start",
            gameplay={
                "indoors": True,
                "room_base_id": LINKS_HOUSE_ROOM,
                "room_hex": f"0x{LINKS_HOUSE_ROOM:04X}",
                "dark_world": False,
            },
            z3_rooms=("Links House",),
            z3_nodes=(
                ExpectedNode("Links House", "Links House Exit"),
                ExpectedNode("Links House", "Link's House"),  # lamp item node
            ),
            # Pin has the cave room/nodes but no Door edge for the exit in
            # connections/main.json — keep optional so validate stays honest.
            z3_connections=(
                ExpectedConnection(
                    origin="Links House Exit",
                    destination="Light World",
                    required=False,
                    note=(
                        "Expected door edge for house exit; absent from "
                        "connections/main.json at the pinned revision "
                        "(nodes still exist on Links House / Light World)."
                    ),
                ),
            ),
            notes=(
                "Fresh-file spawn. z3 room 'Links House' is a cave/logic "
                "region, not screen 0x2C."
            ),
        ),
        OpeningCheckpoint(
            id="links_house_overworld",
            label="Link's House overworld porch",
            role="transit",
            gameplay={
                "indoors": False,
                "screen_id": LINKS_HOUSE_SCREEN,
                "screen_hex": f"0x{LINKS_HOUSE_SCREEN:02X}",
                "dark_world": False,
            },
            z3_rooms=("Light World", "Links House"),
            z3_nodes=(
                ExpectedNode("Light World", "Links House"),
            ),
            notes=(
                "stable-retro overworld screen 0x2C after house exit. "
                "Light World door node 'Links House' is a logic label, "
                "not a pixel coordinate."
            ),
        ),
        OpeningCheckpoint(
            id="overworld_to_castle",
            label="Overworld screens toward castle",
            role="transit",
            gameplay={
                "indoors": False,
                "dark_world": False,
                "screen_path": list(OVERWORLD_SCREEN_PATH),
            },
            z3_rooms=(),
            notes=(
                "Scripted BFS on the 8×8 light-world grid "
                "(alttp.overworld). No 1:1 z3 room per intermediate screen."
            ),
        ),
        OpeningCheckpoint(
            id="hyrule_castle_grounds",
            label="Hyrule Castle grounds (goal)",
            role="goal",
            gameplay={
                "indoors": False,
                "screen_id": HYRULE_CASTLE_SCREEN,
                "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
                "dark_world": False,
                "has_control": True,
                "on_castle_grounds": True,
            },
            # Courtyard is the nearest logic region for the castle exterior;
            # it is associated with screen 0x1B, not identical to it.
            z3_rooms=(
                "Hyrule Castle Courtyard",
                "Hyrule Castle Ledge",
            ),
            z3_nodes=(
                ExpectedNode(
                    "Hyrule Castle Courtyard",
                    "Hyrule Castle Entrance (South)",
                ),
                ExpectedNode(
                    "Hyrule Castle Courtyard",
                    "Hyrule Castle Secret Entrance Stairs",
                ),
                ExpectedNode("Light World", "Hyrule Castle Main Gate"),
            ),
            z3_connections=(
                ExpectedConnection(
                    origin="Hyrule Castle Main Gate",
                    destination="Hyrule Castle Courtyard",
                    required=True,
                    note="Logic edge for the courtyard / main gate.",
                ),
                ExpectedConnection(
                    origin="Hyrule Castle Entrance (South)",
                    destination="Hyrule Castle",
                    required=True,
                    note="South door into the castle interior (post-goal).",
                ),
            ),
            notes=(
                "Acceptance for boot_to_castle: controllable outdoors on "
                "light-world screen 0x1B. z3 'Hyrule Castle Courtyard' is "
                "an associated logic region name, not the screen id."
            ),
        ),
        OpeningCheckpoint(
            id="castle_interior_context",
            label="Hyrule Castle interior (context, not boot goal)",
            role="post_goal_context",
            gameplay={
                "indoors": True,
                "dark_world": False,
            },
            z3_rooms=(
                "Hyrule Castle",
                "Hyrule Castle Secret Entrance",
            ),
            z3_nodes=(
                ExpectedNode("Hyrule Castle", "Hyrule Castle Exit (South)"),
            ),
            z3_connections=(
                ExpectedConnection(
                    origin="Hyrule Castle Exit (South)",
                    destination="Light World",
                    required=True,
                ),
                ExpectedConnection(
                    origin="Hyrule Castle Secret Entrance Stairs",
                    destination="Hyrule Castle Secret Entrance",
                    required=True,
                    note="Uncle hole / secret entrance (next segment).",
                ),
            ),
            notes=(
                "Not required for boot_to_castle acceptance. Curated for "
                "the next uncle / fighter-sword experiment."
            ),
        ),
    )
