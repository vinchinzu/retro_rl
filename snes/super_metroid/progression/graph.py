"""Room progression graph class and pathfinding helpers."""

from __future__ import annotations

from collections import defaultdict

from retro_harness.adventure.graph import RouteGraph
from super_metroid.progression.types import (
    VERIFICATION_RANK,
    DoorEdge,
    ObservedTransition,
    ProgressionMilestone,
    RoomNode,
)


class RoomProgressionGraph:
    """Validated directed room graph with capability-aware pathfinding."""

    def __init__(
        self,
        rooms: tuple[RoomNode, ...],
        edges: tuple[DoorEdge, ...],
        milestones: tuple[ProgressionMilestone, ...],
        *,
        graph_id: str,
    ) -> None:
        self.graph_id = graph_id
        self.rooms = {room.room_id: room for room in rooms}
        self.edges = edges
        self.milestones = milestones
        self._outgoing: dict[int, list[DoorEdge]] = defaultdict(list)
        self._edge_pairs: dict[tuple[int, int], DoorEdge] = {}
        for edge in edges:
            if (
                edge.source_room_id not in self.rooms
                or edge.target_room_id not in self.rooms
            ):
                raise ValueError(f"edge {edge.edge_id} references an unknown room")
            self._outgoing[edge.source_room_id].append(edge)
            self._edge_pairs[(edge.source_room_id, edge.target_room_id)] = edge
        # Shared graph view for pathfinding (single BFS implementation).
        self._route_graph = RouteGraph(
            (room.as_graph_node() for room in rooms),
            (edge.as_graph_edge() for edge in edges),
        )

    def room_name(self, room_id: int) -> str:
        room = self.rooms.get(room_id)
        return room.name if room else f"Unknown 0x{room_id:04X}"

    def edge_for(self, source_room_id: int, target_room_id: int) -> DoorEdge | None:
        return self._edge_pairs.get((source_room_id, target_room_id))

    def observe_transition(
        self,
        frame: int,
        source_room_id: int,
        target_room_id: int,
        *,
        leave_kinematics: dict[str, object] | None = None,
        entry_kinematics: dict[str, object] | None = None,
    ) -> ObservedTransition:
        edge = self.edge_for(source_room_id, target_room_id)
        return ObservedTransition(
            frame=frame,
            source_room_id=source_room_id,
            target_room_id=target_room_id,
            edge_id=edge.edge_id if edge else None,
            leave_kinematics=leave_kinematics,
            entry_kinematics=entry_kinematics,
        )

    def shortest_path(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
    ) -> tuple[DoorEdge, ...] | None:
        """Capability-aware BFS via shared :class:`RouteGraph`."""
        path = self._route_graph.shortest_path(
            source_room_id,
            target_room_id,
            capabilities=capabilities,
        )
        if path is None:
            return None
        by_id = {edge.edge_id: edge for edge in self.edges}
        return tuple(by_id[edge.edge_id] for edge in path)

    def outgoing(
        self,
        room_id: int,
        *,
        capabilities: frozenset[str] | None = None,
        verification: str | None = None,
    ) -> tuple[DoorEdge, ...]:
        """Edges leaving ``room_id``, optionally filtered by caps / verification."""
        edges = tuple(self._outgoing.get(room_id, ()))
        if capabilities is not None:
            edges = tuple(e for e in edges if e.requires <= capabilities)
        if verification is not None:
            edges = tuple(e for e in edges if e.verification == verification)
        return edges

    def suggest_edges(
        self,
        room_id: int,
        *,
        capabilities: frozenset[str] | None = None,
        prefer: str = "continuous",
        exclude_verifications: frozenset[str] = frozenset(),
        include_gated: bool = True,
    ) -> tuple[DoorEdge, ...]:
        """Ranked outbound edges from ``room_id`` (single suggest surface).

        ``prefer`` selects sort priority:
        - ``continuous`` (default): continuous → controller_dev → open work
        - ``pure_work``: unverified → planned → controller_dev (omit continuous)

        When no capability-ready edges exist and ``include_gated`` is True,
        still surfaces gated outbound edges so authors see missing requires.
        """
        if room_id not in self.rooms:
            return ()
        caps = capabilities if capabilities is not None else frozenset()
        ready = self.outgoing(room_id, capabilities=caps)
        if not ready and include_gated:
            ready = self.outgoing(room_id)
        if exclude_verifications:
            ready = tuple(e for e in ready if e.verification not in exclude_verifications)
        if prefer == "pure_work":
            # Open pure work first; continuous edges are usually excluded above.
            order = {
                "unverified": 0,
                "planned": 1,
                "controller_dev": 2,
                "continuous": 3,
            }
        else:
            # Continuous extension: preferred status first, then weaker ranks.
            order = {
                prefer: 0,
                "continuous": 0,
                "controller_dev": 1,
                "unverified": 2,
                "planned": 3,
            }
        ranked = sorted(
            ready,
            key=lambda e: (order.get(e.verification, 9), e.edge_id),
        )
        return tuple(ranked)

    def suggest_next_hops(
        self,
        room_id: int,
        *,
        capabilities: frozenset[str] | None = None,
        prefer_verification: str = "continuous",
    ) -> tuple[DoorEdge, ...]:
        """Ranked next hops from ``room_id`` for continuous extension.

        Thin wrapper over :meth:`suggest_edges` with continuous-first ranking.
        """
        return self.suggest_edges(
            room_id,
            capabilities=capabilities,
            prefer=prefer_verification,
        )

    def suggest_pure_work(
        self,
        room_id: int,
        *,
        capabilities: frozenset[str] | None = None,
    ) -> tuple[DoorEdge, ...]:
        """Next hops that still need pure geometry work from ``room_id``.

        Thin wrapper over :meth:`suggest_edges` excluding continuous edges.
        """
        return self.suggest_edges(
            room_id,
            capabilities=capabilities,
            prefer="pure_work",
            exclude_verifications=frozenset({"continuous"}),
        )

    def path_summary(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
        *,
        min_verification: str = "continuous",
    ) -> dict[str, object]:
        """Shortest-path readiness with a tunable verification floor.

        Single path API: ``min_verification`` is the lowest accepted status
        before a blocking edge is reported (default ``continuous``).
        ``pure_gated`` is True when every edge meets that floor.
        ``all_continuous`` is True when every edge is continuous regardless
        of ``min_verification``.
        """
        need = VERIFICATION_RANK.get(min_verification, VERIFICATION_RANK["controller_dev"])
        path = self.shortest_path(source_room_id, target_room_id, capabilities)
        if path is None:
            return {
                "reachable": False,
                "edges": [],
                "all_continuous": False,
                "pure_gated": False,
                "blocking": None,
                "blocking_edge_id": None,
                "min_verification": min_verification,
            }
        edges_payload: list[dict[str, object]] = []
        blocking: dict[str, object] | None = None
        blocking_edge_id: str | None = None
        for edge in path:
            edges_payload.append(
                {
                    "edgeId": edge.edge_id,
                    "from": f"0x{edge.source_room_id:04X}",
                    "to": f"0x{edge.target_room_id:04X}",
                    "verification": edge.verification,
                    "requires": sorted(edge.requires),
                    "policyId": edge.policy_id,
                }
            )
            if (
                blocking is None
                and VERIFICATION_RANK.get(edge.verification, 0) < need
            ):
                blocking = {
                    "edgeId": edge.edge_id,
                    "from": f"0x{edge.source_room_id:04X}",
                    "to": f"0x{edge.target_room_id:04X}",
                    "verification": edge.verification,
                    "requires": sorted(edge.requires),
                    "policyId": edge.policy_id,
                }
                blocking_edge_id = edge.edge_id
        verifications = [e.verification for e in path]
        return {
            "reachable": True,
            "edges": edges_payload,
            "all_continuous": all(v == "continuous" for v in verifications),
            "pure_gated": blocking is None,
            "blocking": blocking,
            "blocking_edge_id": blocking_edge_id,
            "min_verification": min_verification,
        }

    def pure_gate(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
        *,
        min_verification: str = "controller_dev",
    ) -> dict[str, object]:
        """First path edge below the pure gate (for tip composition).

        Wrapper over :meth:`path_summary` preserving the planner dict shape
        (blocking edge object, slim edge rows without requires).
        """
        summary = self.path_summary(
            source_room_id,
            target_room_id,
            capabilities,
            min_verification=min_verification,
        )
        if not summary["reachable"]:
            return {
                "reachable": False,
                "pure_gated": False,
                "blocking": None,
                "edges": [],
            }
        return {
            "reachable": True,
            "pure_gated": summary["pure_gated"],
            "blocking": summary["blocking"],
            "edges": [
                {
                    "edgeId": e["edgeId"],
                    "from": e["from"],
                    "to": e["to"],
                    "verification": e["verification"],
                }
                for e in summary["edges"]  # type: ignore[union-attr]
            ],
        }

    def path_verification(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
        *,
        min_verification: str = "continuous",
    ) -> dict[str, object]:
        """Summarize shortest-path readiness (edge ids + verification mix).

        Wrapper over :meth:`path_summary`. ``blocking`` remains the first
        edge id below ``min_verification`` (default continuous) for artifact
        stability; use :meth:`path_summary` for the full blocking object.
        """
        summary = self.path_summary(
            source_room_id,
            target_room_id,
            capabilities,
            min_verification=min_verification,
        )
        if not summary["reachable"]:
            return {
                "reachable": False,
                "edges": [],
                "all_continuous": False,
                "blocking": None,
            }
        return {
            "reachable": True,
            "edges": [
                {
                    "edgeId": e["edgeId"],
                    "from": e["from"],
                    "to": e["to"],
                    "verification": e["verification"],
                    "requires": e["requires"],
                }
                for e in summary["edges"]  # type: ignore[union-attr]
            ],
            "all_continuous": summary["all_continuous"],
            "blocking": summary["blocking_edge_id"],
        }

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready navigation-map representation."""
        rooms = [
            {
                "roomId": room.room_id,
                "roomIdHex": f"0x{room.room_id:04X}",
                "name": room.name,
                "area": room.area,
                "tags": sorted(room.tags),
            }
            for room in self.rooms.values()
        ]
        edges = [
            {
                "edgeId": edge.edge_id,
                "sourceRoomId": edge.source_room_id,
                "sourceRoomIdHex": f"0x{edge.source_room_id:04X}",
                "targetRoomId": edge.target_room_id,
                "targetRoomIdHex": f"0x{edge.target_room_id:04X}",
                "exitDirection": edge.exit_direction,
                "entryDirection": edge.entry_direction,
                "requires": sorted(edge.requires),
                "policyId": edge.policy_id,
                "verification": edge.verification,
            }
            for edge in self.edges
        ]
        milestones = [
            {
                "milestoneId": milestone.milestone_id,
                "label": milestone.label,
                "condition": {
                    "roomId": milestone.condition.room_id,
                    "roomIdHex": (
                        f"0x{milestone.condition.room_id:04X}"
                        if milestone.condition.room_id is not None
                        else None
                    ),
                    "gameStates": sorted(milestone.condition.game_states),
                    "collectedItemsMask": milestone.condition.collected_items_mask,
                    "minimumAmmoCapacities": list(
                        milestone.condition.minimum_ammo_capacities
                    ),
                },
                "requires": sorted(milestone.requires),
                "acquires": sorted(milestone.acquires),
                "timeoutFrames": milestone.timeout_frames,
                "policyId": milestone.policy_id,
            }
            for milestone in self.milestones
        ]
        return {
            "schemaVersion": 1,
            "graphId": self.graph_id,
            "rooms": rooms,
            "edges": edges,
            "milestones": milestones,
        }
