"""Walking a room's entry route, and the escape it never had.

``DoorRoute`` waypoints are hand-written against one measured leftover pose,
and the leftover moves whenever an upstream room's timing shifts — the L1
chain is frame-perfect, so a change anywhere reshuffles every room after it.
When that happens the axis walk aims the first leg through geometry nobody
checked and holds one button into it until the stage budget runs out. The
Survival ``clear45_key`` stage spent 8999 of its 9000 frames pressing DOWN at
``(168,141)`` in room ``0x44``, against a statue the live ``$6530`` map knows
about, and reported ``combat_frames=0`` and a single ``timeout`` note.

Every other phase already had an escape (FIGHT has the occupancy walker and
the patrol skip; COLLECT has the stale-waypoint skip). This is that escape for
ROUTE_ENTRY, and it is deliberately *stall-triggered*: below the threshold the
route is the same axis walk it has always been, so a green chain is
frame-identical and only a route that was going to time out behaves
differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.tilemap import has_room_tile_map
from zelda_i.ram import ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker, measured_walker

if TYPE_CHECKING:
    # ``dungeon.engine`` imports this module at import time, so the annotation
    # may only be resolved under ``from __future__ import annotations``.
    from zelda_i.dungeon.engine import DoorRoute

__all__ = (
    "ENTER_STALL_FRAMES",
    "EntryRouteWalker",
    "ROUTE_BOUNDS",
    "ROUTE_STALL_FRAMES",
)

# Identical-pose frames on a held entry-route button before the route is
# treated as walled. Link moves ~1px/frame, so 24 frames of no movement while
# a direction is pressed is geometry, not lag.
ROUTE_STALL_FRAMES = 24
# The ENTER push is not steered (a key door eats frames while it opens); this
# is only how long it may stand still before the pose is worth a note.
ENTER_STALL_FRAMES = 120
# Whole-playfield bounds for the entry-route replan. Entry routes walk door
# mouths at x=16/224 and y=80/208, which the inland fight box excludes.
ROUTE_BOUNDS: tuple[int, int, int, int] = (16, 224, 80, 208)


@dataclass
class EntryRouteWalker:
    """ROUTE_ENTRY state and rules, mixed into the room controller.

    Every field is ``init=False``: this is a base of a dataclass whose own
    first field (``spec``) has no default, and only ``__init__`` fields
    participate in the default-ordering rule.

    The owner supplies ``self._env`` (for the tile map), ``self.frames``,
    ``self.notes`` and ``self.waypoint_index``.
    """

    _resolved_route: "object | None" = field(default=None, init=False, repr=False)
    _resolved_waypoints: tuple[tuple[int, int], ...] | None = field(
        default=None, init=False, repr=False
    )
    # Entry-route stall guard: pose repeat counter, the room the route walker
    # was measured in, and the walker itself (built lazily, only on a stall).
    _route_stall_frames: int = field(default=0, init=False, repr=False)
    _route_stall_xy: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    _route_walker: OccupancyWalker | None = field(
        default=None, init=False, repr=False
    )
    _route_walker_room: int | None = field(default=None, init=False, repr=False)
    _route_skips: int = field(default=0, init=False, repr=False)
    # Latched once a leg has walled: the axis rule put Link there, so handing
    # the leg back to it after one replanned step just walls him again (a
    # 24:1 duty cycle across the same wall). Cleared when the leg advances.
    _route_replanning: bool = field(default=False, init=False, repr=False)

    def _reset_entry_route(self) -> None:
        """Forget the resolved route and every stall latch (phase change)."""
        self._resolved_route = None
        self._resolved_waypoints = None
        self._route_stall_frames = 0
        self._route_stall_xy = None
        self._route_skips = 0
        self._route_walker = None
        self._route_walker_room = None
        self._route_replanning = False

    def _note_enter_stall(self, snap: ZeldaSnapshot, direction: str) -> None:
        """The door push is held by design; leave the pose behind anyway.

        A stage that dies in ENTER otherwise reports only ``timeout``.
        """
        xy = (int(snap.link_x), int(snap.link_y))
        if self._route_stall_xy == xy:
            self._route_stall_frames += 1
        else:
            self._route_stall_xy = xy
            self._route_stall_frames = 0
        if self._route_stall_frames == ENTER_STALL_FRAMES:
            self.notes.append(f"enter_stall_f{self.frames}_{xy}_{direction}")

    def _route_walker_for(self, snap: ZeldaSnapshot) -> OccupancyWalker | None:
        """Measured ``$6530`` walker for the room the entry route is walking.

        Built only when the naive route has already stalled, and rebuilt when
        the route crosses into another room, so the geometry is always the
        live one. ``None`` when the tile map is unavailable (no bound env, or
        a RAM window without the cart-WRAM half) — the caller then falls back
        to skipping the waypoint rather than pretending to know the walls.
        """
        room = int(snap.screen)
        if self._route_walker is not None and self._route_walker_room == room:
            return self._route_walker
        if self._env is None:
            return None
        ram = self._env.get_ram()
        if not has_room_tile_map(ram):
            return None
        # Sticky (the ``measured_walker`` default): a miss here is kept, not
        # forgotten and re-tried. The tile map is sampled at one point under
        # Link, but Link is 16px wide, so a statue his *right edge* clips
        # reads as open floor — measured at L1 0x44 (168,141), where the map
        # says the cell south is free and the ROM will not let him walk into
        # it. A forgetful walker replans straight back into that cell and
        # yo-yos until the stage budget is gone.
        self._route_walker = measured_walker(ram, ROUTE_BOUNDS)
        self._route_walker_room = room
        return self._route_walker

    def _route_escape(
        self, snap: ZeldaSnapshot, target: tuple[int, int], n_waypoints: int
    ) -> FrameAction:
        """The held button is into a wall. Replan, or give up on this waypoint.

        Hand-written entry waypoints are written against one measured leftover
        pose, and the leftover moves whenever an upstream room's timing shifts
        (the L1 chain is frame-perfect). When that happens the axis walk aims
        the first leg through geometry nobody checked. Measure the walls and
        BFS instead; if even that has no path, drop the waypoint and try the
        next one. Bounded by the waypoint count so a walled route still fails
        on its own budget rather than cycling.
        """
        xy = (int(snap.link_x), int(snap.link_y))
        walker = self._route_walker_for(snap)
        if walker is not None:
            direction = walker.next_dir(xy, target)
            if direction is not None:
                self._route_replanning = True
                return FrameAction(nes_action(direction), "entry_route_replan")
        if self._route_skips < n_waypoints:
            self._route_skips += 1
            self.waypoint_index += 1
            self._route_stall_frames = 0
            self._route_replanning = False
            self.notes.append(
                f"entry_route_skip_f{self.frames}_{xy}_to{target}"
            )
            return FrameAction(nes_idle_action(), "entry_route_skip")
        return FrameAction(nes_idle_action(), "entry_route_walled")

    def _follow_route(self, snap: ZeldaSnapshot, route: DoorRoute) -> FrameAction:
        if self._resolved_route is not route or self._resolved_waypoints is None:
            self._resolved_route = route
            self._resolved_waypoints = (
                route.waypoints(snap) if callable(route.waypoints) else route.waypoints
            )
            self.waypoint_index = 0
        waypoints = self._resolved_waypoints
        if self.waypoint_index >= len(waypoints):
            return FrameAction(nes_idle_action(), "entry_route_done")
        tx, ty = waypoints[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            self._route_stall_frames = 0
            self._route_replanning = False
            if self.waypoint_index >= len(waypoints):
                return FrameAction(nes_idle_action(), "entry_route_done")
            return FrameAction(nes_idle_action(), "entry_waypoint_idle")
        xy = (int(snap.link_x), int(snap.link_y))
        if self._route_stall_xy == xy:
            self._route_stall_frames += 1
        else:
            self._route_stall_xy = xy
            self._route_stall_frames = 0
        if self._route_stall_frames == ROUTE_STALL_FRAMES:
            # One note per stall, with the pose and the leg that walled — the
            # measurement a timeout never leaves behind.
            self.notes.append(
                f"entry_route_stall_f{self.frames}_{xy}"
                f"_wp{self.waypoint_index}{(tx, ty)}"
            )
        if self._route_replanning or self._route_stall_frames >= ROUTE_STALL_FRAMES:
            return self._route_escape(snap, (tx, ty), len(waypoints))
        y_first = route.y_first and abs(dy) > 2
        if not y_first and abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "entry_route")
