"""Shared hop-path controller for Zelda I overworld level approaches.

Level modules (L2/L3/L5/L6/L8) keep geometry and stop predicates locally;
this module owns the common hop-advance / stuck / swing / maze / door core.
Level 1 remains on the phase-machine in ``overworld_nav.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import manhattan
from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE
from zelda_i.overworld.common import (
    EDGE_EAST_X,
    EDGE_NORTH_Y,
    EDGE_SOUTH_Y,
    EDGE_WEST_X,
    align_and_push,
    on_arrival_edge,
    recover_off_edge,
    swing_action,
    track_knockback,
    track_stuck,
    unstick_wiggle,
    wake_or_wait_mode,
    walk_or_swing,
)
from zelda_i.overworld.heart_farm import (
    BAND_SWEEP_WAYPOINTS,
    HeartFarmController,
    HeartFarmPhase,
)
from zelda_i.overworld.graph import (
    MAZE_WAYPOINT_TOL,
    SCREEN_5C_MAZE,
    ScreenHop,
    is_5c_maze_hop,
)
from zelda_i.overworld.locations import farm_at, restock_for, worth_rupee_farm
from zelda_i.overworld.rupee_farm import RupeeFarmController, RupeeFarmPhase
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

DEFAULT_SWING_PERIOD = 10
DEFAULT_SWING_HOLD = 3
DEFAULT_STUCK_THRESHOLD = 50
DEFAULT_MAX_FRAMES = 30000
DEFAULT_SCOOP_RADIUS = 48
_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
# Dungeon OccupancyGrid xmax=216 traps OW east-mouth leftover x≈240.
_OW_OCC_BOUNDS = (0, 255, 0, 239)
_ALIGN_X_TOL = 5


def _ow_hop_grid() -> OccupancyGrid:
    xmin, xmax, ymin, ymax = _OW_OCC_BOUNDS
    return OccupancyGrid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


class PathNavPhase(Enum):
    """Generic hop-path phases. Level modules may use their own enums with
    at least HOP / DONE / FAILED members (and often DOOR)."""

    HOP = auto()
    DOOR = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldPathController:
    """Frame policy: walk a ``ScreenHop`` table, optional maze, optional door.

    Subclasses override ``_at_stop``, ``_after_hops``, ``_before_play``, and
    ``_extra_hop_action`` for level-specific phases (Lost Hills, burn bush, …).
    Phase enums may differ per level; helpers resolve members by name.
    """

    hops: tuple[ScreenHop, ...] = ()
    hop_index: int = 0
    phase: Any = PathNavPhase.HOP
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    # Knockback: mode 8 keeps Link moving, so hits are counted, not frames.
    last_health: int = -1
    hits_taken: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    swing_period: int = DEFAULT_SWING_PERIOD
    swing_hold: int = DEFAULT_SWING_HOLD
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD
    max_frames: int = DEFAULT_MAX_FRAMES
    allowed_modes: frozenset[int] = field(
        default_factory=lambda: frozenset({PLAY_MODE, 8, 11})
    )

    # Maze (0x5C → 0x5D style). Default pred is is_5c_maze_hop when waypoints set.
    maze_hop_pred: Callable[[ScreenHop], bool] | None = None
    maze_waypoints: tuple[tuple[int, int], ...] = ()
    maze_wp_index: int = 0
    maze_screen: int = SCREEN_5C_MAZE
    maze_tol: int = MAZE_WAYPOINT_TOL

    # Door hunt after hops complete
    door_x: int | None = None
    door_dir: str = "UP"
    door_screen: int | None = None
    entry_level: int | None = None
    entry_room: int | None = None
    require_dungeon: bool = False
    require_entrance_screen: bool = False

    # Low-heart recovery (Phase 4.9). Default 3 farms when assist is off;
    # Survival health assist keeps ``filled_hearts`` full so the hook stays
    # inert. ``farm_below_hearts=0`` restores the old run-to-death hop.
    farm_below_hearts: int = 3
    farm_min_filled: int = 3
    farm_max_frames: int = 3600
    max_farm_attempts: int = 2
    farm_attempts: int = 0
    _farm: HeartFarmController | None = field(default=None, repr=False)

    # Walk onto nearby type-0x60 drops while short of a later shop price.
    # 0 keeps the old hop (ignore drops). Shops set this to ``price``.
    need_rupees: int = 0
    scoop_radius: int = DEFAULT_SCOOP_RADIUS
    # In-route kill+restock so we arrive at the shop closer to ``need_rupees``.
    rupee_farm_attempts: int = 0
    _rupee_farm: RupeeFarmController | None = field(default=None, repr=False)
    # Leftover-relative column walk for UP/DOWN hops with align_x.
    _hop_walker: OccupancyWalker | None = field(default=None, repr=False)
    _hop_walker_key: tuple[int, int] | None = field(default=None, repr=False)

    # Default hop-complete stop extras
    require_sword: bool = False
    require_triforce_bit: int | None = None
    stop_y_lo: int = 40
    stop_y_hi: int = 210

    # ------------------------------------------------------------------ #
    # Phase helpers
    # ------------------------------------------------------------------ #

    def _phase_member(self, name: str) -> Any:
        enum_cls = type(self.phase)
        return enum_cls[name]

    def _set_phase(self, phase: Any, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.stuck = 0
            if note:
                self.notes.append(note)

    def _set_phase_name(self, name: str, note: str = "") -> None:
        self._set_phase(self._phase_member(name), note)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        # Gate A on nearby threats (``_nav_snap`` set each ``step``).
        return walk_or_swing(
            self.phase_frames,
            direction,
            reason,
            getattr(self, "_nav_snap", None),
            period=self.swing_period,
            hold=self.swing_hold,
        )

    def _finish(self, note: str = "path_stop") -> FrameAction:
        self.success = True
        self._set_phase_name("DONE", note)
        return FrameAction(nes_idle_action(), "done")

    def _fail(self, note: str) -> FrameAction:
        self._set_phase_name("FAILED", note)
        return FrameAction(nes_idle_action(), note)

    # ------------------------------------------------------------------ #
    # Reset / report
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.hop_index = 0
        self.phase = self._phase_member("HOP")
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.last_health = -1
        self.hits_taken = 0
        self.farm_attempts = 0
        self._farm = None
        self.rupee_farm_attempts = 0
        self._rupee_farm = None
        self._hop_walker = None
        self._hop_walker_key = None
        self.success = False
        self.notes.clear()
        self.maze_wp_index = 0

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            current = self.hops[self.hop_index]
            hop = {
                "index": self.hop_index,
                "target": current.target,
                "direction": current.direction,
            }
            if self._is_maze_hop(current):
                hop["maze"] = True
        out: dict[str, Any] = {
            "success": self.success,
            "phase": self.phase.name if hasattr(self.phase, "name") else str(self.phase),
            "frames": self.frames,
            "hop_index": self.hop_index,
            "hop": hop,
            "notes": list(self.notes),
            "stuck": self.stuck,
            "hits_taken": self.hits_taken,
            "farm_attempts": self.farm_attempts,
            "rupee_farm_attempts": self.rupee_farm_attempts,
            "need_rupees": self.need_rupees,
        }
        if self.maze_waypoints:
            out["maze_wp_index"] = self.maze_wp_index
        if self.require_dungeon or self.require_entrance_screen:
            out["require_dungeon"] = self.require_dungeon
            out["require_entrance_screen"] = self.require_entrance_screen
        return out

    # ------------------------------------------------------------------ #
    # Stop / post-hop policy (override in subclasses)
    # ------------------------------------------------------------------ #

    def _wants_post_hop(self) -> bool:
        """True when hops complete should continue (door hunt / dungeon enter)."""
        return self.require_dungeon or self.require_entrance_screen

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon and self.entry_level is not None:
            if snap.level != self.entry_level or snap.mode != PLAY_MODE:
                return False
            if self.entry_room is not None and snap.screen != self.entry_room:
                return False
            return True
        if self.require_entrance_screen and self.door_screen is not None:
            if not (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == self.door_screen
            ):
                return False
            if self.require_sword and not snap.has_sword:
                return False
            if self.require_triforce_bit is not None:
                if not (snap.triforce & self.require_triforce_bit):
                    return False
            return True
        end_screen = (
            self.hops[-1].target
            if self.hops
            else (self.door_screen if self.door_screen is not None else -1)
        )
        if end_screen < 0:
            return False
        if not (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == end_screen
            and self.stop_y_lo < snap.link_y < self.stop_y_hi
        ):
            return False
        if self.require_sword and not snap.has_sword:
            return False
        if self.require_triforce_bit is not None:
            if not (snap.triforce & self.require_triforce_bit):
                return False
        return True

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        """Default post-hop: align door_x and push door_dir; idle in dungeon."""
        if self.entry_level is not None and snap.level == self.entry_level:
            return FrameAction(nes_idle_action(), "dungeon_settle")
        if self.door_x is not None and abs(snap.link_x - self.door_x) > 5:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing(self.door_dir, "door_hunt")

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._wants_post_hop():
            if "DOOR" in type(self.phase).__members__:
                if self.phase.name == "HOP":
                    self._set_phase_name("DOOR", "door_hunt")
            return self._simple_door_hunt(snap)
        return self._finish("hops_complete")

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Hook before hop/door play logic (e.g. exit wrong dungeon)."""
        return None

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """Hook after advance check, before maze/align (e.g. 0x5B north corridor)."""
        return None

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.hop_index < len(self.hops):
            return FrameAction(
                nes_action(self.hops[self.hop_index].direction), "scroll"
            )
        return FrameAction(nes_idle_action(), "scroll_idle")

    # ------------------------------------------------------------------ #
    # Hop advance + maze
    # ------------------------------------------------------------------ #

    def _is_maze_hop(self, hop: ScreenHop) -> bool:
        pred = self.maze_hop_pred
        if pred is None:
            if self.maze_waypoints:
                pred = is_5c_maze_hop
            else:
                return False
        return pred(hop)

    def _in_maze_phase(self, snap: ZeldaSnapshot, hop: ScreenHop) -> bool:
        if not self._is_maze_hop(hop) or not self.maze_waypoints:
            return False
        return snap.screen == self.maze_screen

    def _advance_hop(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """If arrived off the entry edge, advance hop index. Return action if handled."""
        if (
            snap.screen != hop.target
            or snap.mode not in (PLAY_MODE, 8)
            or snap.transitioning
            or on_arrival_edge(hop.direction, snap)
        ):
            return None

        self.notes.append(f"hop_{self.hop_index}_{hop.target:02x}")
        if self._is_maze_hop(hop):
            self.notes.append("maze_complete")
        self.hop_index += 1
        self.stuck = 0
        self.hits_taken = 0
        self.phase_frames = 0
        self.maze_wp_index = 0
        return self._on_hop_advanced(snap, hop)

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        """Called after hop_index increments. Default: done or idle advance."""
        if self.hop_index >= len(self.hops) and not self._wants_post_hop():
            return self._finish("path_complete")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _follow_maze(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self.maze_waypoints:
            return self._swing("RIGHT", "maze_no_waypoints")

        if "maze_start" not in self.notes:
            self.notes.append("maze_start")

        if self.maze_wp_index >= len(self.maze_waypoints):
            return self._swing("RIGHT", "maze_exit")

        tx, ty = self.maze_waypoints[self.maze_wp_index]
        if (
            abs(snap.link_x - tx) <= self.maze_tol
            and abs(snap.link_y - ty) <= self.maze_tol
        ):
            self.maze_wp_index += 1
            self.stuck = 0
            if self.maze_wp_index >= len(self.maze_waypoints):
                return self._swing("RIGHT", "maze_exit")
            tx, ty = self.maze_waypoints[self.maze_wp_index]

        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="maze_unstick")
            return action

        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) > self.maze_tol:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > self.maze_tol:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "RIGHT"
        return self._swing(direction, f"maze_wp{self.maze_wp_index}")

    def _farm_action(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Divert into a heart farm while low, then hand the hop back.

        Fail-soft on both sides: the farm gives up on its own timeout or when
        Link leaves the screen, and ``max_farm_attempts`` stops a farm/starve
        loop from replacing the hop entirely. Keep control through restock
        phases (anything other than DONE/FAILED), not only ``FARM``.
        """
        if self._farm is not None:
            action = self._farm.step(snap)
            if self._farm.phase not in (HeartFarmPhase.DONE, HeartFarmPhase.FAILED):
                return action
            note = "farm_ok" if self._farm.success else "farm_gave_up"
            self.notes.append(f"{note}_{snap.filled_hearts}")
            self._farm = None
            self.stuck = 0
            return FrameAction(nes_idle_action(), note)
        if (
            self.farm_below_hearts <= 0
            or snap.level != 0
            or snap.filled_hearts >= self.farm_below_hearts
            or self.farm_attempts >= self.max_farm_attempts
            or farm_at(int(snap.screen)) is None
        ):
            return None
        self.farm_attempts += 1
        kwargs: dict[str, Any] = {
            "min_filled": self.farm_min_filled,
            "max_frames": self.farm_max_frames,
            "farm_screen": int(snap.screen),
            "waypoints": BAND_SWEEP_WAYPOINTS,
        }
        restock = self._restock_for(snap)
        if restock is not None:
            kwargs["restock_neighbor_screen"] = restock[0]
            kwargs["restock_direction"] = restock[1]
        self._farm = HeartFarmController(**kwargs)
        self.notes.append(f"farm_start_{snap.screen:02x}_{snap.filled_hearts}")
        return self._farm.step(snap)

    def _worth(self, screen: int) -> Any:
        """Catalog rupee-farm spot, skipping lynel/peahat/zora."""
        return worth_rupee_farm(int(screen))

    def _restock_for(self, snap: ZeldaSnapshot) -> tuple[int, str] | None:
        """Catalog restock, else current hop target/direction (leave toward hop)."""
        pair = restock_for(int(snap.screen))
        if pair is not None:
            return pair
        if self.hop_index < len(self.hops):
            hop = self.hops[self.hop_index]
            return (int(hop.target), hop.direction)
        return None

    def _rupee_farm_action(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Kill+restock on a farmable screen while short of ``need_rupees``.

        Fail-soft: a farm timeout or left-screen hands the hop back. Does not
        run after hops complete (door hunt / shop farm-after-hops owns that).
        """
        if self._rupee_farm is not None:
            action = self._rupee_farm.step(snap)
            phase = self._rupee_farm.phase
            if phase in (RupeeFarmPhase.FARM, RupeeFarmPhase.RETURN):
                return action
            if phase is RupeeFarmPhase.DONE:
                note = f"rupee_farm_ok_{snap.rupees}"
            else:
                note = "rupee_farm_gave_up"
            self.notes.append(note)
            self._rupee_farm = None
            self.stuck = 0
            return FrameAction(nes_idle_action(), note)
        if (
            self.need_rupees <= 0
            or snap.rupees >= self.need_rupees
            or snap.level != 0
            or snap.mode not in (PLAY_MODE, 8)
            or self.hop_index >= len(self.hops)
            or self.rupee_farm_attempts >= self.max_farm_attempts
            or self._worth(int(snap.screen)) is None
        ):
            return None
        hop = self.hops[self.hop_index]
        if self._in_maze_phase(snap, hop):
            return None
        restock = self._restock_for(snap)
        if restock is None or restock[1] not in _OPPOSITE:
            return None
        neighbor, direction = restock
        self.rupee_farm_attempts += 1
        self._rupee_farm = RupeeFarmController(
            target_rupees=self.need_rupees,
            farm_screen=int(snap.screen),
            restock_neighbor_screen=int(neighbor),
            restock_direction=direction,
            leftover_screen=int(snap.screen),
            max_frames=self.farm_max_frames,
        )
        self.notes.append(f"rupee_farm_start_{snap.screen:02x}")
        return self._rupee_farm.step(snap)

    def _rupee_scoop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """Walk onto a nearby rupee drop when ``snap.rupees < need_rupees``.

        Stays on the current screen and refuses a drop that sits on the
        hop's opposite edge (would scroll away). Does not swing — pickup
        is contact. ``need_rupees=0`` (default) never diverts.
        """
        if self.need_rupees <= 0 or snap.rupees >= self.need_rupees:
            return None
        if snap.mode != PLAY_MODE or snap.level != 0:
            return None
        drops = [
            obj
            for obj in snap.objects
            if obj.slot >= 1
            and int(obj.type_id) == RUPEE_DROP_OBJECT_TYPE
            and 40 < obj.y < 220
            and 8 < obj.x < 248
        ]
        if not drops:
            return None
        nearest = min(
            drops,
            key=lambda obj: manhattan(snap.link_x, snap.link_y, obj.x, obj.y),
        )
        dist = manhattan(snap.link_x, snap.link_y, nearest.x, nearest.y)
        if dist > self.scoop_radius:
            return None
        if hop.direction == "RIGHT" and nearest.x < EDGE_WEST_X + 16:
            return None
        if hop.direction == "LEFT" and nearest.x > EDGE_EAST_X - 16:
            return None
        if hop.direction == "DOWN" and nearest.y < EDGE_NORTH_Y + 16:
            return None
        if hop.direction == "UP" and nearest.y > EDGE_SOUTH_Y - 16:
            return None
        if dist <= 4:
            return FrameAction(nes_idle_action(), "scoop_rupee")
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) >= abs(dy) and abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 2:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            return FrameAction(nes_idle_action(), "scoop_rupee")
        return FrameAction(nes_action(direction), "scoop_rupee")

    def _occupancy_align_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """Walk to ``align_x`` on a walkable y; peel on miss. None on-column.

        East-mouth leftover (x≈240): LEFT toward the door column; occupancy
        miss → block cell → y-peel; no path → stand. Do not RIGHT-scroll.
        """
        if (
            hop.direction not in ("UP", "DOWN")
            or hop.align_x is None
            or hop.y_band is not None
        ):
            return None
        ax = int(hop.align_x)
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - ax) <= _ALIGN_X_TOL:
            self._hop_walker = None
            self._hop_walker_key = None
            return None
        # Inland off-column stays on align_and_push (credits-tape hops).
        if x < EDGE_EAST_X and x > EDGE_WEST_X:
            return None
        if self.stuck > self.stuck_threshold:
            return FrameAction(nes_idle_action(), f"hop{self.hop_index}_stand")

        key = (self.hop_index, int(snap.screen))
        if self._hop_walker is None or self._hop_walker_key != key:
            self._hop_walker = OccupancyWalker(grid=_ow_hop_grid(), slide=True)
            self._hop_walker_key = key
        walker = self._hop_walker
        xy = (x, y)
        direction = walker.next_dir(xy, (ax, y))
        # RIGHT at x≥232 scrolls to 0x4D; LEFT at west edge leaves the screen.
        # Hop UP/DOWN stays a legal y-peel after a LEFT miss (0x4C corridor).
        forbidden: set[str] = set()
        if x >= EDGE_EAST_X:
            forbidden.add("RIGHT")
        if x <= EDGE_WEST_X:
            forbidden.add("LEFT")
        for _ in range(4):
            if direction is None or direction not in forbidden:
                break
            walker.grid.mark_blocked_ahead(*xy, direction)
            walker.path = None
            direction = walker.next_dir(xy)
        if direction is None or direction in forbidden:
            return FrameAction(nes_idle_action(), f"hop{self.hop_index}_stand")
        # Keep the occupancy lane; off-axis face would RIGHT-scroll the mouth.
        return swing_action(
            self.phase_frames,
            direction,
            f"hop{self.hop_index}_occ",
            period=self.swing_period,
            hold=self.swing_hold,
        )

    def _do_hop(self, snap: ZeldaSnapshot) -> FrameAction:
        hop = self.hops[self.hop_index]
        advanced = self._advance_hop(snap, hop)
        if advanced is not None:
            return advanced

        extra = self._extra_hop_action(snap, hop)
        if extra is not None:
            return extra

        if self._in_maze_phase(snap, hop):
            return self._follow_maze(snap)

        occ = self._occupancy_align_action(snap, hop)
        if occ is not None:
            return occ

        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck)
            return action

        edge = recover_off_edge(snap, hop.direction, swing=self._swing)
        if edge is not None:
            return edge

        scoop = self._rupee_scoop(snap, hop)
        if scoop is not None:
            return scoop

        return align_and_push(
            snap,
            direction=hop.direction,
            reason=f"hop{self.hop_index}",
            align_x=hop.align_x,
            align_y=hop.align_y,
            y_band=hop.y_band,
            stuck=0,
            stuck_threshold=self.stuck_threshold,
            swing=self._swing,
        )

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self._nav_snap = snap
        self.frames += 1
        self.phase_frames += 1
        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )
        self.hits_taken, self.last_health, self.stuck = track_knockback(
            snap,
            last_health=self.last_health,
            hits=self.hits_taken,
            stuck=self.stuck,
        )

        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if snap.mode == 17:
            return self._fail("link_death")

        if self._at_stop(snap):
            return self._finish("path_stop")

        early = self._before_play(snap)
        if early is not None:
            return early

        # Active farms own their own scroll/mode handling (restock leave/return).
        farm_busy = self._farm is not None or self._rupee_farm is not None
        if snap.transitioning and not farm_busy:
            return self._handle_transition(snap)

        if snap.mode not in self.allowed_modes and not farm_busy:
            return wake_or_wait_mode(self.phase_frames, snap.mode)

        farm = self._farm_action(snap)
        if farm is not None:
            return farm

        rupee = self._rupee_farm_action(snap)
        if rupee is not None:
            return rupee

        if self.hop_index >= len(self.hops):
            return self._after_hops(snap)

        return self._do_hop(snap)


__all__ = [
    "DEFAULT_SWING_PERIOD",
    "DEFAULT_SWING_HOLD",
    "DEFAULT_STUCK_THRESHOLD",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_SCOOP_RADIUS",
    "PathNavPhase",
    "OverworldPathController",
    "is_5c_maze_hop",
    "MAZE_WAYPOINT_TOL",
    "SCREEN_5C_MAZE",
]
