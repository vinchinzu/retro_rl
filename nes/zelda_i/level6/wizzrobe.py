"""Level 6 wizzrobe backstep combat for rooms 0x7a and 0x78.

Sword misses when overlapping at the door; controllers retreat when stuck
too close without a kill, then re-engage. Specs live in ``level6_dungeon``.

Clean leftover 0x7a ``(64,117)``: patrol-nearest ``(64,109)`` walks UP into
the cart-WRAM block ``(64,112)–(64,128)``. Occupancy contract: that pocket
has no N/S path → replan RIGHT onto the open floor; no path → stand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import GenericDungeonRoomController
from zelda_i.dungeon.threat import assess, dodgeable, in_firing_line
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = (
    "Level6EastKeyController",
    "Level6WestWizzrobeController",
    "make_east_key_controller",
    "make_west_wizzrobe_controller",
)

# West 0x7a statue/block column. Link's (64,117) is inside it.
_BLOCK_POCKET_X = 72
_BLOCK_POCKET_Y = (100, 136)
_PATROL_STUCK = 24
_KEY_XY = (120, 141)

# 0x78 east-waist leftover: live beams are type 0x59 (not 0x55). Local
# literal — do not add to dungeon/ids.py from this lane.
_EMPTY_TYPES = frozenset({0, 0xFF})
_WIZZROBE_TYPES = frozenset({0x23, 0x24})
_WIZZ_BEAM_TYPE = 0x59
_Y_BAND = 8
_BEAM_DY = 12
_THREAT_DX = 64
_WAIST_Y = 141
_NORTH_SAFE_Y = 109
_SOUTH_SAFE_Y = 173
_HOLD_SOUTH_Y = 157
_HOLD_NORTH_Y = 109
_OPEN_X_LO = 80
_OPEN_X_HI = 160
_FACE_EAST = 0x01
_FACE_WEST = 0x02
_INLAND_X = 120
_BODY_RAD = 16


def _census_shots(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Live object slots that are not empty and not a wizzrobe body."""
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and (int(obj.type_id) & 0xFF) not in _EMPTY_TYPES
        and (int(obj.type_id) & 0xFF) not in _WIZZROBE_TYPES
    )


def _shot_inbound_waist(
    snap: ZeldaSnapshot,
    obj: ZeldaObject,
    prev_xy: tuple[int, int] | None,
) -> bool:
    """True when ``obj`` shares Link's y-band and is closing on x."""
    if abs(int(obj.y) - int(snap.link_y)) > _Y_BAND:
        return False
    lx, ox = int(snap.link_x), int(obj.x)
    facing = int(obj.facing)
    if facing == _FACE_EAST and ox <= lx:
        return True
    if facing == _FACE_WEST and ox >= lx:
        return True
    if prev_xy is not None:
        vx = ox - int(prev_xy[0])
        if vx > 0 and ox <= lx:
            return True
        if vx < 0 and ox >= lx:
            return True
        return False
    return abs(ox - lx) <= _THREAT_DX


def _beams_59(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Live 0x59 slots (wizzrobe shot; v6 census)."""
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1 and (int(obj.type_id) & 0xFF) == _WIZZ_BEAM_TYPE
    )


def _same_x_wizz(live: tuple[ZeldaObject, ...], x: int) -> bool:
    """True when a 0x24/0x23 shares Link's column (v8 leftover 0x24@(160,141))."""
    return any(
        abs(int(o.x) - x) <= _BODY_RAD
        for o in live
        if (int(o.type_id) & 0xFF) in _WIZZROBE_TYPES
    )


def _sidestep_x(snap: ZeldaSnapshot) -> str:
    """Leave the 0x24 column. Prefer open floor x=80..160 / center."""
    x = int(snap.link_x)
    if x <= _OPEN_X_LO:
        return "RIGHT"
    if x >= _OPEN_X_HI:
        return "LEFT"
    return "LEFT" if x >= _INLAND_X else "RIGHT"


# The open floor 0x78 evades inside; walls and the west block sit outside it.
_EVADE_BOUNDS = (56, 200, _NORTH_SAFE_Y, _SOUTH_SAFE_Y)


def _off_band_dir(
    snap: ZeldaSnapshot,
    beams: tuple[ZeldaObject, ...] = (),
    live: tuple[ZeldaObject, ...] = (),
) -> str:
    """Leave the beam row. Fallback only — the evader owns live beams.

    v4-v9 committed south from a y table and walked back onto y=141 three
    times. Breaking the shared row is the whole move, so pick the side of
    the row Link is already nearer to and keep going that way.
    """
    del beams, live
    y = int(snap.link_y)
    if y >= _SOUTH_SAFE_Y:
        return "UP"
    if y <= _NORTH_SAFE_Y:
        return "DOWN"
    mid = (_NORTH_SAFE_Y + _SOUTH_SAFE_Y) // 2
    # Toward the larger free span, so the peel has room to finish.
    return "UP" if y >= mid else "DOWN"


def _object_rows(snap: ZeldaSnapshot) -> list[dict[str, int]]:
    """Non-empty slots for a fail census (slot, type, x, y, hp)."""
    rows: list[dict[str, int]] = []
    for obj in snap.objects:
        tid = int(obj.type_id) & 0xFF
        if obj.slot < 1 or tid in _EMPTY_TYPES:
            continue
        rows.append(
            {
                "slot": int(obj.slot),
                "type": tid,
                "x": int(obj.x),
                "y": int(obj.y),
                "hp": int(obj.hp),
            }
        )
    return rows


@dataclass
class Level6EastKeyController(GenericDungeonRoomController):
    """0x7a clear + backstep. ``0x24_E`` at (88,133) is undodgeable.

    Census: parked ``0x24@(96,125)``, ``d=8``, ``ttc=0``, ``dodgeable=False``,
    ``in_firing_line``. In-place A, y=173 stand, and always-slash engage
    each died in-room to ``0x59`` on that axis; do not peel (0 frames of
    warning vs MIN_DODGE_BODY=16).
    """

    last_progress_frame: int = 0
    prev_live_count: int = -1
    backstep_frames: int = 0
    _ttc_ring: list[str] = field(default_factory=list)
    _hit_diag: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Record ttc/dodgeable around each hit so 0x24_E contacts are measured."""
        n_hits = len(self.damage.hits)
        tracked = self.tracker.observe(snap)
        link = (int(snap.link_x), int(snap.link_y))
        impact = assess(link, tracked)
        src = impact.source
        on_line = bool(src is not None and in_firing_line(link, src))
        src_s = (
            "none"
            if src is None
            else (
                f"0x{src.type_id:02x}@({src.x},{src.y})"
                f"v=({src.vx:+.1f},{src.vy:+.1f})"
            )
        )
        self._ttc_ring.append(
            f"f{self.frames} ({link[0]},{link[1]}) ttc={impact.frames} "
            f"dodgeable={dodgeable(impact)} line={int(on_line)} src={src_s}"
        )
        del self._ttc_ring[:-20]
        action = super().step(snap)
        if len(self.damage.hits) > n_hits:
            hit = self.damage.hits[-1]
            tid = 0 if hit.type_id is None else int(hit.type_id)
            diag = (
                f"hit_f{hit.frame}_0x{tid:02x}_{hit.bearing}"
                f"_d{hit.distance}_ttc{impact.frames}"
                f"_dodgeable={dodgeable(impact)}_line={int(on_line)}"
                f"_xy=({link[0]},{link[1]})_act={hit.action}"
            )
            self._hit_diag.append(diag)
            self.notes.append(diag)
        # The reason histogram and action tail this room used to keep locally
        # are on the engine now (``_record_reason``): every room times out the
        # same way, so every room reports it. What stays here is the part that
        # is actually 0x7a's — ttc/dodgeable around each contact.
        return action

    def _go_key(self, snap: ZeldaSnapshot, *, reason: str) -> FrameAction:
        """Walk onto (120,141) then wiggle. Standing 2px off does not collect."""
        tx, ty = _KEY_XY
        dx = tx - int(snap.link_x)
        dy = ty - int(snap.link_y)
        if (
            int(snap.link_x) <= _BLOCK_POCKET_X
            and _BLOCK_POCKET_Y[0] <= int(snap.link_y) <= _BLOCK_POCKET_Y[1]
        ):
            return FrameAction(nes_action("RIGHT"), f"{reason}_leave_block")
        if abs(dx) > 1:
            btn = "RIGHT" if dx > 0 else "LEFT"
            return FrameAction(nes_action(btn), reason)
        if abs(dy) > 1:
            btn = "DOWN" if dy > 0 else "UP"
            return FrameAction(nes_action(btn), reason)
        btn = "RIGHT" if (self.frames % 8) < 4 else "LEFT"
        return FrameAction(nes_action(btn), f"{reason}_wiggle")

    def _collect_reward(self, snap: ZeldaSnapshot) -> FrameAction:
        return self._go_key(snap, reason="wizzrobe_key")

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        self.combat_frames += 1
        n_live = len(live)
        if self.prev_live_count < 0:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
        elif n_live < self.prev_live_count:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
            self.backstep_frames = 0
            self.notes.append(f"kill_to_{n_live}_f{self.frames}")

        if not live:
            return self._go_key(snap, reason="wizzrobe_key")

        self._update_stuck(snap)

        # No N/S path through the west block. Replan RIGHT; do not UP/DOWN.
        if (
            int(snap.link_x) <= _BLOCK_POCKET_X
            and _BLOCK_POCKET_Y[0] <= int(snap.link_y) <= _BLOCK_POCKET_Y[1]
        ):
            return FrameAction(nes_action("RIGHT"), "wizzrobe_leave_block")

        nearest = min(
            live,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        stuck_close = (
            dist < 16 and (self.frames - self.last_progress_frame) > 100
        )
        if stuck_close or self.backstep_frames > 0:
            if self.backstep_frames <= 0:
                self.backstep_frames = 24
                self.notes.append(f"backstep_f{self.frames}_d{dist}")
            self.backstep_frames -= 1
            if self.backstep_frames == 0:
                # Allow a fresh engage window after retreat.
                self.last_progress_frame = self.frames
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                direction = "LEFT" if dx >= 0 else "RIGHT"
            else:
                direction = "UP" if dy >= 0 else "DOWN"
            # Prefer center when pinned on a door edge.
            if snap.link_x < 40:
                direction = "RIGHT"
            elif snap.link_x > 200:
                direction = "LEFT"
            return FrameAction(nes_action(direction), "wizzrobe_backstep")

        if dist < self.spec.combat.engage_distance:
            if self._stuck_frames >= _PATROL_STUCK:
                detour = self._lattice_dir(snap, (int(nearest.x), int(nearest.y)))
                if detour is not None:
                    return FrameAction(nes_action(detour), "wizzrobe_chase_lattice")
            return self._engage(snap, nearest)
        if self._stuck_frames >= _PATROL_STUCK:
            n = len(self.spec.combat.patrol)
            self.patrol_index = (self.patrol_index + 1) % n
            self._stuck_frames = 0
            self.notes.append(f"patrol_skip_f{self.frames}")
        return self._patrol(snap)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["last_progress_frame"] = self.last_progress_frame
        base["prev_live_count"] = self.prev_live_count
        damage = dict(base.get("damage") or {})
        if self._hit_diag:
            damage["hit_diagnosis"] = list(self._hit_diag)
            damage["ttc_ring"] = list(self._ttc_ring)
            base["damage"] = damage
        return base


def make_east_key_controller() -> Level6EastKeyController:
    """Factory: GenericDungeonRoomController subclass bound to ROOM_7A_SPEC."""
    from zelda_i.level6.dungeon import ROOM_7A_SPEC

    return Level6EastKeyController(spec=ROOM_7A_SPEC)


@dataclass
class Level6WestWizzrobeController(Level6EastKeyController):
    """Peel off y=141 on 0x59, then inland LEFT. No fight on the east lip."""

    _prev_shot_xy: dict[int, tuple[int, int]] = field(default_factory=dict)
    _logged_shot_types: set[int] = field(default_factory=set)
    _last_objects: list[dict[str, int]] = field(default_factory=list)
    _dumped_objects: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if int(snap.mode) != 17:
            self._last_objects = _object_rows(snap)
        elif not self._dumped_objects:
            self._dumped_objects = True
            rows = self._last_objects or _object_rows(snap)
            self._last_objects = rows
            blob = ",".join(
                f"s{r['slot']}:0x{r['type']:02x}@({r['x']},{r['y']})hp{r['hp']}"
                for r in rows
            )
            self.notes.append(f"objects_{blob or 'none'}_f{self.frames}")
        return super().step(snap)

    def _dodge_waist_beam(self, snap: ZeldaSnapshot) -> FrameAction | None:
        inbound: list[ZeldaObject] = []
        seen: set[int] = set()
        for obj in _census_shots(snap):
            seen.add(obj.slot)
            tid = int(obj.type_id) & 0xFF
            if tid not in self._logged_shot_types:
                self._logged_shot_types.add(tid)
                self.notes.append(f"shot_type_0x{tid:02x}_s{obj.slot}_f{self.frames}")
            prev = self._prev_shot_xy.get(obj.slot)
            if _shot_inbound_waist(snap, obj, prev):
                inbound.append(obj)
            self._prev_shot_xy[obj.slot] = (int(obj.x), int(obj.y))
        for slot in list(self._prev_shot_xy):
            if slot not in seen:
                del self._prev_shot_xy[slot]
        if not inbound:
            return None
        return FrameAction(
            nes_action(_off_band_dir(snap, tuple(inbound))),
            "wizzrobe_waist_dodge",
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        # 0x78 is a shooting gallery: 0x24 wizzrobes fire along the row they
        # face. Leaving that row before the beam spawns is the only window a
        # 1 px/frame walker has (threat.MIN_DODGE_SHOT is 12 frames).
        self.evader.bounds = _EVADE_BOUNDS
        self.evader.avoid_firing_lines = True

    def _reactive(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction | None:
        """Time-to-contact dodge, then pre-emptive step off a firing line."""
        decision = self.evader.decide(
            snap,
            self.tracked,
            goal=None,
        )
        if decision is None:
            return None
        note = f"threat_{decision.reason}"
        if note not in self.notes:
            self.notes.append(note)
        if decision.stands:
            return FrameAction(nes_idle_action(), f"wizzrobe_{decision.reason}")
        # Keep the room's trace vocabulary: a vertical break leaves the beam
        # row, a horizontal one leaves a wizzrobe column.
        reason = (
            "wizzrobe_beam_peel"
            if decision.direction in ("UP", "DOWN")
            else "wizzrobe_sidestep"
        )
        return FrameAction(nes_action(decision.direction), reason)

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        # Block pocket has no N/S path — RIGHT before any UP/DOWN dodge.
        lx = int(snap.link_x)
        ly = int(snap.link_y)
        if (
            lx <= _BLOCK_POCKET_X
            and _BLOCK_POCKET_Y[0] <= ly <= _BLOCK_POCKET_Y[1]
        ):
            return FrameAction(nes_action("RIGHT"), "wizzrobe_leave_block")
        reactive = self._reactive(snap, live)
        if reactive is not None:
            return reactive
        if _same_x_wizz(live, lx):
            return FrameAction(nes_action(_sidestep_x(snap)), "wizzrobe_sidestep")
        if ly >= _HOLD_SOUTH_Y:
            return FrameAction(nes_idle_action(), "wizzrobe_south_hold")
        if ly <= _HOLD_NORTH_Y:
            return FrameAction(nes_idle_action(), "wizzrobe_north_hold")
        on_band = tuple(
            b for b in _beams_59(snap) if abs(int(b.y) - ly) <= _BEAM_DY
        )
        if on_band:
            tid = _WIZZ_BEAM_TYPE
            if tid not in self._logged_shot_types:
                self._logged_shot_types.add(tid)
                self.notes.append(
                    f"shot_type_0x{tid:02x}_s{on_band[0].slot}_f{self.frames}"
                )
            return FrameAction(
                nes_action(_off_band_dir(snap, on_band, live)),
                "wizzrobe_beam_peel",
            )
        dodge = self._dodge_waist_beam(snap)
        if dodge is not None:
            return dodge
        # Off the 0x59 band: inland toward x=120. Prefer open floor 80..160.
        if self.spec.combat.inland_dash > 0 and lx > _INLAND_X:
            return FrameAction(nes_action("LEFT"), "wizzrobe_inland_dash")
        return super()._combat(snap, live)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["shot_types"] = sorted(self._logged_shot_types)
        base["objects"] = list(self._last_objects)
        return base


def make_west_wizzrobe_controller() -> Level6WestWizzrobeController:
    """Factory: backstep combat controller for 0x78 west wizzrobes."""
    from zelda_i.level6.dungeon import ROOM_78_SPEC

    return Level6WestWizzrobeController(spec=ROOM_78_SPEC)
