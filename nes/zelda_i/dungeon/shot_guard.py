"""Wizzrobe-aware safety filter: read the countdowns, not a velocity.

Level 9's damage is Wizzrobes (Clean at 10 hearts: 0x05 4-5h, 0x20 5-7h,
0x10 dies, the join dies). Their shots are not reactive-evader material,
because the ROM decides them from bytes that are already in RAM
(aldonunez/zelda1-disassembly, Z_04.asm ``UpdateBlueWizzrobe`` /
``UpdateRedWizzrobe``):

* **Blue ($23)** shoots magic ``$58`` only on frames where
  ``FrameCounter & $1F == 0``, only while not teleporting (``ObjRemDistance``
  is 0), only when Link's ``Y & $F0`` equals its own ``Y & $F0`` and it faces
  him horizontally -- or when Link's X equals its ``X & $F0`` exactly and it
  faces him vertically (``BlueWizzrobe_TryShooting``).
* **Red/orange ($24)** counts ``ObjState`` down every frame: ``$FF`` places it
  on Link's row or column 32-80 px away with a random facing, ``$C0-$FE`` fades
  in, ``$80-$BF`` is solid and fires magic ``$59`` in its facing at ``$B0``,
  ``$40-$7F`` fades out (from ``$4F``), ``$00-$3F`` is invisible.
* Magic shots fly straight at 3 px/frame (measured on L9 0x05) in the
  shooter's ``ObjDir``; ``$58`` costs 2 hearts and ``$59`` 4 (``ObjTypeToDamagePoints``),
  halved per ring level. Without the Magical Shield nothing parries them.

So the future is known, not guessed: this module simulates Link on the turn
lattice for a few candidate plans (hold one direction for *k* frames, then
stand) against those shots and the bodies, and swaps the inner controller's
press only when that press walks into a predicted hit. It never writes RAM.
Link's own motion is the ROM's: 1.5 px/frame, and a press across the current
axis keeps him moving the way he faces until he reaches the 8 px grid line.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.species import species_of
from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

ADDR_FRAME_COUNTER = 0x15
ADDR_OBJ_X = 0x70
ADDR_OBJ_Y = 0x84
ADDR_OBJ_DIR = 0x98
ADDR_OBJ_STATE = 0xAC
ADDR_OBJ_TYPE = 0x34F
ADDR_OBJ_REM_DISTANCE = 0x394
ADDR_OBJ_METASTATE = 0x405
ADDR_OBJ_HP = 0x485
ADDR_OBJ_INVINCIBILITY = 0x4F0

PATRA_BODIES = (0x47, 0x48)
PATRA_EYES = (0x25, 0x26)
BLADE_TRAP = 0x49
BLUE_WIZZROBE = 0x23
RED_WIZZROBE = 0x24
BLUE_MAGIC = 0x58
RED_MAGIC = 0x59
MAGIC_SHOTS = (BLUE_MAGIC, RED_MAGIC)
# Shots that fly one straight axis at 3 px/f (rock, arrow, beam, magic). The
# fireballs ($55/$56) aim at Link on their own vector and are not modelled.
STRAIGHT_SHOTS = frozenset({0x53, 0x54, 0x57, 0x58, 0x59, 0x5B})
RED_FIRE_STATE = 0xB0
RED_VISIBLE_STATE = 0x40
BLUE_SHOT_PERIOD = 0x20
SHOT_SPEED = 3.0
BLUE_WALK_SPEED = 0.5
BLUE_TELEPORT_SPEED = 1.0
LINK_SPEED = 1.5
CONTACT = 9
# A room box the shots die outside of (the playfield ring is wall).
ROOM_X = (8, 232)
ROOM_Y = (61, 221)

DIRS = {"RIGHT": (1, 0), "LEFT": (-1, 0), "DOWN": (0, 1), "UP": (0, -1)}
_DIR_BITS = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
_FACING_NAME = {1: "RIGHT", 2: "LEFT", 4: "DOWN", 8: "UP"}


def _dir_vec(bits: int) -> tuple[int, int]:
    """8-way ``ObjDir`` bits as a unit step (diagonals combine)."""
    dx = (1 if bits & 1 else 0) - (1 if bits & 2 else 0)
    dy = (1 if bits & 4 else 0) - (1 if bits & 8 else 0)
    return dx, dy


@dataclass(frozen=True)
class Mover:
    """One predicted hazard: straight-line motion over a frame window."""

    x: float
    y: float
    vx: float
    vy: float
    t0: int
    t1: int
    mid_x: int
    slot: int
    kind: str

    def at(self, t: int) -> tuple[float, float] | None:
        if t < self.t0 or t > self.t1:
            return None
        dt = t - self.t0
        x = self.x + self.vx * dt
        y = self.y + self.vy * dt
        if self.kind == "shot" and not (ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1]):
            return None
        return x, y


@dataclass(frozen=True)
class OrbitMover:
    """A Patra eye: polar motion about its (moving) body, fit on two frames."""

    bx: float
    by: float
    bvx: float
    bvy: float
    r: float
    dr: float
    th: float
    dth: float
    t0: int
    t1: int
    mid_x: int
    slot: int
    kind: str = "body"

    def at(self, t: int) -> tuple[float, float] | None:
        if t < self.t0 or t > self.t1:
            return None
        r = max(0.0, self.r + self.dr * t)
        th = self.th + self.dth * t
        return (self.bx + self.bvx * t + r * math.cos(th), self.by + self.bvy * t + r * math.sin(th))


@dataclass(frozen=True)
class BlueShooter:
    """A blue Wizzrobe that may spawn a ``$58`` on a period frame."""

    x: float
    y: float
    vx: float
    vy: float
    dir_bits: int
    teleport_left: int
    slot: int


@dataclass
class Forecast:
    """Everything the RAM says about the next ``horizon`` frames."""

    movers: list[Mover] = field(default_factory=list)
    shooters: list[BlueShooter] = field(default_factory=list)
    frame_counter: int = 0
    link_invincible: int = 0

    @property
    def empty(self) -> bool:
        return not self.movers and not self.shooters


def read_forecast(
    ram: np.ndarray,
    horizon: int,
    *,
    body_horizon: int = 12,
    bodies: bool = True,
    blue_bodies: bool = True,
) -> Forecast:
    """Hazards from RAM: shots in flight, Wizzrobe shots to come, bodies.

    ``bodies=False`` keeps only shots (in flight and predicted): contact is the
    inner controller's fight, and a guard that also dodges bodies stands Link
    still while an orange Wizzrobe fades in within reach of one cut.
    """
    fc = Forecast(
        frame_counter=int(ram[ADDR_FRAME_COUNTER]),
        link_invincible=int(ram[ADDR_OBJ_INVINCIBILITY]),
    )
    for slot in range(1, 12):
        t = int(ram[ADDR_OBJ_TYPE + slot])
        if t == 0:
            continue
        x = float(ram[ADDR_OBJ_X + slot])
        y = float(ram[ADDR_OBJ_Y + slot])
        d = int(ram[ADDR_OBJ_DIR + slot])
        st = int(ram[ADDR_OBJ_STATE + slot])
        meta = int(ram[ADDR_OBJ_METASTATE + slot])
        mid_x = species_of(t).mid_offset_x
        if t >= 0x53:
            # A monster shot hurts only in state $1x (CheckLinkCollision). The
            # floor drop ($60) and the dead dummy also live up here: skip them.
            if t not in STRAIGHT_SHOTS or st & 0xF0 != 0x10 or d not in _DIR_BITS:
                continue
            ux, uy = _DIR_BITS[d]
            fc.movers.append(
                Mover(x, y, ux * SHOT_SPEED, uy * SHOT_SPEED, 0, horizon, mid_x, slot, "shot")
            )
            continue
        if meta != 0:
            continue
        if t == BLUE_WIZZROBE:
            if int(ram[ADDR_OBJ_HP + slot]) == 0:
                continue
            rd = int(ram[ADDR_OBJ_REM_DISTANCE + slot])
            ux, uy = _dir_vec(d)
            speed = BLUE_TELEPORT_SPEED if rd else BLUE_WALK_SPEED
            vx, vy = ux * speed, uy * speed
            if bodies or blue_bodies:
                # Walks 0.5 px/f: good for ~16 frames, then it may turn.
                fc.movers.append(Mover(x, y, vx, vy, 0, min(horizon, 16), mid_x, slot, "body"))
            fc.shooters.append(BlueShooter(x, y, vx, vy, d, rd, slot))
            continue
        if t == RED_WIZZROBE:
            if int(ram[ADDR_OBJ_HP + slot]) == 0:
                continue
            # Visible (solid or fading) while the countdown stays >= $40.
            if bodies and st >= RED_VISIBLE_STATE:
                fc.movers.append(
                    Mover(x, y, 0.0, 0.0, 0, min(horizon, st - RED_VISIBLE_STATE), mid_x, slot, "body")
                )
            if RED_FIRE_STATE <= st <= 0xFE and d in _DIR_BITS:
                t_fire = st - RED_FIRE_STATE
                if t_fire <= horizon:
                    ux, uy = _DIR_BITS[d]
                    fc.movers.append(
                        Mover(
                            x, y, ux * SHOT_SPEED, uy * SHOT_SPEED, t_fire, horizon,
                            species_of(RED_MAGIC).mid_offset_x, slot, "shot",
                        )
                    )
            continue
        if not bodies:
            continue
        if int(ram[ADDR_OBJ_HP + slot]) == 0 and species_of(t).hp not in (0, None):
            continue
        if species_of(t).contact_hearts <= 0:
            continue
        # Any other body: where it is, for a short horizon only.
        ux, uy = _DIR_BITS.get(d, (0, 0))
        v = species_of(t).px_per_frame(st)
        fc.movers.append(
            Mover(x, y, ux * v, uy * v, 0, min(horizon, body_horizon), mid_x, slot, "body")
        )
    return fc


def _collides(lx: float, ly: float, mx: float, my: float, mid_x: int) -> bool:
    return abs((lx + 8) - (mx + mid_x)) < CONTACT and abs((ly + 8) - (my + 8)) < CONTACT


INTERIOR_X = (32, 208)
INTERIOR_Y = (93, 189)


def _interior(x: float, y: float) -> bool:
    return INTERIOR_X[0] <= x <= INTERIOR_X[1] and INTERIOR_Y[0] <= y <= INTERIOR_Y[1]


class LinkModel:
    """Link on the dungeon turn lattice: 1.5 px/f, turns only on grid lines."""

    def __init__(
        self,
        nodes: frozenset[tuple[int, int]] | None,
        start: tuple[float, float] | None = None,
    ) -> None:
        self.nodes = nodes
        # A dodge must not walk Link out of the room: door lanes count as
        # floor only while he already stands in one (L9 0x05 dodged east
        # through its bomb hole into 0x06 on 2 of 4 offsets).
        self.in_room = start is None or _interior(start[0], start[1])

    def _free(self, x: int, y: int) -> bool:
        if self.nodes is not None and (x, y) not in self.nodes:
            return False
        return not self.in_room or _interior(x, y)

    def step(self, x: float, y: float, facing: str, press: str | None) -> tuple[float, float, str]:
        if press is None:
            return x, y, facing
        # Across the current axis off the grid: slide to the NEAREST grid line
        # first, ties going the way Link faces (measured on L9 0x14: y=152
        # backs up to 149, y=154 runs on to 157, y=145 facing up runs to 141).
        move = press
        if DIRS[press][0] != 0 and (round(y) - 5) % 8 != 0:
            off = (round(y) - 5) % 8
            move = "UP" if off < 4 or (off == 4 and facing == "UP") else "DOWN"
        elif DIRS[press][1] != 0 and round(x) % 8 != 0:
            off = round(x) % 8
            move = "LEFT" if off < 4 or (off == 4 and facing == "LEFT") else "RIGHT"
        dx, dy = DIRS[move]
        nx, ny = x + dx * LINK_SPEED, y + dy * LINK_SPEED
        if self.in_room:
            # A fractional step can overshoot a legal boundary node before
            # the next-node floor check runs (e.g. x=207 -> 208.5).
            nx = max(INTERIOR_X[0], min(nx, INTERIOR_X[1]))
            ny = max(INTERIOR_Y[0], min(ny, INTERIOR_Y[1]))
        if move != press:
            # The slide stops on the grid line (then the press turns him).
            if dy:
                line = (np.floor((y - 5) / 8.0) + (1 if dy > 0 else 0)) * 8 + 5
                ny = min(ny, line) if dy > 0 else max(ny, line)
            else:
                line = (np.floor(x / 8.0) + (1 if dx > 0 else 0)) * 8
                nx = min(nx, line) if dx > 0 else max(nx, line)
            return nx, ny, move
        # Stop short of a grid node that is not floor.
        if dx:
            gy = int(round(y))
            if (gy - 5) % 8 == 0:
                nxt = (np.floor(x / 8.0) + 1) * 8 if dx > 0 else (np.ceil(x / 8.0) - 1) * 8
                if not self._free(int(nxt), gy):
                    nx = min(nx, max(x, np.floor(x / 8.0) * 8)) if dx > 0 else max(nx, min(x, np.ceil(x / 8.0) * 8))
        if dy:
            gx = int(round(x))
            if gx % 8 == 0:
                ry = y - 5
                nxt = (np.floor(ry / 8.0) + 1) * 8 + 5 if dy > 0 else (np.ceil(ry / 8.0) - 1) * 8 + 5
                if not self._free(gx, int(nxt)):
                    ny = min(ny, max(y, np.floor(ry / 8.0) * 8 + 5)) if dy > 0 else max(ny, min(y, np.ceil(ry / 8.0) * 8 + 5))
        return nx, ny, press


@dataclass
class PlanResult:
    first_hit: int | None
    clearance: float
    end: tuple[float, float]


def simulate(
    forecast: Forecast,
    model: LinkModel,
    start: tuple[float, float],
    facing: str,
    plan: list[str | None],
) -> PlanResult:
    """First predicted hit frame (None = safe) holding ``plan`` from now."""
    x, y = start
    spawned: list[Mover] = []
    clearance = 1e9
    for t in range(1, len(plan) + 1):
        x, y, facing = model.step(x, y, facing, plan[t - 1])
        if (forecast.frame_counter + t) % BLUE_SHOT_PERIOD == 0:
            for s in forecast.shooters:
                if s.teleport_left > t:
                    continue
                wx, wy = s.x + s.vx * t, s.y + s.vy * t
                ux, uy = _dir_vec(s.dir_bits)
                same_row = (int(wy) & 0xF0) == (int(y) & 0xF0)
                same_col = int(round(x)) == (int(wx) & 0xF0)
                toward_x = (ux > 0 and wx < x) or (ux < 0 and wx >= x)
                toward_y = (uy > 0 and wy < y) or (uy < 0 and wy >= y)
                # Same square row decides alone; the column test runs only off it.
                if same_row:
                    if uy == 0 and toward_x:
                        spawned.append(Mover(wx, wy, ux * SHOT_SPEED, 0.0, t, t + 64, 8, s.slot, "shot"))
                elif same_col and ux == 0 and toward_y:
                    spawned.append(Mover(wx, wy, 0.0, uy * SHOT_SPEED, t, t + 64, 8, s.slot, "shot"))
        if t <= forecast.link_invincible:
            continue
        for m in (*forecast.movers, *spawned):
            p = m.at(t)
            if p is None:
                continue
            if _collides(x, y, p[0], p[1], m.mid_x):
                return PlanResult(t, 0.0, (x, y))
            clearance = min(clearance, max(abs(x - p[0]), abs(y - p[1])))
    return PlanResult(None, clearance, (x, y))


# A frame the inner controller already checked against the ROM itself (a
# savestate rollout, ``zelda_i.rollout``) carries this in its reason. The
# guard's straight-line and orbit model cannot improve on that answer, and
# replacing the frame would fire or walk into the hit the rollout avoided.
ROM_CHECKED = "rom_checked:"


def _pressed_dir(action: list[int]) -> str | None:
    names = ("B", "", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
    for i, name in enumerate(names):
        if action[i] and name in DIRS:
            return name
    return None


def _presses_a(action: list[int]) -> bool:
    return bool(action[8])


SWING_PIN = 13
# The blade is out on frames 4-11 of the 13-frame pin; a body it meets then
# is hit (a Wizzrobe reverses, attr $80) and no longer walks into Link.
BLADE_OUT = 5


def _cut(forecast: Forecast, snap: ZeldaSnapshot) -> Forecast:
    """The forecast with every body the coming swing will cut removed."""
    from zelda_i.combat import in_sword_hitbox

    keep = []
    for m in forecast.movers:
        if m.kind == "body":
            p = m.at(BLADE_OUT)
            if p is not None and in_sword_hitbox(
                int(snap.link_x), int(snap.link_y), int(snap.facing), int(p[0]), int(p[1])
            ):
                continue
        keep.append(m)
    return Forecast(keep, forecast.shooters, forecast.frame_counter, forecast.link_invincible)


_DEBUG = bool(os.environ.get("SHOT_GUARD_DEBUG"))
HOLDS = (4, 8, 12, 16, 24)


@dataclass
class ShotGuard:
    """Swap the inner press for a safe one when the RAM predicts a hit.

    ``active_types`` gates the (Python-costly) search to rooms that hold a
    Wizzrobe or a magic shot; everything else passes through untouched.
    """

    horizon: int = 32
    bodies: bool = False
    blue_bodies: bool = True
    # The inner press is only replaced for a hit this close: a controller
    # does not hold one direction for the whole horizon.
    trigger: int = 20
    overrides: int = 0
    frames: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    _nodes_room: tuple[int, int] | None = field(default=None, repr=False)
    _nodes: frozenset[tuple[int, int]] | None = field(default=None, repr=False)
    _last: str | None = field(default=None, repr=False)
    _tick: int = field(default=0, repr=False)
    _eye_hist: dict[int, deque] = field(default_factory=dict, repr=False)
    _body_hist: deque = field(default_factory=lambda: deque(maxlen=5), repr=False)

    def _model(self, ram: np.ndarray, snap: ZeldaSnapshot) -> LinkModel:
        key = (int(snap.level), int(snap.screen))
        if key != self._nodes_room:
            self._nodes_room = key
            self._nodes = ow_walkable_nodes(ram, overworld=False) if has_room_tile_map(ram) else None
        return LinkModel(self._nodes, (float(snap.link_x), float(snap.link_y)))

    @staticmethod
    def relevant(ram: np.ndarray) -> bool:
        types = ram[ADDR_OBJ_TYPE + 1 : ADDR_OBJ_TYPE + 12]
        return bool(
            np.isin(
                types,
                (BLUE_WIZZROBE, RED_WIZZROBE, BLUE_MAGIC, RED_MAGIC, *PATRA_BODIES, *PATRA_EYES),
            ).any()
        )

    def _patra_movers(self, ram: np.ndarray) -> list[Any]:
        """The Patra body (linear) and each eye (polar about the body)."""
        body_slot = next(
            (s for s in range(1, 12) if int(ram[ADDR_OBJ_TYPE + s]) in PATRA_BODIES), None
        )
        self._tick += 1
        if body_slot is None:
            self._eye_hist.clear()
            self._body_hist.clear()
            return []
        bx, by = float(ram[ADDR_OBJ_X + body_slot]), float(ram[ADDR_OBJ_Y + body_slot])
        self._body_hist.append((self._tick, bx, by))
        bvx = bvy = 0.0
        if len(self._body_hist) >= 4:
            f0, x0, y0 = self._body_hist[0]
            dt = max(1, self._tick - f0)
            bvx, bvy = (bx - x0) / dt, (by - y0) / dt
        btype = int(ram[ADDR_OBJ_TYPE + body_slot])
        out: list[Any] = [
            Mover(bx, by, bvx, bvy, 0, 16, species_of(btype).mid_offset_x, body_slot, "body")
        ]
        live = set()
        for s in range(1, 12):
            t = int(ram[ADDR_OBJ_TYPE + s])
            if t not in PATRA_EYES or int(ram[ADDR_OBJ_METASTATE + s]):
                continue
            live.add(s)
            rx = ((int(ram[ADDR_OBJ_X + s]) - int(bx) + 128) & 0xFF) - 128
            ry = ((int(ram[ADDR_OBJ_Y + s]) - int(by) + 128) & 0xFF) - 128
            r, th = math.hypot(rx, ry), math.atan2(ry, rx)
            hist = self._eye_hist.setdefault(s, deque(maxlen=3))
            if hist and hist[-1][0] != self._tick - 1:
                hist.clear()
            hist.append((self._tick, r, th))
            dr = dth = 0.0
            if len(hist) >= 2:
                f0, r0, th0 = hist[0]
                dt = max(1, self._tick - f0)
                dr = (r - r0) / dt
                dth = ((th - th0 + math.pi) % (2 * math.pi) - math.pi) / dt
            out.append(
                OrbitMover(bx, by, bvx, bvy, r, dr, th, dth, 0, 12, species_of(t).mid_offset_x, s)
            )
        for s in set(self._eye_hist) - live:
            del self._eye_hist[s]
        return out

    def filter(self, snap: ZeldaSnapshot, ram: np.ndarray, inner: FrameAction) -> FrameAction:
        self.frames += 1
        if snap.mode != PLAY_MODE or snap.transitioning or not snap.level or not self.relevant(ram):
            self._last = None
            return inner
        if ROM_CHECKED in inner.reason:
            self._last = None
            return inner
        patra = self._patra_movers(ram)
        # Mid-swing Link cannot move: a press changes nothing until the pin ends.
        if int(ram[ADDR_OBJ_STATE]) & 0xF0:
            if _DEBUG:
                print(f"  guard f{self.frames} skip: link state {int(ram[ADDR_OBJ_STATE]):02x}")
            return inner
        forecast = read_forecast(
            ram, self.horizon, bodies=self.bodies, blue_bodies=self.blue_bodies
        )
        forecast.movers.extend(patra)
        if forecast.empty:
            self._last = None
            return inner
        model = self._model(ram, snap)
        start = (float(snap.link_x), float(snap.link_y))
        facing = _FACING_NAME.get(int(snap.facing), "UP")
        press = _pressed_dir(inner.action)
        if _presses_a(inner.action):
            inner_plan: list[str | None] = [None] * max(SWING_PIN, self.horizon)
            base = simulate(_cut(forecast, snap), model, start, facing, inner_plan)
        else:
            inner_plan = [press] * self.horizon
            base = simulate(forecast, model, start, facing, inner_plan)
        # Hysteresis: once dodging, hand back only when the inner plan is safe
        # for the whole horizon, not just past the trigger (else the two flip
        # every frame and Link stands in the lane: L9 0x05, x 199<->200).
        limit = self.horizon if self._last is not None else self.trigger
        if _DEBUG:
            print(f"  guard f{self.frames} L{start} press={press} base={base.first_hit} last={self._last} st0={int(ram[ADDR_OBJ_STATE]):02x} movers={[(m.kind, m.slot, m.x, m.y, m.vx, m.vy, m.t0) for m in forecast.movers]}")
        if base.first_hit is None or base.first_hit > limit:
            self._last = None
            return inner
        best: tuple[tuple, str | None, int] | None = None
        for d in (None, *DIRS):
            if d is not None:
                # A press the model says goes nowhere is a stand at best -- and
                # at a door it is the ROM walking Link out of the room.
                nx, ny, _ = model.step(start[0], start[1], facing, d)
                if (nx, ny) == start:
                    continue
            for k in ((0,) if d is None else HOLDS):
                plan = [d] * k + [None] * (self.horizon - k)
                res = simulate(forecast, model, start, facing, plan)
                safe = res.first_hit is None
                score = (
                    safe,
                    res.first_hit or self.horizon + 1,
                    d == self._last,
                    d == press,
                    min(res.clearance, 48.0),
                    -k,
                )
                if _DEBUG:
                    print(f"    alt {d} k={k} hit={res.first_hit} clr={res.clearance:.1f} end={res.end}")
                if best is None or score > best[0]:
                    best = (score, d, k)
        assert best is not None
        score, d, _k = best
        if not score[0] and score[1] <= base.first_hit:
            # Nothing buys a frame: keep the inner controller's plan.
            return inner
        self.overrides += 1
        self._last = d if d is not None else "STAND"
        reason = f"guard_{(d or 'stand').lower()}"
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return FrameAction(nes_action(d) if d else nes_idle_action(), reason)

    def report(self) -> dict[str, Any]:
        return {"frames": self.frames, "overrides": self.overrides, "reasons": dict(self.reasons)}


@dataclass
class GuardedController:
    """Wrap any stage controller with a :class:`ShotGuard`."""

    inner: Any
    guard: ShotGuard = field(default_factory=ShotGuard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def bind_env(self, env: Any) -> None:
        self._env = env
        bind = getattr(self.inner, "bind_env", None)
        if callable(bind):
            bind(env)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        act = self.inner.step(snap)
        if self.inner.success or self.inner.failed:
            return act
        env = self.__dict__.get("_env")
        if env is None:
            from zelda_i.walk import live_env

            env = live_env.current()
        if env is None:
            return act
        return self.guard.filter(snap, env.get_ram(), act)

    def report(self) -> dict[str, Any]:
        rep = self.inner.report() if callable(getattr(self.inner, "report", None)) else {}
        if isinstance(rep, dict):
            rep = dict(rep)
            rep["shot_guard"] = self.guard.report()
        return rep
