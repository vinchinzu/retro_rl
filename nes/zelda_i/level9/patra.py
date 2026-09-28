"""Final Patra combat policy for Level 9 room ``0x52``.

Live fceumm observations from the disclosed full-loadout recon fixture:

- body type ``0x47`` starts in slot 1 with HP ``0xB0``;
- eight orbiting eyes use type ``0x25`` and HP ``0x60``;
- Magical Sword hits move an eye ``0x60 -> 0x20 -> dead``;
- after the eyes are gone, body hits move ``0xB0 -> 0x70 -> 0x30 -> dead``;
- the game raises north-door bit ``0x08`` after the body disappears.

This controller performs no RAM writes.  The start checkpoint remains an
explicit fixture because its full inventory and room-loader setup are composed.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import room_step

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import direction_to_facing, in_sword_hitbox
from zelda_i.level9.ganon import LEVEL9, ROOM_BEFORE_GANON, hazard_dodge_dir
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

OBJ_PATRA = 0x47
OBJ_PATRA_EYE = 0x25
# The first Patra (L9 0x16, 0x27) is type $48 with $26 eyes: a 24 px small
# circle and a wider one, $60 of angle a frame against the $25 eyes' $70
# (Z_04.asm ``UpdatePatraChild``). Every helper below reads both kinds.
OBJ_PATRA_2 = 0x48
OBJ_PATRA_EYE_2 = 0x26
PATRA_BODY_TYPES = (OBJ_PATRA, OBJ_PATRA_2)
PATRA_EYE_TYPES = (OBJ_PATRA_EYE, OBJ_PATRA_EYE_2)
PATRA_BODY_HP_START = 0xB0
PATRA_EYE_HP_START = 0x60
PATRA_EYE_COUNT = 8
NORTH_DOOR = 0x08

# Stand this far south of the body; the stand clamps to the bottom row
# (y=173), outside the eyes' orbit, and the full-health beam still reaches.
# 30 px stood inside the orbit: over 12 RNG offsets from Blue Ring power-on
# 9 pins, 0x52 went 24.0h -> 6.8h (2351f -> 1761f) and 0x61 23h -> 2h.
PATRA_STAND_DY = 84
# 8, not 12: over 12 offsets x 3 pins, 0x52 1690f/13.8h (the clamped south
# stand, cooldown 12) -> 1650f/1.7h; 0x61 2492f/7.0h -> 2135f/3.0h
# (x<=192) or 2483f/1.6h (x<=208). Cooldown 6 was worse.
PATRA_ATTACK_COOLDOWN = 8
# The body roams, and late in a fight it sits in the bottom rows, where the
# clamped south stand put Link inside it: 22 of 22 hits on the power-on 14
# 0x61 pin. Stand on whichever side of the body still leaves this much room
# (the side Link already faces first), and fire once on its lane: the beam
# flies the room, so the exact stand pixel does not matter.
PATRA_MIN_CLEAR = 56
PATRA_LANE_HALF = 8
# x<=192: 0x52's east column holds the stairs down to cellar 0x77 (208,144),
# and a stand there walked Link out of the fight (6 of 36 runs). 0x61 has
# no stairs there and passes the full box.
PATRA_ROOM = (32, 192, 93, 173)
PATRA_ROOM_FULL = (32, 208, 93, 173)
_PATRA_SIDES = (("UP", 0, 1), ("DOWN", 0, -1), ("LEFT", 1, 0), ("RIGHT", -1, 0))
PATRA_MAX_FRAMES = 6000


def in_final_patra_room(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_BEFORE_GANON
    )


def patra_body(snap: ZeldaSnapshot) -> ZeldaObject | None:
    return next(
        (
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and obj.type_id in PATRA_BODY_TYPES
        ),
        None,
    )


def patra_eyes(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id in PATRA_EYE_TYPES
    )


def final_patra_live(snap: ZeldaSnapshot) -> bool:
    return in_final_patra_room(snap) and patra_body(snap) is not None


def final_patra_north_door_earned(snap: ZeldaSnapshot) -> bool:
    return (
        in_final_patra_room(snap)
        and patra_body(snap) is None
        and not patra_eyes(snap)
        and bool(snap.cur_opened_doors & NORTH_DOOR)
    )


def _lane_offsets(xy: tuple[int, int], body: ZeldaObject, facing: str) -> tuple[int, int]:
    """(along, perp) of ``xy`` on ``facing``'s lane of the body."""
    dx = int(body.x) - int(xy[0])
    dy = int(body.y) - int(xy[1])
    return {"UP": (-dy, dx), "DOWN": (dy, dx), "LEFT": (-dx, dy), "RIGHT": (dx, dy)}[facing]


def _lane_node(nodes, body: ZeldaObject, facing: str, goal: tuple[int, int]) -> tuple[int, int] | None:
    """The walkable node on ``facing``'s lane, out of the orbit, nearest ``goal``."""
    best = None
    for node in nodes:
        along, perp = _lane_offsets(node, body, facing)
        if along < PATRA_MIN_CLEAR or abs(perp) > PATRA_LANE_HALF:
            continue
        cost = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        if best is None or cost < best[0]:
            best = (cost, node)
    return None if best is None else best[1]


def _patra_stands(
    snap: ZeldaSnapshot,
    body: ZeldaObject,
    dist: int,
    room,
    *,
    prefer: str | None = None,
    nodes=None,
):
    """``(cost, (x, y), facing)`` per side the room leaves ``PATRA_MIN_CLEAR``.

    ``prefer`` (the side chosen last frame) takes the bonus the held facing
    gets without it: walking to one side's stand faces Link at the other
    side, and the facing bonus flipped the pick every frame (run 23's body
    phase: x 103<->104 for 425 frames). ``nodes`` snaps each stand to a
    walkable lane node: 0x52's east rows are blocks, and a clamped stand in
    them pressed UP into a wall for 200 frames.
    """
    xlo, xhi, ylo, yhi = room
    bx, by = int(body.x), int(body.y)
    lx, ly = int(snap.link_x), int(snap.link_y)
    held = direction_to_facing(prefer) if prefer else int(snap.facing)
    out = []
    for facing, sx, sy in _PATRA_SIDES:
        gx = max(xlo, min(xhi, bx + sx * dist))
        gy = max(ylo, min(yhi, by + sy * dist))
        if nodes is not None:
            node = _lane_node(nodes, body, facing, (gx, gy))
            if node is None:
                continue
            gx, gy = node
        if max(abs(gx - bx), abs(gy - by)) < PATRA_MIN_CLEAR:
            continue
        cost = abs(gx - lx) + abs(gy - ly) - (64 if direction_to_facing(facing) == held else 0)
        out.append((cost, (gx, gy), facing))
    return sorted(out)


def _on_lane(snap: ZeldaSnapshot, body: ZeldaObject, facing: str) -> bool:
    """On the body's row/column, on ``facing``'s side and out of the orbit."""
    along, perp = _lane_offsets((int(snap.link_x), int(snap.link_y)), body, facing)
    return along >= PATRA_MIN_CLEAR and abs(perp) <= PATRA_LANE_HALF


def patra_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
    stand_dy: int = PATRA_STAND_DY,
    room: tuple[int, int, int, int] = PATRA_ROOM,
    prefer: str | None = None,
    nodes=None,
) -> tuple[list[int], str, int]:
    """One frame: on a lane of the body out of the orbit, face it and pulse A.

    The orbiting eyes cross the lane; the full-health beam carries the A
    press the length of the room. Cooldown dodges nearby eyes or stands.
    ``prefer``/``nodes``: see ``_patra_stands``.
    """
    body = patra_body(snap)
    next_cd = max(0, cooldown - 1)
    if body is None:
        return nes_idle_action(), "wait_north_door", next_cd

    stands = _patra_stands(snap, body, int(stand_dy), room, prefer=prefer, nodes=nodes)
    if not stands:
        # Nowhere in the room clears the orbit: step away from the body.
        return nes_action(hazard_dodge_dir(snap, (body,), thr=255) or "DOWN"), "patra_boxed", next_cd
    _, goal, facing = stands[0]
    if not _on_lane(snap, body, facing):
        # ROM lattice to the stand; the greedy axis step pushed into the 0x61
        # blocks at (144,165) for 1041 frames.
        direction = room_step(snap, goal, tol=4)
        if direction is not None:
            return nes_action(direction), f"align_{facing.lower()}", next_cd

    dodge = hazard_dodge_dir(snap, patra_eyes(snap))
    if cooldown > 0:
        if dodge is not None:
            return nes_action(dodge), "attack_dodge", next_cd
        return nes_idle_action(), "cooldown_stand", next_cd
    if int(snap.facing) != direction_to_facing(facing):
        return nes_action(facing), f"face_{facing.lower()}", 0
    return nes_action("A"), f"sword_pulse_{facing.lower()}", PATRA_ATTACK_COOLDOWN


# 0x61 has a walkable east lane beside the eye orbit.  On the last-heart pin
# Link enters below full health, so the sword beam never appears.  Holding this
# turn-lattice node lands ordinary blade hits as the eyes pass; after they die,
# the body needs a short chase because it can roam away from the lane.
PATRA_61_MELEE_STAND = (192, 149)
# The $25 orbit reaches roughly 52 px east/west of the roaming body.
# A blade lane 64..72 px away still intersects it, while the old 56 px
# approach gave the guard no escape during one 13-frame sword pin.
PATRA_MELEE_CLEAR = 64
PATRA_MELEE_SLACK = 8


def patra_melee_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
    stand: tuple[int, int] = PATRA_61_MELEE_STAND,
    facing: str = "LEFT",
    room: tuple[int, int, int, int] = PATRA_ROOM_FULL,
) -> tuple[list[int], str, int]:
    """One no-beam Patra frame, using a room's melee lane and sword input."""
    eyes = tuple(eye for eye in patra_eyes(snap) if eye.hp > 0)
    next_cd = max(0, cooldown - 1)
    if eyes:
        x, y = int(snap.link_x), int(snap.link_y)
        if max(abs(x - stand[0]), abs(y - stand[1])) > 3:
            direction = room_step(snap, stand, tol=2)
            if direction is not None:
                return nes_action(direction), "melee_approach", next_cd
        if int(snap.facing) != direction_to_facing(facing):
            return nes_action(facing), "melee_face", next_cd
        if cooldown > 0:
            return nes_idle_action(), "melee_cooldown", next_cd
        return nes_action("A"), "melee_swing", PATRA_ATTACK_COOLDOWN

    body = patra_body(snap)
    if body is None:
        return nes_idle_action(), "melee_wait_door", next_cd
    lx, ly = int(snap.link_x), int(snap.link_y)
    dx, dy = int(body.x) - lx, int(body.y) - ly
    if abs(dx) >= abs(dy):
        direction = "RIGHT" if dx > 0 else "LEFT"
    else:
        direction = "DOWN" if dy > 0 else "UP"
    if in_sword_hitbox(lx, ly, direction, body.x, body.y):
        if int(snap.facing) != direction_to_facing(direction):
            return nes_action(direction), "melee_body_face", next_cd
        if cooldown > 0:
            return nes_idle_action(), "melee_body_cooldown", next_cd
        return nes_action("A"), "melee_body_swing", PATRA_ATTACK_COOLDOWN
    xlo, xhi, ylo, yhi = room
    goal = (max(xlo, min(xhi, int(body.x))), max(ylo, min(yhi, int(body.y))))
    step = room_step(snap, goal, tol=12)
    return nes_action(step or direction), "melee_body_chase", next_cd


@dataclass
class PatraMelee:
    """Follow a reachable blade lane as the body carries its eyes around.

    A fixed east stand cannot reach eyes while the body roams west. A
    committed side follows the orbit without re-picking when walking turns
    Link away from it. The ordinary sword, including below full health,
    reaches the near edge of the $25 eye orbit from 64..72 px off the body.
    """

    cooldown: int = 0
    facing: str | None = None

    def step(self, snap: ZeldaSnapshot) -> tuple[list[int], str]:
        body = patra_body(snap)
        eyes = tuple(eye for eye in patra_eyes(snap) if eye.hp > 0)
        if body is None or not eyes:
            action, reason, self.cooldown = patra_melee_action(snap, cooldown=self.cooldown)
            return action, reason
        from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
        from zelda_i.walk import live_env

        env = live_env.current()
        ram = env.get_ram() if env is not None else None
        nodes = ow_walkable_nodes(ram, overworld=False) if ram is not None and has_room_tile_map(ram) else None
        if nodes is not None:
            nodes = frozenset(
                n for n in nodes
                if max(abs(n[0] - body.x), abs(n[1] - body.y)) >= PATRA_MELEE_CLEAR
            )
        stands = _patra_stands(
            snap, body, PATRA_MELEE_CLEAR + PATRA_MELEE_SLACK // 2,
            PATRA_ROOM_FULL, prefer=self.facing, nodes=nodes,
        )
        stands = [s for s in stands if _lane_offsets(s[1], body, s[2])[0] >= PATRA_MELEE_CLEAR]
        self.cooldown = max(0, self.cooldown - 1)
        if not stands:
            return nes_idle_action(), "melee_no_lane"
        _, goal, self.facing = stands[0]
        along, perp = _lane_offsets((int(snap.link_x), int(snap.link_y)), body, self.facing)
        if not (
            PATRA_MELEE_CLEAR <= along <= PATRA_MELEE_CLEAR + PATRA_MELEE_SLACK
            and abs(perp) <= PATRA_LANE_HALF
        ):
            step = room_step(snap, goal, tol=2)
            if step is not None:
                return nes_action(step), "melee_follow"
        if int(snap.facing) != direction_to_facing(self.facing):
            return nes_action(self.facing), "melee_face"
        if self.cooldown:
            return nes_idle_action(), "melee_cooldown"
        self.cooldown = PATRA_ATTACK_COOLDOWN
        return nes_action("A"), "melee_swing"


# --- Eye aim (rr-e59v) ------------------------------------------------------
# Measured on run 23's final Patra (``BlueRingFull23_level9_final_patra``,
# 11,006 frames, ~800 A pulses for 24 eye hits under the lane stand):
# - the shot (slot $0E) appears BEAM_LEAD frames after the A press (blade
#   state 1 for 4 frames, 2 for 8, then the shot; Link is held meanwhile),
#   BEAM_SPAWN px ahead of Link, and flies BEAM_SPEED px/f; leaving
#   BEAM_BOUNDS it bursts for 22 frames, and only one shot lives at a time,
#   so a miss holds the next one ~60-80 frames;
# - it hits an eye when eye - shot lies in BEAM_HIT_DX x BEAM_HIT_DY (the
#   offsets of all 20 beam hits on the frame before the eye's HP fell);
# - it flies through the body while an eye lives (no mid-room burst);
# - the shot does 0x20: three per eye, six for the body.
# Each eye laps a point EYE_PERIOD frames a lap, and that point drifts off
# the body (-14..+158 px in y over the fight): the last eye spent 94% of its
# 5,674 frames below the room, and the lane stand fired into an empty room.
EYE_PERIOD = 73
EYE_FIT_FRAMES = 24
BEAM_LEAD = 13
BEAM_SPAWN = 19
BEAM_SPEED = 3
BEAM_BOUNDS = (22, 219, 79, 180)
BEAM_HIT_DX = (-13, 21)
BEAM_HIT_DY = (-17, 10)
# The fit is ~3 px off at 8 frames and ~11 px at 32: shrink the box by this
# and only trust a hit inside AIM_HORIZON frames.
AIM_MARGIN = 3
AIM_HORIZON = 40
# A press, then no A until the shot is out: the slot reads empty for
# BEAM_LEAD frames, and an A inside the swing is dropped.
AIM_COOLDOWN = BEAM_LEAD + 1
AIM_REPLAN = 8
# A stand the lap passes within this (Chebyshev) is contact; the body roams.
AIM_EYE_CLEAR = 20
AIM_BODY_CLEAR = 32
# Keep the stand unless a new one is this many frames cheaper (no flip-flop).
AIM_HYSTERESIS = 24
# Walk onto the stand node, then hold it until Link is this far off: the turn
# press moves him a pixel, and a 4 px arrive tolerance flipped align/face
# every frame on run 23's pin (x 68<->69 for 270 frames).
AIM_STAND_SLACK = 8
# The lane stand is faster while the eyes ring the body (its blind pulse
# crosses many eyes): over 8 pins x 6 offsets the aim cost ~430 frames more
# there. Run 23's lap drifted 23 -> 192 px off the body in ~22 px steps;
# healthy pins stayed <= 24 (the early ellipse). Past this, aim, latched.
AIM_DRIFT = 36
# A fit worse than this RMS (px) is the eyes' spawn, not a lap; an eye
# still parked at its spawn point (48,173) fits exactly with no swing.
AIM_FIT_RMS = 6.0
AIM_FIT_SWING = 16.0
_DIR_VEC = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
_FACING_NAME = {direction_to_facing(name): name for name in _DIR_VEC}


def _s8(value: int) -> int:
    return ((int(value) + 128) & 0xFF) - 128


@dataclass
class PatraEyeModel:
    """Each eye's lap about the body, least-squares fit on its last frames.

    Relative to the body so the body's walk does not smear the fit; y is
    unwrapped across the 8-bit edge (the drifted lap crosses it).
    """

    frame: int = 0
    body: deque = field(default_factory=lambda: deque(maxlen=EYE_FIT_FRAMES))
    tracks: dict[int, deque] = field(default_factory=dict)

    def observe(self, snap: ZeldaSnapshot) -> None:
        self.frame += 1
        body = patra_body(snap)
        if body is None:
            self.tracks.clear()
            self.body.clear()
            return
        bx, by = int(body.x), int(body.y)
        self.body.append((self.frame, bx, by))
        live = set()
        for eye in patra_eyes(snap):
            live.add(eye.slot)
            rx, ry = _s8(int(eye.x) - bx), _s8(int(eye.y) - by)
            track = self.tracks.get(eye.slot)
            if track and track[-1][0] == self.frame - 1:
                _, px, py = track[-1]
                rx, ry = px + _s8(rx - px), py + _s8(ry - py)
            else:
                track = self.tracks[eye.slot] = deque(maxlen=EYE_FIT_FRAMES)
            track.append((self.frame, rx, ry))
        for slot in set(self.tracks) - live:
            del self.tracks[slot]

    def ready(self) -> bool:
        return any(len(t) >= EYE_FIT_FRAMES for t in self.tracks.values())

    def _fit(self, track) -> tuple[np.ndarray, float]:
        w = 2.0 * np.pi / EYE_PERIOD
        t = np.array([f - self.frame for f, _, _ in track], dtype=float)
        basis = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
        rel = np.array([(x, y) for _, x, y in track], dtype=float)
        coef = np.linalg.lstsq(basis, rel, rcond=None)[0]
        rms = float(np.sqrt(np.mean((basis @ coef - rel) ** 2)))
        return coef, rms

    def drift(self) -> int:
        """Largest fitted lap-center offset from the body (px), clean fits only."""
        best = 0
        for track in self.tracks.values():
            if len(track) < EYE_FIT_FRAMES:
                continue
            coef, rms = self._fit(track)
            swing = float(np.hypot(coef[1], coef[2]).max())
            if rms <= AIM_FIT_RMS and swing >= AIM_FIT_SWING:
                best = max(best, int(np.abs(coef[0]).max()))
        return best

    def paths(self, ks: np.ndarray, *, body_motion: bool = True) -> dict[int, np.ndarray]:
        """``{slot: (len(ks), 2)}`` screen positions ``k`` frames ahead (mod 256)."""
        if not self.body:
            return {}
        w = 2.0 * np.pi / EYE_PERIOD
        _, bx, by = self.body[-1]
        vx = vy = 0.0
        if body_motion and len(self.body) > 8:
            _, ox, oy = self.body[-9]
            vx, vy = (bx - ox) / 8.0, (by - oy) / 8.0
        ks = np.asarray(ks, dtype=float)
        basis = np.column_stack([np.ones_like(ks), np.cos(w * ks), np.sin(w * ks)])
        out = {}
        for slot, track in self.tracks.items():
            if len(track) < EYE_FIT_FRAMES:
                continue
            coef, _ = self._fit(track)
            pos = basis @ coef + np.column_stack([bx + vx * ks, by + vy * ks])
            out[slot] = np.mod(pos, 256.0)
        return out

    def predict(self, k: int) -> dict[int, tuple[float, float]]:
        return {s: (float(p[0][0]), float(p[0][1])) for s, p in self.paths(np.array([k])).items()}


def _shot_track(link_xy: tuple[int, int], facing: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Shot positions per frame after the press, and whether it still flies."""
    ux, uy = _DIR_VEC[facing]
    k = np.arange(horizon + 1)
    dist = BEAM_SPAWN + BEAM_SPEED * (k - BEAM_LEAD)
    xy = np.column_stack([link_xy[0] + ux * dist, link_xy[1] + uy * dist]).astype(float)
    xlo, xhi, ylo, yhi = BEAM_BOUNDS
    inside = (xy[:, 0] >= xlo) & (xy[:, 0] <= xhi) & (xy[:, 1] >= ylo) & (xy[:, 1] <= yhi)
    flying = (k >= BEAM_LEAD) & np.minimum.accumulate(inside | (k < BEAM_LEAD))
    return xy, flying


def beam_hit_frame(link_xy: tuple[int, int], facing: str, paths) -> int | None:
    """First frame after an A press now whose shot meets a predicted eye."""
    best = None
    for path in paths.values():
        path = np.asarray(path, dtype=float)
        shot, flying = _shot_track(link_xy, facing, len(path) - 1)
        dx = path[:, 0] - shot[:, 0]
        dy = path[:, 1] - shot[:, 1]
        hit = (
            flying
            & (dx >= BEAM_HIT_DX[0] + AIM_MARGIN) & (dx <= BEAM_HIT_DX[1] - AIM_MARGIN)
            & (dy >= BEAM_HIT_DY[0] + AIM_MARGIN) & (dy <= BEAM_HIT_DY[1] - AIM_MARGIN)
        )
        if hit.any():
            k = int(np.argmax(hit))
            best = k if best is None else min(best, k)
    return best


def _lane_windows(nodes: np.ndarray, facing: str, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per node: lap points its shot lane can hit, and the nearest one's flight."""
    ux, uy = _DIR_VEC[facing]
    xlo, xhi, ylo, yhi = BEAM_BOUNDS
    m = AIM_MARGIN
    sx, sy = nodes[:, 0:1], nodes[:, 1:2]
    px, py = pts[None, :, 0], pts[None, :, 1]
    if ux:
        lane = (py - sy >= BEAM_HIT_DY[0] + m) & (py - sy <= BEAM_HIT_DY[1] - m)
        near, far = (sx + BEAM_SPAWN, xhi) if ux > 0 else (xlo, sx - BEAM_SPAWN)
        reach = (px >= near + BEAM_HIT_DX[0] + m) & (px <= far + BEAM_HIT_DX[1] - m)
        along = np.abs(px - sx)
    else:
        lane = (px - sx >= BEAM_HIT_DX[0] + m) & (px - sx <= BEAM_HIT_DX[1] - m)
        near, far = (sy + BEAM_SPAWN, yhi) if uy > 0 else (ylo, sy - BEAM_SPAWN)
        reach = (py >= near + BEAM_HIT_DY[0] + m) & (py <= far + BEAM_HIT_DY[1] - m)
        along = np.abs(py - sy)
    ok = lane & reach
    flight = np.where(ok, along, np.inf).min(axis=1) / BEAM_SPEED
    return ok.sum(axis=1), flight


@dataclass
class PatraAim:
    """Fire only on a predicted eye hit; stand where the lap crosses a lane.

    The body-lane stand (``patra_action``) is right while the eyes ring the
    body: it plays the fight until the lap drifts ``AIM_DRIFT`` off the body,
    and after the eyes die. (Aiming at <= 2 eyes on healthy pins was no
    faster over 54 runs.)
    """

    model: PatraEyeModel = field(default_factory=PatraEyeModel)
    stand_dy: int = PATRA_STAND_DY
    room: tuple[int, int, int, int] = PATRA_ROOM
    cooldown: int = 0
    stand: tuple[tuple[int, int], str] | None = None
    replan_in: int = 0
    fires: int = 0
    last_fire_hit_frame: int | None = None
    arrived: bool = False
    drifted: bool = False
    lane_side: str | None = None
    walkable: frozenset[tuple[int, int]] | None = None

    def _walkable(self) -> frozenset[tuple[int, int]] | None:
        """The room's walkable turn-lattice nodes inside ``room`` (ROM tiles)."""
        if self.walkable is None:
            from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
            from zelda_i.walk import live_env

            env = live_env.current()
            if env is None or not has_room_tile_map(env.get_ram()):
                return None
            xlo, xhi, ylo, yhi = self.room
            self.walkable = frozenset(
                (x, y)
                for x, y in ow_walkable_nodes(env.get_ram(), overworld=False)
                if xlo <= x <= xhi and ylo <= y <= yhi
            )
        return self.walkable

    def _nodes(self) -> np.ndarray:
        walkable = self._walkable()
        if walkable:
            return np.array(sorted(walkable), dtype=float)
        xlo, xhi, ylo, yhi = self.room
        xs = range(xlo + (-xlo) % 8, xhi + 1, 8)
        ys = range(ylo + (5 - ylo) % 8, yhi + 1, 8)
        return np.array([(x, y) for y in ys for x in xs], dtype=float)

    def _lane_stand(self, snap: ZeldaSnapshot) -> tuple[list[int], str]:
        action, reason, self.cooldown = patra_action(
            snap, cooldown=self.cooldown, stand_dy=self.stand_dy, room=self.room,
            prefer=self.lane_side, nodes=self._walkable(),
        )
        side = reason.rsplit("_", 1)[-1].upper()
        if side in _DIR_VEC:
            self.lane_side = side
        return action, reason

    def choose_stand(self, snap: ZeldaSnapshot) -> tuple[tuple[int, int], str] | None:
        """Cheapest safe lattice node + facing whose lane the next lap crosses."""
        laps = self.model.paths(np.arange(EYE_PERIOD), body_motion=False)
        if not laps:
            return None
        pts = np.concatenate(list(laps.values()))
        nodes = self._nodes()
        gap = np.abs(nodes[:, None, :] - pts[None, :, :]).max(axis=2).min(axis=1)
        safe = gap >= AIM_EYE_CLEAR
        body = patra_body(snap)
        if body is not None:
            safe &= np.maximum(np.abs(nodes[:, 0] - int(body.x)), np.abs(nodes[:, 1] - int(body.y))) >= AIM_BODY_CLEAR
        lx, ly = int(snap.link_x), int(snap.link_y)
        travel = (np.abs(nodes[:, 0] - lx) + np.abs(nodes[:, 1] - ly)) / 1.5
        held = _FACING_NAME.get(int(snap.facing))
        best = None
        for facing in _DIR_VEC:
            count, flight = _lane_windows(nodes, facing, pts)
            cost = travel + flight / 2 - 3 * np.minimum(count, 12) - (4 if facing == held else 0)
            cost = np.where(safe & (count > 0), cost, np.inf)
            i = int(np.argmin(cost))
            if np.isfinite(cost[i]) and (best is None or cost[i] < best[0]):
                best = (float(cost[i]), (int(nodes[i][0]), int(nodes[i][1])), facing)
        if best is None:
            return None
        if self.stand is not None:
            (gx, gy), gf = self.stand
            idx = np.flatnonzero((nodes[:, 0] == gx) & (nodes[:, 1] == gy))
            if idx.size and safe[idx[0]]:
                count, flight = _lane_windows(nodes[idx], gf, pts)
                if count[0] > 0:
                    keep = float(travel[idx[0]] + flight[0] / 2 - 3 * min(int(count[0]), 12))
                    if keep <= best[0] + AIM_HYSTERESIS:
                        return self.stand
        return best[1], best[2]

    def step(self, snap: ZeldaSnapshot) -> tuple[list[int], str]:
        self.model.observe(snap)
        eyes = patra_eyes(snap)
        if eyes and not self.drifted and self.model.drift() > AIM_DRIFT:
            self.drifted = True
        if not eyes or not self.model.ready() or not self.drifted:
            return self._lane_stand(snap)
        dodge = hazard_dodge_dir(snap, eyes)
        if self.cooldown > 0:
            self.cooldown -= 1
            if dodge is not None:
                return nes_action(dodge), "aim_dodge"
            return nes_idle_action(), "aim_cooldown"
        held = _FACING_NAME.get(int(snap.facing))
        if held is not None and int(snap.sword_shot.state) == 0:
            paths = self.model.paths(np.arange(AIM_HORIZON + 1))
            k = beam_hit_frame((int(snap.link_x), int(snap.link_y)), held, paths)
            if k is not None:
                self.fires += 1
                self.last_fire_hit_frame = k
                self.cooldown = AIM_COOLDOWN
                return nes_action("A"), f"sword_pulse_{held.lower()}"
        self.replan_in -= 1
        if self.replan_in <= 0 or self.stand is None:
            stand = self.choose_stand(snap)
            if stand is None or self.stand is None or stand[0] != self.stand[0]:
                self.arrived = False
            self.stand = stand
            self.replan_in = AIM_REPLAN
        if self.stand is None:
            if dodge is not None:
                return nes_action(dodge), "aim_dodge"
            return nes_idle_action(), "aim_no_lane"
        goal, facing = self.stand
        off = max(abs(int(snap.link_x) - goal[0]), abs(int(snap.link_y) - goal[1]))
        if off > AIM_STAND_SLACK:
            self.arrived = False
        if not self.arrived:
            direction = room_step(snap, goal, tol=1)
            if direction is not None:
                return nes_action(direction), f"aim_align_{facing.lower()}"
            self.arrived = True
        if held != facing:
            return nes_action(facing), f"aim_face_{facing.lower()}"
        if dodge is not None:
            return nes_action(dodge), "aim_dodge"
        return nes_idle_action(), "aim_wait"


@dataclass
class FinalPatraFightController:
    """Controller-input-only final Patra clear and north-door earn."""

    max_frames: int = PATRA_MAX_FRAMES
    stand_dy: int = PATRA_STAND_DY
    frames: int = 0
    sword_pulses: int = 0
    max_eyes_seen: int = 0
    eye_count_changes: list[dict[str, int]] = field(default_factory=list)
    body_hp_changes: list[int] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def run(
        self,
        env: Any,
        *,
        assist: Any | None = None,
        total: list[int] | None = None,
    ) -> dict[str, Any]:
        total_frames = total if total is not None else [0]
        start = read_snapshot(env.get_ram())
        if not final_patra_live(start):
            return self.report(ok=False, snap=start, error="final Patra not live")

        cooldown = 0
        last_eye_count: int | None = None
        last_body_hp: int | None = None

        for _ in range(self.max_frames):
            snap = read_snapshot(env.get_ram())
            eyes = patra_eyes(snap)
            body = patra_body(snap)
            eye_count = len(eyes)
            self.max_eyes_seen = max(self.max_eyes_seen, eye_count)
            if eye_count != last_eye_count:
                self.eye_count_changes.append({"frame": self.frames, "eyes": eye_count})
                last_eye_count = eye_count
            if body is not None and body.hp != last_body_hp:
                self.body_hp_changes.append(int(body.hp))
                last_body_hp = body.hp

            if final_patra_north_door_earned(snap):
                return self.report(ok=True, snap=snap)
            if snap.mode == 17:
                return self.report(ok=False, snap=snap, error="link death")

            action, reason, cooldown = patra_action(
                snap,
                cooldown=cooldown,
                stand_dy=self.stand_dy,
            )
            if reason.startswith("sword_pulse"):
                self.sword_pulses += 1
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            env.step(action)
            self.frames += 1
            total_frames[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total_frames[0])

        return self.report(
            ok=False,
            snap=read_snapshot(env.get_ram()),
            error="north door timeout",
        )

    def report(
        self,
        *,
        ok: bool,
        snap: ZeldaSnapshot,
        error: str | None = None,
    ) -> dict[str, Any]:
        body = patra_body(snap)
        result: dict[str, Any] = {
            "ok": ok,
            "frames": self.frames,
            "policy": "south_stand_dodge",
            "stand_dy": self.stand_dy,
            "sword_pulses": self.sword_pulses,
            "max_eyes_seen": self.max_eyes_seen,
            "eye_count_changes": list(self.eye_count_changes),
            "body_hp_changes": list(self.body_hp_changes),
            "reasons": dict(self.reasons),
            "north_door_earned": bool(snap.cur_opened_doors & NORTH_DOOR),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "room_all_dead": int(snap.room_all_dead),
            "room_obj_count": int(snap.room_obj_count),
            "remaining_eyes": len(patra_eyes(snap)),
            "body": (
                {"slot": body.slot, "hp": body.hp, "state": body.state}
                if body is not None
                else None
            ),
            "controller_memory_writes": 0,
        }
        if error is not None:
            result["error"] = error
        return result


__all__ = [
    "FinalPatraFightController",
    "NORTH_DOOR",
    "OBJ_PATRA",
    "OBJ_PATRA_2",
    "OBJ_PATRA_EYE",
    "OBJ_PATRA_EYE_2",
    "PATRA_BODY_TYPES",
    "PATRA_EYE_TYPES",
    "PATRA_ATTACK_COOLDOWN",
    "PATRA_BODY_HP_START",
    "PATRA_EYE_COUNT",
    "PATRA_EYE_HP_START",
    "PATRA_MAX_FRAMES",
    "PATRA_ROOM",
    "PATRA_ROOM_FULL",
    "PATRA_STAND_DY",
    "final_patra_live",
    "final_patra_north_door_earned",
    "in_final_patra_room",
    "patra_action",
    "patra_body",
    "patra_eyes",
]
