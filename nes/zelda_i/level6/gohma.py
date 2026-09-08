"""Level 6 Gohma 0x1C: poke wooden arrows, climb the column, loose one
arrow into the open eye.

Leftover is north2c play 0x1C ``(120,205)``. Bow is earned on the L1
Survival splice. Operator exception: ``ADDR_ARROWS=1`` + B-slot 2 until
the 80R shop splice.

The rr-17co CheckWarp walk-on enters this room +223 global frames later
than the old position poke, so the RNG phase differs and Gohma no longer
parks on the x=120 column — it strafes x 128..160, bobs y 112..144 at
x=128, then drifts left, on a ~260f cycle. A fixed column / mouth shot
misses (v1-v4 all red). The kill is reactive:

1. climb UP off door tile 118 onto the ``STAND_Y`` firing line (LEFT/RIGHT
   do nothing wedged in the doorway);
2. lightly lead ``body.x`` and step LEFT/RIGHT to stay within ``FIRE_TOL``;
3. on the rising edge of an eye-open window (RAM ``0x03C7`` just left the
   ``0xC0`` blink), turn to face NORTH, then loose UP+B the next frame.

The face-NORTH frame matters: UP+B straight off a sideways strafe looses
the arrow sideways — that (not aim or timing) is why v1-v4 and the first
reactive passes never dropped ``ghp``.

Do not write ``ADDR_BOW``. Do not poke doors/keys. Keys stay 2. Isolated
BFS banned. Heart / north 0x0C / TF ``0x20`` are later SpineHops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.assist import poke_wooden_arrows
from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.dungeon.hop_controller import (
    CELLAR_MODE,
    HopController,
    WAIT_SCROLL_B,
)
from zelda_i.dungeon.door_hop import door_hop_stages
from zelda_i.level6.door_hop import NORTH2C_SPEC, SOUTH1D_SPEC, WEST2D_SPEC
from zelda_i.level6.occupancy import record_l6_walk
from zelda_i.level6.overworld import LEVEL6, LEVEL6_GOHMA_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "GOHMA_MAX_FRAMES",
    "SPAWN_WINDOW",
    "EYE_ADDR",
    "EYE_SHUT",
    "STAND_Y",
    "Level6GohmaController",
    "gohma_live",
    "level6_gohma_stages",
    "level6_gohma_success",
    "make_gohma_controller",
]

# Reactive hop: climb the column, track the strafing body, fire on the
# rising edge of each eye-open window. Not a 120f spawn hop.
GOHMA_MAX_FRAMES = 1400
SPAWN_WINDOW = 54

# Gohma's eye: RAM 0x03C7 reads 0xC0 for the ~17f closed blink and a lower
# value (0x70 / 0x60 / 0x58 ...) while open, on a ~65f cycle. An arrow only
# damages Gohma if it arrives while the eye is open. Firing on any fixed
# multiple of the cycle aliases straight onto the closed blink (v1-v4 and the
# first reactive pass: 20+ arrows, 0 connects), so fire only in the first
# EYE_EDGE_WINDOW frames after the eye opens — the arrow (flight ~17f from
# STAND_Y) then arrives well inside the same open window.
EYE_ADDR = 0x03C7
EYE_SHUT = 0xC0
EYE_EDGE_WINDOW = 16
FACE_NORTH = 0x08  # ADDR_LINK_FACING: 0x08 N / 0x04 S / 0x02 W / 0x01 E
SHOT_COOLDOWN = 30  # < eye period: at most one live arrow per open window
STUCK_FRAMES = 260  # no fire this long -> take the next aligned shot, gated or not

# Gohma also lobs its own downward type-0x56 projectile most frames; it is
# not Link's arrow, so this fight never gates firing on "arrow on screen".

# Firing column. Recompose killed from y~168 at dx=8, so the hit box is
# forgiving; stand just below Gohma's y<=144 vertical bob and re-climb after
# contact knockback (recompose took knockback to y=189 and still connected).
STAND_Y = 162
STAND_Y_TOL = 8
ALIGN_TOL = 5    # start strafing again past this
FIRE_TOL = 8     # commit to the fire sequence within this (recompose hit dx=8)
ARROW_SPEED = 2.8  # px/frame up the column (recompose y~168 fire -> kill ~18f)
LEAD_CLAMP = 12
LINK_X_MIN, LINK_X_MAX = 40, 216

GOHMA_TYPES = frozenset({GOHMA_OBJECT_TYPE, GOHMA_BLUE_OBJECT_TYPE})
_SKIP_TYPES = frozenset({0, 0xFF})
GOHMA_WAIT = tuple(sorted(set(WAIT_SCROLL_B) | {CELLAR_MODE}))

# A per-frame WRAM dump for retuning this fight lives in
# ``nes/zelda_i/scripts/gohma_lab.py --dump`` (gitignored scratch), not here.


def gohma_live(snap: ZeldaSnapshot) -> list:
    """Red 0x33 (or blue 0x34) slots 1–12. TYPE presence; HP may be 0."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in GOHMA_TYPES
    ]


@dataclass
class Level6GohmaController(HopController):
    """Poke wooden arrows, climb the column, fire on each eye-open edge."""

    spec_id: str = "level6_gohma_0x1c"
    room: int = LEVEL6_GOHMA_ROOM
    max_frames: int = GOHMA_MAX_FRAMES
    wait_modes: tuple[int, ...] = GOHMA_WAIT
    done_reason: str = "body_gone"
    cooldown: int = 0
    saw_gohma: bool = False
    poked: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    inventory_assist: dict[str, Any] | None = None
    env: Any | None = None
    arrow_pulses: int = 0
    gx_hist: list[int] = field(default_factory=list)
    eye_open_since: int = -1  # frames since 0x03C7 left the blink; -1 = shut now
    last_fire: int = 0
    connect_frame: int | None = None

    def bind_env(self, env: Any) -> None:
        self.env = env

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_bow={int(snap.bow)}_arrows={int(snap.arrows)}"
            f"_pulses={self.arrow_pulses}"
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"body_gone_{snap.link_x}_{snap.link_y}_pulses={self.arrow_pulses}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        bodies = gohma_live(snap)
        body = bodies[0] if bodies else None
        if self.saw_gohma and body is None and self.connect_frame is None:
            self.connect_frame = self.frames
        self.leftover = record_l6_walk(
            self.samples,
            snap,
            reason=action.reason,
            frames=self.frames,
            period=16,
            misses=0,
            force=force,
        )
        if force or self.frames <= SPAWN_WINDOW or self.frames % 16 == 0:
            types = [
                int(obj.type_id)
                for obj in snap.objects
                if 1 <= obj.slot <= 12 and int(obj.type_id) not in _SKIP_TYPES
            ]
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "gx": None if body is None else int(body.x),
                    "gy": None if body is None else int(body.y),
                    "ghp": None if body is None else int(body.hp),
                    "gst": None if body is None else int(body.state),
                    "eye": self._eye_byte(),
                    "eye_since": self.eye_open_since,
                    "n": len(bodies),
                    "types": types,
                    "bow": int(snap.bow),
                    "arrows": int(snap.arrows),
                    "rupees": int(snap.rupees),
                    "misses": 0,
                }
            )
        return action

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        bodies = gohma_live(snap)
        if bodies:
            self.saw_gohma = True
            return False
        return self.saw_gohma

    def _eye_byte(self) -> int | None:
        if self.env is None:
            return None
        try:
            return int(self.env.get_ram()[EYE_ADDR])
        except Exception:  # pragma: no cover - defensive
            return None

    def _track_eye(self) -> None:
        byte = self._eye_byte()
        if byte is None or byte == EYE_SHUT:
            self.eye_open_since = -1
        elif self.eye_open_since < 0:
            self.eye_open_since = 0
        else:
            self.eye_open_since = min(9999, self.eye_open_since + 1)

    def _eye_fresh_open(self) -> bool:
        """Eye left the blink within EYE_EDGE_WINDOW frames (arrow will land)."""
        return 0 <= self.eye_open_since <= EYE_EDGE_WINDOW

    def _poke(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.poked:
            return None
        if int(snap.bow) < 1:
            return self.mark_fail("unarmed_no_bow")
        if self.env is None:
            if int(snap.arrows) >= 1:
                self.poked = True
                self.notes.append("arrows_already_set")
                return None
            return self.mark_fail("no_env_for_arrow_write")
        self.inventory_assist = poke_wooden_arrows(
            self.env, from_arrows=int(snap.arrows), select=True
        )
        self.poked = True
        n = int(self.inventory_assist.get("inventory_writes") or 0)
        self.notes.append(
            f"arrow_poke_writes={n}_from={int(snap.arrows)}"
        )
        if int(self.inventory_assist.get("progression_writes") or 0) != 0:
            return self.mark_fail("arrow_poke_progression")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")

        poked = self._poke(snap)
        if poked is not None:
            return poked

        bodies = gohma_live(snap)
        if not bodies:
            return FrameAction(nes_idle_action(), "wait_body")

        self._track_eye()

        body = bodies[0]
        gx = int(body.x)
        self.gx_hist.append(gx)
        del self.gx_hist[:-8]
        gvx = (
            (self.gx_hist[-1] - self.gx_hist[0]) / (len(self.gx_hist) - 1)
            if len(self.gx_hist) >= 4
            else 0.0
        )

        ly = int(snap.link_y)

        # 1. climb the column onto the firing line. LEFT/RIGHT do nothing
        #    wedged in the south doorway, so climb straight up (recompose
        #    walked column x~120 the whole way and still hit at dx=8).
        if ly > STAND_Y + STAND_Y_TOL:
            return FrameAction(nes_action("UP"), "climb")

        # 2. lightly lead the strafing body; hit box is ~+-8 so a rough
        #    column is enough during the vertical bob (gvx ~ 0) and the sweep.
        flight = max(1.0, (ly - int(body.y)) / ARROW_SPEED)
        lead = int(round(gvx * flight))
        lead = max(-LEAD_CLAMP, min(LEAD_CLAMP, lead))
        target_x = max(LINK_X_MIN, min(LINK_X_MAX, gx + lead))
        dx = target_x - int(snap.link_x)

        # 3. fire on the rising edge of an eye-open window (or force a shot if
        #    nothing has connected for STUCK_FRAMES). Commit as soon as we are
        #    within FIRE_TOL so a 1px body drift doesn't kick us back to
        #    strafing and flip Link's facing mid-sequence.
        if int(snap.rupees) <= 0:
            return self.mark_fail("out_of_ammo")
        forced = self.frames - self.last_fire >= STUCK_FRAMES
        ready = self.cooldown <= 0 and (self._eye_fresh_open() or forced)
        if ready and abs(dx) <= FIRE_TOL:
            # UP+B in one frame from a sideways strafe looses the arrow
            # sideways (facing does not flip in time — that is why every
            # earlier pass missed). Face NORTH first, fire next frame.
            if int(snap.facing) != FACE_NORTH:
                return FrameAction(nes_action("UP"), "face_up")
            self.cooldown = SHOT_COOLDOWN
            self.last_fire = self.frames
            self.arrow_pulses += 1
            return FrameAction(nes_action("UP", "B"), "arrow_shot")

        if abs(dx) > ALIGN_TOL:
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "strafe"
            )
        if ly < STAND_Y - STAND_Y_TOL:
            return FrameAction(nes_action("DOWN"), "settle")
        if self.cooldown > 0:
            return FrameAction(nes_idle_action(), "cooldown")
        return FrameAction(nes_idle_action(), "eye_wait")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.cooldown > 0:
            self.cooldown -= 1
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "leftover": dict(self.leftover),
            "inventory_assist": self.inventory_assist,
            "policy": "poke arrows; climb to STAND_Y; face N; UP+B on the eye edge",
            "saw_gohma": self.saw_gohma,
            "arrow_pulses": self.arrow_pulses,
            "connect_frame": self.connect_frame,
            "spec_id": self.spec_id,
            "room": self.room,
            "body_type": GOHMA_OBJECT_TYPE,
        }


def make_gohma_controller() -> Level6GohmaController:
    """Kill Gohma 0x1C with poked wooden arrows. Bow already earned."""
    return Level6GohmaController()


def level6_gohma_stages():
    """West 0x2C KEY-UP leftover → poke arrows → Gohma gone."""
    return (
        *door_hop_stages(SOUTH1D_SPEC),
        *door_hop_stages(WEST2D_SPEC),
        *door_hop_stages(NORTH2C_SPEC),
        ("level6_gohma_0x1c", make_gohma_controller(), GOHMA_MAX_FRAMES),
    )


def level6_gohma_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x1C, Gohma gone, bow+arrows set. TF 0x20 is the next hop."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if snap.mode != PLAY_MODE or snap.transitioning or snap.screen != LEVEL6_GOHMA_ROOM:
        return False
    if int(snap.bow) < 1 or int(snap.arrows) < 1:
        return False
    return not gohma_live(snap)
