"""Level 8 Magical-Key chapter one-frame policies (fixture-live).

Promotes the ``scratch/probe_l8_1e_gohma.py`` kill into a spine-composable
``HopController``.  The fight is the Level 6 ``0x1C`` eye-edge arrow policy
re-aimed at the live ``0x1E`` body (RAM type ``0x33``; ``ids.py`` labels that
``gohma_red`` -- colour is *not* asserted, only the mechanic).  Wooden arrows
and the bow are the naturally owned Survival inventory: no arrow poke, no L6
``0x1C`` detour, no ``$0656`` write (B is cycled through the pause menu).

Not route eligible; not a ``DungeonRoomSpec``; never a power-on claim on its
own.  ``rr-6o7.2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import direction_to_facing
from zelda_i.dungeon.behaviors import projectile_threats
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon.gleeok import FIREBALL_DODGE_DIST, _fireball_dodge_dir
from zelda_i.dungeon.gohma import (
    GOHMA_TYPES,
    advance_eye,
    arrow_aim_x,
    eye_fresh_open,
    read_eye,
)
from zelda_i.dungeon.hop_controller import (
    BLOCK_Y_OFFSET,
    HopController,
    WAIT_SCROLL_B,
    lattice_goto,
    room_step,
    stairs_step,
)
from zelda_i.dungeon.ids import (
    DARKNUT_OBJECT_TYPE,
    FIREBALL_OBJECT_TYPE,
    MANHANDLA_PROJECTILE_TYPE,
    POLS_VOICE_OBJECT_TYPE,
)
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.dungeon.pause_select import B_SLOT_ARROWS, B_SLOT_BOMBS, PauseSelectController
from zelda_i.level8.cellar import magic_key_cellar_return_step
from zelda_i.level8.north_column import TYPE_0C, _SWORD_PATROL
from zelda_i.level6.gohma import (
    ALIGN_TOL,
    FACE_NORTH,
    FIRE_TOL,
    SHOT_COOLDOWN,
    STAND_Y_TOL,
    STUCK_FRAMES,
)
from zelda_i.ram import (
    ADDR_MAGIC_KEY,
    ADDR_SELECTED_ITEM,
    PASSAGE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

LEVEL8 = 8
GOHMA_ROOM_1E = 0x1E
GOHMA_DEST_1F = 0x1F
# Live census in 0x1E (rr-gw0x probe_l8_1e_gohma D1b/D2): one body, RAM type
# 0x33 HP 96, killed by three connecting wooden arrows.  0x34 is accepted too
# so a blue re-observation is not a false miss; colour is not asserted.  Same
# set L6 0x1C uses -- one Gohma, one table (``dungeon.gohma``).
GOHMA_BODY_TYPES_1E = GOHMA_TYPES
BLUE_GOHMA_ARROWS_REQUIRED = 3  # walkthrough number; connects, not shots
BODY_GONE_FRAMES = 45
# door_graph.core: RIGHT 0x01.  Kill raised cur_opened_doors 0x04 -> 0x0D
# (RIGHT+DOWN+UP); the open_doorway_mask stayed 0x04, so gate on the
# doors byte that actually rose (same as L6 post-Gleeok / L9 Patra).
RIGHT_DOOR_BIT = 0x01
EAST_DOOR = DOOR_TARGETS["RIGHT"]  # (208, 141)
_DOOR_TOL = 4
EAST_WAIT_FRAMES = 150  # idle budget for the doors byte to rise after the kill
GOHMA_1E_MAX_FRAMES = 12_000
# South door lip: LEFT/RIGHT are no-ops. Inland, dodge type-0x56 and Gohma
# contact — Clean 1–2 hearts cannot tank the L6 stand-line fire stream.
GOHMA_DOOR_LIP_Y = 189
GOHMA_BODY_CONTACT = 20
# L6 STAND_Y=162 is the 0x1E death band (leftover y=163/165). Stand south,
# still inland of the door lip so strafe/dodge work (TOL=8 → climb while y>189).
STAND_Y = 181
# Inland of both statue columns. Leftover (50,149) was LEFT into the west 0x55.
INLAND_X_MIN = 88
INLAND_X_MAX = 168
# Once B=arrows, hold the entry column. LEFT of 112 is the west 0x55 stream
# (h8 leftover (88,181), shots 0, B=arrows).
COLUMN_X_MIN = 112
COLUMN_X_MAX = 128
# Statue 0x55 and Gohma/Manhandla 0x56 both stream this room (rr-npv.4).
_SHOT_TYPES = frozenset({FIREBALL_OBJECT_TYPE, MANHANDLA_PROJECTILE_TYPE})


@dataclass(kw_only=True)
class Level8BlueGohma1EController(HopController):
    """0x1E: pause-select arrows, eye-edge arrow kill, RIGHT shutter to 0x1F.

    Fails closed without a naturally owned bow + wooden arrows (never pokes an
    arrow or L6's one-time grant).  Dest is RAM; ``0x1F`` is the recon-observed
    east neighbour but the controller only asserts "left 0x1E east into play".
    """

    spec_id: str = "level8_blue_gohma"
    room: int = GOHMA_ROOM_1E
    dest: int = GOHMA_DEST_1F
    max_frames: int = GOHMA_1E_MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "arrived_0x1f"
    route_eligible: bool = False
    writes: int = 0

    cooldown: int = 0
    shots: int = 0
    last_fire: int = 0
    eye_open_since: int = -1
    saw_body: bool = False
    body_gone: int = 0
    east_open: bool = False
    doors_in: int | None = None
    gx_hist: list[int] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController | None = field(
        default=None, init=False, repr=False
    )

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def bind_env(self, env: Any) -> None:
        self._env = env
        if self._select is None:
            self._select = PauseSelectController(want=B_SLOT_ARROWS, name="arrows")
        self._select.bind_env(env)

    # -- lifecycle -----------------------------------------------------------
    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL8
            and snap.screen == self.dest
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_shots={self.shots}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("RIGHT"), "east_scroll")

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_shots={self.shots}_body_gone={self.body_gone}"
        )

    # -- eye tracker (RAM 0x03C7, shared with L6 0x1C) ----------------------
    def _eye_byte(self) -> int | None:
        return read_eye(self._env)

    def _track_eye(self) -> None:
        self.eye_open_since = advance_eye(self.eye_open_since, self._eye_byte())

    def _bodies(self, snap: ZeldaSnapshot) -> list:
        return [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and int(obj.type_id) in GOHMA_BODY_TYPES_1E
        ]

    def _shots(self, snap: ZeldaSnapshot) -> list:
        return [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and int(obj.type_id) in _SHOT_TYPES
        ]

    def _se_shot(self, snap: ZeldaSnapshot) -> bool:
        """True if a 0x55/0x56 sits east of Link on or south of the stand line."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        south = min(ly, STAND_Y)
        return any(int(obj.x) > lx and int(obj.y) >= south for obj in self._shots(snap))

    def _inland(self, snap: ZeldaSnapshot) -> bool:
        lx, ly = int(snap.link_x), int(snap.link_y)
        return ly <= STAND_Y and INLAND_X_MIN <= lx <= INLAND_X_MAX

    def _arrows_on_b(self) -> bool:
        if self._select is not None and self._select.success:
            return True
        if self._env is None:
            return False
        try:
            return int(read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM)) == B_SLOT_ARROWS
        except Exception:  # pragma: no cover - defensive
            return False

    def _hold_column(self, snap: ZeldaSnapshot) -> bool:
        return int(snap.link_y) <= STAND_Y and self._arrows_on_b()

    def _column_shot(self, snap: ZeldaSnapshot) -> bool:
        """0x55/0x56 on the entry column, y approaching STAND_Y (h10 idle)."""
        lx = int(snap.link_x)
        return any(
            abs(int(o.x) - lx) <= 16 and int(o.y) >= STAND_Y - 24
            for o in self._shots(snap)
        )

    def _column_btn(self, snap: ZeldaSnapshot, btn: str) -> str | None:
        """Keep x in [112, 128]. Cooldown LEFT toward 112; else RIGHT off fire."""
        if btn not in ("LEFT", "RIGHT"):
            return btn
        lx = int(snap.link_x)
        if lx < COLUMN_X_MIN:
            return "RIGHT"
        if lx > COLUMN_X_MAX:
            return "LEFT"
        if lx < COLUMN_X_MAX and self._column_shot(snap):
            return "LEFT" if self.cooldown > 0 and lx > COLUMN_X_MIN else "RIGHT"
        return None

    def _inland_escape(self, snap: ZeldaSnapshot, reason: str) -> FrameAction:
        """Off a statue column: return to STAND_Y, else step toward [88, 168]."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        if ly < STAND_Y - STAND_Y_TOL:
            return FrameAction(nes_action("DOWN"), "settle")
        if ly > STAND_Y + STAND_Y_TOL:
            return FrameAction(nes_action("UP"), "climb")
        if lx <= INLAND_X_MIN:
            return FrameAction(nes_action("RIGHT"), reason)
        if lx >= INLAND_X_MAX:
            return FrameAction(nes_action("LEFT"), reason)
        return FrameAction(nes_idle_action(), "eye_wait")

    def _arrow_fire(self, snap: ZeldaSnapshot) -> FrameAction:
        """One-frame UP+B. Cooldown + inbound 0x55 peels in-band, never idle."""
        ly = int(snap.link_y)
        if ly < STAND_Y - STAND_Y_TOL:
            return FrameAction(nes_action("DOWN"), "settle")
        if self.cooldown <= 0:
            if int(snap.facing) != FACE_NORTH:
                return FrameAction(nes_action("UP"), "face_up")
            self.cooldown = SHOT_COOLDOWN
            self.last_fire = self.frames
            self.shots += 1
            return FrameAction(nes_action("UP", "B"), "arrow_shot")
        if ly < STAND_Y:
            return FrameAction(nes_action("DOWN"), "settle")
        if self._column_shot(snap):
            lx = int(snap.link_x)
            btn = "RIGHT" if lx <= COLUMN_X_MIN else "LEFT"
            return FrameAction(nes_action(btn), "column_peel")
        return FrameAction(nes_idle_action(), "eye_wait")

    def _clamp_hmove(
        self, snap: ZeldaSnapshot, btn: str, reason: str
    ) -> FrameAction:
        lx = int(snap.link_x)
        if self._hold_column(snap):
            if btn == "LEFT" and lx <= COLUMN_X_MAX:
                if lx < COLUMN_X_MIN:
                    return FrameAction(nes_action("RIGHT"), "column_recover")
                if lx >= COLUMN_X_MAX:
                    return self._arrow_fire(snap)
                if self.cooldown > 0 and self._column_shot(snap):
                    return FrameAction(nes_action("LEFT"), "column_peel")
                return FrameAction(nes_idle_action(), "column_hold")
            if btn == "RIGHT" and lx >= COLUMN_X_MAX:
                return self._arrow_fire(snap)
        if btn == "LEFT" and lx <= INLAND_X_MIN:
            return self._inland_escape(snap, reason)
        if btn == "RIGHT" and lx >= INLAND_X_MAX:
            return self._inland_escape(snap, reason)
        return FrameAction(nes_action(btn), reason)

    def _sidestep(
        self, snap: ZeldaSnapshot, hazard_x: int, reason: str
    ) -> FrameAction:
        lx = int(snap.link_x)
        if int(hazard_x) >= lx:
            btn = "LEFT"
        else:
            btn = "RIGHT"
        return self._clamp_hmove(snap, btn, reason)

    def _maybe_hmove(
        self, snap: ZeldaSnapshot, btn: str, reason: str
    ) -> FrameAction | None:
        """Column-hold: never LEFT of 112; cooldown LEFT, else RIGHT off fire."""
        if self._hold_column(snap):
            mapped = self._column_btn(snap, btn)
            if mapped is None:
                return None
            if mapped != btn:
                lx = int(snap.link_x)
                reason = (
                    "column_peel"
                    if mapped == "RIGHT" and lx >= COLUMN_X_MIN
                    else "column_recover"
                )
            btn = mapped
        return self._clamp_hmove(snap, btn, reason)

    def _hazard_dodge(self, snap: ZeldaSnapshot, body) -> FrameAction | None:
        """Sidestep 0x55/0x56 / Gohma contact once off the south door lip."""
        ly = int(snap.link_y)
        lx = int(snap.link_x)
        if ly > STAND_Y:
            return None
        if self._hold_column(snap) and self._column_shot(snap):
            if COLUMN_X_MIN <= lx < COLUMN_X_MAX:
                btn = "LEFT" if self.cooldown > 0 and lx > COLUMN_X_MIN else "RIGHT"
                return FrameAction(nes_action(btn), "column_peel")
            if lx >= COLUMN_X_MAX:
                return self._arrow_fire(snap)
        dodge = _fireball_dodge_dir(snap, thr=FIREBALL_DODGE_DIST)
        if dodge is not None:
            return self._maybe_hmove(snap, dodge, "climb_dodge_fb")
        shots = self._shots(snap)
        if shots:
            fb = min(
                shots, key=lambda o: abs(int(o.x) - lx) + abs(int(o.y) - ly)
            )
            dist = abs(int(fb.x) - lx) + abs(int(fb.y) - ly)
            if dist <= FIREBALL_DODGE_DIST:
                btn = "LEFT" if int(fb.x) >= lx else "RIGHT"
                return self._maybe_hmove(snap, btn, "climb_dodge_fb")
        hits = projectile_threats(
            lx,
            ly,
            shots,
            direction="UP",
            ahead=FIREBALL_DODGE_DIST,
            behind=4,
            half_width=8,
        )
        if hits:
            fb = min(
                hits, key=lambda o: abs(int(o.x) - lx) + abs(int(o.y) - ly)
            )
            btn = "LEFT" if int(fb.x) >= lx else "RIGHT"
            return self._maybe_hmove(snap, btn, "climb_dodge_fb")
        if body is None:
            return None
        bx, by = int(body.x), int(body.y)
        if max(abs(bx - lx), abs(by - ly)) <= GOHMA_BODY_CONTACT:
            btn = "LEFT" if bx >= lx else "RIGHT"
            return self._maybe_hmove(snap, btn, "climb_dodge_body")
        return None

    def _emit_leftover(self, snap: ZeldaSnapshot) -> None:
        if not self.leftover or self.frames % 12 == 0:
            self.leftover = {
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "shots": self.shots,
                "cur_opened_doors": int(snap.cur_opened_doors),
            }

    # -- per-frame policy --------------------------------------------------
    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.cooldown > 0:
            self.cooldown -= 1
        return super().step(snap)

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self._emit_leftover(snap)
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL8:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self.mark_fail(
                f"l8_gohma_unexpected_room_0x{snap.screen:02x}"
            )
        if int(snap.bow) < 1 or int(snap.arrows) < 1:
            return self.mark_fail("l8_gohma_requires_natural_bow_arrows")

        if self.doors_in is None:
            self.doors_in = int(snap.cur_opened_doors)

        bodies = self._bodies(snap)
        if bodies:
            self.saw_body = True
            self.body_gone = 0
        elif self.saw_body:
            self.body_gone += 1

        # kill complete -> east kill-clear shutter
        if self.saw_body and not bodies and self.body_gone >= BODY_GONE_FRAMES:
            if not self.east_open:
                rose = bool(int(snap.cur_opened_doors) & RIGHT_DOOR_BIT) and not (
                    int(self.doors_in or 0) & RIGHT_DOOR_BIT
                )
                if rose:
                    self.east_open = True
                elif self.body_gone < BODY_GONE_FRAMES + EAST_WAIT_FRAMES:
                    return FrameAction(nes_idle_action(), "east_wait_right_bit")
                else:
                    return self.mark_fail(
                        "east_gate_0x1e_not_a_kill_clear_shutter"
                    )
            return self._east_push(snap)

        # Lip (y>STAND_Y): only UP. LEFT dodge at y=189 walked to (88,181)
        # before pause-select (l8clr_lab_h4). Climb at entry x, then select.
        body = bodies[0] if bodies else None
        lx = int(snap.link_x)
        ly = int(snap.link_y)
        if ly > STAND_Y:
            return FrameAction(nes_action("UP"), "climb")
        if self._hold_column(snap) and ly < STAND_Y - STAND_Y_TOL:
            return FrameAction(nes_action("DOWN"), "settle")

        inland = self._inland(snap)
        if not inland:
            dodge = self._hazard_dodge(snap, body)
            if dodge is not None:
                return dodge
            if lx < INLAND_X_MIN or lx > INLAND_X_MAX:
                return self._inland_escape(snap, "climb_dodge_fb")

        if self._select is not None and inland:
            driven = self._select.drive(snap)
            if self._select.failed:
                return self.mark_fail(
                    self._select.fail_reason or "l8_gohma_arrow_select_failed"
                )
            if driven is not None:
                return driven

        dodge = self._hazard_dodge(snap, body)
        if dodge is not None:
            return dodge

        if not bodies:
            return FrameAction(nes_idle_action(), "wait_body")

        self._track_eye()
        body = bodies[0]
        bounds = (
            (COLUMN_X_MIN, COLUMN_X_MAX)
            if self._hold_column(snap)
            else (INLAND_X_MIN, INLAND_X_MAX)
        )
        target_x = arrow_aim_x(self.gx_hist, body, ly, bounds)
        dx = target_x - int(snap.link_x)

        if int(snap.rupees) <= 0:
            return self.mark_fail("l8_gohma_out_of_ammo")
        forced = self.frames - self.last_fire >= STUCK_FRAMES
        fresh = eye_fresh_open(self.eye_open_since)
        if self.cooldown <= 0 and (fresh or forced) and abs(dx) <= FIRE_TOL:
            return self._arrow_fire(snap)

        if abs(dx) > ALIGN_TOL:
            go_right = dx > 0
            # y=189 RIGHT-strafe walked into the SE statue stream (140,189).
            if go_right and (ly >= GOHMA_DOOR_LIP_Y or self._se_shot(snap)):
                if ly > STAND_Y:
                    return FrameAction(nes_action("UP"), "climb")
                return self._sidestep(snap, int(snap.link_x) + 16, "climb_dodge_fb")
            btn = "RIGHT" if go_right else "LEFT"
            return self._clamp_hmove(snap, btn, "strafe")
        if ly < STAND_Y - STAND_Y_TOL:
            return FrameAction(nes_action("DOWN"), "settle")
        return FrameAction(nes_idle_action(), "eye_wait")

    def _east_push(self, snap: ZeldaSnapshot) -> FrameAction:
        gx, gy = EAST_DOOR
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(y - gy) > _DOOR_TOL:
            return FrameAction(
                nes_action("UP" if y > gy else "DOWN"), "east_align"
            )
        if x < gx - _DOOR_TOL:
            return FrameAction(nes_action("RIGHT"), "east_approach")
        return FrameAction(nes_action("RIGHT"), "east_push")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "shots": self.shots,
            "arrows_required": BLUE_GOHMA_ARROWS_REQUIRED,
            "body_types": tuple(sorted(GOHMA_BODY_TYPES_1E)),
            "east_open": self.east_open,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "poked_arrows": False,
            "writes": int(self.writes),
            "door": "RIGHT",
            "leftover": dict(self.leftover),
            "select": None if self._select is None else self._select.report(),
        }


def make_blue_gohma_1e_controller() -> Level8BlueGohma1EController:
    return Level8BlueGohma1EController()


# --------------------------------------------------------------------------
# level8_magic_key_stairs: 0x1F centre 0x68 block -> cellar 0x0F -> Magical
# Key -> two-ladder return to play 0x1F.  Promotes probe_l8_1f_magic_key +
# probe_l8_0f_cellar_return + level8.cellar.magic_key_cellar_return_step.
# --------------------------------------------------------------------------

STAIRS_ROOM_1F = 0x1F
CELLAR_ROOM_0F = 0x0F
STAIRS_OBJECT = 0x68
BLOCK_SLIDE_PX = 8
STAIRS_STAND_CENTER = (128, 141)  # L1 CheckWarps pose after a west-block slide
STAIRS_TILES = frozenset(range(0x70, 0x74))
# 0x1F census (LEVEL8_INTERIOR_0X1F_RECON): 2x pols voice 0x16, 2x 0x0C
# HP128, 2x blue darknut 0x0B.  Hitstun from these blocks the 0x68 push.
STAIRS_1F_CENSUS_TYPES = (POLS_VOICE_OBJECT_TYPE, TYPE_0C, DARKNUT_OBJECT_TYPE)
MK_CELLAR_FLOOR_Y = 189
MK_CELLAR_EAST_X = 176
MK_CELLAR_PEDESTAL = (136, 141)
# 0x1F is a 5-block diamond around the centre stairs: blocks at x in
# {96,112,128,144,160} on rows y in {112..176} (tilemap dump).  x>=176 and
# x<=80 columns are fully open.  The clear patrol can leave Link boxed south
# of the diamond, so route out via the open x=192 vertical lane up to the
# fully-open north band (y~104), then west to the pushable 0x68 column
# (x=96) and hold DOWN to shove it south into the open (96,160) cell
# (probe_l8_1f_magic_key E2).
_PUSH_LANE_X = 192
# y=93 north band: Link's 16px body clears the diamond's top block at
# (128,112) only above y~109, so route along the top wall, not y~104.
_PUSH_NORTH_Y = 93
_CLEAR_1F_FRAMES = 16_000
MAGIC_KEY_STAIRS_MAX_FRAMES = 34_000
_PROX = 5
# The two Darknuts spawned inside the centre diamond are unreachable by
# ordinary sword. These perimeter stands hit them through its gaps.
_INNER_BOMB_STANDS = (
    ((144, 109), "DOWN"), ((160, 125), "LEFT"), ((144, 165), "UP"),
    ((80, 141), "RIGHT"), ((144, 165), "UP"),
)


def _clear_1f_spec() -> DungeonRoomSpec:
    return DungeonRoomSpec(
        spec_id="l8_clear_0x1f_stairs_census",
        source_room=STAIRS_ROOM_1F,
        room_id=STAIRS_ROOM_1F,
        entry=DoorRoute("LEFT", ((16, 141),)),
        enemy_types=STAIRS_1F_CENSUS_TYPES,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        # pols voice can read hp=0 while still hopping.
        type_only_enemy_types=(POLS_VOICE_OBJECT_TYPE,),
        combat=CombatTuning(
            patrol=_SWORD_PATROL,
            engage_distance=48,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=_CLEAR_1F_FRAMES,
        level=LEVEL8,
    )


@dataclass(kw_only=True)
class Level8MagicKeyStairsController(HopController):
    """0x1F: clear the census, slide the centre 0x68, drop the revealed
    stairs, walk the cellar to the Magical Key, then the two-ladder return
    to play 0x1F carrying ``ADDR_MAGIC_KEY`` 0->1.  Fixture-live; no writes.
    """

    spec_id: str = "level8_magic_key_stairs"
    room: int = STAIRS_ROOM_1F
    max_frames: int = MAGIC_KEY_STAIRS_MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "magic_key_and_back_in_0x1f"
    route_eligible: bool = False
    writes: int = 0

    phase: str = "clear"
    mk_before: int | None = None
    mk_gained: bool = False
    block_xy0: tuple[int, int] | None = None
    push_frames: int = 0
    cellar_frames: int = 0
    cellar_entry_x: int | None = None
    leftover: dict[str, Any] = field(default_factory=dict)
    _env: Any = field(default=None, init=False, repr=False)
    _clear: GenericDungeonRoomController | None = field(
        default=None, init=False, repr=False
    )
    _saw_census: bool = False
    _no_beam: bool = False
    _bomb_wait: int = 0
    _bomb_count: int = 0
    _bombs_before: int = 0
    _bomb_select: PauseSelectController = field(
        default_factory=lambda: PauseSelectController(want=B_SLOT_BOMBS),
        init=False, repr=False,
    )

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._bomb_select.bind_env(env)

    def _mk(self) -> int:
        if self._env is None:
            return 0
        try:
            return int(read_u8(self._env.get_ram(), ADDR_MAGIC_KEY))
        except Exception:  # pragma: no cover - defensive
            return 0

    def _blocks(self, snap: ZeldaSnapshot) -> list:
        return [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and int(obj.type_id) == STAIRS_OBJECT
        ]

    def _live_1f(self, snap: ZeldaSnapshot) -> list:
        want = frozenset(STAIRS_1F_CENSUS_TYPES)
        return [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and int(obj.type_id) in want and obj.hp > 0
        ]

    # -- lifecycle ---------------------------------------------------------
    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.mk_gained
            and snap.level == LEVEL8
            and snap.screen == STAIRS_ROOM_1F
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"magic_key_{self._mk()}_back_0x1f_{snap.link_x}_{snap.link_y}"

    def _emit_leftover(self, snap: ZeldaSnapshot) -> None:
        if not self.leftover or self.frames % 12 == 0:
            self.leftover = {
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "phase": self.phase,
                "magic_key": self._mk(),
            }

    def _goto(self, snap: ZeldaSnapshot, tx: int, ty: int, reason: str):
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(y - ty) > _PROX:
            return FrameAction(
                nes_action("DOWN" if y < ty else "UP"), f"{reason}_y"
            )
        if abs(x - tx) > _PROX:
            return FrameAction(
                nes_action("RIGHT" if x < tx else "LEFT"), f"{reason}_x"
            )
        return None

    def _inner_bomb_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self._live_1f(snap):
            self.phase = "push"
            self._note(f"inner_clear_{self._bomb_count}_bombs")
            return self._push_policy(snap)
        if self._bomb_wait:
            if self._bomb_wait == 105 and snap.bombs >= self._bombs_before:
                self._bomb_wait = 0
                self._bomb_count -= 1
                return FrameAction(nes_idle_action(), "bomb_retry_edge")
            self._bomb_wait -= 1
            return FrameAction(nes_idle_action(), "inner_bomb_wait")
        selected = self._bomb_select.drive(snap)
        if self._bomb_select.failed:
            return self.mark_fail(self._bomb_select.fail_reason)
        if selected is not None:
            return selected
        if snap.bombs <= 0:
            return self.mark_fail("inner_bombs_exhausted")
        goal, face = _INNER_BOMB_STANDS[min(self._bomb_count, 4)]
        if max(abs(snap.link_x - goal[0]), abs(snap.link_y - goal[1])) > 3:
            direction = room_step(snap, goal, tol=2)
            if direction is None:
                return self.mark_fail("inner_bomb_stand_unreachable")
            return FrameAction(nes_action(direction), "inner_bomb_approach")
        if snap.facing != direction_to_facing(face):
            return FrameAction(nes_action(face), "inner_bomb_face")
        self._bomb_count += 1
        self._bombs_before = snap.bombs
        self._bomb_wait = 105
        return FrameAction(nes_action("B"), "inner_bomb_place")

    # -- per-frame policy -------------------------------------------------
    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.mk_before is None:
            self.mk_before = self._mk()
        self._emit_leftover(snap)

        # cellar / return legs run in passage mode (9); everything else needs
        # controllable play in 0x1F.
        if self.phase in ("cellar", "return"):
            return self._cellar_policy(snap)
        if self.phase == "inner_bombs":
            return self._inner_bomb_policy(snap)

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL8:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != STAIRS_ROOM_1F:
            return self.mark_fail(
                f"l8_mk_stairs_unexpected_room_0x{snap.screen:02x}"
            )

        if self.phase == "clear":
            self._no_beam |= not snap.health_is_full
            live = self._live_1f(snap)
            self._saw_census |= len(live) >= 6
            outer = any(
                o.slot not in (2, 3) and o.type_id in STAIRS_1F_CENSUS_TYPES
                for o in snap.objects
            )
            if self._no_beam and self._saw_census and len(live) <= 2 and not outer:
                self.phase = "inner_bombs"
                return self._inner_bomb_policy(snap)
            if self._clear is None:
                self._clear = GenericDungeonRoomController(_clear_1f_spec())
                self._clear.phase = DungeonPhase.FIGHT
            # Run the clear engine to its own DONE -- a pols voice (0x16) can
            # sit at hp=0 while still hopping, so a bare hp>0 census check
            # would advance to the push while an enemy is still hitstunning
            # Link off the 0x68 (E1).
            if self._clear.phase not in (DungeonPhase.DONE, DungeonPhase.FAILED):
                act = self._clear.step(snap)
                if self._clear.phase is DungeonPhase.FAILED:
                    note = (
                        self._clear.notes[-1]
                        if self._clear.notes
                        else "clear_0x1f_failed"
                    )
                    return self.mark_fail(f"l8_mk_clear_0x1f:{note}")
                if self._clear.phase is not DungeonPhase.DONE:
                    return act
            # settle a few frames so late hitstun clears before the push
            if self._live_1f(snap):
                return FrameAction(nes_idle_action(), "clear_1f_settle")
            self.phase = "push"

        if self.phase == "push":
            return self._push_policy(snap)

        if self.phase == "stairs":
            return self._stairs_policy(snap)

        return self.mark_fail(f"l8_mk_stairs_bad_phase_{self.phase}")

    def _push_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        blocks = self._blocks(snap)
        if blocks:
            bx, by = int(blocks[0].x), int(blocks[0].y)
            if self.block_xy0 is None:
                self.block_xy0 = (bx, by)
        elif self.block_xy0 is None:
            # no block object at all -> already open; go straight for the stairs
            self.phase = "stairs"
            return self._stairs_policy(snap)
        else:
            bx, by = self.block_xy0

        b0x, b0y = self.block_xy0
        slid = (not blocks) or (abs(by - b0y) >= BLOCK_SLIDE_PX) or (
            abs(bx - b0x) >= BLOCK_SLIDE_PX
        )
        if slid:
            self.phase = "stairs"
            return self._stairs_policy(snap)

        # probe_l8_1f_magic_key E2: the winning slide is from the NORTH -- get
        # to the block's x in the open y=93 north band, then hold DOWN so Link
        # shoves the 0x68 south and vacates the west diamond slot.  The old
        # "stage west / south-face UP" line wedged on the diamond geometry when
        # the clear left Link south-east of the block (power-on: (144,165),
        # tile 178 wall).  Route out via the fully-open x=192 column.
        x, y = int(snap.link_x), int(snap.link_y)
        # ROM lattice to the north face first: the lattice treats the block
        # tile as solid, so it never shoves the 0x68 the wrong way. The
        # timed waypoints below walked RIGHT along y=141 from a west pose
        # (power-on gathered spine) and slid it into the diamond, sealing
        # the stairs for good.
        stand = (b0x, b0y - BLOCK_Y_OFFSET - 16)
        if abs(x - stand[0]) <= 2 and stand[1] - 2 <= y <= b0y - BLOCK_Y_OFFSET:
            return FrameAction(nes_action("DOWN"), "push_0x68_down")
        step = lattice_goto(None, snap, stand, slack=0)
        if step is not None:
            return FrameAction(nes_action(step), "push_lattice")
        waypoints = (
            (_PUSH_LANE_X, y),          # RIGHT to the open east vertical lane
            (_PUSH_LANE_X, _PUSH_NORTH_Y),  # UP the east lane to the north band
            (b0x, _PUSH_NORTH_Y),       # LEFT along the north band to block x
            (b0x, b0y - 4),             # DOWN toward the block's north face
        )
        self.push_frames += 1
        if self.push_frames > 6000:
            return self.mark_fail("l8_mk_0x68_push_blocked")
        wp_i = min(self.push_frames // 400, len(waypoints) - 1)
        wx, wy = waypoints[wp_i]
        if abs(x - wx) <= 5 and abs(y - wy) <= 5:
            wp_i = min(wp_i + 1, len(waypoints) - 1)
            wx, wy = waypoints[wp_i]
        # last waypoint: commit to a straight DOWN push through the block
        if wp_i == len(waypoints) - 1:
            if abs(x - b0x) > 4:
                return FrameAction(
                    nes_action("RIGHT" if x < b0x else "LEFT"), "push_align_x"
                )
            return FrameAction(nes_action("DOWN"), "push_0x68_down")
        if abs(x - wx) > 4:
            return FrameAction(
                nes_action("RIGHT" if x < wx else "LEFT"), "push_route_x"
            )
        if abs(y - wy) > 4:
            return FrameAction(
                nes_action("DOWN" if y < wy else "UP"), "push_route_y"
            )
        return FrameAction(nes_idle_action(), "push_route_wait")

    def _stairs_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        # ROM stair cell first (the L9 0x61 diamond shape): the fixed
        # (128,141) goto held (96,141) for 34000f from a different push pose
        # on the power-on gathered spine.
        rom = stairs_step(None, snap)
        if rom is not None:
            return FrameAction(nes_action(rom), "rom_stairs")
        tx, ty = STAIRS_STAND_CENTER
        step = self._goto(snap, tx, ty, "revealed_stairs")
        if step is not None:
            return step
        return FrameAction(nes_action("UP"), "revealed_stairs_hold_up")

    def _cellar_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        mk = self._mk()
        if not self.mk_gained and mk >= 1 and mk > int(self.mk_before or 0):
            self.mk_gained = True
            self._note(f"magic_key_{mk}_in_cellar")

        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.screen == STAIRS_ROOM_1F:
                if self.mk_gained:
                    return FrameAction(nes_idle_action(), "cellar_return_settled")
                # bounced straight back off the entry stairs without the key
                return self.mark_fail("l8_mk_cellar_rewarped_no_key")
            return self.mark_fail(
                f"l8_mk_cellar_left_to_play_0x{snap.screen:02x}"
            )
        if snap.mode in WAIT_SCROLL_B or snap.transitioning:
            return FrameAction(nes_idle_action(), f"cellar_settle_{snap.mode}")
        if snap.mode != PASSAGE_MODE:
            return FrameAction(nes_idle_action(), f"cellar_wait_mode_{snap.mode}")

        self.cellar_frames += 1

        # key taken -> the two-ladder return navigator does the rest
        # (probe_l8_0f_cellar_return / magic_key_cellar_return_step).
        if self.mk_gained:
            return magic_key_cellar_return_step(snap)

        # Before the key: the centre stairs Link just rode down are a warp
        # tile -- staying on them re-triggers back to play 0x1F.  The pad at
        # x=136 has brick to its south, so a straight edge-east stalls and
        # the key never spawns.  Walk the full DOWN->RIGHT->UP->LEFT loop
        # around the pit, mirroring probe_l8_1f_magic_key._cellar_walk
        # (E2 recording: MK 0->1 at f567, key at ~(141,141)).
        if self.cellar_entry_x is None:
            self.cellar_entry_x = int(snap.link_x)
        waypoints = (
            (self.cellar_entry_x, MK_CELLAR_FLOOR_Y),
            (MK_CELLAR_EAST_X, MK_CELLAR_FLOOR_Y),
            (MK_CELLAR_EAST_X, MK_CELLAR_PEDESTAL[1]),
            MK_CELLAR_PEDESTAL,
        )
        wp_i = min(self.cellar_frames // 180, len(waypoints) - 1)
        wx, wy = waypoints[wp_i]
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - wx) <= 5 and abs(y - wy) <= 5:
            wp_i = min(wp_i + 1, len(waypoints) - 1)
            wx, wy = waypoints[wp_i]
        if abs(y - wy) > 4:
            return FrameAction(
                nes_action("UP" if y > wy else "DOWN"), "cellar_pickup_y"
            )
        if abs(x - wx) > 4:
            return FrameAction(
                nes_action("RIGHT" if x < wx else "LEFT"), "cellar_pickup_x"
            )
        return FrameAction(nes_idle_action(), "cellar_pickup_wait")

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        # In the cellar legs mode 9 must reach ``policy`` (HopController.guard
        # would treat a non-play mode as a scroll wait forever).
        if self.phase in ("cellar", "return") and snap.mode == PASSAGE_MODE:
            if self.success:
                return FrameAction(nes_idle_action(), "done")
            if self.failed or self.frames >= self.max_frames:
                self.failed = True
                self._note(self.timeout_note(snap))
                return FrameAction(nes_idle_action(), "timeout")
            return None
        return HopController.guard(self, snap)

    def _maybe_enter_cellar(self, snap: ZeldaSnapshot) -> None:
        if (
            self.phase in ("push", "stairs")
            and snap.mode == PASSAGE_MODE
            and snap.screen == CELLAR_ROOM_0F
        ):
            self.phase = "cellar"

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self._maybe_enter_cellar(snap)
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "phase": self.phase,
            "magic_key_before": self.mk_before,
            "magic_key_gained": self.mk_gained,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "STAIRS",
            "leftover": dict(self.leftover),
        }


def make_magic_key_stairs_live_controller() -> Level8MagicKeyStairsController:
    return Level8MagicKeyStairsController()


__all__ = [
    "BLUE_GOHMA_ARROWS_REQUIRED",
    "CELLAR_ROOM_0F",
    "GOHMA_BODY_CONTACT",
    "GOHMA_BODY_TYPES_1E",
    "GOHMA_DEST_1F",
    "GOHMA_DOOR_LIP_Y",
    "GOHMA_ROOM_1E",
    "COLUMN_X_MAX",
    "COLUMN_X_MIN",
    "INLAND_X_MAX",
    "INLAND_X_MIN",
    "STAND_Y",
    "STAIRS_ROOM_1F",
    "Level8BlueGohma1EController",
    "Level8MagicKeyStairsController",
    "make_blue_gohma_1e_controller",
    "make_magic_key_stairs_live_controller",
]
