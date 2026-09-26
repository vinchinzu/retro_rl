"""Level 7 room 0x1C: whistle-shrink FORCED_DIGDOGGER, sword, north 0x0C.

Leftover-relative HopController: door-row leftover binds the whistle stand;
dest is RAM (type ``0x38`` gone / shrunk ``0x18`` dead / play ``0x0C``).
Pause-select recorder B-slot 5 (cycle past Red Candle=4; no ``$0656`` poke).
Blow B until RAM shrinks, sword the bodies, UP. No ``idle(n)`` settle hop.

Do not call ``level5.boss_path.fight_digdogger`` (it mutates env). OccupancyWalker
is banned. No RAM writes. ``route_eligible`` stays False.
"""

from __future__ import annotations

from zelda_i.walk.physics import OPPOSITE
from zelda_i.dungeon.passage import passage_step
from zelda_i.dungeon.threat import ReactiveEvader
from zelda_i.dungeon.tracking import ObjectTracker
from zelda_i.overworld.common import walk_or_swing

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.dungeon.behaviors import (
    DIGDOGGER_SHRUNK_TYPE,
    DIGDOGGER_TYPE,
    EnemyKind,
    engagement_hint,
)
from zelda_i.dungeon.door_hop import hop_leftover
from zelda_i.dungeon.hop_controller import (
    WAIT_SCROLL_B,
    HopController,
    axis_dir,
    dungeon_align_then_push,
)
from zelda_i.level7.graph import (
    CANDLE_PUSH,
    FORCED_DIGDOGGER,
    GORIYA_PRE_HUNGRY,
    LEVEL7_ROOM_BY_ID,
)
from zelda_i.level7.path import (
    DOOR_Y_TOL,
    EAST_DOOR_X,
    EAST_DOOR_Y,
    ENTRY_SCREEN,
    NORTH_DOOR_X,
    NORTH_X_TOL,
    ROOM_49,
    ROOM_69,
)
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "DEST",
    "LEVEL7",
    "ROOM",
    "WHISTLE_B_SLOT",
    "WHISTLE_STAND",
    "DigdoggerPhase",
    "Level7ForcedDigdoggerController",
    "Room4AReturnController",
    "Room1BKeyEastController",
    "Room39LeftController",
    "east_of_room1b_ram_id",
    "make_level7_forced_digdogger_controller",
    "play_of_room4a_ram_id",
    "room_1b_key_east_step",
    "room_39_left_step",
    "room_4a_return_step",
    "west_of_room39_ram_id",
]

LEVEL7 = 7
ROOM = 0x1C
DEST = 0x0C
WHISTLE_B_SLOT = B_SLOT_RECORDER
WHISTLE_STAND = (120, 141)
NORTH_DOOR = (120, 93)
ARRIVE_TOL = 3
BLOW_PULSE_FRAMES = 1
BLOW_WAIT_FRAMES = 240
BLOW_ATTEMPTS = 4
EMPTY_SWORD_FRAMES = 30
DIGDOGGER_MAX_FRAMES = 16000


class DigdoggerPhase(Enum):
    WALK = auto()
    SELECT = auto()
    BLOW = auto()
    SWORD = auto()
    EXIT = auto()
    DONE = auto()
    FAILED = auto()


def _boss_types(snap: ZeldaSnapshot) -> set[int]:
    return {
        int(obj.type_id) & 0xFF
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12
        and (int(obj.type_id) & 0xFF) in (DIGDOGGER_TYPE, DIGDOGGER_SHRUNK_TYPE)
    }


def _large(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12 and (int(obj.type_id) & 0xFF) == DIGDOGGER_TYPE
    ]


def _shrunk_live(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12
        and (int(obj.type_id) & 0xFF) == DIGDOGGER_SHRUNK_TYPE
        and int(obj.hp) > 0
    ]


def _live_digdogger(snap: ZeldaSnapshot) -> bool:
    return bool(_large(snap) or _shrunk_live(snap))


@dataclass(kw_only=True)
class Level7ForcedDigdoggerController(HopController):
    """Kill live ``$EB=0x1C`` Digdogger. Dest is RAM (dead + play ``0x0C``).

    Leftover pose binds the whistle stand on the door row; dest is shrink /
    body-gone / room change, not a scripted 12×B or ``idle(n)`` settle.
    """

    spec_id: str = "level7_forced_digdogger"
    max_frames: int = DIGDOGGER_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "dest_0x0c"
    dest: int = DEST
    room: int = ROOM
    stand: tuple[int, int] = WHISTLE_STAND
    phase: DigdoggerPhase = DigdoggerPhase.WALK
    phase_frames: int = 0
    saw_boss: bool = False
    saw_large: bool = False
    shrunk: bool = False
    killed: bool = False
    blow_presses: int = 0
    blow_attempts: int = 0
    sword_frames: int = 0
    empty_sword_frames: int = 0
    selected_before: int | None = None
    leftover: dict[str, Any] = field(default_factory=dict)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)
    _stand_bound: bool = field(default=False, init=False, repr=False)
    _tracker: ObjectTracker = field(default_factory=ObjectTracker, init=False, repr=False)
    _evader: ReactiveEvader = field(
        default_factory=lambda: ReactiveEvader(bounds=(56, 200, 93, 173)),
        init=False,
        repr=False,
    )
    tracked: tuple[Any, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(
            want=WHISTLE_B_SLOT, name="recorder"
        )

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    def _set_phase(self, phase: DigdoggerPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _observe(self, snap: ZeldaSnapshot) -> None:
        self.tracked = self._tracker.observe(snap)
        types = _boss_types(snap)
        if DIGDOGGER_TYPE in types:
            self.saw_boss = True
            self.saw_large = True
        if DIGDOGGER_SHRUNK_TYPE in types:
            self.saw_boss = True
            self.shrunk = True

    def _bind_stand(self, snap: ZeldaSnapshot) -> None:
        if self._stand_bound:
            return
        if snap.mode != PLAY_MODE or snap.transitioning:
            return
        if int(snap.screen) != self.room:
            return
        y = int(snap.link_y)
        stand_y = (
            y if abs(y - WHISTLE_STAND[1]) <= ARRIVE_TOL else WHISTLE_STAND[1]
        )
        self.stand = (WHISTLE_STAND[0], stand_y)
        self._stand_bound = True

    def _walk_to(
        self, snap: ZeldaSnapshot, dest: tuple[int, int], reason: str
    ) -> FrameAction | None:
        direction = axis_dir(
            (int(snap.link_x), int(snap.link_y)), dest, y_first=True, tol=ARRIVE_TOL
        )
        if direction is None:
            return None
        axis = "y" if direction in ("UP", "DOWN") else "x"
        return FrameAction(nes_action(direction), f"{reason}_{axis}")

    def _begin_blow(self, note: str) -> FrameAction:
        self.blow_attempts += 1
        self.blow_presses = 1
        self.phase = DigdoggerPhase.BLOW
        self.phase_frames = 0
        if note:
            self._note(note)
        return FrameAction(nes_action("B"), "whistle_blow")

    def _run_select(self, snap: ZeldaSnapshot) -> FrameAction:
        action = self._select.drive(snap)
        for note in self._select.notes:
            self._note(note)
        if self._select.failed:
            return self.mark_fail(self._select.fail_reason)
        if action is None:
            if (
                snap.level != LEVEL7
                or snap.mode != PLAY_MODE
                or int(snap.screen) != self.room
            ):
                return self.mark_fail("pause_close_contract_mismatch")
            return self._begin_blow("recorder_ready")
        return action

    def _sword(self, snap: ZeldaSnapshot) -> FrameAction:
        self.sword_frames += 1
        evade = self._evader.decide(snap, self.tracked)
        if evade is not None and getattr(evade, "source_type", None) == 0x55:
            return FrameAction(nes_action(evade.direction), "sword_evade_fb")
        live = _shrunk_live(snap)
        if not live:
            if _large(snap):
                return self._begin_blow("large_still_present")
            self.empty_sword_frames += 1
            if self.empty_sword_frames >= EMPTY_SWORD_FRAMES:
                self.killed = True
                self._set_phase(DigdoggerPhase.EXIT, "digdogger_killed")
                return self._exit_north(snap)
            return FrameAction(nes_idle_action(), "wait_shrunk_slots")
        self.empty_sword_frames = 0
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "sword_idle")
        hint = engagement_hint(EnemyKind.DIGDOGGER, snap, target)
        if should_swing_at(
            snap.link_x, snap.link_y, hint.face, (target,), hint=hint
        ):
            if (self.sword_frames % 8) < 4:
                return FrameAction(nes_action(hint.face, "A"), "sword_swing")
            return FrameAction(nes_idle_action(), "sword_hold")
        if hint.retreat:
            back = OPPOSITE.get(hint.face, "DOWN")
            return FrameAction(nes_action(back), "sword_retreat")
        if (self.sword_frames % 8) < 4:
            return FrameAction(nes_action(hint.face, "A"), "sword_chase_swing")
        return FrameAction(nes_action(hint.face), "sword_chase")

    def _exit_north(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.link_y >= 100:
            evade = self._evader.decide(snap, self.tracked)
            if evade is not None and evade.direction != "UP":
                return FrameAction(
                    nes_action(evade.direction), f"north_evade_{evade.reason}"
                )
        return dungeon_align_then_push(
            snap,
            push_dir="UP",
            target_x=NORTH_DOOR[0],
            x_tol=ARRIVE_TOL,
            reason="north",
        )

    def _kill_edge(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.saw_boss
            and self.shrunk
            and not _live_digdogger(snap)
        )

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        dest_play = (
            int(snap.level) == LEVEL7
            and int(snap.screen) == self.dest
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )
        return dest_play and self._kill_edge(snap)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        self.killed = True
        self._set_phase(DigdoggerPhase.DONE, "dest_0x0c")
        return (
            f"dest_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_tf={snap.triforce:02x}"
        )

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_phase={self.phase.name}"
            f"_shrunk={int(self.shrunk)}_killed={int(self.killed)}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        if self.phase is DigdoggerPhase.EXIT or self.killed:
            return FrameAction(nes_action("UP"), "north_scroll")
        return FrameAction(nes_idle_action(), "wait_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        del force
        self.leftover = {
            **hop_leftover(snap),
            "level": int(snap.level),
        }
        return action

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self.phase_frames += 1
        waited = self.wait_not_play(snap)
        if waited is not None:
            return waited
        if int(snap.screen) != self.room:
            if int(snap.screen) == self.dest:
                return self.mark_fail("dest_without_kill_edge")
            return self.mark_fail(
                f"left_room_L{snap.level}_0x{snap.screen:02x}"
            )

        if self.phase is DigdoggerPhase.WALK:
            walked = self._walk_to(snap, self.stand, "stand")
            if walked is not None:
                return walked
            self._set_phase(DigdoggerPhase.SELECT, "at_stand")
            return self._run_select(snap)
        if self.phase is DigdoggerPhase.SELECT:
            return self._run_select(snap)
        if self.phase is DigdoggerPhase.BLOW:
            if _shrunk_live(snap):
                self._set_phase(DigdoggerPhase.SWORD, "shrunk")
                return self._sword(snap)
            if self.phase_frames >= BLOW_WAIT_FRAMES:
                if _large(snap):
                    if self.blow_attempts >= BLOW_ATTEMPTS:
                        return self.mark_fail("whistle_did_not_shrink")
                    walked = self._walk_to(snap, self.stand, "restand")
                    if walked is not None:
                        return walked
                    return self._begin_blow("whistle_retry")
                self.shrunk = True
                self.killed = True
                self._set_phase(DigdoggerPhase.EXIT, "large_gone_after_blow")
                return self._exit_north(snap)
            if self.blow_presses < BLOW_PULSE_FRAMES:
                self.blow_presses += 1
                return FrameAction(nes_action("B"), "whistle_blow")
            return FrameAction(nes_idle_action(), "whistle_wait")
        if self.phase is DigdoggerPhase.SWORD:
            return self._sword(snap)
        if self.phase is DigdoggerPhase.EXIT:
            return self._exit_north(snap)
        return FrameAction(nes_idle_action(), "done")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self._observe(snap)
        self._bind_stand(snap)
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.spec_id,
            "live_room": f"0x{self.room:02X}",
            "dest": f"0x{self.dest:02X}",
            "phase": self.phase.name,
            "saw_boss": self.saw_boss,
            "shrunk": self.shrunk,
            "killed": self.killed,
            "stand": list(self.stand),
            "cursor_moves": self.cursor_moves,
            "blow_attempts": self.blow_attempts,
            "selected_before": self.selected_before,
            "leftover": dict(self.leftover) if self.leftover else None,
            "writes": 0,
            "normal_pause_input": True,
            "route_eligible": False,
            "notes": list(self.notes),
        }


def make_level7_forced_digdogger_controller() -> Level7ForcedDigdoggerController:
    """Fresh 0x1C whistle-shrink + north 0x0C controller (never share instances)."""
    return Level7ForcedDigdoggerController()


ROOM_4A = 0x4A

# 0x39 (DIGDOGGER_2): entry (120,205) S mouth. LEFT door is OPEN on spawn —
# skip the 0x38 fight. Rise the centre column to y=141, hold LEFT. Dest live
# $EB=0x38 (GORIYA_PRE_HUNGRY). 2/2 (39_left_v2/v3). Do not hug SW statue.
ROOM_39 = 0x39
ROOM_39_DOOR_Y = 141
ROOM_39_WEST_PLANE = 16
ROOM39_LEFT_MAX_FRAMES = 4000


def west_of_room39_ram_id() -> int | None:
    """Live ``$EB`` of the room west of ``0x39`` (GORIYA_PRE_HUNGRY), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_PRE_HUNGRY].ram_id

def room_39_left_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x39 south mouth → OPEN west door (skip Digdogger).

    Stay on the centre column until ``y=141`` — the SW statue boxes
    ``(48,189)``. Then hold LEFT on the door row.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("LEFT"), "left39_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "left39_arrived")
    if snap.screen != ROOM_39:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_39_DOOR_Y + DOOR_Y_TOL:
        if abs(x - NORTH_DOOR_X) > NORTH_X_TOL:
            btn = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(btn), "left39_center")
        return FrameAction(nes_action("UP"), "left39_rise")
    if y < ROOM_39_DOOR_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "left39_drop")
    return dungeon_align_then_push(
        snap,
        push_dir="LEFT",
        target_y=ROOM_39_DOOR_Y,
        y_tol=DOOR_Y_TOL,
        door_plane=ROOM_39_WEST_PLANE,
        reason="left39",
    )


@dataclass(kw_only=True)
class Room39LeftController(HopController):
    """0x39 (DIGDOGGER_2) south mouth → OPEN west door to live dest 0x38.

    Skips the Digdogger fight.  2/2 (recordings/39_left_v2/v3.json).
    Recon-wired only.
    """

    spec_id: str = "level7_room39_left"
    max_frames: int = ROOM39_LEFT_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x39_west"
    dest: int | None = field(default_factory=west_of_room39_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_49, ROOM_39}
        ):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("LEFT"), "left39_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_39_left_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_49:
                return self.mark_fail("south_backtrack")
            return self.mark_fail(action.reason)
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "LEFT",
        }


ROOM_4A_WEST_X = 48
ROOM_4A_EAST_COL = 176
ROOM_4A_FLOOR_Y = 189
ROOM_4A_ALIGN = 4
ROOM4A_RETURN_MAX_FRAMES = 4000


def play_of_room4a_ram_id() -> int | None:
    """Live ``$EB`` of CANDLE_PUSH after the 0x4A west-ladder stairs return."""
    return LEVEL7_ROOM_BY_ID[CANDLE_PUSH].ram_id


def room_4a_return_step(snap: ZeldaSnapshot) -> FrameAction:
    """One-frame 0x4A stairs return: east drop, floor west, west-ladder UP."""
    return passage_step(
        snap, east_x=ROOM_4A_EAST_COL, align=ROOM_4A_ALIGN, drop=("LEFT", "DOWN")
    )


@dataclass(kw_only=True)
class Room4AReturnController(HopController):
    """0x4A cellar leftover → west-ladder stairs → live play ``0x1A``.

    2/2 (4a_ret_v7/v8).  Recon-wired only.  ``route_eligible=false``.
    """

    spec_id: str = "level7_room4a_return"
    max_frames: int = ROOM4A_RETURN_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x4a_stairs"
    dest: int | None = field(default_factory=play_of_room4a_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.dest is not None
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_4A and snap.mode != 9:
            if self.dest is not None and snap.screen == self.dest:
                return FrameAction(nes_idle_action(), "wait_dest")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        return room_4a_return_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "STAIRS",
        }


ROOM_1B = 0x1B
ROOM_1C = 0x1C
ROOM1B_KEY_EAST_MAX_FRAMES = 4000


def east_of_room1b_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x1B`` (FORCED_DIGDOGGER)."""
    return LEVEL7_ROOM_BY_ID[FORCED_DIGDOGGER].ram_id


def room_1b_key_east_step(snap: ZeldaSnapshot, frames: int = 0) -> FrameAction:
    """One-frame 0x1B y=141 KEY-RIGHT. Goriyas are tanked, not cleared."""
    x, y = int(snap.link_x), int(snap.link_y)
    if abs(y - EAST_DOOR_Y) > DOOR_Y_TOL:
        return FrameAction(
            nes_action("UP" if y > EAST_DOOR_Y else "DOWN"), "keyeast_align_y"
        )
    if x < EAST_DOOR_X - 2:
        return walk_or_swing(frames, "RIGHT", "keyeast_approach", snap)
    return FrameAction(nes_action("RIGHT"), "keyeast_push")


@dataclass(kw_only=True)
class Room1BKeyEastController(HopController):
    """0x1B GORIYA_PRE_DIG KEY-east → live play ``0x1C`` (FORCED_DIGDOGGER).

    2/2 (1b_ke_v2/v3).  Recon-wired only.  Natural key spend 3→2.
    """

    spec_id: str = "level7_room1b_key_east"
    max_frames: int = ROOM1B_KEY_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x1b_east"
    dest: int | None = field(default_factory=east_of_room1b_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.dest is not None
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "keyeast_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_1B:
            if self.dest is not None and snap.screen == self.dest:
                return FrameAction(nes_idle_action(), "wait_dest")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        return room_1b_key_east_step(snap, self.frames)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "RIGHT",
        }
