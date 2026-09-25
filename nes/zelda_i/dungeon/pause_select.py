"""Pause-select an already-owned B-item. Never poke ``$0656``.

START / idle 20 / RIGHT / idle 8 / START close / idle 24, matching the
live L5 ``select_b_item_menu`` timings — but RIGHT only counts as a
cursor move when ``ADDR_SELECTED_ITEM`` actually changes, and the
controller never emits B (the parent blows/places after ``success``).

Snapshot has no ``selected_item``; read ``ADDR_SELECTED_ITEM`` after
``bind_env``.

``PotionDrinkGuard`` wraps any stage controller: at the last heart with a
potion owned it takes the frame, pause-selects the potion, presses B, waits
out the refill and puts the previous B item back. Measured 2026-09-23 on
0x64: the potion is B slot 7 (the letter slot 15), ``$065E`` steps down on
the B frame (red 2 -> blue 1 -> 0), ``$E0`` holds 2 while the hearts fill
(world frozen, mode stays 5) and drops to 0 on the frame they are full,
about 22 + 43 frames per missing heart. START is accepted on that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MENU_STATE,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_WORLD_PAUSED,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

B_SLOT_BOOMERANG = 0
B_SLOT_BOMBS = 1
B_SLOT_ARROWS = 2
B_SLOT_CANDLE = 4
B_SLOT_RECORDER = 5
B_SLOT_BAIT = 6
B_SLOT_POTION = 7
B_SLOT_LETTER = 15

SLOT_NAMES = {
    B_SLOT_BOOMERANG: "boomerang",
    B_SLOT_BOMBS: "bombs",
    B_SLOT_ARROWS: "arrows",
    B_SLOT_CANDLE: "candle",
    B_SLOT_RECORDER: "recorder",
    B_SLOT_BAIT: "bait",
    B_SLOT_POTION: "potion",
    B_SLOT_LETTER: "letter",
}

DEATH_MODE = 17
SELECT_MAX_FRAMES = 240
OPEN_SETTLE_FRAMES = 20
CURSOR_SETTLE_FRAMES = 8
CLOSE_SETTLE_FRAMES = 64  # NES Zelda pause scroll-up takes 59 frames to accept input
MAX_CURSOR_MOVES = 8


class PauseSelectPhase(Enum):
    CHECK = auto()
    OPEN_SETTLE = auto()
    CYCLE = auto()
    CURSOR_SETTLE = auto()
    CLOSE = auto()
    CLOSE_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PauseSelectController:
    """Cycle the pause menu until ``want`` is on B, then close back to play."""

    want: int
    name: str = ""
    max_frames: int = SELECT_MAX_FRAMES
    open_settle: int = OPEN_SETTLE_FRAMES
    cursor_settle: int = CURSOR_SETTLE_FRAMES
    close_settle: int = CLOSE_SETTLE_FRAMES
    max_cursor_moves: int = MAX_CURSOR_MOVES
    phase: PauseSelectPhase = PauseSelectPhase.CHECK
    frames: int = 0
    phase_frames: int = 0
    cursor_moves: int = 0
    success: bool = False
    failed: bool = False
    skipped: bool = False
    fail_reason: str = ""
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _last_selected: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = SLOT_NAMES.get(int(self.want), f"slot_{self.want}")

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: PauseSelectPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _selected(self) -> int | None:
        if self._env is None:
            return None
        return int(read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM))

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self.fail_reason = reason
        self._set_phase(PauseSelectPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, reason: str) -> FrameAction:
        self.success = True
        self._set_phase(PauseSelectPhase.DONE, reason)
        return FrameAction(nes_idle_action(), reason)

    def drive(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """``None`` means the slot is selected and the parent may act now."""
        if self.success:
            return None
        action = self.step(snap)
        if self.failed:
            return action
        if self.success:
            return None
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("pause_select_env_not_bound")
        selected = self._selected()
        if selected is None:
            return self._fail("pause_select_env_not_bound")
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        if self.frames > self.max_frames:
            return self._fail(f"select_{self.name}_timeout")

        if self.phase is PauseSelectPhase.CHECK:
            if selected == self.want:
                self.skipped = True
                return self._finish(f"{self.name}_already_selected")
            self._set_phase(PauseSelectPhase.OPEN_SETTLE, "pause_open")
            return FrameAction(nes_action("START"), "pause_open")

        if self.phase is PauseSelectPhase.OPEN_SETTLE:
            if self.phase_frames >= self.open_settle:
                self._set_phase(PauseSelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_settle")

        if self.phase is PauseSelectPhase.CYCLE:
            if selected == self.want:
                self._set_phase(
                    PauseSelectPhase.CLOSE, f"{self.name}_cursor_selected"
                )
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= self.max_cursor_moves:
                return self._fail(f"{self.name}_cursor_not_found")
            self._last_selected = selected
            self._set_phase(PauseSelectPhase.CURSOR_SETTLE)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")

        if self.phase is PauseSelectPhase.CURSOR_SETTLE:
            if (
                self._last_selected is not None
                and selected != self._last_selected
            ):
                self.cursor_moves += 1
                self._last_selected = selected
            if self.phase_frames >= self.cursor_settle:
                self._set_phase(PauseSelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_cursor_settle")

        if self.phase is PauseSelectPhase.CLOSE:
            self._set_phase(PauseSelectPhase.CLOSE_SETTLE, "pause_close")
            return FrameAction(nes_action("START"), "pause_close")

        if self.phase is PauseSelectPhase.CLOSE_SETTLE:
            if self.phase_frames < self.close_settle:
                return FrameAction(nes_idle_action(), "pause_resume")
            if selected == self.want:
                return self._finish(f"{self.name}_selected_naturally")
            return self._fail("pause_close_contract_mismatch")

        return FrameAction(nes_idle_action(), "select")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "want": self.want,
            "name": self.name,
            "frames": self.frames,
            "phase": self.phase.name,
            "cursor_moves": self.cursor_moves,
            "fail_reason": self.fail_reason,
            "writes": 0,
            "normal_pause_input": True,
            "notes": list(self.notes),
        }


def pause_dropped(select: PauseSelectController, ram: Any) -> bool:
    """True when ``select`` is past its open settle but no menu is up.

    START is dropped on some frames (~40 after a cave's mode 11). The next
    CYCLE step would press RIGHT and walk Link, so a caller that sees this
    discards the select and tries again later.
    """
    cycling = select.phase in (PauseSelectPhase.CYCLE, PauseSelectPhase.CURSOR_SETTLE)
    return cycling and not read_u8(ram, ADDR_MENU_STATE)


def b_slot_owned(ram: Any, slot: int) -> bool:
    """True when B slot ``slot`` holds an item the cursor can land on.

    Slot ``n`` is the inventory byte at ``$0657 + n`` (measured: bombs 1,
    arrows 2, candle 4, recorder 5, bait 6, potion 7 at ``$065E``, letter 15
    at ``$0666``). Arrows also need the bow; slot 0 is the boomerang pair.
    """
    slot = int(slot)
    if slot == B_SLOT_BOOMERANG:
        return bool(read_u8(ram, ADDR_BOOMERANG) or read_u8(ram, ADDR_MAGIC_BOOMERANG))
    if slot == B_SLOT_ARROWS:
        return bool(read_u8(ram, ADDR_ARROWS) and read_u8(ram, ADDR_BOW))
    return bool(read_u8(ram, ADDR_SWORD + slot))


def potion_drink_window(snap: ZeldaSnapshot, *, paused: int, menu: int) -> bool:
    """A frame the guard may open the menu and press B on.

    Settled play only (mode 5 past its init frame): no scroll, stairs, cave
    or passage, nothing already frozen (``$E0``) or open (``$E1``), and
    Link's own slot idle — a swing, a knockback or a text halt is nonzero
    there, and B on such a frame is dropped.
    """
    link = snap.objects[0] if snap.objects else None
    link_busy = link is not None and int(link.slot) == 0 and int(link.state) != 0
    return (
        int(snap.mode) == PLAY_MODE
        and int(snap.is_updating_mode) != 0
        and not snap.transitioning
        and int(paused) == 0
        and int(menu) == 0
        and not link_busy
    )


# Frames a due refill waits for a drink window before the assist writes it.
DRINK_WAIT_BUDGET = 90
# Longest refill: 16 hearts x 43 frames plus the lead-in.
DRINK_BUDGET = 900
DRINK_PRESS_TRIES = 3
DRINK_PRESS_GAP = 4
# After an aborted drink, pass through this long before trying again.
DRINK_COOLDOWN = 120
MENU_CLOSE_BUDGET = 120
# Dropped STARTs one drink tolerates before it gives the frame back.
DRINK_PAUSE_TRIES = 3


class DrinkPhase(Enum):
    IDLE = auto()
    WAIT = auto()
    SELECT = auto()
    PRESS = auto()
    DRINK = auto()
    RESTORE = auto()
    CLOSE = auto()


@dataclass
class PotionDrinkGuard:
    """Drink a real potion at the last heart; otherwise ``inner`` plays.

    No potion, or more than ``drink_at_whole_hearts``, and every frame is
    ``inner.step`` untouched. With a potion at the last heart it waits for
    :func:`potion_drink_window` (``inner`` keeps playing meanwhile), then
    owns the frames: select the potion, B, the refill, the old B item back.
    ``inner`` is not stepped while the guard owns a frame; the world is
    frozen for almost all of them (menu or refill).

    ``holds_refill`` is the Survival assist's ``refill_hold``: True while a
    drink is running, and for up to ``wait_budget`` frames of waiting, so the
    last-heart refill does not spend itself first.
    """

    inner: Any
    drink_at_whole_hearts: int = 1
    wait_budget: int = DRINK_WAIT_BUDGET
    drink_budget: int = DRINK_BUDGET
    phase: DrinkPhase = DrinkPhase.IDLE
    phase_frames: int = 0
    frames: int = 0
    drinks: int = 0
    restores: int = 0
    drink_frames: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController | None = field(default=None, init=False, repr=False)
    _prior: int | None = field(default=None, init=False, repr=False)
    _potion_before: int = field(default=0, init=False, repr=False)
    _presses: int = field(default=0, init=False, repr=False)
    _wait: int = field(default=0, init=False, repr=False)
    _cooldown: int = field(default=0, init=False, repr=False)
    _drink_start: int = field(default=0, init=False, repr=False)
    _drops: int = field(default=0, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _wants(self, snap: ZeldaSnapshot) -> bool:
        return (
            int(snap.potion) > 0
            and int(snap.mode) != DEATH_MODE
            and int(snap.whole_hearts) <= int(self.drink_at_whole_hearts)
        )

    def holds_refill(self, snap: ZeldaSnapshot) -> bool:
        if self._env is None:
            return False
        if self.phase in (DrinkPhase.SELECT, DrinkPhase.PRESS, DrinkPhase.DRINK):
            return True
        if self.phase in (DrinkPhase.IDLE, DrinkPhase.WAIT) and self._cooldown <= 0:
            return self._wants(snap) and self._wait < self.wait_budget
        return False

    def _set(self, phase: DrinkPhase, note: str = "") -> None:
        self.phase = phase
        self.phase_frames = 0
        if note:
            self.notes.append(note)

    def _selector(self, want: int) -> PauseSelectController:
        ctl = PauseSelectController(want=want)
        ctl.bind_env(self._env)
        return ctl

    def _own(self, action: FrameAction) -> FrameAction:
        self.frames += 1
        return action

    def _abort(self, ram: Any, note: str) -> FrameAction:
        self._cooldown = DRINK_COOLDOWN
        if read_u8(ram, ADDR_MENU_STATE):
            self._set(DrinkPhase.CLOSE, note)
            return self._own(FrameAction(nes_action("START"), "potion_menu_close"))
        self._set(DrinkPhase.IDLE, note)
        return self._own(FrameAction(nes_idle_action(), note))

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._env is None:
            return self.inner.step(snap)
        self.phase_frames += 1
        ram = self._env.get_ram()
        if self._cooldown > 0:
            self._cooldown -= 1
        if self.phase is DrinkPhase.IDLE:
            if self._cooldown > 0 or not self._wants(snap):
                self._wait = 0
                return self.inner.step(snap)
            self._set(DrinkPhase.WAIT)
        if self.phase is DrinkPhase.WAIT:
            if not self._wants(snap):
                self._set(DrinkPhase.IDLE)
                self._wait = self._drops = 0
                return self.inner.step(snap)
            window = potion_drink_window(
                snap,
                paused=read_u8(ram, ADDR_WORLD_PAUSED),
                menu=read_u8(ram, ADDR_MENU_STATE),
            )
            if not window:
                self._wait += 1
                return self.inner.step(snap)
            # The item from before the *first* try. An aborted try leaves
            # the potion selected; re-reading it then made the restore a
            # no-op, and 0x5B's burn press drank the last charge (CL63).
            if self._prior is None:
                self._prior = int(read_u8(ram, ADDR_SELECTED_ITEM))
            self._select = self._selector(B_SLOT_POTION)
            self._set(DrinkPhase.SELECT, "potion_select")
        if self.phase is DrinkPhase.SELECT:
            assert self._select is not None
            if pause_dropped(self._select, ram):
                self._select = None
                self._drops += 1
                if self._drops >= DRINK_PAUSE_TRIES:
                    self._drops = 0
                    return self._abort(ram, "potion_pause_never_opened")
                self._set(DrinkPhase.WAIT, "potion_pause_dropped")
                return self._own(FrameAction(nes_idle_action(), "potion_pause_dropped"))
            act = self._select.drive(snap)
            if self._select.failed:
                return self._abort(ram, f"potion_{self._select.fail_reason}")
            if act is not None:
                return self._own(act)
            self._potion_before = int(snap.potion)
            self._presses = 0
            self._set(DrinkPhase.PRESS)
        if self.phase is DrinkPhase.PRESS:
            if int(snap.potion) < self._potion_before or read_u8(ram, ADDR_WORLD_PAUSED):
                self._drink_start = self.frames
                self._set(DrinkPhase.DRINK, "potion_drink")
            elif self._presses >= DRINK_PRESS_TRIES and self.phase_frames > DRINK_PRESS_GAP:
                return self._abort(ram, "potion_b_ignored")
            else:
                free = potion_drink_window(
                    snap,
                    paused=read_u8(ram, ADDR_WORLD_PAUSED),
                    menu=read_u8(ram, ADDR_MENU_STATE),
                )
                if free and (self._presses == 0 or self.phase_frames > DRINK_PRESS_GAP):
                    self._presses += 1
                    self.phase_frames = 0
                    return self._own(FrameAction(nes_action("B"), "potion_press_b"))
                return self._own(FrameAction(nes_idle_action(), "potion_press_wait"))
        if self.phase is DrinkPhase.DRINK:
            if read_u8(ram, ADDR_WORLD_PAUSED) and self.phase_frames <= self.drink_budget:
                return self._own(FrameAction(nes_idle_action(), "potion_refill"))
            if self.phase_frames > self.drink_budget:
                return self._abort(ram, "potion_refill_timeout")
            self.drinks += 1
            self.drink_frames.append(self.frames - self._drink_start)
            prior = self._prior
            selected = int(read_u8(ram, ADDR_SELECTED_ITEM))
            if prior is None or prior == selected or not b_slot_owned(ram, prior):
                self._prior = None
                self._set(DrinkPhase.IDLE, "potion_done")
                return self.inner.step(snap)
            self._select = self._selector(prior)
            self._set(DrinkPhase.RESTORE, "potion_restore_b")
        if self.phase is DrinkPhase.RESTORE:
            assert self._select is not None
            if pause_dropped(self._select, ram):
                self._drops += 1
                if self._drops >= DRINK_PAUSE_TRIES:
                    self._drops = 0
                    return self._abort(ram, "restore_pause_never_opened")
                self._select = self._selector(int(self._prior or 0))
                return self._own(FrameAction(nes_idle_action(), "restore_pause_dropped"))
            act = self._select.drive(snap)
            if self._select.failed:
                return self._abort(ram, f"restore_{self._select.fail_reason}")
            if act is not None:
                return self._own(act)
            self.restores += 1
            self._drops = 0
            self._prior = None
            self._set(DrinkPhase.IDLE, "potion_done")
            return self.inner.step(snap)
        if self.phase is DrinkPhase.CLOSE:
            if read_u8(ram, ADDR_MENU_STATE) and self.phase_frames <= MENU_CLOSE_BUDGET:
                return self._own(FrameAction(nes_idle_action(), "potion_menu_close"))
            self._set(DrinkPhase.IDLE)
            return self.inner.step(snap)
        return self.inner.step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "drinks": self.drinks,
            "restores": self.restores,
            "frames": self.frames,
            "drink_frames": list(self.drink_frames),
            "phase": self.phase.name,
            "notes": list(self.notes[-12:]),
        }


__all__ = [
    "B_SLOT_ARROWS",
    "B_SLOT_BAIT",
    "B_SLOT_BOMBS",
    "B_SLOT_BOOMERANG",
    "B_SLOT_CANDLE",
    "B_SLOT_LETTER",
    "B_SLOT_POTION",
    "B_SLOT_RECORDER",
    "CLOSE_SETTLE_FRAMES",
    "CURSOR_SETTLE_FRAMES",
    "DrinkPhase",
    "MAX_CURSOR_MOVES",
    "OPEN_SETTLE_FRAMES",
    "PauseSelectController",
    "PauseSelectPhase",
    "PotionDrinkGuard",
    "SELECT_MAX_FRAMES",
    "SLOT_NAMES",
    "b_slot_owned",
    "pause_dropped",
    "potion_drink_window",
]
