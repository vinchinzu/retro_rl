"""Glass Joe bout policy for Mike Tyson's Punch-Out!! (NES).

Verified recipe (Match1 → M3 bout win):

1. **Wait** for Glass Joe's Vive La France backup (``opp_pattern_set == 150`` only).
2. **Taunt counter**: left face jabs (A, 2 on / 3 off) → knockdown.
3. **Get-up**: 2-frame A / idle / 2-frame B / idle mash (not hold).
4. **Survive**: on attack act change, wait ~32f then 5f LEFT/RIGHT pulse.
5. **Counter**: short left-jab burst after a successful dodge.

Priority each frame (after KD / get-up / between edge updates):

    get-up → between → watch count → taunt jab → timed dodge → counter → idle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    FIGHT_BETWEEN,
    FIGHT_IN_RING,
    is_taunt_window,
)

# Acts that deal Mac damage on Glass Joe (probed post-KD1 idle timeline).
# Hit lands ~40–58 frames after act enters these ids (timer often stuck at 4).
ATTACK_ACTS = frozenset({4, 6, 7, 10, 13, 17, 20, 23})
# Acts that dodge LEFT (others dodge RIGHT) in the probed window.
DODGE_LEFT_ACTS = frozenset({7, 13, 17, 23})

DODGE_WAIT = 32
DODGE_HOLD = 5
COUNTER_FRAMES = 28


class BoutMode(Enum):
    """Coarse bout phase. Dodge/counter timing lives in dedicated timers, not modes."""

    STAND = auto()  # in-ring standing: idle / dodge / counter (priority stack)
    PUNCH_TAUNT = auto()
    WATCH_KD = auto()
    GETUP = auto()
    BETWEEN = auto()


@dataclass
class GlassJoePolicy:
    """Taunt-counter + timed dodge/counter + get-up policy for Glass Joe."""

    mode: BoutMode = BoutMode.STAND
    mode_t: int = 0
    phase: int = 0
    getup_g: int = 0
    opp_kd: int = 0
    mac_kd: int = 0
    hits: int = 0
    _prev_opp: int = 96
    _prev_mac: int = 96
    _prev_opp0: bool = False
    _prev_fight: int = FIGHT_IN_RING
    _prev_act: int = -1
    _dodge_wait: int = 0
    _dodge_hold: int = 0
    _dodge_side: str = "LEFT"
    _counter_t: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def tick(self, ram) -> FrameAction:
        """Choose one frame of input and update knockdown counters."""
        opp = int(ram[ADDR_OPP_HEALTH])
        mac = int(ram[ADDR_HEALTH])
        fight = int(ram[ADDR_FIGHT_FLAG])
        act = int(ram[ADDR_OPP_ACTION])
        taunt = is_taunt_window(ram)
        opp0 = opp == 0

        if opp < self._prev_opp:
            self.hits += 1
        if opp0 and not self._prev_opp0:
            self.opp_kd += 1
            self._enter(BoutMode.WATCH_KD)
            self._clear_combat()
        self._prev_opp0 = opp0

        if mac == 0 and self._prev_mac > 0:
            self.mac_kd += 1
            self._enter(BoutMode.GETUP)
            self.getup_g = 0
            self._clear_combat()
        if mac > 0 and self._prev_mac == 0:
            self._enter(BoutMode.STAND)
        self._prev_mac = mac
        self._prev_opp = opp

        if fight != self._prev_fight:
            if fight == FIGHT_BETWEEN:
                self._enter(BoutMode.BETWEEN)
                self._clear_combat()
            elif self._prev_fight == FIGHT_BETWEEN and fight == FIGHT_IN_RING:
                # Glass Joe backups each round — re-arm standing wait.
                self._enter(BoutMode.STAND)
                self._clear_combat()
            self._prev_fight = fight

        self.mode_t += 1

        # Always track act while Mac is up in-ring so rise-from-KD cannot false-arm.
        if fight == FIGHT_IN_RING and mac > 0:
            standing = self.mode in (BoutMode.STAND, BoutMode.PUNCH_TAUNT)
            if standing and not opp0 and act != self._prev_act and act in ATTACK_ACTS and not taunt:
                self._arm_dodge(act)
            self._prev_act = act

        if self.mode == BoutMode.STAND and taunt:
            self._enter(BoutMode.PUNCH_TAUNT)
            self._clear_combat()
        if self.mode == BoutMode.WATCH_KD and not opp0 and opp > 0:
            self._enter(BoutMode.STAND)
        if self.mode == BoutMode.PUNCH_TAUNT and self.mode_t > 200 and not opp0:
            self._enter(BoutMode.STAND)

        act_out = self._action(taunt=taunt, mac=mac, fight=fight)
        self.reasons[act_out.reason] = self.reasons.get(act_out.reason, 0) + 1
        return act_out

    def _enter(self, mode: BoutMode) -> None:
        self.mode = mode
        self.mode_t = 0

    def _clear_combat(self) -> None:
        self._dodge_wait = 0
        self._dodge_hold = 0
        self._counter_t = 0

    def _arm_dodge(self, act: int) -> None:
        self._dodge_wait = DODGE_WAIT
        self._dodge_hold = 0  # armed only when wait completes
        self._dodge_side = "LEFT" if act in DODGE_LEFT_ACTS else "RIGHT"
        self._counter_t = 0

    def _action(self, *, taunt: bool, mac: int, fight: int) -> FrameAction:
        # Priority stack — order is the design.
        if self.mode == BoutMode.GETUP or (mac == 0 and fight == FIGHT_IN_RING):
            return self._getup()
        if self.mode == BoutMode.BETWEEN:
            return self._between()
        if self.mode == BoutMode.WATCH_KD:
            return self._dodge_pulse("watch_kd")

        # Taunt offense beats dodge/counter (also catch taunt while mid-dodge).
        if taunt or self.mode == BoutMode.PUNCH_TAUNT:
            if self.mode != BoutMode.PUNCH_TAUNT:
                self._enter(BoutMode.PUNCH_TAUNT)
                self._clear_combat()
            return self._jab(self.mode_t, "taunt_a", "taunt_rec")

        if self._dodge_wait > 0 or self._dodge_hold > 0:
            return self._do_dodge()

        if self._counter_t > 0:
            return self._counter_jab()

        return FrameAction(nes_idle_action(), "stand_idle")

    def _jab(self, t: int, on: str, rec: str) -> FrameAction:
        if t % 5 < 2:
            return FrameAction(nes_action("A"), on)
        return FrameAction(nes_idle_action(), rec)

    def _do_dodge(self) -> FrameAction:
        if self._dodge_wait > 0:
            self._dodge_wait -= 1
            if self._dodge_wait == 0:
                self._dodge_hold = DODGE_HOLD
            return FrameAction(nes_idle_action(), "dodge_wait")
        if self._dodge_hold > 0:
            self._dodge_hold -= 1
            if self._dodge_hold == 0:
                self._counter_t = COUNTER_FRAMES
            side = self._dodge_side
            return FrameAction(nes_action(side), f"dodge_{side.lower()}")
        return FrameAction(nes_idle_action(), "dodge_idle")

    def _counter_jab(self) -> FrameAction:
        if self._counter_t <= 0:
            return FrameAction(nes_idle_action(), "counter_done")
        self._counter_t -= 1
        t = COUNTER_FRAMES - self._counter_t
        return self._jab(t, "counter_a", "counter_rec")

    def _getup(self) -> FrameAction:
        g = self.getup_g
        self.getup_g += 1
        # A,A,idle,B,B,idle — required; single-frame mash does not get up.
        if g % 6 in (0, 1):
            return FrameAction(nes_action("A"), "getup_a")
        if g % 6 in (3, 4):
            return FrameAction(nes_action("B"), "getup_b")
        return FrameAction(nes_idle_action(), "getup_idle")

    def _between(self) -> FrameAction:
        t = self.mode_t % 10
        if t < 2:
            return FrameAction(nes_action("A"), "between_a")
        if t < 4:
            return FrameAction(nes_action("START"), "between_start")
        return FrameAction(nes_idle_action(), "between_wait")

    def _dodge_pulse(self, prefix: str) -> FrameAction:
        """Short L/R pulses used only during knockdown count watch."""
        s = self.phase % 12
        self.phase += 1
        if s < 3:
            return FrameAction(nes_action("LEFT"), f"{prefix}_left")
        if 6 <= s < 9:
            return FrameAction(nes_action("RIGHT"), f"{prefix}_right")
        return FrameAction(nes_idle_action(), f"{prefix}_idle")
