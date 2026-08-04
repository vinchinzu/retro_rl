"""Glass Joe bout policy for Mike Tyson's Punch-Out!! (NES).

Verified pieces (from Match1 / Level1 probes):

1. **Wait** for fight clock, then for Glass Joe's Vive La France backup
   (``opp_pattern_set == 150``).
2. **Taunt counter**: left face jabs (A) during the backup window → knockdown
   (~0:42 R1, ~0:32 R2).
3. **Get-up**: 2-frame A / idle / 2-frame B / idle mash (not hold).
4. **Survive**: alternating LEFT/RIGHT dodge pulses (3 on / 3 off).

Full bout win (3 opp knockdowns or decision) still needs stronger post-KD2
offense; this policy reliably scores knockdown #1 and often #2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_KNOCKDOWN,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    ADDR_ROUND,
    FIGHT_BETWEEN,
    FIGHT_IN_RING,
    hearts,
    is_taunt_window,
)


class BoutMode(Enum):
    WAIT_TAUNT = auto()
    PUNCH_TAUNT = auto()
    WATCH_KD = auto()
    SURVIVE = auto()
    GETUP = auto()
    BETWEEN = auto()


@dataclass
class GlassJoePolicy:
    """Taunt-counter + dodge/get-up policy for the Glass Joe bout."""

    mode: BoutMode = BoutMode.WAIT_TAUNT
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
    reasons: dict[str, int] = field(default_factory=dict)

    def tick(self, ram) -> FrameAction:
        """Choose one frame of input and update knockdown counters."""
        opp = int(ram[ADDR_OPP_HEALTH])
        mac = int(ram[ADDR_HEALTH])
        fight = int(ram[ADDR_FIGHT_FLAG])
        kd = int(ram[ADDR_KNOCKDOWN])
        h = hearts(ram)
        taunt = is_taunt_window(ram)
        opp0 = opp == 0

        if opp < self._prev_opp:
            self.hits += 1
        if opp0 and not self._prev_opp0:
            self.opp_kd += 1
            self.mode = BoutMode.WATCH_KD
            self.mode_t = 0
        self._prev_opp0 = opp0

        if mac == 0 and self._prev_mac > 0:
            self.mac_kd += 1
            self.mode = BoutMode.GETUP
            self.getup_g = 0
            self.mode_t = 0
        if mac > 0 and self._prev_mac == 0:
            self.mode = BoutMode.SURVIVE
            self.mode_t = 0
        self._prev_mac = mac
        self._prev_opp = opp

        if fight != self._prev_fight:
            if fight == FIGHT_BETWEEN:
                self.mode = BoutMode.BETWEEN
                self.mode_t = 0
            elif self._prev_fight == FIGHT_BETWEEN and fight == FIGHT_IN_RING:
                self.mode = (
                    BoutMode.WAIT_TAUNT if self.opp_kd < 2 else BoutMode.SURVIVE
                )
                self.mode_t = 0
            self._prev_fight = fight

        self.mode_t += 1
        if self.mode == BoutMode.WAIT_TAUNT and taunt:
            self.mode = BoutMode.PUNCH_TAUNT
            self.mode_t = 0
        if self.mode == BoutMode.WATCH_KD and (not opp0) and opp > 0:
            self.mode = BoutMode.SURVIVE
            self.mode_t = 0
        if self.mode == BoutMode.PUNCH_TAUNT and self.mode_t > 200 and not opp0:
            self.mode = BoutMode.SURVIVE
            self.mode_t = 0

        act = self._action(ram, taunt=taunt, mac=mac, fight=fight, hearts=h)
        self.reasons[act.reason] = self.reasons.get(act.reason, 0) + 1
        return act

    def _action(self, ram, *, taunt: bool, mac: int, fight: int, hearts: int) -> FrameAction:
        if self.mode == BoutMode.GETUP or (mac == 0 and fight == FIGHT_IN_RING):
            return self._getup()
        if self.mode == BoutMode.BETWEEN:
            return self._between()
        if self.mode == BoutMode.WAIT_TAUNT:
            return FrameAction(nes_idle_action(), "wait_taunt")
        if self.mode == BoutMode.WATCH_KD:
            # Dodge during the count so the first post-rise punch is covered.
            return self._spam_lr("watch_kd")
        if self.mode == BoutMode.PUNCH_TAUNT:
            if self.mode_t % 5 < 2:
                return FrameAction(nes_action("A"), "taunt_a")
            return FrameAction(nes_idle_action(), "taunt_rec")
        # SURVIVE
        if taunt and self.opp_kd < 2:
            if self.phase % 5 < 2:
                self.phase += 1
                return FrameAction(nes_action("A"), "survive_taunt")
            self.phase += 1
            return FrameAction(nes_idle_action(), "survive_taunt_rec")
        return self._spam_lr("survive")

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

    def _spam_lr(self, prefix: str) -> FrameAction:
        s = self.phase % 12
        self.phase += 1
        if s < 3:
            return FrameAction(nes_action("LEFT"), f"{prefix}_left")
        if 6 <= s < 9:
            return FrameAction(nes_action("RIGHT"), f"{prefix}_right")
        return FrameAction(nes_idle_action(), f"{prefix}_idle")
