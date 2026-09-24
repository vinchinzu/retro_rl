"""Patra arms for ``ab_stage.sh`` / A-B evals: the old lane stand vs ``PatraAim``. Scratch."""
from dataclasses import dataclass

from zelda_i.level9.natural_path import NaturalFinalPatraController
from zelda_i.level9.patra import patra_action


@dataclass
class LaneStandFinalPatra(NaturalFinalPatraController):
    """The pre-rr-e59v policy: body-lane stand, pulse A on the lane."""

    cooldown: int = 0

    def step(self, snap):
        if self.success or self.failed or not self.start_checked or snap.mode == 17:
            return super().step(snap)
        from zelda_i.level9.patra import final_patra_north_door_earned
        from retro_harness.nes import nes_idle_action
        if final_patra_north_door_earned(snap):
            self.success = True
            return self._action(nes_idle_action(), "patra_north_door_earned")
        action, reason, self.cooldown = patra_action(snap, cooldown=self.cooldown, stand_dy=self.stand_dy)
        return self._action(action, reason)


def silver_arrows_measured():
    """The spine's L9 silver-arrows stage (measured post-L8 handoff), for replays."""
    from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
    from zelda_i.level9.natural_path import make_natural_silver_arrows_controller

    return make_natural_silver_arrows_controller(handoff=MEASURED_POST_L8_HANDOFF)
