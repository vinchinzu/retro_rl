"""Level 7 pre-boss 0x29 east BOMB wall → live Aquamentus 0x2A.

Geometry from ``scratch/probe_l7_7b_cellar_cross.py`` (``EAST_BOMB_STAND`` /
``EAST_BOMB_APPROACH``). Configures ``BombWallController``; no new phase
machine. Factory does not write bombs — Survival top-up stays the
probe/spine's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
from zelda_i.level7.cellar import DEST_ROOM
from zelda_i.level7.path import Level7BombWall
from zelda_i.level7.stairs import AQUAMENTUS_ROM
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "L7_ROOM29_EAST_APPROACH",
    "L7_ROOM29_EAST_BOMB",
    "make_room29_east_bomb_controller",
]

LEVEL7 = 7

# 0x29 PRE_BOSS (cellar DEST_ROOM / graph ram_id) east BOMB wall -> live 0x2A.
# South-around from the cellar leftover: (96,189)->(208,189)->(208,141) face
# RIGHT. Probe ``EAST_BOMB_STAND`` / ``EAST_BOMB_APPROACH``.
L7_ROOM29_EAST_BOMB = Level7BombWall(
    room=DEST_ROOM, stand=(208, 141), face="RIGHT", opens_to=AQUAMENTUS_ROM
)
L7_ROOM29_EAST_APPROACH = ((96, 189), (208, 189), (208, 141))


@dataclass
class Room29EastBombController(BombWallController):
    """Let the Goriya wave leave the cellar landing before crossing south."""

    wave_settle_frames: int = 40
    wave_frames: int = 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.wave_frames < self.wave_settle_frames and snap.mode != 17:
            self.wave_frames += 1
            return FrameAction(nes_idle_action(), "room29_wave_settle")
        return super().step(snap)


def make_room29_east_bomb_controller() -> Room29EastBombController:
    """0x29 PRE_BOSS east BOMB wall → live 0x2A Aquamentus.

    South-around to stand (208,141) face RIGHT. Needs bombs + bomb on B.
    Factory does not write bombs. Recon-wired only.
    """
    return Room29EastBombController(
        wall=L7_ROOM29_EAST_BOMB,
        level=LEVEL7,
        approach_waypoints=L7_ROOM29_EAST_APPROACH,
        approach_tol=4,
        select_item=B_SLOT_BOMBS,
    )
