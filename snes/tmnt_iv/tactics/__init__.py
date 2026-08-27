"""TMNT IV next() tactics and the fight() CombatProfile seam."""

from tmnt_iv.tactics.alleycat import AlleycatPackTactics
from tmnt_iv.tactics.baxter import BaxterTactics
from tmnt_iv.tactics.fight import CombatProfile, fight
from tmnt_iv.tactics.hazards import HazardAvoid, SewerSpikeAvoid
from tmnt_iv.tactics.pizza import PizzaSeek
from tmnt_iv.tactics.raph_air import raph_starbase_jump_action
from tmnt_iv.tactics.recovery import (
    CombatPositionStall,
    PlayerXStallWalk,
    PrehistoricCaveRecovery,
)
from tmnt_iv.tactics.shredder_f2 import SuperShredderForm2Tactics
from tmnt_iv.tactics.slash import SlashTactics
from tmnt_iv.tactics.technodrome import TechnodromeTactics

__all__ = [
    "AlleycatPackTactics",
    "BaxterTactics",
    "CombatPositionStall",
    "CombatProfile",
    "HazardAvoid",
    "PizzaSeek",
    "PlayerXStallWalk",
    "PrehistoricCaveRecovery",
    "SewerSpikeAvoid",
    "SlashTactics",
    "SuperShredderForm2Tactics",
    "TechnodromeTactics",
    "fight",
    "raph_starbase_jump_action",
]
