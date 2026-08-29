"""Segment policy behavior tree for TMNT IV (all stages; Clean path Stage 0).

Clean lessons (do not relearn): ``docs/CLEAN_PLAYBOOK.md``.

Production rules already burned in:

- Pizza is the only Clean heal; full ``PizzaSeek`` is **stage-0-only**.
  Alleycat may **underfoot-pickup** only (no seek path change).
- Walk uses ``pickup_every=0`` — no empty-screen RIGHT+Y stutter.
- Stage 1 wrecking-ball jump-dodge is **offline** in ``Stage1Policy.tick``
  (jump-through caused Clean deaths). ``HazardAvoid`` stays for future
  phase-safe work.
- Baxter uses left standoff + elevated jump-slash; never approach into body.
- Jump-slash only where grounded Y fails; suppress for Slash / Rat King /
  form-2 / Mode-7 depth.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import (
    ActionNode,
    Condition,
    NodeStatus,
    Selector,
    Sequence,
    TickResult,
)
from retro_harness.combat import AttackCadence, WalkProgress
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.tactics.alleycat import AlleycatPackTactics
from tmnt_iv.tactics.baxter import BaxterTactics
from tmnt_iv.tactics.fight import CombatProfile, fight
from tmnt_iv.tactics.hazards import HazardAvoid, SewerSpikeAvoid
from tmnt_iv.tactics.neon import NeonLaneTactics
from tmnt_iv.tactics.pizza import PizzaSeek
from tmnt_iv.tactics.recovery import (
    CombatPositionStall,
    PlayerXStallWalk,
    PrehistoricCaveRecovery,
)
from tmnt_iv.tactics.shredder_f2 import SuperShredderForm2Tactics
from tmnt_iv.tactics.slash import SlashTactics
from tmnt_iv.tactics.technodrome import TechnodromeTactics

# Public seam: scripts import PizzaSeek / HazardAvoid from here.
__all__ = [
    "CombatPositionStall",
    "CombatProfile",
    "HazardAvoid",
    "PizzaSeek",
    "PlayerXStallWalk",
    "PrehistoricCaveRecovery",
    "SewerSpikeAvoid",
    "Stage1Policy",
    "build_stage1_tree",
    "fight",
]

# Default poke cadence; fight() overwrites hold/gap from CombatProfile.
_ATTACK_HOLD = 2
_ATTACK_GAP = 5


def _needs_continue(state: GameState) -> bool:
    """True on continue / KO with lives remaining."""
    if state.mode is GameMode.CONTINUE or state.player_dead:
        return True
    if state.lives <= 0:
        return False
    if state.health == 0 or state.health > 0x60:
        return True
    return False


def _continue_action(state: GameState) -> FrameAction:
    """START on continue; idle through mid-life KO respawn."""
    if state.mode is GameMode.CONTINUE or state.player_dead:
        return FrameAction(action=buttons("START"), reason="continue")
    return FrameAction(action=idle_action(), reason="ko_wait")


def build_stage1_tree(
    *,
    cadence: AttackCadence | None = None,
    walk_progress: PlayerXStallWalk | WalkProgress | None = None,
) -> Selector:
    """Segment policy: continue → clear → fight nearest → walk right."""
    cadence = cadence or AttackCadence(hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP)
    # PizzaSeek handles real pizza boxes; blind RIGHT+Y every N frames while
    # the screen is empty is the ugly "stutter-step attack" on clear walks.
    walk_progress = walk_progress or PlayerXStallWalk(pickup_every=0)

    def fight_action(state: GameState) -> FrameAction:
        return fight(state, cadence)

    def walk_action(state: GameState) -> FrameAction:
        # Mode-7 empty-lane hold; far-band living enemies go through fight()
        # so CombatPositionStall can still overlay a freeze.
        lane = NeonLaneTactics().next(state)
        if lane is not None:
            return lane
        return walk_progress.next(state)

    return Selector(
        [
            Sequence(
                [
                    Condition(_needs_continue, name="needs_continue"),
                    ActionNode(_continue_action, name="handle_continue"),
                ],
                name="continue_seq",
            ),
            Condition(lambda s: s.level_complete, name="level_complete"),
            Sequence(
                [
                    Condition(
                        lambda s: bool(s.living_enemies),
                        name="enemies_present",
                    ),
                    ActionNode(fight_action, name="fight_nearest"),
                ],
                name="fight_seq",
            ),
            ActionNode(walk_action, name="walk_right"),
        ],
        name="segment_clear",
    )


class Stage1Policy:
    """Stateful multi-stage policy (name is historical; covers Stage 1–9).

    Tick order: pizza → Alleycat pack → sewer spikes → Baxter →
    Technodrome → cave → Slash → form-2 Shredder → combat tree →
    combat-stall escape. Neon lane hold lives in the tree (walk + fight)
    so stall can still overlay a freeze. HazardAvoid is not ticked.

    Clean production choices are intentional — see module docstring and
    ``docs/CLEAN_PLAYBOOK.md`` before reordering or re-enabling hazard dodge.
    """

    def __init__(self) -> None:
        self._cadence = AttackCadence(hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP)
        # No blind walk-Y: PizzaSeek owns pizza; empty-screen Y is visual junk.
        self._walk = PlayerXStallWalk(pickup_every=0)
        self._bind()

    def reset(self) -> None:
        """Reset cadence / walk stall and rebuild tactics + tree."""
        self._cadence.reset()
        self._walk.reset()
        self._bind()

    def _bind(self) -> None:
        """Construct pizza / stage tactics and rebuild the combat tree."""
        self._pizza = PizzaSeek()
        self._alleycat = AlleycatPackTactics()
        self._sewer_spikes = SewerSpikeAvoid()
        self._baxter = BaxterTactics()
        self._technodrome = TechnodromeTactics()
        self._prehistoric_cave = PrehistoricCaveRecovery()
        self._slash = SlashTactics()
        self._shredder_f2 = SuperShredderForm2Tactics()
        self._combat_stall = CombatPositionStall()
        self._tree = build_stage1_tree(
            cadence=self._cadence,
            walk_progress=self._walk,
        )

    def tick(self, state: GameState) -> TickResult:
        """Choose one frame of action for the current state."""
        for tactic in (
            self._pizza,
            self._alleycat,
            self._sewer_spikes,
            self._baxter,
            self._technodrome,
            self._prehistoric_cave,
            self._slash,
            self._shredder_f2,
        ):
            action = tactic.next(state)
            if action is not None:
                return TickResult(
                    status=NodeStatus.RUNNING,
                    action=action,
                    reason=action.reason,
                )
        result = self._tree.tick(state)
        combat_stall = self._combat_stall.next(state)
        if combat_stall is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=combat_stall,
                reason=combat_stall.reason,
            )
        if result.action is None and result.status is NodeStatus.SUCCESS:
            return TickResult(
                status=NodeStatus.SUCCESS,
                action=FrameAction(action=idle_action(), reason="segment_done"),
                reason=result.reason,
            )
        if result.action is None:
            return TickResult(
                status=result.status,
                action=FrameAction(action=idle_action(), reason="policy_idle"),
                reason=result.reason,
            )
        return result
