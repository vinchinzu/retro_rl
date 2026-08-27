"""Clean Baxter: HP-adaptive left standoff + elevated jump-slash."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.tactics.fight import _STAGE1_ATTACK_RANGE

_BAXTER_CHAR = 0x44
# Baxter elev peaks ~13px on Stage1 probes — fire jump-slash at 10+.
_BAXTER_JUMP_DY = 10
_BAXTER_JUMP_ADX = 72


class BaxterTactics:
    """Clean Baxter: HP-adaptive left standoff + elevated jump-slash.

    Entry X/HP vary with wave path RNG (checkpoint vs power-on). Always
    force a left standoff; widen it and slow cadence when HP is low so
    Clean never walks into his body trading at HP 2–6.
    """

    _LANE_Y = 171
    _STANDOFF = 36
    _STANDOFF_LOW = 48  # more space when Clean is stressed
    _STANDOFF_SLACK = 14
    _MIN_ADX = 20

    def __init__(self) -> None:
        self._saw_arena_pizza = False

    def next(self, state: GameState) -> FrameAction | None:
        """Return a Clean Baxter action, or ``None`` to fall through."""
        if state.mode is not GameMode.PLAYING:
            return None
        if not (state.boss_active and state.stage == 0 and state.health > 0):
            self._saw_arena_pizza = False
            return None
        if state.extras.get("pickups"):
            self._saw_arena_pizza = True
            # PizzaSeek owns critical grabs; otherwise leave healthy fights.
            if state.health > 32:
                return None
        enemies = state.living_enemies
        baxter = next(
            (e for e in enemies if e.kind == _BAXTER_CHAR),
            enemies[0] if enemies else None,
        )
        if baxter is None:
            return None

        low = state.health <= 40
        standoff = self._STANDOFF_LOW if low else self._STANDOFF
        elev = state.player_y - baxter.y
        adx = abs(baxter.x - state.player_x)
        dy = baxter.y - state.player_y
        ideal_x = baxter.x - standoff

        # 1) Wrong side / body overlap — never approach_right into him.
        if state.player_x >= baxter.x - self._MIN_ADX:
            return FrameAction(action=buttons("LEFT"), reason="baxter_releft")

        # 2) Elevated jump-slash before micro standoff thrash (elev peaks ~13).
        # Skip jump when critically low — jump trades are riskier than poke.
        if (
            elev >= _BAXTER_JUMP_DY
            and adx <= _BAXTER_JUMP_ADX
            and state.health > 20
        ):
            toward: list[str] = []
            if baxter.x > state.player_x + 10:
                toward.append("RIGHT")
            elif baxter.x < state.player_x - 10:
                toward.append("LEFT")
            if state.frame % 4 == 0:
                return FrameAction(
                    action=buttons("B", "Y", *toward),
                    reason="baxter_jump_slash",
                )
            if toward and state.frame % 4 == 1:
                return FrameAction(
                    action=buttons(*toward),
                    reason="baxter_reclose",
                )
            return FrameAction(
                action=buttons("Y"),
                reason="baxter_ground_poke",
            )

        # 3) Hold left standoff band (Clean chip comes from walking in).
        if state.player_x > ideal_x + self._STANDOFF_SLACK:
            return FrameAction(action=buttons("LEFT"), reason="baxter_releft")
        if state.player_x < ideal_x - self._STANDOFF_SLACK - 10:
            if state.player_x < baxter.x - standoff - 8:
                return FrameAction(
                    action=buttons("RIGHT"), reason="baxter_reclose"
                )

        if abs(dy) > 16:
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="baxter_lane",
            )

        # 4) Cadence: slower when low HP so we are not locked in his hitbox.
        period = 6 if low else 4
        hold = 1 if low else 2
        if adx <= _STAGE1_ATTACK_RANGE + 12:
            if state.frame % period < hold:
                return FrameAction(action=buttons("Y"), reason="baxter_ground_poke")
            return FrameAction(action=idle_action(), reason="baxter_gap")

        if state.player_x < ideal_x - 4:
            return FrameAction(action=buttons("RIGHT"), reason="baxter_reclose")
        return FrameAction(action=buttons("Y"), reason="baxter_ground_poke")
