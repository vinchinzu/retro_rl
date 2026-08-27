"""PizzaSeek: Clean's only heal. Full seek is stage-0-only."""

from __future__ import annotations

from retro_harness.actions import buttons
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.ram import LEO_MAX_HP

# Post-pickup step-out; 6f lines up Baxter entry for the Clean clear.
_PIZZA_DISENGAGE_FRAMES = 6


class PizzaSeek:
    """Walk to ground pizza (char ``0x30``) and tap Y when HP is not full.

    Pizza is the **only** Clean heal (no emergency HP writes).

    **Stage 0 (Big Apple):** full HP-adaptive seek (see thresholds below).

    **Stage 1 (Alleycat) / Stage 2 (Sewer):** underfoot pickup always;
    far seek **only between waves** (no living enemies). Mid-wave far
    chase desynced emergency Stage2 (190→479 dmg). Global seek soft-locks
    Skull (see CLEAN_PLAYBOOK). Sewer needs the same between-wave grab
    for Clean boss entry HP.

    Clean suite thresholds (heal=none Stage1 + power-on):
    - Within ~64px → always grab when HP not full.
    - HP ≤ 68 → seek out to 260px.
    - HP ≤ 48 → any on-screen box (≤320).
    - Baxter: seek only when HP ≤ 32 (survival); otherwise leave poke lane.
    """

    _NEAR_DIST = 64
    _MID_DIST = 160
    _FAR_DIST = 260
    _SCREEN_DIST = 320
    _MID_HP = LEO_MAX_HP - 12  # 68
    _LOW_HP = 48
    _CRITICAL_HP = 36
    _BOSS_PIZZA_HP = 32
    # Underfoot + between-wave only (no mid-wave far chase).
    _BETWEEN_WAVE_STAGES = frozenset({1, 2})
    # Power-on Alleycat locked at (66,135) mashing Y on a box at (70,118)
    # (dy=17 inside the underfoot band, never collected). Give up so the
    # dumpster walk can resume.
    _PICKUP_GIVE_UP_FRAMES = 48

    def __init__(self) -> None:
        self._disengage_frames = 0
        self._pickup_frames = 0
        self._pickup_hp = -1
        self._pickup_key: tuple[int, int] | None = None
        self._skip: set[tuple[int, int]] = set()

    def next(self, state: GameState) -> FrameAction | None:
        """Return a seek/pickup action, or ``None`` when pizza is not useful."""
        if state.mode is not GameMode.PLAYING:
            return None
        # Alleycat / Sewer: underfoot always; far seek only between waves.
        # Mid-wave seek desynced emergency Stage2 190→479 dmg.
        if state.stage in self._BETWEEN_WAVE_STAGES:
            if not (0 < state.health < LEO_MAX_HP):
                return None
            pickups = state.extras.get("pickups") or ()
            if not pickups:
                self._reset_pickup_mash()
                return None
            live = {
                (int(p[0]), int(p[1]))
                for p in pickups
            }
            self._skip &= live
            reachable = [
                p
                for p in pickups
                if (int(p[0]), int(p[1])) not in self._skip
            ]
            if not reachable:
                self._reset_pickup_mash()
                return None
            target = min(
                reachable,
                key=lambda p: abs(p[0] - state.player_x)
                + abs(p[1] - state.player_y),
            )
            dx = int(target[0]) - state.player_x
            dy = int(target[1]) - state.player_y
            dist = abs(dx) + abs(dy)
            key = (int(target[0]), int(target[1]))
            if abs(dx) <= 14 and abs(dy) <= 18:
                if self._give_up_uncollected(state, key):
                    return None
                return FrameAction(action=buttons("Y"), reason="pizza_pickup")
            # Between waves only — dumpster-aware RIGHT toward the box.
            if state.living_enemies or dist > self._FAR_DIST:
                self._reset_pickup_mash()
                return None
            dirs: list[str] = []
            if abs(dx) > 4:
                dirs.append("RIGHT" if dx > 0 else "LEFT")
            if abs(dy) > 4:
                dirs.append("DOWN" if dy > 0 else "UP")
            if not dirs:
                if self._give_up_uncollected(state, key):
                    return None
                return FrameAction(action=buttons("Y"), reason="pizza_pickup")
            self._reset_pickup_mash()
            return FrameAction(action=buttons(*dirs), reason="pizza_seek")
        if state.stage != 0:
            return None
        # After a pickup in a crowd, step out before resuming the poke.
        if self._disengage_frames > 0:
            self._disengage_frames -= 1
            return FrameAction(action=buttons("LEFT"), reason="pizza_disengage")
        if not (0 < state.health < LEO_MAX_HP):
            return None
        pickups = state.extras.get("pickups") or ()
        if not pickups:
            return None
        # Baxter: only break the poke for pizza when Clean survival needs it.
        if state.boss_active and state.health > self._BOSS_PIZZA_HP:
            return None
        target = min(
            pickups,
            key=lambda p: abs(p[0] - state.player_x) + abs(p[1] - state.player_y),
        )
        tx, ty = int(target[0]), int(target[1])
        dx = tx - state.player_x
        dy = ty - state.player_y
        dist = abs(dx) + abs(dy)
        if dist <= self._NEAR_DIST:
            max_dist = self._NEAR_DIST
        elif state.health <= self._CRITICAL_HP or state.boss_active:
            max_dist = self._SCREEN_DIST
        elif state.health <= self._LOW_HP:
            max_dist = self._SCREEN_DIST
        elif state.health <= self._MID_HP:
            # heal=none Stage1 missed pizza at dist≈186 with old MID=96.
            max_dist = self._FAR_DIST
        else:
            # Scratch only: pizza we are already walking over.
            max_dist = self._NEAR_DIST
        if dist > max_dist:
            self._reset_pickup_mash()
            return None
        close_threats = sum(
            1
            for enemy in state.living_enemies
            if abs(enemy.x - state.player_x) < 24
            and abs(enemy.y - state.player_y) < 18
        )
        # Stay in the fray if surrounded unless pizza is underfoot / low HP.
        if (
            close_threats > 0
            and dist >= 40
            and state.health > self._LOW_HP
            and not state.boss_active
        ):
            return None
        if abs(dx) <= 14 and abs(dy) <= 18:
            self._disengage_frames = _PIZZA_DISENGAGE_FRAMES
            return FrameAction(action=buttons("Y"), reason="pizza_pickup")
        dirs: list[str] = []
        if abs(dx) > 4:
            dirs.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > 4:
            dirs.append("DOWN" if dy > 0 else "UP")
        if state.frame % 3 == 0:
            dirs.append("Y")
        if not dirs:
            self._disengage_frames = _PIZZA_DISENGAGE_FRAMES
            return FrameAction(action=buttons("Y"), reason="pizza_pickup")
        return FrameAction(action=buttons(*dirs), reason="pizza_seek")

    def _reset_pickup_mash(self) -> None:
        self._pickup_frames = 0
        self._pickup_hp = -1
        self._pickup_key = None

    def _give_up_uncollected(
        self, state: GameState, key: tuple[int, int]
    ) -> bool:
        """True when Y-mash failed to collect; skip this box."""
        if key != self._pickup_key or state.health != self._pickup_hp:
            self._pickup_key = key
            self._pickup_hp = state.health
            self._pickup_frames = 1
            return False
        self._pickup_frames += 1
        if self._pickup_frames < self._PICKUP_GIVE_UP_FRAMES:
            return False
        self._skip.add(key)
        self._reset_pickup_mash()
        return True
