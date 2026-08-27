"""Technodrome pink-Foot ram and tank-throw tactics.

Hard-mode Tonfa / pink Foot (``0x6C``) block standing Y. Raphael's close
opener is jump-behind → tap Y → screen throw. The long retreat → pure-run
charge → toward+Y ram is the fallback when the Foot is far, off the Y
band after a missed hop, or not Raphael. Never mix vertical taps into
charge; never emit A or grounded Y+B (Power Attack).
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import EnemyState, GameState
from tmnt_iv.grind_knobs import active_knobs
from tmnt_iv.stages import RAPH_CHAR

_TECHNODROME_STAGE = 3
_SHREDDER_TANK_EVENT = 0x18
_BLOCKING_FOOT_CHAR = 0x6C
_TANK_FOOT_CHARS: frozenset[int] = frozenset({0x66, 0x6C})
_MOUSER_CHAR = 0x58

# Jump-behind when already close on the same lane. Far / facing-block
# Foot keep the 40f retreat + ≥34f pure-run ram.
_JB_ADX = 48
_JB_Y_BAND = 10
_JB_JUMP_FRAMES = 16
_JB_LAND_FRAMES = 8
_JB_STUN_FRAMES = 2
_JB_STUN_GAP = 8
_JB_BUDGET = 56
_JB_BEHIND_DX = 10
_JB_WALL_LEFT = 48
_JB_WALL_RIGHT = 208


class TechnodromeTactics:
    """Handle blocking Foot, right-wall Mousers, and the tank throw fight.

    The ordinary align-and-poke policy can clear most of the game, but it
    cannot finish the SNES Technodrome.  Hard-mode pink Foot block normal
    attacks.  Raphael jumps behind a close Foot, taps Y, and throws;
    otherwise a short retreat, running shoulder hit, and close grab.
    During event ``0x18`` the grab is deliberately finished with toward+Y,
    which throws the Foot into Shredder's foreground tank.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the current target and input phase."""
        self._target_slot = -1
        self._target_kind = -1
        self._target_health = 0
        self._phase = ""
        self._timer = 0
        self._phase_frames = 0
        self._before_hp = 0
        self._mouser_cadence = 0
        self._in_tank = False
        self._force_ram = False
        self._jb_side = "RIGHT"

    @staticmethod
    def _toward(state: GameState, enemy_x: int) -> str:
        return "RIGHT" if enemy_x > state.player_x else "LEFT"

    @staticmethod
    def _away(toward: str) -> str:
        return "LEFT" if toward == "RIGHT" else "RIGHT"

    @staticmethod
    def _align(state: GameState, enemy_y: int) -> FrameAction | None:
        dy = enemy_y - state.player_y
        if abs(dy) <= _JB_Y_BAND:
            return None
        return FrameAction(
            action=buttons("UP" if dy < 0 else "DOWN"),
            reason="technodrome_align",
        )

    @staticmethod
    def _jump_side(state: GameState, target: EnemyState) -> str:
        """Direction that lands on the far side (through), unless walled."""
        if target.x > state.player_x:
            through = "RIGHT"
        elif target.x < state.player_x:
            through = "LEFT"
        else:
            through = "LEFT" if state.player_x > 128 else "RIGHT"
        if through == "RIGHT" and state.player_x >= _JB_WALL_RIGHT:
            return "LEFT"
        if through == "LEFT" and state.player_x <= _JB_WALL_LEFT:
            return "RIGHT"
        return through

    def _is_behind(self, state: GameState, target: EnemyState) -> bool:
        if self._jb_side == "RIGHT":
            return state.player_x >= target.x + _JB_BEHIND_DX
        return state.player_x <= target.x - _JB_BEHIND_DX

    def _want_jump_behind(self, state: GameState, target: EnemyState) -> bool:
        """Raph tank, already close, already on-lane. Else ram."""
        if self._force_ram or not self._in_tank:
            return False
        if int(state.extras.get("char_id", -1)) != RAPH_CHAR:
            return False
        if abs(target.x - state.player_x) > _JB_ADX:
            return False
        return abs(target.y - state.player_y) <= _JB_Y_BAND

    def _choose_target(
        self,
        state: GameState,
        *,
        kinds: frozenset[int],
        max_y: int = 255,
    ) -> EnemyState | None:
        candidates = [
            enemy
            for enemy in state.living_enemies
            if (enemy.kind in kinds and 0 < enemy.x < 256 and 0 < enemy.y <= max_y)
        ]
        current = next(
            (enemy for enemy in candidates if enemy.slot == self._target_slot),
            None,
        )
        if current is not None:
            # A slot can despawn and be reused between policy ticks. Do not
            # carry a grab/throw phase onto a fresh Foot in the same slot.
            if (
                current.kind != self._target_kind
                or current.health > self._target_health
            ):
                self._phase = ""
                self._timer = 0
                self._phase_frames = 0
                self._before_hp = current.health
                self._force_ram = False
            self._target_kind = current.kind
            self._target_health = current.health
            return current
        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda enemy: abs(enemy.x - state.player_x)
            + abs(enemy.y - state.player_y),
        )
        if target.slot != self._target_slot:
            self._phase = ""
            self._timer = 0
            self._phase_frames = 0
            self._force_ram = False
        self._target_slot = target.slot
        self._target_kind = target.kind
        self._target_health = target.health
        self._before_hp = target.health
        return target

    def _start_blocker(self, health: int) -> None:
        self._phase = "retreat"
        # Need a real runway so the shoulder stun actually registers before
        # the grab/throw into Shredder's tank. Too-short retreats whiff forever.
        self._timer = active_knobs().blocker_retreat_frames
        self._phase_frames = 0
        self._before_hp = health

    def _start_jump_behind(self, state: GameState, target: EnemyState) -> None:
        self._phase = "jb_jump"
        self._timer = _JB_JUMP_FRAMES
        self._phase_frames = 0
        self._before_hp = target.health
        self._jb_side = self._jump_side(state, target)

    def _jump_behind_action(
        self,
        state: GameState,
        target: EnemyState,
    ) -> FrameAction:
        self._phase_frames += 1
        if self._phase_frames > _JB_BUDGET:
            self._force_ram = True
            self._start_blocker(target.health)
            away = self._away(self._toward(state, target.x))
            return FrameAction(action=buttons(away), reason="blocker_retreat")

        if self._phase == "jb_jump":
            self._timer -= 1
            if self._timer <= 0 or self._is_behind(state, target):
                self._phase = "jb_land"
                self._timer = _JB_LAND_FRAMES
            return FrameAction(
                action=buttons("B", self._jb_side),
                reason="blocker_jump_behind",
            )
        if self._phase == "jb_land":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "jb_stun"
                self._timer = _JB_STUN_FRAMES
            return FrameAction(
                action=buttons(self._jb_side),
                reason="blocker_jump_behind",
            )
        if self._phase == "jb_stun":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "jb_stun_gap"
                self._timer = _JB_STUN_GAP
            # Standing Y only — never Y+B (Power Attack spends HP).
            return FrameAction(action=buttons("Y"), reason="blocker_behind_stun")

        # jb_stun_gap
        self._timer -= 1
        if self._timer <= 0:
            if target.health < self._before_hp:
                self._before_hp = target.health
                self._phase = "grab"
                self._phase_frames = 0
            else:
                self._force_ram = True
                self._start_blocker(target.health)
        return FrameAction(action=idle_action(), reason="blocker_behind_gap")

    def _blocker_action(
        self,
        state: GameState,
        target: EnemyState,
    ) -> FrameAction:
        if not self._phase or self._phase.startswith("tank_"):
            # Same-band hop only. Off-lane / far / corridor Foot start the
            # ram immediately — do not sit on empty-phase UP/DOWN (Stage4
            # locked ~24k frames of technodrome_align that way).
            if self._want_jump_behind(state, target):
                self._start_jump_behind(state, target)
            else:
                self._start_blocker(target.health)
        if self._phase.startswith("jb_"):
            return self._jump_behind_action(state, target)

        knobs = active_knobs()
        self._phase_frames += 1
        toward = self._toward(state, target.x)
        away = self._away(toward)
        dx = abs(target.x - state.player_x)

        # Align only while spacing / closing for grab. Never during pure-run
        # charge or dash-hit — vertical taps cancel run momentum and the pink
        # Foot block never breaks (was ~1 tank chip / 8k frames).
        if self._phase in {"retreat", "grab"}:
            aligned = self._align(state, target.y)
            if aligned is not None:
                return aligned

        if self._phase == "retreat":
            self._timer -= 1
            if self._timer <= 0 or dx >= knobs.blocker_retreat_dx:
                self._phase = "charge"
                self._timer = 0
                self._before_hp = target.health
            return FrameAction(action=buttons(away), reason="blocker_retreat")
        if self._phase == "charge":
            self._timer += 1
            # Pink Foot stuns after a sustained pure run (~34f) then Y.
            # Old path ended at dx<16 with a 2f Y tap and whiffed ~75%.
            if (
                self._timer >= knobs.blocker_charge_min
                and dx < knobs.blocker_charge_dx
            ):
                self._phase = "hit"
                self._timer = knobs.blocker_hit_frames
                self._before_hp = target.health
            elif self._timer >= knobs.blocker_charge_timeout:
                self._start_blocker(target.health)
            return FrameAction(action=buttons(toward), reason="blocker_charge")
        if self._phase == "hit":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "hit_gap"
                self._timer = 8
            return FrameAction(action=buttons(toward, "Y"), reason="blocker_dash_hit")
        if self._phase == "hit_gap":
            self._timer -= 1
            if self._timer <= 0:
                if target.health < self._before_hp:
                    self._before_hp = target.health
                    self._phase = "grab"
                    self._phase_frames = 0
                else:
                    self._start_blocker(target.health)
            return FrameAction(action=idle_action(), reason="blocker_hit_gap")
        if self._phase == "grab":
            if self._phase_frames > 120:
                self._force_ram = True
                self._start_blocker(target.health)
                return FrameAction(action=buttons(away), reason="blocker_retry")
            if dx > 12:
                return FrameAction(action=buttons(toward), reason="blocker_grab_close")
            self._phase = "grab_hold"
            self._timer = 3
            return FrameAction(action=buttons(toward, "Y"), reason="screen_throw")
        if self._phase == "grab_hold":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "grab_gap"
                self._timer = 14
            return FrameAction(action=buttons(toward, "Y"), reason="screen_throw")

        # grab_gap: release Y, then retry grab. Keep phase_frames so the
        # 120f grab budget still expires across throw retries.
        self._timer -= 1
        if self._timer <= 0:
            self._phase = "grab"
        return FrameAction(action=idle_action(), reason="screen_throw_gap")

    def _tank_action(self, state: GameState) -> FrameAction:
        # Foot below the bottom edge (Y > 220) cannot be reached; wait for
        # it to re-enter instead of holding DOWN forever at Leo's Y=215 cap.
        target = self._choose_target(state, kinds=_TANK_FOOT_CHARS, max_y=220)
        if target is None:
            self._target_slot = -1
            self._phase = ""
            if state.player_x < 72:
                return FrameAction(action=buttons("RIGHT"), reason="tank_center")
            if state.player_x > 184:
                return FrameAction(action=buttons("LEFT"), reason="tank_center")
            return FrameAction(action=idle_action(), reason="tank_wait")

        if target.kind == _BLOCKING_FOOT_CHAR:
            return self._blocker_action(state, target)

        # Non-blocking 0x66: short standing Y, then shared grab/throw.
        if not self._phase:
            self._phase = "tank_attack"
            self._timer = 0
            self._phase_frames = 0
            self._before_hp = target.health
        toward = self._toward(state, target.x)
        dx = abs(target.x - state.player_x)
        aligned = self._align(state, target.y)
        if aligned is not None and self._phase not in {"tank_hit", "grab_hold"}:
            return aligned

        if self._phase == "tank_attack":
            if dx > 48:
                return FrameAction(action=buttons(toward), reason="tank_foot_approach")
            self._phase = "tank_hit"
            self._timer = 2
            self._before_hp = target.health
        if self._phase == "tank_hit":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "tank_hit_gap"
                self._timer = 6
            return FrameAction(action=buttons("Y"), reason="tank_foot_stun")
        if self._phase == "tank_hit_gap":
            self._timer -= 1
            if self._timer <= 0:
                if target.health < self._before_hp:
                    self._before_hp = target.health
                    self._phase = "grab"
                    self._phase_frames = 0
                else:
                    self._phase = "tank_attack"
            return FrameAction(action=idle_action(), reason="tank_foot_stun_gap")

        return self._blocker_action(state, target)

    def _right_wall_mouser_action(self, state: GameState) -> FrameAction | None:
        if state.player_x < 210:
            return None
        mousers = [
            enemy
            for enemy in state.living_enemies
            if enemy.kind == _MOUSER_CHAR and enemy.health <= 2
        ]
        if not mousers:
            return None
        target = min(
            mousers,
            key=lambda enemy: abs(enemy.x - state.player_x)
            + abs(enemy.y - state.player_y),
        )
        aligned = self._align(state, target.y)
        if aligned is not None:
            return aligned
        self._mouser_cadence = (self._mouser_cadence + 1) % 7
        return FrameAction(
            action=(buttons("Y") if self._mouser_cadence < 2 else idle_action()),
            reason="mouser_wall_attack",
        )

    def next(self, state: GameState) -> FrameAction | None:
        """Return a Technodrome-specific action, or ``None``."""
        if state.stage != _TECHNODROME_STAGE:
            if self._phase or self._in_tank:
                self.reset()
            return None

        in_tank = int(state.extras.get("event", -1)) == _SHREDDER_TANK_EVENT
        if in_tank:
            if not self._in_tank:
                self.reset()
                self._in_tank = True
            return self._tank_action(state)
        if self._in_tank:
            self.reset()

        # Tokka/Rahzar fight: leave targeting to the duo left-flank poke.
        # Mouser chip must not steal the target forever while bosses sit full.
        if state.boss_active or any(
            enemy.kind in {0x48, 0xA0} for enemy in state.living_enemies
        ):
            if self._phase:
                self.reset()
            return None

        target = self._choose_target(state, kinds=frozenset({_BLOCKING_FOOT_CHAR}))
        if target is not None:
            return self._blocker_action(state, target)
        self._target_slot = -1
        self._phase = ""
        return self._right_wall_mouser_action(state)
