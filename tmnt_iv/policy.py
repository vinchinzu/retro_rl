"""Segment policy behavior tree for TMNT IV Stage 1 clears."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.controls import SNES_LEFT, SNES_RIGHT
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.behavior import (
    ActionNode,
    Condition,
    NodeStatus,
    Selector,
    Sequence,
    TickResult,
)
from snes_oneshot.combat import (
    AttackCadence,
    PreferredFlank,
    WalkProgress,
    fight_nearest_action,
)
from snes_oneshot.game_state import EnemyState, GameMode, GameState
from snes_oneshot.primitives import FrameAction

# Screen coords: UP decreases Y (probe-confirmed). Do NOT invert.
_Y_TOLERANCE = 8
# Sewer Surfin' (stage byte >= 2): hanging spikes punish chasing UP.
_SEWER_Y_TOLERANCE = 20
_SEWER_MIN_FIGHT_Y = 160
# Rat King: long horizontal poke (dx≈120 from left/mid still hits).
_RAT_KING_Y_TOLERANCE = 16
# Leo weapon reach is generous (~50–68); keep a modest poke band.
_ATTACK_RANGE = 56
_MIN_RANGE = 8
_STANDOFF = 24
_ATTACK_HOLD = 2
_ATTACK_GAP = 5
# Rat King Footski: extended range + tight cadence; jump only to
# escape the auto-scroll left chip (standing jump-slashes whiff).
_BOSS_ATTACK_RANGE = 140
_BOSS_MIN_RANGE = 8
_BOSS_STANDOFF = 24
_BOSS_ATTACK_HOLD = 2
_BOSS_ATTACK_GAP = 1
_BOSS_LEFT_CHIP_X = 118
# Player/enemy X are screen-space; combat sees camera_x=0 (see fight_action).
_CAMERA_LEFT_MARGIN = 24
_CAMERA_RIGHT_MARGIN = 220
_EDGE_ATTACK_BONUS = 16
_LEFT_THREAT_X = 80
# 0x003A can tick while Leo is stuck on Stage 2 dumpster collision.
_PLAYER_X_STALL_FRAMES = 40
# Deep dumpsters block mid/upper lanes; bottom-lane JUMP+RIGHT clears.
_STALL_DOWN_FRAMES = 36
_STALL_JUMP_RIGHT_FRAMES = 24
_STALL_RIGHT_FRAMES = 40
_STALL_UP_FRAMES = 36
_STALL_UP_RIGHT_FRAMES = 48
_STALL_SMASH_FRAMES = 24


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


def _is_sewer(state: GameState) -> bool:
    """True on Stage 3 Sewer Surfin' (stage byte 2)."""
    return state.stage == 2


def _is_prehistoric(state: GameState) -> bool:
    """True on Stage 5 Prehistoric (stage byte 4)."""
    return state.stage == 4


def _is_wounded_knee(state: GameState) -> bool:
    """True on Stage 7 Bury My Shell at Wounded Knee (byte 6)."""
    return state.stage == 6


def _is_neon_highway(state: GameState) -> bool:
    """True on Stage 8 Neon Night Riders Mode-7 highway (byte 7)."""
    return state.stage == 7


def _is_starbase(state: GameState) -> bool:
    """True on Stage 9 Starbase waves / Super Shredder (bytes 8–9)."""
    return state.stage in {8, 9}


# Stage 7 stacked / elevated Foot (bazooka stack top is char 0xb0).
_WOUNDED_KNEE_JUMP_CHARS: frozenset[int] = frozenset({0xB0})
# Starbase: hover Foot / teleporter spawns / stack tops whiff grounded Y.
_STARBASE_JUMP_CHARS: frozenset[int] = frozenset({0x6A, 0x6C, 0xB0, 0xB2, 0xB4, 0xF2})
# Mode-7: enemies approach in depth (rising Y). Player Y clamps ~160–213;
# chasing far slots (y≪player) only burns frames. Fight the near band.
_NEON_MIN_FIGHT_Y = 140
_NEON_Y_TOLERANCE = 48
_NEON_ATTACK_RANGE = 68
_KRANG_CHAR = 0x4E
_KRANG_LEFT_STANDOFF = 36

# The SNES-only Technodrome finale is a special interaction, not a normal
# beat-'em-up lock.  Pink Foot block standing Y attacks and the Shredder tank
# only takes damage when a stunned Foot is grabbed and thrown at the screen.
_TECHNODROME_STAGE = 3
_SHREDDER_TANK_EVENT = 0x18
_BLOCKING_FOOT_CHAR = 0x6C
_TANK_FOOT_CHARS: frozenset[int] = frozenset({0x66, 0x6C})
_MOUSER_CHAR = 0x58
_TECHNODROME_JUMP_CHARS: frozenset[int] = frozenset({0x6A})
_SLASH_CHAR = 0x50
_SUPER_SHREDDER_F2 = 0xAE


class SuperShredderForm2Tactics:
    """Left-flank poke with periodic hop dodges for form-2 demutation.

    Hard-mode form 2 fires a demutation projectile that bypasses ordinary HP.
    Standing still and mashing Y is what forced the old iframe assist. Keep a
    left standoff, chip with grounded Y, and hop laterally every short cycle so
    the projectile whiffs without writing invulnerability RAM.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear flank / dodge phase."""
        self._phase = "approach"
        self._timer = 0
        self._cadence = 0

    def next(self, state: GameState) -> FrameAction | None:
        """Return a form-2 action, or ``None`` outside the finale arena."""
        if state.stage != 9 or int(state.extras.get("event", -1)) != 0x0A:
            if self._phase != "approach" or self._timer:
                self.reset()
            return None
        boss = next(
            (
                enemy
                for enemy in state.living_enemies
                if enemy.kind == _SUPER_SHREDDER_F2
            ),
            None,
        )
        if boss is None:
            return None

        # The left standoff is unreachable when Shredder teleports against
        # the left wall. Switch sides there instead of walking into the wall
        # for thousands of frames; dodge and spacing use the same open side.
        use_right_flank = boss.x < 64
        target_x = boss.x + 40 if use_right_flank else boss.x - 40
        open_side = "RIGHT" if use_right_flank else "LEFT"
        inward_side = "LEFT" if use_right_flank else "RIGHT"
        dx = target_x - state.player_x
        dy = boss.y - state.player_y
        adx_boss = abs(boss.x - state.player_x)

        # Hop dodge cycle — short jump left/right off the projectile lane.
        if self._phase == "dodge":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "attack"
                self._timer = 36
            side = (
                open_side
                if (self._timer // 6) % 2 == 0
                else inward_side
            )
            return FrameAction(action=buttons("B", side), reason="shredder_dodge")

        if abs(dy) > 14:
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="shredder_align",
            )
        if abs(dx) > 12:
            self._phase = "approach"
            return FrameAction(
                action=buttons("LEFT" if dx < 0 else "RIGHT"),
                reason="shredder_approach",
            )

        # In standoff band: cadence Y, then hop.
        if self._phase != "attack":
            self._phase = "attack"
            self._timer = 40
        self._cadence = (self._cadence + 1) % 8
        self._timer -= 1
        if self._timer <= 0:
            self._phase = "dodge"
            self._timer = 18
            return FrameAction(
                action=buttons("B", open_side),
                reason="shredder_dodge",
            )
        if self._cadence < 2 and adx_boss <= 72:
            return FrameAction(action=buttons("Y"), reason="shredder_attack")
        if adx_boss < 24:
            return FrameAction(
                action=buttons(open_side),
                reason="shredder_space",
            )
        return FrameAction(action=idle_action(), reason="shredder_wait")


class SlashTactics:
    """Hybrid whiplash: lab winner ported from ``slash_pattern_lab``.

    FullHardBoss5 emergency-heal probe (lab): **18,570f / 918 dmg / 13 heals**.
    Rules (also ``docs/SLASH_VULN_MAP.md``):

    * Claw ``0x83``/``0x09`` → hop away (main player damage).
    * Spin ``0xEE`` close → hop away.
    * Punish ``0x3E``/``0x2E``/``0x17``/``0xB7``/``0x23`` or iframes → mash.
    * Else thrash approach@48 → cross → toward+Y; shorter cross at low HP.
    """

    _SPIN = 0xEE
    _CLAW = frozenset({0x83, 0x09})
    _PUNISH = frozenset({0x3E, 0x2E, 0x17, 0xB7, 0x23, 0x40})

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget thrash phase."""
        self._active = False
        self._phase = "approach"
        self._timer = 0
        self._punish_tick = 0

    def next(self, state: GameState) -> FrameAction | None:
        """Return a Slash-specific action, or ``None`` outside his fight."""
        if state.stage != 4 or int(state.extras.get("event", -1)) != 0x0A:
            if self._active:
                self.reset()
            return None

        slash = next(
            (enemy for enemy in state.living_enemies if enemy.kind == _SLASH_CHAR),
            None,
        )
        if slash is None:
            if not self._active:
                return None
            return FrameAction(action=idle_action(), reason="slash_wait")

        self._active = True
        dy = slash.y - state.player_y
        dx = slash.x - state.player_x
        adx = abs(dx)
        status = int(slash.animation)
        iframes = int(state.extras.get("iframes", 0))
        toward = "RIGHT" if dx > 0 else "LEFT"
        away = "LEFT" if dx >= 0 else "RIGHT"

        # Off-screen parking (x≳256): hold mid-lane.
        if slash.x > 256:
            if state.player_x > 180:
                return FrameAction(action=buttons("LEFT"), reason="slash_approach")
            if state.player_x < 90:
                return FrameAction(action=buttons("RIGHT"), reason="slash_approach")
            return FrameAction(action=idle_action(), reason="slash_wait")

        # Shell spin (and claw often follows). Lab winner only flees 0xEE;
        # over-dodging 0x83/0x09 starved DPS in probes.
        if status == self._SPIN and iframes <= 0 and adx < 52 and abs(dy) <= 22:
            self._phase = "approach"
            return FrameAction(action=buttons("B", away), reason="slash_dodge")
        # Claw active only when already inside the hit band — don't kite.
        if status in self._CLAW and iframes <= 0 and adx < 44:
            self._phase = "approach"
            return FrameAction(action=buttons("B", away), reason="slash_dodge")

        # Iframe / punish windows — full aggression (lab core).
        if iframes > 0 or status in self._PUNISH:
            if abs(dy) > 14 and adx < 40 and iframes <= 0:
                return FrameAction(
                    action=buttons("UP" if dy < 0 else "DOWN"),
                    reason="slash_align",
                )
            if adx > 54:
                return FrameAction(action=buttons(toward), reason="slash_approach")
            if adx < 8:
                return FrameAction(
                    action=buttons(away, "Y"), reason="slash_back_attack"
                )
            # Brief re-flank hop every ~48f of punish.
            self._punish_tick = (self._punish_tick + 1) % 48
            if self._punish_tick < 10 and adx < 40:
                return FrameAction(action=buttons("B", toward), reason="slash_cross")
            return FrameAction(action=buttons(toward, "Y"), reason="slash_back_attack")

        if abs(dy) > 10:
            self._phase = "approach"
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="slash_align",
            )

        # Thrash cycle — wider approach, shorter cross at low HP.
        approach_band = 48
        cross_frames = 16 if slash.health <= 48 else 22
        attack_frames = 40 if slash.health <= 48 else 36

        if self._phase == "approach":
            if adx > approach_band:
                return FrameAction(action=buttons(toward), reason="slash_approach")
            self._phase = "cross"
            self._timer = cross_frames
            return FrameAction(action=buttons("B", toward), reason="slash_cross")

        if self._phase == "cross":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "attack"
                self._timer = attack_frames
            return FrameAction(action=buttons("B", toward), reason="slash_cross")

        self._timer -= 1
        if self._timer <= 0:
            self._phase = "approach"
        return FrameAction(action=buttons(toward, "Y"), reason="slash_back_attack")


class PrehistoricCaveRecovery:
    """Return to the Stage 5 cave rendezvous after a frozen right edge.

    One Prehistoric cave fork stops scrolling at player X≈207.  The next
    wave is triggered near the upper-middle of the room, so continuing to
    press RIGHT can wait forever.  Activate only after a sustained empty
    screen at that exact edge, then walk to the observed trigger point.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_x = -1
        self._stall_frames = 0
        self._active = False
        self._wait_frames = 0

    def next(self, state: GameState) -> FrameAction | None:
        if (
            state.stage != 4
            or int(state.extras.get("event", -1)) != 0x0A
            or state.living_enemies
        ):
            self.reset()
            return None
        if state.player_x == self._last_x:
            self._stall_frames += 1
        else:
            self._last_x = state.player_x
            self._stall_frames = 0
        if state.player_x >= 200 and self._stall_frames >= 120:
            self._active = True
        if not self._active:
            return None
        if abs(state.player_y - 134) > 2:
            return FrameAction(
                action=buttons("UP" if state.player_y > 134 else "DOWN"),
                reason="cave_recenter_y",
            )
        target_x = 125
        if self._wait_frames > 240:
            # If the exact center does not fire immediately, sweep across it
            # instead of waiting forever at the edge of a trigger rectangle.
            target_x = 145 if (self._wait_frames // 240) % 2 else 105
        if abs(state.player_x - target_x) > 3:
            return FrameAction(
                action=buttons("LEFT" if state.player_x > target_x else "RIGHT"),
                reason="cave_recenter_x",
            )
        self._wait_frames += 1
        return FrameAction(action=idle_action(), reason="cave_wait")


class TechnodromeTactics:
    """Handle blocking Foot, right-wall Mousers, and the tank throw fight.

    The ordinary align-and-poke policy can clear most of the game, but it
    cannot finish the SNES Technodrome.  Hard-mode pink Foot block normal
    attacks, so they need a short retreat, a running shoulder hit, and a
    close grab.  During event ``0x18`` the grab is deliberately finished
    with toward+Y, which throws the Foot into Shredder's foreground tank.
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

    @staticmethod
    def _toward(state: GameState, enemy_x: int) -> str:
        return "RIGHT" if enemy_x > state.player_x else "LEFT"

    @staticmethod
    def _away(toward: str) -> str:
        return "LEFT" if toward == "RIGHT" else "RIGHT"

    @staticmethod
    def _align(state: GameState, enemy_y: int) -> FrameAction | None:
        dy = enemy_y - state.player_y
        if abs(dy) <= 10:
            return None
        return FrameAction(
            action=buttons("UP" if dy < 0 else "DOWN"),
            reason="technodrome_align",
        )

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
        self._target_slot = target.slot
        self._target_kind = target.kind
        self._target_health = target.health
        self._before_hp = target.health
        return target

    def _start_blocker(self, health: int) -> None:
        self._phase = "retreat"
        # Need a real runway so the shoulder stun actually registers before
        # the grab/throw into Shredder's tank. Too-short retreats whiff forever.
        self._timer = 40
        self._phase_frames = 0
        self._before_hp = health

    def _blocker_action(
        self,
        state: GameState,
        target: EnemyState,
    ) -> FrameAction:
        if not self._phase or self._phase.startswith("tank_"):
            self._start_blocker(target.health)
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
            if self._timer <= 0 or dx >= 55:
                self._phase = "charge"
                self._timer = 0
                self._before_hp = target.health
            return FrameAction(action=buttons(away), reason="blocker_retreat")
        if self._phase == "charge":
            self._timer += 1
            # Pink Foot stuns after a sustained pure run (~34f) then Y.
            # Old path ended at dx<16 with a 2f Y tap and whiffed ~75%.
            if self._timer >= 34 and dx < 22:
                self._phase = "hit"
                self._timer = 10
                self._before_hp = target.health
            elif self._timer >= 70:
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


def _sewer_combat_state(state: GameState) -> GameState:
    """Zero progress-camera; clamp fight Y out of Stage 3 spike band."""
    combat = replace(state, camera_x=0)
    # Rat King: keep real Y and all living slots — long pokes from the
    # water lane are what actually reduce boss HP (top-lane whiffs).
    if state.boss_active or not _is_sewer(state):
        return combat
    enemies = tuple(
        replace(e, y=max(e.y, _SEWER_MIN_FIGHT_Y)) if e.active and e.health > 0 else e
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


def _keep_sewer_pace(state: GameState, action: FrameAction) -> FrameAction:
    """Hold RIGHT during Sewer Surfin'; jump out of Rat King left chip."""
    if not _is_sewer(state):
        return action
    # Escape auto-scroll left chip before anything else.
    if state.boss_active and state.player_x <= _BOSS_LEFT_CHIP_X:
        return FrameAction(action=buttons("B", "RIGHT"), reason="boss_jump_right")
    held = list(action.action)
    # Spacing LEFT is intentional poke-retreat — do not force RIGHT.
    if held[SNES_LEFT] and action.reason in {
        "space_left",
        "approach_left",
    }:
        return action
    held[SNES_RIGHT] = 1
    return FrameAction(action=held, reason=action.reason)


def _neon_combat_state(state: GameState) -> GameState:
    """Zero progress-camera; keep only near-band / Krang Mode-7 targets."""
    combat = replace(state, camera_x=0)
    if not _is_neon_highway(state):
        return combat
    enemies = tuple(
        e
        if (
            e.active
            and e.health > 0
            and (e.kind == _KRANG_CHAR or e.y >= _NEON_MIN_FIGHT_Y)
        )
        else replace(e, active=False, health=0)
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


# Duo boss fights spawn endless chip adds (Mousers, ship trash). Fighting
# those forever leaves Tokka/Rahzar/Bebop/Rocksteady at full HP.
_DUO_BOSS_CHARS: frozenset[int] = frozenset({0x48, 0xA0, 0xA8, 0xAC})


def _duo_boss_combat_state(state: GameState) -> GameState:
    """Keep only the duo boss slots so adds cannot steal targeting."""
    combat = replace(state, camera_x=0)
    enemies = tuple(
        e
        if (e.active and e.health > 0 and e.kind in _DUO_BOSS_CHARS)
        else replace(e, active=False, health=0)
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


def _duo_boss_fight_action(
    state: GameState,
    *,
    cadence: AttackCadence,
) -> FrameAction | None:
    """Left-flank poke for Tokka/Rahzar and Bebop/Rocksteady.

    Shared ``fight_nearest_action`` treats dx≤70 as in-range even when Leo is
    on the wrong side of the duo, so he freezes at screen-right mashing Y
    forever. Force a left standoff first, then cadence Y.
    """
    if not (
        (state.boss_active and state.stage in {3, 5})
        or any(e.kind in _DUO_BOSS_CHARS for e in state.living_enemies)
    ):
        return None
    if state.stage not in {3, 5}:
        return None
    bosses = [
        e for e in state.living_enemies if e.kind in _DUO_BOSS_CHARS and e.health > 0
    ]
    if not bosses:
        return None
    cadence.hold_frames = _BOSS_ATTACK_HOLD
    cadence.gap_frames = _BOSS_ATTACK_GAP
    # Prefer the rightmost living boss so we work left through the pair.
    target = max(bosses, key=lambda e: e.x)
    ideal_x = target.x - 36
    dx = state.player_x - target.x
    dy = target.y - state.player_y

    # In the continuous route the right-wall door can pin Leo at x≈224
    # just as Tokka/Rahzar spawn. Plain LEFT never clears that collision,
    # while a short jump-left does; checkpoint boss states start left of it.
    if state.player_x >= 216:
        return FrameAction(
            action=buttons("B", "LEFT"),
            reason="duo_wall_escape",
        )
    if abs(dy) > 12:
        return FrameAction(
            action=buttons("UP" if dy < 0 else "DOWN"),
            reason="align_up" if dy < 0 else "align_down",
        )
    # Not yet on left flank (or overlapping) — close/space horizontally.
    if state.player_x > target.x - 18:
        return FrameAction(action=buttons("LEFT"), reason="approach_left")
    if state.player_x < ideal_x - 14:
        return FrameAction(action=buttons("RIGHT"), reason="approach_right")
    if abs(dx) < 14:
        return FrameAction(action=buttons("LEFT"), reason="space_left")
    # On left standoff band — poke.
    return cadence.next_attack(button="Y")


def _neon_fight_action(state: GameState) -> FrameAction | None:
    """Mode-7 lane wait / lateral match, or left-flank Krang poke.

    Returns ``None`` when the shared ``fight_nearest_action`` path should
    run on the filtered neon combat state.
    """
    if not _is_neon_highway(state):
        return None
    near = [
        e
        for e in state.living_enemies
        if e.kind == _KRANG_CHAR or e.y >= _NEON_MIN_FIGHT_Y
    ]
    if not near:
        if state.player_x < 90:
            return FrameAction(action=buttons("RIGHT"), reason="neon_drift_right")
        if state.player_x > 180:
            return FrameAction(action=buttons("LEFT"), reason="neon_drift_left")
        return FrameAction(action=idle_action(), reason="neon_wait")
    if state.boss_active or any(e.kind == _KRANG_CHAR for e in near):
        krang = next((e for e in near if e.kind == _KRANG_CHAR), near[0])
        target_x = krang.x - _KRANG_LEFT_STANDOFF
        dx = target_x - state.player_x
        dy = krang.y - state.player_y
        if abs(dx) > 14:
            return FrameAction(
                action=buttons("LEFT" if dx < 0 else "RIGHT"),
                reason=("approach_left" if dx < 0 else "approach_right"),
            )
        if abs(dy) > 24:
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="align_up" if dy < 0 else "align_down",
            )
        return FrameAction(action=buttons("Y"), reason="attack")
    return None


class PlayerXStallWalk:
    """Walk right; break Stage 2 dumpster soft-locks when player X freezes.

    ``0x003A`` keeps ticking while Leo is glued to alley dumpsters. Deep
    dumpsters block the mid/upper lanes — drop to the bottom lane and
    JUMP+RIGHT, with UP / smash as fallbacks for earlier obstacles.
    """

    def __init__(self, *, pickup_every: int = 24) -> None:
        self._walk = WalkProgress(pickup_every=pickup_every)
        self._last_player_x: int = -1
        self._stall_frames: int = 0
        self._escape_frames: int = 0

    def reset(self) -> None:
        """Clear walk + player-X stall tracking."""
        self._walk.reset()
        self._last_player_x = -1
        self._stall_frames = 0
        self._escape_frames = 0

    def _stall_escape(self) -> FrameAction:
        """Cycle dumpster breakers while X remains frozen."""
        phase = self._escape_frames
        self._escape_frames += 1
        down_end = _STALL_DOWN_FRAMES
        jump_end = down_end + _STALL_JUMP_RIGHT_FRAMES
        right_end = jump_end + _STALL_RIGHT_FRAMES
        up_end = right_end + _STALL_UP_FRAMES
        up_right_end = up_end + _STALL_UP_RIGHT_FRAMES
        smash_end = up_right_end + _STALL_SMASH_FRAMES
        cycle = smash_end
        slot = phase % cycle
        if slot < down_end:
            return FrameAction(action=buttons("DOWN"), reason="stall_down")
        if slot < jump_end:
            return FrameAction(action=buttons("B", "RIGHT"), reason="stall_jump_right")
        if slot < right_end:
            return FrameAction(action=buttons("RIGHT"), reason="stall_right")
        if slot < up_end:
            return FrameAction(action=buttons("UP"), reason="stall_up")
        if slot < up_right_end:
            return FrameAction(action=buttons("UP", "RIGHT"), reason="stall_up_right")
        return FrameAction(action=buttons("Y"), reason="stall_smash")

    def next(self, state: GameState) -> FrameAction:
        """Walk via WalkProgress; on X-stall run dumpster escape."""
        if state.living_enemies:
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
            return self._walk.next(state)
        if self._last_player_x == state.player_x:
            self._stall_frames += 1
        else:
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
        if self._stall_frames >= _PLAYER_X_STALL_FRAMES:
            return self._stall_escape()
        return self._walk.next(state)


class CombatPositionStall:
    """Jump out when combat position and total enemy HP never change.

    Disabled while Tokka/Rahzar or Bebop/Rocksteady are live: the duo left-flank
    poke is intentionally stationary for long stretches, and the jump-escape
    walks into the pack (Boss4 probe: 364→116 damage with stall suppressed).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the frozen-position signature and escape phase."""
        self._signature: tuple[int, int, int, int] | None = None
        self._stalled_frames = 0
        self._escape_frame = -1

    def next(self, state: GameState) -> FrameAction | None:
        """Return a short no-special escape after a four-second freeze."""
        if not state.living_enemies or state.mode is not GameMode.PLAYING:
            self.reset()
            return None
        # Duo left-flank poke holds X/Y while chipping; do not override it.
        if any(
            enemy.kind in _DUO_BOSS_CHARS and enemy.health > 0
            for enemy in state.living_enemies
        ):
            self.reset()
            return None
        signature = (
            state.stage,
            state.player_x,
            state.player_y,
            sum(enemy.health for enemy in state.living_enemies),
        )
        if self._escape_frame >= 0:
            phase = self._escape_frame // 32
            self._escape_frame += 1
            if self._escape_frame >= 160:
                self.reset()
            patterns = (
                ("B", "DOWN", "RIGHT"),
                ("B", "DOWN", "LEFT"),
                ("B", "UP", "RIGHT"),
                ("B", "UP", "LEFT"),
                ("B", "Y"),
            )
            return FrameAction(
                action=buttons(*patterns[min(phase, 4)]),
                reason="combat_stall_escape",
            )
        if signature == self._signature:
            self._stalled_frames += 1
        else:
            self._signature = signature
            self._stalled_frames = 0
        if self._stalled_frames < 240:
            return None
        self._escape_frame = 0
        return FrameAction(
            action=buttons("B", "DOWN", "RIGHT"),
            reason="combat_stall_escape",
        )


def build_stage1_tree(
    *,
    cadence: AttackCadence | None = None,
    walk_progress: PlayerXStallWalk | WalkProgress | None = None,
) -> Selector:
    """Segment policy: continue → clear → fight nearest → walk right."""
    cadence = cadence or AttackCadence(hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP)
    walk_progress = walk_progress or PlayerXStallWalk(pickup_every=24)

    def fight_action(state: GameState) -> FrameAction:
        neon = _neon_fight_action(state)
        if neon is not None:
            return neon
        duo = _duo_boss_fight_action(state, cadence=cadence)
        if duo is not None:
            return duo
        # 0x003A progress is not a scroll origin; zero it so screen-edge
        # clamps use player_x / enemy.x directly. Sewer Surfin' also
        # clamps target Y below hanging spikes. Neon filters far-depth
        # Foot so we do not chase Mode-7 vanishing-point slots.
        # Technodrome duo: stay on the LEFT flank so we do not overshoot
        # past Rahzar into the right-wall door and whiff forever.
        technodrome_boss = state.boss_active and state.stage == 3
        prehistoric_boss = state.boss_active and _is_prehistoric(state)
        # Skull and Crossbones (stage byte 5): Bebop + Rocksteady duo.
        pirate_boss = state.boss_active and state.stage == 5
        neon_boss = state.boss_active and _is_neon_highway(state)
        # Super Shredder form 1 (byte 8) / form 2 (byte 9).
        shredder_boss = state.boss_active and _is_starbase(state)
        if technodrome_boss or pirate_boss:
            combat = _duo_boss_combat_state(state)
        elif _is_neon_highway(state):
            combat = _neon_combat_state(state)
        else:
            combat = _sewer_combat_state(state)
        if state.boss_active and _is_sewer(state):
            y_tol = _RAT_KING_Y_TOLERANCE
            attack_range = _BOSS_ATTACK_RANGE
            min_range = _BOSS_MIN_RANGE
            standoff = _BOSS_STANDOFF
            cadence.hold_frames = _BOSS_ATTACK_HOLD
            cadence.gap_frames = _BOSS_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif technodrome_boss or pirate_boss:
            # Duo bosses: stay LEFT so we do not overshoot into the
            # right-wall door / ship edge and whiff forever.
            y_tol = _Y_TOLERANCE
            attack_range = 70
            min_range = 12
            standoff = 36
            cadence.hold_frames = _BOSS_ATTACK_HOLD
            cadence.gap_frames = _BOSS_ATTACK_GAP
            flank = PreferredFlank.LEFT
        elif prehistoric_boss or shredder_boss:
            # Slash / Super Shredder: grounded Y; wide right margin.
            y_tol = _Y_TOLERANCE
            attack_range = 72
            min_range = 10
            standoff = 28
            cadence.hold_frames = _BOSS_ATTACK_HOLD
            cadence.gap_frames = _BOSS_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif _is_neon_highway(state):
            y_tol = _NEON_Y_TOLERANCE
            attack_range = _NEON_ATTACK_RANGE
            min_range = _MIN_RANGE
            standoff = _STANDOFF
            cadence.hold_frames = _ATTACK_HOLD
            cadence.gap_frames = _ATTACK_GAP
            flank = PreferredFlank.NONE
        elif _is_sewer(state):
            y_tol = _SEWER_Y_TOLERANCE
            attack_range = _ATTACK_RANGE
            min_range = _MIN_RANGE
            standoff = _STANDOFF
            cadence.hold_frames = _ATTACK_HOLD
            cadence.gap_frames = _ATTACK_GAP
            flank = PreferredFlank.NONE
        else:
            y_tol = _Y_TOLERANCE
            attack_range = _ATTACK_RANGE
            min_range = _MIN_RANGE
            standoff = _STANDOFF
            cadence.hold_frames = _ATTACK_HOLD
            cadence.gap_frames = _ATTACK_GAP
            flank = PreferredFlank.NONE
        # Alleycat / Technodrome Foot sometimes park past ~256 and never
        # close. Shared right-edge wait then soft-locks; widen the margin
        # so we walk in. Sewer Surfin' auto-scrolls — Rat King also sits
        # past the alley walk-band, so keep the wide margin there too.
        far_park = any(e.x > _CAMERA_RIGHT_MARGIN + 24 for e in state.living_enemies)
        right_margin = (
            400
            if (
                far_park
                or _is_sewer(state)
                or technodrome_boss
                or prehistoric_boss
                or pirate_boss
                or neon_boss
                or shredder_boss
                or _is_neon_highway(state)
                or _is_starbase(state)
            )
            else _CAMERA_RIGHT_MARGIN
        )
        action = _keep_sewer_pace(
            state,
            fight_nearest_action(
                combat,
                y_tolerance=y_tol,
                attack_range=attack_range,
                min_range=min_range,
                attack_button="Y",
                invert_vertical=False,
                cadence=cadence,
                preferred_flank=flank,
                standoff=standoff,
                use_throw=False,
                prefer_left_threat=not state.boss_active,
                left_threat_x=_LEFT_THREAT_X,
                camera_left_margin=_CAMERA_LEFT_MARGIN,
                camera_right_margin=right_margin,
                edge_attack_bonus=_EDGE_ATTACK_BONUS,
            ),
        )
        # Prehistoric dinos (0x6C) ignore grounded Y — jump-slash (B+Y).
        # Slash (boss) chips with grounded Y; jump-slash whiffs his shell.
        if (
            _is_prehistoric(state)
            and action.reason == "attack"
            and not prehistoric_boss
            and any(e.kind == 0x6C for e in state.living_enemies)
        ):
            return FrameAction(action=buttons("B", "Y"), reason="jump_slash")
        # Hard Technodrome spear Foot (0x6A) can hold a standing guard
        # forever; a jump-slash breaks it immediately.
        if (
            state.stage == _TECHNODROME_STAGE
            and action.reason == "attack"
            and not state.boss_active
            and any(e.kind in _TECHNODROME_JUMP_CHARS for e in state.living_enemies)
        ):
            return FrameAction(action=buttons("B", "Y"), reason="jump_slash")
        # Stage 7 train: stacked bazooka Foot (0xb0 on shoulders) needs
        # jump-slash; grounded Y only chips the carrier.
        if (
            _is_wounded_knee(state)
            and action.reason == "attack"
            and not state.boss_active
            and any(e.kind in _WOUNDED_KNEE_JUMP_CHARS for e in state.living_enemies)
        ):
            return FrameAction(action=buttons("B", "Y"), reason="jump_slash")
        # Starbase: hover / teleporter Foot and bruiser stacks need B+Y.
        if (
            _is_starbase(state)
            and action.reason == "attack"
            and not state.boss_active
            and any(e.kind in _STARBASE_JUMP_CHARS for e in state.living_enemies)
        ):
            return FrameAction(action=buttons("B", "Y"), reason="jump_slash")
        return action

    def walk_action(state: GameState) -> FrameAction:
        # Mode-7 auto-scrolls; frozen X must not trigger dumpster escapes.
        if _is_neon_highway(state):
            if state.player_x < 90:
                return FrameAction(action=buttons("RIGHT"), reason="neon_drift_right")
            if state.player_x > 180:
                return FrameAction(action=buttons("LEFT"), reason="neon_drift_left")
            return FrameAction(action=idle_action(), reason="neon_wait")
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
    """Stateful wrapper around the Stage 1 / Stage 2 segment tree."""

    def __init__(self) -> None:
        self._cadence = AttackCadence(hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP)
        self._walk = PlayerXStallWalk(pickup_every=24)
        self._technodrome = TechnodromeTactics()
        self._prehistoric_cave = PrehistoricCaveRecovery()
        self._slash = SlashTactics()
        self._shredder_f2 = SuperShredderForm2Tactics()
        self._combat_stall = CombatPositionStall()
        self._tree = build_stage1_tree(
            cadence=self._cadence,
            walk_progress=self._walk,
        )

    def reset(self) -> None:
        """Reset cadence / walk stall and rebuild the tree."""
        self._cadence.reset()
        self._walk.reset()
        self._technodrome.reset()
        self._prehistoric_cave.reset()
        self._slash.reset()
        self._shredder_f2.reset()
        self._combat_stall.reset()
        self._tree = build_stage1_tree(
            cadence=self._cadence,
            walk_progress=self._walk,
        )

    def tick(self, state: GameState) -> TickResult:
        """Choose one frame of action for the current state."""
        technodrome = self._technodrome.next(state)
        if technodrome is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=technodrome,
                reason=technodrome.reason,
            )
        cave_recovery = self._prehistoric_cave.next(state)
        if cave_recovery is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=cave_recovery,
                reason=cave_recovery.reason,
            )
        slash = self._slash.next(state)
        if slash is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=slash,
                reason=slash.reason,
            )
        shredder = self._shredder_f2.next(state)
        if shredder is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=shredder,
                reason=shredder.reason,
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
