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
from tmnt_iv.grind_knobs import active_knobs
from tmnt_iv.ram import LEO_MAX_HP

# Screen coords: UP decreases Y (probe-confirmed). Do NOT invert.
_Y_TOLERANCE = 8
# Big Apple (stage 0): wider poke + tighter cadence clears Baxter
# heal=none from Boss.state; also cuts wave chip before the pizza windows.
_STAGE1_ATTACK_RANGE = 64
_STAGE1_MIN_RANGE = 8
_STAGE1_STANDOFF = 32
_STAGE1_ATTACK_HOLD = 2
_STAGE1_ATTACK_GAP = 2
# Alleycat Blues (stage byte 1): a one-frame poke avoids trading into the
# alley packs.  Stage2 checkpoint probe: 15,453f / 124 dmg / 1 emergency
# heal, down from 15,550f / 293 dmg / 4 heals.
_ALLEY_Y_TOLERANCE = 6
_ALLEY_ATTACK_RANGE = 65
_ALLEY_MIN_RANGE = 8
_ALLEY_STANDOFF = 24
_ALLEY_ATTACK_HOLD = 1
_ALLEY_ATTACK_GAP = 2
# Skull & Crossbones / Wounded Knee: one less release frame improves the
# exact-entry clears without raising risk.  FullHardStage6: 19,892→16,793f,
# 802→526 damage; FullHardStage7: 18,856→17,484f, 924→830 damage.
_LATE_ATTACK_HOLD = 2
_LATE_ATTACK_GAP = 4
# Raphael's shorter weapon benefits from one fewer release frame on Wounded
# Knee waves, then a one-frame release against Leatherhead. Natural-entry
# probes: waves 18,156→10,573f / 497→362 dmg; boss 6,712→5,249f /
# 356→276 dmg.
_RAPH_WOUNDED_ATTACK_GAP = 3
_RAPH_LEATHERHEAD_ATTACK_GAP = 1
# Post-pickup step-out; 6f lines up Baxter entry for the Clean clear.
_PIZZA_DISENGAGE_FRAMES = 6
# Sewer Surfin' (stage byte 2): a broad lane tolerance avoids chasing Foot
# through the hanging spikes while keeping the normal two-frame poke.
_SEWER_Y_TOLERANCE = 36
_SEWER_MIN_FIGHT_Y = 160
_SEWER_ATTACK_RANGE = 64
_SEWER_MIN_RANGE = 8
_SEWER_STANDOFF = 24
_SEWER_ATTACK_HOLD = 2
_SEWER_ATTACK_GAP = 2
# Rat King: close far enough for Leo's swing to connect, but accept a wider
# water lane.  Combined Stage3 probe: 7,768f / 112 dmg / 2 emergency heals,
# down from 20,172f / 572 dmg / 9 heals.
_RAT_KING_Y_TOLERANCE = 32
# Leo weapon reach is generous (~50–68); keep a modest poke band.
# Runtime values come from ``active_knobs()`` so local grind can A/B.
_ATTACK_RANGE = 56
_MIN_RANGE = 8
_STANDOFF = 24
_ATTACK_HOLD = 2
_ATTACK_GAP = 5
# Rat King Footski: extended range + tight cadence; jump only to
# escape the auto-scroll left chip (standing jump-slashes whiff).
_BOSS_ATTACK_RANGE = 120
_BOSS_MIN_RANGE = 8
_BOSS_STANDOFF = 24
_BOSS_ATTACK_HOLD = 2
_BOSS_ATTACK_GAP = 1
_BOSS_LEFT_CHIP_X = 80
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


def _combat_knobs() -> tuple[int, int, int, int, int]:
    """Live poke-band knobs (defaults match module constants above)."""
    k = active_knobs()
    return (
        k.attack_range,
        k.min_range,
        k.standoff,
        k.attack_hold,
        k.attack_gap,
    )


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
_RAPH_CHAR = 8
# Raphael needs an aggressive closing jump for the short/elevated Foot stacks,
# but applying it to Starbase bruisers (0xb2/0xb4) jump-locks beside them.
_RAPH_STARBASE_CLOSE_CHARS: frozenset[int] = frozenset({0x6A, 0xB0, 0xBA})
_RAPH_STARBASE_GROUND_CHARS: frozenset[int] = frozenset({0xB2, 0xB4})
# Mode-7: enemies approach in depth (rising Y). Player Y clamps ~160–213;
# chasing far slots (y≪player) only burns frames. Fight the near band.
_NEON_MIN_FIGHT_Y = 140
_NEON_Y_TOLERANCE = 48
_NEON_ATTACK_RANGE = 68
_KRANG_CHAR = 0x4E
_KRANG_LEFT_STANDOFF = 36
_SHREDDER_F1_ATTACK_RANGE = 72
_SHREDDER_F1_Y_TOLERANCE = 8

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

        knobs = active_knobs()
        # Hop dodge cycle — short jump left/right off the projectile lane.
        if self._phase == "dodge":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "attack"
                self._timer = knobs.shredder_post_dodge_attack
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
            self._timer = knobs.shredder_attack_window
        self._cadence = (self._cadence + 1) % 8
        self._timer -= 1
        if self._timer <= 0:
            self._phase = "dodge"
            self._timer = knobs.shredder_dodge_frames
            return FrameAction(
                action=buttons("B", open_side),
                reason="shredder_dodge",
            )
        if self._cadence < 2 and adx_boss <= knobs.shredder_attack_adx:
            return FrameAction(action=buttons("Y"), reason="shredder_attack")
        if adx_boss < knobs.shredder_space_adx:
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
        knobs = active_knobs()
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
        if (
            status == self._SPIN
            and iframes <= 0
            and adx < knobs.slash_spin_dodge_adx
            and abs(dy) <= knobs.slash_spin_dodge_ady
        ):
            self._phase = "approach"
            return FrameAction(action=buttons("B", away), reason="slash_dodge")
        # Claw active only when already inside the hit band — don't kite.
        if (
            status in self._CLAW
            and iframes <= 0
            and adx < knobs.slash_claw_dodge_adx
        ):
            self._phase = "approach"
            return FrameAction(action=buttons("B", away), reason="slash_dodge")

        # Iframe / punish windows — full aggression (lab core).
        if iframes > 0 or status in self._PUNISH:
            if abs(dy) > 14 and adx < 40 and iframes <= 0:
                return FrameAction(
                    action=buttons("UP" if dy < 0 else "DOWN"),
                    reason="slash_align",
                )
            if adx > knobs.slash_punish_approach_adx:
                return FrameAction(action=buttons(toward), reason="slash_approach")
            if adx < knobs.slash_back_attack_adx:
                return FrameAction(
                    action=buttons(away, "Y"), reason="slash_back_attack"
                )
            # Brief re-flank hop every ~punish_cycle frames.
            self._punish_tick = (self._punish_tick + 1) % knobs.slash_punish_cycle
            if self._punish_tick < knobs.slash_punish_cross and adx < 40:
                return FrameAction(action=buttons("B", toward), reason="slash_cross")
            return FrameAction(action=buttons(toward, "Y"), reason="slash_back_attack")

        if abs(dy) > 10:
            self._phase = "approach"
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="slash_align",
            )

        # Thrash cycle — wider approach, shorter cross at low HP.
        low_hp = slash.health <= knobs.slash_low_hp
        approach_band = knobs.slash_approach_band
        cross_frames = (
            knobs.slash_cross_frames_low if low_hp else knobs.slash_cross_frames
        )
        attack_frames = (
            knobs.slash_attack_frames_low if low_hp else knobs.slash_attack_frames
        )

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
        self._timer = active_knobs().blocker_retreat_frames
        self._phase_frames = 0
        self._before_hp = health

    def _blocker_action(
        self,
        state: GameState,
        target: EnemyState,
    ) -> FrameAction:
        if not self._phase or self._phase.startswith("tank_"):
            self._start_blocker(target.health)
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


class PizzaSeek:
    """Walk to ground pizza (char ``0x30``) and tap Y when HP is not full.

    Pizza slots keep HP 0 so they never appear in ``living_enemies``. Seeking
    them is the main Clean-path heal for Stage 1 Big Apple (and later stages).

    Rules of thumb from Big Apple probes:
    - Any missing HP + pizza within ~56px → grab it (do not walk past).
    - Larger HP holes allow a longer seek band.
    - At critical HP (≤32), cross the screen for any visible box — Stage1
      heal=none died at HP 12/28 with pizza ~173–233px to the right while
      Leo kept walking left into the pack (``FAR_DIST`` 140 was too tight).
    - Do not abandon a boss fight for pizza — Boss.state's left box is
      collected by natural walk-in; active seek during Baxter regresses DPS.
    """

    _NEAR_DIST = 56
    _MID_DIST = 96
    _FAR_DIST = 180
    _SCREEN_DIST = 320
    _MID_HP = LEO_MAX_HP - 16
    _LOW_HP = 40
    _CRITICAL_HP = 32

    def __init__(self) -> None:
        self._disengage_frames = 0

    def next(self, state: GameState) -> FrameAction | None:
        """Return a seek/pickup action, or ``None`` when pizza is not useful."""
        # This tactic is tuned for Big Apple's fixed pizza placements. Later
        # stages can expose unreachable pickup slots (notably Skull &
        # Crossbones), so applying it globally can stall the continuous run.
        if state.mode is not GameMode.PLAYING or state.stage != 0:
            return None
        # After a pickup in a crowd, step out before resuming the poke.
        if self._disengage_frames > 0:
            self._disengage_frames -= 1
            return FrameAction(action=buttons("LEFT"), reason="pizza_disengage")
        if not (0 < state.health < LEO_MAX_HP):
            return None
        # Boss fights: natural walk-in only (active seek during Baxter
        # pulled Leo off the poke lane and failed Boss.state heal=none).
        if state.boss_active:
            return None
        pickups = state.extras.get("pickups") or ()
        if not pickups:
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
        elif state.health <= self._CRITICAL_HP:
            # Any on-screen pizza — Clean survival depends on this grab.
            max_dist = self._SCREEN_DIST
        elif state.health <= self._LOW_HP:
            max_dist = self._FAR_DIST
        elif state.health <= self._MID_HP:
            max_dist = self._MID_DIST
        else:
            # Scratch damage: only snag pizza we are already walking over.
            max_dist = self._NEAR_DIST
        if dist > max_dist:
            return None
        close_threats = sum(
            1
            for enemy in state.living_enemies
            if abs(enemy.x - state.player_x) < 24
            and abs(enemy.y - state.player_y) < 18
        )
        # Stay in the fray if surrounded unless pizza is already underfoot
        # or HP is low enough that the box is the survival play.
        if (
            close_threats > 0
            and dist >= 40
            and state.health > self._LOW_HP
        ):
            return None
        if abs(dx) <= 14 and abs(dy) <= 18:
            # Step out after a grab — length tunes Baxter entry alignment.
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


class BaxterTactics:
    """Hold left/mid lane when Baxter has no arena pizza.

    Continuous Stage1→Baxter enters on the right and stalls in
    ``edge_press``. Boss.state keeps a left pizza that naturally recenters
    Leo — skip lane overrides while that box (or any pickup) is present,
    and after it is eaten stay on the proven poke path (no lane thrash).
    """

    _LANE_X = 110
    _LANE_Y = 171

    def __init__(self) -> None:
        self._saw_arena_pizza = False

    def next(self, state: GameState) -> FrameAction | None:
        """Return a lane-correct step, or ``None`` when already in band."""
        if state.mode is not GameMode.PLAYING:
            return None
        if not (state.boss_active and state.stage == 0 and state.health > 0):
            self._saw_arena_pizza = False
            return None
        if state.extras.get("pickups"):
            self._saw_arena_pizza = True
            return None
        # Boss.state path: pizza already taught the lane — do not override.
        if self._saw_arena_pizza:
            return None
        dx = self._LANE_X - state.player_x
        dy = self._LANE_Y - state.player_y
        enemies = state.living_enemies
        baxter = enemies[0] if enemies else None
        if abs(dx) > 28 or abs(dy) > 24:
            if state.player_x > 150 or abs(dy) > 30:
                dirs: list[str] = []
                if abs(dx) > 6:
                    dirs.append("RIGHT" if dx > 0 else "LEFT")
                if abs(dy) > 6:
                    dirs.append("DOWN" if dy > 0 else "UP")
                if dirs:
                    return FrameAction(
                        action=buttons(*dirs), reason="baxter_lane"
                    )
        if (
            baxter is not None
            and baxter.x + 20 < state.player_x
            and state.player_x > 100
        ):
            return FrameAction(action=buttons("LEFT"), reason="baxter_releft")
        return None


class PlayerXStallWalk:
    """Walk right; break Stage 2 dumpster soft-locks when player X freezes.

    ``0x003A`` keeps ticking while Leo is glued to alley dumpsters. Deep
    dumpsters block the mid/upper lanes — drop to the bottom lane and
    JUMP+RIGHT, with UP / smash as fallbacks for earlier obstacles.
    Stage 0 (Big Apple) skips dumpster escapes — frozen X there is usually
    a wave lock, and DOWN thrash walks into chip.
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
        # Leonardo's frozen Stage 1 X is usually a wave lock, but Raphael
        # (player char 8) physically catches on the late dumpster at x=128.
        # Reuse the proven alley escape only for that character.
        if (
            state.stage == 0
            and int(state.extras.get("char_id", -1)) != _RAPH_CHAR
        ):
            return self._walk.next(state)
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
        # Raphael's short sai make generic Y-align oscillate forever beside
        # Starbase hover/stack targets. Jump toward the close elevated target;
        # this remains a normal B+Y attack and never uses the HP-draining A.
        if (
            state.stage == 8
            and not state.boss_active
            and int(state.extras.get("char_id", -1)) == _RAPH_CHAR
            and not any(
                enemy.kind in _RAPH_STARBASE_GROUND_CHARS
                for enemy in state.living_enemies
            )
        ):
            raph_targets = [
                enemy
                for enemy in state.living_enemies
                if enemy.kind in _RAPH_STARBASE_CLOSE_CHARS
            ]
            if raph_targets:
                target = min(
                    raph_targets,
                    key=lambda enemy: abs(enemy.x - state.player_x)
                    + abs(enemy.y - state.player_y),
                )
                dx = target.x - state.player_x
                dy = target.y - state.player_y
                if abs(dx) <= 80 and abs(dy) <= 36:
                    toward = "RIGHT" if dx > 0 else "LEFT"
                    steering = [toward]
                    if abs(dy) > 8:
                        steering.append("DOWN" if dy > 0 else "UP")
                    # Release B/Y between jumps. Holding them continuously
                    # lands one hit and leaves Raphael jump-locked beside the
                    # surviving stack.
                    if state.frame % 4:
                        return FrameAction(
                            action=buttons(*steering),
                            reason="raph_starbase_close_gap",
                        )
                    return FrameAction(
                        action=buttons("B", "Y", *steering),
                        reason="raph_starbase_jump",
                    )
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
            y_tol = (
                _SHREDDER_F1_Y_TOLERANCE
                if shredder_boss
                else _Y_TOLERANCE
            )
            attack_range = (
                _SHREDDER_F1_ATTACK_RANGE if shredder_boss else 72
            )
            min_range = 10
            standoff = 28
            cadence.hold_frames = _BOSS_ATTACK_HOLD
            cadence.gap_frames = _BOSS_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif _is_neon_highway(state):
            y_tol = _NEON_Y_TOLERANCE
            attack_range = _NEON_ATTACK_RANGE
            _, min_range, standoff, hold, gap = _combat_knobs()
            cadence.hold_frames = hold
            cadence.gap_frames = gap
            flank = PreferredFlank.NONE
        elif _is_sewer(state):
            y_tol = _SEWER_Y_TOLERANCE
            attack_range = _SEWER_ATTACK_RANGE
            min_range = _SEWER_MIN_RANGE
            standoff = _SEWER_STANDOFF
            cadence.hold_frames = _SEWER_ATTACK_HOLD
            cadence.gap_frames = _SEWER_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif state.stage == 1:
            y_tol = _ALLEY_Y_TOLERANCE
            attack_range = _ALLEY_ATTACK_RANGE
            min_range = _ALLEY_MIN_RANGE
            standoff = _ALLEY_STANDOFF
            cadence.hold_frames = _ALLEY_ATTACK_HOLD
            cadence.gap_frames = _ALLEY_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif state.stage == 0:
            # Big Apple waves + Baxter: wider standoff / tighter cadence.
            y_tol = _Y_TOLERANCE
            attack_range = _STAGE1_ATTACK_RANGE
            min_range = _STAGE1_MIN_RANGE
            standoff = _STAGE1_STANDOFF
            cadence.hold_frames = _STAGE1_ATTACK_HOLD
            cadence.gap_frames = _STAGE1_ATTACK_GAP
            flank = PreferredFlank.NONE
        elif state.stage in {5, 6}:
            # Pirate / train waves: exact-entry probes favor a slightly
            # tighter cadence. Pirate duo bosses use their branch above.
            y_tol = _Y_TOLERANCE
            attack_range = _ATTACK_RANGE
            min_range = _MIN_RANGE
            standoff = _STANDOFF
            cadence.hold_frames = _LATE_ATTACK_HOLD
            if (
                state.stage == 6
                and int(state.extras.get("char_id", -1)) == _RAPH_CHAR
            ):
                cadence.gap_frames = (
                    _RAPH_LEATHERHEAD_ATTACK_GAP
                    if state.boss_active
                    else _RAPH_WOUNDED_ATTACK_GAP
                )
            else:
                cadence.gap_frames = _LATE_ATTACK_GAP
            flank = PreferredFlank.NONE
        else:
            y_tol = _Y_TOLERANCE
            attack_range, min_range, standoff, hold, gap = _combat_knobs()
            cadence.hold_frames = hold
            cadence.gap_frames = gap
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
                or (state.boss_active and state.stage == 0)
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
            and not (
                int(state.extras.get("char_id", -1)) == _RAPH_CHAR
                and any(
                    e.kind in _RAPH_STARBASE_GROUND_CHARS
                    for e in state.living_enemies
                )
            )
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
        # Starbase holds Raphael at x=64 during its opening spawn delay.
        # Feeding those frames into the dumpster-stall detector pushes him
        # down a lane and desynchronizes the later wave triggers. Keep the
        # intended launch input until the stage actually starts moving.
        if state.stage == 8 and state.player_x <= 64:
            return FrameAction(
                action=buttons("RIGHT"),
                reason="starbase_launch_right",
            )
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
        self._pizza = PizzaSeek()
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

    def reset(self) -> None:
        """Reset cadence / walk stall and rebuild the tree."""
        self._cadence.reset()
        self._walk.reset()
        self._pizza = PizzaSeek()
        self._baxter = BaxterTactics()
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
        pizza = self._pizza.next(state)
        if pizza is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=pizza,
                reason=pizza.reason,
            )
        baxter = self._baxter.next(state)
        if baxter is not None:
            return TickResult(
                status=NodeStatus.RUNNING,
                action=baxter,
                reason=baxter.reason,
            )
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
