"""Reusable beat-em-up combat helpers for oneshot agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.behavior import (
    ActionNode,
    BehaviorNode,
    Condition,
    Selector,
    Sequence,
)
from snes_oneshot.game_state import EnemyState, GameState
from snes_oneshot.primitives import FrameAction

DEFAULT_Y_TOLERANCE = 8
DEFAULT_ATTACK_RANGE_X = 28
DEFAULT_MIN_RANGE_X = 10
DEFAULT_ATTACK_HOLD = 2
DEFAULT_ATTACK_GAP = 10
DEFAULT_GRAB_RANGE_X = 14
DEFAULT_PREFERRED_STANDOFF = 18
DEFAULT_LEFT_THREAT_X = 96
# Screen-lock edges: chasing past these bands walks into the scroll wall
# while thugs still chip from just off the playable band.
DEFAULT_CAMERA_LEFT_MARGIN = 40
# Playable band tops out near cam+170 during Slum locks.
DEFAULT_CAMERA_RIGHT_MARGIN = 160
DEFAULT_EDGE_ATTACK_BONUS = 24


def near_camera_left(
    state: GameState,
    *,
    margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
) -> bool:
    """True when the player is hugging the left side of the camera."""
    return state.player_x - state.camera_x <= margin


def near_camera_right(
    state: GameState,
    *,
    margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
) -> bool:
    """True when the player is hugging the right side of the camera."""
    return state.player_x - state.camera_x >= margin


class PreferredFlank(Enum):
    """Which side of the enemy the player should prefer to stand on."""

    NONE = auto()
    RIGHT = auto()
    LEFT = auto()


def vertical_delta(player_y: int, target_y: int) -> int:
    """Signed delta from player Y toward target Y."""
    return target_y - player_y


def is_vertically_aligned(
    player_y: int,
    target_y: int,
    *,
    tolerance: int = DEFAULT_Y_TOLERANCE,
) -> bool:
    """True when player Y is within tolerance of target Y."""
    return abs(vertical_delta(player_y, target_y)) <= tolerance


def align_vertical_action(
    state: GameState,
    target_y: int,
    *,
    tolerance: int = DEFAULT_Y_TOLERANCE,
    invert_vertical: bool = False,
) -> FrameAction:
    """Walk up/down toward target_y, or idle when already aligned.

    ``invert_vertical``: games where UP increases world Y (e.g. Final Fight
    SNES) should pass True so alignment presses the correct button.
    """
    dy = vertical_delta(state.player_y, target_y)
    if abs(dy) <= tolerance:
        return FrameAction(action=idle_action(), reason="aligned")
    # Normal screen coords: smaller Y is "up" → press UP when dy < 0.
    press_up = dy < 0
    if invert_vertical:
        press_up = not press_up
    if press_up:
        return FrameAction(action=buttons("UP"), reason="align_up")
    return FrameAction(action=buttons("DOWN"), reason="align_down")


def approach_x_action(
    state: GameState,
    target_x: int,
    *,
    attack_range: int = DEFAULT_ATTACK_RANGE_X,
    min_range: int = 0,
    camera_left_margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
    camera_right_margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
) -> FrameAction:
    """Walk toward target X, or idle when within attack range.

    When ``min_range`` > 0 and the player is closer than that, step away so
    melee does not sit inside the enemy collision box.
    """
    dx = target_x - state.player_x
    adx = abs(dx)
    if min_range > 0 and adx < min_range:
        if dx >= 0:
            return FrameAction(action=buttons("LEFT"), reason="space_left")
        return FrameAction(action=buttons("RIGHT"), reason="space_right")
    if adx <= attack_range:
        return FrameAction(action=idle_action(), reason="in_range")
    if dx > 0:
        if near_camera_right(state, margin=camera_right_margin):
            return FrameAction(action=idle_action(), reason="edge_wait")
        return FrameAction(action=buttons("RIGHT"), reason="approach_right")
    if near_camera_left(state, margin=camera_left_margin):
        return FrameAction(action=buttons("RIGHT"), reason="edge_space")
    return FrameAction(action=buttons("LEFT"), reason="approach_left")


def flank_approach_x_action(
    state: GameState,
    target_x: int,
    *,
    attack_range: int = DEFAULT_ATTACK_RANGE_X,
    min_range: int = DEFAULT_MIN_RANGE_X,
    preferred_flank: PreferredFlank = PreferredFlank.RIGHT,
    standoff: int = DEFAULT_PREFERRED_STANDOFF,
    camera_left_margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
    camera_right_margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
) -> FrameAction:
    """Close on X while preferring a safer side of the enemy.

    ``PreferredFlank.RIGHT`` keeps the player on the enemy's right when
    possible so left-spawning thugs walk into punches instead of the back.
    """
    if preferred_flank is PreferredFlank.NONE:
        return approach_x_action(
            state,
            target_x,
            attack_range=attack_range,
            min_range=min_range,
            camera_left_margin=camera_left_margin,
            camera_right_margin=camera_right_margin,
        )
    side = 1 if preferred_flank is PreferredFlank.RIGHT else -1
    stand = max(min_range, min(standoff, attack_range))
    ideal_x = target_x + side * stand
    dx_enemy = target_x - state.player_x
    adx = abs(dx_enemy)
    on_preferred = (state.player_x - target_x) * side >= 0
    if min_range > 0 and adx < min_range:
        # Too overlapped: step toward the preferred flank.
        if side > 0:
            return FrameAction(action=buttons("RIGHT"), reason="space_right")
        return FrameAction(action=buttons("LEFT"), reason="space_left")
    if on_preferred and adx <= attack_range:
        return FrameAction(action=idle_action(), reason="in_range")
    # Wrong side with the enemy ahead: close and punch from this face.
    # Circling past (old flank_right through the body) chips HP hard.
    if not on_preferred:
        return approach_x_action(
            state,
            target_x,
            attack_range=attack_range,
            min_range=min_range,
            camera_left_margin=camera_left_margin,
            camera_right_margin=camera_right_margin,
        )
    # On preferred flank but outside the band — hold the standoff lane.
    dx_ideal = ideal_x - state.player_x
    if abs(dx_ideal) <= 2:
        return FrameAction(action=idle_action(), reason="in_range")
    if dx_ideal > 0:
        if near_camera_right(state, margin=camera_right_margin):
            return FrameAction(action=idle_action(), reason="edge_wait")
        return FrameAction(action=buttons("RIGHT"), reason="approach_right")
    # Never walk into the left scroll wall during a screen lock.
    if near_camera_left(state, margin=camera_left_margin):
        return FrameAction(action=buttons("RIGHT"), reason="edge_space")
    return FrameAction(action=buttons("LEFT"), reason="approach_left")


def attack_action(*, button: str = "Y") -> FrameAction:
    """Emit a single-frame melee attack."""
    return FrameAction(action=buttons(button), reason="attack")


def grab_throw_action(
    state: GameState,
    target_x: int,
    *,
    attack_button: str = "Y",
) -> FrameAction:
    """Toward + attack — Cody shoulder throw / Haggar toss once grabbed.

    Walking into an enemy establishes the grab; toward+Y throws. Direction
    is toward the enemy so the throw fires even if not yet latched.
    """
    if target_x >= state.player_x:
        return FrameAction(
            action=buttons("RIGHT", attack_button),
            reason="throw_right",
        )
    return FrameAction(
        action=buttons("LEFT", attack_button),
        reason="throw_left",
    )


def pickup_action(*, button: str = "Y") -> FrameAction:
    """Press attack to pick up food / weapons underfoot."""
    return FrameAction(action=buttons(button), reason="pickup")


def walk_right_action() -> FrameAction:
    """Walk right (screen unlock / progress)."""
    return FrameAction(action=buttons("RIGHT"), reason="walk_right")


def select_combat_target(
    state: GameState,
    *,
    prefer_left_threat: bool = True,
    left_threat_x: int = DEFAULT_LEFT_THREAT_X,
) -> EnemyState | None:
    """Pick a living enemy, preferring nearby threats behind/left."""
    living = state.living_enemies
    if not living:
        return None
    if prefer_left_threat:
        behind = [
            e
            for e in living
            if e.x <= state.player_x
            and state.player_x - e.x <= left_threat_x
        ]
        if behind:
            return min(
                behind,
                key=lambda e: abs(e.x - state.player_x)
                + abs(e.y - state.player_y),
            )
    return state.nearest_enemy()


@dataclass
class WalkProgress:
    """Walk right, with periodic Y-lane nudges when the camera stalls."""

    stall_frames: int = 90
    nudge_period: int = 45
    pickup_every: int = 24
    _last_camera_x: int = -1
    _stalled_frames: int = 0
    _walk_frames: int = 0

    def reset(self) -> None:
        """Clear stall tracking."""
        self._last_camera_x = -1
        self._stalled_frames = 0
        self._walk_frames = 0

    def next(self, state: GameState) -> FrameAction:
        """Advance one walk frame; nudge UP/DOWN+RIGHT after a camera stall.

        Periodically taps attack while walking so food / weapons underfoot
        get picked up. Stall-phase also taps Y to break barrels in the path.
        """
        self._walk_frames += 1
        if (
            self.pickup_every > 0
            and self._walk_frames % self.pickup_every == 0
            and not state.living_enemies
        ):
            return FrameAction(
                action=buttons("RIGHT", "Y"),
                reason="pickup_walk",
            )
        if self._last_camera_x < 0:
            self._last_camera_x = state.camera_x
        if state.camera_x > self._last_camera_x:
            self._last_camera_x = state.camera_x
            self._stalled_frames = 0
            return walk_right_action()
        if state.living_enemies:
            self._stalled_frames = 0
            return walk_right_action()
        self._stalled_frames += 1
        if self._stalled_frames < self.stall_frames:
            return walk_right_action()
        phase = (
            (self._stalled_frames - self.stall_frames) // self.nudge_period
        ) % 4
        if phase == 0:
            return FrameAction(
                action=buttons("UP", "RIGHT"),
                reason="stall_up_right",
            )
        if phase == 1:
            return FrameAction(
                action=buttons("DOWN", "RIGHT"),
                reason="stall_down_right",
            )
        if phase == 2:
            return FrameAction(
                action=buttons("Y"),
                reason="stall_smash",
            )
        return walk_right_action()


@dataclass
class AttackCadence:
    """Press/release timing so melee is not held every frame."""

    hold_frames: int = DEFAULT_ATTACK_HOLD
    gap_frames: int = DEFAULT_ATTACK_GAP
    _phase: int = 0

    def reset(self) -> None:
        """Restart the press/release cycle."""
        self._phase = 0

    def next_attack(self, *, button: str = "Y") -> FrameAction:
        """Advance one frame; return attack press or idle gap."""
        period = self.hold_frames + self.gap_frames
        if period <= 0:
            return attack_action(button=button)
        in_hold = self._phase < self.hold_frames
        self._phase = (self._phase + 1) % period
        if in_hold:
            return attack_action(button=button)
        return FrameAction(action=idle_action(), reason="attack_gap")

    def next_throw(
        self,
        state: GameState,
        target_x: int,
        *,
        attack_button: str = "Y",
    ) -> FrameAction:
        """Cadenced toward+attack so throws are not mashed every frame."""
        period = self.hold_frames + self.gap_frames
        if period <= 0:
            return grab_throw_action(
                state, target_x, attack_button=attack_button
            )
        in_hold = self._phase < self.hold_frames
        self._phase = (self._phase + 1) % period
        if in_hold:
            return grab_throw_action(
                state, target_x, attack_button=attack_button
            )
        return FrameAction(action=idle_action(), reason="throw_gap")


def fight_nearest_action(
    state: GameState,
    *,
    y_tolerance: int = DEFAULT_Y_TOLERANCE,
    attack_range: int = DEFAULT_ATTACK_RANGE_X,
    min_range: int = DEFAULT_MIN_RANGE_X,
    attack_button: str = "Y",
    invert_vertical: bool = False,
    cadence: AttackCadence | None = None,
    preferred_flank: PreferredFlank = PreferredFlank.NONE,
    standoff: int = DEFAULT_PREFERRED_STANDOFF,
    use_throw: bool = False,
    grab_range: int = DEFAULT_GRAB_RANGE_X,
    prefer_left_threat: bool = False,
    left_threat_x: int = DEFAULT_LEFT_THREAT_X,
    camera_left_margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
    camera_right_margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
    edge_attack_bonus: int = DEFAULT_EDGE_ATTACK_BONUS,
    patient_approach: bool = False,
) -> FrameAction:
    """Align vertically, close X distance, then punch/throw nearest threat."""
    enemy = select_combat_target(
        state,
        prefer_left_threat=prefer_left_threat,
        left_threat_x=left_threat_x,
    )
    if enemy is None:
        return FrameAction(action=idle_action(), reason="no_enemy")
    dx = abs(enemy.x - state.player_x)
    on_left_edge = near_camera_left(
        state, margin=camera_left_margin
    )
    on_right_edge = near_camera_right(
        state, margin=camera_right_margin
    )
    enemy_left = enemy.x < state.player_x
    enemy_right = enemy.x > state.player_x
    enemy_screen_x = enemy.x - state.camera_x
    # Hard clamp: never stand past the scroll walk-limit during a lock.
    # Skip when any living thug is on/behind us (do not walk into them).
    hold_limit = state.camera_x + camera_right_margin + 4
    left_threat = any(e.x <= state.player_x for e in state.living_enemies)
    if (
        state.screen_locked
        and state.player_x > hold_limit + 6
        and not left_threat
    ):
        return FrameAction(
            action=buttons("LEFT"), reason="edge_recenter"
        )
    # Thugs parked past the right walk limit (~cam+190): close X to the
    # hold first. Y-aligning mid-screen while they chip burns the last life.
    unreachable_right = (
        state.screen_locked
        and enemy_right
        and enemy_screen_x > camera_right_margin + 8
    )
    right_edge_fight = unreachable_right or (
        state.screen_locked and on_right_edge and enemy_right
    )
    # Also treat mid-screen vs far-right park as right-edge so flank
    # approach cannot walk past the hold toward a closing thug.
    if (
        state.screen_locked
        and enemy_right
        and enemy_screen_x > camera_right_margin - 20
    ):
        right_edge_fight = True
    if right_edge_fight:
        left_other = any(
            e.slot != enemy.slot and e.x <= state.player_x
            for e in state.living_enemies
        )
        hold_x = state.camera_x + camera_right_margin + 4
        # Wait further left than the walk-limit so far-park jump kicks whiff.
        # Tough/patient: deeper wait keeps dx>95 vs park~sx197.
        wait_x = state.camera_x + (72 if patient_approach else 100)
        # Never stand past the walk limit — RIGHT+throw walks off the band.
        if not left_other and state.player_x > hold_x + 6:
            return FrameAction(
                action=buttons("LEFT"), reason="edge_recenter"
            )
        # Engage when the thug has closed into poke distance (~cam+205).
        engage = enemy_screen_x <= camera_right_margin + 45
        # Hold sits at ~cam+164; engage-boundary thugs are dx≈40 — need a
        # little slack past attack_range so the first poke connects.
        in_punch = dx <= attack_range + 8
        dy = abs(enemy.y - state.player_y)
        if in_punch:
            # Poke spacing: step out of overlap before trading.
            if (
                not left_other
                and min_range > 0
                and dx < min_range + 2
                and enemy_right
            ):
                return FrameAction(
                    action=buttons("LEFT"), reason="space_left"
                )
            # Attack steps can drift past the walk limit — correct first.
            if not left_other and state.player_x >= hold_x:
                return FrameAction(
                    action=buttons("LEFT"), reason="edge_recenter"
                )
            # Only micro-align at punch X. Large Y-chase / punch trades hurt.
            if dy > y_tolerance + 4 and dy <= y_tolerance + 12:
                return align_vertical_action(
                    state,
                    enemy.y,
                    tolerance=y_tolerance,
                    invert_vertical=invert_vertical,
                )
            if dy > y_tolerance + 12:
                return FrameAction(
                    action=idle_action(), reason="edge_wait"
                )
            # Tough thug: keep a lane offset before poking to cut trades.
            if patient_approach and dy < 6:
                if enemy.y >= state.player_y:
                    away = "DOWN" if invert_vertical else "UP"
                else:
                    away = "UP" if invert_vertical else "DOWN"
                return FrameAction(
                    action=buttons(away), reason="edge_desync"
                )
            # Throw only toward LEFT here — throw_right walks past walk X.
            if (
                use_throw
                and dx <= grab_range + 8
                and enemy.x <= state.player_x
            ):
                if cadence is not None:
                    return cadence.next_throw(
                        state, enemy.x, attack_button=attack_button
                    )
                return grab_throw_action(
                    state, enemy.x, attack_button=attack_button
                )
            if cadence is not None:
                return cadence.next_attack(button=attack_button)
            return attack_action(button=attack_button)
        # Tough thug: park at deep wait + Y-desync; never walk the kick band.
        # Let them close into punch range (avoids press/retreat stalls).
        if patient_approach and not left_other:
            if state.player_x > wait_x + 14:
                return FrameAction(
                    action=buttons("LEFT"), reason="edge_recenter"
                )
            if state.player_x < wait_x - 10:
                return FrameAction(
                    action=buttons("RIGHT"), reason="edge_mid"
                )
            if dy < y_tolerance + 14:
                # Increase |dy| so jump kicks on our old lane whiff.
                if enemy.y >= state.player_y:
                    away = "DOWN" if invert_vertical else "UP"
                else:
                    away = "UP" if invert_vertical else "DOWN"
                return FrameAction(
                    action=buttons(away), reason="edge_desync"
                )
            return FrameAction(action=idle_action(), reason="edge_wait")
        target_x = hold_x if engage else wait_x
        # Kick band (dx 45–95): never idle at the HOLD (w3 chips dx≈78–88
        # at psx≈170). Mid-screen idle is safer than walking through kicks.
        in_kick_dx = enemy_right and 45 <= dx <= 95
        player_sx = state.player_x - state.camera_x
        holdish = player_sx >= camera_right_margin - 20
        if (
            not left_other
            and state.player_x < target_x - 10
            and not (in_kick_dx and not engage)
        ):
            return FrameAction(
                action=buttons("RIGHT"),
                reason="edge_press" if engage else "edge_mid",
            )
        if not left_other and state.player_x > target_x + 14:
            if in_kick_dx and not engage:
                if holdish:
                    return FrameAction(
                        action=buttons("LEFT"), reason="edge_recenter"
                    )
                return FrameAction(
                    action=idle_action(), reason="edge_wait"
                )
            return FrameAction(
                action=buttons("LEFT"), reason="edge_recenter"
            )
        # Engaged but still in kick dx at the hold: fall back to wait
        # instead of idling into jump kicks (w3 chips at psx≈170).
        if engage and in_kick_dx and not left_other:
            if state.player_x > wait_x + 8:
                return FrameAction(
                    action=buttons("LEFT"), reason="edge_recenter"
                )
            return FrameAction(action=idle_action(), reason="edge_wait")
        # Far park: do NOT chase enemy Y (jump-kick lane). Hold lane.
        if not engage:
            return FrameAction(action=idle_action(), reason="edge_wait")
        if not is_vertically_aligned(
            state.player_y, enemy.y, tolerance=y_tolerance
        ):
            return align_vertical_action(
                state,
                enemy.y,
                tolerance=y_tolerance,
                invert_vertical=invert_vertical,
            )
        return FrameAction(action=idle_action(), reason="edge_wait")
    # Grab/throw from above/below is preferred in Final Fight — allow throw
    # before perfect Y align when already close on X. During a lock, never
    # throw_right — RIGHT+Y walks into the scroll wall.
    can_throw = use_throw and dx <= grab_range
    if can_throw and state.screen_locked and enemy.x >= state.player_x:
        can_throw = False
    if can_throw:
        if cadence is not None:
            return cadence.next_throw(
                state, enemy.x, attack_button=attack_button
            )
        return grab_throw_action(
            state, enemy.x, attack_button=attack_button
        )
    if not is_vertically_aligned(
        state.player_y, enemy.y, tolerance=y_tolerance
    ):
        return align_vertical_action(
            state,
            enemy.y,
            tolerance=y_tolerance,
            invert_vertical=invert_vertical,
        )
    # Left scroll wall: punch early so left-spawns walk into hits.
    effective_range = attack_range
    if on_left_edge and enemy_left:
        effective_range = attack_range + max(0, edge_attack_bonus)
    in_band = dx <= effective_range and (
        min_range <= 0 or dx >= min_range
    )
    if in_band:
        if cadence is not None:
            return cadence.next_attack(button=attack_button)
        return attack_action(button=attack_button)
    if on_left_edge and enemy_left:
        return FrameAction(action=buttons("RIGHT"), reason="edge_space")
    # Sandwiched between two living thugs: throw/space the nearer one;
    # never walk deeper into the pocket.
    living = state.living_enemies
    if len(living) >= 2:
        left_any = any(e.x < state.player_x for e in living)
        right_any = any(e.x > state.player_x for e in living)
        if left_any and right_any and use_throw and dx <= grab_range + 6:
            if cadence is not None:
                return cadence.next_throw(
                    state, enemy.x, attack_button=attack_button
                )
            return grab_throw_action(
                state, enemy.x, attack_button=attack_button
            )
        if left_any and right_any and dx <= attack_range + 8:
            if cadence is not None:
                return cadence.next_attack(button=attack_button)
            return attack_action(button=attack_button)
    if preferred_flank is not PreferredFlank.NONE:
        return flank_approach_x_action(
            state,
            enemy.x,
            attack_range=attack_range,
            min_range=min_range,
            preferred_flank=preferred_flank,
            standoff=standoff,
            camera_left_margin=camera_left_margin,
            camera_right_margin=camera_right_margin,
        )
    return approach_x_action(
        state,
        enemy.x,
        attack_range=attack_range,
        min_range=min_range,
        camera_left_margin=camera_left_margin,
        camera_right_margin=camera_right_margin,
    )


def enemies_present(state: GameState) -> bool:
    """True when any living enemy remains."""
    return bool(state.living_enemies)


def build_segment_tree(
    *,
    handle_continue: BehaviorNode | None = None,
    boss_policy: BehaviorNode | None = None,
    y_tolerance: int = DEFAULT_Y_TOLERANCE,
    attack_range: int = DEFAULT_ATTACK_RANGE_X,
    min_range: int = DEFAULT_MIN_RANGE_X,
    attack_button: str = "Y",
    invert_vertical: bool = False,
    cadence: AttackCadence | None = None,
    walk_progress: WalkProgress | None = None,
    preferred_flank: PreferredFlank = PreferredFlank.NONE,
    standoff: int = DEFAULT_PREFERRED_STANDOFF,
    use_throw: bool = False,
    grab_range: int = DEFAULT_GRAB_RANGE_X,
    prefer_left_threat: bool = False,
    left_threat_x: int = DEFAULT_LEFT_THREAT_X,
    camera_left_margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
    camera_right_margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
    edge_attack_bonus: int = DEFAULT_EDGE_ATTACK_BONUS,
) -> Selector:
    """Build the standard beat-em-up segment selector.

    Priority: continue → level complete idle → boss → fight nearest → walk
    right. Callers supply game-specific continue/boss stubs when needed.
    """
    children: list[BehaviorNode] = []
    if handle_continue is not None:
        children.append(handle_continue)
    children.append(
        Condition(lambda s: s.level_complete, name="level_complete")
    )
    if boss_policy is not None:
        children.append(boss_policy)

    def _fight(s: GameState) -> FrameAction:
        return fight_nearest_action(
            s,
            y_tolerance=y_tolerance,
            attack_range=attack_range,
            min_range=min_range,
            attack_button=attack_button,
            invert_vertical=invert_vertical,
            cadence=cadence,
            preferred_flank=preferred_flank,
            standoff=standoff,
            use_throw=use_throw,
            grab_range=grab_range,
            prefer_left_threat=prefer_left_threat,
            left_threat_x=left_threat_x,
            camera_left_margin=camera_left_margin,
            camera_right_margin=camera_right_margin,
            edge_attack_bonus=edge_attack_bonus,
        )

    # Fail (not succeed) when clear so the selector falls through to walk.
    children.append(
        Sequence(
            [
                Condition(enemies_present, name="enemies_present"),
                ActionNode(_fight, name="fight_nearest"),
            ],
            name="fight_seq",
        )
    )
    walker = walk_progress or WalkProgress()

    def _walk(s: GameState) -> FrameAction:
        return walker.next(s)

    children.append(ActionNode(_walk, name="walk_right"))
    return Selector(children, name="segment_clear")
