"""Final Fight edge-lock / patient-approach combat policy.

Extracted from shared ``retro_harness.combat`` so the harness stays as pure
melee primitives. Covers right-edge park waits, kick-band retreats, and the
tough-thug patient_approach desync used on Slum locks and wave-3/4 chips.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.combat import (
    DEFAULT_ATTACK_RANGE_X,
    DEFAULT_CAMERA_LEFT_MARGIN,
    DEFAULT_CAMERA_RIGHT_MARGIN,
    DEFAULT_EDGE_ATTACK_BONUS,
    DEFAULT_GRAB_RANGE_X,
    DEFAULT_LEFT_THREAT_X,
    DEFAULT_MIN_RANGE_X,
    DEFAULT_PREFERRED_STANDOFF,
    DEFAULT_Y_TOLERANCE,
    AttackCadence,
    PreferredFlank,
    align_vertical_action,
    attack_action,
    fight_nearest_action,
    grab_throw_action,
    is_vertically_aligned,
    near_camera_left,
    near_camera_right,
    select_combat_target,
)
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import EnemyState, GameState


def _right_edge_fight_action(
    state: GameState,
    enemy: EnemyState,
    *,
    y_tolerance: int,
    attack_range: int,
    min_range: int,
    attack_button: str,
    invert_vertical: bool,
    cadence: AttackCadence | None,
    use_throw: bool,
    grab_range: int,
    camera_right_margin: int,
    patient_approach: bool,
) -> FrameAction | None:
    """Handle right-edge park / kick-band fights, or return None to fall through."""
    dx = abs(enemy.x - state.player_x)
    on_right_edge = near_camera_right(state, margin=camera_right_margin)
    enemy_right = enemy.x > state.player_x
    enemy_screen_x = enemy.x - state.camera_x
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
    if not right_edge_fight:
        return None

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
        return FrameAction(action=buttons("LEFT"), reason="edge_recenter")
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
            return FrameAction(action=buttons("LEFT"), reason="space_left")
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
            return FrameAction(action=idle_action(), reason="edge_wait")
        # Tough thug: keep a lane offset before poking to cut trades.
        if patient_approach and dy < 6:
            if enemy.y >= state.player_y:
                away = "DOWN" if invert_vertical else "UP"
            else:
                away = "UP" if invert_vertical else "DOWN"
            return FrameAction(action=buttons(away), reason="edge_desync")
        # Throw only toward LEFT here — throw_right walks past walk X.
        if use_throw and dx <= grab_range + 8 and enemy.x <= state.player_x:
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
            return FrameAction(action=buttons("RIGHT"), reason="edge_mid")
        if dy < y_tolerance + 14:
            # Increase |dy| so jump kicks on our old lane whiff.
            if enemy.y >= state.player_y:
                away = "DOWN" if invert_vertical else "UP"
            else:
                away = "UP" if invert_vertical else "DOWN"
            return FrameAction(action=buttons(away), reason="edge_desync")
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
            return FrameAction(action=idle_action(), reason="edge_wait")
        return FrameAction(action=buttons("LEFT"), reason="edge_recenter")
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


def ff_fight_nearest_action(
    state: GameState,
    *,
    y_tolerance: int = DEFAULT_Y_TOLERANCE,
    attack_range: int = DEFAULT_ATTACK_RANGE_X,
    min_range: int = DEFAULT_MIN_RANGE_X,
    attack_button: str = "Y",
    invert_vertical: bool = True,
    cadence: AttackCadence | None = None,
    preferred_flank: PreferredFlank = PreferredFlank.RIGHT,
    standoff: int = DEFAULT_PREFERRED_STANDOFF,
    use_throw: bool = False,
    grab_range: int = DEFAULT_GRAB_RANGE_X,
    prefer_left_threat: bool = True,
    left_threat_x: int = DEFAULT_LEFT_THREAT_X,
    camera_left_margin: int = DEFAULT_CAMERA_LEFT_MARGIN,
    camera_right_margin: int = DEFAULT_CAMERA_RIGHT_MARGIN,
    edge_attack_bonus: int = DEFAULT_EDGE_ATTACK_BONUS,
    patient_approach: bool = False,
) -> FrameAction:
    """FF melee: right-edge park / patient tactics, then shared melee path."""
    enemy = select_combat_target(
        state,
        prefer_left_threat=prefer_left_threat,
        left_threat_x=left_threat_x,
    )
    if enemy is None:
        return FrameAction(action=idle_action(), reason="no_enemy")

    # Hard clamp: never stand past the scroll walk-limit during a lock.
    # Skip when any living thug is on/behind us (do not walk into them).
    hold_limit = state.camera_x + camera_right_margin + 4
    left_threat = any(e.x <= state.player_x for e in state.living_enemies)
    if (
        state.screen_locked
        and state.player_x > hold_limit + 6
        and not left_threat
    ):
        return FrameAction(action=buttons("LEFT"), reason="edge_recenter")

    edge = _right_edge_fight_action(
        state,
        enemy,
        y_tolerance=y_tolerance,
        attack_range=attack_range,
        min_range=min_range,
        attack_button=attack_button,
        invert_vertical=invert_vertical,
        cadence=cadence,
        use_throw=use_throw,
        grab_range=grab_range,
        camera_right_margin=camera_right_margin,
        patient_approach=patient_approach,
    )
    if edge is not None:
        return edge

    # Non-edge: shared generic melee (align → approach → attack).
    # Suppress shared right-hold clamp when a left threat is present; it
    # already skips on left_threat, so fall-through is safe.
    return fight_nearest_action(
        state,
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
