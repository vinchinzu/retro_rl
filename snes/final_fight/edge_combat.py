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

# West Area1 Andore (HP≈250): face-Y punch band, grab-throw on overlap.
# Continuous LEFT+Y whiffs; chasing knockdown flyaway walks the gutter
# or right fence and dies. Probe: ``scripts/stage3_area1_probe.py``.
AREA1_FENCE_SX = 165
AREA1_GUTTER_SX = 52
AREA1_FAR_ADX = 70
AREA1_PUNCH_LO = 24
AREA1_PUNCH_HI = 70
AREA1_GRAB_ADX = 16


def area1_andore_action(
    *,
    frame: int,
    sx: int,
    dx: int | None,
    dy: int = 0,
    status: int = 3,
    faced: bool = False,
    enemy_hp: int = 250,
) -> tuple[FrameAction, bool]:
    """One-frame Area1 Andore recipe: clamp, wait-far, face-Y, throw.

    ``dx`` is enemy_x - player_x (behind is negative). ``status`` is the
    entity byte (``3`` fighting, ``1`` spawn/KD flyaway). ``faced`` tracks
    a brief LEFT/RIGHT face before bare Y — holding dir+Y whiffs.
    HP≤50 waits for Andore to walk into grab, then UP+Y (chasing
    after the first throw walks the fence and dies).
    """
    if sx > AREA1_FENCE_SX:
        # JD-left off the lock edge — ground LEFT is too slow vs Andore push.
        return FrameAction(action=buttons("B", "LEFT"), reason="clamp_l"), faced
    if sx < AREA1_GUTTER_SX:
        return FrameAction(action=buttons("RIGHT"), reason="clamp_r"), faced
    if dx is None:
        if sx < 110:
            return FrameAction(action=buttons("RIGHT"), reason="walk"), faced
        return FrameAction(action=idle_action(), reason="wait"), faced

    adx = abs(dx)
    toward = "RIGHT" if dx >= 0 else "LEFT"
    # Knockdown flyaway (~40-dmg throw launches adx 100+). Hold mid;
    # chasing walks the gutter/fence and dies.
    if (status == 1 and adx > 50) or adx > AREA1_FAR_ADX:
        if sx > 140:
            return FrameAction(action=buttons("LEFT"), reason="clamp_l"), faced
        if sx < 75:
            return FrameAction(action=buttons("RIGHT"), reason="clamp_r"), faced
        return FrameAction(action=idle_action(), reason="wait_far"), faced
    # Crumb: do not chase. Hop out of Andore's grab, throw at 16–32.
    if enemy_hp <= 50:
        if adx < 12:
            away = "RIGHT" if dx < 0 else "LEFT"
            if away == "LEFT" and sx < 70:
                away = "RIGHT"
            if away == "RIGHT" and sx > 140:
                away = "LEFT"
            return FrameAction(action=buttons("B", away), reason="space"), faced
        if adx > 32:
            if abs(dy) < 8:
                away_v = "DOWN" if dy >= 0 else "UP"
                return FrameAction(action=buttons(away_v), reason="desync"), faced
            return FrameAction(action=idle_action(), reason="wait_far"), faced
        cycle = frame % 4
        if cycle == 0:
            return FrameAction(action=buttons("UP", "Y"), reason="throw"), faced
        if cycle == 1:
            return (
                FrameAction(action=buttons(toward, "Y"), reason="throw"),
                faced,
            )
        if cycle == 2:
            return FrameAction(action=buttons("LEFT", "Y"), reason="throw"), faced
        return FrameAction(action=idle_action(), reason="gap"), faced
    if abs(dy) > 10 and adx > 20:
        vert = "UP" if dy > 0 else "DOWN"
        return FrameAction(action=buttons(vert), reason="align"), faced
    if adx <= 18:
        cycle = frame % 4
        if cycle == 0:
            return FrameAction(action=buttons("UP", "Y"), reason="throw"), faced
        if cycle == 1:
            return (
                FrameAction(action=buttons(toward, "Y"), reason="throw"),
                faced,
            )
        if cycle == 2:
            return FrameAction(action=buttons("LEFT", "Y"), reason="throw"), True
        return FrameAction(action=buttons("RIGHT"), reason="space"), faced
    if adx <= 32:
        if dx < 0 and not faced:
            return FrameAction(action=buttons("LEFT"), reason="face"), True
        if frame % 3 < 2:
            return FrameAction(action=buttons("UP", "Y"), reason="throw"), faced
        return FrameAction(action=buttons(toward), reason="close"), faced
    # Walk into grab range. Bare Y only as a 2/12 pulse while closing
    # behind — LEFT+Y from here is the documented 0-dmg lock.
    if dx < 0 and sx > 70:
        if not faced:
            return FrameAction(action=buttons("LEFT"), reason="face"), True
        if frame % 12 < 2:
            return FrameAction(action=buttons("Y"), reason="y"), faced
        return FrameAction(action=buttons("LEFT"), reason="close"), True
    if toward == "RIGHT" and sx >= 150:
        return FrameAction(action=buttons("B", "LEFT"), reason="clamp_l"), faced
    return FrameAction(action=buttons(toward), reason="close"), faced


def area1_andore_from_snap(
    frame: int,
    enemy: dict[str, int] | None,
    *,
    sx: int,
    faced: bool,
) -> tuple[FrameAction, bool]:
    """Probe adapter: enemy snap dict ``{dx, dy, st, hp}`` or ``None``."""
    if enemy is None:
        return area1_andore_action(frame=frame, sx=sx, dx=None, faced=faced)
    return area1_andore_action(
        frame=frame,
        sx=sx,
        dx=int(enemy["dx"]),
        dy=int(enemy.get("dy", 0)),
        status=int(enemy.get("st", 3)),
        faced=faced,
        enemy_hp=int(enemy.get("hp", 250)),
    )


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
