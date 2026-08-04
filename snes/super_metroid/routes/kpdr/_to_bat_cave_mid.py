"""Bubble → Bat Cave mid-budget loop: save-door runway / lip + climb.

Internal helper for :mod:`to_bat_cave`. One call covers Phase B→D under
a shared outer-iteration budget (:data:`bubble_to_bat.MID_FRAMES`).
Control state is local to :func:`run_mid_loop` — not on ``ClimbTrack``.
"""

from __future__ import annotations

from typing import Literal

from super_metroid.routes.controller_common import hold
from super_metroid.routes.skills.policies import bubble_to_bat as P
from super_metroid.routes.skills.geometry import (
    ClimbTrack,
    avoid_wrong_door,
    is_stand_pin_pose,
    is_true_ground,
    on_launch_lip,
    on_mid_iso_pin,
    on_right_shelf,
    on_save_runway,
    phase_c_usable_right_contact,
    phase_d_top_band,
    track_state,
)
from super_metroid.routes.skills.runway import (
    save_runway_fire_recipe,
    seat_max_left_fire,
)
from super_metroid.routes.runtime import ControllerSession

MidStart = Literal["launch", "climb"]

ROOM_ID = P.ROOM_ID


def run_mid_loop(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    start: MidStart = "launch",
    policy=P,
) -> None:
    """Shared mid-frame budget: lip launch (optional) + right-structure climb.

    ``start="launch"`` (product): approach lip → charged HJ → climb.
    ``start="climb"`` (handoff): skip lip; sticky height class already earned.

    Outer iterations ≤ ``policy.MID_FRAMES`` (multi-frame holds inside a tick do not
    each consume a budget unit — same accounting as pre-split controller).
    """
    label = track.label
    lip_lo, lip_hi = policy.LIP_X
    lip_y_lo, lip_y_hi = policy.LIP_Y
    stand_lo, stand_hi = policy.MID_STAND_X

    phase: MidStart = start
    mid_i = 0
    frames_used = 0
    height_class = start == "climb"
    if start == "climb":
        track.launched = True
        track.mid_reached = True
        # Climb handoff (Phase-C dump) must sticky-right immediately — dump
        # at ~(301,429) falls out of Phase C in a few idle frames.
        st0 = session.state
        if (
            phase_c_usable_right_contact(st0, policy)
            or (
                st0.room_id == ROOM_ID
                and st0.samus_x >= policy.RIGHT_SHELF_X - 20
                and st0.samus_y <= policy.PHASE_C_Y_MAX + 40
            )
        ):
            track.phase_c_hit = True
            height_class = True

    while frames_used < policy.MID_FRAMES:
        frames_used += 1
        state = session.state
        if state.room_id != ROOM_ID:
            break
        track_state(session, track, state, policy)
        if phase_d_top_band(state, policy):
            track.top_reached = True
            break
        if avoid_wrong_door(session, track, state, policy):
            continue
        if state.pose in (137, 138):
            for _ in range(10):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_kb")
            continue

        x = state.samus_x
        y = state.samus_y
        if y <= policy.HEIGHT_CLASS_Y:
            height_class = True
        if x > policy.CAVITY_X_MAX and y > policy.TOP_Y:
            hold(session, 1, "LEFT", "B", reason=f"{label}_mid_cap")
            continue

        mid_i += 1

        # --- launch: R6 lip (height class) + R15/R18 max-left save runway ---
        if phase == "launch" and not track.launched:
            # R18: fire only from human max-left seat band (x∈[25,32]).
            # Wider fire window (25–60) shortens runway / no p132 from x~48–50.
            # Never LEFT+X while walking (KB p138). Stationary clear + brake.
            # Do NOT steal solid lip seats (x~79 y~427) — pure regress.
            human_lo, human_hi = policy.SAVE_HUMAN_SEAT_X
            fire_lo, fire_hi = policy.SAVE_RUNWAY_FIRE_X
            on_runway = on_save_runway(state, policy) and not on_launch_lip(state, policy)
            seated_max_left = (
                on_runway
                and human_lo <= x <= human_hi
                and is_true_ground(state, poses=policy.TRUE_GROUND)
                and state.pose not in (137, 138)
            )
            if seated_max_left:
                # R19: phase-wait (idle) → prepare → dash → spin → double WJ.
                # R18: arm-pump GREEN on human pin, RED on pure max-left seat.
                # Bare dash + enemy-phase wait clears Phase D on full pure seat
                # (R18 open-loop alone hard-caps mx200≈251 — Geruta AI phase).
                save_runway_fire_recipe(
                    session,
                    track,
                    policy=policy,
                    y_clear=True,
                    crouch=False,
                    arm_pump=policy.SAVE_ARM_PUMP,  # R18 pure: False
                    wj_count=2,
                    phase_wait=True,
                )
                state = session.state
                if track.min_y <= policy.HEIGHT_CLASS_Y or state.samus_y <= policy.HEIGHT_CLASS_Y:
                    height_class = True
                track.launched = True
                phase = "climb"
                mid_i = 0
                if track.top_reached or state.room_id != ROOM_ID:
                    break
                continue

            # On save platform: seat max-left (stationary X + brake), not fire yet.
            if on_runway and (x > human_hi or x < human_lo or x > fire_hi):
                if x < fire_lo:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_save_align_r")
                    continue
                # Cap seat attempts so lip fallback still runs for height class.
                if mid_i < 8:
                    seat_max_left_fire(session, track, policy=policy)
                    continue
                # Fall through to lip / mid-iso after several seat tries.

            # R6 solid lip — proven pure height class min_y=260. Keep for
            # regression; R15 Phase D needs max-left fire window seat separately.
            if on_launch_lip(state, policy):
                if x < 70:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_lip_align")
                    continue
                if x > 90:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_lip_align_l")
                    continue
                for _ in range(policy.LIP_CHARGE):
                    hold(session, 1, "A", reason=f"{label}_lip_charge")
                for _ in range(policy.LIP_SPIN):
                    state = hold(
                        session, 1, "RIGHT", "B", "A", reason=f"{label}_lip_hj"
                    )
                    track_state(session, track, state, policy)
                    if state.room_id != ROOM_ID:
                        break
                    if state.samus_y <= policy.HEIGHT_CLASS_Y:
                        height_class = True
                    if phase_d_top_band(state, policy):
                        track.top_reached = True
                        break
                if (
                    not track.top_reached
                    and height_class
                    and state.room_id == ROOM_ID
                ):
                    for _ in range(policy.LIP_EXTEND):
                        state = hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason=f"{label}_ol_extend",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if phase_d_top_band(state, policy):
                            track.top_reached = True
                            break
                        if (
                            state.samus_x >= policy.RIGHT_SHELF_X
                            and state.samus_y <= policy.MIDHIGH_Y
                        ):
                            break
                        if (
                            is_true_ground(state, poses=policy.TRUE_GROUND)
                            and state.samus_y <= policy.MID_RESEAT_Y
                            and 140 <= state.samus_x < policy.RIGHT_SHELF_X
                        ):
                            break
                track.launched = True
                phase = "climb"
                mid_i = 0
                if track.top_reached or state.room_id != ROOM_ID:
                    break
                continue

            # From mid-iso pin (not lip): edge left onto save runway seat.
            if (
                on_mid_iso_pin(state, policy)
                and not on_launch_lip(state, policy)
                and y <= policy.SAVE_RUNWAY_Y[1]
            ):
                run_lo, run_hi = policy.SAVE_RUNWAY_X
                if x > run_hi:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_to_save_l")
                elif x < run_lo:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_to_save_r")
                else:
                    hold(session, 1, reason=f"{label}_save_wait")
                continue

            # Unstable mid float (y≤410, pin band): drop left onto solid lip.
            if y <= lip_y_lo and stand_lo - 10 <= x <= stand_hi + 20:
                if x > lip_hi:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_drop_left")
                else:
                    hold(session, 1, reason=f"{label}_drop_idle")
                continue

            # Too far right / low cavity: pull left then HJ toward lip.
            if x > 160 and y > lip_y_lo:
                hold(session, 1, "LEFT", "B", reason=f"{label}_to_lip_left")
                continue
            if is_stand_pin_pose(state, poses=policy.STAND_PIN):
                dir_h = "LEFT" if x > lip_hi else "RIGHT"
                if y > lip_y_hi + 20:
                    for _ in range(10):
                        hold(session, 1, "A", reason=f"{label}_below_charge")
                    for _ in range(36):
                        state = hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            "A",
                            reason=f"{label}_below_hj",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if on_launch_lip(state, policy) or on_save_runway(state, policy):
                            break
                    continue
                hold(session, 1, dir_h, "B", reason=f"{label}_to_lip_walk")
                continue
            dir_h = "LEFT" if x > 100 else "RIGHT"
            hold(session, 1, dir_h, "B", "A", reason=f"{label}_to_lip_air")
            if mid_i > 600:
                # Budget escape: attempt climb from wherever we are.
                phase = "climb"
                mid_i = 0
            continue

        # --- climb (R7/R10/R13): mid-high open-loop / floor-reclimb / top ---
        if phase == "climb":
            grounded = is_true_ground(state, poses=policy.TRUE_GROUND)
            # Sticky Phase-C right-structure mode: once usable right contact
            # fired, never re-drop to floor runway — climb the right wall.
            phase_c_sticky = track.phase_c_hit or phase_c_usable_right_contact(state, policy)
            if grounded:
                # R9: right-structure shelf → LEFT charged HJ to top band.
                if height_class and on_right_shelf(state, policy):
                    for _ in range(12):
                        hold(session, 1, "A", reason=f"{label}_shelf_charge")
                    for _ in range(56):
                        state = hold(
                            session,
                            1,
                            "LEFT",
                            "B",
                            "A",
                            reason=f"{label}_shelf_hj",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if phase_d_top_band(state, policy):
                            track.top_reached = True
                            break
                    if track.top_reached or state.room_id != ROOM_ID:
                        break
                    continue

                # R11 mid reseat: only true ground nubs (not spin apex).
                if (
                    height_class
                    and y <= policy.MID_RESEAT_Y
                    and x < policy.RIGHT_SHELF_X
                    and not phase_c_sticky
                ):
                    for _ in range(2):
                        hold(session, 1, reason=f"{label}_reseat_settle")
                    for _ in range(6):
                        hold(session, 1, "A", reason=f"{label}_reseat_charge")
                    for _ in range(56):
                        state = hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason=f"{label}_reseat_hop",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if phase_d_top_band(state, policy):
                            track.top_reached = True
                            break
                        if state.samus_y <= policy.HEIGHT_CLASS_Y:
                            height_class = True
                    if track.top_reached or state.room_id != ROOM_ID:
                        break
                    continue

                # Phase-C sticky grounded: charge hop up-right along structure.
                if phase_c_sticky and height_class and x >= 250:
                    for _ in range(10):
                        hold(session, 1, "A", reason=f"{label}_pc_charge")
                    dir_h = "RIGHT" if x < 370 else "LEFT"
                    if x > policy.CAVITY_X_MAX - 15:
                        dir_h = "LEFT"
                    for _ in range(48):
                        state = hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            "A",
                            reason=f"{label}_pc_hj",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if phase_d_top_band(state, policy):
                            track.top_reached = True
                            break
                        if on_right_shelf(state, policy):
                            break
                    if track.top_reached or state.room_id != ROOM_ID:
                        break
                    continue

                # R13: after height class, deep floor runway → right climb.
                # Place: (288,531) charge12 + spin/WJ p8i2b2 hits Phase C
                # ~(302,428). Do not re-seat left lip (that is pre-height only).
                # Skip once Phase C sticky (stay on right structure).
                if (
                    height_class
                    and y >= policy.FLOOR_RECLIMB_Y
                    and not phase_c_sticky
                ):
                    r_lo, r_hi = policy.FLOOR_RUNWAY_X
                    if y >= policy.FLOOR_RUNWAY_Y:
                        if x < r_lo:
                            hold(
                                session,
                                1,
                                "RIGHT",
                                "B",
                                reason=f"{label}_floor_align_r",
                            )
                            continue
                        if x > r_hi:
                            hold(
                                session,
                                1,
                                "LEFT",
                                "B",
                                reason=f"{label}_floor_align_l",
                            )
                            continue
                    for _ in range(policy.FLOOR_RECLIMB_CHARGE):
                        hold(session, 1, "A", reason=f"{label}_floor_charge")
                    for _ in range(policy.FLOOR_RECLIMB_SPIN):
                        state = hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason=f"{label}_floor_hj",
                        )
                        track_state(session, track, state, policy)
                        if state.room_id != ROOM_ID:
                            break
                        if phase_d_top_band(state, policy):
                            track.top_reached = True
                            break
                        if on_right_shelf(state, policy):
                            break
                    if track.top_reached or state.room_id != ROOM_ID:
                        break
                    continue

                # Pre-height or mid (not deep): R6-style lip re-seat / left column.
                if y >= lip_y_lo and not (
                    lip_lo <= x <= lip_hi and y <= lip_y_hi
                ):
                    if y <= lip_y_hi + 10 and x > lip_hi:
                        hold(
                            session, 1, "LEFT", "B", reason=f"{label}_climb_relip"
                        )
                        continue
                for _ in range(10):
                    hold(session, 1, "A", reason=f"{label}_climb_charge")
                if height_class:
                    if x < 340:
                        dir_h = "RIGHT"
                    else:
                        dir_h = "LEFT"
                elif y <= 220:
                    dir_h = "RIGHT"
                elif y <= 280:
                    dir_h = "RIGHT" if x < 280 else "LEFT"
                else:
                    if x < 70:
                        dir_h = "RIGHT"
                    elif x > 130:
                        dir_h = "LEFT"
                    else:
                        dir_h = "RIGHT" if (mid_i // 40) % 2 == 0 else "LEFT"
                if x > policy.CAVITY_X_MAX - 15:
                    dir_h = "LEFT"
                for _ in range(44):
                    state = hold(
                        session,
                        1,
                        dir_h,
                        "B",
                        "A",
                        reason=f"{label}_climb_hj",
                    )
                    track_state(session, track, state, policy)
                    if state.room_id != ROOM_ID:
                        break
                    if state.samus_y <= policy.HEIGHT_CLASS_Y:
                        height_class = True
                    if phase_d_top_band(state, policy):
                        track.top_reached = True
                        break
                if track.top_reached or state.room_id != ROOM_ID:
                    break
                continue

            # Air after Phase C: sticky right-wall WJ (do not re-drop to floor).
            if phase_c_sticky and height_class:
                if x > policy.CAVITY_X_MAX - 15:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_pc_sc")
                    continue
                # Prefer wall contact ~x360-390; period-8 WJ like place shelf path.
                if x < 300:
                    dir_h = "RIGHT"
                elif x > 380:
                    dir_h = "LEFT"
                else:
                    dir_h = "RIGHT" if x < 365 else "LEFT"
                wj_phase = mid_i % policy.RIGHT_WJ_PERIOD
                if wj_phase < policy.RIGHT_WJ_INTO:
                    hold(
                        session,
                        1,
                        "RIGHT" if x < 375 else "LEFT",
                        "B",
                        reason=f"{label}_pc_into",
                    )
                elif wj_phase < (policy.RIGHT_WJ_INTO + policy.RIGHT_WJ_BOUNCE):
                    hold(session, 1, "LEFT", "A", reason=f"{label}_pc_wj")
                else:
                    hold(
                        session,
                        1,
                        dir_h,
                        "B",
                        "A",
                        reason=f"{label}_pc_spin",
                    )
                continue

            # Air (R10): after height class while still mid-high.
            if height_class and y <= policy.MIDHIGH_Y:
                if x > policy.CAVITY_X_MAX - 15:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_peak_sc")
                    continue
                if (
                    x >= policy.RIGHT_SHELF_X
                    and 280 <= y <= policy.MIDHIGH_Y
                    and state.velocity_y >= 0
                ):
                    dir_h = "RIGHT" if x < 365 else "LEFT"
                    hold(
                        session,
                        1,
                        dir_h,
                        "B",
                        reason=f"{label}_shelf_drop",
                    )
                    continue
                if x >= 250 and y > policy.TOP_Y:
                    wj_phase = mid_i % policy.RIGHT_WJ_PERIOD
                    if wj_phase < policy.RIGHT_WJ_INTO:
                        hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            reason=f"{label}_ol_into",
                        )
                    elif wj_phase < (policy.RIGHT_WJ_INTO + policy.RIGHT_WJ_BOUNCE):
                        hold(
                            session, 1, "LEFT", "A", reason=f"{label}_ol_wj"
                        )
                    else:
                        hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason=f"{label}_ol_spin",
                        )
                    continue
                hold(
                    session, 1, "RIGHT", "B", "A", reason=f"{label}_ol_cross"
                )
                continue

            # R13 deep air after height (pre-Phase-C only): floor runway approach.
            if height_class and y > policy.MIDHIGH_Y:
                if x > policy.CAVITY_X_MAX - 15:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_deep_sc")
                    continue
                r_lo, r_hi = policy.FLOOR_RUNWAY_X
                # Still airborne above floor: bias toward runway x, WJ for height.
                if y < policy.FLOOR_RUNWAY_Y - 20:
                    if x < r_lo - 20:
                        dir_h = "RIGHT"
                    elif x > r_hi + 30:
                        dir_h = "LEFT"
                    else:
                        dir_h = "RIGHT" if x < 320 else "LEFT"
                    wj_phase = mid_i % policy.RIGHT_WJ_PERIOD
                    if wj_phase < policy.RIGHT_WJ_INTO:
                        hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            reason=f"{label}_deep_into",
                        )
                    elif wj_phase < (policy.RIGHT_WJ_INTO + policy.RIGHT_WJ_BOUNCE):
                        opp = "LEFT" if dir_h == "RIGHT" else "RIGHT"
                        hold(
                            session, 1, opp, "A", reason=f"{label}_deep_wj"
                        )
                    else:
                        hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            "A",
                            reason=f"{label}_deep_spin",
                        )
                    continue
                # Near/on floor altitude: drop in, then grounded branch takes over.
                if x < r_lo:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_deep_drop_r")
                elif x > r_hi:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_deep_drop_l")
                else:
                    hold(session, 1, reason=f"{label}_deep_drop")
                continue

            # Pre-height air.
            if y <= 240:
                dir_h = "RIGHT" if x < 340 else "LEFT"
            elif x < 70:
                dir_h = "RIGHT"
            elif x > 150 and y > 300:
                dir_h = "LEFT"
            else:
                dir_h = "RIGHT" if x < 120 else "LEFT"
            if x > policy.CAVITY_X_MAX - 15:
                dir_h = "LEFT"
            air_phase = mid_i % 12
            if air_phase < 2:
                hold(session, 1, dir_h, "B", reason=f"{label}_climb_rel")
            elif air_phase < 4:
                opp = "LEFT" if dir_h == "RIGHT" else "RIGHT"
                hold(session, 1, opp, "A", reason=f"{label}_climb_wj")
            else:
                hold(
                    session, 1, dir_h, "B", "A", reason=f"{label}_climb_spin"
                )
            continue

        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_fallback")



# Historical name used by product hop re-exports / docs.
run_bubble_mid_loop = run_mid_loop
