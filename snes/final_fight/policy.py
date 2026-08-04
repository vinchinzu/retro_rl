"""Segment policy behavior tree for Final Fight Stage 1 clears."""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import (
    ActionNode,
    Condition,
    NodeStatus,
    Selector,
    Sequence,
    TickResult,
)
from retro_harness.combat import (
    AttackCadence,
    PreferredFlank,
    WalkProgress,
)
from retro_harness.ram_state import GameMode, GameState
from retro_harness.input_script import FrameAction

from final_fight.edge_combat import ff_fight_nearest_action


@dataclass
class _PatientWave:
    """Sticky flag: stay patient for the whole tough-thug wave."""

    active: bool = False

# Final Fight SNES: UP increases world Y (probe-confirmed).
_Y_TOLERANCE = 10
# Longer poke range + tighter cadence: alley thugs punish slow walk-ins.
# Alley probes: reliable connects around dx 28–35.
_ATTACK_RANGE = 34
_MIN_RANGE = 10
_STANDOFF = 18
_GRAB_RANGE = 18
_ATTACK_HOLD = 2
_ATTACK_GAP = 5
_LEFT_THREAT_X = 96
# Cam locks: playable band ~cam+40 … cam+170; park-hold uses right margin.
_CAMERA_LEFT_MARGIN = 48
_CAMERA_RIGHT_MARGIN = 160
# Modest left-edge early poke (large bonus whiffs at dx 60+).
_EDGE_ATTACK_BONUS = 20
# Toward+Y throw helps when latched; keep grab band tight so punches remain
# the primary clear tool. Right-edge policy forbids throw_right.
_USE_THROW = True
# Wave-4 alley thug starts at HP 80.
_TOUGH_ENEMY_HP = 60
# Door / Damnd: idle in dx 45–95 eats jump kicks; ground-nudge the near
# band and jump-dash the deep band into punch range (dx≈28–35).
_DOOR_PUNCH_LO = 28
_DOOR_PUNCH_HI = 35
_DOOR_NUDGE_HI = 39
_DOOR_KICK_HI = 103


def _needs_continue(state: GameState) -> bool:
    """True on continue screen, final death, or mid-life KO."""
    if state.mode is GameMode.CONTINUE or state.player_dead:
        return True
    if state.lives <= 0:
        return False
    # 0 HP or underflow corpse bytes while lives remain.
    if state.health == 0 or state.health > 128:
        return True
    if not bool(state.extras.get("player_active", True)):
        return True
    return False


def _continue_action(state: GameState) -> FrameAction:
    """START on continue / game-over; idle through mid-life KO respawn."""
    if state.mode is GameMode.CONTINUE or state.player_dead:
        return FrameAction(action=buttons("START"), reason="continue")
    # Death animation with lives left — do not mash START (can soft-lock).
    return FrameAction(action=idle_action(), reason="ko_wait")


# Face-Y while HP still connects; walk_past once ≤72 (stall band).
_SUBWAY_BEHIND_FACE_HP = 72
_SUBWAY_BEHIND_PLAYER_MIN = 40


def _subway_behind_action(
    *,
    adx: int,
    sx: int,
    enemy_hp: int,
    player_hp: int,
    cadence: AttackCadence,
    punch_hi: int,
    area1: bool = False,
) -> FrameAction | None:
    """Reposition when subway tough is behind (dx<0).

    Early face-dir Y chips HP148→~72; punches then stall at dx≈−37.
    Walk past / throw at HP≤72, or sooner when player HP is critical,
    to finish before Sid/J chips the clear. Area1 gutter: never
    walk_past (sx≈24 dies); face-Y / throw only.
    """
    # Critical player HP: abandon face-Y and grab/walk immediately.
    face_ok = (
        enemy_hp > _SUBWAY_BEHIND_FACE_HP
        and player_hp > _SUBWAY_BEHIND_PLAYER_MIN
        and _DOOR_PUNCH_LO <= adx <= punch_hi
    )
    if face_ok:
        return None
    if adx <= _GRAB_RANGE:
        punched = cadence.next_attack(button="Y")
        if punched.reason == "attack_gap":
            return punched
        return FrameAction(
            action=buttons("LEFT", "Y"), reason="throw_behind"
        )
    # Critical + stall: walk_past on area0 only. Area1 gutter walk_past
    # into the left wall dies (cam1792 dual-pack).
    if (
        not area1
        and player_hp <= _SUBWAY_BEHIND_PLAYER_MIN
        and enemy_hp <= _SUBWAY_BEHIND_FACE_HP
        and adx <= _DOOR_KICK_HI
    ):
        return FrameAction(action=buttons("LEFT"), reason="walk_past")
    # High HP behind: never walk_past (walks into Sid/J). Face-Y in
    # punch / near-punch; otherwise close left into punch band.
    # Area1+/area2: JD-left when far — ground LEFT sticks adx≈50
    # (e28 @cam3969). Widen face-Y to adx≤70 so JD does not overshoot
    # the punch band into kicks (e78 stall @cam3968).
    if enemy_hp > _SUBWAY_BEHIND_FACE_HP:
        face_adx = 70 if area1 else punch_hi + 14
        if sx < 55 or adx <= face_adx:
            punched = cadence.next_attack(button="Y")
            if punched.reason == "attack_gap":
                return punched
            return FrameAction(
                action=buttons("LEFT", "Y"), reason="attack"
            )
        if adx <= _DOOR_KICK_HI:
            if area1:
                return FrameAction(
                    action=buttons("B", "LEFT"),
                    reason="jump_dash",
                )
            return FrameAction(action=buttons("LEFT"), reason="nudge")
        return FrameAction(action=buttons("B", "LEFT"), reason="jump_dash")
    # Stall band (HP≤72): ground walk_past when there's room. Gutter
    # (sx<40) or area1: face-Y / throw — walk_past into the wall dies.
    # Close with JD-left — ground LEFT matches the thug's walk speed so
    # adx stays ~50 forever (e28 @cam3969); right-edge also sticks ~30f.
    if adx <= _DOOR_KICK_HI:
        if area1 or sx < 40:
            if adx > punch_hi:
                return FrameAction(
                    action=buttons("B", "LEFT"),
                    reason="jump_dash",
                )
            punched = cadence.next_attack(button="Y")
            if punched.reason == "attack_gap":
                return punched
            return FrameAction(
                action=buttons("LEFT", "Y"), reason="attack"
            )
        return FrameAction(action=buttons("LEFT"), reason="walk_past")
    return FrameAction(action=buttons("B", "LEFT"), reason="jump_dash")


def _is_subway_area1(state: GameState) -> bool:
    """True past the cam994 CLEAR_AREA bridge (subway room/area ≥1)."""
    return state.stage == 1 and state.room >= 1


def _is_subway_area2(state: GameState) -> bool:
    """True on subway Sodom stretch (round 01, room/area ≥2)."""
    return state.stage == 1 and state.room >= 2


def _subway_ultra_dual(state: GameState) -> bool:
    """Area1+ dual with an HP>100 thug (area2 HP112/134 pack)."""
    return (
        _is_subway_area1(state)
        and len(state.living_enemies) >= 2
        and any(e.health > 100 for e in state.living_enemies)
    )


def _living_all_far_behind(state: GameState, *, min_dx: int = -80) -> bool:
    """True when every living thug is far left (scroll leftovers)."""
    living = state.living_enemies
    if not living:
        return False
    return all((e.x - state.player_x) < min_dx for e in living)


def _door_thug_action(
    state: GameState,
    *,
    cadence: AttackCadence,
) -> FrameAction:
    """Kick-band safe close for Damnd door thugs.

    Never idle at dx 40–103. Near band (36–39) ground-nudges. Tough
    (HP>50): jump-dash 40–103 into punch dx≈28–35. Peak≤50: park-bait /
    retreat — hop_in and idle@dx≈53 chip or one-shot. Subway area0 keeps
    always-JD (Sid/J); area1 dual-pack uses door park-bait (always-JD
    suicide at cam1950). Area2 ultra dual (HP112/134): tighter space +
    lighter JD pulse; finish lone leftovers up to HP80.
    """
    enemy = state.nearest_enemy()
    if enemy is None:
        return FrameAction(action=idle_action(), reason="no_enemy")
    area1 = _is_subway_area1(state)
    area2 = _is_subway_area2(state)
    ultra_dual = _subway_ultra_dual(state)
    # Subway dual living: focus HP≈148. Prefer a thug already in the
    # kick/punch band over a far tough (area1 idle-on-far-tough lets the
    # closer pack chip hold_left to death). Cam994: front weak before
    # far-behind tough. Area2 ultra: prefer weaker in-band to reach 1v1.
    # Area2 non-ultra: prefer behind leftover (face-Y) over front JD.
    # Area2 HP≤40 leftovers: always finish the crumb first (underflow kill).
    if state.stage >= 1 and len(state.living_enemies) >= 2:
        crumbs = tuple(
            e for e in state.living_enemies if e.health <= 40
        )
        tough_e = max(state.living_enemies, key=lambda e: e.health)
        tdx = tough_e.x - state.player_x
        in_band = tuple(
            e
            for e in state.living_enemies
            if abs(e.x - state.player_x) <= _DOOR_KICK_HI
        )
        behind = tuple(
            e
            for e in state.living_enemies
            if (e.x - state.player_x) < 0
            and abs(e.x - state.player_x) <= _DOOR_KICK_HI
        )
        if area2 and crumbs:
            enemy = min(
                crumbs, key=lambda e: abs(e.x - state.player_x)
            )
        elif area2 and behind and not ultra_dual:
            enemy = min(behind, key=lambda e: e.health)
        elif in_band and (
            area1 or abs(tdx) > _DOOR_KICK_HI
        ):
            if ultra_dual:
                enemy = min(
                    in_band,
                    key=lambda e: (e.health, abs(e.x - state.player_x)),
                )
            else:
                enemy = min(
                    in_band, key=lambda e: abs(e.x - state.player_x)
                )
        elif (
            state.camera_x >= 990
            and not area1
            and tdx < 0
            and abs(tdx) > _DOOR_KICK_HI
        ):
            front = tuple(
                e
                for e in state.living_enemies
                if (e.x - state.player_x) > 0
                and abs(e.x - state.player_x) <= _DOOR_KICK_HI
            )
            enemy = (
                min(front, key=lambda e: abs(e.x - state.player_x))
                if front
                else tough_e
            )
        else:
            enemy = tough_e
    dx = enemy.x - state.player_x
    adx = abs(dx)
    dy = enemy.y - state.player_y
    sx = state.player_x - state.camera_x
    tough = enemy.health > 50
    # Subway: slightly wider punch/nudge so dx≈38 punches instead of
    # walking into kicks; kick-band JD only above nudge (avoids the
    # old JD↔retreat oscillation that never dealt damage).
    punch_hi = 38 if state.stage >= 1 else _DOOR_PUNCH_HI
    nudge_hi = 45 if state.stage >= 1 else _DOOR_NUDGE_HI
    area2_1v1 = (
        area2
        and len(state.living_enemies) == 1
        and enemy.health <= 80
    )
    # HP≤40 front leftovers / HP≤8 any: JD-scroll + toward+Y kill window.
    # Behind mid-HP still closes via normal face-Y path (adx52 Y whiffs).
    area2_crumb = area2 and (
        enemy.health <= 8
        or (enemy.health <= 40 and dx >= 0)
    )
    crumb_punch_hi = 70 if area2_crumb else punch_hi
    # Area2: face-Y from behind chips ~7/hit. Park only when far in
    # front AND sx very high — sx≈100 park/nudge never dealt damage.
    # Never park below sx≈55 (1v1f gutter death). Crumbs chase instead.
    if (
        area2
        and dx > 0
        and adx > _DOOR_KICK_HI
        and sx > 125
        and not area2_crumb
    ):
        return FrameAction(action=buttons("LEFT"), reason="park")
    # Area2 high-sx / low-HP behind: face-Y only in punch band.
    # Outside punch, fall through so stall-band can nudge close
    # (LEFT+Y at adx≈52–100 whiffed on e28 @cam3969).
    if (
        area2
        and dx < 0
        and _DOOR_PUNCH_LO <= adx <= punch_hi
        and (sx > 150 or enemy.health <= 50)
    ):
        punched = cadence.next_attack(button="Y")
        if punched.reason == "attack_gap":
            return punched
        return FrameAction(
            action=buttons("LEFT", "Y"), reason="attack"
        )
    # Area2 ultra dual: survive HP134 — prefer sx>125; only refuse
    # chips when the tough is inside ~55 (old kick-band gate never
    # attacked and L1 bled out to space-spam). Crumb focus: allow
    # toward+Y when sx>100 and tough not overlapping.
    if ultra_dual and area2:
        tough_e = max(state.living_enemies, key=lambda e: e.health)
        tdx = tough_e.x - state.player_x
        tadx = abs(tdx)
        if sx < 55:
            return FrameAction(
                action=buttons("RIGHT"), reason="nudge"
            )
        if area2_crumb and sx > 100 and (
            tadx >= 30 or abs(dx) <= 60
        ):
            pass  # finish crumb even if tough mid-range
        elif sx <= 100:
            away_t = "LEFT" if tdx > 0 else "RIGHT"
            if away_t == "LEFT" and sx < 70:
                away_t = "RIGHT"
            return FrameAction(
                action=buttons("B", away_t), reason="space"
            )
        elif tadx < 55:
            away = "LEFT" if tdx > 0 else "RIGHT"
            if away == "LEFT" and sx < 70:
                away = "RIGHT"
            return FrameAction(
                action=buttons("B", away), reason="space"
            )
    # Ultra dual: space any tight overlap (dx≈4 kick one-shots L1).
    if ultra_dual:
        for other in state.living_enemies:
            odx = other.x - state.player_x
            if abs(odx) < 26 and abs(other.y - state.player_y) < 16:
                away = "LEFT" if odx > 0 else "RIGHT"
                if area2 and away == "LEFT" and sx < 70:
                    away = "RIGHT"
                return FrameAction(
                    action=buttons("B", away), reason="space"
                )
    # Weak overlapping while chasing HP148 — hop away first.
    if state.stage >= 1 and len(state.living_enemies) >= 2:
        for other in state.living_enemies:
            if other.slot == enemy.slot:
                continue
            odx = other.x - state.player_x
            if abs(odx) < 32 and abs(other.y - state.player_y) < 14:
                away = "LEFT" if odx > 0 else "RIGHT"
                if area2 and away == "LEFT" and sx < 70:
                    away = "RIGHT"
                return FrameAction(
                    action=buttons("B", away), reason="space"
                )
    # Area1: door-tight align (≤48). Subway area0 keeps ≤80 so JD can
    # land after rise; area1 align@dx80 stalls in Sid/J kick forever.
    # Area2: dy-only (absolute py<70 rose past ey=59 then DOWN-oscillated).
    align_dx = 48 if (area1 or state.stage < 1) else 80
    align_dy = 6 if area2 else 10
    if abs(dy) > align_dy and adx <= align_dx:
        return FrameAction(
            action=buttons("UP") if dy > 0 else buttons("DOWN"),
            reason="align",
        )
    # Area2 HP≤40 leftover: JD-scroll into cam≈3968 then grounded toward+Y
    # (HP underflow at dx≈56; jd90_faceY). After cam3960, do not JD into
    # HP134 — space if tough overlaps, else toward+Y / ground-close.
    if area2_crumb:
        toward = "RIGHT" if dx > 0 else "LEFT"
        if ultra_dual:
            tough_e = max(
                state.living_enemies, key=lambda e: e.health
            )
            tadx = abs(tough_e.x - state.player_x)
            # Kill window at right-edge cam3968: dual can sit ~dx90.
            # Only hop if overlapping (scripted kill had tadx≈91).
            if tadx < 28 and sx > 100:
                away = (
                    "LEFT"
                    if tough_e.x > state.player_x
                    else "RIGHT"
                )
                if away == "LEFT" and sx < 70:
                    away = "RIGHT"
                return FrameAction(
                    action=buttons("B", away), reason="space"
                )
        if (
            dx > 0
            and state.camera_x < 3960
            and sx < 170
        ):
            if sx < 55:
                return FrameAction(
                    action=buttons("RIGHT"), reason="nudge"
                )
            return FrameAction(
                action=buttons("B", "RIGHT"), reason="jump_dash"
            )
        # Post-scroll front kill window: spam toward+Y (dx≈56 connect).
        # Behind leftovers still close first (adx52 Y whiffs).
        if (
            state.camera_x >= 3960
            and sx >= 125
            and adx <= 70
            and dx >= 0
        ):
            if state.frame % 4 < 2:
                return FrameAction(
                    action=buttons(toward, "Y"), reason="attack"
                )
            return FrameAction(
                action=idle_action(), reason="attack_gap"
            )
        if adx > crumb_punch_hi:
            if sx < 55:
                return FrameAction(
                    action=buttons(toward), reason="nudge"
                )
            if state.camera_x >= 3960:
                return FrameAction(
                    action=buttons(toward), reason="nudge"
                )
            return FrameAction(
                action=buttons("B", toward), reason="jump_dash"
            )
        if adx <= _GRAB_RANGE:
            punched = cadence.next_attack(button="Y")
            if punched.reason == "attack_gap":
                return punched
            return FrameAction(
                action=buttons(toward, "Y"), reason="throw"
            )
        if state.frame % 4 < 2:
            return FrameAction(
                action=buttons(toward, "Y"), reason="attack"
            )
        return FrameAction(action=idle_action(), reason="attack_gap")
    # Subway behind kick-band: always walk_past/throw — do NOT gate on
    # tough (HP>50). After face-Y chips 148→48 the old gate flipped to
    # JD-left and Sid/J burned the clear (HP46→death, L2→L1).
    # Area2: never walk_past (sx≈24 gutter dies); face-Y / throw only.
    # Area2 sx>200: walk left first (right-edge trap @cam3968 px≈4200).
    # sx≈188 still JD/face-Y closes (high_sx behind tests).
    if state.stage >= 1 and dx < 0 and adx <= _DOOR_KICK_HI:
        if area2 and sx > 200:
            return FrameAction(
                action=buttons("LEFT"), reason="scroll_edge"
            )
        if area2 and sx < 55:
            return FrameAction(action=buttons("RIGHT"), reason="nudge")
        behind = _subway_behind_action(
            adx=adx,
            sx=sx,
            enemy_hp=enemy.health,
            player_hp=state.health,
            cadence=cadence,
            punch_hi=punch_hi,
            area1=area1 or area2,
        )
        if behind is not None:
            return behind
    # Area2 in front: JD-pass into behind (face-Y). Front punches whiff
    # on mid-HP thugs. Crumbs handled above. Cam≥3915: never JD-scroll
    # into HP134 — ground-close / punch instead (old park-in-kick-band
    # let e69 walk in and kill while we held LEFT).
    if (
        area2
        and dx > 0
        and 16 <= adx <= _DOOR_KICK_HI
        and (area2_1v1 or punch_hi < adx)
    ):
        if state.camera_x >= 3915 and area2_1v1:
            if adx <= punch_hi:
                pass  # fall through to face-Y / throw
            else:
                return FrameAction(
                    action=buttons("RIGHT"), reason="nudge"
                )
        elif sx < 55:
            return FrameAction(
                action=buttons("RIGHT"), reason="nudge"
            )
        else:
            return FrameAction(
                action=buttons("B", "RIGHT"), reason="jump_dash"
            )
    # Area2 grab: throw ASAP (toward+Y).
    if area2 and adx <= _GRAB_RANGE:
        toward = "RIGHT" if dx > 0 else "LEFT"
        punched = cadence.next_attack(button="Y")
        if punched.reason == "attack_gap":
            return punched
        return FrameAction(
            action=buttons(toward, "Y"), reason="throw"
        )
    if _DOOR_PUNCH_LO <= adx <= punch_hi:
        # Face the target — bare Y whiffs when HP148 is behind (dx<0).
        punched = cadence.next_attack(button="Y")
        if punched.reason == "attack_gap":
            return punched
        toward = "RIGHT" if dx > 0 else "LEFT"
        return FrameAction(
            action=buttons(toward, "Y"), reason="attack"
        )
    if adx < _DOOR_PUNCH_LO:
        if adx >= 16:
            punched = cadence.next_attack(button="Y")
            if punched.reason == "attack_gap":
                return punched
            toward = "RIGHT" if dx > 0 else "LEFT"
            return FrameAction(
                action=buttons(toward, "Y"), reason="attack"
            )
        away = "LEFT" if dx > 0 else "RIGHT"
        if area2 and away == "LEFT" and sx < 70:
            away = "RIGHT"
        return FrameAction(action=buttons(away), reason="space")
    if adx <= nudge_hi:
        toward = "RIGHT" if dx > 0 else "LEFT"
        return FrameAction(action=buttons(toward), reason="nudge")
    if adx <= _DOOR_KICK_HI:
        # Area1+ lone leftover (dual→1v1 ≤80): finish in place. Area2
        # parks only outside nudge; kick band: aggressive walk-in
        # (14/24) so e79 dies before cam≈3928 (old 14/50 never punched).
        if (
            area1
            and enemy.health <= 80
            and len(state.living_enemies) == 1
        ):
            toward = "RIGHT" if dx > 0 else "LEFT"
            if (
                area2
                and sx > 125
                and dx > 0
                and adx > nudge_hi
                and not area2_crumb
            ):
                return FrameAction(
                    action=buttons("LEFT"), reason="park"
                )
            if area2:
                if sx < 55:
                    return FrameAction(
                        action=buttons(toward), reason="nudge"
                    )
                # Prefer JD-pass to behind over retreat stall.
                # Crumbs: ground-close only (JD scrolls dual @3928).
                if dx > 0 and not area2_crumb:
                    return FrameAction(
                        action=buttons("B", "RIGHT"),
                        reason="jump_dash",
                    )
                if dx > 0 and area2_crumb:
                    return FrameAction(
                        action=buttons(toward), reason="nudge"
                    )
                if state.frame % 24 < 14:
                    return FrameAction(
                        action=buttons(toward), reason="nudge"
                    )
                if sx > 70:
                    return FrameAction(
                        action=buttons("LEFT"), reason="retreat"
                    )
                return FrameAction(
                    action=buttons("RIGHT"), reason="nudge"
                )
            return FrameAction(
                action=buttons("B", toward), reason="jump_dash"
            )
        if area1:
            # Mid-screen hit-and-run: pulse JD into punch, then retreat,
            # but never park below sx≈55 (gutter space-spam dies).
            # Ultra dual: lighter JD duty (14/40 suicided vs HP112/134).
            toward = "RIGHT" if dx > 0 else "LEFT"
            if sx < 55:
                return FrameAction(
                    action=buttons("B", toward), reason="jump_dash"
                )
            pulse = 8 if ultra_dual else 14
            period = 48 if ultra_dual else 40
            if state.frame % period < pulse:
                return FrameAction(
                    action=buttons("B", toward), reason="jump_dash"
                )
            if area2 and sx <= 70:
                return FrameAction(
                    action=buttons("RIGHT"), reason="nudge"
                )
            return FrameAction(
                action=buttons("LEFT"), reason="retreat"
            )
        if not tough:
            # Area0 Sid/J: always JD (retreat still eats kicks).
            if state.stage >= 1:
                if dx > 0:
                    return FrameAction(
                        action=buttons("B", "RIGHT"),
                        reason="jump_dash",
                    )
                return FrameAction(
                    action=buttons("B", "LEFT"),
                    reason="jump_dash",
                )
            if sx > 55:
                return FrameAction(
                    action=buttons("LEFT"), reason="retreat"
                )
            return FrameAction(
                action=idle_action(), reason="hold_left"
            )
        if dx > 0:
            return FrameAction(
                action=buttons("B", "RIGHT"), reason="jump_dash"
            )
        return FrameAction(
            action=buttons("B", "LEFT"), reason="jump_dash"
        )
    # Beyond kick band: door park-bait. Area0+ghost / dual / cam≥900
    # must keep closing (train stretch). Area1 uses park-bait — cam is
    # always ≥1792 so the old cam≥900 gate forced suicide JD forever.
    # West Side (stage≥2): engage often starts dx≈109 (>kick) — park-bait
    # never closes; JD in like subway dual/ghost.
    ghost_near = any(
        e.health == 0 and abs(e.x - state.player_x) < 120
        for e in state.threat_enemies
    )
    if state.stage >= 1 and not area1 and (
        ghost_near
        or len(state.living_enemies) >= 2
        or state.camera_x >= 900
        or state.stage >= 2
    ):
        toward = "RIGHT" if dx > 0 else "LEFT"
        return FrameAction(
            action=buttons("B", toward), reason="jump_dash"
        )
    # Area1+: finish a lone leftover. Dual packs still park-bait.
    # Area2: walk-close beyond kick (JD scrolls / gutters); brief nudge
    # pulses in kick band so far_bait alone never connects.
    if (
        area1
        and enemy.health <= 80
        and len(state.living_enemies) == 1
    ):
        toward = "RIGHT" if dx > 0 else "LEFT"
        if area2:
            if sx > 100 and dx > 0:
                return FrameAction(
                    action=buttons("LEFT"), reason="park"
                )
            return FrameAction(
                action=buttons(toward), reason="nudge"
            )
        return FrameAction(
            action=buttons("B", toward), reason="jump_dash"
        )
    # Area1: keep mid-screen bait (sx≈55–75); deep left park traps.
    park_sx = 75 if area1 else (60 if not tough else 65)
    if sx > park_sx:
        return FrameAction(action=buttons("LEFT"), reason="park")
    return FrameAction(action=idle_action(), reason="far_bait")


def _ghost_punch_action(state: GameState) -> FrameAction | None:
    """Cadenced plant-punch for damaging HP0 / UF status-03 chasers.

    Standing Y pulses (hold 3 / gap 5) despawn door/subway post-kill
    hurtboxes without chasing into them. Adapter normalizes underflow
    HP to 0 on ``EnemyState``. Returns None when no ghost is near.
    """
    ghosts = tuple(
        e for e in state.threat_enemies if e.health == 0
    )
    if not ghosts:
        return None
    ghost = min(
        ghosts,
        key=lambda e: abs(e.x - state.player_x) + abs(e.y - state.player_y),
    )
    dx = ghost.x - state.player_x
    adx = abs(dx)
    if adx > 160:
        return None
    # Door: flee dx<36. Subway: when a living thug remains, only flee the
    # tight overlap (dx<40) so we still JD the HP148; ghost-only uses 50.
    flee_dx = 40
    if state.stage >= 1 and not state.living_enemies:
        flee_dx = 50
    if adx < flee_dx:
        away = "LEFT" if dx > 0 else "RIGHT"
        return FrameAction(
            action=buttons("B", away), reason="z_flee"
        )
    # Cam≥840: ignore distant HP0 (scroll softlock at cam848), but
    # plant-punch kick-band ghosts — leaving UF behind the lock wall
    # at cam994 chips with no living parse and never sets GO.
    # Area2 open leftovers sit at dx≈105–145; still plant when unlocked
    # with no living thug (else they walk in and chip the next pack).
    plant_hi = _DOOR_KICK_HI
    if (
        _is_subway_area2(state)
        and not state.living_enemies
        and not state.screen_locked
    ):
        plant_hi = 160
    if (
        state.stage >= 1
        and state.camera_x >= 840
        and adx > plant_hi
    ):
        return None
    # Frame-cadence plant punch (matches door_jump_clear probe).
    # Face the corpse so subway behind-UF connects.
    if state.frame % 8 < 3:
        if dx < 0:
            return FrameAction(
                action=buttons("LEFT", "Y"), reason="z_punch"
            )
        if dx > 0:
            return FrameAction(
                action=buttons("RIGHT", "Y"), reason="z_punch"
            )
        return FrameAction(action=buttons("Y"), reason="z_punch")
    return FrameAction(action=idle_action(), reason="z_gap")


def _boss_stub(
    state: GameState,
    *,
    cadence: AttackCadence,
) -> FrameAction:
    """Door / Damnd / subway: kick-band close, punch, plant-punch ghosts.

    ``Boss.state`` lights ``0x11E0=01`` before Damnd is drawn. Door thugs
    (peaks ~36/60/95) spawn in regular slots first; idle inside kick dx
    chips hard. Jump-dash the deep band; ground-nudge 36–54. After kills,
    plant-punch HP0/UF chasers before parking for the next spawn. Subway
    dual packs leave a UF ghost beside a living thug — flee the corpse
    first or it chips through the living fight.
    """
    # Overlapping corpse outranks living targets (subway cam844 dual pack).
    close_ghost = _ghost_punch_action(state)
    if close_ghost is not None and close_ghost.reason == "z_flee":
        return close_ghost
    # Ghost-only kick-band plant before cam≥840 / cam994 scroll mash —
    # otherwise UF leftovers sit while RIGHT+Y chips at the lock wall.
    if (
        close_ghost is not None
        and state.stage >= 1
        and state.camera_x >= 840
        and not state.living_enemies
    ):
        return close_ghost
    # Train escape: cam994 unlock is brief — mash right before the
    # re-lock pack pins the left gutter (walk_past at sx≈24 dies).
    # Area-0 cam994 still needs CLEAR_AREA bridge (stage2_advance) when
    # scroll will not advance; mash keeps HP until the poke. Area1 must
    # plant-punch ghosts first — unlocked mash at cam2048 chipped 54→2
    # with living=0.
    if (
        state.stage >= 1
        and state.room == 0
        and state.camera_x >= 990
        and not state.screen_locked
    ):
        sx = state.player_x - state.camera_x
        if sx > 170:
            return FrameAction(
                action=buttons("LEFT"), reason="scroll_edge"
            )
        if state.player_y < 70:
            return FrameAction(
                action=buttons("UP", "RIGHT"), reason="scroll_rise"
            )
        return FrameAction(
            action=buttons("RIGHT", "Y"), reason="scroll_mash"
        )
    # Unlocked scroll — or softlock approach with only far-behind
    # leftovers (cam2488 HP134 behind; area2 cam4130 e80 @dx≈−174).
    # Area1+ uses plain RIGHT; RIGHT+Y chips 54→38 at cam≈2523 with no
    # living parse. Area2 locked: never hijack near behind combat into
    # scroll_edge (e28 @cam3969), but do scroll past far-behind.
    far_behind = _living_all_far_behind(state, min_dx=-100)
    softlock_behind = (
        state.room >= 1 and far_behind and state.screen_locked
    )
    if (
        state.stage >= 1
        and state.camera_x >= 840
        and (
            not state.living_enemies
            or far_behind
        )
        and (
            not state.screen_locked
            or softlock_behind
        )
    ):
        sx = state.player_x - state.camera_x
        if sx > 170:
            return FrameAction(
                action=buttons("LEFT"), reason="scroll_edge"
            )
        if state.player_y < 70:
            return FrameAction(
                action=buttons("UP", "RIGHT"), reason="scroll_rise"
            )
        if state.room >= 1:
            return FrameAction(
                action=buttons("RIGHT"), reason="scroll_mash"
            )
        return FrameAction(
            action=buttons("RIGHT", "Y"), reason="scroll_mash"
        )
    if state.living_enemies:
        return _door_thug_action(state, cadence=cadence)
    if close_ghost is not None:
        return close_ghost
    # Cam≥840 ghost-only / clear: mash right — do not idle on distant
    # HP0. Rise out of the low Sid/J kick lane first. Stay in the
    # playable sx band — overshoot sx≈232 at cam2048 drained HP with
    # no living parse (pit / off-edge). Area1+: plain RIGHT (Y chips).
    if state.stage >= 1 and state.camera_x >= 840:
        sx = state.player_x - state.camera_x
        if sx > 170:
            return FrameAction(
                action=buttons("LEFT"), reason="scroll_edge"
            )
        if state.player_y < 70:
            return FrameAction(
                action=buttons("UP", "RIGHT"), reason="scroll_rise"
            )
        if state.room >= 1:
            return FrameAction(
                action=buttons("RIGHT"), reason="scroll_mash"
            )
        return FrameAction(
            action=buttons("RIGHT", "Y"), reason="scroll_mash"
        )
    if state.boss_active:
        boss_status = int(state.extras.get("boss_status", 0))
        boss_hp = int(state.extras.get("boss_hp", 0))
        boss_x = int(state.extras.get("boss_x", state.player_x))
        boss_y = int(state.extras.get("boss_y", state.player_y))
        sx = state.player_x - state.camera_x
        # Undrawn (01, hp 0): after door pack clears, creep right until
        # cam≈2675 draws Damnd. Rise out of the low knife/corpse lane.
        if boss_status < 0x03 and boss_hp <= 0:
            if state.player_y < 70:
                return FrameAction(
                    action=buttons("UP"), reason="boss_rise"
                )
            if sx < 150:
                return FrameAction(
                    action=buttons("RIGHT"), reason="boss_creep"
                )
            return FrameAction(
                action=idle_action(), reason="boss_hold_door"
            )
        # Drawn Damnd (HP≈44): spam-Y at dx 24–40; JD the kick band.
        dx = boss_x - state.player_x
        adx = abs(dx)
        dy = boss_y - state.player_y
        if abs(dy) > 10 and adx <= 48:
            return FrameAction(
                action=buttons("UP") if dy > 0 else buttons("DOWN"),
                reason="b_align",
            )
        if 24 <= adx <= 40:
            return FrameAction(action=buttons("Y"), reason="b_yp")
        if adx < 24:
            away = "LEFT" if dx > 0 else "RIGHT"
            return FrameAction(action=buttons(away), reason="b_space")
        if adx <= _DOOR_KICK_HI:
            if dx > 0:
                return FrameAction(
                    action=buttons("B", "RIGHT"), reason="b_jump_dash"
                )
            return FrameAction(
                action=buttons("B", "LEFT"), reason="b_jump_dash"
            )
        toward = "RIGHT" if dx > 0 else "LEFT"
        return FrameAction(action=buttons(toward), reason="b_close")
    return FrameAction(action=idle_action(), reason="boss_wait")


def _is_tough_thug(state: GameState) -> bool:
    """True for the wave-4 alley thug (starts at HP 80)."""
    enemy = state.nearest_enemy()
    return enemy is not None and enemy.health >= _TOUGH_ENEMY_HP


def _fight_action(
    state: GameState,
    *,
    cadence: AttackCadence,
) -> FrameAction:
    """Normal alley fight, or spaced mode for the HP-80 thug.

    Throws are disabled in the alley: throw_gap trades burned waves 2–3.
    Punch-only at dx≈28–35 once aligned. Post-unlock uses softer cadence
    in Stage1Policy.tick (gap=3); patient_approach stays tough-only.
    """
    use_patient = _is_tough_thug(state)
    return ff_fight_nearest_action(
        state,
        y_tolerance=_Y_TOLERANCE,
        attack_range=_ATTACK_RANGE,
        min_range=_MIN_RANGE + (6 if use_patient else 0),
        invert_vertical=True,
        cadence=cadence,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=_STANDOFF + (6 if use_patient else 0),
        use_throw=False,
        grab_range=_GRAB_RANGE + (10 if use_patient else 0),
        prefer_left_threat=True,
        left_threat_x=_LEFT_THREAT_X,
        camera_left_margin=_CAMERA_LEFT_MARGIN,
        camera_right_margin=_CAMERA_RIGHT_MARGIN,
        edge_attack_bonus=_EDGE_ATTACK_BONUS,
        patient_approach=use_patient,
    )


def build_stage1_tree(
    *,
    cadence: AttackCadence | None = None,
    walk_progress: WalkProgress | None = None,
) -> Selector:
    """Segment policy: continue → clear → boss stub → fight → walk right."""
    cadence = cadence or AttackCadence(
        hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP
    )
    walk_progress = walk_progress or WalkProgress(pickup_every=18)

    def boss_action(state: GameState) -> FrameAction:
        return _boss_stub(state, cadence=cadence)

    def fight_action(state: GameState) -> FrameAction:
        return _fight_action(state, cadence=cadence)

    def walk_action(state: GameState) -> FrameAction:
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
                    Condition(lambda s: s.boss_active, name="boss_active"),
                    ActionNode(boss_action, name="boss_segment"),
                ],
                name="boss_seq",
            ),
            # Subway (round 01): same kick-band rules as door thugs —
            # idle at dx≈78 one-shots (Sid/J). Park-bait / JD, not alley
            # edge_wait. Also plant-punch UF/HP0 ghosts before walk.
            Sequence(
                [
                    Condition(
                        lambda s: s.stage >= 1 and bool(s.threat_enemies),
                        name="subway_threats",
                    ),
                    ActionNode(boss_action, name="subway_kickband"),
                ],
                name="subway_seq",
            ),
            # Cam≥840 clear/unlock: mash right — WalkProgress stalls
            # softlock the train approach after cam994 pack clear.
            Sequence(
                [
                    Condition(
                        lambda s: s.stage >= 1 and s.camera_x >= 840,
                        name="subway_scroll",
                    ),
                    ActionNode(boss_action, name="subway_scroll_mash"),
                ],
                name="subway_scroll_seq",
            ),
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
            # Door/slum post-kill HP0 ghosts when boss_active is false.
            Sequence(
                [
                    Condition(
                        lambda s: bool(s.threat_enemies)
                        and not s.living_enemies,
                        name="ghost_only",
                    ),
                    ActionNode(boss_action, name="ghost_plant"),
                ],
                name="ghost_seq",
            ),
            ActionNode(walk_action, name="walk_right"),
        ],
        name="segment_clear",
    )


class Stage1Policy:
    """Stateful wrapper around the Stage 1 segment behavior tree."""

    def __init__(self) -> None:
        self._cadence = AttackCadence(
            hold_frames=_ATTACK_HOLD, gap_frames=_ATTACK_GAP
        )
        self._walk = WalkProgress(pickup_every=18)
        self._patient = _PatientWave()
        self._tree = build_stage1_tree(
            cadence=self._cadence,
            walk_progress=self._walk,
        )

    def reset(self) -> None:
        """Reset cadence / walk stall and rebuild the tree."""
        self._cadence.reset()
        self._walk.reset()
        self._patient.active = False
        self._tree = build_stage1_tree(
            cadence=self._cadence,
            walk_progress=self._walk,
        )

    def tick(self, state: GameState) -> TickResult:
        """Choose one frame of action for the current state."""
        if any(
            e.health >= _TOUGH_ENEMY_HP for e in state.living_enemies
        ):
            self._patient.active = True
        if not state.living_enemies:
            self._patient.active = False
        if state.camera_x >= 1600:
            # Door / post-unlock: spaced punches (scripted door used ~gap 6).
            self._cadence.hold_frames = 2
            self._cadence.gap_frames = 6
        elif self._patient.active:
            self._cadence.hold_frames = 2
            self._cadence.gap_frames = 3
        else:
            self._cadence.hold_frames = _ATTACK_HOLD
            self._cadence.gap_frames = _ATTACK_GAP
        result = self._tree.tick(state)
        if result.action is None and result.status is NodeStatus.SUCCESS:
            return TickResult(
                status=NodeStatus.SUCCESS,
                action=FrameAction(
                    action=idle_action(), reason="segment_done"
                ),
                reason=result.reason,
            )
        if result.action is None:
            return TickResult(
                status=result.status,
                action=FrameAction(
                    action=idle_action(), reason="policy_idle"
                ),
                reason=result.reason,
            )
        return result
