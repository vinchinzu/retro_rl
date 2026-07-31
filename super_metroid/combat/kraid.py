"""Full-knowledge Kraid strategy (RAM positions + Super Missile spray).

Deterministic controller only — no vision and no RL. Expects the session
already inside Kraid's Room (``0xA59F``) from a real doorway entry (or a
doorway-natural save). Primary weapon: Super Missiles.

Recipe (proven on doorway entry + mid-arena saves):
  1. Select Supers.
  2. Hold a left-mid lane (avoid walking into the body).
  3. Face right and pulse Super fire + periodic jumps.
  4. Win when body HP hits 0; wait for Brinstar boss bit 0.
  5. Push the rear (right) blue door into Varia Suit Room.
  6. Shoot the Chozo, touch the real Varia PLM, wait fanfare.

Body HP and multi-slot layout come from live probes: enemy0 is the body
(HP 1000); nails/projectiles occupy other slots.

Measured doorway-entry closeout (``eye_hj_kraid_entry``, unlimited assist):
body zero ~1321f, boss bit ~1520f, Varia room ~1635f, Varia bit ~1975f,
zero energy restored. Not continuous evidence until composed on the power-on
KPDR prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from super_metroid.combat.features import kraid_catalog
from super_metroid.combat.primitives import (
    ensure_weapon,
    lane_hold_action,
    settle_standing,
    spray_action,
)
from super_metroid.ram import GameplayPhase, SuperMetroidState, read_bank7e_wram
from super_metroid.routes.controller_common import select_weapon, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_KRAID = 0xA59F
ROOM_VARIA = 0xA6E2
VARIA_MASK = 0x0001
# Brinstar boss bits live at $7E:D829; bit 0 = Kraid defeated.
ADDR_BRINSTAR_BOSS_BITS = 0xD829
KRAID_BOSS_BIT = 0x01
# Selected item index for Super Missiles / beams.
WEAPON_SUPERS = 2
WEAPON_BEAM = 0


@dataclass(frozen=True)
class KraidStrategy:
    """Tunable left-lane Super spray parameters."""

    min_x: int = 50
    max_x: int = 260
    jump_hold_frames: int = 10
    jump_period: int = 50
    fire_hold_frames: int = 6
    fire_period: int = 12
    max_fight_frames: int = 15_000
    # After body HP 0, wait this many frames for the boss bit (death anim).
    boss_bit_grace_frames: int = 1_200


@dataclass(frozen=True)
class KraidEvidence:
    start_frame: int
    body_zero_frame: int | None
    boss_bit_frame: int | None
    end_frame: int
    peak_body_hp: int
    min_body_hp: int
    action_frames: int
    final_body_hp: int
    boss_bit_set: bool
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "body_zero_frame": self.body_zero_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp,
            "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames,
            "final_body_hp": self.final_body_hp,
            "boss_bit_set": self.boss_bit_set,
            "outcome": self.outcome,
        }


def brinstar_boss_bits(env: Any) -> int:
    """Read Brinstar boss flags ($7E:D829) via bank-$7E WRAM."""
    return int(read_bank7e_wram(env)[ADDR_BRINSTAR_BOSS_BITS])


def kraid_defeated(env: Any) -> bool:
    """True when Brinstar boss bit 0 is set (Kraid)."""
    return bool(brinstar_boss_bits(env) & KRAID_BOSS_BIT)


def body_hp(state: SuperMetroidState) -> int:
    """Kraid body HP from enemy0 (live probes pin body to slot 0)."""
    return int(state.enemy0_hp)


def fight_kraid_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: KraidStrategy = KraidStrategy(),
    *,
    body_dead: bool = False,
) -> tuple[str, ...]:
    """One-frame button names for the Super-spray Kraid policy.

    ``body_dead`` is True after body HP has hit 0 (death animation / exit
    prep). Returns button name tuples suitable for ``hold(session, 1, *names)``.

    Measured clear: doorway entry (``eye_hj_kraid_entry``) body-zero ~1321f,
    boss bit ~1520f, zero energy restored under unlimited-assist contract.
    """
    x = state.samus_x
    # Off-map / wall wrap: idle and let the runner re-place if needed.
    if x > 60_000:
        return ()

    if body_dead:
        # Death animation: keep some motion so the camera unlocks.
        if state.pose in (137, 138):  # knockback / spin variants
            if (frame_index // 30) % 2 == 0:
                return ("LEFT",)
            return ("RIGHT", "A", "B")
        return ("RIGHT", "B", "A")

    # Lane: stay left-mid so Supers hit the rising body without walking into it.
    if x > strategy.max_x or x < strategy.min_x:
        return lane_hold_action(
            x,
            min_x=strategy.min_x,
            max_x=strategy.max_x,
            face="RIGHT",
            dash=True,
        )

    return spray_action(
        frame_index,
        face="RIGHT",
        fire_period=strategy.fire_period,
        fire_hold_frames=strategy.fire_hold_frames,
        jump_period=strategy.jump_period,
        jump_hold_frames=strategy.jump_hold_frames,
        dash_when_not_jumping=True,
    )


def play_kraid_fight(
    session: ControllerSession,
    *,
    strategy: KraidStrategy = KraidStrategy(),
    require_boss_bit: bool = True,
) -> KraidEvidence:
    """Fight Kraid from the current doorway/arena entry until body dies.

    Expects ``session`` already in room ``0xA59F``. Does not door-warp or
    place Samus — start from a natural (or doorway-natural) entry state.
    """
    catalog = kraid_catalog()
    start = session.frame
    if session.state.room_id != ROOM_KRAID:
        raise RuntimeError(
            f"Kraid fight expected room 0x{ROOM_KRAID:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    if session.state.max_super_missiles > 0:
        ensure_weapon(session, WEAPON_SUPERS)

    # Composed doorway entries often load mid-air (pose 81). Idle until the
    # floor so the spray starts from a stable standing distribution. Settled
    # saves (y≥390, not falling) skip immediately.
    settle_standing(
        session,
        min_y=390,
        bad_poses=frozenset({81, 164}),
        max_frames=60,
        reason="fight_kraid_land",
    )

    peak_hp = 0
    min_hp = catalog.max_hp
    body_zero_frame: int | None = None
    boss_bit_frame: int | None = None
    prev_hp = body_hp(session.state)
    if 0 < prev_hp <= catalog.max_hp:
        peak_hp = prev_hp
        min_hp = prev_hp

    for index in range(strategy.max_fight_frames):
        state = session.state
        if state.room_id != ROOM_KRAID:
            break

        body_dead = body_zero_frame is not None
        names = fight_kraid_action(state, index, strategy, body_dead=body_dead)
        if names:
            hold(session, 1, *names, reason="fight_kraid")
        else:
            hold(session, 1, reason="fight_kraid_idle")

        post = session.state
        hp = body_hp(post)
        if 0 <= hp <= catalog.max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)

        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        if boss_bit_frame is None and kraid_defeated(session.env):
            boss_bit_frame = session.frame

        if require_boss_bit:
            if boss_bit_frame is not None:
                break
            if (
                body_zero_frame is not None
                and session.frame - body_zero_frame > strategy.boss_bit_grace_frames
            ):
                break
        elif body_zero_frame is not None:
            break

        prev_hp = hp

    final_hp = body_hp(session.state)
    boss_set = kraid_defeated(session.env)
    if boss_set:
        outcome = "kraid_defeated"
    elif body_zero_frame is not None:
        outcome = "kraid_body_zero_no_boss_bit"
    elif session.state.room_id != ROOM_KRAID:
        outcome = "left_room"
    else:
        outcome = "timeout"

    return KraidEvidence(
        start_frame=start,
        body_zero_frame=body_zero_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_body_hp=peak_hp,
        min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=final_hp,
        boss_bit_set=boss_set,
        outcome=outcome,
    )


@dataclass(frozen=True)
class VariaEvidence:
    """Rear-door exit + real Varia PLM collect after Kraid is defeated."""

    start_frame: int
    varia_room_frame: int | None
    collect_frame: int | None
    end_frame: int
    final_items: int
    final_room_id: int
    samus_x: int
    samus_y: int
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "varia_room_frame": self.varia_room_frame,
            "collect_frame": self.collect_frame,
            "end_frame": self.end_frame,
            "final_items": self.final_items,
            "final_items_hex": f"0x{self.final_items:04X}",
            "final_room_id": self.final_room_id,
            "final_room_id_hex": f"0x{self.final_room_id:04X}",
            "samus_x": self.samus_x,
            "samus_y": self.samus_y,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class KraidVariaEvidence:
    """Boss-only closeout: fight + rear door + Varia PLM."""

    fight: KraidEvidence
    varia: VariaEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "fight": self.fight.to_dict(),
            "varia": self.varia.to_dict(),
            "success": (
                self.fight.outcome == "kraid_defeated"
                and self.varia.outcome == "varia_collected"
            ),
        }


def play_kraid_rear_exit(
    session: ControllerSession,
    *,
    max_frames: int = 1_200,
) -> SuperMetroidState:
    """Exit Kraid's Room through the rear (right) door into Varia Suit Room.

    Expects Kraid already defeated (boss bit 0). Works from the mid-air
    post-death pose (~x=475) and from a grounded right-wall pin: hold right,
    pulse jump + beam shots so the blue door opens and the transition fires.
    """
    if session.state.room_id == ROOM_VARIA:
        return session.state
    if session.state.room_id != ROOM_KRAID:
        raise RuntimeError(
            f"kraid rear exit expected 0x{ROOM_KRAID:04X} or 0x{ROOM_VARIA:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )
    if not kraid_defeated(session.env):
        raise RuntimeError("kraid rear exit: Brinstar boss bit 0 not set")

    # Beams open the blue door; Supers are fine too but beam is the default.
    if session.state.selected_item != WEAPON_BEAM:
        try:
            select_weapon(session, WEAPON_BEAM)
        except RuntimeError:
            pass

    for index in range(max_frames):
        state = session.state
        if state.room_id == ROOM_VARIA:
            break
        if state.room_id != ROOM_KRAID:
            break

        # Mid-air death-anim pose: keep drifting right into the door zone.
        if state.samus_y < 360 and state.pose in (81, 19, 20, 25, 26, 27, 28):
            hold(session, 1, "RIGHT", reason="kraid_rear_fall")
            continue

        if state.pose in (137, 138):
            # Knockback near the right wall — keep pushing through rather than
            # idling forever on residual nails.
            hold(session, 1, "RIGHT", "B", "A", reason="kraid_rear_knockback")
            continue

        if state.samus_x < 400:
            hold(session, 1, "RIGHT", "B", reason="kraid_rear_run")
            continue

        phase = index % 24
        if phase < 3:
            hold(session, 1, "RIGHT", reason="kraid_rear_face")
        elif phase < 6:
            hold(session, 1, "X", reason="kraid_rear_shot")
        elif phase < 14:
            hold(session, 1, "RIGHT", "A", "B", reason="kraid_rear_jump")
        else:
            hold(session, 1, "RIGHT", "B", reason="kraid_rear_push")
    else:
        raise TimeoutError(
            f"kraid rear exit: door failed @ frame {session.frame}: {session.state}"
        )

    if session.state.room_id != ROOM_VARIA:
        raise RuntimeError(
            f"kraid rear exit: expected Varia 0x{ROOM_VARIA:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    # Door transition lands with Kraid-room x during ROOM_TRANSITION; wait for
    # ordinary gameplay in the 1-screen Varia room.
    for frame in range(400):
        state = hold(session, 1, reason="kraid_rear_settle")
        if (
            state.room_id == ROOM_VARIA
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 10
        ):
            break
    for _ in range(30):
        st = session.state
        if st.samus_y >= 130 and st.pose not in (81,):
            break
        hold(session, 1, reason="kraid_rear_land")
    return session.state


def play_varia_collect(
    session: ControllerSession,
    *,
    max_frames: int = 1_200,
    fanfare_frames: int = 480,
) -> int:
    """Shoot the Varia Chozo and collect the real PLM (item bit ``0x0001``).

    Expects ordinary gameplay in Varia Suit Room ``0xA6E2``. Chozo items need
    a shot to open the hand (same pattern as Hi-Jump); then walk into the orb
    near block (7, 9) ≈ pixel (112, 144).

    Returns the frame when the Varia bit first set.
    """
    if session.state.room_id != ROOM_VARIA:
        raise RuntimeError(
            f"varia collect expected room 0x{ROOM_VARIA:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )
    if session.state.collected_items & VARIA_MASK:
        return session.frame

    unmorph(session)
    if session.state.selected_item != WEAPON_BEAM:
        try:
            select_weapon(session, WEAPON_BEAM)
        except RuntimeError:
            pass

    collect_frame: int | None = None
    for index in range(max_frames):
        state = session.state
        if state.collected_items & VARIA_MASK:
            collect_frame = session.frame
            break
        if state.room_id != ROOM_VARIA:
            break

        # Pose 137/138 is knockback (and some fall recoveries). Idling never
        # leaves it — pulse direction + jump to stand, then resume the cycle.
        if state.pose in (137, 138, 9, 10):
            if index % 20 < 8:
                hold(session, 1, "UP", reason="varia_recover")
            elif index % 20 < 14:
                hold(session, 1, "A", reason="varia_recover")
            else:
                hold(session, 1, "LEFT" if state.samus_x > 90 else "RIGHT", reason="varia_recover")
            continue

        x = state.samus_x
        if x < 45:
            hold(session, 1, "RIGHT", "B", reason="varia_approach")
            continue
        if x > 150:
            hold(session, 1, "LEFT", "B", reason="varia_backoff")
            continue

        phase = index % 50
        if phase < 3:
            hold(session, 1, "RIGHT", reason="varia_face")
        elif phase < 5:
            hold(session, 1, reason="varia_face_release")
        elif phase < 8:
            # Standing beam shot opens the Chozo hand (Hi-Jump pattern).
            hold(session, 1, "X", reason="varia_statue_shot")
        elif phase < 16:
            hold(session, 1, "A", reason="varia_jump")
        elif phase < 20:
            hold(session, 1, "RIGHT", "X", reason="varia_air_shot")
        elif phase < 38:
            # Touch band: Chozo orb ~block (7,9) ≈ x=112.
            if x < 112:
                hold(session, 1, "RIGHT", "B", reason="varia_touch")
            elif x > 125:
                hold(session, 1, "LEFT", reason="varia_touch")
            else:
                hold(session, 1, "RIGHT", reason="varia_touch")
        else:
            hold(session, 1, reason="varia_wait")

        if session.state.collected_items & VARIA_MASK:
            collect_frame = session.frame
            break
    else:
        raise TimeoutError(
            f"varia collect: PLM not collected @ frame {session.frame}: {session.state}"
        )

    if collect_frame is None or not (session.state.collected_items & VARIA_MASK):
        raise RuntimeError(
            f"varia collect: left room or failed; items=0x{session.state.collected_items:04X}"
        )

    # Item fanfare locks controls substantially longer than the pickup flash.
    for _ in range(fanfare_frames):
        hold(session, 1, reason="varia_fanfare")
    return collect_frame


def play_kraid_to_varia(
    session: ControllerSession,
    *,
    max_exit_frames: int = 1_200,
    max_collect_frames: int = 1_200,
) -> VariaEvidence:
    """From defeated Kraid, rear-door exit and collect Varia (no fight)."""
    start = session.frame
    if session.state.collected_items & VARIA_MASK:
        return VariaEvidence(
            start_frame=start,
            varia_room_frame=start if session.state.room_id == ROOM_VARIA else None,
            collect_frame=start,
            end_frame=session.frame,
            final_items=session.state.collected_items,
            final_room_id=session.state.room_id,
            samus_x=session.state.samus_x,
            samus_y=session.state.samus_y,
            outcome="varia_collected",
        )

    varia_room_frame: int | None = None
    if session.state.room_id != ROOM_VARIA:
        play_kraid_rear_exit(session, max_frames=max_exit_frames)
    if session.state.room_id == ROOM_VARIA and varia_room_frame is None:
        varia_room_frame = session.frame

    collect_frame: int | None = None
    if session.state.room_id == ROOM_VARIA and not (
        session.state.collected_items & VARIA_MASK
    ):
        collect_frame = play_varia_collect(
            session, max_frames=max_collect_frames
        )

    if session.state.collected_items & VARIA_MASK:
        outcome = "varia_collected"
    elif session.state.room_id == ROOM_VARIA:
        outcome = "varia_room_no_item"
    else:
        outcome = "no_varia_room"

    return VariaEvidence(
        start_frame=start,
        varia_room_frame=varia_room_frame,
        collect_frame=collect_frame,
        end_frame=session.frame,
        final_items=session.state.collected_items,
        final_room_id=session.state.room_id,
        samus_x=session.state.samus_x,
        samus_y=session.state.samus_y,
        outcome=outcome,
    )


def play_kraid_fight_to_varia(
    session: ControllerSession,
    *,
    strategy: KraidStrategy = KraidStrategy(),
    require_boss_bit: bool = True,
) -> KraidVariaEvidence:
    """Boss-only closeout: Super-spray fight, rear door, Varia PLM.

    Expects doorway entry into Kraid's Room (``0xA59F``). Does not door-warp
    or place Samus. Not continuous evidence until composed on the power-on
    KPDR prefix after ``play_eye_to_kraid``.
    """
    fight = play_kraid_fight(
        session, strategy=strategy, require_boss_bit=require_boss_bit
    )
    if fight.outcome != "kraid_defeated":
        return KraidVariaEvidence(
            fight=fight,
            varia=VariaEvidence(
                start_frame=session.frame,
                varia_room_frame=None,
                collect_frame=None,
                end_frame=session.frame,
                final_items=session.state.collected_items,
                final_room_id=session.state.room_id,
                samus_x=session.state.samus_x,
                samus_y=session.state.samus_y,
                outcome="skipped_fight_failed",
            ),
        )
    varia = play_kraid_to_varia(session)
    return KraidVariaEvidence(fight=fight, varia=varia)
