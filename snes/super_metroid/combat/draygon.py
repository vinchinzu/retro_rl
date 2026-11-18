"""Development-only full-knowledge Draygon fight strategy.

This is a bounded strategy scaffold for the Maridia boss room. It is
explicitly developmentOnly: natural entry, continuous integration, and the
Space Jump collect/closeout are deferred. The controller does not warp,
place Samus, or write boss, event, item, or progression RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from retro_harness.actions import snes_action
from super_metroid.combat.features import (
    boss_defeated_in_state,
    draygon_catalog,
    features_from_state,
)
from super_metroid.combat.primitives import ensure_weapon, range_kite_action
from super_metroid.ram import SuperMetroidState, read_bank7e_wram
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_DRAYGON = 0xDA60
ROOM_SPACE_JUMP = 0xD9AA
WEAPON_SUPERS = 2
WEAPON_GRAPPLE = 4
_GRAB_LEFT_POSES = range(186, 192)


@dataclass(frozen=True)
class DraygonStrategy:
    """Tunable naive Super spray parameters for Draygon's body phase."""

    min_range: int = 90
    max_range: int = 220
    jump_range: int = 170
    jump_hold_frames: int = 16
    jump_period: int | None = 48
    fire_period: int = 3
    max_fight_frames: int = 15_000
    weapon: int = WEAPON_SUPERS


@dataclass(frozen=True)
class DraygonEvidence:
    """Measured result of one development-only Draygon fight attempt."""

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


def fight_draygon_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: DraygonStrategy = DraygonStrategy(),
) -> tuple[str, ...]:
    """Return one pure, RAM-driven Super spray action for Draygon.

    This intentionally models only a bounded, naive body spray. The selected
    weapon is controlled by the caller, so ``X`` does not grant Supers or
    change inventory state.
    """
    catalog = draygon_catalog()
    features = features_from_state(state, catalog)
    if features.enemy_defeated or state.enemy0_hp == 0:
        return ()
    gunk_action = draygon_gunk_clear_action(state, frame_index, strategy)
    if gunk_action:
        return gunk_action
    return range_kite_action(
        state.samus_x,
        features.enemy_x,
        min_range=strategy.min_range,
        max_range=strategy.max_range,
        jump_range=strategy.jump_range,
        frame_index=frame_index,
        jump_period=strategy.jump_period or 0,
        jump_hold_frames=strategy.jump_hold_frames,
        fire_period=strategy.fire_period,
        fire_button="X",
    )


def draygon_gunk_clear_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: DraygonStrategy = DraygonStrategy(),
) -> tuple[str, ...]:
    """Escape a contact-overlap gunk/grab window for one bounded frame.

    Contact overlap is the available structured-state signal for the dangerous
    close phase. Jumping away immediately prevents the normal kite action from
    spending another frame inside Draygon's large body hitbox; firing remains
    cadence-controlled so this helper does not change weapon throughput.
    """
    catalog = draygon_catalog()
    features = features_from_state(state, catalog)
    if features.enemy_defeated or state.enemy0_hp == 0 or not features.contact_overlap:
        return ()

    away = "LEFT" if features.dx >= 0 else "RIGHT"
    names = [away, "A"]
    if strategy.fire_period > 0 and frame_index % strategy.fire_period == 0:
        names.append("X")
    return tuple(names)


def play_draygon_fight(
    session: ControllerSession,
    *,
    strategy: DraygonStrategy = DraygonStrategy(),
) -> DraygonEvidence:
    """Fight Draygon until HP zero, the boss bit, or a bounded timeout.

    The session must already be in room ``0xDA60``. This is developmentOnly
    evidence; it makes no Space Jump collect and has no continuous claim.
    """
    catalog = draygon_catalog()
    start = session.frame
    if session.state.room_id != ROOM_DRAYGON:
        raise RuntimeError(
            f"Draygon fight expected room 0x{ROOM_DRAYGON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = session.state.enemy0_hp
    peak_hp = initial_hp
    min_hp = initial_hp
    body_zero_frame: int | None = None
    boss_bit_frame: int | None = None
    prev_hp = initial_hp
    if initial_hp == 0:
        body_zero_frame = start

    for index in range(strategy.max_fight_frames):
        state = session.state
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = session.frame
            break
        if body_zero_frame is not None:
            break

        names = fight_draygon_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_draygon")
        else:
            hold(session, 1, reason="fight_draygon_idle")

        post = session.state
        hp = post.enemy0_hp
        if 0 <= hp <= catalog.max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)
        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        if boss_bit_frame is None and boss_defeated_in_state(post, catalog):
            boss_bit_frame = session.frame
        if boss_bit_frame is not None or body_zero_frame is not None:
            break
        prev_hp = hp

    final_hp = session.state.enemy0_hp
    boss_set = boss_defeated_in_state(session.state, catalog)
    if boss_set:
        outcome = "draygon_defeated"
    elif body_zero_frame is not None:
        outcome = "draygon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"

    return DraygonEvidence(
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


def _draygon_defeated(env: object) -> bool:
    return bool(read_bank7e_wram(env)[0xD82C] & 1)


def _enemy0_hp(env: object) -> int:
    ram = env.get_ram()
    return int(ram[0x0F8C]) | (int(ram[0x0F8D]) << 8)


def play_clean_draygon_leave(session: ControllerSession) -> SimpleNamespace:
    """Defeat Draygon via Clean Grapple electrocution and settle in Space Jump Room.

    Uses zero ammo and zero RAM writes. The upper-left turret ``(40, 184)`` is
    already sparking on entry; after patrolling to dodge turret shots and
    entering Draygon's left-facing grab, a short save/restore lookahead finds
    the first frame where ``LEFT + X`` latches onto that turret and drains
    Draygon's 6000 HP. Once ``$7E:D82C`` bit 0 sets, Samus jumps onto the left
    ledge, shoots the blue door, and settles in ``0xD9AA``.
    """
    if session.state.room_id != ROOM_DRAYGON:
        raise RuntimeError(f"Draygon clean leave expected 0x{ROOM_DRAYGON:04X}, got 0x{session.state.room_id:04X}")

    env = session.env
    idle = snes_action()
    left_run = snes_action("LEFT", "B")
    left_only = snes_action("LEFT")
    left_grapple = snes_action("LEFT", "X")
    select_btn = snes_action("SELECT")
    restores = 0

    for _ in range(25):
        session.step(left_run, "draygon_drop")
    for _ in range(95):
        session.step(idle, "draygon_land")
    for _ in range(12):
        if session.state.selected_item == WEAPON_GRAPPLE:
            break
        session.step(select_btn, "draygon_select_grapple")
        session.step(idle, "draygon_select_settle")
    if session.state.selected_item != WEAPON_GRAPPLE:
        raise RuntimeError(f"failed to select Grapple Beam (selected={session.state.selected_item})")

    moving_right = True
    grabbed = False
    for frame in range(2400):
        st = session.state
        if st.pose in _GRAB_LEFT_POSES:
            grabbed = True
            break
        if st.samus_x >= 420:
            moving_right = False
        if st.samus_x <= 140:
            moving_right = True
        buttons = ["RIGHT" if moving_right else "LEFT", "B"]
        if (frame % 40) < 18:
            buttons.append("A")
        session.step(snes_action(*buttons), "draygon_patrol")
    if not grabbed:
        raise RuntimeError("Draygon did not grab Samus in left-facing pose")

    latched = False
    for _ in range(1500):
        st = session.state
        if st.samus_x <= 185 and 160 <= st.samus_y <= 215:
            saved = env.em.get_state()
            hit = False
            for _ in range(25):
                env.step(left_grapple)
                if _enemy0_hp(env) < 6000:
                    hit = True
                    break
            env.em.set_state(saved)
            restores += 1
            if hit:
                latched = True
                break
        session.step(left_only, "draygon_climb")
    if not latched:
        raise RuntimeError("failed to find Grapple latch window on upper-left turret")

    for _ in range(400):
        if _enemy0_hp(env) == 0:
            break
        session.step(left_grapple, "draygon_electrocute")
    if _enemy0_hp(env) != 0:
        raise RuntimeError(f"Draygon survived Grapple electrocution (hp={_enemy0_hp(env)})")

    for _ in range(12):
        if session.state.selected_item == 0:
            break
        session.step(select_btn, "draygon_deselect_grapple")
        session.step(idle, "draygon_deselect_settle")

    for _ in range(1200):
        if _draygon_defeated(env):
            break
        st = session.state
        if st.samus_y == 443 and st.samus_x > 115:
            session.step(left_run, "draygon_approach_exit")
        else:
            session.step(idle, "draygon_death_wait")
    if not _draygon_defeated(env):
        raise RuntimeError("Draygon boss bit $7E:D82C bit 0 did not set")

    right_run = snes_action("RIGHT", "B")
    left_jump = snes_action("LEFT", "B", "A")
    for k in range(600):
        st = session.state
        if st.room_id == ROOM_SPACE_JUMP and st.game_state == 8 and not st.door_transition:
            return SimpleNamespace(
                fight=SimpleNamespace(rollout_restores=restores),
                room_id=st.room_id,
                health=st.health,
            )
        if st.room_id == ROOM_SPACE_JUMP:
            session.step(idle, "draygon_settle_space_jump")
            continue
        if st.samus_y > 400:
            if st.samus_x < 95:
                for _ in range(15):
                    session.step(right_run, "draygon_exit_runup")
            for _ in range(28):
                session.step(left_jump, "draygon_exit_jump")
            for _ in range(15):
                session.step(left_only, "draygon_exit_land")
        elif k % 6 == 0:
            session.step(left_grapple, "draygon_open_door")
        else:
            session.step(left_run, "draygon_enter_door")
    raise RuntimeError("failed to settle in Space Jump Room 0xD9AA after defeating Draygon")
