"""No-assist Spore Spawn policy: left-ledge morph seat + live-eye missiles.

Human tape (full_start_v1 s6 hop 11, room ``0x9DC7``) sits morphed at the
left-corner ledge top ``(x≈21, y≈697)``. Windows 1–3 are two-missile
dash-bounces under the live eye. Later parks collapse after one hit.
Missiles are farmable from bouncing-spore droplets (sm-json 80% missiles)
so a short fight can clear on the natural 10-cap without resource writes.

This is the Clean-track fight. The assisted floor-bounce loop in
``routes/kpdr/spore_spawn.py`` remains as the old continuous path until this
policy is dual-green from natural entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from super_metroid.combat.features import (
    boss_defeated_in_state,
    spore_spawn_catalog,
)
from super_metroid.combat.primitives import ensure_weapon, settle_standing
from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    is_morph,
    select_weapon,
    unmorph,
)
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_SPORE_SPAWN = 0x9DC7
WEAPON_MISSILES = 1

# Human-tape morph seat (left corner of the raised pit ledge).
SEAT_X = 21
SEAT_Y = 697
LEDGE_Y_MIN = 668
LEDGE_Y_MAX = 705
FLOOR_Y = 710

# Mouth-open / fully-open hold spritemaps (same set as the continuous fight).
VULNERABLE_SPRITEMAPS = frozenset(
    {
        0xEE79,
        0xEE8B,
        0xEE9D,
        0xEEAF,
        0xEEC1,
        0xEED3,
        0xEEE5,
        0xEF3D,
        0xEF4F,
        0xEF61,
    }
)
# Fully-open holds — transition maps (EE79…) are not a reliable hit.
FULLY_OPEN_SPRITEMAPS = frozenset({0xEF3D, 0xEF4F, 0xEF61})

# Enemy-projectile tables (Kejardon / PJBoy bank $86): 18 slots × 2 bytes.
# $1997 is the header pointer (not a 1-byte type). $19BB is graphics/palette.
# Pickups are projectile $F337; kind is the instruction list at $1B47.
N_ENEMY_PROJECTILES = 18
ADDR_PROJ_ID = 0x1997
ADDR_PROJ_X = 0x1A4B
ADDR_PROJ_Y = 0x1A93
ADDR_PROJ_ILIST = 0x1B47
PICKUP_PROJ_ID = 0xF337
ILIST_SMALL_ENERGY = 0xED8D
ILIST_BIG_ENERGY = 0xEDA3
ILIST_MISSILES = 0xEDB9
PICKUP_SMALL_ENERGY = 0x16
PICKUP_BIG_ENERGY = 0x17
PICKUP_MISSILE = 0x18
_ILIST_TO_KIND = {
    ILIST_SMALL_ENERGY: PICKUP_SMALL_ENERGY,
    ILIST_BIG_ENERGY: PICKUP_BIG_ENERGY,
    ILIST_MISSILES: PICKUP_MISSILE,
}


@dataclass(frozen=True)
class SporeSpawnStrategy:
    """Left-ledge ball + two-missile eye windows + droplet farm."""

    seat_x: int = SEAT_X
    seat_x_max: int = 28
    missiles_per_window: int = 2
    # Late parks collapse after the first hit (10f cooldown is too long).
    # Allow a 1-missile chip so the last 60 HP does not go to farm.
    min_missiles_to_fire: int = 1
    farm_until: int = 2
    jump_hold_frames: int = 48
    seat_hop_frames: int = 8
    hop_x_min: int = 62
    hop_x_max: int = 78
    missile_cooldown: int = 10
    fire_hold_frames: int = 2
    max_fight_frames: int = 24_000
    farm_sweep_frames: int = 1_600
    boss_bit_grace_frames: int = 1_200
    # Leave the seat once the eye is on the right half. Later windows park
    # around (142, 604), not the first-window (185, 586) — fire under the
    # live eye, not a hardcoded x=180–195 band.
    fire_enemy_x_min: int = 120
    fire_enemy_x_max: int = 200
    fire_enemy_y_min: int = 540
    fire_enemy_y_max: int = 640
    fire_airborne_y: int = 660
    fire_x_slop: int = 10
    fire_x_left: int = 4
    fire_x_right: int = 12
    fire_y_max: int = 624
    takeoff_behind: int = 16
    fire_x_bias: int = 6
    fire_standoff: int = 32
    fire_close_x: int = 64
    fire_close_y: int = 70


@dataclass(frozen=True)
class SporeSpawnEvidence:
    start_frame: int
    activation_seen: bool
    defeat_frame: int | None
    boss_bit_frame: int | None
    end_frame: int
    peak_hp: int
    min_enemy_hp: int
    action_frames: int
    final_enemy_hp: int
    shots_fired: int
    farm_frames: int
    windows: int
    outcome: str
    vulnerable_spritemaps: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "activation_seen": self.activation_seen,
            "defeat_frame": self.defeat_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "peak_hp": self.peak_hp,
            "min_enemy_hp": self.min_enemy_hp,
            "action_frames": self.action_frames,
            "final_enemy_hp": self.final_enemy_hp,
            "shots_fired": self.shots_fired,
            "farm_frames": self.farm_frames,
            "windows": self.windows,
            "outcome": self.outcome,
            "vulnerable_spritemaps": list(self.vulnerable_spritemaps),
        }


@dataclass(frozen=True)
class Pickup:
    slot: int
    kind: int
    x: int
    y: int


def mouth_open(state: SuperMetroidState) -> bool:
    """True while Spore Spawn's eye/mouth spritemap is a damage window."""
    return int(state.enemy0_spritemap) in VULNERABLE_SPRITEMAPS


def eye_fully_open(state: SuperMetroidState) -> bool:
    """True on the long open-hold maps where missiles actually connect."""
    return int(state.enemy0_spritemap) in FULLY_OPEN_SPRITEMAPS


def on_left_ledge(state: SuperMetroidState) -> bool:
    """True when Samus is on the raised left pit ledge (not the floor)."""
    return LEDGE_Y_MIN <= int(state.samus_y) <= LEDGE_Y_MAX and int(state.samus_x) <= 80


def under_eye(
    state: SuperMetroidState,
    slop: int | None = None,
    *,
    left: int = 4,
    right: int = 12,
) -> bool:
    """True when a straight-up missile from Samus can hit the live eye.

    Hits sit to the right of the eye: (193, 185)=+8, (193, 182)=+11 after
    the flinch. x=174 vs 185 (−11) misses. ``slop`` (legacy) uses ±slop.
    """
    dx = int(state.samus_x) - int(state.enemy0_x)
    if slop is not None:
        return abs(dx) <= slop
    return -left <= dx <= right


def in_fire_height(state: SuperMetroidState, below: int = 30) -> bool:
    """True when Samus is at/below the eye, close enough to clear the shell.

    y grows downward. Window-1 hits were 18–26px below (612 vs 586);
    36px below (622 vs 586) spent into the stalk. Later parks sit lower
    (604) so the same 30px window still includes the 622 bounce. A
    floor park (eye y≥650) is next to standing Samus — allow the floor.
    """
    eye_y = int(state.enemy0_y)
    samus_y = int(state.samus_y)
    if eye_y >= 650:
        # Must be at/below the eye (missiles go up). 35px below hits the
        # stalk; 8px above spends into empty air.
        return eye_y <= samus_y <= eye_y + 20
    return eye_y <= samus_y <= eye_y + below


def _low_floor_park(state: SuperMetroidState) -> bool:
    """Eye parked near the floor (window 4 sat at (96, 666))."""
    return int(state.enemy0_y) >= 650


def _high_right_park(state: SuperMetroidState) -> bool:
    """First-window park: eye on the right wall at y≈586, not the later (142, 604)."""
    return int(state.enemy0_x) >= 170 and int(state.enemy0_y) < 595


def _fire_target_x(state: SuperMetroidState, bias: int) -> int:
    """Bounce peak we want: proven 188 band, or just right of a low park."""
    if _high_right_park(state):
        return 188
    return int(state.enemy0_x) + bias


def _in_fire_band(state: SuperMetroidState, strategy: SporeSpawnStrategy) -> bool:
    return (
        strategy.fire_enemy_x_min <= state.enemy0_x <= strategy.fire_enemy_x_max
        and strategy.fire_enemy_y_min <= state.enemy0_y <= strategy.fire_enemy_y_max
    )


def seated(state: SuperMetroidState, strategy: SporeSpawnStrategy | None = None) -> bool:
    """Morphed in the left-corner seat from the human tape."""
    strat = strategy or SporeSpawnStrategy()
    return (
        is_morph(int(state.pose))
        and int(state.samus_x) <= strat.seat_x_max
        and LEDGE_Y_MIN <= int(state.samus_y) <= LEDGE_Y_MAX
    )


def _read_u16(ram: Any, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def list_pickups(env: Any) -> tuple[Pickup, ...]:
    """Live enemy-projectile pickups (energy / missiles). Empty if no RAM."""
    if env is None:
        return ()
    try:
        ram = env.get_ram()
    except Exception:
        return ()
    need = ADDR_PROJ_ILIST + N_ENEMY_PROJECTILES * 2
    if ram is None or len(ram) < need:
        return ()
    found: list[Pickup] = []
    for slot in range(N_ENEMY_PROJECTILES):
        header = _read_u16(ram, ADDR_PROJ_ID + slot * 2)
        if header == PICKUP_PROJ_ID:
            kind = _ILIST_TO_KIND.get(_read_u16(ram, ADDR_PROJ_ILIST + slot * 2), 0)
        else:
            kind = header & 0xFF
        if kind not in (PICKUP_SMALL_ENERGY, PICKUP_BIG_ENERGY, PICKUP_MISSILE):
            continue
        x = _read_u16(ram, ADDR_PROJ_X + slot * 2)
        y = _read_u16(ram, ADDR_PROJ_Y + slot * 2)
        if x == 0 and y == 0:
            continue
        found.append(Pickup(slot=slot, kind=kind, x=x, y=y))
    return tuple(found)


def fight_spore_spawn_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: SporeSpawnStrategy = SporeSpawnStrategy(),
) -> tuple[str, ...]:
    """One-frame seat / fire / farm hint (tests + AP). Play loop owns morph."""
    if state.enemy0_hp == 0:
        return ()
    if seated(state, strategy):
        if (
            mouth_open(state)
            and state.missiles >= strategy.min_missiles_to_fire
            and state.enemy0_x >= strategy.fire_enemy_x_min
        ):
            return ("UP",)
        return ()
    if on_left_ledge(state) and state.samus_x > strategy.seat_x_max:
        return ("LEFT",)
    if state.samus_y >= FLOOR_Y:
        if state.samus_x > 80:
            return ("LEFT", "B")
        return ("LEFT", "A")
    if mouth_open(state) and state.missiles >= 1 and not is_morph(state.pose):
        names: list[str] = []
        if state.samus_y > strategy.fire_airborne_y:
            names.append("A")
        if state.enemy0_y + 20 < state.samus_y:
            names.append("UP")
        if frame_index % (strategy.missile_cooldown + strategy.fire_hold_frames) < (
            strategy.fire_hold_frames
        ):
            names.append("X")
        return tuple(names)
    return ()


def _dead(session: ControllerSession) -> bool:
    st = session.state
    return int(st.health) == 0 or st.phase is GameplayPhase.DEATH_OR_GAME_OVER


def _morph_in_corner(session: ControllerSession, strategy: SporeSpawnStrategy) -> None:
    """Walk LEFT on the ledge to the human corner, then morph."""
    for _ in range(40):
        st = session.state
        if _dead(session) or seated(st, strategy):
            return
        if not on_left_ledge(st):
            return
        if st.samus_x <= strategy.seat_x_max:
            break
        hold(session, 1, "LEFT", reason="spore_ledge_left")
    if _dead(session) or seated(session.state, strategy):
        return
    if on_left_ledge(session.state) and not is_morph(session.state.pose):
        try:
            ensure_morph(session)
        except TimeoutError:
            return


def _go_to_seat(session: ControllerSession, strategy: SporeSpawnStrategy) -> None:
    """Land, one short hop from x≈70, morph in the human-tape corner.

    Tune-hop from the enter pin: LEFT+A ×8 at x≈74 lands (21, 697) pose 65.
    Holding A against the left wall (x<50) wall-jumps out of the pit.
    """
    if session.state.samus_y < 500:
        for _ in range(180):
            if session.state.samus_y >= LEDGE_Y_MIN or _dead(session):
                break
            hold(session, 1, reason="spore_fall_in")
    settle_standing(session, min_y=LEDGE_Y_MIN, max_frames=80, reason="spore_land")
    if seated(session.state, strategy) or _dead(session) or session.state.enemy0_hp == 0:
        return
    if on_left_ledge(session.state):
        _morph_in_corner(session, strategy)
        return

    for _ in range(90):
        st = session.state
        if _dead(session) or seated(st, strategy) or st.enemy0_hp == 0:
            return
        if on_left_ledge(st):
            _morph_in_corner(session, strategy)
            return
        if st.samus_y < LEDGE_Y_MIN:
            hold(session, 1, reason="spore_fall_in")
            continue
        if st.samus_y >= FLOOR_Y and strategy.hop_x_min <= st.samus_x <= strategy.hop_x_max:
            break
        if st.samus_x < strategy.hop_x_min:
            hold(session, 1, "RIGHT", reason="spore_off_wall")
        else:
            hold(session, 1, "LEFT", "B", reason="spore_floor_left")

    st = session.state
    if seated(st, strategy) or _dead(session) or st.enemy0_hp == 0:
        return
    if on_left_ledge(st):
        _morph_in_corner(session, strategy)
        return
    if st.samus_y >= FLOOR_Y and 60 <= st.samus_x <= 80:
        hold(session, strategy.seat_hop_frames, "LEFT", "A", reason="spore_ledge_hop")
        for _ in range(28):
            st = session.state
            if _dead(session) or on_left_ledge(st):
                break
            if st.samus_x < 50:
                hold(session, 1, reason="spore_hop_idle")
            else:
                hold(session, 1, "LEFT", reason="spore_ledge_left")
    if on_left_ledge(session.state):
        _morph_in_corner(session, strategy)


def _keep_seat(session: ControllerSession, strategy: SporeSpawnStrategy) -> None:
    """Nudge back into the morph corner if a shot or hit pushed Samus out."""
    st = session.state
    if seated(st, strategy):
        return
    if is_morph(st.pose) and on_left_ledge(st) and st.samus_x > strategy.seat_x:
        hold(session, 4, "LEFT", reason="spore_seat_nudge")
        return
    if on_left_ledge(st):
        if st.samus_x > strategy.seat_x_max:
            hold(session, 4, "LEFT", reason="spore_seat_nudge")
        if not is_morph(session.state.pose):
            ensure_morph(session)
        return
    _go_to_seat(session, strategy)


def _fire_window(
    session: ControllerSession, strategy: SporeSpawnStrategy
) -> int:
    """Dash-bounce under the live eye and tap two straight-up missiles.

    Window 1 parks at (185, 586); later windows park near (142, 604). A
    hardcoded x=180–195 band spends both shots into empty air. Missiles go
    straight up — no LEFT/RIGHT on the fire frame. Dash-bounce (B+A) is
    required: standing hops stall at y≈660. Take off from ~30px left of
    the eye so the bounce peaks underneath.
    """
    if is_morph(session.state.pose):
        unmorph(session)
    if session.state.missiles <= 0:
        return 0
    try:
        ensure_weapon(session, WEAPON_MISSILES)
    except RuntimeError:
        return 0

    left = strategy.fire_x_left
    right = strategy.fire_x_right
    bias = strategy.fire_x_bias
    for _ in range(100):
        st = session.state
        if _dead(session) or st.enemy0_hp == 0:
            return 0
        if st.samus_x < 55:
            hold(session, 1, "RIGHT", reason="spore_off_wall")
            continue
        target = _fire_target_x(st, bias)
        close = 40 if _high_right_park(st) else 12 if _low_floor_park(st) else 18
        if st.samus_y >= FLOOR_Y and abs(st.samus_x - target) <= close:
            break
        face = "RIGHT" if st.samus_x < target else "LEFT"
        if st.samus_y >= FLOOR_Y:
            # Dash only while far — a long run-up through a low park
            # overlaps the body (pose 143) and skips the fire frame.
            dash = _high_right_park(st) or abs(st.samus_x - target) > 50
            names = (face, "B") if dash else (face,)
            hold(session, 1, *names, reason="spore_close")
        else:
            hold(session, 1, face, reason="spore_close")

    shots = 0
    jump_hold = 0
    last_spend = -99
    hurt = frozenset({81, 83, 84, 109, 143, 158, 159, 160})
    for _index in range(220):
        st = session.state
        if st.enemy0_hp == 0 or _dead(session):
            break
        on_floor = st.samus_y >= FLOOR_Y
        if st.samus_x < 55:
            jump_hold = 0
            hold(session, 1, "RIGHT", reason="spore_off_wall")
            continue
        # Steer at the live eye (not eye+bias). Bias is only the approach
        # target; using it here leaves a dead zone at x=157 vs eye 142
        # (inside ±(eye+6) so no steer, outside ±eye so no fire).
        aim = int(st.enemy0_x)
        dx = st.samus_x - aim
        if dx < -left:
            jump_dir = "RIGHT"
        elif dx > right:
            jump_dir = "LEFT"
        else:
            jump_dir = ""
        if on_floor and jump_hold == 0:
            if _low_floor_park(st):
                # Short hop: 18f peaks above the eye (y≈651) and contacts.
                # 8f stays in the 20px-below band for both taps.
                jump_hold = 12 if mouth_open(st) else 0
            else:
                jump_hold = 52 if mouth_open(st) else 36
        hold_jump = jump_hold > 0
        # Don't rise through the eye — the second tap then spends above it.
        if not on_floor and st.samus_y <= int(st.enemy0_y) + 12:
            hold_jump = False
        jump_hold = max(0, jump_hold - 1)
        fire = (
            under_eye(st, left=left, right=right)
            and in_fire_height(st)
            and mouth_open(st)
            and st.pose not in hurt
            and st.missiles > 0
            and shots < strategy.missiles_per_window
        )
        tap_x = fire and (session.frame - last_spend) >= 10
        if fire and tap_x:
            names_list = ["UP", "A", "X"]
        elif fire:
            # Stay put through the 10f cooldown. Steering here walked
            # window 1 off the proven (193, 612) seat; if the eye leaves
            # slop, `fire` drops and the airborne steer tracks it.
            names_list = ["UP", "A"]
        elif on_floor:
            if _low_floor_park(st):
                names_list = [jump_dir or "RIGHT"]
                if hold_jump:
                    names_list.append("A")
            else:
                names_list = [jump_dir or "RIGHT", "A"]
                if hold_jump:
                    names_list.append("B")
        else:
            names_list = ["UP"]
            if jump_dir:
                names_list.insert(0, jump_dir)
            if hold_jump:
                names_list.append("A")
                if not _low_floor_park(st):
                    names_list.append("B")
        ms_before = st.missiles
        hold(session, 1, *tuple(dict.fromkeys(names_list)), reason="spore_eye_shot")
        if session.state.missiles < ms_before:
            shots += 1
            last_spend = session.frame
        if shots >= strategy.missiles_per_window:
            break

    if not _dead(session) and session.state.enemy0_hp > 0:
        for _ in range(24):
            if session.state.pose not in hurt:
                break
            hold(session, 1, reason="spore_land")
        _go_to_seat(session, strategy)
        if not seated(session.state, strategy) and not _dead(session):
            _go_to_seat(session, strategy)
    return shots


def _walk_toward_x(session: ControllerSession, target_x: int, frames: int) -> None:
    for _ in range(frames):
        st = session.state
        if abs(st.samus_x - target_x) <= 4:
            return
        hold(
            session,
            1,
            "RIGHT" if st.samus_x < target_x else "LEFT",
            reason="spore_farm_walk",
        )


def _farm_drops(session: ControllerSession, strategy: SporeSpawnStrategy) -> int:
    """Shoot bouncing spores with the beam; collect $F337 missile pickups.

    Spores only drop when shot (bank $86 DC34). select_weapon(missiles) at 0
    ammo raises — stay on beam until missiles > 0.
    """
    start = session.frame
    want = min(strategy.farm_until, int(session.state.max_missiles) or strategy.farm_until)
    if is_morph(session.state.pose):
        unmorph(session)
    try:
        select_weapon(session, 0)
    except RuntimeError:
        pass
    env = getattr(session, "env", None)
    walk_right = True
    for index in range(strategy.farm_sweep_frames):
        st = session.state
        if _dead(session) or st.enemy0_hp == 0 or st.missiles >= want:
            break
        if st.samus_y < 500:
            hold(session, 1, reason="spore_farm_fall")
            continue
        drops = list_pickups(env)
        missiles = [d for d in drops if d.kind == PICKUP_MISSILE]
        if missiles:
            _walk_toward_x(session, missiles[0].x, 6)
            continue
        if st.samus_x >= 200:
            walk_right = False
        elif st.samus_x <= 50:
            walk_right = True
        face = "RIGHT" if walk_right else "LEFT"
        names = [face, "UP"]
        if index % 40 < 18:
            names.append("A")
        if index % 8 < 2:
            names.append("X")
        hold(session, 1, *names, reason="spore_farm_shoot")
    if session.state.missiles > 0:
        try:
            ensure_weapon(session, WEAPON_MISSILES)
        except RuntimeError:
            pass
    _go_to_seat(session, strategy)
    return session.frame - start


def play_spore_spawn_fight(
    session: ControllerSession,
    *,
    strategy: SporeSpawnStrategy = SporeSpawnStrategy(),
    require_boss_bit: bool = True,
) -> SporeSpawnEvidence:
    """Fight Spore Spawn from room ``0x9DC7`` until HP 0 (+ optional boss bit)."""
    catalog = spore_spawn_catalog()
    start = session.frame
    if session.state.room_id != ROOM_SPORE_SPAWN:
        raise RuntimeError(
            f"Spore Spawn fight expected room 0x{ROOM_SPORE_SPAWN:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    if session.state.missiles > 0:
        try:
            ensure_weapon(session, WEAPON_MISSILES)
        except RuntimeError:
            pass
    _go_to_seat(session, strategy)

    peak_hp = session.state.enemy0_hp
    min_hp = session.state.enemy0_hp
    activation_seen = peak_hp >= 900 or mouth_open(session.state)
    defeat_frame: int | None = start if peak_hp == 0 else None
    boss_bit_frame: int | None = None
    shots_fired = 0
    farm_frames = 0
    windows = 0
    seen: set[int] = set()
    prev_hp = session.state.enemy0_hp

    for _ in range(strategy.max_fight_frames):
        state = session.state
        if state.room_id != ROOM_SPORE_SPAWN:
            break
        if _dead(session):
            break
        peak_hp = max(peak_hp, state.enemy0_hp)
        min_hp = min(min_hp, state.enemy0_hp)
        if mouth_open(state):
            seen.add(int(state.enemy0_spritemap))
            activation_seen = True

        if defeat_frame is None and state.enemy0_hp == 0 and prev_hp > 0:
            defeat_frame = session.frame
            min_hp = 0
        prev_hp = state.enemy0_hp

        if defeat_frame is not None:
            if boss_defeated_in_state(session.state, catalog):
                boss_bit_frame = session.frame
                break
            if not require_boss_bit:
                break
            if session.frame - defeat_frame >= strategy.boss_bit_grace_frames:
                break
            hold(session, 1, reason="spore_death_anim")
            continue

        if state.missiles < strategy.min_missiles_to_fire:
            farm_frames += _farm_drops(session, strategy)
            continue

        # Leave the ball on the first vulnerable spritemap of the right-wall open.
        # Leave the ball on the first right-side open; _fire_window waits
        # at the standoff for the fully-open hold before spending missiles.
        ready = (
            mouth_open(state)
            and state.enemy0_x >= 120
            and state.missiles >= strategy.min_missiles_to_fire
            and (seated(state, strategy) or on_left_ledge(state))
        )
        if ready:
            windows += 1
            shots_fired += _fire_window(session, strategy)
            continue

        if not seated(state, strategy):
            _keep_seat(session, strategy)
        else:
            hold(session, 1, reason="spore_wait_eye")

    if _dead(session):
        outcome = "died"
    elif defeat_frame is not None and (
        not require_boss_bit or boss_bit_frame is not None
    ):
        outcome = "spore_spawn_defeated"
    elif defeat_frame is not None:
        outcome = "boss_bit_timeout"
    else:
        outcome = "timeout"

    return SporeSpawnEvidence(
        start_frame=start,
        activation_seen=activation_seen,
        defeat_frame=defeat_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_hp=peak_hp,
        min_enemy_hp=min_hp,
        action_frames=session.frame - start,
        final_enemy_hp=session.state.enemy0_hp,
        shots_fired=shots_fired,
        farm_frames=farm_frames,
        windows=windows,
        outcome=outcome,
        vulnerable_spritemaps=tuple(sorted(seen)),
    )
