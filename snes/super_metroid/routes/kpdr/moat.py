"""Crateria Kihunter → Moat shinespark → West Ocean (K6).

Natural source: ``scratch/speed_with_spazer_human_end.state`` (or promoted
``scratch/post_kihunter_pre_moat_spark.state``) — Speed + HJ + Varia + PBs,
beams Charge+Wave+Ice+Spazer, standing in Crateria Kihunter ``0x948C``.

Harness buttons (canonical — do not swap for VOD A/B labels):

* **B = dash / speed-charge**, **A = jump / shine-activate**, **DOWN = store**.
* Verified: ``RIGHT+B`` builds echoes; ``RIGHT+A`` does not. After store,
  ``RIGHT+B`` only walks while $0A68 drains.

Phases (bot / pure — verified hop path from pin):

1. **Clear** flying Kihunters so knockback cannot cancel store.
2. **Runway** — left floor band; charge start off both doors.
3. **Charge** — grounded ``RIGHT+B`` to sc≥4 pose 9 (trench x≈503 y≈178 OK).
4. **Store** — DOWN from pose 9 arms $0A68. Do **not** store from spin
   pose 25/166 (DOWN crouches but wipes echoes without arming).
5. **Hop-unspin-activate** — stand → micro-run (leave crouch) →
   ``RIGHT+B+A`` spin over x555 → UP mid-air → horizontal spark pose 201
   into Moat at y≈115–122 → jam ~x475 → after spark dies, ``RIGHT+X``
   opens blue door into West Ocean ``0x93FE``.

Residual / probe: ``docs/tasks/SM-MOAT-SHINESPARK-residual.md``,
``scripts/probe/moat_spark_watch.py hop``.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, select_weapon
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback

ROOM_KIHUNTER = 0x948C
ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE

# Enemy table: 0x40-byte slots from $0F78 (low WRAM mirror).
_ENEMY_BASE = 0x0F78
_ENEMY_STRIDE = 0x40
_ENEMY_HP_OFF = 0x14
_ENEMY_X_OFF = 0x02
_ENEMY_Y_OFF = 0x06
_MAX_ENEMIES = 8
# Kihunters ~60 HP; ignore deep-pit / high-HP junk while clearing air.
_KIHUNTER_HP_MAX = 120

_CLEAR_BUDGET = 2400
_RUNWAY_BUDGET = 700
_CHARGE_BUDGET = 500
_STORE_FRAMES = 18
# Hop-unspin-activate band (verified pure from post_kihunter_pre_moat_spark)
_STAND_FRAMES = 4
_MICRO_RUN = 2
_HOP_FRAMES = 14
_UNSPIN_FRAMES = 3
_SPARK_ACTIVATE = 16
_SPARK_TRAVEL = 700
_TOTAL_OK_ROOMS = frozenset({ROOM_KIHUNTER, ROOM_MOAT, ROOM_WEST_OCEAN})
_SPARK_POSES = frozenset({199, 200, 201, 202})


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def list_air_enemies(env) -> list[tuple[int, int, int, int, int]]:
    """Return ``(slot, x, y, hp, enemy_id)`` for living upper-room Kihunters.

    Super Metroid leaves stale HP in freed slots — **enemy id at +0x00 must be
    non-zero** or the slot is dead/garbage (was the stuck ``air_enemies=2`` bug).
    Kihunters are paired body/parts (``0xEABF`` / ``0xEAFF``); both count until
    their ids clear. Scisers in the pit (y large / high HP) are ignored.
    """
    ram = env.get_ram()
    out: list[tuple[int, int, int, int, int]] = []
    for i in range(_MAX_ENEMIES):
        base = _ENEMY_BASE + i * _ENEMY_STRIDE
        enemy_id = _u16(ram, base)  # +0x00 — 0 means free/dead slot
        if enemy_id == 0:
            continue
        hp = _u16(ram, base + _ENEMY_HP_OFF)
        if hp <= 0 or hp > _KIHUNTER_HP_MAX:
            continue
        x = _u16(ram, base + _ENEMY_X_OFF)
        y = _u16(ram, base + _ENEMY_Y_OFF)
        # Ignore deep-pit Scisers (y large).
        if y > 350:
            continue
        out.append((i, x, y, hp, enemy_id))
    return out


def air_enemies_alive(env) -> bool:
    return bool(list_air_enemies(env))


def air_enemy_count(env) -> int:
    return len(list_air_enemies(env))


def _session_env(session: ControllerSession):
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("moat controller needs session.env for enemy scan")
    return env


# Crateria Kihunter doors (room is 3 screens wide; left = Tube, right = Moat).
# Human left-door entry ~x≤22; never hold LEFT inside the lip band.
_LEFT_DOOR_LIP_X = 80  # hard: steer RIGHT if x under this
_WALK_STOP_X = 180  # stop walking left; plant-fire from here
_RUNWAY_X = (140, 200)  # charge start band (stay off both doors)
_RIGHT_DOOR_LIP_X = 720  # Moat door — only enter on purpose during spark
_FACING_LEFT = 8


def near_left_door(state: SuperMetroidState) -> bool:
    """True when Samus is on the Tube-door lip (must not hold LEFT)."""
    return (
        state.room_id == ROOM_KIHUNTER
        and state.samus_x < _LEFT_DOOR_LIP_X
        and state.samus_y < 250
    )


def near_right_door(state: SuperMetroidState) -> bool:
    """True when Samus is on the Moat-door lip."""
    return (
        state.room_id == ROOM_KIHUNTER
        and state.samus_x > _RIGHT_DOOR_LIP_X
        and state.samus_y < 250
    )


def avoid_kihunter_doors(
    session: ControllerSession,
    *,
    label: str,
    allow_right: bool = False,
) -> bool:
    """Hard door guard. Returns True if a steer frame was applied.

    Detects left Tube lip / right Moat lip by x-band (door shells sit at the
    room edges). Holding into a blue door walks through it — so clear/runway
    must never hold LEFT inside the left lip.
    """
    st = session.state
    if st.room_id != ROOM_KIHUNTER:
        return False
    if st.door_transition != 0:
        # Transition already started — idle; caller should treat as fail soon.
        hold(session, 1, reason=f"{label}_door_transition")
        return True
    if near_left_door(st):
        hold(session, 1, "RIGHT", reason=f"{label}_avoid_tube")
        return True
    if not allow_right and near_right_door(st):
        hold(session, 1, "LEFT", reason=f"{label}_avoid_moat")
        return True
    return False


def play_clear_kihunter_room(session: ControllerSession) -> SuperMetroidState:
    """Clear flying Kihunters: walk LEFT and shoot. No chase / L-R thrash.

    Start is on the right (Moat door). Walk left while firing, then **plant**
    facing into the room (usually RIGHT — enemies are not in the Tube wall).
    Shoot only (no direction hold) so we never walk through doors.
    """
    require_room(session, ROOM_KIHUNTER, "kihunter_clear")
    select_weapon(session, 0)
    env = _session_env(session)
    label = "kihunter_clear"
    plant_face: str | None = None
    zero_streak = 0

    for frame in range(_CLEAR_BUDGET):
        st = session.state
        if st.room_id != ROOM_KIHUNTER:
            raise TimeoutError(
                f"{label}: left room 0x{st.room_id:04X} during clear "
                f"(Tube/Moat door — guard failed)"
            )
        if avoid_kihunter_doors(session, label=label, allow_right=False):
            plant_face = None
            continue

        if is_knockback(st):
            prefer = "RIGHT" if st.samus_x < _WALK_STOP_X else "LEFT"
            escape_knockback_spin(
                session,
                prefer_dir=prefer,
                run_frames=4,
                spin_frames=12,
                label=label,
                run_with=("B", "X"),
                spin_with=("B", "X"),
            )
            avoid_kihunter_doors(session, label=label, allow_right=False)
            continue

        enemies = list_air_enemies(env)
        if not enemies:
            zero_streak += 1
            hold(session, 1, reason=f"{label}_confirm")
            # Need a few clean frames — free slots can flicker.
            if zero_streak >= 20:
                return session.state
            continue
        zero_streak = 0

        # Plant zone: stand still, face toward remaining enemies, shoot only.
        if st.samus_x <= _WALK_STOP_X:
            # Enemies after a left walk are almost always to the RIGHT (in-room).
            # Shooting LEFT only hits the Tube wall — that was the "stuck glitching".
            mean_x = sum(e[1] for e in enemies) / len(enemies)
            want = "RIGHT" if mean_x >= st.samus_x - 8 else "LEFT"
            # Never hold toward Tube lip.
            if want == "LEFT" and st.samus_x < _LEFT_DOOR_LIP_X + 40:
                want = "RIGHT"
            if plant_face != want:
                hold(session, 2, want, reason=f"{label}_face")
                hold(session, 3, reason=f"{label}_face_release")
                plant_face = want
            pulse = frame % 32
            if pulse < 26:
                hold(session, 1, "X", reason=f"{label}_plant")
            else:
                hold(session, 1, reason=f"{label}_plant_rel")
            continue

        # Approach: walk left + shoot (still outside door lip).
        plant_face = None
        pulse = frame % 30
        if pulse < 24:
            hold(session, 1, "LEFT", "X", reason=f"{label}_walk_shot")
        elif pulse < 26:
            hold(session, 1, "LEFT", reason=f"{label}_release")
        else:
            hold(session, 1, "LEFT", "X", reason=f"{label}_walk_shot")

    raise TimeoutError(
        f"{label}: air enemies still alive after left-walk clear: "
        f"{list_air_enemies(env)} state={session.state}"
    )


def play_kihunter_left_runway(session: ControllerSession) -> SuperMetroidState:
    """Walk to left floor runway for speed charge — stay off Tube door."""
    require_room(session, ROOM_KIHUNTER, "kihunter_runway")
    label = "kihunter_runway"
    lo, hi = _RUNWAY_X
    for _ in range(_RUNWAY_BUDGET):
        st = session.state
        if avoid_kihunter_doors(session, label=label, allow_right=False):
            continue
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT" if st.samus_x < lo else "LEFT",
                run_frames=4,
                spin_frames=14,
                label=label,
            )
            continue
        if (
            lo <= st.samus_x <= hi
            and st.samus_y <= 180
            and st.velocity_y == 0
            and not is_knockback(st)
        ):
            hold(session, 6, reason=f"{label}_settle")
            hold(session, 4, "RIGHT", reason=f"{label}_face")
            hold(session, 4, reason=f"{label}_face_settle")
            return session.state
        if st.samus_x < lo:
            hold(session, 1, "RIGHT", reason=f"{label}_too_left")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_left")
    raise TimeoutError(f"{label}: never reached left runway: {session.state}")


def play_kihunter_charge_store(session: ControllerSession) -> SuperMetroidState:
    """Ground-charge Speed Booster (RIGHT+B dash) and crouch-store."""
    require_room(session, ROOM_KIHUNTER, "kihunter_charge")
    label = "kihunter_charge"
    for frame in range(_CHARGE_BUDGET):
        st = session.state
        if st.room_id != ROOM_KIHUNTER:
            return st
        if is_knockback(st):
            # Mid-room x≈555 trap: idle clear then step left and re-run.
            hold(session, 20, reason=f"{label}_kb_idle")
            if is_knockback(session.state):
                escape_knockback_spin(
                    session,
                    prefer_dir="LEFT",
                    run_frames=6,
                    spin_frames=14,
                    label=label,
                    run_with=("B",),
                    spin_with=("B", "A"),
                )
            continue

        # Store only on trench floor (y≥170). Elevated mid-path can hit sc≥4
        # around x≈460 y≈176; hop from there activates too high/low and jams
        # before the Moat door. Verified green band: store ~x503 y178 pose 9.
        if (
            st.speed_boosting
            and st.velocity_y == 0
            and st.pose not in (137, 138)
            and st.samus_y >= 170
        ):
            # Immediate crouch — do not keep holding RIGHT into the 555 trap.
            for _ in range(_STORE_FRAMES):
                hold(session, 1, "DOWN", reason=f"{label}_store")
            return session.state

        x = st.samus_x
        # Hop only the mid collision band; dash is B (not A).
        if 545 <= x <= 575 and st.velocity_y == 0:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_trap_hop")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_run")

    raise TimeoutError(
        f"{label}: no grounded speed store in {_CHARGE_BUDGET}f: {session.state}"
    )


def play_moat_shinespark(
    session: ControllerSession,
    *,
    skip_clear: bool = False,
) -> SuperMetroidState:
    """Clear → charge-store → hop-unspin-activate → Moat → West Ocean.

    Store-first trench charge, then micro-run + spin hop over the x555 wall,
    UP mid-air (often arms pose 199), horizontal spark into Moat, then after
    spark dies at the ~x475 wall walk RIGHT+X to open the blue door.
    """
    require_room(session, ROOM_KIHUNTER, "moat_spark")
    if not skip_clear:
        play_clear_kihunter_room(session)
    # Left pin (x≲100, already clear) charges in place with pure RIGHT+B
    # (same as probe). Runway walk from mid-room; do not pre-walk without B.
    st0 = session.state
    if not (st0.samus_x < 100 and st0.samus_y <= 180 and st0.velocity_y == 0):
        play_kihunter_left_runway(session)
    play_kihunter_charge_store(session)

    label = "moat_spark"
    # Leave crouch-store (pose 39) — idle alone does not stand; need RIGHT+B.
    for _ in range(_STAND_FRAMES):
        hold(session, 1, reason=f"{label}_stand")
    for _ in range(_MICRO_RUN):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_micro_run")
    # Spin over wall (continuous A while spinning does not early-activate
    # once standing pose 9; from crouch it would shine into the wall).
    for _ in range(_HOP_FRAMES):
        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hop")
    for _ in range(_UNSPIN_FRAMES):
        hold(session, 1, "UP", reason=f"{label}_unspin")

    # Activate / travel (UP may already have started windup pose 199).
    for _ in range(_SPARK_ACTIVATE):
        hold(session, 1, "RIGHT", "A", reason=f"{label}_activate")
        st = session.state
        if st.pose in _SPARK_POSES or st.room_id in (ROOM_MOAT, ROOM_WEST_OCEAN):
            break

    for frame in range(_SPARK_TRAVEL):
        st = session.state
        if st.room_id == ROOM_WEST_OCEAN and st.door_transition == 0 and st.game_state == 8:
            hold(session, 8, reason=f"{label}_west_settle")
            return session.state
        if st.room_id == ROOM_WEST_OCEAN:
            # Finish door transition
            hold(session, 1, "RIGHT", reason=f"{label}_west_trans")
            continue
        if st.room_id == ROOM_MOAT and st.pose not in _SPARK_POSES:
            # Spark died on mid-Moat wall (~x475) — open blue door + walk in.
            if frame % 8 < 5:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_door_open")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_door_walk")
            continue
        hold(session, 1, "RIGHT", "A", reason=f"{label}_travel")

    st = session.state
    if st.room_id == ROOM_WEST_OCEAN:
        return st
    raise TimeoutError(
        f"{label}: did not reach West Ocean 0x93FE "
        f"(room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
        f"spark={st.shinespark_timer} sc={st.speed_counter}): {st}"
    )


def play_moat_cross(session: ControllerSession) -> SuperMetroidState:
    """Entry for pure registry / scaffold: full Kihunter→Moat spark path.

    If already in Moat (legacy callers), attempt a short in-room spark/push.
    """
    st = session.state
    if st.room_id == ROOM_KIHUNTER:
        return play_moat_shinespark(session)
    if st.room_id == ROOM_MOAT:
        require_room(session, ROOM_MOAT, "moat_cross")
        for attempt in range(3):
            hold(session, 80, "RIGHT", "B", reason=f"moat_inroom_charge_{attempt}")
            hold(session, 12, "DOWN", reason=f"moat_inroom_store_{attempt}")
            hold(session, 120, "RIGHT", "A", reason=f"moat_inroom_spark_{attempt}")
            if session.state.room_id == ROOM_WEST_OCEAN:
                return session.state
        raise TimeoutError(f"moat_cross: stuck in Moat: {session.state}")
    raise TimeoutError(
        f"moat_cross: expected Kihunter 0x948C or Moat 0x95FF, got 0x{st.room_id:04X}"
    )


__all__ = [
    "ROOM_KIHUNTER",
    "ROOM_MOAT",
    "ROOM_WEST_OCEAN",
    "air_enemies_alive",
    "avoid_kihunter_doors",
    "list_air_enemies",
    "near_left_door",
    "near_right_door",
    "play_clear_kihunter_room",
    "play_kihunter_charge_store",
    "play_kihunter_left_runway",
    "play_moat_cross",
    "play_moat_shinespark",
]
