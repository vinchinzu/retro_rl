"""Controller-only post-Spore Super collect (and planned PB suffix).

Emits ordinary 12-button controller actions from typed state only. Never loads
emulator state or writes RAM. Development states may seed probes; continuous
acceptance must compose this after the power-on → Spore prefix.

Living route board: ``docs/routes/ROUTE_SUPERS_TO_PHANTOON.md``.

Proven so far from ``natural_post_spore_spawn`` (room ``0x9B5B``):

1. Break into the vertical shaft (crouch-shot + bombs + jump-right).
2. Fall to the bottom and collect Super Missiles (capacity 0 → 5).
3. Bomb through the bottom gate, shoot the left door, enter farming ``0xA0A4``.
4. Jump-run left across farming, Super the green door into Big Pink ``0x9D19``.
5. Crest the farm-pocket lip at x≈1157 via run-right then spin-jump-left
   (lands ~x=1125, y=1387 standing on the raised platform).
6. **Segment 3b:** crouch-Super clears permanent Super-only shot block
   **(69, 87)**; **double-tap DOWN** morphs into the y87 tunnel (standing
   y≈1387 → morph y≈1401 — same floor, pose height); morph-roll west + **X**
   bombs open scroll (3,5), clear bomb blocks (62–63, 87), reach main shaft
   x≲750 (``play_big_pink_into_main_shaft``).

Pink PB progress (Big Pink exit → Mission Impossible Room):

- **Bottom door ``0x8E02``:** place-bridge midair ``(580,1136)`` run-shoot-spin →
  alcove ~``(530,1163)`` → bottom spawn ~y=395. Not a hop-to island; y1051 is
  corridor *roof*; east falls hit main min_x=613 (full-height wall).
- **Top door ``0x8DDE`` (preferred):** solid ledge **x≈520–548, y≈907**; entry
  → top spawn ~y=130. **Drop-air:** free-fall from place **x∈[535,555],
  y∈[850,910]** lands the ledge (``dev_b1_pb_top_ledge``). Pure path into that
  air still open: east blocked by wall@613; west (`left_upper`) needs height
  (spin peak Δy≈79 only).
- **Maze wall@437:** double-tap morph + y-safe bomb-roll → ~x=405 (pure).
- **Collect:** left-zone ``play_pink_pb_from_left_zone`` ~(180,360); mid solid
  at band y (x≈230–400); pit y≈455 continuous but dead-end (~2px headroom).
- **Still open:** pure into drop-air/y907; pure mid-maze door→left-volume;
  top floor sealed (bombs/DOWN do not open crumble from standing).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold as _hold

ROOM_SUPER = 0x9B5B
ROOM_FARMING = 0xA0A4
ROOM_BIG_PINK = 0x9D19
ROOM_PINK_PB = 0x9E11

# Morph-ball poses observed on this route (and facing/air/fall variants).
# 29=0x1D, 30=0x1E falling; 31=0x1F, 32=0x20; 49=0x31, 50=0x32;
# 65=0x41, 66=0x42 ground/move. Expand from live logs only.
MORPH_POSES = frozenset({29, 30, 31, 32, 49, 50, 65, 66})


@dataclass(frozen=True)
class SuperCollectEvidence:
    entry_frame: int
    collect_frame: int
    exit_frame: int | None
    max_super_missiles: int
    final_room_id: int
    samus_x: int
    samus_y: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PowerBombEvidence:
    entry_frame: int
    collect_frame: int | None
    max_super_missiles: int
    max_power_bombs: int
    final_room_id: int
    samus_x: int
    samus_y: int
    reached_big_pink: bool
    reached_pb_room: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_room(session: ControllerSession, room_id: int, label: str) -> None:
    state = session.state
    if state.room_id != room_id:
        raise RuntimeError(
            f"{label}: expected room 0x{room_id:04X}, got 0x{state.room_id:04X} "
            f"at frame {session.frame}"
        )


def _select_weapon(session: ControllerSession, target: int, *, max_cycles: int = 8) -> None:
    """Cycle SELECT until ``selected_item`` matches ``target`` (0–3)."""
    for _ in range(max_cycles):
        if session.state.selected_item == target:
            return
        _hold(session, 1, "SELECT", reason="select_weapon")
        _hold(session, 25, reason="select_weapon_settle")
    if session.state.selected_item != target:
        raise RuntimeError(
            f"could not select weapon {target}, still {session.state.selected_item}"
        )


def _unmorph(session: ControllerSession) -> None:
    pose = session.state.pose
    if pose in (39, 40, 137, 138, 9, 10) or is_morph(pose):
        _hold(session, 8, "UP", reason="unmorph")
        if not is_morph(session.state.pose):
            _hold(session, 8, "A", reason="unmorph")
        _hold(session, 10, reason="unmorph_settle")


def is_morph(pose: int) -> bool:
    """True when Samus is in morph/spring-ball pose (any facing)."""
    return pose in MORPH_POSES


def wait_until(
    session: ControllerSession,
    pred: Callable[[SuperMetroidState], bool],
    *,
    timeout: int = 120,
    reason: str = "wait",
) -> SuperMetroidState:
    """Idle one frame at a time until ``pred(state)`` or raise ``TimeoutError``."""
    for _ in range(timeout):
        if pred(session.state):
            return session.state
        _hold(session, 1, reason=reason)
    raise TimeoutError(f"{reason} timed out: {session.state}")


def ensure_morph(
    session: ControllerSession,
    *,
    max_attempts: int = 5,
) -> SuperMetroidState:
    """Pose-confirmed morph via double-tap DOWN (held DOWN only crouches).

    Each attempt: brief UP to leave crouch, idle, tap–release–tap DOWN, then
    poll for a morph pose. Later attempts hold DOWN longer and re-UP before
    retry. Replaces fixed ``_double_tap_morph`` call sites.
    """
    for attempt in range(max_attempts):
        if is_morph(session.state.pose):
            return session.state
        # Extra stand/idle on later attempts (leave crouch / super pose).
        up_frames = 4 + min(attempt, 3) * 2
        _hold(session, up_frames, "UP", reason="morph_pre")
        _hold(session, 3 + attempt, reason="morph_idle")
        tap1 = 5 + attempt * 2
        tap2 = 6 + attempt * 3
        _hold(session, tap1, "DOWN", reason="morph_tap1")
        _hold(session, 3, reason="morph_release")
        _hold(session, tap2, "DOWN", reason="morph_tap2")
        try:
            state = wait_until(
                session,
                lambda s: is_morph(s.pose),
                timeout=28 + attempt * 4,
                reason="morph_poll",
            )
        except TimeoutError:
            continue
        if is_morph(state.pose):
            return state
    raise TimeoutError(f"ensure_morph failed, pose={session.state.pose}")


def bomb_roll_left_safe(
    session: ControllerSession,
    target_x: int,
    *,
    max_y: int = 415,
    pit_y: int = 430,
    max_frames: int = 280,
    cycle_len: int = 38,
    elev_y: int = 400,
    log_every: int = 0,
    stall_frames: int = 0,
) -> SuperMetroidState:
    """Morph-bomb-roll left toward ``target_x`` with y-band / pit recovery.

    Uses the proven wall-open cycle: lay bomb (X), wait for bomb-jump height,
    then roll left only while on the upper band and rising/flat. Prefer short
    LEFT while ``y <= max_y`` and not falling hard.

    Pit recovery (``y > pit_y`` or hard downward velocity): force unmorph →
    short RIGHT/UP hop back toward the door ledge (not deeper left into the
    pit) → re-``ensure_morph``. Deep pit (``y > 445``) has ~2px morph
    headroom — recovery is best-effort; mid x≈230–400 is solid at band y.

    Progress watchdog (``stall_frames > 0`` only): if x has not decreased for
    that many frames, force a bomb + short pause. Leave at 0 for wall-open
    (x stalls against the block by design). Optional ``log_every`` prints
    ``x,y,pose,vel_y,is_morph`` every N frames. Early-exits when PB capacity
    appears.
    """
    frames = 0
    last_progress_x = session.state.samus_x
    frames_since_progress = 0
    deep_pit_y = 445
    pit_recoveries = 0

    def _log(tag: str) -> None:
        if log_every <= 0:
            return
        if frames % log_every != 0 and tag == "tick":
            return
        s = session.state
        print(
            f"[bomb_roll {tag} f={frames}] "
            f"x={s.samus_x} y={s.samus_y} pose={s.pose} "
            f"vy={s.velocity_y} vx={s.velocity_x} morph={is_morph(s.pose)} "
            f"pb={s.max_power_bombs}",
            flush=True,
        )

    while session.state.samus_x > target_x and frames < max_frames:
        s = session.state
        if s.max_power_bombs > 0:
            return s
        _log("tick")

        falling_hard = s.velocity_y > 80
        in_pit = s.samus_y > pit_y or falling_hard
        deep_pit = s.samus_y > deep_pit_y

        if in_pit:
            pit_recoveries += 1
            # Deep pit: morph headroom ~2px — unmorph usually fails. Nudge
            # RIGHT toward door ledge first; avoid re-rolling deeper left.
            if deep_pit:
                _hold(session, 4, "UP", reason="pit_unmorph")
                for _ in range(18):
                    _hold(session, 1, "RIGHT", reason="pit_right")
                    if session.state.samus_y <= pit_y:
                        break
                # Brief hop if we got any headroom.
                if not is_morph(session.state.pose) or session.state.samus_y <= pit_y + 5:
                    _hold(session, 6, "A", "RIGHT", reason="pit_jump")
                    _hold(session, 8, reason="pit_settle")
                try:
                    if not is_morph(session.state.pose):
                        ensure_morph(session)
                except TimeoutError:
                    pass
                frames += 36
            else:
                # Shallow fall: unmorph, hop back onto band (prefer right).
                _hold(session, 6, "UP", reason="pit_unmorph")
                _hold(session, 8, "A", "RIGHT", reason="pit_jump")
                _hold(session, 10, reason="pit_settle")
                try:
                    ensure_morph(session)
                except TimeoutError:
                    pass
                frames += 24
            # Bail if stuck in deep pit with no x progress (geometry trap).
            if deep_pit and pit_recoveries >= 3 and session.state.samus_x >= last_progress_x - 2:
                _log("deep_pit_stuck")
                return session.state
            continue

        if not is_morph(session.state.pose):
            ensure_morph(session)
            frames += 20
            continue

        # Progress watchdog (opt-in): no leftward progress → bomb + pause.
        # Disabled when stall_frames==0 (wall-open needs multi-cycle stalls).
        if stall_frames > 0 and frames_since_progress >= stall_frames:
            _hold(session, 2, "X", reason="safe_watchdog_bomb")
            _hold(session, 10, reason="safe_watchdog_pause")
            frames += 12
            frames_since_progress = 0
            _log("watchdog")
            continue

        # Bomb-jump cycle (matches proven wall@437 open timing).
        _hold(session, 2, "X", reason="safe_bomb")
        frames += 2
        for step in range(cycle_len):
            if frames >= max_frames:
                break
            s = session.state
            if s.max_power_bombs > 0:
                return s
            if s.samus_x <= target_x and s.samus_y <= max_y + 5:
                return s
            if s.samus_y > pit_y or s.velocity_y > 80:
                break  # outer loop recovers
            if not is_morph(s.pose):
                break

            # Prefer left while elevated by bomb jump or late in cycle on band.
            # Never roll left once past max_y toward the pit (band-keeping).
            if s.samus_y < elev_y or (s.samus_y <= max_y and step > cycle_len // 2):
                _hold(session, 1, "LEFT", reason="safe_roll")
            elif s.samus_y <= max_y + 8:
                # On band but early cycle: short wait so bomb can pop.
                if step < 8:
                    _hold(session, 1, reason="safe_bomb_wait")
                else:
                    _hold(session, 1, "LEFT", reason="safe_roll")
            else:
                _hold(session, 1, reason="safe_band_wait")
            frames += 1
            frames_since_progress += 1

        if session.state.samus_x < last_progress_x - 2:
            last_progress_x = session.state.samus_x
            frames_since_progress = 0
        else:
            # Stalled a full cycle against a barrier — pause then re-bomb.
            _hold(session, 6, reason="safe_stall_pause")
            frames += 6
            frames_since_progress += 6
    _log("done")
    return session.state


def play_super_room_collect(session: ControllerSession) -> SuperCollectEvidence:
    """From natural Super-room entry, descend and collect Super Missiles.

    Entry expectation: ordinary gameplay in ``0x9B5B``, ``max_super_missiles==0``.
    """
    entry_frame = session.frame
    state = session.state
    if state.room_id != ROOM_SUPER:
        raise RuntimeError(
            f"Super collect entry: room 0x{state.room_id:04X} != 0x{ROOM_SUPER:04X}"
        )
    if state.max_super_missiles > 0:
        raise RuntimeError("Super collect entry: supers already collected")

    # Walk to shaft entrance.
    for _ in range(80):
        state = _hold(session, 1, "RIGHT", "B", reason="super_shaft_approach")
        if state.samus_x >= 140:
            break

    # Crouch-shot floor blocks.
    for _ in range(15):
        _hold(session, 3, "DOWN", "X", reason="super_shaft_shot")
        _hold(session, 2, "DOWN", reason="super_shaft_shot")
    _hold(session, 10, "DOWN", reason="super_shaft_morph")
    for _ in range(8):
        _hold(session, 2, "A", reason="super_shaft_bomb")
        _hold(session, 30, reason="super_shaft_bomb_wait")

    # Jump right into the cleared shaft path.
    for _ in range(50):
        state = _hold(session, 2, "RIGHT", "A", "B", reason="super_shaft_jump")
        if state.samus_x > 250:
            break

    # Explore right/down until free-fall begins.
    for i in range(200):
        phase = i % 10
        if phase < 3:
            state = _hold(session, 4, "RIGHT", "B", "X", reason="super_shaft_explore")
        elif phase < 5:
            state = _hold(session, 4, "RIGHT", "A", "B", reason="super_shaft_explore")
        elif phase < 7:
            state = _hold(session, 3, "DOWN", "X", reason="super_shaft_explore")
        elif phase < 8:
            _hold(session, 8, "DOWN", reason="super_shaft_explore")
            _hold(session, 2, "A", reason="super_shaft_explore")
            state = _hold(session, 20, reason="super_shaft_explore")
        else:
            state = _hold(session, 4, "RIGHT", "B", reason="super_shaft_explore")
        if state.samus_y > 500:
            break

    for _ in range(800):
        state = _hold(session, 2, reason="super_shaft_fall")
        if state.samus_y > 2100:
            break
    if state.samus_y <= 2000:
        raise TimeoutError(f"Super shaft fall failed: {state}")

    # Approach Chozo Super and collect.
    collect_frame: int | None = None
    for i in range(400):
        if state.samus_x < 412:
            state = _hold(session, 2, "RIGHT", "B", reason="super_item_approach")
        elif state.samus_x > 428:
            state = _hold(session, 2, "LEFT", "B", reason="super_item_approach")
        else:
            state = _hold(session, 2, reason="super_item_approach")
        if i % 12 == 0:
            state = _hold(session, 4, "X", reason="super_item_shoot")
        if i % 40 == 20:
            state = _hold(session, 6, "A", reason="super_item_jump")
        if state.max_super_missiles > 0:
            collect_frame = session.frame
            break
    if collect_frame is None or state.max_super_missiles <= 0:
        raise TimeoutError(f"Super Missile PLM never collected: {state}")

    # Fanfare / control return.
    for i in range(300):
        state = _hold(session, 1, reason="super_item_fanfare")
        if (
            state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.max_super_missiles > 0
            and i > 80
        ):
            break

    return SuperCollectEvidence(
        entry_frame=entry_frame,
        collect_frame=collect_frame,
        exit_frame=None,
        max_super_missiles=state.max_super_missiles,
        final_room_id=state.room_id,
        samus_x=state.samus_x,
        samus_y=state.samus_y,
    )


def play_super_room_to_farming(session: ControllerSession) -> SuperMetroidState:
    """After Super collect at bottom, bomb left gate and enter farming."""
    state = session.state
    _require_room(session, ROOM_SUPER, "to_farming")
    if state.max_super_missiles <= 0:
        raise RuntimeError("to_farming requires Super Missiles")

    # Prefer missiles for blue door shot; SELECT may be flaky mid-pose.
    try:
        _select_weapon(session, 1)
    except RuntimeError:
        pass

    for _ in range(80):
        state = _hold(session, 2, "LEFT", "B", reason="super_gate_approach")
        if state.samus_x <= 320:
            break

    _hold(session, 12, "DOWN", reason="super_gate_bomb")
    for _ in range(15):
        _hold(session, 2, "A", reason="super_gate_bomb")
        _hold(session, 35, reason="super_gate_bomb")
        state = _hold(session, 4, "LEFT", reason="super_gate_bomb")
        if state.samus_x < 200:
            break
    _unmorph(session)

    for _ in range(120):
        state = _hold(session, 2, "LEFT", "B", reason="super_door_approach")
        if state.samus_x < 50:
            break

    for _ in range(50):
        _hold(session, 3, "LEFT", "X", reason="super_door_shot")
        state = _hold(session, 5, "LEFT", "B", reason="super_door_enter")
        if state.room_id == ROOM_FARMING:
            break
    for _ in range(200):
        state = _hold(session, 1, reason="farming_settle")
        if (
            state.room_id == ROOM_FARMING
            and state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ):
            break
    _require_room(session, ROOM_FARMING, "farming entry")
    return state


def play_farming_to_big_pink(session: ControllerSession) -> SuperMetroidState:
    """Cross farming left and Super the green door into Big Pink."""
    state = session.state
    _require_room(session, ROOM_FARMING, "farming_to_pink")
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass
    _unmorph(session)

    for i in range(500):
        if session.state.pose in (39, 40, 137, 138):
            _unmorph(session)
        state = _hold(session, 3, "LEFT", "A", "B", reason="farming_cross")
        if i % 5 == 0:
            _hold(session, 2, "LEFT", "X", reason="farming_super")
        if i % 25 == 12:
            _hold(session, 8, "DOWN", reason="farming_bomb")
            _hold(session, 2, "A", reason="farming_bomb")
            _hold(session, 30, reason="farming_bomb")
            _unmorph(session)
        if state.room_id == ROOM_BIG_PINK:
            break
    for _ in range(150):
        state = _hold(session, 1, reason="big_pink_settle")
        if (
            state.room_id == ROOM_BIG_PINK
            and state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ):
            break
    _require_room(session, ROOM_BIG_PINK, "big pink entry")
    return state


def play_big_pink_crest_pocket(session: ControllerSession) -> SuperMetroidState:
    """Crest the farm-pocket lip at x≈1157 (run-right, spin-jump left).

    Expects ordinary Big Pink near the farm door (~x≥1180). On success Samus
    is roughly at **(1125, 1387)** standing on the raised platform (same floor
    as morph y≈1401 — pose height). Double-tap DOWN morphs into the y87 tunnel.
    Raises ``TimeoutError`` if the lip is not crossed.
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "crest_pocket")
    _unmorph(session)
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass

    # Walk into the pocket wall, then reverse for run speed.
    for _ in range(40):
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_pocket_approach")
        if state.samus_x <= 1160:
            break
    _hold(session, 25, "RIGHT", "B", reason="big_pink_pocket_runup")

    for _ in range(40):
        state = _hold(session, 12, "LEFT", "A", "B", reason="big_pink_pocket_crest")
        state = _hold(session, 6, "LEFT", "B", reason="big_pink_pocket_crest")
        if state.samus_x <= 1135:
            break
    if state.samus_x > 1135:
        raise TimeoutError(
            f"Big Pink pocket crest failed (still x={state.samus_x}, y={state.samus_y})"
        )
    _hold(session, 12, reason="big_pink_pocket_crest_settle")
    return session.state


def play_big_pink_clear_super_block(session: ControllerSession) -> SuperMetroidState:
    """Clear the permanent Super-only shot block at tile (69, 87) from crest.

    The second barrier is level type ``0xC`` / BTS ``0x0B`` (Super Missile only,
    permanent). Standing/angle supers from the wall-top often miss; crouch +
    Super left from the crest ledge clears it in one shot.

    Expects Big Pink after ``play_big_pink_crest_pocket`` (~x≤1135, y≈1387).
    Does **not** by itself reach the open main shaft — Samus must still enter
    the raised morph tunnel (~y=1401) and clear bomb blocks (62–63, 87).
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "clear_super_block")
    # Unspin from crest land (pose often 0x8A ran-into-wall).
    _hold(session, 4, "A", reason="big_pink_unspin")
    _hold(session, 40, reason="big_pink_unspin_settle")
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass
    _hold(session, 15, "DOWN", reason="big_pink_crouch_super")
    for _ in range(8):
        _hold(session, 3, "LEFT", "X", reason="big_pink_crouch_super")
        state = _hold(session, 18, "LEFT", "DOWN", reason="big_pink_crouch_super")
    _hold(session, 10, reason="big_pink_super_block_settle")
    return session.state


def play_big_pink_morph_to_tunnel(session: ControllerSession) -> SuperMetroidState:
    """Double-tap DOWN to morph into the y87 tunnel on the raised platform.

    Standing center is ~y=1387 on the same floor as morph ~y=1401 (pose height).
    Holding DOWN only crouches (pose 40) and cannot enter the 1-tile tunnel;
    a short tap–release–tap completes morph ball (pose 65) so Samus fits under
    the y86 ceiling and can roll west.

    Expects Big Pink after Super block clear, roughly x≤1140 on the platform.
    Raises ``TimeoutError`` if morph tunnel height is not reached.
    """
    _require_room(session, ROOM_BIG_PINK, "morph_to_tunnel")
    # Leave crouch/super pose if still holding down from the Super clear.
    _hold(session, 12, reason="big_pink_morph_stand")
    # Double-tap DOWN: crouch then morph (hold-DOWN alone stays crouch).
    _hold(session, 1, "DOWN", reason="big_pink_morph_tap1")
    _hold(session, 4, reason="big_pink_morph_gap")
    _hold(session, 1, "DOWN", reason="big_pink_morph_tap2")
    state = _hold(session, 18, "DOWN", reason="big_pink_morph_hold")
    on_tunnel = 1395 <= state.samus_y <= 1410 and state.samus_x <= 1155
    if not on_tunnel:
        # One retry with a slightly longer second tap.
        _hold(session, 8, reason="big_pink_morph_retry_stand")
        _hold(session, 2, "DOWN", reason="big_pink_morph_retry_tap1")
        _hold(session, 4, reason="big_pink_morph_retry_gap")
        _hold(session, 8, "DOWN", reason="big_pink_morph_retry_tap2")
        state = _hold(session, 20, "DOWN", reason="big_pink_morph_retry_hold")
        on_tunnel = 1395 <= state.samus_y <= 1410 and state.samus_x <= 1155
    if not on_tunnel:
        raise TimeoutError(
            "Big Pink morph-to-tunnel failed: "
            f"({state.samus_x}, {state.samus_y}) pose={state.pose}; "
            "expected morph on raised floor y≈1401 after double-tap DOWN"
        )
    return session.state


def play_big_pink_tunnel_west(
    session: ControllerSession,
    *,
    target_x: int = 750,
    max_frames: int = 400,
) -> SuperMetroidState:
    """Morph-roll west through the y87 tunnel into the open main shaft.

    Expects Big Pink on the **raised tunnel floor** (~y 1395–1410, x≲1140) with
    the Super shot block at (69, 87) already cleared. Sequence:

    1. Morph and roll left (opens scroll PLM at (64, 87) → screen (3,5)).
    2. Lay morph bombs with **X** (not A) on permanent bomb blocks (62–63, 87).
    3. Continue west into open main-shaft volume (default x≤``target_x``).

    Proven from crest after Super clear + ``play_big_pink_morph_to_tunnel``.
    Raises ``TimeoutError`` if the target is not reached.
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "tunnel_west")
    try:
        _select_weapon(session, 0)
    except RuntimeError:
        pass
    # Already morphed after morph_to_tunnel; short DOWN keeps ball if needed.
    _hold(session, 8, "DOWN", reason="big_pink_tunnel_morph")

    for i in range(max_frames):
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_tunnel_roll")
        if i % 18 == 5:
            # Morph bombs: X (A is unreliable here against these BTS-4 blocks).
            _hold(session, 2, "X", reason="big_pink_tunnel_bomb")
            _hold(session, 50, reason="big_pink_tunnel_bomb_wait")
        if state.room_id != ROOM_BIG_PINK:
            raise RuntimeError(
                f"tunnel west left Big Pink at frame {session.frame}: {state}"
            )
        if state.samus_x <= target_x:
            _hold(session, 10, reason="big_pink_tunnel_west_settle")
            return session.state

    raise TimeoutError(
        f"Big Pink tunnel west failed: x={state.samus_x}, y={state.samus_y} "
        f"(target x≤{target_x})"
    )


def play_big_pink_drop_to_pocket(session: ControllerSession) -> SuperMetroidState:
    """From crest wall-top, walk off east into the deep farm pocket floor.

    Lands roughly x≥1157, y≈1419. Used before bomb-jump attempts onto the
    raised tunnel ledge.
    """
    _require_room(session, ROOM_BIG_PINK, "drop_to_pocket")
    _hold(session, 10, reason="big_pink_pocket_drop_settle")
    _hold(session, 4, "A", reason="big_pink_pocket_drop_unspin")
    _hold(session, 25, reason="big_pink_pocket_drop_unspin")
    _hold(session, 40, "RIGHT", "B", reason="big_pink_pocket_drop")
    _hold(session, 50, reason="big_pink_pocket_drop_land")
    _unmorph(session)
    return session.state


def play_big_pink_bomb_to_walkway_edge(
    session: ControllerSession,
    *,
    fuse_frames: int = 15,
    jump_frames: int = 14,
) -> SuperMetroidState:
    """From deep pocket lip, morph-bomb jump onto the floating walkway edge.

    Lands near **(1151, 1387)** — the east edge of the wall-top walkway above
    the raised tunnel floor. Does **not** yet drop the last ~9px onto tunnel
    floor y≈1401 (that hop remains open: morph-right falls past x=1152 into
    the deep pocket).

    Expects Super block already cleared and Samus near the lip (~x 1160–1210).
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "bomb_to_edge")
    # Approach lip if still east.
    for _ in range(60):
        if state.samus_x <= 1168:
            break
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_edge_approach")
    try:
        _select_weapon(session, 0)
    except RuntimeError:
        pass
    _hold(session, 12, "DOWN", reason="big_pink_edge_morph")
    _hold(session, 3, "LEFT", reason="big_pink_edge_press")
    _hold(session, 2, "X", reason="big_pink_edge_bomb")
    _hold(session, fuse_frames, "LEFT", reason="big_pink_edge_fuse")
    _hold(session, jump_frames, "LEFT", "A", reason="big_pink_edge_boost")
    # Idle fall/land on walkway edge (no left — left pulls back to crest).
    for _ in range(50):
        state = _hold(session, 1, reason="big_pink_edge_land")
    _hold(session, 10, reason="big_pink_edge_settle")
    return session.state


def play_big_pink_into_main_shaft(
    session: ControllerSession,
    *,
    target_x: int = 750,
) -> SuperMetroidState:
    """Crest → Super clear → double-tap morph → tunnel west into main shaft.

    Fully controller (no place/WRAM). Standing y≈1387 and morph y≈1401 are the
    same raised floor (pose height); the former “hop” was a morph-input issue.
    """
    play_big_pink_crest_pocket(session)
    play_big_pink_clear_super_block(session)
    play_big_pink_morph_to_tunnel(session)
    return play_big_pink_tunnel_west(session, target_x=target_x)


def play_big_pink_enter_pb_door_from_sill(
    session: ControllerSession,
    *,
    settle_frames: int = 200,
) -> SuperMetroidState:
    """Enter Pink PB room ``0x9E11`` from a Big Pink PB door ledge.

    Proven controller sequence (no place/WRAM once on the door ledge):

    1. Run left + shoot to open the blue door.
    2. Spin-jump left through the door alcove.
    3. Hold left until the room transition (~45–100 frames).
    4. Idle until ordinary gameplay (game state 8, no door transition).

    Works from either door once Samus is on the ledge:

    - **Top door** ``0x8DDE`` (preferred): solid ledge **x≈520–548, y≈907**
      → spawn ~y=130 (Mission Impossible top / crumble path).
    - **Bottom door** ``0x8E02``: place-bridge midair ~``(580,1136)`` or alcove
      ~``(530,1163)`` → spawn ~y=395. Not a hop-to island from main/upper
      (wall@613 full height; y1051 ledge is corridor roof).

    Pure climb onto either ledge from main shaft is still open.
    """
    _require_room(session, ROOM_BIG_PINK, "enter_pb_door_from_sill")
    state = session.state
    # Run-shoot opens blue door; spin carries through alcove.
    _hold(session, 10, "LEFT", "B", reason="pb_door_run")
    _hold(session, 4, "LEFT", "B", "X", reason="pb_door_shoot")
    _hold(session, 30, "LEFT", "B", "A", reason="pb_door_spin")
    entered = False
    for _ in range(120):
        state = _hold(session, 1, "LEFT", reason="pb_door_hold")
        if state.room_id == ROOM_PINK_PB:
            entered = True
            break
    if not entered:
        raise TimeoutError(
            f"pb_door_from_sill: did not reach 0x{ROOM_PINK_PB:04X}: {session.state}"
        )
    # Wait for multi-frame door load to ordinary gameplay.
    for frame in range(settle_frames):
        state = _hold(session, 1, reason="pb_door_settle")
        if (
            state.room_id == ROOM_PINK_PB
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 20
        ):
            break
    if state.room_id != ROOM_PINK_PB:
        raise RuntimeError(
            f"pb_door_settle: left 0x{ROOM_PINK_PB:04X}: {session.state}"
        )
    return state


def play_big_pink_enter_pb_door_from_top_ledge(
    session: ControllerSession,
    *,
    settle_frames: int = 200,
) -> SuperMetroidState:
    """Enter Pink PB via top door ledge (~532, 907). Alias of sill entry.

    Expects Big Pink on the solid top-door ledge (x≈520–548, y≈900–920).
    Lands ``0x9E11`` top spawn (~y=130). Prefer this over bottom place-bridge
    for the Mission Impossible crumble/collect path.
    """
    return play_big_pink_enter_pb_door_from_sill(
        session, settle_frames=settle_frames
    )


def _double_tap_morph(session: ControllerSession) -> SuperMetroidState:
    """Deprecated alias for ``ensure_morph`` (pose-confirmed double-tap)."""
    return ensure_morph(session)


def play_pink_pb_break_maze_wall(
    session: ControllerSession,
    *,
    max_frames: int = 640,
) -> SuperMetroidState:
    """Open bottom-spawn maze wall at x≈437 with morph bombs.

    From natural bottom-door spawn ~(460, 395): walk left into the wall,
    ``ensure_morph``, y-safe bomb-roll past the wall (~x=405, upper band
    y≈395–410). Missiles/supers do **not** open this wall; crouch-bombs do
    not either — morph is required.
    """
    _require_room(session, ROOM_PINK_PB, "pink_pb_break_maze_wall")
    if session.state.max_power_bombs > 0:
        return session.state
    if session.state.samus_x <= 410 and session.state.samus_y <= 415:
        return session.state
    # Approach wall.
    for _ in range(40):
        if session.state.samus_x <= 440:
            break
        _hold(session, 1, "LEFT", reason="pb_maze_to_wall")
    ensure_morph(session)
    state = bomb_roll_left_safe(
        session,
        410,
        max_y=412,
        pit_y=430,
        max_frames=max_frames,
    )
    if state.max_power_bombs > 0:
        return state
    if state.samus_x > 420:
        raise TimeoutError(
            f"pink_pb_break_maze_wall: still blocked: {session.state}"
        )
    return state


def play_pink_pb_morph_bomb_collect(
    session: ControllerSession,
    *,
    max_frames: int = 400,
) -> SuperMetroidState:
    """Morph-bomb-roll left from collect pocket (x≤225, y≈395) to the PB PLM.

    Proven from place ``(220–225, 395)`` after bottom-door entry: ensure morph,
    bomb-roll left; capacity 0→5 near the item (~x=100–120, y≈370–400).
    Unmorph+walk fallback if we overshoot the PLM in ball form.

    Expects Pink PB room ``0x9E11`` already west of mid-maze barriers (not
    merely past wall@437 — that only reaches ~x=405). Use
    ``play_pink_pb_mid_maze_to_collect`` when starting near x≈405.
    """
    _require_room(session, ROOM_PINK_PB, "pink_pb_morph_collect")
    if session.state.max_power_bombs > 0:
        return session.state
    ensure_morph(session)
    state = bomb_roll_left_safe(
        session,
        100,
        max_y=415,
        pit_y=430,
        max_frames=max_frames,
    )
    if state.max_power_bombs > 0:
        return state
    # Unmorph + walk fallback near PLM (item may need standing contact).
    if state.samus_x < 160:
        _hold(session, 5, "UP", reason="pb_collect_unmorph")
        _hold(session, 10, reason="pb_collect_unmorph")
        for _ in range(100):
            state = _hold(session, 1, "LEFT", reason="pb_collect_walk")
            if state.max_power_bombs > 0:
                return state
        for _ in range(40):
            state = _hold(session, 1, "RIGHT", reason="pb_collect_walk_back")
            if state.max_power_bombs > 0:
                return state
    if session.state.max_power_bombs <= 0:
        raise TimeoutError(
            f"pink_pb_morph_collect: still 0 PB capacity: {session.state}"
        )
    return session.state


def play_pink_pb_from_left_zone(
    session: ControllerSession,
) -> SuperMetroidState:
    """Left free volume (~x≤220, y≈310–380) → drop into pocket → collect.

    Proven from place ``(180, 360)`` after wall@437 open: walk/fall into the
    collect band (y≳385, x≲220) then ``play_pink_pb_morph_bomb_collect``.
    Pure entry into this left volume from the bottom door is still open.
    """
    _require_room(session, ROOM_PINK_PB, "from_left_zone")
    if session.state.max_power_bombs > 0:
        return session.state
    # Drop / walk into pocket band if still elevated in left volume.
    if session.state.samus_y < 385 and session.state.samus_x <= 230:
        for i in range(80):
            d = "LEFT" if (i // 8) % 2 == 0 else "RIGHT"
            state = _hold(session, 1, d, reason="pb_leftzone_drop")
            if state.samus_y >= 385 or state.max_power_bombs > 0:
                break
            if state.samus_x > 230:
                _hold(session, 5, "LEFT", reason="pb_leftzone_back")
    if session.state.max_power_bombs > 0:
        return session.state
    return play_pink_pb_morph_bomb_collect(session)


def play_pink_pb_mid_maze_to_collect(
    session: ControllerSession,
    *,
    max_frames: int = 500,
    log_every: int = 0,
) -> SuperMetroidState:
    """After wall break (~408,398) → collect pocket without place (OPEN).

    Walkthrough / map notes (wiki.supermetroid.run “Mission Impossible Room”,
    room map ``PinkBrinstarPowerBombRoom.png``):

    - Room is two-tier: pink upper (top door, sidehoppers) + metal lower maze
      (bottom door, item). 100% often Quick-Drops a **crumble** from above.
    - After wall@437 open, continuous morph-roll sampling shows **no mid
      bridge** at band y: door-side ledge ~x=412 and left volume x≲228 are
      rollable; mid x≈230–400 is solid. Pit y≈455 is continuous x=90–420 but
      morph headroom ~2px (cannot unmorph/climb to item). Top corridor y≈171
      spans full x but is sealed from below (bombs do not open the floor).
    - **Working suffix:** once in left volume (~180,360), walk/fall into
      pocket and collect (``play_pink_pb_from_left_zone``).
    - **Still open:** pure door-side → left volume (or pure top → crumble).

    Tries y-safe bomb-roll (strict band-keeping + pit recovery toward the
    door ledge) then left-zone collect; times out with geometry notes if
    still east of the left volume or stuck in the pit.
    """
    _require_room(session, ROOM_PINK_PB, "mid_maze")
    if session.state.max_power_bombs > 0:
        return session.state
    # Already in left volume or pocket (not deep pit).
    if session.state.samus_x <= 230 and session.state.samus_y <= 420:
        if session.state.samus_y < 385 and session.state.samus_x <= 220:
            return play_pink_pb_from_left_zone(session)
        return play_pink_pb_morph_bomb_collect(session)
    ensure_morph(session)
    start_x = session.state.samus_x
    start_y = session.state.samus_y
    bomb_roll_left_safe(
        session,
        225,
        max_y=412,
        pit_y=420,
        max_frames=max_frames,
        elev_y=400,
        log_every=log_every,
        stall_frames=50,
    )
    s = session.state
    if s.max_power_bombs > 0:
        return s
    if s.samus_x <= 230 and s.samus_y <= 420:
        return play_pink_pb_from_left_zone(session)
    # Stuck in pit after transit: still not collectable (no climb-out).
    pit_note = ""
    if s.samus_y > 440:
        pit_note = (
            " deep-pit trap y≈457 (rollable under mid but ~2px headroom — "
            "no climb to item band y≈360–395);"
        )
    raise TimeoutError(
        f"pink_pb_mid_maze: no pure path yet "
        f"(start=({start_x},{start_y}) → x={s.samus_x} y={s.samus_y} "
        f"pose={s.pose});{pit_note} "
        f"mid solid at band — need door→left-volume or top→crumble "
        f"(see Mission Impossible Room)"
    )


def play_post_spore_supers(
    session: ControllerSession,
    *,
    continue_to_farming: bool = True,
    continue_to_big_pink: bool = True,
    continue_to_crest: bool = False,
) -> SuperCollectEvidence:
    """Collect Supers and optionally reach Big Pink / pocket crest."""
    evidence = play_super_room_collect(session)
    exit_frame = None
    if continue_to_farming:
        play_super_room_to_farming(session)
        if continue_to_big_pink:
            play_farming_to_big_pink(session)
            if continue_to_crest:
                play_big_pink_crest_pocket(session)
        exit_frame = session.frame
        state = session.state
        return SuperCollectEvidence(
            entry_frame=evidence.entry_frame,
            collect_frame=evidence.collect_frame,
            exit_frame=exit_frame,
            max_super_missiles=state.max_super_missiles,
            final_room_id=state.room_id,
            samus_x=state.samus_x,
            samus_y=state.samus_y,
        )
    return evidence

