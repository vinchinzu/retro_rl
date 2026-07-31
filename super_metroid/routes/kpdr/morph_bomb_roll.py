"""Morph bomb-roll left with y-band / pit recovery.

Morph bomb-roll helper for KPDR Pink Brinstar segments. Uses a small phase enum so pit
recovery, band rolling, and stall watchdog are not nested ad-hoc branches.
"""

from __future__ import annotations

from enum import Enum, auto

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold


class MorphBombRollPhase(Enum):
    ON_BAND = auto()
    IN_PIT = auto()
    RECOVERING = auto()
    STALLED = auto()
    DONE = auto()


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
    phase = MorphBombRollPhase.ON_BAND

    def _log(tag: str) -> None:
        if log_every <= 0:
            return
        if frames % log_every != 0 and tag == "tick":
            return
        s = session.state
        print(
            f"[bomb_roll {tag} f={frames} phase={phase.name}] "
            f"x={s.samus_x} y={s.samus_y} pose={s.pose} "
            f"vy={s.velocity_y} vx={s.velocity_x} morph={is_morph(s.pose)} "
            f"pb={s.max_power_bombs}",
            flush=True,
        )

    while session.state.samus_x > target_x and frames < max_frames:
        s = session.state
        if s.max_power_bombs > 0:
            phase = MorphBombRollPhase.DONE
            return s
        _log("tick")

        falling_hard = s.velocity_y > 80
        in_pit = s.samus_y > pit_y or falling_hard
        deep_pit = s.samus_y > deep_pit_y

        if in_pit:
            phase = MorphBombRollPhase.IN_PIT
            pit_recoveries += 1
            phase = MorphBombRollPhase.RECOVERING
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
                phase = MorphBombRollPhase.DONE
                return session.state
            phase = MorphBombRollPhase.ON_BAND
            continue

        if not is_morph(session.state.pose):
            ensure_morph(session)
            frames += 20
            continue

        # Progress watchdog (opt-in): no leftward progress → bomb + pause.
        # Disabled when stall_frames==0 (wall-open needs multi-cycle stalls).
        if stall_frames > 0 and frames_since_progress >= stall_frames:
            phase = MorphBombRollPhase.STALLED
            _hold(session, 2, "X", reason="safe_watchdog_bomb")
            _hold(session, 10, reason="safe_watchdog_pause")
            frames += 12
            frames_since_progress = 0
            _log("watchdog")
            phase = MorphBombRollPhase.ON_BAND
            continue

        # Bomb-jump cycle (matches proven wall@437 open timing).
        phase = MorphBombRollPhase.ON_BAND
        _hold(session, 2, "X", reason="safe_bomb")
        frames += 2
        for step in range(cycle_len):
            if frames >= max_frames:
                break
            s = session.state
            if s.max_power_bombs > 0:
                phase = MorphBombRollPhase.DONE
                return s
            if s.samus_x <= target_x and s.samus_y <= max_y + 5:
                phase = MorphBombRollPhase.DONE
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
    phase = MorphBombRollPhase.DONE
    return session.state
