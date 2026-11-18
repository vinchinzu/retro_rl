"""Rollout on SNES. Step cost (AP-054), ``Rollout.fan`` (AP-055), ``walk`` (AP-056).

The ROM is the enemy model, same claim as ``nes/zelda_i/rollout.py``.
``fan`` replays candidate press lists from one saved frame and restores the
env in a ``finally``. The numbers below are the lookahead budget. A later
guard must not copy the NES 24-frame horizon or the NES 73× figure.

Measured 2026-09-29 on an AMD Ryzen 9 5900XT, stable-retro,
``SDL_VIDEODRIVER=dummy``. Each ``em.step`` sample is 60 frames timed after
``set_state`` back to the pin, so the figure is still in that scene. A long
run that never restores walks out of the room and the step gets cheaper.
Do not quote that number.

``CastleB1Guard`` (room ``0x72``), 15 batches. A held RIGHT was 630.7 us,
not slower than idle:

    em.get_state()   25.5 us      em.set_state()   71.1 us
    em.step()       635.5 us      (1574 fps, 26x real time)

``FirstAction`` (overworld screen ``0x1B``), 15 batches. Idle is the slower
step, so it sets the budget. A held RIGHT was 649.8 us:

    em.get_state()   26.1 us      em.set_state()   70.4 us
    em.step()       655.7 us      (1525 fps, 25x real time)

1× real time is 60 Hz, the fps in stable-retro's metadata. One real frame
is 16667 us. Frames of lookahead per real frame is ``16667 / em.step``.

Budget: 25 frames of lookahead per real frame at 1×. 25 × 655.7 us fits in
one real frame on ``FirstAction``; 26 does not. ``CastleB1Guard`` still fits
26. One ``get_state`` plus one ``set_state`` is about 96 us and does not
change either integer.

Restore plus 60 replayed frames is bit-exact on both pins. The script was
15 frames each of RIGHT, DOWN, LEFT, UP, applied with ``set_button_mask``
and ``em.step``. ``get_ram()`` (147456 bytes) matched, and so did the
430181-byte state blob. The guard room stayed in ``0x72`` (281 RAM bytes
changed). The overworld walk left screen ``0x1B`` for ``0x2C`` and still
matched.

Report:
    bit_exact: true
    pins: CastleB1Guard, FirstAction
    ram_bytes: 147456
    state_bytes: 430181
    lookahead_frames_per_real_frame: 25

``contact_frame`` is the first stepped frame whose Link HP (``$F36D``,
verified on CastleB1Key) is lower than the previous frame. It is not an
i-frame edge. ``moved`` is Manhattan distance of Link's world xy.
``room_changed`` is a change of the room word ``read_snapshot`` stores from
``$A0``. On CastleB1Guard that word is ``0x0072`` (high byte 0).

``PolicyGuard`` (AP-057) plays a deep copy of the controller for
``POLICY_GUARD_HORIZON`` frames. That horizon is the AP-054 1× budget (25),
not the NES 24. A hit is an HP drop. A held-direction detour is committed
only when it stays on the same room word and, if the live frame had
control, stays in control. The inner's own room change is its plan.

``Rollout.walk`` (AP-056) records a ``room_step`` waypoint walk and restores.
Replaying those presses ends on the same xy: the ROM is deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from retro_harness.actions import SNES_ACTION_SIZE, snes_action
from retro_harness.controls import SNES_BUTTON_NAMES
from retro_harness.input_script import FrameAction
from alttp.primitives import (
    SPRITE_FLOOR,
    SPRITE_HIT_TIMER,
    SPRITE_HP,
    SPRITE_SLOTS,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
)
from alttp.ram import (
    LINK_HP,
    LINK_LAYER,
    AlttpSnapshot,
    read_snapshot,
    read_u8_safe,
    wram_index,
)
from alttp.sprite_ids import vulnerability

__all__ = [
    "HAZARD_REACH",
    "POLICY_GUARD_DETOURS",
    "POLICY_GUARD_DIRECTIONS",
    "POLICY_GUARD_HORIZON",
    "ROM_CHECKED",
    "Frame",
    "Outcome",
    "Plan",
    "PolicyGuard",
    "Rollout",
    "guard_press",
    "hazard_in_reach",
    "hold",
    "press",
    "restores",
    "strike_if_safe",
    "safe_combat_plan",
    "strike_soldier",
    "strike_verdict",
]

# One frame is the SNES action vector ``snes_action`` builds. ``set_button_mask``
# wants it as ``uint8``. A Python int is read as a one-button array (B), so
# every plan in a fan lands on the same pixel.
Frame = tuple[int, ...]
Plan = tuple[Frame, ...]

# Lookahead restores this process has made. A run report sets this beside
# ``set_state`` so a rollout load is not counted as a mid-run state load.
_RESTORES = 0


def restores() -> int:
    """Lookahead restores (``em.set_state`` back to the saved frame) so far."""
    return _RESTORES


def _restore(em: Any, state: bytes) -> None:
    global _RESTORES
    em.set_state(state)
    _RESTORES += 1


def press(*names: str) -> Frame:
    """One plan frame. ``press()`` is idle. Raises on an unknown button."""
    cleaned = tuple(str(name).strip().upper() for name in names)
    for name in cleaned:
        if name not in SNES_BUTTON_NAMES:
            raise ValueError(f"unknown SNES button: {name}")
    return tuple(int(value) for value in snes_action(*cleaned))


def hold(direction: str | None, frames: int) -> Plan:
    """A plan that holds one direction (or nothing) for ``frames``."""
    if direction in (None, "STAND"):
        frame = press()
    else:
        frame = press(direction)
    return (frame,) * max(int(frames), 0)


def _pad(script: Plan, frames: int | None) -> Plan:
    if frames is None:
        return script
    want = max(int(frames), 0)
    if len(script) >= want:
        return script[:want]
    tail = script[-1] if script else press()
    return script + (tail,) * (want - len(script))


def _chebyshev(x: int, y: int, gx: int, gy: int) -> int:
    return max(abs(int(x) - int(gx)), abs(int(y) - int(gy)))


def _toward(x: int, y: int, goal: tuple[int, int], tol: int) -> str | None:
    """Cardinal toward ``goal``, or None once Chebyshev is inside ``tol``."""
    dx = int(goal[0]) - int(x)
    dy = int(goal[1]) - int(y)
    if max(abs(dx), abs(dy)) <= int(tol):
        return None
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return None


def _link_hp(ram: np.ndarray) -> int:
    return read_u8_safe(ram, wram_index(LINK_HP))


def _mask(frame: Frame) -> np.ndarray:
    values = tuple(int(value) for value in frame)
    if len(values) != SNES_ACTION_SIZE:
        raise ValueError(
            f"plan frame is {len(values)} buttons, want {SNES_ACTION_SIZE}"
        )
    return np.asarray(values, dtype=np.uint8)


@dataclass(frozen=True)
class Outcome:
    """What one plan did, measured from the frame the fan started on.

    ``contact_frame`` is the first HP drop (1-based), not an i-frame edge.
    ``moved`` is Manhattan world distance. ``room_changed`` is the ``$A0``
    room word.
    """

    label: str
    frames: int
    snap: AlttpSnapshot
    hp_before: int
    hp_after: int
    contact_frame: int | None
    moved: int
    room_changed: bool
    room_before: int
    room_after: int

    def report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "frames": self.frames,
            "contact_frame": self.contact_frame,
            "moved": self.moved,
            "room_changed": self.room_changed,
            "room_before": self.room_before,
            "room_after": self.room_after,
            "hp_before": self.hp_before,
            "hp_after": self.hp_after,
            "xy": [int(self.snap.link_x), int(self.snap.link_y)],
        }


class Rollout:
    """Branch the live emulator over candidate plans and restore it."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.em = env.em
        self.rollouts = 0
        self.frames_rolled = 0

    def fan(
        self,
        plans: Mapping[str, Sequence[Frame]],
        *,
        frames: int | None = None,
    ) -> tuple[Outcome, ...]:
        """Replay every plan from this one frame. Order follows ``plans``.

        ``frames`` pads or truncates every plan to one length. A short plan
        holds its last frame (idle if it is empty). The env's RAM on return
        is the RAM from before the call.
        """
        if not plans:
            return ()
        ram0 = np.asarray(self.env.get_ram(), dtype=np.uint8)
        before = read_snapshot(ram0)
        hp0 = _link_hp(ram0)
        state = bytes(self.em.get_state())
        outcomes: list[Outcome] = []
        try:
            for label, plan in plans.items():
                script = _pad(
                    tuple(tuple(int(value) for value in frame) for frame in plan),
                    frames,
                )
                _restore(self.em, state)
                outcomes.append(self._replay(label, script, before, hp0))
        finally:
            _restore(self.em, state)
        return tuple(outcomes)

    def run(self, label: str, plan: Sequence[Frame]) -> Outcome:
        """Replay one button script from the live frame. Restores after."""
        return self.fan({label: plan})[0]

    def walk(
        self,
        waypoints: Sequence[tuple[int, int]],
        *,
        prefix: Sequence[Frame] = (),
        frames: int = 300,
        tol: int = 2,
    ) -> tuple[Plan, Outcome, bool]:
        """Roll ``prefix`` then a ``room_step`` walk. Restore when done.

        The presses are what ``room_step`` chose on the rolled frames, so
        replaying ``plan`` live ends on ``outcome``'s xy. ``reached`` is
        whether the last waypoint was inside ``tol`` within ``frames``.
        """
        from alttp.dungeon.nav import WalkCommit, room_step

        state = bytes(self.em.get_state())
        script: list[Frame] = [tuple(int(value) for value in frame) for frame in prefix]
        commit = WalkCommit()
        index = 0
        try:
            for frame in script:
                self.em.set_button_mask(_mask(frame), 0)
                self.em.step()
            for _ in range(max(int(frames), 0)):
                snap = read_snapshot(np.asarray(self.env.get_ram(), dtype=np.uint8))
                while index < len(waypoints) and _chebyshev(
                    snap.link_x, snap.link_y, waypoints[index][0], waypoints[index][1]
                ) <= int(tol):
                    index += 1
                if index >= len(waypoints):
                    break
                button, commit = room_step(
                    commit, _toward(snap.link_x, snap.link_y, waypoints[index], tol)
                )
                frame = press(button) if button else press()
                script.append(frame)
                self.em.set_button_mask(_mask(frame), 0)
                self.em.step()
        finally:
            _restore(self.em, state)
        plan = tuple(script)
        return plan, self.run("walk", plan), index >= len(waypoints)

    def _replay(
        self,
        label: str,
        script: Sequence[Frame],
        before: AlttpSnapshot,
        hp0: int,
    ) -> Outcome:
        self.rollouts += 1
        self.frames_rolled += len(script)
        contact: int | None = None
        hp_prev = hp0
        snap = before
        hp_after = hp0
        for index, frame in enumerate(script):
            self.em.set_button_mask(_mask(frame), 0)
            self.em.step()
            ram = np.asarray(self.env.get_ram(), dtype=np.uint8)
            snap = read_snapshot(ram)
            hp_after = _link_hp(ram)
            if contact is None and hp_after < hp_prev:
                contact = index + 1
            hp_prev = hp_after
        return Outcome(
            label=label,
            frames=len(script),
            snap=snap,
            hp_before=hp0,
            hp_after=hp_after,
            contact_frame=contact,
            moved=(
                abs(int(snap.link_x) - int(before.link_x))
                + abs(int(snap.link_y) - int(before.link_y))
            ),
            room_changed=int(snap.room_id) != int(before.room_id),
            room_before=int(before.room_id),
            room_after=int(snap.room_id),
        )

    def report(self) -> dict[str, int]:
        return {"rollouts": self.rollouts, "frames_rolled": self.frames_rolled}


# AP-054: 25 lookahead frames fit in one real frame. Do not copy the NES 24.
POLICY_GUARD_HORIZON = 25
# Shorter than the horizon, so every committed frame was actually stepped.
# 8 frames is about a body width at the measured 1-then-2 px step.
POLICY_GUARD_DETOURS = (8, 16, 24)
POLICY_GUARD_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
# A frame already answered on the ROM. The guard must not roll it again.
ROM_CHECKED = "rom_checked:"
# CastleB1Guard, pad released. The floor-0 soldier's Chebyshev to Link was
# 45 at frame 140; the HP drop was frame 164, 24 frames later. A body inside
# this radius can touch Link inside the horizon. Past it, only a measured
# per-frame speed whose ETA is inside the horizon opens the gate.
HAZARD_REACH = 48
_SpriteXY = dict[int, tuple[int, int, int]]


def _is_hostile(kind: int) -> bool:
    """Contact damage, minus pickups and followers.

    ``ignores_damage`` plus contact class 0 is the NPC / pickup / follower
    row (heart ``0xD8``, uncle ``0x73``, Zelda ``0x76``). Enemy shots are
    sprites, including arrow ``0x1B``; they have a real contact class.
    """
    try:
        row = vulnerability(kind)
    except (TypeError, ValueError):
        return False
    if row.contact_damage[0] <= 0:
        return False
    if row.ignores_damage and row.contact_class == 0:
        return False
    return True


def hazard_in_reach(
    ram: np.ndarray,
    prev: _SpriteXY | None = None,
    *,
    reach: int = HAZARD_REACH,
    horizon: int = POLICY_GUARD_HORIZON,
) -> tuple[bool, _SpriteXY]:
    """True when a hostile sprite or shot can arrive inside ``horizon``.

    Distance ``reach`` is the standing-soldier measurement. ETA uses the
    Chebyshev px that slot moved since ``prev`` (one play frame). The
    returned map is the next ``prev``. Other-floor slots are ignored.
    """
    snap = read_snapshot(ram)
    layer = int(ram[LINK_LAYER]) if len(ram) > LINK_LAYER else 0
    seen: _SpriteXY = {}
    near = False
    prior = prev or {}
    for slot in range(SPRITE_SLOTS):
        if int(ram[SPRITE_STATE + slot]) == 0:
            continue
        kind = int(ram[SPRITE_TYPE + slot])
        if int(ram[SPRITE_FLOOR + slot]) != layer:
            continue
        x = int(ram[SPRITE_X_LOW + slot]) | (int(ram[SPRITE_X_HIGH + slot]) << 8)
        y = int(ram[SPRITE_Y_LOW + slot]) | (int(ram[SPRITE_Y_HIGH + slot]) << 8)
        seen[slot] = (x, y, kind)
        if not _is_hostile(kind):
            continue
        dist = max(abs(x - int(snap.link_x)), abs(y - int(snap.link_y)))
        if dist <= int(reach):
            near = True
            continue
        old = prior.get(slot)
        if old is None or old[2] != kind:
            continue
        speed = max(abs(x - old[0]), abs(y - old[1]))
        if speed and dist <= speed * int(horizon):
            near = True
    return near, seen


class PlannedPress:
    """One held direction for ``PolicyGuard``. The walker sets ``button`` each frame."""

    def __init__(self, max_frames: int = 0) -> None:
        self.button = "NONE"
        self.max_frames = max_frames

    def step(self, _snap: AlttpSnapshot) -> FrameAction:
        from alttp.startup import action_for, no_action

        act = no_action() if self.button in (None, "NONE") else action_for(self.button)
        return FrameAction([int(value) for value in act], "ow")

    def report(self) -> dict[str, object]:
        return {}


def guard_press(env: Any, max_frames: int = 0) -> tuple[PlannedPress, PolicyGuard]:
    """Bind a direction holder so an overworld walk can dodge under ``PolicyGuard``."""
    holder = PlannedPress(max_frames)
    policy = PolicyGuard(holder)
    policy.bind_env(env)
    return holder, policy


class PolicyGuard:
    """Follow ``inner`` unless the ROM says its own next frames get Link hit.

    Each play frame, a deep copy of ``inner`` is played ``horizon`` frames.
    On an HP drop, held-direction detours are rolled the same way. The first
    one that stays in the room and is hit later than staying on the plan
    (a clean roll counts as later than the horizon) is committed. Detour
    frames do not step ``inner``.
    """

    def __init__(
        self,
        inner: Any,
        *,
        horizon: int = POLICY_GUARD_HORIZON,
        detours: Sequence[int] = POLICY_GUARD_DETOURS,
        directions: Sequence[str] = POLICY_GUARD_DIRECTIONS,
        reach: int | None = HAZARD_REACH,
    ) -> None:
        self.inner = inner
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.detours = tuple(int(k) for k in detours)
        self.directions = tuple(directions)
        # None rolls every controllable frame. The default rolls only when
        # hazard_in_reach says a hostile or a shot can arrive in time.
        self.reach = None if reach is None else int(reach)
        self.max_frames = int(getattr(inner, "max_frames", 0))
        self.checks = 0
        self.quiet = 0
        self.play_frames = 0
        self.detours_taken = 0
        self.detour_frames = 0
        self.frames_rolled = 0
        self._env: Any = None
        self._queue: list[Frame] = []
        self._last_reason = ""
        self._sprite_xy: _SpriteXY = {}
        self._t0: float | None = None

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def bind_env(self, env: Any) -> None:
        self._env = env
        bind = getattr(self.inner, "bind_env", None)
        if callable(bind):
            bind(env)

    @property
    def success(self) -> bool:
        return bool(getattr(self.inner, "success", False))

    @property
    def failed(self) -> bool:
        return bool(getattr(self.inner, "failed", False))

    def drop_detour(self) -> None:
        """Forget a committed detour so the next step follows ``inner``."""
        self._queue.clear()

    def _copy(self) -> Any:
        import copy

        env = self._env
        memo = {id(env): env}
        em = getattr(env, "em", None)
        if em is not None:
            memo[id(em)] = em
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            memo[id(unwrapped)] = unwrapped
        return copy.deepcopy(self.inner, memo)

    def _contact(self, detour: Sequence[Frame]) -> tuple[int | None, bool]:
        """``(hit frame, left the room during the detour)``.

        ``hit frame`` is the first HP drop (1-based), or None when the
        horizon stays at the starting HP. The inner's own room change is
        not a hit and not a reason to reject: ``left`` is set only while
        the detour itself is playing.
        """
        em = self._env.em
        state = bytes(em.get_state())
        start = read_snapshot(np.asarray(self._env.get_ram(), dtype=np.uint8))
        room = int(start.room_id)
        armed = bool(start.has_control)
        hp_prev = int(start.hp)
        try:
            probe = self._copy()
            for t in range(self.horizon):
                ram = np.asarray(self._env.get_ram(), dtype=np.uint8)
                snap = read_snapshot(ram)
                if t < len(detour):
                    frame: Frame = detour[t]
                else:
                    frame = _action_frame(probe.step(snap))
                    # The next stage owns input after the inner's goal.
                    # Forecasting idle here can refuse a safe door approach
                    # because an enemy would arrive after the real handoff.
                    if getattr(probe, "success", False):
                        return (None, False)
                em.set_button_mask(_mask(frame), 0)
                em.step()
                self.frames_rolled += 1
                snapped = read_snapshot(np.asarray(self._env.get_ram(), dtype=np.uint8))
                in_detour = bool(detour) and (t + 1) <= len(detour)
                left = (
                    int(snapped.room_id) != room or snapped.layer != start.layer
                    or snapped.floor != start.floor or (armed and not snapped.has_control)
                )
                if left:
                    if in_detour:
                        return (t + 1, True)
                    return (None, False)
                if int(snapped.hp) < hp_prev:
                    return (t + 1, False)
                hp_prev = int(snapped.hp)
        finally:
            _restore(em, state)
        return (None, False)

    def _inner_step(self, snap: AlttpSnapshot) -> FrameAction:
        act = self.inner.step(snap)
        frame = _action_frame(act)
        self._last_reason = str(getattr(act, "reason", ""))
        return FrameAction(list(frame), self._last_reason)

    def _mark_play(self) -> None:
        self.play_frames += 1
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def step(self, snap: AlttpSnapshot) -> FrameAction:
        self._mark_play()
        detour_reason = f"{ROM_CHECKED}policy_guard_detour"
        if self._queue and snap.has_control and not (self.success or self.failed):
            self.detour_frames += 1
            return FrameAction(list(self._queue.pop(0)), detour_reason)
        self._queue.clear()
        if ROM_CHECKED in self._last_reason:
            return self._inner_step(snap)
        if getattr(self._env, "em", None) is None:
            return self._inner_step(snap)
        if not snap.has_control or self.success or self.failed:
            return self._inner_step(snap)
        if self.reach is not None:
            ram = np.asarray(self._env.get_ram(), dtype=np.uint8)
            near, self._sprite_xy = hazard_in_reach(
                ram, self._sprite_xy, reach=self.reach, horizon=self.horizon
            )
            if not near:
                self.quiet += 1
                return self._inner_step(snap)
        self.checks += 1
        base, _base_left = self._contact(())
        if base is None:
            return self._inner_step(snap)
        best: tuple[tuple[int, int], tuple[Frame, ...]] | None = None
        for k in self.detours:
            for direction in self.directions:
                # Frames past the horizon were not stepped. Do not commit them.
                presses = (press(direction),) * min(int(k), self.horizon)
                hit, left = self._contact(presses)
                if left:
                    continue
                key = (self.horizon + 1 if hit is None else hit, -len(presses))
                if best is None or key > best[0]:
                    best = (key, presses)
            if best is not None and best[0][0] > self.horizon:
                break
        if best is None or best[0][0] <= base:
            return self._inner_step(snap)
        self.detours_taken += 1
        self._queue = list(best[1][1:])
        self.detour_frames += 1
        return FrameAction(list(best[1][0]), detour_reason)

    def report(self) -> dict[str, Any]:
        rep = self.inner.report() if callable(getattr(self.inner, "report", None)) else {}
        if not isinstance(rep, dict):
            rep = {}
        else:
            rep = dict(rep)
        elapsed = 0.0 if self._t0 is None else time.perf_counter() - self._t0
        played = self.play_frames
        rep["policy_guard"] = {
            "checks": self.checks,
            "quiet": self.quiet,
            "play_frames": played,
            "fraction_rolled": (self.checks / played) if played else 0.0,
            "detours": self.detours_taken,
            "detour_frames": self.detour_frames,
            "frames_rolled": self.frames_rolled,
            "horizon": self.horizon,
            "reach": self.reach,
            "x_realtime": (played / 60.0 / elapsed) if elapsed > 0 else None,
        }
        return rep


def _action_frame(act: Any) -> Frame:
    action = getattr(act, "action", act)
    return tuple(int(value) for value in action)


def _sprite_at(ram: np.ndarray, slot: int) -> tuple[int, int, int] | None:
    """Sprite origin and HP, or None when the slot is empty."""
    if not 0 <= int(slot) < SPRITE_SLOTS:
        raise ValueError(f"sprite slot {slot} is outside 0..{SPRITE_SLOTS - 1}")
    if int(ram[SPRITE_STATE + slot]) == 0:
        return None
    return (
        int(ram[SPRITE_X_LOW + slot]) | (int(ram[SPRITE_X_HIGH + slot]) << 8),
        int(ram[SPRITE_Y_LOW + slot]) | (int(ram[SPRITE_Y_HIGH + slot]) << 8),
        int(ram[SPRITE_HP + slot]),
    )


def _play(env: Any, frame: Frame) -> None:
    env.em.set_button_mask(_mask(frame), 0)
    env.em.step()


def safe_combat_plan(env: Any, slot: int) -> Plan | None:
    """Choose a complete approach, swing and recovery on the live ROM.

    Forecast 32 frames, including body and weapon contact, then commit eight
    and replan. This avoids ending a long recovery in an inescapable flail arc.
    Rank damage to the target before closing distance. Reject any HP drop or
    room/floor/layer change; do not substitute an unsafe plan when none pass.
    This combat search is slower than the 25-frame real-time guard budget.
    """
    before = read_snapshot(np.asarray(env.get_ram(), dtype=np.uint8))
    plans = [hold(None, 32), (press("B"),) + hold(None, 31)]
    for direction in POLICY_GUARD_DIRECTIONS:
        for approach in (1, 4, 8):
            plans.append(hold(direction, approach) + (press("B"),)
                         + hold(None, 31 - approach))
        plans.append(hold(direction, 32))
    state = bytes(env.em.get_state())
    best: tuple[tuple[int, int, int], Plan] | None = None
    try:
        for plan in plans:
            _restore(env.em, state)
            safe = True
            hp_previous = before.hp
            for frame in plan:
                _play(env, frame)
                snap = read_snapshot(np.asarray(env.get_ram(), dtype=np.uint8))
                if (snap.hp < hp_previous or not snap.has_control
                        or (snap.room_id, snap.floor, snap.layer)
                        != (before.room_id, before.floor, before.layer)):
                    safe = False
                    break
                hp_previous = snap.hp
            if not safe:
                continue
            target = _sprite_at(np.asarray(env.get_ram(), dtype=np.uint8), slot)
            hp = target[2] if target else 0
            distance = _chebyshev(snap.link_x, snap.link_y, *target[:2]) if target else 0
            moved = abs(snap.link_x - before.link_x) + abs(snap.link_y - before.link_y)
            rank = (-hp, -distance, moved)
            if best is None or rank > best[0]:
                best = (rank, plan)
    finally:
        _restore(env.em, state)
    return best[1][:8] if best else None


def strike_verdict(env: Any, slot: int) -> str:
    """Roll one B swing and restore. ``hit``, ``hurt``, ``whiff``, ``stun``, or ``gone``.

    A hit is a target HP drop or a shove of at least 4px while bit 7 of
    ``$0EF0`` is set. ``hurt`` is a Link HP drop. ``strike_if_safe`` commits
    only ``hit``. The env is restored either way.
    """
    from alttp.walk_physics import SWORD_SWING_FRAMES

    ram0 = np.asarray(env.get_ram(), dtype=np.uint8)
    before = _sprite_at(ram0, slot)
    if before is None:
        return "gone"
    if int(ram0[SPRITE_HIT_TIMER + slot]) & 0x80:
        return "stun"
    link0 = read_snapshot(ram0)
    hp0 = _link_hp(ram0)
    start = max(abs(before[0] - link0.link_x), abs(before[1] - link0.link_y))
    state = bytes(env.em.get_state())
    script = (press("B"),) + hold(None, SWORD_SWING_FRAMES - 1)
    hp_after = before[2]
    link_hp = hp0
    pushed = False
    try:
        for frame in script:
            _play(env, frame)
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            link_hp = _link_hp(ram)
            if link_hp < hp0:
                return "hurt"
            viewed = _sprite_at(ram, slot)
            snap = read_snapshot(ram)
            if viewed is None:
                hp_after = -1
                continue
            hp_after = viewed[2]
            dist = max(abs(viewed[0] - snap.link_x), abs(viewed[1] - snap.link_y))
            if dist >= start + 4 and int(ram[SPRITE_HIT_TIMER + slot]) & 0x80:
                pushed = True
    finally:
        _restore(env.em, state)
    if link_hp != hp0:
        return "hurt"
    if hp_after < before[2] or pushed:
        return "hit"
    return "whiff"


def strike_if_safe(env: Any, slot: int) -> tuple[FrameAction, ...] | None:
    """Commit one B swing only when :func:`strike_verdict` is ``hit``.

    Every committed frame is marked ``ROM_CHECKED``. A target already in
    sword hitstun is left for a later frame.
    """
    from alttp.walk_physics import SWORD_SWING_FRAMES

    if strike_verdict(env, slot) != "hit":
        return None
    script = (press("B"),) + hold(None, SWORD_SWING_FRAMES - 1)
    reason = f"{ROM_CHECKED}strike"
    return tuple(FrameAction(list(frame), reason) for frame in script)


def strike_soldier(
    env: Any,
    slot: int,
    *,
    approach: str = "DOWN",
    frames: int = 900,
) -> dict[str, int | bool]:
    """Swing with ``strike_if_safe`` and idle while sword hitstun is set.

    Bit 7 of ``$0EF0`` blocks another roll. Waiting it out, then swinging
    again, is what keeps Link off the body after the first shove. Between
    safe swings this holds ``approach``. CastleB1Guard's sword line is DOWN.
    No RAM writes. ``dead`` is the slot gone or at 0 HP. ``hp`` is Link.
    """
    start = _link_hp(np.asarray(env.get_ram(), dtype=np.uint8))
    swings = 0
    used = 0
    for _ in range(max(int(frames), 0)):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        link_hp = _link_hp(ram)
        sprite = _sprite_at(ram, slot)
        if link_hp < start:
            break
        if sprite is None or sprite[2] <= 0:
            return {"dead": True, "swings": swings, "hp": link_hp, "frames": used}
        if int(ram[SPRITE_HIT_TIMER + slot]) & 0x80:
            _play(env, press())
            used += 1
            continue
        script = strike_if_safe(env, slot)
        if script:
            for act in script:
                _play(env, tuple(int(value) for value in act.action))
            swings += 1
            used += len(script)
            continue
        _play(env, press(approach))
        used += 1
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    sprite = _sprite_at(ram, slot)
    return {
        "dead": sprite is None or sprite[2] <= 0,
        "swings": swings,
        "hp": _link_hp(ram),
        "frames": used,
    }
