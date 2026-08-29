"""Walk stall, cave rendezvous, and frozen-combat escape."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.combat import WalkProgress
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.stages import (
    DUO_BOSS_CHARS,
    RAPH_CHAR,
    SEWER_SAFE_WALK_Y,
    STARBASE_WAVES,
    WOUNDED_KNEE_JUMP_CHARS,
    is_sewer,
    is_wounded_knee,
)

# 0x003A can tick while Leo is stuck on Stage 2 dumpster collision.
_PLAYER_X_STALL_FRAMES = 40
# Deep dumpsters block mid/upper lanes; bottom-lane JUMP+RIGHT clears.
_STALL_DOWN_FRAMES = 36
_STALL_JUMP_RIGHT_FRAMES = 24
_STALL_RIGHT_FRAMES = 40
_STALL_UP_FRAMES = 36
_STALL_UP_RIGHT_FRAMES = 48
_STALL_SMASH_FRAMES = 24
# Starbase holds Raphael at x=64 during its opening spawn delay.
# Dumpster-stall on those frames pushes him down a lane.
_STARBASE_LAUNCH_X = 64
# Starbase right-rail form-1 vanish: X glued at ~229 while cam still
# ticks. Immediate RIGHT (no dumpster) is the Diag 33,825→24,645 cut.
# A 96f dumpster budget then RIGHT 40k-timeout Diag. Y-steer to 156
# made Boss9 8,880. Wave dumpsters stay at x=126 / 207.
_STARBASE_RAIL_X = 220
# Continuous power-on sticks at x=207 (cam auto-scrolls, dumpster Y-sweeps
# 113–194, damage frozen). Fast pin 207 dumpsters recover in <600f; after
# three full cycles the 207 freeze is the encode loop, not a dumpster.
# Do not exhaust-skip x=126 (Sewer-like always-RIGHT 40k-timeout).
_STARBASE_EXHAUST_X = 200
_STARBASE_DUMPSTER_CYCLES_BEFORE_RIGHT = 3


class PrehistoricCaveRecovery:
    """Return to the Stage 5 cave rendezvous after a frozen right edge.

    One Prehistoric cave fork stops scrolling at player X≈207.  The next
    wave is triggered near the upper-middle of the room, so continuing to
    press RIGHT can wait forever.  Activate only after a sustained empty
    screen at that exact edge, then walk to the observed trigger point.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_x = -1
        self._stall_frames = 0
        self._active = False
        self._wait_frames = 0

    def next(self, state: GameState) -> FrameAction | None:
        if (
            state.stage != 4
            or int(state.extras.get("event", -1)) != 0x0A
            or state.living_enemies
        ):
            self.reset()
            return None
        if state.player_x == self._last_x:
            self._stall_frames += 1
        else:
            self._last_x = state.player_x
            self._stall_frames = 0
        if state.player_x >= 200 and self._stall_frames >= 120:
            self._active = True
        if not self._active:
            return None
        if abs(state.player_y - 134) > 2:
            return FrameAction(
                action=buttons("UP" if state.player_y > 134 else "DOWN"),
                reason="cave_recenter_y",
            )
        target_x = 125
        if self._wait_frames > 240:
            # If the exact center does not fire immediately, sweep across it
            # instead of waiting forever at the edge of a trigger rectangle.
            target_x = 145 if (self._wait_frames // 240) % 2 else 105
        if abs(state.player_x - target_x) > 3:
            return FrameAction(
                action=buttons("LEFT" if state.player_x > target_x else "RIGHT"),
                reason="cave_recenter_x",
            )
        self._wait_frames += 1
        return FrameAction(action=idle_action(), reason="cave_wait")


class PlayerXStallWalk:
    """Walk right; break Stage 2 dumpster soft-locks when player X freezes.

    ``0x003A`` keeps ticking while Leo is glued to alley dumpsters. Deep
    dumpsters block the mid/upper lanes — drop to the bottom lane and
    JUMP+RIGHT, with UP / smash as fallbacks for earlier obstacles.
    Stage 0 (Big Apple) skips dumpster escapes — frozen X there is usually
    a wave lock, and DOWN thrash walks into chip. Starbase (byte 8) must
    **keep** dumpster DOWN+JUMP on mid-lane frozen X (x=126 / 207) —
    Sewer-like always-RIGHT and climb-only both 40k-timeout this pin.
    Opening spawn (x<=64) and right-rail frozen X (x>=220, form-1 vanish)
    hold RIGHT — dumpster DOWN+JUMP on the rail is the Diag 7k-frame loop.
    """

    def __init__(self, *, pickup_every: int = 24) -> None:
        self._walk = WalkProgress(pickup_every=pickup_every)
        self._last_player_x: int = -1
        self._stall_frames: int = 0
        self._escape_frames: int = 0

    def reset(self) -> None:
        """Clear walk + player-X stall tracking."""
        self._walk.reset()
        self._last_player_x = -1
        self._stall_frames = 0
        self._escape_frames = 0

    def _stall_escape(self, state: GameState) -> FrameAction:
        """Cycle dumpster breakers while X remains frozen."""
        down_end = _STALL_DOWN_FRAMES
        jump_end = down_end + _STALL_JUMP_RIGHT_FRAMES
        right_end = jump_end + _STALL_RIGHT_FRAMES
        up_end = right_end + _STALL_UP_FRAMES
        up_right_end = up_end + _STALL_UP_RIGHT_FRAMES
        smash_end = up_right_end + _STALL_SMASH_FRAMES
        cycle = smash_end
        if (
            state.stage == STARBASE_WAVES
            and _STARBASE_EXHAUST_X <= state.player_x < _STARBASE_RAIL_X
            and self._escape_frames >= cycle * _STARBASE_DUMPSTER_CYCLES_BEFORE_RIGHT
        ):
            self._escape_frames += 1
            return FrameAction(
                action=buttons("RIGHT"),
                reason="starbase_unstick_right",
            )
        phase = self._escape_frames
        self._escape_frames += 1
        slot = phase % cycle
        if slot < down_end:
            return FrameAction(action=buttons("DOWN"), reason="stall_down")
        if slot < jump_end:
            return FrameAction(action=buttons("B", "RIGHT"), reason="stall_jump_right")
        if slot < right_end:
            return FrameAction(action=buttons("RIGHT"), reason="stall_right")
        if slot < up_end:
            return FrameAction(action=buttons("UP"), reason="stall_up")
        if slot < up_right_end:
            return FrameAction(action=buttons("UP", "RIGHT"), reason="stall_up_right")
        return FrameAction(action=buttons("Y"), reason="stall_smash")

    def next(self, state: GameState) -> FrameAction:
        """Walk via WalkProgress; on X-stall run dumpster escape."""
        # Leonardo's frozen Stage 1 X is usually a wave lock, but Raphael
        # (player char 8) physically catches on the late dumpster at x=128.
        # Reuse the proven alley escape only for that character.
        if (
            state.stage == 0
            and int(state.extras.get("char_id", -1)) != RAPH_CHAR
        ):
            return self._walk.next(state)
        # Sewer Surfin' auto-scroll freezes X (and often the progress word).
        # Dumpster thrash AND WalkProgress camera-stall nudges (UP/DOWN+Y)
        # walk into hanging spikes. Hold RIGHT only (+ drop lane if high).
        # Clean (2026-07-27): dumpster stall thrash is **offline** on sewer —
        # auto-scroll freezes player X between waves and UP/DOWN thrash walks
        # into hanging spikes (4×16 dmg with empty enemy list). Walk RIGHT only.
        if is_sewer(state):
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
            if (
                not state.living_enemies
                and state.player_y < SEWER_SAFE_WALK_Y
            ):
                return FrameAction(
                    action=buttons("DOWN", "RIGHT"),
                    reason="sewer_drop_lane",
                )
            return FrameAction(action=buttons("RIGHT"), reason="walk_right")
        if (
            state.stage == STARBASE_WAVES
            and state.player_x <= _STARBASE_LAUNCH_X
            and not state.living_enemies
        ):
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
            return FrameAction(
                action=buttons("RIGHT"),
                reason="starbase_launch_right",
            )
        if (
            state.stage == STARBASE_WAVES
            and state.player_x >= _STARBASE_RAIL_X
            and not state.living_enemies
        ):
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
            return FrameAction(
                action=buttons("RIGHT"),
                reason="starbase_rail_right",
            )
        if state.living_enemies:
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
            return self._walk.next(state)
        if self._last_player_x == state.player_x:
            self._stall_frames += 1
        else:
            self._stall_frames = 0
            self._escape_frames = 0
            self._last_player_x = state.player_x
        if self._stall_frames >= _PLAYER_X_STALL_FRAMES:
            return self._stall_escape(state)
        return self._walk.next(state)


class CombatPositionStall:
    """Jump out when combat position and total enemy HP never change.

    Disabled while Tokka/Rahzar or Bebop/Rocksteady are live: the duo left-flank
    poke is intentionally stationary for long stretches, and the jump-escape
    walks into the pack (Boss4 probe: 364→116 damage with stall suppressed).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the frozen-position signature and escape phase."""
        self._signature: tuple[int, int, int, int] | None = None
        self._stalled_frames = 0
        self._escape_frame = -1

    def next(self, state: GameState) -> FrameAction | None:
        """Return a short no-special escape after a four-second freeze."""
        if not state.living_enemies or state.mode is not GameMode.PLAYING:
            self.reset()
            return None
        # Duo left-flank poke holds X/Y while chipping; do not override it.
        if any(
            enemy.kind in DUO_BOSS_CHARS and enemy.health > 0
            for enemy in state.living_enemies
        ):
            self.reset()
            return None
        # Rat King long poke is intentionally stationary; jump-escape into
        # the Footski / left chip is the heal=none Boss3 death train.
        if state.boss_active and is_sewer(state):
            self.reset()
            return None
        # Ignore fine Y bob so endless lane thrash (Wounded Knee 0xb0
        # stacks) still freezes the signature and triggers an escape.
        # Use coarser Y on Stage 7; elsewhere keep modest quantization.
        y_bin = state.player_y // (16 if is_wounded_knee(state) else 8)
        signature = (
            state.stage,
            state.player_x,
            y_bin,
            sum(enemy.health for enemy in state.living_enemies),
        )
        if self._escape_frame >= 0:
            phase = self._escape_frame // 32
            self._escape_frame += 1
            if self._escape_frame >= 160:
                self.reset()
            # Wounded Knee elevated stacks: jump-slash laterally so escape
            # is not a pure hop that leaves the 0xb0 untouched.
            if is_wounded_knee(state) and any(
                e.kind in WOUNDED_KNEE_JUMP_CHARS for e in state.living_enemies
            ):
                side = "LEFT" if (phase % 2) == 0 else "RIGHT"
                if phase < 3:
                    return FrameAction(
                        action=buttons("B", "Y", side),
                        reason="combat_stall_escape",
                    )
                return FrameAction(
                    action=buttons(side),
                    reason="combat_stall_escape",
                )
            patterns = (
                ("B", "DOWN", "RIGHT"),
                ("B", "DOWN", "LEFT"),
                ("B", "UP", "RIGHT"),
                ("B", "UP", "LEFT"),
                ("B", "Y"),
            )
            return FrameAction(
                action=buttons(*patterns[min(phase, 4)]),
                reason="combat_stall_escape",
            )
        if signature == self._signature:
            self._stalled_frames += 1
        else:
            self._signature = signature
            self._stalled_frames = 0
        if self._stalled_frames < 240:
            return None
        self._escape_frame = 0
        # First escape frame — same branch as the ongoing escape cycle so
        # Wounded Knee stacks get a jump-slash instead of a pure hop.
        if is_wounded_knee(state) and any(
            e.kind in WOUNDED_KNEE_JUMP_CHARS for e in state.living_enemies
        ):
            return FrameAction(
                action=buttons("B", "Y", "LEFT"),
                reason="combat_stall_escape",
            )
        return FrameAction(
            action=buttons("B", "DOWN", "RIGHT"),
            reason="combat_stall_escape",
        )
