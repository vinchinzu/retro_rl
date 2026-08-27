"""Super Shredder form-2 vertical-offset tactics (Clean path off iframes).

Wiki (RetroMaggedon Scene 10 / Hard): invulnerable inside the aura; stand
just above, below, or behind while it is up; step in and combo when it
drops; follow the teleport. Never stand in front — green fireball is a
life loss. Blue = ice 45°, yellow = floor flame, green = straight fireball.

This arena is a floor: UP/DOWN is vertical, not Mode-7 depth. No A, no
grounded Y+B Power Attack. B+direction hops are allowed.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameState
from tmnt_iv.grind_knobs import active_knobs

_SUPER_SHREDDER_F2 = 0xAE
_FORM2_STAGE = 9
_FORM2_EVENT = 0x0A

# Wiki: occupy a 16–28px vertical band while the aura is up.
_OFFSET_MIN = 16
_OFFSET_MAX = 28
_OFFSET_TARGET = 22
# Overlapping X on his lane = in front of the green fireball.
_FACE_ADX = 12
_FACE_ADY = 10
# Horizontal beside / behind — not overlapping his body.
_BESIDE = 32
_BESIDE_TOL = 10
# Step-in chip band once the projectile has passed.
_CHIP_ADX = 24
_CHIP_ADY = 12
_POST_FIRE_DELAY = 8
_TELEPORT_DX = 40
_LEFT_WALL = 64
_RIGHT_WALL = 200
_Y_FLOOR_MIN = 148
_Y_FLOOR_MAX = 214
# Idle/probe: 0xEE windup then 0xFE fire; leaving those is the drop.
_DROP_ANIMS = frozenset({0xEE, 0xFE})


def _in_form2_arena(state: GameState) -> bool:
    return state.stage == _FORM2_STAGE and int(state.extras.get("event", -1)) == _FORM2_EVENT


def _preferred_offset_y(boss_y: int, player_y: int) -> int:
    """Pick a 16–28px vertical slot on the roomier side of the boss."""
    above = boss_y - _OFFSET_TARGET
    below = boss_y + _OFFSET_TARGET
    above_ok = above >= _Y_FLOOR_MIN
    below_ok = below <= _Y_FLOOR_MAX
    if player_y <= boss_y - _OFFSET_MIN and above_ok:
        return max(_Y_FLOOR_MIN, above)
    if player_y >= boss_y + _OFFSET_MIN and below_ok:
        return min(_Y_FLOOR_MAX, below)
    if above_ok and not below_ok:
        return max(_Y_FLOOR_MIN, above)
    if below_ok and not above_ok:
        return min(_Y_FLOOR_MAX, below)
    if abs(player_y - above) <= abs(player_y - below):
        return max(_Y_FLOOR_MIN, above)
    return min(_Y_FLOOR_MAX, below)


def _behind_side(*, boss_x: int, player_x: int) -> str:
    """Hop toward the open side, never into the left wall."""
    if boss_x < _LEFT_WALL:
        return "RIGHT"
    if boss_x > _RIGHT_WALL:
        return "LEFT"
    if player_x >= boss_x:
        return "RIGHT"
    return "LEFT"


class SuperShredderForm2Tactics:
    """Vertical offset while aura is up; step in and Y only on the drop."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear offset / drop-window phase."""
        self._phase = "offset"
        self._timer = 0
        self._delay = 0
        self._cadence = 0
        self._boss_x = -1
        self._anim = -1
        self._offset_wait = 0
        self._saw_drop = False

    def next(self, state: GameState) -> FrameAction | None:
        """Return a form-2 action, or ``None`` outside the finale arena."""
        if not _in_form2_arena(state):
            if self._phase != "offset" or self._timer or self._offset_wait:
                self.reset()
            return None
        boss = next(
            (
                enemy
                for enemy in state.living_enemies
                if enemy.kind == _SUPER_SHREDDER_F2
            ),
            None,
        )
        if boss is None:
            return None

        knobs = active_knobs()
        status = int(boss.animation)
        if self._boss_x >= 0 and abs(boss.x - self._boss_x) >= _TELEPORT_DX:
            self._phase = "offset"
            self._timer = 0
            self._delay = 0
            self._offset_wait = 0
        self._boss_x = boss.x

        drop_anim = status in _DROP_ANIMS
        if drop_anim:
            self._saw_drop = True
            # Projectile / drop windup: abort any step-in and hold offset.
            self._phase = "offset"
            self._timer = 0
            self._delay = 0
            self._offset_wait = 0
        elif self._anim in _DROP_ANIMS:
            self._phase = "attack"
            self._timer = knobs.shredder_drop_chip_frames
            self._delay = _POST_FIRE_DELAY
        self._anim = status

        adx = abs(boss.x - state.player_x)
        ady = abs(boss.y - state.player_y)
        in_front = adx <= _FACE_ADX and ady <= _FACE_ADY
        in_offset = _OFFSET_MIN <= ady <= _OFFSET_MAX
        use_right = boss.x < _LEFT_WALL
        open_side = "RIGHT" if use_right else "LEFT"
        target_x = boss.x + _BESIDE if use_right else boss.x - _BESIDE
        target_x = max(24, min(220, target_x))
        target_y = _preferred_offset_y(boss.y, state.player_y)

        if self._phase == "behind":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "offset"
                self._timer = 0
            return FrameAction(
                action=buttons("B", open_side),
                reason="shredder_behind",
            )

        # In front during aura / projectile: vertical first, then hop behind.
        if in_front and self._phase != "attack":
            return self._escape_front(
                state,
                target_y=target_y,
                open_side=open_side,
                player_x=state.player_x,
                boss_x=boss.x,
                behind_hop_frames=knobs.shredder_behind_hop_frames,
            )

        if self._phase == "attack":
            if self._delay > 0:
                self._delay -= 1
                # Hold the vertical band but slide onto the chip x-band so
                # the post-delay step-in is only a short UP/DOWN.
                chip_x = boss.x + _CHIP_ADX if use_right else boss.x - _CHIP_ADX
                chip_x = max(24, min(220, chip_x))
                return self._hold_offset(
                    state,
                    target_x=chip_x,
                    target_y=target_y,
                    in_offset=in_offset,
                    adx=adx,
                    open_side=open_side,
                    space_adx=knobs.shredder_space_adx,
                )
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "offset"
                self._timer = 0
                self._delay = 0
                self._offset_wait = 0
            else:
                return self._chip(
                    state, boss_x=boss.x, boss_y=boss.y, adx=adx, ady=ady
                )

        return self._hold_offset(
            state,
            target_x=target_x,
            target_y=target_y,
            in_offset=in_offset,
            adx=adx,
            open_side=open_side,
            space_adx=knobs.shredder_space_adx,
            allow_fallback=not drop_anim,
        )

    def _hold_offset(
        self,
        state: GameState,
        *,
        target_x: int,
        target_y: int,
        in_offset: bool,
        adx: int,
        open_side: str,
        space_adx: int,
        allow_fallback: bool = False,
    ) -> FrameAction:
        knobs = active_knobs()
        if not in_offset:
            if state.player_y > target_y:
                return FrameAction(action=buttons("UP"), reason="shredder_offset")
            if state.player_y < target_y:
                return FrameAction(action=buttons("DOWN"), reason="shredder_offset")
        dx = target_x - state.player_x
        if abs(dx) > _BESIDE_TOL:
            return FrameAction(
                action=buttons("LEFT" if dx < 0 else "RIGHT"),
                reason="shredder_approach",
            )
        if adx < space_adx:
            return FrameAction(action=buttons(open_side), reason="shredder_space")
        self._offset_wait += 1
        # Anim-blind fallback only if this fight never showed 0xEE/0xFE.
        if (
            allow_fallback
            and not self._saw_drop
            and self._offset_wait >= knobs.shredder_blind_offset_timeout
        ):
            self._phase = "attack"
            self._timer = knobs.shredder_drop_chip_frames
            self._delay = 0
            self._offset_wait = 0
            return self._chip(
                state,
                boss_x=state.player_x + (1 if open_side == "LEFT" else -1),
                boss_y=target_y,
                adx=adx,
                ady=abs(state.player_y - target_y),
            )
        return FrameAction(action=idle_action(), reason="shredder_wait")

    def _escape_front(
        self,
        state: GameState,
        *,
        target_y: int,
        open_side: str,
        player_x: int,
        boss_x: int,
        behind_hop_frames: int,
    ) -> FrameAction:
        if state.player_y != target_y:
            return FrameAction(
                action=buttons("UP" if state.player_y > target_y else "DOWN"),
                reason="shredder_offset",
            )
        self._phase = "behind"
        self._timer = behind_hop_frames
        side = _behind_side(boss_x=boss_x, player_x=player_x)
        if boss_x < _LEFT_WALL:
            side = open_side
        return FrameAction(action=buttons("B", side), reason="shredder_behind")

    def _chip(
        self,
        state: GameState,
        *,
        boss_x: int,
        boss_y: int,
        adx: int,
        ady: int,
    ) -> FrameAction:
        """Step onto his x-band and mash Y. No A, no Y+B."""
        knobs = active_knobs()
        if ady > _CHIP_ADY:
            return FrameAction(
                action=buttons("UP" if state.player_y > boss_y else "DOWN"),
                reason="shredder_close",
            )
        if adx > knobs.shredder_attack_adx:
            return FrameAction(
                action=buttons("LEFT" if state.player_x > boss_x else "RIGHT"),
                reason="shredder_approach",
            )
        if adx > _CHIP_ADX:
            return FrameAction(
                action=buttons("LEFT" if state.player_x > boss_x else "RIGHT"),
                reason="shredder_close",
            )
        self._cadence = (self._cadence + 1) % 4
        if self._cadence == 3:
            return FrameAction(action=idle_action(), reason="shredder_attack")
        return FrameAction(action=buttons("Y"), reason="shredder_attack")
