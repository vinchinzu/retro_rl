"""CombatProfile + fight() seam for Stage1Policy's fight_seq.

Elevated / Raphael Starbase / Krang poke / duo stay here (not pre-tree
tactics). Mode-7 empty-lane hold lives in ``tactics.neon``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from retro_harness.actions import buttons
from retro_harness.combat import AttackCadence, PreferredFlank, fight_nearest_action
from retro_harness.controls import SNES_LEFT, SNES_RIGHT
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameState
from tmnt_iv.grind_knobs import active_knobs
from tmnt_iv.stages import (
    DUO_BOSS_CHARS,
    KRANG_CHAR,
    NEON_MIN_FIGHT_Y,
    RAPH_CHAR,
    RAPH_STARBASE_GROUND_CHARS,
    WOUNDED_KNEE_JUMP_CHARS,
    is_neon_highway,
    is_prehistoric,
    is_sewer,
    is_starbase,
    is_wounded_knee,
)
from tmnt_iv.tactics.neon import NeonLaneTactics
from tmnt_iv.tactics.raph_air import raph_starbase_jump_action
from tmnt_iv.tactics.slash import SLASH_CHAR

# Screen coords: UP decreases Y (probe-confirmed). Do NOT invert.
_Y_TOLERANCE = 8
# Big Apple (stage 0): wider poke + tighter cadence clears Baxter
# heal=none from Boss.state; also cuts wave chip before the pizza windows.
_STAGE1_ATTACK_RANGE = 64
_STAGE1_MIN_RANGE = 8
_STAGE1_STANDOFF = 32
_STAGE1_ATTACK_HOLD = 2
_STAGE1_ATTACK_GAP = 2
# Alleycat Blues (stage byte 1): one-frame poke + left shoulder.
# Clean track (2026-07-27): left flank + standoff 36 → emergency Stage2
# **14,231f / 159 dmg / 2 heals** (was ~196/3 without left shoulder).
# heal=none still dies mid-alley (~68 dmg) — pizza seek desyncs packs;
# underfoot pickup only. Historic tuned row 15,453f / 124 / 1 may need
# the pre-Stage1-Clean policy snapshot.
# Clean rejects (2026-08-01) — keep left flank + standoff 36 / min_range 8:
# - min_range 8→20: worse (7310f / 130 dmg vs 8391f / 104)
# - standoff 36→48: no-op (identical 8391/104)
# - AlleycatPackSpace LEFT on ≥2 near: worse (5349f / 88, earlier death)
# Residual: post-pizza 0x5E pile-ons (2×24 dmg) at progress ~21126–21464.
# Pack overrides live in tactics/alleycat.py (REACH to the 0x5E window at
# HP 80; CKPT still dies on a left-clump 24). See docs/tasks/rr-t4s2-residual.md.
_ALLEY_Y_TOLERANCE = 6
_ALLEY_ATTACK_RANGE = 65
_ALLEY_MIN_RANGE = 0
_ALLEY_STANDOFF = 36
_ALLEY_ATTACK_HOLD = 1
_ALLEY_ATTACK_GAP = 2
_ALLEY_PREFERRED_FLANK = PreferredFlank.LEFT
# Skull & Crossbones / Wounded Knee: one less release frame improves the
# exact-entry clears without raising risk.  FullHardStage6: 19,892→16,793f,
# 802→526 damage; FullHardStage7: 18,856→17,484f, 924→830 damage.
_LATE_ATTACK_HOLD = 2
_LATE_ATTACK_GAP = 4
# Raphael's shorter weapon benefits from one fewer release frame on Wounded
# Knee waves, then a one-frame release against Leatherhead. Natural-entry
# probes: waves 18,156→10,573f / 497→362 dmg; boss 6,712→5,249f /
# 356→276 dmg.
_RAPH_WOUNDED_ATTACK_GAP = 3
_RAPH_LEATHERHEAD_ATTACK_GAP = 1
# Sewer Surfin' (stage byte 2): a broad lane tolerance avoids chasing Foot
# through the hanging spikes while keeping the normal two-frame poke.
_SEWER_Y_TOLERANCE = 36
_SEWER_MIN_FIGHT_Y = 160
_SEWER_ATTACK_RANGE = 64
_SEWER_MIN_RANGE = 8
_SEWER_STANDOFF = 24
_SEWER_ATTACK_HOLD = 2
_SEWER_ATTACK_GAP = 2
# Rat King: long poke from water lane. heal=none Boss3 ~68 dmg to kill
# (min HP 12). Grounded RIGHT at the left wall (continuous B+RIGHT
# soft-locks x≈24). boss_active stays true down to HP 1 so finishers land.
_RAT_KING_CHAR = 0x4A
_RAT_KING_Y_TOLERANCE = 32
_RAT_KING_ATTACK_RANGE = 120
_RAT_KING_MIN_RANGE = 8
_RAT_KING_STANDOFF = 24
_RAT_KING_ATTACK_HOLD = 2
_RAT_KING_ATTACK_GAP = 1
_BOSS_LEFT_CHIP_X = 80
# Leo weapon reach is generous (~50–68); keep a modest poke band.
# Runtime values come from ``active_knobs()`` so local grind can A/B.
_ATTACK_RANGE = 56
_MIN_RANGE = 8
_STANDOFF = 24
_ATTACK_HOLD = 2
_ATTACK_GAP = 5
# Legacy aliases (other stages / knobs); Rat King uses _RAT_KING_* above.
_BOSS_ATTACK_RANGE = _RAT_KING_ATTACK_RANGE
_BOSS_MIN_RANGE = _RAT_KING_MIN_RANGE
_BOSS_STANDOFF = _RAT_KING_STANDOFF
_BOSS_ATTACK_HOLD = _RAT_KING_ATTACK_HOLD
_BOSS_ATTACK_GAP = _RAT_KING_ATTACK_GAP
# Player/enemy X are screen-space; combat sees camera_x=0 (see fight).
_CAMERA_LEFT_MARGIN = 24
_CAMERA_RIGHT_MARGIN = 220
_EDGE_ATTACK_BONUS = 16
_LEFT_THREAT_X = 80

# Starbase: hover Foot / teleporter spawns / stack tops whiff grounded Y.
_STARBASE_JUMP_CHARS: frozenset[int] = frozenset({0x6A, 0x6C, 0xB0, 0xB2, 0xB4, 0xF2})
# Mode-7 poke band. Near-band Y / Krang id live in stages.py (NeonLane).
_NEON_Y_TOLERANCE = 48
_NEON_ATTACK_RANGE = 68
_KRANG_LEFT_STANDOFF = 36
_SHREDDER_F1_ATTACK_RANGE = 72
_SHREDDER_F1_Y_TOLERANCE = 8

# The SNES-only Technodrome finale is a special interaction, not a normal
# beat-'em-up lock.  Pink Foot / tank throw live in ``tactics.technodrome``.
_TECHNODROME_STAGE = 3
_TECHNODROME_JUMP_CHARS: frozenset[int] = frozenset({0x6A})
# General elevated jump: only true air / high platforms — ordinary lane
# offsets (~20–40px) still use walk-align + grounded Y.
_ELEVATED_JUMP_DY = 44
_ELEVATED_JUMP_ADX = 72


def _combat_knobs() -> tuple[int, int, int, int, int]:
    """Live poke-band knobs (defaults match module constants above)."""
    k = active_knobs()
    return (
        k.attack_range,
        k.min_range,
        k.standoff,
        k.attack_hold,
        k.attack_gap,
    )


def _sewer_combat_state(state: GameState) -> GameState:
    """Zero progress-camera; clamp fight Y out of Stage 3 spike band."""
    combat = replace(state, camera_x=0)
    # Rat King: keep real Y and all living slots — long pokes from the
    # water lane are what actually reduce boss HP (top-lane whiffs).
    if state.boss_active or not is_sewer(state):
        return combat
    enemies = tuple(
        replace(e, y=max(e.y, _SEWER_MIN_FIGHT_Y)) if e.active and e.health > 0 else e
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


def _keep_sewer_pace(state: GameState, action: FrameAction) -> FrameAction:
    """Hold RIGHT during Sewer Surfin'; recover from Rat King left chip.

    Continuous B+RIGHT at the left wall soft-locks x≈24 (jump thrash).
    Prefer grounded RIGHT until mid-screen; hop only in the 56–80 band.
    Empty-screen drop-lane lives in ``PlayerXStallWalk`` (fight_seq is
    living-enemies only).
    """
    if not is_sewer(state):
        return action
    # Escape auto-scroll left chip before anything else.
    if state.boss_active and state.player_x <= _BOSS_LEFT_CHIP_X:
        if state.player_x <= 56:
            # Grounded run — continuous jump freezes recovery on the wall.
            return FrameAction(action=buttons("RIGHT"), reason="boss_run_right")
        return FrameAction(action=buttons("B", "RIGHT"), reason="boss_jump_right")
    held = list(action.action)
    if not state.boss_active and held[SNES_LEFT]:
        # Wave LEFT walks into hanging spikes (LiveHard 4-dmg at x=96).
        # Keep only a short overlap retreat when a Foot is on the right.
        overlapping_right = any(
            0 < enemy.x - state.player_x <= _SEWER_STANDOFF
            for enemy in state.living_enemies
        )
        if action.reason == "space_left" and overlapping_right:
            return action
        held[SNES_LEFT] = 0
        held[SNES_RIGHT] = 1
        return FrameAction(action=held, reason="walk_right")
    held[SNES_RIGHT] = 1
    return FrameAction(action=held, reason=action.reason)


def _neon_combat_state(state: GameState) -> GameState:
    """Zero progress-camera; keep only near-band / Krang Mode-7 targets."""
    combat = replace(state, camera_x=0)
    if not is_neon_highway(state):
        return combat
    enemies = tuple(
        e
        if (
            e.active
            and e.health > 0
            and (e.kind == KRANG_CHAR or e.y >= NEON_MIN_FIGHT_Y)
        )
        else replace(e, active=False, health=0)
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


def _duo_boss_combat_state(state: GameState) -> GameState:
    """Keep only the duo boss slots so adds cannot steal targeting."""
    combat = replace(state, camera_x=0)
    enemies = tuple(
        e
        if (e.active and e.health > 0 and e.kind in DUO_BOSS_CHARS)
        else replace(e, active=False, health=0)
        for e in combat.enemies
    )
    return replace(combat, enemies=enemies)


def _duo_boss_fight_action(
    state: GameState,
    *,
    cadence: AttackCadence,
) -> FrameAction | None:
    """Left-flank poke for Tokka/Rahzar and Bebop/Rocksteady.

    Shared ``fight_nearest_action`` treats dx≤70 as in-range even when Leo is
    on the wrong side of the duo, so he freezes at screen-right mashing Y
    forever. Force a left standoff first, then cadence Y.
    """
    if not (
        (state.boss_active and state.stage in {3, 5})
        or any(e.kind in DUO_BOSS_CHARS for e in state.living_enemies)
    ):
        return None
    if state.stage not in {3, 5}:
        return None
    bosses = [
        e for e in state.living_enemies if e.kind in DUO_BOSS_CHARS and e.health > 0
    ]
    if not bosses:
        return None
    cadence.hold_frames = _BOSS_ATTACK_HOLD
    cadence.gap_frames = _BOSS_ATTACK_GAP
    # Prefer the rightmost living boss so we work left through the pair.
    target = max(bosses, key=lambda e: e.x)
    ideal_x = target.x - 36
    dx = state.player_x - target.x
    dy = target.y - state.player_y

    # In the continuous route the right-wall door can pin Leo at x≈224
    # just as Tokka/Rahzar spawn. Plain LEFT never clears that collision,
    # while a short jump-left does; checkpoint boss states start left of it.
    if state.player_x >= 216:
        return FrameAction(
            action=buttons("B", "LEFT"),
            reason="duo_wall_escape",
        )
    if abs(dy) > 12:
        return FrameAction(
            action=buttons("UP" if dy < 0 else "DOWN"),
            reason="align_up" if dy < 0 else "align_down",
        )
    # Not yet on left flank (or overlapping) — close/space horizontally.
    if state.player_x > target.x - 18:
        return FrameAction(action=buttons("LEFT"), reason="approach_left")
    if state.player_x < ideal_x - 14:
        return FrameAction(action=buttons("RIGHT"), reason="approach_right")
    if abs(dx) < 14:
        return FrameAction(action=buttons("LEFT"), reason="space_left")
    # On left standoff band — poke.
    return cadence.next_attack(button="Y")


def _neon_fight_action(state: GameState) -> FrameAction | None:
    """Left-flank Krang poke on the Mode-7 near band.

    Lane hold is ``NeonLaneTactics`` (same object as empty-screen walk).
    Returns ``None`` when the shared ``fight_nearest_action`` path should run.
    """
    if not is_neon_highway(state):
        return None
    near = [
        e
        for e in state.living_enemies
        if e.kind == KRANG_CHAR or e.y >= NEON_MIN_FIGHT_Y
    ]
    if not near:
        return NeonLaneTactics().next(state)
    if state.boss_active or any(e.kind == KRANG_CHAR for e in near):
        krang = next((e for e in near if e.kind == KRANG_CHAR), near[0])
        target_x = krang.x - _KRANG_LEFT_STANDOFF
        dx = target_x - state.player_x
        dy = krang.y - state.player_y
        if abs(dx) > 14:
            return FrameAction(
                action=buttons("LEFT" if dx < 0 else "RIGHT"),
                reason=("approach_left" if dx < 0 else "approach_right"),
            )
        if abs(dy) > 24:
            return FrameAction(
                action=buttons("UP" if dy < 0 else "DOWN"),
                reason="align_up" if dy < 0 else "align_down",
            )
        return FrameAction(action=buttons("Y"), reason="attack")
    return None


def _suppress_elevated_jump(state: GameState) -> bool:
    """True when elevated jump-slash is known to whiff or desync the fight."""
    # Generic elev≥44 jump is Stage-0 (Big Apple air Foot / Baxter assist).
    # On Alleycat it false-fires into packs and blew Stage2 emergency from
    # ~190 dmg / 2 heals to ~443 / 5. Stage-specific B+Y (dino, stack,
    # hover, spear Foot) still run via their own fight branches.
    if state.stage != 0:
        return True
    # Mode-7 Y is depth, not height — jump-slash on "high" Foot is wrong.
    if is_neon_highway(state):
        return True
    # Rat King: standing jump-slashes whiff; use long grounded poke.
    if state.boss_active and is_sewer(state):
        return True
    # Slash shell / Super Shredder form 2: jump-slash whiffs.
    if state.boss_active and is_prehistoric(state):
        return True
    if state.stage == 9:
        return True
    if any(e.kind == SLASH_CHAR for e in state.living_enemies):
        return True
    return False


def _elevated_jump_slash(state: GameState) -> FrameAction | None:
    """Jump-slash flying / elevated targets that grounded Y cannot reach.

    Positive elevation = enemy above player (smaller screen Y). Used for
    Baxter, airborne Foot, and any mid-air thug within a generous X band.
    """
    if _suppress_elevated_jump(state):
        return None
    living = state.living_enemies
    if not living:
        return None
    # Prefer the nearest elevated threat; fall back to nearest living.
    elevated = [
        e
        for e in living
        if (state.player_y - e.y) >= _ELEVATED_JUMP_DY
    ]
    if not elevated:
        return None
    target = min(
        elevated,
        key=lambda e: abs(e.x - state.player_x) + abs(e.y - state.player_y),
    )
    adx = abs(target.x - state.player_x)
    if adx > _ELEVATED_JUMP_ADX:
        return None
    toward: list[str] = []
    if target.x > state.player_x + 6:
        toward.append("RIGHT")
    elif target.x < state.player_x - 6:
        toward.append("LEFT")
    return FrameAction(
        action=buttons("B", "Y", *toward),
        reason="jump_slash",
    )


@dataclass(frozen=True)
class _PokeRow:
    """Six poke-band fields + flank. ``None`` means overlay live knobs."""

    y_tolerance: int
    attack_range: int | None
    min_range: int | None
    standoff: int | None
    hold_frames: int | None
    gap_frames: int | None
    flank: PreferredFlank


_DUO_ROW = _PokeRow(_Y_TOLERANCE, 70, 12, 36, _BOSS_ATTACK_HOLD, _BOSS_ATTACK_GAP, PreferredFlank.LEFT)
_SHREDDER_ROW = _PokeRow(
    _SHREDDER_F1_Y_TOLERANCE,
    _SHREDDER_F1_ATTACK_RANGE,
    10,
    28,
    _BOSS_ATTACK_HOLD,
    _BOSS_ATTACK_GAP,
    PreferredFlank.NONE,
)
_NEON_ROW = _PokeRow(_NEON_Y_TOLERANCE, _NEON_ATTACK_RANGE, None, None, None, None, PreferredFlank.NONE)
_ALLEY_ROW = _PokeRow(
    _ALLEY_Y_TOLERANCE,
    _ALLEY_ATTACK_RANGE,
    _ALLEY_MIN_RANGE,
    _ALLEY_STANDOFF,
    _ALLEY_ATTACK_HOLD,
    _ALLEY_ATTACK_GAP,
    _ALLEY_PREFERRED_FLANK,
)
_STAGE0_ROW = _PokeRow(
    _Y_TOLERANCE,
    _STAGE1_ATTACK_RANGE,
    _STAGE1_MIN_RANGE,
    _STAGE1_STANDOFF,
    _STAGE1_ATTACK_HOLD,
    _STAGE1_ATTACK_GAP,
    PreferredFlank.NONE,
)
_LATE_ROW = _PokeRow(
    _Y_TOLERANCE,
    _ATTACK_RANGE,
    _MIN_RANGE,
    _STANDOFF,
    _LATE_ATTACK_HOLD,
    _LATE_ATTACK_GAP,
    PreferredFlank.NONE,
)

# Keyed by (stage byte, boss_active). Miss → knobs + _Y_TOLERANCE.
_POKE_TABLE: dict[tuple[int, bool], _PokeRow] = {
    (2, True): _PokeRow(
        _RAT_KING_Y_TOLERANCE,
        _RAT_KING_ATTACK_RANGE,
        _RAT_KING_MIN_RANGE,
        _RAT_KING_STANDOFF,
        _RAT_KING_ATTACK_HOLD,
        _RAT_KING_ATTACK_GAP,
        PreferredFlank.NONE,
    ),
    (3, True): _DUO_ROW,
    (5, True): _DUO_ROW,
    (4, True): _PokeRow(_Y_TOLERANCE, 72, 10, 28, _BOSS_ATTACK_HOLD, _BOSS_ATTACK_GAP, PreferredFlank.NONE),
    (8, True): _SHREDDER_ROW,
    (9, True): _SHREDDER_ROW,
    (7, False): _NEON_ROW,
    (7, True): _NEON_ROW,
    (2, False): _PokeRow(
        _SEWER_Y_TOLERANCE,
        _SEWER_ATTACK_RANGE,
        _SEWER_MIN_RANGE,
        _SEWER_STANDOFF,
        _SEWER_ATTACK_HOLD,
        _SEWER_ATTACK_GAP,
        PreferredFlank.NONE,
    ),
    (1, False): _ALLEY_ROW,
    (1, True): _ALLEY_ROW,
    (0, False): _STAGE0_ROW,
    (0, True): _STAGE0_ROW,
    (5, False): _LATE_ROW,
    (6, False): _LATE_ROW,
    (6, True): _LATE_ROW,
}


def _poke_fields(
    state: GameState,
) -> tuple[int, int, int, int, int, int, PreferredFlank]:
    """Table poke-band plus knobs / Raphael Leatherhead gap overlays."""
    row = _POKE_TABLE.get((state.stage, bool(state.boss_active)))
    if row is None:
        attack_range, min_range, standoff, hold, gap = _combat_knobs()
        y_tol = _Y_TOLERANCE
        flank = PreferredFlank.NONE
    else:
        knobs = _combat_knobs()
        y_tol = row.y_tolerance
        attack_range = knobs[0] if row.attack_range is None else row.attack_range
        min_range = knobs[1] if row.min_range is None else row.min_range
        standoff = knobs[2] if row.standoff is None else row.standoff
        hold = knobs[3] if row.hold_frames is None else row.hold_frames
        gap = knobs[4] if row.gap_frames is None else row.gap_frames
        flank = row.flank
    if (
        state.stage == 6
        and int(state.extras.get("char_id", -1)) == RAPH_CHAR
    ):
        gap = (
            _RAPH_LEATHERHEAD_ATTACK_GAP
            if state.boss_active
            else _RAPH_WOUNDED_ATTACK_GAP
        )
    return y_tol, attack_range, min_range, standoff, hold, gap, flank


def _jump_kinds(state: GameState) -> frozenset[int]:
    """Wave-only jump-slash kinds (empty during bosses)."""
    if state.boss_active:
        return frozenset()
    if is_prehistoric(state):
        return frozenset({0x6C})
    if state.stage == _TECHNODROME_STAGE:
        return _TECHNODROME_JUMP_CHARS
    if is_wounded_knee(state):
        return WOUNDED_KNEE_JUMP_CHARS
    if is_starbase(state):
        raph_ground = (
            int(state.extras.get("char_id", -1)) == RAPH_CHAR
            and any(
                e.kind in RAPH_STARBASE_GROUND_CHARS
                for e in state.living_enemies
            )
        )
        if not raph_ground:
            return _STARBASE_JUMP_CHARS
    return frozenset()


@dataclass(frozen=True)
class CombatProfile:
    """Poke-band + overlay; ``action`` emits one fight frame after specials."""

    y_tolerance: int
    attack_range: int
    min_range: int
    standoff: int
    hold_frames: int
    gap_frames: int
    flank: PreferredFlank
    combat_state: Callable[[GameState], GameState]
    wide_right_margin: bool
    jump_kinds: frozenset[int]

    def action(self, state: GameState, cadence: AttackCadence) -> FrameAction:
        """Emit fight_nearest + sewer pace + jump-kinds overlay."""
        cadence.hold_frames = self.hold_frames
        cadence.gap_frames = self.gap_frames
        action = _keep_sewer_pace(
            state,
            fight_nearest_action(
                self.combat_state(state),
                y_tolerance=self.y_tolerance,
                attack_range=self.attack_range,
                min_range=self.min_range,
                attack_button="Y",
                invert_vertical=False,
                cadence=cadence,
                preferred_flank=self.flank,
                standoff=self.standoff,
                use_throw=False,
                prefer_left_threat=not state.boss_active,
                left_threat_x=_LEFT_THREAT_X,
                camera_left_margin=_CAMERA_LEFT_MARGIN,
                camera_right_margin=(
                    400 if self.wide_right_margin else _CAMERA_RIGHT_MARGIN
                ),
                edge_attack_bonus=_EDGE_ATTACK_BONUS,
            ),
        )
        if (
            action.reason == "attack"
            and not state.boss_active
            and any(e.kind in self.jump_kinds for e in state.living_enemies)
        ):
            return FrameAction(action=buttons("B", "Y"), reason="jump_slash")
        return action

    @classmethod
    def from_state(cls, state: GameState) -> CombatProfile:
        """Build the per-stage poke-band for this frame."""
        technodrome_boss = state.boss_active and state.stage == 3
        prehistoric_boss = state.boss_active and is_prehistoric(state)
        pirate_boss = state.boss_active and state.stage == 5
        neon_boss = state.boss_active and is_neon_highway(state)
        shredder_boss = state.boss_active and is_starbase(state)
        if technodrome_boss or pirate_boss:
            combat_state = _duo_boss_combat_state
        elif is_neon_highway(state):
            combat_state = _neon_combat_state
        else:
            combat_state = _sewer_combat_state
        far_park = any(e.x > _CAMERA_RIGHT_MARGIN + 24 for e in state.living_enemies)
        wide_right = (
            far_park
            or (state.boss_active and state.stage == 0)
            or is_sewer(state)
            or technodrome_boss
            or prehistoric_boss
            or pirate_boss
            or neon_boss
            or shredder_boss
            or is_neon_highway(state)
            or is_starbase(state)
        )
        y_tol, attack_range, min_range, standoff, hold, gap, flank = _poke_fields(state)
        return cls(
            y_tolerance=y_tol,
            attack_range=attack_range,
            min_range=min_range,
            standoff=standoff,
            hold_frames=hold,
            gap_frames=gap,
            flank=flank,
            combat_state=combat_state,
            wide_right_margin=wide_right,
            jump_kinds=_jump_kinds(state),
        )


def fight(state: GameState, cadence: AttackCadence) -> FrameAction:
    """One fight frame: elevated → raph → neon → duo → profile poke/jump."""
    elevated = _elevated_jump_slash(state)
    if elevated is not None:
        return elevated
    raph = raph_starbase_jump_action(state)
    if raph is not None:
        return raph
    neon = _neon_fight_action(state)
    if neon is not None:
        return neon
    duo = _duo_boss_fight_action(state, cadence=cadence)
    if duo is not None:
        return duo
    return CombatProfile.from_state(state).action(state, cadence)
