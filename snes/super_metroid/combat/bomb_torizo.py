"""Full-knowledge Bomb Torizo strategy (RAM positions + hitbox catalog).

Mirrors the Spore Spawn pattern: a deterministic controller that reads enemy
X/Y/HP/spritemap and chooses face / range / jump / fire each frame. This is
the baseline policy; structured RL can later refine the same feature vector.

Vision BC from the legacy project is intentionally not used here — it only
wins on its training save distribution and fails natural Flyway entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    bomb_torizo_catalog,
    features_from_state,
)
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import select_weapon
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_BOMB_TORIZO = 0x9804
# Idle statue spritemap (no combat AI) observed on incomplete entry states.
STATUE_SPRITEMAP = 0x87D0
# Room-load / chozo spawn before the idle statue settles.
SPAWN_SPRITEMAP = 0x804F


@dataclass(frozen=True)
class BombTorizoStrategy:
    """Tunable range-kite + missile spray parameters."""

    min_range: int = 70
    max_range: int = 120
    jump_range: int = 100
    jump_hold_frames: int = 18
    jump_period: int = 50
    fire_period: int = 2  # fire every N frames
    max_fight_frames: int = 8_000


@dataclass(frozen=True)
class BombTorizoEvidence:
    start_frame: int
    activation_seen: bool
    defeat_frame: int | None
    end_frame: int
    peak_hp: int
    min_enemy_hp: int
    action_frames: int
    final_enemy_hp: int
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "activation_seen": self.activation_seen,
            "defeat_frame": self.defeat_frame,
            "end_frame": self.end_frame,
            "peak_hp": self.peak_hp,
            "min_enemy_hp": self.min_enemy_hp,
            "action_frames": self.action_frames,
            "final_enemy_hp": self.final_enemy_hp,
            "outcome": self.outcome,
        }


def fight_bomb_torizo_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: BombTorizoStrategy = BombTorizoStrategy(),
) -> tuple[str, ...]:
    """One-frame button names from full-knowledge features (no pixels)."""
    catalog = bomb_torizo_catalog()
    feat = features_from_state(state, catalog)
    if feat.enemy_defeated or state.enemy0_hp == 0:
        return ()
    if not feat.enemy_active and state.enemy0_spritemap in (
        STATUE_SPRITEMAP,
        SPAWN_SPRITEMAP,
    ):
        # Pre-combat chozo/statue: walk right to touch (live entries).
        return ("RIGHT",)

    dx = feat.dx
    face = "RIGHT" if dx >= 0 else "LEFT"
    abs_dx = abs(dx)
    if abs_dx < strategy.min_range:
        move = "LEFT" if dx >= 0 else "RIGHT"
    elif abs_dx > strategy.max_range:
        move = face
    else:
        move = face

    names: list[str] = [move]
    if abs_dx < strategy.jump_range and (
        frame_index % strategy.jump_period < strategy.jump_hold_frames
    ):
        names.append("A")
    if frame_index % strategy.fire_period == 0:
        names.append("X")
    # Deduplicate while preserving order.
    return tuple(dict.fromkeys(names))


def play_bomb_torizo_fight(
    session: ControllerSession,
    *,
    strategy: BombTorizoStrategy = BombTorizoStrategy(),
    require_active: bool = True,
) -> BombTorizoEvidence:
    """Fight Bomb Torizo until HP 0 using the structured strategy.

    Expects the session already in room ``0x9804`` with combat-active Torizo
    (or a live entry that activates when approached). Incomplete save-states
    that freeze on spritemap ``0x87D0`` will fail activation and return a
    non-success outcome.
    """
    catalog = bomb_torizo_catalog()
    start = session.frame
    if session.state.room_id != ROOM_BOMB_TORIZO:
        raise RuntimeError(
            f"Bomb Torizo fight expected room 0x{ROOM_BOMB_TORIZO:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    if session.state.selected_item != 1 and session.state.max_missiles > 0:
        select_weapon(session, 1)

    peak_hp = session.state.enemy0_hp
    min_hp = session.state.enemy0_hp
    activation_seen = session.state.enemy0_spritemap != STATUE_SPRITEMAP
    defeat_frame: int | None = None
    prev_hp = session.state.enemy0_hp

    for index in range(strategy.max_fight_frames):
        state = session.state
        peak_hp = max(peak_hp, state.enemy0_hp)
        min_hp = min(min_hp, state.enemy0_hp)
        feat = features_from_state(state, catalog)
        if feat.enemy_active:
            activation_seen = True

        names = fight_bomb_torizo_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_bomb_torizo")
        else:
            hold(session, 1, reason="fight_bomb_torizo_idle")

        if (
            defeat_frame is None
            and session.state.enemy0_hp == 0
            and prev_hp > 0
        ):
            defeat_frame = session.frame
            min_hp = 0
            break
        prev_hp = session.state.enemy0_hp

    if defeat_frame is not None:
        outcome = "bomb_torizo_defeated"
    elif require_active and not activation_seen:
        outcome = "torizo_inactive_statue"
    else:
        outcome = "timeout"

    return BombTorizoEvidence(
        start_frame=start,
        activation_seen=activation_seen,
        defeat_frame=defeat_frame,
        end_frame=session.frame,
        peak_hp=peak_hp,
        min_enemy_hp=min_hp,
        action_frames=session.frame - start,
        final_enemy_hp=session.state.enemy0_hp,
        outcome=outcome,
    )
