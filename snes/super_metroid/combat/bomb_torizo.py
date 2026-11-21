"""Full-knowledge Bomb Torizo strategy (RAM positions + hitbox catalog).

Mirrors the Spore Spawn pattern: a deterministic controller that reads enemy
X/Y/HP/spritemap and chooses face / range / jump / fire each frame. This is
the baseline policy; structured RL can later refine the same feature vector.

Vision BC from the legacy project is intentionally not used here — it only
wins on its training save distribution and fails natural Flyway entry.

Clean-track economy: defaults hold a slightly wider kite band so natural
10-missile capacity + drops clear the fight without ammo writes (see
``docs/CLEAN_TRACK.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from super_metroid.combat.features import (
    bomb_torizo_catalog,
    features_from_state,
)
from super_metroid.ram import (
    BOMBS_MASK,
    FACING_RIGHT,
    GameplayPhase,
    SuperMetroidState,
    parse_env_state,
    read_bank7e_wram,
)
from super_metroid.routes.controller_common import select_weapon
from super_metroid.routes.kpdr.room_ids import ROOM_BOMB_TORIZO, ROOM_FLYWAY
from super_metroid.routes.runtime import ControllerSession, hold
from super_metroid.takeoff import walk_toward_x

# Idle statue spritemap (no combat AI) observed on incomplete entry states.
STATUE_SPRITEMAP = 0x87D0
# Room-load / chozo spawn before the idle statue settles.
SPAWN_SPRITEMAP = 0x804F
# Crateria boss bits $7E:D828 — bit 0x04 = Bomb Torizo defeated.
ADDR_CRATERIA_BOSS_BITS = 0xD828
BOMB_TORIZO_BOSS_BIT = 0x04


@dataclass(frozen=True)
class BombTorizoStrategy:
    """Tunable range-kite + missile spray parameters.

    Defaults proven clean on ``natural_bomb_torizo_active`` (10 missiles, no
    resource writes): wider band + longer jump holds vs the original spray so
    contact damage stays under starting energy while drops refill ammo.
    """

    # Continuous clean entry often has only ~3/10 missiles; wider kite + longer
    # jump holds farm egg drops while staying under contact lethal.
    min_range: int = 100
    max_range: int = 160
    jump_range: int = 100
    jump_hold_frames: int = 28
    jump_period: int = 40
    fire_period: int = 2  # fire every N frames
    max_fight_frames: int = 12_000
    # After body HP 0, wait this many frames for the Crateria boss bit.
    boss_bit_grace_frames: int = 1_200


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
    boss_bit_frame: int | None = None

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
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class BombTorizoSearchEvidence:
    """Development lookahead; only the resulting buttons belong on a Clean tape."""

    fight: BombTorizoEvidence
    strategy: BombTorizoStrategy
    candidates: int
    rollout_restores: int
    search_frames: int


class BombTorizoSearchError(RuntimeError):
    def __init__(self, candidates: int, search_frames: int, rollout_restores: int) -> None:
        self.candidates = candidates
        self.search_frames = search_frames
        self.rollout_restores = rollout_restores
        super().__init__(
            f"Bomb Torizo search found no survivor in {candidates} candidates "
            f"({search_frames} frames, {rollout_restores} restores)"
        )


@dataclass(frozen=True)
class BombTorizoLeaveEvidence:
    fight: BombTorizoSearchEvidence
    recovery_frames: int
    exit_frames: int
    final_health: int
    final_missiles: int


def play_flyway_to_bomb_torizo(session: ControllerSession) -> None:
    """Spend one full Missile pack at the Flyway hatch from the upper ledge."""
    if session.state.room_id != ROOM_FLYWAY or session.state.missiles < 5:
        raise RuntimeError("Flyway requires five Missiles for the pink hatch")

    if session.state.selected_item:
        hold(session, 1, "Y", reason="flyway_select_beam")
    jump_frame = None
    missiles_after_hatch = None
    for tick in range(650):
        state = session.state
        if state.room_id == ROOM_BOMB_TORIZO and state.game_state == 8 and not state.door_transition:
            return
        if (
            state.room_id != ROOM_FLYWAY
            or state.game_state != 8
            or (missiles_after_hatch is not None and state.missiles <= missiles_after_hatch)
        ):
            names = ("RIGHT", "B")
        elif state.samus_x < 700:
            names = ("RIGHT", "B")
            if state.samus_x >= 642 and jump_frame is None:
                jump_frame = tick
            if jump_frame is not None and tick - jump_frame < 31:
                names += ("A",)
            if jump_frame is None and tick % 2 == 0:
                names += ("X", "R")
        else:
            if missiles_after_hatch is None:
                missiles_after_hatch = state.missiles - 5
            if not state.selected_item:
                names = ("SELECT",)
            elif state.samus_y == 139 and state.vertical_direction == 0 and tick % 2 == 0:
                names = ("X",)
            else:
                names = ()
        hold(session, 1, *names, reason="flyway_missile_hatch")
    raise TimeoutError(f"Flyway hatch did not lead to Bomb Torizo: {session.state}")


def play_bomb_torizo_pickup(session: ControllerSession) -> None:
    """Clear the entry step and take Bombs from the statue on the left approach."""
    if session.state.room_id != ROOM_BOMB_TORIZO:
        raise RuntimeError("Bombs pickup requires Bomb Torizo's room")
    for tick in range(700):
        state = session.state
        if state.collected_items & BOMBS_MASK:
            return
        if state.samus_x < 112:
            names = ("RIGHT", "B")
        else:
            names = walk_toward_x(state.samus_x, 193, slack=1)
            if state.samus_y > 173 and state.vertical_direction != 2:
                names += ("A",)
        if state.selected_item and tick % 2 == 0:
            names += ("Y",)
        if state.samus_x >= 166 and state.samus_y >= 157 and tick % 2 == 0:
            names += ("X",)
        hold(session, 1, *names, reason="bomb_torizo_pickup")
    raise RuntimeError(f"Bombs pickup missed the statue: {session.state}")


def play_bomb_torizo_fast_leave(
    session: ControllerSession, *, statue_side_x: int = 165,
) -> None:
    """Fight from the statue side, then jump through the unlocked Flyway hatch."""
    state = session.state
    if state.room_id != ROOM_BOMB_TORIZO or not state.collected_items & BOMBS_MASK:
        raise RuntimeError("Bomb Torizo fight requires the natural Bombs pickup")
    env = session.env  # type: ignore[attr-defined]
    # Staying near the statue keeps the beam in range of both the boss and
    # its eggs. Pulse jump on close approaches, and use natural missile drops.
    for tick in range(3_600):
        state = session.state
        if state.health <= 0:
            raise RuntimeError("Bomb Torizo depleted Samus's energy")
        if bomb_torizo_boss_bit_set(env):
            break
        if state.enemy0_hp == 0:
            hold(session, 1, reason="bomb_torizo_unlock")
            continue
        dx = state.enemy0_x - state.samus_x
        toward = "RIGHT" if dx >= 0 else "LEFT"
        if state.samus_x > statue_side_x + 3:
            move = "LEFT"
        elif state.samus_x < statue_side_x - 3:
            move = "RIGHT"
        elif (state.facing == FACING_RIGHT) != (dx >= 0):
            move = toward
        else:
            move = None
        names = (move,) if move else ()
        if tick % 2 == 0 and move in (None, toward):
            names += ("X",)
        if abs(dx) < 96 and tick % 41 < 16:
            names += ("A",)
        if state.missiles and state.selected_item == 0:
            names += ("SELECT",)
        hold(session, 1, *names, reason="bomb_torizo_statue_side")
    else:
        raise RuntimeError(f"Bomb Torizo fight timed out: {session.state}")

    for tick in range(500):
        state = session.state
        if state.room_id == ROOM_FLYWAY and state.game_state == 8 and not state.door_transition:
            return
        names = ("LEFT",)
        if tick % 50 < 28:
            names += ("A",)
        if tick % 2 == 0:
            names += ("X",)
        hold(session, 1, *names, reason="bomb_torizo_fast_flyway_exit")
    raise RuntimeError(f"Bomb Torizo missed Flyway: {session.state}")


def play_bomb_torizo_one_pack_leave(session: ControllerSession) -> None:
    """Hold a deeper statue-side position to collect eggs on the one-pack route."""
    play_bomb_torizo_fast_leave(session, statue_side_x=158)


def play_clean_bomb_torizo_leave(
    session: ControllerSession,
) -> BombTorizoLeaveEvidence:
    """Search the live low-ammo fight, collect natural drops, leave for Flyway.

    The short fan tries the two survivors measured on the power-on human
    chain. If that entry has drifted enough that neither survives, expand to
    the full fan from the same entry. No resource or selected-weapon writes
    occur: every interaction, including selection, uses controller buttons.
    """
    proven = tuple(
        BombTorizoStrategy(jump_hold_frames=12, jump_period=period)
        for period in (48, 32)
    )
    try:
        fight = search_bomb_torizo_fight(session, strategies=proven)
    except BombTorizoSearchError as failed:
        expanded = search_bomb_torizo_fight(session)
        fight = replace(
            expanded,
            candidates=expanded.candidates + failed.candidates,
            search_frames=expanded.search_frames + failed.search_frames,
            rollout_restores=expanded.rollout_restores + failed.rollout_restores,
        )

    # Dead Torizo and destroyed eggs release ordinary game pickups. Sweep
    # under their fall before leaving; the measured entry heals 28 -> 99.
    recovery_start = session.frame
    for index in range(900):
        state = session.state
        names = ["LEFT"]
        if index % 2 == 0:
            names.append("X")
        if state.samus_x < 120 and state.samus_y > 150 and index % 50 < 18:
            names.append("A")
        hold(session, 1, *names, reason="clean_bomb_torizo_natural_drops")
    recovery_frames = session.frame - recovery_start

    # Hold the jump through its ascent. Releasing A as soon as y<150 makes
    # the hop too short to shoot and enter this raised left doorway.
    exit_start = session.frame
    for index in range(600):
        state = session.state
        if state.room_id == ROOM_FLYWAY and state.game_state == 8 and not state.door_transition:
            return BombTorizoLeaveEvidence(
                fight=fight,
                recovery_frames=recovery_frames,
                exit_frames=session.frame - exit_start,
                final_health=state.health,
                final_missiles=state.missiles,
            )
        names = ["LEFT"]
        if index < 28:
            names.append("A")
        if index % 2 == 0:
            names.append("X")
        hold(session, 1, *names, reason="clean_bomb_torizo_flyway_exit")
    raise RuntimeError(f"Clean Bomb Torizo missed settled Flyway: {session.state}")


def search_bomb_torizo_fight(
    session: ControllerSession,
    *,
    strategies: tuple[BombTorizoStrategy, ...] | None = None,
) -> BombTorizoSearchEvidence:
    """Try a strategy fan from the live frame, then play the winning buttons.

    Every branch uses the same real entry, natural energy/ammo, and the ROM.
    Branches never run through the caller's assist or recording callbacks.
    Prefer surviving energy, then ammo, then time; failed branches cannot
    modify the caller's state. A loadless replay of the recorded buttons is
    the proof, rather than this development search.

    The fan includes the 2026-11-16 winner from the human power-on chain's
    94-energy, three-missile entry: range 100, jump hold 12, period 48.
    """
    env = getattr(session, "env", None)
    if env is None:
        raise TypeError("Bomb Torizo search needs a live emulator session")
    if strategies is None:
        strategies = tuple(
            BombTorizoStrategy(
                min_range=minimum,
                jump_range=jump_range,
                jump_hold_frames=hold_frames,
                jump_period=period,
                max_fight_frames=7000,
            )
            for minimum, jump_range, hold_frames, period in product(
                (60, 100, 140), (100, 160), (12, 28), (32, 48)
            )
        )
    saved = env.em.get_state()
    search_frames = 0
    restores = 0
    best: tuple[tuple[int, int, int], BombTorizoStrategy] | None = None

    class Branch:
        def __init__(self) -> None:
            self.env = env
            self.frame = 0
            self.state = parse_env_state(env, mode="nav")

        def step(self, action: Any, reason: str = "") -> SuperMetroidState:
            env.step(action)
            self.frame += 1
            self.state = parse_env_state(env, mode="nav")
            return self.state

    try:
        for strategy in strategies:
            env.em.set_state(saved)
            restores += 1
            branch = Branch()
            evidence = play_bomb_torizo_fight(branch, strategy=strategy)
            search_frames += branch.frame
            if evidence.outcome != "bomb_torizo_defeated":
                continue
            score = (branch.state.health, branch.state.missiles, -branch.frame)
            if best is None or score > best[0]:
                best = score, strategy
    finally:
        env.em.set_state(saved)
        restores += 1
    if best is None:
        raise BombTorizoSearchError(len(strategies), search_frames, restores)

    # The winning policy is deterministic from the restored entry. Play it
    # through the owner to preserve the caller's frame/transition telemetry.
    evidence = play_bomb_torizo_fight(session, strategy=best[1])
    return BombTorizoSearchEvidence(
        fight=evidence,
        strategy=best[1],
        candidates=len(strategies),
        rollout_restores=restores,
        search_frames=search_frames,
    )


def crateria_boss_bits(env: Any) -> int:
    """Read Crateria boss flags ($7E:D828) via bank-$7E WRAM."""
    return int(read_bank7e_wram(env)[ADDR_CRATERIA_BOSS_BITS])


def bomb_torizo_boss_bit_set(env: Any) -> bool:
    """True when Crateria boss bit 0x04 is set (Bomb Torizo)."""
    return bool(crateria_boss_bits(env) & BOMB_TORIZO_BOSS_BIT)


def fight_bomb_torizo_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: BombTorizoStrategy = BombTorizoStrategy(),
) -> tuple[str, ...]:
    """One-frame button names from full-knowledge features (no pixels).

    Defeat is keyed on ``enemy0_hp == 0`` only. Low-WRAM ``boss_bits`` from
    ``env.get_ram()`` are open-bus garbage and must not idle the controller.
    """
    catalog = bomb_torizo_catalog()
    feat = features_from_state(state, catalog)
    if state.enemy0_hp == 0:
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
    # Only pulse fire when missiles remain (or beam as fallback when empty —
    # still faces the boss). fire_period gates the spray rate.
    if frame_index % strategy.fire_period == 0:
        names.append("X")
    # Deduplicate while preserving order.
    return tuple(dict.fromkeys(names))


def play_bomb_torizo_fight(
    session: ControllerSession,
    *,
    strategy: BombTorizoStrategy = BombTorizoStrategy(),
    require_active: bool = True,
    require_boss_bit: bool = True,
) -> BombTorizoEvidence:
    """Fight Bomb Torizo until HP 0 (and optionally Crateria boss bit).

    Expects the session already in room ``0x9804`` with combat-active Torizo
    (or a live entry that activates when approached). Incomplete save-states
    that freeze on spritemap ``0x87D0`` will fail activation and return a
    non-success outcome.

    When ``require_boss_bit`` is True and the session exposes ``env``, wait
    after body HP 0 for $7E:D828 bit 0x04 (door unlock precondition).
    """
    catalog = bomb_torizo_catalog()
    start = session.frame
    if session.state.room_id != ROOM_BOMB_TORIZO:
        raise RuntimeError(
            f"Bomb Torizo fight expected room 0x{ROOM_BOMB_TORIZO:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    if session.state.selected_item != 1 and session.state.missiles > 0:
        select_weapon(session, 1)

    peak_hp = session.state.enemy0_hp
    min_hp = session.state.enemy0_hp
    activation_seen = session.state.enemy0_spritemap != STATUE_SPRITEMAP
    defeat_frame: int | None = None
    boss_bit_frame: int | None = None
    prev_hp = session.state.enemy0_hp
    env = getattr(session, "env", None)

    for index in range(strategy.max_fight_frames):
        state = session.state
        if state.health == 0 or state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
            break
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
            if not require_boss_bit or env is None:
                break
        prev_hp = session.state.enemy0_hp

        if (
            defeat_frame is not None
            and require_boss_bit
            and env is not None
            and bomb_torizo_boss_bit_set(env)
        ):
            boss_bit_frame = session.frame
            break

        if (
            defeat_frame is not None
            and require_boss_bit
            and env is not None
            and (session.frame - defeat_frame) >= strategy.boss_bit_grace_frames
        ):
            break

    if defeat_frame is not None and (
        not require_boss_bit
        or env is None
        or boss_bit_frame is not None
        or bomb_torizo_boss_bit_set(env)
    ):
        if boss_bit_frame is None and env is not None and bomb_torizo_boss_bit_set(env):
            boss_bit_frame = session.frame
        outcome = "bomb_torizo_defeated"
    elif session.state.health == 0 or session.state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
        outcome = "death"
    elif require_active and not activation_seen:
        outcome = "torizo_inactive_statue"
    elif defeat_frame is not None and require_boss_bit:
        outcome = "boss_bit_timeout"
    else:
        outcome = "timeout"

    return BombTorizoEvidence(
        start_frame=start,
        activation_seen=activation_seen,
        defeat_frame=defeat_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_hp=peak_hp,
        min_enemy_hp=min_hp,
        action_frames=session.frame - start,
        final_enemy_hp=session.state.enemy0_hp,
        outcome=outcome,
    )
